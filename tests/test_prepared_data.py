import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import torch

from tim_compat.prepared_data import prepare_manifest


class PreparedDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "prepared"
        directory = self.root / "samples" / "one"
        directory.mkdir(parents=True)
        self.directory = directory
        for name, channels in (("conversation.wav", 2), ("voice_prompt_left.wav", 1)):
            with wave.open(str(directory / name), "wb") as audio:
                audio.setnchannels(channels)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b"\0\0" * channels * 24000)
        self.words = [{"word": "hello", "start": 0.1, "end": 0.4, "speaker": "agent"},
                      {"word": "yes", "start": 0.5, "end": 0.8, "speaker": "user"}]
        (directory / "words.json").write_text(json.dumps(self.words))
        (directory / "metadata.json").write_text(json.dumps({
            "agent_channel": "left", "user_channel": "right", "text_prompt": "Be helpful."}))
        self.manifest = self.root / "train.jsonl"
        self.manifest.write_text(json.dumps({"sample_id": "one", "sample_dir": "samples/one"}) + "\n")

    def test_text_modes_transform_only_agent_targets_and_keep_raw_cache(self):
        words = [{"word": "tiếng", "start": .1, "end": .4, "speaker": "agent"},
                 {"word": "Đặng", "start": .5, "end": .8, "speaker": "user"}]
        original = json.dumps(words, ensure_ascii=False).encode()
        (self.directory / "words.json").write_bytes(original)
        expected = {'diacritics': 'tiếng', 'no_diacritics': 'tieng', 'telex': 'tieengs'}
        for mode, target in expected.items():
            result = prepare_manifest(self.manifest, self.base / mode,
                                      cache_dir=self.base / 'cache', vietnamese_text_mode=mode)
            row = json.loads(result.read_text())
            sidecar = json.loads(Path(row['path']).with_suffix('.json').read_text())
            self.assertEqual(sidecar['alignments'][0], [target, [.1, .4], 'SPEAKER_BROKER'])
            self.assertEqual(sidecar['alignments'][1], ['Đặng', [.5, .8], 'SPEAKER_CLIENT'])
            self.assertEqual(sidecar['text_prompt'], 'Be helpful.')
            self.assertEqual(sidecar['vietnamese_text_mode'], mode)
            self.assertEqual(row['vietnamese_text_mode'], mode)
        self.assertEqual((self.directory / 'words.json').read_bytes(), original)
        with self.assertRaises(ValueError):
            prepare_manifest(self.manifest, self.base / 'bad', vietnamese_text_mode='invalid')
        self.assertFalse((self.base / 'bad').exists())

    def test_acceptance_export_threads_mode_into_native_worker_fixture(self):
        import yaml
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from _common import prepare_fixture
        (self.directory / 'words.json').write_text(json.dumps([
            {'word':'tiếng','start':.1,'end':.4,'speaker':'agent'}], ensure_ascii=False))
        config = self.base / 'acceptance.yaml'
        config.write_text(yaml.safe_dump({'acceptance':{'model_root':str(self.base),
            'prepared_manifest':str(self.manifest)},'data':{'vietnamese_text_mode':'telex'},
            'lora':{'enable':True},'max_steps':2}))
        assets = SimpleNamespace(moshi_weights=self.base / 'model', mimi_weights=self.base / 'mimi',
                                 tokenizer=self.base / 'tokenizer')
        with patch('tim_compat.local_checkpoint.LocalAssets.resolve', return_value=assets):
            result = prepare_fixture(config, self.base / 'acceptance')
        resolved = yaml.safe_load(result.read_text())
        self.assertNotIn('vietnamese_text_mode', resolved['data'])
        row = json.loads(Path(resolved['data']['train_data']).read_text())
        self.assertEqual(row['vietnamese_text_mode'], 'telex')
        sidecar = json.loads(Path(row['path']).with_suffix('.json').read_text())
        self.assertEqual(sidecar['alignments'][0][0], 'tieengs')
        fixture = json.loads((result.parent / 'fixture.json').read_text())
        self.assertEqual(fixture['vietnamese_text_mode'], 'telex')

    def test_prepare_cli_reads_mode_and_emits_train_config(self):
        import subprocess
        import yaml
        from tim_compat.training_config import native_training_config
        config = self.base / 'train.yaml'
        config.write_text(yaml.safe_dump({'data':{'vietnamese_text_mode':'telex',
            'train_data':'/unused.jsonl','eval_data':''}, 'run_dir':'/unused-run'}))
        before = config.read_bytes()
        output = self.base / 'cli-export'
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(project / 'prepare_data.py'),
            '--manifest',str(self.manifest),'--output',str(output),'--config',str(config)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        derived = output / 'train.yaml'
        self.assertEqual(yaml.safe_load(derived.read_text())['data']['vietnamese_text_mode'], 'telex')
        with native_training_config(derived) as native:
            self.assertNotIn('vietnamese_text_mode', yaml.safe_load(native.read_text())['data'])
        self.assertEqual(config.read_bytes(), before)

    def test_out_of_bounds_quarantine_preserves_conversation_and_cache(self):
        words = [{'word':'tiếng','start':.1,'end':.4,'speaker':'agent'},
                 {'word':'mà','start':.8,'end':1.2,'speaker':'user'}]
        source = json.dumps(words, ensure_ascii=False).encode()
        (self.directory / 'words.json').write_bytes(source)
        exports = []
        for index, workers in enumerate((0, 2, 0)):
            result = prepare_manifest(self.manifest, self.base / f'quarantine-{index}',
                workers=workers, cache_dir=(self.base / 'cache') if workers == 0 else None,
                vietnamese_text_mode='telex')
            row = json.loads(result.read_text())
            sidecar = json.loads(Path(row['path']).with_suffix('.json').read_text())
            self.assertEqual(sidecar['alignments'], [['tieengs',[.1,.4],'SPEAKER_BROKER']])
            self.assertEqual(sidecar['source_alignment_errors'][0]['word'], words[1])
            self.assertEqual(sidecar['audio_duration_sec'], 1)
            report = json.loads((result.parent / 'source_rejections.jsonl').read_text())
            self.assertEqual(report['sample_id'], 'one')
            exports.append(sidecar)
        self.assertEqual(exports[0], exports[1]); self.assertEqual(exports[1], exports[2])
        self.assertEqual((self.directory / 'words.json').read_bytes(), source)

    def test_parity_preflight_exports_tail_error_without_changing_selected_id(self):
        import yaml
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from check_interleaver_parity import prepare_parity_fixture
        (self.directory / 'words.json').write_text(json.dumps([
            {'word':'good','start':.1,'end':.4,'speaker':'agent'},
            {'word':'mà','start':.8,'end':1.0169,'speaker':'user'}]))
        config = self.base / 'parity.yaml'
        config.write_text(yaml.safe_dump({'acceptance':{'model_root':str(self.base),
            'prepared_manifest':str(self.manifest)},'lora':{'enable':True},'max_steps':2}))
        assets = SimpleNamespace(moshi_weights=self.base / 'model', mimi_weights=self.base / 'mimi',
                                 tokenizer=self.base / 'tokenizer')
        with patch('tim_compat.local_checkpoint.LocalAssets.resolve', return_value=assets):
            resolved = prepare_parity_fixture(config, self.base / 'parity', 0)
        fixture = json.loads((resolved.parent / 'fixture.json').read_text())
        self.assertEqual(fixture['samples'][0]['sample_id'], 'one')
        selected = json.loads((resolved.parent / 'selected.jsonl').read_text())
        self.assertEqual(Path(selected['path']).stem, 'one')
        self.assertEqual(json.loads((resolved.parent / 'prepared/source_rejections.jsonl').read_text())['reason'],
                         'timestamp_out_of_bounds')

    def test_disabled_returns_reference_manifest_untouched(self):
        original = self.base / "not-required-to-exist.jsonl"
        self.assertEqual(prepare_manifest(original, enabled=False), original)
        self.assertFalse(original.exists())

    def test_disabled_preserves_real_tim_manifest_bytes(self):
        original = self.base / "reference.jsonl"
        content = b'{"path": "/reference.wav", "duration": 1.0}\n'
        original.write_bytes(content)
        actual = prepare_manifest(original, self.base / "unused", enabled=False)
        self.assertEqual(actual.read_bytes(), content)
        self.assertFalse((self.base / "unused").exists())

    def test_enabled_sidecars_and_reference_interleaver_parity(self):
        before = (self.directory / "words.json").read_bytes()
        result = prepare_manifest(self.manifest, self.base / "tim")
        record = json.loads(result.read_text())
        sidecar = json.loads(Path(record["path"]).with_suffix(".json").read_text())
        expected = [["hello", [0.1, 0.4], "SPEAKER_BROKER"],
                    ["yes", [0.5, 0.8], "SPEAKER_CLIENT"]]
        self.assertEqual(sidecar["alignments"], expected)
        self.assertEqual(record["duration"], 1.0)
        self.assertEqual(sidecar["text_prompt"], "Be helpful.")
        self.assertEqual(Path(record["path"]).read_bytes(), (self.directory / "conversation.wav").read_bytes())
        self.assertEqual((self.directory / "words.json").read_bytes(), before)
        # Execute the actual frozen Tim Interleaver, not a port of its algorithm.
        path = Path(__file__).resolve().parents[1] / "moshi-finetune/finetune/data/interleaver.py"
        spec = importlib.util.spec_from_file_location("tim_parity_interleaver", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        class Tokenizer:
            def encode(self, lines):
                return [[10 + len(word) for word in line.split()] for line in lines]

            def piece_to_id(self, piece):
                return 9

            def bos_id(self):
                return 1

            def eos_id(self):
                return 2

        interleaver = module.Interleaver(Tokenizer(), 12.5, 3, 4, -1,
                                         keep_main_only=True, device="cpu")
        reference = interleaver.prepare_item(expected, 1.0)
        adapted = interleaver.prepare_item(sidecar["alignments"], 1.0)
        self.assertTrue(torch.equal(reference, adapted))

    def test_text_prompt_left_precedence_and_legacy_fallback(self):
        metadata_path = self.directory / "metadata.json"
        for index, (fields, expected) in enumerate((
            ({"text_prompt_left": "Agent prompt."}, "Agent prompt."),
            ({"text_prompt_left": "Agent prompt.", "text_prompt": "Legacy."}, "Agent prompt."),
            ({"text_prompt": "Legacy."}, "Legacy."),
            ({}, "File prompt."),
        )):
            with self.subTest(fields=fields):
                metadata_path.write_text(json.dumps({
                    "agent_channel": "left", "user_channel": "right", **fields}))
                if not fields:
                    (self.directory / "prompt.txt").write_text(expected)
                before = metadata_path.read_bytes()
                result = prepare_manifest(self.manifest, self.base / f"prompt-{index}")
                record = json.loads(result.read_text())
                sidecar = json.loads(Path(record["path"]).with_suffix(".json").read_text())
                self.assertEqual(sidecar["text_prompt"], expected)
                self.assertEqual(metadata_path.read_bytes(), before)

    def test_invalid_text_prompt_left_does_not_use_legacy_prompt(self):
        for index, prompt in enumerate(("", "   ", 123)):
            with self.subTest(prompt=prompt):
                (self.directory / "metadata.json").write_text(json.dumps({
                    "agent_channel": "left", "user_channel": "right",
                    "text_prompt_left": prompt, "text_prompt": "Legacy."}))
                output = self.base / f"invalid-prompt-{index}"
                with self.assertRaisesRegex(ValueError, "missing prepared text prompt"):
                    prepare_manifest(self.manifest, output)
                self.assertFalse(output.exists())

    def test_legacy_voice_name_and_manifest_order(self):
        (self.directory / "voice_prompt_left.wav").rename(self.directory / "voice_prompt.wav")
        second = {"sample_id": "two", "sample_dir": "samples/one"}
        first = {"sample_id": "one", "sample_dir": "samples/one"}
        self.manifest.write_text(json.dumps(second) + "\n" + json.dumps(first) + "\n")
        result = prepare_manifest(self.manifest, self.base / "ordered")
        rows = [json.loads(line) for line in result.read_text().splitlines()]
        self.assertEqual([Path(row["path"]).stem for row in rows], ["two", "one"])
        for row in rows:
            sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
            self.assertEqual(Path(sidecar["voice_prompt"]).name, "voice_prompt.wav")

    def test_invalid_inputs_leave_no_output(self):
        for mutate in (lambda words: words.reverse(),
                       lambda words: words[0].update(end=float('inf')),
                       lambda words: words[0].update(speaker="unknown")):
            words = json.loads(json.dumps(self.words))
            mutate(words)
            (self.directory / "words.json").write_text(json.dumps(words))
            with self.assertRaises(ValueError):
                prepare_manifest(self.manifest, self.base / "invalid")
            self.assertFalse((self.base / "invalid").exists())

    def test_root_escape_and_duplicate_rejected(self):
        row = {"sample_id": "one", "audio_path": "../escape.wav"}
        self.manifest.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "escapes root"):
            prepare_manifest(self.manifest, self.base / "escape")
        row = {"sample_id": "one", "sample_dir": "samples/one"}
        self.manifest.write_text((json.dumps(row) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_manifest(self.manifest, self.base / "duplicate")