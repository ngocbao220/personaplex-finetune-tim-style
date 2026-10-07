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
                       lambda words: words[0].update(end=2.0),
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