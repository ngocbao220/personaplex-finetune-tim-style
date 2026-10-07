import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tim_compat.training_config import bind_training_manifest


class TrainingConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / 'original.yaml'
        self.values = {
            'data': {'train_data': '/original.jsonl', 'eval_data': '', 'shuffle': False},
            'system_prompt': {'enable': True, 'audio_silence_frames': 6},
            'lora': {'enable': True, 'rank': 64}, 'seed': 0,
            'first_codebook_weight_multiplier': 100., 'text_padding_weight': .5,
            'run_dir': '/run', 'duration_sec': 10,
        }
        self.config.write_text(yaml.safe_dump(self.values))
        self.manifest = self.root / 'tim.jsonl'
        self.manifest.write_text(json.dumps({'path': '/export/one.wav', 'duration': 1.0}) + '\n')
        self.output = self.root / 'derived.yaml'

    def test_native_config_consumes_mode_and_rejects_wrong_export(self):
        from tim_compat.training_config import native_training_config
        self.values['data']['vietnamese_text_mode'] = 'telex'
        self.values['data']['train_data'] = str(self.manifest)
        self.config.write_text(yaml.safe_dump(self.values))
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ValueError, 'mode'):
            with native_training_config(self.config):
                pass
        wav = self.root / 'one.wav'
        wav.with_suffix('.json').write_text(json.dumps({'vietnamese_text_mode':'telex'}))
        self.manifest.write_text(json.dumps({'path':str(wav),'duration':1.,'vietnamese_text_mode':'telex'})+'\n')
        with native_training_config(self.config) as native:
            normalized = yaml.safe_load(native.read_text())
            self.assertNotIn('vietnamese_text_mode', normalized['data'])
            self.assertEqual(normalized['data']['train_data'], str(self.manifest))
            self.assertTrue(native.exists())
        self.assertFalse(native.exists())
        self.assertEqual(before, self.config.read_bytes())
        self.values['data'].pop('vietnamese_text_mode')
        self.config.write_text(yaml.safe_dump(self.values))
        with native_training_config(self.config, vietnamese_text_mode='telex') as native:
            self.assertNotIn('vietnamese_text_mode', yaml.safe_load(native.read_text())['data'])

    def test_checkpoint_records_mode_without_leaking_patch(self):
        from tim_compat.training_config import checkpoint_text_mode
        class Checkpointer:
            def __init__(self, config):
                self.config = config
        original = Checkpointer.__init__
        raw = {'lora_rank':64}
        with checkpoint_text_mode(Checkpointer, 'telex'):
            checkpoint = Checkpointer(raw)
            self.assertEqual(checkpoint.config['vietnamese_text_mode'], 'telex')
            self.assertNotIn('vietnamese_text_mode', raw)
        self.assertIs(Checkpointer.__init__, original)

    def test_disabled_returns_original_path_without_reading_anything(self):
        missing = self.root / 'missing.yaml'
        self.assertEqual(bind_training_manifest(missing, None, None, enabled=False), missing)
        self.assertFalse(self.output.exists())

    def test_only_train_manifest_changes_source_bytes_preserved(self):
        before = self.config.read_bytes()
        expected = copy.deepcopy(self.values)
        expected['data']['train_data'] = str(self.manifest.resolve())
        actual = bind_training_manifest(self.config, self.manifest, self.output)
        self.assertEqual(yaml.safe_load(actual.read_text()), expected)
        self.assertEqual(self.config.read_bytes(), before)

    def test_disabled_real_config_preserves_reference_content(self):
        before = self.config.read_bytes()
        actual = bind_training_manifest(self.config, self.manifest, self.output, enabled=False)
        self.assertEqual(actual, self.config)
        self.assertEqual(actual.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_no_overwrite_even_original_or_manifest(self):
        for output in (self.config, self.manifest):
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                bind_training_manifest(self.config, self.manifest, output)
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_manifest_fails_before_config_write(self):
        for text in ('', '{"sample_id":"prepared-not-tim"}\n',
                     '{"path":"one.wav","duration":0}\n',
                     '{"path":"one.wav","duration":NaN}\n'):
            self.manifest.write_text(text)
            with self.assertRaises(ValueError):
                bind_training_manifest(self.config, self.manifest, self.output)
            self.assertFalse(self.output.exists())

    def test_prompt_policy_is_not_silently_enabled(self):
        self.values['system_prompt']['enable'] = False
        self.config.write_text(yaml.safe_dump(self.values))
        actual = bind_training_manifest(self.config, self.manifest, self.output)
        self.assertFalse(yaml.safe_load(actual.read_text())['system_prompt']['enable'])

    def test_invalid_config_mapping_fails_without_output(self):
        for content in ('[]', 'data: null', 'data: []'):
            self.config.write_text(content)
            with self.assertRaises(ValueError):
                bind_training_manifest(self.config, self.manifest, self.output)
            self.assertFalse(self.output.exists())