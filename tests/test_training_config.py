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