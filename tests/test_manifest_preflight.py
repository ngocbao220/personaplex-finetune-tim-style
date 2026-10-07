import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tim_compat import training_config


class ManifestPreflightTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / 'train.jsonl'
        self.config = self.root / 'train.yaml'

    def write_config(self, train, evaluation=''):
        self.config.write_text(yaml.safe_dump({'data': {
            'train_data': str(train), 'eval_data': str(evaluation)}}))

    def test_launcher_rejects_prepared_manifest_before_gpu_import(self):
        import train_local
        self.manifest.write_text('{"sample_id":"one","sample_dir":"one"}\n')
        self.write_config(self.manifest)
        with patch.object(train_local.LocalAssets, 'resolve'):
            with self.assertRaisesRegex(ValueError, r'train_data.*train.jsonl:1.*path.*prepare_data.py'):
                train_local.main(['--config', str(self.config), '--model-root', str(self.root)])

    def test_weighted_sources_and_nested_eval_directory(self):
        self.manifest.write_text('{"path":"one.wav","duration":1}\n')
        directory = self.root / 'eval'
        directory.mkdir()
        evaluation = directory / 'eval.jsonl'
        evaluation.write_bytes(self.manifest.read_bytes())
        self.write_config(f'{self.manifest}:0.5,{evaluation}:1', directory)
        before = self.config.read_bytes()
        training_config.validate_training_sources(self.config)
        self.assertEqual(self.config.read_bytes(), before)
        evaluation.write_text('{"sample_id":"wrong"}\n')
        with self.assertRaisesRegex(ValueError, r'eval_data.*eval.jsonl:1'):
            training_config.validate_training_sources(self.config)

    def test_bad_records_include_filename_and_line(self):
        for row in ({'path': 'one.wav', 'duration': 0},
                    {'path': 'one.wav', 'duration': True},
                    {'path': 'one.wav', 'duration': float('nan')}, [], None):
            with self.subTest(row=row):
                self.manifest.write_text('{"path":"one.wav","duration":1}\n' + json.dumps(row) + '\n')
                self.write_config(self.manifest)
                with self.assertRaisesRegex(ValueError, r'train.jsonl:2'):
                    training_config.validate_training_sources(self.config)

    def test_empty_or_malformed_manifest(self):
        self.write_config(self.manifest)
        for content in ('', '\n', '{broken}\n'):
            self.manifest.write_text(content)
            with self.assertRaises(ValueError):
                training_config.validate_training_sources(self.config)


if __name__ == '__main__':
    unittest.main()
