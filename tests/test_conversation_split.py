import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import yaml

from tim_compat.prepared_data import prepare_manifest
from tim_compat.training_config import native_training_config


class ConversationSplitTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        source = self.root / 'prepared'
        source.mkdir()
        rows = []
        for index in range(10):
            sample_id = f'conversation_{index}'
            directory = source / sample_id
            directory.mkdir()
            for name, channels in [('conversation.wav', 2), ('voice_prompt.wav', 1)]:
                with wave.open(str(directory / name), 'wb') as audio:
                    audio.setnchannels(channels)
                    audio.setsampwidth(2)
                    audio.setframerate(24000)
                    audio.writeframes(b'\0\0' * channels * 24000)
            (directory / 'metadata.json').write_text(json.dumps({
                'agent_channel': 'left', 'user_channel': 'right', 'text_prompt': 'Hello'}))
            (directory / 'words.json').write_text(json.dumps([
                {'word': 'hello', 'start': .1, 'end': .4, 'speaker': 'agent'}]))
            rows.append({'sample_id': sample_id, 'sample_dir': sample_id})
        self.manifest = source / 'train.jsonl'
        self.manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_split_is_disjoint_complete_and_independent_of_manifest_order_and_cache(self):
        memberships = []
        for index in range(2):
            output = self.root / f'export_{index}'
            prepare_manifest(self.manifest, output, eval_split_from_train=.2,
                             seed=7, cache_dir=self.root / 'cache')
            split = json.loads((output / 'split.json').read_text())
            train, val = set(split['train_sample_ids']), set(split['val_sample_ids'])
            self.assertEqual((len(train), len(val)), (8, 2))
            self.assertFalse(train & val)
            self.assertEqual(len(train | val), 10)
            train_audio = {Path(json.loads(line)['path']).resolve()
                           for line in (output / 'train.jsonl').read_text().splitlines()}
            val_audio = {Path(json.loads(line)['path']).resolve()
                         for line in (output / 'val.jsonl').read_text().splitlines()}
            self.assertFalse(train_audio & val_audio)
            memberships.append((train, val))
        self.assertEqual(memberships[0], memberships[1])
        self.manifest.write_text('\n'.join(reversed(self.manifest.read_text().splitlines())) + '\n')
        output = self.root / 'reordered'
        prepare_manifest(self.manifest, output, eval_split_from_train=.2, seed=7)
        split = json.loads((output / 'split.json').read_text())
        self.assertEqual(set(split['val_sample_ids']), memberships[0][1])

    def test_invalid_ratio_small_dataset_and_duplicate_audio_fail_without_output(self):
        for ratio in (-.1, 1, True, '0.2', float('nan')):
            with self.assertRaises(ValueError):
                prepare_manifest(self.manifest, self.root / 'bad', eval_split_from_train=ratio)
            self.assertFalse((self.root / 'bad').exists())
        rows = self.manifest.read_text().splitlines()
        self.manifest.write_text(rows[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'two conversations'):
            prepare_manifest(self.manifest, self.root / 'small', eval_split_from_train=.1)
        self.assertFalse((self.root / 'small').exists())
        alias = json.loads(rows[0])
        alias['sample_id'] = 'alias'
        self.manifest.write_text(rows[0] + '\n' + json.dumps(alias) + '\n')
        with self.assertRaisesRegex(ValueError, 'same conversation audio'):
            prepare_manifest(self.manifest, self.root / 'duplicate', eval_split_from_train=.1)
        self.assertFalse((self.root / 'duplicate').exists())

    def test_cli_binds_both_manifests_and_native_loader_consumes_split_key(self):
        project = Path(__file__).resolve().parents[1]
        config = self.root / 'config.yaml'
        values = {'data': {'eval_split_from_train': .2, 'train_data': '', 'eval_data': ''},
                  'seed': 7, 'do_eval': True, 'eval_freq': 64}
        config.write_text(yaml.safe_dump(values))
        before = config.read_bytes()
        with self.assertRaisesRegex(ValueError, 'prepare_data.py'):
            with native_training_config(config):
                pass
        output = self.root / 'cli'
        result = subprocess.run([sys.executable, str(project / 'prepare_data.py'),
            '--manifest', str(self.manifest), '--output', str(output), '--config', str(config)],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        derived = yaml.safe_load((output / 'train.yaml').read_text())
        self.assertEqual(derived['data']['eval_data'], str(output / 'val.jsonl'))
        self.assertEqual(derived['data']['train_data'], str(output / 'train.jsonl'))
        self.assertEqual(config.read_bytes(), before)
        with native_training_config(output / 'train.yaml') as native:
            self.assertNotIn('eval_split_from_train', yaml.safe_load(native.read_text())['data'])
        values['data']['eval_data'] = '/existing/val.jsonl'
        config.write_text(yaml.safe_dump(values))
        result = subprocess.run([sys.executable, str(project / 'prepare_data.py'),
            '--manifest', str(self.manifest), '--output', str(self.root / 'conflict'),
            '--config', str(config)], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'conflict').exists())

    def test_acceptance_consumes_zero_and_rejects_holdout_split(self):
        project = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project / 'scripts'))
        from _common import read_config
        values = {'acceptance': {'model_root': '/models', 'prepared_manifest': str(self.manifest)},
                  'data': {'eval_split_from_train': 0}}
        config = self.root / 'acceptance.yaml'
        config.write_text(yaml.safe_dump(values))
        native, _ = read_config(config)
        self.assertNotIn('eval_split_from_train', native['data'])
        values['data']['eval_split_from_train'] = .2
        config.write_text(yaml.safe_dump(values))
        with self.assertRaisesRegex(ValueError, 'fixed-set'):
            read_config(config)

    def test_zero_preserves_unsplit_export(self):
        output = self.root / 'unsplit'
        result = prepare_manifest(self.manifest, output, eval_split_from_train=0)
        self.assertEqual(len(result.read_text().splitlines()), 10)
        self.assertFalse((output / 'val.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
