import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ServerLauncherTest(unittest.TestCase):
    def test_commands_and_fail_fast_without_gpu(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_synthetic_server.sh'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / 'python'
            calls = root / 'calls.jsonl'
            fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1].endswith('prepare_data.py'):
    if os.environ.get('FAIL_PREPARE'):
        sys.exit(7)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir()
    (output / 'train.yaml').write_text('data: {}')
if sys.argv[1:3] == ['-m', 'torch.distributed.run']:
    sys.exit(int(os.environ.get('TRAIN_EXIT', '0')))
''')
            fake.chmod(0o755)
            env = dict(os.environ, PYTHON=str(fake), CALLS=str(calls),
                       EXPORT_DIR=str(root / 'export'), TRAIN_EXIT='9',
                       RESUME_FROM='/checkpoints/64')
            result = subprocess.run(['bash', str(script), 'all'], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 9, result.stderr)
            commands = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            self.assertTrue(commands[0][0].endswith('prepare_data.py'))
            train = commands[-1]
            self.assertEqual(train[:2], ['-m', 'torch.distributed.run'])
            self.assertIn('--nproc-per-node=1', train)
            self.assertEqual(train[train.index('--config') + 1], str(root / 'export/train.yaml'))
            self.assertIn('--free-running-config', train)
            self.assertIn('--token-cache', train)
            self.assertEqual(train[train.index('--resume-from') + 1], '/checkpoints/64')
            calls.write_text('')
            env['FAIL_PREPARE'] = '1'
            result = subprocess.run(['bash', str(script), 'all'], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(calls.read_text().splitlines()), 1)

    def test_disabled_inference_preflight_uses_real_bridge(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_synthetic_server.sh'
        text = script.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'disabled.yaml'
            config.write_text('acceptance:\n  inference:\n    free_running_every_steps: 0\n')
            import sys
            result = subprocess.run([sys.executable, '-', str(config)], input=text,
                                    cwd=script.parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
