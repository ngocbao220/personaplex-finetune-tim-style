"""CPU harness checks, explicitly not evidence of GPU numerical parity."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _common import compare_tensors


class GPUAcceptanceScriptsTest(unittest.TestCase):
    def test_cli_help_does_not_require_gpu(self):
        for name in ('check_interleaver_parity.py', 'check_one_step_parity.py',
                     'run_gpu_acceptance.py', 'run_free_running_inference.py', '_tim_worker.py', '_run_all.py'):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), '--help'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
    def test_tensor_comparison_detects_update_and_missing_keys(self):
        import torch
        a = {'lora_A': torch.tensor([1.]), 'lora_B': torch.tensor([0.])}
        b = {'lora_A': torch.tensor([1.]), 'lora_B': torch.tensor([1e-7])}
        result = compare_tensors(a,b)
        self.assertTrue(result['lora_A']['pass_'])
        self.assertFalse(result['lora_B']['pass_'])
        self.assertFalse(compare_tensors(a, {})['lora_A']['pass_'])

    def test_reload_parity_tolerates_bf16_evaluation_and_filters_frozen_lora(self):
        before = dict(total=0.5552079677581787, text=0.0003757586528081447, audio=0.5548322200775146)
        after = dict(total=0.5559874773025513, text=0.00037130602868273854, audio=0.5556161999702454)
        loss_atol = 1e-3
        loss_rtol = 5e-3
        for key in ('total', 'text', 'audio'):
            diff = abs(before[key] - after[key])
            tol = loss_atol + loss_rtol * abs(before[key])
            self.assertLessEqual(diff, tol, f'{key} difference {diff} exceeds tolerance {tol}')

        import torch
        pre = {
            'model.layers.0.self_attn.q_proj.lora_A': torch.tensor([1.0]),
            'model.depformer.layers.0.self_attn.q_proj.lora_A': torch.tensor([2.0]),
        }
        post = {
            'model.layers.0.self_attn.q_proj.lora_A': torch.tensor([1.0]),
            'model.depformer.layers.0.self_attn.q_proj.lora_A': torch.tensor([9.9]),  # frozen, untracked
        }
        trainable_keys = {'model.layers.0.self_attn.q_proj.lora_A'}
        trainable_pre = {k: v for k, v in pre.items() if k in trainable_keys}
        trainable_post = {k: v for k, v in post.items() if k in trainable_keys}
        diffs = compare_tensors(trainable_pre, trainable_post)
        self.assertTrue(all(v['pass_'] for v in diffs.values()))

    def test_harness_uses_real_trainer_not_custom_update(self):
        worker = (ROOT / 'scripts/_tim_worker.py').read_text()
        self.assertIn('module.train(str(opts.config))', worker)
        self.assertIn('train_local.main', worker)
        self.assertIn('evaluate(model,', worker)
        self.assertNotIn('total.backward()', worker)
        self.assertNotIn('optimizer.step()', worker)
        self.assertNotIn('torch.optim.AdamW(', worker)
        source = (ROOT / 'moshi-finetune/train.py').read_text()
        for anchor in ('# Host sync', 'if args.do_ckpt and (',
                       'torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)'):
            self.assertEqual(sum(line.strip() == anchor for line in source.splitlines()), 1)

    def test_free_running_delegates_to_original_source(self):
        source = (ROOT / 'scripts/run_free_running_inference.py').read_text()
        self.assertIn('native.run_inference(', source)
        self.assertIn('merge(checkpoint, merged, base_weights=', source)
        self.assertNotIn('lm_gen.step(', source)
        self.assertNotIn('lm_kwargs[', source)
        self.assertIn("if 'WARNING:' in merge_log", source)

    def test_free_running_local_file_validation(self):
        from run_free_running_inference import require_file
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            with self.assertRaises(FileNotFoundError):
                require_file(path)
            path.touch()
            with self.assertRaises(FileNotFoundError):
                require_file(path)
            path.write_bytes(b'local')
            self.assertEqual(require_file(path), path.resolve())

    def test_dialogue_exports_preserve_channel_contract(self):
        import numpy as np
        from run_free_running_inference import dialogue_arrays, text_from_pieces
        user = np.array([[.1, .2, .3]])
        original = np.concatenate((user * 0, user))
        base, current = dialogue_arrays(original, user, user * 2, user * 3)
        np.testing.assert_array_equal(base[0], user[0] * 2)
        np.testing.assert_array_equal(current[0], user[0] * 3)
        np.testing.assert_array_equal(base[1], user[0])
        np.testing.assert_array_equal(current[1], user[0])
        with self.assertRaises(ValueError):
            dialogue_arrays(original, user * 2, user, user)
        with self.assertRaises(ValueError):
            dialogue_arrays(original, user, user[:, :2], user)
        with self.assertRaises(ValueError):
            dialogue_arrays(original, user, user * float('nan'), user)
        left_original = original[::-1]
        left_base, left_current = dialogue_arrays(left_original, user, user * 2, user * 3, 'left')
        np.testing.assert_array_equal(left_base[0], user[0])
        np.testing.assert_array_equal(left_current[1], user[0] * 3)
        self.assertEqual(text_from_pieces(['BOS', ' Xin', ' chào', 'PAD', 'EOS']), 'Xin chào')


if __name__ == '__main__':
    unittest.main()