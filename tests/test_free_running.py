import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tim_compat.free_running import due, frequency, periodic_free_running, cpu_training_state


class PeriodicFreeRunningTest(unittest.TestCase):
    def test_frequency_and_schedule(self):
        self.assertEqual(frequency({}), 0)
        self.assertEqual([i for i in range(11) if due(i, 3)], [0, 3, 6, 9])
        self.assertFalse(due(0, 0))
        for value in (-1, True, 1.5, '3'):
            with self.assertRaises(ValueError):
                frequency({'free_running_every_steps': value})

    def test_disabled_does_not_read_config_or_change_trace(self):
        previous = sys.gettrace()
        with periodic_free_running(None, '/nonexistent', {}):
            self.assertIs(sys.gettrace(), previous)
        self.assertIs(sys.gettrace(), previous)

    def test_cpu_parameter_optimizer_identity_on_failure(self):
        import torch
        from unittest.mock import patch
        model = torch.nn.Linear(2, 2)
        mimi = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        params = list(model.parameters())
        copies = [p.detach().clone() for p in params]
        with patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                with cpu_training_state(model, optimizer, mimi):
                    raise RuntimeError('child failed')
        for p, saved, group_p in zip(params, copies, optimizer.param_groups[0]['params']):
            self.assertIs(p, group_p)
            self.assertTrue(torch.equal(p, saved))

    def test_source_hook_runs_after_update_and_restores_previous_trace(self):
        import tempfile
        import yaml
        source = '''def train():
    state = State(step=0)
    args = State(max_steps=5)
    loss = State(item=lambda: 0)
    updates = []
    while state.step < args.max_steps:
        state.step += 1
        updates.append(state.step)
        loss_item = loss.item()
    return updates
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trainer.py'
            path.write_text(source)
            config = Path(directory) / 'config.yaml'
            config.write_text(yaml.safe_dump(dict(lora=dict(enable=True), do_ckpt=True, run_dir=directory)))
            namespace = {'State': SimpleNamespace}
            exec(compile(source, str(path), 'exec'), namespace)
            calls, traced = [], []
            old = sys.gettrace()
            def observer(frame, event, arg):
                if frame.f_code.co_filename == str(path):
                    traced.append(event)
                return observer
            def evaluate(f):
                calls.append((f['state'].step, list(f['updates'])))
            try:
                sys.settrace(observer)
                with periodic_free_running(SimpleNamespace(__file__=str(path)), config,
                                           {'free_running_every_steps': 2}, evaluator=evaluate):
                    self.assertEqual(namespace['train'](), [1, 2, 3, 4, 5])
                self.assertIs(sys.gettrace(), observer)
            finally:
                sys.settrace(old)
            self.assertEqual(calls, [(0, []), (2, [1, 2]), (4, [1, 2, 3, 4])])
            self.assertIn('line', traced)
            self.assertIn('return', traced)


if __name__ == '__main__':
    unittest.main()