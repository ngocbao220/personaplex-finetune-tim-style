"""Fixed smoke data identity and bounded coverage, without CUDA/model execution."""
import itertools
import ast
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from typing import Any, Iterator
from contextlib import nullcontext, redirect_stdout
from unittest.mock import patch

import torch
import numpy as np

from tim_compat.sample_filter import FilterPolicy
from tim_compat.smoke_subset import save_smoke_samples, load_smoke_samples, replay_samples, smoke_dataset_loader, prepare_smoke_samples


class SmokeSubsetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.policy = FilterPolicy(100, 2048, (0, 1, 2), require_agent_text=False)

    def sample(self, index):
        codes = torch.zeros(1, 17, 5, dtype=torch.long)
        codes[0, 0, 2] = 10 + index % 80
        return SimpleNamespace(codes=codes, prompt_length=1, context_mask=None,
            condition_attributes=None, injection_stats=None,
            provenance=dict(path='/export/one.wav', start_sec=index * 10,
                            chunk_id=f'chunk-{index}', wav_shape=[2, 240000]))

    def test_900_second_source_stops_at_eight_valid_chunks_without_peeking(self):
        seen = []
        def source():
            for index in range(90):
                seen.append(index)
                yield self.sample(index)
        selection = save_smoke_samples(source(), self.policy, 8, self.output)
        self.assertEqual(seen, list(range(8)))
        self.assertEqual(selection['num_chunks'], 8)
        self.assertEqual(selection['scope'], 'fixed_chunk_smoke')
        self.assertEqual([r['start_sec'] for r in selection['chunks']], list(range(0, 80, 10)))
        loaded, restored = load_smoke_samples(self.output)
        self.assertEqual(restored, selection)
        self.assertEqual(len(loaded), 8)

    def test_rejections_recorded_and_valid_chunks_keep_native_order(self):
        samples = [self.sample(i) for i in range(4)]
        samples[0].codes[0, 0, 2] = 100
        samples[2].validation = dict(bounds=[], prompt_frames_dropped=0,
            context_tokens_dropped=0, dialogue_lexical_dropped=1, dialogue=[])
        selection = save_smoke_samples(iter(samples), self.policy, 2, self.output)
        self.assertEqual([r['candidate_index'] for r in selection['chunks']], [1, 3])
        self.assertEqual(selection['chunks_rejected'], 2)
        rows = [json.loads(line) for line in (self.output / 'smoke_rejections.jsonl').read_text().splitlines()]
        self.assertEqual([r['reason'] for r in rows], ['text_token_out_of_bounds', 'text_overflow'])

    def test_partial_and_empty_sets_are_explicit(self):
        selection = save_smoke_samples(iter([self.sample(0)]), self.policy, 8, self.output)
        self.assertEqual(selection['requested_chunks'], 8)
        self.assertEqual(selection['num_chunks'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'no valid smoke chunk'):
                save_smoke_samples(iter([]), self.policy, 8, output)
            self.assertFalse(json.loads((output / 'smoke_selection.json').read_text())['pass_'])

    def test_count_validation_no_overwrite_and_corruption(self):
        for count in (0, -1, 41, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    save_smoke_samples(iter([]), self.policy, count, self.output)
        save_smoke_samples(iter([self.sample(0)]), self.policy, 1, self.output)
        before = (self.output / 'smoke_samples.pt').read_bytes()
        with self.assertRaises(FileExistsError):
            save_smoke_samples(iter([self.sample(1)]), self.policy, 1, self.output)
        self.assertEqual((self.output / 'smoke_samples.pt').read_bytes(), before)
        (self.output / 'smoke_samples.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_smoke_samples(self.output)

    def test_train_cycles_eval_is_finite_and_replays_do_not_mutate_snapshot(self):
        samples = [self.sample(0), self.sample(1)]
        train = replay_samples(samples, device='cpu', is_eval=False)
        first_five = list(itertools.islice(train, 5))
        self.assertEqual([s.provenance['chunk_id'] for s in first_five],
                         ['chunk-0', 'chunk-1', 'chunk-0', 'chunk-1', 'chunk-0'])
        first_five[0].codes.fill_(99)
        self.assertEqual(samples[0].codes[0, 0, 2].item(), 10)
        batches = list(replay_samples(samples, device='cpu', is_eval=True))
        self.assertEqual(len(batches), 2)
        self.assertTrue(torch.equal(batches[0].codes, samples[0].codes))

    def test_loader_bridge_restores_binding_even_when_train_fails(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        import _common  # adds reference package path, no GPU import
        tokenizer = SimpleNamespace(interleaver=SimpleNamespace(device='cpu'))
        samples = [self.sample(0)]
        original = lambda **kwargs: iter(samples)
        loader = SimpleNamespace(build_dataset=original)
        save_smoke_samples(iter(samples), self.policy, 1, self.output)
        with self.assertRaisesRegex(RuntimeError, 'train failed'):
            with smoke_dataset_loader(loader, samples, tokenizer):
                eval_samples = list(loader.build_dataset(is_eval=True, rank=0, world_size=1))
                self.assertEqual(len(eval_samples), 1)
                train = loader.build_dataset(is_eval=False, rank=0, world_size=1)
                self.assertEqual(len(list(itertools.islice(train, 3))), 3)
                raise RuntimeError('train failed')
        self.assertIs(loader.build_dataset, original)

    def test_prepare_snapshot_and_native_batch_loader_use_same_samples(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        import _common
        from test_interleaver_filter import CPUChunkTokenizer
        tokenizer = CPUChunkTokenizer([([10], (0, 1), 'agent')])
        # Execute the exact native batching function without importing the GPU
        # trainer's unrelated simple_parsing dependency in this CPU environment.
        source = root / 'moshi-finetune/finetune/data/data_loader.py'
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'build_data_loader')
        data_loader = ModuleType('smoke_cpu_native_loader')
        data_loader.__dict__.update(Any=Any, Iterator=Iterator, DataArgs=SimpleNamespace,
            Batch=sys.modules[type(tokenizer.interleaver).__module__].Batch, build_dataset=None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), data_loader.__dict__)
        tokenizer.mimi = SimpleNamespace(sample_rate=4)
        path = self.output / 'one.wav'
        path.with_suffix('.json').write_text('{}')
        config = self.output / 'config.yaml'
        config.write_text('seed: 0\n')
        model = SimpleNamespace(text_card=100, card=2048, zero_token_id=-1,
            text_initial_token_id=100, initial_token_id=2048)
        data_args = SimpleNamespace(train_data='long-conversation', eval_data='same', shuffle=False)
        seen = []
        def native(**kwargs):
            self.assertTrue(kwargs['is_eval'])
            self.assertFalse(kwargs['shuffle_pretrain'])
            for index in range(90):
                seen.append(index)
                yield kwargs['instruct_tokenizer'](np.zeros((2, 8), dtype=np.float32), index * 2, path)
        with patch.object(data_loader, 'build_dataset', native), \
                patch('torch.cuda.current_device', return_value=0), \
                patch('torch.random.fork_rng', return_value=nullcontext()):
            samples, selection = prepare_smoke_samples(data_loader, tokenizer, model, data_args,
                                                       8, self.output, config)
        self.assertEqual(seen, list(range(8)))
        self.assertEqual(selection['chunks'][0]['start_sec'], 0)
        self.assertEqual(selection['chunks'][0]['end_sec'], 2)
        before = [s.codes.clone() for s in samples]
        with smoke_dataset_loader(data_loader, samples, tokenizer):
            train = data_loader.build_data_loader(tokenizer, data_args, 1, 0, 0, 1, False)
            first_ten = list(itertools.islice(train, 10))
            self.assertTrue(torch.equal(first_ten[8].codes, before[0]))
            baseline = list(data_loader.build_data_loader(tokenizer, data_args, 1, None, 0, 1, True))
            final = list(data_loader.build_data_loader(tokenizer, data_args, 1, None, 0, 1, True))
        restored, restored_selection = load_smoke_samples(self.output)
        with smoke_dataset_loader(data_loader, restored, tokenizer):
            reload_batches = list(data_loader.build_data_loader(tokenizer, data_args, 1, None, 0, 1, True))
        self.assertEqual(restored_selection['samples_sha256'], selection['samples_sha256'])
        self.assertEqual(tokenizer.calls, 8)  # no re-tokenization for train/eval/reload
        for index in range(8):
            for batches in (baseline, final, reload_batches):
                self.assertTrue(torch.equal(batches[index].codes, before[index]))
                self.assertEqual(batches[index].prompt_lengths, [1])

    def test_smoke_cli_and_config_precedence_reaches_worker_fixture(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        import _common
        import run_gpu_acceptance
        config = self.output / 'resolved.yaml'
        config.write_text('seed: 0\n')
        for configured, flag, expected, requested_samples in (
                (8, None, 8, None), (8, 0, 0, None),
                (8, 4, 4, None), (0, None, 0, None), (8, None, 8, 3)):
            with self.subTest(configured=configured, flag=flag):
                fixture = self.output / 'fixture.json'
                fixture.write_text(json.dumps(dict(acceptance=dict(smoke_chunks=configured))))
                argv = ['run_gpu_acceptance', '--config', str(config), '--output-dir', str(self.output)]
                if flag is not None:
                    argv += ['--smoke-chunks', str(flag)]
                if requested_samples is not None:
                    argv += ['--num-samples', str(requested_samples)]
                with patch.object(sys, 'argv', argv), \
                        patch.object(_common, 'read_config', return_value=({'batch_size': 1}, {'smoke_chunks': configured})), \
                        patch.object(run_gpu_acceptance, 'prepare_fixture', return_value=config) as prepare, \
                        patch.object(run_gpu_acceptance, 'launch_worker', side_effect=RuntimeError('worker stopped for CPU test')), \
                        redirect_stdout(io.StringIO()):
                    self.assertTrue(run_gpu_acceptance.main())
                expected_samples = requested_samples if requested_samples is not None else (1 if expected else 10)
                self.assertEqual(prepare.call_args.args[2], expected_samples)
                self.assertEqual(json.loads(fixture.read_text())['acceptance']['smoke_chunks'], expected)
                report = json.loads((self.output / 'comparison.json').read_text())
                self.assertEqual(report['coverage'], 'fixed_chunk_smoke' if expected else 'full_selected_conversations')

    def test_outside_manifest_reports_counts_without_creating_run_directory(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        from _common import prepare_fixture
        manifest = self.output / 'manifest.jsonl'
        manifest.write_text(json.dumps(dict(sample_id='one', sample_dir='one')) + '\n')
        config = self.output / 'acceptance.yaml'
        config.write_text(json.dumps(dict(acceptance=dict(
            model_root=str(self.output / 'models'), prepared_manifest=str(manifest)))))
        run = self.output / 'run'
        with self.assertRaisesRegex(ValueError, 'total=1, sample_index=0, num_samples=10'):
            prepare_fixture(config, run, num_samples=10)
        self.assertFalse(run.exists())


if __name__ == '__main__':
    unittest.main()
