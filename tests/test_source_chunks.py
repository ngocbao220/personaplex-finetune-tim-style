import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tim_compat.source_chunks import SourceChunkGate, source_chunk_loader


class SourceChunksTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'sample.wav'
        self.path.with_suffix('.json').write_text(json.dumps({'sample_id':'sample',
            'audio_duration_sec':30, 'source_alignment_errors':[
                {'reason':'timestamp_out_of_bounds','word_index':1,
                 'word':{'word':'bad','start':29.5,'end':31}}]}))

    def test_only_affected_windows_rejected_before_tokenizer(self):
        gate = SourceChunkGate()
        self.assertIsNone(gate.rejection(self.path, 0, 10))
        self.assertIsNone(gate.rejection(self.path, 10, 20))
        self.assertEqual(gate.rejection(self.path, 20, 30)['reason'], 'timestamp_out_of_bounds')
        self.assertEqual(gate.rejection(self.path, 29, 29.8)['reason'], 'timestamp_out_of_bounds')

    def test_outside_audio_and_half_open_boundaries(self):
        for start, end, rejected in [(-4,-1,0), (32,33,20), (10,11,10)]:
            self.path.with_suffix('.json').write_text(json.dumps({'audio_duration_sec':30,
                'source_alignment_errors':[{'word':{'start':start,'end':end}}]}))
            gate = SourceChunkGate()
            for window in (0, 10, 20):
                self.assertEqual(gate.rejection(self.path, window, window+10) is not None,
                                 window == rejected)

    def test_native_loader_keeps_iterator_running_and_restores_on_failure(self):
        calls = []
        tokenizer = SimpleNamespace(mimi=SimpleNamespace(sample_rate=1))
        class Tokenizer:
            mimi = tokenizer.mimi
            def __call__(self, wav, start, path):
                calls.append(start); return start
        def native(pretrain_data, instruct_tokenizer, **kwargs):
            for start in (0, 20, 10):
                yield instruct_tokenizer(np.ones((2, 10)), start, str(self.path))
        loader = SimpleNamespace(build_dataset=native)
        report = io.StringIO()
        with source_chunk_loader(loader, report=report):
            actual = list(loader.build_dataset(pretrain_data='manifest', instruct_tokenizer=Tokenizer()))
            self.assertEqual(actual, [0, 10]); self.assertEqual(calls, [0, 10])
        self.assertIs(loader.build_dataset, native)
        rejected = json.loads(report.getvalue())
        self.assertEqual(rejected['chunk_start'], 20)
        self.assertEqual(rejected['sample_id'], 'sample')

    def test_no_usable_chunk_has_bounded_failure(self):
        def native(pretrain_data, instruct_tokenizer):
            while True: yield instruct_tokenizer(np.ones((2, 10)), 20, str(self.path))
        class Tokenizer:
            mimi = SimpleNamespace(sample_rate=1)
            def __call__(self, *args): raise AssertionError('bad chunk reached tokenizer')
        loader = SimpleNamespace(build_dataset=native)
        with source_chunk_loader(loader, max_consecutive_rejections=3):
            with self.assertRaisesRegex(RuntimeError, 'consecutive'):
                next(loader.build_dataset('manifest', Tokenizer()))
        self.assertIs(loader.build_dataset, native)

    def test_parity_selects_next_chunk_without_tokenizing_source_error(self):
        import sys
        from unittest.mock import Mock, patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from check_interleaver_parity import select_valid_chunk
        output = self.path.parent / 'parity'; output.mkdir()
        chunks = [dict(data=np.ones((2,10)), unpadded_len=10, start_time_sec=start)
                  for start in (20, 0)]
        sample = SimpleNamespace(provenance={})
        tokenizer = Mock(return_value=sample)
        with patch('tim_compat.sample_filter.chunk_rejection', return_value=None):
            _, result, selection = select_valid_chunk(chunks, tokenizer, self.path,
                                                      'sample', 1, None, output)
        self.assertIs(result, sample)
        self.assertEqual(tokenizer.call_count, 1)
        self.assertEqual(tokenizer.call_args.args[1], 0)
        self.assertEqual(selection['candidate_index'], 1)
        self.assertEqual(selection['chunks_rejected'], 1)
        self.assertEqual(json.loads((output / 'rejections.jsonl').read_text())['stage'],
                         'prepared_source_chunk')

    def test_lazy_report_does_not_create_run_dir_before_native_trainer(self):
        run = self.path.parent / 'run'
        def native(pretrain_data, instruct_tokenizer):
            run.mkdir()
            for start in (20, 0):
                yield instruct_tokenizer(np.ones((2, 10)), start, str(self.path))
        class Tokenizer:
            mimi = SimpleNamespace(sample_rate=1)
            def __call__(self, wav, start, path): return start
        loader = SimpleNamespace(build_dataset=native)
        with source_chunk_loader(loader, report_path=run / 'rejections.jsonl'):
            self.assertFalse(run.exists())
            self.assertEqual(list(loader.build_dataset('manifest', Tokenizer())), [0])
        self.assertEqual(json.loads((run / 'rejections.jsonl').read_text())['chunk_start'], 20)

    def test_finite_all_invalid_fails_instead_of_returning_empty_dataset(self):
        def native(pretrain_data, instruct_tokenizer):
            yield instruct_tokenizer(np.ones((2, 10)), 20, str(self.path))
        class Tokenizer:
            mimi = SimpleNamespace(sample_rate=1)
            def __call__(self, *args): raise AssertionError('invalid chunk tokenized')
        loader = SimpleNamespace(build_dataset=native)
        with source_chunk_loader(loader):
            with self.assertRaisesRegex(RuntimeError, 'no valid source chunk'):
                list(loader.build_dataset('manifest', Tokenizer()))
