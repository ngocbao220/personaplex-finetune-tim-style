"""CPU regressions for phase-1 filtering; not GPU parity evidence."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tim_compat.sample_filter import FilterPolicy, chunk_rejection, runtime_filter_policy
from tim_compat.tokenization import ObservedTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_interleaver_parity as phase1


def reference_interleaver():
    spec = importlib.util.spec_from_file_location('tim_filter_reference',
        ROOT / 'moshi-finetune/finetune/data/interleaver.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    text = SimpleNamespace(bos_id=lambda: 1, eos_id=lambda: 2)
    return module.Interleaver(text, 2, 0, 3, -1, device='cpu')


class CPUChunkTokenizer:
    """Use the real Tim builder and tensor crop, with CPU-only prompt/audio."""
    duration_sec = 2
    num_audio_frames = 4
    system_prompt_enabled = True
    audio_silence_frames = 0
    prompt_budget_frames = 0

    def __init__(self, alignments, prompt_frames=1, injection_tokens=0):
        self.interleaver = reference_interleaver()
        self.alignments = alignments
        self.prompt_frames = prompt_frames
        self.injection_tokens = injection_tokens
        self.calls = 0

    def _build_system_prompt_prefix(self, *args):
        return torch.zeros(1, 17, self.prompt_frames, dtype=torch.long), self.prompt_frames

    def __call__(self, wav, start_sec, path):
        self.calls += 1
        prompt, length = self._build_system_prompt_prefix('prompt', None, path)
        length = min(length, self.num_audio_frames - 1)
        prompt = prompt[..., :length]
        frames = self.num_audio_frames - length
        text = self.interleaver.build_token_stream(self.alignments, frames)
        text = torch.nn.functional.pad(text, (0, frames - text.shape[-1]), value=-1)
        conversation = torch.cat((text, torch.zeros(1, 16, frames, dtype=torch.long)), dim=1)
        mask = None
        stats = None
        if self.injection_tokens:
            injection = torch.zeros(1, 17, self.injection_tokens, dtype=torch.long)
            injection[:, 0] = 30
            conversation = torch.cat((injection, conversation), dim=-1)[..., :frames]
            mask = torch.zeros(self.num_audio_frames, dtype=torch.bool)
            mask[length:length + min(frames, self.injection_tokens)] = True
            stats = SimpleNamespace(tokens_requested=self.injection_tokens,
                                    tokens_placed=self.injection_tokens)
        return SimpleNamespace(codes=torch.cat((prompt, conversation), dim=-1),
            prompt_length=length, context_mask=mask, injection_stats=stats,
            condition_attributes=None)


class InterleaverFilterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'audio.wav'
        self.path.with_suffix('.json').write_text('{}')
        self.wav = np.zeros((2, 8), dtype=np.float32)
        self.policy = FilterPolicy(100, 2048, (0, 1, 2, 3), require_agent_text=False)

    def observe(self, tokenizer, cache=False):
        return ObservedTokenizer(tokenizer, 'runtime', validation_policy=self.policy,
            cache_dir=self.root / 'cache' if cache else None, report=io.StringIO())

    def test_crop_overflow_missed_by_old_pre_crop_diagnostics(self):
        observed = self.observe(CPUChunkTokenizer([([10, 11, 12, 13], (0, 2), 'agent')]))
        sample = observed(self.wav, 0, self.path)
        row = json.loads(observed.report.getvalue())
        self.assertEqual(row['dense_overflow'][0]['tail_pending'], 0)
        failure = chunk_rejection(sample, self.policy)
        self.assertEqual(failure['reason'], 'text_overflow')
        self.assertEqual(failure['details']['dialogue'][0]['tail_pending'], 1)

    def test_overwrite_and_unstarted_tokens_are_rejected(self):
        for alignments in (
            [([10, 11, 12], (0, 1), 'agent'), ([13], (.5, 1), 'agent')],
            [([10], (1.6, 2), 'agent')],
        ):
            with self.subTest(alignments=alignments):
                sample = self.observe(CPUChunkTokenizer(alignments))(self.wav, 0, self.path)
                self.assertEqual(chunk_rejection(sample, self.policy)['reason'], 'text_overflow')

    def test_prompt_and_context_truncation_and_dialogue_displacement(self):
        for tokenizer, expected in (
            (CPUChunkTokenizer([], prompt_frames=5), 'prompt_overflow'),
            (CPUChunkTokenizer([], injection_tokens=4), 'context_overflow'),
            (CPUChunkTokenizer([([10], (1, 2), 'agent')], injection_tokens=2), 'text_overflow'),
        ):
            with self.subTest(expected=expected):
                sample = self.observe(tokenizer)(self.wav, 0, self.path)
                self.assertEqual(chunk_rejection(sample, self.policy)['reason'], expected)

    def test_bounds_checked_before_crop_and_allowed_runtime_sentinels(self):
        sample = self.observe(CPUChunkTokenizer([([100], (2, 3), 'agent')]))(self.wav, 0, self.path)
        self.assertEqual(chunk_rejection(sample, self.policy)['reason'], 'text_token_out_of_bounds')
        model = SimpleNamespace(text_card=100, card=2048, zero_token_id=-1,
            text_initial_token_id=100, initial_token_id=2048)
        policy = runtime_filter_policy(model, SimpleNamespace(special_tokens={0, 1, 2, -1}))
        codes = torch.zeros(1, 17, 3, dtype=torch.long)
        codes[0, 0] = torch.tensor([-1, 100, 99])
        codes[0, 1] = torch.tensor([-1, 2048, 2047])
        sample = SimpleNamespace(codes=codes, prompt_length=0, context_mask=None)
        self.assertIsNone(chunk_rejection(sample, policy))
        codes[0, 1, 0] = 2049
        self.assertEqual(chunk_rejection(sample, policy)['reason'], 'audio_token_out_of_bounds')

    def test_valid_sample_cache_diagnostics_and_reference_fields_unchanged(self):
        tokenizer = CPUChunkTokenizer([([10], (0, 1), 'agent')])
        reference = tokenizer(self.wav, 0, self.path)
        observed = self.observe(tokenizer, cache=True)
        miss = observed(self.wav, 0, self.path)
        hit = observed(self.wav, 0, self.path)
        self.assertIsNone(chunk_rejection(miss, self.policy))
        self.assertIsNone(chunk_rejection(hit, self.policy))
        self.assertEqual(miss.validation, hit.validation)
        self.assertTrue(all(phase1.compare_samples(reference, hit, torch).values()))
        self.assertEqual(tokenizer.calls, 2)

    def test_exported_symlink_keeps_adjacent_tim_sidecar(self):
        source = self.root / 'source' / 'conversation.wav'
        source.parent.mkdir()
        source.write_bytes(b'prepared audio placeholder')
        self.path.symlink_to(source)
        observed = self.observe(CPUChunkTokenizer([([10], (0, 1), 'agent')]), cache=True)
        sample = observed(self.wav, 0, self.path)
        self.assertIsNone(chunk_rejection(sample, self.policy))
        self.assertEqual(sample.provenance['path'], str(self.path))
        self.assertFalse(source.with_suffix('.json').exists())

    def test_invalid_sample_structure_reaches_filter_without_observer_indexing(self):
        original = CPUChunkTokenizer.__call__
        for mutate, reason in (
            (lambda s: setattr(s, 'codes', s.codes[:, :9]), 'invalid_stream_layout'),
            (lambda s: setattr(s, 'prompt_length', 99), 'invalid_prompt_length'),
            (lambda s: setattr(s, 'context_mask', torch.zeros(2, dtype=torch.bool)), 'invalid_context_mask'),
        ):
            with self.subTest(reason=reason):
                class InvalidSample(CPUChunkTokenizer):
                    def __call__(self, *args):
                        sample = original(self, *args)
                        mutate(sample)
                        return sample

                tokenizer = InvalidSample([([10], (0, 1), 'agent')])
                sample = self.observe(tokenizer)(self.wav, 0, self.path)
                self.assertEqual(chunk_rejection(sample, self.policy)['reason'], reason)

    def test_strict_cache_preserves_rejection_and_corruption_still_fails(self):
        tokenizer = CPUChunkTokenizer([([10, 11, 12, 13], (0, 2), 'agent')])
        observed = self.observe(tokenizer, cache=True)
        miss = observed(self.wav, 0, self.path)
        hit = observed(self.wav, 0, self.path)
        self.assertEqual(chunk_rejection(miss, self.policy),
                         chunk_rejection(hit, self.policy))
        self.assertEqual(tokenizer.calls, 1)
        cache = self.root / 'cache' / (miss.provenance['chunk_id'] + '.pt')
        cache.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt token cache'):
            observed(self.wav, 0, self.path)

    def test_selection_skips_invalid_and_keeps_candidate_identity(self):
        tokenizer = CPUChunkTokenizer([([10], (0, 1), 'agent')])
        original = tokenizer.__class__.__call__

        class InvalidFirst(CPUChunkTokenizer):
            def __call__(self, wav, start_sec, path):
                sample = original(self, wav, start_sec, path)
                if start_sec == 0:
                    sample.codes[0, 0, 1] = 100
                return sample

        observed = self.observe(InvalidFirst(tokenizer.alignments))
        chunks = [{'data': self.wav, 'unpadded_len': 8, 'start_time_sec': start} for start in (0, 2)]
        chunk, sample, selection = phase1.select_valid_chunk(chunks, observed, self.path,
            'one', 4, self.policy, self.root)
        self.assertIs(chunk, chunks[1])
        self.assertEqual(selection['chunks_examined'], 2)
        self.assertEqual(selection['chunks_rejected'], 1)
        self.assertEqual(selection['candidate_index'], 1)
        rejection = json.loads((self.root / 'rejections.jsonl').read_text())
        self.assertEqual(rejection['chunk_start'], 0)
        self.assertEqual(rejection['chunk_end'], 2)
        self.assertEqual(rejection['sample_id'], 'one')

    def test_all_invalid_and_runtime_error_preserve_failure_reports(self):
        observed = self.observe(CPUChunkTokenizer([([100], (0, 1), 'agent')]))
        chunk = {'data': self.wav, 'unpadded_len': 8, 'start_time_sec': 0}
        with self.assertRaisesRegex(RuntimeError, 'no valid chunk'):
            phase1.select_valid_chunk([chunk], observed, self.path, 'one', 4, self.policy, self.root)
        summary = json.loads((self.root / 'summary.json').read_text())
        self.assertFalse(summary['pass_'])
        self.assertEqual(summary['chunks_rejected'], 1)
        self.assertEqual(summary['filter_policy']['text_cardinality'], 100)

        def failed(*args):
            raise RuntimeError('CUDA failed')

        with self.assertRaisesRegex(RuntimeError, 'CUDA failed'):
            phase1.select_valid_chunk([chunk], failed, self.path, 'one', 4, self.policy, self.root)

    def test_native_tokenizer_boundary_tail_context_and_binding_restore(self):
        interleaver = reference_interleaver()
        module = sys.modules[type(interleaver).__module__]
        text_ids = {'dense': [10, 11, 12, 13], 'late': [100],
                    '<context> bad </context>': [100] * 4,
                    '<context> long </context>': [20] * 4}

        def encode(lines):
            if isinstance(lines, str):
                return [9]
            return [text_ids.get(line, [10]) for line in lines]

        interleaver.tokenizer.encode = encode
        mimi = SimpleNamespace(frame_rate=2, sample_rate=4,
            encode=lambda audio: torch.zeros(audio.shape[0], 8, max(1, audio.shape[-1] // 2), dtype=torch.long))
        tokenizer = module.InterleavedTokenizer(mimi, interleaver, duration_sec=2,
                                               system_prompt_enabled=True, audio_silence_frames=0)
        originals = {name: getattr(torch, name) for name in ('tensor', 'zeros', 'full')}

        def cpu_factory(name):
            def factory(*args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return originals[name](*args, **kwargs)
            return factory

        original_tokenize = module.tokenize
        original_builder = interleaver.build_token_stream.__func__
        original_prompt = tokenizer._build_system_prompt_prefix.__func__
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self))
            for name in originals:
                stack.enter_context(patch.object(torch, name, cpu_factory(name)))
            # Run the frozen native tokenizer body, replacing only CUDA/codec execution.
            for word, end, injection, expected in (
                ('hello', 2.5, None, None),  # word crosses the chunk boundary
                ('dense', 1, None, 'text_overflow'),
                ('hello', 1, 'long', 'context_overflow'),
                ('hello', 1, 'bad', 'text_token_out_of_bounds'),
            ):
                with self.subTest(word=word, injection=injection):
                    data = dict(text_prompt='prompt', alignments=[[word, [0, end], 'agent']])
                    if injection:
                        data['context_injections'] = [dict(frame_offset=0, text=injection)]
                    self.path.with_suffix('.json').write_text(json.dumps(data))
                    observed = self.observe(tokenizer)
                    sample = observed(self.wav, 0, self.path)
                    rejection = chunk_rejection(sample, self.policy)
                    self.assertEqual(rejection['reason'] if rejection else None, expected)
                    self.assertIs(module.tokenize, original_tokenize)
                    self.assertIs(interleaver.build_token_stream.__func__, original_builder)
                    self.assertIs(tokenizer._build_system_prompt_prefix.__func__, original_prompt)
            # Short final chunk: native padding is not lexical overflow.
            self.path.with_suffix('.json').write_text(json.dumps(dict(
                text_prompt='prompt', alignments=[['hello', [0, .4], 'agent']])))
            sample = self.observe(tokenizer)(self.wav[..., :2], 0, self.path)
            self.assertIsNone(chunk_rejection(sample, self.policy))


class SourceAlignmentReportTest(unittest.TestCase):
    def test_timestamp_errors_fail_before_model_load_with_source_reports(self):
        from tim_compat.prepared_data import PreparedAlignmentError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / 'prepared'
            directory = prepared / 'one'
            directory.mkdir(parents=True)
            for name, channels in (('conversation.wav', 2), ('voice_prompt_left.wav', 1)):
                with wave.open(str(directory / name), 'wb') as wav:
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(b'\0\0' * channels * 24000)
            (directory / 'metadata.json').write_text(json.dumps(dict(
                agent_channel='left', user_channel='right', text_prompt='prompt')))
            manifest = prepared / 'manifest.jsonl'
            manifest.write_text(json.dumps(dict(sample_id='one', sample_dir='one')) + '\n')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(acceptance=dict(
                model_root=str(root / 'no-model-required'), prepared_manifest=str(manifest)))))
            words_path = directory / 'words.json'
            for index, (start, end, reason) in enumerate((
                (-.1, .4, 'timestamp_out_of_bounds'),
                (.1, 2, 'timestamp_out_of_bounds'),
                (float('nan'), .4, 'invalid_timestamp'),
                (.1, float('inf'), 'invalid_timestamp'),
                ('bad', .4, 'invalid_timestamp'),
            )):
                with self.subTest(start=start, end=end):
                    words_path.write_text(json.dumps([dict(word='hello', start=start, end=end, speaker='agent')]))
                    before = words_path.read_bytes()
                    output = root / f'run-{index}'
                    with self.assertRaises(PreparedAlignmentError):
                        phase1.prepare_parity_fixture(config, output, 0)
                    summary = json.loads((output / 'summary.json').read_text())
                    self.assertFalse(summary['pass_'])
                    self.assertEqual(summary['reason'], reason)
                    self.assertEqual(summary['stage'], 'prepared_source')
                    row = json.loads((output / 'rejections.jsonl').read_text())
                    self.assertEqual(row['sample_id'], 'one')
                    self.assertEqual(row['details']['word_index'], 0)
                    self.assertEqual(row['details']['audio_duration_sec'], 1)
                    self.assertEqual(words_path.read_bytes(), before)
                    summary_before = (output / 'summary.json').read_bytes()
                    with self.assertRaises(FileExistsError):
                        phase1.prepare_parity_fixture(config, output, 0)
                    self.assertEqual((output / 'summary.json').read_bytes(), summary_before)


if __name__ == '__main__':
    unittest.main()
