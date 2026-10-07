import io
import json
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import torch

from tim_compat.sample_filter import FilterPolicy, filter_samples, sample_filter_loader


class SampleFilterTest(unittest.TestCase):
    def sample(self):
        codes = torch.zeros(1, 17, 6, dtype=torch.long)
        codes[0, 0, 2] = 20
        return SimpleNamespace(codes=codes, prompt_length=2, context_mask=None)

    def policy(self):
        return FilterPolicy(text_cardinality=100, audio_cardinality=2048,
                            nonlexical_text_ids=(0, 1, 2, 3))

    def test_disabled_returns_exact_iterator_without_inspection_or_report(self):
        source = iter([object()])
        report = io.StringIO()
        self.assertIs(filter_samples(source, None, enabled=False, report=report), source)
        self.assertEqual(report.getvalue(), "")

    def test_valid_samples_unchanged_in_original_order(self):
        samples = [self.sample(), self.sample()]
        before = [s.codes.clone() for s in samples]
        result = list(filter_samples(iter(samples), self.policy()))
        for actual, expected, codes in zip(result, samples, before):
            self.assertIs(actual, expected)
            self.assertTrue(torch.equal(actual.codes, codes))

    def test_retained_text_excludes_prompt_and_injections(self):
        sample = self.sample()
        sample.codes[0, 0, 0] = 30
        sample.context_mask = torch.tensor([False, False, True, False, False, False])
        report = io.StringIO()
        self.assertEqual(list(filter_samples(iter([sample]), self.policy(), report=report)), [])
        row = json.loads(report.getvalue())
        self.assertEqual(row['reason'], 'no_retained_agent_text')
        self.assertEqual(row['candidate_index'], 0)

    def test_oob_checks_retained_codes_not_original_word_counts(self):
        bad = self.sample()
        bad.codes[0, 16, -1] = 2048
        good = self.sample()
        report = io.StringIO()
        self.assertEqual(list(filter_samples(iter([bad, good]), self.policy(), report=report)), [good])
        self.assertEqual(json.loads(report.getvalue())['reason'], 'audio_token_out_of_bounds')

    def test_padding_sentinel_and_short_tail(self):
        sample = self.sample()
        sample.codes = sample.codes[..., :3]
        sample.codes[0, 1:, -1] = -1
        self.assertEqual(list(filter_samples(iter([sample]), self.policy())), [sample])

    def test_layout_mask_and_text_bounds_rejected(self):
        for mutate, reason in (
            (lambda s: setattr(s, 'codes', s.codes[:, :9]), 'invalid_stream_layout'),
            (lambda s: setattr(s, 'context_mask', torch.zeros(2, dtype=torch.bool)), 'invalid_context_mask'),
            (lambda s: s.codes[0, 0, 2].fill_(100), 'text_token_out_of_bounds'),
            (lambda s: setattr(s, 'prompt_length', 8), 'invalid_prompt_length'),
        ):
            sample = self.sample()
            mutate(sample)
            report = io.StringIO()
            self.assertEqual(list(filter_samples(iter([sample]), self.policy(), report=report)), [])
            self.assertEqual(json.loads(report.getvalue())['reason'], reason)

    def test_reference_errors_are_not_silently_swallowed(self):
        def source():
            yield self.sample()
            raise RuntimeError('reference failure')
        with self.assertRaisesRegex(RuntimeError, 'reference failure'):
            list(filter_samples(source(), self.policy()))

    def test_rejection_limit_prevents_infinite_empty_training(self):
        policy = FilterPolicy(100, 2048, (0, 1, 2, 3), max_consecutive_rejections=2)
        sample = self.sample()
        sample.codes.zero_()
        with self.assertRaisesRegex(RuntimeError, 'consecutive rejections'):
            list(filter_samples(iter([sample, sample]), policy))

    def test_loader_disable_call_parity_and_restore_on_exception(self):
        source = iter([self.sample()])
        original = Mock(return_value=source)
        loader = SimpleNamespace(build_dataset=original)
        with sample_filter_loader(loader, None, enabled=False):
            self.assertIs(loader.build_dataset, original)
            self.assertIs(loader.build_dataset('manifest', rank=0), source)
        original.assert_called_once_with('manifest', rank=0)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with sample_filter_loader(loader, self.policy()):
                result = list(loader.build_dataset('manifest', rank=0))
                self.assertEqual(len(result), 1)
                raise RuntimeError('test')
        self.assertIs(loader.build_dataset, original)