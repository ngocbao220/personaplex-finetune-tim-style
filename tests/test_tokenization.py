import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tim_compat.tokenization import ObservedTokenizer, dense_overflow, tokenizer_bridge


class FakeTokenizer:
    def __init__(self):
        self.duration_sec = 1
        self.system_prompt_enabled = False
        self.audio_silence_frames = 6
        self.prompt_budget_frames = 0
        self.interleaver = SimpleNamespace(device='cpu', audio_frame_rate=2,
            keep_and_shift=False, build_token_stream=lambda a, d: torch.tensor([3]))
        self.calls = 0

    def __call__(self, wav, start_sec, path):
        self.calls += 1
        self.interleaver.build_token_stream([([1, 2, 3], (0, 1), 'agent')], 1)
        return SimpleNamespace(codes=torch.ones(1, 17, 2, dtype=torch.long),
                               context_mask=None, prompt_length=0)


class TokenizationTest(unittest.TestCase):
    def test_queue_diagnostics(self):
        alignments = [([1, 2, 3], (0, 1), 'a'), ([4, 5], (.5, 1), 'a')]
        self.assertEqual(dense_overflow(alignments, 2, 2, False),
                         dict(requested=5, overwritten=2, tail_pending=1))
        self.assertEqual(dense_overflow(alignments, 2, 2, True)['tail_pending'], 3)

    def test_hit_invalidation_corruption_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audio.wav'
            path.with_suffix('.json').write_text('{}')
            fake = FakeTokenizer()
            original = fake.interleaver.build_token_stream
            report = io.StringIO()
            wrapped = ObservedTokenizer(fake, 'model', cache_dir=Path(tmp)/'cache', report=report)
            wav = np.zeros((2, 4), dtype=np.float32)
            first = wrapped(wav, 0, path)
            second = wrapped(wav, 0, path)
            self.assertEqual(fake.calls, 1)
            self.assertTrue(torch.equal(first.codes, second.codes))
            self.assertEqual(first.provenance, second.provenance)
            self.assertIs(original, fake.interleaver.build_token_stream)
            third = wrapped(wav, 1, path)
            self.assertNotEqual(first.provenance['chunk_id'], third.provenance['chunk_id'])
            path.with_suffix('.json').write_text('{"changed": true}')
            wrapped(wav, 0, path)
            self.assertEqual(fake.calls, 3)
            cache = Path(tmp)/'cache'/(first.provenance['chunk_id']+'.pt')
            cache.write_bytes(b'corrupt')
            path.with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                wrapped(wav, 0, path)

    def test_disabled_no_io(self):
        with tokenizer_bridge(None, None, enabled=False):
            pass


if __name__ == '__main__':
    unittest.main()