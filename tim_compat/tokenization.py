"""Opt-in observation/cache around the actual Tim tokenizer, not a replacement.

Single-process GPU encoding only. Cache hits deserialize CPU Samples then restore
the original device. Inputs/model fingerprint must describe immutable assets.
"""
import hashlib
import io
import json
import math
import os
import tempfile
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import torch


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dense_overflow(alignments, frames, frame_rate, keep_and_shift):
    """Replay queue occupancy only; never change reference placement decisions."""
    queue = deque()
    index = dropped = requested = 0
    for frame in range(frames):
        while index < len(alignments) and alignments[index][1][0] * frame_rate < frame + 1:
            tokens = alignments[index][0]
            requested += len(tokens)
            if not keep_and_shift:
                dropped += len(queue)
                queue.clear()
            queue.extend(tokens)
            index += 1
        if queue:
            queue.popleft()
    return {'requested': requested, 'overwritten': dropped, 'tail_pending': len(queue)}


class ObservedTokenizer:
    def __init__(self, tokenizer, fingerprint, *, cache_dir=None, report=None):
        if not fingerprint:
            raise ValueError('explicit model/tokenizer/source fingerprint required')
        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.report = report
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, wav, start_sec, path):
        path = Path(path).resolve()
        metadata = path.with_suffix('.json')
        data = json.loads(metadata.read_text())
        voice_hash = None
        if data.get('voice_prompt'):
            voice = Path(data['voice_prompt'])
            if not voice.is_absolute():
                voice = path.parent / voice
            voice_hash = digest_file(voice)
        config = {key: getattr(self.tokenizer, key) for key in (
            'duration_sec', 'system_prompt_enabled', 'audio_silence_frames', 'prompt_budget_frames')}
        identity = {'path': str(path), 'start_sec': float(start_sec).hex(),
                    'metadata': digest_file(metadata), 'voice': voice_hash,
                    'fingerprint': self.fingerprint,
                    'config': config, 'wav_shape': list(wav.shape),
                    'wav_dtype': str(wav.dtype),
                    'wav': hashlib.sha256(wav.tobytes()).hexdigest()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = self.cache_dir / (key + '.pt') if self.cache_dir else None
        diagnostics = []
        hit = destination is not None and destination.exists()
        if hit:
            payload = destination.read_bytes()
            checksum = destination.with_suffix('.sha256').read_text().strip()
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f'corrupt token cache: {destination}')
            # Trusted local cache only: Sample includes reference dataclasses.
            saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            sample, diagnostics = saved['sample'], saved['diagnostics']
            sample.codes = sample.codes.to(self.interleaver.device)
            if sample.context_mask is not None:
                sample.context_mask = sample.context_mask.to(sample.codes.device)
        else:
            interleaver = self.interleaver
            original = interleaver.build_token_stream

            def observed(alignments, segment_duration):
                if alignments is not None:
                    diagnostics.append(dense_overflow(alignments,
                        math.ceil(segment_duration * interleaver.audio_frame_rate),
                        interleaver.audio_frame_rate, interleaver.keep_and_shift))
                return original(alignments, segment_duration)

            with temporary_binding(interleaver, 'build_token_stream', observed):
                sample = self.tokenizer(wav, start_sec, str(path))
            if destination is not None:
                import copy
                cpu = copy.copy(sample)
                cpu.codes = sample.codes.detach().cpu()
                if sample.context_mask is not None:
                    cpu.context_mask = sample.context_mask.detach().cpu()
                buffer = io.BytesIO()
                torch.save({'sample': cpu, 'diagnostics': diagnostics}, buffer)
                payload = buffer.getvalue()
                # Lock publication of the two files. Other publishers must fail,
                # never read an incomplete pair or silently overwrite.
                lock = destination.with_suffix('.lock')
                with lock.open('x'):
                    try:
                        publish(destination.with_suffix('.sha256'),
                                hashlib.sha256(payload).hexdigest().encode())
                        publish(destination, payload)
                    finally:
                        lock.unlink()
        sample.provenance = {'chunk_id': key, **identity}
        if self.report is not None:
            self.report.write(json.dumps({'chunk_id': key, 'path': str(path),
                'start_sec': start_sec, 'cache_hit': hit, 'dense_overflow': diagnostics}) + '\n')
            self.report.flush()
        return sample


def publish(path, payload):
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


@contextmanager
def temporary_binding(owner, name, replacement):
    previous = getattr(owner, name)
    owned = name in vars(owner)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


@contextmanager
def tokenizer_bridge(module, fingerprint, *, cache_dir=None, report=None, enabled=True):
    if not enabled:
        yield
        return
    original = module.InterleavedTokenizer
    def factory(*args, **kwargs):
        return ObservedTokenizer(original(*args, **kwargs), fingerprint,
                                 cache_dir=cache_dir, report=report)
    with temporary_binding(module, 'InterleavedTokenizer', factory):
        yield