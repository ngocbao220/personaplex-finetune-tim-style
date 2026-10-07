"""Opt-in observation/cache around the actual Tim tokenizer, not a replacement.

Single-process GPU encoding only. Cache hits deserialize CPU Samples then restore
the original device. Inputs/model fingerprint must describe immutable assets.
"""
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import deque
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
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
    def __init__(self, tokenizer, fingerprint, *, cache_dir=None, report=None,
                 validation_policy=None):
        if not fingerprint:
            raise ValueError('explicit model/tokenizer/source fingerprint required')
        self.tokenizer = tokenizer
        self.fingerprint = fingerprint
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.report = report
        self.validation_policy = validation_policy
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, wav, start_sec, path):
        # Exported WAVs are symlinks; Tim JSON belongs beside the export link.
        path = Path(os.path.abspath(path))
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
        if self.validation_policy is not None:
            # Separate strict diagnostics from legacy cache entries.
            identity['validation'] = {'version': 1, 'policy': asdict(self.validation_policy)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = self.cache_dir / (key + '.pt') if self.cache_dir else None
        diagnostics = []
        validation = None
        hit = destination is not None and destination.exists()
        if hit:
            payload = destination.read_bytes()
            checksum = destination.with_suffix('.sha256').read_text().strip()
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f'corrupt token cache: {destination}')
            # Trusted local cache only: Sample includes reference dataclasses.
            saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            sample, diagnostics = saved['sample'], saved['diagnostics']
            if self.validation_policy is not None:
                validation = saved['validation']
            sample.codes = sample.codes.to(self.interleaver.device)
            if sample.context_mask is not None:
                sample.context_mask = sample.context_mask.to(sample.codes.device)
        else:
            interleaver = self.interleaver
            original = interleaver.build_token_stream
            streams = []
            prompt_frames = 0
            bounds = []

            def check_bounds(codes, region):
                from .sample_filter import token_bounds_error
                policy = self.validation_policy
                for tokens, cardinality, sentinels, reason in (
                    (codes[:, 0], policy.text_cardinality, policy.text_sentinel_ids,
                     'text_token_out_of_bounds'),
                    (codes[:, 1:], policy.audio_cardinality, policy.audio_sentinel_ids,
                     'audio_token_out_of_bounds'),
                ):
                    error = token_bounds_error(tokens, cardinality, policy.zero_padding_id, sentinels)
                    if error:
                        bounds.append(dict(reason=reason, region=region, **error))

            def observe_prompt(*args, **kwargs):
                nonlocal prompt_frames
                codes, prompt_frames = build_prompt(*args, **kwargs)
                check_bounds(codes, 'prompt_before_clamp')
                return codes, prompt_frames

            def observed(alignments, segment_duration):
                if alignments is not None:
                    diagnostics.append(dense_overflow(alignments,
                        math.ceil(segment_duration * interleaver.audio_frame_rate),
                        interleaver.audio_frame_rate, interleaver.keep_and_shift))
                codes = original(alignments, segment_duration)
                if self.validation_policy is not None:
                    if alignments is not None:
                        raw_ids = [int(token) for alignment in alignments for token in alignment[0]]
                        check_bounds(torch.tensor(raw_ids, dtype=torch.long).view(1, 1, -1),
                                     'dialogue_before_placement')
                    check_bounds(codes, 'dialogue_before_crop')
                    if alignments is not None:
                        streams.append((alignments, codes.shape[-1], lexical_count(codes[:, 0], interleaver)))
                return codes

            def observe_tokens(*args, **kwargs):
                tokens = tokenize(*args, **kwargs)
                check_bounds(torch.tensor(tokens, dtype=torch.long).view(1, 1, -1),
                             'text_before_placement')
                return tokens

            with ExitStack() as stack:
                stack.enter_context(temporary_binding(interleaver, 'build_token_stream', observed))
                if self.validation_policy is not None and self.system_prompt_enabled:
                    build_prompt = self.tokenizer._build_system_prompt_prefix
                    stack.enter_context(temporary_binding(self.tokenizer, '_build_system_prompt_prefix', observe_prompt))
                if self.validation_policy is not None:
                    module = sys.modules[type(self.tokenizer).__module__]
                    if hasattr(module, 'tokenize'):
                        tokenize = module.tokenize
                        stack.enter_context(temporary_binding(module, 'tokenize', observe_tokens))
                sample = self.tokenizer(wav, start_sec, str(path))
            if self.validation_policy is not None:
                validation = retained_validation(sample, self.validation_policy, interleaver,
                                                 streams, prompt_frames, bounds)
            if destination is not None:
                import copy
                cpu = copy.copy(sample)
                cpu.codes = sample.codes.detach().cpu()
                if sample.context_mask is not None:
                    cpu.context_mask = sample.context_mask.detach().cpu()
                buffer = io.BytesIO()
                torch.save({'sample': cpu, 'diagnostics': diagnostics, 'validation': validation}, buffer)
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
        if self.validation_policy is not None:
            sample.validation = validation
        self.last_observation = {'chunk_id': key, 'path': str(path),
                'start_sec': start_sec, 'cache_hit': hit, 'dense_overflow': diagnostics,
                'validation': validation}
        if self.report is not None:
            self.report.write(json.dumps(self.last_observation) + '\n')
            self.report.flush()
        return sample


def lexical_count(tokens, interleaver):
    lexical = torch.ones_like(tokens, dtype=torch.bool)
    for token_id in interleaver.special_tokens:
        lexical &= tokens != token_id
    return int(lexical.sum().item())


def retained_validation(sample, policy, interleaver, streams, prompt_frames, bounds):
    """Measure loss against retained frames, without reconstructing token codes."""
    from .sample_filter import rejection_reason
    if rejection_reason(sample, policy) in (
            'invalid_stream_layout', 'invalid_prompt_length', 'invalid_context_mask'):
        # These Samples must reach the filter, not fail inside observation indexing.
        return dict(bounds=bounds)
    dialogue_frames = sample.codes.shape[-1] - sample.prompt_length
    effective = []
    for alignments, built_frames, lexical_built in streams:
        frames = min(built_frames, dialogue_frames)
        row = dense_overflow(alignments, frames, interleaver.audio_frame_rate,
                             interleaver.keep_and_shift)
        row.update(retained_frames=frames,
            unstarted=sum(len(a[0]) for a in alignments
                          if a[1][0] * interleaver.audio_frame_rate >= frames),
            lexical_built=lexical_built)
        effective.append(row)
    dialogue = sample.codes[:, 0, sample.prompt_length:]
    if sample.context_mask is not None:
        dialogue = dialogue[:, ~sample.context_mask[sample.prompt_length:]]
    lexical_retained = lexical_count(dialogue, interleaver)
    context_retained = (int(sample.context_mask.sum().item())
                        if sample.context_mask is not None else 0)
    stats = getattr(sample, 'injection_stats', None)
    context_requested = stats.tokens_requested if stats is not None else 0
    return dict(bounds=bounds, dialogue=effective,
        prompt_frames_requested=prompt_frames, prompt_frames_retained=sample.prompt_length,
        prompt_frames_dropped=max(0, prompt_frames - sample.prompt_length),
        dialogue_lexical_dropped=max(0, sum(s[2] for s in streams) - lexical_retained),
        context_tokens_requested=context_requested, context_tokens_retained=context_retained,
        context_tokens_dropped=max(0, context_requested - context_retained))


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
