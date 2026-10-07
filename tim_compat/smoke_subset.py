"""Small immutable native Samples for smoke train/eval/reload, not full coverage."""
import copy
import hashlib
import io
import itertools
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from .sample_filter import chunk_rejection
from .tokenization import publish, temporary_binding


def validate_smoke_count(count):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 40:
        raise ValueError('smoke chunks must be an integer from 1 to 40 (native eval cap)')
    return count


def save_smoke_samples(samples, policy, count, output):
    """Take first N valid Samples without consuming N+1; preserve rejection evidence."""
    validate_smoke_count(count)
    output = Path(output)
    for name in ('smoke_samples.pt', 'smoke_selection.json', 'smoke_rejections.jsonl'):
        if (output / name).exists():
            raise FileExistsError(output / name)
    selected = []
    selection = dict(scope='fixed_chunk_smoke', requested_chunks=count, num_chunks=0,
        chunks_examined=0, chunks_rejected=0, chunks=[],
        filter_policy=json.loads(json.dumps(asdict(policy))), pass_=False)
    with (output / 'smoke_rejections.jsonl').open('x') as report:
        for index, sample in enumerate(samples):
            selection['chunks_examined'] += 1
            provenance = getattr(sample, 'provenance', {})
            row = dict(candidate_index=index, **provenance)
            if provenance.get('path'):
                row['sample_id'] = Path(provenance['path']).stem
            start = provenance.get('start_sec')
            if start is not None:
                row['start_sec'] = float.fromhex(start) if isinstance(start, str) else start
                if provenance.get('sample_rate') and provenance.get('wav_shape'):
                    row['end_sec'] = row['start_sec'] + provenance['wav_shape'][-1] / provenance['sample_rate']
            failure = chunk_rejection(sample, policy)
            if failure is not None:
                selection['chunks_rejected'] += 1
                report.write(json.dumps(dict(**row, **failure), ensure_ascii=False) + '\n')
                report.flush()
                print(f'Smoke rejected chunk {index}: {failure["reason"]}', flush=True)
                continue
            cpu = copy.copy(sample)
            cpu.codes = sample.codes.detach().cpu().clone()
            if sample.context_mask is not None:
                cpu.context_mask = sample.context_mask.detach().cpu().clone()
            selected.append(cpu)
            selection['chunks'].append(row)
            print(f'Smoke selected chunk {index}: {len(selected)}/{count}', flush=True)
            if len(selected) == count:
                break
    selection['num_chunks'] = len(selected)
    if not selected:
        publish(output / 'smoke_selection.json', json.dumps(selection, indent=2).encode())
        raise RuntimeError('no valid smoke chunk; see smoke_rejections.jsonl')
    buffer = io.BytesIO()
    torch.save(selected, buffer)
    payload = buffer.getvalue()
    selection.update(pass_=True, samples_sha256=hashlib.sha256(payload).hexdigest())
    publish(output / 'smoke_samples.pt', payload)
    publish(output / 'smoke_selection.json', json.dumps(selection, indent=2, ensure_ascii=False).encode())
    return selection


def load_smoke_samples(output):
    output = Path(output)
    selection = json.loads((output / 'smoke_selection.json').read_text())
    payload = (output / 'smoke_samples.pt').read_bytes()
    if hashlib.sha256(payload).hexdigest() != selection['samples_sha256']:
        raise ValueError('smoke snapshot checksum mismatch')
    # Trusted artifacts generated locally by this run, like the existing token cache.
    samples = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    validate_smoke_count(len(samples))
    if len(samples) != selection['num_chunks'] or len(samples) != len(selection['chunks']):
        raise ValueError('smoke snapshot count mismatch')
    return samples, selection


def prepare_smoke_samples(data_loader, tokenizer, model, data_args, count, output, config):
    """Use native finite chunking/tokenization once, then freeze valid Samples."""
    from .sample_filter import runtime_filter_policy
    from .tokenization import ObservedTokenizer, digest_file
    base = tokenizer.tokenizer if isinstance(tokenizer, ObservedTokenizer) else tokenizer
    policy = runtime_filter_policy(model, base.interleaver)
    fingerprint = tokenizer.fingerprint if isinstance(tokenizer, ObservedTokenizer) else digest_file(config)
    with (Path(output) / 'smoke_observations.jsonl').open('x') as report:
        observed = ObservedTokenizer(base, fingerprint, report=report, validation_policy=policy)
        native = data_loader.build_dataset(pretrain_data=data_args.train_data,
            instruct_tokenizer=observed, seed=None, rank=0, world_size=1,
            is_eval=True, shuffle_pretrain=False)
        def samples():
            for sample in native:
                sample.provenance['sample_rate'] = base.mimi.sample_rate
                yield sample
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            save_smoke_samples(samples(), policy, count, output)
    return load_smoke_samples(output)


def replay_samples(samples, *, device, is_eval):
    """Fresh native Samples, finite eval and deterministic cycling train."""
    if not samples:
        raise ValueError('empty smoke snapshot')
    source = iter(samples) if is_eval else itertools.cycle(samples)
    for sample in source:
        replay = copy.copy(sample)
        replay.codes = sample.codes.to(device).clone()
        if sample.context_mask is not None:
            replay.context_mask = sample.context_mask.to(device).clone()
        yield replay


@contextmanager
def smoke_dataset_loader(data_loader, samples, tokenizer):
    """Scoped dataset seam; Tim's batching/training/loss code remains unchanged."""
    def dataset(*args, **kwargs):
        if kwargs.get('rank', 0) != 0 or kwargs.get('world_size', 1) != 1:
            raise ValueError('smoke subset supports a single GPU only')
        return replay_samples(samples, device=tokenizer.interleaver.device,
                              is_eval=kwargs['is_eval'])
    with temporary_binding(data_loader, 'build_dataset', dataset):
        yield
