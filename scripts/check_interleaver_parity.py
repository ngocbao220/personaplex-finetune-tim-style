"""Phase 1: inspect actual Tim tokenization and prove observation/cache parity.

Example (one visible CUDA GPU, explicit local assets in acceptance config)::

    python scripts/check_interleaver_parity.py --config configs/acceptance.yaml \
        --sample-index 0 --output-dir outputs/phase1

--help imports only the standard library. Execution requires Tim's training
Moshi loader and CUDA: the reference tokenizer hardcodes CUDA internally.
This does not prove transcript completeness, loss semantics, or trainer parity.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys


def compare_samples(reference, candidate, torch):
    """Compare every reference Sample field; provenance is observer-only."""
    checks = {
        'codes': (reference.codes.shape == candidate.codes.shape
                  and reference.codes.dtype == candidate.codes.dtype
                  and torch.equal(reference.codes, candidate.codes)),
        'prompt_length': reference.prompt_length == candidate.prompt_length,
        'condition_attributes': reference.condition_attributes == candidate.condition_attributes,
        'injection_stats': reference.injection_stats == candidate.injection_stats,
    }
    a, b = reference.context_mask, candidate.context_mask
    checks['context_mask'] = ((a is None and b is None) or
                              (a is not None and b is not None
                               and a.shape == b.shape and a.dtype == b.dtype
                               and torch.equal(a, b)))
    return checks


def inspect_sample(sample, interleaver, torch):
    """Counts are measured on retained codes, not inferred lost-token counts."""
    codes = sample.codes
    if codes.ndim != 3 or tuple(codes.shape[:2]) != (1, 17):
        raise ValueError(f'expected [1,17,T] (text, LEFT agent 8, RIGHT user 8): {codes.shape}')
    frames = codes.shape[-1]
    if not 0 <= sample.prompt_length <= frames:
        raise ValueError('prompt_length outside codes')
    context = torch.zeros(frames, dtype=torch.bool, device=codes.device)
    if sample.context_mask is not None:
        if sample.context_mask.shape != (frames,) or sample.context_mask.dtype != torch.bool:
            raise ValueError('expected boolean context_mask [T]')
        context = sample.context_mask
    dialogue = codes[0, 0, sample.prompt_length:]
    dialogue_context = context[sample.prompt_length:]
    special = interleaver.special_tokens
    lexical = torch.ones_like(dialogue, dtype=torch.bool)
    for token in special:
        lexical &= dialogue != token
    return dict(
        codes_shape=list(codes.shape), codes_dtype=str(codes.dtype),
        stream_layout=['agent_text'] + [f'agent_audio_{i}' for i in range(8)]
                      + [f'user_audio_{i}' for i in range(8)],
        channel_mapping='LEFT=agent, RIGHT=user',
        total_frames=frames, prompt_length=sample.prompt_length,
        dialogue_frames=frames - sample.prompt_length,
        context_mask_present=sample.context_mask is not None,
        context_mask_frames=int(context.sum().item()),
        dialogue_non_special_text_tokens=int((lexical & ~dialogue_context).sum().item()),
        injection_non_special_text_tokens=int((lexical & dialogue_context).sum().item()),
        dialogue_text_padding_tokens=int((dialogue == interleaver.text_padding).sum().item()),
        dialogue_end_padding_tokens=int((dialogue == interleaver.end_of_text_padding).sum().item()),
        dialogue_zero_padding_tokens=int((dialogue == interleaver.zero_padding).sum().item()),
        zero_padding_positions_per_stream=[int((row == interleaver.zero_padding).sum().item())
                                           for row in codes[0]],
        special_token_ids=sorted(special),
        injection_stats=asdict(sample.injection_stats) if sample.injection_stats is not None else None,
        dropped_dialogue_tokens=None,
        dropped_dialogue_tokens_reason='Reference Sample exposes no dialogue drop counter; retained counts cannot recover it.',
    )


def runtime_filter_policy(model, interleaver):
    from tim_compat.sample_filter import FilterPolicy
    return FilterPolicy(text_cardinality=model.text_card, audio_cardinality=model.card,
        nonlexical_text_ids=tuple(sorted(interleaver.special_tokens | {model.text_initial_token_id})),
        zero_padding_id=model.zero_token_id, require_agent_text=False,
        text_sentinel_ids=(model.text_initial_token_id,), audio_sentinel_ids=(model.initial_token_id,))


def chunk_rejection(sample, policy):
    """Validate native Sample and observed losses; never repair its contents."""
    from tim_compat.sample_filter import rejection_reason, token_bounds_error
    reason = rejection_reason(sample, policy)
    if reason is not None:
        details = {}
        if reason in ('text_token_out_of_bounds', 'audio_token_out_of_bounds'):
            text = reason == 'text_token_out_of_bounds'
            details = token_bounds_error(sample.codes[:, 0] if text else sample.codes[:, 1:],
                policy.text_cardinality if text else policy.audio_cardinality, policy.zero_padding_id,
                policy.text_sentinel_ids if text else policy.audio_sentinel_ids)
        return dict(reason=reason, details=details)
    validation = getattr(sample, 'validation', None)
    if validation is None:
        return None
    if validation['bounds']:
        error = validation['bounds'][0]
        return dict(reason=error['reason'], details=error)
    if validation['prompt_frames_dropped']:
        return dict(reason='prompt_overflow', details=validation)
    if validation['context_tokens_dropped']:
        return dict(reason='context_overflow', details=validation)
    if (validation['dialogue_lexical_dropped'] or
            any(row['overwritten'] or row['tail_pending'] or row['unstarted']
                for row in validation['dialogue'])):
        return dict(reason='text_overflow', details=validation)
    return None


def select_valid_chunk(dataset, observed, path, sample_id, sample_rate, policy, output):
    """Scan only the selected conversation, retaining native order and errors."""
    from _common import write_json
    selection = dict(sample_id=sample_id, chunks_examined=0, chunks_rejected=0,
                     candidate_index=None)
    with (output / 'rejections.jsonl').open('a') as report:
        try:
            for index, chunk in enumerate(dataset):
                selection['chunks_examined'] += 1
                wav = chunk['data'][..., :chunk['unpadded_len']]
                start = chunk['start_time_sec']
                if wav.ndim != 2 or wav.shape[0] != 2:
                    raise ValueError('LEFT=agent, RIGHT=user stereo required')
                sample = observed(wav, start, str(path))
                rejection = chunk_rejection(sample, policy)
                if rejection is None:
                    selection['candidate_index'] = index
                    print(f'Accepted chunk {index}: start={start}s; '
                          f'rejected={selection["chunks_rejected"]}', flush=True)
                    return chunk, sample, selection
                selection['chunks_rejected'] += 1
                row = dict(sample_id=sample_id, candidate_index=index, chunk_start=start,
                    chunk_end=start + wav.shape[-1] / sample_rate,
                    provenance=getattr(sample, 'provenance', None), **rejection)
                report.write(json.dumps(row, ensure_ascii=False) + '\n')
                report.flush()
                print(f'Rejected chunk {index}: start={start}s; reason={row["reason"]}', flush=True)
        except Exception as error:
            write_json(output / 'summary.json', dict(pass_=False, reason='runtime_error',
                error_type=type(error).__name__, error=str(error), filter_policy=asdict(policy), **selection))
            raise
    write_json(output / 'summary.json', dict(pass_=False, reason='no_valid_chunk',
                                           filter_policy=asdict(policy), **selection))
    raise RuntimeError(f'no valid chunk in selected conversation {sample_id}; see {output / "summary.json"}')


def prepare_parity_fixture(config, output, sample_index):
    """Preserve typed source-validation evidence before CUDA/model initialization."""
    from _common import prepare_fixture, write_json
    from tim_compat.prepared_data import PreparedAlignmentError
    try:
        return prepare_fixture(config, output, sample_index=sample_index)
    except PreparedAlignmentError as error:
        output = Path(output).resolve()
        row = dict(sample_id=error.sample_id, candidate_index=None, stage='prepared_source',
                   reason=error.reason, details=error.details)
        with (output / 'rejections.jsonl').open('x') as report:
            report.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(output / 'summary.json', dict(pass_=False, chunks_examined=0,
            chunks_rejected=0, error=str(error), **row))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    opts = p.parse_args(argv)
    if opts.sample_index < 0:
        p.error('--sample-index must be nonnegative')

    from _common import ROOT, offline, read_config, write_json
    offline()
    # Validate local assets/export prepared sources before expensive model imports.
    resolved = prepare_parity_fixture(opts.config, opts.output_dir, opts.sample_index)
    _, acceptance = read_config(opts.config)
    sys.path.insert(0, str(ROOT / 'moshi-finetune'))
    import numpy as np
    import sphn
    import torch
    from moshi.models import loaders
    from finetune.args import TrainArgs
    from finetune.data.interleaver import Interleaver, InterleavedTokenizer
    from tim_compat.local_checkpoint import LocalAssets, LocalCheckpointInfo
    from tim_compat.tokenization import ObservedTokenizer, digest_file

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES=0; CPU --help only')
    torch.cuda.set_device(0)
    args = TrainArgs.load(str(resolved), drop_extra_fields=False)
    if not args.system_prompt.enable:
        raise ValueError('phase1 requires hybrid system prompts enabled')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assets = LocalAssets.resolve(acceptance['model_root'],
        moshi_path=args.moshi_paths.moshi_path, mimi_path=args.moshi_paths.mimi_path,
        tokenizer_path=args.moshi_paths.tokenizer_path, config_path=args.moshi_paths.config_path)
    info = LocalCheckpointInfo(assets, loaders)
    # Actual training constructor, meta-only: obtain LM properties without 7B allocation.
    model = info.get_moshi(device='meta', load_weight=False)
    mimi = info.get_mimi(device='cuda').eval()
    for parameter in mimi.parameters():
        parameter.requires_grad = False
    interleaver = Interleaver(info.get_text_tokenizer(), mimi.frame_rate,
        model.text_padding_token_id, model.end_of_text_padding_id, model.zero_token_id,
        keep_main_only=True)
    tokenizer = InterleavedTokenizer(mimi, interleaver, duration_sec=args.duration_sec,
        system_prompt_enabled=args.system_prompt.enable,
        audio_silence_frames=args.system_prompt.audio_silence_frames,
        prompt_budget_frames=args.system_prompt.prompt_budget_frames)
    if tokenizer.chunk_step_sec <= 0:
        raise ValueError('prompt budget leaves no positive chunk step')
    records = [json.loads(line) for line in Path(args.data.train_data).read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError('phase1 fixture must contain one selected sample')
    # Preserve the exported WAV link so its adjacent Tim JSON remains addressable.
    path = Path(records[0]['path']).absolute()
    metadata = json.loads(path.with_suffix('.json').read_text())
    if not metadata.get('text_prompt') or not metadata.get('voice_prompt'):
        raise ValueError('selected sample needs both text and voice prompts')
    # Scan this conversation through Tim's exact sphn pipeline, no manual resampling.
    dataset = sphn.dataset_jsonl(str(args.data.train_data), duration_sec=tokenizer.chunk_step_sec,
        num_threads=4, sample_rate=mimi.sample_rate, pad_last_segment=True).seq(skip=0, step_by=1)
    source_sr = mimi.sample_rate
    fingerprint_paths = [assets.mimi_weights, assets.tokenizer, resolved,
                         ROOT / 'moshi-finetune/finetune/data/interleaver.py',
                         ROOT / 'tim_compat/tokenization.py']
    hashes = {str(path): digest_file(path) for path in fingerprint_paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    output = opts.output_dir.resolve()
    policy = runtime_filter_policy(model, interleaver)
    sample_id = json.loads((resolved.parent / 'fixture.json').read_text())['samples'][0]['sample_id']
    with (output / 'observations.jsonl').open('x') as report:
        observed = ObservedTokenizer(tokenizer, fingerprint, cache_dir=output / 'token_cache',
                                     report=report, validation_policy=policy)
        chunk, miss, selection = select_valid_chunk(dataset, observed, path, sample_id,
                                                    mimi.sample_rate, policy, output)
        observations = [observed.last_observation]
        wav = chunk['data'][..., :chunk['unpadded_len']]
        start_sec = chunk['start_time_sec']
        hit = observed(wav, start_sec, str(path))
        observations.append(observed.last_observation)
    from tim_compat.tokenization import temporary_binding
    captured_alignments = []
    builder = interleaver.build_token_stream
    def capture(alignments, segment_duration):
        if alignments is not None:
            captured_alignments.extend(alignments)
        return builder(alignments, segment_duration)
    with temporary_binding(interleaver, 'build_token_stream', capture):
        reference = tokenizer(wav, start_sec, str(path))
    comparisons = dict(observation_miss=compare_samples(reference, miss, torch),
                       cache_hit=compare_samples(reference, hit, torch))
    passed = (all(all(checks.values()) for checks in comparisons.values())
              and [row['cache_hit'] for row in observations] == [False, True]
              and chunk_rejection(hit, policy) is None)
    dump = inspect_sample(reference, interleaver, torch)
    ids = [int(token) for alignment in captured_alignments for token in alignment[0]]
    real_ids = [token for token in ids if token not in interleaver.special_tokens]
    text_tokenizer = info.get_text_tokenizer()
    placed = dump['dialogue_non_special_text_tokens']
    dump.update(**selection, filter_policy=asdict(policy), validation=miss.validation,
        chunk_start=start_sec, chunk_end=start_sec+wav.shape[-1]/mimi.sample_rate,
        number_of_real_text_tokens=len(real_ids), number_of_placed_text_tokens=placed,
        dropped_tokens=max(0, len(real_ids)-placed),
        dropped_tokens_definition='Requested lexical occurrences minus retained non-special dialogue occurrences; no token identity repair.',
        overflow_status=chunk_rejection(miss, policy) is not None,
        pass_=passed, comparisons=comparisons, sample_path=str(path), start_sec=start_sec,
        source_sample_rate=source_sr, mimi_sample_rate=mimi.sample_rate,
        mimi_frame_rate=mimi.frame_rate, input_wav_shape=list(wav.shape),
        duration_sec=args.duration_sec, chunk_step_sec=tokenizer.chunk_step_sec,
        prompt_budget_frames=tokenizer.prompt_budget_frames,
        source_hashes=hashes, observations=observations,
        caveats=[
            'Parity is direct actual Tim tokenizer vs opt-in observation/cache, not independent native checkpoint discovery.',
            'No loss masks are invented: prompt_length and context_mask are the reference metadata consumed by Tim loss.',
            'Counts describe retained codes, not transcript completeness or lost dialogue/injection tails.',
            'Reference passes audio frame count to prepare_item segment_duration; this script preserves that behavior.',
            'Legacy dense_overflow is pre-crop; validation measures effective dialogue capacity and retained lexical/context counts.',
            'Chunk obtained from native sphn dataset iterator, ordered first valid chunk of selected conversation.',
        ])
    torch.save(dict(codes=reference.codes.detach().cpu(), prompt_length=reference.prompt_length,
                    context_mask=None if reference.context_mask is None else reference.context_mask.detach().cpu()),
               output / 'streams.pt')
    write_json(output / 'summary.json', dump)
    write_json(output / 'text_tokens.json', dict(token_ids=ids, decoded=text_tokenizer.decode(ids),
        decoded_tokens=[text_tokenizer.id_to_piece(x) for x in ids],
        text_stream=reference.codes[0,0].tolist()))
    rows = [json.dumps(dump, indent=2, ensure_ascii=False)]
    rows += [f'{name}: {reference.codes[0,i].tolist()}' for i,name in enumerate(dump['stream_layout'])]
    (output / 'dump.txt').write_text('\n'.join(rows))
    print(json.dumps(dump, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError(f'Tokenizer parity failed; see {output / "summary.json"}')


if __name__ == '__main__':
    main()
