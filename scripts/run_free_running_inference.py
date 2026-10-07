"""Merge with original pipeline and run bundled PersonaPlex offline inference unchanged."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
import shutil
from _common import ROOT, offline, read_config, write_json


def require_file(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'missing or empty local file: {path}')
    return path


def dialogue_arrays(original, user, base, current):
    """Assert a matched full-window comparison; never silently crop/pad exports."""
    import numpy as np
    original, user, base, current = map(np.asarray, (original, user, base, current))
    if original.ndim != 2 or original.shape[0] != 2 or not original.shape[-1]:
        raise ValueError('original dialogue must have LEFT=agent, RIGHT=user')
    if any(x.shape != (1, original.shape[-1]) for x in (user, base, current)):
        raise ValueError('original/user/base/current audio windows must have identical length')
    if not all(np.isfinite(x).all() for x in (original, user, base, current)):
        raise ValueError('nonfinite dialogue audio')
    if not np.allclose(original[1], user[0], atol=1e-4, rtol=0):
        raise ValueError('input user WAV differs from RIGHT channel of original window')
    return np.concatenate((base, user)), np.concatenate((current, user))


def text_from_pieces(pieces):
    return ''.join(piece for piece in pieces if piece not in ('EPAD', 'PAD', 'BOS', 'EOS')).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path, help='Tim consolidated adapter directory')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--input-wav', type=Path, help='Mono user WAV; extract RIGHT channel externally')
    p.add_argument('--voice-prompt', type=Path, help='Prepared agent WAV or native .pt voice embeddings')
    p.add_argument('--text-prompt-file', type=Path, help='UTF-8 system prompt')
    p.add_argument('--original-wav', type=Path, help='Stereo original window, LEFT=agent RIGHT=user')
    p.add_argument('--reference-text-file', type=Path, help='Optional agent transcript for this window')
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--step', type=int, help='Actual optimizer step, overriding config label')
    p.add_argument('--baseline-dir', type=Path, help='Reuse a verified comparison baseline')
    p.add_argument('--native-weight', type=Path, help=argparse.SUPPRESS)
    opts = p.parse_args()
    offline()
    values, acceptance = read_config(opts.config)
    infer = acceptance.get('inference', {})
    paths = {}
    for key in ('input_wav', 'voice_prompt', 'text_prompt_file'):
        value = getattr(opts, key) or infer.get(key)
        if not value:
            p.error(f'provide --{key.replace("_", "-")} or acceptance.inference.{key}')
        paths[key] = require_file(value)
    if opts.native_weight is None:
        original = opts.original_wav or infer.get('original_wav')
        if not original:
            p.error('provide --original-wav or acceptance.inference.original_wav')
        paths['original_wav'] = require_file(original)
        reference = opts.reference_text_file or infer.get('reference_text_file')
        if reference:
            paths['reference_text_file'] = require_file(reference)
    checkpoint = opts.checkpoint.resolve()
    require_file(checkpoint / 'lora.safetensors')
    adapter_config = require_file(checkpoint / 'config.json')
    metadata = json.loads(adapter_config.read_text())
    if not {'lora_rank', 'lora_scaling'}.issubset(metadata):
        raise ValueError('adapter config missing explicit lora_rank/lora_scaling; refuse merge defaults')
    from tim_compat.local_checkpoint import LocalAssets
    assets = LocalAssets.resolve(acceptance['model_root'], **{
        key: values.get('moshi_paths', {}).get(key)
        for key in ('moshi_path', 'mimi_path', 'tokenizer_path', 'config_path')})
    if assets.config is None:
        raise ValueError('native offline inference requires explicit local moshi_paths.config_path')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU')
    # Use bundled inference package, not installed training Moshi. Fresh CLI process only.
    if 'moshi' in sys.modules:
        raise RuntimeError('STOP: Moshi already imported; run inference in a fresh process')
    sys.path.insert(0, str(ROOT / 'personaplex/moshi'))
    sys.path.insert(0, str(ROOT / 'pipeline'))
    import sphn
    from merge_lora import merge
    from moshi import offline as native
    from safetensors import safe_open
    with safe_open(str(checkpoint / 'lora.safetensors'), framework='pt', device='cpu') as adapter:
        keys = list(adapter.keys())
        allowed = ('.lora_A.weight', '.lora_B.weight', '.frozen_W.weight')
        if not keys or any(not key.endswith(allowed) for key in keys):
            raise ValueError('STOP: empty adapter or keys unsupported by native merger')
        if not any(key.endswith('.lora_B.weight') for key in keys):
            raise ValueError('STOP: adapter contains no LoRA B tensors')
    audio, _ = sphn.read(str(paths['input_wav']))
    if audio.ndim != 2 or audio.shape[0] != 1 or not audio.shape[-1]:
        raise ValueError('input WAV must be nonempty mono user audio, not stereo conversation')
    output = opts.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAIL', backend='bundled PersonaPlex offline (not training LMGen parity)',
                  checkpoint=str(checkpoint), inputs={k: str(v) for k, v in paths.items()})
    try:
        if opts.native_weight is None:
            signature = dict(inputs=report['inputs'], seed=values.get('seed', 0), greedy=opts.greedy,
                             base_weights=str(assets.moshi_weights))
            import hashlib
            signature['input_hashes'] = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
            if opts.baseline_dir:
                baseline = opts.baseline_dir.resolve()
                prior = json.loads(require_file(baseline / 'manifest.json').read_text())
                if prior.get('comparison_signature') != signature:
                    raise ValueError('baseline input/seed/generation/base model mismatch')
                shutil.copytree(baseline / 'base', output / 'base')
            # Each child loads one model and exits before the next starts.
            for route, weight in (('base', assets.moshi_weights), ('current', None)):
                if route == 'base' and opts.baseline_dir:
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), '--config', str(opts.config.resolve()),
                       '--checkpoint', str(checkpoint), '--output-dir', str(output / route),
                       '--input-wav', str(paths['input_wav']), '--voice-prompt', str(paths['voice_prompt']),
                       '--text-prompt-file', str(paths['text_prompt_file']),
                       '--native-weight', str(weight) if weight else 'MERGE']
                if opts.greedy:
                    cmd.append('--greedy')
                with (output / f'{route}.log').open('w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{route} inference failed; see {route}.log')
            import numpy as np
            loaded = {}
            for name, path in [('original', paths['original_wav']), ('user', paths['input_wav']),
                               ('base', output / 'base/agent.wav'), ('current', output / 'current/agent.wav')]:
                data, rate = sphn.read(str(path))
                if rate != 24000:
                    data = sphn.resample(data, src_sample_rate=rate, dst_sample_rate=24000)
                loaded[name] = data
            base_dialogue, step_dialogue = dialogue_arrays(**loaded)
            for filename, data in [('dialogue_original.wav', loaded['original']),
                                   ('dialogue_base.wav', base_dialogue), ('dialogue_step.wav', step_dialogue)]:
                sphn.write_wav(str(output / filename), np.ascontiguousarray(data), 24000)
            hypotheses = {}
            for route in ('base', 'current'):
                hypotheses[route] = text_from_pieces(json.loads((output / route / 'agent_text.json').read_text()))
            reference = paths['reference_text_file'].read_text().strip() if 'reference_text_file' in paths else None
            manifest = dict(sample_id=infer.get('sample_id', paths['original_wav'].stem),
                step=opts.step if opts.step is not None else infer.get('step'), checkpoint=str(checkpoint),
                comparison_signature=signature,
                hypothesis=hypotheses['current'], transcript=hypotheses['current'],
                base_hypothesis=hypotheses['base'], reference=reference, raw_reference=reference,
                cer=None, wer=None, metrics_status='not computed; native token text is not ASR scoring',
                window_start_sec=infer.get('window_start_sec', 0.0),
                window_duration_sec=loaded['original'].shape[-1] / 24000,
                sample_rate=24000, channels={'left': 'agent', 'right': 'user'},
                seed=values.get('seed', 0), generation=dict(greedy=opts.greedy, temp_audio=.8,
                    temp_text=.7, topk_audio=250, topk_text=25),
                inputs=report['inputs'], base_weights=str(assets.moshi_weights),
                audio_files=dict(original='dialogue_original.wav', base='dialogue_base.wav',
                                 current_step='dialogue_step.wav'))
            manifest['window_end_sec'] = manifest['window_start_sec'] + manifest['window_duration_sec']
            write_json(output / 'manifest.json', manifest)
            report.update(status='PASS', manifest=str(output / 'manifest.json'))
            return 0
        merged = output / 'merged_model.safetensors'
        captured = io.StringIO()
        # Native merger warns and skips unsupported groups: treat any warning as failure.
        with contextlib.redirect_stdout(captured):
            if str(opts.native_weight) == 'MERGE':
                merge(checkpoint, merged, base_weights=str(assets.moshi_weights), dtype=torch.bfloat16)
            else:
                merged = require_file(opts.native_weight)
        merge_log = captured.getvalue()
        (output / 'merge.log').write_text(merge_log)
        print(merge_log)
        if 'WARNING:' in merge_log:
            raise RuntimeError('STOP: native merge skipped weights; see merge.log')
        # offline.run_inference unconditionally fetches config for download counting.
        # Redirect only this local asset lookup; generation/model semantics stay native.
        original_download = native.hf_hub_download
        def local_config(repo, filename):
            if filename != 'config.json' or assets.config is None:
                raise RuntimeError(f'STOP: unexpected native download request: {filename}')
            return str(assets.config)
        native.hf_hub_download = local_config
        try:
            with torch.no_grad():
                native.run_inference(
                    input_wav=str(paths['input_wav']), output_wav=str(output / 'agent.wav'),
                    output_text=str(output / 'agent_text.json'),
                    text_prompt=paths['text_prompt_file'].read_text(),
                    voice_prompt_path=str(paths['voice_prompt']), tokenizer_path=str(assets.tokenizer),
                    moshi_weight=str(merged), mimi_weight=str(assets.mimi_weights),
                    hf_repo=values['moshi_paths'].get('hf_repo_id'), device='cuda',
                    seed=values.get('seed', 0), temp_audio=.8, temp_text=.7,
                    topk_audio=250, topk_text=25, greedy=opts.greedy,
                    save_voice_prompt_embeddings=False, cpu_offload=False)
        finally:
            native.hf_hub_download = original_download
        require_file(output / 'agent.wav')
        tokens = json.loads(require_file(output / 'agent_text.json').read_text())
        import numpy as np
        generated, sr = sphn.read(str(output / 'agent.wav'))
        if not np.isfinite(generated).all() or not generated.size or not tokens:
            raise RuntimeError('invalid native audio/text outputs')
        report.update(status='PASS', sample_rate=sr, samples=generated.shape[-1],
                      text_token_count=len(tokens), merged_weights=str(merged))
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        write_json(output / 'summary.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())