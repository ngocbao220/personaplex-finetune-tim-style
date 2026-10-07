"""Acceptance infrastructure only; no model, placement or training implementation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'moshi-finetune'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    return p


def read_config(path):
    import yaml
    values = yaml.safe_load(Path(path).read_text())
    acceptance = values.pop('acceptance', {})
    if not acceptance.get('model_root') or not acceptance.get('prepared_manifest'):
        raise ValueError('config requires acceptance.model_root and acceptance.prepared_manifest (absolute paths)')
    for key in ('model_root', 'prepared_manifest'):
        if not Path(acceptance[key]).is_absolute():
            raise ValueError(f'acceptance.{key} must be absolute')
    return values, acceptance


def prepare_fixture(config, output, num_samples=1, sample_index=0, max_steps=None, checkpoint_step=None):
    import yaml
    from tim_compat.prepared_data import prepare_manifest
    from tim_compat.local_checkpoint import LocalAssets
    values, acceptance = read_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'use a fresh output directory: {output}')
    rows = [json.loads(x) for x in Path(acceptance['prepared_manifest']).read_text().splitlines() if x.strip()]
    selected = rows[sample_index:sample_index + num_samples]
    if sample_index < 0 or num_samples < 1 or len(selected) != num_samples:
        raise ValueError('sample selection outside prepared manifest: '
                         f'total={len(rows)}, sample_index={sample_index}, num_samples={num_samples}')
    output.mkdir(parents=True, exist_ok=False)
    # Preserve prepared-root path resolution; export first, select exported rows.
    exported = prepare_manifest(acceptance['prepared_manifest'], output / 'prepared')
    tim_rows = [json.loads(x) for x in exported.read_text().splitlines() if x.strip()]
    manifest = output / 'selected.jsonl'
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in tim_rows[sample_index:sample_index+num_samples]))
    assets = LocalAssets.resolve(acceptance['model_root'], **{
        k: values.get('moshi_paths', {}).get(k) for k in
        ('moshi_path', 'mimi_path', 'tokenizer_path', 'config_path')})
    values.setdefault('moshi_paths', {}).update(moshi_path=str(assets.moshi_weights),
        mimi_path=str(assets.mimi_weights), tokenizer_path=str(assets.tokenizer))
    values.setdefault('data', {}).update(train_data=str(manifest), eval_data=str(manifest), shuffle=False)
    if max_steps is not None:
        values['max_steps'] = max_steps
    if not values.get('lora', {}).get('enable') or values.get('full_finetuning', False):
        raise ValueError('LoRA-only acceptance required')
    if values.get('lora', {}).get('ft_embed', False):
        raise ValueError('STOP: embedding tuning outside this LoRA-only checkpoint acceptance coverage')
    if values.get('neftune_alpha', 0) != 0:
        raise ValueError('STOP: augmentation enabled; provide deterministic acceptance config')
    values.update(run_dir=str(output / 'tim_run'), overwrite_run_dir=False, log_freq=1,
        do_eval=False, skip_zero_eval=True, do_ckpt=True, save_adapters=True,
        ckpt_freq=checkpoint_step or values['max_steps'], num_ckpt_keep=None)
    values.setdefault('gen_eval', {})['enable'] = False
    values.setdefault('wandb', {})['project'] = None
    resolved = output / 'resolved.yaml'
    resolved.write_text(yaml.safe_dump(values, sort_keys=False))
    write_json(output / 'fixture.json', dict(acceptance=acceptance, samples=selected,
        overrides='ordered sample subset, no shuffle, logging/checkpoint/eval routing; optimizer/scheduler/precision unchanged',
        config=str(Path(config).resolve()), resolved=str(resolved)))
    return resolved


def launch_worker(config, output, route, stop_step=0, reload=None):
    import yaml
    values = yaml.safe_load(Path(config).read_text())
    if route == 'original':
        values['run_dir'] = str(Path(output).resolve() / 'original_tim_run')
    routed_config = Path(output) / (route+'_config.yaml')
    routed_config.write_text(yaml.safe_dump(values, sort_keys=False))
    cmd = [sys.executable, str(ROOT / 'scripts/_tim_worker.py'), '--config', str(routed_config),
           '--output', str(output), '--route', route, '--stop-step', str(stop_step)]
    if reload:
        cmd += ['--reload', str(reload)]
    with (Path(output) / ('reload.log' if reload else route+'.log')).open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'worker failed ({result.returncode}); see {log.name}')


def compare_tensors(a, b, *, atol=0., rtol=0.):
    import torch
    results = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            results[key] = dict(pass_=False, reason='missing key')
            continue
        x, y = a[key], b[key]
        same_shape = x.shape == y.shape and x.dtype == y.dtype
        results[key] = dict(pass_=same_shape and torch.allclose(x, y, atol=atol, rtol=rtol),
            max_abs=float((x.float()-y.float()).abs().max()) if same_shape and x.numel() else None)
    return results


def offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('single GPU only')
