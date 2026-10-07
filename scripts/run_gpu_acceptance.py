"""Real Tim mini-overfit and fresh-process checkpoint reload acceptance."""
import json
from _common import parser, prepare_fixture, launch_worker, write_json, compare_tensors


def main():
    p = parser(__doc__)
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--checkpoint-step', type=int)
    p.add_argument('--required-loss-ratio', type=float, default=.8)
    p.add_argument('--smoke-chunks', type=int, default=None,
                   help='1..40 fixed valid chunks for smoke train/eval/reload; 0=full fixed set; overrides acceptance.smoke_chunks')
    opts = p.parse_args()
    from _common import read_config
    supplied, acceptance = read_config(opts.config)
    smoke_chunks = opts.smoke_chunks if opts.smoke_chunks is not None else acceptance.get('smoke_chunks', 0)
    if isinstance(smoke_chunks, bool) or not isinstance(smoke_chunks, int) or not 0 <= smoke_chunks <= 40:
        p.error('smoke-chunks must be an integer from 0 to 40')
    if supplied.get('batch_size', 1) != 1:
        p.error('fixed-set coverage requires batch_size=1 in supplied config; no silent batch-size override')
    if opts.max_steps < 2 or not 0 < opts.required_loss_ratio < 1:
        p.error('max_steps >=2 and 0 < required_loss_ratio <1 required')
    config = prepare_fixture(opts.config, opts.output_dir, opts.num_samples, opts.sample_index,
                             opts.max_steps, opts.checkpoint_step)
    out = config.parent
    fixture = json.loads((out / 'fixture.json').read_text())
    fixture['acceptance']['smoke_chunks'] = smoke_chunks
    write_json(out / 'fixture.json', fixture)
    report = dict(mini_overfit='FAIL', checkpoint_reload='FAIL', mismatches=[],
                  coverage='fixed_chunk_smoke' if smoke_chunks else 'full_selected_conversations')
    try:
        launch_worker(config, out, 'bridge')
        baseline = json.loads((out / 'bridge/baseline.json').read_text())
        final = json.loads((out / 'bridge/final_eval.json').read_text())
        if smoke_chunks:
            selection = json.loads((out / 'bridge/smoke_selection.json').read_text())
            report['smoke_selection'] = selection
            if any(row.get('samples_sha256') != selection['samples_sha256'] for row in (baseline, final)):
                raise ValueError('baseline/final smoke snapshot identity mismatch')
        report.update(baseline=baseline, final=final, ratio=final['total']/baseline['total'])
        if report['ratio'] < opts.required_loss_ratio:
            report['mini_overfit'] = 'PASS'
        else:
            report['mismatches'].append('fixed-set Tim eval loss did not fall by required ratio')
        checkpoint = out / 'tim_run/checkpoints' / f'checkpoint_{opts.max_steps:06d}' / 'consolidated'
        # Training subprocess has exited: all CUDA/runtime objects are gone.
        launch_worker(config, out, 'bridge', reload=checkpoint)
        import torch
        from safetensors.torch import load_file
        pre = torch.load(out / 'bridge/pre_save_tensors.pt', weights_only=False)
        post = torch.load(out / 'bridge/reload_tensors.pt', weights_only=False)
        diffs = compare_tensors(pre, post)
        expected = {k.replace('_checkpoint_wrapped_module.', '').replace('_fsdp_wrapped_module.', '')
                    for k in pre if k in json.loads((out / 'bridge/trainable.json').read_text())}
        saved = load_file(str(checkpoint / 'lora.safetensors'))
        key_mismatch = sorted(set(saved) ^ expected)
        train_state = torch.load(checkpoint / 'train_state.pt', weights_only=False)
        if not {'optimizer','scheduler','train_state'}.issubset(train_state):
            report['mismatches'].append('checkpoint missing Tim training state keys')
        before = json.loads((out / 'bridge/pre_save.json').read_text())
        after = json.loads((out / 'bridge/reload.json').read_text())
        if smoke_chunks and any(row.get('samples_sha256') != selection['samples_sha256']
                                for row in (before, after)):
            report['mismatches'].append('reload smoke snapshot identity mismatch')
        import math
        for key in ('total','text','audio'):
            if not (math.isfinite(before[key]) and math.isfinite(after[key])) or abs(before[key]-after[key]) > 1e-6 + 1e-5*abs(before[key]):
                report['mismatches'].append(f'reload {key}: {before[key]} != {after[key]}')
        if key_mismatch:
            report['mismatches'].append('checkpoint semantic keys differ: '+repr(key_mismatch))
        if not all(v['pass_'] for v in diffs.values()):
            report['mismatches'].append('LoRA tensors not identical after reload')
        reload_errors = [x for x in report['mismatches'] if 'fixed-set' not in x]
        if not reload_errors:
            report['checkpoint_reload'] = 'PASS'
        report.update(pre_save=before, post_reload=after, tensor_diff=diffs, checkpoint=str(checkpoint),
            tolerance='LoRA exact; teacher-forced Tim eval atol=1e-6 rtol=1e-5')
    except Exception as exc:
        report['mismatches'].append(str(exc))
    write_json(out / 'comparison.json', report)
    print(json.dumps(report, indent=2))
    return report['mini_overfit'] != 'PASS' or report['checkpoint_reload'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
