"""Compare one actual Tim trainer step through native discovery vs train_local."""
import json
from _common import parser, prepare_fixture, launch_worker, write_json, compare_tensors


def main():
    p = parser(__doc__)
    opts = p.parse_args()
    config = prepare_fixture(opts.config, opts.output_dir, sample_index=opts.sample_index)
    out = config.parent
    mismatches = []
    try:
        for route in ('original', 'bridge'):
            launch_worker(config, out, route, stop_step=1)
        import torch
        a = torch.load(out / 'original/state.pt', weights_only=False)
        b = torch.load(out / 'bridge/state.pt', weights_only=False)
        for route, state in (('original',a), ('bridge',b)):
            for field in ('exp_avg','exp_avg_sq'):
                if not any(n.endswith('.'+field) for n in state['optimizer']):
                    mismatches.append(route+': missing optimizer '+field)
            if not any(not torch.equal(state['initial'][n], state['final'][n])
                       for n in state['initial'] if 'lora_B' in n):
                mismatches.append(route+': no LoRA B update after optimizer step')
        # Same-device seeded paths should be identical: tight FP32 tolerance;
        # BF16 parameters exact to avoid swallowing a one-step update.
        diffs = {}
        for category in ('initial', 'final', 'optimizer'):
            diffs[category] = compare_tensors(a[category], b[category],
                atol=1e-8 if category == 'optimizer' else 0.,
                rtol=1e-6 if category == 'optimizer' else 0.)
            mismatches += [category+':'+k for k,v in diffs[category].items() if not v['pass_']]
        deltas = [{k: x['final'][k].float()-x['initial'][k].float() for k in x['initial']} for x in (a,b)]
        diffs['parameter_delta'] = compare_tensors(*deltas)
        mismatches += ['delta:'+k for k,v in diffs['parameter_delta'].items() if not v['pass_']]
        if a['scheduler'] != b['scheduler']:
            mismatches.append('scheduler state')
        if a['trainable'] != b['trainable']:
            mismatches.append('trainable coverage/dtype/shape')
        reports = [json.loads((out / (route+'.json')).read_text()) for route in ('original','bridge')]
        for key in ('lr_before','lr_after'):
            if reports[0]['metrics'][key] != reports[1]['metrics'][key]:
                mismatches.append('exact '+key)
        def compare_scalar(x,y,path):
            if isinstance(x, dict) and isinstance(y, dict):
                if x.keys() != y.keys():
                    mismatches.append(path+':keys')
                for k in x.keys() & y.keys():
                    compare_scalar(x[k], y[k], path+'.'+k)
            elif isinstance(x, (int,float)) and isinstance(y,(int,float)):
                import math
                if not (math.isfinite(x) and math.isfinite(y)) or abs(x-y) > 1e-6 + 1e-5*abs(y):
                    mismatches.append(f'{path}: {x} != {y}')
            elif x != y:
                mismatches.append(path)
        compare_scalar(reports[0]['metrics'], reports[1]['metrics'], 'metrics')
        # Require actual first/middle/last transformer A/B coverage, not arbitrary tensors.
        import re
        layers = sorted({int(m.group(1)) for n in a['initial']
                         if (m := re.search(r'(?<!depformer\.)transformer\.layers\.(\d+)\.', n))
                         and 'depformer' not in n})
        if len(layers) < 3:
            mismatches.append('cannot identify >=3 transformer LoRA layers')
        selected = [layers[i] for i in (0,len(layers)//2,-1)] if layers else []
        for layer in selected:
            for kind in ('lora_A','lora_B'):
                if not any(f'transformer.layers.{layer}.' in n and kind in n for n in a['initial']):
                    mismatches.append(f'missing {kind} transformer layer {layer}')
        import yaml
        if yaml.safe_load(config.read_text())['lora'].get('skip_depformer', False):
            for route, x in zip(('original','bridge'), (a,b)):
                for n in x['initial']:
                    if 'depformer' in n and not torch.equal(x['initial'][n], x['final'][n]):
                        mismatches.append(route+': depformer changed: '+n)
        # Batch equality includes prompt/context masks; trusted local artifacts only.
        batches = [torch.load(out / route / 'first_batch.pt', weights_only=False) for route in ('original','bridge')]
        if not torch.equal(batches[0].codes.cpu(), batches[1].codes.cpu()):
            mismatches.append('input batch codes')
        if batches[0].prompt_lengths != batches[1].prompt_lengths:
            mismatches.append('input batch prompt lengths')
        masks = [x.context_masks for x in batches]
        if (masks[0] is None) != (masks[1] is None) or (masks[0] is not None and not torch.equal(masks[0].cpu(), masks[1].cpu())):
            mismatches.append('input context masks')
        write_json(out / 'tensor_diff.json', dict(selected_layers=selected, comparisons=diffs))
    except Exception as exc:
        mismatches.append(str(exc))
    write_json(out / 'comparison.json', dict(status='FAIL' if mismatches else 'PASS', mismatches=mismatches,
        tolerance=dict(parameters='exact (including BF16)', delta='exact', optimizer='atol=1e-8 rtol=1e-6',
                       losses_grad_norm='atol=1e-6 rtol=1e-5', scheduler='exact')))
    print('ONE-STEP PARITY: '+('FAIL' if mismatches else 'PASS'))
    for mismatch in mismatches:
        print(mismatch)
    return bool(mismatches)


if __name__ == '__main__':
    raise SystemExit(main())