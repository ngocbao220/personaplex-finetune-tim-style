"""Fresh-process observation of original Tim _train, never a substitute trainer.

Trace anchors are source statements, not hard-coded line numbers. Observations
do not replace optimizer/scheduler/loss calls. An exception stops AFTER a full
step for one-step testing without changing the OneCycle total_steps horizon.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from _common import ROOT, offline, write_json


class StepComplete(Exception):
    pass


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--route', choices=['original', 'bridge'], required=True)
    p.add_argument('--stop-step', type=int, default=0)
    p.add_argument('--reload', type=Path)
    opts = p.parse_args()
    offline()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES=0')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.environ.update(LOCAL_RANK='0', RANK='0', WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    sys.path.insert(0, str(ROOT / 'moshi-finetune'))
    from moshi.models import loaders
    from finetune.eval import evaluate
    from finetune.args import TrainArgs
    from finetune.wrapped_model import get_fsdp_model
    from finetune.utils import TrainState, set_random_seed
    import torch.distributed as dist
    fixture = json.loads((opts.output / 'fixture.json').read_text())
    model_root = fixture['acceptance']['model_root']
    args = TrainArgs.load(str(opts.config), drop_extra_fields=False)
    artifacts = opts.output / opts.route
    artifacts.mkdir(exist_ok=True)

    def evaluation(model, batch):
        state = TrainState(args.max_steps)
        was_training = model.training
        # Preserve RNG state; call Tim's actual teacher-forced evaluator.
        with torch.random.fork_rng(devices=[0]):
            evaluate(model, iter([batch]), state, args)
        model.train(was_training)
        return dict(total=state.this_eval_loss, text=state.this_text_loss, audio=state.this_audio_loss,
                    definition='Tim evaluate: text + audio, independent of training audio_loss_weight/L2')

    if opts.reload:
        from tim_compat.local_checkpoint import local_checkpoint_loader
        with local_checkpoint_loader(loaders, model_root):
            dist.init_process_group('nccl')
            set_random_seed(args.seed)
            info = loaders.CheckpointInfo.from_hf_repo(hf_repo=args.moshi_paths.hf_repo_id,
                moshi_weights=args.moshi_paths.moshi_path, mimi_weights=args.moshi_paths.mimi_path,
                tokenizer=args.moshi_paths.tokenizer_path, config_path=args.moshi_paths.config_path)
            model = get_fsdp_model(args, info, resume_lora_path=str(opts.reload / 'lora.safetensors'))
            batch = torch.load(artifacts / 'last_batch.pt', weights_only=False)
            batch.codes = batch.codes.cuda()
            if batch.context_masks is not None:
                batch.context_masks = batch.context_masks.cuda()
            write_json(artifacts / 'reload.json', evaluation(model, batch))
            torch.save({n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'lora' in n},
                       artifacts / 'reload_tensors.pt')
            del model, batch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            dist.destroy_process_group()
        return

    source = ROOT / 'moshi-finetune/train.py'
    lines = source.read_text().splitlines()
    def anchor(text):
        hits = [i+1 for i, line in enumerate(lines) if line.strip() == text]
        if len(hits) != 1:
            raise RuntimeError(f'STOP: ambiguous Tim observation anchor {text}: {hits}')
        return hits[0]
    before_forward = anchor('output = model(codes=codes, condition_tensors=condition_tensors)')
    before_clip = anchor('torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)')
    after_step = anchor('loss_item = loss.item()')
    # Trace executable statement inside final checkpoint branch.
    checkpoint_branch = anchor('if args.do_ckpt and (')
    save_line = next(i+1 for i in range(checkpoint_branch, len(lines)) if lines[i].strip() == 'checkpointer.save_checkpoint(')
    captured = {}
    metrics = []
    def tensors(model):
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'lora' in n}
    def trace(frame, event, arg):
        if event != 'line' or frame.f_code.co_name != '_train' or Path(frame.f_code.co_filename) != source:
            return trace
        f = frame.f_locals
        model = f.get('model')
        if frame.f_lineno == before_forward and 'initial' not in captured:
            captured['initial'] = tensors(model)
            captured['trainable'] = {n: dict(shape=list(p.shape), dtype=str(p.dtype))
                for n, p in model.named_parameters() if p.requires_grad}
            assert captured['trainable'], 'no trainable parameters'
            assert all('lora' in n for n in captured['trainable']), 'base parameter not frozen'
            if args.lora.skip_depformer:
                assert not any('depformer' in n for n in captured['trainable']), 'depformer remains trainable'
            write_json(artifacts / 'trainable.json', captured['trainable'])
            first_batch = copy.copy(f['batch'])
            first_batch.codes = first_batch.codes.cpu()
            if first_batch.context_masks is not None:
                first_batch.context_masks = first_batch.context_masks.cpu()
            torch.save(first_batch, artifacts / 'first_batch.pt')
            if not opts.stop_step:
                from finetune.data.data_loader import build_data_loader
                def fixed_eval():
                    state = TrainState(args.max_steps)
                    loader = build_data_loader(f['interleaved_tokenizer'], args.data,
                        args.batch_size, None, 0, 1, True)
                    seen = [0]
                    def bounded():
                        for batch in loader:
                            seen[0] += 1
                            if seen[0] > 40:
                                raise RuntimeError('STOP: fixed set exceeds native Tim eval 40-batch cap')
                            yield batch
                    with torch.random.fork_rng(devices=[0]):
                        evaluate(model, bounded(), state, args)
                    if seen[0] == 0:
                        raise RuntimeError('STOP: no complete evaluation batch; reduce batch_size')
                    return dict(total=state.this_eval_loss, text=state.this_text_loss, audio=state.this_audio_loss)
                captured['fixed_eval'] = fixed_eval
                write_json(artifacts / 'baseline.json', fixed_eval())
        if frame.f_lineno == before_clip:
            grads = [p.grad.detach().float().norm().square() for p in model.parameters() if p.grad is not None]
            captured['grad_norm'] = torch.stack(grads).sum().sqrt().item()
            import math
            assert math.isfinite(captured['grad_norm']), 'nonfinite gradient norm'
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if 'lora' in n)
        if frame.f_lineno == after_step:
            step = f['state'].step
            row = dict(step=step, total=f['loss'].item(),
                text=f['text_loss_accum']/args.num_microbatches,
                audio=f['audio_loss_accum']/args.num_microbatches,
                per_codebook=f['per_cb_accum'].copy(), grad_norm=captured['grad_norm'],
                lr_before=f['last_lr'], lr_after=f['scheduler'].get_last_lr()[0],
                real_token_count=f['real_token_count_accum'], pad_token_count=f['pad_token_count_accum'])
            metrics.append(row)
            print(json.dumps(row), flush=True)
            write_json(artifacts / 'metrics.json', metrics)
            if opts.stop_step and step == opts.stop_step:
                captured['final'] = tensors(model)
                captured['optimizer'] = {n+'.'+key: value.detach().cpu().clone()
                    for n,p in model.named_parameters() for key,value in f['optimizer'].state.get(p, {}).items()
                    if isinstance(value, torch.Tensor)}
                captured['scheduler'] = f['scheduler'].state_dict()
                torch.save(captured, artifacts / 'state.pt')
                write_json(opts.output / (opts.route+'.json'), dict(metrics=row,
                    trainable=captured['trainable'], scheduler=captured['scheduler']))
                raise StepComplete()
        if frame.f_lineno == save_line and f['state'].step == args.max_steps:
            batch = f['batch']
            write_json(artifacts / 'pre_save.json', evaluation(model, batch))
            write_json(artifacts / 'final_eval.json', captured['fixed_eval']())
            torch.save(tensors(model), artifacts / 'pre_save_tensors.pt')
            batch = copy.copy(batch)
            batch.codes = batch.codes.cpu()
            if batch.context_masks is not None:
                batch.context_masks = batch.context_masks.cpu()
            torch.save(batch, artifacts / 'last_batch.pt')
        return trace

    from contextlib import ExitStack
    with ExitStack() as stack:
        if opts.route == 'original':
            # Independent native discovery with explicit paths; no LocalCheckpointInfo.
            # Offline native loader failure is a prerequisite failure, never a fallback.
            spec = importlib.util.spec_from_file_location('acceptance_original_tim', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            invoke = lambda: module.train(str(opts.config))
        else:
            import train_local
            bridge_args = ['--config', str(opts.config), '--model-root', model_root]
            acceptance = fixture['acceptance']
            if not opts.stop_step and acceptance.get('inference', {}).get('free_running_every_steps', 0):
                bridge_args += ['--free-running-config', fixture['config']]
            if acceptance.get('token_cache', False):
                bridge_args += ['--token-cache', str(opts.output / 'bridge_cache'),
                                '--token-report', str(artifacts / 'token_report.jsonl')]
            if acceptance.get('filter_policy'):
                bridge_args += ['--filter-policy', acceptance['filter_policy'],
                                '--filter-report-dir', str(artifacts / 'filter_reports')]
            invoke = lambda: train_local.main(bridge_args)
        try:
            sys.settrace(trace)
            invoke()
        except StepComplete:
            pass
        finally:
            sys.settrace(None)
            if dist.is_initialized():
                dist.destroy_process_group()
            import gc
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()