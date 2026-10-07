"""Opt-in synchronous evaluation hook; copied trainer remains unchanged."""
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager


def frequency(settings):
    value = settings.get('free_running_every_steps', 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('free_running_every_steps must be a nonnegative integer')
    return value


def due(step, every):
    return every > 0 and (step == 0 or step % every == 0)


@contextmanager
def cpu_training_state(model, optimizer, mimi):
    """Move tensor storage, not Parameter identities; restore even on child failure."""
    import torch
    slots = []
    seen = set()

    def move(owner, key, mapping=False):
        identity = (id(owner), key)
        if identity in seen:
            return
        seen.add(identity)
        tensor = owner[key] if mapping else getattr(owner, key)
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cuda':
            return
        device = tensor.device
        cpu = tensor.to('cpu')
        if mapping:
            owner[key] = cpu
        else:
            setattr(owner, key, cpu)
        slots.append((owner, key, mapping, device))

    try:
        torch.cuda.synchronize()
        for module in (model, mimi):
            for p in module.parameters():
                move(p, 'data')
                if p.grad is not None:
                    move(p.grad, 'data')
                for attr in ('_mp_param', '_temp'):
                    if hasattr(p, attr):
                        move(p, attr)
            for child in module.modules():
                for key in child._buffers:
                    move(child._buffers, key, True)
        for state in optimizer.state.values():
            for key in state:
                move(state, key, True)
        torch.cuda.empty_cache()
        yield
    finally:
        for owner, key, mapping, device in reversed(slots):
            tensor = owner[key] if mapping else getattr(owner, key)
            if mapping:
                owner[key] = tensor.to(device)
            else:
                setattr(owner, key, tensor.to(device))


@contextmanager
def periodic_free_running(module, config_path, settings, *, evaluator=None):
    every = frequency(settings)
    if not every:
        yield
        return
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('periodic free-running supports single GPU only')
    import yaml
    values = yaml.safe_load(Path(config_path).read_text())
    if values.get('full_finetuning') or not values.get('lora', {}).get('enable'):
        raise ValueError('periodic free-running requires LoRA-only training')
    if not values.get('do_ckpt') or values.get('gen_eval', {}).get('enable'):
        raise ValueError('require do_ckpt=true and gen_eval.enable=false')
    source = Path(module.__file__).resolve()
    lines = source.read_text().splitlines()

    def anchor(text):
        matches = [i + 1 for i, line in enumerate(lines) if line.strip() == text]
        if len(matches) != 1:
            raise RuntimeError(f'unsupported trainer hook anchor: {text}')
        return matches[0]

    initial = anchor('while state.step < args.max_steps:')
    completed = anchor('loss_item = loss.item()')
    previous = sys.gettrace()
    previous_local = {}
    visited = set()
    baseline = None
    initialized = False
    root = Path(__file__).resolve().parents[1]
    output = Path(values['run_dir']).resolve() / 'free_running'

    def evaluate(f):
        nonlocal baseline
        import torch
        import safetensors.torch
        step = f['state'].step
        checkpoint = output / f'adapter_{step:06d}'
        checkpoint.mkdir(parents=True, exist_ok=False)
        with torch.no_grad():
            states = f['checkpointer'].retrieve_save_states(True, f['param_dtype'])
            states = {key: value.detach().cpu().contiguous() for key, value in states.items()}
            safetensors.torch.save_file(states, str(checkpoint / 'lora.safetensors'))
        (checkpoint / 'config.json').write_text(json.dumps(f['checkpointer'].config))
        del states
        result_dir = output / f'step_{step:06d}'
        cmd = [sys.executable, str(root / 'scripts/run_free_running_inference.py'),
               '--config', str(Path(config_path).resolve()), '--checkpoint', str(checkpoint),
               '--output-dir', str(result_dir), '--step', str(step)]
        if baseline:
            cmd += ['--baseline-dir', str(baseline)]
        print(f'Free-running evaluation at optimizer step {step}: {result_dir}', flush=True)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with cpu_training_state(f['model'], f['optimizer'], f['mimi']):
                with (output / f'step_{step:06d}.log').open('w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'free-running failed at step {step}; see {log.name}')
        if baseline is None:
            baseline = result_dir

    def trace(frame, event, arg):
        nonlocal initialized
        # Chain the acceptance observer, including its frame-local trace function.
        old = previous_local.get(id(frame), previous)
        if old:
            previous_local[id(frame)] = old(frame, event, arg)
        if event == 'return':
            previous_local.pop(id(frame), None)
        if event == 'line' and Path(frame.f_code.co_filename).resolve() == source:
            if frame.f_lineno in (initial, completed):
                step = frame.f_locals['state'].step
                if step not in visited and (due(step, every) or (not initialized and frame.f_lineno == initial)):
                    (evaluator or evaluate)(frame.f_locals)
                    visited.add(step)
                    initialized = True
        return trace

    try:
        sys.settrace(trace)
        yield
    finally:
        sys.settrace(previous)