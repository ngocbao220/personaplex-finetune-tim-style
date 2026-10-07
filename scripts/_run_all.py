"""Fail-fast orchestration; always publish statuses, never infer a GPU pass."""
import json
import subprocess
import sys
from pathlib import Path
from _common import ROOT, parser, write_json


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    opts = p.parse_args()
    checks = {k: 'SKIPPED' for k in ('Real interleaver parity','Cache parity',
        'One-step loss parity','One-step parameter parity','Optimizer parity','Scheduler parity',
        'Tim mini-overfit','Checkpoint reload parity','Native free-running smoke')}
    phases = [('check_interleaver_parity.py', opts.runs_dir / 'parity/interleaver', list(checks)[:2]),
              ('check_one_step_parity.py', opts.runs_dir / 'parity/one_step', list(checks)[2:6]),
              ('run_gpu_acceptance.py', opts.runs_dir / 'gpu_acceptance', list(checks)[6:8])]
    failed = False
    native_smoke = 'SKIPPED: configure acceptance.inference input_wav, voice_prompt, text_prompt_file'
    try:
        for script, output, names in phases:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                '--config', str(opts.config.resolve()), '--output-dir', str(output.resolve())])
            for name in names:
                checks[name] = 'FAIL' if result.returncode else 'PASS'
            if script == 'run_gpu_acceptance.py' and (output / 'comparison.json').is_file():
                evidence = json.loads((output / 'comparison.json').read_text())
                checks['Tim mini-overfit'] = evidence['mini_overfit']
                checks['Checkpoint reload parity'] = evidence['checkpoint_reload']
            if result.returncode:
                failed = True
                break
        if not failed:
            from _common import read_config
            _, acceptance = read_config(opts.config)
            infer = acceptance.get('inference', {})
            if infer:
                evidence = json.loads((opts.runs_dir / 'gpu_acceptance/comparison.json').read_text())
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_free_running_inference.py'),
                    '--config', str(opts.config.resolve()), '--checkpoint', evidence['checkpoint'],
                    '--output-dir', str((opts.runs_dir / 'free_running').resolve())])
                checks['Native free-running smoke'] = 'FAIL' if result.returncode else 'PASS'
                native_smoke = 'Bundled PersonaPlex offline, merged Tim adapter; not training LMGen parity'
                failed = bool(result.returncode)
    except Exception as exc:
        failed = True
        print(str(exc), file=sys.stderr)
    finally:
        opts.runs_dir.mkdir(parents=True, exist_ok=True)
        write_json(opts.runs_dir / 'gpu_acceptance_report.json', dict(checks=checks,
            native_smoke=native_smoke,
            config=str(opts.config.resolve())))
        print('\nCheck                         Status\n----------------------------------------')
        for name,status in checks.items():
            print(f'{name:30} {status}')
    return failed


if __name__ == '__main__':
    raise SystemExit(main())