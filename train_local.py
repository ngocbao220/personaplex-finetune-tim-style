"""Local-only launch bridge to the unchanged Tim trainer (original YAML format)."""

import argparse
import importlib.util
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from tim_compat.local_checkpoint import LocalAssets, local_checkpoint_loader
from tim_compat.sample_filter import FilterPolicy, sample_filter_loader
from tim_compat.training_config import bind_training_manifest
from tim_compat.tokenization import tokenizer_bridge, digest_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--filter-policy", type=Path, default=None,
                        help="Explicit retained-Sample policy JSON; omitted = original loader")
    parser.add_argument("--filter-report-dir", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="Exported Tim manifest; bind to a fresh derived config")
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--token-report", type=Path)
    parser.add_argument("--free-running-config", type=Path,
                        help="Acceptance YAML containing inference inputs and periodic frequency")
    args = parser.parse_args(argv)
    LocalAssets.resolve(args.model_root)  # Fail before importing GPU training code.
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    policy = None
    if args.filter_policy is not None:
        policy = FilterPolicy(**json.loads(args.filter_policy.read_text()))
        if args.filter_report_dir is None:
            parser.error("--filter-policy requires --filter-report-dir")
    elif args.filter_report_dir is not None:
        parser.error("--filter-report-dir requires --filter-policy")
    if (args.train_manifest is None) != (args.resolved_config is None):
        parser.error("--train-manifest and --resolved-config must be specified together")
    if args.train_manifest is not None and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error("materialize config once before multi-rank launch; pass it with --config")
    config = bind_training_manifest(args.config, args.train_manifest, args.resolved_config,
                                    enabled=args.train_manifest is not None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "moshi-finetune"))
    from moshi.models import loaders

    spec = importlib.util.spec_from_file_location("tim_reference_train", root / "moshi-finetune/train.py")
    module = importlib.util.module_from_spec(spec)
    with ExitStack() as stack:
        stack.enter_context(local_checkpoint_loader(loaders, args.model_root))
        spec.loader.exec_module(module)
        if args.free_running_config is not None:
            import yaml
            from tim_compat.free_running import periodic_free_running
            settings = yaml.safe_load(args.free_running_config.read_text()).get('acceptance', {}).get('inference', {})
            # Use actual trainer run_dir/LoRA settings with external inference assets.
            inference_values = yaml.safe_load(args.free_running_config.read_text())
            native_values = yaml.safe_load(config.read_text())
            inference_values.update(native_values)
            import tempfile
            handle = stack.enter_context(tempfile.NamedTemporaryFile(mode='w', suffix='.yaml'))
            yaml.safe_dump(inference_values, handle)
            handle.flush()
            stack.enter_context(periodic_free_running(module, Path(handle.name), settings))
        if args.token_cache is not None or args.token_report is not None:
            if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                parser.error('token observation/cache is single-process until GPU parity passes')
            assets = LocalAssets.resolve(args.model_root)
            import hashlib
            fingerprint = hashlib.sha256(''.join(digest_file(path) for path in (
                assets.moshi_weights, assets.mimi_weights, assets.tokenizer,
                root / 'moshi-finetune/finetune/data/interleaver.py',
                root / 'tim_compat/tokenization.py', config)).encode()).hexdigest()
            report = None
            if args.token_report is not None:
                args.token_report.parent.mkdir(parents=True, exist_ok=True)
                report = stack.enter_context(args.token_report.open('x'))
            stack.enter_context(tokenizer_bridge(module, fingerprint,
                                                  cache_dir=args.token_cache, report=report))
        if policy is not None:
            from finetune.data import data_loader

            args.filter_report_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            report = stack.enter_context((args.filter_report_dir / f"rejects-rank-{rank}.jsonl").open("x"))
            stack.enter_context(sample_filter_loader(data_loader, policy, report=report))
        module.train(str(config.resolve()), resume_from=args.resume_from)


if __name__ == "__main__":
    main()