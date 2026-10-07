"""Export prepared sources to Tim sidecars; no model/GPU or raw-data preparation."""

import argparse
from pathlib import Path

import yaml

from tim_compat.prepared_data import prepare_manifest, validate_eval_split
from tim_compat.training_config import bind_training_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        help="Read text mode, seed and data.eval_split_from_train from this YAML")
    parser.add_argument("--resolved-config", type=Path,
                        help="Explicit output YAML with exported train/val paths; requires --config; never overwritten")
    parser.add_argument("--workers", type=int, default=0,
                        help="Spawn validation workers; default uses serial reference bridge")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional content-addressed export cache outside source root")
    parser.add_argument('--vietnamese-text-mode', choices=('diacritics', 'no_diacritics', 'telex'),
                        help='Agent target representation; defaults to data.vietnamese_text_mode in --config')
    args = parser.parse_args()
    if args.resolved_config is not None:
        if args.config is None:
            parser.error('--resolved-config requires --config')
        if args.resolved_config.exists():
            parser.error(f'resolved config already exists: {args.resolved_config}')
        if not args.resolved_config.parent.is_dir():
            parser.error(f'resolved config directory does not exist: {args.resolved_config.parent}')
    from tim_compat.text_normalization import text_mode_from_config
    values = yaml.safe_load(args.config.read_text()) if args.config else {}
    mode = args.vietnamese_text_mode or text_mode_from_config(values)
    if args.config and mode != text_mode_from_config(values):
        parser.error('--vietnamese-text-mode must match data.vietnamese_text_mode in --config')
    ratio = validate_eval_split(values.get('data', {}).get('eval_split_from_train', 0))
    if ratio and values['data'].get('eval_data', '').strip():
        parser.error('eval_split_from_train conflicts with explicit data.eval_data')
    result = prepare_manifest(args.manifest, args.output, workers=args.workers,
                              cache_dir=args.cache_dir, vietnamese_text_mode=mode,
                              eval_split_from_train=ratio, seed=values.get('seed', 0))
    print(f"Vietnamese text mode: {mode}")
    print(f"Set original Tim config data.train_data to: {result}")
    if ratio:
        print(f"Set original Tim config data.eval_data to: {result.parent / 'val.jsonl'}")
    print("Enable system_prompt.enable to use the prepared voice/text conditioning.")
    if args.resolved_config is not None:
        config = bind_training_manifest(args.config, result, args.resolved_config,
                                       eval_manifest=result.parent / 'val.jsonl' if ratio else None)
        print(f"Derived reference config: {config.resolve()}")


if __name__ == "__main__":
    main()
