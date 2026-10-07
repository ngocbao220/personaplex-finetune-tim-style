"""Export prepared sources to Tim sidecars; no model/GPU or raw-data preparation."""

import argparse
from pathlib import Path

import yaml

from tim_compat.prepared_data import prepare_manifest
from tim_compat.training_config import bind_training_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        help="Optional reference YAML; emit derived YAML with only train_data changed")
    parser.add_argument("--workers", type=int, default=0,
                        help="Spawn validation workers; default uses serial reference bridge")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional content-addressed export cache outside source root")
    parser.add_argument('--vietnamese-text-mode', choices=('diacritics', 'no_diacritics', 'telex'),
                        help='Agent target representation; defaults to data.vietnamese_text_mode in --config')
    args = parser.parse_args()
    from tim_compat.text_normalization import text_mode_from_config
    values = yaml.safe_load(args.config.read_text()) if args.config else {}
    mode = args.vietnamese_text_mode or text_mode_from_config(values)
    if args.config and mode != text_mode_from_config(values):
        parser.error('--vietnamese-text-mode must match data.vietnamese_text_mode in --config')
    result = prepare_manifest(args.manifest, args.output, workers=args.workers,
                              cache_dir=args.cache_dir, vietnamese_text_mode=mode)
    print(f"Vietnamese text mode: {mode}")
    print(f"Set original Tim config data.train_data to: {result}")
    print("Enable system_prompt.enable to use the prepared voice/text conditioning.")
    if args.config is not None:
        config = bind_training_manifest(args.config, result, args.output / 'train.yaml')
        print(f"Derived reference config: {config.resolve()}")


if __name__ == "__main__":
    main()