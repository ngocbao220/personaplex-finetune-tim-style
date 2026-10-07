"""Export prepared sources to Tim sidecars; no model/GPU or raw-data preparation."""

import argparse
from pathlib import Path

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
    args = parser.parse_args()
    result = prepare_manifest(args.manifest, args.output, workers=args.workers,
                              cache_dir=args.cache_dir)
    print(f"Set original Tim config data.train_data to: {result}")
    print("Enable system_prompt.enable to use the prepared voice/text conditioning.")
    if args.config is not None:
        config = bind_training_manifest(args.config, result, args.output / 'train.yaml')
        print(f"Derived reference config: {config.resolve()}")


if __name__ == "__main__":
    main()