# Repository Guidelines

## Project Structure & Module Organization

This repository adapts Tim’s PersonaPlex LoRA trainer for prepared OtoSpeech conversations. Add local integration logic in `tim_compat/`; root entrypoints include `prepare_data.py`, `train_local.py`, and optional `train_hydra.py`. GPU validation and inference tools live in `scripts/`, CPU regressions in `tests/`, and workflow documentation in `docs/`. `config.full.yaml` configures the acceptance harness; `configs/` contains reference recipes.

`moshi-finetune/`, `personaplex/`, and inventoried upstream utilities are preserved references. Check `reference_files.sha256.json` before editing them; extend bridges instead of rewriting frozen sources. Keep datasets, model snapshots, and run artifacts outside the source tree.

## Build, Test, and Development Commands

Run commands from the repository root using an existing compatible environment (Python ≥ 3.10). Dependency declarations are in `moshi-finetune/pyproject.toml`; training requires a Tim-compatible Moshi runtime and Linux/CUDA.

- `python3 -m unittest discover -s tests -v`: run CPU contract and regression tests.
- `python3 tests/test_reference_integrity.py`: verify preserved source hashes.
- `python prepare_data.py --manifest /absolute/prepared/manifest.jsonl --output /absolute/exports/tim-prepared`: validate/export prepared data for Tim.
- `CUDA_VISIBLE_DEVICES=0 python scripts/_run_all.py --config config.full.yaml --runs-dir /absolute/runs/new_checks`: run ordered GPU checks after configuring local assets. Use a fresh output directory.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Match neighboring code and explain non-obvious contracts in docstrings. The reference package declares Ruff, Black/isort settings with an 88-column target; avoid repository-wide formatting of frozen files.

## Testing Guidelines

Use `unittest` with files named `tests/test_<feature>.py` and methods named `test_<behavior>`. Add focused regressions for changed bridge behavior, including invalid inputs and artifact preservation. No numeric coverage threshold is configured. CPU tests do not establish successful GPU training, native generation, or offload/restore; record actual GPU evidence separately.

## Commit & Pull Request Guidelines

History currently contains only `first commit`, so no established message convention exists. Use concise imperative subjects. PRs should explain the behavior changed, relevant issue, exact validation commands/results, configuration requirements, and remaining GPU limitations. Preserve unrelated work.

## Data & Configuration Safety

Require local model assets without silent downloads. Prepared audio is 24 kHz stereo: LEFT=agent, RIGHT=user. Mask voice/text conditioning prompts from loss. Keep acceptance YAML separate from native trainer YAML, preserve prepared sources, and retain diagnostic reports when checks fail.
