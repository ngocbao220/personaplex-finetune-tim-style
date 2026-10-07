# Local-only checkpoint bridge

Provide one absolute model directory with three nonempty files:

```text
/home/data/models/personaplex-7b-v1/
  model.safetensors
  tokenizer-e351c8d8-checkpoint125.safetensors
  tokenizer_spm_32k_3.model
```

`config.json` is not required. Explicit `moshi_paths.config_path` is optional. Individual file overrides in the original Tim config are supported (relative overrides resolve under model root).

## Launch (original Tim data/config contract only)

```bash
torchrun --standalone --nproc-per-node=1 \
  /path/to/personaplex-finetune-tim-style/train_local.py \
  --model-root /home/data/models/personaplex-7b-v1 \
  --config /path/to/original-tim-config.yaml
```

This bridge is not the requested Hydra/prepared-data/DDP trainer. Multi-rank execution still follows original Tim FSDP behavior. Do not use it with prepared manifests yet.

The bridge temporarily replaces only checkpoint discovery (`CheckpointInfo.from_hf_repo`) while executing the unchanged Tim trainer. It restores the original discovery method even on exceptions. Assets are resolved explicitly without Hub calls; offline environment variables are set before importing the trainer.

The installed **training** Moshi loader must support `lm_kwargs` and `lm_kwargs_overrides`. The copied inference-only PersonaPlex loader is deliberately rejected: it cannot accept Tim's construction-time LoRA overrides. No fallback to that loader is performed.

Model defaults are delegated to the training loader; dep_q=16 and Mimi codebooks=8 reproduce the loader patches documented by Tim in `refs/personaplex-finetune/docs/SETUP_GUIDE.md`, Phase 8. Full explicit configs override defaults; sparse PersonaPlex metadata does not. LoRA rank/scaling/checkpointing overrides from `get_fsdp_model()` pass through unchanged. Tim still loads base weights and initializes/freezes LoRA through its original wrapper.

## Verification / limitations

```bash
cd /path/to/personaplex-finetune-tim-style
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Tests verify missing/empty files, no config requirement, sparse config, override forwarding, Mimi codebook count, discovery restoration and frozen-source hashes. They do not validate safetensors contents, real GPU loading, or training-runtime version parity. The original dependency is an unpinned Git Moshi package, and Tim's patched training environment is not vendored in the reference. That environment still needs to be pinned/audited before claiming training parity.

The context manager now supports `enabled=False`: no patching or validation occurs, and original reference discovery calls/results are preserved by a regression test. Construction-call parity is separately checked against SETUP_GUIDE Phase 8's documented patched-loader contract for loaded and meta/no-weight construction. This is not numerical model parity or proof that the installed unpatched runtime is Tim's exact runtime.