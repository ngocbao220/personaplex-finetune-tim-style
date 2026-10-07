# Tim reference baseline

Source: `refs/personaplex-finetune/` in the parent workspace.

Git revision: **d8c69e03390b2a97cfc926a2c52623e9b512b549**.
The source working tree was clean when inspected.

Copied unchanged:

- `moshi-finetune/`: trainer, data/interleaver, losses, optimizer/scheduler call sites, LoRA/model wrapper, checkpointing, monitoring and generation evaluation.
- `personaplex/moshi/`: native PersonaPlex Python runtime, model, LoRA modules, loaders, LMGen, packaging and licenses.
- `configs/`: original reference recipes.
- `pipeline/merge_lora.py`: original adapter merging utility.
- `LICENSE`, `THIRD_PARTY_NOTICES.md`: original license and attribution. Original reference README is preserved byte-for-byte as `README.upstream.md`; the user-facing `README.md` now documents the local OtoSpeech workflow.

Excluded: Git internals, bytecode/caches, frontend, VibeVoice, external data preparation pipeline, and historical documentation. Copied trainer/runtime files were not rewritten.

`reference_files.sha256.json` records original SHA-256 for every copied file. Markdown has been relocated under `docs/`, preserving original relative subdirectories and bytes. The integrity test maps original Markdown paths into `docs/` and its original `README.md` entry to `docs/README.upstream.md`, retaining original hashes and source comparisons. The root and scripts READMEs remain in place. Verify from the project root with:

```bash
python tests/test_reference_integrity.py
```

The frozen reference remains unchanged. An additive local-checkpoint bridge is now available; this is not yet an operational prepared-data/DDP baseline. See `LOCAL_CHECKPOINT.md` and `TIM_STYLE_CHANGES.md`.