# GPU acceptance (not yet GPU verified)

Use the installed **Tim patched training Moshi runtime**, whose loader supports
`lm_kwargs`/`lm_kwargs_overrides`, and native `LMGen` hybrid prompts. The copied
inference-only loader is not a training substitute. All assets must be local.

Run from the same working directory used for the original config's relative
paths. Supply an absolute exported Tim JSONL containing exactly ten distinct
prepared WAVs and JSON sidecars, LEFT=agent/RIGHT=user. Output must not exist.

```bash
python /absolute/path/personaplex-finetune-tim-style/gpu_acceptance.py \
  --config /absolute/path/tim.yaml \
  --model-root /absolute/path/personaplex-7b-v1 \
  --manifest /absolute/path/ten/train.jsonl \
  --output /absolute/path/acceptance/run-01 \
  --steps 100 --required-loss-ratio 0.8 --allow-tim-objective
```

The explicit objective flag acknowledges a policy mismatch, not a fix: Tim
trains 16 audio outputs and weights both semantic streams. This script retains
that objective exactly. It does **not** certify the workspace agent-only loss.

Checks: deterministic first crop for each sample, actual reference/miss/hit
token equality including masks, forward/backward, nonzero LoRA gradients,
frozen-base gradients absent, finite gradient norm, fixed-ten loss before/after,
original Checkpointer adapter save, fresh base+adapter reload objective equality.
Each step logs losses/per-codebook values, LR, trainable count, norm and peak
CUDA memory. Uses constant LR for this acceptance harness (not reference
OneCycle trainer/scheduler parity); does not claim optimizer-schedule parity.

## Important fail-closed native inference boundary

Standard native `LMGen.prepare_step_input` takes `n_q - dep_q` external user
codebooks. Local Tim construction enforces `n_q=16, dep_q=16`, hence zero user
slots; PersonaPlex inference needs eight. The script records adapter/reload and
loss evidence in `summary.json`, then fails with explicit `inference=BLOCKED`
instead of silently changing dep_q, dropping user context or mislabeling a pass.
An installed runtime implementing a compatible 8-user-slot interface must be
audited before this inference branch can pass. This is a demonstrated semantic
boundary, not proof that DDP infrastructure is impossible.

DDP/checkpoint transport and multi-rank acceptance remain pending. Do not run
distributed experiments before single-GPU overfit acceptance. No GPU pass has
been obtained on this host.

## Optional training observation and Sample cache

Add `--token-report /absolute/path/new-report.jsonl` and optionally
`--token-cache /absolute/path/cache` to `train_local.py`. Omit both for the
unchanged reference tokenizer. These flags are single-process only.

The cache contains actual reference Samples, not reconstructed sequences.
Keys hash actual chunk PCM, sidecar metadata, model/Mimi/SPM files, config and
tokenizer implementation and voice WAV bytes. Keep every referenced asset
immutable during a run (reference voice cache is path-based). Local cache files must be
trusted (PyTorch dataclass deserialization). Checksum detects accidental damage,
not malicious modification. Concurrent publishers fail rather than overwrite.
Dense queue diagnostics count overwritten and end-pending tokens before final
reference cropping; they are **not** a measurement of retained lexical loss.