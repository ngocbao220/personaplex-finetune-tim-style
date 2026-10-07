# GPU parity acceptance continuation

Add scripts only: native-tokenizer/cache inspection → subprocess original Tim
vs local bridge one-step observation → real Tim mini-overfit → fresh subprocess
adapter reload. Preserve protected source hashes and all original construction,
loss, mixed precision, optimizer and scheduler calls. Fail on discovery/API
prerequisites; do not substitute the older constant-LR harness. Native generation
is SKIPPED at the documented dep_q/user-slot semantic boundary.

# Reference-preserving implementation plan

1. Copy and hash frozen trainer/runtime; record source revision. Verify byte identity.
2. Classify each subsystem A/B before edits; capture reference behavior and add enabled/disabled parity tests. Audit the existing local bridge, including its model configuration overrides.
3. Add independent prepared reader and materialized Tim sidecars. Test valid/invalid inputs, root confinement and ordering. Never alter source data.
4. Filter original chunk candidates with actual retained Tim streams, including tail cropping; test dense-overflow rejection and normal-input parity.
5. Add spawn workers and atomic fingerprinted caches; compare serial/parallel results and invalidate stale inputs.
6. Add native LMGen orchestration on current replicated model; synchronize ranks; test state restoration and artifacts.
7. Add class-A DDP orchestration and replicated checkpoint transport with unchanged FSDP fallback; test equal steps, disjoint samples, synchronized gradients and checkpoint reload. Do not change model construction or checkpoint meaning/content.
8. Add one-file OmegaConf/Hydra-compatible launch wrapper exposing reference arguments; test overrides and offline path preflight.
9. Run CPU parity, single-GPU step, two-B200 DDP, step-zero generation, then fixed-ten-sample overfit/save/reload report. Do not scale before milestone verification.

Checkpoint: step 1 complete; local bridge implemented but not yet accepted under the new disable-to-reference parity gate. Infrastructure work is not blocked merely by FSDP. No GPU acceptance claims on this macOS host.

## Mandatory change classification and acceptance

Before editing each subsystem, record its classification and invariant/tests:

- **A — Additive infrastructure/compatibility:** prepared adapter, validation/filtering, workers/cache, Hydra, local asset resolution, DDP orchestration, replicated-DDP checkpoint transport, rank-0 native LMGen evaluation, logging/export. Implement directly with regression/parity tests.
- **B — Reference semantic change:** token placement, temporal alignment/delays, causal inputs/targets, teacher forcing, loss, targets/masks, LoRA initialization/coverage, model construction, checkpoint meaning/content, or native LMGen semantics. Do not implement automatically.
- Every infrastructure modification needs a test showing that **disabling it recovers reference behavior**. Source hashes or enabled-path mock tests alone are insufficient. Compare equivalent inputs, seeds, configuration and initial state; filtering can intentionally reject inputs, but disabled filtering must reproduce reference candidate ordering/results.
- A valid STOP must identify the invariant that would change, why an additive approach cannot preserve it, exact reference file/function/code, and the minimal required semantic change. Missing implementation or FSDP usage alone is not evidence.

## Per-subsystem parity gates

| Subsystem (initial classification A) | Disabled-to-reference acceptance, plus enabled checks |
|---|---|
| Local discovery | Disable bridge: original discovery calls/arguments/results; enable: offline asset lookup only, identical construction kwargs and LoRA policy for equivalent assets. Audit dep_q overrides before claiming parity. |
| Prepared adapter | Bypass adapter on equivalent Tim sidecars: identical ordered samples, streams, targets/masks; enable: prepared inputs produce those same sidecars. |
| Filters | Disable: original candidates/order/crops/retained streams; enable: valid candidates unchanged, explicit invalid rejection. |
| Workers/cache | Disable: reference serial results; enable: same results/order for serial, spawn and cache hit; stale inputs invalidate cache. |
| Hydra | Disable wrapper: original trainer arguments/config; enable: equivalent resolved config and overrides. |
| DDP/transport | Disable strategy extension: original single-rank/FSDP behavior; enable: same model/trainable names, gradients/update for equivalent global batch, checkpoint contents and resume state. |
| Native evaluation | Disable: reference training RNG/model/optimizer behavior; enable: native generation args/outputs for fixed state, state restoration and rank synchronization. |
| Logging/export | Disable: reference outputs/state/update; enable: same training behavior with additional artifacts only. |

Run CPU checks per slice; hardware-dependent parity remains pending until executed on GPU. Preserve the documented objective-policy conflict without silently changing the Tim objective.