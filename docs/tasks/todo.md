# GPU parity acceptance continuation

- [x] Add tokenizer/cache parity script and inspectable artifacts.
- [x] Add original-vs-bridge actual trainer one-step observers/comparisons.
- [x] Add actual trainer mini-overfit and fresh-process save/reload comparison.
- [x] Add fail-fast master, prerequisites/tolerances documentation, CPU tests.
- [x] CPU suite: 50 tests PASS, exit 0; syntax and shell checks PASS.
- [ ] Execute on one B200 with real local assets and prepared data.
- [ ] Establish numerical parity, clear overfit and reload GPU evidence.
- [ ] Native free-running: STOP/SKIPPED pending compatible user-input contract.

# Tasks

- [x] Phase 0: copy needed reference code, record revision, verify every copied file against source.
- [x] Document protected-boundary blockers before modifying reference implementation.
- [x] Add local-only asset discovery bridge with three-file contract, training-loader delegation and CPU tests; no reference edits.
- [x] Add local bridge disabled-to-reference discovery and documented Tim-patch construction-call parity tests.
- [ ] Pin/audit actual patched training runtime; verify numerical model/LoRA parity on GPU.
- [ ] Classify each subsystem before editing and attach disable-to-reference parity tests to every infrastructure modification.
- [ ] Preserve/report workspace-objective conflict; do not silently change Tim objective.
- [x] Prepared format adapter, validation, bypass parity and frozen Interleaver alignment tensor parity; export smoke on real ten samples.
- [x] Prepared export/launcher config binding with only train_data substitution, no-write bypass and source/config-value preservation tests.
- [x] Prepared export spawn workers and atomic content-addressed cache; serial/spawn/cache-hit synthetic and real-ten CPU parity. This is not a tokenization cache.
- [x] Opt-in Hydra launch wrapper; original CLI bypass, strict launch fields, interpolation/override/call parity CPU tests.
- [ ] Tokenization workers/cache, dense-overflow diagnostics and stable chunk provenance remain pending.
- [ ] Integrate sample rejection/report filtering and full tokenizer codes/prompt/masks parity.
- [x] Opt-in retained-Sample validation/report bridge: no-op disable, valid identity/order parity, OOB and prompt/injection exclusion CPU tests.
- [ ] Dense-overflow diagnostics and stable source/chunk rejection identities; retained-code bounds do not establish these.
- [ ] Original-chunk OOB and retained-Tim-stream overflow tests.
- [ ] Spawn-worker filtering and fingerprinted cache tests.
- [ ] Native current-model free-running evaluation tests.
- [ ] DDP strategy, synchronized eval and checkpoint parity tests.
- [ ] Self-contained Hydra configuration and launch override tests.
- [ ] Synthetic objective/model/optimizer regression parity tests.
- [ ] Single-GPU and two-B200 smoke tests.
- [ ] Fixed ten-sample overfit, save/reload and inference report.

Uncompleted tasks are pending, not automatically blocked. DDP infrastructure is class A if semantics are preserved. STOP only for a demonstrated class-B requirement with the four required pieces of evidence. GPU acceptance remains pending on suitable hardware.