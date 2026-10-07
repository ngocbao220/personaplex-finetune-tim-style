# Experiment integrity / STOP report

## Status

Class-A additions: spawn prepared-export validation/cache, Hydra wrapper, and opt-in single-process actual-Sample token cache/queue diagnostics/provenance. No copied trainer/runtime files changed. Prior claims of 47/50/52 passing CPU tests were not backed by visible retained results and must not be used as acceptance evidence. Earlier real-ten serial/spawn/cache claims are not reverified. New `scripts/` acceptance observes actual Tim training, unlike the older constant-LR `gpu_acceptance.py`; see `scripts/README.md`. GPU execution remains pending. Native replicated-model generation and DDP/checkpoint transport remain unaccepted; single-GPU overfit gate has not been met on this macOS CPU host. Native user-slot incompatibility remains a generation boundary, not an executed inference result.

Free-running comparison extension: `scripts/run_free_running_inference.py` runs base and merged checkpoint sequentially in separate native inference processes. It exports original/base/current stereo dialogue WAVs (LEFT=agent RIGHT=user), checks matching windows and RIGHT-channel identity, and writes `manifest.json` with input/checkpoint/seed/settings provenance. Reference text is optional; CER/WER remain explicitly null (not computed). This is analogous to the original artifacts, not numerical equivalence to the original trainer's generation. GPU execution is still pending. Focused CPU verification visibly returned 6 tests OK, compile exit 0 and CLI help exit 0 on this continuation.

Class-A prepared training integration: `tim_compat/training_config.py` and tests, CPU export `--config`, launcher `--train-manifest`/`--resolved-config`. Only parsed `data.train_data` changes; disabled binding preserves the original config path/bytes. Source YAML is never overwritten. Prompt enablement and objective remain untouched. Config materialization is serial; no DDP acceptance is implied.

Additional class-A slice: `tim_compat/sample_filter.py`, `tests/test_sample_filter.py`, and `SAMPLE_FILTER.md`; opt-in integration in `train_local.py`. Validates actual retained reference Samples without modifying them. Disabled iterator/binding parity and enabled identity/order tests cover the CPU boundary. Dense overflow, original chunk provenance, full tokenizer/GPU parity, cache/workers and DDP remain pending.

Phase 0 completed. Additive local-checkpoint loading and prepared-format export are implemented (see `LOCAL_CHECKPOINT.md` and `PREPARED_DATA.md`). Local discovery disable parity and documented Tim-patch construction-call tests pass; prepared export has bypass and frozen Interleaver alignment parity tests. Full tokenizer/model parity and the remaining baseline integration are pending. No protected Tim source has been modified. The requested two-B200 DDP launch is **not supported by this snapshot**.

## DDP boundary analysis (previous blanket STOP superseded)

The user's A/B classification rule supersedes the earlier blanket permission blocker. Additive DDP orchestration and replicated checkpoint transport are authorized directly, provided reference semantics and disabled-path parity are proven. The code observations below identify integration risks, not proof that an additive implementation is impossible.

### Model construction

`moshi-finetune/finetune/wrapped_model.py:get_fsdp_model()` constructs on meta, loads base and LoRA weights on rank 0 only, leaves other ranks on meta, freezes parameters, then relies on FSDP `param_init_fn` and `sync_module_states=True` to materialize/broadcast on multiple ranks. The only non-FSDP return is `get_world_size() == 1`.

Adding DDP cannot safely be done by wrapping this return: it would wrap an already sharded FSDP model. Changing the rank predicates or pretending the world size is one would alter initialization/control-flow assumptions. A proper strategy branch must explicitly materialize replicas and preserve the exact initialization and trainable-parameter policy. That touches the user-protected **model construction** boundary, even if intended to preserve mathematical behavior.

### Checkpointing and resume

`moshi-finetune/finetune/checkpointing.py:Checkpointer.retrieve_save_states()` selects leaf FSDP modules for adapter extraction when world size exceeds one. DDP modules would not pass this selection: adapter extraction can return an empty state. Full-weight extraction asserts FSDP and invokes `summon_full_params`. `save_checkpoint()` also uses FSDP full optimizer-state gathering. `moshi-finetune/train.py` uses `shard_full_optim_state_dict` during distributed resume.

A DDP strategy requires different extraction and optimizer serialization paths while retaining adapter names, checkpoint directories, retention, resume and merge semantics. This touches the explicitly protected **checkpoint behavior**. Silently disabling saves or changing global world-size helpers is not an acceptable workaround.

**Current decision:** first attempt additive orchestration/transport preserving original construction, LoRA policy and checkpoint meaning/content. No separate DDP permission is needed. A semantic STOP is valid only with the exact invariant, impossibility rationale, reference code and minimal required change. Neither the absence of a DDP implementation nor FSDP assumptions alone satisfies that requirement.

## Additional reference findings (do not fix silently)

1. `InterleavedTokenizer.__call__()` passes `this_num_audio_frames` to `Interleaver.prepare_item()` as `segment_duration`. `build_token_stream()` computes `ceil(segment_duration * audio_frame_rate)`. The subsequent pad uses `conv_frames - text_tokens.shape[-1]`, potentially causing negative-padding cropping. An overflow filter must inspect **the final retained stream**, not merely count tokens in `prepare_item()`. Changing the argument to seconds would change protected placement semantics. This finding has not been demonstrated with a real Mimi encode here.
2. Tim's audio objective upweights the first codebook of **each** stream and supports 16 depformer outputs. Workspace rules instead prescribe agent-only targets and a specific non-semantic weighting. These are not automatically equivalent. Do not rewrite Tim loss to satisfy a different objective and still call it reference-preserving. Resolve the policy conflict explicitly before training.
3. Native rank-0 generation is feasible for an unwrapped replicated model; the existing reference generation evaluation instead tears down/reloads for external bot-to-bot evaluation. Reuse requires additive orchestration, synchronization and restoration of streaming/model state, not reuse of the current implementation's sequence builder.

## Changes table

| Change | Taken from current implementation? | Changes Tim training semantics? | Status |
|---|---|---|---|
| Reference copy | No, Tim | NO | Complete, byte-for-byte |
| Prepared data adapter | Independent format bridge (A) | No placement/delay/objective implementation | Export and CPU alignment parity implemented |
| Sample validation | Independent prepared-contract checks (A) | Rejects invalid inputs; no repair | Implemented; rejection/report integration pending |
| Chunk OOB filter | Planned concept only | Must be NO | Not implemented |
| Tim-semantic overflow filter | Concept only | Must be NO | Not implemented |
| Free-running orchestration | Planned orchestration only | Must be NO | Not implemented |
| Native generation | Tim runtime copied | NO | Runtime copied; integration not implemented |
| DDP | Infrastructure (A) | Must preserve construction and checkpoint semantics | Pending implementation and parity |
| Hydra config | Infrastructure | Must be NO | Not implemented |

Local path validation borrows the contract of `RuntimePaths.validate()` from `personaplex-finetuning/`: resolve a root and require three local assets. It does not copy its model construction. No `sequence.py`, objective, optimizer or LoRA implementation from that source was ported. The local bridge delegates construction/overrides to the installed training loader, with the dep_q=16/Mimi=8 settings documented in Tim's SETUP_GUIDE Phase 8. This loader compatibility configuration is explicit, not evidence that the unpinned installed training runtime matches Tim's original environment.

## File inventory / proof

- **Unchanged copied files:** complete inventory and hashes in `reference_files.sha256.json`.
- **Modified reference files:** none.
- **New files:** `REFERENCE_BASELINE.md`, `TIM_STYLE_CHANGES.md`, `reference_files.sha256.json`, `tests/test_reference_integrity.py`, `docs/tasks/plan.md`, `docs/tasks/todo.md`.
- **Subsequently added local bridge:** `LOCAL_CHECKPOINT.md`, `train_local.py`, `tim_compat/__init__.py`, `tim_compat/local_checkpoint.py`, `tests/test_local_checkpoint.py`.
- Exact source identity proves the copied objective is unchanged. It does **not** prove an unimplemented adapter or DDP path has parity.
- Every infrastructure change requires disabled-to-reference parity, as specified in `docs/tasks/plan.md`. Local bridge now has discovery bypass and documented patched-construction call parity, but still needs actual runtime/numerical audit. Prepared export has bypass and CPU frozen-Interleaver parity, not full CUDA tokenizer/model parity.
- Additive prepared files: `tim_compat/prepared_data.py`, `prepare_data.py`, `tests/test_prepared_data.py`, `PREPARED_DATA.md`. A temporary export of the real ten-conversation manifest succeeded without source edits.

## Verification and hardware limitation

Periodic free-running is now opt-in via `acceptance.inference.free_running_every_steps` and `train_local --free-running-config`. A scoped source-anchor hook observes pre-loop state and completed optimizer/scheduler updates, snapshots adapters using native Checkpointer retrieval, and runs native comparison synchronously with CPU offload/restore. Baseline reuse checks input content hashes and matching generation provenance. No copied trainer/runtime file was edited. Disabled mode does not install a hook. GPU memory round-trip and real training/inference acceptance remain pending.

Visible local verification for this addition: `python3 -m unittest discover -s tests` ran **57 tests, OK**; focused periodic tests ran **4 tests, OK**, audio/acceptance script tests ran **6 tests, OK**. `python3 -m compileall -q tim_compat/free_running.py train_local.py scripts/_tim_worker.py scripts/run_free_running_inference.py` exited 0. CLI help exposed the new flags and YAML parsed frequency 0. These are CPU/mock checks, not real checkpoint/native inference or GPU offload acceptance. The workspace has no enclosing Git repository, so `git diff --check` could not provide verification.

Host: macOS; installed PyTorch 2.2.1; CUDA unavailable. No PersonaPlex GPU forward/backward, B200 DDP, overfit run or native generation smoke was executed. CPU integrity checks are separate from those acceptance tests.

The requested one-GPU, two-B200, overfit and full-data compatibility commands cannot yet be supplied as working commands. The copied original launcher remains `moshi-finetune/train.py` with the original reference configuration format and data contract; it does not read prepared manifests or support the requested `--config config.full.yaml` wrapper. Do not run it against prepared data expecting compatibility.