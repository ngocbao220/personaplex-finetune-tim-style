# Retained-Sample validation — class A

Opt-in validation of the actual Tim `Sample`, after reference crops, prompts and injections. Valid samples retain object identity, codes, masks and order. No sequence/objective reconstruction.

`train_local.py` accepts `--filter-policy /absolute/policy.json --filter-report-dir /absolute/new-directory`. Omit both for the original loader binding and behavior. Policy JSON fields:

- `text_cardinality`, `audio_cardinality`: explicit bounds from the actual model/runtime.
- `nonlexical_text_ids`: explicit reference padding/marker IDs; include BOS/EOS if not considered lexical. Never assume IDs from a different tokenizer.
- `zero_padding_id`: reference sentinel, default `-1`.
- `require_agent_text`: default `true`; `false` allows audio-only samples.
- `max_consecutive_rejections`: default `1000`, preventing infinite rejection loops.

Checks integer `[1,17,T]` layout, prompt bounds, context mask shape/device/dtype, token bounds and lexical text outside prompt/injection regions. Allows sentinel padding and short tails. Reference exceptions propagate.

Exclusive-create `rejects-rank-N.jsonl` reports record iterator-local `candidate_index` and `reason`. Indices are not stable source/chunk identities; train/eval can repeat indices. Use a fresh report directory. Filtering applies to both train and eval. Uneven rejection across ranks still needs dedicated DDP orchestration; this is not multi-rank acceptance.

CPU tests cover disabled call parity, valid identity/order/tensor preservation, prompt/injection exclusion, OOB, tails, exceptions and rejection limits. No Mimi/GPU/numerical training acceptance is claimed.

**Not dense-overflow detection:** valid retained codes cannot prove Tim preserved all queued tokens before cropping. Original-candidate metadata, dense-overflow diagnostics, spawn/cache and full tokenizer parity remain pending. Filtering intentionally changes membership when enabled, never retained sample contents.