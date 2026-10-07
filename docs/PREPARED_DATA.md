# Prepared-data format adapter (class A)

The adapter exports ordered Tim-compatible WAV/JSON sidecars and a `path`/`duration` JSONL manifest. It does not encode audio, tokenize prompts, build sequences, repair transcripts, align words, select voice regions, change cropping/delays or implement loss.

```bash
python /Users/ngocbao/Documents/Document/research/main/speech/exps/finetune-personalplex/personaplex-finetune-tim-style/prepare_data.py \
  --manifest /Users/ngocbao/Documents/Document/research/main/speech/exps/finetune-personalplex/otospeech-prepared/train.jsonl \
  --output /tmp/tim-prepared-export
```

Use a fresh output directory outside the source root. Set original Tim config `data.train_data` to the emitted manifest, and `system_prompt.enable: true` to use voice/text conditioning. This command is a CPU export command, not a claim of working GPU training.

## Training config binding (class A)

## Spawn validation and export cache (class A)

`--workers 2` uses spawn processes for per-row validation and sidecar preparation. Results are collected in manifest order; the default `--workers 0` stays serial. `--cache-dir /absolute/cache` optionally caches the validated export records, **not** Mimi codes or model/tokenizer output. Keep cache and output outside the prepared source root.

Cache keys include adapter version, manifest identity, absolute source root and the content of every file under that root. This deliberately conservative fingerprint invalidates unrelated source-root changes too, and reads audio bytes on hits; it is a correctness-first cache, not a throughput claim. Atomic no-overwrite publication prevents incomplete JSON entries; payload corruption fails before output. Sources must stay immutable during export and training because exported audio/voice paths reference source files. No cache or process inspection occurs when the adapter is disabled.

CPU acceptance: synthetic and real ten-conversation serial/spawn/cache-hit exports match in ordering, durations, audio bytes and sidecar bytes after normalizing the output-directory paths. Existing frozen Interleaver parity remains passing. Tokenization workers/cache are still pending.

## Config binding details

Add `--config /absolute/reference.yaml` to the export command to write `train.yaml` alongside the exported manifest. Only `data.train_data` changes; all other parsed YAML values, including objective weights, prompt enablement, shuffle, crop and LoRA settings remain identical. The source YAML is not modified. The copied example has prompt disabled; this bridge deliberately does not silently enable it. Choose the intended reference config before export.

Pass the derived YAML to `train_local.py --config`. Alternatively use `--train-manifest /absolute/export/train.jsonl --resolved-config /absolute/fresh-derived.yaml` together with `--config /absolute/reference.yaml`. Omit both options to bypass binding and use the original config path without reading or rewriting it in the bridge. Output is exclusive-create; existing files cannot be overwritten. Relative non-data paths keep their reference current-working-directory meaning.

For multi-rank launches, generate the config once with the CPU export command, then pass it to all ranks. In-launch config materialization rejects `WORLD_SIZE>1` to avoid concurrent writers; this does not claim DDP support. Manifest binding validates `path`/positive finite `duration` records, not audio/tokenizer parity.

## Input

Each JSONL row has `sample_id` and `sample_dir`, or explicit root-confined `audio_path`, `words_path`, `metadata_path`, `voice_prompt_left` and optional `text_prompt_path`. Defaults under the sample directory:

- `conversation.wav`: two-channel 24kHz PCM, LEFT=agent, RIGHT=user.
- `words.json`: ordered list of `word`, `start`, `end`, `speaker` (`agent`/`user`). Overlap is allowed; decreasing start times, nonfinite/out-of-bounds/zero durations and unknown speakers are rejected, not repaired.
- `metadata.json`: explicit `agent_channel: left`, `user_channel: right`, prepared top-level `text_prompt_left`. If absent or null, use legacy `text_prompt`; if that is also absent or null, read existing `prompt.txt`. A non-null prompt must be a nonempty string; invalid values are rejected rather than bypassed. Exported Tim sidecars still use `text_prompt`. Export cache version is bumped to invalidate entries created with the old field selection.
- `voice_prompt_left.wav`: mono 24kHz prepared agent voice; legacy `voice_prompt.wav` filename is supported. No region selection or synthesis.

Agent/user labels map to Tim's `SPEAKER_BROKER`/`SPEAKER_CLIENT`. Words and prompt content retain their original order/text. Conversation WAVs are symlinked without rewriting channels or source files. Tim's unchanged `InterleavedTokenizer.__call__()` reads adjacent JSON and builds hybrid prompt/codes/masks itself.

## Disable-to-reference gate

`prepare_manifest(reference_manifest, enabled=False)` returns the original path without validation, filesystem writes or conversion. Tests verify original Tim manifest bytes, enabled alignment/prompt/audio parity, ordering, legacy filename support, invalid-input rejection and root confinement. A CPU test executes the frozen Tim `Interleaver.prepare_item()` on independently specified reference alignments versus exported alignments and compares tensors.

The real local ten-conversation manifest validated and exported successfully into a temporary directory. Full `InterleavedTokenizer` prompt/Mimi/codes/masks, model forward/loss/update parity is still pending; the reference hardcodes CUDA in its tokenizer. No GPU acceptance is claimed.