"""Class-A format bridge: prepared sources to unchanged Tim WAV/JSON sidecars."""

import json
import math
import re
import wave
from pathlib import Path


class PreparedAlignmentError(ValueError):
    """Invalid source alignment, distinct from runtime/asset failures."""
    def __init__(self, sample_id, index, word, duration, reason):
        self.sample_id = sample_id
        self.reason = reason
        self.details = dict(word_index=index, audio_duration_sec=duration,
                            word=repr(word))
        super().__init__(f'invalid or unsorted alignment: {sample_id}: {word}')


def _asset(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"prepared path escapes root: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _wav_duration(path, channels):
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != channels or audio.getframerate() != 24000:
            raise ValueError(f"expected {channels}-channel 24kHz PCM WAV: {path}")
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise ValueError(f"empty audio: {path}")
    return duration


def prepare_manifest(manifest, output=None, *, enabled=True, workers=0, cache_dir=None,
                     vietnamese_text_mode='diacritics'):
    """Materialize ordered Tim sidecars, without changing any source or sequence.

    Disabled mode returns the exact original manifest path without reading it.
    Enabled mode requires a fresh output directory; invalid input produces no output.
    Prompt construction, crop/delay and loss masks remain the Tim tokenizer's job.
    """
    if not enabled:
        return Path(manifest)
    from .text_normalization import normalize_vietnamese_text
    normalize_vietnamese_text('', vietnamese_text_mode)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    manifest = Path(manifest).resolve()
    root = manifest.parent
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"use a fresh adapter output directory: {output}")
    if output.is_relative_to(root):
        raise ValueError("adapter output must be outside prepared source root")
    cache = None
    fingerprint = None
    cached = None
    if cache_dir is not None:
        cache = Path(cache_dir).resolve()
        if cache.is_relative_to(root):
            raise ValueError("cache must be outside prepared source root")
        fingerprint = _fingerprint(root, manifest)
        cache_file = cache / (fingerprint + ".json")
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("fingerprint") != fingerprint:
                raise ValueError("invalid export cache fingerprint")
            if cached.get("payload_hash") != _payload_hash(cached.get("samples")):
                raise ValueError("corrupt export cache payload")
    samples = []
    seen = set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        samples.append((root, row))
    if cached is not None:
        samples = [(row[0], Path(row[1]), Path(row[2]), row[3], row[4])
                   for row in cached["samples"]]
    elif workers:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            samples = list(pool.map(_prepare_row, samples))
    else:
        samples = list(map(_prepare_row, samples))
    if not samples:
        raise ValueError("empty prepared manifest")
    if cache is not None:
        if _fingerprint(root, manifest) != fingerprint:
            raise RuntimeError("prepared sources changed during export; retry on immutable inputs")
        if cached is None:
            _publish_cache(cache, fingerprint, samples)
    output.mkdir(parents=True)
    records = []
    for sample_id, audio, voice, duration, sidecar in samples:
        # Cache stores raw prepared text; apply each mode exactly once on export.
        sidecar = dict(sidecar, vietnamese_text_mode=vietnamese_text_mode,
            alignments=[[normalize_vietnamese_text(text, vietnamese_text_mode)
                         if speaker == 'SPEAKER_BROKER' else text, timestamps, speaker]
                        for text, timestamps, speaker in sidecar['alignments']])
        wav_path = output / f"{sample_id}.wav"
        wav_path.symlink_to(audio)
        wav_path.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False) + "\n")
        records.append(json.dumps({"path": str(wav_path), "duration": duration,
                                   "vietnamese_text_mode": vietnamese_text_mode}))
    result = output / "train.jsonl"
    result.write_text("\n".join(records) + "\n")
    print(f"Prepared adapter: {len(samples)} samples; LEFT=agent, RIGHT=user; first={samples[0][0]}")
    return result


def _fingerprint(root, manifest):
    """Conservative content hash: invalidate on any source-root file change."""
    import hashlib

    digest = hashlib.sha256(b"tim-prepared-export-v2\0")
    digest.update(str(root).encode())
    digest.update(str(manifest).encode() + b"\0")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        _asset(root, path.relative_to(root))
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _publish_cache(cache, fingerprint, samples):
    """Publish complete JSON with an atomic no-overwrite hard link."""
    import os
    import tempfile

    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (fingerprint + ".json")
    payload = {"fingerprint": fingerprint, "samples": samples,
               "payload_hash": _payload_hash(samples)}
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, default=str, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if json.loads(target.read_text()) != json.loads(temporary.read_text()):
                    raise ValueError("conflicting export cache entry")
        finally:
            temporary.unlink(missing_ok=True)


def _payload_hash(samples):
    import hashlib

    return hashlib.sha256(json.dumps(samples, default=str, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()

def _prepare_row(item):
    root, row = item
    sample_id = row["sample_id"]
    if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError(f"unsafe sample_id: {sample_id!r}")
    directory = Path(row.get("sample_dir", f"samples/{sample_id}"))
    audio = _asset(root, row.get("audio_path", directory / "conversation.wav"))
    words = json.loads(_asset(root, row.get("words_path", directory / "words.json")).read_text())
    metadata = json.loads(_asset(root, row.get("metadata_path", directory / "metadata.json")).read_text())
    # Older externally prepared exports use voice_prompt.wav for the agent.
    # This is filename compatibility, not voice-region selection.
    voice_default = directory / "voice_prompt_left.wav"
    if not (root / voice_default).exists():
        voice_default = directory / "voice_prompt.wav"
    voice = _asset(root, row.get("voice_prompt_left", voice_default))
    if metadata.get("agent_channel") not in ("left", 0) or metadata.get("user_channel") not in ("right", 1):
        raise ValueError(f"LEFT=agent / RIGHT=user required: {sample_id}")
    duration = _wav_duration(audio, 2)
    _wav_duration(voice, 1)
    prompt = metadata.get("text_prompt_left")
    if prompt is None:
        prompt = metadata.get("text_prompt")
    if prompt is None:
        prompt = _asset(root, row.get("text_prompt_path", directory / "prompt.txt")).read_text()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"missing prepared text prompt: {sample_id}")
    alignments = []
    last_start = -1
    labels = {"agent": "SPEAKER_BROKER", "user": "SPEAKER_CLIENT"}
    if not isinstance(words, list) or not words:
        raise ValueError(f"empty/non-list words: {sample_id}")
    for index, word in enumerate(words):
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise PreparedAlignmentError(sample_id, index, word, duration, 'invalid_timestamp') from error
        if not all(map(math.isfinite, (start, end))):
            raise PreparedAlignmentError(sample_id, index, word, duration, 'invalid_timestamp')
        if not 0 <= start < end <= duration:
            raise PreparedAlignmentError(sample_id, index, word, duration, 'timestamp_out_of_bounds')
        text = word["word"]
        if (start < last_start or
                not isinstance(text, str) or not text.strip() or word["speaker"] not in labels):
            raise PreparedAlignmentError(sample_id, index, word, duration, 'invalid_alignment')
        alignments.append([text, [start, end], labels[word["speaker"]]])
        last_start = start
    return (sample_id, audio, voice, duration, {
        "alignments": alignments, "text_prompt": prompt,
        "voice_prompt": str(voice),
    })
