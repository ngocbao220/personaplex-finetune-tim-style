"""Resolve prepared/external inference audio without importing a model runtime."""
import json
import math
from pathlib import Path


def select_source(settings, manifest=None):
    # An external file wins; sample_id can still label its artifacts.
    external = settings.get('input_file') or settings.get('original_wav')
    if external:
        path = Path(external).resolve()
    else:
        sample_id = settings.get('sample_id')
        if not sample_id or not manifest:
            raise ValueError('provide input_file/original_wav or sample_id and prepared_manifest')
        root, row = find_sample(manifest, sample_id)
        directory = row.get('sample_dir', f'samples/{sample_id}')
        path = prepared_path(root, row.get('audio_path', f'{directory}/conversation.wav'))
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    return path


def audio_window(audio, rate, user_channel='right', start_sec=0., window_seconds=None, *, exact=False):
    import numpy as np
    audio = np.asarray(audio)
    if user_channel not in ('left', 'right'):
        raise ValueError('user_channel must be left or right')
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or not audio.shape[1]:
        raise ValueError('input audio must be nonempty mono or stereo')
    if not np.isfinite(audio).all() or rate <= 0:
        raise ValueError('invalid audio samples or sample rate')
    start = 0.0 if start_sec is None else float(start_sec)
    duration = audio.shape[1] / rate
    if not math.isfinite(start) or not 0 <= start < duration:
        raise ValueError('start_sec outside audio duration')
    if window_seconds is not None:
        window_seconds = float(window_seconds)
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('window_seconds must be positive and finite')
    if exact and window_seconds is not None and start + window_seconds > duration:
        raise ValueError('requested prepared window exceeds conversation duration')
    end = duration if window_seconds is None else min(duration, start + window_seconds)
    first, last = round(start * rate), round(end * rate)
    if last <= first:
        raise ValueError('selected window contains no samples')
    original = np.ascontiguousarray(audio[:, first:last])
    channel = 0 if audio.shape[0] == 1 or user_channel == 'left' else 1
    return original, np.ascontiguousarray(original[channel:channel+1]), first / rate


def prepared_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('prepared path escapes manifest root')
    return path


def find_sample(manifests, sample_id):
    # A single prepared manifest or an ordered list of train/validation/test manifests.
    if isinstance(manifests, (str, Path)):
        manifests = [manifests]
    matches = []
    seen = set()
    for value in manifests or []:
        manifest = Path(value).resolve()
        if manifest in seen:
            continue
        seen.add(manifest)
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('sample_id') == sample_id:
                    matches.append((manifest.parent, row))
    if len(matches) != 1:
        raise ValueError(f'sample_id {sample_id!r}: expected one manifest match, got {len(matches)}')
    return matches[0]


def conditioning(settings, manifests=None):
    """External audio replaces audio only; missing prompts still come from sample_id."""
    voice = settings.get('voice_prompt')
    text_file = settings.get('text_prompt_file')
    text = settings.get('text_prompt')
    if text is not None and text_file:
        raise ValueError('provide text_prompt or text_prompt_file, not both')
    text_file = Path(text_file).resolve() if text_file else None
    if text_file:
        text = text_file.read_text(encoding='utf-8')
    if not voice or text is None:
        sample_id = settings.get('sample_id')
        if not sample_id or not manifests:
            raise ValueError('missing prompts: provide voice/text prompts or a prepared sample_id')
        root, row = find_sample(manifests, sample_id)
        directory = row.get('sample_dir', f'samples/{sample_id}')
        metadata_path = prepared_path(root, row.get('metadata_path', f'{directory}/metadata.json'))
        metadata = json.loads(metadata_path.read_text())
        if not voice:
            name = 'voice_prompt_right.wav' if settings.get('user_channel', 'right') == 'left' else 'voice_prompt_left.wav'
            default = prepared_path(root, f'{directory}/{name}')
            if not default.is_file() and name == 'voice_prompt_left.wav':
                default = prepared_path(root, f'{directory}/voice_prompt.wav')
            voice = prepared_path(root, row.get(name.removesuffix('.wav'), str(default)))
        if text is None:
            key = 'text_prompt_right' if settings.get('user_channel', 'right') == 'left' else 'text_prompt_left'
            text = metadata.get(key)
            if text is None and key == 'text_prompt_left':
                text = metadata.get('text_prompt')
                if text is None:
                    text_file = prepared_path(root, row.get('text_prompt_path', f'{directory}/prompt.txt'))
                    text = text_file.read_text(encoding='utf-8')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('agent text prompt must be nonempty; provide prompt for the selected agent channel')
    voice = Path(voice).resolve()
    if not voice.is_file() or not voice.stat().st_size:
        raise FileNotFoundError(voice)
    return voice, text, text_file


def load_audio_window(path, settings):
    """Crop source once, normalize to 24 kHz, and take user from that same window."""
    import numpy as np
    import sphn
    audio, rate = sphn.read(str(path))
    original, _, start = audio_window(audio, rate, settings.get('user_channel', 'right'),
        settings.get('start_sec', settings.get('start', 0.0)), settings.get('window_seconds'),
        exact=not (settings.get('input_file') or settings.get('original_wav')))
    if rate != 24000:
        expected_samples = round(original.shape[-1] * 24000 / rate)
        original = sphn.resample(original, src_sample_rate=rate, dst_sample_rate=24000)
        # sphn emits padded resampler blocks; retain only the requested duration.
        if original.shape[-1] < expected_samples:
            raise ValueError('resampled audio shorter than selected window')
        original = original[:, :expected_samples]
    channel = 0 if original.shape[0] == 1 or settings.get('user_channel', 'right') == 'left' else 1
    return np.ascontiguousarray(original), np.ascontiguousarray(original[channel:channel+1]), start
