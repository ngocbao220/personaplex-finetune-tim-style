"""Class-A manifest binding; no changes to reference training policies."""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import yaml


def bind_training_manifest(config, manifest, output, *, enabled=True, eval_manifest=None):
    """Bind exported train and optionally val manifests in an independent YAML.

    Bypass returns the original path without reads/writes. Relative non-data
    paths retain Tim's current-working-directory meaning; we do not rebase them.
    Enabled output is exclusive-create and must not overwrite any existing file.
    """
    if not enabled:
        return Path(config)
    config, manifest, output = Path(config), Path(manifest).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    values = yaml.safe_load(config.read_text())
    if not isinstance(values, dict) or not isinstance(values.get('data'), dict):
        raise ValueError('reference config requires a data mapping')
    manifests = [manifest]
    if eval_manifest is not None:
        eval_manifest = Path(eval_manifest).resolve()
        manifests.append(eval_manifest)
    for source in manifests:
        _validate_tim_manifest(source)
    values['data']['train_data'] = str(manifest)
    if eval_manifest is not None:
        values['data']['eval_data'] = str(eval_manifest)
    text = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    with output.open('x') as stream:
        stream.write(text)
    return output


def _validate_tim_manifest(manifest):
    count = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get('path'), str) or
                not row['path'].strip() or
                not isinstance(row.get('duration'), (int, float)) or
                isinstance(row['duration'], bool) or
                not math.isfinite(row['duration']) or row['duration'] <= 0):
            raise ValueError('expected Tim path/duration manifest record')
        count += 1
    if count == 0:
        raise ValueError('empty Tim manifest')

@contextmanager
def native_training_config(config, *, vietnamese_text_mode=None):
    """Consume bridge-only text mode after checking exported train/eval representations."""
    from .text_normalization import text_mode_from_config
    import tempfile
    config = Path(config)
    values = yaml.safe_load(config.read_text())
    from .prepared_data import validate_eval_split
    ratio = validate_eval_split(values.get('data', {}).get('eval_split_from_train', 0))
    if ratio and not values['data'].get('eval_data', '').strip():
        raise ValueError('eval_split_from_train requires exported val manifest; run prepare_data.py --config first')
    configured_mode = text_mode_from_config(values)
    mode = vietnamese_text_mode or configured_mode
    text_mode_from_config({'data': {'vietnamese_text_mode': mode}})
    if ('vietnamese_text_mode' in values.get('data', {}) and mode != configured_mode):
        raise ValueError('CLI text mode differs from data.vietnamese_text_mode')
    if (vietnamese_text_mode is None and 'vietnamese_text_mode' not in values.get('data', {})
            and 'eval_split_from_train' not in values.get('data', {})):
        yield config
        return
    # Match native comma-separated path[:weight] sources, including manifest directories.
    for key in ('train_data', 'eval_data'):
        for source in values['data'].get(key, '').split(','):
            if not source.strip():
                continue
            path = Path(source.strip().split(':')[0])
            manifests = sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]
            for manifest in manifests:
                for line in manifest.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    recorded = row.get('vietnamese_text_mode', 'diacritics')
                    if recorded != mode:
                        raise ValueError(f'text mode {mode!r} differs from export {recorded!r}: {manifest}; re-export prepared data')
                    if mode != 'diacritics':
                        # Do not resolve the WAV symlink away from its exported sidecar.
                        sidecar = json.loads(Path(row['path']).with_suffix('.json').read_text())
                        if sidecar.get('vietnamese_text_mode') != mode:
                            raise ValueError(f'text mode metadata missing/mismatched for {row["path"]}')
    values['data'].pop('vietnamese_text_mode', None)
    values['data'].pop('eval_split_from_train', None)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', encoding='utf-8') as handle:
        yaml.safe_dump(values, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        yield Path(handle.name)


@contextmanager
def checkpoint_text_mode(checkpointer_class, mode):
    """Include representation in checkpoint metadata and periodic inference snapshots."""
    from .text_normalization import normalize_vietnamese_text
    normalize_vietnamese_text('', mode)
    original = checkpointer_class.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.config = dict(self.config, vietnamese_text_mode=mode)
    checkpointer_class.__init__ = initialize
    try:
        yield
    finally:
        checkpointer_class.__init__ = original
