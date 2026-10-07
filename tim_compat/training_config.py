"""Class-A manifest binding; no changes to reference training policies."""

import json
import math
from pathlib import Path

import yaml


def bind_training_manifest(config, manifest, output, *, enabled=True):
    """Persist an independent YAML with only data.train_data replaced.

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
    values['data']['train_data'] = str(manifest)
    text = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    with output.open('x') as stream:
        stream.write(text)
    return output