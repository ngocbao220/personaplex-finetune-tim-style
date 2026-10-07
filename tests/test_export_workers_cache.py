import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_prepared_data
from tim_compat.prepared_data import prepare_manifest


class ExportWorkersCacheTest(unittest.TestCase):
    setUp = test_prepared_data.PreparedDataTest.setUp
    def normalized(self, manifest):
        result = []
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            path = Path(row.pop("path"))
            result.append((path.name, row, path.read_bytes(),
                           path.with_suffix(".json").read_bytes()))
        return result

    def test_spawn_serial_cache_parity_and_hit(self):
        rows = [{"sample_id": name, "sample_dir": "samples/one"}
                for name in ("two", "one")]
        self.manifest.write_text("\n".join(map(json.dumps, rows)))
        serial = prepare_manifest(self.manifest, self.base / "serial")
        cache = self.base / "cache"
        spawned = prepare_manifest(self.manifest, self.base / "spawn", workers=2, cache_dir=cache)
        self.assertEqual(self.normalized(serial), self.normalized(spawned))
        with patch("tim_compat.prepared_data._prepare_row", side_effect=AssertionError("cache miss")):
            hit = prepare_manifest(self.manifest, self.base / "hit", cache_dir=cache)
        self.assertEqual(self.normalized(serial), self.normalized(hit))
        self.assertEqual(len(list(cache.iterdir())), 1)

    def test_stale_metadata_invalidates(self):
        cache = self.base / "cache"
        prepare_manifest(self.manifest, self.base / "initial", cache_dir=cache)
        metadata = self.directory / "metadata.json"
        row = json.loads(metadata.read_text())
        row["text_prompt"] = "A different prompt."
        metadata.write_text(json.dumps(row))
        result = prepare_manifest(self.manifest, self.base / "changed", cache_dir=cache)
        sidecar = result.parent / "one.json"
        self.assertEqual(json.loads(sidecar.read_text())["text_prompt"], row["text_prompt"])
        self.assertEqual(len(list(cache.iterdir())), 2)

    def test_disabled_ignores_workers_and_cache(self):
        original = self.base / "absent"
        self.assertEqual(prepare_manifest(original, enabled=False, workers=-1,
                                          cache_dir=self.base / "unused"), original)
        self.assertFalse((self.base / "unused").exists())

    def test_invalid_parallel_input_no_output_or_cache(self):
        (self.directory / "words.json").write_text("[]")
        with self.assertRaises(ValueError):
            prepare_manifest(self.manifest, self.base / "invalid", workers=2,
                             cache_dir=self.base / "cache")
        self.assertFalse((self.base / "invalid").exists())
        self.assertFalse((self.base / "cache").exists())

    def test_corrupt_cache_fails_without_output(self):
        cache = self.base / "cache"
        prepare_manifest(self.manifest, self.base / "initial", cache_dir=cache)
        path = next(cache.glob("*.json"))
        payload = json.loads(path.read_text())
        payload["samples"][0][4]["text_prompt"] = "corrupted"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "corrupt"):
            prepare_manifest(self.manifest, self.base / "bad", cache_dir=cache)
        self.assertFalse((self.base / "bad").exists())

    def test_manifest_identity_and_audio_bytes_invalidate(self):
        cache = self.base / "cache"
        prepare_manifest(self.manifest, self.base / "initial", cache_dir=cache)
        other = self.root / "other.jsonl"
        other.write_bytes(self.manifest.read_bytes())
        prepare_manifest(other, self.base / "other", cache_dir=cache)
        self.assertEqual(len(list(cache.glob("*.json"))), 2)
        audio = self.directory / "conversation.wav"
        data = bytearray(audio.read_bytes())
        data[-1] = 1
        audio.write_bytes(data)
        result = prepare_manifest(other, self.base / "audio_changed", cache_dir=cache)
        self.assertEqual((result.parent / "one.wav").read_bytes(), bytes(data))
        self.assertEqual(len(list(cache.glob("*.json"))), 3)