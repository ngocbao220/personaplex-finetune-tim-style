"""CPU-only, dependency-free identity test for the frozen reference baseline."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "refs" / "personaplex-finetune"


def copied_path(relative):
    # Markdown is relocated to docs; original inventory and hashes stay intact.
    if relative == "README.md":
        return ROOT / "docs" / "README.upstream.md"
    if relative.endswith(".md"):
        return ROOT / "docs" / relative
    return ROOT / relative


class ReferenceIntegrityTest(unittest.TestCase):
    def test_inventory_is_nonempty(self):
        inventory = json.loads((ROOT / "reference_files.sha256.json").read_text())
        self.assertGreater(len(inventory), 0)
        self.assertIn("moshi-finetune/finetune/loss.py", inventory)
        self.assertIn("moshi-finetune/finetune/data/interleaver.py", inventory)

    def test_copied_files_match_recorded_hashes(self):
        inventory = json.loads((ROOT / "reference_files.sha256.json").read_text())
        for relative, expected in inventory.items():
            with self.subTest(path=relative):
                actual = hashlib.sha256(copied_path(relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    @unittest.skipUnless(SOURCE.is_dir(), "original reference not available")
    def test_copied_files_match_original_source(self):
        inventory = json.loads((ROOT / "reference_files.sha256.json").read_text())
        for relative in inventory:
            with self.subTest(path=relative):
                self.assertEqual(copied_path(relative).read_bytes(), (SOURCE / relative).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)