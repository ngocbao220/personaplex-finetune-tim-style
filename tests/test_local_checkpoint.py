import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from tim_compat.local_checkpoint import LocalAssets, LocalCheckpointInfo, local_checkpoint_loader


class LocalCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
                     "tokenizer_spm_32k_3.model"):
            (self.root / name).write_bytes(b"test asset")
        self.calls = []

        def get_moshi_lm(filename, lm_kwargs=None, lm_kwargs_overrides=None, **kwargs):
            self.calls.append((filename, lm_kwargs, lm_kwargs_overrides, kwargs))
            return "model"

        class CheckpointInfo:
            @staticmethod
            def from_hf_repo(*args, **kwargs):
                raise AssertionError("Hub path must never be called")

        self.loaders = types.SimpleNamespace(
            _lm_kwargs={"dim": 4096, "dep_q": 8, "n_q": 16},
            get_moshi_lm=get_moshi_lm, get_mimi=Mock(return_value="mimi"),
            CheckpointInfo=CheckpointInfo,
        )

    def test_three_local_assets_without_config(self):
        assets = LocalAssets.resolve(self.root)
        self.assertIsNone(assets.config)
        self.assertEqual(assets.moshi_weights, self.root / "model.safetensors")
        info = LocalCheckpointInfo(assets, self.loaders)
        overrides = {"lora": True, "lora_rank": 32, "lora_scaling": 2.0,
                     "gradient_checkpointing": True}
        self.assertEqual(info.get_moshi(device="meta", load_weight=False,
                                       lm_kwargs_overrides=overrides), "model")
        filename, config, forwarded, kwargs = self.calls[-1]
        self.assertIsNone(filename)
        self.assertEqual(config["dep_q"], 16)
        self.assertEqual(forwarded, overrides)
        self.assertEqual(kwargs["device"], "meta")
        self.assertEqual(self.loaders._lm_kwargs["dep_q"], 8)
        info.get_mimi(device="cuda:1")
        self.loaders.get_mimi.assert_called_once_with(assets.mimi_weights,
                                                     device="cuda:1", num_codebooks=8)

    def test_missing_and_empty_assets_fail(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
            LocalAssets.resolve(self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            LocalAssets.resolve(self.root)

    def test_optional_sparse_config(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"model_type": "personaplex", "version": "7b-v1"}))
        info = LocalCheckpointInfo(LocalAssets.resolve(self.root, config_path="config.json"), self.loaders)
        self.assertEqual(info.lm_config, {"dim": 4096, "dep_q": 16, "n_q": 16})

    def test_inference_loader_rejected(self):
        self.loaders.get_moshi_lm = lambda filename, device="cpu": None
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            LocalCheckpointInfo(LocalAssets.resolve(self.root), self.loaders)

    def test_discovery_is_local_and_restored_on_error(self):
        original = self.loaders.CheckpointInfo.from_hf_repo
        with self.assertRaisesRegex(ValueError, "intentional"):
            with local_checkpoint_loader(self.loaders, self.root):
                info = self.loaders.CheckpointInfo.from_hf_repo(hf_repo="never/download")
                self.assertEqual(info.moshi_weights, self.root / "model.safetensors")
                raise ValueError("intentional")
        self.assertIs(self.loaders.CheckpointInfo.from_hf_repo, original)

    def test_disabled_layer_recovers_reference_discovery(self):
        reference = Mock(return_value=object())
        self.loaders.CheckpointInfo.from_hf_repo = staticmethod(reference)
        arguments = dict(hf_repo="reference", moshi_weights="explicit-local")
        expected = self.loaders.CheckpointInfo.from_hf_repo(**arguments)
        reference.reset_mock()
        with local_checkpoint_loader(self.loaders, "/nonexistent", enabled=False):
            actual = self.loaders.CheckpointInfo.from_hf_repo(**arguments)
            self.assertIs(self.loaders.CheckpointInfo.from_hf_repo, reference)
        self.assertIs(actual, expected)
        reference.assert_called_once_with(**arguments)

    def test_construction_matches_documented_tim_patch_contract(self):
        # SETUP_GUIDE Phase 8 is the reference for the non-vendored patched loader.
        # This tests call parity, not numerical model/runtime version parity.
        assets = LocalAssets.resolve(self.root)
        overrides = {"lora": True, "lora_rank": 32, "lora_scaling": 2.0}
        for load_weight in (False, True):
            reference_config = dict(self.loaders._lm_kwargs, dep_q=16)
            self.loaders.get_moshi_lm(
                assets.moshi_weights if load_weight else None,
                lm_kwargs=reference_config, device="meta",
                lm_kwargs_overrides=overrides,
            )
            expected = self.calls[-1]
            LocalCheckpointInfo(assets, self.loaders).get_moshi(
                device="meta", load_weight=load_weight, lm_kwargs_overrides=overrides,
            )
            self.assertEqual(self.calls[-1], expected)


if __name__ == "__main__":
    unittest.main()