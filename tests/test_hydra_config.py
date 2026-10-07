import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tim_compat.hydra_config import launch_arguments


class HydraConfigTest(unittest.TestCase):
    def test_disabled_exact_identity_no_inspection(self):
        sentinel = object()
        self.assertIs(launch_arguments(sentinel, enabled=False), sentinel)

    def test_composition_overrides_and_reference_args(self):
        directory = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
            config = compose(config_name="launch", overrides=[
                "launch.config=/tmp/reference.yaml", "launch.model_root=/tmp/local",
                "launch.resume_from=/tmp/checkpoint"])
        self.assertEqual(launch_arguments(config), [
            "--config", str(Path("/tmp/reference.yaml").resolve()), "--model-root", str(Path("/tmp/local").resolve()),
            "--resume-from", str(Path("/tmp/checkpoint").resolve())])

    def test_interpolation_resolved_without_changing_trainer(self):
        config = OmegaConf.create({"launch": {"config": "/tmp/reference.yaml",
                                              "model_root": "${launch.config}"}})
        self.assertEqual(launch_arguments(config)[3], str(Path("/tmp/reference.yaml").resolve()))

    def test_bad_fields_and_missing_paths_fail(self):
        for fields in ({}, {"config": "x", "model_root": "y", "loss": 0.2},
                       {"config": "x", "model_root": False}):
            with self.assertRaises(ValueError):
                launch_arguments(OmegaConf.create({"launch": fields}))

    def test_wrapper_calls_existing_launcher_once(self):
        import train_hydra
        config = OmegaConf.create({"launch": {"config": "/tmp/reference.yaml",
                                              "model_root": "/tmp/local"}})
        with patch.object(train_hydra, "train_local") as train:
            train_hydra.main.__wrapped__(config)
        train.assert_called_once_with(["--config", str(Path("/tmp/reference.yaml").resolve()),
                                       "--model-root", str(Path("/tmp/local").resolve())])

    def test_directory_change_rejected_before_launch(self):
        import train_hydra
        with patch.object(train_hydra.HydraConfig, "initialized", return_value=True), \
                patch.object(train_hydra.HydraConfig, "get", return_value=OmegaConf.create({"job": {"chdir": True}})), \
                patch.object(train_hydra, "train_local") as train:
            with self.assertRaisesRegex(ValueError, "chdir"):
                train_hydra.main.__wrapped__(None)
            train.assert_not_called()