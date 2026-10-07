"""Resolve opt-in launch configuration without changing trainer semantics."""

from pathlib import Path

from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd


def launch_arguments(config, *, enabled=True):
    """Map the launch namespace to the existing argparse interface.

    Disabled mode returns its input verbatim without inspecting it. Trainer YAML
    remains a separate reference-format file; Hydra overrides launch fields only.
    """
    if not enabled:
        return config
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(values, dict) or set(values) != {"launch"}:
        raise ValueError("expected only a launch mapping")
    fields = values["launch"]
    allowed = {"config", "model_root", "resume_from", "filter_policy",
               "filter_report_dir", "train_manifest", "resolved_config"}
    if not isinstance(fields, dict) or set(fields) - allowed:
        raise ValueError("unknown launch fields")
    for key in ("config", "model_root"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            raise ValueError(f"launch.{key} must be an explicit local path")
    args = []
    for key, value in fields.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"launch.{key} must be a nonempty string")
        # Hydra chdir is disabled in shipped config. Absolute paths also protect
        # callers that explicitly override hydra.job.chdir.
        path = Path(value).expanduser()
        if not path.is_absolute() and HydraConfig.initialized():
            path = Path(get_original_cwd()) / path
        value = str(path.resolve())
        args.extend(["--" + key.replace("_", "-"), value])
    return args