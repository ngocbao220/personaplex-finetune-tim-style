"""Opt-in Hydra launch wrapper; the original trainer YAML stays untouched."""

import hydra
from hydra.core.hydra_config import HydraConfig

from tim_compat.hydra_config import launch_arguments
from train_local import main as train_local


@hydra.main(version_base="1.3", config_path="configs", config_name="launch")
def main(config):
    if HydraConfig.initialized() and HydraConfig.get().job.chdir:
        raise ValueError("hydra.job.chdir must stay false to preserve trainer relative paths")
    train_local(launch_arguments(config))


if __name__ == "__main__":
    main()