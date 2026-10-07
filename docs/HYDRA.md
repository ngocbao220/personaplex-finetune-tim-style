# Opt-in Hydra launch (class A)

Run the wrapper by absolute path. Set `launch.config` to the original or derived Tim YAML and `launch.model_root` to the local PersonaPlex assets. Example override syntax: `launch.config=/absolute/train.yaml launch.model_root=/absolute/assets`.

The wrapper accepts only the existing launcher fields: config, model_root, resume_from, filter_policy, filter_report_dir, train_manifest and resolved_config. It resolves interpolation and calls the same local launcher once. It does not reinterpret the trainer YAML or override loss, prompt, crop or LoRA settings. Offline asset preflight remains in the existing launcher before importing GPU training code.

The shipped configuration disables Hydra directory changes and output directories. Keep `hydra.job.chdir=false`: relative paths inside the reference trainer YAML retain their original working-directory contract. Use `--cfg job` to inspect launch configuration without loading any model. Bypass Hydra entirely by using the existing local launcher; the launcher still accepts its original CLI, and the argument builder has a tested no-inspection identity bypass.

Hydra composition, interpolation and call parity are CPU-tested. No GPU/DDP acceptance is implied.