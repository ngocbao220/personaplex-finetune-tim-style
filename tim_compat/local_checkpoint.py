"""Local assets behind the CheckpointInfo interface used by the Tim trainer.

Model construction and LoRA overrides are delegated to the training loader.
Do not use the inference-only PersonaPlex loader as a training replacement.
"""

import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalAssets:
    root: Path
    moshi_weights: Path
    mimi_weights: Path
    tokenizer: Path
    config: Path | None

    @classmethod
    def resolve(cls, root, *, moshi_path=None, mimi_path=None,
                tokenizer_path=None, config_path=None):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"local PersonaPlex model directory missing: {root}")

        def file_path(override, default):
            path = Path(override).expanduser() if override is not None else root / default
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"required local PersonaPlex asset missing or empty: {path}")
            return path

        config = file_path(config_path, "config.json") if config_path is not None else None
        return cls(root, file_path(moshi_path, "model.safetensors"),
                   file_path(mimi_path, "tokenizer-e351c8d8-checkpoint125.safetensors"),
                   file_path(tokenizer_path, "tokenizer_spm_32k_3.model"), config)


class LocalCheckpointInfo:
    """Delegate training construction, with explicit PersonaPlex configuration.

    The dep_q=16 and Mimi=8 compatibility settings match Tim's SETUP_GUIDE
    Phase 8. All other model defaults come from the supplied training loader.
    """

    def __init__(self, assets, loaders):
        signature = inspect.signature(loaders.get_moshi_lm)
        if not {"lm_kwargs", "lm_kwargs_overrides"}.issubset(signature.parameters):
            raise RuntimeError(
                "Tim training loader must support lm_kwargs and lm_kwargs_overrides; "
                "the inference-only PersonaPlex loader cannot preserve Tim LoRA construction"
            )
        self.assets = assets
        self._loaders = loaders
        self.moshi_weights = assets.moshi_weights
        self.mimi_weights = assets.mimi_weights
        self.tokenizer = assets.tokenizer
        self.raw_config = json.loads(assets.config.read_text()) if assets.config else {}
        if not isinstance(self.raw_config, dict):
            raise ValueError("local model config must be a JSON object")
        self.model_type = self.raw_config.get("model_type", "personaplex")
        if self.model_type != "personaplex":
            raise ValueError(f"expected PersonaPlex configuration, got {self.model_type!r}")
        self.lm_config = dict(loaders._lm_kwargs)
        # Tim documents sparse configs as metadata, not complete LM kwargs.
        if "dim" in self.raw_config:
            metadata = {"model_type", "version", "moshi_name", "mimi_name",
                        "tokenizer_name", "lora_name", "lm_gen_config"}
            self.lm_config.update({k: v for k, v in self.raw_config.items() if k not in metadata})
        self.lm_config["dep_q"] = 16
        if self.lm_config.get("n_q") != 16:
            raise ValueError("PersonaPlex training requires n_q=16")
        self.lm_gen_config = self.raw_config.get("lm_gen_config", {})

    def get_mimi(self, device="cpu"):
        return self._loaders.get_mimi(self.mimi_weights, device=device, num_codebooks=8)

    def get_moshi(self, device="cpu", dtype=None, load_weight=True, **kwargs):
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self._loaders.get_moshi_lm(
            self.moshi_weights if load_weight else None,
            lm_kwargs=dict(self.lm_config), device=device, **kwargs,
        )

    def get_text_tokenizer(self):
        import sentencepiece
        return sentencepiece.SentencePieceProcessor(str(self.tokenizer))


@contextmanager
def local_checkpoint_loader(loaders, root, *, enabled=True):
    """Scope only Tim's checkpoint discovery to local assets; never invoke Hub."""
    if not enabled:
        # No validation, wrapping or default changes on the reference path.
        yield
        return
    original = inspect.getattr_static(loaders.CheckpointInfo, "from_hf_repo")

    def from_local(hf_repo=None, moshi_weights=None, mimi_weights=None,
                   tokenizer=None, config_path=None):
        assets = LocalAssets.resolve(root, moshi_path=moshi_weights,
                                     mimi_path=mimi_weights, tokenizer_path=tokenizer,
                                     config_path=config_path)
        return LocalCheckpointInfo(assets, loaders)

    loaders.CheckpointInfo.from_hf_repo = staticmethod(from_local)
    try:
        yield
    finally:
        loaders.CheckpointInfo.from_hf_repo = original