# nu_markdown/configuration/vlm_config.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------
# Load local .env (standalone-safe)
# ---------------------------------------------------------
from .env_adapter import load_local_env

load_local_env()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = (dtype_str or "").strip().lower()

    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_str in ("float16", "fp16"):
        return torch.float16
    if dtype_str in ("float32", "fp32"):
        return torch.float32

    raise ValueError(f"Unsupported VLM_TORCH_DTYPE: {dtype_str}")


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


# ---------------------------------------------------------
# Config Class
# ---------------------------------------------------------

@dataclass(frozen=True)
class VLMConfig:
    # -----------------------------
    # Model
    # -----------------------------
    model_path: Path
    device: str
    torch_dtype: torch.dtype
    device_map: str

    # -----------------------------
    # Rendering
    # -----------------------------
    dpi: int
    max_image_long_side: int

    # -----------------------------
    # Generation
    # -----------------------------
    max_new_tokens: int
    temperature: float

    # -----------------------------
    # Offline
    # -----------------------------
    transformers_offline: bool
    hf_hub_offline: bool

    # -----------------------------
    # Logging
    # -----------------------------
    log_level: str

    # -----------------------------------------------------
    # Factory
    # -----------------------------------------------------

    @classmethod
    def from_env(cls) -> "VLMConfig":

        # -----------------------------
        # Model path
        # -----------------------------
        model_path_str = os.getenv(
            "VLM_MODEL_PATH",
            "C:/models/nu_markdown"
        )

        model_path = Path(model_path_str)

        if not model_path.exists():
            raise FileNotFoundError(
                f"VLM_MODEL_PATH does not exist: {model_path}"
            )

        # -----------------------------
        # Device
        # -----------------------------
        device = os.getenv("VLM_DEVICE", "cuda")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "VLM_DEVICE=cuda but CUDA is not available."
            )

        # -----------------------------
        # Torch dtype
        # -----------------------------
        torch_dtype = resolve_torch_dtype(
            os.getenv("VLM_TORCH_DTYPE", "bfloat16")
        )

        # -----------------------------
        # Numeric values
        # -----------------------------
        dpi = int(os.getenv("VLM_DPI", 200))
        max_image_long_side = int(
            os.getenv("VLM_MAX_IMAGE_LONG_SIDE", 2048)
        )
        max_new_tokens = int(
            os.getenv("VLM_MAX_NEW_TOKENS", 1800)
        )
        temperature = float(
            os.getenv("VLM_TEMPERATURE", 0.0)
        )

        # -----------------------------
        # Offline flags
        # -----------------------------
        transformers_offline = parse_bool(
            os.getenv("TRANSFORMERS_OFFLINE"),
            True,
        )

        hf_hub_offline = parse_bool(
            os.getenv("HF_HUB_OFFLINE"),
            True,
        )

        # -----------------------------
        # Logging
        # -----------------------------
        log_level = os.getenv("VLM_LOG_LEVEL", "INFO")

        return cls(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            device_map=os.getenv("VLM_DEVICE_MAP", "auto"),
            dpi=dpi,
            max_image_long_side=max_image_long_side,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            transformers_offline=transformers_offline,
            hf_hub_offline=hf_hub_offline,
            log_level=log_level,
        )
