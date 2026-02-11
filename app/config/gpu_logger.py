"""
app/configuration/gpu_logger.py
============================================================

GPU Service Logger (DOMAIN-LEVEL LOGGER)

WHAT THIS MODULE DOES
---------------------
• Provides a **dedicated logger for GPU-related activity**
• Safe for **FastAPI + uvicorn** (no Flask assumptions)
• Thread-safe **request ID tracking** using thread-local storage
• Logs to **both file and console**
• Designed for:
    - Model loading / eviction
    - GPU routing decisions
    - Memory pressure / OOM diagnostics
    - Per-request inference tracing
    - Performance and lifecycle events
    - Tensor / model parallel diagnostics

WHAT THIS MODULE DOES *NOT* DO
------------------------------
• Does NOT configure uvicorn / FastAPI logging
• Does NOT use dictConfig / LOGGING_CONFIG
• Does NOT touch access logs or framework logs
• Does NOT rely on fragile LogRecord fields

INTENDED USAGE
--------------
Import and use directly inside GPU / model / inference code:

    from app.configuration.gpu_logger import gpu_info, gpu_warning

    gpu_info("Loading mistral on cuda:1")
    gpu_warning("Evicting qwen due to memory pressure")

This logger intentionally exists **alongside** (not instead of)
the uvicorn/FastAPI logging configuration.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------
# Resolve project root safely (…/app/)
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GPU_LOG_FILE = LOG_DIR / "gpu_service.log"

# ---------------------------------------------------------
# Thread-local request ID (FastAPI-safe)
# ---------------------------------------------------------
# Used to correlate GPU events across a single request
_local = threading.local()


def get_request_id() -> str:
    """Return current request ID or create one if missing."""
    if hasattr(_local, "request_id"):
        return _local.request_id
    rid = str(uuid.uuid4())[:8]
    _local.request_id = rid
    return rid


def set_request_id(request_id: Optional[str] = None) -> str:
    """Explicitly set a request ID (useful at request entry)."""
    rid = request_id or str(uuid.uuid4())[:8]
    _local.request_id = rid
    return rid


def clear_request_id():
    """Clear request ID after request completion."""
    if hasattr(_local, "request_id"):
        delattr(_local, "request_id")


# ---------------------------------------------------------
# GPU Logger definition
# ---------------------------------------------------------
gpu_logger = logging.getLogger("ematac_gpu")

LOG_LEVEL = os.getenv("GPU_LOG_LEVEL", "INFO").upper()
gpu_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

gpu_logger.propagate = False

# Prevent duplicate handlers on reload (uvicorn, dev mode)
if not gpu_logger.handlers:

    # ---------------- File handler ----------------
    file_handler = RotatingFileHandler(
        GPU_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | GPU | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # ---------------- Console handler ----------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)

    gpu_logger.addHandler(file_handler)
    gpu_logger.addHandler(console_handler)

# ---------------------------------------------------------
# Core logging helpers (request-ID aware)
# ---------------------------------------------------------
def gpu_log(level: int, message: str, request_id: Optional[str] = None):
    """
    Core logging helper.
    Automatically injects request ID if not provided.
    """
    rid = request_id or get_request_id()
    gpu_logger.log(level, f"[REQ-{rid}] {message}")


def gpu_debug(msg: str, request_id: Optional[str] = None):
    gpu_log(logging.DEBUG, msg, request_id)


def gpu_info(msg: str, request_id: Optional[str] = None):
    gpu_log(logging.INFO, msg, request_id)


def gpu_warning(msg: str, request_id: Optional[str] = None):
    gpu_log(logging.WARNING, msg, request_id)


def gpu_error(msg: str, request_id: Optional[str] = None):
    gpu_log(logging.ERROR, msg, request_id)


def gpu_critical(msg: str, request_id: Optional[str] = None):
    gpu_log(logging.CRITICAL, msg, request_id)


# ---------------------------------------------------------
# Diagnostic helpers (NEW — drop-in safe)
# ---------------------------------------------------------
def gpu_snapshot(prefix: str = "", request_id: Optional[str] = None):
    """
    Log a lightweight GPU memory snapshot.

    Safe to call frequently.
    Zero allocations.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            gpu_debug(f"{prefix} | CUDA not available", request_id)
            return

        parts = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            parts.append(
                f"cuda:{i} free={free // (1024**2)}MB total={total // (1024**2)}MB"
            )

        gpu_debug(
            f"{prefix} | GPU snapshot | " + " | ".join(parts),
            request_id,
        )
    except Exception as e:
        gpu_warning(f"{prefix} | GPU snapshot failed: {e}", request_id)


def gpu_phase(phase: str, seconds: float, request_id: Optional[str] = None):
    """
    Structured phase timing log (machine-parseable).
    """
    gpu_info(f"PHASE={phase} duration={seconds:.3f}s", request_id)


def gpu_oom(
    model: str,
    device: str,
    context: str,
    request_id: Optional[str] = None,
):
    """
    Explicit OOM diagnostic entry.
    """
    gpu_error(
        f"OOM | model={model} device={device} context={context}",
        request_id,
    )


def gpu_shard_info(
    model: str,
    devices: List[str],
    request_id: Optional[str] = None,
):
    """
    Log tensor/model parallel shard placement.
    """
    gpu_info(
        f"MODEL_SHARDED | model={model} devices={','.join(devices)}",
        request_id,
    )
