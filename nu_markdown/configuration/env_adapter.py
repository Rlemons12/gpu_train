"""
Standalone WSL / Windows Environment Adapter
---------------------------------------------

Loads a LOCAL .env file located at the NuMarkdown project root.

Design goals:
  • Self-contained
  • No dependency on external EMTAC env
  • Automatic Windows → WSL path translation
  • Safe to import multiple times
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from dotenv import dotenv_values

log = logging.getLogger(__name__)

# ---------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------
# configuration/ → parents[1] = nu_markdown
BASE_DIR = Path(__file__).resolve().parents[1]

LOCAL_ENV_PATH = BASE_DIR / ".env"

# Optional override
ENV_OVERRIDE_KEY = "NUMARKDOWN_ENV"


# ---------------------------------------------------------
# REGEX
# ---------------------------------------------------------
_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.+)")
_WSL_PATH_RE = re.compile(r"^/mnt/[a-z]/")


# ---------------------------------------------------------
# PATH NORMALIZATION
# ---------------------------------------------------------
def windows_to_wsl_path(value: str) -> str:
    value = value.strip().strip('"').strip("'")

    match = _WINDOWS_PATH_RE.match(value)
    if not match:
        return value

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")

    return f"/mnt/{drive}/{rest}"


def normalize_path(value: str) -> str:
    if not isinstance(value, str):
        return value

    value = value.strip().strip('"').strip("'")

    if _WSL_PATH_RE.match(value):
        return value

    if _WINDOWS_PATH_RE.match(value):
        return windows_to_wsl_path(value)

    return value


# ---------------------------------------------------------
# ENV PATH RESOLUTION
# ---------------------------------------------------------
def resolve_env_path() -> Path:
    override = os.environ.get(ENV_OVERRIDE_KEY)
    if override:
        p = Path(normalize_path(override))
        if p.exists():
            return p
        raise FileNotFoundError(f"{ENV_OVERRIDE_KEY} set but not found: {p}")

    if LOCAL_ENV_PATH.exists():
        return LOCAL_ENV_PATH

    raise FileNotFoundError(
        f".env not found at: {LOCAL_ENV_PATH}"
    )


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
def load_local_env(*, override_existing: bool = False) -> Path:
    env_path = resolve_env_path()
    values = dotenv_values(env_path)

    for key, val in values.items():
        if not val:
            continue

        adapted = normalize_path(val)

        if override_existing or key not in os.environ:
            os.environ[key] = adapted
            log.debug(f"[ENV] {key} = {adapted}")

    log.info(f"[ENV] Loaded local env from: {env_path}")
    return env_path


# ---------------------------------------------------------
# AUTO-LOAD
# ---------------------------------------------------------
_loaded_env_path = load_local_env()
