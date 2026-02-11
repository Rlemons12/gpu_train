"""
WSL / Windows Environment Adapter
---------------------------------

Loads a GLOBAL Windows-authored .env file and adapts it
for WSL execution by converting Windows paths → /mnt/* paths.

Design goals:
  • Single source of truth (.env authored on Windows)
  • Automatic Windows → WSL path translation
  • Safe to import multiple times
  • Works in:
        - Windows Python
        - WSL Python
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from dotenv import dotenv_values

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# Authoritative Windows-authored .env
WINDOWS_ENV_PATH = r"C:\Users\operator\emtac\dev_env\.env"

# WSL-visible equivalent
WSL_ENV_PATH = "/mnt/c/Users/operator/emtac/dev_env/.env"

# Optional override (highest priority)
ENV_OVERRIDE_KEY = "EMTAC_GLOBAL_ENV"


# ------------------------------------------------------------------
# REGEX
# ------------------------------------------------------------------

_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.+)")
_WSL_PATH_RE = re.compile(r"^/mnt/[a-z]/")


# ------------------------------------------------------------------
# PATH NORMALIZATION
# ------------------------------------------------------------------

def windows_to_wsl_path(value: str) -> str:
    """
    Convert a Windows absolute path to a WSL path.

    Example:
        C:\\Users\\operator\\foo
        → /mnt/c/Users/operator/foo
    """
    value = value.strip().strip('"').strip("'")

    match = _WINDOWS_PATH_RE.match(value)
    if not match:
        return value

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")

    return f"/mnt/{drive}/{rest}"


def normalize_path(value: str) -> str:
    """
    Normalize environment values:
      • Windows path → WSL path
      • Leave WSL paths untouched
      • Leave non-path values untouched
    """
    if not isinstance(value, str):
        return value

    value = value.strip().strip('"').strip("'")

    if _WSL_PATH_RE.match(value):
        return value

    if _WINDOWS_PATH_RE.match(value):
        return windows_to_wsl_path(value)

    return value


def looks_like_path(value: str) -> bool:
    return isinstance(value, str) and (
        _WINDOWS_PATH_RE.match(value)
        or _WSL_PATH_RE.match(value)
    )


# ------------------------------------------------------------------
# ENV PATH RESOLUTION
# ------------------------------------------------------------------

def resolve_global_env_path() -> Path:
    """
    Resolve the global .env path with priority:

      1. EMTAC_GLOBAL_ENV (explicit override)
      2. Windows path (native Windows execution)
      3. WSL-mounted path
    """
    override = os.environ.get(ENV_OVERRIDE_KEY)
    if override:
        p = Path(normalize_path(override))
        if p.exists():
            return p
        raise FileNotFoundError(f"{ENV_OVERRIDE_KEY} set but not found: {p}")

    win = Path(WINDOWS_ENV_PATH)
    if win.exists():
        return win

    wsl = Path(WSL_ENV_PATH)
    if wsl.exists():
        return wsl

    raise FileNotFoundError(
        "Global .env not found. Tried:\n"
        f"  • {WINDOWS_ENV_PATH}\n"
        f"  • {WSL_ENV_PATH}\n"
        f"  • ${ENV_OVERRIDE_KEY}"
    )


# ------------------------------------------------------------------
# LOAD + ADAPT
# ------------------------------------------------------------------

def load_global_env_for_wsl(*, override_existing: bool = False) -> Path:
    """
    Load the global .env and adapt Windows paths → WSL paths.

    Args:
        override_existing:
            False (default) → do NOT clobber existing os.environ
            True            → force overwrite

    Returns:
        Path to the .env that was loaded
    """
    env_path = resolve_global_env_path()
    values = dotenv_values(env_path)

    for key, val in values.items():
        if not val:
            continue

        adapted = normalize_path(val)

        if override_existing or key not in os.environ:
            os.environ[key] = adapted
            log.debug(f"[ENV] {key} = {adapted}")

    log.info(f"[ENV] Loaded global env from: {env_path}")
    return env_path


# ------------------------------------------------------------------
# AUTO-LOAD (INTENTIONAL, SAFE)
# ------------------------------------------------------------------

_loaded_env_path = load_global_env_for_wsl(override_existing=True)

