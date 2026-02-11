"""
setup_environment.py

Sets up a self-contained environment for the NuMarkdown VLM PDF extractor.

What it does:
- Creates a venv in ./venv if missing
- Installs required packages
- Verifies CUDA
- Verifies model snapshot path
- Verifies PyMuPDF backend
- Creates required working folders
- Runs optional smoke test

This script is portable and safe to re-run.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import venv


# =========================
# CONFIGURE THIS
# =========================

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"

# Change to your real model snapshot path
MODEL_PATH = Path(r"C:\Users\operator\emtac\models\vlm\numarkdown-8b-thinking")

RENDER_DIR = PROJECT_ROOT / "rendered_pages"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_output"


REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "pillow",
    "pymupdf",
]

# =========================
# UTILITIES
# =========================

def run(cmd, cwd=None):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def create_venv():
    if VENV_DIR.exists():
        print("Virtual environment already exists.")
        return

    print("Creating virtual environment...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    print("Virtual environment created.")


def get_venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_packages():
    python = get_venv_python()
    print("Installing required packages...")

    for pkg in REQUIRED_PACKAGES:
        run(f'"{python}" -m pip install {pkg}')

    print("Dependencies installed.")


def verify_cuda():
    python = get_venv_python()

    code = """
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"""

    print("Checking CUDA...")
    run(f'"{python}" -c "{code}"')


def verify_model_snapshot():
    print("Checking model snapshot path...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model snapshot not found at: {MODEL_PATH}")

    required_files = ["configuration.json"]
    missing = [f for f in required_files if not (MODEL_PATH / f).exists()]
    if missing:
        print("Warning: Some expected model files are missing:", missing)
    else:
        print("Model snapshot looks valid.")


def verify_pymupdf():
    python = get_venv_python()
    print("Checking PyMuPDF...")

    code = """
import fitz
print("PyMuPDF version:", fitz.__doc__[:50])
"""
    run(f'"{python}" -c "{code}"')


def create_directories():
    print("Creating working directories...")
    RENDER_DIR.mkdir(exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    print("Directories ready.")


def run_smoke_test():
    python = get_venv_python()
    print("Running basic import smoke test...")

    code = """
from vlm_scanned_pdf_to_markdown import VLMExtractorConfig
print("Module import successful.")
"""

    run(f'"{python}" -c "{code}"')


# =========================
# MAIN
# =========================

def main():
    print("\n=== NuMarkdown VLM Setup Script ===\n")

    create_venv()
    install_packages()
    verify_cuda()
    verify_model_snapshot()
    verify_pymupdf()
    create_directories()
    run_smoke_test()

    print("\nSETUP COMPLETE")
    print(f"\nTo activate the environment:")
    if os.name == "nt":
        print(f'  {VENV_DIR}\\Scripts\\activate')
    else:
        print(f'  source {VENV_DIR}/bin/activate')


if __name__ == "__main__":
    main()
