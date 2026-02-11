"""
scan_project.py

Robust project scanner for Windows + WSL mixed environments.
Handles:
- Broken symlinks
- Inaccessible directories
- Corrupt cache folders
- Permission errors

Outputs project_scan_report.txt
"""

import os
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(".")
OUTPUT_NAME = "project_scan_report.txt"
LARGE_FILE_MB = 50


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def format_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def should_flag(path: Path):
    name = path.name.lower()

    ignore_patterns = [
        "__pycache__",
        ".venv",
        "venv",
        ".cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "node_modules",
        ".ipynb_checkpoints",
    ]

    binary_exts = [
        ".safetensors",
        ".pt",
        ".bin",
        ".ckpt",
        ".log",
        ".sqlite",
        ".db",
    ]

    if any(p in name for p in ignore_patterns):
        return "COMMON_IGNORE"

    if path.suffix.lower() in binary_exts:
        return "MODEL_OR_BINARY"

    try:
        if path.is_file() and path.stat().st_size > LARGE_FILE_MB * 1024 * 1024:
            return "LARGE_FILE"
    except Exception:
        return "UNREADABLE"

    return None


# ------------------------------------------------------------
# Safe Scanner
# ------------------------------------------------------------

def scan_directory(root: Path):

    lines = []
    lines.append("Project Scan Report")
    lines.append(f"Root: {root.resolve()}")
    lines.append(f"Generated: {datetime.now()}")
    lines.append("=" * 60)
    lines.append("")

    for current_root, dirs, files in os.walk(root, topdown=True, onerror=lambda e: None):

        current_path = Path(current_root)

        # Remove problematic directories BEFORE walking into them
        dirs[:] = [
            d for d in dirs
            if not any(x in d.lower() for x in ["venv", "lib64", ".cache"])
        ]

        indent = "  " * len(current_path.relative_to(root).parts)

        lines.append(f"{indent}[DIR] {current_path.name}")

        for file in files:
            file_path = current_path / file

            try:
                flag = should_flag(file_path)

                try:
                    size = format_size(file_path.stat().st_size)
                except Exception:
                    size = "UNREADABLE"

                if flag:
                    lines.append(
                        f"{indent}  [FILE] {file} ({size})  <-- {flag}"
                    )
                else:
                    lines.append(
                        f"{indent}  [FILE] {file} ({size})"
                    )

            except Exception as e:
                lines.append(f"{indent}  [ERROR] {file} -> {e}")

    return lines


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    report_lines = scan_directory(ROOT_DIR)

    output_path = Path(OUTPUT_NAME)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nReport written to: {output_path.resolve()}")
