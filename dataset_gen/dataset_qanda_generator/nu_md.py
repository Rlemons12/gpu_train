"""
nu_md.py

Interactive CLI wrapper for NuMarkdown VLM.

• Prompts user for file path
• Auto-converts Windows paths to WSL paths
• Runs offline VLM extraction
• Writes Markdown output to current working directory
"""

import os
import sys
from pathlib import Path

# -----------------------------------------------------------
# Ensure nu_markdown package is importable
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from nu_markdown.vlm_scanned_pdf_to_markdown import (
    build_config_from_env,
    scanned_pdf_to_stage1_schema,
)


# -----------------------------------------------------------
# Windows → WSL path converter
# -----------------------------------------------------------
def windows_to_wsl_path(path_str: str) -> str:
    path_str = path_str.strip('"').strip("'")

    # If already Linux style
    if path_str.startswith("/mnt/"):
        return path_str

    # Convert C:\Users\... → /mnt/c/Users/...
    if ":" in path_str:
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    return path_str


# -----------------------------------------------------------
# Markdown writer
# -----------------------------------------------------------
def write_markdown_output(result: dict, original_path: str):
    file_name = Path(original_path).stem
    output_file = Path.cwd() / f"{file_name}.md"

    full_markdown = "\n\n".join(
        chunk["markdown"] for chunk in result["chunks"]
    )

    output_file.write_text(full_markdown, encoding="utf-8")

    print("\n✅ Markdown written to:")
    print(output_file)


# -----------------------------------------------------------
# Main CLI
# -----------------------------------------------------------
def main():
    print("\n=== NuMarkdown Interactive CLI ===")
    print("Paste file path (PDF or image).")
    print("Example:")
    print(r'C:\Users\operator\Pictures\example.pdf')
    print()

    user_input = input("File path: ").strip()

    if not user_input:
        print("No file provided. Exiting.")
        return

    linux_path = windows_to_wsl_path(user_input)

    if not Path(linux_path).exists():
        print("\n❌ File does not exist:")
        print(linux_path)
        return

    print("\n🔄 Processing:", linux_path)

    cfg = build_config_from_env()

    result = scanned_pdf_to_stage1_schema(
        pdf_path=linux_path,
        cfg=cfg,
    )

    print("\n📄 Total Pages:", result["total_pages"])

    write_markdown_output(result, linux_path)


if __name__ == "__main__":
    main()
