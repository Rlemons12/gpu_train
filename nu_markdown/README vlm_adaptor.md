# VLM README Generator Script

## Overview

This document provides a complete solution for automatically generating `README_VLM_ADAPTER.md` documentation for the NuMarkdown Vision-Language Model adapter.

---

## Quick Start

### Single Copy-Paste Script

Save this as `create_vlm_readme.py` anywhere inside your `NuMarkdown-8B-Thinking` folder:

```python
"""
create_vlm_readme.py

Creates README_VLM_ADAPTER.md in the current directory.
Run:
    python create_vlm_readme.py
"""

from pathlib import Path

README_CONTENT = """# VLM Adapter Usage Guide

## Overview

The VLMAdapter wraps the NuMarkdown Vision-Language model
into simple reusable methods for:

- Image description
- Image → Markdown extraction
- Scanned PDF → Markdown extraction
- Generic multimodal chat

It ensures:
- Model loads once
- GPU memory handled correctly
- Offline operation supported
- Clean Markdown output (no <think> blocks)

---

# Requirements

- Python 3.10+
- torch (CUDA-enabled)
- transformers
- Pillow
- PyMuPDF or pdf2image
- Local offline model snapshot

Environment variable required:
VLM_MODEL_PATH

---

# Basic Usage

## Import Adapter

```python
from adapters.vlm_adapter import VLMAdapter

adapter = VLMAdapter()
```

## Describe Image

```python
description = adapter.describe_image(
    "C:/Users/operator/Pictures/example.jpg"
)
print(description)
```

## Extract Markdown From Image

```python
markdown = adapter.extract_markdown_from_image(
    "C:/Users/operator/Pictures/document_page.jpg"
)
print(markdown)
```

## Extract Markdown From PDF

```python
result = adapter.extract_markdown_from_pdf(
    "test/Robot I-O 1467-99-31.pdf"
)

print("Total Pages:", result["total_pages"])
print(result["chunks"][0]["markdown"])
```

### Returned structure:

```json
{
  "source_path": "...",
  "doc_type": "pdf_scanned_markdown_vlm",
  "total_pages": 1,
  "chunks": [
    {
      "page_number": 1,
      "text": "...",
      "markdown": "..."
    }
  ],
  "images": [
    {
      "page_number": 1,
      "image_path": "...",
      "dpi": 200
    }
  ]
}
```

---

# Performance Notes

- Large pages auto-resize
- VRAM usage logged
- CUDA cache cleared between pages
- Offline mode supported
- Safe for 32GB GPUs

---

# Recommended Pattern

```python
adapter = VLMAdapter()

for pdf_path in pdf_list:
    result = adapter.extract_markdown_from_pdf(pdf_path)
    save_to_database(result)
```

**Do NOT re-instantiate adapter per page.**
"""

def main():
    output_file = Path("README_VLM_ADAPTER.md")
    output_file.write_text(README_CONTENT, encoding="utf-8")
    print(f"README created at: {output_file.resolve()}")

if __name__ == "__main__":
    main()
```

---

## How to Use

1. **Save the script** as `create_vlm_readme.py`
2. **Navigate** to your `NuMarkdown-8B-Thinking` directory
3. **Run** the script:

```bash
python create_vlm_readme.py
```

4. **Output**: You'll see:

```
README created at: C:\Users\operator\PycharmProjects\gpu_train\NuMarkdown-8B-Thinking\README_VLM_ADAPTER.md
```

---

## Alternative: Production Script

For more advanced logging and directory handling, use this version:

Save as `NuMarkdown-8B-Thinking/scripts/generate_vlm_readme.py`:

```python
"""
generate_vlm_readme.py

Creates README_VLM_ADAPTER.md automatically.
"""

from pathlib import Path
import logging

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("generate_vlm_readme")


# -------------------------------------------------------------
# Markdown Content
# -------------------------------------------------------------

README_CONTENT = """# VLM Adapter Usage Guide

## Overview

`VLMAdapter` is a high-level wrapper around the NuMarkdown Vision-Language Model.  
It provides a simple, reusable interface for:

- Image description
- Document image → Markdown conversion
- Scanned PDF → structured Stage-1 schema extraction
- Generic multimodal chat

The adapter ensures:

- Model loads only once
- Prompt formatting is centralized
- GPU memory is handled consistently
- Offline operation is supported

---

# Project Structure

```
NuMarkdown-8B-Thinking/
│
├── adapters/
│   └── vlm_adapter.py
│
├── vlm_scanned_pdf_to_markdown.py
├── configuration/
│   └── vlm_config.py
└── models/
    └── NuMarkdown-8B-Thinking/
```

---

# Requirements

- Python 3.10+
- torch (CUDA-enabled)
- transformers
- Pillow
- PyMuPDF or pdf2image
- Offline model snapshot in:
  `models/NuMarkdown-8B-Thinking/`

Environment variable required:
```
VLM_MODEL_PATH
```

---

# Basic Usage

## Import Adapter

```python
from adapters.vlm_adapter import VLMAdapter
adapter = VLMAdapter()
```

## Image Description

```python
description = adapter.describe_image(
    "/mnt/c/Users/operator/Pictures/example.jpg"
)
print(description)
```

## Extract Markdown From Image

```python
markdown = adapter.extract_markdown_from_image(
    "/mnt/c/Users/operator/Pictures/document_page.jpg"
)
print(markdown)
```

## Extract Markdown From PDF

```python
result = adapter.extract_markdown_from_pdf(
    "test/Robot I-O 1467-99-31.pdf"
)

print("Total Pages:", result["total_pages"])
print(result["chunks"][0]["markdown"])
```

### Returned schema:

```json
{
  "source_path": "...",
  "doc_type": "pdf_scanned_markdown_vlm",
  "total_pages": 1,
  "chunks": [
    {
      "page_number": 1,
      "text": "...",
      "markdown": "..."
    }
  ],
  "images": [
    {
      "page_number": 1,
      "image_path": "...",
      "dpi": 200
    }
  ]
}
```

## Generic Multimodal Chat

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain industrial automation."}
        ],
    }
]

response = adapter.chat(messages)
print(response)
```

---

# Performance Notes

- Large pages auto-resize (default max side: 2048px)
- VRAM usage logged
- CUDA cache cleared between pages
- Offline mode supported

For 32GB GPUs:
- Full 8B VLM runs comfortably
- Multi-page PDFs are safe

---

# Recommended Usage Pattern

```python
adapter = VLMAdapter()

for pdf_path in pdf_list:
    result = adapter.extract_markdown_from_pdf(pdf_path)
    save_to_database(result)
```

**Do not re-instantiate adapter per page.**

---

# Summary

`VLMAdapter` centralizes:

- Prompt logic
- Device handling
- Model loading
- Output cleaning

This keeps your GPU pipeline modular and stable.
"""

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "README_VLM_ADAPTER.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_path.write_text(README_CONTENT, encoding="utf-8")
    
    LOGGER.info("README generated at: %s", output_path)


if __name__ == "__main__":
    main()
```

### Run from project root:

```bash
python scripts/generate_vlm_readme.py
```

Output:
```
INFO:generate_vlm_readme:README generated at: .../README_VLM_ADAPTER.md
```

---

## Project Context

### File Structure

```
NuMarkdown-8B-Thinking/
│
├── adapters/
│   └── vlm_adapter.py          # Main VLM wrapper
│
├── scripts/
│   └── generate_vlm_readme.py  # Production README generator
│
├── configuration/
│   └── vlm_config.py           # VLM configuration
│
├── models/
│   └── NuMarkdown-8B-Thinking/ # Local model snapshot
│
├── vlm_scanned_pdf_to_markdown.py
└── README_VLM_ADAPTER.md       # Generated documentation
```

---

## Future Enhancements

Potential improvements for the generator:

1. **Auto-detect model path** and inject into README
2. **Detect CUDA version** and add to requirements
3. **Extract from docstrings** to generate API documentation
4. **Version metadata** from git tags
5. **Multiple docs** (API.md, ARCHITECTURE.md, etc.)

---

## Benefits

✅ **Automated documentation** - No manual copy-paste needed  
✅ **Version control friendly** - Regenerate on demand  
✅ **Consistent formatting** - Single source of truth  
✅ **Easy updates** - Modify content in one place  
✅ **Project-aware** - Automatically finds correct paths