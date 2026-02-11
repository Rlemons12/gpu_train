from pathlib import Path
from vlm_scanned_pdf_to_markdown import (
    build_config_from_env,
    scanned_pdf_to_stage1_schema,
)

# --------------------------------------------------
# Input PDF
# --------------------------------------------------
pdf_path = "Robot I-O 1467-99-31.pdf"

# --------------------------------------------------
# Build config from .env
# --------------------------------------------------
cfg = build_config_from_env()

# --------------------------------------------------
# Run extraction
# --------------------------------------------------
result = scanned_pdf_to_stage1_schema(
    pdf_path=pdf_path,
    cfg=cfg,
)

print("Total Pages:", result["total_pages"])

# --------------------------------------------------
# Export Markdown
# --------------------------------------------------
output_dir = Path("out_put")
output_dir.mkdir(parents=True, exist_ok=True)

full_md = "\n\n--- PAGE BREAK ---\n\n".join(
    chunk["markdown"] for chunk in result["chunks"]
)

pdf_name = Path(pdf_path).stem
output_file = output_dir / f"{pdf_name}.md"

output_file.write_text(full_md, encoding="utf-8")

print(f"\nMarkdown written to:\n{output_file}")
