from vlm_scanned_pdf_to_markdown import (
    build_config_from_env,
    VLMMarkdownExtractor,
)

image_path = "/mnt/c/Users/operator/Pictures/HWbl-vtAv_rtpp_DFmLCMDIHFYBhiy88sdJsCOabkeQ.jpg"

cfg = build_config_from_env()
extractor = VLMMarkdownExtractor(cfg)

output = extractor.page_to_markdown(
    image_path=image_path,
    page_number=1,
)

print("\n--- IMAGE OUTPUT ---\n")
print(output)
