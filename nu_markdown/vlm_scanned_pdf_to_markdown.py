"""
vlm_scanned_pdf_to_markdown.py

Offline-first scanned-PDF -> structured Markdown extractor using a Vision-Language Model.
Designed as a separate module that outputs Stage-1 schema:

{
  "source_path": str,
  "doc_type": str,
  "total_pages": int,
  "chunks": list,
  "images": list
}
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq



try:
    import fitz
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

try:
    from pdf2image import convert_from_path
    _HAVE_PDF2IMAGE = True
except Exception:
    _HAVE_PDF2IMAGE = False

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

from .configuration.vlm_config import VLMConfig

# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger("vlm_pdf_md")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    LOGGER.addHandler(h)



def build_config_from_env() -> VLMExtractorConfig:
    vlm_cfg = VLMConfig.from_env()

    # Convert torch dtype to string expected by extractor
    if vlm_cfg.torch_dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    elif vlm_cfg.torch_dtype == torch.float16:
        dtype_str = "float16"
    elif vlm_cfg.torch_dtype == torch.float32:
        dtype_str = "float32"
    else:
        raise ValueError(f"Unsupported dtype: {vlm_cfg.torch_dtype}")

    return VLMExtractorConfig(
        model_path=str(vlm_cfg.model_path),
        torch_dtype=dtype_str,
        device_map=vlm_cfg.device_map,
        max_new_tokens=vlm_cfg.max_new_tokens,
        temperature=vlm_cfg.temperature,
        dpi=vlm_cfg.dpi,
        max_image_long_side=vlm_cfg.max_image_long_side,
        transformers_offline=vlm_cfg.transformers_offline,
    )


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class VLMExtractorConfig:
    model_path: str
    doc_type: str = "pdf_scanned_markdown_vlm"

    dpi: int = 200
    max_image_long_side: int = 2048
    image_format: str = "png"

    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True

    max_new_tokens: int = 1800
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.05

    strip_model_fences: bool = True
    persist_page_images: bool = True
    page_image_dir: Optional[str] = None

    transformers_offline: bool = True




# ============================================================
# Utilities
# ============================================================

def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _strip_code_fences(md: str) -> str:
    s = md.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s.strip()


# ============================================================
# PDF Rendering
# ============================================================

class PDFPageRenderer:

    def __init__(self, dpi: int, image_format: str = "png"):
        if not _HAVE_PIL:
            raise RuntimeError("Pillow required")
        self.dpi = dpi
        self.image_format = image_format

    def render(self, pdf_path: str, out_dir: str) -> Tuple[int, List[str]]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if _HAVE_PYMUPDF:
            return self._render_fitz(pdf_path, out_dir)
        if _HAVE_PDF2IMAGE:
            return self._render_pdf2image(pdf_path, out_dir)

        raise RuntimeError("No PDF backend available")

    def _render_fitz(self, pdf_path: str, out_dir: str):
        doc = fitz.open(pdf_path)
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        img_paths = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = str(Path(out_dir) / f"page_{i+1:04d}.{self.image_format}")
            pix.save(img_path)
            img_paths.append(img_path)

        doc.close()
        return len(img_paths), img_paths

    def _render_pdf2image(self, pdf_path: str, out_dir: str):
        images = convert_from_path(pdf_path, dpi=self.dpi)
        img_paths = []
        for i, img in enumerate(images):
            img_path = str(Path(out_dir) / f"page_{i+1:04d}.{self.image_format}")
            img.save(img_path)
            img_paths.append(img_path)
        return len(img_paths), img_paths


# ============================================================
# VLM Extractor
# ============================================================

class VLMMarkdownExtractor:

    def __init__(self, cfg: VLMExtractorConfig):
        self.cfg = cfg

        if cfg.transformers_offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        dtype = _dtype_from_str(cfg.torch_dtype)


        LOGGER.info("Loading processor: %s", cfg.model_path)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            local_files_only=True,
            trust_remote_code=cfg.trust_remote_code,
        )

        LOGGER.info("Loading model: %s", cfg.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_path,
            local_files_only=True,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            device_map=cfg.device_map,
        )

        self.model.eval()

    def page_to_markdown(self, image_path: str, page_number: int) -> str:
        try:
            img = Image.open(image_path).convert("RGB")

            # ---------------------------------------------------
            # Resize guard (VERY important for large PDF pages)
            # ---------------------------------------------------
            max_side = self.cfg.max_image_long_side
            w, h = img.size
            long_side = max(w, h)

            if long_side > max_side:
                scale = max_side / long_side
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size)
                LOGGER.info(
                    "Page %d resized from %sx%s to %sx%s",
                    page_number,
                    w, h,
                    new_size[0], new_size[1]
                )

            # ---------------------------------------------------
            # Build multimodal chat message
            # ---------------------------------------------------
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Convert this scanned document page into clean structured Markdown.\n"
                                "Preserve headings, tables, lists, indentation exactly as they appear.\n"
                                "Do NOT include reasoning.\n"
                                "Do NOT include analysis.\n"
                                "Do NOT include <think> blocks.\n"
                                "Output ONLY the final Markdown."
                            ),
                        },
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text],
                images=[img],
                return_tensors="pt",
            )

            # ---------------------------------------------------
            # Move tensors to GPU
            # ---------------------------------------------------
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.model.device)

            if torch.cuda.is_available():
                LOGGER.info(
                    "Page %d VRAM before generate: %.2f GB",
                    page_number,
                    torch.cuda.memory_allocated() / 1e9
                )

            # ---------------------------------------------------
            # Generate
            # ---------------------------------------------------
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=self.cfg.repetition_penalty,
                )

            # ---------------------------------------------------
            # Remove prompt tokens from output
            # ---------------------------------------------------
            generated_ids = output_ids[:, inputs["input_ids"].size(1):]

            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            # Remove thinking blocks if present
            if "<think>" in output_text:
                output_text = output_text.split("</think>")[-1].strip()

            if self.cfg.strip_model_fences:
                output_text = _strip_code_fences(output_text)

            if torch.cuda.is_available():
                LOGGER.info(
                    "Page %d VRAM after generate: %.2f GB",
                    page_number,
                    torch.cuda.memory_allocated() / 1e9
                )
                torch.cuda.empty_cache()

            LOGGER.info("Page %d complete", page_number)

            return output_text

        except Exception as e:
            LOGGER.exception("Page %d failed: %s", page_number, str(e))
            return f"<!-- PAGE {page_number} FAILED: {str(e)} -->"


# ============================================================
# Stage-1 Builder
# ============================================================

def scanned_pdf_to_stage1_schema(pdf_path: str, cfg: VLMExtractorConfig) -> Dict[str, Any]:

    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        raise FileNotFoundError(pdf_path)

    page_dir = Path(cfg.page_image_dir or tempfile.mkdtemp())
    renderer = PDFPageRenderer(cfg.dpi, cfg.image_format)
    total_pages, images = renderer.render(pdf_path, str(page_dir))

    extractor = VLMMarkdownExtractor(cfg)

    chunks = []
    image_meta = []

    for i, img in enumerate(images):
        page_number = i + 1

        image_meta.append({
            "page_number": page_number,
            "image_path": img,
            "dpi": cfg.dpi,
        })

        md = extractor.page_to_markdown(img, page_number)

        chunks.append({
            "page_number": page_number,
            "text": md,
            "markdown": md,
        })

    return {
        "source_path": pdf_path,
        "doc_type": cfg.doc_type,
        "total_pages": total_pages,
        "chunks": chunks,
        "images": image_meta,
    }


# ============================================================
# CLI
# ============================================================

def _cli():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--model-path", required=False)
    p.add_argument("--use-env", action="store_true")

    args = p.parse_args()

    if args.use_env:
        cfg = build_config_from_env()
    else:
        if not args.model_path:
            raise RuntimeError("Provide --model-path or use --use-env")
        cfg = VLMExtractorConfig(model_path=args.model_path)

    result = scanned_pdf_to_stage1_schema(args.pdf, cfg)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    LOGGER.info("Output written: %s", out)


if __name__ == "__main__":
    _cli()
