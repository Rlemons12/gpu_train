"""
vlm_adapter.py

Unified Vision-Language Model Adapter

Provides:
    - describe_image()
    - extract_markdown_from_image()
    - extract_markdown_from_pdf()
    - chat()
"""

from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image

from .vlm_scanned_pdf_to_markdown import (
    build_config_from_env,
    VLMMarkdownExtractor,
    scanned_pdf_to_stage1_schema,
)


class VLMAdapter:
    """
    High-level wrapper around VLMMarkdownExtractor.
    """

    def __init__(self):
        self.cfg = build_config_from_env()
        self.extractor = VLMMarkdownExtractor(self.cfg)

    # --------------------------------------------------
    # Core Chat Method (lowest level)
    # --------------------------------------------------
    def _run_image_prompt(self, image_path: str, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.extractor.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.extractor.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.extractor.model.device)

        with torch.inference_mode():
            output_ids = self.extractor.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=self.cfg.repetition_penalty,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]

        output_text = self.extractor.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        if "<think>" in output_text:
            output_text = output_text.split("</think>")[-1]

        return output_text.strip()

    # --------------------------------------------------
    # Public Methods
    # --------------------------------------------------

    def describe_image(self, image_path: str) -> str:
        prompt = (
            "Describe this image in precise detail.\n"
            "Be factual and structured.\n"
            "Do NOT include reasoning.\n"
            "Do NOT include <think> blocks.\n"
            "Output ONLY the description."
        )
        return self._run_image_prompt(image_path, prompt)

    def extract_markdown_from_image(self, image_path: str) -> str:
        prompt = (
            "Convert this document image into clean structured Markdown.\n"
            "Preserve headings, tables, lists, indentation exactly.\n"
            "Do NOT include reasoning.\n"
            "Do NOT include <think> blocks.\n"
            "Output ONLY the Markdown."
        )
        return self._run_image_prompt(image_path, prompt)

    def extract_markdown_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        return scanned_pdf_to_stage1_schema(
            pdf_path=pdf_path,
            cfg=self.cfg,
        )

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generic multimodal chat interface.
        """
        text = self.extractor.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.extractor.processor(
            text=[text],
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.extractor.model.device)

        with torch.inference_mode():
            output_ids = self.extractor.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]

        return self.extractor.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

    def answer_from_chunk(self, question: str, context: str) -> str:

        messages = [
            {
                "role": "system",
                "content": "You are a precise industrial documentation assistant."
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer ONLY using the provided context."
                ),
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
