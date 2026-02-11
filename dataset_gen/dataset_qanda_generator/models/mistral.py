# dataset_qanda_generator/models/mistral.py
# -*- coding: utf-8 -*-

import os
import time
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_causal import CausalLMAnswerBase

log = logging.getLogger(__name__)


def _looks_like_windows_path(p: str) -> bool:
    return len(p) > 2 and p[1] == ":" and p[2] in ("\\", "/")


def _win_to_wsl_path(p: str) -> str:
    drive = p[0].lower()
    rest = p[2:].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


class MistralAnswerGenerator(CausalLMAnswerBase):
    """
    Mistral-7B-Instruct answer generator.

    Strict behavior:
      • Answers using ONLY the provided chunk
      • No document metadata injected into the answer
      • Deterministic, training-safe output
    """

    def __init__(self, model_dir: str | None = None, device: str | None = None):
        t0 = time.time()

        if model_dir is None:
            model_dir = os.environ.get("MODELS_MISTRAL_7B_DIR")
            if not model_dir:
                raise RuntimeError("MODELS_MISTRAL_7B_DIR is not set")

        raw_dir = model_dir
        if _looks_like_windows_path(raw_dir):
            model_dir = _win_to_wsl_path(raw_dir)
            log.info("[MISTRAL] Normalized model_dir (win→wsl): %s → %s", raw_dir, model_dir)

        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Mistral model not found: {model_path}")

        self.device = device or "cpu"
        self.name = "mistral-7b-instruct"

        log.info("[%s] Loading from %s", self.name, model_path)
        log.info("[%s] Device=%s | CUDA=%s", self.name, self.device, torch.cuda.is_available())

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self.device)

        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

        log.info(
            "[%s] Ready in %.2fs | eos_token_id=%s",
            self.name,
            time.time() - t0,
            self.eos_token_id,
        )

    # ------------------------------------------------------------------
    # PROMPT (STRICT + TRAINING-SAFE)
    # ------------------------------------------------------------------
    def build_prompt(self, *, context: str, question: str) -> str:
        return (
            "You are an expert document analyst.\n"
            "Your task is to ANSWER the question below.\n\n"
            "Rules:\n"
            "- Use ONLY the provided context\n"
            "- Do NOT ask questions\n"
            "- Do NOT return JSON\n"
            "- Do NOT restate the question\n"
            "- Return a direct, complete answer in plain English\n"
            "- If the answer is not explicitly stated, reply exactly:\n"
            "  Insufficient information in context.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )

    # ------------------------------------------------------------------
    # GENERATION ENTRY POINT
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
            self,
            question: str | None = None,
            context: str | None = None,
            prompt: str | None = None,
            max_new_tokens: int = 256,
            temperature: float = 0.0,
            top_p: float = 1.0,
            **kwargs,
    ) -> str:
        """
        Backward-compatible generate().

        Accepts:
          • generate(question=..., context=...)
          • generate(prompt=...)            (legacy pipeline)
          • generate(prompt_text)           (positional legacy)

        Internally normalizes to strict grounding.
        """

        # ---------------------------------
        # Handle positional legacy call
        # ---------------------------------
        if question and not context and not prompt:
            # positional call like generate(prompt_text)
            prompt = question
            question = None

        # ---------------------------------
        # Normalize inputs
        # ---------------------------------
        if question is not None and context is not None:
            prompt_text = self.build_prompt(
                question=question,
                context=context,
            )
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError(
                "MistralAnswerGenerator.generate() requires either "
                "(question + context) or a prompt"
            )

        # ---------------------------------
        # Tokenize
        # ---------------------------------
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # ---------------------------------
        # Generate
        # ---------------------------------
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )

        generated = output_ids[0][input_ids.shape[-1]:]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return answer

