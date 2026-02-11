# dataset_qanda_generator/models/tiny_llama.py
# -*- coding: utf-8 -*-

import logging
import os
import re
from pathlib import Path
from typing import Optional

from .base_causal import CausalLMAnswerBase

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# SAFE MODEL DIR RESOLUTION (NO IMPORT-TIME ENV ACCESS)
# ----------------------------------------------------------------------
def _resolve_tiny_llama_dir(explicit_dir: Optional[str | Path] = None) -> Path:
    """
    Resolve TinyLlama model directory safely.

    Priority:
        1) Explicit constructor argument
        2) MODELS_TINY_LLAMA_DIR environment variable

    Raises a clear error ONLY if TinyLlama is instantiated
    and the directory cannot be resolved.
    """
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()

    env_dir = os.environ.get("MODELS_TINY_LLAMA_DIR")
    if not env_dir:
        raise RuntimeError(
            "TinyLlama is enabled but MODELS_TINY_LLAMA_DIR is not set.\n"
            "Set it via:\n\n"
            "  export MODELS_TINY_LLAMA_DIR=/path/to/tinyllama\n\n"
            "or disable TinyLlama in the qna_models table."
        )

    return Path(env_dir).expanduser().resolve()


class TinyLlamaAnswerGenerator(CausalLMAnswerBase):
    """
    Unified TinyLlama wrapper for Option-C.

    Design goals:
        • Strict factual grounding
        • One-sentence enforcement
        • Unified cleanup pipeline
        • Identical interface to Qwen / Gemma / OpenELM
        • Prevents TinyLlama-style hallucinations
        • SAFE lazy initialization
    """

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        device: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        resolved_dir = _resolve_tiny_llama_dir(model_dir)

        super().__init__(
            model_dir=str(resolved_dir),
            name="TINY",
            device=device,
            trust_remote_code=False,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

        log.info("[TINY] Unified Option-C TinyLlama wrapper initialized")
        log.debug("[TINY] Model dir resolved to: %s", resolved_dir)

    # ================================================================
    # PROMPT
    # ================================================================
    def build_prompt(self, context: str, question: str) -> str:
        """
        TinyLlama performs best with a concise instruction format.
        No multi-turn chat, no role tags, no fluff.
        """

        return (
            "You answer questions using ONLY the information in the context.\n"
            "If the answer cannot be found in the context, respond exactly:\n"
            "\"Insufficient information in context.\"\n"
            "Respond in one concise sentence.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    # ================================================================
    # OUTPUT CLEANUP
    # ================================================================
    def _clean_output(
        self,
        answer: str,
        strict_factual: bool,
        one_sentence: bool,
    ) -> str:
        if not answer:
            return ""

        ans = answer.strip()

        # Remove TinyLlama junk prefixes
        bad_patterns = [
            r"^assistant[:\-]\s*",
            r"^answer[:\-]\s*",
            r"^response[:\-]\s*",
            r"^a:\s*",
            r"^q:\s*",
            r"^here is.*?:\s*",
        ]
        for pat in bad_patterns:
            ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()

        # Delegate to unified cleanup
        return super()._clean_output(
            answer=ans,
            strict_factual=strict_factual,
            one_sentence=one_sentence,
        )

    # ================================================================
    # FINAL ANSWER ENTRY POINT
    # ================================================================
    def generate_answer(
        self,
        context: str,
        question: str,
        temperature: float = 0.4,
        max_new_tokens: int = 80,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        do_sample: bool = True,
        strict_factual: bool = True,
        one_sentence: bool = True,
        aggressive_grounding: bool = False,
        debug_raw: bool = False,
    ) -> str:

        return super().generate_answer(
            context=context,
            question=question,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            strict_factual=strict_factual,
            one_sentence=one_sentence,
            aggressive_grounding=aggressive_grounding,
            debug_raw=debug_raw,
        )
