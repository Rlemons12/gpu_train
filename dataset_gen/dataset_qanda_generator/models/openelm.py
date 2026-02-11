# dataset_qanda_generator/models/openelm.py
# -*- coding: utf-8 -*-

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_causal import CausalLMAnswerBase

log = logging.getLogger(__name__)

DEFAULT_OPENELM_DIR = Path(os.environ["MODELS_APPLE_ELM_DIR"])


class OpenELMAnswerGenerator(CausalLMAnswerBase):
    """
    Apple OpenELM-1.1B Instruct unified Option-C wrapper.

    Upgrades:
        • Uses CausalLMAnswerBase for:
            - strict factual mode
            - one-sentence answers
            - grounding check
            - unified cleanup pipeline
            - debug_raw support
        • Uses ChatML formatting required by OpenELM
        • Converts system/user messages automatically
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_OPENELM_DIR,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__(
            model_dir=model_dir,
            name="OPENELM",
            device=device,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        log.info("[OPENELM] Unified Option-C wrapper initialized")

    # ==================================================================
    # PROMPT
    # ==================================================================
    def build_prompt(self, context: str, question: str) -> str:
        """
        ChatML prompt, but designed to behave EXACTLY like the new base class:
            - No rambling
            - No multi-turn garbage
            - Factual, concise, grounded
        """

        return (
            "<|im_start|>system\n"
            "You answer ONLY using the context provided.\n"
            "If the answer cannot be found in the context, reply:\n"
            "\"Insufficient information in context.\"\n"
            "Only one concise sentence.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # ==================================================================
    # CUSTOM CLEANUP (OpenELM emits weird ChatML leftovers)
    # ==================================================================
    def _clean_output(
        self,
        answer: str,
        strict_factual: bool,
        one_sentence: bool,
    ) -> str:

        if not answer:
            return ""

        ans = answer.strip()

        # Remove ChatML artifacts
        ans = re.sub(r"<\|im_start\|>assistant", "", ans).strip()
        ans = re.sub(r"<\|im_start\|>user", "", ans).strip()
        ans = re.sub(r"<\|im_end\|>", "", ans).strip()
        ans = ans.replace("<<<", "").replace(">>", "")
        ans = ans.replace("<|im_start|>", "").replace("<|im_end|>", "")

        # Delegate the rest of the cleanup to the unified base class
        return super()._clean_output(
            answer=ans,
            strict_factual=strict_factual,
            one_sentence=one_sentence,
        )

    # ==================================================================
    # ENTRY POINT – inherits grounding, strict factual, one-sentence
    # ==================================================================
    @torch.no_grad()
    def generate_answer(
        self,
        context: str,
        question: str,
        temperature: float = 0.3,
        max_new_tokens: int = 80,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        do_sample: bool = False,
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
