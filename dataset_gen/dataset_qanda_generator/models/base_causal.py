# dataset_qanda_generator/models/base_causal.py
# -*- coding: utf-8 -*-

import logging
import re
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger(__name__)


class CausalLMAnswerBase:
    """
    Unified base class for all causal Option-C models:
        - TinyLlama
        - Qwen
        - Gemma
        - OpenELM
        - FLAN causal variants

    Adds:
        • Unified output cleanup
        • Strict factual mode
        • One-sentence mode
        • Grounding/hallucination filter
        • Debug support
        • Consistent decoding
    """

    # ---------------------------------------------------------
    # Base settings (can be overridden per model)
    # ---------------------------------------------------------
    STRICT_FACTUAL_DEFAULT = True
    ONE_SENTENCE_DEFAULT = True
    AGGRESSIVE_GROUNDING_DEFAULT = False
    DEBUG_RAW_DEFAULT = False

    # ---------------------------------------------------------
    def __init__(
        self,
        model_dir: str,
        name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"{name} model not found: {model_path}")

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"[{name}] Loading from: {model_path}")
        log.info(f"[{name}] Device: {self.device}")

        # ------------------------------
        # Tokenizer
        # ------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

        # ------------------------------
        # Model Loading
        # ------------------------------
        model_kwargs = dict(
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **model_kwargs
        ).to(self.device)

        self.model.eval()

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    # =====================================================================
    # MAIN PUBLIC ENTRY POINT
    # =====================================================================
    @torch.no_grad()
    def generate_answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        seed: Optional[int] = None,
        strict_factual: Optional[bool] = None,
        one_sentence: Optional[bool] = None,
        aggressive_grounding: Optional[bool] = None,
        debug_raw: Optional[bool] = None,
    ) -> str:
        """
        Unified generation pipeline with:
            - Prompt building
            - Model inference
            - Cleanup / safety pass
            - Grounding check
        """

        strict_factual = self.STRICT_FACTUAL_DEFAULT if strict_factual is None else strict_factual
        one_sentence = self.ONE_SENTENCE_DEFAULT if one_sentence is None else one_sentence
        aggressive_grounding = (
            self.AGGRESSIVE_GROUNDING_DEFAULT if aggressive_grounding is None else aggressive_grounding
        )
        debug_raw = self.DEBUG_RAW_DEFAULT if debug_raw is None else debug_raw

        prompt = self.build_prompt(context, question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        if seed is not None:
            torch.manual_seed(seed)

        generation_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=False,
        )

        if self.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.eos_token_id

        output_ids = self.model.generate(**generation_kwargs)[0]
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Remove prompt prefix
        answer = self._strip_prompt_prefix(raw, prompt)

        # Cleanup
        cleaned = self._clean_output(
            answer,
            strict_factual=strict_factual,
            one_sentence=one_sentence,
        )

        # Debug mode (prints raw + cleaned)
        if debug_raw:
            print("\n========== DEBUG RAW ==========")
            print(raw)
            print("========== DEBUG CLEANED ==========")
            print(cleaned)
            print("====================================\n")

        # Grounding check — fallback answer
        if not self._is_grounded(cleaned, context, aggressive_grounding):
            return "Insufficient information in context."

        return cleaned

    # =====================================================================
    # PROMPT BUILDING (models override this)
    # =====================================================================
    def build_prompt(self, context: str, question: str) -> str:
        """
        Default prompt; models override for better formatting.
        """
        return (
            "CONTEXT:\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER: "
        )

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    def _strip_prompt_prefix(self, text: str, prompt: str) -> str:
        if text.startswith(prompt):
            return text[len(prompt):].strip()

        # fallback: last occurrence
        idx = text.rfind(prompt)
        if idx != -1:
            return text[idx + len(prompt):].strip()

        return text.strip()

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------
    def _clean_output(
        self,
        answer: str,
        strict_factual: bool,
        one_sentence: bool,
    ) -> str:

        if not answer:
            return ""

        ans = answer.strip()

        # Remove chat-style prefixes
        bad_prefixes = [
            r"^Human:\s*", r"^Assistant:\s*", r"^User:\s*", r"^System:\s*",
            r"^Q:\s*", r"^A:\s*", r"^Answer:\s*", r"^Context:\s*",
        ]
        for pat in bad_prefixes:
            ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()

        # Remove filler/meta content
        if strict_factual:
            bad_starts = [
                "Based on the context", "According to the context",
                "From the context", "As stated", "As described",
                "In summary", "Overall", "To summarize",
                "The answer is",
            ]
            for phrase in bad_starts:
                if ans.lower().startswith(phrase.lower()):
                    ans = ans[len(phrase):].strip(" :,-").strip()
                    break

        # Remove quotes
        ans = ans.strip("`'\"")

        # Collapse extra spaces
        ans = re.sub(r"\s+", " ", ans)

        # One sentence only
        if one_sentence:
            m = re.match(r"(.+?[.!?])(\s|$)", ans)
            if m:
                ans = m.group(1).strip()

        # Ensure punctuation
        if ans and ans[-1] not in ".!?":
            ans += "."

        return ans.strip()

    # ---------------------------------------------------------
    # Grounding / Hallucination Checks
    # ---------------------------------------------------------
    def _is_grounded(self, answer: str, context: str, aggressive: bool) -> bool:
        """
        Basic grounding check:
            ✓ answer words must appear in context
            ✓ avoid hallucinations
            ✓ optional aggressive mode
        """

        if not answer:
            return False

        ans_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())

        overlap = len(ans_words & ctx_words)

        # Loose threshold
        if not aggressive:
            return overlap >= 2

        # Aggressive threshold
        return overlap >= 4
