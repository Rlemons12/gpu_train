# dataset_qanda_generator/models/qwen.py
# -*- coding: utf-8 -*-

import logging
import os
import re
from .base_causal import CausalLMAnswerBase
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_QWEN_DIR = Path(os.environ["MODELS_QWEN_DIR"])


class QwenAnswerGenerator(CausalLMAnswerBase):
    """
    Robust, strictly factual Qwen2.5-3B answer generator for Option-C.

    Improvements:
        • Better grounding → almost no false "Insufficient information" cases
        • One-sentence factual answers
        • Automatic hallucination filtering
        • Sentence cleanup + normalization
        • Optional aggressive grounding mode
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_QWEN_DIR,
        device: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__(
            model_dir=model_dir,
            name="QWEN",
            device=device,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        log.info("[QWEN] Strict factual answer generator loaded.")

    # ==================================================================
    # Public API
    # ==================================================================

    def generate_answer(
        self,
        context: str,
        question: str,
        temperature: float = 0.25,
        max_new_tokens: int = 80,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        strict_factual: bool = True,
        one_sentence: bool = True,
        aggressive_grounding: bool = True,
        debug_raw: bool = False,
    ) -> str:
        """
        Generate a strictly context-grounded, one-sentence factual answer.
        """

        prompt = self._build_prompt(context, question, aggressive_grounding)

        raw = super().generate_answer(
            context=prompt,
            question="",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        cleaned = self._clean_output(raw, strict_factual, one_sentence)

        if debug_raw:
            print("\n========= QWEN DEBUG =========")
            print("RAW:\n", raw)
            print("\nCLEAN:\n", cleaned)
            print("==============================\n")

        # Final grounding check:
        if not self._is_grounded(cleaned, context):
            return "Insufficient information in context."

        return cleaned

    # ==================================================================
    # PROMPT CONSTRUCTION
    # ==================================================================

    def _build_prompt(self, context: str, question: str, aggressive: bool) -> str:
        """
        A refined, highly stable instruction prompt that:
        • Forces single-sentence output
        • Prevents multi-turn or meta stuff
        • Reduces false "insufficient info"
        """

        grounding_clause = (
            "\n- Use the exact wording from the context where possible."
            "\n- If unsure, still provide the most direct factual sentence supported by the context."
            if aggressive
            else "\n- If the answer cannot be found, respond exactly: \"Insufficient information in context.\""
        )

        return f"""
You are a technical manufacturing documentation assistant.

INSTRUCTIONS:
- Answer ONLY using the information from the context.
- Respond in ONE short factual sentence.
- No lists. No explanations. No multi-turn dialogue labels.
- No filler such as "Based on the context" or "According to the document." 
- No guessing or speculation.
{grounding_clause}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""".strip()

    # ==================================================================
    # OUTPUT CLEANING
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

        # Remove junk prefixes
        bad_prefixes = [
            r"^Human:\s*", r"^Assistant:\s*", r"^User:\s*",
            r"^System:\s*", r"^Q:\s*", r"^A:\s*",
            r"^Context:\s*", r"^Answer:\s*",
            r"^Response:\s*", r"^Explanation:\s*",
        ]
        for pat in bad_prefixes:
            ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()

        # Remove filler meta-statements
        fillers = [
            "Based on the context",
            "According to the context",
            "From the context",
            "The answer is",
            "As stated",
            "As described",
            "In summary",
            "Overall",
        ]
        for f in fillers:
            if ans.lower().startswith(f.lower()):
                ans = ans[len(f):].strip(" .:-").strip()
                break

        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)

        # One-sentence enforcement
        if one_sentence:
            m = re.match(r"(.+?[.!?])(\s|$)", ans)
            if m:
                ans = m.group(1).strip()

        # Add punctuation
        if ans and ans[-1] not in ".!?":
            ans += "."

        return ans.strip()

    # ==================================================================
    # FINAL GROUNDING CHECK
    # ==================================================================

    def _is_grounded(
        self,
        answer: str,
        context: str,
        aggressive_grounding: bool = False,  # <- extra arg to match base class
    ) -> bool:
        """
        Simple grounding check:
            • Does answer contain at least 1–2 words present in the context?
            • If answer == "Insufficient information..." → pass it through.
        """

        # If the model already said "Insufficient information...", let caller
        # handle it (generate_answer will return that message).
        if answer.lower().startswith("insufficient information"):
            return False

        # Word overlap check
        ans_words = set(w.lower() for w in answer.split() if len(w) > 3)
        ctx_words = set(w.lower() for w in context.split() if len(w) > 3)

        overlap = len(ans_words & ctx_words)

        # Require at least 1 shared non-trivial word
        return overlap >= 1

