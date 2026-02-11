# dataset_qanda_generator/models/gemma.py
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import os
from .base_causal import CausalLMAnswerBase
from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger

log = get_qna_logger("gemma_model")

DEFAULT_GEMMA_DIR = Path(os.environ["MODELS_GEMMA_DIR"])




class GemmaAnswerGenerator(CausalLMAnswerBase):
    """
    Gemma 2B/9B Instruct – plain prompt version.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_GEMMA_DIR,
        device: str | None = None,
    ):
        super().__init__(
            model_dir=model_dir,
            name="GEMMA",
            device=device,
            trust_remote_code=True,
        )

    def build_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful assistant. Use ONLY the information in the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

    # Optional: override generate_answer for deterministic behavior
    # (this keeps your previous behavior exactly)
    @torch.no_grad()
    def generate_answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 60,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        from transformers import AutoTokenizer  # type: ignore
        import torch

        prompt = self.build_prompt(context, question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
        )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full[len(prompt) :].strip()
        answer = answer.split("\n")[0].strip()
        return answer
