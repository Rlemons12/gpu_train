# gpu_adapter.py
# ==========================================================
# GPUAdapter — LOCAL GPU BYPASS USING MISTRAL
# ==========================================================
# Temporary adapter:
#   • Keeps pipeline interface stable
#   • Routes directly to local CUDA GPU
#   • Uses existing MistralAnswerGenerator
#   • No GPU service yet
#
# Later: swap internals with HTTP GPU service client
# ==========================================================

import logging
import os
from typing import Dict, Optional

import torch

from dataset_gen.dataset_qanda_generator.models.mistral import (
    MistralAnswerGenerator,
)

log = logging.getLogger("GPUAdapter")


class GPUAdapter:
    """
    Local GPU-bypass adapter.

    Pipeline expects:
        GPUAdapter().generate(
            model: str,
            prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float
        ) -> str
    """

    def __init__(self, *, mistral_model_dir: Optional[str] = None):
        self._models: Dict[str, MistralAnswerGenerator] = {}

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPUAdapter requires CUDA, but torch.cuda.is_available() is False"
            )

        # Optional explicit override (useful for debugging)
        self._mistral_model_dir = mistral_model_dir

        log.info(
            "[GPUAdapter] Initialized LOCAL GPU BYPASS | device=%s",
            torch.cuda.get_device_name(0),
        )

        if mistral_model_dir:
            log.info(
                "[GPUAdapter] Using explicit Mistral model dir override: %s",
                mistral_model_dir,
            )
        else:
            log.info(
                "[GPUAdapter] Using MODELS_MISTRAL_7B_DIR=%s",
                os.environ.get("MODELS_MISTRAL_7B_DIR"),
            )

    # --------------------------------------------------
    # Internal: lazy-load model onto GPU
    # --------------------------------------------------
    def _get_model(self, model_name: str) -> MistralAnswerGenerator:
        key = model_name.lower()

        if key not in self._models:
            log.info(
                "[GPUAdapter] Loading model '%s' via MistralAnswerGenerator",
                model_name,
            )

            if key in ("mistral", "mistral-7b", "mistral-7b-instruct"):
                try:
                    self._models[key] = MistralAnswerGenerator(
                        model_dir=self._mistral_model_dir,
                        device="cuda",
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to initialize MistralAnswerGenerator.\n"
                        "Check MODELS_MISTRAL_7B_DIR or provided override.\n"
                        f"Original error: {e}"
                    ) from e
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        return self._models[key]

    def _normalize_output(self, output) -> str:
        if output is None:
            return ""

        if not isinstance(output, str):
            try:
                output = str(output)
            except Exception:
                return ""

        output = output.strip()

        # Hard guardrails
        if not output:
            return ""

        if output.endswith("?"):
            log.warning("[GPUAdapter] Rejected question-like output: %r", output)
            return ""

        if output.startswith("{") or output.startswith("["):
            log.warning("[GPUAdapter] Rejected structured output: %r", output)
            return ""

        return output

    # --------------------------------------------------
    # Pipeline entry point
    # --------------------------------------------------
    def generate(
            self,
            *,
            model: str,
            prompt: str,
            max_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.95,
    ) -> str:
        llm = self._get_model(model)

        raw = llm.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # --------------------------------------------------
        # Normalize + enforce answer contract
        # --------------------------------------------------
        if raw is None:
            return ""

        if not isinstance(raw, str):
            try:
                raw = str(raw)
            except Exception:
                log.warning("[GPUAdapter] Non-string output could not be coerced")
                return ""

        answer = raw.strip()

        # Reject empty output
        if not answer:
            return ""

        # Reject question-like outputs
        if answer.endswith("?"):
            log.warning(
                "[GPUAdapter] Rejected question-like output from model '%s': %r",
                model,
                answer,
            )
            return ""

        # Reject structured / JSON-like outputs
        if answer.startswith("{") or answer.startswith("["):
            log.warning(
                "[GPUAdapter] Rejected structured output from model '%s': %r",
                model,
                answer,
            )
            return ""

        return answer

