from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import re

from dataset_gen.dataset_qanda_generator.configuration import cfg


class PromptBuilder:
    """
    Canonical prompt construction layer.

    Responsibilities:
      • Load prompt templates from disk
      • Enforce required fields
      • Render final prompt strings
      • Optional length enforcement
    """

    def __init__(
        self,
        template_path: Path,
        *,
        max_chars: Optional[int] = None,
        truncate_from: str = "tail",
    ):
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        self.template_path = template_path
        self.template_text = template_path.read_text(encoding="utf-8")

        self.max_chars = max_chars
        self.truncate_from = truncate_from

    # --------------------------------------------------
    # Core render
    # --------------------------------------------------
    def render(self, **kwargs: Any) -> str:
        missing = self._missing_placeholders(kwargs)
        if missing:
            raise ValueError(
                f"Missing required prompt fields: {sorted(missing)}"
            )

        rendered = self.template_text.format(**kwargs)

        if self.max_chars and len(rendered) > self.max_chars:
            rendered = self._truncate(rendered)

        return rendered

    # --------------------------------------------------
    # Truncation
    # --------------------------------------------------
    def _truncate(self, text: str) -> str:
        if self.truncate_from == "head":
            return text[-self.max_chars :]
        if self.truncate_from == "tail":
            return text[: self.max_chars]
        raise ValueError(
            f"Invalid truncate_from='{self.truncate_from}', expected 'head' or 'tail'"
        )

    # --------------------------------------------------
    # Validation helpers
    # --------------------------------------------------
    def _missing_placeholders(self, values: Dict[str, Any]) -> set[str]:
        placeholders = set(re.findall(r"{([a-zA-Z0-9_]+)}", self.template_text))
        return placeholders - set(values.keys())

    # --------------------------------------------------
    # Factories (INTENT-SPECIFIC)
    # --------------------------------------------------
    @classmethod
    def for_document_metadata(cls) -> "PromptBuilder":
        """
        Used for:
          • Document metadata extraction
          • Structured outputs
          • Non-answer tasks
        """
        return cls(cfg.DOCUMENT_METADATA_PROMPT)

    @classmethod
    def for_flan_question_generation(cls) -> "PromptBuilder":
        """
        Used ONLY for FLAN question generation.
        """
        return cls(
            cfg.QA_GENERATION_FLAN_PROMPT,
            max_chars=2400,
            truncate_from="tail",
        )

    @classmethod
    def for_answer_generation(cls) -> "PromptBuilder":
        """
        Used ONLY for answer generation (Stage 4).
        Enforces plain-text, grounded answers.
        """
        return cls(
            cfg.ANSWER_GENERATION_PROMPT,
            max_chars=2400,
            truncate_from="tail",
        )
