from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import (
    get_qna_session,
)
from dataset_gen.dataset_qanda_generator.configuration.logging_config import (
    get_qna_logger,
)
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document,
    Criticality,
)

# ---------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------
log = get_qna_logger("dataset_builder")

# =====================================================================
# DatasetBuilder (READ-ONLY)
# =====================================================================
class DatasetBuilder:
    """
    Dataset Builder Pipeline (READ-ONLY)

    Responsibilities:
      • Select documents by MANUAL metadata
      • Validate eligibility (strict by default)
      • Resolve authoritative Q&A (best answers only)
      • Assemble canonical training records
      • Export to multiple dataset formats

    This class NEVER mutates the database.
    """

    # -----------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------
    def __init__(
        self,
        *,
        session: Optional[Session] = None,
        output_dir: Optional[Path] = None,
        strict: bool = True,
    ):
        self.session = session or get_qna_session()
        self.strict = strict

        self.output_dir = Path(output_dir or "dataset_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "[INIT] DatasetBuilder strict=%s output_dir=%s",
            self.strict,
            self.output_dir,
        )

    # =================================================================
    # Internal helpers
    # =================================================================
    def _resolve_output_dir(self, output_dir: Optional[Path]) -> Path:
        """
        Resolve final output directory.

        Allows dataset-scoped subfolders while remaining backward compatible.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out

    # =================================================================
    # Stage 1 — Document Selection
    # =================================================================
    def select_documents(
        self,
        *,
        include_tags: Optional[Iterable[str]] = None,
        exclude_tags: Optional[Iterable[str]] = None,
        domain: Optional[str] = None,
        audience: Optional[str] = None,
        max_criticality: Optional[str] = None,
        document_ids: Optional[Iterable[int]] = None,
    ) -> List[Document]:

        q = self.session.query(Document)

        if document_ids:
            q = q.filter(Document.id.in_(list(document_ids)))

        if domain:
            q = q.join(Document.domain).filter_by(code=domain)

        if audience:
            q = q.join(Document.audience).filter_by(code=audience)

        if max_criticality:
            max_obj = (
                self.session.query(Criticality)
                .filter_by(code=max_criticality)
                .first()
            )
            if not max_obj:
                raise ValueError(f"Unknown criticality code: {max_criticality}")

            q = (
                q.join(Document.criticality)
                .filter(Criticality.severity <= max_obj.severity)
            )

        docs = q.all()

        include_tags = set(include_tags or [])
        exclude_tags = set(exclude_tags or [])

        filtered: List[Document] = []

        for doc in docs:
            doc_tags = {t.code for t in doc.tags}

            if include_tags and not include_tags.issubset(doc_tags):
                continue

            if exclude_tags and doc_tags.intersection(exclude_tags):
                continue

            filtered.append(doc)

        log.info(
            "[SELECT] %d documents selected (include_tags=%s exclude_tags=%s)",
            len(filtered),
            include_tags,
            exclude_tags,
        )

        return filtered

    # =================================================================
    # Stage 2 — Eligibility Validation
    # =================================================================
    def validate_document(self, doc: Document) -> None:
        errors: List[str] = []

        if not doc.tags:
            errors.append("missing tags")
        if not doc.domain:
            errors.append("missing domain")
        if not doc.audience:
            errors.append("missing audience")
        if not doc.criticality:
            errors.append("missing criticality")
        if not doc.chunks:
            errors.append("no chunks")

        has_ranked_answer = any(
            q.ranking is not None
            for c in doc.chunks
            for q in c.questions
        )

        if not has_ranked_answer:
            errors.append("no ranked answers")

        if errors:
            msg = f"Document id={doc.id} invalid: {', '.join(errors)}"
            if self.strict:
                log.error("[INVALID] %s", msg)
                raise RuntimeError(msg)
            else:
                log.warning("[SKIP] %s", msg)

    # =================================================================
    # Stage 3 — Resolve Q&A
    # =================================================================
    def iter_resolved_qna(self, doc: Document):
        for chunk in doc.chunks:
            for question in chunk.questions:
                if not question.ranking:
                    continue
                yield chunk, question, question.ranking

    # =================================================================
    # Stage 4 — Record Assembly
    # =================================================================
    def assemble_records(self, docs: List[Document]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for doc in docs:
            try:
                self.validate_document(doc)
            except RuntimeError as e:
                if self.strict:
                    raise
                log.warning("[SKIP] %s", str(e))
                continue

            domain_code = doc.domain.code if doc.domain else None
            audience_code = doc.audience.code if doc.audience else None
            criticality_code = doc.criticality.code if doc.criticality else None
            tags_list = sorted(t.code for t in doc.tags) if doc.tags else []

            for chunk, question, ranking in self.iter_resolved_qna(doc):
                records.append(
                    {
                        "instruction": question.question,
                        "input": chunk.context,
                        "output": ranking.best_answer,
                        "metadata": {
                            "document": {
                                "id": doc.id,
                                "title": doc.title,
                                "domain": domain_code,
                                "audience": audience_code,
                                "criticality": criticality_code,
                                "tags": tags_list,
                            },
                            "chunk": {
                                "id": chunk.id,
                                "page": chunk.page,
                                "section": chunk.section,
                                "subsection": chunk.subsection,
                            },
                            "question": {"id": question.id},
                            "answer": {
                                "best_model": ranking.best_model,
                                "scores": ranking.answer_scores,
                            },
                        },
                    }
                )

        log.info("[ASSEMBLE] %d dataset records created", len(records))
        return records

    # =================================================================
    # Public Build API
    # =================================================================
    def build(
        self,
        *,
        include_tags: Optional[Iterable[str]] = None,
        exclude_tags: Optional[Iterable[str]] = None,
        domain: Optional[str] = None,
        audience: Optional[str] = None,
        max_criticality: Optional[str] = None,
        document_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:

        docs = self.select_documents(
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            domain=domain,
            audience=audience,
            max_criticality=max_criticality,
            document_ids=document_ids,
        )

        return self.assemble_records(docs)

    # =================================================================
    # Exporters (dataset-folder aware)
    # =================================================================
    def export_jsonl(
            self,
            records: List[Dict[str, Any]],
            filename: str,
            *,
            output_dir: Optional[Path] = None,
    ) -> Path:
        out = self._resolve_output_dir(output_dir)
        path = out / filename

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                out_row = {
                    "prompt": r.get("instruction"),
                    "context": r.get("input"),
                    "response": r.get("output"),
                    "metadata": r.get("metadata"),
                }

                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

        log.info("[EXPORT] JSONL -> %s", path)
        return path

    def export_alpaca(
        self,
        records: List[Dict[str, Any]],
        filename: str,
        *,
        output_dir: Optional[Path] = None,
    ) -> Path:
        out = self._resolve_output_dir(output_dir)
        path = out / filename

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(
                    json.dumps(
                        {
                            "instruction": r["instruction"],
                            "input": r["input"],
                            "output": r["output"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        log.info("[EXPORT] Alpaca -> %s", path)
        return path

    def export_chatml(
        self,
        records: List[Dict[str, Any]],
        filename: str,
        *,
        output_dir: Optional[Path] = None,
    ) -> Path:
        out = self._resolve_output_dir(output_dir)
        path = out / filename

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Answer using only the provided context.",
                                },
                                {"role": "user", "content": r["instruction"]},
                                {"role": "assistant", "content": r["output"]},
                            ],
                            "context": r["input"],
                            "metadata": r["metadata"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        log.info("[EXPORT] ChatML -> %s", path)
        return path

    def export_prompt_response(
        self,
        records: List[Dict[str, Any]],
        filename: str,
        *,
        output_dir: Optional[Path] = None,
    ) -> Path:
        out = self._resolve_output_dir(output_dir)
        path = out / filename

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                prompt = f"{r['instruction']}\n\n{r['input']}".strip()
                response = r["output"].strip()

                if not prompt or not response:
                    continue

                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "response": response,
                            "metadata": r.get("metadata"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        log.info("[EXPORT] Prompt/Response -> %s", path)
        return path


