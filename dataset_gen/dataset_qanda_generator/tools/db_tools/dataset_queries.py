#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset_queries.py

Centralized DB read queries for Option-C reporting tools.

Responsibilities:
    - Open training DB session
    - List documents
    - List pipeline runs
    - Fetch Q&A rows (chunks → questions → answers → rankings)
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
from dataset_gen.dataset_qanda_generator.configuration import cfg

# ------------------------------------------------------------
# DATABASE + ORM
# ------------------------------------------------------------
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_training_session
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import (
    Document,
    Chunk,
    Question,
    Answer,
    LLMModel,
    PipelineRun,
    AnswerRanking,
)

# ======================================================================
# SESSION HELPERS
# ======================================================================

def open_training_session():
    """Return a SQLAlchemy session connected to the training DB."""
    return get_training_session()


# ======================================================================
# DOCUMENT QUERIES
# ======================================================================

def list_documents(
    session,
    name_fragment: Optional[str] = None,
) -> List[Document]:
    """
    List documents in DB, optional filtering by name fragment.
    """
    q = session.query(Document)

    if name_fragment:
        q = q.filter(Document.file_name.ilike(f"%{name_fragment}%"))

    return q.order_by(Document.file_name).all()


def get_document_by_name_fragment(
    session,
    name_fragment: str,
) -> Optional[Document]:
    """Return the first document whose name contains the fragment."""
    docs = list_documents(session, name_fragment)
    return docs[0] if docs else None


# ======================================================================
# PIPELINE RUN QUERIES (UPDATED: timestamps + duration)
# ======================================================================

def _compute_duration(run: PipelineRun) -> Optional[float]:
    """Compute duration_seconds for a pipeline run."""
    if not run.started_at or not run.finished_at:
        return None
    return (run.finished_at - run.started_at).total_seconds()


def list_pipeline_runs_for_document(
    session,
    document_id: Optional[int] = None,
    document_name_fragment: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return pipeline runs, optionally filtered by:
        • document_id
        • document_name_fragment

    Returns list[dict]:
        {
            "run_id": int,
            "document_id": int | None,
            "document_name": str | None,
            "run_type": str,
            "success": bool | None,
            "started_at": datetime,
            "finished_at": datetime,
            "duration_seconds": float | None,
            "options": dict,
            "models": dict,
        }
    """
    q = (
        session.query(PipelineRun, Document.file_name)
        .outerjoin(Document, PipelineRun.document_id == Document.id)
    )

    if document_id is not None:
        q = q.filter(PipelineRun.document_id == document_id)

    if document_name_fragment:
        like = f"%{document_name_fragment}%"
        q = q.filter(Document.file_name.ilike(like))

    rows = q.order_by(PipelineRun.started_at.desc()).all()

    output = []
    for run, file_name in rows:
        output.append({
            "run_id": run.id,
            "document_id": run.document_id,
            "document_name": file_name,
            "run_type": run.run_type,
            "success": run.success,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration_seconds": _compute_duration(run),
            "options": run.options_json,
            "models": run.models_json,
        })

    return output


# ======================================================================
# Q&A EXTRACTION (UPDATED: model_name, answer_text, ranks)
# ======================================================================

def load_qna_rows_for_document(
    session,
    document: Document,
    include_rankings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load full Q&A rows for a single document.

    Returns list[dict] with a normalized format suitable for:
        - Excel/CSV export
        - Q&A reports
        - CLI tools
        - Dataset builders

    Row fields:
        document → chunk → question → answer → model → is_best_model
    """

    # ------------------------------------------------------------
    # 1. Load chunks
    # ------------------------------------------------------------
    chunks: List[Chunk] = (
        session.query(Chunk)
        .filter(Chunk.document_id == document.id)
        .order_by(Chunk.id)
        .all()
    )

    # ------------------------------------------------------------
    # 2. Ranking lookup: question_id → best_model_name
    # ------------------------------------------------------------
    best_model_by_question: Dict[int, str] = {}

    if include_rankings:
        rankings = (
            session.query(AnswerRanking)
            .join(Question, AnswerRanking.question_id == Question.id)
            .join(Chunk, Question.chunk_id == Chunk.id)
            .filter(Chunk.document_id == document.id)
            .all()
        )
        for r in rankings:
            best_model_by_question[r.question_id] = r.best_model

    # ------------------------------------------------------------
    # 3. Build flat row list
    # ------------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    for ch in chunks:

        # Fetch questions
        questions: List[Question] = (
            session.query(Question)
            .filter(Question.chunk_id == ch.id)
            .order_by(Question.question_index)
            .all()
        )

        for q in questions:

            # Fetch answers for each question
            answers = (
                session.query(Answer)
                .filter(Answer.question_id == q.id)
                .order_by(Answer.model_name)
                .all()
            )

            # No answers → placeholder row
            if not answers:
                rows.append({
                    "document_id": document.id,
                    "document": document.file_name,
                    "file_path": document.file_path,

                    "chunk_db_id": ch.id,
                    "chunk_id": ch.chunk_id,
                    "page": ch.page,
                    "section": ch.section,
                    "subsection": ch.subsection,
                    "context": ch.context,

                    "question_id": q.id,
                    "question_index": q.question_index,
                    "question": q.question,

                    "model": None,
                    "answer": None,
                    "is_best_model": False,
                })
                continue

            # Ranking: best model NAME for this question
            best_model_name = best_model_by_question.get(q.id)

            for ans in answers:
                rows.append({
                    "document_id": document.id,
                    "document": document.file_name,
                    "file_path": document.file_path,

                    "chunk_db_id": ch.id,
                    "chunk_id": ch.chunk_id,
                    "page": ch.page,
                    "section": ch.section,
                    "subsection": ch.subsection,
                    "context": ch.context,

                    "question_id": q.id,
                    "question_index": q.question_index,
                    "question": q.question,

                    "model": ans.model_name,
                    "answer": ans.answer_text,   # correct ORM field
                    "is_best_model": (
                        best_model_name is not None
                        and ans.model_name == best_model_name
                    ),
                })

    return rows
