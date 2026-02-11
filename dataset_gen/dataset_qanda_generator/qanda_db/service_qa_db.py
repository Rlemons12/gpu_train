#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q&A Dataset Service Layer (FULL RUN-TRACKED VERSION)
----------------------------------------------------

This version:
    ✔ Every create operation attaches to EXACTLY ONE PipelineRun
    ✔ All methods accept run_id
    ✔ Creates PipelineRunItem automatically
    ✔ Matches new polymorphic schema
    ✔ Clean logging and session control
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import time
from pathlib import Path

from .qa_db import (
    Document,
    Chunk,
    Question,
    Answer,
    AnswerRanking,
    LLMModel,
    Embedding,
    PipelineRunItem,
)

from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger


# ================================================================
# LOGGER
# ================================================================
log = get_qna_logger("qna_service")


# ================================================================
# Helper: timing decorator
# ================================================================
def timed_log(operation_name: str):
    def wrapper(func):
        def inner(*args, **kwargs):
            start = time.time()
            log.debug(f"[START] {operation_name}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                log.info(f"[OK] {operation_name} completed in {elapsed:.3f}s")
                return result
            except Exception as exc:
                elapsed = time.time() - start
                log.error(f"[FAIL] {operation_name} failed after {elapsed:.3f}s")
                log.exception(exc)
                raise
        return inner
    return wrapper


# ================================================================
# SERVICE LAYER
# ================================================================
class QADatabaseService:

    def __init__(self, session: Optional[Session] = None, request_id: Optional[str] = None):
        self._session = session
        self.request_id = request_id

        if request_id:
            log.info(f"[CTX] Request-ID={request_id}")

        if session:
            log.info("Using caller-supplied DB session.")
        else:
            log.info("Internal auto-session mode enabled.")

    # -------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------
    def _auto_session(self) -> Session:
        if self._session is None:
            log.debug("[DB] Creating internal DB session...")
            return get_qna_session()
        return self._session

    def _finalize(self, session: Session, obj):
        if self._session is None:
            try:
                session.commit()
                try:
                    session.refresh(obj)
                except Exception:
                    pass
            except SQLAlchemyError as exc:
                session.rollback()
                log.error("[DB] Commit failed — rolled back")
                log.exception(exc)
                raise
            finally:
                session.close()
        return obj

    def _add_run_item(self, session, run_id: Optional[int], parent_type: str, parent_id: int):
        """
        Creates a pipeline run item ONLY if run_id is valid.
        Prevents NOT NULL violations during early pipeline stages.
        """
        if run_id is None:
            log.debug(f"[RUN ITEM] Skipped creating run item for {parent_type}:{parent_id} (run_id=None)")
            return None

        run_item = PipelineRunItem(
            run_id=run_id,
            parent_type=parent_type,
            parent_id=parent_id,
        )
        session.add(run_item)
        return run_item

    # ==================================================================
    # DOCUMENT
    # ==================================================================
    @timed_log("Add Document")
    def add_document(self, run_id: int, file_name: str, file_path: str) -> Document:
        """
        STRICT MODE.

        - file_path MUST be absolute
        - file_path MUST exist
        - file_path is stored canonically
        """

        log.info(
            "[DOCUMENT] add_document called file_name='%s', file_path='%s'",
            file_name,
            file_path,
        )

        # ------------------------------------
        # STRICT PATH VALIDATION
        # ------------------------------------
        path = Path(file_path)

        if not path.is_absolute():
            raise ValueError(
                f"DocumentService requires absolute file paths. "
                f"Received: '{file_path}'"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Document file does not exist: '{file_path}'"
            )

        # Canonicalize (resolves symlinks, .., etc.)
        path = path.resolve(strict=True)
        canonical_path = str(path)

        log.info(
            "[DOCUMENT] canonical file_path='%s'",
            canonical_path,
        )

        session = self._auto_session()

        try:
            # ------------------------------------
            # DEDUPE BY CANONICAL PATH ONLY
            # ------------------------------------
            existing = (
                session.query(Document)
                .filter(Document.file_path == canonical_path)
                .first()
            )

            if existing:
                log.info("[DOCUMENT] Already exists: id=%s", existing.id)

                if run_id is not None:
                    self._add_run_item(session, run_id, "document", existing.id)

                return existing

            # ------------------------------------
            # CREATE DOCUMENT
            # ------------------------------------
            doc = Document(
                file_name=file_name,
                file_path=canonical_path,
            )

            session.add(doc)
            session.flush()

            if run_id is not None:
                self._add_run_item(session, run_id, "document", doc.id)

            return self._finalize(session, doc)

        except Exception:
            session.rollback()
            raise

    # ==================================================================
    # CHUNK
    # ==================================================================
    @timed_log("Add Chunk")
    def add_chunk(
        self,
        run_id: int,
        document_id: int,
        chunk_id: str,
        context: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
        subsection: Optional[str] = None,
    ) -> Chunk:

        log.info(f"[CHUNK] doc={document_id}, chunk='{chunk_id}'")

        session = self._auto_session()

        chunk = Chunk(
            document_id=document_id,
            chunk_id=chunk_id,
            context=context,
            page=page,
            section=section,
            subsection=subsection,
        )

        session.add(chunk)
        session.flush()

        # NEW
        self._add_run_item(session, run_id, "chunk", chunk.id)

        return self._finalize(session, chunk)

    # ==================================================================
    # QUESTION
    # ==================================================================
    @timed_log("Add Question")
    def add_question(
        self,
        run_id: int,
        chunk_id: int,
        question_text: str,
        question_index: int = 1,
    ) -> Question:

        log.info(f"[QUESTION] chunk={chunk_id}, idx={question_index}")

        session = self._auto_session()

        q = Question(
            chunk_id=chunk_id,
            question=question_text,
            question_index=question_index,
        )

        session.add(q)
        session.flush()

        # NEW
        self._add_run_item(session, run_id, "question", q.id)

        return self._finalize(session, q)

    # ==================================================================
    # ANSWER
    # ==================================================================
    @timed_log("Add Answer")
    def add_answer(
            self,
            run_id: int,
            question_id: int,
            model_name: str,
            answer_text: str,
            model_type: str = "causal_lm",
            model_path: str = None,
            score: float = None,
            is_best: bool = False,
            is_worst: bool = False,
    ) -> Answer:

        log.info(f"[ANSWER] q_id={question_id}, model='{model_name}'")

        session = self._auto_session()

        # Ensure the model is registered
        model = session.query(LLMModel).filter_by(name=model_name).first()

        if not model:
            log.error(
                f"[MODEL] LLM '{model_name}' does NOT exist in qna_models. "
                f"Every model MUST be registered with class_path before running pipeline."
            )
            raise RuntimeError(
                f"LLM model '{model_name}' is not registered in qna_models."
            )

        a = Answer(
            question_id=question_id,
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            answer_text=answer_text,
            score=score,
            is_best=is_best,
            is_worst=is_worst,
        )

        session.add(a)
        session.flush()

        # Track pipeline run item
        self._add_run_item(session, run_id, "answer", a.id)

        return self._finalize(session, a)

    # ==================================================================
    # ANSWER RANKING
    # ==================================================================
    @timed_log("Add Answer Ranking")
    def add_answer_ranking(
        self,
        run_id: int,
        question_id: int,
        best_model: str,
        best_answer: str,
        worst_model: str,
        worst_answer: str,
        answer_scores: Dict[str, float],
    ) -> AnswerRanking:

        log.info(f"[RANK] q_id={question_id}, best={best_model}")

        session = self._auto_session()

        r = AnswerRanking(
            question_id=question_id,
            best_model=best_model,
            best_answer=best_answer,
            worst_model=worst_model,
            worst_answer=worst_answer,
            answer_scores=answer_scores,
        )

        session.add(r)
        session.flush()

        # NEW
        self._add_run_item(session, run_id, "ranking", r.id)

        return self._finalize(session, r)

    # ==================================================================
    # EMBEDDING
    # ==================================================================
    @timed_log("Add Embedding")
    def add_embedding(
        self,
        run_id: int,
        parent_type: str,
        parent_id: int,
        model_name: str,
        embedding_vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Embedding:

        log.info(f"[EMBED] {parent_type}:{parent_id}, model={model_name}")

        session = self._auto_session()

        emb = Embedding(
            parent_type=parent_type,
            parent_id=parent_id,
            model_name=model_name,
            embedding_vector=embedding_vector,
            actual_dimensions=len(embedding_vector),
            meta_json=metadata or {},
        )

        session.add(emb)
        session.flush()

        # NEW
        self._add_run_item(session, run_id, "embedding", emb.id)

        return self._finalize(session, emb)


# ======================================================================
# GLOBAL FACTORY
# ======================================================================
def get_qa_service(session: Optional[Session] = None, request_id: Optional[str] = None) -> QADatabaseService:
    log.info("Creating QADatabaseService via factory.")
    return QADatabaseService(session=session, request_id=request_id)
