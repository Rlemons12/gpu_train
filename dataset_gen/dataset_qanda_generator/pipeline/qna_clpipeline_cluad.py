#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C v3.0 – Q&A Dataset Pipeline (Service-Layer + Run Tracking)
-------------------------------------------------------------------

Primary usage (production path):

    from qanda_pipeline import QandaPipeline

    pipeline = QandaPipeline()
    pipeline.run_full_pipeline(
        doc_path="path/to/document.docx",
        max_chunks=3,
        embed=True,
    )

Stages:
    1. structure   -> extract structure.json (filesystem only)
    2. clean       -> structure.json -> cleaned chunks  + DB insert
    3. questions   -> generate questions only           + DB insert (multi-pass)
    4. answers     -> generate answers only             + DB insert + ranking
    5. full        -> run 1–4 end-to-end:
                        - creates PipelineRun
                        - all creates go through QADatabaseService
                        - all created objects are attached to that run
    6. export      -> export best/worst Q&A to fine-tuning formats
    7. rank        -> recompute rankings from DB (optional, maintenance)

Design notes:
    - Full pipeline (run_full_pipeline) is the "real" production path.
      It:
        * Creates a PipelineRun
        * Uses QADatabaseService for Document, Chunk, Question, Answer,
          AnswerRanking, Embedding
        * Updates PipelineRun.document_id
        * Marks run finished / failed
    - Single-stage modes are mostly for debugging / development and use
      simpler DB logic without run tracking.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from option_c_qna.qanda_db.qa_db import LLMModel
from option_c_qna.models.flan_qa import FLAN_QA_Model
from option_c_qna.configuration import cfg
from option_c_qna.configuration.logging_config import get_qna_logger
from option_c_qna.configuration.pg_db_config import get_qna_session
from option_c_qna.qanda_db.service_qa_db import QADatabaseService
from option_c_qna.qanda_db import get_qa_service
from option_c_qna.qanda_db.qa_db import (
    Document,
    Chunk,
    Question,
    Answer,
    AnswerRanking,
    Embedding,
    PipelineRun,
)
from option_c_qna.document_structure_extractor.structure_extractor import (
    DocumentStructureExtractor,
)
from option_c_qna.document_structure_extractor.structure_chunk_loader import (
    StructureChunkLoader,
)

# -------------------------------------------------------------------
# OPTIONAL – Semantic Similarity / Embedding (MiniLM)
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util

    _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SIM_MODEL = True
except Exception as e:
    _similarity_model = None
    HAS_SIM_MODEL = False


class QandaPipeline:
    """
    Q&A Dataset Pipeline with Service-Layer + Run Tracking.

    This class encapsulates the full pipeline for generating Q&A datasets
    from documents, including structure extraction, chunk cleaning,
    question generation, answer generation, and export functionality.
    """

    # -------------------------------------------------------------------
    # HYPERPARAMETERS / DEFAULTS
    # -------------------------------------------------------------------
    DEFAULT_NUM_QUESTIONS = 3
    DEFAULT_MAX_Q_RETRIES = 3
    DEFAULT_SIMILARITY_THRESHOLD = 0.70
    DEFAULT_MIN_QUESTION_LEN = 12

    # Hybrid sampling: per-model answer samples
    NUM_DETERMINISTIC_SAMPLES = 3
    NUM_STOCHASTIC_SAMPLES = 5
    TOTAL_SAMPLES_PER_MODEL = NUM_DETERMINISTIC_SAMPLES + NUM_STOCHASTIC_SAMPLES

    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

    # Question templates
    QUESTION_TEMPLATES = [
        "What is {focus}?",
        "What does {focus} refer to?",
        "Where is {focus} located?",
        "Which {focus} is mentioned?",
        "When does {focus} occur?",
    ]

    def __init__(self):
        """Initialize the QandaPipeline."""
        self.log = get_qna_logger("qanda_pipeline")

        # Output directories
        self.STRUCT_DIR = cfg.STRUCTURE_DIR
        self.CLEAN_DIR = cfg.CLEAN_DIR
        self.QUESTION_DIR = cfg.QUESTIONS_DIR
        self.ANSWER_DIR = cfg.ANSWERS_DIR

        # Model caches
        self._flan_qa_model: Optional[FLAN_QA_Model] = None
        self._answer_model_cache: Optional[Dict[str, Any]] = None

        # Similarity model
        self._similarity_model = _similarity_model
        self._has_sim_model = HAS_SIM_MODEL

        if self._has_sim_model:
            self.log.info("[SIM] Loaded all-MiniLM-L6-v2 for question similarity + embeddings.")
        else:
            self.log.warning(
                "[SIM] Could not load SentenceTransformer (all-MiniLM-L6-v2). "
                "Similarity-based dedupe + embeddings disabled."
            )

    # ==================================================================
    # MODEL LOADING METHODS
    # ==================================================================

    def get_flan_model(self) -> FLAN_QA_Model:
        """
        Lazily load FLAN-T5-Large once per process and reuse.
        Used for question generation and (optionally) as an answer model.
        """
        if self._flan_qa_model is None:
            self.log.info("[MODEL] Loading FLAN-T5-Large (FLAN_QA_Model) once...")
            self._flan_qa_model = FLAN_QA_Model()
        return self._flan_qa_model

    def _init_answer_models(
            self,
            selected_models: Optional[List[str]] = None,
            model_cache: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load answer LLMs with caching support.

        Priority of sources:
            1. If model_cache is provided → use that (direct injection)
            2. Else if instance cache exists → reuse it
            3. Else → load from qna_models table and cache

        Args:
            selected_models: Optional list of model names supplied via CLI
            model_cache: Optional dict {name: model_instance} (preloaded externally)

        Returns:
            Dict[str, Any]: {model_name: model_instance}
        """
        # STEP 0 — Directly supplied preloaded cache
        if model_cache is not None:
            self.log.info("[ANSWERS] Using externally provided preloaded model cache.")
            all_models = {name.lower(): obj for name, obj in model_cache.items()}

        # STEP 0.5 — Use instance cache if already populated
        elif self._answer_model_cache is not None:
            self.log.info("[ANSWERS] Using instance preloaded model cache.")
            all_models = {name.lower(): obj for name, obj in self._answer_model_cache.items()}

        else:
            # STEP 1 — Load from DB and populate the instance cache
            try:
                raw_models = LLMModel.load_all_enabled()
            except Exception as exc:
                self.log.error("[ANSWERS] Failed to load enabled models from qna_models.")
                self.log.exception(exc)
                raise

            all_models = {name.lower(): obj for name, obj in raw_models.items()}

            if not all_models:
                self.log.error("[ANSWERS] No enabled LLM models found in qna_models!")
                raise RuntimeError("No enabled LLM models available.")

            self._answer_model_cache = all_models
            self.log.info("[ANSWERS] Cached %d answer models.", len(all_models))

        # STEP 2 — No CLI filter → return all models
        if not selected_models:
            self.log.info("[ANSWERS] Using ALL enabled models: %s", list(all_models.keys()))
            return all_models

        # STEP 3 — Normalize the requested model list
        selected_lower = [m.strip().lower() for m in selected_models if m.strip()]

        # STEP 4 — Apply filter
        filtered = {
            name: obj for name, obj in all_models.items()
            if name in selected_lower
        }

        # STEP 5 — Warn about invalid model names
        for req in selected_lower:
            if req not in all_models:
                self.log.warning(
                    f"[ANSWERS] Requested model '{req}' is NOT enabled or not found in qna_models."
                )

        # STEP 6 — Enforce valid result set
        if not filtered:
            self.log.error(
                "[ANSWERS] No valid answer models remain after filtering.\n"
                "Requested=%s\nEnabled=%s",
                selected_lower,
                list(all_models.keys())
            )
            raise RuntimeError("No valid answer models remain after filtering.")

        self.log.info("[ANSWERS] Using models: %s", list(filtered.keys()))
        return filtered

    def preload_all_answer_models(self) -> Dict[str, Any]:
        """
        Loads all enabled answer models from qna_models table ONE TIME.
        Returns a dict: {model_name: model_instance}

        - Safe offline loading
        - Logs failures but continues with remaining models
        - Populates the instance _answer_model_cache
        """
        # Already loaded → return cache
        if self._answer_model_cache is not None:
            self.log.info("[PRELOAD] Using existing instance model cache.")
            return self._answer_model_cache

        # Load enabled model records from DB
        try:
            enabled_models = LLMModel.load_all_enabled()
        except Exception as exc:
            self.log.error("[PRELOAD] Failed to load enabled models from qna_models.")
            self.log.exception(exc)
            raise

        if not enabled_models:
            raise RuntimeError("[PRELOAD] No enabled models found in qna_models table.")

        self.log.info("[PRELOAD] Found %d enabled models: %s",
                      len(enabled_models), list(enabled_models.keys()))

        loaded: Dict[str, Any] = {}

        # Loop over each enabled model and load instance
        for name, model_obj in enabled_models.items():
            try:
                self.log.info(f"[PRELOAD] Initializing model '{name}'...")
                _ = model_obj  # ensures constructor ran

                loaded[name.lower()] = model_obj
                self.log.info(f"[PRELOAD] Model '{name}' loaded successfully.")

            except Exception as exc:
                self.log.error(f"[PRELOAD] FAILED to load model '{name}'. Skipping.")
                self.log.exception(exc)
                continue

        if not loaded:
            raise RuntimeError("[PRELOAD] All models failed to load!")

        # Store in instance cache
        self._answer_model_cache = loaded

        self.log.info("[PRELOAD] Successfully preloaded %d models.", len(loaded))
        return loaded

    # ==================================================================
    # EMBEDDING HELPER
    # ==================================================================

    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute a vector embedding for text using MiniLM, if available.
        Returns list[float] or None.
        """
        if not self._has_sim_model or not text:
            return None
        try:
            vec = self._similarity_model.encode(text)
            return vec.tolist()
        except Exception as e:
            self.log.warning("[EMBED] Failed to compute embedding: %s", e)
            return None

    # ==================================================================
    # QUESTION QUALITY + SIMILARITY LOGIC
    # ==================================================================

    def apply_question_templates(self, raw_questions: List[str]) -> List[str]:
        """
        Normalizes questions into a small set of controlled stems using a
        very light heuristic. This keeps them more consistent for training.
        """
        import re

        normalized: List[str] = []

        for idx, q in enumerate(raw_questions):
            q = (q or "").strip()
            if not q:
                continue

            # If it already looks fine, keep as-is
            if q[0].isupper() and q.endswith("?"):
                normalized.append(q)
                continue

            # Extract a rough "focus" phrase after the question word
            match = re.search(
                r"(what|where|which|when|who)\s+(.*)",
                q,
                flags=re.IGNORECASE,
            )
            if match:
                focus = match.group(2).strip().rstrip("?")
            else:
                focus = q.rstrip("?")

            template = self.QUESTION_TEMPLATES[idx % len(self.QUESTION_TEMPLATES)]
            normalized.append(template.format(focus=focus))

        return normalized

    def question_quality_pass(self, question: str, context: str) -> bool:
        """
        PASS 1 – Basic quality gate for generated questions.
        Returns True if the question is worth keeping.
        """
        if not question:
            return False

        q = question.strip()
        if len(q) < self.DEFAULT_MIN_QUESTION_LEN:
            return False

        lower_q = q.lower()
        if not lower_q.startswith(
                ("what", "where", "which", "when", "who")
        ):
            return False

        if not q.endswith("?"):
            return False

        # Must share some words with context (simple overlap check)
        q_words = set(lower_q.rstrip("?").split())
        c_words = set(context.lower().split())
        overlap = len(q_words & c_words)
        if overlap < 2:
            return False

        return True

    def dedupe_questions(
            self,
            questions: List[str],
            similarity_threshold: float = None,
    ) -> List[str]:
        """
        PASS 2 – Remove near-duplicate questions using MiniLM embeddings,
        if available. Falls back to naive set-based dedupe if model is missing.
        """
        if similarity_threshold is None:
            similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        cleaned: List[str] = []

        if not questions:
            return cleaned

        if not self._has_sim_model:
            # Fallback: simple unique filter with lowercasing
            seen = set()
            for q in questions:
                key = q.strip().lower()
                if key and key not in seen:
                    cleaned.append(q)
                    seen.add(key)
            return cleaned

        embeddings = []
        for q in questions:
            q = q.strip()
            if not q:
                continue
            emb = self._similarity_model.encode(q)
            if embeddings:
                sims = util.cos_sim(emb, embeddings)[0]
                if float(max(sims)) > similarity_threshold:
                    # Too similar to something we already kept
                    continue

            embeddings.append(emb)
            cleaned.append(q)

        return cleaned

    def generate_questions_multi_pass(
            self,
            flan_model: FLAN_QA_Model,
            context: str,
            n: int = None,
            max_retries: int = None,
            similarity_threshold: float = None,
    ) -> List[str]:
        """
        Core question generation routine:

            1) Generate with FLAN
            2) Apply templates
            3) PASS 1 (quality)
            4) PASS 2 (similarity dedupe)
            5) Retry if we end up with nothing
        """
        if n is None:
            n = self.DEFAULT_NUM_QUESTIONS
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_Q_RETRIES
        if similarity_threshold is None:
            similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        last_raw: List[str] = []

        for attempt in range(1, max_retries + 1):
            raw = flan_model.generate_questions(context, n=n) or []
            last_raw = raw

            templated = self.apply_question_templates(raw)
            passed = [q for q in templated if self.question_quality_pass(q, context)]

            deduped = self.dedupe_questions(passed, similarity_threshold=similarity_threshold)

            if deduped:
                self.log.debug(
                    "[QUESTION] PASS pipeline success on attempt %d – kept %d/%d "
                    "questions (context length=%d chars).",
                    attempt,
                    len(deduped),
                    len(raw),
                    len(context),
                )
                return deduped

            self.log.debug(
                "[QUESTION] PASS pipeline failed on attempt %d – retrying. "
                "raw=%d, passed=%d, deduped=%d",
                attempt,
                len(raw),
                len(passed),
                len(deduped),
            )

        # Fallback: return whatever we last got, even if junk
        if last_raw:
            self.log.warning(
                "[QUESTION] Multi-pass generation exhausted retries; returning "
                "last raw FLAN questions without filtering."
            )
            templated = self.apply_question_templates(last_raw)
            return templated

        self.log.warning("[QUESTION] Multi-pass generation produced no questions at all.")
        return []

    # ==================================================================
    # ANSWER RELEVANCE + RANKING
    # ==================================================================

    def answer_relevance_check(self, answer: str, context: str) -> bool:
        """
        Simple check that an answer is at least somewhat grounded in the context.
        """
        if not answer:
            return False

        ans = answer.strip()
        if len(ans.split()) < 3:
            return False

        a_words = set(ans.lower().split())
        c_words = set(context.lower().split())
        overlap = len(a_words & c_words)

        return overlap >= 2

    def rank_answers(
            self,
            answers: Dict[str, str],
            context: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Compute a simple score for each model's answer and return sorted list:
            [(model_name, answer_text, score), ...] (descending by score)

        Scoring:
            - +overlap word count with context
            - +1 if length in [5, 40] words
            - -5 if overlap == 0 (likely hallucination)
        """
        ranked: List[Tuple[str, str, float]] = []
        c_words = set(context.lower().split())

        for model_name, ans in answers.items():
            if not ans:
                ranked.append((model_name, ans, float("-inf")))
                continue

            a_words = set(ans.lower().split())
            overlap = len(a_words & c_words)
            score = float(overlap)

            if overlap == 0:
                score -= 5.0

            length = len(a_words)
            if 5 <= length <= 40:
                score += 1.0

            ranked.append((model_name, ans, score))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    # ==================================================================
    # PIPELINE RUN HELPERS
    # ==================================================================

    def _bind_run_to_document(self, session, run_id: int, document_id: int):
        """
        Links a PipelineRun to the Document it processed.
        MUST run early so PipelineRun exists before adding PipelineRunItems.
        """
        if run_id is None:
            self.log.error("[RUN] WARNING: run_id=None — cannot bind document to run")
            return

        run = (
            session.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .first()
        )

        if run is None:
            self.log.error(f"[RUN] ERROR: No PipelineRun found with id={run_id}")
            return

        self.log.info(f"[RUN] Binding document_id={document_id} → run_id={run_id}")

        run.document_id = document_id

        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            self.log.error(f"[RUN] ERROR: Failed to bind run_id={run_id} to document_id={document_id}")
            self.log.info(exc)
            raise

    def _create_pipeline_run(self, session, document_id, run_type, options_json, models_json, env_json):
        """Create a new PipelineRun record."""
        run = PipelineRun(
            document_id=document_id,
            run_type=run_type,
            options_json=options_json,
            models_json=models_json,
            env_json=env_json,
            started_at=datetime.now(timezone.utc),
        )

        session.add(run)
        session.flush()

        try:
            session.commit()
        except Exception:
            session.rollback()
            raise

        return run.id

    def _attach_document_to_run(self, run_id: int, document_id: int) -> None:
        """
        Set PipelineRun.document_id once the Document exists.
        """
        session = get_qna_session()
        try:
            run = session.get(PipelineRun, run_id)
            if not run:
                self.log.warning("[RUN] attach_document: run id=%s not found", run_id)
                return
            run.document_id = document_id
            session.commit()
            self.log.info(
                "[RUN] Attached document_id=%s to run_id=%s",
                document_id,
                run_id,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _finish_pipeline_run(self, run_id, success: bool, error_message: str | None):
        """
        Marks a pipeline run as finished.
        Updates finished_at, success state, and error_message.
        """
        qa_service = get_qa_service()
        session = qa_service._auto_session()

        run = session.get(PipelineRun, run_id)
        if not run:
            raise RuntimeError(f"PipelineRun id={run_id} not found.")

        now = datetime.now(timezone.utc)

        run.finished_at = now
        run.success = success

        if error_message:
            run.error_message = str(error_message)[:4000]

        try:
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to finish pipeline run id={run_id}: {e}")

    # ======================================================================
    # STAGE 1 — STRUCTURE EXTRACTION
    # ======================================================================

    def stage_structure_only(self, doc_path: Path) -> Path:
        """Extract structure.json -> filesystem only."""
        self.STRUCT_DIR.mkdir(parents=True, exist_ok=True)
        out = self.STRUCT_DIR / f"{doc_path.stem}_structure.json"

        if out.exists():
            self.log.info(f"[STRUCTURE] Using existing: {out}")
            return out

        self.log.info(f"[STRUCTURE] Extracting -> {out}")

        extractor = DocumentStructureExtractor(str(doc_path))
        structure = extractor.extract()

        extractor.save(structure, out)
        self.log.info(f"[STRUCTURE] Wrote structure JSON: {out}")

        return out

    # ======================================================================
    # STAGE 2 — CLEAN CHUNKS + DB INSERT (legacy, no run tracking)
    # ======================================================================

    def stage_clean_chunks(
            self,
            structure_json: Path,
            min_len: int = 40,
    ) -> Path:
        """
        Legacy per-stage version:
            Parse structure.json -> cleaned chunks + insert into DB.

        Creates:
            - qna_documents row
            - qna_chunks rows

        This does NOT use QADatabaseService nor PipelineRun.
        """
        self.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        out = self.CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

        self.log.info(f"[CLEAN] Loading structure -> {structure_json}")

        loader = StructureChunkLoader(
            structure_path=str(structure_json),
            min_length=min_len,
            dedupe=True,
            merge_headings=True,
        )

        chunks = loader.load_clean_chunks()
        self.log.info(f"[CLEAN] Loaded {len(chunks)} cleaned chunks")

        session = get_qna_session()
        try:
            document = Document(
                file_name=structure_json.stem.replace("_structure", ""),
                file_path=str(structure_json),
            )
            session.add(document)
            session.flush()

            with open(out, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    chk = Chunk(
                        document_id=document.id,
                        chunk_id=c["chunk_id"],
                        context=c["pipeline_context"],
                        page=c.get("page"),
                        section=c.get("section"),
                        subsection=c.get("subsection"),
                    )
                    session.add(chk)

            session.commit()

            self.log.info(
                "[CLEAN] Inserted document_id=%s with %d chunks",
                document.id,
                len(chunks),
            )
            self.log.info(f"[CLEAN] Wrote cleaned JSONL -> {out}")
            return out

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ======================================================================
    # STAGE 2 – CLEAN + RUN TRACKING (FULL PIPELINE, SERVICE LAYER)
    # ======================================================================

    def stage_clean_chunks_tracked(
            self,
            doc_path: Path,
            structure_json: Path,
            min_len: int,
            run_id: int,
            embed: bool,
    ) -> Tuple[Path, int, List[Dict[str, Any]]]:
        """
        Full-pipeline version (optimized):
            - uses QADatabaseService
            - creates Document via service
            - creates Chunk rows via service
            - attaches Document to PipelineRun
            - optionally stores chunk embeddings
            - RETURNS cleaned_chunks IN MEMORY

        Returns:
            (clean_jsonl_path, document_id, chunks_list)
        """
        svc = get_qa_service()

        self.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        out = self.CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

        self.log.info(f"[CLEAN/TRACK] Loading structure -> {structure_json}")

        loader = StructureChunkLoader(
            structure_path=str(structure_json),
            min_length=min_len,
            dedupe=True,
            merge_headings=True,
        )
        chunks = loader.load_clean_chunks()
        self.log.info(f"[CLEAN/TRACK] Loaded {len(chunks)} cleaned chunks")

        # 1. Create Document (service-layer)
        file_name = doc_path.stem
        document = svc.add_document(
            run_id=run_id,
            file_name=file_name,
            file_path=str(doc_path),
        )
        document_id = document.id

        # 2. Attach Document to PipelineRun
        self._attach_document_to_run(run_id, document_id)

        # 3. Write JSONL (artifact) + insert chunks into DB
        with open(out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

                chunk_obj = svc.add_chunk(
                    run_id=run_id,
                    document_id=document_id,
                    chunk_id=c["chunk_id"],
                    context=c["pipeline_context"],
                    page=c.get("page"),
                    section=c.get("section"),
                    subsection=c.get("subsection"),
                )

                if embed:
                    vec = self.compute_embedding(chunk_obj.context)
                    if vec is not None:
                        svc.add_embedding(
                            run_id=run_id,
                            parent_type="chunk",
                            parent_id=chunk_obj.id,
                            model_name=self.EMBED_MODEL_NAME,
                            embedding_vector=vec,
                            metadata={
                                "source": "pipeline_clean",
                                "doc_id": document_id,
                                "chunk_id": chunk_obj.chunk_id,
                            },
                        )

        self.log.info(
            "[CLEAN/TRACK] Document id=%s, %d chunks, run_id=%s",
            document_id,
            len(chunks),
            run_id,
        )
        self.log.info(f"[CLEAN/TRACK] Wrote cleaned JSONL -> {out}")

        return out, document_id, chunks

    # ======================================================================
    # STAGE 3 — QUESTIONS (legacy, no run tracking)
    # ======================================================================

    def stage_generate_questions(
            self,
            clean_jsonl: Path,
            num_questions: int = None,
            max_q_retries: int = None,
            similarity_threshold: float = None,
            max_chunks: Optional[int] = None,
    ) -> Path:
        """
        Legacy per-stage version used by CLI --stage questions (no run tracking).
        """
        if num_questions is None:
            num_questions = self.DEFAULT_NUM_QUESTIONS
        if max_q_retries is None:
            max_q_retries = self.DEFAULT_MAX_Q_RETRIES
        if similarity_threshold is None:
            similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        self.QUESTION_DIR.mkdir(parents=True, exist_ok=True)
        out = self.QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

        self.log.info("[QUESTION] Loading FLAN model for question generation...")
        flan = self.get_flan_model()

        with open(clean_jsonl, "r", encoding="utf-8") as fin:
            clean_chunks = [json.loads(line) for line in fin]

        if max_chunks is not None:
            clean_chunks = clean_chunks[:max_chunks]

        if not clean_chunks:
            self.log.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
            return out

        chunk_ids = {c["chunk_id"] for c in clean_chunks}
        self.log.info(f"[QUESTION] Resolving {len(chunk_ids)} chunk_ids to DB rows...")

        session = get_qna_session()
        try:
            db_chunks = (
                session.query(Chunk)
                .filter(Chunk.chunk_id.in_(list(chunk_ids)))
                .all()
            )
            chunk_map = {c.chunk_id: c.id for c in db_chunks}
        finally:
            session.close()

        missing_chunk_ids = chunk_ids - set(chunk_map.keys())
        if missing_chunk_ids:
            self.log.warning(
                "[QUESTION] %d chunk_ids not found in DB (first few: %s)",
                len(missing_chunk_ids),
                list(missing_chunk_ids)[:5],
            )

        written_groups = 0
        total_questions = 0

        svc = get_qa_service()

        with open(out, "w", encoding="utf-8") as fout:
            for chunk in clean_chunks:
                context = chunk["pipeline_context"]
                pipeline_chunk_id = chunk["chunk_id"]

                flan_questions = self.generate_questions_multi_pass(
                    flan_model=flan,
                    context=context,
                    n=num_questions,
                    max_retries=max_q_retries,
                    similarity_threshold=similarity_threshold,
                ) or []

                record = {
                    "chunk_id": pipeline_chunk_id,
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "subsection": chunk.get("subsection"),
                    "context": context,
                    "questions": flan_questions,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                db_chunk_id = chunk_map.get(pipeline_chunk_id)
                if db_chunk_id is None:
                    self.log.warning(
                        "[QUESTION] No DB chunk found for chunk_id=%s; "
                        "skipping DB insert for this chunk.",
                        pipeline_chunk_id,
                    )
                    continue

                session = get_qna_session()
                try:
                    for idx, question_text in enumerate(flan_questions, start=1):
                        q_obj = Question(
                            chunk_id=db_chunk_id,
                            question=question_text,
                            question_index=idx,
                        )
                        session.add(q_obj)
                        total_questions += 1
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

                written_groups += 1

        self.log.info(
            "[QUESTION] Wrote %d chunk question groups to %s",
            written_groups,
            out,
        )
        self.log.info("[QUESTION] Inserted %d questions into DB", total_questions)

        return out

    # ======================================================================
    # STAGE 3 (FULL PIPELINE) — QUESTIONS + RUN TRACKING (SERVICE)
    # ======================================================================

    def stage_generate_questions_tracked(
            self,
            clean_chunks: List[Dict[str, Any]],
            document_id: int,
            run_id: int,
            num_questions: int,
            max_q_retries: int,
            similarity_threshold: float,
            max_chunks: Optional[int],
            embed: bool,
    ) -> Tuple[Path, List[Dict[str, Any]]]:
        """
        Optimized full pipeline version:
            - uses clean_chunks *already in memory*
            - generates questions
            - inserts qna_questions via service
            - attaches PipelineRunItem (handled inside service)
            - optionally stores question embeddings
            - returns (artifact_path, question_records_in_memory)
        """
        self.QUESTION_DIR.mkdir(parents=True, exist_ok=True)
        out = self.QUESTION_DIR / "questions.jsonl"

        self.log.info("[QUESTION/TRACK] Loading FLAN model for question generation...")
        flan = self.get_flan_model()
        svc = get_qa_service()

        if max_chunks is not None:
            clean_chunks = clean_chunks[:max_chunks]

        if not clean_chunks:
            self.log.warning("[QUESTION/TRACK] No cleaned chunks received; nothing to do.")
            return out, []

        session = get_qna_session()
        try:
            doc = session.get(Document, document_id)
            if not doc:
                self.log.error("[QUESTION/TRACK] Document id=%s not found", document_id)
                return out, []

            chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}
        finally:
            session.close()

        records = []
        for chunk in clean_chunks:
            context = chunk["pipeline_context"]
            pipeline_chunk_id = chunk["chunk_id"]

            flan_questions = self.generate_questions_multi_pass(
                flan_model=flan,
                context=context,
                n=num_questions,
                max_retries=max_q_retries,
                similarity_threshold=similarity_threshold,
            ) or []

            records.append(
                {
                    "chunk_id": pipeline_chunk_id,
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "subsection": chunk.get("subsection"),
                    "context": context,
                    "questions": flan_questions,
                }
            )

        with open(out, "w", encoding="utf-8") as fout:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_questions = 0
        for rec in records:
            pipeline_chunk_id = rec["chunk_id"]
            q_texts = rec["questions"] or []

            chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
            if not chunk_obj:
                self.log.warning(
                    "[QUESTION/TRACK] No DB chunk found for chunk_id=%s; skipping.",
                    pipeline_chunk_id,
                )
                continue

            for idx, question_text in enumerate(q_texts, start=1):
                q_obj = svc.add_question(
                    run_id=run_id,
                    chunk_id=chunk_obj.id,
                    question_text=question_text,
                    question_index=idx,
                )
                total_questions += 1

                if embed:
                    vec = self.compute_embedding(q_obj.question)
                    if vec is not None:
                        svc.add_embedding(
                            run_id=run_id,
                            parent_type="question",
                            parent_id=q_obj.id,
                            model_name=self.EMBED_MODEL_NAME,
                            embedding_vector=vec,
                            metadata={
                                "source": "pipeline_questions",
                                "doc_id": document_id,
                                "chunk_id": chunk_obj.chunk_id,
                                "question_index": idx,
                            },
                        )

        self.log.info(
            "[QUESTION/TRACK] Inserted %d questions into DB for document_id=%s (run_id=%s)",
            total_questions,
            document_id,
            run_id,
        )
        self.log.info("[QUESTION/TRACK] Wrote questions JSONL -> %s", out)

        return out, records

    # ======================================================================
    # STAGE 4 — ANSWERS (legacy, no run tracking)
    # ======================================================================

    def stage_generate_answers(
            self,
            question_jsonl: Path,
            models: Optional[List[str]] = None,
            max_chunks: Optional[int] = None,
    ) -> Path:
        """
        Legacy per-stage version, used by CLI --stage answers.
        Keeps behavior unchanged (no run tracking).
        """
        self.ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        out = self.ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

        self.log.info("[ANSWERS] Loading answer models...")
        model_objects = self._init_answer_models(models)

        with open(question_jsonl, "r", encoding="utf-8") as fin:
            question_items = [json.loads(line) for line in fin]

        if max_chunks is not None:
            question_items = question_items[:max_chunks]

        if not question_items:
            self.log.warning("[ANSWERS] No question records found; nothing to do.")
            return out

        pipeline_chunk_ids = {item["chunk_id"] for item in question_items}

        session = get_qna_session()
        try:
            db_chunks = (
                session.query(Chunk)
                .filter(Chunk.chunk_id.in_(list(pipeline_chunk_ids)))
                .all()
            )
            chunk_map = {c.chunk_id: c.id for c in db_chunks}

            db_questions = (
                session.query(Question)
                .filter(Question.chunk_id.in_(list(chunk_map.values())))
                .all()
            )
            question_map = {(q.chunk_id, q.question_index): q.id for q in db_questions}
        finally:
            session.close()

        missing_chunk_ids = pipeline_chunk_ids - set(chunk_map.keys())
        if missing_chunk_ids:
            self.log.warning(
                "[ANSWERS] %d chunk_ids have questions but no DB chunk; first few: %s",
                len(missing_chunk_ids),
                list(missing_chunk_ids)[:5],
            )

        written_records = 0
        total_answers = 0
        total_rankings = 0

        with open(out, "w", encoding="utf-8") as fout:
            for item in question_items:
                context = item["context"]
                pipeline_chunk_id = item["chunk_id"]
                questions = item.get("questions") or []

                db_chunk_id = chunk_map.get(pipeline_chunk_id)
                if db_chunk_id is None:
                    self.log.warning(
                        "[ANSWERS] No DB chunk found for chunk_id=%s; "
                        "skipping DB answer inserts for this chunk's questions.",
                        pipeline_chunk_id,
                    )
                    continue

                for idx, q_text in enumerate(questions, start=1):
                    per_model_best_answer: Dict[str, str] = {}
                    per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                    for model_name, model_obj in model_objects.items():
                        sample_answers: Dict[str, str] = {}

                        for i in range(self.NUM_DETERMINISTIC_SAMPLES):
                            sample_key = f"{model_name}#det{i + 1}"
                            ans = model_obj.generate_answer(context, q_text)
                            sample_answers[sample_key] = ans

                        for i in range(self.NUM_STOCHASTIC_SAMPLES):
                            sample_key = f"{model_name}#stoch{i + 1}"
                            ans = model_obj.generate_answer(context, q_text)
                            sample_answers[sample_key] = ans

                        ranked_samples = self.rank_answers(sample_answers, context)

                        if not ranked_samples:
                            self.log.warning(
                                "[ANSWERS] Model %s produced no usable samples "
                                "for chunk_id=%s question_index=%d",
                                model_name,
                                pipeline_chunk_id,
                                idx,
                            )
                            continue

                        best_key, best_ans, best_score = ranked_samples[0]
                        if len(ranked_samples) > 1:
                            worst_key, worst_ans, worst_score = ranked_samples[-1]
                        else:
                            worst_key, worst_ans, worst_score = (
                                best_key,
                                best_ans,
                                best_score,
                            )

                        per_model_best_answer[model_name] = best_ans

                        per_model_samples[model_name] = [
                            {
                                "sample_id": key,
                                "answer": ans_text,
                                "score": float(score),
                            }
                            for (key, ans_text, score) in ranked_samples
                        ]

                    if not per_model_best_answer:
                        self.log.warning(
                            "[ANSWERS] No models produced valid answers for "
                            "chunk_id=%s question_index=%d",
                            pipeline_chunk_id,
                            idx,
                        )
                        continue

                    cross_model_ranked = self.rank_answers(per_model_best_answer, context)

                    best_model, best_answer, best_score = cross_model_ranked[0]
                    if len(cross_model_ranked) > 1:
                        worst_model, worst_answer, worst_score = cross_model_ranked[-1]
                    else:
                        worst_model, worst_answer, worst_score = (
                            best_model,
                            best_answer,
                            best_score,
                        )

                    answer_scores = {m: float(s) for (m, _, s) in cross_model_ranked}

                    rec = {
                        "chunk_id": pipeline_chunk_id,
                        "question_index": idx,
                        "question": q_text,
                        "context": context,
                        "best_model": best_model,
                        "best_answer": best_answer,
                        "worst_model": worst_model,
                        "worst_answer": worst_answer,
                        "answer_scores": answer_scores,
                        **{f"answer_{m}": a for m, a in per_model_best_answer.items()},
                        "per_model_samples": per_model_samples,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    db_question_id = question_map.get((db_chunk_id, idx))
                    if db_question_id is None:
                        self.log.warning(
                            "[ANSWERS] No DB question row for chunk_id=%s, db_chunk_id=%s, "
                            "question_index=%d; skipping DB inserts for this question.",
                            pipeline_chunk_id,
                            db_chunk_id,
                            idx,
                        )
                        continue

                    session = get_qna_session()
                    try:
                        for rank_idx, (model_name, best_ans_for_model, score) in enumerate(
                                cross_model_ranked
                        ):
                            is_best = rank_idx == 0
                            is_worst = rank_idx == len(cross_model_ranked) - 1 and len(
                                cross_model_ranked
                            ) > 1

                            a_obj = Answer(
                                question_id=db_question_id,
                                model_name=model_name,
                                model_type="causal_lm",
                                model_path=None,
                                answer_text=best_ans_for_model,
                                score=float(score),
                                is_best=is_best,
                                is_worst=is_worst,
                            )
                            session.add(a_obj)
                            total_answers += 1

                        r_obj = AnswerRanking(
                            question_id=db_question_id,
                            best_model=best_model,
                            best_answer=best_answer,
                            worst_model=worst_model,
                            worst_answer=worst_answer,
                            answer_scores=answer_scores,
                        )
                        session.add(r_obj)
                        total_rankings += 1

                        session.commit()
                    except Exception:
                        session.rollback()
                        raise
                    finally:
                        session.close()

                    written_records += 1

        self.log.info(
            "[ANSWERS] Wrote %d Q/A records → %s",
            written_records,
            out,
        )
        self.log.info("[ANSWERS] Inserted %d answers into DB", total_answers)
        self.log.info("[ANSWERS] Inserted %d ranking rows into DB", total_rankings)

        return out

    # ======================================================================
    # STAGE 4 (FULL PIPELINE) — ANSWERS + RUN TRACKING (SERVICE)
    # ======================================================================

    def stage_generate_answers_tracked(
            self,
            question_records: List[Dict[str, Any]],
            document_id: int,
            run_id: int,
            models: List[str],
            max_chunks: Optional[int],
            embed: bool,
            model_cache: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Optimized full-pipeline version (NO JSONL reload):
            - Accepts in-memory question_records from Stage 3
            - Loads answer LLMs (with caching support)
            - Generates multiple samples per model
            - Ranks answers (per model + cross-model)
            - Inserts qna_answers + qna_answer_ranking into DB
            - Optional embeddings
            - Writes JSONL for audit, but DOES NOT read it
        """
        self.ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        out = self.ANSWER_DIR / "answers.jsonl"

        self.log.info("[ANSWERS/TRACK] Loading answer models (cached if available)...")

        model_objects = self._init_answer_models(
            selected_models=models,
            model_cache=model_cache,
        )

        svc = get_qa_service()

        if max_chunks is not None:
            question_records = question_records[:max_chunks]

        if not question_records:
            self.log.warning("[ANSWERS/TRACK] No question_records provided; nothing to do.")
            return out

        session = get_qna_session()
        try:
            doc = session.get(Document, document_id)
            if not doc:
                self.log.error(
                    "[ANSWERS/TRACK] Document id=%s not found; aborting answers stage",
                    document_id,
                )
                return out

            chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}

            db_questions = (
                session.query(Question)
                .join(Chunk, Question.chunk_id == Chunk.id)
                .filter(Chunk.document_id == document_id)
                .all()
            )

            question_map = {(q.chunk_id, q.question_index): q for q in db_questions}

        finally:
            session.close()

        written_records = 0
        total_answers = 0
        total_rankings = 0

        with open(out, "w", encoding="utf-8") as fout:
            for item in question_records:

                context = item["context"]
                pipeline_chunk_id = item["chunk_id"]
                questions = item.get("questions") or []

                chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
                if not chunk_obj:
                    self.log.warning(
                        "[ANSWERS/TRACK] No DB chunk found for chunk_id=%s; skipping.",
                        pipeline_chunk_id,
                    )
                    continue

                for idx, q_text in enumerate(questions, start=1):

                    per_model_best_answer: Dict[str, str] = {}
                    per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                    for model_name, model_obj in model_objects.items():

                        sample_answers: Dict[str, str] = {}

                        for i in range(self.NUM_DETERMINISTIC_SAMPLES):
                            key = f"{model_name}#det{i + 1}"
                            sample_answers[key] = model_obj.generate_answer(context, q_text)

                        for i in range(self.NUM_STOCHASTIC_SAMPLES):
                            key = f"{model_name}#stoch{i + 1}"
                            sample_answers[key] = model_obj.generate_answer(context, q_text)

                        ranked = self.rank_answers(sample_answers, context)
                        if not ranked:
                            self.log.warning(
                                "[ANSWERS/TRACK] Model %s produced no valid ranked answers for chunk_id=%s q_index=%d",
                                model_name,
                                pipeline_chunk_id,
                                idx,
                            )
                            continue

                        best_key, best_ans, best_score = ranked[0]
                        worst_key, worst_ans, worst_score = ranked[-1] if len(ranked) > 1 else ranked[0]

                        per_model_best_answer[model_name] = best_ans
                        per_model_samples[model_name] = [
                            {"sample_id": k, "answer": a, "score": float(s)}
                            for (k, a, s) in ranked
                        ]

                    if not per_model_best_answer:
                        self.log.warning(
                            "[ANSWERS/TRACK] No LLMs produced answers for chunk_id=%s question_index=%d",
                            pipeline_chunk_id,
                            idx,
                        )
                        continue

                    cross_ranked = self.rank_answers(per_model_best_answer, context)

                    best_model, best_answer, best_score = cross_ranked[0]
                    worst_model, worst_answer, worst_score = (
                        cross_ranked[-1] if len(cross_ranked) > 1 else cross_ranked[0]
                    )

                    answer_scores = {m: float(s) for (m, _, s) in cross_ranked}

                    rec = {
                        "chunk_id": pipeline_chunk_id,
                        "question_index": idx,
                        "question": q_text,
                        "context": context,
                        "best_model": best_model,
                        "best_answer": best_answer,
                        "worst_model": worst_model,
                        "worst_answer": worst_answer,
                        "answer_scores": answer_scores,
                        **{f"answer_{m}": a for m, a in per_model_best_answer.items()},
                        "per_model_samples": per_model_samples,
                    }

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    q_obj = question_map.get((chunk_obj.id, idx))
                    if not q_obj:
                        self.log.warning(
                            "[ANSWERS/TRACK] No DB question row for chunk_id=%s question_index=%d",
                            pipeline_chunk_id,
                            idx,
                        )
                        continue

                    for rank_idx, (model_name, best_ans_for_model, score) in enumerate(cross_ranked):

                        is_best = (rank_idx == 0)
                        is_worst = (rank_idx == len(cross_ranked) - 1 and len(cross_ranked) > 1)

                        a_obj = svc.add_answer(
                            run_id=run_id,
                            question_id=q_obj.id,
                            model_name=model_name,
                            answer_text=best_ans_for_model,
                            model_type="causal_lm",
                            model_path=None,
                            score=float(score),
                            is_best=is_best,
                            is_worst=is_worst,
                        )
                        total_answers += 1

                        if embed:
                            vec = self.compute_embedding(a_obj.answer_text)
                            if vec is not None:
                                svc.add_embedding(
                                    run_id=run_id,
                                    parent_type="answer",
                                    parent_id=a_obj.id,
                                    model_name=self.EMBED_MODEL_NAME,
                                    embedding_vector=vec,
                                    metadata={
                                        "source": "pipeline_answers",
                                        "doc_id": document_id,
                                        "chunk_id": chunk_obj.chunk_id,
                                        "question_index": idx,
                                        "model_name": model_name,
                                    },
                                )

                    svc.add_answer_ranking(
                        run_id=run_id,
                        question_id=q_obj.id,
                        best_model=best_model,
                        best_answer=best_answer,
                        worst_model=worst_model,
                        worst_answer=worst_answer,
                        answer_scores=answer_scores,
                    )
                    total_rankings += 1
                    written_records += 1

        self.log.info("[ANSWERS/TRACK] Wrote %d answer records → %s", written_records, out)
        self.log.info("[ANSWERS/TRACK] Inserted %d answers into DB", total_answers)
        self.log.info("[ANSWERS/TRACK] Inserted %d ranking rows into DB", total_rankings)

        return out

    # ======================================================================
    # STAGE 6 — EXPORT DATASET (ALPACA / CHATML / ORPO)
    # ======================================================================

    def stage_export_dataset(
            self,
            answers_jsonl: Path,
            export_format: str = "alpaca",
    ) -> Path:
        """
        Convert the *_answers.jsonl file into a fine-tuning dataset.

        export_format:
            - alpaca : {"instruction","input","output"}
            - chatml : {"messages":[...]}
            - orpo   : {"prompt","chosen","rejected"}
        """
        if not answers_jsonl.exists():
            raise FileNotFoundError(f"Answers JSONL not found: {answers_jsonl}")

        export_path = answers_jsonl.with_suffix(f".{export_format}.jsonl")
        self.log.info(
            "[EXPORT] Exporting %s -> %s (format=%s)",
            answers_jsonl,
            export_path,
            export_format,
        )

        qna_items = []
        with open(answers_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                rec = json.loads(line)
                question = rec.get("question") or ""
                context = rec.get("context") or ""
                best_answer = rec.get("best_answer") or ""
                worst_answer = rec.get("worst_answer") or ""

                if not question or not best_answer:
                    continue

                qna_items.append(
                    {
                        "question": question,
                        "context": context,
                        "best_answer": best_answer,
                        "worst_answer": worst_answer,
                    }
                )

        with open(export_path, "w", encoding="utf-8") as fout:
            if export_format == "alpaca":
                for item in qna_items:
                    block = {
                        "instruction": item["question"],
                        "input": item["context"],
                        "output": item["best_answer"],
                    }
                    fout.write(json.dumps(block, ensure_ascii=False) + "\n")

            elif export_format == "chatml":
                for item in qna_items:
                    block = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a manufacturing training assistant. Answer strictly using the provided context.",
                            },
                            {"role": "user", "content": item["context"]},
                            {"role": "assistant", "content": item["best_answer"]},
                        ]
                    }
                    fout.write(json.dumps(block, ensure_ascii=False) + "\n")

            elif export_format == "orpo":
                for item in qna_items:
                    block = {
                        "prompt": item["question"],
                        "chosen": item["best_answer"],
                        "rejected": item["worst_answer"],
                    }
                    fout.write(json.dumps(block, ensure_ascii=False) + "\n")
            else:
                raise ValueError(f"Unknown export_format: {export_format}")

        self.log.info("[EXPORT] Wrote %d Q&A items to %s", len(qna_items), export_path)
        return export_path

    # ======================================================================
    # STAGE 7 — RANK EXISTING ANSWERS (DB MAINTENANCE, NO RUN TRACKING)
    # ======================================================================

    def stage_rank_answers(self):
        """
        Rank ALL answers already in the database.

        For every question:
            - pull all answers
            - compute best/worst from existing score field
            - store in qna_answer_ranking

        NOTE: This is a DB maintenance utility (re-ranks); it does NOT use
        the service layer nor create PipelineRun rows.
        """
        self.log.info("[RANK] Running answer ranking stage...")

        session = get_qna_session()
        try:
            all_questions = session.query(Question).all()
            self.log.info(f"[RANK] Found {len(all_questions)} questions.")

            for q in all_questions:
                answers = (
                    session.query(Answer)
                    .filter(Answer.question_id == q.id)
                    .all()
                )

                if not answers:
                    self.log.warning(f"[RANK] Question {q.id} has no answers. Skipping.")
                    continue

                answer_scores: Dict[str, float] = {}
                for a in answers:
                    s = a.score if a.score is not None else 0.0
                    if a.model_name not in answer_scores:
                        answer_scores[a.model_name] = float(s)
                    else:
                        answer_scores[a.model_name] = max(answer_scores[a.model_name], float(s))

                best_model = max(answer_scores, key=answer_scores.get)
                worst_model = min(answer_scores, key=answer_scores.get)

                best_answer_obj = next(
                    a for a in answers if a.model_name == best_model and a.score == answer_scores[best_model]
                )
                worst_answer_obj = next(
                    a for a in answers if a.model_name == worst_model and a.score == answer_scores[worst_model]
                )

                r_obj = AnswerRanking(
                    question_id=q.id,
                    best_model=best_model,
                    best_answer=best_answer_obj.answer_text,
                    worst_model=worst_model,
                    worst_answer=worst_answer_obj.answer_text,
                    answer_scores=answer_scores,
                )
                session.add(r_obj)

                self.log.info(
                    f"[RANK] Ranked question {q.id}: best={best_model}, worst={worst_model}"
                )

            session.commit()
            self.log.info("[RANK] Ranking stage complete.")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ======================================================================
    # FULL PIPELINE (SERVICE-LAYER + RUN TRACKING)
    # ======================================================================

    def run_full_pipeline(
            self,
            doc_path: Path,
            max_chunks: Optional[int] = None,
            min_context_len: int = 40,
            num_questions: int = None,
            max_q_retries: int = None,
            similarity_threshold: float = None,
            models: Optional[List[str]] = None,
            embed: bool = False,
            test_mode: bool = False,
    ) -> Path:
        """
        Run the full Q&A pipeline with service-layer and run tracking.

        Args:
            doc_path: Path to the source document
            max_chunks: Maximum number of chunks to process
            min_context_len: Minimum context length for chunks
            num_questions: Number of questions to generate per chunk
            max_q_retries: Maximum retries for question generation
            similarity_threshold: Threshold for question deduplication
            models: List of model names to use for answer generation
            embed: Whether to compute and store embeddings
            test_mode: If True, run minimal test (1 chunk, 1 question)

        Returns:
            Path to the generated answers JSONL file
        """
        if num_questions is None:
            num_questions = self.DEFAULT_NUM_QUESTIONS
        if max_q_retries is None:
            max_q_retries = self.DEFAULT_MAX_Q_RETRIES
        if similarity_threshold is None:
            similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        doc_path = Path(doc_path)

        qa_service = get_qa_service()
        session = qa_service._auto_session()

        self.log.info("=== run_full_pipeline() STARTED ===")
        self.log.info(f"Document path: {doc_path}")

        model_list = models or []
        self.log.debug(f"Requested answer models: {model_list}")

        # TEST MODE OVERRIDES
        if test_mode:
            self.log.warning("TEST MODE ENABLED: Limiting to 1 chunk, 1 question, no retries.")

            max_chunks = 1
            num_questions = 1
            max_q_retries = 0
            similarity_threshold = 0.0
            min_context_len = 1
            os.environ["OPTION_C_TEST_MODE"] = "1"

        # PRELOAD MODELS
        self.log.info("Preloading all enabled answer models...")
        preloaded_answer_models = self.preload_all_answer_models()
        self.log.debug(f"Preloaded models: {list(preloaded_answer_models.keys())}")

        # Metadata for DB
        options = {
            "max_chunks": max_chunks,
            "min_context_len": min_context_len,
            "num_questions": num_questions,
            "max_q_retries": max_q_retries,
            "similarity_threshold": similarity_threshold,
            "embed": embed,
            "test_mode": test_mode,
        }

        env = {
            "source_path": str(doc_path),
            "cwd": str(os.getcwd()),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # CREATE RUN
        run_id = self._create_pipeline_run(
            session=session,
            document_id=None,
            run_type="full",
            options_json=options,
            models_json={"answers": model_list},
            env_json=env,
        )

        self.log.info(f"Created pipeline run_id = {run_id}")

        try:
            # STAGE 1 — STRUCTURE
            self.log.info("Stage 1 – structure")
            struct_path = self.stage_structure_only(doc_path)

            # STAGE 2 — CLEAN + DOCUMENT + CHUNKS (IN MEMORY)
            self.log.info("Stage 2 – clean + insert")
            clean_path, document_id, chunks = self.stage_clean_chunks_tracked(
                doc_path=doc_path,
                structure_json=struct_path,
                min_len=min_context_len,
                run_id=run_id,
                embed=embed,
            )
            self.log.info(f"Clean complete: document_id={document_id}")

            self._bind_run_to_document(session, run_id, document_id)

            # STAGE 3 — QUESTIONS (IN MEMORY)
            self.log.info("Stage 3 – questions")
            questions_path, question_records = self.stage_generate_questions_tracked(
                clean_chunks=chunks,
                document_id=document_id,
                run_id=run_id,
                num_questions=num_questions,
                max_q_retries=max_q_retries,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                embed=embed,
            )

            # STAGE 4 — ANSWERS (CACHED MODELS)
            self.log.info("Stage 4 – answers (cached models)")
            answers_path = self.stage_generate_answers_tracked(
                question_records=question_records,
                document_id=document_id,
                run_id=run_id,
                models=model_list,
                max_chunks=max_chunks,
                embed=embed,
                model_cache=preloaded_answer_models,
            )

            # STAGE 5 — FINISH RUN
            self.log.info("Stage 5 – finish run")
            self._finish_pipeline_run(run_id, success=True, error_message=None)

            self.log.info("=== run_full_pipeline() COMPLETE ===")
            return answers_path

        except Exception as e:
            self.log.error("EXCEPTION TRIGGERED IN PIPELINE")
            self.log.exception(e)
            self._finish_pipeline_run(run_id, success=False, error_message=str(e))
            raise

    # ======================================================================
    # CLI SUPPORT METHODS
    # ======================================================================

    @staticmethod
    def get_default_models_from_db() -> str:
        """
        Returns a comma-separated list of all enabled model names
        from the qna_models table.

        Falls back to empty string if the DB cannot be queried.
        """
        try:
            enabled = LLMModel.load_all_enabled()

            if not enabled:
                return ""

            return ",".join(enabled.keys())

        except Exception:
            return ""


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Option C v3.0 Q&A Pipeline (Service-layer + Run Tracking)"
    )

    p.add_argument(
        "input",
        help="Path to document OR structure/clean/questions/answers JSONL",
    )

    p.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=[
            "structure",
            "clean",
            "questions",
            "answers",
            "rank",
            "export",
            "full",
        ],
        help="Which part of the pipeline to run",
    )

    p.add_argument("--min-context-len", type=int, default=40)
    p.add_argument("--max-chunks", type=int, default=None)

    p.add_argument("--num-questions", type=int, default=QandaPipeline.DEFAULT_NUM_QUESTIONS)
    p.add_argument("--max-q-retries", type=int, default=QandaPipeline.DEFAULT_MAX_Q_RETRIES)
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=QandaPipeline.DEFAULT_SIMILARITY_THRESHOLD,
    )

    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models. Default: all enabled models.",
    )

    p.add_argument(
        "--embed",
        action="store_true",
        help="If set, store embeddings.",
    )

    p.add_argument(
        "--export-format",
        type=str,
        choices=["alpaca", "chatml", "orpo"],
        default="alpaca",
    )

    p.add_argument(
        "--test",
        action="store_true",
        help="Run minimal 1-chunk/1-question/1-answer wiring test.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    log = get_qna_logger("qanda_pipeline")

    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    # MODEL RESOLUTION — dynamic defaults from DB + CLI override
    if args.models:
        selected_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    else:
        log.info("[CLI] --models not supplied; loading enabled LLMs from qna_models...")
        selected_models = LLMModel.get_default_model_names()
        log.info("[CLI] Default enabled models from DB = %s", selected_models)

    if not selected_models:
        log.error("[CLI] No enabled LLM models found and none specified via --models.")
        sys.exit(1)

    # Create pipeline instance
    pipeline = QandaPipeline()

    # STAGE-SPECIFIC HANDLERS
    if args.stage == "structure":
        pipeline.stage_structure_only(inp)
        return

    elif args.stage == "clean":
        struct = inp if inp.name.endswith("_structure.json") else pipeline.stage_structure_only(inp)
        pipeline.stage_clean_chunks(struct, min_len=args.min_context_len)
        return

    elif args.stage == "questions":
        if inp.name.endswith("_clean.jsonl"):
            clean = inp
        else:
            struct = pipeline.stage_structure_only(inp)
            clean = pipeline.stage_clean_chunks(struct, min_len=args.min_context_len)

        pipeline.stage_generate_questions(
            clean,
            num_questions=args.num_questions,
            max_q_retries=args.max_q_retries,
            similarity_threshold=args.similarity_threshold,
            max_chunks=args.max_chunks,
        )
        return

    elif args.stage == "answers":
        if inp.name.endswith("_questions.jsonl"):
            qs = inp
        else:
            struct = pipeline.stage_structure_only(inp)
            clean = pipeline.stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = pipeline.stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )

        pipeline.stage_generate_answers(
            qs,
            models=selected_models,
            max_chunks=args.max_chunks,
        )
        return

    elif args.stage == "rank":
        pipeline.stage_rank_answers()
        return

    elif args.stage == "export":
        if inp.name.endswith("_answers.jsonl"):
            answers_jsonl = inp

        elif inp.name.endswith("_questions.jsonl"):
            answers_jsonl = pipeline.stage_generate_answers(
                inp,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        elif inp.name.endswith("_clean.jsonl"):
            qs = pipeline.stage_generate_questions(
                inp,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = pipeline.stage_generate_answers(
                qs,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        else:
            struct = pipeline.stage_structure_only(inp)
            clean = pipeline.stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = pipeline.stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = pipeline.stage_generate_answers(
                qs,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        pipeline.stage_export_dataset(
            answers_jsonl=answers_jsonl,
            export_format=args.export_format,
        )
        return

    # FULL PIPELINE (service-layer + run tracking)
    else:
        if args.test:
            log.info("\n=== TEST MODE ENABLED ===")
            log.info("Limiting to 1 chunk, 1 question, 1 answer per model.\n")
            pipeline.run_full_pipeline(
                inp,
                max_chunks=1,
                min_context_len=20,
                num_questions=1,
                max_q_retries=0,
                similarity_threshold=0.0,
                models=selected_models,
                embed=False,
                test_mode=True,
            )
            return

        pipeline.run_full_pipeline(
            inp,
            max_chunks=args.max_chunks,
            min_context_len=args.min_context_len,
            num_questions=args.num_questions,
            max_q_retries=args.max_q_retries,
            similarity_threshold=args.similarity_threshold,
            models=selected_models,
            embed=args.embed,
        )


if __name__ == "__main__":
    main()