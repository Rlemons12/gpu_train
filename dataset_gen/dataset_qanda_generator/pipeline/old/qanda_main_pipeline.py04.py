#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C v3.0 – Q&A Dataset Pipeline (Service-Layer + Run Tracking)
-------------------------------------------------------------------

Primary usage (production path):

    python -m dataset_qanda_generator.pipeline.qanda_main_pipeline \
        "path/to/document.docx" \
        --stage full \
        --models qwen \
        --max-chunks 3 \
        --embed

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
    - Full pipeline (--stage full) is the "real" production path.
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
    Embedding,       # retained to keep ORM imports complete
    PipelineRun,
)

# -------------------------------------------------------------------
# LOGGER
# -------------------------------------------------------------------
log = get_qna_logger("qanda_pipeline")

# -------------------------------------------------------------------
# MODELS (LLMs)
# -------------------------------------------------------------------
from option_c_qna.models import (
    FLAN_QA_Model,
    TinyLlamaAnswerGenerator,
    QwenAnswerGenerator,
    GemmaAnswerGenerator,
    OpenELMAnswerGenerator,Mistral7BAnswerGenerator
)

# -------------------------------------------------------------------
# STRUCTURE EXTRACTION
# -------------------------------------------------------------------
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
    log.info("[SIM] Loaded all-MiniLM-L6-v2 for question similarity + embeddings.")
except Exception as e:  # noqa: BLE001
    _similarity_model = None
    HAS_SIM_MODEL = False
    log.warning(
        "[SIM] Could not load SentenceTransformer (all-MiniLM-L6-v2). "
        "Similarity-based dedupe + embeddings disabled. Error: %s",
        e,
    )

# -------------------------------------------------------------------
# OUTPUT DIRECTORIES
# -------------------------------------------------------------------
STRUCT_DIR = cfg.STRUCTURE_DIR
CLEAN_DIR = cfg.CLEAN_DIR
QUESTION_DIR = cfg.QUESTIONS_DIR
ANSWER_DIR = cfg.ANSWERS_DIR

# -------------------------------------------------------------------
# HYPERPARAMETERS / DEFAULTS
# -------------------------------------------------------------------
DEFAULT_NUM_QUESTIONS = 3
DEFAULT_MAX_Q_RETRIES = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_MIN_QUESTION_LEN = 12

# Hybrid sampling: per-model answer samples
NUM_DETERMINISTIC_SAMPLES = 3   # e.g. low-temperature style calls
NUM_STOCHASTIC_SAMPLES = 5      # e.g. sampling-style calls
TOTAL_SAMPLES_PER_MODEL = NUM_DETERMINISTIC_SAMPLES + NUM_STOCHASTIC_SAMPLES

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------------------------------------------
# GLOBAL MODEL CACHES (load-once-and-reuse)
# -------------------------------------------------------------------
_FLAN_QA_MODEL: Optional[FLAN_QA_Model] = None
_ANSWER_MODEL_CACHE: Dict[str, Any] = {}


# ==================================================================
# Bind run to document (safe commit)
# ==================================================================
# ==================================================================
# Bind run to document (safe commit)
# ==================================================================
def _bind_run_to_document(session, run_id: int, document_id: int):
    """
    Links a PipelineRun to the Document it processed.
    MUST run early so PipelineRun exists before adding PipelineRunItems.
    """

    if run_id is None:
        print("[RUN] WARNING: run_id=None — cannot bind document to run")
        return

    from option_c_qna.qanda_db import PipelineRun  # local import to avoid circulars

    run = (
        session.query(PipelineRun)
        .filter(PipelineRun.id == run_id)
        .first()
    )

    if run is None:
        print(f"[RUN] ERROR: No PipelineRun found with id={run_id}")
        return

    print(f"[RUN] Binding document_id={document_id} → run_id={run_id}")

    run.document_id = document_id

    try:
        session.commit()   # REQUIRED so that run_id becomes durable
    except Exception as exc:
        session.rollback()
        print(f"[RUN] ERROR: Failed to bind run_id={run_id} to document_id={document_id}")
        print(exc)
        raise

def get_flan_model() -> FLAN_QA_Model:
    """
    Lazily load FLAN-T5-Large once per process and reuse.
    Used for question generation and (optionally) as an answer model.
    """
    global _FLAN_QA_MODEL
    if _FLAN_QA_MODEL is None:
        log.info("[MODEL] Loading FLAN-T5-Large (FLAN_QA_Model) once...")
        _FLAN_QA_MODEL = FLAN_QA_Model()
    return _FLAN_QA_MODEL

def _init_answer_models(selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Instantiate only the requested answer models, with caching.

    - If FLAN is included, reuse the same FLAN instance used for QG.
    - Other models (TinyLlama, Qwen, Gemma, OpenELM) are instantiated once
      and cached in _ANSWER_MODEL_CACHE.
    """
    global _ANSWER_MODEL_CACHE

    all_creators = {
        "flan": FLAN_QA_Model,                # handled via get_flan_model
        "tinyllama": TinyLlamaAnswerGenerator,
        "qwen": QwenAnswerGenerator,
        "gemma": GemmaAnswerGenerator,
        "openelm": OpenELMAnswerGenerator,
        "mistral": Mistral7BAnswerGenerator,
    }

    if not selected_models:
        selected_models = list(all_creators.keys())

    models: Dict[str, Any] = {}

    for name in selected_models:
        if name not in all_creators:
            log.warning("[ANSWERS] Unknown model '%s' requested; skipping.", name)
            continue

        if name == "flan":
            models[name] = get_flan_model()
            continue

        if name not in _ANSWER_MODEL_CACHE:
            log.info("[ANSWERS] Initializing model: %s", name)
            _ANSWER_MODEL_CACHE[name] = all_creators[name]()

        models[name] = _ANSWER_MODEL_CACHE[name]

    return models

# -------------------------------------------------------------------
# EMBEDDING HELPER
# -------------------------------------------------------------------
def compute_embedding(text: str) -> Optional[List[float]]:
    """
    Compute a vector embedding for text using MiniLM, if available.
    Returns list[float] or None.
    """
    if not HAS_SIM_MODEL or not text:
        return None
    try:
        vec = _similarity_model.encode(text)
        # Convert to Python list so SQLAlchemy can store it in pgvector column
        return vec.tolist()
    except Exception as e:  # noqa: BLE001
        log.warning("[EMBED] Failed to compute embedding: %s", e)
        return None

# =====================================================================
# QUESTION TEMPLATE + QUALITY + SIMILARITY LOGIC
# =====================================================================
QUESTION_TEMPLATES = [
    "What is {focus}?",
    "What does {focus} refer to?",
    "Where is {focus} located?",
    "Which {focus} is mentioned?",
    "When does {focus} occur?",
]

def apply_question_templates(raw_questions: List[str]) -> List[str]:
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

        template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        normalized.append(template.format(focus=focus))

    return normalized

def question_quality_pass(question: str, context: str) -> bool:
    """
    PASS 1 – Basic quality gate for generated questions.
    Returns True if the question is worth keeping.
    """
    if not question:
        return False

    q = question.strip()
    if len(q) < DEFAULT_MIN_QUESTION_LEN:
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
    questions: List[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[str]:
    """
    PASS 2 – Remove near-duplicate questions using MiniLM embeddings,
    if available. Falls back to naive set-based dedupe if model is missing.
    """
    cleaned: List[str] = []

    if not questions:
        return cleaned

    if not HAS_SIM_MODEL:
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
        emb = _similarity_model.encode(q)
        if embeddings:
            sims = util.cos_sim(emb, embeddings)[0]
            if float(max(sims)) > similarity_threshold:
                # Too similar to something we already kept
                continue

        embeddings.append(emb)
        cleaned.append(q)

    return cleaned

def generate_questions_multi_pass(
    flan_model: FLAN_QA_Model,
    context: str,
    *,
    document_title: str = "Unknown",
    domain_type: str = "Training",
    intent_type: str = "question_generation",
    section: str = "Unknown",
    subsection: str = "N/A",
    page: str = "N/A",
    n: int = DEFAULT_NUM_QUESTIONS,
    max_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    flan_timeout_s: int | None = None,
) -> List[str]:
    """
    Strict, document-grounded question generation.

    Rules:
      • Questions MUST come from the chunk (context)
      • PromptBuilder controls behavior
      • FLAN only generates text
      • JSON-only output is enforced
      • EMPTY is a valid outcome
    """

    if not context or len(context.split()) < 40:
        log.info("[QUESTION] Context too small — skipping question generation.")
        return []

    prompt_builder = PromptBuilder.for_document_metadata()

    prompt = prompt_builder.render(
        document_title=document_title,
        domain_type=domain_type,
        intent_type=intent_type,
        section=section,
        subsection=subsection,
        page=page,
        context=context,
        num_questions=n,
    )

    preview = " ".join(prompt.split())[:180] + "…"
    log.info(
        "[QUESTION] START | n=%d retries=%d prompt_len=%d preview=%r",
        n,
        max_retries,
        len(prompt),
        preview,
    )

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        log.info("[QUESTION] Attempt %d/%d BEGIN", attempt, max_retries)

        try:
            raw_text = flan_model.generate_from_prompt(
                prompt,
                max_new_tokens=512,
                timeout_s=flan_timeout_s,
            )
        except Exception:
            log.exception("[QUESTION] FLAN generation failed on attempt %d", attempt)
            continue

        log.info(
            "[QUESTION] FLAN returned %d chars in %.2fs",
            len(raw_text),
            time.time() - t0,
        )

        # -------------------------------------------------
        # STRICT JSON PARSING (hard gate)
        # -------------------------------------------------
        try:
            questions = json.loads(raw_text)
            if not isinstance(questions, list):
                raise ValueError("Output is not a JSON list")
        except Exception:
            log.warning(
                "[QUESTION] Invalid JSON output on attempt %d — retrying",
                attempt,
            )
            continue

        # Normalize + sanity
        normalized = [
            q.strip()
            for q in questions
            if isinstance(q, str) and q.strip().endswith("?")
        ]

        # PASS 1 — grounding + quality
        passed = [
            q for q in normalized
            if question_quality_pass(q, context)
        ]

        # PASS 2 — semantic dedupe
        deduped = dedupe_questions(
            passed,
            similarity_threshold=similarity_threshold,
        )

        if deduped:
            log.info(
                "[QUESTION] SUCCESS attempt=%d kept=%d/%d "
                "(context_chars=%d)",
                attempt,
                len(deduped),
                len(questions),
                len(context),
            )
            return deduped

        log.info(
            "[QUESTION] Attempt %d rejected all questions "
            "(raw=%d normalized=%d passed=%d)",
            attempt,
            len(questions),
            len(normalized),
            len(passed),
        )

    # -------------------------------------------------
    # FINAL BEHAVIOR — DO NOT FORCE JUNK
    # -------------------------------------------------
    log.warning(
        "[QUESTION] Exhausted retries — returning EMPTY question set "
        "(chunk does not support valid questions)."
    )
    return []


# =====================================================================
# ANSWER RELEVANCE + RANKING
# =====================================================================
def answer_relevance_check(answer: str, context: str) -> bool:
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

# =====================================================================
# PIPELINE RUN HELPERS (PipelineRun only – everything else via service)
# =====================================================================

def _create_pipeline_run(session, document_id, run_type, options_json, models_json, env_json):
    run = PipelineRun(
        run_type=run_type,
        options_json=options_json,
        models_json=models_json,
        env_json=env_json,
    )

    session.add(run)
    session.flush()      # get run.id

    try:
        session.commit()   # <-- REQUIRED FIX
    except Exception as e:
        session.rollback()
        raise

    return run.id

def _attach_document_to_run(run_id: int, document_id: int) -> None:
    """
    Set PipelineRun.document_id once the Document exists.
    """
    session = get_qna_session()
    try:
        run = session.get(PipelineRun, run_id)
        if not run:
            log.warning("[RUN] attach_document: run id=%s not found", run_id)
            return
        run.document_id = document_id
        session.commit()
        log.info(
            "[RUN] Attached document_id=%s to run_id=%s",
            document_id,
            run_id,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def _finish_pipeline_run(
    run_id: int,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    """
    Mark PipelineRun as finished.
    """
    session = get_qna_session()
    try:
        run = session.get(PipelineRun, run_id)
        if not run:
            log.warning("[RUN] finish: no PipelineRun found for id=%s", run_id)
            return
        run.finished_at = datetime.now(timezone.utc)
        run.success = success
        run.error_message = error_message
        session.commit()
        log.info(
            "[RUN] Finished pipeline run id=%s success=%s",
            run_id,
            success,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ======================================================================
# STAGE 1 — STRUCTURE EXTRACTION (shared by all modes)
# ======================================================================
def stage_structure_only(doc_path: Path) -> Path:
    """Extract structure.json -> filesystem only."""
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    out = STRUCT_DIR / f"{doc_path.stem}_structure.json"

    if out.exists():
        log.info(f"[STRUCTURE] Using existing: {out}")
        return out

    log.info(f"[STRUCTURE] Extracting -> {out}")

    extractor = DocumentStructureExtractor(str(doc_path))
    structure = extractor.extract()

    extractor.save(structure, out)
    log.info(f"[STRUCTURE] Wrote structure JSON: {out}")

    return out

# ======================================================================
# STAGE 2 — CLEAN CHUNKS + DB INSERT (legacy, no run tracking)
# ======================================================================
def stage_clean_chunks(
    structure_json: Path,
    min_len: int = 40,
) -> Path:
    """
    Legacy per-stage version:
        Parse structure.json -> cleaned chunks + insert into DB.

    Creates:
        - qna_documents row
        - qna_chunks rows

    This does NOT use QADatabaseService nor PipelineRun. It is mainly
    for debugging / manual single-stage runs.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

    log.info(f"[CLEAN] Loading structure -> {structure_json}")

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_len,
        dedupe=True,
        merge_headings=True,
    )

    chunks = loader.load_clean_chunks()
    log.info(f"[CLEAN] Loaded {len(chunks)} cleaned chunks")

    session = get_qna_session()
    try:
        document = Document(
            file_name=structure_json.stem.replace("_structure", ""),
            file_path=str(structure_json),
        )
        session.add(document)
        session.flush()  # assign id

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

        log.info(
            "[CLEAN] Inserted document_id=%s with %d chunks",
            document.id,
            len(chunks),
        )
        log.info(f"[CLEAN] Wrote cleaned JSONL -> {out}")
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
    doc_path: Path,
    structure_json: Path,
    min_len: int,
    run_id: int,
    embed: bool,
) -> Tuple[Path, int]:
    """
    Full-pipeline version:
        - uses QADatabaseService
        - creates Document via service
        - creates Chunk rows via service
        - attaches Document to PipelineRun
        - optionally stores chunk embeddings

    Returns:
        (clean_jsonl_path, document_id)
    """
    svc = get_qa_service()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

    log.info(f"[CLEAN/TRACK] Loading structure -> {structure_json}")

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_len,
        dedupe=True,
        merge_headings=True,
    )
    chunks = loader.load_clean_chunks()
    log.info(f"[CLEAN/TRACK] Loaded {len(chunks)} cleaned chunks")

    # 1) Create Document (service-layer)
    file_name = doc_path.stem
    document = svc.add_document(
        run_id=run_id,
        file_name=file_name,
        file_path=str(doc_path),   # store original doc path for easier lookup
    )
    document_id = document.id

    # 2) Attach Document to run
    _attach_document_to_run(run_id, document_id)

    # 3) Save cleaned chunks to disk + insert Chunk rows via service
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
                vec = compute_embedding(chunk_obj.context)
                if vec is not None:
                    svc.add_embedding(
                        run_id=run_id,
                        parent_type="chunk",
                        parent_id=chunk_obj.id,
                        model_name=EMBED_MODEL_NAME,
                        embedding_vector=vec,
                        metadata={
                            "source": "pipeline_clean",
                            "doc_id": document_id,
                            "chunk_id": chunk_obj.chunk_id,
                        },
                    )

    log.info(
        "[CLEAN/TRACK] Document id=%s, %d chunks, run_id=%s",
        document_id,
        len(chunks),
        run_id,
    )
    log.info(f"[CLEAN/TRACK] Wrote cleaned JSONL -> {out}")

    return out, document_id

# ======================================================================
# STAGE 3 — QUESTIONS (legacy, no run tracking)
# ======================================================================
def stage_generate_questions(
    clean_jsonl: Path,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_chunks: Optional[int] = None,
) -> Path:
    """
    Legacy per-stage version used by CLI --stage questions (no run tracking).
    """
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    out = QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

    log.info("[QUESTION] Loading FLAN model for question generation...")
    flan = get_flan_model()

    # Load all cleaned chunks
    with open(clean_jsonl, "r", encoding="utf-8") as fin:
        clean_chunks = [json.loads(line) for line in fin]

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    if not clean_chunks:
        log.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
        return out

    # Resolve chunk_id -> DB chunk.id
    chunk_ids = {c["chunk_id"] for c in clean_chunks}
    log.info(f"[QUESTION] Resolving {len(chunk_ids)} chunk_ids to DB rows...")

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
        log.warning(
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

            flan_questions = generate_questions_multi_pass(
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
                log.warning(
                    "[QUESTION] No DB chunk found for chunk_id=%s; "
                    "skipping DB insert for this chunk.",
                    pipeline_chunk_id,
                )
                continue

            # Legacy mode: no run_id, so we go straight to ORM session
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

    log.info(
        "[QUESTION] Wrote %d chunk question groups to %s",
        written_groups,
        out,
    )
    log.info("[QUESTION] Inserted %d questions into DB", total_questions)

    return out

# ======================================================================
# STAGE 3 (FULL PIPELINE) — QUESTIONS + RUN TRACKING (SERVICE)
# ======================================================================
def stage_generate_questions_tracked(
    clean_jsonl: Path,
    document_id: int,
    run_id: int,
    num_questions: int,
    max_q_retries: int,
    similarity_threshold: float,
    max_chunks: Optional[int],
    embed: bool,
) -> Path:
    """
    Full pipeline version:
        - generates questions
        - inserts qna_questions via service
        - attaches PipelineRunItem (handled inside service)
        - optionally stores embeddings for questions
    """
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    out = QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

    log.info("[QUESTION/TRACK] Loading FLAN model for question generation...")
    flan = get_flan_model()
    svc = get_qa_service()

    # Load all cleaned chunks from JSONL
    with open(clean_jsonl, "r", encoding="utf-8") as fin:
        clean_chunks = [json.loads(line) for line in fin]

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    if not clean_chunks:
        log.warning("[QUESTION/TRACK] No chunks found in clean JSONL; nothing to do.")
        return out

    # Map pipeline chunk_id -> Chunk object
    session = get_qna_session()
    try:
        doc = session.get(Document, document_id)
        if not doc:
            log.error("[QUESTION/TRACK] Document id=%s not found", document_id)
            return out
        chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}
    finally:
        session.close()

    # Pre-generate questions for each chunk
    records = []
    for chunk in clean_chunks:
        context = chunk["pipeline_context"]
        pipeline_chunk_id = chunk["chunk_id"]

        flan_questions = generate_questions_multi_pass(
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

    # Write JSONL
    with open(out, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Insert into DB via service
    total_questions = 0

    for rec in records:
        pipeline_chunk_id = rec["chunk_id"]
        q_texts = rec["questions"] or []

        chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
        if not chunk_obj:
            log.warning(
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
                vec = compute_embedding(q_obj.question)
                if vec is not None:
                    svc.add_embedding(
                        run_id=run_id,
                        parent_type="question",
                        parent_id=q_obj.id,
                        model_name=EMBED_MODEL_NAME,
                        embedding_vector=vec,
                        metadata={
                            "source": "pipeline_questions",
                            "doc_id": document_id,
                            "chunk_id": chunk_obj.chunk_id,
                            "question_index": idx,
                        },
                    )

    log.info(
        "[QUESTION/TRACK] Inserted %d questions into DB for document_id=%s (run_id=%s)",
        total_questions,
        document_id,
        run_id,
    )
    log.info("[QUESTION/TRACK] Wrote questions JSONL -> %s", out)

    return out

# ======================================================================
# STAGE 4 — ANSWERS (legacy, no run tracking)
# ======================================================================
def stage_generate_answers(
    question_jsonl: Path,
    models: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
) -> Path:
    """
    Legacy per-stage version, used by CLI --stage answers.
    Keeps behavior unchanged (no run tracking).
    """
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

    log.info("[ANSWERS] Loading answer models...")
    model_objects = _init_answer_models(models)

    with open(question_jsonl, "r", encoding="utf-8") as fin:
        question_items = [json.loads(line) for line in fin]

    if max_chunks is not None:
        question_items = question_items[:max_chunks]

    if not question_items:
        log.warning("[ANSWERS] No question records found; nothing to do.")
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
        log.warning(
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
                log.warning(
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

                    for i in range(NUM_DETERMINISTIC_SAMPLES):
                        sample_key = f"{model_name}#det{i+1}"
                        ans = model_obj.generate_answer(context, q_text)
                        sample_answers[sample_key] = ans

                    for i in range(NUM_STOCHASTIC_SAMPLES):
                        sample_key = f"{model_name}#stoch{i+1}"
                        ans = model_obj.generate_answer(context, q_text)
                        sample_answers[sample_key] = ans

                    ranked_samples = rank_answers(sample_answers, context)

                    if not ranked_samples:
                        log.warning(
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
                    log.warning(
                        "[ANSWERS] No models produced valid answers for "
                        "chunk_id=%s question_index=%d",
                        pipeline_chunk_id,
                        idx,
                    )
                    continue

                cross_model_ranked = rank_answers(per_model_best_answer, context)

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
                    log.warning(
                        "[ANSWERS] No DB question row for chunk_id=%s, db_chunk_id=%s, "
                        "question_index=%d; skipping DB inserts for this question.",
                        pipeline_chunk_id,
                        db_chunk_id,
                        idx,
                    )
                    continue

                # Insert answers + ranking directly via ORM
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

    log.info(
        "[ANSWERS] Wrote %d Q/A records → %s",
        written_records,
        out,
    )
    log.info("[ANSWERS] Inserted %d answers into DB", total_answers)
    log.info("[ANSWERS] Inserted %d ranking rows into DB", total_rankings)

    return out

# ======================================================================
# STAGE 4 (FULL PIPELINE) — ANSWERS + RUN TRACKING (SERVICE)
# ======================================================================
def stage_generate_answers_tracked(
    question_jsonl: Path,
    document_id: int,
    run_id: int,
    models: List[str],
    max_chunks: Optional[int],
    embed: bool,
) -> Path:
    """
    Full pipeline version:
        - generates answers with multiple models
        - inserts qna_answers + qna_answer_ranking via service
        - attaches all created objects to the run
        - optionally stores embeddings for answers
    """
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

    log.info("[ANSWERS/TRACK] Loading answer models...")
    model_objects = _init_answer_models(models)
    svc = get_qa_service()

    with open(question_jsonl, "r", encoding="utf-8") as fin:
        question_items = [json.loads(line) for line in fin]

    if max_chunks is not None:
        question_items = question_items[:max_chunks]

    if not question_items:
        log.warning("[ANSWERS/TRACK] No question records found; nothing to do.")
        return out

    # Map pipeline chunk_id string -> Chunk object
    session = get_qna_session()
    try:
        doc = session.get(Document, document_id)
        if not doc:
            log.error(
                "[ANSWERS/TRACK] Document id=%s not found; aborting answers stage",
                document_id,
            )
            return out

        chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}

        # Map (chunk_id, question_index) -> Question
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
        for item in question_items:
            context = item["context"]
            pipeline_chunk_id = item["chunk_id"]
            questions = item.get("questions") or []

            chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
            if not chunk_obj:
                log.warning(
                    "[ANSWERS/TRACK] No DB chunk found for chunk_id=%s; skipping.",
                    pipeline_chunk_id,
                )
                continue

            for idx, q_text in enumerate(questions, start=1):
                per_model_best_answer: Dict[str, str] = {}
                per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                for model_name, model_obj in model_objects.items():
                    sample_answers: Dict[str, str] = {}

                    for i in range(NUM_DETERMINISTIC_SAMPLES):
                        sample_key = f"{model_name}#det{i+1}"
                        ans = model_obj.generate_answer(context, q_text)
                        sample_answers[sample_key] = ans

                    for i in range(NUM_STOCHASTIC_SAMPLES):
                        sample_key = f"{model_name}#stoch{i+1}"
                        ans = model_obj.generate_answer(context, q_text)
                        sample_answers[sample_key] = ans

                    ranked_samples = rank_answers(sample_answers, context)

                    if not ranked_samples:
                        log.warning(
                            "[ANSWERS/TRACK] Model %s produced no usable samples "
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
                    log.warning(
                        "[ANSWERS/TRACK] No models produced valid answers for "
                        "chunk_id=%s question_index=%d",
                        pipeline_chunk_id,
                        idx,
                    )
                    continue

                cross_model_ranked = rank_answers(per_model_best_answer, context)

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

                q_obj = question_map.get((chunk_obj.id, idx))
                if not q_obj:
                    log.warning(
                        "[ANSWERS/TRACK] No DB question row for chunk_id=%s, question_index=%d",
                        pipeline_chunk_id,
                        idx,
                    )
                    continue

                # Insert answers via service
                for rank_idx, (model_name, best_ans_for_model, score) in enumerate(
                    cross_model_ranked
                ):
                    is_best = rank_idx == 0
                    is_worst = rank_idx == len(cross_model_ranked) - 1 and len(
                        cross_model_ranked
                    ) > 1

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
                        vec = compute_embedding(a_obj.answer_text)
                        if vec is not None:
                            svc.add_embedding(
                                run_id=run_id,
                                parent_type="answer",
                                parent_id=a_obj.id,
                                model_name=EMBED_MODEL_NAME,
                                embedding_vector=vec,
                                metadata={
                                    "source": "pipeline_answers",
                                    "doc_id": document_id,
                                    "chunk_id": chunk_obj.chunk_id,
                                    "question_index": idx,
                                    "model_name": model_name,
                                },
                            )

                # Insert ranking via service
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

    log.info(
        "[ANSWERS/TRACK] Wrote %d Q/A records → %s",
        written_records,
        out,
    )
    log.info("[ANSWERS/TRACK] Inserted %d answers into DB", total_answers)
    log.info("[ANSWERS/TRACK] Inserted %d ranking rows into DB", total_rankings)

    return out

# ======================================================================
# STAGE 6 — EXPORT DATASET (ALPACA / CHATML / ORPO)
# ======================================================================
def stage_export_dataset(
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
    log.info(
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

    log.info("[EXPORT] Wrote %d Q&A items to %s", len(qna_items), export_path)
    return export_path

# ======================================================================
# STAGE 7 — RANK EXISTING ANSWERS (DB MAINTENANCE, NO RUN TRACKING)
# ======================================================================
def stage_rank_answers():
    """
    Rank ALL answers already in the database.

    For every question:
        - pull all answers
        - compute best/worst from existing score field
        - store in qna_answer_ranking

    NOTE: This is a DB maintenance utility (re-ranks); it does NOT use
    the service layer nor create PipelineRun rows.
    """
    log.info("[RANK] Running answer ranking stage...")

    session = get_qna_session()
    try:
        all_questions = session.query(Question).all()
        log.info(f"[RANK] Found {len(all_questions)} questions.")

        for q in all_questions:
            answers = (
                session.query(Answer)
                .filter(Answer.question_id == q.id)
                .all()
            )

            if not answers:
                log.warning(f"[RANK] Question {q.id} has no answers. Skipping.")
                continue

            answer_scores: Dict[str, float] = {}
            for a in answers:
                s = a.score if a.score is not None else 0.0
                # If multiple answers with same model_name exist, keep max
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

            log.info(
                f"[RANK] Ranked question {q.id}: best={best_model}, worst={worst_model}"
            )

        session.commit()
        log.info("[RANK] Ranking stage complete.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ======================================================================
# FULL PIPELINE (SERVICE-LAYER + RUN TRACKING)
# ======================================================================

def run_full_pipeline(
    doc_path: Path,
    max_chunks: Optional[int] = None,
    min_context_len: int = 40,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    models: Optional[List[str]] = None,
    embed: bool = False,
) -> Path:

    from option_c_qna.qanda_db import get_qa_service
    qa_service = get_qa_service()

    session = qa_service._auto_session()
    qa = qa_service

    print("DEBUG: run_full_pipeline() STARTED")
    print(f"DEBUG: doc_path={doc_path}")

    model_list = models or []

    options = {
        "max_chunks": max_chunks,
        "min_context_len": min_context_len,
        "num_questions": num_questions,
        "max_q_retries": max_q_retries,
        "similarity_threshold": similarity_threshold,
        "embed": embed,
    }

    env = {
        "source_path": str(doc_path),
        "cwd": str(os.getcwd()),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # -------------------------------------------------------
    # 0. CREATE RUN FIRST (document_id not known yet)
    # -------------------------------------------------------
    run_id = _create_pipeline_run(
        session=session,
        document_id=None,             # TEMPORARY
        run_type="full",
        options_json=options,
        models_json={"answers": model_list},
        env_json=env,
    )

    print(f"DEBUG: Created run_id = {run_id}")

    try:
        # ---------------------------------------------------
        # 1. STRUCTURE EXTRACTION
        # ---------------------------------------------------
        print("DEBUG: Stage 1 – structure")
        struct = stage_structure_only(doc_path)

        # ---------------------------------------------------
        # 2. CLEAN + INSERT DOCUMENT + CHUNKS
        # ---------------------------------------------------
        print("DEBUG: Stage 2 – clean + insert")

        clean, document_id = stage_clean_chunks_tracked(
            doc_path=doc_path,
            structure_json=struct,
            min_len=min_context_len,
            run_id=run_id,          # <-- NOW run_id EXISTS
            embed=embed,
        )

        print(f"DEBUG: document_id = {document_id}")

        # -----------------------------------------------
        # 2b. Bind run to document (UPDATE run.document_id)
        # -----------------------------------------------
        _bind_run_to_document(session, run_id, document_id)

        # ---------------------------------------------------
        # 3. QUESTIONS
        # ---------------------------------------------------
        print("DEBUG: Stage 3 – questions")
        questions = stage_generate_questions_tracked(
            clean_jsonl=clean,
            document_id=document_id,
            run_id=run_id,
            num_questions=num_questions,
            max_q_retries=max_q_retries,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks,
            embed=embed,
        )

        # ---------------------------------------------------
        # 4. ANSWERS
        # ---------------------------------------------------
        print("DEBUG: Stage 4 – answers")
        answers = stage_generate_answers_tracked(
            question_jsonl=questions,
            document_id=document_id,
            run_id=run_id,
            models=model_list,
            max_chunks=max_chunks,
            embed=embed,
        )

        # ---------------------------------------------------
        # 5. FINISH RUN
        # ---------------------------------------------------
        print("DEBUG: Stage 5 – finish run")
        _finish_pipeline_run(run_id, success=True, error_message=None)

        return answers

    except Exception as e:
        print("DEBUG: EXCEPTION TRIGGERED")
        print(str(e))
        _finish_pipeline_run(run_id, success=False, error_message=str(e))
        raise

# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Option C v3.0 Q&A Pipeline (Service-layer + Run Tracking)"
    )

    # Input document or preprocessed file
    p.add_argument(
        "input",
        help="Path to document OR structure/clean/questions/answers JSONL",
    )

    # Pipeline stage selector
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

    # Chunk filtering
    p.add_argument("--min-context-len", type=int, default=40)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks/questions for quick tests.",
    )

    # Question generation hyperparameters
    p.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    p.add_argument("--max-q-retries", type=int, default=DEFAULT_MAX_Q_RETRIES)
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Question similarity threshold for dedupe (requires MiniLM).",
    )

    # Answer LLM models used
    p.add_argument(
        "--models",
        type=str,
        default="flan,tinyllama,qwen,gemma,mistral",   # OpenELM optional via CLI
        help="Comma-separated list of answer models to use.",
    )

    # Optional embedding
    p.add_argument(
        "--embed",
        action="store_true",
        help="If set, store embeddings for chunks/questions/answers (requires MiniLM).",
    )

    # Export stage
    p.add_argument(
        "--export-format",
        type=str,
        choices=["alpaca", "chatml", "orpo"],
        default="alpaca",
        help="Format for 'export' stage.",
    )

    return p.parse_args()

def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # ------------------------------
    # STAGE: structure
    # ------------------------------
    if args.stage == "structure":
        stage_structure_only(inp)
        return

    # ------------------------------
    # STAGE: clean
    # ------------------------------
    elif args.stage == "clean":
        struct = inp if inp.name.endswith("_structure.json") else stage_structure_only(inp)
        stage_clean_chunks(struct, min_len=args.min_context_len)
        return

    # ------------------------------
    # STAGE: questions
    # ------------------------------
    elif args.stage == "questions":
        if inp.name.endswith("_clean.jsonl"):
            clean = inp
        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)

        stage_generate_questions(
            clean,
            num_questions=args.num_questions,
            max_q_retries=args.max_q_retries,
            similarity_threshold=args.similarity_threshold,
            max_chunks=args.max_chunks,
        )
        return

    # ------------------------------
    # STAGE: answers
    # ------------------------------
    elif args.stage == "answers":
        if inp.name.endswith("_questions.jsonl"):
            qs = inp
        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )

        stage_generate_answers(
            qs,
            models=selected_models,
            max_chunks=args.max_chunks,
        )
        return

    # ------------------------------
    # STAGE: rank
    # ------------------------------
    elif args.stage == "rank":
        stage_rank_answers()
        return

    # ------------------------------
    # STAGE: export
    # ------------------------------
    elif args.stage == "export":

        if inp.name.endswith("_answers.jsonl"):
            answers_jsonl = inp

        elif inp.name.endswith("_questions.jsonl"):
            answers_jsonl = stage_generate_answers(
                inp,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        elif inp.name.endswith("_clean.jsonl"):
            qs = stage_generate_questions(
                inp,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = stage_generate_answers(
                qs,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = stage_generate_answers(
                qs,
                models=selected_models,
                max_chunks=args.max_chunks,
            )

        stage_export_dataset(
            answers_jsonl=answers_jsonl,
            export_format=args.export_format,
        )
        return

    # -------------------------------------------------------------
    # FULL (tracked, service-layer version)
    # -------------------------------------------------------------
    else:
        # Session is created inside run_full_pipeline()
        run_full_pipeline(
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
