#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C v2.0 – Modular Q&A Dataset Pipeline (DB-Enhanced, Multi-Pass)
----------------------------------------------------------------------

Stages:
    1. structure   -> extract structure.json
    2. clean       -> structure.json -> cleaned chunks  + DB insert
    3. questions   -> generate questions only           + DB insert (multi-pass)
    4. answers     -> generate answers only             + DB insert + ranking
    5. full        -> run 1–4 end-to-end
    6. export      -> export best/worst Q&A to fine-tuning formats

Features:
    - 100% offline.
    - Multi-pass question generation:
        * Template normalization
        * Quality gate (PASS 1)
        * Similarity-based dedupe (PASS 2)
        * Retry loop
    - Answer generation from multiple local models + relevance scoring.
    - Best / worst answer ranking for ORPO preference data.
    - Saves outputs to filesystem AND PostgreSQL training DB.

Note:
    This file is self-contained for the pipeline. It relies on:
        dataset_qanda_generator.configuration.cfg
        dataset_qanda_generator.configuration.logging_config.get_qna_log
        dataset_qanda_generator.configuration.pg_db_config.get_training_session
        dataset_qanda_generator.models
        dataset_qanda_generator.document_structure_extractor
        dataset_qanda_generator.qanda_db
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from option_c_qna.configuration import cfg
from option_c_qna.configuration.logging_config import get_qna_logger
from option_c_qna.configuration.pg_db_config import get_qna_session

log = get_qna_logger("qanda_pipeline")

# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------
from option_c_qna.models import (
    FLAN_QA_Model,
    TinyLlamaAnswerGenerator,
    QwenAnswerGenerator,
    GemmaAnswerGenerator,
    OpenELMAnswerGenerator,
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
# DATABASE SERVICE + ORM MODELS
# -------------------------------------------------------------------
from option_c_qna.qanda_db import get_qa_service
from option_c_qna.qanda_db.qa_db import Chunk, Question, Answer

# -------------------------------------------------------------------
# OPTIONAL – Semantic Similarity (MiniLM)
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util

    _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SIM_MODEL = True
    log.info("[SIM] Loaded all-MiniLM-L6-v2 for question similarity dedupe.")
except Exception as e:  # noqa: BLE001
    _similarity_model = None
    HAS_SIM_MODEL = False
    log.warning(
        "[SIM] Could not load SentenceTransformer (all-MiniLM-L6-v2). "
        "Similarity-based dedupe will be disabled. Error: %s",
        e,
    )

# -------------------------------------------------------------------
# OUTPUT DIRECTORIES
# -------------------------------------------------------------------
STRUCT_DIR = cfg.STRUCTURE_DIR
CLEAN_DIR = cfg.CLEAN_DIR
QUESTION_DIR = cfg.QUESTIONS_DIR
ANSWER_DIR = cfg.ANSWERS_DIR

# If you want, you could later add cfg.DATASET_EXPORT_DIR; for now we keep
# exports alongside the answers JSONL.
# -------------------------------------------------------------------
# HYPERPARAMETERS / DEFAULTS
# -------------------------------------------------------------------
DEFAULT_NUM_QUESTIONS = 3
DEFAULT_MAX_Q_RETRIES = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_MIN_QUESTION_LEN = 12


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
    n: int = DEFAULT_NUM_QUESTIONS,
    max_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[str]:
    """
    Core question generation routine:

        1) Generate with FLAN
        2) Apply templates
        3) PASS 1 (quality)
        4) PASS 2 (similarity dedupe)
        5) Retry if we end up with nothing
    """
    last_raw: List[str] = []

    for attempt in range(1, max_retries + 1):
        raw = flan_model.generate_questions(context, n=n) or []
        last_raw = raw

        templated = apply_question_templates(raw)
        passed = [q for q in templated if question_quality_pass(q, context)]

        deduped = dedupe_questions(passed, similarity_threshold=similarity_threshold)

        if deduped:
            log.debug(
                "[QUESTION] PASS pipeline success on attempt %d – kept %d/%d "
                "questions (context length=%d chars).",
                attempt,
                len(deduped),
                len(raw),
                len(context),
            )
            return deduped

        log.debug(
            "[QUESTION] PASS pipeline failed on attempt %d – retrying. "
            "raw=%d, passed=%d, deduped=%d",
            attempt,
            len(raw),
            len(passed),
            len(deduped),
        )

    # Fallback: return whatever we last got, even if junk
    if last_raw:
        log.warning(
            "[QUESTION] Multi-pass generation exhausted retries; returning "
            "last raw FLAN questions without filtering."
        )
        templated = apply_question_templates(last_raw)
        return templated

    log.warning("[QUESTION] Multi-pass generation produced no questions at all.")
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


# ======================================================================
# STAGE 1 — STRUCTURE EXTRACTION
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
# STAGE 2 — CLEAN CHUNKS + DB INSERT
# ======================================================================
def stage_clean_chunks(structure_json: Path, min_len: int = 40) -> Path:
    """
    Parse structure.json -> cleaned chunks + insert into DB.
    Creates:
        - qna_documents row
        - qna_chunks rows
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

    # -----------------------------
    # DB INSERT – DOCUMENT
    # -----------------------------
    svc = get_qa_service()

    document = svc.add_document(
        file_name=structure_json.stem.replace("_structure", ""),
        file_path=str(structure_json),
    )

    # -----------------------------
    # SAVE OUTPUT + INSERT CHUNKS
    # -----------------------------
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            # Write JSONL line
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

            # DB INSERT – CHUNK
            svc.add_chunk(
                document_id=document.id,
                chunk_id=c["chunk_id"],
                context=c["pipeline_context"],
                page=c.get("page"),
                section=c.get("section"),
                subsection=c.get("subsection"),
            )

    log.info(f"[CLEAN] Wrote cleaned JSONL -> {out}")
    log.info(
        "[CLEAN] Inserted %d chunks into DB for document_id=%s",
        len(chunks),
        document.id,
    )

    return out


# ======================================================================
# STAGE 3 — QUESTION GENERATION + DB INSERT (MULTI-PASS)
# ======================================================================
def stage_generate_questions(
    clean_jsonl: Path,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_chunks: Optional[int] = None,
) -> Path:
    """
    Generate questions for each cleaned chunk and insert into DB.

    Files:
        Input  : *_clean.jsonl
        Output : *_clean_questions.jsonl

    DB:
        - Reads qna_chunks by chunk_id (string) from clean_jsonl
        - Inserts qna_questions rows via QADatabaseService
    """
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    out = QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

    log.info("[QUESTION] Loading FLAN model for question generation...")
    flan = FLAN_QA_Model()
    svc = get_qa_service()

    # -------------------------------------------------
    # Load all cleaned chunks into memory
    # -------------------------------------------------
    with open(clean_jsonl, "r", encoding="utf-8") as fin:
        clean_chunks = [json.loads(line) for line in fin]

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    if not clean_chunks:
        log.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
        return out

    # -------------------------------------------------
    # Build mapping: pipeline chunk_id (string) -> DB chunk primary key
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Generate questions, write JSONL, and insert into DB
    # -------------------------------------------------
    with open(out, "w", encoding="utf-8") as fout:
        for chunk in clean_chunks:
            context = chunk["pipeline_context"]
            pipeline_chunk_id = chunk["chunk_id"]

            # Multi-pass FLAN generation
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

            for idx, question_text in enumerate(flan_questions, start=1):
                svc.add_question(
                    chunk_id=db_chunk_id,
                    question_text=question_text,
                    question_index=idx,
                )
                total_questions += 1

            written_groups += 1

    log.info(
        "[QUESTION] Wrote %d chunk question groups to %s",
        written_groups,
        out,
    )
    log.info("[QUESTION] Inserted %d questions into DB", total_questions)

    return out


# ======================================================================
# STAGE 4 — ANSWER GENERATION + DB INSERT (+ RANKING)
# ======================================================================
def _init_answer_models(selected_models: Optional[List[str]] = None):
    """
    Helper: instantiate only the answer models requested.
    """
    all_creators = {
        "flan": FLAN_QA_Model,
        "tinyllama": TinyLlamaAnswerGenerator,
        "qwen": QwenAnswerGenerator,
        "gemma": GemmaAnswerGenerator,
        "openelm": OpenELMAnswerGenerator,
    }

    if not selected_models:
        selected_models = list(all_creators.keys())

    models = {}
    for name, ctor in all_creators.items():
        if name in selected_models:
            log.info("[ANSWERS] Initializing model: %s", name)
            models[name] = ctor()

    return models


def stage_generate_answers(
    question_jsonl: Path,
    models: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
) -> Path:
    """
    Generate answers for each question record and insert into DB.

    Files:
        Input  : *_questions.jsonl
        Output : *_questions_answers.jsonl

    DB:
        - Resolves pipeline chunk_id -> DB chunk_id
        - Resolves (db_chunk_id, question_index) -> qna_questions.id
        - Inserts qna_answers rows for each model (with scores + best/worst flags)
        - Inserts one qna_answer_ranking row per question
    """
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

    log.info("[ANSWERS] Loading answer models...")
    model_objects = _init_answer_models(models)
    svc = get_qa_service()

    # -------------------------------------------------
    # Load all question records from JSONL
    # -------------------------------------------------
    with open(question_jsonl, "r", encoding="utf-8") as fin:
        question_items = [json.loads(line) for line in fin]

    if max_chunks is not None:
        question_items = question_items[:max_chunks]

    if not question_items:
        log.warning("[ANSWERS] No question records found; nothing to do.")
        return out

    # -------------------------------------------------
    # Build mapping for questions in DB
    #   pipeline_chunk_id (string) -> db_chunk_id
    #   (db_chunk_id, question_index) -> question_id
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Generate answers, write JSONL, and insert into DB
    # -------------------------------------------------
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
                # -------------------------------------------------
                # 1) Generate answers with each selected model
                # -------------------------------------------------
                answers: Dict[str, str] = {}
                for model_name, model_obj in model_objects.items():
                    if model_name == "flan":
                        ans = model_obj.generate_answer(context, q_text)
                    elif model_name == "openelm":
                        ans = model_obj.answer(context, q_text)
                    else:
                        ans = model_obj.generate_answer(context, q_text)
                    answers[model_name] = ans

                # -------------------------------------------------
                # 2) Rank answers for ORPO-style preference data
                #    rank_answers should return:
                #    List[(model_name, answer_text, score)], sorted best→worst
                # -------------------------------------------------
                ranked = rank_answers(answers, context)

                if ranked:
                    best_model, best_answer, best_score = ranked[0]
                    if len(ranked) > 1:
                        worst_model, worst_answer, worst_score = ranked[-1]
                    else:
                        worst_model, worst_answer, worst_score = (
                            best_model,
                            best_answer,
                            best_score,
                        )
                    answer_scores = {m: s for (m, _, s) in ranked}
                else:
                    # Fallback: no ranking info
                    best_model = ""
                    best_answer = ""
                    best_score = 0.0
                    worst_model = ""
                    worst_answer = ""
                    worst_score = 0.0
                    answer_scores = {}

                # -------------------------------------------------
                # 3) Write JSONL record (for inspection / downstream)
                # -------------------------------------------------
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
                    **{f"answer_{k}": v for k, v in answers.items()},
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # -------------------------------------------------
                # 4) Resolve DB question row
                # -------------------------------------------------
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

                # -------------------------------------------------
                # 5) Insert answers into DB (one row per model)
                #    with score + best/worst flags
                # -------------------------------------------------
                for rank_idx, (model_name, answer_text, score) in enumerate(ranked):
                    is_best = (rank_idx == 0)
                    is_worst = (rank_idx == len(ranked) - 1 and len(ranked) > 1)

                    svc.add_answer(
                        question_id=db_question_id,
                        model_name=model_name,
                        answer_text=answer_text,
                        model_type="causal_lm",
                        model_path=None,
                        score=score,
                        is_best=is_best,
                        is_worst=is_worst,
                    )
                    total_answers += 1

                # -------------------------------------------------
                # 6) Insert / update the 1–1 AnswerRanking record
                # -------------------------------------------------
                if ranked:
                    svc.add_answer_ranking(
                        question_id=db_question_id,
                        best_model=best_model,
                        best_answer=best_answer,
                        worst_model=worst_model,
                        worst_answer=worst_answer,
                        answer_scores=answer_scores,
                    )
                    total_rankings += 1

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
# STAGE 5 — FULL PIPELINE
# ======================================================================
def run_full_pipeline(
    doc_path: Path,
    max_chunks: Optional[int] = None,
    min_context_len: int = 40,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    models: Optional[List[str]] = None,
) -> Path:
    struct = stage_structure_only(doc_path)
    clean = stage_clean_chunks(struct, min_len=min_context_len)
    questions = stage_generate_questions(
        clean,
        num_questions=num_questions,
        max_q_retries=max_q_retries,
        similarity_threshold=similarity_threshold,
        max_chunks=max_chunks,
    )
    answers = stage_generate_answers(
        questions,
        models=models,
        max_chunks=max_chunks,
    )
    return answers


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

def stage_rank_answers():
    """
    Rank ALL answers already in the database.
    For every question: pull all answers, compute best/worst, store in qna_answer_ranking.
    """
    log.info("[RANK] Running answer ranking stage...")

    svc = get_qa_service()

    # We need Answer + Question models
    from option_c_qna.qanda_db.qa_db import Question, Answer
    session = get_qna_session()

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

        # -----------------------------
        # Compute ranking scores
        # -----------------------------
        answer_scores = {}

        for a in answers:
            # If no score exists, treat it as 0.0
            s = a.score if a.score is not None else 0.0
            answer_scores[a.model_name] = float(s)

        # Identify best/worst
        best_model = max(answer_scores, key=answer_scores.get)
        worst_model = min(answer_scores, key=answer_scores.get)

        # Collect text for best/worst
        best_answer_obj = next(a for a in answers if a.model_name == best_model)
        worst_answer_obj = next(a for a in answers if a.model_name == worst_model)

        # -----------------------------
        # Save to qna_answer_ranking
        # -----------------------------
        svc.add_answer_ranking(
            question_id=q.id,
            best_model=best_model,
            best_answer=best_answer_obj.answer_text,
            worst_model=worst_model,
            worst_answer=worst_answer_obj.answer_text,
            answer_scores=answer_scores,
        )

        log.info(f"[RANK] Ranked question {q.id}: best={best_model}, worst={worst_model}")

    log.info("[RANK] Ranking stage complete.")


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Option C v2.0 Q&A Pipeline (Modular + DB)")

    # Input document or preprocessed file
    p.add_argument(
        "input",
        help="Path to document OR structure/clean/questions/answers JSONL"
    )

    # Pipeline stage selector (UPDATED)
    p.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=[
            "structure",
            "clean",
            "questions",
            "answers",
            "rank",        # NEW
            "export",
            "full"
        ],
        help="Which part of the pipeline to run",
    )

    # Chunk filtering
    p.add_argument("--min-context-len", type=int, default=40)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks/questions for quick tests."
    )

    # Question generation hyperparameters
    p.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    p.add_argument("--max-q-retries", type=int, default=DEFAULT_MAX_Q_RETRIES)
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Question similarity threshold for dedupe (requires MiniLM)."
    )

    # Answer LLM models used
    p.add_argument(
        "--models",
        type=str,
        default="flan,tinyllama,qwen,gemma,openelm",
        help="Comma-separated list of answer models to use."
    )

    # Export stage
    p.add_argument(
        "--export-format",
        type=str,
        choices=["alpaca", "chatml", "orpo"],
        default="alpaca",
        help="Format for 'export' stage."
    )

    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.stage == "structure":
        stage_structure_only(inp)

    elif args.stage == "clean":
        struct = inp if inp.name.endswith("_structure.json") else stage_structure_only(inp)
        stage_clean_chunks(struct, min_len=args.min_context_len)

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

    elif args.stage == "export":
        # input is expected to be *_answers.jsonl or something earlier
        if inp.name.endswith("_answers.jsonl"):
            answers_jsonl = inp
        elif inp.name.endswith("_questions.jsonl"):
            # Need to generate answers first, then export
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

    else:
        # full
        run_full_pipeline(
            inp,
            max_chunks=args.max_chunks,
            min_context_len=args.min_context_len,
            num_questions=args.num_questions,
            max_q_retries=args.max_q_retries,
            similarity_threshold=args.similarity_threshold,
            models=selected_models,
        )


if __name__ == "__main__":
    main()

"""python -m dataset_qanda_generator.qanda_pipeline "E:\emtac\data\raw_documention\FB4-GENERAL\AFL31600 Overview.docx" --stage full --max-chunks 5 --num-questions 2
python -m dataset_qanda_generator.qanda_pipeline "path\to\doc.docx" --stage questions --similarity-threshold 0.75 --num-questions 4
python -m dataset_qanda_generator.qanda_pipeline "path\to\*_questions.jsonl" --stage answers --models flan,qwen
python -m dataset_qanda_generator.qanda_pipeline "path\to\*_answers.jsonl" --stage export --export-format orpo
python -m dataset_qanda_generator.qanda_pipeline "E:\emtac\data\raw_documention\FB4-GENERAL\AFL31600 Overview.docx" --stage full

"""