#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C – Modular Q&A Dataset Pipeline (DB-Enhanced)
----------------------------------------------------

Stages:
    1. structure   -> extract structure.json
    2. clean       -> convert structure.json -> cleaned chunks  + DB insert
    3. questions   -> generate questions only                   + DB insert
    4. answers     -> generate answers only                     + DB insert
    5. full        -> run everything end-to-end

100% offline.
Saves outputs to filesystem AND PostgreSQL training DB.
"""

import sys
import json
import argparse
from pathlib import Path

from option_c_qna.configuration import cfg
from option_c_qna.configuration.logging_config import get_qna_logger
from option_c_qna.configuration.pg_db_config import get_training_session

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
# OUTPUT DIRECTORIES
# -------------------------------------------------------------------
STRUCT_DIR = cfg.STRUCTURE_DIR
CLEAN_DIR = cfg.CLEAN_DIR
QUESTION_DIR = cfg.QUESTIONS_DIR
ANSWER_DIR = cfg.ANSWERS_DIR


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
    log.info(f"[CLEAN] Inserted {len(chunks)} chunks into DB for document_id={document.id}")

    return out


# ======================================================================
# STAGE 3 — QUESTION GENERATION + DB INSERT
# ======================================================================
def stage_generate_questions(clean_jsonl: Path) -> Path:
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

    if not clean_chunks:
        log.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
        return out

    # -------------------------------------------------
    # Build mapping: pipeline chunk_id (string) -> DB chunk primary key
    # -------------------------------------------------
    chunk_ids = {c["chunk_id"] for c in clean_chunks}
    log.info(f"[QUESTION] Resolving {len(chunk_ids)} chunk_ids to DB rows...")

    session = get_training_session()
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
        log.warning(f"[QUESTION] {len(missing_chunk_ids)} chunk_ids not found in DB "
                   f"(first few: {list(missing_chunk_ids)[:5]})")

    written_groups = 0
    total_questions = 0

    # -------------------------------------------------
    # Generate questions, write JSONL, and insert into DB
    # -------------------------------------------------
    with open(out, "w", encoding="utf-8") as fout:
        for chunk in clean_chunks:
            context = chunk["pipeline_context"]
            pipeline_chunk_id = chunk["chunk_id"]

            flan_questions = flan.generate_questions(context, n=3) or []

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
                log.warning(f"[QUESTION] No DB chunk found for chunk_id={pipeline_chunk_id}; "
                            "skipping DB insert for this chunk.")
                continue

            for idx, question_text in enumerate(flan_questions, start=1):
                svc.add_question(
                    chunk_id=db_chunk_id,
                    question_text=question_text,
                    question_index=idx,
                )
                total_questions += 1

            written_groups += 1

    log.info(f"[QUESTION] Wrote {written_groups} chunk question groups to {out}")
    log.info(f"[QUESTION] Inserted {total_questions} questions into DB")

    return out


# ======================================================================
# STAGE 4 — ANSWER GENERATION + DB INSERT
# ======================================================================
def stage_generate_answers(question_jsonl: Path) -> Path:
    """
    Generate answers for each question record and insert into DB.

    Files:
        Input  : *_questions.jsonl
        Output : *_questions_answers.jsonl

    DB:
        - Resolves pipeline chunk_id -> DB chunk_id
        - Resolves (chunk_db_id, question_index) -> qna_questions.id
        - Inserts qna_answers rows for each model
    """
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

    log.info("[ANSWERS] Loading answer models...")

    flan = FLAN_QA_Model()
    tiny = TinyLlamaAnswerGenerator()
    qwen = QwenAnswerGenerator()
    gemma = GemmaAnswerGenerator()
    openelm = OpenELMAnswerGenerator()

    svc = get_qa_service()

    # -------------------------------------------------
    # Load all question records from JSONL
    # -------------------------------------------------
    with open(question_jsonl, "r", encoding="utf-8") as fin:
        question_items = [json.loads(line) for line in fin]

    if not question_items:
        log.warning("[ANSWERS] No question records found; nothing to do.")
        return out

    # -------------------------------------------------
    # Build mapping for questions in DB
    #   pipeline_chunk_id (string) -> db_chunk_id
    #   (db_chunk_id, question_index) -> question_id
    # -------------------------------------------------
    pipeline_chunk_ids = {item["chunk_id"] for item in question_items}

    session = get_training_session()
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

        question_map = {
            (q.chunk_id, q.question_index): q.id
            for q in db_questions
        }
    finally:
        session.close()

    missing_chunk_ids = pipeline_chunk_ids - set(chunk_map.keys())
    if missing_chunk_ids:
        log.warning(f"[ANSWERS] {len(missing_chunk_ids)} chunk_ids have questions but no DB chunk; "
                   f"first few: {list(missing_chunk_ids)[:5]}")

    written_records = 0
    total_answers = 0

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
                log.warning(f"[ANSWERS] No DB chunk found for chunk_id={pipeline_chunk_id}; "
                            "skipping DB answer inserts for this chunk's questions.")
                continue

            for idx, q_text in enumerate(questions, start=1):
                # Generate answers with each model
                answers = {
                    "flan": flan.generate_answer(context, q_text),
                    "tinyllama": tiny.generate_answer(context, q_text),
                    "qwen": qwen.generate_answer(context, q_text),
                    "gemma": gemma.generate_answer(context, q_text),
                    "openelm": openelm.answer(context, q_text),
                }

                rec = {
                    "chunk_id": pipeline_chunk_id,
                    "question_index": idx,
                    "question": q_text,
                    "context": context,
                    **{f"answer_{k}": v for k, v in answers.items()},
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Resolve DB question row
                db_question_id = question_map.get((db_chunk_id, idx))
                if db_question_id is None:
                    log.warning(
                        f"[ANSWERS] No DB question row for chunk_id={pipeline_chunk_id}, "
                        f"db_chunk_id={db_chunk_id}, question_index={idx}; "
                        "skipping DB answer inserts for this question."
                    )
                    continue

                # Insert answers into DB
                for model_name, answer_text in answers.items():
                    svc.add_answer(
                        question_id=db_question_id,
                        model_name=model_name,
                        model_type="causal_lm",
                        model_path="LOCAL_MODEL",
                        answer_text=answer_text,
                    )
                    total_answers += 1

                written_records += 1

    log.info(f"[ANSWERS] Wrote {written_records} question+answer records to {out}")
    log.info(f"[ANSWERS] Inserted {total_answers} answers into DB")

    return out


# ======================================================================
# FULL PIPELINE
# ======================================================================
def run_full_pipeline(doc_path: Path, max_chunks=None, min_context_len=40):
    struct = stage_structure_only(doc_path)
    clean = stage_clean_chunks(struct, min_len=min_context_len)
    questions = stage_generate_questions(clean)
    answers = stage_generate_answers(questions)
    return answers


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Option C Q&A Pipeline (Modular + DB)")
    p.add_argument("input", help="Path to document or _structure.json")
    p.add_argument(
        "--stage",
        choices=["structure", "clean", "questions", "answers", "full"],
        default="full",
    )
    p.add_argument("--min-context-len", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error(f"Input not found: {inp}")
        sys.exit(1)

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
        stage_generate_questions(clean)

    elif args.stage == "answers":
        if inp.name.endswith("_questions.jsonl"):
            qs = inp
        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = stage_generate_questions(clean)
        stage_generate_answers(qs)

    else:
        run_full_pipeline(inp, min_context_len=args.min_context_len)


if __name__ == "__main__":
    main()
