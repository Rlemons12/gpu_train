#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OPTION C – Q&A SYSTEM MODULE CHECK
==================================

This script validates:
    • .env accessibility
    • Model paths
    • Pipeline directories
    • Training DB connectivity
    • Portable PostgreSQL auto-start (pg_ctl)
    • pgvector extension
    • Required tables
    • QADatabaseService functionality
    • Lists all tables in the current DB
"""

import os
import sys
import time
from pathlib import Path

from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError

# -----------------------------------------
# Logging
# -----------------------------------------
from option_c_qna.configuration.logging_config import get_qna_logger
log = get_qna_logger("module_check")

# -----------------------------------------
# Configuration
# -----------------------------------------
from option_c_qna.configuration import cfg
from option_c_qna.configuration.pg_db_config import (
    train_engine,
    get_training_session,
)

# -----------------------------------------
# Database Service Layer + ORM
# -----------------------------------------
from option_c_qna.qanda_db import (
    create_schema,
)

# -----------------------------------------
# Portable PostgreSQL control
# -----------------------------------------
from option_c_qna.tools.start_postgres import start_postgres


# ---------------------------------------------------------
# 1. .env file check
# ---------------------------------------------------------
def check_env_loaded():
    log.info("Checking which .env was loaded...")

    try:
        env_path = cfg.ENV_PATH
        log.info(f".env loaded from: {env_path}")
        return True
    except Exception:
        log.error("Could not determine .env path via cfg")
        return False


# ---------------------------------------------------------
# 2. Training DB connectivity — with auto-start
# ---------------------------------------------------------
def check_training_db_connection():
    log.info("Checking Training DB connection...")

    try:
        with train_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Training database connection OK")
        return True

    except OperationalError:
        log.warning("Could not connect. Attempting to start portable PostgreSQL...")

        try:
            start_postgres()
            time.sleep(3)

            with train_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            log.info("DB reachable after auto-start")
            return True

        except Exception as e:
            log.error(f"Still cannot connect to DB: {e}")
            return False


# ---------------------------------------------------------
# 3. pgvector check
# ---------------------------------------------------------
def check_pgvector():
    log.info("Checking pgvector extension...")

    try:
        with train_engine.connect() as conn:
            res = conn.execute(text(
                "SELECT extname FROM pg_extension WHERE extname='vector';"
            )).fetchone()

        if res:
            log.info("pgvector installed")
            return True
        else:
            log.error("pgvector NOT installed in training DB")
            return False

    except Exception as e:
        log.error(f"pgvector check failed: {e}")
        return False


# ---------------------------------------------------------
# 4. Required tables
# ---------------------------------------------------------
REQUIRED_TABLES = {
    "qna_documents",
    "qna_chunks",
    "qna_questions",
    "qna_answers",
    "qna_models",
    "qna_embeddings",
}


def check_tables_exist(auto_create=True):
    log.info("Checking required tables...")

    try:
        inspector = inspect(train_engine)
        existing = set(inspector.get_table_names())
    except Exception as e:
        log.error(f"Could not inspect training DB tables: {e}")
        return False

    missing = REQUIRED_TABLES - existing

    if not missing:
        log.info("All required tables exist")
        return True

    log.warning(f"Missing tables: {missing}")

    if auto_create:
        log.info("Creating missing schema...")
        try:
            create_schema()
            log.info("Schema created")
            return True
        except Exception as e:
            log.error(f"Failed creating schema: {e}")
            return False

    return False


# ---------------------------------------------------------
# 4b. List all tables in DB (new)
# ---------------------------------------------------------
def list_all_tables():
    """Return and print a list of all tables in the database."""
    try:
        inspector = inspect(train_engine)
        tables = inspector.get_table_names()

        log.info("Listing all tables in DB:")
        for t in tables:
            log.info(f"  • {t}")

        return tables

    except Exception as e:
        log.error(f"Could not list tables: {e}")
        return []


# ---------------------------------------------------------
# 5. Model paths from .env
# ---------------------------------------------------------
MODEL_KEYS = [
    "MODELS_LLM_DIR",
    "MODELS_QWEN_DIR",
    "MODELS_TINY_LLAMA_DIR",
    "MODELS_APPLE_ELM_DIR",
    "MODELS_GEMMA_DIR",
    "MODELS_FLAN_DIR",
    "MODEL_MINILM_DIR",
]


def check_model_paths():
    log.info("Checking model paths...")

    ok = True

    for key in MODEL_KEYS:
        if not hasattr(cfg, key):
            log.error(f"Missing cfg.{key}")
            ok = False
            continue

        p = Path(getattr(cfg, key))
        if not p.exists():
            log.error(f"Model path missing: {key} → {p}")
            ok = False
        else:
            log.info(f"{key} OK")

    return ok


# ---------------------------------------------------------
# 6. Pipeline directories
# ---------------------------------------------------------
REQUIRED_DIRS = [
    cfg.STRUCTURE_DIR,
    cfg.CLEAN_DIR,
    cfg.QUESTIONS_DIR,
    cfg.ANSWERS_DIR,
    cfg.LOG_DIR,
]


def check_directories():
    log.info("Checking pipeline directories...")

    ok = True

    for d in REQUIRED_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            log.info(f"Directory OK: {d}")
        except Exception as e:
            log.error(f"Could not create directory {d}: {e}")
            ok = False

    return ok


# ---------------------------------------------------------
# 7. Service Layer Test
# ---------------------------------------------------------
def check_service_layer():
    log.info("Testing QADatabaseService...")

    try:
        from option_c_qna.qanda_db import (
            Document,
            Chunk,
            Question,
            Answer,
            Embedding,
            LLMModel,
        )

        session = get_training_session()

        session.query(Document).limit(1).all()
        session.query(Chunk).limit(1).all()
        session.query(Question).limit(1).all()
        session.query(Answer).limit(1).all()
        session.query(Embedding).limit(1).all()
        session.query(LLMModel).limit(1).all()

        log.info("QADatabaseService basic load OK")
        return True

    except Exception as e:
        log.error(f"Service layer FAILED: {e}")
        return False

# ---------------------------------------------------------
# EXTRA CHECK A: Row counts for Option C tables only
# ---------------------------------------------------------
def check_table_row_counts():
    log.info("Checking row counts for Option C tables...")

    try:
        inspector = inspect(train_engine)
        tables = inspector.get_table_names()

        # Only count rows for Option C tables
        qna_tables = [t for t in tables if t.startswith("qna_")]

        session = get_training_session()

        for tbl in qna_tables:
            count = session.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            log.info(f"  {tbl:20} rows: {count}")

        return True

    except Exception as e:
        log.error(f"Row count check failed: {e}")
        return False


# ---------------------------------------------------------
# EXTRA CHECK B: Schema column verification
# ---------------------------------------------------------
EXPECTED_SCHEMA = {
    "qna_documents": {
        "id", "file_name", "file_path", "created_at"
    },

    "qna_chunks": {
        "id", "document_id", "chunk_id", "page",
        "section", "subsection", "context"
    },

    "qna_questions": {
        "id", "chunk_id", "question_index", "question"
    },

    # UPDATED FOR NEW SCHEMA
    "qna_answers": {
        "id",
        "question_id",
        "model_name",
        "model_type",
        "model_path",
        "answer_text",
        "score",
        "is_best",
        "is_worst",
    },

    "qna_models": {
        "id", "name", "model_type", "model_path"
    },

    "qna_embeddings": {
        "id",
        "parent_type",
        "parent_id",
        "model_name",
        "embedding_vector",
        "actual_dimensions",
        "meta_json",
        "created_at",
        "updated_at",
    },

    # NEW: answer ranking schema
    "qna_answer_ranking": {
        "id",
        "question_id",
        "best_model",
        "best_answer",
        "worst_model",
        "worst_answer",
        "answer_scores",
    },

    # NEW: pipeline run schema
    "qna_pipeline_runs": {
        "id",
        "document_id",
        "run_type",
        "options_json",
        "models_json",
        "env_json",
        "started_at",
        "finished_at",
        "success",
        "error_message",
    },

    # NEW: pipeline run items schema
    "qna_pipeline_run_items": {
        "id",
        "run_id",
        "parent_type",
        "parent_id",
        "created_at",
    },
}



def check_schema_columns():
    log.info("Checking schema column definitions...")

    inspector = inspect(train_engine)
    ok = True

    for table, expected_cols in EXPECTED_SCHEMA.items():
        try:
            actual_cols = {col["name"] for col in inspector.get_columns(table)}
        except Exception:
            log.error(f"  Table missing: {table}")
            ok = False
            continue

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            log.error(f"  {table}: missing columns: {missing}")
            ok = False
        if extra:
            log.warning(f"  {table}: unexpected extra columns: {extra}")

        if not missing:
            log.info(f"  {table}: columns OK")

    return ok


# ---------------------------------------------------------
# EXTRA CHECK C: DB integrity rules
# ---------------------------------------------------------
def check_integrity():
    log.info("Checking referential integrity...")

    session = get_training_session()
    ok = True

    # 1. Chunks must map to valid documents
    invalid_chunks = session.execute(text("""
        SELECT c.id, c.document_id
        FROM qna_chunks c
        LEFT JOIN qna_documents d ON c.document_id = d.id
        WHERE d.id IS NULL
    """)).fetchall()

    if invalid_chunks:
        log.error(f"Invalid chunks referencing missing documents: {invalid_chunks}")
        ok = False

    # 2. Questions must map to valid chunks
    invalid_questions = session.execute(text("""
        SELECT q.id, q.chunk_id
        FROM qna_questions q
        LEFT JOIN qna_chunks c ON q.chunk_id = c.id
        WHERE c.id IS NULL
    """)).fetchall()

    if invalid_questions:
        log.error(f"Invalid questions referencing missing chunks: {invalid_questions}")
        ok = False

    # 3. Answers must map to valid questions
    invalid_answers = session.execute(text("""
        SELECT a.id, a.question_id
        FROM qna_answers a
        LEFT JOIN qna_questions q ON a.question_id = q.id
        WHERE q.id IS NULL
    """)).fetchall()

    if invalid_answers:
        log.error(f"Invalid answers referencing missing questions: {invalid_answers}")
        ok = False

    # 4. Embeddings must reference a valid parent entity
    invalid_embeddings = session.execute(text("""
        SELECT id, parent_type, parent_id
        FROM qna_embeddings
        WHERE (
            (parent_type='document' AND parent_id NOT IN (SELECT id FROM qna_documents)) OR
            (parent_type='chunk'    AND parent_id NOT IN (SELECT id FROM qna_chunks)) OR
            (parent_type='question' AND parent_id NOT IN (SELECT id FROM qna_questions)) OR
            (parent_type='answer'   AND parent_id NOT IN (SELECT id FROM qna_answers))
        )
    """)).fetchall()

    if invalid_embeddings:
        log.error(f"Invalid embeddings referencing missing parent rows: {invalid_embeddings}")
        ok = False

    if ok:
        log.info("Referential integrity OK")

    return ok


# ---------------------------------------------------------
# EXTRA CHECK D: Vector dimension validation
# ---------------------------------------------------------
def check_vector_dimensions(expected_dim=384):
    log.info("Checking embedding vector dimensions...")

    session = get_training_session()

    rows = session.execute(text("""
        SELECT id, actual_dimensions
        FROM qna_embeddings
        WHERE actual_dimensions IS NOT NULL
    """)).fetchall()

    ok = True

    for row_id, dim in rows:
        if dim != expected_dim:
            log.error(f"Embedding {row_id} has incorrect dimension {dim} (expected {expected_dim})")
            ok = False

    if ok:
        log.info("All embedding vector dimensions OK")

    return ok


# ---------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------
def main():
    log.info("\n=== OPTION C Q&A SYSTEM – MODULE CHECK START ===\n")
    log.info("Starting environment + system validation...\n")

    results = {
        "env_loaded": check_env_loaded(),
        "db_connection": check_training_db_connection(),
        "pgvector": check_pgvector(),
        "tables": check_tables_exist(auto_create=True),
        "model_paths": check_model_paths(),
        "directories": check_directories(),
        "service_layer": check_service_layer(),
        "schema_columns": check_schema_columns(),
        "row_counts": check_table_row_counts(),
        "integrity": check_integrity(),
        "vector_dimensions": check_vector_dimensions(),

    }

    # Immediately list tables AFTER table checks run
    list_all_tables()

    log.info("\n=== MODULE CHECK RESULTS ===\n")
    for key, value in results.items():
        log.info(f"{key:15}: {'PASS' if value else 'FAIL'}")

    if all(results.values()):
        log.info("\nALL CHECKS PASSED — SYSTEM READY\n")
        sys.exit(0)
    else:
        log.error("\nCHECKS FAILED — SEE LOG OUTPUT ABOVE\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
