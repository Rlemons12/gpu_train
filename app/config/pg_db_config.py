# app/configuration/pg_db_config.py

from __future__ import annotations

from app.config.env_adapter import load_global_env_for_wsl

# 🔑 Ensure env is loaded BEFORE reading os.environ
load_global_env_for_wsl()

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# env already loaded by env_adapter
POSTGRES_TRAIN_DB = os.getenv("POSTGRES_TRAIN_DB")
POSTGRES_TRAIN_USER = os.getenv("POSTGRES_TRAIN_USER")
POSTGRES_TRAIN_PASSWORD = os.getenv("POSTGRES_TRAIN_PASSWORD")
POSTGRES_TRAIN_HOST = os.getenv("POSTGRES_TRAIN_HOST", "127.0.0.1")
POSTGRES_TRAIN_PORT = os.getenv("POSTGRES_TRAIN_PORT", "5432")

if not all([POSTGRES_TRAIN_DB, POSTGRES_TRAIN_USER, POSTGRES_TRAIN_PASSWORD]):
    raise RuntimeError("Missing training DB credentials")

TRAIN_DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_TRAIN_USER}:{POSTGRES_TRAIN_PASSWORD}"
    f"@{POSTGRES_TRAIN_HOST}:{POSTGRES_TRAIN_PORT}/{POSTGRES_TRAIN_DB}"
)

# --------------------------------------------------
# Engine (shared)
# --------------------------------------------------
engine = create_engine(TRAIN_DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# --------------------------------------------------
# Schema helpers
# --------------------------------------------------
def get_session(schema: str | None = None):
    """
    Get a session with optional schema binding.
    """
    db = SessionLocal()
    if schema:
        db.execute(text(f"SET search_path TO {schema}"))
    return db


def get_qna_session():
    return get_session(schema="public")


def get_governance_session():
    return get_session(schema="ml_governance")
