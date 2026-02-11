# DATASET_GEN/dataset_qanda_generator/configuration/pg_db_config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------
# Load local gpu_train/.env FIRST (for DOTENV_PATH)
# ---------------------------------------------------------
load_dotenv(override=False)


def resolve_env_path() -> Path:
    """
    Resolve EMTAC environment file.
    """
    override = os.getenv("OPTIONC_ENV_PATH")
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"OPTIONC_ENV_PATH not found: {p}")
        return p

    dotenv_path = os.getenv("DOTENV_PATH")
    if not dotenv_path:
        raise RuntimeError("DOTENV_PATH not set in gpu_train/.env")

    p = Path(dotenv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"DOTENV_PATH not found: {p}")

    return p


ENV_PATH = resolve_env_path()
load_dotenv(ENV_PATH, override=True)


# =========================================================
# 🟩 TRAINING DATABASE (REQUIRED)
# =========================================================
POSTGRES_TRAIN_DB = os.getenv("POSTGRES_TRAIN_DB")
POSTGRES_TRAIN_USER = os.getenv("POSTGRES_TRAIN_USER")
POSTGRES_TRAIN_PASSWORD = os.getenv("POSTGRES_TRAIN_PASSWORD")
POSTGRES_TRAIN_HOST = os.getenv("POSTGRES_TRAIN_HOST", "127.0.0.1")
POSTGRES_TRAIN_PORT = os.getenv("POSTGRES_TRAIN_PORT", "5432")

missing = [
    k for k, v in {
        "POSTGRES_TRAIN_DB": POSTGRES_TRAIN_DB,
        "POSTGRES_TRAIN_USER": POSTGRES_TRAIN_USER,
        "POSTGRES_TRAIN_PASSWORD": POSTGRES_TRAIN_PASSWORD,
    }.items() if not v
]

if missing:
    raise RuntimeError(f"Missing training DB env vars: {missing}")

TRAIN_DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_TRAIN_USER}:{POSTGRES_TRAIN_PASSWORD}"
    f"@{POSTGRES_TRAIN_HOST}:{POSTGRES_TRAIN_PORT}/{POSTGRES_TRAIN_DB}"
)

train_engine = create_engine(TRAIN_DATABASE_URL, pool_pre_ping=True)
TrainSessionLocal = sessionmaker(bind=train_engine, autoflush=False, autocommit=False)

# Unified access for Q&A pipeline
QNA_ENGINE = train_engine
QNA_SessionLocal = TrainSessionLocal


# =========================================================
# 🟦 EMTAC PRODUCTION DATABASE (OPTIONAL)
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

emtac_engine = None
EMTACSessionLocal = None

if DATABASE_URL:
    emtac_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    EMTACSessionLocal = sessionmaker(bind=emtac_engine, autoflush=False, autocommit=False)


# =========================================================
# Helpers
# =========================================================
def get_qna_session():
    return QNA_SessionLocal()


def get_training_session():
    return QNA_SessionLocal()


def get_emtac_session():
    if EMTACSessionLocal is None:
        raise RuntimeError("EMTAC DB not configured")
    return EMTACSessionLocal()
