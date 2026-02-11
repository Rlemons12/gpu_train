# dataset_gen/dataset_qanda_generator/configuration/configuration.py

from __future__ import annotations

import os
from pathlib import Path


class Config:
    """
    Runtime configuration object.

    IMPORTANT:
    - This class assumes ALL environment variables are already loaded
      and adapted (Windows → WSL) BEFORE instantiation.
    - Singleton creation is handled in configuration/__init__.py
    """

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # BASE ROOT DIR (repo-local, movable)
        # ---------------------------------------------------------
        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]

        # ---------------------------------------------------------
        # ENV METADATA (informational only)
        # ---------------------------------------------------------
        self.ENV_PATH = os.getenv("DOTENV_PATH")

        # ---------------------------------------------------------
        # MODEL DIRECTORIES (from EMTAC env)
        # ---------------------------------------------------------
        self.MODELS_LLM_DIR = self._env_path("MODELS_LLM_DIR")
        self.MODELS_QWEN_DIR = self._env_path("MODELS_QWEN_DIR")
        self.MODELS_TINY_LLAMA_DIR = self._env_path("MODELS_TINY_LLAMA_DIR")
        self.MODELS_APPLE_ELM_DIR = self._env_path("MODELS_APPLE_ELM_DIR")
        self.MODELS_GEMMA_DIR = self._env_path("MODELS_GEMMA_DIR")
        self.MODELS_FLAN_DIR = self._env_path("MODEL_FLAN_DIR")
        self.MODEL_MINILM_DIR = self._env_path("MODEL_MINILM_DIR")
        self.MODEL_mrm8488_DIR = self._env_path("MODEL_mrm8488_DIR")
        self.MODELS_MISTRAL_7B_DIR = self._env_path("MODELS_MISTRAL_7B_DIR")

        # ---------------------------------------------------------
        # TRAINING DATABASE SETTINGS (NO ENGINE HERE)
        # ---------------------------------------------------------
        self.POSTGRES_TRAIN_DB = os.getenv("POSTGRES_TRAIN_DB")
        self.POSTGRES_TRAIN_USER = os.getenv("POSTGRES_TRAIN_USER")
        self.POSTGRES_TRAIN_PASSWORD = os.getenv("POSTGRES_TRAIN_PASSWORD")
        self.POSTGRES_TRAIN_HOST = os.getenv("POSTGRES_TRAIN_HOST", "127.0.0.1")
        self.POSTGRES_TRAIN_PORT = os.getenv("POSTGRES_TRAIN_PORT", "5432")

        if all(
            [
                self.POSTGRES_TRAIN_DB,
                self.POSTGRES_TRAIN_USER,
                self.POSTGRES_TRAIN_PASSWORD,
            ]
        ):
            self.TRAIN_DATABASE_URL = (
                f"postgresql+psycopg2://{self.POSTGRES_TRAIN_USER}:***"
                f"@{self.POSTGRES_TRAIN_HOST}:{self.POSTGRES_TRAIN_PORT}/"
                f"{self.POSTGRES_TRAIN_DB}"
            )
        else:
            self.TRAIN_DATABASE_URL = None

        # ---------------------------------------------------------
        # PIPELINE OUTPUT DIRECTORIES (repo-local)
        # ---------------------------------------------------------
        qna_base = self.PROJECT_ROOT / "dataset_qanda_generator" / "qna"

        self.STRUCTURE_DIR = qna_base / "structure"
        self.CLEAN_DIR = qna_base / "clean_chunks"
        self.QUESTIONS_DIR = qna_base / "questions"
        self.ANSWERS_DIR = qna_base / "answers"

        # ---------------------------------------------------------
        # LOGS DIR
        # ---------------------------------------------------------
        self.LOG_DIR = self.PROJECT_ROOT / "dataset_qanda_generator" / "logs"

        # ---------------------------------------------------------
        # PROMPTING LAYER (NEW)
        # ---------------------------------------------------------
        self.PROMPTING_DIR = (
            self.PROJECT_ROOT / "dataset_qanda_generator" / "prompting"
        )

        self.PROMPT_TEMPLATES_DIR = (
            self.PROJECT_ROOT / "dataset_qanda_generator" / "prompt_templates"
        )

        # Canonical templates
        self.DOCUMENT_METADATA_PROMPT = (
            self.PROMPT_TEMPLATES_DIR / "document_metadata.json"
        )

        # ---------------------------------------------------------
        # DB TOOLS DIRECTORIES
        # ---------------------------------------------------------
        self.DB_TOOLS_DIR = (
            self.PROJECT_ROOT / "dataset_qanda_generator" / "tools" / "db_tools"
        )
        self.DB_TOOLS_OUTPUT_DIR = self.DB_TOOLS_DIR / "output"

        # ---------------------------------------------------------
        # Create required directories
        # ---------------------------------------------------------
        for d in [
            self.STRUCTURE_DIR,
            self.CLEAN_DIR,
            self.QUESTIONS_DIR,
            self.ANSWERS_DIR,
            self.LOG_DIR,
            self.PROMPTING_DIR,
            self.PROMPT_TEMPLATES_DIR,
            self.DB_TOOLS_DIR,
            self.DB_TOOLS_OUTPUT_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # Validate required prompt templates
        # ---------------------------------------------------------
        if not self.DOCUMENT_METADATA_PROMPT.exists():
            raise FileNotFoundError(
                f"Required prompt template missing: {self.DOCUMENT_METADATA_PROMPT}"
            )

    # ---------------------------------------------------------
    # QA GENERATION PROMPTS
    # ---------------------------------------------------------
        self.QA_GENERATION_FLAN_PROMPT = (
                self.PROMPT_TEMPLATES_DIR
                / "qa_generation_flan"
                / "qa_generation_flan.txt"
        )

    # ---------------------------------------------------------
    # ANSWER PROMPT (NEW)
    # ---------------------------------------------------------
        self.ANSWER_GENERATION_PROMPT = (
                self.PROMPT_TEMPLATES_DIR / "answer_generation.txt"
        )

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _env_path(self, var_name: str) -> Path | None:
        val = os.getenv(var_name)
        if not val:
            return None
        return Path(val)
