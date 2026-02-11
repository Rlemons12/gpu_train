#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Register QNA LLM Models (Variant-Based, Safe)
--------------------------------------------

• Curated model families (code-level contract)
• Discovers concrete model variants on disk
• One DB row per runnable model directory
• Idempotent (safe to re-run)
• DB is the source of truth
"""

from pathlib import Path

from dataset_gen.dataset_qanda_generator.configuration import cfg
from dataset_gen.dataset_qanda_generator.configuration.pg_db_config import get_qna_session
from dataset_gen.dataset_qanda_generator.configuration.logging_config import get_qna_logger
from dataset_gen.dataset_qanda_generator.qanda_db.qa_db import LLMModel

log = get_qna_logger("register_qna_models")

# ----------------------------------------------------------------------
# SUPPORTED MODEL FAMILIES
# ----------------------------------------------------------------------

MODEL_DEFINITIONS = {
    "mistral": {
        "model_type": "causal_lm",
        "class_path": "dataset_gen.dataset_qanda_generator.models.mistral.MistralAnswerGenerator",
        "disk_prefixes": ["mistral-"],
        "default_enabled": True,
    },
    "gemma": {
        "model_type": "causal_lm",
        "class_path": "dataset_gen.dataset_qanda_generator.models.gemma.GemmaAnswerGenerator",
        "disk_prefixes": ["google_gemma-"],
        "default_enabled": True,
    },
    "openelm": {
        "model_type": "causal_lm",
        "class_path": "dataset_gen.dataset_qanda_generator.models.openelm.OpenELMAnswerGenerator",
        "disk_prefixes": ["apple_OpenELM", "openelm_"],
        "default_enabled": True,
    },
    "devstral": {
        "model_type": "causal_lm",
        "class_path": "dataset_gen.dataset_qanda_generator.models.devstral.DevstralAnswerGenerator",
        "disk_prefixes": ["Devstral-"],
        "default_enabled": False,
    },
}

# ----------------------------------------------------------------------
# DISCOVERY
# ----------------------------------------------------------------------

def discover_models_on_disk():
    base_dir = Path(cfg.MODELS_LLM_DIR)
    discovered = []

    if not base_dir.exists():
        log.error(f"Model base directory missing: {base_dir}")
        return discovered

    for family, spec in MODEL_DEFINITIONS.items():
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            if any(subdir.name.startswith(p) for p in spec["disk_prefixes"]):
                discovered.append({
                    "name": subdir.name.lower(),
                    "model_type": spec["model_type"],
                    "model_path": str(subdir),
                    "class_path": spec["class_path"],
                    "enabled": spec["default_enabled"],
                })
                log.info(f"[FOUND] {family}: {subdir.name}")

    return discovered

# ----------------------------------------------------------------------
# REGISTRATION
# ----------------------------------------------------------------------

def register_models():
    session = get_qna_session()
    models = discover_models_on_disk()

    try:
        for spec in models:
            name = spec["name"]

            model = session.query(LLMModel).filter_by(name=name).first()

            if model:
                updated = False
                for field in ("model_type", "model_path", "class_path", "enabled"):
                    if getattr(model, field) != spec[field]:
                        setattr(model, field, spec[field])
                        updated = True

                if updated:
                    log.info(f"[UPDATE] {name}")
                else:
                    log.info(f"[SKIP] {name} unchanged")

            else:
                session.add(LLMModel(**spec))
                log.info(f"[INSERT] {name}")

        session.commit()
        log.info("Model registration complete")

    except Exception:
        session.rollback()
        log.exception("Model registration FAILED")
        raise

    finally:
        session.close()

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    register_models()
