"""
SFT + FSDP launcher with MLflow tracking enabled.
(Governed dataset selection, smoke-test friendly)
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path

from app.config.pg_db_config import get_governance_session
from app.launchers.launch_base import launch_training
from app.ml_governance.registry import DatasetRegistry
from app.ml_governance.health import check_governance_db


# ---------------------------------------------------------------------
# GOVERNANCE PRE-LAUNCH HOOK
# ---------------------------------------------------------------------
from pathlib import Path
from app.config.pg_db_config import get_governance_session
from app.ml_governance.registry import DatasetRegistry
from dotenv import load_dotenv

load_dotenv()

if os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING") != "true":
    print("[WARN] MLflow system metrics logging is NOT enabled")

if not os.getenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"):
    print("[WARN] MLflow system metrics sampling interval is NOT set")

print("[ENV] MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING =",
    os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"),)

print("[ENV] MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL =",
    os.getenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"),)

def resolve_dataset_pre_launch(payload: dict) -> None:
    dataset_name = payload.get("dataset_name")
    dataset_version = payload.get("dataset_version")

    if not dataset_name or not dataset_version:
        raise RuntimeError("dataset_name and dataset_version are required")

    with get_governance_session() as db:
        registry = DatasetRegistry(db, actor="launcher")
        dataset = registry.get_dataset(
            name=dataset_name,
            version=dataset_version,
        )

    train_path = Path(dataset.path)
    if not train_path.exists():
        raise RuntimeError(f"Training file not found: {train_path}")

    # Always set the training file path (this key survives)
    payload["train_data_path"] = str(train_path)

    # Inject governance metadata (TOP-LEVEL may be stripped by API)
    payload["dataset_hash"] = dataset.content_hash
    payload["dataset_version"] = dataset.version

    # ALSO embed in training_policy (this survives API whitelisting)
    tp = payload.setdefault("training_policy", {}) or {}
    tp["dataset_hash"] = dataset.content_hash
    tp["dataset_name"] = dataset_name
    tp["dataset_version"] = dataset.version
    payload["training_policy"] = tp

# ---------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Launch SFT FSDP training job with MLflow"
    )

    # --------------------------------------------------
    # Dataset selection (NEW)
    # --------------------------------------------------
    parser.add_argument(
        "--dataset-name",
        help="Registered dataset name (e.g. qna_training__tags_SPEC)",
    )
    parser.add_argument(
        "--dataset-version",
        help="Dataset version (e.g. v001)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and infrastructure without launching training",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Governance health check
    # --------------------------------------------------
    check_governance_db()

    # --------------------------------------------------
    # Base payload (NO train_data_path yet)
    # --------------------------------------------------
    payload = {
        "trainer": "sft_fsdp",
        "enable_lora": True,

        "job_name": "mistral7b_mini_lora",

        "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
        "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/mistral7b_mini",

        "num_train_epochs": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-6,
        "max_seq_length": 1024,
        "mixed_precision": "bf16",
        "max_steps": 2,

        "training_policy": {
            "register_model": True,
            "registry_name": "emtac_mistral_sft",
        },
    }

    # --------------------------------------------------
    # 🔑 Inject dataset reference into payload
    # --------------------------------------------------
    if args.dataset_name and args.dataset_version:
        payload["dataset_name"] = args.dataset_name
        payload["dataset_version"] = args.dataset_version

    # --------------------------------------------------
    # Launch (dry-run respected)
    # --------------------------------------------------
    launch_training(
        payload,
        enable_mlflow=True,
        experiment="gpu_train",
        run_name=payload["job_name"],
        dry_run=args.dry_run,
        pre_launch_hook=resolve_dataset_pre_launch,
    )

if __name__ == "__main__":
    main()
