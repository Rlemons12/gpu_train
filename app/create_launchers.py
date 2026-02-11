from pathlib import Path
import textwrap

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ROOT_DIR = Path.cwd()
LAUNCHERS_DIR = ROOT_DIR / "launchers"

# -------------------------------------------------
# TEMPLATE: launch_base.py
# -------------------------------------------------

BASE_TEMPLATE = """
\"\"\"
Shared launcher utilities.

This file contains:
- Payload validation
- Optional MLflow wrapping
- Dry-run support
- HTTP POST to the GPU training service

This file should rarely need modification.
\"\"\"

import requests
import mlflow
import argparse
import json
from typing import Dict, Optional

# FastAPI GPU training service endpoint
TRAIN_URL = "http://localhost:8001/train/start"


def validate_payload(payload: Dict):
    \"\"\"
    Validate that the minimum required keys exist before
    touching the GPU or starting a job.
    \"\"\"
    required = [
        "job_name",          # Human-readable identifier for the training job
        "base_model_path",   # Path to the pretrained model directory
        "train_data_path",   # Path to JSONL training dataset
        "output_dir",        # Where checkpoints/logs will be written
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


def launch_training(
    payload: Dict,
    *,
    enable_mlflow: bool = False,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    dry_run: bool = False,
):
    \"\"\"
    Launch a training job.

    Parameters:
    - payload: full training configuration dictionary
    - enable_mlflow: wrap the launch in an MLflow run
    - experiment: MLflow experiment name
    - run_name: MLflow run display name
    - dry_run: validate + print payload without launching
    \"\"\"
    validate_payload(payload)

    if dry_run:
        print("[DRY-RUN] Payload validated successfully")
        print(json.dumps(payload, indent=2))
        return None

    if enable_mlflow:
        if not experiment:
            raise ValueError("experiment must be set when enable_mlflow=True")

        with mlflow.start_run(
            experiment_name=experiment,
            run_name=run_name or payload.get("job_name"),
        ):
            mlflow.log_params(payload)

            resp = requests.post(TRAIN_URL, json=payload)
            resp.raise_for_status()

            job = resp.json()
            mlflow.set_tag("job_id", job["job_id"])
            print("Launched job:", job)
            return job

    resp = requests.post(TRAIN_URL, json=payload)
    resp.raise_for_status()

    job = resp.json()
    print("Launched job:", job)
    return job
"""

# -------------------------------------------------
# TEMPLATE: launch_sft_fsdp.py
# -------------------------------------------------

SFT_FSDP_TEMPLATE = """
\"\"\"
Standalone launcher for supervised fine-tuning (SFT) using FSDP.

This is a production-equivalent, single-run launcher.
\"\"\"

from launchers.launch_base import launch_training
import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch SFT FSDP training job")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without launching")
    args = parser.parse_args()

    payload = {
        # ------------------------------
        # Identification
        # ------------------------------
        "job_name": "mistral7b_mini_lora",   # Name shown in logs, MLflow, and job status APIs

        # ------------------------------
        # Model & data
        # ------------------------------
        "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
        # Directory containing configuration.json, safetensors, tokenizer, etc.

        "train_data_path": "/mnt/c/Users/operator/PycharmProjects/gpu_train/dataset_output/dataset.jsonl",
        # JSONL file with training samples (instruction / response format)

        "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/mistral7b_mini",
        # Where checkpoints, logs, and trainer state are written

        # ------------------------------
        # Training schedule
        # ------------------------------
        "num_train_epochs": 1.0,            # Number of passes over the dataset
        "per_device_train_batch_size": 1,   # Micro-batch size per GPU
        "gradient_accumulation_steps": 1,   # Steps to accumulate before optimizer step

        # ------------------------------
        # Optimization
        # ------------------------------
        "learning_rate": 2e-6,               # Base learning rate (LoRA-safe default)
        "max_seq_length": 1024,              # Max tokens per sample
        "mixed_precision": "bf16",           # bf16 or fp16 (bf16 preferred on modern GPUs)

        # ------------------------------
        # FSDP configuration
        # ------------------------------
        "fsdp_sharding_strategy": "FULL_SHARD",
        # FULL_SHARD shards parameters, gradients, optimizer state

        "fsdp_auto_wrap_policy": "transformer",
        # Automatically wraps transformer layers for sharding
    }

    launch_training(payload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
"""

# -------------------------------------------------
# TEMPLATE: launch_sft_fsdp_mlflow.py
# -------------------------------------------------

SFT_FSDP_MLFLOW_TEMPLATE = """
\"\"\"
SFT + FSDP launcher with MLflow tracking enabled.
\"\"\"

from launchers.launch_base import launch_training
import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch SFT FSDP training job with MLflow")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without launching")
    args = parser.parse_args()

    payload = {
        "job_name": "mistral7b_mini_lora",

        # Model + dataset
        "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
        "train_data_path": "/mnt/c/Users/operator/PycharmProjects/gpu_train/dataset_output/dataset.jsonl",
        "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/mistral7b_mini",

        # Training
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 2e-6,
        "max_seq_length": 1024,
        "mixed_precision": "bf16",
    }

    launch_training(
        payload,
        enable_mlflow=True,
        experiment="gpu_train",              # MLflow experiment name
        run_name=payload["job_name"],        # MLflow run display name
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
"""

# -------------------------------------------------
# TEMPLATE: launch_resume_last.py
# -------------------------------------------------

RESUME_TEMPLATE = """
\"\"\"
Resume a previous MLflow run using its logged parameters.
\"\"\"

from launchers.launch_base import launch_training
import mlflow
import argparse


def main():
    parser = argparse.ArgumentParser(description="Resume training from an MLflow run")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load parameters from the previous run
    run = mlflow.get_run(args.run_id)
    payload = {k: v for k, v in run.data.params.items()}

    payload["resume"] = True  # Signal trainer to resume from checkpoint

    launch_training(
        payload,
        enable_mlflow=True,
        experiment=run.info.experiment_id,
        run_name=f"resume_{run.info.run_name}",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
"""

# -------------------------------------------------
# TEMPLATE: launch_grid_search.py
# -------------------------------------------------

GRID_TEMPLATE = """
\"\"\"
Simple learning-rate grid search launcher.
\"\"\"

from launchers.launch_base import launch_training

# Values to sweep over
LEARNING_RATES = [2e-6, 5e-6, 1e-5]

BASE_PAYLOAD = {
    "base_model_path": "/mnt/c/Users/operator/emtac/models/llm/mistral-7b-instruct",
    "train_data_path": "/mnt/c/Users/operator/PycharmProjects/gpu_train/dataset_output/dataset.jsonl",
    "output_dir": "/mnt/c/Users/operator/PycharmProjects/gpu_train/runs/grid",

    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "max_seq_length": 1024,
    "mixed_precision": "bf16",
}

for lr in LEARNING_RATES:
    payload = BASE_PAYLOAD | {
        "job_name": f"grid_lr_{lr}",
        "learning_rate": lr,
    }

    launch_training(
        payload,
        enable_mlflow=True,
        experiment="gpu_train",
        run_name=payload["job_name"],
    )
"""

# -------------------------------------------------
# FILE MAP
# -------------------------------------------------

FILES = {
    "__init__.py": "",
    "launch_base.py": BASE_TEMPLATE,
    "launch_sft_fsdp.py": SFT_FSDP_TEMPLATE,
    "launch_sft_fsdp_mlflow.py": SFT_FSDP_MLFLOW_TEMPLATE,
    "launch_resume_last.py": RESUME_TEMPLATE,
    "launch_grid_search.py": GRID_TEMPLATE,
}

# -------------------------------------------------
# SCRIPT
# -------------------------------------------------
def main():
    LAUNCHERS_DIR.mkdir(exist_ok=True)
    print(f"[OK] launchers/ directory ready: {LAUNCHERS_DIR}")

    for name, template in FILES.items():
        path = LAUNCHERS_DIR / name
        if path.exists():
            print(f"[SKIP] {name} already exists")
            continue

        path.write_text(textwrap.dedent(template).strip() + "\\n")
        print(f"[CREATE] {name}")

    print("\\nLauncher system ready (documented).")


if __name__ == "__main__":
    main()
