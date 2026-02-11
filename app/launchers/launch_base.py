"""
Shared launcher utilities.

This file contains:
- Payload validation
- Optional MLflow wrapping
- Dry-run support
- HTTP POST to the GPU training service

This file should rarely need modification.
"""

import requests
import mlflow
import argparse
import json
from typing import Dict, Optional, Callable

# FastAPI GPU training service endpoint
TRAIN_URL = "http://localhost:8001/train/start"

def validate_payload(payload: Dict):
    """
    Validate that the minimum required keys exist before
    touching the GPU or starting a job.

    Supports BOTH:
      • Direct train_data_path
      • Dataset-based resolution via dataset_name + dataset_version

    This function runs AFTER pre_launch_hook.
    """

    # --------------------------------------------------
    # Always-required keys (never optional)
    # --------------------------------------------------
    required = [
        "job_name",        # Human-readable identifier
        "base_model_path", # Pretrained model path
        "output_dir",      # Checkpoint / artifact output
    ]

    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # --------------------------------------------------
    # Dataset / data source rules
    # --------------------------------------------------
    has_direct_path = bool(payload.get("train_data_path"))
    has_dataset_ref = bool(
        payload.get("dataset_name") and payload.get("dataset_version")
    )

    if not has_direct_path and not has_dataset_ref:
        raise ValueError(
            "Training data not specified. "
            "Provide either 'train_data_path' OR "
            "both 'dataset_name' and 'dataset_version'."
        )

    # --------------------------------------------------
    # Post-resolution enforcement
    # --------------------------------------------------
    # At this point, pre_launch_hook MUST have resolved
    # train_data_path if dataset_name/version were used.
    if not payload.get("train_data_path"):
        raise RuntimeError(
            "train_data_path was not resolved after pre-launch processing.\n"
            "This indicates a dataset resolution failure or a missing "
            "pre_launch_hook."
        )

def launch_training(
    payload: Dict,
    *,
    enable_mlflow: bool = False,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    dry_run: bool = False,
    pre_launch_hook: Optional[Callable[[Dict], None]] = None,
):
    """
    Launch a training job.

    Parameters:
    - payload: full training configuration dictionary
    - enable_mlflow: wrap the launch in an MLflow run
    - experiment: MLflow experiment name
    - run_name: MLflow run display name
    - dry_run: validate + print payload without launching
    - pre_launch_hook: optional callable(payload) run ONCE before validation
    """

    # --------------------------------------------------
    # Pre-launch hook (run ONCE)
    # --------------------------------------------------
    if pre_launch_hook:
        pre_launch_hook(payload)

    # --------------------------------------------------
    # Validate resolved payload
    # --------------------------------------------------
    validate_payload(payload)

    # --------------------------------------------------
    # HARD REQUIRE trainer for worker compatibility
    # --------------------------------------------------
    if "trainer" not in payload:
        raise ValueError(
            "launch_training requires payload['trainer'] "
            "(expected 'sft_fsdp' or 'fsdp')"
        )

    # --------------------------------------------------
    # Dry run mode
    # --------------------------------------------------
    if dry_run:
        print("[DRY-RUN] Payload validated successfully")
        print(json.dumps(payload, indent=2))
        return None

    # --------------------------------------------------
    # MLflow-wrapped launch
    # --------------------------------------------------
    if enable_mlflow:
        if not experiment:
            raise ValueError("experiment must be set when enable_mlflow=True")

        mlflow.set_experiment(experiment)

        with mlflow.start_run(
            run_name=run_name or payload.get("job_name")
        ):
            mlflow.log_params(payload)

            resp = requests.post(TRAIN_URL, json=payload)
            resp.raise_for_status()

            job = resp.json()
            mlflow.set_tag("job_id", job["job_id"])
            print("Launched job:", job)
            return job

    # --------------------------------------------------
    # Non-MLflow launch path
    # --------------------------------------------------
    resp = requests.post(TRAIN_URL, json=payload)
    resp.raise_for_status()

    job = resp.json()
    print("Launched job:", job)
    return job

