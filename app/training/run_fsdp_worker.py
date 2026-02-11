from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import torch

from app.config.gpu_logger import gpu_info, gpu_warning


# ============================================================
# Rank helpers (torchrun safe)
# ============================================================
def _get_rank_env() -> tuple[int, int, int]:
    """
    Returns (rank, local_rank, world_size) from torchrun environment.
    Defaults are safe for single-process runs.
    """
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _is_main(rank: int) -> bool:
    return rank == 0


# ============================================================
# 🔧 Ensure repo root on sys.path (torchrun safe)
# ============================================================
def _ensure_repo_root_on_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../gpu_train

    if not (repo_root / "app").exists():
        raise RuntimeError(
            f"Repo root detection failed — 'app/' not found under {repo_root}"
        )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return repo_root


# ============================================================
# 🧹 Filter incoming configuration strictly to dataclass fields
# ============================================================
def _filter_to_dataclass_fields(
    raw: Dict[str, Any],
    dc_type,
) -> tuple[Dict[str, Any], list[str]]:
    allowed = {f.name for f in fields(dc_type)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    dropped = sorted(k for k in raw.keys() if k not in allowed)
    return filtered, dropped


# ============================================================
# 🧠 MLflow (SERVICE-LEVEL, MODEL-AGNOSTIC)
# ============================================================
def _start_mlflow_run(raw_cfg: Dict[str, Any], world_size: int) -> None:
    """
    Start an MLflow run (rank-0 only).
    """
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "The GPU service must inject it before torchrun starts."
        )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "gpu_train")
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run() is not None:
        gpu_warning("[MLFLOW] Active run already exists; skipping start_run()")
        return

    run_name = str(raw_cfg.get("job_name") or "unnamed-job")

    tags = {
        "service": "gpu_train",
        "world_size": str(world_size),
        "trainer": str(raw_cfg.get("trainer")),
        "job_name": run_name,
    }

    mlflow.start_run(run_name=run_name, tags=tags)

    params: Dict[str, str | int | float] = {}
    for k, v in raw_cfg.items():
        if isinstance(v, (int, float, str)):
            params[k] = v
        elif v is not None:
            params[k] = str(v)

    if params:
        mlflow.log_params(params)

    gpu_info(
        f"[MLFLOW] Run started | uri={tracking_uri} "
        f"exp={experiment_name} name={run_name}"
    )


def _end_mlflow_run(status: str) -> None:
    import mlflow

    if mlflow.active_run() is None:
        gpu_warning("[MLFLOW] end_run() called but no active run exists")
        return

    mlflow.end_run(status=status)
    gpu_info(f"[MLFLOW] Run ended | status={status}")


# ============================================================
# 🔥 CUDA device resolution (multi-GPU safe)
# ============================================================
def _resolve_cuda_device(local_rank: int, world_size: int) -> torch.device:
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError(
                f"world_size={world_size} but CUDA is not available"
            )
        return torch.device("cpu")

    gpu_count = torch.cuda.device_count()

    if world_size > gpu_count:
        raise RuntimeError(
            f"world_size={world_size} but only {gpu_count} CUDA device(s) available"
        )

    if local_rank < 0 or local_rank >= gpu_count:
        raise RuntimeError(
            f"Invalid local_rank={local_rank}, gpu_count={gpu_count}"
        )

    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


# ============================================================
# Main torchrun entrypoint
# ============================================================
def main():
    repo_root = _ensure_repo_root_on_path()
    gpu_info(f"[FSDP-WORKER] repo_root={repo_root}")

    rank, local_rank, world_size = _get_rank_env()

    gpu_info(
        f"[FSDP-WORKER] rank={rank} local_rank={local_rank} "
        f"world_size={world_size} cuda_available={torch.cuda.is_available()}"
    )

    device = _resolve_cuda_device(local_rank, world_size)
    gpu_info(f"[FSDP-WORKER] bound_device={device}")

    cfg_json = os.environ.get("FSDP_TRAIN_CONFIG_JSON")
    if not cfg_json:
        raise RuntimeError("Missing env var: FSDP_TRAIN_CONFIG_JSON")

    raw_cfg = json.loads(cfg_json)
    gpu_info(f"[DEBUG] raw_cfg keys = {sorted(raw_cfg.keys())}")

    trainer = raw_cfg.get("trainer")

    # --------------------------------------------------
    # ENFORCE EXPLICIT TRAINER INTENT
    # --------------------------------------------------
    if trainer is None:
        raise RuntimeError(
            "Missing 'trainer' in configuration. "
            "You MUST specify one of: 'sft_fsdp' or 'fsdp'."
        )

    if trainer not in {"sft_fsdp", "fsdp"}:
        raise RuntimeError(
            f"Invalid trainer='{trainer}'. "
            "Allowed values: 'sft_fsdp', 'fsdp'."
        )

    job_name = str(raw_cfg.get("job_name", ""))
    if "lora" in job_name.lower() and trainer != "sft_fsdp":
        raise RuntimeError(
            "Job name implies LoRA but trainer!='sft_fsdp'. "
            "Set trainer='sft_fsdp' explicitly."
        )

    mlflow_started = False
    status = "FINISHED"

    try:
        if _is_main(rank):
            _start_mlflow_run(raw_cfg, world_size)
            mlflow_started = True

        # --------------------------------------------------
        # TRAINER DISPATCH
        # --------------------------------------------------
        if trainer == "sft_fsdp":
            gpu_info("[FSDP-WORKER] Selected SFT FSDP trainer")

            from app.training.sft_fsdp_trainer import (
                SFTFSDPTrainConfig,
                run_sft_fsdp_training,
            )

            # Filter only dataclass fields
            cfg_dict, dropped = _filter_to_dataclass_fields(raw_cfg, SFTFSDPTrainConfig)

            # Orchestration-only metadata (never injected into cfg_dict)
            training_policy = raw_cfg.get("training_policy") or {}

            # --------------------------------------------------
            # 🔐 DATASET GOVERNANCE RESOLUTION (ROBUST)
            # --------------------------------------------------
            dataset_hash = raw_cfg.get("dataset_hash") or training_policy.get("dataset_hash")

            if not dataset_hash:
                train_path = raw_cfg.get("train_data_path")
                if not train_path:
                    raise RuntimeError(
                        "[FSDP-WORKER] dataset_hash missing and train_data_path missing; "
                        "cannot resolve dataset identity."
                    )

                import hashlib

                h = hashlib.sha256()
                with open(train_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                dataset_hash = h.hexdigest()

                gpu_warning(
                    "[FSDP-WORKER] dataset_hash was missing; computed from train_data_path "
                    "(dev fallback)."
                )

            # Ensure BOTH cfg and policy contain governance identity
            training_policy["dataset_hash"] = dataset_hash
            raw_cfg["dataset_hash"] = dataset_hash
            raw_cfg["training_policy"] = training_policy

            # Inject into dataclass configuration (SFTFSDPTrainConfig has dataset_hash field)
            cfg_dict["dataset_hash"] = dataset_hash

            # Optionally pass through dataset_version if present in policy/raw
            dataset_version = raw_cfg.get("dataset_version") or training_policy.get("dataset_version")
            if dataset_version:
                cfg_dict["dataset_version"] = dataset_version
                training_policy["dataset_version"] = dataset_version

            # Ensure output dir exists
            if cfg_dict.get("output_dir"):
                Path(cfg_dict["output_dir"]).mkdir(parents=True, exist_ok=True)

            gpu_info(f"[FSDP-WORKER] dropped_keys={dropped}")

            # Construct dataclass CLEANLY
            cfg = SFTFSDPTrainConfig(**cfg_dict)

            # Call worker-facing entrypoint (NOTE: no dataset_hash kwarg)
            run_sft_fsdp_training(
                cfg,
                training_policy=training_policy,
            )

        else:
            gpu_info("[FSDP-WORKER] Selected generic FSDP trainer")

            from app.training.fsdp_trainer import (
                FSdpTrainConfig,
                run_fsdp_training,
            )

            cfg_dict, dropped = _filter_to_dataclass_fields(raw_cfg, FSdpTrainConfig)

            if cfg_dict.get("output_dir"):
                Path(cfg_dict["output_dir"]).mkdir(parents=True, exist_ok=True)

            gpu_info(f"[FSDP-WORKER] dropped_keys={dropped}")

            cfg = FSdpTrainConfig(**cfg_dict)
            run_fsdp_training(cfg)

    except Exception:
        status = "FAILED"
        raise

    finally:
        if _is_main(rank) and mlflow_started:
            _end_mlflow_run(status)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
