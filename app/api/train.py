from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from fastapi import APIRouter, HTTPException

from app.schemas.train import (
    TrainStartRequest,
    TrainStartResponse,
    TrainStatusResponse,
)
from app.training.job_manager import TRAINING_JOBS
from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error, get_request_id


router = APIRouter(prefix="/train", tags=["training"])


def _project_root() -> Path:
    # app/api/train.py -> app/api -> app -> PROJECT_ROOT
    return Path(__file__).resolve().parents[2]


def _torchrun_cmd_module(nproc: int, module: str) -> List[str]:
    # Prefer torchrun; run worker as a python module for consistent imports
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "-m",
        module,
    ]


@router.post("/start", response_model=TrainStartResponse)
def start_train(req: TrainStartRequest) -> TrainStartResponse:
    rid = get_request_id()

    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available on this machine.")

    gpu_count = torch.cuda.device_count()
    nproc = int(req.nproc_per_node) if req.nproc_per_node else gpu_count
    if nproc < 1 or nproc > gpu_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid nproc_per_node={nproc}. GPU count={gpu_count}.",
        )

    if not Path(req.base_model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {req.base_model_path}")
    if not Path(req.train_data_path).exists():
        raise HTTPException(status_code=400, detail=f"Train data not found: {req.train_data_path}")

    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "train_job.log"
    worker_module = "app.training.run_fsdp_worker"

    cfg_dict: Dict[str, Any] = req.model_dump()
    cfg_json = json.dumps(cfg_dict)

    env = dict(os.environ)
    env["FSDP_TRAIN_CONFIG_JSON"] = cfg_json

    # -------------------------------------------------
    # CRITICAL: ensure child processes can import "app"
    # -------------------------------------------------
    project_root = _project_root()

    existing_pp = env.get("PYTHONPATH", "")
    root_str = str(project_root)
    env["PYTHONPATH"] = root_str + (os.pathsep + existing_pp if existing_pp else "")

    # -------------------------------------------------
    # MLflow: force a single global tracking location
    # -------------------------------------------------
    mlflow_root = project_root / "mlruns"
    mlflow_root.mkdir(parents=True, exist_ok=True)

    env["MLFLOW_TRACKING_URI"] = f"file:{mlflow_root}"
    env["MLFLOW_EXPERIMENT_NAME"] = "gpu_train"

    # Merge extra env
    for k, v in (req.extra_env or {}).items():
        env[str(k)] = str(v)

    cmd = _torchrun_cmd_module(nproc=nproc, module=worker_module)
    cmd.extend(req.extra_args or [])

    gpu_info(
        f"[API] /train/start | job_name={req.job_name} nproc={nproc} project_root={project_root}",
        rid,
    )

    log_file.write_text("")

    rec = TRAINING_JOBS.start_job(
        job_name=req.job_name,
        cmd=cmd,
        env=env,
        output_dir=str(out_dir),
        log_file=str(log_file),
    )

    return TrainStartResponse(
        job_id=rec.job_id,
        status=rec.status,
        message="Training job queued",
        launch_cmd=cmd,
    )


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str, tail_lines: int = 80) -> TrainStatusResponse:
    if job_id not in {j.job_id for j in TRAINING_JOBS.list_jobs()} and job_id not in TRAINING_JOBS._jobs:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    rec = TRAINING_JOBS.status(job_id)
    lines = TRAINING_JOBS.tail_log(rec.log_file, lines=tail_lines)

    def _fmt(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        import datetime
        return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    return TrainStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        return_code=rec.return_code,
        started_at=_fmt(rec.started_at),
        finished_at=_fmt(rec.finished_at),
        output_dir=rec.output_dir,
        log_file=rec.log_file,
        last_lines=lines,
        error=rec.error,
    )
