from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TrainStartRequest(BaseModel):
    # --------------------------------------------------
    # REQUIRED: Trainer routing
    # --------------------------------------------------
    trainer: str = Field(
        ...,
        description="Training backend selector: 'sft_fsdp' or 'fsdp'"
    )

    enable_lora: bool = Field(
        False,
        description="Enable LoRA adapters (used by SFT FSDP trainer)"
    )

    job_name: str = Field(..., description="Friendly name for the training job")
    base_model_path: str = Field(..., description="Local path to HF model (offline-safe)")
    output_dir: str = Field(..., description="Where checkpoints/logs will be written")

    train_data_path: str = Field(..., description="Local path to training data file")
    eval_data_path: Optional[str] = Field(None, description="Optional local path to eval data file")

    # Training hyperparams
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    mixed_precision: str = Field("bf16", description="bf16|fp16|no")
    seed: int = 42

    # FSDP configuration
    fsdp_sharding_strategy: str = Field(
        "FULL_SHARD",
        description="FULL_SHARD|SHARD_GRAD_OP|NO_SHARD (maps to torch.distributed.fsdp.ShardingStrategy)",
    )
    fsdp_auto_wrap_policy: str = Field("transformer", description="transformer|size_based")
    fsdp_min_num_params: int = Field(100_000_000, description="Used for size_based auto wrap policy")

    # Resume
    resume_from_checkpoint: Optional[str] = Field(
        None,
        description="Path to a checkpoint directory containing pytorch_model_fsdp.pt (+ optimizer.pt optional)",
    )
    resume_strict: bool = Field(True, description="Strict load for state_dict when resuming")

    # Multi-node / torchrun rendezvous
    nproc_per_node: Optional[int] = Field(None, description="Default: all visible GPUs")
    nnodes: int = Field(1, description="Number of nodes in the job (default 1)")
    node_rank: int = Field(0, description="Rank of this node [0..nnodes-1]")
    rdzv_backend: str = Field("c10d", description="torchrun rendezvous backend, usually c10d")
    rdzv_endpoint: Optional[str] = Field(
        None,
        description="Rendezvous endpoint host:port (required if nnodes>1)",
    )
    rdzv_id: Optional[str] = Field(
        None,
        description="Rendezvous id; if omitted a job-scoped id will be generated",
    )

    # Metrics (optional)
    enable_mlflow: bool = Field(False, description="Enable MLflow logging from rank 0")
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI")
    mlflow_experiment: Optional[str] = Field(None, description="MLflow experiment name")
    mlflow_run_name: Optional[str] = Field(None, description="MLflow run name (defaults to job_name)")

    # Extra
    extra_env: Dict[str, str] = Field(default_factory=dict)
    extra_args: List[str] = Field(default_factory=list, description="Extra args forwarded to worker module")

    training_policy: Optional[Dict[str, Any]] = None

class TrainStartResponse(BaseModel):
    job_id: str
    status: str
    message: str
    launch_cmd: List[str]


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    return_code: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    last_lines: List[str] = []
    error: Optional[str] = None


class TrainCancelResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainListResponse(BaseModel):
    jobs: List[TrainStatusResponse]
