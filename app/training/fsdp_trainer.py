from __future__ import annotations

import os
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import functools
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error, gpu_snapshot


@dataclass
class FSdpTrainConfig:
    job_name: str
    base_model_path: str
    output_dir: str
    train_data_path: str
    eval_data_path: Optional[str] = None

    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    mixed_precision: str = "bf16"  # bf16|fp16|no
    seed: int = 42

    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_auto_wrap_policy: str = "transformer"
    fsdp_min_num_params: int = 100_000_000

    resume_from_checkpoint: Optional[str] = None
    resume_strict: bool = True

    enable_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    mlflow_run_name: Optional[str] = None


class JsonlTextDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Dataset not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        r = self.rows[idx]
        if "text" in r:
            return str(r["text"])
        prompt = str(r.get("prompt", ""))
        response = str(r.get("response", ""))
        return (prompt + "\n" + response).strip()


def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_main() -> bool:
    return _rank() == 0


def _parse_sharding_strategy(s: str) -> ShardingStrategy:
    s = (s or "").upper().strip()
    if s == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    if s == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    if s == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    raise ValueError(f"Unknown sharding strategy: {s}")


def _build_mixed_precision(mode: str) -> Optional[MixedPrecision]:
    mode = (mode or "").lower().strip()
    if mode == "no":
        return None
    if mode == "bf16":
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("mixed_precision must be bf16|fp16|no")

    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _build_auto_wrap_policy(model, cfg: FSdpTrainConfig):
    """
    Return an auto_wrap_policy callable for FSDP.

    IMPORTANT:
    - wrap policies must be returned as callables
    - they are invoked internally by FSDP with:
        policy(module, recurse, nonwrapped_numel)
    """

    # If single process / single GPU, skip auto-wrapping entirely
    # (FSDP will effectively behave like NO_SHARD)
    try:
        if dist.get_world_size() <= 1:
            return None
    except Exception:
        return None

    ap = (cfg.fsdp_auto_wrap_policy or "").lower().strip()

    # --------------------------------------
    # Size-based policy
    # --------------------------------------
    if ap == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(cfg.fsdp_min_num_params),
        )

    # --------------------------------------
    # Transformer-based policy (LLaMA / Mistral)
    # --------------------------------------
    try:
        import transformers
        layer_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer

        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
    except Exception:
        # Fallback to size-based
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(cfg.fsdp_min_num_params),
        )


def _maybe_init_mlflow(cfg: FSdpTrainConfig):
    if not cfg.enable_mlflow or not _is_main():
        return None
    try:
        import mlflow
    except Exception:
        gpu_warning("[MLFLOW] enable_mlflow=True but mlflow not installed; skipping MLflow logging")
        return None

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    if cfg.mlflow_experiment:
        mlflow.set_experiment(cfg.mlflow_experiment)

    run_name = cfg.mlflow_run_name or cfg.job_name
    run = mlflow.start_run(run_name=run_name)

    # Log configuration as params (safe subset)
    mlflow.log_params(
        {
            "job_name": cfg.job_name,
            "base_model_path": cfg.base_model_path,
            "train_data_path": cfg.train_data_path,
            "num_train_epochs": cfg.num_train_epochs,
            "batch_size": cfg.per_device_train_batch_size,
            "grad_accum": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "max_seq_length": cfg.max_seq_length,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "mixed_precision": cfg.mixed_precision,
            "fsdp_sharding_strategy": cfg.fsdp_sharding_strategy,
            "fsdp_auto_wrap_policy": cfg.fsdp_auto_wrap_policy,
            "resume_from_checkpoint": cfg.resume_from_checkpoint or "",
        }
    )
    return run


def _mlflow_log_metric(step: int, key: str, value: float):
    try:
        import mlflow
        mlflow.log_metric(key, float(value), step=step)
    except Exception:
        pass


def _write_metrics_jsonl(metrics_path: Path, rec: Dict[str, Any]):
    try:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _load_trainer_state(ckpt_dir: Path) -> Dict[str, Any]:
    p = ckpt_dir / "trainer_state.json"
    if not p.exists():
        return {"global_step": 0, "epoch": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"global_step": 0, "epoch": 0}


def _save_trainer_state(ckpt_dir: Path, state: Dict[str, Any]):
    p = ckpt_dir / "trainer_state.json"
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_fsdp_training(cfg: FSdpTrainConfig):
    _seed_everything(cfg.seed)
    _init_dist()

    rank = _rank()
    world = _world()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics_path = out_root / "metrics.jsonl"

    if _is_main():
        gpu_info(
            f"[FSDP] Starting | job={cfg.job_name} world={world} model={cfg.base_model_path} out={cfg.output_dir}"
        )

    mlflow_run = _maybe_init_mlflow(cfg)

    # Tokenizer/model load
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_snapshot("[FSDP] before-model-load")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else None,
        trust_remote_code=True,
    )
    model.train()

    # Wrap with FSDP
    sharding = _parse_sharding_strategy(cfg.fsdp_sharding_strategy)
    mp = _build_mixed_precision(cfg.mixed_precision)
    auto_wrap = _build_auto_wrap_policy(model, cfg)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding,
        mixed_precision=mp,
        device_id=device if device.type == "cuda" else None,
    )

    gpu_snapshot("[FSDP] after-fsdp-wrap")

    # Data
    train_ds = JsonlTextDataset(cfg.train_data_path)

    def collate(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(cfg.max_seq_length),
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return {k: v.to(device) for k, v in enc.items()}

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.per_device_train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    total_steps = int(math.ceil(len(train_loader) * float(cfg.num_train_epochs)))
    warmup_steps = int(total_steps * float(cfg.warmup_ratio))

    def lr_for_step(step: int) -> float:
        if warmup_steps <= 0:
            return float(cfg.learning_rate)
        if step < warmup_steps:
            return float(cfg.learning_rate) * (step / max(1, warmup_steps))
        return float(cfg.learning_rate)

    # Resume
    global_step = 0
    start_epoch = 0

    if cfg.resume_from_checkpoint:
        ckpt_dir = Path(cfg.resume_from_checkpoint)
        model_path = ckpt_dir / "pytorch_model_fsdp.pt"
        opt_path = ckpt_dir / "optimizer.pt"

        if _is_main():
            gpu_info(f"[RESUME] Loading checkpoint from {ckpt_dir}")

        # Load trainer state (all ranks)
        state = _load_trainer_state(ckpt_dir)
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", 0))

        # Load model state dict (all ranks)
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd, strict=bool(cfg.resume_strict))

        # Load optimizer state if present
        if opt_path.exists():
            try:
                opt_sd = torch.load(opt_path, map_location="cpu")
                opt.load_state_dict(opt_sd)
            except Exception as e:
                if _is_main():
                    gpu_warning(f"[RESUME] Failed to load optimizer state: {e}")

        dist.barrier()

        if _is_main():
            gpu_info(f"[RESUME] Done | global_step={global_step} start_epoch={start_epoch}")

    grad_accum = max(1, int(cfg.gradient_accumulation_steps))

    # Train
    for epoch in range(start_epoch, int(math.ceil(float(cfg.num_train_epochs)))):
        if _is_main():
            gpu_info(f"[FSDP] Epoch {epoch + 1} starting")

        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                global_step += 1

                lr = lr_for_step(global_step)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                opt.step()
                opt.zero_grad(set_to_none=True)

                if _is_main() and global_step % 10 == 0:
                    loss_val = float(loss.item())
                    gpu_info(f"[FSDP] step={global_step}/{total_steps} loss={loss_val:.4f} lr={lr:.2e}")

                    _write_metrics_jsonl(
                        metrics_path,
                        {
                            "ts": time.time(),
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": loss_val,
                            "lr": lr,
                        },
                    )

                    if cfg.enable_mlflow:
                        _mlflow_log_metric(global_step, "loss", loss_val)
                        _mlflow_log_metric(global_step, "lr", lr)

        # Save checkpoint (rank 0 only)
        if _is_main():
            ckpt_dir = out_root / f"checkpoint-epoch-{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            gpu_info(f"[FSDP] Saving checkpoint to {ckpt_dir}")

            sd = model.state_dict()
            torch.save(sd, ckpt_dir / "pytorch_model_fsdp.pt")
