from __future__ import annotations

import os
import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_snapshot
import hashlib
from app.config.pg_db_config import get_governance_session
from app.ml_governance.registry import DatasetRegistry
from app.training.training_hooks import MLflowGPUTrainingHooks

# Optional orchestration hook object.
# The service can inject something with:
#   on_train_start(cfg), on_epoch_start(...), on_step(...), on_train_end()
hooks = None


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ==================================================
# DATASET: Instruction SFT
# ==================================================
# =========================
# REPLACE InstructionDataset WITH THIS
# =========================
class InstructionDataset(Dataset):
    """
    Supports JSONL rows in either format:
      - {"instruction": "...", "response": "..."}   (legacy)
      - {"prompt": "...", "response": "..."}        (your current file)
      - {"text": "..."}                             (fallback single-field)
    Returns raw text strings; tokenization happens in collate_fn.
    """
    def __init__(self, jsonl_path: str):
        self.rows: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)

                instr = r.get("instruction")
                prompt = r.get("prompt")
                resp = r.get("response")

                if resp is None:
                    # allow single-field datasets but make it explicit
                    text = r.get("text")
                    if text:
                        self.rows.append(str(text))
                        continue
                    raise RuntimeError("Row missing 'response' (and no 'text' fallback).")

                # prefer prompt if present, else instruction
                user_part = prompt if prompt is not None else instr
                if user_part is None:
                    raise RuntimeError("Row missing 'prompt' or 'instruction'.")

                # Simple SFT formatting
                self.rows.append(f"Instruction: {user_part}\nResponse: {resp}")

        if not self.rows:
            raise RuntimeError("Training dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        return self.rows[idx]



# ==================================================
# DATASET: Intent head
#   JSONL rows: {"text": "...", "label": "intent_name"} or {"text": "...", "label_id": 3}
#   label_map.json: {"intent_name": 0, ...} (optional)
# ==================================================
class IntentDataset(Dataset):
    def __init__(self, jsonl_path: str, label_map: Optional[Dict[str, int]] = None):
        self.samples: List[Tuple[str, int]] = []
        self.label_map = label_map or {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)

                text = r.get("text") or r.get("query") or r.get("instruction")
                if not text:
                    raise RuntimeError("IntentDataset row missing 'text' (or 'query'/'instruction').")

                if "label_id" in r and r["label_id"] is not None:
                    y = int(r["label_id"])
                else:
                    label = r.get("label")
                    if label is None:
                        raise RuntimeError("IntentDataset row missing 'label' or 'label_id'.")
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)
                    y = int(self.label_map[label])

                self.samples.append((text, y))

        if not self.samples:
            raise RuntimeError("Intent dataset is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


# ==================================================
# DISTRIBUTED HELPERS
# ==================================================
def _init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_main() -> bool:
    return _rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _destroy_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================================================
# TOKENIZER (Production-safe with OpenELM)
# ==================================================
def _load_tokenizer(cfg) -> Any:
    tokenizer_path = cfg.tokenizer_path or cfg.base_model_path
    tokenizer_impl = (cfg.tokenizer_impl or "auto").lower()

    if tokenizer_impl == "llama":
        tok = LlamaTokenizerFast.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            tok = LlamaTokenizerFast.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=True,
            )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


# ==================================================
# Optional LoRA (PEFT)
# ==================================================
def _force_uniform_param_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """
    FSDP requires flattened params to have uniform dtype.
    PEFT LoRA adapters can default to fp32; normalize everything.
    """
    for _, p in model.named_parameters(recurse=True):
        if p is not None and p.data is not None and p.data.dtype != dtype:
            p.data = p.data.to(dtype)

    for _, b in model.named_buffers(recurse=True):
        if b is not None and b.data is not None and b.data.dtype != dtype:
            b.data = b.data.to(dtype)


def _maybe_apply_lora(model, cfg):
    enable_lora = bool(getattr(cfg, "enable_lora", False))
    if not enable_lora:
        return model, False

    try:
        from peft import LoraConfig, get_peft_model
        from peft.utils.peft_types import TaskType
    except Exception as e:
        raise RuntimeError(
            "LoRA requested (enable_lora=true) but 'peft' is not installed or failed to import.\n"
            "Fix: pip install -U peft"
        ) from e

    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str):
            return [s.strip() for s in x.split(",") if s.strip()]
        return None

    def _leaf(name: str) -> str:
        return name.split(".")[-1]

    linear_leafs = set()
    attn_leafs = set()
    attn_linear_names = []

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            leaf = _leaf(n)
            linear_leafs.add(leaf)

            ln = n.lower()
            if any(k in ln for k in ("attn", "attention", "self_attn", "mha")):
                attn_leafs.add(leaf)
                attn_linear_names.append(n)

    target_modules = _as_list(getattr(cfg, "lora_target_modules", None))

    if not target_modules:
        candidate_sets = [
            ["qkv_proj", "out_proj"],                 # OpenELM-like
            ["q_proj", "k_proj", "v_proj", "o_proj"], # LLaMA/Mistral/Gemma/Qwen
            ["in_proj", "out_proj"],                  # packed variants
            ["c_attn", "c_proj"],                     # GPT-like
        ]
        for cands in candidate_sets:
            if all(c in attn_leafs for c in cands):
                target_modules = cands
                break

        if not target_modules:
            projection_like = []
            for leaf in sorted(attn_leafs):
                ll = leaf.lower()
                if any(k in ll for k in ("proj", "qkv", "out", "in", "query", "key", "value")):
                    projection_like.append(leaf)
            if projection_like:
                target_modules = projection_like[:4]

    if not target_modules:
        sample = "\n  - " + "\n  - ".join(attn_linear_names[:50]) if attn_linear_names else "\n  (no attention linears found)"
        raise RuntimeError(
            "LoRA injection failed: could not resolve any target modules.\n"
            "Fix: set cfg.lora_target_modules explicitly.\n\n"
            "Attention Linear module name samples:\n"
            f"{sample}"
        )

    missing = [t for t in target_modules if t not in linear_leafs]
    if missing:
        sample = "\n  - " + "\n  - ".join(attn_linear_names[:50]) if attn_linear_names else "\n  (no attention linears found)"
        raise RuntimeError(
            "LoRA targets do not exist as nn.Linear leaf module names in this model.\n"
            f"Missing: {missing}\n"
            f"Provided/selected targets: {target_modules}\n\n"
            "Fix: set cfg.lora_target_modules to valid leaf names.\n\n"
            "Attention Linear module name samples:\n"
            f"{sample}"
        )

    r = int(getattr(cfg, "lora_r", 8))
    alpha = int(getattr(cfg, "lora_alpha", 16))
    dropout = float(getattr(cfg, "lora_dropout", 0.05))
    bias = str(getattr(cfg, "lora_bias", "none"))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
    )

    gpu_info(
        "[SFT-FSDP][LoRA] enabling | "
        f"r={r} alpha={alpha} dropout={dropout} bias={bias} targets={target_modules}"
    )

    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model, True


# ==================================================
# FSDP: Gather full state dict (all ranks participate)
# ==================================================
def _gather_fsdp_full_state_dict(model: FSDP) -> Optional[Dict[str, torch.Tensor]]:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    if _is_main():
        gpu_info("[SFT-FSDP] Gathering FULL_STATE_DICT for export (all ranks participate)")

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    if not _is_main():
        return None
    return state


# ==================================================
# HF Export + Verification
# ==================================================
def _export_hf_folder(*, cfg, tokenizer, merged_state_dict_path: Path, hf_dir: Path, dtype: torch.dtype):
    hf_dir.mkdir(parents=True, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )

    state = torch.load(merged_state_dict_path, map_location="cpu")

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = base_model.load_state_dict(state, strict=False)

    base_model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)

    if _is_main():
        gpu_info(f"[SFT-FSDP] HF export saved → {hf_dir}")
        if missing:
            gpu_warning(f"[SFT-FSDP] HF export missing keys: {len(missing)}")
        if unexpected:
            gpu_warning(f"[SFT-FSDP] HF export unexpected keys: {len(unexpected)}")


# ==================================================
# LoRA merge into HF export
# ==================================================
def _maybe_merge_lora_into_hf(*, cfg, hf_dir: Path, lora_adapter_dir: Path):
    if not getattr(cfg, "enable_lora", False):
        return

    if _is_main():
        gpu_info("[SFT-FSDP][LoRA] Merging LoRA adapter into HF export")

    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "LoRA merge requested but 'peft' is not installed. "
            "Install it in this venv: pip install peft"
        ) from e

    base = AutoModelForCausalLM.from_pretrained(hf_dir, local_files_only=True, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, lora_adapter_dir, local_files_only=True)
    merged = merged.merge_and_unload()

    merged.save_pretrained(hf_dir)
    if _is_main():
        gpu_info(f"[SFT-FSDP][LoRA] Merge complete → {hf_dir}")


# ==================================================
# Intent head training (rank0 only)
# ==================================================
class IntentHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)


def _train_intent_head(*, cfg, hf_dir: Path, out_dir: Path):
    intent_jsonl = getattr(cfg, "intent_train_data_path", None)
    if not intent_jsonl:
        return
    if not _is_main():
        return

    gpu_info("[INTENT] Starting intent head training (rank0 only)")

    tok = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True, trust_remote_code=True, use_fast=True)
    lm = AutoModelForCausalLM.from_pretrained(hf_dir, local_files_only=True, trust_remote_code=True)
    lm.eval()

    for p in lm.parameters():
        p.requires_grad = False

    cfg_obj = lm.config
    hidden = getattr(cfg_obj, "hidden_size", None) or getattr(cfg_obj, "n_embd", None)
    if hidden is None:
        raise RuntimeError("Could not infer hidden size from model configuration for intent head training.")

    label_map_path = getattr(cfg, "intent_label_map_path", None)
    label_map: Optional[Dict[str, int]] = None
    if label_map_path:
        label_map = json.loads(Path(label_map_path).read_text(encoding="utf-8"))

    ds = IntentDataset(intent_jsonl, label_map=label_map)
    num_labels = int(getattr(cfg, "intent_num_labels", len(ds.label_map)))
    if num_labels <= 1:
        raise RuntimeError("intent_num_labels must be > 1 (or provide label map / multiple labels).")

    intent_out = out_dir / "hf_export_intent"
    intent_out.mkdir(parents=True, exist_ok=True)
    (intent_out / "label_map.json").write_text(json.dumps(ds.label_map, indent=2), encoding="utf-8")

    head = IntentHead(hidden_size=int(hidden), num_labels=num_labels)
    head.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm.to(device)
    head.to(device)

    bs = int(getattr(cfg, "intent_batch_size", 8))
    lr = float(getattr(cfg, "intent_learning_rate", 1e-3))
    epochs = int(getattr(cfg, "intent_num_epochs", 3))
    max_len = int(getattr(cfg, "intent_max_length", 256))

    def collate_intent(batch: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        texts = [t for t, _ in batch]
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_intent)

    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    step = 0
    for ep in range(epochs):
        gpu_info(f"[INTENT] Epoch {ep + 1}/{epochs}")
        for batch in loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                out = lm(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states[-1]
                lengths = attn.sum(dim=1) - 1
                pooled = hs[torch.arange(hs.size(0), device=device), lengths]

            logits = head(pooled)
            loss = loss_fn(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 25 == 0:
                gpu_info(f"[INTENT] step={step} loss={loss.item():.4f}")

    torch.save(head.state_dict(), intent_out / "intent_head.pt")
    meta = {"base_hf_dir": str(hf_dir), "num_labels": num_labels, "hidden_size": int(hidden), "pooling": "last_nonpad"}
    (intent_out / "intent_head_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    gpu_info(f"[INTENT] Saved intent head artifact → {intent_out}")


# ==================================================
# CONFIG
# ==================================================
@dataclass
class SFTFSDPTrainConfig:
    # Job / paths
    job_name: str
    base_model_path: str
    train_data_path: str
    output_dir: str

    # --------------------------------------------------
    # Governance identity (REQUIRED)
    # --------------------------------------------------
    dataset_hash: Optional[str] = None
    dataset_version: Optional[str] = None

    # Core SFT training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # Optional early stop (honors JSON "max_steps")
    max_steps: Optional[int] = None

    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"

    max_seq_length: int = 1024
    mixed_precision: str = "bf16"  # "bf16" | "fp16"
    max_grad_norm: float = 1.0

    # Tokenizer routing
    tokenizer_impl: str = "auto"  # "auto" | "llama"
    tokenizer_path: Optional[str] = None

    # HF export behavior
    verify_hf_export: bool = True

    # LoRA (optional, PEFT)
    enable_lora: bool = False
    merge_lora_into_hf: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: Optional[List[str]] = None

    # Intent classifier head (optional)
    train_intent_head: bool = False
    intent_train_data_path: Optional[str] = None
    intent_label_map_path: Optional[str] = None
    intent_num_labels: int = 0
    intent_num_epochs: int = 3
    intent_batch_size: int = 8
    intent_learning_rate: float = 1e-3
    intent_max_length: int = 256

    # Orchestration metadata
    training_policy: Optional[Dict[str, Any]] = None


# ==================================================
# TRAIN ENTRYPOINT (INTERNAL TRAINER)
# ==================================================
def run_sft_fsdp(
    *,
    training_policy: Dict[str, Any] | None = None,
    request_id: str | None = None,
    **kwargs,
):
    """
    SFT FSDP Trainer Entrypoint (Production-Grade)

    Responsibilities:
      - Load JSONL SFT dataset
      - Run supervised fine-tuning with FSDP (+ optional LoRA)
      - Log structured metrics to MLflow (launcher-owned run)
      - Track dataset provenance
      - Export HF-compatible model artifacts
      - Register/version model in MLflow Registry (optional)

    Design guarantees:
      - Rank-safe (rank0-only side effects)
      - Deterministic optimizer step accounting
      - Safe for world_size=1 and >1
    """

    # ------------------------------------------------------------------
    # Sanitize orchestration-only kwargs
    # ------------------------------------------------------------------
    kwargs.pop("training_policy", None)
    kwargs.pop("_training_policy", None)
    kwargs.pop("request_id", None)

    cfg = SFTFSDPTrainConfig(**kwargs)
    cfg.training_policy = training_policy

    # --------------------------------------------------
    # Enforce dataset governance
    # --------------------------------------------------
    if not cfg.dataset_hash:
        # Allow fallback from training_policy if invoked indirectly
        if cfg.training_policy and cfg.training_policy.get("dataset_hash"):
            cfg.dataset_hash = cfg.training_policy["dataset_hash"]
            cfg.dataset_version = cfg.training_policy.get("dataset_version")
        else:
            raise RuntimeError(
                "[SFT-FSDP] dataset_hash is required. "
                "Datasets must be registered via the launcher before training."
            )

    if request_id is not None:
        setattr(cfg, "request_id", request_id)

    _init_dist()

    # ------------------------------------------------------------------
    # MLflow (rank0 only, RUN IS OWNED BY LAUNCHER)
    # ------------------------------------------------------------------
    mlflow = None
    if _is_main():
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
        except Exception:
            mlflow = None

    # ------------------------------------------------------------------
    # Imports local to trainer
    # ------------------------------------------------------------------
    import json
    import math
    import os
    import functools

    import torch
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

    from transformers import AutoModelForCausalLM

    from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    class _SFTJsonlDataset(Dataset):
        """
        JSONL dataset loader for SFT training.

        Expected schema per row:
          - prompt / instruction
          - response

        Fallback:
          - text (single-field samples)
        """

        def __init__(self, path: str):
            path = str(Path(path).resolve())
            self.rows: list[str] = []

            gpu_info(f"[DATASET] Loading SFT dataset | path={path}")

            fallback_text_rows = 0
            total_rows = 0
            first_schema = None

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln_no, ln in enumerate(f, start=1):
                        if not ln.strip():
                            continue

                        total_rows += 1

                        try:
                            r = json.loads(ln)
                        except json.JSONDecodeError as e:
                            gpu_error(
                                f"[DATASET] JSON parse error | line={ln_no} | error={e}"
                            )
                            raise

                        if first_schema is None:
                            first_schema = list(r.keys())

                        resp = r.get("response")
                        prompt = r.get("prompt") or r.get("instruction")

                        if resp is not None and prompt is not None:
                            self.rows.append(f"Instruction: {prompt}\nResponse: {resp}")
                            continue

                        text = r.get("text")
                        if text:
                            fallback_text_rows += 1
                            self.rows.append(str(text))
                            continue

                        gpu_error(
                            "[DATASET] Invalid SFT row | "
                            f"line={ln_no} keys={list(r.keys())}"
                        )
                        raise RuntimeError(
                            "Missing required fields: "
                            "'prompt'/'instruction' + 'response' "
                            "or fallback 'text'"
                        )

            except Exception:
                gpu_error("[DATASET] Dataset load FAILED")
                raise

            if not self.rows:
                gpu_error(f"[DATASET] Dataset empty after parsing | path={path}")
                raise RuntimeError(f"Empty training dataset: {path}")

            gpu_info(
                "[DATASET] Loaded successfully | "
                f"rows={len(self.rows)} "
                f"fallback_text_rows={fallback_text_rows}"
            )
            if first_schema:
                gpu_info(f"[DATASET] Detected schema | keys={first_schema}")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    try:
        # ------------------------------------------------------------------
        # Normalize configuration
        # ------------------------------------------------------------------
        cfg.num_train_epochs = int(cfg.num_train_epochs)
        cfg.per_device_train_batch_size = int(cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps = max(1, int(cfg.gradient_accumulation_steps))
        cfg.max_seq_length = int(cfg.max_seq_length)
        cfg.learning_rate = float(cfg.learning_rate)
        if cfg.max_steps is not None:
            cfg.max_steps = int(cfg.max_steps) or None

        ws = _world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        if _is_main():
            gpu_info(
                "[SFT-FSDP] cfg | "
                f"epochs={cfg.num_train_epochs} "
                f"batch={cfg.per_device_train_batch_size} "
                f"grad_accum={cfg.gradient_accumulation_steps} "
                f"seq={cfg.max_seq_length} "
                f"lr={cfg.learning_rate} "
                f"mp={cfg.mixed_precision} "
                f"max_steps={cfg.max_steps}"
            )

        # ------------------------------------------------------------------
        # Hooks: ensure symbol exists and initialize at runtime (never at import time)
        #
        # IMPORTANT:
        # - This function assumes a module-level "hooks = None" exists.
        # - If it does not, define it once at top of file:
        #       hooks = None
        # ------------------------------------------------------------------
        global hooks
        if "hooks" not in globals():
            hooks = None

        if hooks is None:
            try:
                from app.training.training_hooks import MLflowGPUTrainingHooks

                hooks = MLflowGPUTrainingHooks(
                    log_every_n_steps=int(getattr(cfg, "gpu_log_every_n_steps", 1)),
                    device_index=local_rank,
                    is_main_process=_is_main(),
                )
                if _is_main():
                    gpu_info("[HOOKS] MLflowGPUTrainingHooks initialized")
            except Exception as e:
                hooks = None
                if _is_main():
                    gpu_warning(f"[HOOKS] GPU hooks disabled: {e}")

        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        train_ds = _SFTJsonlDataset(cfg.train_data_path)

        dataset_path = cfg.train_data_path
        dataset_hash = cfg.dataset_hash  # authoritative (from launcher)

        # ------------------------------------------------------------------
        # MLflow params + governance + provenance (ONLY if run exists)
        # ------------------------------------------------------------------
        if mlflow and _is_main() and mlflow.active_run():
            mlflow.log_params(
                {
                    "trainer": "sft_fsdp",
                    "base_model_path": cfg.base_model_path,
                    "enable_lora": bool(getattr(cfg, "enable_lora", False)),
                    "epochs": cfg.num_train_epochs,
                    "per_device_train_batch_size": cfg.per_device_train_batch_size,
                    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                    "effective_batch_size": (
                        cfg.per_device_train_batch_size
                        * cfg.gradient_accumulation_steps
                        * ws
                    ),
                    "max_seq_length": cfg.max_seq_length,
                    "learning_rate": cfg.learning_rate,
                    "mixed_precision": cfg.mixed_precision,
                    "dataset_path": dataset_path,
                    "dataset_size": len(train_ds),
                    "dataset_hash": dataset_hash,
                    "dataset_version": getattr(cfg, "dataset_version", "unknown"),
                    "world_size": ws,
                    "device_type": "cuda" if torch.cuda.is_available() else "cpu",
                    "gpu_name": (
                        torch.cuda.get_device_name(local_rank)
                        if torch.cuda.is_available()
                        else "none"
                    ),
                }
            )

            # Record dataset usage in governance DB (AUDIT)
            from app.config.pg_db_config import get_governance_session
            from app.ml_governance.registry import DatasetRegistry

            with get_governance_session() as db:
                registry = DatasetRegistry(db, actor="gpu_train")
                registry.mark_used(
                    dataset_hash=dataset_hash,
                    run_id=mlflow.active_run().info.run_id,
                )

            # MLflow dataset provenance (UI-only, best-effort)
            try:
                import pandas as pd

                gpu_info("[MLFLOW] Registering dataset provenance (from_pandas)")
                df = pd.read_json(dataset_path, lines=True)

                dataset_display_name = Path(dataset_path).stem
                ds = mlflow.data.from_pandas(
                    df,
                    name=dataset_display_name,
                    source=dataset_path,
                )
                mlflow.log_input(ds, context="training")

            except Exception as e:
                gpu_warning(f"[MLFLOW] Dataset provenance skipped: {e}")

        # ------------------------------------------------------------------
        # Hooks: train start (independent of MLflow)
        # ------------------------------------------------------------------
        if hooks:
            try:
                hooks.on_train_start(cfg)
            except Exception as e:
                if _is_main():
                    gpu_warning(f"[HOOKS] on_train_start skipped: {e}")

        # ------------------------------------------------------------------
        # Model + Tokenizer
        # ------------------------------------------------------------------
        tokenizer = _load_tokenizer(cfg)
        dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            local_files_only=True,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        base_model, _lora_enabled = _maybe_apply_lora(base_model, cfg)
        _force_uniform_param_dtype(base_model, dtype)
        base_model.train()

        auto_wrap_policy = None
        if ws > 1:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=int(getattr(cfg, "fsdp_min_num_params", 50_000_000)),
            )

        model = FSDP(
            base_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            ),
            device_id=device if device.type == "cuda" else None,
            use_orig_params=True,
        )

        # ------------------------------------------------------------------
        # Dataloader / Optimizer / Scheduler
        # ------------------------------------------------------------------
        def collate(batch):
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_length,
                return_tensors="pt",
            )
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels
            return enc

        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=ws, rank=_rank(), shuffle=True
            )
            if ws > 1
            else None
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.per_device_train_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

        total_opt_steps = math.ceil(
            len(train_loader) / cfg.gradient_accumulation_steps
        ) * cfg.num_train_epochs
        if cfg.max_steps:
            total_opt_steps = min(total_opt_steps, cfg.max_steps)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(total_opt_steps))
        )

        # ------------------------------------------------------------------
        # Training Loop
        # ------------------------------------------------------------------
        global_step = 0
        opt_step = 0

        for epoch in range(cfg.num_train_epochs):
            if sampler:
                sampler.set_epoch(epoch)

            for batch in train_loader:
                global_step += 1
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                loss = model(**batch, use_cache=False).loss
                (loss / cfg.gradient_accumulation_steps).backward()

                if global_step % cfg.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    opt_step += 1

                    if mlflow and _is_main() and mlflow.active_run():
                        mlflow.log_metric("train_loss", float(loss.item()), step=opt_step)
                        mlflow.log_metric("lr", float(scheduler.get_last_lr()[0]), step=opt_step)
                        mlflow.log_metric("grad_norm", float(grad_norm), step=opt_step)

                    if hooks:
                        try:
                            hooks.on_step(opt_step, float(loss.item()))
                        except Exception as e:
                            if _is_main():
                                gpu_warning(f"[HOOKS] on_step skipped: {e}")

                if cfg.max_steps and opt_step >= cfg.max_steps:
                    break

            if cfg.max_steps and opt_step >= cfg.max_steps:
                break

        if _is_main():
            gpu_info(f"[SFT-FSDP] Training complete | optimizer_steps={opt_step}")

        # ------------------------------------------------------------------
        # Export + Registry (rank0 only)
        # ------------------------------------------------------------------
        _barrier()
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        state = _gather_fsdp_full_state_dict(model)

        if _is_main() and state is not None:
            torch.save(state, out_dir / "merged_state_dict.pt")
            hf_dir = out_dir / "hf_export"

            _export_hf_folder(
                cfg=cfg,
                tokenizer=tokenizer,
                merged_state_dict_path=out_dir / "merged_state_dict.pt",
                hf_dir=hf_dir,
                dtype=dtype,
            )

            if (
                mlflow
                and cfg.training_policy
                and cfg.training_policy.get("register_model")
                and mlflow.active_run()
            ):
                import mlflow.transformers

                task = cfg.training_policy.get("mlflow_task", "text-generation")
                registry_name = cfg.training_policy["registry_name"]

                mlflow.transformers.log_model(
                    transformers_model=str(hf_dir),
                    name="model",
                    task=task,
                )

                run_id = mlflow.active_run().info.run_id
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name=registry_name,
                )

                gpu_info(
                    f"[MLFLOW] Model registered | name={registry_name} run_id={run_id}"
                )

            if mlflow and mlflow.active_run():
                from app.config.pg_db_config import get_governance_session
                from app.ml_governance.registry import DatasetRegistry

                with get_governance_session() as db:
                    registry = DatasetRegistry(db, actor="gpu_train")
                    registry.mark_completed(
                        dataset_hash=cfg.dataset_hash,
                        run_id=mlflow.active_run().info.run_id,
                        status="success",
                    )

            # --------------------------------------------------
            # Generate Training Report (rank0 only, safe)
            # --------------------------------------------------
            if mlflow and mlflow.active_run():
                try:
                    from app.reporting.generate_training_report import generate_training_report

                    current_run = mlflow.active_run()
                    current_run_id = current_run.info.run_id

                    report_path = generate_training_report(
                        run_id=current_run_id,
                        export_pdf=True,
                    )

                    # Optional but recommended: tag run with report pointer
                    mlflow.set_tag("training_report", "reports/training_report.html")

                    gpu_info(
                        f"[REPORT] Training report attached | run_id={current_run_id}"
                    )

                except Exception as e:
                    gpu_warning(f"[REPORT] Report generation failed: {e}")

        _barrier()

    except Exception:
        if mlflow and _is_main() and mlflow.active_run():
            from app.config.pg_db_config import get_governance_session
            from app.ml_governance.registry import DatasetRegistry

            try:
                with get_governance_session() as db:
                    registry = DatasetRegistry(db, actor="gpu_train")
                    registry.mark_completed(
                        dataset_hash=cfg.dataset_hash,
                        run_id=mlflow.active_run().info.run_id,
                        status="failed",
                    )
            except Exception:
                pass

        raise

    finally:
        # Hooks train end should never break teardown
        try:
            if "hooks" in globals() and hooks:
                hooks.on_train_end()
        except Exception:
            pass

        _destroy_dist()


def run_sft_fsdp_training(
    cfg: Any,
    *,
    training_policy: Dict[str, Any] | None = None,
    request_id: str | None = None,
):
    """
    Worker-facing entrypoint expected by run_fsdp_worker.

    - Accepts dataclass or dict-like configuration
    - Ensures training_policy is passed EXACTLY ONCE
    - Ensures request_id never gets forwarded into the dataclass
    """


# Accept dataclass or dict-like configuration
    if cfg is None:
        cfg_dict: Dict[str, Any] = {}
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        if hasattr(cfg, "__dataclass_fields__"):
            cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}
        elif hasattr(cfg, "dict"):
            cfg_dict = dict(cfg.dict())
        else:
            cfg_dict = dict(cfg)

    cfg_dict = dict(cfg_dict)  # defensive copy

    # Pull policy out of cfg to avoid duplicate passing
    cfg_policy = cfg_dict.pop("training_policy", None)
    cfg_dict.pop("_training_policy", None)

    if training_policy is None:
        training_policy = cfg_policy

    # Never forward orchestration-only metadata as dataclass kwargs
    cfg_dict.pop("request_id", None)

    return run_sft_fsdp(
        training_policy=training_policy,
        request_id=request_id,
        **cfg_dict,
    )

