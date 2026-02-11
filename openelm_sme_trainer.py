from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import AutoModelForCausalLM, LlamaTokenizerFast

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_snapshot


# ==================================================
# DATASET (instruction / response format preserved)
# ==================================================
class InstructionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[str] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = (
                    f"Instruction: {r['instruction']}\n"
                    f"Response: {r['response']}"
                )
                self.rows.append(text)

        if not self.rows:
            raise RuntimeError("Training dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
        return self.rows[idx]


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


def _is_main() -> bool:
    return _rank() == 0


# ==================================================
# TRAIN ENTRYPOINT (called by run_fsdp_worker)
# ==================================================
def run_openelm_sme_training(cfg):
    """
    cfg is FSdpTrainConfig injected by gpu_train service
    """

    _init_dist()

    local_rank = int(torch.cuda.current_device())
    device = torch.device(f"cuda:{local_rank}")

    if _is_main():
        gpu_info(
            f"[OpenELM-SME] Starting training | model={cfg.base_model_path}"
        )

    # ==================================================
    # TOKENIZER
    # ==================================================
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ==================================================
    # MODEL
    # ==================================================
    dtype = (
        torch.bfloat16
        if cfg.mixed_precision == "bf16"
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.train()

    auto_wrap = transformer_auto_wrap_policy(
        transformer_layer_cls=set(
            type(m)
            for m in model.modules()
            if m.__class__.__name__.endswith("DecoderLayer")
        )
    )

    mp = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        device_id=device,
    )

    gpu_snapshot("[OpenELM-SME] after-fsdp-wrap")

    # ==================================================
    # DATA
    # ==================================================
    dataset = InstructionDataset(cfg.train_data_path)

    def collate(texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return {k: v.to(device) for k, v in enc.items()}

    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    # ==================================================
    # OPTIMIZER
    # ==================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    total_steps = int(
        math.ceil(len(loader) * cfg.num_train_epochs)
    )

    step = 0

    # ==================================================
    # TRAIN LOOP (your logic, service-safe)
    # ==================================================
    for epoch in range(int(cfg.num_train_epochs)):
        if _is_main():
            gpu_info(f"[OpenELM-SME] Epoch {epoch + 1}")

        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            if _is_main() and step % 25 == 0:
                gpu_info(
                    f"[OpenELM-SME] step={step}/{total_steps} "
                    f"loss={loss.item():.4f}"
                )

    # ==================================================
    # SAVE FINAL CHECKPOINT (rank 0)
    # ==================================================
    if _is_main():
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gpu_info("[OpenELM-SME] Saving final checkpoint")

        state_dict = model.state_dict()
        torch.save(
            state_dict,
            out_dir / "pytorch_model_fsdp.pt",
        )

    dist.barrier()

    if _is_main():
        gpu_info("[OpenELM-SME] Training complete")

    if dist.is_initialized():
        dist.barrier()

    if _is_main():
        gpu_info("[OpenELM-SME] Training complete")

    if dist.is_initialized():
        dist.destroy_process_group()
