from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM


class IntentJsonl(Dataset):
    def __init__(self, jsonl_path: str, label2id: Dict[str, int]):
        self.rows: List[Tuple[str, int]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.rows.append((r["text"], label2id[r["label"]]))
        if not self.rows:
            raise RuntimeError("Intent dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.rows[idx]


class CausalIntentHead(nn.Module):
    """
    Minimal classifier head on top of a decoder-only model:
    - run base model forward with output_hidden_states=True
    - pool last token (based on attention_mask)
    - linear -> logits
    """
    def __init__(self, base_lm: AutoModelForCausalLM, num_labels: int):
        super().__init__()
        self.base_lm = base_lm
        hidden = getattr(base_lm.config, "hidden_size", None)
        if hidden is None:
            # Some remote-code models name it differently
            hidden = getattr(base_lm.config, "d_model", None)
        if hidden is None:
            raise RuntimeError("Could not infer hidden size from model configuration")

        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states[-1]  # (B, T, H)

        # last valid token per row
        lengths = attention_mask.sum(dim=1) - 1  # (B,)
        pooled = hs[torch.arange(hs.size(0), device=hs.device), lengths]  # (B, H)

        logits = self.classifier(pooled)
        return logits


@dataclass
class IntentTrainCfg:
    base_model_path: str
    tokenizer_path: str
    train_jsonl: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-3
    max_len: int = 128
    dtype: str = "bf16"           # bf16|fp16
    freeze_base: bool = True      # fast routing head training


def _load_labels(jsonl_path: str) -> Dict[str, int]:
    labels = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels.append(r["label"])
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}


def train_intent_head(cfg: IntentTrainCfg) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label2id = _load_labels(cfg.train_jsonl)
    id2label = {v: k for k, v in label2id.items()}

    torch_dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    base.train()

    # Freeze base if requested (FAST)
    if cfg.freeze_base:
        for p in base.parameters():
            p.requires_grad = False

    model = CausalIntentHead(base, num_labels=len(label2id)).to(device)
    model.train()

    ds = IntentJsonl(cfg.train_jsonl, label2id)

    def collate(batch):
        texts = [x[0] for x in batch]
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(
        model.classifier.parameters() if cfg.freeze_base else model.parameters(),
        lr=cfg.lr,
    )
    loss_fn = nn.CrossEntropyLoss()

    steps = math.ceil(len(dl) * cfg.num_epochs)
    step = 0

    for epoch in range(cfg.num_epochs):
        print(f"[INTENT] epoch {epoch+1}/{cfg.num_epochs}")
        for batch in dl:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(f"[INTENT] step {step}/{steps} loss={loss.item():.4f}")

    # Save artifact folder (HF-style)
    # - base model path reference
    # - tokenizer files (so it’s portable)
    # - head weights + metadata
    (out_dir / "intent_head").mkdir(parents=True, exist_ok=True)

    head_path = out_dir / "intent_head" / "intent_head.pt"
    torch.save(model.classifier.state_dict(), head_path)

    meta = {
        "base_model_path": cfg.base_model_path,
        "tokenizer_path": cfg.tokenizer_path,
        "label2id": label2id,
        "id2label": id2label,
        "max_len": cfg.max_len,
        "freeze_base": cfg.freeze_base,
    }
    with open(out_dir / "intent_head" / "intent_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tokenizer.save_pretrained(out_dir / "intent_head")

    print(f"[INTENT] saved → {out_dir / 'intent_head'}")
    return out_dir / "intent_head"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--freeze_base", action="store_true")
    args = p.parse_args()

    cfg = IntentTrainCfg(
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        dtype=args.dtype,
        freeze_base=args.freeze_base,
    )
    train_intent_head(cfg)


if __name__ == "__main__":
    main()
