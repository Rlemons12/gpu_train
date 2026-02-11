from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception as e:
    raise RuntimeError(
        "peft is required for LoRA merge. Install with: pip install -U peft"
    ) from e


def merge_lora_to_hf(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    tokenizer_path: str | None = None,
    dtype: str = "bf16",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok_path = tokenizer_path or base_model_path

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    print(f"[MERGE] base_model_path={base_model_path}")
    print(f"[MERGE] adapter_path={adapter_path}")
    print(f"[MERGE] output_dir={out}")
    print(f"[MERGE] tokenizer_path={tok_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    peft_model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
    )

    print("[MERGE] Merging LoRA weights into base model...")
    merged = peft_model.merge_and_unload()

    print("[MERGE] Saving HuggingFace folder...")
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)

    print(f"[MERGE] Done → {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()

    merge_lora_to_hf(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
