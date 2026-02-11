from __future__ import annotations

import argparse
from pathlib import Path
import torch

from transformers import AutoModelForCausalLM, AutoConfig


def export_fsdp_checkpoint(
    base_model_path: str,
    fsdp_state_path: str,
    output_dir: str,
):
    base_model_path = Path(base_model_path)
    fsdp_state_path = Path(fsdp_state_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("▶ Loading base configuration...")
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("▶ Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
    )

    print("▶ Loading FSDP state_dict...")
    state_dict = torch.load(
        fsdp_state_path,
        map_location="cpu",
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print("⚠ Missing keys:", missing)
    if unexpected:
        print("⚠ Unexpected keys:", unexpected)

    print("▶ Saving HuggingFace model...")
    model.save_pretrained(output_dir)

    print("✅ Export complete")
    print(f"📁 HF model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--fsdp_state", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    export_fsdp_checkpoint(
        base_model_path=args.base_model,
        fsdp_state_path=args.fsdp_state,
        output_dir=args.out,
    )
