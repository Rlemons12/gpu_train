#!/usr/bin/env python3
"""
EMTAC Portable Model CLI Tester

Usage:
    python test_model_cli.py --model-path portable_models/emtac_mistral_sft_v21_portable
    python test_model_cli.py --model-path path/to/model --prompt "Explain hydraulic cavitation"

Features:
    - Auto-detects inner MLflow "model" folder
    - Verifies HF model structure
    - Confirms GPU usage
    - Measures latency
    - Interactive mode supported
"""

import argparse
import time
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------------------------------
# Device Detection
# -------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        print(f"[INFO] CUDA detected: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
        return "cuda"
    else:
        print("[INFO] Using CPU")
        return "cpu"


# -------------------------------------------------------
# Resolve Model Path
# -------------------------------------------------------

def resolve_model_path(model_path: Path) -> Path:
    """
    Handles MLflow export structure:
    portable_model/
        model/
            config.json
            tokenizer.json
            ...
    """

    if (model_path / "config.json").exists():
        return model_path

    if (model_path / "model").exists():
        print("[INFO] Detected MLflow structure. Using inner 'model' directory.")
        return model_path / "model"

    raise FileNotFoundError(
        f"Could not find HuggingFace model files in: {model_path}"
    )


# -------------------------------------------------------
# Model Load
# -------------------------------------------------------

def load_model(model_path: Path, device: str):

    model_path = resolve_model_path(model_path)

    print(f"[INFO] Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        legacy=True  # avoids llama tokenizer warning confusion
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    model.eval()

    print("[SUCCESS] Model loaded successfully\n")

    return tokenizer, model


# -------------------------------------------------------
# Generate
# -------------------------------------------------------

def generate(tokenizer, model, prompt: str, device: str):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    latency = time.time() - start

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output cleanly
    response = full_output[len(prompt):].strip()

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.shape[1] - input_tokens

    return response, latency, input_tokens, output_tokens


# -------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to exported model")
    parser.add_argument("--prompt", help="Optional single prompt (otherwise interactive mode)")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = get_device()

    tokenizer, model = load_model(model_path, device)

    # ---------------------------------------------------
    # Single prompt mode
    # ---------------------------------------------------
    if args.prompt:

        print("\n[INFO] Running inference...")
        print(f"[PROMPT] {args.prompt}\n")

        response, latency, input_tokens, output_tokens = generate(
            tokenizer, model, args.prompt, device
        )

        print("=" * 80)
        print("[RESPONSE]")
        print(response)
        print("=" * 80)

        print(f"[LATENCY] {latency:.2f} seconds")
        print(f"[INPUT TOKENS] {input_tokens}")
        print(f"[OUTPUT TOKENS] {output_tokens}")
        print("[DONE]")

        return

    # ---------------------------------------------------
    # Interactive mode
    # ---------------------------------------------------
    print("\n[INTERACTIVE MODE]")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("EMTAC> ")

        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        response, latency, input_tokens, output_tokens = generate(
            tokenizer, model, prompt, device
        )

        print("\n--- RESPONSE ---")
        print(response)
        print(f"\n[LATENCY] {latency:.2f}s | TOKENS: {input_tokens}->{output_tokens}\n")


if __name__ == "__main__":
    main()
