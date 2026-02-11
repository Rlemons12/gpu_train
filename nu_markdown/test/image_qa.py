"""
image_qa.py

Simple interactive CLI:
Ask a question about an image using NuMarkdown (Qwen2.5-VL).

Usage:
python -m nu_markdown.image_qa
"""

import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from nu_markdown.configuration.vlm_config import VLMConfig


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def windows_to_wsl_path(path: str) -> str:
    """
    Convert Windows path to WSL path.
    """
    path = path.strip().strip('"')

    if ":" in path:
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    return path





def load_model():
    cfg = VLMConfig.from_env()

    model_path = Path(cfg.model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    print("Loading processor from:", model_path)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading model from:", model_path)

    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
    )

    model.eval()

    print("Model loaded.")
    print("Device:", model.device)

    return processor, model




# ------------------------------------------------------------
# Core QA
# ------------------------------------------------------------

def answer_question(processor, model, image_path: str, question: str):

    img = Image.open(image_path).convert("RGB")

    # ------------------------------------------------------------
    # Resize guard (prevent VRAM explosions)
    # ------------------------------------------------------------
    max_side = 2048
    w, h = img.size
    long_side = max(w, h)

    if long_side > max_side:
        scale = max_side / long_side
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size)
        print(f"Image resized to {new_size}")

    # ------------------------------------------------------------
    # Build multimodal message
    # ------------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        question
                        + "\n\n"
                        + "Answer clearly and directly. "
                        + "Do NOT include reasoning. "
                        + "Do NOT include <think> blocks."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # Strip accidental reasoning blocks
    if "<think>" in output_text:
        output_text = output_text.split("</think>")[-1]

    return output_text.strip()



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():

    print("\n=== NuMarkdown Image QA CLI ===\n")

    processor, model = load_model()

    print("\nModel ready.\n")

    while True:
        image_input = input("Image path (or 'exit'): ").strip()

        if image_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        image_path = windows_to_wsl_path(image_input)

        if not Path(image_path).exists():
            print("File not found:", image_path)
            continue

        question = input("Question: ").strip()

        if not question:
            print("Empty question.")
            continue

        print("\nThinking...\n")

        answer = answer_question(processor, model, image_path, question)

        print("\n=== ANSWER ===\n")
        print(answer)
        print()



if __name__ == "__main__":
    main()
