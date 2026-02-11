import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_DIR = r"/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/mistral7b_mini/hf_export"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# --------------------------------------------------
# LOAD MODEL + TOKENIZER
# --------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
)

model.eval()

print(f"Model loaded on {DEVICE}")
print("Type 'exit' to quit\n")

# --------------------------------------------------
# INTERACTIVE LOOP
# --------------------------------------------------
while True:
    prompt = input("You: ").strip()
    if not prompt or prompt.lower() in {"exit", "quit"}:
        break

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    print("\nModel:")
    print(response)
    print("-" * 60)
