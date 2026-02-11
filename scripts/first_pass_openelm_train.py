import json
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from torch.optim import AdamW

# ==================================================
# CONFIG
# ==================================================
MODEL_DIR = "/mnt/c/Users/operator/E_vdrive/models/llm/apple_OpenELM-1_1B-Instruct"
DATASET_PATH = "/mnt/c/Users/operator/PycharmProjects/gpu_train/fanuc_lrmate200id_section7_120qa.jsonl"

LR = 2e-6
TRAIN_STEPS = 360            # ~3 passes over 120 examples
LOG_EVERY = 25
MAX_NEW_TOKENS = 200
USE_INTERACTIVE = True

# ==================================================
# TOKENIZER
# ==================================================
tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# DTYPE (bf16 preferred if available)
# ==================================================
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"Using dtype: {dtype}")

# ==================================================
# MODEL
# ==================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
)
model.train()

device = model.device
print(f"Model loaded on {device}")

# ==================================================
# LOAD SECTION 7 SME TRAINING DATA
# ==================================================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    train_texts = [
        f"Instruction: {r['instruction']}\nResponse: {r['response']}"
        for r in map(json.loads, f)
    ]

print(f"Loaded {len(train_texts)} Section 7 training samples")

# ==================================================
# OPTIMIZER + SCALER
# ==================================================
optimizer = AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

# ==================================================
# GENERATION FUNCTION
# ==================================================
@torch.no_grad()
def generate(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    model.train()
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ==================================================
# BEFORE TRAINING (BASELINE)
# ==================================================
test_prompt = (
    "Instruction: Explain the purpose of routine checks and maintenance "
    "on a FANUC LR Mate 200iD robot.\nResponse:"
)

print("\n========== BEFORE TRAINING ==========\n")
print(generate(test_prompt))

# ==================================================
# TRAIN LOOP
# ==================================================
print("\n========== TRAINING (SECTION 7 SME) ==========\n")

step = 0
while step < TRAIN_STEPS:
    for text in train_texts:
        step += 1
        if step > TRAIN_STEPS:
            break

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # critical pad masking

        with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % LOG_EVERY == 0:
            print(f"Step {step}/{TRAIN_STEPS} | loss={loss.item():.4f}")

# ==================================================
# AFTER TRAINING
# ==================================================
print("\n========== AFTER TRAINING ==========\n")
print(generate(test_prompt))

# ==================================================
# INTERACTIVE MODE (STAKEHOLDER DEMO)
# ==================================================
if USE_INTERACTIVE:
    print("\n========== INTERACTIVE MODE ==========")
    print("Ask Section 7 (Checks & Maintenance) questions.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except EOFError:
            break

        if user_input.lower() in {"exit", "quit"}:
            break

        prompt = f"Instruction: {user_input}\nResponse:"
        print("\n--- Model Response ---")
        print(generate(prompt))
        print("----------------------")

print("\n✅ DONE")
