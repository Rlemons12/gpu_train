import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_MODEL_PATH = "/mnt/c/Users/operator/E_vdrive/models/llm/mistral-7b-instruct"
FINETUNED_MODEL_PATH = (
    "/mnt/c/Users/operator/PycharmProjects/gpu_train/prod_out/"
    "mistral7b_mini/hf_export"
)

assert Path(FINETUNED_MODEL_PATH).exists(), "❌ Finetuned model path does not exist"

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Using device: {device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------------------------------
# TOKENIZER
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    FINETUNED_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

# -------------------------------------------------
# LOAD BASE MODEL
# -------------------------------------------------
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
)
base_model.eval()

print("BASE model identity:")
print("  name_or_path:", base_model.config._name_or_path)

# -------------------------------------------------
# LOAD FINETUNED MODEL
# -------------------------------------------------
print("\nLoading FINETUNED model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_PATH,
    local_files_only=True,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
)
ft_model.eval()

print("FINETUNED model identity:")
print("  name_or_path:", ft_model.config._name_or_path)

# -------------------------------------------------
# SANITY CHECKS
# -------------------------------------------------
print("\nSanity checks:")

base_params = sum(p.numel() for p in base_model.parameters())
ft_params = sum(p.numel() for p in ft_model.parameters())

print(f"  Base params     : {base_params:,}")
print(f"  Finetuned params: {ft_params:,}")

if base_params == ft_params:
    print("  ⚠️ Param count identical (expected with merged LoRA)")
else:
    print("  ✅ Param count differs")

# lightweight tensor check (first weight)
with torch.no_grad():
    same_tensor = torch.equal(
        next(base_model.parameters()).cpu(),
        next(ft_model.parameters()).cpu(),
    )

print("  Weights identical:", same_tensor)
if same_tensor:
    print("  ⚠️ Models may be identical — verify training output")
else:
    print("  ✅ Finetune weights differ from base")

print("\nModels loaded.")
print("Type 'exit' to quit.\n")

# -------------------------------------------------
# GENERATION HELPER
# -------------------------------------------------
def generate(model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,     # 🔒 deterministic
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# INTERACTIVE LOOP
# -------------------------------------------------
while True:
    prompt = input("You> ").strip()
    if prompt.lower() in {"exit", "quit"}:
        break

    print("\n--- BASE MODEL ---")
    print(generate(base_model, prompt))

    print("\n--- FINETUNED MODEL ---")
    print(generate(ft_model, prompt))

    print("\n" + "=" * 60 + "\n")

