import os
from pathlib import Path
from huggingface_hub import snapshot_download

# ============================================================
# HARD STALL–SAFE CONFIG
# ============================================================

MODEL_ID = "google/flan-t5-large"

BASE_DIR = Path(r"E:\emtac\models\llm\flan_t5_large")
HF_CACHE_DIR = BASE_DIR / "hf_cache"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Force safe modes
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_DISABLE_XET"] = "1"          # 🔴 CRITICAL
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # 🔴 CRITICAL
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

print(f"[INFO] HF_HOME = {HF_CACHE_DIR}")
print("[INFO] XET disabled, HTTP forced")

# ============================================================
# DOWNLOAD
# ============================================================

snapshot_path = snapshot_download(
    repo_id=MODEL_ID,
    cache_dir=HF_CACHE_DIR,
    local_files_only=False,
    resume_download=True,
    max_workers=1,              # 🔴 single stream = stable
)

print("\n[SUCCESS] Download complete")
print(snapshot_path)
