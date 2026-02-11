import time
import sys
import json
import requests
from pathlib import Path
import time
from pathlib import Path
import time

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/tmp/gpu_train_smoke_{RUN_ID}"


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_URL = "http://localhost:8000"

MODEL_PATH = "/mnt/c/Users/operator/E_vdrive/models/llm/Qwen2.5-3B-Instruct"
DATA_PATH = "/tmp/smoke_data.jsonl"
OUT_DIR = "/tmp/gpu_train_smoke"

POLL_INTERVAL = 1.0
STARTUP_TIMEOUT = 30
CHECKPOINT_TIMEOUT = 90
FINISH_TIMEOUT = 120


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def fail(msg):
    print(f"[SMOKE ❌] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[SMOKE ✅] {msg}")


def wait_for_status(job_id, target, timeout):
    for _ in range(int(timeout / POLL_INTERVAL)):
        r = requests.get(f"{BASE_URL}/train/status/{job_id}")
        if r.status_code != 200:
            fail(f"Status check failed: {r.text}")
        s = r.json()["status"]
        if s == target:
            return True
        if s == "failed":
            fail(f"Job failed: {r.json()}")
        time.sleep(POLL_INTERVAL)
    return False


# --------------------------------------------------
# 0. Prepare smoke dataset
# --------------------------------------------------
Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

if not Path(DATA_PATH).exists():
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": "Say hello.",
            "response": "Hello."
        }) + "\n")

ok("Smoke dataset ready")


# --------------------------------------------------
# 1. Health check
# --------------------------------------------------
r = requests.get(f"{BASE_URL}/health")
if r.status_code != 200:
    fail("Service health check failed")
ok("Service health OK")


# --------------------------------------------------
# 2. Start training job
# --------------------------------------------------
payload = {
    "job_name": "smoke_test_job",
    "base_model_path": MODEL_PATH,
    "train_data_path": DATA_PATH,
    "output_dir": OUT_DIR,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "nproc_per_node": 1,
}

r = requests.post(f"{BASE_URL}/train/start", json=payload)
if r.status_code != 200:
    fail(f"Failed to start job: {r.text}")

job = r.json()
job_id = job["job_id"]
ok(f"Job started (job_id={job_id})")


# --------------------------------------------------
# 3. Wait for running
# --------------------------------------------------
if not wait_for_status(job_id, "running", STARTUP_TIMEOUT):
    fail("Job never entered running state")

ok("Job is running")


# --------------------------------------------------
# 4. Wait for checkpoint (optional)
# --------------------------------------------------
ckpt_dir = Path(OUT_DIR)
checkpoint_seen = False

for _ in range(int(CHECKPOINT_TIMEOUT / POLL_INTERVAL)):
    if ckpt_dir.exists() and any(p.is_dir() for p in ckpt_dir.glob("checkpoint*")):
        checkpoint_seen = True
        break
    time.sleep(POLL_INTERVAL)

if checkpoint_seen:
    ok("Checkpoint directory created")
else:
    print("[SMOKE ⚠️] No checkpoint detected (non-fatal for minimal smoke test)")


# --------------------------------------------------
# 5. Wait for finish
# --------------------------------------------------
if not wait_for_status(job_id, "finished", FINISH_TIMEOUT):
    fail("Job did not finish in time")

ok("Job finished successfully")


# --------------------------------------------------
# 6. Metrics endpoint (basic sanity)
# --------------------------------------------------
r = requests.get(f"{BASE_URL}/metrics")
if r.status_code != 200:
    fail("/metrics endpoint failed")

ok("Prometheus metrics endpoint reachable")


print("\n🎉 ALL SMOKE TESTS PASSED 🎉")
