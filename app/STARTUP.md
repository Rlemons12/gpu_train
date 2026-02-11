# GPU Training Service – Startup & Preflight Runbook

This document defines the **authoritative, verified startup procedure**
for the GPU training stack.

Follow these steps **in order** before running any training scripts.
Skipping steps risks wasted GPU time and invalid experiment runs.

---

## 0. Scope

This runbook validates:

- Python environment
- CUDA + GPU availability
- Hugging Face model cache
- MLflow experiment tracking
- FastAPI GPU training service
- Prometheus metrics
- GPU idle state prior to training

---

## 1. Activate Python Environment

Activate the training virtual environment:

```bash
source ~/venvs/gpu_train/bin/activate
Verify:

bash
Copy code
which python
python --version
Expected:

Python 3.10.x

Path under ~/venvs/gpu_train/

2. Verify Core Dependencies
bash
Copy code
pip list | grep -E "torch|fastapi|mlflow|prometheus"
Expected:

torch (CUDA-enabled)

fastapi

mlflow

prometheus_client

If any are missing, stop and fix the environment.

3. GPU + CUDA Sanity Check (REQUIRED)
Run before starting any service:

bash
Copy code
python - <<EOF
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
EOF
Expected:

CUDA available: True

Device count ≥ 1

Correct GPU name

If this fails, do not proceed.

4. Hugging Face Cache Validation
bash
Copy code
echo "HF_HOME=$HF_HOME"
ls "$HF_HOME"
Expected:

HF_HOME is set

Directory exists

Contains hub/, transformers/, etc.

TRANSFORMERS_CACHE may be unset.
This is correct and preferred.

5. Start MLflow UI
Start MLflow before the training service:

bash
Copy code
mlflow ui \
  --backend-store-uri file:/mnt/c/Users/operator/PycharmProjects/gpu_train/mlruns \
  --host 0.0.0.0 \
  --port 5000
Verify in browser:

arduino
Copy code
http://localhost:5000
If the port is already in use, confirm MLflow is running:

bash
Copy code
ss -lptn | grep 5000
6. Start GPU Training Service
From the project root:

bash
Copy code
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --log-level info
Expected log lines:

Logging initialized

GPU training service booted

Application startup complete

No CUDA warnings or tracebacks

7. Health Check
bash
Copy code
curl http://localhost:8001/health
Expected:

json
Copy code
{"ok": true}
8. Metrics Check
bash
Copy code
curl http://localhost:8001/metrics | head
Expected:

Prometheus text output

SERVICE_UP 1

9. Route Verification (Optional)
bash
Copy code
curl http://localhost:8001/openapi.json
Confirm /train/* endpoints are present.

10. GPU Idle Safety Check
In a second terminal:

bash
Copy code
nvidia-smi
Expected:

Training service process visible

Near-zero VRAM usage

GPU utilization ~0%

READY STATE
Once all steps above pass:

Infrastructure is healthy

GPU is available

MLflow is tracking

Training scripts may be executed safely

Failures after this point are training logic issues, not platform issues.

Operational Rules
Always follow this startup order

Never run long training jobs without passing this checklist

Do not bypass GPU idle verification

This runbook is intentionally conservative to protect GPU time.

yaml
Copy code

---

If you want the **shutdown runbook**, **automation script**, or **Makefile version**, say the word.
::contentReference[oaicite:0]{index=0}


# cd /mnt/c/Users/operator/PycharmProjects/gpu_train
python -m dataset_gen.app.app
