#!/usr/bin/env bash
set -e

############################################
# GPU STACK CONTROLLER
#
# Commands:
#   ./start_gpu_stack.sh           -> Start & validate
#   ./start_gpu_stack.sh stop      -> Stop services
#   ./start_gpu_stack.sh restart   -> Restart stack
#   ./start_gpu_stack.sh status    -> Show service status
#   ./start_gpu_stack.sh --strict-gpu
#
############################################

MLFLOW_PORT=5000
SERVICE_PORT=8001
HF_DEFAULT="/mnt/c/Users/operator/.cache/huggingface"

STRICT_GPU=0

############################################
# Helpers
############################################

is_port_open() {
    ss -lptn | grep -q ":$1"
}

start_mlflow() {
    if ! is_port_open $MLFLOW_PORT; then
        echo "Starting MLflow..."
        nohup mlflow ui \
            --backend-store-uri file:/mnt/c/Users/operator/PycharmProjects/gpu_train/mlruns \
            --host 0.0.0.0 \
            --port $MLFLOW_PORT > mlflow.log 2>&1 &
        sleep 3
    else
        echo "MLflow already running."
    fi
}

start_service() {
    if ! is_port_open $SERVICE_PORT; then
        echo "Starting GPU service..."
        nohup uvicorn app.main:app \
            --host 0.0.0.0 \
            --port $SERVICE_PORT \
            --log-level info > gpu_service.log 2>&1 &
        sleep 5
    else
        echo "GPU service already running."
    fi
}

stop_port() {
    if is_port_open $1; then
        echo "Stopping service on port $1..."
        PID=$(ss -lptn | grep ":$1" | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2)
        kill -9 $PID || true
    fi
}

health_check() {
    for i in {1..10}; do
        if curl -s -f http://localhost:$SERVICE_PORT/health > /dev/null; then
            echo "Health check passed."
            return 0
        fi
        sleep 1
    done

    echo "Health check FAILED."
    echo "Last 20 lines of service log:"
    tail -n 20 gpu_service.log
    exit 1
}


status() {
    echo "MLflow:"
    is_port_open $MLFLOW_PORT && echo "  Running on port $MLFLOW_PORT" || echo "  Not running"
    echo "GPU Service:"
    is_port_open $SERVICE_PORT && echo "  Running on port $SERVICE_PORT" || echo "  Not running"
}

############################################
# Argument Handling
############################################

case "$1" in
    stop)
        stop_port $SERVICE_PORT
        stop_port $MLFLOW_PORT
        echo "Stack stopped."
        exit 0
        ;;
    restart)
        stop_port $SERVICE_PORT
        stop_port $MLFLOW_PORT
        ;;
    status)
        status
        exit 0
        ;;
    --strict-gpu)
        STRICT_GPU=1
        ;;
esac

############################################
# STARTUP SEQUENCE
############################################

echo "======================================================"
echo "GPU STACK – STARTUP & PREFLIGHT"
echo "======================================================"

############################################
# 1. Activate Venv
############################################
source ~/venvs/gpu_train/bin/activate

PY_PATH=$(which python)
[[ "$PY_PATH" == *"venvs/gpu_train"* ]] || { echo "Wrong Python env."; exit 1; }

############################################
# 2. Dependency Check
############################################
pip list | grep -E "torch|fastapi|mlflow|prometheus_client" > /dev/null || {
    echo "Missing dependencies."; exit 1;
}

############################################
# 3. CUDA Check
############################################
python - <<EOF
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))
EOF

############################################
# 4. HF Cache Setup
############################################
export HF_HOME=${HF_HOME:-$HF_DEFAULT}
mkdir -p "$HF_HOME"
echo "HF_HOME -> $HF_HOME"

############################################
# 5. GPU Idle Check
############################################
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    if [ "$STRICT_GPU" -eq 1 ]; then
        echo "GPU busy. Aborting."
        exit 1
    else
        echo "Warning: GPU busy."
    fi
else
    echo "GPU idle."
fi

############################################
# 6. Start Services
############################################
start_mlflow
start_service

############################################
# 7. Health Validation
############################################
health_check

############################################
# 8. Metrics Validation
############################################
if curl -s -f http://localhost:$SERVICE_PORT/metrics > /dev/null; then
    echo "Metrics endpoint reachable."
else
    echo "Metrics endpoint failed."
    exit 1
fi


############################################
# READY
############################################
echo "======================================================"
echo "STACK READY"
echo "MLflow:   http://localhost:$MLFLOW_PORT"
echo "Service:  http://localhost:$SERVICE_PORT"
echo "======================================================"
