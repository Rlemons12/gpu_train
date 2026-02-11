
# source ~/venvs/gpu_train/bin/activate
"""mlflow ui \
  --backend-store-uri file:/mnt/c/Users/operator/PycharmProjects/gpu_train/mlruns \
  --host 0.0.0.0 \
  --port 5000"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response

import logging.config
import torch

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.train import router as train_router
from app.config.settings import SERVICE_NAME, SERVICE_VERSION
from app.config.gpu_logger import gpu_info
from app.config.gpu_log_config import LOGGING_CONFIG
from app.metrics.prometheus import SERVICE_UP

# ==================================================
# LOGGING CONFIGURATION (CRITICAL FIX)
# ==================================================
logging.config.dictConfig(LOGGING_CONFIG)

# ==================================================
# FASTAPI APP
# ==================================================
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
)

# ==================================================
# ROUTERS
# ==================================================
app.include_router(train_router)

# ==================================================
# PROMETHEUS METRICS
# ==================================================
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# ==================================================
# STARTUP / SHUTDOWN H



def create_app() -> FastAPI:
    app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

    # -------------------------------------------------
    # Training API only
    # -------------------------------------------------
    app.include_router(train_router)

    # -------------------------------------------------
    # Health + metrics
    # -------------------------------------------------
    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
SERVICE_UP.set(1)
gpu_info("GPU training service booted")
