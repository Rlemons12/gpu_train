from __future__ import annotations

import os
from prometheus_client import Counter, Gauge, Histogram

# Service-level metrics (not per-training-process loss; those live in MLflow/metrics.jsonl)
SERVICE_UP = Gauge("gpu_train_service_up", "GPU train service is up (1=up)")
TRAIN_JOBS_STARTED = Counter("gpu_train_jobs_started_total", "Total training jobs started")
TRAIN_JOBS_FINISHED = Counter("gpu_train_jobs_finished_total", "Total training jobs finished")
TRAIN_JOBS_FAILED = Counter("gpu_train_jobs_failed_total", "Total training jobs failed")
TRAIN_JOBS_CANCELED = Counter("gpu_train_jobs_canceled_total", "Total training jobs canceled")

TRAIN_RUNNING = Gauge("gpu_train_jobs_running", "Number of running training jobs")
TRAIN_QUEUED = Gauge("gpu_train_jobs_queued", "Number of queued training jobs")

API_REQUESTS = Counter("gpu_train_api_requests_total", "API requests", ["route"])
JOB_LAUNCH_LATENCY = Histogram("gpu_train_job_launch_latency_seconds", "Job launch latency seconds")

