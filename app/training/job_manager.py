from __future__ import annotations

import os
import uuid
import time
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Deque
from collections import deque

from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error
from app.metrics.prometheus import (
    TRAIN_RUNNING,
    TRAIN_QUEUED,
    TRAIN_JOBS_STARTED,
    TRAIN_JOBS_FINISHED,
    TRAIN_JOBS_FAILED,
    TRAIN_JOBS_CANCELED,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


MAX_CONCURRENT_TRAIN_JOBS = _env_int("MAX_CONCURRENT_TRAIN_JOBS", 1)
POLL_INTERVAL_SECONDS = float(os.getenv("TRAIN_JOB_POLL_INTERVAL", "1.0"))


@dataclass
class JobRecord:
    job_id: str
    job_name: str
    status: str  # queued|running|finished|failed|canceled
    proc: Optional[subprocess.Popen]
    started_at: float
    finished_at: Optional[float]
    return_code: Optional[int]
    output_dir: str
    log_file: str
    error: Optional[str]
    cmd: List[str]
    env: Dict[str, str]


class TrainingJobManager:
    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._queue: Deque[str] = deque()
        self._lock = threading.Lock()

        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    def _now(self) -> float:
        return time.time()

    def _running_count_locked(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    def _update_metrics_locked(self):
        TRAIN_RUNNING.set(self._running_count_locked())
        TRAIN_QUEUED.set(len(self._queue))

    def start_job(
        self,
        job_name: str,
        cmd: List[str],
        env: Dict[str, str],
        output_dir: str,
        log_file: str,
    ) -> JobRecord:
        job_id = str(uuid.uuid4())[:8]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        lf = Path(log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)

        rec = JobRecord(
            job_id=job_id,
            job_name=job_name,
            status="queued",
            proc=None,
            started_at=self._now(),
            finished_at=None,
            return_code=None,
            output_dir=str(out_dir),
            log_file=str(lf),
            error=None,
            cmd=cmd,
            env=env,
        )

        with self._lock:
            self._jobs[job_id] = rec
            self._queue.append(job_id)
            self._update_metrics_locked()

        gpu_info(f"[TRAIN] Queued job | job_id={job_id} job_name={job_name}")
        return rec

    def _launch_job_locked(self, job_id: str) -> JobRecord:
        rec = self._jobs[job_id]
        out_dir = Path(rec.output_dir)
        lf = Path(rec.log_file)

        gpu_info(f"[TRAIN] Launching job | job_id={rec.job_id} job_name={rec.job_name}")
        gpu_info(f"[TRAIN] Launch cmd: {' '.join(rec.cmd)}")

        f = open(lf, "a", encoding="utf-8")

        try:
            # Create a new process group so we can kill the entire torchrun tree.
            preexec_fn = os.setsid if os.name != "nt" else None

            proc = subprocess.Popen(
                rec.cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=rec.env,
                cwd=str(out_dir),
                preexec_fn=preexec_fn,
            )
        except Exception as e:
            gpu_error(f"[TRAIN] Launch failed | job_id={rec.job_id} error={e}")
            try:
                f.close()
            except Exception:
                pass

            rec.status = "failed"
            rec.proc = None
            rec.finished_at = self._now()
            rec.error = str(e)
            self._jobs[job_id] = rec

            TRAIN_JOBS_FAILED.inc()
            return rec

        rec.status = "running"
        rec.proc = proc
        rec.started_at = self._now()
        rec.error = None
        self._jobs[job_id] = rec

        TRAIN_JOBS_STARTED.inc()
        return rec

    def _pump_loop(self):
        while True:
            try:
                self._pump_once()
            except Exception:
                pass
            time.sleep(POLL_INTERVAL_SECONDS)

    def _pump_once(self):
        with self._lock:
            # Poll running jobs
            for job_id in list(self._jobs.keys()):
                self._poll_job_locked(job_id)

            # Launch queued jobs up to concurrency limit
            while self._queue and self._running_count_locked() < MAX_CONCURRENT_TRAIN_JOBS:
                next_id = self._queue.popleft()
                rec = self._jobs.get(next_id)
                if not rec or rec.status != "queued":
                    continue
                self._launch_job_locked(next_id)

            self._update_metrics_locked()

    def _poll_job_locked(self, job_id: str) -> JobRecord:
        rec = self._jobs[job_id]
        if rec.proc is None:
            return rec

        if rec.status not in {"running"}:
            return rec

        rc = rec.proc.poll()
        if rc is None:
            return rec

        rec.return_code = rc
        rec.finished_at = self._now()

        if rc == 0:
            rec.status = "finished"
            gpu_info(f"[TRAIN] Job finished | job_id={job_id} rc=0")
            TRAIN_JOBS_FINISHED.inc()
        else:
            rec.status = "failed"
            rec.error = f"Non-zero return code: {rc}"
            gpu_warning(f"[TRAIN] Job failed | job_id={job_id} rc={rc}")
            TRAIN_JOBS_FAILED.inc()

        self._jobs[job_id] = rec
        return rec

    def cancel_job(self, job_id: str, force: bool = False) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)

            rec = self._jobs[job_id]

            # If queued, just remove from queue and mark canceled
            if rec.status == "queued":
                try:
                    self._queue = deque([jid for jid in self._queue if jid != job_id])
                except Exception:
                    pass
                rec.status = "canceled"
                rec.finished_at = self._now()
                rec.error = "Canceled while queued"
                self._jobs[job_id] = rec
                TRAIN_JOBS_CANCELED.inc()
                self._update_metrics_locked()
                return rec

            # If running, kill process group
            if rec.status == "running" and rec.proc is not None:
                try:
                    if os.name != "nt":
                        pgid = os.getpgid(rec.proc.pid)
                        os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
                    else:
                        rec.proc.kill() if force else rec.proc.terminate()
                except Exception as e:
                    rec.error = f"Cancel failed: {e}"
                    self._jobs[job_id] = rec
                    self._update_metrics_locked()
                    return rec

                rec.status = "canceled"
                rec.finished_at = self._now()
                rec.return_code = rec.proc.poll()
                rec.error = "Canceled by user"
                self._jobs[job_id] = rec
                TRAIN_JOBS_CANCELED.inc()
                self._update_metrics_locked()
                return rec

            # finished/failed/canceled: no-op
            return rec

    def status(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self._jobs[job_id]
            return rec

    def tail_log(self, log_file: str, lines: int = 80) -> List[str]:
        p = Path(log_file)
        if not p.exists():
            return []
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                data = f.read().splitlines()
            return data[-lines:]
        except Exception:
            return []

    def list_jobs(self) -> List[JobRecord]:
        with self._lock:
            return list(self._jobs.values())


TRAINING_JOBS = TrainingJobManager()
