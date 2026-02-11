# app/training/training_hooks.py

from typing import Any, Optional, Dict

import torch


class TrainingHooks:
    """
    Lightweight, model-agnostic training callbacks.

    Trainers may call these if provided.
    Implementations (MLflow, logging, DB, etc.) live elsewhere.
    """

    def on_train_start(self, cfg: Any) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_step(self, step: int, loss: float) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_train_end(self) -> None:
        pass


# ----------------------------------------------------------------------
# Optional MLflow + GPU implementation
# ----------------------------------------------------------------------
import torch
from typing import Any, Dict

class MLflowGPUTrainingHooks(TrainingHooks):
    """
    MLflow-backed TrainingHooks that log GPU utilization metrics.

    Safe for:
      - FSDP
      - LoRA
      - single or multi-GPU
      - rank-0 only usage
    """

    def __init__(
        self,
        *,
        log_every_n_steps: int = 1,  # IMPORTANT: default to 1 for validation runs
        device_index: int = 0,
        is_main_process: bool = True,
    ) -> None:
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.device_index = device_index
        self.is_main_process = is_main_process
        self._nvml_initialized = False
        self._last_logged_step: int | None = None

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _init_nvml(self) -> None:
        if self._nvml_initialized:
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception:
            self._nvml_initialized = False

    def _get_gpu_metrics(self) -> Dict[str, float]:
        if not self.is_main_process:
            return {}

        if not torch.cuda.is_available():
            return {}

        self._init_nvml()
        if not self._nvml_initialized:
            return {}

        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        metrics: Dict[str, float] = {
            # Utilization
            "gpu_utilization_pct": float(util.gpu),

            # Memory
            "gpu_memory_used_mb": mem.used / 1024**2,
            "gpu_memory_total_mb": mem.total / 1024**2,
            "gpu_memory_utilization_pct": (mem.used / mem.total) * 100.0,

            # Torch-side memory (often more insightful)
            "gpu_torch_memory_allocated_mb": (
                torch.cuda.memory_allocated(self.device_index) / 1024**2
            ),
            "gpu_torch_memory_reserved_mb": (
                torch.cuda.memory_reserved(self.device_index) / 1024**2
            ),
            "gpu_torch_max_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(self.device_index) / 1024**2
            ),
        }

        try:
            metrics["gpu_power_watts"] = (
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            )
        except Exception:
            pass

        try:
            metrics["gpu_temperature_c"] = float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception:
            pass

        return metrics

    # -----------------------------
    # Hook implementations
    # -----------------------------

    def on_train_start(self, cfg: Any) -> None:
        if not self.is_main_process:
            return

        try:
            import mlflow
            import pynvml

            self._init_nvml()
            if not self._nvml_initialized:
                return

            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            mlflow.log_params(
                {
                    "gpu_device_index": self.device_index,
                    "gpu_name": torch.cuda.get_device_name(self.device_index),
                    "gpu_total_memory_mb": mem.total / 1024**2,
                }
            )

        except Exception:
            # GPU params are optional
            pass

    def on_step(self, step: int, loss: float | None = None) -> None:
        if not self.is_main_process:
            return

        # Ensure we log at least once, even for 1-step runs
        if (
            self._last_logged_step is not None
            and step - self._last_logged_step < self.log_every_n_steps
        ):
            return

        self._last_logged_step = step

        try:
            import mlflow

            metrics = self._get_gpu_metrics()
            if metrics:
                mlflow.log_metrics(metrics, step=step)

        except Exception:
            # Never fail training due to telemetry
            pass

    def on_train_end(self) -> None:
        if not self.is_main_process:
            return

        try:
            import mlflow

            if torch.cuda.is_available():
                mlflow.log_metric(
                    "gpu_peak_memory_allocated_mb",
                    torch.cuda.max_memory_allocated(self.device_index) / 1024**2,
                )
        except Exception:
            pass

