"""Compute provenance writer (Story 7.1 / FR40 / NFR13).

Records GPU model, CUDA version, PyTorch version, wall-clock duration, peak
GPU memory, and total GPU-hours per reportable run. Written to
``results/<run-id>/compute.json`` in canonical sorted-keys JSON. ML-conference
"compute statements" auto-generate from this file.
"""

from __future__ import annotations

import contextlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class ComputeProvenance:
    """One run's compute fingerprint."""

    python_version: str
    pytorch_version: str
    cuda_version: str | None
    gpu_model: str | None
    host_machine: str
    wall_clock_seconds: float
    peak_gpu_memory_bytes: int | None
    gpu_hours: float

    def to_canonical_json(self) -> str:
        payload = {
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "gpu_model": self.gpu_model,
            "host_machine": self.host_machine,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "gpu_hours": round(self.gpu_hours, 6),
        }
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _current_gpu_info() -> tuple[str | None, int | None]:
    """Return ``(gpu_model, peak_memory_bytes)`` for the active accelerator.

    Detection order:
      1. CUDA — full info (name + ``max_memory_allocated``).
      2. MPS (Apple Silicon) — model string includes ``"(MPS)"`` so the
         backend is identifiable from ``compute.json``; peak memory comes
         from ``torch.mps.driver_allocated_memory()`` when available
         (PyTorch 2.x+); older builds return None for the memory field.
      3. CPU — both fields None.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0), int(torch.cuda.max_memory_allocated(0))
    if torch.backends.mps.is_available():
        # MPS doesn't expose a per-device name the way CUDA does; the host's
        # CPU is the chip identifier on Apple Silicon (e.g. "arm").
        chip = platform.processor() or "unknown"
        model = f"Apple GPU (MPS, {chip})"
        peak: int | None = None
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None:
            driver_alloc = getattr(mps_module, "driver_allocated_memory", None)
            if callable(driver_alloc):
                with contextlib.suppress(RuntimeError, TypeError, ValueError):
                    raw_peak: Any = driver_alloc()
                    peak = int(raw_peak)
        return model, peak
    return None, None


def _current_accelerator_version() -> str | None:
    """Best-effort accelerator-version string. CUDA toolkit version on CUDA;
    PyTorch version-tagged MPS marker on MPS; None on CPU."""
    if torch.cuda.is_available():
        return torch.version.cuda
    if torch.backends.mps.is_available():
        return f"mps-{torch.__version__}"
    return None


def record_provenance(start_time: float) -> ComputeProvenance:
    """Take a measurement now relative to ``start_time`` (from ``time.perf_counter()``)."""
    wall = max(0.0, time.perf_counter() - start_time)
    gpu_model, peak = _current_gpu_info()
    return ComputeProvenance(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        pytorch_version=str(torch.__version__),
        cuda_version=_current_accelerator_version(),
        gpu_model=gpu_model,
        host_machine=platform.node() or "unknown",
        wall_clock_seconds=wall,
        peak_gpu_memory_bytes=peak,
        gpu_hours=wall / 3600.0,
    )
