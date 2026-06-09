"""GPU preflight for the remote sweep — fail fast if CUDA is unusable.

Run before a multi-seed sweep so a mis-provisioned pod (NVIDIA driver too
old for the installed torch build, no GPU attached, CPU-only image)
aborts at setup instead of silently falling back to CPU and burning
hours producing zero usable training steps.

Exit 0 + a one-line summary when CUDA is available; exit 1 with a
diagnostic when it is not.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    """Return 0 if CUDA is usable, 1 otherwise."""
    if not torch.cuda.is_available():
        sys.stderr.write(
            "GPU PREFLIGHT FAIL: torch.cuda.is_available() is False.\n"
            "The pod cannot run CUDA. Most common cause: the NVIDIA driver "
            "is older than this PyTorch build supports (check the pod's CUDA "
            "version — it must match torch). Aborting before the sweep so no "
            "compute is wasted on a silent CPU fallback.\n"
        )
        return 1
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # is_available() and the metadata queries above can ALL falsely succeed
    # while actual context creation / allocation fails — e.g.
    # cudaErrorDevicesUnavailable on a host-contended or wedged GPU, or an
    # empty/incorrect CUDA_VISIBLE_DEVICES masking the device. nvidia-smi may
    # still show the GPU idle. Force a real allocation + compute + sync so the
    # sweep never starts against a GPU it cannot actually use.
    try:
        probe = torch.randn(1024, 1024, device="cuda")
        result = (probe @ probe).sum()
        torch.cuda.synchronize()
        _ = float(result)
    except Exception as exc:
        sys.stderr.write(
            f"GPU PREFLIGHT FAIL: {name} reports available, but a real CUDA "
            f"allocation + matmul failed: {exc}\n"
            "The device cannot create a usable compute context. Common causes: "
            "the physical GPU is held by another container (host contention), a "
            "wedged context from a crashed run, or an empty/incorrect "
            "CUDA_VISIBLE_DEVICES. nvidia-smi often still shows the GPU idle. "
            "Restart or reprovision the pod, then re-run. Aborting before the "
            "sweep so no compute is wasted.\n"
        )
        return 1
    sys.stderr.write(
        f"GPU preflight ok: {name}, {total_gb:.0f} GB VRAM, real allocation + matmul verified.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
