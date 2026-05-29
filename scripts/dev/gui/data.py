"""Read-only loaders for the `results/<run-id>/` contract.

Every loader takes a path and returns plain Python / dataclass / pydantic
objects. There is no write path through this module by design — the GUI
must not perturb the reportable artifact tree (§"Read-only against
`results/<run-id>/`").

Loaders are deliberately tolerant of partial/in-progress runs: a live-tail
view may open a run mid-flight, before `metrics.json` or `report.md`
exist. Missing files surface as `None` rather than exceptions so the UI can
render a "waiting" state rather than crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wormjepa.eval.metrics_schema import MetricsOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — filenames per architecture's results contract
# ---------------------------------------------------------------------------

FILE_CONFIG: str = "config.yaml"
FILE_METRICS: str = "metrics.json"
FILE_COMPUTE: str = "compute.json"
FILE_SEED: str = "seed.txt"
FILE_MANIFEST: str = "manifest_at_run.lock"
FILE_REPORT: str = "report.md"
FILE_LOG: str = "log.jsonl"
DIR_CHECKPOINTS: str = "checkpoints"


@dataclass(slots=True, frozen=True)
class RunSummary:
    """One row in the run-table (the basic-variant `RunRow` content).

    Phase 1 keeps this intentionally narrow — seed/config-slug/git-sha are
    extracted on a best-effort basis from the files that already exist,
    falling back to `None` if any one of them is missing.
    """

    run_id: str
    path: Path
    mtime: float
    seed: str | None
    config_slug: str | None
    git_sha: str | None
    gpu_hours: float | None
    has_metrics: bool
    has_report: bool
    has_log: bool


@dataclass(slots=True, frozen=True)
class LogEntry:
    """One parsed line of `log.jsonl`.

    The full record is preserved in `payload` for downstream consumers; the
    convenience fields (`step`, `loss`) are populated when present under
    the conventional `extra` key the project's `JSONLFormatter` writes.
    """

    line_number: int
    payload: dict[str, Any]
    step: int | None
    loss: float | None


@dataclass(slots=True, frozen=True)
class LatentStatsSeries:
    """Per-step latent-geometry diagnostics extracted from `log.jsonl`.

    Mirrors the schema emitted by `wormjepa.training.loop._compute_latent_stats`
    (UX spec §"C4 LatentGeometryPanel"). All four arrays are parallel and indexed
    by training step; rows where any of the four fields are missing are dropped.

    `std_per_dim` is rectangular — every row has the same `D`. If the encoder
    were ever re-instantiated mid-run with a different latent dim (it isn't,
    but defensively), `from_log_entries` would skip mismatched rows rather
    than ragged-array.
    """

    steps: list[int]
    std_per_dim: list[list[float]]
    std_min: list[float]
    std_mean: list[float]
    cov_offdiag_frobenius: list[float]

    @property
    def n_dims(self) -> int:
        """Latent dimensionality (rows of the std_per_dim matrix)."""
        return len(self.std_per_dim[0]) if self.std_per_dim else 0

    @property
    def is_empty(self) -> bool:
        """True when no log entry carried the `extra.latent` block."""
        return not self.steps


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_run_directories(results_root: Path) -> list[Path]:
    """Return every immediate subdirectory of `results_root` as a candidate run.

    Filters out hidden dirs (`.foo`) but does not validate the contract —
    that's the caller's job once it tries to read individual files. This
    keeps "no runs yet" indistinguishable from "results root does not exist
    yet" at the top level, which is what the empty-state UX wants.
    """
    if not results_root.is_dir():
        return []
    candidates: list[Path] = []
    for child in results_root.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            candidates.append(child)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def load_run_summary(run_dir: Path) -> RunSummary:
    """Build a `RunSummary` from whatever files exist in `run_dir` right now."""
    seed = _read_seed(run_dir / FILE_SEED)
    config_slug = _read_config_slug(run_dir / FILE_CONFIG)
    compute = load_compute(run_dir)
    git_sha = _read_git_sha(run_dir / FILE_MANIFEST)
    gpu_hours = compute.get("gpu_hours") if compute else None
    if gpu_hours is not None and not isinstance(gpu_hours, (int, float)):
        gpu_hours = None
    return RunSummary(
        run_id=run_dir.name,
        path=run_dir,
        mtime=run_dir.stat().st_mtime,
        seed=seed,
        config_slug=config_slug,
        git_sha=git_sha,
        gpu_hours=float(gpu_hours) if gpu_hours is not None else None,
        has_metrics=(run_dir / FILE_METRICS).is_file(),
        has_report=(run_dir / FILE_REPORT).is_file(),
        has_log=(run_dir / FILE_LOG).is_file(),
    )


# ---------------------------------------------------------------------------
# Per-file loaders
# ---------------------------------------------------------------------------


def load_metrics(run_dir: Path) -> MetricsOutput | None:
    """Parse `<run_dir>/metrics.json` via the canonical pydantic schema.

    Returns `None` if the file does not exist yet (mid-training case) or
    cannot be parsed — the GUI renders a "waiting / unreadable" hint
    rather than crashing.
    """
    path = run_dir / FILE_METRICS
    if not path.is_file():
        return None
    try:
        return MetricsOutput.from_canonical_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("metrics.json unreadable at %s: %s", path, exc)
        return None


def load_compute(run_dir: Path) -> dict[str, Any] | None:
    """Parse `<run_dir>/compute.json` to a raw dict.

    Returns `None` if the file is missing or malformed. We deliberately do
    not depend on the producer's `ComputeProvenance` dataclass — its
    `record_provenance` reaches into `torch.cuda` which is irrelevant for
    a reader and would pull a torch import into the GUI hot path.
    """
    path = run_dir / FILE_COMPUTE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("compute.json unreadable at %s: %s", path, exc)
        return None


def load_log_entries(run_dir: Path, max_lines: int | None = None) -> list[LogEntry]:
    """Stream-parse `<run_dir>/log.jsonl` into a list of `LogEntry`.

    Bad lines are skipped (logged at WARNING) rather than aborting the
    whole load — Streamlit re-renders are frequent and a single torn write
    from a live run should not blank the chart.
    """
    path = run_dir / FILE_LOG
    if not path.is_file():
        return []
    entries: list[LogEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, raw in enumerate(fh):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("log.jsonl L%d in %s is not valid JSON", line_idx + 1, path)
                continue
            if not isinstance(payload, dict):
                continue
            step, loss = _extract_step_loss(payload)
            entries.append(
                LogEntry(line_number=line_idx + 1, payload=payload, step=step, loss=loss)
            )
            if max_lines is not None and len(entries) >= max_lines:
                break
    return entries


def trajectory_from_log(entries: list[LogEntry]) -> tuple[list[int], list[float]]:
    """Project a list of log entries onto (step, loss) parallel arrays.

    Entries without a step+loss pair are dropped. Used by `TrajectoryChart`.
    """
    steps: list[int] = []
    losses: list[float] = []
    for entry in entries:
        if entry.step is not None and entry.loss is not None:
            steps.append(entry.step)
            losses.append(entry.loss)
    return steps, losses


def latent_stats_from_log(entries: list[LogEntry]) -> LatentStatsSeries:
    """Project log entries onto a `LatentStatsSeries` (Phase 2 C4 input).

    Drops entries that lack `extra.latent`. Returns an empty series (all
    lists empty) if no entry carries the block — the GUI surfaces this as
    "no latent-stats data; re-run training with the Phase 2 loop."

    Defensive: if a row's `std_per_dim` width disagrees with the first
    accepted row's width, it is skipped (logged at WARNING). The latent dim
    is fixed at config-time today; this only triggers if the schema drifts.
    """
    steps: list[int] = []
    std_per_dim: list[list[float]] = []
    std_min: list[float] = []
    std_mean: list[float] = []
    cov_offdiag: list[float] = []
    expected_width: int | None = None
    for entry in entries:
        if entry.step is None:
            continue
        latent = _extract_latent_block(entry.payload)
        if latent is None:
            continue
        dims = latent.get("std_per_dim")
        if not isinstance(dims, list):
            continue
        try:
            row = [float(x) for x in dims]
        except (TypeError, ValueError):
            continue
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            logger.warning(
                "log.jsonl L%d: latent_std_per_dim width %d != expected %d; row dropped",
                entry.line_number,
                len(row),
                expected_width,
            )
            continue
        std_min_v = _coerce_float(latent.get("std_min"))
        std_mean_v = _coerce_float(latent.get("std_mean"))
        cov_v = _coerce_float(latent.get("cov_offdiag_frobenius"))
        if std_min_v is None or std_mean_v is None or cov_v is None:
            continue
        steps.append(entry.step)
        std_per_dim.append(row)
        std_min.append(std_min_v)
        std_mean.append(std_mean_v)
        cov_offdiag.append(cov_v)
    return LatentStatsSeries(
        steps=steps,
        std_per_dim=std_per_dim,
        std_min=std_min,
        std_mean=std_mean,
        cov_offdiag_frobenius=cov_offdiag,
    )


def apply_ema_smoothing(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average over `values` with TensorBoard semantics.

    The convention is ``s[i] = alpha * s[i-1] + (1 - alpha) * x[i]`` with
    ``s[0] = x[0]``. `alpha=0` returns the input unchanged; `alpha→1` flattens
    to a constant equal to the first value. `alpha` is clamped to `[0, 0.99]`
    to match the `SmoothingSlider` widget bounds; values outside are silently
    clamped (rather than raising) so a stale URL param can't crash the GUI.

    Returns a parallel list of the same length. `NaN` values are preserved
    (smoothing skips over them so a single bad step doesn't poison the line).
    """
    if not values:
        return []
    alpha = max(0.0, min(0.99, float(alpha)))
    if alpha == 0.0:
        return list(values)
    smoothed: list[float] = []
    prev: float | None = None
    for x in values:
        if x != x:  # NaN check without importing math
            smoothed.append(x)
            continue
        prev = x if prev is None else alpha * prev + (1.0 - alpha) * x
        smoothed.append(prev)
    return smoothed


def latent_norm_trajectory_from_log(
    entries: list[LogEntry],
) -> tuple[list[int], list[float]] | None:
    """Optional: (step, latent_norm) pairs if the log carries them.

    Returns `None` if no log entry exposes a `latent_norm` (or
    `latent_std_min`) field — the chart simply isn't rendered in Phase 1.
    """
    steps: list[int] = []
    norms: list[float] = []
    for entry in entries:
        candidate = _extract_latent_norm(entry.payload)
        if candidate is None or entry.step is None:
            continue
        steps.append(entry.step)
        norms.append(candidate)
    if not steps:
        return None
    return steps, norms


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_seed(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _read_config_slug(path: Path) -> str | None:
    """Heuristic config slug — first line that looks like a `name:` field.

    Avoid a full yaml dependency on the read path for what is a presentation
    string. The config file is the source of truth either way; the slug is
    a label, not a number.
    """
    if not path.is_file():
        return None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("name:"):
                value = stripped.split(":", 1)[1].strip().strip("\"'")
                return value or None
    except OSError:
        return None
    return None


def _read_git_sha(path: Path) -> str | None:
    """Pull the `git_sha_at_lock` value from `manifest_at_run.lock`.

    Same heuristic-parser rationale as `_read_config_slug`.
    """
    if not path.is_file():
        return None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("git_sha_at_lock:"):
                value = stripped.split(":", 1)[1].strip().strip("\"'")
                return value or None
    except OSError:
        return None
    return None


def _extract_step_loss(payload: dict[str, Any]) -> tuple[int | None, float | None]:
    """Pull `step` and `loss` from a JSONL payload.

    The training-loop writer puts `step` at the top level and `loss` under
    `extra`. Older or test fixtures occasionally put both under `extra`. We
    look in both places and combine — top-level wins for `step`, `extra`
    wins for `loss` (which is where the trainer puts it).
    """
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else None
    step = _coerce_int(payload.get("step"))
    if step is None and extra is not None:
        step = _coerce_int(extra.get("step"))
    loss: float | None = None
    if extra is not None:
        loss = _coerce_float(extra.get("loss"))
    if loss is None:
        loss = _coerce_float(payload.get("loss"))
    return step, loss


def _extract_latent_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the `extra.latent` sub-block from a JSONL payload.

    Returns `None` if the payload does not carry it (e.g., an older run
    written before the Phase 2 latent-stats addition to the training loop).
    """
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        return None
    latent = extra.get("latent")
    if not isinstance(latent, dict):
        return None
    return latent


def _extract_latent_norm(payload: dict[str, Any]) -> float | None:
    """Pull a latent-norm-like value from a JSONL payload.

    Accepts any of `latent_norm`, `latent_std_min`, `latent_std_mean` for
    Phase 1. Live-tail GUIs render whichever of these the trainer happens
    to log, no schema negotiation required.
    """
    extra = payload.get("extra")
    candidates = ("latent_norm", "latent_std_min", "latent_std_mean")
    if isinstance(extra, dict):
        for key in candidates:
            value = _coerce_float(extra.get(key))
            if value is not None:
                return value
    for key in candidates:
        value = _coerce_float(payload.get(key))
        if value is not None:
            return value
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
