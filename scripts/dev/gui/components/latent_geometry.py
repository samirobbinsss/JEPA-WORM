"""C4 — LatentGeometryPanel component.

Phase 2 visualises encoder latent-collapse risk (Growth G1 / VICReg signal
per the UX spec §"C4 LatentGeometryPanel"). The panel reads the per-step
`extra.latent.{std_per_dim, std_min, std_mean, cov_offdiag_frobenius}` block
that `wormjepa.training.loop` writes to `log.jsonl` and renders three
stacked sub-panels:

1. Per-dim std heatmap over time (x = step, y = dim, color = std).
2. Diagonal std-vs-step line chart (std_min in critical-red zone,
   std_mean as the research-blue primary line).
3. Off-diagonal covariance Frobenius norm over step.

A verdict caption at the top summarises the panel's state in words (a11y +
"never make the user reason from a chart alone" per §"Feedback Patterns").

Implementation note: the spec calls for `st.pyplot` with matplotlib for the
heatmap. The project's `pyproject.toml` does not depend on matplotlib (and
is off-limits for this story), so we render the heatmap with Plotly's
`go.Heatmap` instead. Visually equivalent for the panel's purpose and
keeps a single chart library throughout the GUI.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.dev.gui import cache, data, state, theme

# Thresholds tied to the verdict colors (UX spec §"Feedback Patterns →
# Verdict pattern"). Tunable per environment if real-data collapse profiles
# differ — kept as module-level constants so a future override is one edit.
STD_HEALTHY_THRESHOLD: float = 0.05
STD_CRITICAL_THRESHOLD: float = 0.01


def render_latent_geometry_panel(run_dir: Path | None, run_id: str | None) -> None:
    """Render the C4 LatentGeometryPanel for the currently-selected run.

    No-op-with-caption pattern when:
    - No run is selected.
    - The run's log.jsonl lacks the `extra.latent` block (older runs before
      the Phase 2 latent-stats addition to the training loop).
    """
    st.markdown("##### Latent geometry")
    if run_dir is None or run_id is None:
        st.caption("Select a run to view its encoder latent-collapse diagnostics.")
        return

    entries = cache.log_entries(run_dir)
    if not entries:
        st.caption("No `log.jsonl` entries yet — waiting for the first step.")
        return

    series = data.latent_stats_from_log(entries)
    if series.is_empty:
        st.caption(
            "No `extra.latent.*` fields found in `log.jsonl`. Re-run training "
            "with the Phase 2 training loop (latent stats logging gated on "
            "`src/wormjepa/training/loop.py`)."
        )
        return

    _render_verdict_caption(series)
    _render_std_heatmap(series, key=f"gui.latent.heatmap.{run_id}")
    _render_std_diagonal(series, run_id=run_id, key=f"gui.latent.std.{run_id}")
    _render_cov_offdiag(series, run_id=run_id, key=f"gui.latent.cov.{run_id}")


def _verdict(series: data.LatentStatsSeries) -> tuple[str, str, str]:
    """Categorise the latest std_min into healthy / warning / critical.

    Returns `(color, label, message)` where `color` is a theme hex.
    """
    n_dims = series.n_dims
    latest_min = series.std_min[-1]
    active_dims = sum(1 for v in series.std_per_dim[-1] if v >= STD_CRITICAL_THRESHOLD)
    if latest_min < STD_CRITICAL_THRESHOLD:
        return (
            theme.COLOR_CRITICAL,
            "critical",
            f"lowest std {latest_min:.4f} below {STD_CRITICAL_THRESHOLD:g}; "
            f"encoder near-collapsed ({active_dims}/{n_dims} dims active).",
        )
    if latest_min < STD_HEALTHY_THRESHOLD:
        return (
            theme.COLOR_WARNING,
            "warning",
            f"lowest std {latest_min:.4f} trending toward 0 (threshold "
            f"{STD_HEALTHY_THRESHOLD:g}); check encoder collapse "
            f"({active_dims}/{n_dims} dims active).",
        )
    return (
        theme.COLOR_HEALTHY,
        "healthy",
        f"{active_dims}/{n_dims} dims active; lowest std {latest_min:.4f} "
        f"(above {STD_HEALTHY_THRESHOLD:g} threshold).",
    )


def _render_verdict_caption(series: data.LatentStatsSeries) -> None:
    """Render a colored verdict badge above the charts.

    Pairs an icon with a text label (§"Accessibility — color paired with
    icons, never color-alone") and a one-line plain-language description.
    """
    color, label, message = _verdict(series)
    icon = {"healthy": "✓", "warning": "⚠", "critical": "✗"}[label]
    st.markdown(
        f'<div role="status" aria-live="polite" '
        f'style="color:{color}; font-family:{theme.FONT_SANS}; '
        f"font-size:{theme.FONT_SIZE_BODY_PX}px; font-weight:600; "
        f'padding:{theme.SPACE_SM_PX}px 0;">'
        f"{icon} {label}: {message}"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_std_heatmap(series: data.LatentStatsSeries, key: str) -> None:
    """Render the per-dim std heatmap (x = step, y = dim, color = std)."""
    # Transpose: rows = dims, columns = steps so y axis stacks dims.
    n_steps = len(series.steps)
    n_dims = series.n_dims
    matrix: list[list[float]] = [
        [series.std_per_dim[s][d] for s in range(n_steps)] for d in range(n_dims)
    ]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=series.steps,
                y=list(range(n_dims)),
                colorscale="Viridis",
                colorbar={"title": "std"},
                hovertemplate=("step %{x}<br>dim %{y}<br>std %{z:.4g}<extra></extra>"),
            )
        ]
    )
    fig.update_layout(
        autosize=True,
        height=220,
        margin={"l": 48, "r": 16, "t": 16, "b": 32},
        plot_bgcolor=theme.COLOR_SURFACE,
        paper_bgcolor=theme.COLOR_BACKGROUND,
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
        xaxis={"title": "training step", "gridcolor": theme.COLOR_BORDER},
        yaxis={"title": "latent dim", "gridcolor": theme.COLOR_BORDER},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        f"Per-dim latent std across {n_steps} steps x {n_dims} dims. "
        "Dark cells = low std (a dim the encoder has stopped using)."
    )


def _render_std_diagonal(series: data.LatentStatsSeries, run_id: str, key: str) -> None:
    """Render the std_min + std_mean line chart with a healthy-threshold band."""
    alpha = state.get_smoothing()
    raw_min = list(series.std_min)
    raw_mean = list(series.std_mean)
    smooth_min = data.apply_ema_smoothing(raw_min, alpha) if alpha > 0.0 else None
    smooth_mean = data.apply_ema_smoothing(raw_mean, alpha) if alpha > 0.0 else None

    fig = go.Figure()
    # Healthy/critical threshold shading. `add_hrect` is the simplest way to
    # call attention to the danger zone without obscuring the lines.
    fig.add_hrect(
        y0=0.0,
        y1=STD_CRITICAL_THRESHOLD,
        line_width=0,
        fillcolor=theme.COLOR_CRITICAL,
        opacity=0.08,
    )
    fig.add_hrect(
        y0=STD_CRITICAL_THRESHOLD,
        y1=STD_HEALTHY_THRESHOLD,
        line_width=0,
        fillcolor=theme.COLOR_WARNING,
        opacity=0.06,
    )

    raw_opacity = 0.25 if smooth_min is not None else 1.0
    fig.add_trace(
        go.Scatter(
            x=series.steps,
            y=raw_min,
            mode="lines+markers",
            line={"color": theme.COLOR_CRITICAL, "width": 2},
            marker={"size": 4, "color": theme.COLOR_CRITICAL},
            opacity=raw_opacity,
            name="std_min" + (" (raw)" if smooth_min is not None else ""),
            hovertemplate="step %{x}<br>std_min %{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series.steps,
            y=raw_mean,
            mode="lines+markers",
            line={"color": theme.COLOR_ACCENT, "width": 2},
            marker={"size": 4, "color": theme.COLOR_ACCENT},
            opacity=raw_opacity,
            name="std_mean" + (" (raw)" if smooth_mean is not None else ""),
            hovertemplate="step %{x}<br>std_mean %{y:.4g}<extra></extra>",
        )
    )
    if smooth_min is not None:
        fig.add_trace(
            go.Scatter(
                x=series.steps,
                y=smooth_min,
                mode="lines",
                line={"color": theme.COLOR_CRITICAL, "width": 3},
                name="std_min (EMA)",
                hovertemplate="step %{x}<br>EMA %{y:.4g}<extra></extra>",
            )
        )
    if smooth_mean is not None:
        fig.add_trace(
            go.Scatter(
                x=series.steps,
                y=smooth_mean,
                mode="lines",
                line={"color": theme.COLOR_ACCENT, "width": 3},
                name="std_mean (EMA)",
                hovertemplate="step %{x}<br>EMA %{y:.4g}<extra></extra>",
            )
        )

    fig.update_layout(
        autosize=True,
        height=240,
        margin={"l": 48, "r": 16, "t": 16, "b": 32},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        plot_bgcolor=theme.COLOR_SURFACE,
        paper_bgcolor=theme.COLOR_BACKGROUND,
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
        xaxis={"title": "training step", "gridcolor": theme.COLOR_BORDER},
        yaxis={"title": "latent std", "gridcolor": theme.COLOR_BORDER, "rangemode": "tozero"},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        f"std_min (red) and std_mean (blue) for run {run_id}. "
        f"Amber zone = [{STD_CRITICAL_THRESHOLD:g}, {STD_HEALTHY_THRESHOLD:g}], "
        f"red zone < {STD_CRITICAL_THRESHOLD:g}."
    )


def _render_cov_offdiag(series: data.LatentStatsSeries, run_id: str, key: str) -> None:
    """Render the off-diagonal covariance Frobenius norm over step."""
    alpha = state.get_smoothing()
    raw = list(series.cov_offdiag_frobenius)
    smooth = data.apply_ema_smoothing(raw, alpha) if alpha > 0.0 else None

    fig = go.Figure()
    raw_opacity = 0.25 if smooth is not None else 1.0
    fig.add_trace(
        go.Scatter(
            x=series.steps,
            y=raw,
            mode="lines+markers",
            line={"color": theme.COLOR_NEUTRAL, "width": 2},
            marker={"size": 4, "color": theme.COLOR_NEUTRAL},
            opacity=raw_opacity,
            name="‖Σ_off‖_F" + (" (raw)" if smooth is not None else ""),
            hovertemplate="step %{x}<br>‖Σ_off‖_F %{y:.4g}<extra></extra>",
        )
    )
    if smooth is not None:
        fig.add_trace(
            go.Scatter(
                x=series.steps,
                y=smooth,
                mode="lines",
                line={"color": theme.COLOR_NEUTRAL, "width": 3},
                name="‖Σ_off‖_F (EMA)",
                hovertemplate="step %{x}<br>EMA %{y:.4g}<extra></extra>",
            )
        )

    fig.update_layout(
        autosize=True,
        height=220,
        margin={"l": 48, "r": 16, "t": 16, "b": 32},
        showlegend=smooth is not None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        plot_bgcolor=theme.COLOR_SURFACE,
        paper_bgcolor=theme.COLOR_BACKGROUND,
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
        xaxis={"title": "training step", "gridcolor": theme.COLOR_BORDER},
        yaxis={"title": "off-diag Frobenius", "gridcolor": theme.COLOR_BORDER},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        f"Off-diagonal covariance Frobenius norm for run {run_id}. "
        "High values = redundant dims (VICReg decorrelation signal); "
        "rising in tandem with falling std_min flags collapse."
    )
