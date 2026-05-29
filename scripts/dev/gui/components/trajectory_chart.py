"""C2 — TrajectoryChart (single-run variant) component.

Phase 1 renders a Plotly line chart of (training step → scalar) for one
run. Hover-emit-to-session-state and cross-run-overlay variants are Phase 2
/ Phase 3 items.

Every chart is paired with an `st.caption` that summarises the current
state in plain language (§"Alt text — `st.caption` below every chart").
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from scripts.dev.gui import data, state, theme

# Opacity of the raw (unsmoothed) line when a smoothing overlay is drawn.
# TensorBoard renders the raw series faintly behind the smoothed one; this
# matches that convention.
RAW_LINE_OPACITY_WITH_SMOOTHING: float = 0.25


def render_trajectory_chart(
    run_id: str,
    label: str,
    steps: Sequence[int],
    values: Sequence[float],
    color: str | None = None,
    y_axis_title: str = "value",
    key: str | None = None,
) -> None:
    """Render a single-run trajectory chart with plain-language caption.

    Args:
        run_id: Run identifier (used in the caption).
        label: Human-readable label for the series (e.g., "Loss",
            "Latent std min").
        steps: Training-step x values.
        values: Scalar y values, parallel to `steps`.
        color: Hex color for the line; defaults to the semantic accent color.
        y_axis_title: Plotly y-axis label.
        key: Optional `st.plotly_chart` key for stable widget IDs.
    """
    if not steps or not values or len(steps) != len(values):
        st.caption(f"{label} for run {run_id}: no data yet.")
        return

    alpha = state.get_smoothing()
    smoothed = data.apply_ema_smoothing(list(values), alpha) if alpha > 0.0 else None

    fig = _build_figure(
        label=label,
        steps=steps,
        values=values,
        color=color or theme.COLOR_ACCENT,
        y_axis_title=y_axis_title,
        smoothed=smoothed,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    caption_tail = f" (EMA alpha={alpha:.2f})" if smoothed is not None else ""
    st.caption(
        f"{label} for run {run_id} across {len(steps)} steps, "
        f"currently {values[-1]:.4g} at step {steps[-1]}.{caption_tail}"
    )


def render_trajectory_chart_overlay(
    series: list[tuple[str, Sequence[int], Sequence[float], str]],
    label: str,
    y_axis_title: str,
    key: str,
) -> None:
    """Render a cross-run overlay: one trace per run on a single figure.

    Args:
        series: List of ``(run_id, steps, values, color)`` tuples. Empty
            tuples (no data yet) are skipped silently.
        label: Series label used in tooltips and the caption.
        y_axis_title: Plotly y-axis label.
        key: Stable widget key.

    Smoothing slider value applies to every trace (TensorBoard convention).
    """
    drawable = [
        (rid, list(steps), list(values), color)
        for rid, steps, values, color in series
        if steps and values and len(steps) == len(values)
    ]
    if not drawable:
        st.caption(f"{label}: no overlay data yet across the selected runs.")
        return

    alpha = state.get_smoothing()
    fig = go.Figure()
    summary_bits: list[str] = []
    for run_id, steps, values, color in drawable:
        smoothed = data.apply_ema_smoothing(values, alpha) if alpha > 0.0 else None
        raw_opacity = RAW_LINE_OPACITY_WITH_SMOOTHING if smoothed is not None else 1.0
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 4, "color": color},
                opacity=raw_opacity,
                name=f"{run_id}" + (" (raw)" if smoothed is not None else ""),
                hovertemplate=run_id + " · step %{x}<br>" + label + " %{y:.4g}<extra></extra>",
            )
        )
        if smoothed is not None:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=smoothed,
                    mode="lines",
                    line={"color": color, "width": 3},
                    name=f"{run_id} (EMA)",
                    hovertemplate=run_id + " · EMA %{y:.4g}<extra></extra>",
                )
            )
        summary_bits.append(f"{run_id}={values[-1]:.4g} at step {steps[-1]}")

    fig.update_layout(
        autosize=True,
        height=300,
        margin={"l": 48, "r": 16, "t": 16, "b": 32},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        plot_bgcolor=theme.COLOR_SURFACE,
        paper_bgcolor=theme.COLOR_BACKGROUND,
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
        xaxis={
            "title": "training step",
            "showgrid": True,
            "gridcolor": theme.COLOR_BORDER,
            "zeroline": False,
        },
        yaxis={
            "title": y_axis_title,
            "showgrid": True,
            "gridcolor": theme.COLOR_BORDER,
            "zeroline": False,
        },
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    tail = f" (EMA alpha={alpha:.2f})" if alpha > 0.0 else ""
    st.caption(f"{label} across {len(drawable)} runs — " + "; ".join(summary_bits) + tail + ".")


def _build_figure(
    label: str,
    steps: Sequence[int],
    values: Sequence[float],
    color: str,
    y_axis_title: str,
    smoothed: Sequence[float] | None = None,
) -> go.Figure:
    """Build the Plotly figure with consistent theme tokens.

    When `smoothed` is supplied the raw line is rendered faintly behind the
    smoothed line (TensorBoard convention). When `smoothed` is `None` the
    raw line is rendered at full opacity as before.
    """
    raw_opacity = RAW_LINE_OPACITY_WITH_SMOOTHING if smoothed is not None else 1.0
    traces: list[go.Scatter] = [
        go.Scatter(
            x=list(steps),
            y=list(values),
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 4, "color": color},
            opacity=raw_opacity,
            name=label + (" (raw)" if smoothed is not None else ""),
            hovertemplate="step %{x}<br>" + label + " %{y:.4g}<extra></extra>",
        )
    ]
    if smoothed is not None:
        traces.append(
            go.Scatter(
                x=list(steps),
                y=list(smoothed),
                mode="lines",
                line={"color": color, "width": 3},
                name=label + " (EMA)",
                hovertemplate="step %{x}<br>EMA %{y:.4g}<extra></extra>",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=True,
        height=260,
        margin={"l": 48, "r": 16, "t": 16, "b": 32},
        showlegend=smoothed is not None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        plot_bgcolor=theme.COLOR_SURFACE,
        paper_bgcolor=theme.COLOR_BACKGROUND,
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
        xaxis={
            "title": "training step",
            "showgrid": True,
            "gridcolor": theme.COLOR_BORDER,
            "zeroline": False,
        },
        yaxis={
            "title": y_axis_title,
            "showgrid": True,
            "gridcolor": theme.COLOR_BORDER,
            "zeroline": False,
        },
    )
    return fig
