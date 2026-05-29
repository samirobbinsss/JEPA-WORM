"""C10 — SmoothingSlider component.

Phase 2 renders an EMA smoothing slider above the middle-column trajectory
charts (UX spec §"C10 SmoothingSlider"). The chosen value persists in
`st.session_state["gui.smoothing"]` via `state.set_smoothing`; downstream
charts (TrajectoryChart, LatentGeometryPanel) read it and overlay a
smoothed line on top of the raw series.

Semantics follow the TensorBoard convention:

    s[i] = alpha * s[i-1] + (1 - alpha) * x[i],  s[0] = x[0]

with `alpha = 0` returning the raw input and `alpha → 0.99` aggressively
smoothing. The math itself lives in `data.apply_ema_smoothing` so non-GUI
contexts (tests, future CLI exports) can reuse it.
"""

from __future__ import annotations

import streamlit as st

from scripts.dev.gui import state


def render_smoothing_slider(key: str = "gui.smoothing.widget") -> float:
    """Render the smoothing slider and return its current value.

    The widget writes its value to `st.session_state[state.KEY_SMOOTHING]`
    so any component re-rendering in the same Streamlit run sees the new
    alpha without prop drilling.

    Args:
        key: Streamlit widget key; tests can pass distinct keys when
            instantiating the component twice in one page.

    Returns:
        The chosen smoothing factor in ``[SMOOTHING_MIN, SMOOTHING_MAX]``.
    """
    current = state.get_smoothing()
    chosen = st.slider(
        "Smoothing (EMA alpha)",
        min_value=state.SMOOTHING_MIN,
        max_value=state.SMOOTHING_MAX,
        value=current,
        step=0.01,
        key=key,
        help=(
            "EMA over plotted lines. 0 = raw trajectory, higher = smoother. "
            "Tendency: 0.6-0.9 reveals macro trends; > 0.95 erases them."
        ),
    )
    state.set_smoothing(float(chosen))
    return float(chosen)
