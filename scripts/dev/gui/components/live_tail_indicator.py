"""C8 — LiveTailIndicator component.

Status dot + "Last update: 3s ago" text. The dot is rendered via colored
text (no animation per §"Motion: no auto-play, no auto-scroll, no animation
on data updates") because Streamlit's stock components don't support a
pulsing dot, and CSS injection is forbidden beyond the theme tokens.

States (from the UX spec):
- live + recent: bright accent dot, "Last update: Ns ago"
- live + stale:  dim warning dot, "Stale (Ns ago)"
- static:        gray dot, "Loaded at <UTC>"
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from scripts.dev.gui import theme

STALE_THRESHOLD_SECONDS: float = 30.0


def render_live_tail_indicator(
    live: bool,
    seconds_since_update: float | None,
    loaded_at: datetime | None = None,
) -> None:
    """Render the live-tail status indicator.

    Args:
        live: Whether live-tail is currently enabled.
        seconds_since_update: Wall-clock seconds since the last detected
            filesystem event. `None` when no event has been observed yet.
        loaded_at: UTC datetime captured at static-load time. Required only
            when `live=False`; ignored otherwise.
    """
    color, dot, label = _state_for(live, seconds_since_update, loaded_at)
    # role=status + aria-live=polite via st.markdown for screen-reader friendliness.
    st.markdown(
        f'<span role="status" aria-live="polite" '
        f'style="color:{color}; font-family:monospace; font-size:13px;">'
        f"{dot}&nbsp;{label}</span>",
        unsafe_allow_html=True,
    )


def _state_for(
    live: bool,
    seconds_since_update: float | None,
    loaded_at: datetime | None,
) -> tuple[str, str, str]:
    """Compute (color, dot-glyph, label) for the given mode + freshness."""
    if not live:
        when = loaded_at or datetime.now(tz=UTC)
        return theme.COLOR_NEUTRAL, "○", f"Static · loaded {when.strftime('%H:%M:%S UTC')}"
    if seconds_since_update is None:
        return theme.COLOR_ACCENT, "●", "Live · waiting for first update"
    if seconds_since_update > STALE_THRESHOLD_SECONDS:
        return theme.COLOR_WARNING, "●", f"Live · stale ({_fmt_seconds(seconds_since_update)} ago)"
    return theme.COLOR_HEALTHY, "●", f"Live · last update {_fmt_seconds(seconds_since_update)} ago"


def _fmt_seconds(seconds: float) -> str:
    """Format seconds for human display: '3s', '12s', '4m'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"
