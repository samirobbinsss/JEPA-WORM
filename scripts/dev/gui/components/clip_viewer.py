"""C3 — ClipViewer component (Phase 2, single-run, no-pose variant).

The Phase 0 training loop runs on synthetic data and does not write any
clips to disk; the spec's hover-linked clip + mask overlay + per-frame
distance heatmap therefore has nothing to render for the smoke runs that
land in `results/` today. The component still occupies the top of the
right column so the layout reads as the planned 3-column cockpit, and it
swaps in the rich content the moment a future run writes clips into
`results/<run-id>/clips/`.

Three states (UX spec §"Empty and Loading States" + §"2.5 Experience
Mechanics"):

1. *No clip data* — common case: render the documented "no clip data"
   panel + a caption explaining that clip storage is gated on Story 8.3+
   real-data wiring.
2. *Clip data present + step pinned* — render the clip for that step
   (HTML5 `<video>` via `st.video` for `.mp4`; `st.image` for a single
   `.png` representative frame). Mask overlay (`<step>.mask.png` next
   to the clip) is composited if present.
3. *Clip data present + no step pinned* — render the most recent step's
   clip (auto-follow head).

The pinned step is read from `st.session_state["gui.pinned_step"]` so
when a future hover-pin gesture lands, this component automatically
reacts without prop drilling.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from scripts.dev.gui import state, theme
from scripts.dev.gui.components import empty_state

# Filename suffixes recognised as clip media. Order matters: when both an
# .mp4 and .png exist for the same step, the MP4 wins (richer view).
_VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".webm")
_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
_MASK_SUFFIX: str = ".mask.png"

CLIPS_SUBDIR: str = "clips"

# Filename for the training-evolution rollup MP4 emitted by
# :class:`wormjepa.training.clip_writer.RolloutRecorder`. When present, the
# ClipViewer renders it at the top of the panel so the watcher's first
# read is the convergence story across all training steps.
ROLLUP_FILENAME: str = "training_evolution.mp4"


def render_clip_viewer(run_dir: Path | None, run_id: str | None) -> None:
    """Render the clip viewer at the top of the right column.

    Dispatches between the three states based on what's on disk and what's
    pinned in session state. Always renders something (never a blank box)
    so the layout reads consistently.
    """
    st.markdown("##### Clip viewer")
    if run_dir is None or run_id is None:
        st.caption("Select a run to view its training clips (when available).")
        return

    clips_dir = run_dir / CLIPS_SUBDIR
    rollup_path = clips_dir / ROLLUP_FILENAME
    if rollup_path.is_file():
        st.markdown("**Training-evolution rollup** — predicted pose (red) vs ground truth (green)")
        st.video(str(rollup_path))
        st.caption(f"One frame per training step on the fixed reference clip · {rollup_path.name}")
    available = _list_available_clips(clips_dir)

    if not available:
        if not rollup_path.is_file():
            _render_no_clip_data_state()
        return

    pinned = state.get_pinned_step()
    chosen_step = _resolve_chosen_step(pinned, available)
    if chosen_step is None:
        # Should not happen: `available` is non-empty so the resolver picks one.
        _render_no_clip_data_state()
        return

    clip_path = available[chosen_step]
    pin_label = "pinned" if pinned is not None and pinned in available else "head"
    _render_clip(clip_path, run_id=run_id, step=chosen_step, pin_label=pin_label)


def _list_available_clips(clips_dir: Path) -> dict[int, Path]:
    """Map step → preferred clip path under `clips_dir`.

    Filename convention: `<step>.<ext>`. Conflicts pick the highest-priority
    extension via `_VIDEO_SUFFIXES + _IMAGE_SUFFIXES` order. Mask sidecars
    (`<step>.mask.png`) are excluded from the dispatch list — `_render_clip`
    finds them on demand.

    Returns an empty dict if the directory is missing, empty, or contains
    no parseable filenames.
    """
    if not clips_dir.is_dir():
        return {}
    candidates: dict[int, list[Path]] = {}
    for path in clips_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(_MASK_SUFFIX):
            continue
        suffix = path.suffix.lower()
        if suffix not in (_VIDEO_SUFFIXES + _IMAGE_SUFFIXES):
            continue
        try:
            step = int(path.stem)
        except ValueError:
            continue
        candidates.setdefault(step, []).append(path)

    priority = _VIDEO_SUFFIXES + _IMAGE_SUFFIXES
    resolved: dict[int, Path] = {}
    for step, paths in candidates.items():
        paths.sort(key=lambda p: priority.index(p.suffix.lower()))
        resolved[step] = paths[0]
    return resolved


def _resolve_chosen_step(pinned: int | None, available: dict[int, Path]) -> int | None:
    """Pick which step's clip to display.

    Order of preference:
    1. The pinned step, if a clip exists for it.
    2. The most recent (max) step among the available clips.
    """
    if pinned is not None and pinned in available:
        return pinned
    if not available:
        return None
    return max(available)


def _render_no_clip_data_state() -> None:
    """The common case for Phase 0 / synthetic-data runs."""
    empty_state.render_empty_state(
        title="No clip data for this run",
        message=(
            "The training run did not write any clips to "
            f"`{CLIPS_SUBDIR}/`. The hover-linked clip viewer is gated on "
            "real-data wiring (Story 8.3+); synthetic smoke runs do not "
            "persist video frames."
        ),
        variant="waiting",
        icon="🎬",
        hint="Re-open this panel once a run with real video clips lands.",
    )


def _render_clip(clip_path: Path, run_id: str, step: int, pin_label: str) -> None:
    """Render the clip at `clip_path` plus a mask overlay if available.

    Mask overlay logic is simple by design: if a sibling `<step>.mask.png`
    exists, render it as a second image below the clip with a caption
    explaining the magenta-alpha encoding. Compositing the mask *onto* the
    frame would need an image library (Pillow / matplotlib) that this
    phase does not require; the side-by-side rendering is the documented
    no-pose-variant fallback.
    """
    suffix = clip_path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        st.video(str(clip_path))
    else:
        st.image(str(clip_path), use_container_width=True)

    mask_path = clip_path.with_suffix("").with_suffix(_MASK_SUFFIX)
    if not mask_path.exists():
        # Try `<stem>.mask.png` for the case where `.with_suffix` produced
        # something unexpected (e.g., the stem already carried a `.mask`).
        mask_path = clip_path.parent / f"{clip_path.stem}{_MASK_SUFFIX}"
    if mask_path.is_file():
        st.image(
            str(mask_path),
            caption="Mask overlay (magenta alpha=0.35 over masked frames)",
            use_container_width=True,
        )

    st.markdown(
        f'<div style="color:{theme.COLOR_NEUTRAL}; font-family:monospace; '
        f'font-size:{theme.FONT_SIZE_CAPTION_PX}px;">'
        f"run {run_id} · step {step} ({pin_label}) · {clip_path.name}"
        "</div>",
        unsafe_allow_html=True,
    )
