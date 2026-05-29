"""Theme tokens for the dev-loop inspection GUI.

Mirrors `_bmad-output/planning-artifacts/ux-design-specification.md`
§"Visual Design Foundation" so the Python code path can reference the
same constants the Streamlit theme file declares. Kept as plain constants
(no dataclass) because the values are read-only and the dependency footprint
should stay minimal.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Semantic palette (§"Color System")
# ---------------------------------------------------------------------------

COLOR_HEALTHY: Final[str] = "#2E7D32"
COLOR_WARNING: Final[str] = "#E68A00"
COLOR_CRITICAL: Final[str] = "#C62828"
COLOR_NEUTRAL: Final[str] = "#424242"
COLOR_ACCENT: Final[str] = "#1565C0"
COLOR_BACKGROUND: Final[str] = "#FAFAFA"
COLOR_SURFACE: Final[str] = "#FFFFFF"
COLOR_BORDER: Final[str] = "#E0E0E0"
COLOR_MASK_OVERLAY: Final[str] = "rgba(216,27,96,0.35)"

# ---------------------------------------------------------------------------
# Tableau-colorblind10 categorical palette for run-id encoding
# (§"Run-id categorical palette"). Order is the matplotlib reference order.
# ---------------------------------------------------------------------------

TABLEAU_COLORBLIND_10: Final[tuple[str, ...]] = (
    "#006BA4",
    "#FF800E",
    "#ABABAB",
    "#595959",
    "#5F9ED1",
    "#C85200",
    "#898989",
    "#A2C8EC",
    "#FFBC79",
    "#CFCFCF",
)

# ---------------------------------------------------------------------------
# Typography (§"Typography System")
# ---------------------------------------------------------------------------

FONT_SANS: Final[str] = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
FONT_MONO: Final[str] = '"JetBrains Mono", Menlo, monospace'

FONT_SIZE_H1_PX: Final[int] = 32
FONT_SIZE_H2_PX: Final[int] = 22
FONT_SIZE_H3_PX: Final[int] = 18
FONT_SIZE_BODY_PX: Final[int] = 16
FONT_SIZE_CAPTION_PX: Final[int] = 13
FONT_SIZE_DATA_PX: Final[int] = 14

LINE_HEIGHT_BODY: Final[float] = 1.5
LINE_HEIGHT_HEADING: Final[float] = 1.3

# ---------------------------------------------------------------------------
# Spacing scale, base 8 (§"Spacing & Layout Foundation")
# ---------------------------------------------------------------------------

SPACE_XS_PX: Final[int] = 4
SPACE_SM_PX: Final[int] = 8
SPACE_MD_PX: Final[int] = 16
SPACE_LG_PX: Final[int] = 24
SPACE_XL_PX: Final[int] = 32
SPACE_2XL_PX: Final[int] = 48
SPACE_3XL_PX: Final[int] = 64

# ---------------------------------------------------------------------------
# Layout (§"Design Direction Decision" — three-column cockpit)
# ---------------------------------------------------------------------------

COCKPIT_COLUMN_RATIOS: Final[tuple[int, int, int]] = (2, 5, 5)


def run_color(index: int) -> str:
    """Return a stable color for a run, cycling tableau-colorblind10."""
    if index < 0:
        msg = f"run_color index must be non-negative; got {index}"
        raise ValueError(msg)
    return TABLEAU_COLORBLIND_10[index % len(TABLEAU_COLORBLIND_10)]
