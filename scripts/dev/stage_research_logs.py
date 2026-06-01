"""Stage the committed collapse-research logs into a results/ layout so the
dev-loop GUI can render them.

The collapse study's per-step logs live as flat files under
`_bmad-output/implementation-artifacts/collapse-research-logs/*.jsonl`.
The dev-loop GUI (`scripts/dev/gui/main.py`) reads the
`results/<run-id>/log.jsonl` contract. This script copies each research
log into `results/collapse-research/<EN-label>/log.jsonl` plus a minimal
`config.yaml` / `seed.txt` so the GUI shows a readable run label.

Usage::

    uv run python scripts/dev/stage_research_logs.py
    uv run streamlit run scripts/dev/gui/main.py -- results/collapse-research/

Idempotent — re-running overwrites the staged copies.
"""
# ruff: noqa: T201

from __future__ import annotations

import shutil
import sys

from wormjepa.paths import project_root

# (research-log stem, GUI run-id label) — ordered E1..E7.
EXPERIMENTS: tuple[tuple[str, str], ...] = (
    ("jepa_debug", "E1-bare-pure-jepa"),
    ("jepa_debug_e2", "E2-variance-reg"),
    ("jepa_debug_e3", "E3-covariance-reg"),
    ("jepa_debug_e4", "E4-rebalanced-RECIPE"),
    ("jepa_debug_e5", "E5-warm-start-heads-on"),
    ("jepa_debug_e6", "E6-heads-0.1x"),
    ("jepa_debug_e7", "E7-curriculum"),
)


def main() -> None:
    """Stage every research log into results/collapse-research/."""
    root = project_root()
    src_dir = root / "_bmad-output" / "implementation-artifacts" / "collapse-research-logs"
    if not src_dir.is_dir():
        print(
            "stage_research_logs.py: required source dir not found.\n"
            f"  expected: {src_dir}\n\n"
            "This is an author-tooling dev script that depends on a local-only "
            "workspace directory (`_bmad-output/implementation-artifacts/"
            "collapse-research-logs/`) which is gitignored and therefore absent "
            "from the public repository. Public users should consult REPRODUCE.md "
            "for the supported collapse-study reproduction path; the raw per-step "
            "research logs this script stages are not part of the public artifact "
            "set.",
            file=sys.stderr,
        )
        sys.exit(78)  # sysexits.h: EX_CONFIG
    dst_root = root / "results" / "collapse-research"
    dst_root.mkdir(parents=True, exist_ok=True)

    staged = 0
    for stem, label in EXPERIMENTS:
        src = src_dir / f"{stem}.jsonl"
        if not src.is_file():
            print(f"  skip {label}: {src.name} not found")
            continue
        run_dir = dst_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, run_dir / "log.jsonl")
        (run_dir / "config.yaml").write_text(
            f"# staged collapse-research run\nconfig_name: {stem}\n", encoding="utf-8"
        )
        (run_dir / "seed.txt").write_text("42\n", encoding="utf-8")
        staged += 1
        print(f"  staged {label}: {(run_dir / 'log.jsonl')}")

    print(f"\n{staged} run(s) staged at {dst_root}")
    print("View in the GUI:")
    print(f"  uv run streamlit run scripts/dev/gui/main.py -- {dst_root}/")


if __name__ == "__main__":
    main()
