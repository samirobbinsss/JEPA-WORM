"""One-shot downloader for WormID DANDI dandisets (Story 8.12c pre-flight).

Pulls a single dandiset (at the version pinned in
`src/wormjepa/data/sources/wormid.py`) into
`data/downloads/wormid/<dandiset_id>/` so the `WormIDLoader` can iterate
it. Re-runnable: existing files at the expected size are skipped.

Usage:
    uv run python scripts/dev/fetch_wormid_cohort.py 000714
    uv run python scripts/dev/fetch_wormid_cohort.py 000692 --resume

Not a public CLI subcommand. A real `wormjepa fetch wormid` would
deserve its own story; this script is the cheapest viable shim for
Story 8.12c's data prerequisite.

SF (000776) is deferred to Phase 0 Growth per the 2026-05-18
materialization deferral (see CHANGELOG + splits/wormid_train_eval.yaml).
This script refuses to fetch 000776 unless --override-sf-deferral is
passed.
"""

# ruff: noqa: T201
# T201 (print): this is a CLI script; print is the intended output channel.
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from wormjepa.data.sources.wormid import SPEC

_API = "https://api.dandiarchive.org/api"


def _list_assets(dandiset_id: str, version: str) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    url = f"{_API}/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=200"
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        assets.extend(page["results"])
        url = page.get("next")
    return assets


def _download_one(
    asset: dict[str, Any],
    dest_dir: Path,
    dandiset_id: str,
    version: str,
) -> tuple[bool, int]:
    """Return (was_new_download, bytes_written). Skips if file already correct size."""
    asset_path = str(asset["path"])
    asset_size = int(asset["size"])
    asset_id = str(asset["asset_id"])
    dest = dest_dir / asset_path
    if dest.exists() and dest.stat().st_size == asset_size:
        return False, asset_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_API}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/download/"
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    tmp.rename(dest)
    return True, written


def _fmt_bytes(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dandiset_id", help="7-digit DANDI dandiset id (e.g. 000714).")
    parser.add_argument(
        "--override-sf-deferral",
        action="store_true",
        help=(
            "Force fetch of 000776 (SF). Deferred to Phase 0 Growth per "
            "the 2026-05-18 materialization deferral; pass this flag only "
            "after the deferral is reversed in pre-reg."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/downloads/wormid"),
        help="Destination root. Default: data/downloads/wormid/",
    )
    args = parser.parse_args()

    if args.dandiset_id == "000776" and not args.override_sf_deferral:
        sys.stderr.write(
            "Refusing to fetch 000776 (SF): materialization deferred to Phase 0 "
            "Growth on 2026-05-18 (see splits/wormid_train_eval.yaml). Pass "
            "--override-sf-deferral if you intend to bypass.\n"
        )
        return 2

    pin = next((p for p in SPEC.dandisets if p.dandiset_id == args.dandiset_id), None)
    if pin is None:
        sys.stderr.write(f"Unknown dandiset {args.dandiset_id}; not in wormid federation lock.\n")
        return 2

    dest_dir = args.root / pin.dandiset_id
    print(f"Fetching {pin.dandiset_id} @ {pin.version} → {dest_dir}", flush=True)
    print("  Listing assets ...", flush=True)
    assets = _list_assets(pin.dandiset_id, pin.version)
    total_bytes = sum(int(a["size"]) for a in assets)
    print(f"  {len(assets)} assets, total {_fmt_bytes(total_bytes)}", flush=True)

    start = time.time()
    fetched = 0
    skipped = 0
    bytes_done = 0
    for i, asset in enumerate(assets, 1):
        try:
            new, written = _download_one(asset, dest_dir, pin.dandiset_id, pin.version)
        except urllib.error.URLError as exc:
            sys.stderr.write(f"\nERROR on {asset.get('path')!r}: {exc}\n")
            return 1
        if new:
            fetched += 1
        else:
            skipped += 1
        bytes_done += written
        elapsed = time.time() - start
        rate = bytes_done / elapsed if elapsed > 0 else 0.0
        msg = (
            f"  [{i}/{len(assets)}] {asset['path']!s:<50} "
            f"{_fmt_bytes(written):>10} | total {_fmt_bytes(bytes_done)} "
            f"({rate / 1e6:.1f} MB/s)"
        )
        print(msg, flush=True)
    print(
        f"Done. Fetched {fetched}, skipped {skipped} (already present). "
        f"Wall: {time.time() - start:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
