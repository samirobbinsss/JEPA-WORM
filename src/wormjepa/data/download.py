"""DOI-pinned dataset downloader for JEPA-WORM.

Generic primitive: ``download_file(url, dest, expected_sha256)``. Per-dataset
entry: ``download_dataset(name)`` which looks up ``data/sources/<name>.SPEC``
and applies the generic primitive.

Reliability properties (NFR20):

- **Retry** with exponential backoff (1s, 4s, 16s) on transient failures:
  ``URLError``, ``HTTPError`` with status 5xx or 429, ``TimeoutError``, and
  ``ConnectionResetError``. 4xx errors other than 429 raise immediately.
- **Resume** an interrupted download via the ``Range: bytes=<n>-`` header.
  Bytes are staged in ``<dest>.partial``; a resumed attempt picks up from the
  current size of that file. If the server ignores ``Range`` (HTTP 200), the
  partial is truncated and the download restarts from byte 0.
- **Integrity** verified via SHA-256, streamed while bytes arrive (no second
  disk read). Mismatch raises :class:`wormjepa.DatasetIntegrityError` (FR7)
  and leaves the ``.partial`` file in place for inspection.
- **Atomic rename**: ``<dest>.partial`` is renamed to ``dest`` only after the
  digest matches, so a partially-written ``dest`` is never observable.

The project never redistributes dataset payloads. Reproducers fetch from the
canonical source via this module (FR10).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import time
import urllib.request
from http import HTTPStatus
from http.client import HTTPResponse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wormjepa import DatasetIntegrityError, WormJEPAError
from wormjepa.paths import project_root

if TYPE_CHECKING:
    from wormjepa.data.sources.base import AnyDatasetSource

logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES: frozenset[int] = frozenset({429})
_DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 1.0


def _backoff_seconds(attempt: int, base: float = _DEFAULT_BACKOFF_BASE_SECONDS) -> float:
    """Exponential backoff: ``base * 4**attempt`` -> 1s, 4s, 16s for base=1."""
    return base * (4**attempt)


def _is_retriable_http(exc: HTTPError) -> bool:
    """Retry 5xx and 429; do not retry other 4xx."""
    return exc.code >= 500 or exc.code in _RETRY_STATUS_CODES


def _open_with_range(url: str, start_byte: int) -> HTTPResponse:
    """Open ``url`` with a ``Range: bytes=<start>-`` header if ``start_byte > 0``.

    Returns the response (caller closes it via context manager).
    """
    request = Request(url)
    if start_byte > 0:
        request.add_header("Range", f"bytes={start_byte}-")
    return urlopen(request, timeout=60)  # type: ignore[return-value]


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    backoff_base: float = _DEFAULT_BACKOFF_BASE_SECONDS,
) -> Path:
    """Fetch ``url`` to ``dest``, retrying transient failures and resuming partials.

    Bytes are staged in ``<dest>.partial`` and the digest is computed in a
    streaming fashion. The file is atomically renamed to ``dest`` only after
    the SHA-256 matches ``expected_sha256``.

    Args:
        url: Source URL (HTTPS).
        dest: Destination path. Parent directory is created if missing. If a
            ``<dest>.partial`` already exists, the download resumes from its
            current size. If the server replies HTTP 200 to a ``Range``
            request, the partial is truncated and the download restarts.
        expected_sha256: Hex SHA-256 of the full payload. Verified as bytes
            arrive; on mismatch the ``.partial`` is left in place and
            :class:`DatasetIntegrityError` is raised.
        max_retries: Number of retries on transient failures (default 3).
            Total attempts = ``max_retries + 1``. Retries fire on
            ``URLError``, ``HTTPError`` 5xx/429, ``TimeoutError``, and
            ``ConnectionResetError``. 4xx other than 429 abort immediately.
        chunk_size: Per-read chunk size in bytes (default 1 MiB).
        backoff_base: Backoff base seconds (multiplied by ``4**attempt``).

    Returns:
        ``dest`` on success.

    Raises:
        DatasetIntegrityError: If the SHA-256 of the completed download does
            not match ``expected_sha256``.
        WormJEPAError: For non-retriable HTTP / URL errors after retries
            exhaust.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = expected_sha256.lower()
    partial = dest.with_suffix(dest.suffix + ".partial")
    hasher = hashlib.sha256()

    for attempt in range(max_retries + 1):
        try:
            start_byte = partial.stat().st_size if partial.exists() else 0
            response = _open_with_range(url, start_byte)
            with response:
                # If we requested a Range but the server returned 200, it
                # ignored the header — restart from byte 0.
                if start_byte > 0 and response.status == HTTPStatus.OK:
                    logger.info(
                        "server ignored Range header; restarting from byte 0",
                        extra={"url": url},
                    )
                    partial.unlink(missing_ok=True)
                    start_byte = 0

                # Re-hash the already-staged bytes (if any) so the digest
                # covers the full payload.
                hasher = hashlib.sha256()
                if start_byte > 0:
                    with partial.open("rb") as existing:
                        while True:
                            block = existing.read(chunk_size)
                            if not block:
                                break
                            hasher.update(block)

                mode = "ab" if start_byte > 0 else "wb"
                with partial.open(mode) as out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        hasher.update(chunk)
            break
        except HTTPError as exc:
            if _is_retriable_http(exc) and attempt < max_retries:
                delay = _backoff_seconds(attempt, backoff_base)
                logger.warning(
                    "transient HTTP %s on %s; retrying in %.1fs (attempt %d/%d)",
                    exc.code,
                    url,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            msg = f"HTTP {exc.code} fetching {url}: {exc.reason}"
            raise WormJEPAError(msg) from exc
        except (URLError, TimeoutError, ConnectionResetError) as exc:
            if attempt < max_retries:
                delay = _backoff_seconds(attempt, backoff_base)
                reason = getattr(exc, "reason", exc)
                logger.warning(
                    "network error on %s: %s; retrying in %.1fs (attempt %d/%d)",
                    url,
                    reason,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            reason = getattr(exc, "reason", exc)
            msg = f"Network error fetching {url}: {reason}"
            raise WormJEPAError(msg) from exc

    actual_sha256 = hasher.hexdigest()
    if actual_sha256 != expected_sha256:
        msg = (
            f"SHA-256 mismatch for {url} -> {dest}: "
            f"expected {expected_sha256}, got {actual_sha256}. "
            f"Partial file retained at {partial} for inspection."
        )
        raise DatasetIntegrityError(msg)
    partial.replace(dest)
    return dest


def load_source_spec(name: str) -> AnyDatasetSource:
    """Import ``wormjepa.data.sources.<name>`` and return its ``SPEC`` constant.

    The returned SPEC may be any of the four canonical shapes:
    :class:`DatasetSource` (single-DOI), :class:`DandiFederationSource`
    (multi-dandiset), :class:`ZenodoSubsetSource` (per-experiment Zenodo
    subset), or :class:`GithubGeneratorSource` (code-only repo pin).
    Callers that need a specific shape should narrow with an
    ``isinstance`` check.

    Raises:
        WormJEPAError: If the source module is missing or does not export a
            ``SPEC`` of the right type.
    """
    try:
        module = importlib.import_module(f"wormjepa.data.sources.{name}")
    except ModuleNotFoundError as exc:
        msg = (
            f"Unknown dataset {name!r}: no module wormjepa.data.sources.{name}. "
            f"Add one (see Story 2.3) before invoking the downloader."
        )
        raise WormJEPAError(msg) from exc
    spec = getattr(module, "SPEC", None)
    if spec is None:
        msg = f"wormjepa.data.sources.{name} does not export SPEC."
        raise WormJEPAError(msg)
    return spec


def download_zenodo_record(record_id: str, dest_dir: Path) -> list[Path]:
    """Fetch every file in a Zenodo record to ``dest_dir``, idempotently.

    Mirrors the behaviour of ``scripts/dev/fetch_zenodo_anchors.py``: queries
    the Zenodo public records API for the ``record_id``, then downloads each
    file in the record. Files already present at the expected byte size are
    skipped, making this safe to re-run for resume-style behaviour.

    No SHA verification is performed here: Zenodo's record API does not
    surface the per-file SHA-256 hash uniformly across record types, and
    the integrity contract for per-experiment Zenodo subsets is enforced
    at the loader level (via the :class:`ZenodoSubsetSource` SPEC) rather
    than at fetch time. Use :func:`download_file` for SHA-pinned single-
    file downloads.

    Args:
        record_id: Numeric Zenodo record identifier (e.g. ``"1031550"``).
        dest_dir: Destination directory. Created if missing.

    Returns:
        The list of file paths now present under ``dest_dir`` (one entry
        per file in the record, whether newly fetched or already
        cached).

    Raises:
        WormJEPAError: On unrecoverable network / HTTP errors.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=60) as response:
            payload: dict[str, Any] = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        msg = f"Failed to query Zenodo record {record_id!r} at {api_url}: {exc}"
        raise WormJEPAError(msg) from exc

    files = payload.get("files", [])
    if not isinstance(files, list):
        msg = f"Zenodo record {record_id!r} returned no 'files' list."
        raise WormJEPAError(msg)

    results: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        key = str(entry["key"])
        size = int(entry["size"])
        dest = dest_dir / key
        if dest.exists() and dest.stat().st_size == size:
            logger.debug("zenodo cache hit", extra={"record": record_id, "key": key})
            results.append(dest)
            continue
        url = entry["links"]["self"]
        try:
            urllib.request.urlretrieve(url, dest)
        except (HTTPError, URLError, TimeoutError) as exc:
            dest.unlink(missing_ok=True)
            msg = f"Failed to fetch Zenodo file {key!r} from {url}: {exc}"
            raise WormJEPAError(msg) from exc
        results.append(dest)
        logger.info(
            "zenodo file fetched",
            extra={"record": record_id, "key": key, "bytes": size},
        )
    return results


def download_dataset(name: str) -> Path:
    """Look up ``data/sources/<name>.SPEC`` and download to ``data/downloads/``.

    Only single-DOI datasets (SPEC type :class:`DatasetSource`) are handled
    here. Federated DANDI corpora, per-experiment Zenodo subsets, and
    GitHub-pinned generators require dataset-specific fetch logic, which
    lands in their respective loader stories (8.3, 8.5, 8.6, 8.7).

    Args:
        name: Dataset key matching a module under ``wormjepa.data.sources``.

    Returns:
        Path to the downloaded payload.

    Raises:
        DatasetIntegrityError: On SHA-256 mismatch.
        WormJEPAError: On unknown dataset, non-single-DOI SPEC, or
            unrecoverable network error.
    """
    # Imported here (not at module scope) to avoid the TYPE_CHECKING guard.
    from wormjepa.data.sources.base import DatasetSource

    spec = load_source_spec(name)
    if not isinstance(spec, DatasetSource):
        msg = (
            f"Dataset {name!r} uses a non-single-DOI SPEC "
            f"({type(spec).__name__}); use the dataset-specific loader. "
            f"See Stories 8.3 / 8.5 / 8.6 / 8.7 for federation / subset / "
            f"generator fetch logic."
        )
        raise WormJEPAError(msg)
    dest = project_root() / "data" / "downloads" / spec.dest_filename
    logger.info("downloading dataset", extra={"name": name, "url": spec.url, "dest": str(dest)})
    return download_file(spec.url, dest, spec.sha256)
