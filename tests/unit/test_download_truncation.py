"""Mid-stream truncation / short-read convergence tests for ``wormjepa.data.download``.

Companion to ``test_download_resume.py``. Where that file scripts *clean* HTTP
errors (5xx/429 with empty bodies), this file scripts the failure mode that
actually broke the ~45 GB Zenodo pull: the server advertises the full
``Content-Length`` but drops the connection mid-body, so the client receives
fewer bytes than promised. ``urllib.request.urlretrieve`` discarded the partial
on that error and never converged; the robust resumable path must instead keep
the staged bytes in ``<dest>.partial`` and finish via a ``Range`` request on the
next attempt.

The handler is a superset of the resume-file handler: it adds a
``truncate_sequence`` that, per request, advertises the true total length but
writes only the first ``k`` bytes and then closes the socket.

These tests target the size-based completion path: ``download_file`` with
``expected_sha256=None`` and ``expected_size=<bytes>`` (the SHA-less primitive
the fix introduces), and a ``download_zenodo_record`` that routes per-file
fetches through that same robust resumable path.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

import wormjepa.data.download as dl_mod
from wormjepa.data.download import download_file, download_zenodo_record


class _TruncatingHandler(BaseHTTPRequestHandler):
    """Handler driven by class-level state, with mid-stream truncation support.

    Per request, in priority order:
      * ``status_sequence`` pops an ``int`` -> send that status, empty body.
      * ``truncate_sequence`` pops an ``int k`` -> advertise the *full*
        ``Content-Length`` for the requested range, but write only the first
        ``k`` bytes of the body and then close the connection (simulating a
        mid-stream drop / ``IncompleteRead`` on the client).
      * otherwise -> serve the (ranged) payload honestly.

    ``range_supported`` toggles 206-with-``Content-Range`` vs 200-full-payload.
    ``request_log`` records the ``Range`` header and the action taken so tests
    can assert the client resumed.
    """

    payload: ClassVar[bytes] = b""
    status_sequence: ClassVar[list[int | None]] = []
    truncate_sequence: ClassVar[list[int | None]] = []
    range_supported: ClassVar[bool] = True
    sequence_lock: ClassVar[threading.Lock] = threading.Lock()
    request_log: ClassVar[list[dict[str, str | int]]] = []

    def log_message(self, *_args: object, **_kwargs: object) -> None:  # silence
        return

    def do_GET(self) -> None:
        range_header = self.headers.get("Range", "")
        with type(self).sequence_lock:
            forced_status: int | None = None
            if type(self).status_sequence:
                forced_status = type(self).status_sequence.pop(0)
            truncate_at: int | None = None
            if forced_status is None and type(self).truncate_sequence:
                truncate_at = type(self).truncate_sequence.pop(0)
            type(self).request_log.append(
                {
                    "range": range_header,
                    "forced_status": forced_status if forced_status else 0,
                    "truncate_at": truncate_at if truncate_at is not None else -1,
                }
            )

        if forced_status is not None:
            self.send_response(forced_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        payload = type(self).payload
        start = 0
        if range_header and type(self).range_supported:
            spec = range_header.removeprefix("bytes=")
            start = int(spec.split("-", 1)[0])
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
        else:
            self.send_response(HTTPStatus.OK)
        body = payload[start:]

        # Advertise the FULL remaining length regardless of truncation: the whole
        # point is that the client is promised Content-Length bytes but receives
        # fewer, then the socket closes.
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        if truncate_at is not None:
            self.wfile.write(body[:truncate_at])
            self.wfile.flush()
            # Force the connection shut so the client's reader sees EOF early.
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.wfile.close()
            return

        self.wfile.write(body)


def _ephemeral_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def http_server() -> Iterator[tuple[str, int, type[_TruncatingHandler]]]:
    """Yield ``(host, port, handler_cls)`` for a per-test threaded server."""
    port = _ephemeral_port()
    _TruncatingHandler.payload = b""
    _TruncatingHandler.status_sequence = []
    _TruncatingHandler.truncate_sequence = []
    _TruncatingHandler.range_supported = True
    _TruncatingHandler.request_log = []

    server = ThreadingHTTPServer(("127.0.0.1", port), _TruncatingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ("127.0.0.1", port, _TruncatingHandler)
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# Test 1: file-level — size-based download converges across a mid-stream drop. #
# --------------------------------------------------------------------------- #
def test_sized_download_converges_after_midstream_truncation(
    tmp_path: Path, http_server: tuple[str, int, type[_TruncatingHandler]]
) -> None:
    """First GET advertises the full Content-Length but drops the connection
    after ``prefix_len`` bytes; the retry resumes via ``Range`` and completes.

    This is the regression for the 45 GB Zenodo pull: with the old
    ``urlretrieve`` path a ContentTooShortError discarded the partial and the
    download never converged. The size-based path must (a) treat the short read
    as retriable/resumable and (b) verify final byte count, not just EOF.
    """
    host, port, handler = http_server
    full = b"sized-payload-resumes-after-a-mid-stream-drop-" * 16  # 736 bytes
    handler.payload = full
    prefix_len = 200
    # Attempt 1: write only the first `prefix_len` bytes, then close.
    # Attempt 2 (None): serve the rest honestly (via Range -> 206).
    handler.truncate_sequence = [prefix_len, None]

    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")

    # SHA-less, size-based completion: we know the expected total bytes only.
    download_file(
        f"http://{host}:{port}/x",
        dest,
        None,
        expected_size=len(full),
        max_retries=3,
        backoff_base=0.0,
    )

    assert dest.exists()
    assert dest.read_bytes() == full, "final file must be byte-for-byte complete"
    assert dest.stat().st_size == len(full)
    assert not partial.exists(), ".partial must be renamed away after completion"

    # Prove convergence happened *via resume*, not a full restart: the second
    # request must carry a Range header anchored at the bytes already staged.
    assert len(handler.request_log) >= 2, "expected a retry after the truncation"
    second = handler.request_log[1]
    assert second["range"] == f"bytes={prefix_len}-", (
        "retry must resume from the staged partial, not restart from byte 0"
    )


def test_sized_download_rejects_when_no_completion_check(
    tmp_path: Path, http_server: tuple[str, int, type[_TruncatingHandler]]
) -> None:
    """A download with neither SHA nor size is refused fail-closed: permitting
    it would reintroduce the silent-truncation footgun this change exists to
    kill."""
    host, port, handler = http_server
    handler.payload = b"no-completion-guarantee-possible"

    dest = tmp_path / "payload.bin"
    with pytest.raises(dl_mod.WormJEPAError, match="completion check"):
        download_file(
            f"http://{host}:{port}/x",
            dest,
            None,
            max_retries=0,
            backoff_base=0.0,
        )
    assert not dest.exists()
    assert not handler.request_log, "must reject before any network request"


# --------------------------------------------------------------------------- #
# Test 2: record-level — download_zenodo_record converges over a flaky server. #
# --------------------------------------------------------------------------- #
def test_zenodo_record_converges_over_truncating_server(
    tmp_path: Path,
    http_server: tuple[str, int, type[_TruncatingHandler]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``download_zenodo_record`` must converge when a record file is served by
    a server that truncates the first GET, and must return the cached path.

    The Zenodo *records API* call is stubbed (the module hits
    ``https://zenodo.org/api/records/<id>``); we point the per-file ``links.self``
    URL at the local truncating server so the file fetch exercises the real
    resumable path rather than the old ``urlretrieve`` discard-on-failure path.
    """
    host, port, handler = http_server
    file_body = b"zenodo-record-file-body-that-survives-a-truncation-" * 20
    handler.payload = file_body
    # First GET of the file truncates; retry resumes and finishes.
    handler.truncate_sequence = [128, None]

    record_id = "1031550"
    file_key = "wormbehavior_subset.h5"
    file_url = f"http://{host}:{port}/files/{file_key}"

    api_json = {
        "files": [
            {
                "key": file_key,
                "size": len(file_body),
                "links": {"self": file_url},
            }
        ]
    }

    class _FakeJSONResponse:
        """Context-manager stand-in for the records-API urlopen() response."""

        def __init__(self, data: Mapping[str, object]) -> None:
            self._raw = json.dumps(data).encode()

        def read(self, *_a: object) -> bytes:
            return self._raw

        def __enter__(self) -> _FakeJSONResponse:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def _fake_urlopen(url: str, *_a: object, **_kw: object) -> _FakeJSONResponse:
        # Only the records-API endpoint goes through urlopen in this function;
        # per-file fetches go through the robust download path (real sockets).
        assert "/api/records/" in str(url), f"unexpected urlopen for {url!r}"
        return _FakeJSONResponse(api_json)

    # Stub the records-API call; let the file download hit the local server.
    monkeypatch.setattr(dl_mod.urllib.request, "urlopen", _fake_urlopen)

    dest_dir = tmp_path / "zenodo" / record_id
    results = download_zenodo_record(record_id, dest_dir)

    expected_path = dest_dir / file_key
    assert results == [expected_path], "returns one path per record file"
    assert expected_path.exists()
    assert expected_path.read_bytes() == file_body, "file must be complete + correct"
    assert expected_path.stat().st_size == len(file_body)
    assert not expected_path.with_suffix(expected_path.suffix + ".partial").exists()

    # Idempotent re-run is a pure cache hit (size matches) -> no new GETs.
    requests_after_first = len(handler.request_log)
    results_again = download_zenodo_record(record_id, dest_dir)
    assert results_again == [expected_path]
    assert len(handler.request_log) == requests_after_first, (
        "second call must be a size-based cache hit, issuing no further GETs"
    )


# --------------------------------------------------------------------------- #
# Test 3: the 416 boundary — a complete .partial must finalize, never wedge.   #
# --------------------------------------------------------------------------- #
def test_complete_partial_finalizes_on_416_range_not_satisfiable(
    tmp_path: Path, http_server: tuple[str, int, type[_TruncatingHandler]]
) -> None:
    """A ``.partial`` that already holds the full payload (a crash between the
    final read and the atomic rename) must finalize, not wedge.

    On the next attempt the client issues ``Range: bytes=<size>-``; the server
    answers ``416 Range Not Satisfiable``. The first cut of the fix classified
    416 as a fatal 4xx and raised, stranding a fully-downloaded record forever.
    416 must instead be a finalize-or-reset signal: the staged bytes here are
    complete, so the file is renamed into place with no further body transfer.
    """
    host, port, handler = http_server
    full = b"already-complete-before-the-rename-" * 24
    handler.payload = full
    # Pre-stage a COMPLETE .partial (size == expected_size).
    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(full)
    # Force 416 on the resume request the client makes.
    handler.status_sequence = [HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE]

    download_file(
        f"http://{host}:{port}/x",
        dest,
        None,
        expected_size=len(full),
        max_retries=2,
        backoff_base=0.0,
    )

    assert dest.exists()
    assert dest.read_bytes() == full, "complete partial must finalize byte-for-byte"
    assert not partial.exists(), ".partial must be renamed away, not left wedged"
    # Exactly one GET (the 416 probe); no body re-transfer, no extra retries.
    assert len(handler.request_log) == 1
    assert handler.request_log[0]["forced_status"] == int(
        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
    )


def test_oversized_partial_discarded_and_restarted(
    tmp_path: Path, http_server: tuple[str, int, type[_TruncatingHandler]]
) -> None:
    """A stale ``.partial`` larger than ``expected_size`` must be discarded
    before any request (so it never provokes a 416) and the download restarted
    clean from byte 0, yielding the correct bytes."""
    host, port, handler = http_server
    full = b"correct-final-payload-" * 8
    handler.payload = full
    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")
    # Stale partial larger than the real resource (e.g. expected_size shrank).
    partial.write_bytes(full + b"STALE-EXTRA-BYTES-FROM-AN-OLD-RUN")

    download_file(
        f"http://{host}:{port}/x",
        dest,
        None,
        expected_size=len(full),
        max_retries=3,
        backoff_base=0.0,
    )

    assert dest.exists()
    assert dest.read_bytes() == full, "restart must yield the correct bytes"
    assert not partial.exists()
    # The discard happens before the network, so the single GET starts at 0.
    assert len(handler.request_log) == 1
    assert handler.request_log[0].get("range") in (None, "", "bytes=0-")
