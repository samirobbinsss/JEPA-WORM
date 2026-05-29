"""Resume / retry / atomic-rename tests for ``wormjepa.data.download``.

Each test spins up a ``ThreadingHTTPServer`` bound to ``127.0.0.1:0`` so
multiple tests can run in parallel without port collisions. The handler's
behavior is driven by class-level state (``payload``, ``status_sequence``,
``range_supported``) so individual tests can script the exact HTTP
responses they need without monkey-patching urllib.
"""

from __future__ import annotations

import hashlib
import socket
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from wormjepa import DatasetIntegrityError, WormJEPAError
from wormjepa.data.download import download_file


class _ScriptedHandler(BaseHTTPRequestHandler):
    """HTTP handler driven by class-level state.

    ``status_sequence`` pops one entry per request: ``None`` means "serve the
    payload normally", an int means "return that status with an empty body".
    ``range_supported`` toggles whether ``Range: bytes=N-`` is honoured (206)
    or ignored (200 + full payload).
    """

    payload: ClassVar[bytes] = b""
    status_sequence: ClassVar[list[int | None]] = []
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
            type(self).request_log.append(
                {"range": range_header, "forced_status": forced_status if forced_status else 0}
            )

        if forced_status is not None:
            self.send_response(forced_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        payload = type(self).payload
        start = 0
        if range_header and type(self).range_supported:
            # Expect "bytes=<n>-".
            spec = range_header.removeprefix("bytes=")
            start = int(spec.split("-", 1)[0])
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
        else:
            self.send_response(HTTPStatus.OK)
        body = payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)


def _ephemeral_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def http_server() -> Iterator[tuple[str, int, type[_ScriptedHandler]]]:
    """Yield ``(host, port, handler_cls)`` for a per-test threaded server."""
    port = _ephemeral_port()
    _ScriptedHandler.payload = b""
    _ScriptedHandler.status_sequence = []
    _ScriptedHandler.range_supported = True
    _ScriptedHandler.request_log = []

    server = ThreadingHTTPServer(("127.0.0.1", port), _ScriptedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ("127.0.0.1", port, _ScriptedHandler)
    finally:
        server.shutdown()
        server.server_close()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_resume_appends_to_partial(
    tmp_path: Path, http_server: tuple[str, int, type[_ScriptedHandler]]
) -> None:
    """Pre-seed ``<dest>.partial`` with a prefix; server returns 206 for the
    remaining bytes; final file matches the full payload's SHA."""
    host, port, handler = http_server
    full = b"resume-me-please-" * 32  # 544 bytes
    handler.payload = full

    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")
    prefix_len = 100
    partial.write_bytes(full[:prefix_len])

    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(full),
        max_retries=0,
        backoff_base=0.0,
    )

    assert dest.exists()
    assert dest.read_bytes() == full
    assert not partial.exists(), ".partial must be removed after atomic rename"

    # Confirm the client actually sent a Range header.
    assert handler.request_log, "expected at least one request"
    range_value = handler.request_log[0]["range"]
    assert isinstance(range_value, str)
    assert range_value == f"bytes={prefix_len}-"


def test_retry_on_5xx(tmp_path: Path, http_server: tuple[str, int, type[_ScriptedHandler]]) -> None:
    """Two 503s then success; final file matches expected SHA."""
    host, port, handler = http_server
    handler.payload = b"recoverable-after-two-503s"
    handler.status_sequence = [503, 503]  # None for the third request

    dest = tmp_path / "payload.bin"
    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(handler.payload),
        max_retries=3,
        backoff_base=0.0,
    )

    assert dest.read_bytes() == handler.payload
    assert len(handler.request_log) == 3, "expected three HTTP requests (two retries)"


def test_no_retry_on_404(
    tmp_path: Path, http_server: tuple[str, int, type[_ScriptedHandler]]
) -> None:
    """404 raises immediately without consuming any retry budget."""
    host, port, handler = http_server
    handler.payload = b"never-served"
    handler.status_sequence = [404, 404, 404, 404]  # plenty available

    dest = tmp_path / "payload.bin"
    with pytest.raises(WormJEPAError, match="HTTP 404"):
        download_file(
            f"http://{host}:{port}/x",
            dest,
            _sha(handler.payload),
            max_retries=3,
            backoff_base=0.0,
        )

    assert len(handler.request_log) == 1, "404 must not trigger retries"
    assert not dest.exists()


def test_atomic_rename(
    tmp_path: Path, http_server: tuple[str, int, type[_ScriptedHandler]]
) -> None:
    """SHA mismatch keeps the ``.partial`` for inspection and never produces dest."""
    host, port, handler = http_server
    handler.payload = b"contents-that-wont-match-the-claimed-sha"

    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")
    bogus_sha = _sha(b"a-different-payload-entirely")

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        download_file(
            f"http://{host}:{port}/x",
            dest,
            bogus_sha,
            max_retries=0,
            backoff_base=0.0,
        )

    assert not dest.exists(), "dest must not exist after SHA failure"
    assert partial.exists(), ".partial must survive for inspection"
    assert partial.read_bytes() == handler.payload


def test_server_ignores_range_restarts_from_zero(
    tmp_path: Path, http_server: tuple[str, int, type[_ScriptedHandler]]
) -> None:
    """If the server returns 200 to a Range request, the partial is truncated
    and the digest still matches the full payload."""
    host, port, handler = http_server
    full = b"server-ignores-range-headers-and-returns-200"
    handler.payload = full
    handler.range_supported = False

    dest = tmp_path / "payload.bin"
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(b"STALE-PREFIX-THAT-MUST-BE-DISCARDED")

    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(full),
        max_retries=0,
        backoff_base=0.0,
    )

    assert dest.read_bytes() == full
    assert not partial.exists()
