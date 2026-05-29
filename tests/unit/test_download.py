"""Unit tests for ``wormjepa.data.download``.

Uses an in-process ``http.server`` running in a thread as the fixture HTTP
server. Avoids new test dependencies and mirrors real urllib behavior
including Range headers and HTTP status codes.
"""

from __future__ import annotations

import hashlib
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from wormjepa import DatasetIntegrityError, WormJEPAError
from wormjepa.data.download import download_file, load_source_spec


class _ScriptedHandler(BaseHTTPRequestHandler):
    """HTTP handler whose behavior is parameterized by class-level state."""

    payload: ClassVar[bytes] = b""
    failure_sequence: ClassVar[list[int]] = []
    range_supported: ClassVar[bool] = True
    sequence_lock: ClassVar[threading.Lock] = threading.Lock()

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:  # silence
        return

    def do_GET(self) -> None:
        with type(self).sequence_lock:
            if type(self).failure_sequence:
                code = type(self).failure_sequence.pop(0)
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        payload = type(self).payload
        range_header = self.headers.get("Range")
        start = 0
        if range_header and type(self).range_supported:
            # Expect "bytes=<n>-"
            spec = range_header.removeprefix("bytes=")
            start = int(spec.split("-", 1)[0])
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
        else:
            self.send_response(HTTPStatus.OK)
        body = payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_server() -> Any:
    """Start a local HTTP server in a thread; yield (host, port, handler_cls)."""
    # Bind to ephemeral port.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    # Reset handler state per test.
    _ScriptedHandler.payload = b""
    _ScriptedHandler.failure_sequence = []
    _ScriptedHandler.range_supported = True

    server = HTTPServer(("127.0.0.1", port), _ScriptedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ("127.0.0.1", port, _ScriptedHandler)
    finally:
        server.shutdown()
        server.server_close()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_download_file_happy_path(tmp_path: Path, http_server: Any) -> None:
    host, port, handler = http_server
    handler.payload = b"hello-worm" * 100
    dest = tmp_path / "payload.bin"
    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(handler.payload),
        max_retries=0,
        backoff_base=0.0,
    )
    assert dest.read_bytes() == handler.payload


def test_download_file_sha_mismatch_raises_and_removes_partial(
    tmp_path: Path, http_server: Any
) -> None:
    host, port, handler = http_server
    handler.payload = b"hello-worm"
    dest = tmp_path / "payload.bin"
    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        download_file(
            f"http://{host}:{port}/x",
            dest,
            _sha(b"wrong"),
            max_retries=0,
            backoff_base=0.0,
        )
    assert not dest.exists()


def test_download_file_retries_on_503(tmp_path: Path, http_server: Any) -> None:
    host, port, handler = http_server
    handler.payload = b"hello"
    handler.failure_sequence = [503, 503]
    dest = tmp_path / "payload.bin"
    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(handler.payload),
        max_retries=3,
        backoff_base=0.0,
    )
    assert dest.read_bytes() == handler.payload


def test_download_file_retries_on_429(tmp_path: Path, http_server: Any) -> None:
    host, port, handler = http_server
    handler.payload = b"rate-limited-payload"
    handler.failure_sequence = [429]
    dest = tmp_path / "payload.bin"
    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(handler.payload),
        max_retries=2,
        backoff_base=0.0,
    )
    assert dest.read_bytes() == handler.payload


def test_download_file_gives_up_after_max_retries(tmp_path: Path, http_server: Any) -> None:
    host, port, handler = http_server
    handler.payload = b"unreachable"
    handler.failure_sequence = [503, 503, 503]
    dest = tmp_path / "payload.bin"
    with pytest.raises(WormJEPAError, match="HTTP 503"):
        download_file(
            f"http://{host}:{port}/x",
            dest,
            _sha(handler.payload),
            max_retries=1,
            backoff_base=0.0,
        )


def test_download_file_resumes_partial(tmp_path: Path, http_server: Any) -> None:
    host, port, handler = http_server
    full = b"the-full-payload-with-many-bytes-of-content"
    handler.payload = full

    dest = tmp_path / "payload.bin"
    # Pretend a previous download stopped mid-way.
    prefix_len = 10
    dest.write_bytes(full[:prefix_len])

    download_file(
        f"http://{host}:{port}/x",
        dest,
        _sha(full),
        max_retries=0,
        backoff_base=0.0,
    )
    assert dest.read_bytes() == full


def test_load_source_spec_unknown_raises() -> None:
    with pytest.raises(WormJEPAError, match="Unknown dataset"):
        load_source_spec("definitely_not_a_real_dataset")
