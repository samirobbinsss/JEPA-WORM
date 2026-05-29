"""Unit tests for ``wormjepa.reporting.results_writer``."""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa.reporting import (
    ALLOWED_FILES,
    ALLOWED_SUBDIRS,
    ResultsContractViolation,
    ResultsWriter,
)


@pytest.fixture
def writer(tmp_path: Path) -> ResultsWriter:
    return ResultsWriter("20260512T000000Z__abcdef12__test", results_root=tmp_path)


def test_initialize_creates_directory(writer: ResultsWriter) -> None:
    path = writer.initialize()
    assert path.is_dir()
    assert path == writer.path


def test_initialize_creates_required_placeholder_files(writer: ResultsWriter) -> None:
    writer.initialize()
    for filename in ResultsWriter.REQUIRED_INITIAL_FILES:
        assert (writer.path / filename).is_file()


def test_initialize_refuses_to_overwrite_existing(writer: ResultsWriter) -> None:
    writer.initialize()
    with pytest.raises(ResultsContractViolation, match="already exists"):
        writer.initialize()


def test_write_text_allowed_file(writer: ResultsWriter) -> None:
    writer.initialize()
    target = writer.write_text("metrics.json", '{"row1": null}')
    assert target.read_text(encoding="utf-8") == '{"row1": null}'


def test_write_text_disallowed_file_raises(writer: ResultsWriter) -> None:
    writer.initialize()
    with pytest.raises(ResultsContractViolation, match="forbids file"):
        writer.write_text("scratch.txt", "data")


def test_write_text_inside_allowed_subdir(writer: ResultsWriter) -> None:
    writer.initialize()
    target = writer.write_text("checkpoints/step_42.pt", "binary-ish")
    assert target.is_file()
    assert target.parent.name == "checkpoints"


def test_write_text_inside_disallowed_subdir_raises(writer: ResultsWriter) -> None:
    writer.initialize()
    with pytest.raises(ResultsContractViolation, match="forbids writes outside"):
        writer.write_text("scratch/notes.txt", "data")


def test_write_bytes_allowed(writer: ResultsWriter) -> None:
    writer.initialize()
    target = writer.write_bytes("bootstrap_samples.parquet", b"PAR1...")
    assert target.read_bytes() == b"PAR1..."


def test_allowed_files_includes_documented_contract() -> None:
    """The contract files documented in architecture must be in ALLOWED_FILES."""
    documented = {
        "config.yaml",
        "metrics.json",
        "compute.json",
        "seed.txt",
        "manifest_at_run.lock",
        "report.md",
        "log.jsonl",
        "bootstrap_samples.parquet",
    }
    assert documented.issubset(ALLOWED_FILES)


def test_allowed_subdirs_includes_checkpoints() -> None:
    assert "checkpoints" in ALLOWED_SUBDIRS


def test_write_strips_leading_dot_slash(writer: ResultsWriter) -> None:
    writer.initialize()
    target = writer.write_text("./metrics.json", "{}")
    assert target.name == "metrics.json"


def test_results_contract_violation_is_wormjepa_error() -> None:
    from wormjepa import WormJEPAError

    assert issubclass(ResultsContractViolation, WormJEPAError)
