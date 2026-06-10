"""Unit tests for ``wormjepa.data.contract``."""

from __future__ import annotations

import torch

from wormjepa.data import (
    DatasetSample,
    SessionID,
    SourceDataset,
    WormID,
)


def test_dataset_sample_with_all_fields_populated() -> None:
    """A sample with every optional field present round-trips correctly."""
    video = torch.zeros((8, 3, 64, 64))
    pose = torch.zeros((8, 22, 3))
    neural = torch.zeros((8, 180))
    sample = DatasetSample(
        video_clip=video,
        pose=pose,
        neural=neural,
        worm_id=WormID("wormid-2023-014"),
        session_id=SessionID("flavell_lab_2023-04-15_session_03"),
        source_dataset=SourceDataset("wormid"),
    )
    assert torch.equal(sample.video_clip, video)
    assert sample.pose is not None
    assert sample.neural is not None
    assert sample.worm_id == "wormid-2023-014"
    assert sample.session_id == "flavell_lab_2023-04-15_session_03"
    assert sample.source_dataset == "wormid"


def test_dataset_sample_with_no_pose_or_neural() -> None:
    """A sample with only video (e.g., WormBehavior DB) is valid."""
    video = torch.zeros((8, 3, 64, 64))
    sample = DatasetSample(
        video_clip=video,
        pose=None,
        neural=None,
        worm_id=WormID("wb_db_w01"),
        session_id=SessionID("wb_db_w01_2013-06-01"),
        source_dataset=SourceDataset("wormbehavior_db"),
    )
    assert sample.pose is None
    assert sample.neural is None


def test_dataset_sample_field_order_is_documented() -> None:
    """The tuple's positional order matches the documented contract.

    Downstream loaders may construct samples positionally; the field order
    is part of the contract and changing it would break every loader.
    ``behavioral_state`` and ``frame_rate`` are appended at the end with
    ``None`` defaults so existing positional and keyword call sites continue
    to work.
    """
    assert DatasetSample._fields == (
        "video_clip",
        "pose",
        "neural",
        "worm_id",
        "session_id",
        "source_dataset",
        "behavioral_state",
        "frame_rate",
    )


def test_behavioral_state_defaults_to_none() -> None:
    """Loaders that don't surface labels can construct samples without the field."""
    sample = DatasetSample(
        video_clip=torch.zeros((1, 3, 8, 8)),
        pose=None,
        neural=None,
        worm_id=WormID("w"),
        session_id=SessionID("s"),
        source_dataset=SourceDataset("wormid"),
    )
    assert sample.behavioral_state is None


def test_frame_rate_defaults_to_none() -> None:
    """Loaders that don't know their rate can construct samples without the field."""
    sample = DatasetSample(
        video_clip=torch.zeros((1, 3, 8, 8)),
        pose=None,
        neural=None,
        worm_id=WormID("w"),
        session_id=SessionID("s"),
        source_dataset=SourceDataset("synthetic"),
    )
    assert sample.frame_rate is None


def test_frame_rate_round_trips_when_set() -> None:
    """A loader-supplied rate survives on the constructed sample."""
    sample = DatasetSample(
        video_clip=torch.zeros((1, 3, 8, 8)),
        pose=None,
        neural=None,
        worm_id=WormID("w"),
        session_id=SessionID("s"),
        source_dataset=SourceDataset("flavell_2023"),
        frame_rate=3.0,
    )
    assert sample.frame_rate == 3.0


def test_newtype_aliases_distinguish_identifier_strings_at_runtime() -> None:
    """``NewType`` is a no-op at runtime but documented for type checkers.

    A regular ``str`` is interchangeable at runtime; the type checker (pyright)
    is responsible for enforcing the distinction. This test exists to make the
    runtime behavior explicit and warn anyone tempted to add runtime checks.
    """
    raw = "wormid-2023-014"
    typed = WormID(raw)
    assert typed == raw
    assert isinstance(typed, str)


def test_dataset_sample_is_immutable() -> None:
    """NamedTuples are immutable; reassignment raises ``AttributeError``."""
    sample = DatasetSample(
        video_clip=torch.zeros((1, 3, 8, 8)),
        pose=None,
        neural=None,
        worm_id=WormID("w"),
        session_id=SessionID("s"),
        source_dataset=SourceDataset("baaiworm"),
    )
    try:
        sample.worm_id = WormID("other")  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("DatasetSample.worm_id should be immutable")
