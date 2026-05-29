"""Unit tests for ``wormjepa.baselines.transformer_eigenworms``."""

from __future__ import annotations

import torch

from wormjepa.baselines.transformer_eigenworms import TransformerEigenwormsBaseline
from wormjepa.data.loaders.synthetic import SyntheticLoader


def _small_baseline() -> TransformerEigenwormsBaseline:
    return TransformerEigenwormsBaseline(
        n_eigen=3,
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_epochs=1,
    )


def test_name() -> None:
    assert TransformerEigenwormsBaseline().name == "transformer_eigenworms"


def test_fit_then_predict_returns_three_horizons_per_clip() -> None:
    loader = SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    )
    # 2 worms x 2 clips x 3 horizons = 12 entries.
    assert len(predictions.future_pose) == 12
    assert {fp.horizon_seconds for fp in predictions.future_pose} == {0.1, 1.0, 5.0}


def test_predict_before_fit_raises() -> None:
    baseline = TransformerEigenwormsBaseline()
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4)
    try:
        baseline.predict(loader)
    except RuntimeError as exc:
        assert "fit" in str(exc)
        return
    raise AssertionError("predict() before fit() should raise RuntimeError")


def test_predicted_shape_matches_ground_truth() -> None:
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=8, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=8, seed=0)
    )
    for fp in predictions.future_pose:
        assert fp.predicted.shape == fp.ground_truth.shape


def test_fit_without_pose_raises() -> None:
    from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID

    def _iter() -> object:
        yield DatasetSample(
            video_clip=torch.zeros((4, 3, 8, 8)),
            pose=None,
            neural=None,
            worm_id=WormID("w"),
            session_id=SessionID("s"),
            source_dataset=SourceDataset("synthetic"),
        )

    baseline = TransformerEigenwormsBaseline()
    try:
        baseline.fit(_iter())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "No pose data" in str(exc)
        return
    raise AssertionError("fit() with no pose data should raise ValueError")


def test_no_latent_exposed() -> None:
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=6, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=6, seed=0)
    )
    assert predictions.latent_by_worm is None
