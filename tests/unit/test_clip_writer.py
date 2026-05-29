"""Unit tests for the per-step training-clip writer (MP4 output)."""

from __future__ import annotations

from pathlib import Path

import av
import torch

from wormjepa.training.clip_writer import write_step_clip


def _frame_count(path: Path) -> int:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))
    finally:
        container.close()


def test_writes_mp4_per_step(tmp_path: Path) -> None:
    t_n, h, w = 4, 8, 8
    video = torch.rand((1, t_n, 3, h, w))
    mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    write_step_clip(tmp_path, step=42, video=video, mask=mask, pose=None)
    mp4 = tmp_path / "42.mp4"
    assert mp4.is_file()
    # All input frames make it into the encoded stream.
    assert _frame_count(mp4) == t_n


def test_pose_dots_do_not_crash_writer(tmp_path: Path) -> None:
    t_n, h, w, k = 3, 16, 16, 5
    video = torch.rand((1, t_n, 3, h, w))
    mask = torch.zeros(1, t_n)
    pose = torch.randn((1, t_n, k, 2)) * 0.1
    write_step_clip(tmp_path, step=1, video=video, mask=mask, pose=pose)
    assert (tmp_path / "1.mp4").is_file()


def test_predicted_pose_rendered_alongside_ground_truth(tmp_path: Path) -> None:
    t_n, h, w, k = 2, 16, 16, 4
    video = torch.rand((1, t_n, 3, h, w))
    mask = torch.zeros(1, t_n)
    pose = torch.randn((1, t_n, k, 2)) * 0.1
    predicted = torch.randn((1, t_n, k, 2)) * 0.1
    write_step_clip(tmp_path, step=3, video=video, mask=mask, pose=pose, predicted_pose=predicted)
    mp4 = tmp_path / "3.mp4"
    assert mp4.is_file()
    assert _frame_count(mp4) == t_n


def test_nonfinite_pose_falls_back_gracefully(tmp_path: Path) -> None:
    t_n, h, w, k = 2, 8, 8, 3
    video = torch.rand((1, t_n, 3, h, w))
    mask = torch.zeros(1, t_n)
    pose = torch.full((1, t_n, k, 2), float("nan"))
    write_step_clip(tmp_path, step=7, video=video, mask=mask, pose=pose)
    assert (tmp_path / "7.mp4").is_file()


def test_writer_swallows_io_errors(tmp_path: Path) -> None:
    """A bad target path logs a WARNING rather than crashing the training loop."""
    bad_target = tmp_path / "not_a_dir"
    bad_target.write_text("placeholder")
    video = torch.rand((1, 2, 3, 4, 4))
    mask = torch.zeros(1, 2)
    write_step_clip(bad_target, step=0, video=video, mask=mask)
    assert not (bad_target / "0.mp4").exists()


def test_rollout_recorder_saves_mp4(tmp_path: Path) -> None:
    """RolloutRecorder records per-step predictions and emits a single MP4."""
    from wormjepa.training.clip_writer import RolloutRecorder

    t_n, h, w, k = 4, 16, 16, 4
    video = torch.rand((t_n, 3, h, w))
    pose = torch.randn((t_n, k, 2)) * 0.1
    recorder = RolloutRecorder(video=video, pose=pose, max_steps=10)
    for step in range(1, 6):
        pred = torch.randn((t_n, k, 2)) * 0.1
        recorder.record(step=step, predicted_pose=pred)
    out_path = tmp_path / "training_evolution.mp4"
    recorder.save(out_path, fps=4)
    assert out_path.is_file()
    assert _frame_count(out_path) == 5


def test_rollout_recorder_caps_at_max_steps(tmp_path: Path) -> None:
    """``max_steps`` clamps how many predictions land in the rollup."""
    from wormjepa.training.clip_writer import RolloutRecorder

    t_n, h, w, k = 2, 8, 8, 3
    video = torch.rand((t_n, 3, h, w))
    pose = torch.randn((t_n, k, 2)) * 0.1
    recorder = RolloutRecorder(video=video, pose=pose, max_steps=2)
    for step in range(1, 6):
        recorder.record(step=step, predicted_pose=torch.randn((t_n, k, 2)) * 0.1)
    out_path = tmp_path / "rollup.mp4"
    recorder.save(out_path, fps=2)
    assert out_path.is_file()
    assert _frame_count(out_path) == 2


def test_rollout_recorder_save_noop_when_empty(tmp_path: Path) -> None:
    """No predictions recorded → save is a no-op (no file created)."""
    from wormjepa.training.clip_writer import RolloutRecorder

    recorder = RolloutRecorder(
        video=torch.rand((2, 3, 8, 8)),
        pose=torch.randn((2, 3, 2)) * 0.1,
    )
    out_path = tmp_path / "empty.mp4"
    recorder.save(out_path)
    assert not out_path.exists()
