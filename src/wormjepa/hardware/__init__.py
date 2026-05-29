"""Hardware-pilot scaffolding (Phase A prep).

Software-first per the project's memory policy: this module defines
the **contract** the eventual hardware pilot will satisfy, with a
mock implementation that lets the rest of the pipeline run without
any physical camera. Real OpenCV / pyav / pylon / spinnaker drivers
slot in behind the same interface at Phase A kickoff.

Per-module surface:

- :mod:`wormjepa.hardware.camera_pair` — :class:`CameraPair` ABC plus
  the :class:`MockCameraPair` driver-free implementation that yields
  deterministic synthetic stereo frames.
- :mod:`wormjepa.hardware.two_view_contract` — :class:`TwoViewSample`
  dataclass; the loader-facing record shape. Maps onto
  :class:`wormjepa.data.contract.DatasetSample` via
  :meth:`TwoViewSample.to_dataset_sample`, which today collapses to
  the primary view only (the encoder is single-view per FR17). The
  secondary view is preserved on :class:`TwoViewSample` so a future
  Phase A multi-view encoder can read it without further contract
  churn.

The hardware module ships zero pre-registered artefacts and zero
gate-evaluating code. It is purely infrastructure for when the
project graduates from public-data to in-house experiments.
"""

from wormjepa.hardware.camera_pair import CameraPair, MockCameraPair
from wormjepa.hardware.file_backed_camera import FileBackedCameraPair
from wormjepa.hardware.live_camera import LiveCameraPair
from wormjepa.hardware.two_view_contract import TwoViewSample

__all__ = [
    "CameraPair",
    "FileBackedCameraPair",
    "LiveCameraPair",
    "MockCameraPair",
    "TwoViewSample",
]
