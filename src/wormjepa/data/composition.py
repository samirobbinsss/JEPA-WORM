"""Loader composition — instantiate + chain loaders from a config spec.

Both ``wormjepa.training.runner`` (Story 8.9) and
``wormjepa.baselines._runner`` (Story 8.10) consume the same
:class:`wormjepa.configs.dataset.DatasetSection`; this module is the
single place that translates a list of :class:`DatasetLoaderSpec`s into
an iterable of :class:`DatasetSample`.
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from wormjepa import WormJEPAError
from wormjepa.data import DatasetSample
from wormjepa.data.loaders.baaiworm import BAAIWormLoader
from wormjepa.data.loaders.flavell_2023 import Flavell2023Loader
from wormjepa.data.loaders.openworm_movement import OpenWormMovementLoader
from wormjepa.data.loaders.synthetic import SyntheticLoader
from wormjepa.data.loaders.two_camera_mock import TwoCameraMockLoader
from wormjepa.data.loaders.wormbehavior_db import WormBehaviorDBLoader
from wormjepa.data.loaders.wormid import WormIDLoader

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from wormjepa.configs.dataset import DatasetLoaderSpec


class ChainedLoader:
    """Wrap multiple loaders so iteration chains through them in declared order.

    The training and baseline loops expect an object with
    ``__iter__() -> Iterator[DatasetSample]``; Python's ``itertools.chain``
    doesn't support re-iteration, so wrap explicitly.
    """

    def __init__(self, loaders: list[Iterable[DatasetSample]]) -> None:
        self._loaders = loaders

    def __iter__(self) -> Iterator[DatasetSample]:
        return chain.from_iterable(iter(loader) for loader in self._loaders)


def build_loader(specs: list[DatasetLoaderSpec], seed: int) -> Iterable[DatasetSample]:
    """Construct an iterable from a list of :class:`DatasetLoaderSpec`.

    A single spec returns the underlying loader directly. Multiple specs are
    wrapped in :class:`ChainedLoader` which iterates them in declared order.

    Raises:
        WormJEPAError: An unknown loader name was provided or a real loader is
            missing its required ``local_root``.
    """
    if not specs:
        msg = "dataset.loaders must contain at least one entry"
        raise WormJEPAError(msg)
    built: list[Iterable[DatasetSample]] = [_build_one(spec, seed) for spec in specs]
    return built[0] if len(built) == 1 else ChainedLoader(built)


def _build_one(spec: DatasetLoaderSpec, seed: int) -> Iterable[DatasetSample]:
    image_size: tuple[int, int] | None
    image_size = (spec.image_size, spec.image_size) if spec.image_size > 0 else None
    if spec.name == "synthetic":
        return SyntheticLoader(
            n_worms=spec.n_worms,
            clips_per_worm=spec.clips_per_worm,
            clip_frames=spec.clip_frames,
            image_size=image_size or (64, 64),
            n_keypoints=spec.n_keypoints,
            n_neurons=spec.n_neurons,
            seed=seed,
        )
    if spec.name == "baaiworm":
        return BAAIWormLoader(
            n_worms=spec.n_worms,
            clips_per_worm=spec.clips_per_worm,
            clip_frames=spec.clip_frames,
            image_size=image_size or (64, 64),
            n_keypoints=spec.n_keypoints,
            n_neurons=spec.n_neurons,
            seed=seed,
        )
    if spec.name == "two_camera_mock":
        return TwoCameraMockLoader(
            n_worms=spec.n_worms,
            clips_per_worm=spec.clips_per_worm,
            clip_frames=spec.clip_frames,
            image_size=image_size or (64, 64),
            seed=seed,
        )
    if spec.local_root is None:
        msg = f"dataset loader {spec.name!r} requires `local_root` to be set"
        raise WormJEPAError(msg)
    if spec.name == "flavell_2023":
        return Flavell2023Loader(
            local_root=spec.local_root,
            clip_frames=spec.clip_frames,
            image_size=image_size,
        )
    if spec.name == "wormbehavior_db":
        return WormBehaviorDBLoader(
            local_root=spec.local_root,
            clip_frames=spec.clip_frames,
            image_size=image_size,
        )
    if spec.name == "openworm_movement":
        return OpenWormMovementLoader(
            local_root=spec.local_root,
            clip_frames=spec.clip_frames,
            image_size=image_size,
        )
    if spec.name == "wormid":
        return WormIDLoader(
            local_dandi_root=spec.local_root,
            cohort=spec.cohort,
            clip_frames=spec.clip_frames,
            image_size=image_size,
        )
    msg = f"unknown loader name: {spec.name!r}"
    raise WormJEPAError(msg)
