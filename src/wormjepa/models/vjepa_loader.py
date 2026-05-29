"""Frozen V-JEPA 2.1 target encoder loader (Story 8.11a).

Closes the carried debt from Stories 5.1 / 5.8 / 7.9: the EMA target branch
was constructed from a deep-copied random-init timm ViT instead of loading
the V-JEPA 2.1 public checkpoint the PRD / architecture committed to as the
NFR2-budget-load-bearing transfer-learning starting point.

This module:

1. Downloads (and SHA-verifies, when the SHA is pinned) the V-JEPA 2.1
   public checkpoint from ``dl.fbaipublicfiles.com`` into the user-level
   cache at ``~/.cache/wormjepa/checkpoints/`` — mirroring the
   :mod:`wormjepa.data.download` pattern for datasets.
2. Constructs the V-JEPA 2.1 encoder via ``torch.hub.load(...)`` against the
   vendored ``third_party/vjepa2`` submodule with ``source='local'`` and
   ``pretrained=False`` (the upstream entry point's pretrained path is
   currently broken because ``VJEPA_BASE_URL`` is hardcoded to a test
   localhost address; we avoid it).
3. Loads the downloaded state dict manually, strips the standard
   ``module.`` / ``backbone.`` prefixes (vendored inline so we do not depend
   on upstream-private ``_clean_backbone_key``), and freezes every
   parameter for use as the JEPA target encoder.
4. Wraps the V-JEPA 2.1 encoder in :class:`FrozenVJEPATarget` so its
   forward signature matches the project-wide ``(B, T, C, H, W) → (B, T, D)``
   contract that the rest of the training loop expects.

The V-JEPA 2.1 SHA is not pinned by this module; pinning lands in Story
8.11b alongside the substantive ``configs/headline.yaml`` populate so the
pre-registration MANIFEST canonicalisation captures it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Self

import torch
from torch import nn

from wormjepa import WormJEPAError
from wormjepa.configs.jepa_config import VJEPAVariant

logger = logging.getLogger(__name__)


_VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
"""Upstream public-checkpoint host. The vendored ``third_party/vjepa2`` ships
with this URL commented out and a localhost placeholder in its place; this
module owns the canonical URL so we do not depend on the upstream constant."""

_CHECKPOINT_FILENAMES: dict[str, str] = {
    "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384.pt",
    "vjepa2_1_vit_giant_384": "vjepa2_1_vitg_384.pt",
    "vjepa2_1_vit_gigantic_384": "vjepa2_1_vitG_384.pt",
}

_TUBELET_SIZE = 2
"""V-JEPA 2.1 temporal tubelet size (frames per spatiotemporal patch)."""


def _default_cache_dir() -> Path:
    base = os.environ.get("WORMJEPA_CHECKPOINT_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "wormjepa" / "checkpoints"


def _sha256_of(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_vjepa_checkpoint(
    variant: VJEPAVariant,
    expected_sha256: str | None = None,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Fetch the V-JEPA 2.1 public checkpoint for ``variant`` into the cache.

    Args:
        variant: V-JEPA 2.1 torch.hub entrypoint name (e.g.
            ``"vjepa2_1_vit_base_384"``).
        expected_sha256: Optional pinned SHA-256. When supplied, the
            downloaded (or cached) file is verified and the function raises
            :class:`WormJEPAError` on mismatch. When ``None``, the function
            logs the observed SHA so the caller can pin it later (e.g. into
            ``configs/headline.yaml`` at Story 8.11b time).
        cache_dir: Override for the cache root. Defaults to
            ``~/.cache/wormjepa/checkpoints/`` (or the ``WORMJEPA_CHECKPOINT_DIR``
            environment variable).

    Returns:
        Local path to the verified checkpoint.

    Raises:
        WormJEPAError: On download failure or SHA mismatch.
    """
    filename = _CHECKPOINT_FILENAMES[variant]
    cache = cache_dir or _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / filename

    if not dest.exists():
        url = f"{_VJEPA_BASE_URL}/{filename}"
        logger.info("downloading V-JEPA 2.1 checkpoint %s → %s", url, dest)
        try:
            urllib.request.urlretrieve(url, dest)
        except OSError as exc:
            dest.unlink(missing_ok=True)
            msg = f"Failed to download V-JEPA 2.1 checkpoint from {url}: {exc}"
            raise WormJEPAError(msg) from exc

    actual_sha = _sha256_of(dest)
    if expected_sha256 is None:
        logger.warning(
            "V-JEPA 2.1 checkpoint %s loaded without SHA pin; observed SHA-256 = %s. "
            "Pin this value in configs/headline.yaml at Story 8.11b.",
            dest,
            actual_sha,
        )
    elif actual_sha.lower() != expected_sha256.lower():
        msg = (
            f"V-JEPA 2.1 checkpoint SHA mismatch at {dest}: "
            f"expected {expected_sha256}, got {actual_sha}."
        )
        raise WormJEPAError(msg)

    return dest


def _clean_backbone_key(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the standard ``module.`` / ``backbone.`` prefixes from a state dict.

    Vendored from ``third_party/vjepa2/src/hub/backbones.py::_clean_backbone_key``
    (MIT licence) so this module does not reach into a private upstream helper.
    """
    cleaned: dict[str, Any] = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned[new_key] = val
    return cleaned


def _vjepa_repo_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "vjepa2"


def build_vjepa_encoder(
    variant: VJEPAVariant,
    *,
    pretrained_checkpoint_sha: str | None = None,
    cache_dir: Path | None = None,
    random_init: bool = False,
) -> tuple[nn.Module, int]:
    """Construct a V-JEPA 2.1 encoder.

    When ``random_init=False`` (default), the V-JEPA 2.1 public weights for
    ``variant`` are downloaded (or read from cache) and loaded into the
    encoder. When ``random_init=True``, the encoder is returned with its
    fresh random initialisation untouched — same architecture, no weight
    load — supporting the Phase 0 Story F1 methodology fix that breaks the
    ``online == target`` symmetry by initialising the online branch
    randomly while the target stays frozen V-JEPA 2.1.

    Args:
        variant: V-JEPA 2.1 torch.hub entrypoint name.
        pretrained_checkpoint_sha: Optional pinned SHA-256 of the checkpoint
            file. Passed through to :func:`download_vjepa_checkpoint`.
            Ignored when ``random_init=True``.
        cache_dir: Override for the checkpoint cache root. Ignored when
            ``random_init=True``.
        random_init: When True, skip the weight load (no checkpoint
            download, no SHA verification, no ``encoder.load_state_dict``).
            The encoder is returned with its torch.hub-provided random
            init at the correct V-JEPA 2.1 architecture.

    Returns:
        ``(encoder, embed_dim)``: the encoder ``nn.Module`` (loaded or
        random-init per ``random_init``) and its ViT embedding dimension
        (768 for ViT-B/16, 1024 for ViT-L/16).

    Notes:
        - The upstream ``vjepa2_1_vit_*`` torch.hub entry points fetch the
          checkpoint themselves when ``pretrained=True``, but they fetch from
          a hardcoded localhost test URL (see ``backbones.py:11``). We
          therefore call the entry point with ``pretrained=False`` to get
          the freshly-initialised architecture and load the real weights
          ourselves from the cached file.
        - The submodule's ``hubconf.py`` imports relative to its own root, so
          we prepend the submodule path to ``sys.path`` before invoking the
          entry point.
    """
    repo_root = _vjepa_repo_root()
    if not (repo_root / "hubconf.py").exists():
        msg = (
            f"V-JEPA 2.1 submodule not found at {repo_root}. Run "
            f"'git submodule update --init third_party/vjepa2' to fetch it."
        )
        raise WormJEPAError(msg)

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # torch.hub.load's typed signature is overly strict: source='local' returns
    # whatever the hubconf entrypoint returns (here a (encoder, predictor) tuple),
    # but the type annotation says ``object``. Cast through Any. trust_repo
    # accepts bool at runtime but the stub only types it as ``str``; pass via
    # **kwargs to bypass the stub mismatch.
    hub_kwargs: dict[str, Any] = {
        "source": "local",
        "pretrained": False,
        "verbose": False,
        "trust_repo": True,
    }
    loaded: Any = torch.hub.load(repo_root_str, variant, **hub_kwargs)
    encoder, _predictor = loaded
    embed_dim = int(encoder.embed_dim)

    if random_init:
        logger.info(
            "build_vjepa_encoder(%s, random_init=True): skipping weight load; "
            "encoder retains torch.hub random init.",
            variant,
        )
        return encoder, embed_dim

    ckpt_path = download_vjepa_checkpoint(
        variant,
        expected_sha256=pretrained_checkpoint_sha,
        cache_dir=cache_dir,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_sd = _clean_backbone_key(state_dict["ema_encoder"])
    encoder.load_state_dict(encoder_sd, strict=True)
    return encoder, embed_dim


class FrozenVJEPATarget(nn.Module):
    """JEPA target encoder backed by frozen V-JEPA 2.1 ViT weights.

    Wraps the V-JEPA 2.1 encoder (which expects ``(B, C, T, H, W)`` and
    returns ``(B, N_tokens, D)``) into the project-wide
    ``(B, T, C, H, W) → (B, T, D)`` contract that :class:`EMATarget` and the
    rest of the training loop consume.

    The wrapper:

    - Transposes the input from ``(B, T, C, H, W)`` to ``(B, C, T, H, W)``.
    - Mean-pools the encoder's output spatial tokens per time-tubelet group,
      yielding ``(B, T // tubelet_size, D)``.
    - Repeat-interleaves along the time axis so the output shape matches the
      caller's frame count ``T``, then trims to the original ``T``.

    All parameters are frozen (``requires_grad=False``) and the module is
    held in ``eval()`` mode; calls run under ``torch.no_grad()`` to match
    :class:`EMATarget`'s stop-grad semantics.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        tubelet_size: int = _TUBELET_SIZE,
    ) -> None:
        super().__init__()
        for p in encoder.parameters():
            p.requires_grad_(False)
        self._encoder = encoder
        self._embed_dim = int(embed_dim)
        self._tubelet_size = int(tubelet_size)
        # Force eval mode for the wrapper and every child. ``nn.Module.__init__``
        # leaves ``self.training=True`` by default; without this call the
        # wrapper would report training mode even though every parameter is
        # frozen. ``train()`` is also overridden below to keep the wrapper
        # eval-only across the training loop's online.train() / target.eval()
        # discipline.
        self.eval()

    def train(self, mode: bool = True) -> Self:
        """No-op override: the frozen target stays in eval() regardless.

        The standard training loop calls ``model.train()`` and ``model.eval()``
        on every member of :class:`JEPATrainingState`. The frozen V-JEPA 2.1
        target must never leave eval mode (dropout / norm statistics must
        stay deterministic), so we ignore the requested mode and force eval.
        """
        return super().train(False)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def latent_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def forward_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``video`` into the per-frame *spatial-token grid*.

        Identical to :meth:`forward` but returns the spatial-token tensor
        *before* the ``.mean(dim=2)`` spatial pool, so callers (the pose
        decoder) can attend over *where* features sit, not just the
        spatially-averaged latent.

        Args:
            video: ``(B, T, C, H, W)`` float tensor.

        Returns:
            ``(B, T, S, D)`` — ``S`` spatial tokens per frame, ``D`` the
            ViT embedding dim. Time is upsampled by ``repeat_interleave``
            over the tubelet and trimmed to the input frame count ``T``.
        """
        if video.ndim != 5:
            msg = (
                f"FrozenVJEPATarget.forward_tokens expects (B, T, C, H, W); "
                f"got {tuple(video.shape)}"
            )
            raise ValueError(msg)
        b, t = video.shape[0], video.shape[1]
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        feats = self._encoder(x)  # (B, N_tokens, D)
        if feats.ndim != 3 or feats.shape[0] != b or feats.shape[2] != self._embed_dim:
            msg = (
                f"Unexpected V-JEPA 2.1 encoder output shape {tuple(feats.shape)}; "
                f"expected (B={b}, N_tokens, D={self._embed_dim})."
            )
            raise WormJEPAError(msg)
        t_eff = max(t // self._tubelet_size, 1)
        n_tokens = feats.shape[1]
        if n_tokens % t_eff != 0:
            msg = (
                f"V-JEPA 2.1 token count {n_tokens} not divisible by effective time "
                f"dimension {t_eff} (T={t}, tubelet={self._tubelet_size}); "
                f"check that input T is a multiple of tubelet_size."
            )
            raise WormJEPAError(msg)
        spatial_tokens = n_tokens // t_eff
        feats = feats.reshape(b, t_eff, spatial_tokens, self._embed_dim)
        upsampled = feats.repeat_interleave(self._tubelet_size, dim=1)
        return upsampled[:, :t, :, :]

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(video).mean(dim=2)


def build_frozen_vjepa_target(
    variant: VJEPAVariant,
    *,
    pretrained_checkpoint_sha: str | None = None,
    cache_dir: Path | None = None,
) -> FrozenVJEPATarget:
    """One-shot constructor: download + load + wrap as a frozen target."""
    encoder, embed_dim = build_vjepa_encoder(
        variant,
        pretrained_checkpoint_sha=pretrained_checkpoint_sha,
        cache_dir=cache_dir,
    )
    return FrozenVJEPATarget(encoder, embed_dim=embed_dim, tubelet_size=_TUBELET_SIZE)


class TrainableVJEPAEncoder(nn.Module):
    """Online (trainable) JEPA encoder initialised from V-JEPA 2.1 weights.

    Same I/O contract as :class:`FrozenVJEPATarget`
    (``(B, T, C, H, W) → (B, T, embed_dim)``), but parameters remain
    trainable so the JEPA training loop can fine-tune the online branch.

    Architectural rationale: when ``frozen_target=True`` the online and
    target encoders must produce latents in the same dimension for the
    predictor's loss to be defined. Both are constructed from the same
    V-JEPA 2.1 variant; the online copy stays trainable, the target copy
    (:class:`FrozenVJEPATarget`) stays frozen.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        tubelet_size: int = _TUBELET_SIZE,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._embed_dim = int(embed_dim)
        self._tubelet_size = int(tubelet_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def latent_dim(self) -> int:
        return self._embed_dim

    def forward_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``video`` into the per-frame *spatial-token grid*.

        Identical to :meth:`forward` but returns the spatial-token tensor
        *before* the ``.mean(dim=2)`` spatial pool, so the pose-decoder
        head can cross-attend over the un-pooled grid.

        Args:
            video: ``(B, T, C, H, W)`` float tensor.

        Returns:
            ``(B, T, S, D)`` — ``S`` spatial tokens per frame.
        """
        if video.ndim != 5:
            msg = (
                f"TrainableVJEPAEncoder.forward_tokens expects (B, T, C, H, W); "
                f"got {tuple(video.shape)}"
            )
            raise ValueError(msg)
        b, t = video.shape[0], video.shape[1]
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        feats = self._encoder(x)  # (B, N_tokens, D)
        if feats.ndim != 3 or feats.shape[0] != b or feats.shape[2] != self._embed_dim:
            msg = (
                f"Unexpected V-JEPA 2.1 encoder output shape {tuple(feats.shape)}; "
                f"expected (B={b}, N_tokens, D={self._embed_dim})."
            )
            raise WormJEPAError(msg)
        t_eff = max(t // self._tubelet_size, 1)
        n_tokens = feats.shape[1]
        if n_tokens % t_eff != 0:
            msg = (
                f"V-JEPA 2.1 token count {n_tokens} not divisible by effective time "
                f"dimension {t_eff} (T={t}, tubelet={self._tubelet_size}); "
                f"check that input T is a multiple of tubelet_size."
            )
            raise WormJEPAError(msg)
        spatial_tokens = n_tokens // t_eff
        feats = feats.reshape(b, t_eff, spatial_tokens, self._embed_dim)
        upsampled = feats.repeat_interleave(self._tubelet_size, dim=1)
        return upsampled[:, :t, :, :]

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(video).mean(dim=2)


def build_trainable_vjepa_encoder(
    variant: VJEPAVariant,
    *,
    pretrained_checkpoint_sha: str | None = None,
    cache_dir: Path | None = None,
    random_init: bool = False,
) -> TrainableVJEPAEncoder:
    """One-shot constructor: download + load (or skip-load when
    ``random_init=True``) + wrap as a trainable online encoder.

    With ``random_init=True``, the online encoder has the same V-JEPA 2.1
    architecture but a fresh random initialisation — the methodological
    fix that breaks the ``online == target`` collapse symmetry. See
    :func:`build_vjepa_encoder` and the Phase 0 R3-outcome doc.
    """
    encoder, embed_dim = build_vjepa_encoder(
        variant,
        pretrained_checkpoint_sha=pretrained_checkpoint_sha,
        cache_dir=cache_dir,
        random_init=random_init,
    )
    return TrainableVJEPAEncoder(encoder, embed_dim=embed_dim, tubelet_size=_TUBELET_SIZE)
