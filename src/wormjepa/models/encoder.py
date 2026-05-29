"""Vision-only JEPA encoder (Story 5.1).

Load-bearing architectural commitment (FR17): the encoder's ``forward()``
signature is exactly ``(self, video: torch.Tensor) -> torch.Tensor``. It does
not — and architecturally cannot — accept neural activity, pose, or any
auxiliary input. Warm-start heads (Stories 5.4-5.7) consume those signals
during training, but they are separate ``nn.Module`` instances composed by
the training loop, never the deployed encoder.

Architecture: V-JEPA 2.1-style ViT (timm-wrapped). The Phase 0 headline
configuration loads frozen V-JEPA 2.1 public weights via
:func:`wormjepa.models.vjepa_loader.build_frozen_vjepa_target` (Story
8.11a, closing the carried debt from Stories 5.1 / 5.8 / 7.9). The
``vit_tiny_patch16_224`` default here remains the *online*-side default
for smoke and ablation configs that do not invoke transfer learning.

Spatial input: ``(B, T, C, H, W)`` — batch, time, channels, height, width.
Time dimension is folded into batch for the ViT pass, then unfolded back.
Output shape: ``(B, T, D)`` per-frame latents.
"""

from __future__ import annotations

from typing import Literal, Self

import timm
import torch
from torch import nn


class WormJEPAEncoder(nn.Module):
    """timm-wrapped ViT operating on ``(B, T, C, H, W)`` video.

    Args:
        model_name: timm model name. Default ``vit_tiny_patch16_224``.
        pretrained: Whether to load timm's ImageNet pretraining (NOT V-JEPA 2.1).
        frozen: If ``True``, freeze all parameters (the V-JEPA target-encoder
            mode used by the EMA target branch; Story 5.2).
        latent_dim: If set, project ViT output to this dimension.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        frozen: bool = False,
        latent_dim: int | None = None,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        # num_classes=0 strips the classifier head; we want raw features.
        if img_size is None:
            self._backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            self._backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        backbone_dim = int(self._backbone.num_features)  # type: ignore[arg-type]
        self._latent_dim = latent_dim or backbone_dim
        if latent_dim is not None and latent_dim != backbone_dim:
            self._projector: nn.Module = nn.Linear(backbone_dim, latent_dim)
        else:
            self._projector = nn.Identity()

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @classmethod
    def from_frozen(cls, model_name: str = "vit_tiny_patch16_224") -> Self:
        """Construct a frozen, random-init timm ViT target encoder.

        Retained for smoke / ablation configs that explicitly want a frozen
        randomly-initialised target. For the Phase 0 headline run, prefer
        :func:`wormjepa.models.vjepa_loader.build_frozen_vjepa_target`,
        which loads the V-JEPA 2.1 public weights the architecture commits
        to (Story 8.11a).
        """
        return cls(model_name=model_name, pretrained=False, frozen=True)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``video`` (B, T, C, H, W) into per-frame latents (B, T, D).

        Args:
            video: ``(B, T, C, H, W)`` float tensor. Channels typically 3
                (RGB) or 1 (grayscale); the backbone is created for the
                channel count it was trained on (timm handles channel
                conversion via the first conv layer).

        Returns:
            ``(B, T, D)`` per-frame latent tensor where ``D = self.latent_dim``.

        Notes:
            No ``neural_activity`` / pose / auxiliary parameter. This is the
            architectural type-level enforcement of FR17.
        """
        if video.ndim != 5:
            msg = f"WormJEPAEncoder.forward expects (B, T, C, H, W); got shape {tuple(video.shape)}"
            raise ValueError(msg)
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        feats = self._backbone(flat)  # (B*T, backbone_dim)
        feats = self._projector(feats)  # (B*T, latent_dim)
        return feats.reshape(b, t, self._latent_dim)

    def forward_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``video`` into a singleton spatial-token grid ``(B, T, 1, D)``.

        The legacy timm-ViT path produces only a pooled per-frame feature
        vector — it has no exposed spatial token grid (``num_classes=0``
        returns the pooled CLS / mean feature). This method exists purely
        so the encoder honours the same ``forward_tokens(video) -> (B,T,S,D)``
        interface as :class:`wormjepa.models.vjepa_loader.TrainableVJEPAEncoder`,
        with ``S = 1``. The pose-decoder head's cross-attention degenerates
        to attending over a single token — equivalent to the old pooled
        path — so the non-V-JEPA smoke configs keep working unchanged.

        Args:
            video: ``(B, T, C, H, W)`` float tensor.

        Returns:
            ``(B, T, 1, D)`` per-frame latent with a singleton spatial axis.
        """
        return self.forward(video).unsqueeze(2)


# Module-level type-level guard: the encoder's forward signature must not gain
# a `neural_activity` parameter. Importing this module re-runs `inspect` at the
# pyright type-check level — a future erroneous edit that adds neural inputs
# trips this in code review (and at runtime via the assertion below).
def _enforce_forward_signature_invariant() -> Literal[True]:
    import inspect

    sig = inspect.signature(WormJEPAEncoder.forward)
    names = list(sig.parameters)
    if names != ["self", "video"]:
        msg = (
            f"FR17 violation: WormJEPAEncoder.forward parameters drifted to {names!r}. "
            "The deployed encoder must accept only video at test time."
        )
        raise AssertionError(msg)
    return True


_enforce_forward_signature_invariant()
