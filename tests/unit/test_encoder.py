"""Unit tests for ``wormjepa.models.encoder``."""

from __future__ import annotations

import inspect

import torch

from wormjepa.models import WormJEPAEncoder


def _small_encoder(latent_dim: int = 32) -> WormJEPAEncoder:
    return WormJEPAEncoder(model_name="vit_tiny_patch16_224", latent_dim=latent_dim)


def test_forward_signature_is_video_only() -> None:
    """FR17 type-level enforcement: forward accepts only video."""
    sig = inspect.signature(WormJEPAEncoder.forward)
    names = list(sig.parameters)
    assert names == ["self", "video"], names


def test_no_neural_activity_parameter() -> None:
    """The encoder cannot accept neural_activity. Catches accidental future edits."""
    sig = inspect.signature(WormJEPAEncoder.forward)
    assert "neural_activity" not in sig.parameters
    assert "neural" not in sig.parameters
    assert "pose" not in sig.parameters


def test_forward_shape_round_trip() -> None:
    encoder = _small_encoder(latent_dim=64)
    video = torch.zeros((2, 3, 3, 224, 224))  # B=2, T=3, C=3, H=W=224
    latents = encoder(video)
    assert latents.shape == (2, 3, 64)


def test_forward_rejects_4d_input() -> None:
    encoder = _small_encoder()
    try:
        encoder(torch.zeros((3, 3, 224, 224)))
    except ValueError as exc:
        assert "(B, T, C, H, W)" in str(exc)
        return
    raise AssertionError("Encoder should reject non-5D input")


def test_frozen_mode_no_grad() -> None:
    encoder = WormJEPAEncoder.from_frozen()
    assert all(not p.requires_grad for p in encoder.parameters())


def test_default_latent_dim_matches_backbone() -> None:
    encoder = WormJEPAEncoder(model_name="vit_tiny_patch16_224")
    assert encoder.latent_dim == encoder._backbone.num_features


def test_explicit_latent_dim_adds_projector() -> None:
    encoder = _small_encoder(latent_dim=16)
    assert encoder.latent_dim == 16
    video = torch.zeros((1, 2, 3, 224, 224))
    assert encoder(video).shape == (1, 2, 16)
