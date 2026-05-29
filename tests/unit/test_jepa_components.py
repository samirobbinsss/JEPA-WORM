"""Unit tests for EMA target, masking, predictor, and pose decoder (Stories 5.2-5.3)."""

from __future__ import annotations

import pytest
import torch

from wormjepa.models.ema import EMATarget
from wormjepa.models.encoder import WormJEPAEncoder
from wormjepa.models.masking import random_temporal_mask
from wormjepa.models.pose_decoder import PoseDecoderHead
from wormjepa.models.predictor import JEPAPredictor


def _encoder() -> WormJEPAEncoder:
    return WormJEPAEncoder(model_name="vit_tiny_patch16_224", latent_dim=32)


# --- masking ---


def test_random_temporal_mask_shape() -> None:
    mask = random_temporal_mask(n_frames=8, n_batches=4, masking_ratio=0.5)
    assert mask.shape == (4, 8)


def test_random_temporal_mask_has_both_classes_per_row() -> None:
    mask = random_temporal_mask(n_frames=10, n_batches=5, masking_ratio=0.5)
    for row in mask:
        assert row.any()
        assert not row.all()


def test_random_temporal_mask_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="masking_ratio"):
        random_temporal_mask(n_frames=8, n_batches=2, masking_ratio=0.0)
    with pytest.raises(ValueError, match="masking_ratio"):
        random_temporal_mask(n_frames=8, n_batches=2, masking_ratio=1.0)


def test_random_temporal_mask_deterministic_with_generator() -> None:
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    m1 = random_temporal_mask(n_frames=8, n_batches=3, masking_ratio=0.5, generator=g1)
    m2 = random_temporal_mask(n_frames=8, n_batches=3, masking_ratio=0.5, generator=g2)
    assert torch.equal(m1, m2)


# --- EMA target ---


def test_ema_target_initial_weights_match_online() -> None:
    online = _encoder()
    target = EMATarget(online, decay=0.99)
    for p_t, p_o in zip(target.target_encoder.parameters(), online.parameters(), strict=True):
        assert torch.equal(p_t, p_o)


def test_ema_target_update_moves_toward_online() -> None:
    online = _encoder()
    target = EMATarget(online, decay=0.5)
    # Mutate online by adding 1 to every parameter.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    target.update(online)
    # After one update with decay=0.5, target = 0.5·target_old + 0.5·online_new.
    # Since target_old was equal to old online (now +1 less), target should be
    # halfway between the two.
    for p_t, p_o in zip(target.target_encoder.parameters(), online.parameters(), strict=True):
        assert torch.allclose(p_t, p_o - 0.5, atol=1e-5)


def test_ema_target_parameters_frozen() -> None:
    online = _encoder()
    target = EMATarget(online)
    for p in target.target_encoder.parameters():
        assert p.requires_grad is False


def test_ema_target_decay_validated() -> None:
    online = _encoder()
    with pytest.raises(ValueError, match="decay"):
        EMATarget(online, decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        EMATarget(online, decay=1.0)


def test_ema_target_forward_runs_in_no_grad() -> None:
    online = _encoder()
    target = EMATarget(online)
    video = torch.zeros((1, 2, 3, 224, 224))
    out = target(video)
    assert out.shape == (1, 2, 32)
    assert out.requires_grad is False


# --- Predictor ---


def test_predictor_output_shape() -> None:
    pred = JEPAPredictor(latent_dim=32, n_layers=1, n_heads=2)
    latents = torch.zeros((2, 6, 32))
    mask = torch.zeros((2, 6), dtype=torch.bool)
    mask[:, 3:] = True
    out = pred(latents, mask)
    assert out.shape == latents.shape


def test_predictor_substitutes_mask_token_at_masked_positions() -> None:
    """Predictor output at masked positions ignores the online latent value at those positions."""
    pred = JEPAPredictor(latent_dim=16, n_layers=1, n_heads=2)
    pred.eval()
    latents_a = torch.randn((1, 4, 16))
    latents_b = latents_a.clone()
    mask = torch.tensor([[False, False, True, True]])
    # Overwrite the masked positions in latents_b with garbage values.
    latents_b[:, 2:] = 999.0
    with torch.no_grad():
        out_a = pred(latents_a, mask)
        out_b = pred(latents_b, mask)
    # Outputs should be identical because the masked positions are replaced
    # by the learned mask token regardless of incoming latent.
    assert torch.allclose(out_a, out_b)


# --- Encoder forward_tokens (spatial-token grid) ---


def test_wormjepa_encoder_forward_tokens_shape_and_pool_identity() -> None:
    """`forward_tokens` returns (B, T, S, D) with forward() == forward_tokens().mean(2).

    The legacy timm encoder has no spatial grid, so S == 1.
    """
    encoder = WormJEPAEncoder(model_name="vit_tiny_patch16_224", latent_dim=24)
    video = torch.zeros((2, 3, 3, 224, 224))
    tokens = encoder.forward_tokens(video)
    assert tokens.shape == (2, 3, 1, 24), tokens.shape
    pooled = encoder(video)
    assert pooled.shape == (2, 3, 24)
    assert torch.allclose(pooled, tokens.mean(dim=2))


# --- PoseDecoderHead (spatial-token cross-attention) ---


def test_pose_decoder_predict_shape_on_spatial_tokens() -> None:
    """`predict` consumes (B, T, S, D) with S>1 and returns (B, T, K, 2)."""
    head = PoseDecoderHead(latent_dim=32, n_keypoints=49)
    tokens = torch.randn(2, 4, 9, 32)  # B=2, T=4, S=9 spatial tokens, D=32
    out = head.predict(tokens)
    assert out.shape == (2, 4, 49, 2)


def test_pose_decoder_forward_returns_scalar_mse() -> None:
    """`forward` returns a scalar MSE loss against ground-truth keypoints."""
    head = PoseDecoderHead(latent_dim=16, n_keypoints=7, n_heads=4)
    tokens = torch.randn(3, 5, 6, 16)  # S=6 spatial tokens
    pose = torch.randn(3, 5, 7, 2)
    loss = head(tokens, pose)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_pose_decoder_rejects_mismatched_keypoints() -> None:
    head = PoseDecoderHead(latent_dim=16, n_keypoints=7)
    tokens = torch.randn(1, 2, 4, 16)
    bad_pose = torch.randn(1, 2, 9, 2)  # K=9 != head's 7
    with pytest.raises(ValueError, match="n_keypoints"):
        head(tokens, bad_pose)


def test_pose_decoder_rejects_non_4d_input() -> None:
    """A pooled (B, T, D) latent must be rejected — the head needs the grid."""
    head = PoseDecoderHead(latent_dim=16, n_keypoints=7)
    pooled = torch.randn(1, 2, 16)  # (B, T, D) — missing spatial axis
    with pytest.raises(ValueError, match=r"\(B, T, S, D\)"):
        head.predict(pooled)


def test_pose_decoder_singleton_spatial_grid_works() -> None:
    """S == 1 (legacy encoder path) decodes without error."""
    head = PoseDecoderHead(latent_dim=24, n_keypoints=4)
    tokens = torch.randn(2, 3, 1, 24)  # singleton spatial grid
    assert head.predict(tokens).shape == (2, 3, 4, 2)
