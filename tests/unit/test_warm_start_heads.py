"""Unit tests for the four warm-start heads (Stories 5.4-5.7)."""

from __future__ import annotations

import pytest
import torch

from wormjepa.models.warm_start import (
    BehavioralHead,
    EigenwormHead,
    GraphPriorHead,
    NeuralAuxiliaryHead,
)


def test_eigenworm_head_requires_fit_first() -> None:
    head = EigenwormHead(latent_dim=32, n_eigen=4)
    latent = torch.zeros((2, 4, 32))
    pose = torch.zeros((2, 4, 10, 2))
    with pytest.raises(RuntimeError, match="fit_basis"):
        head(latent, pose)


def test_eigenworm_head_forward_returns_scalar_loss() -> None:
    head = EigenwormHead(latent_dim=32, n_eigen=4)
    flat_poses = torch.randn((50, 10 * 2))
    head.fit_basis(flat_poses)
    latent = torch.randn((2, 4, 32))
    pose = torch.randn((2, 4, 10, 2))
    loss = head(latent, pose)
    assert loss.ndim == 0
    assert loss >= 0


def test_eigenworm_head_n_eigen_bounded_by_latent_dim() -> None:
    with pytest.raises(ValueError, match="n_eigen"):
        EigenwormHead(latent_dim=4, n_eigen=8)


def test_graph_prior_head_requires_target() -> None:
    head = GraphPriorHead(latent_dim=32, n_neural_aligned_dims=8, n_edges=16)
    latent = torch.zeros((2, 4, 32))
    with pytest.raises(RuntimeError, match="set_target_edges"):
        head(latent)


def test_graph_prior_head_forward_works_after_target_set() -> None:
    head = GraphPriorHead(latent_dim=32, n_neural_aligned_dims=8, n_edges=16)
    head.set_target_edges(torch.randn(16))
    latent = torch.randn((2, 4, 32))
    loss = head(latent)
    assert loss.ndim == 0
    assert loss >= 0


def test_graph_prior_head_target_shape_validated() -> None:
    head = GraphPriorHead(latent_dim=32, n_neural_aligned_dims=8, n_edges=16)
    with pytest.raises(ValueError, match="edges shape"):
        head.set_target_edges(torch.zeros(99))


def test_neural_aux_head_forward() -> None:
    head = NeuralAuxiliaryHead(latent_dim=32, n_neurons=24)
    latent = torch.randn((2, 4, 32))
    neural = torch.randn((2, 4, 24))
    loss = head(latent, neural)
    assert loss.ndim == 0
    assert loss >= 0


def test_neural_aux_head_validates_target_shape() -> None:
    head = NeuralAuxiliaryHead(latent_dim=32, n_neurons=24)
    latent = torch.randn((1, 2, 32))
    neural = torch.randn((1, 2, 99))
    with pytest.raises(ValueError, match="n_neurons"):
        head(latent, neural)


def test_behavioral_head_forward() -> None:
    head = BehavioralHead(latent_dim=32, n_states=5)
    latent = torch.randn((2, 4, 32))
    labels = torch.randint(0, 5, (2, 4))
    loss = head(latent, labels)
    assert loss.ndim == 0
    assert loss >= 0
