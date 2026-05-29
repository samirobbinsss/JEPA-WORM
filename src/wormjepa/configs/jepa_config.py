"""Pydantic schema for ``configs/jepa_*.yaml`` configs (Story 5.11, 8.9, 8.11a).

The ``dataset:`` block (``DatasetLoaderSpec`` / ``DatasetSection`` /
``LoaderName``) moved to :mod:`wormjepa.configs.dataset` in Story 8.10 so
baseline configs can share the same schema. They are re-exported here for
backward compatibility — existing imports like
``from wormjepa.configs.jepa_config import DatasetSection`` still work.

Story 8.11a extends :class:`JEPASection` with V-JEPA 2.1 transfer-learning
fields (``frozen_target`` + ``vjepa_variant``). Defaults preserve legacy
behaviour: configs that omit the new fields construct a random-init timm
backbone exactly as Stories 5.1 / 5.11 / 8.9 shipped.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wormjepa.configs.dataset import (
    DatasetLoaderSpec,
    DatasetSection,
    LoaderName,
)
from wormjepa.configs.models import WormJEPAConfig
from wormjepa.training.loop import WarmStartFlags

__all__ = [
    "DatasetLoaderSpec",
    "DatasetSection",
    "JEPARunConfig",
    "JEPASection",
    "LoaderName",
    "VJEPAVariant",
]

VJEPAVariant = Literal[
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
]
"""V-JEPA 2.1 torch.hub entry points exposed by ``third_party/vjepa2``.

The Phase 0 headline run pins ``vjepa2_1_vit_base_384`` (80M params, the
only V-JEPA 2.1 variant with a plausible NFR1 margin on Apple Silicon MPS).
Larger variants are listed for completeness; selecting one is a deliberate
NFR1/NFR2/NFR3 review, not a default.
"""


class JEPASection(BaseModel):
    """The ``jepa:`` block of a JEPA-run config."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model_name: str = "vit_tiny_patch16_224"
    img_size: int = 64
    latent_dim: int = 32
    masking_ratio: float = 0.5
    n_steps: int = 2
    batch_size: int = Field(default=1, ge=1)
    """Samples per gradient step.

    ``batch_size=1`` reproduces the pre-batching loop (one clip per
    step). ``batch_size>1`` stacks N distinct clips into a real batch —
    required for the VicReg variance regularizer (``variance_reg_weight``)
    to function: with B=1 its "variance" is measured across the 16 frames
    of a single clip rather than across distinct samples, so it cannot
    counter latent collapse. The R3 collapse on the first headline run
    (2026-05-19) was traced to exactly this.
    """
    learning_rate: float = 1.0e-4
    head_learning_rate: float = 1.0e-4
    """Learning rate for the warm-start heads (own optimizer param-group).
    Default equals ``learning_rate`` — a single effective lr. In the
    two-phase curriculum the heads train on a frozen encoder in phase 2;
    a higher head lr lets them fit in the limited phase-2 budget."""
    warmup_steps: int = Field(default=0, ge=0)
    """Linear learning-rate warmup over the first ``warmup_steps`` gradient
    steps (0 -> learning_rate). Default 0 = no warmup (pre-2026-05-20
    behaviour). V-JEPA-style training warms up to stabilise the cold
    start; the headline-collapse research uses this."""
    ema_decay: float = 0.996
    predictor_layers: int = Field(default=1, ge=1)
    """JEPAPredictor depth. Default 1 (the original — far too shallow for
    a real masked-prediction task; V-JEPA's predictor is 6-12 layers)."""
    predictor_heads: int = Field(default=2, ge=1)
    """JEPAPredictor attention-head count. Default 2."""
    standardize_target: bool = True
    """When True (default), the frozen/EMA target latents are standardized
    to zero-mean / unit-variance per dim before the masked MSE. Standard
    I-JEPA / V-JEPA do NOT standardize the target — the EMA + stop-grad is
    the mechanism — and standardizing a collapsing EMA target divides by a
    vanishing std. The headline-collapse research toggles this off."""
    seed: int = 42
    warm_start: WarmStartFlags = Field(default_factory=WarmStartFlags)
    frozen_target: bool = False
    vjepa_variant: VJEPAVariant | None = None
    pretrained_checkpoint_sha: str | None = None
    variance_reg_weight: float = 0.0
    """VicReg-style variance regularizer weight (Story F3, R3-followup).

    When > 0, adds a term to the JEPA loss that penalises online-latent
    dims whose std drops below 1.0. Directly counters the
    online ≡ target collapse the 8.11c + R3 sweeps observed. Default 0
    preserves pre-R3 loss shape. The R3-followup headline.yaml sets a
    positive weight (initial choice: 1.0 per the VicReg paper recipe).
    """
    warm_start_loss_scale: float = 1.0
    """Global multiplier on every warm-start head's loss weight (eigenworm,
    graph_prior, neural, behavioral, pose_decoder). Does NOT scale the
    jepa loss or the variance/covariance regularizers. Default 1.0 keeps
    the per-head weights as-is. Experiment 5 (2026-05-20) showed the heads
    at full weight drag the encoder to collapse, overpowering the
    variance/covariance regs; a value < 1 lets the heads still warm-start
    without dominating the encoder geometry."""
    warm_start_after_step: int = Field(default=0, ge=0)
    """Two-phase curriculum boundary. While ``step < warm_start_after_step``
    the warm-start heads are inactive — the encoder trains pure-JEPA. At
    that step the online encoder is frozen and the heads switch on: they
    warm-start by *reading* the stable representation, never reshaping it.
    Default 0 = heads active from step 1 (joint training). Experiments 5
    and 6 showed joint training collapses the encoder at any head weight;
    the curriculum is the fix."""
    covariance_reg_weight: float = 0.0
    """VICReg-style covariance regularizer weight.

    When > 0, adds ``c_loss = sum_{i!=j} Cov(z)_{ij}^2 / D`` to the loss —
    the squared off-diagonal of the latent covariance, normalised by D.
    Decorrelates the latent dimensions. Experiment 2 (2026-05-20) showed
    the variance regularizer alone lifts per-dim std but lets the dims
    collapse onto a correlated subspace (cov_offdiag 12 -> 1395, jepa
    loss stalls). The covariance term is VICReg's missing third piece."""
    online_init: Literal["vjepa", "random"] = "vjepa"
    """How to initialise the online (trainable) encoder when frozen_target=True.

    - ``"vjepa"`` (default): both online and target start from identical
      V-JEPA 2.1 public weights. Standard transfer-learning posture but
      admits the trivial ``online ≡ target`` collapse the R3 sweep
      observed at step 500 (see phase0-r3-vitl-sweep-2026-05-18.md).
    - ``"random"``: online has the same V-JEPA 2.1 architecture but a
      fresh random init. Target still loads V-JEPA 2.1 weights. Breaks
      the symmetry; the JEPA loss has non-trivial gradient from step 1.
    """

    @model_validator(mode="after")
    def _require_variant_when_frozen(self) -> JEPASection:
        if self.frozen_target and self.vjepa_variant is None:
            msg = (
                "frozen_target=True requires vjepa_variant to be set "
                "(e.g. 'vjepa2_1_vit_base_384'). The frozen target encoder "
                "loads V-JEPA 2.1 public weights via torch.hub against "
                "third_party/vjepa2; there is no default."
            )
            raise ValueError(msg)
        return self


class JEPARunConfig(WormJEPAConfig):
    """Top-level config for a JEPA run."""

    jepa: JEPASection
    dataset: DatasetSection = Field(default_factory=DatasetSection)
