# JEPA-WORM — Experiment Plan: Diagnosing the kill_criterion firing

> Scoped 2026-06-16 from the autonomous overnight run (`AUTONOMOUS_REPORT.md`). Grounded in the eval/readout/encoder code by a 3-agent investigation.

## Diagnosis & single most important next step

Training is healthy (loss 3201→5.84, std_mean→0.54, no collapse), so the encoder is learning *something*. The kill_criterion fired because JEPA future-pose error is ~3.3× the eigenworm baseline — but that headline ratio is **confounded** and likely **artifact-inflated**. Two independent confounds dominate:
1. **The JEPA readout discards real spatial structure** — the eval mean-pools the encoder to `(T,D)` then fakes a singleton `S=1` token (`orchestrator.py:833` `unsqueeze(2)`), even though `PoseDecoderHead` is built to cross-attend a *multi-token* grid and its own docstring says pooled-latent decoding "scatters randomly and never converges" (`pose_decoder.py:10-16`). The real spatial tokens are available via `forward_tokens()` (`vjepa_loader.py:410-450`) but discarded.
2. **The comparison is contaminated** — the eigenworm baseline is fit on the eval cohort, and the JEPA routes through a non-pre-registered dev-loop pose head.

Pose information may well be *present* in the encoder's spatial tokens but *invisible* to the current pooled readout. **Both confounds are fixable EVAL-ONLY (hours, zero GPU retrain) on the existing checkpoint.**

**Most important next step:** run **Tier 0 / Exp A** — re-decode the *existing checkpoint* using the real `forward_tokens()` spatial grid (S>1) instead of the faked S=1 pool. Hours, no retrain. Only if pose info is genuinely absent after Tier 0 do we spend a 14h retrain.

---

## Implementation status (2026-06-18)

**Exp A + Exp E are implemented, unit-tested, committed** (`feat(eval): exploratory Exp A pose-decodability + Exp E held-out baseline`). Both are env-gated and OFF by default, so the canonical frozen eval is byte-identical when unset.
- **Exp A** — `WORMJEPA_EXP_POSE_DECODABILITY=1` adds a `pose_decodability_r2` entry to the eval: leave-one-worm-out ridge pose-R² from the spatial-token summary `[mean,std,max]` vs the mean-pooled latent. `partial = spatial − pooled > 0` ⇒ mean-pooling discards pose-relevant structure (the readout the kill_criterion uses).
- **Exp E** — `WORMJEPA_EXP_BASELINE_HOLDOUT=1` fits the eigenworm baseline on a disjoint `seed+20000` split (eval cohort unseen), removing the train-test contamination.
- **Run both:** `WORMJEPA_EXP_POSE_DECODABILITY=1 WORMJEPA_EXP_BASELINE_HOLDOUT=1 wormjepa eval --run <run-id>`.

**⚠️ Checkpoint reality:** the seed-42 checkpoint was lost when the pod was terminated (2026-06-18). Tier 0 is therefore **no longer eval-only** — it needs a re-train of seed 42 (~14h) first; the instrumented eval then answers Exp A + Exp E in one post-train pass. **Pull the checkpoint local this time.**

## Tier 0 — Eval-only on existing checkpoint (hours, no retrain — once a checkpoint exists)

All load encoder+predictor+pose_decoder from the saved checkpoint, forward-only (`checkpointing.py:62-64`, `orchestrator.py:1006-1063`). Reference = the existing 3.3× number.

### Exp A — Real spatial-token readout (KEYSTONE)
- **Hypothesis:** pose info is present in the spatial grid but destroyed by mean-pooling; passing real `S>1` tokens to the existing decoder recovers most of the gap.
- **Change:** in `orchestrator.py:823-833`, replace `encoder.forward(video).mean → unsqueeze(2)` with `online.forward_tokens(video) → (B,T,S,D)` fed straight to `pose_decoder.predict()` (drop the fake-S). Decode-side only.
- **Cost:** ~1-2h eval-only.
- **Go/No-go:** ratio → **≤1.5×** ⇒ gap was a readout artifact → Exp B/C + the fair comparison (Exp E), **skip Tier 1**. Ratio stays **>2.5×** ⇒ encoder genuinely under-encodes pose → Tier 1.

### Exp B — Attention-weighted / per-location spatial pool
- Learned spatial weighting (or per-location `(K,S,2)` + learned aggregation) reading `forward_tokens()`. Train **only this head** (encoder+predictor frozen). ~2-4h. Run only if Exp A is promising-but-short; informs which readout to pre-register.

### Exp C — Fresh nonlinear probe on pooled+predicted latent
- Bigger MLP probe on the pooled latent (encoder+predictor frozen). ~2-3h. Diagnostic: disambiguates "head too small" (Exp C closes gap) vs "spatial-pooling loss" (only Exp A closes gap).

### Exp E — Fairest comparison: de-contaminate the baseline (run alongside A)
- **Two CRITICAL fixes:** (1) held-out baseline-train split (`seed+20000`), refit `TransformerEigenwormsBaseline` train-only, eval on eval-split (`orchestrator.py:902-906`); (2) pre-registered pose head trained on a separate `seed+30000` split, disjoint from baseline-train + eval. Sanity-check Ridge `alpha=1.0` condition number on the tiny cohort (`residualization.py:41-47`).
- **Cost:** ~2-3h (small refits, no encoder retrain; materialize 2 small splits).
- **Headline Tier-0 verdict = Exp A readout + Exp E fair baseline.** Combined ≤1.5× ⇒ firing was largely methodological, escalate, **no retrain.** Still >2.5× ⇒ Tier 1.
- **Frozen-artifact note:** Exp E's pre-registered head + baseline split are *new frozen artifacts* IF promoted to the official metric → `PRE-REGISTRATION.md` + `CHANGELOG.md` Frozen-artifact-changes entry. Exploratory Tier-0 runs don't require re-lock.

---

## Tier 1 — One targeted retrain (~14h/seed) — only if Tier 0 ratio >2.5×

### Exp F — Pose-coupled auxiliary objective on the online encoder
- **Hypothesis:** pure latent-prediction doesn't preserve fine spatial pose; a light auxiliary pose loss forces the encoder to retain keypoint-relevant structure.
- **F1 (preferred):** add an auxiliary pose-decode loss (λ≈0.1) on `forward_tokens()` during training. Single loss/config change.
- **F2 (fallback):** predict the spatial-token grid instead of the pooled latent (if Exp A/C show pooling is where info dies).
- **Cost:** ~14h/1 seed (gate).
- **Go/No-go:** seed-42 ratio **<1.0×** (beats baseline) or clearly trending down ⇒ Tier 2. Still **>1.5×** ⇒ stop; document encoder limitation, defer to Phase 0 Growth.
- **Frozen-artifact note:** F1/F2 change the **pretraining objective** (pre-registered) → re-lock + CHANGELOG. Also fix the STALE headline.yaml pose/neural-skip comment in the same change.

---

## Tier 2 — Broader sweep — only if Tier 1 single-seed shows promise

### Exp G — Full pre-registered seed sweep
- Winning Tier-1 variant on all three seeds {42, 1337, 8675309}; report **median ratio ± seed-spread CI** (`gates.py:78`). Verify the eigenworm baseline transformer converges on 16-frame clips (raise `n_epochs` if oscillating, `transformer_eigenworms.py:125-166`).
- **Cost:** ~3×14h ≈ 42h.
- **Go/No-go:** median ratio robustly **≤1.0×**, CI excludes 3.3× ⇒ kill_criterion retracted, promote variant + frozen artifacts. Otherwise genuine negative result.

---

## Out of scope (Phase 0 Growth)
- Real WormID HL+SF / Flavell video eval cohort (current proxy pose is synthetic baaiworm) — gated behind materialization.
- Real paired neural gates — the 7 neural gates stay synthetic-proxy; the negative neural/behaviour probes are *expected* under γ-deferral and not falsifiable here. No GPU spend until the real cohort exists.

## Cost ladder (decision-efficient order)
| step | cost | retrain? | gates |
|---|---|---|---|
| Exp A + Exp E | ~3-5h | no | the whole verdict may flip here |
| Exp B / C | ~2-4h each | head-only | only if A is borderline |
| Exp F (1 seed) | ~14h | yes | only if Tier 0 ratio >2.5× |
| Exp G (3 seeds) | ~42h | yes | only if F promises |
