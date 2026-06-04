# Changelog

All notable changes to this project are documented here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with one JEPA-WORM-specific addition: a `### Frozen-artifact changes` subsection records any modification to a file listed in `pre-registration/MANIFEST.lock`.

The pre-commit hook chain (after Epic 4) refuses commits that modify a MANIFEST.lock-listed file without a corresponding `### Frozen-artifact changes` entry.

## [Unreleased]

### Changed

- **2026-06-04** — `flavell_2023` SPEC annotated with a module-level `MATERIALIZATION_STATUS = "phase-0-growth-deferred"` constant (Story 9.7, γ-pattern deferral close). This is **not** a `DatasetSource` field and **not** part of the `doi_manifest` canonicalization, so the `flavell_2023` lock SHA (`dad038ae…`) and `pre-registration/MANIFEST.lock` are unchanged — hence no `### Frozen-artifact changes` entry. The deferral itself was already enforced (flavell removed from `pre-registration/configs/headline.yaml` `dataset.loaders`) and documented (`PRE-REGISTRATION.md` deviation block) on 2026-05-18; this annotation just makes the per-SPEC status programmatically discoverable. WBDB and OWMD were kept (Stories 9.5 / 9.6), so Flavell is Epic 9's only dataset deferral. Closes Story 9.7.

### Frozen-artifact changes (post-lock modifications; latest 2026-06-03, WBDB + OWMD SPECs extended to 100-record subsets)

- **2026-06-03** — WormBehavior DB SPEC (`src/wormjepa/data/sources/wormbehavior_db.py`, Story 9.5) and Open Worm Movement DB SPEC (`src/wormjepa/data/sources/openworm_movement.py`, Story 9.6) extended from 2 anchor records each to the full 100-record stratified subset (20 records × 5 strains), per the `PRE-REGISTRATION.md` stratification policy (N2 wild-type + 4 mutants, mutants distinct between the two datasets so their union maximizes genotype diversity). This closes the Phase 0 dataset-substance gap: the headline previously trained on baaiworm synthetic clips because WBDB/OWMD carried only 2 anchor records each. **Strain disclosure (the new substantive content):** WBDB picks N2 + `CB1112` (cat-2, dopamine-deficient) / `MT2611` (unc-8, degenerin channel) / `RB557` (asic-2, acid-sensing) / `NL1137` (gpa-5, G-protein chemosensory); OWMD picks N2 + `CB120` (unc-4, cholinergic motor-neuron fate) / `ED3054` (*C. briggsae* wild isolate — the only cross-species strain) / `RB1883` (W03B1.2 KO) / `MT1078` (egl-13, Sox-family TF). The two mutant sets share no strain code. Record-ID disjointness additionally holds: the `open-worm-movement-database` Zenodo community is the shared deposit corpus for both datasets, and every OWMD record ID is disjoint from the WBDB 100 and the two WBDB anchors (`1031550`, `1029149`). Raw-video presence was probed 2026-05-29 on the thin strains: WBDB `MT2611` 20/20 and OWMD `MT1078` 20/20 video-bearing; OWMD `RB1883` had 2 video-less off-food picks (`1033265`, `1033151`) replaced with `1020621` + `1005780` → 20/20. The video-less OWMD mutant anchor `1033265` (12 MB, skeleton-only, silently skipped by the loader) was **dropped**, not replaced — the RB1883 strain block already carries 20 video-bearing records. WBDB retains both its anchors (`1031550`, `1029149`) outside the stratified 100 (`SPEC.records` = 102); OWMD retains only its N2 anchor `1031550` outside the 100 (`SPEC.records` = 101). The `zenodo_subset` canonicalization hashes `(zenodo_record_id, doi)` tuples; the per-record `description` strings (strain/gene/allele) are advisory and not hashed. `name`, `license` (`CC-BY-4.0`), `citation`, and `redistribution_restrictions` unchanged on both SPECs. Strain selection signed off by project lead 2026-06-03 (pre-registration commitment). Enumeration + probe provenance: `_bmad-output/implementation-artifacts/9-5-wbdb-full-100-record-extension-original-story-8-5-ac.md` and `9-6-owmd-...md`.

- **2026-06-03** — `src/wormjepa/data/sources/wormbehavior_db.py`
  - prior SHA: `0608bf36050a45e33f937514e3ff8f3e690c6e554ff39a4e643966a4ab4dfa74`
  - new SHA: `394c971f539b50df6dceb12acc84e6ec84884fa8c4410a816dc66966313ecbe8`
  - justification: `records` grown from 2 anchors to 2 anchors + 100 stratified entries (5 strains × 20). No other SPEC field changed.

- **2026-06-03** — `src/wormjepa/data/sources/openworm_movement.py`
  - prior SHA: `da06ac357dbe8aa86098a6966cb0893c20772ff606519d6fb2173a77021876cc`
  - new SHA: `7b27a5580d37a23c74c526e63a9db396a87143657201def0bc1b379701e0c747`
  - justification: `records` grown from 2 anchors to 1 N2 anchor + 100 stratified entries (5 strains × 20); the video-less mutant anchor `1033265` dropped. No other SPEC field changed.

- **2026-06-03** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the `wormbehavior_db` and `openworm_movement` `zenodo_subset` SHA flips. Only those two entries' SHAs changed; the other 9 artifacts re-canonicalised byte-identical.

### Frozen-artifact changes (post-lock modifications; latest 2026-05-21, two-phase curriculum ported to headline)

- **2026-05-21** — `headline.yaml` gains the two-phase curriculum: `jepa.warm_start_after_step: 4000` and `jepa.head_learning_rate: 0.001`. The E4 anti-collapse recipe (ported 2026-05-20) holds with the warm-start heads *off*, but Experiments 5 and 6 showed the heads co-trained with the encoder collapse it at any head weight — their predict-pose/neural objective fights JEPA's predict-the-target. The curriculum (Experiment 7) trains the encoder pure-JEPA for the first `warm_start_after_step` steps, then freezes the online encoder and switches the heads on: they warm-start by reading a stable representation, and a frozen encoder cannot collapse. `head_learning_rate` puts the heads in their own optimizer param-group at 10x the encoder lr so they fit in the phase-2 budget. For the 5000-step headline, phase 1 = 4000 steps (encoder), phase 2 = 1000 steps (heads). Experiment 8 verified the full recipe (E4 + curriculum + head_lr + the spatial-token pose head) end-to-end: latent `std_mean` held ~0.29 across the freeze, jepa loss 3201 -> 20. Code support (`warm_start_after_step`, `head_learning_rate`, the 2-group optimizer) shipped earlier in the session; defaults are no-curriculum / single-lr so this is the headline opting in. The collapse research is logged in `_bmad-output/implementation-artifacts/phase0-collapse-research-2026-05-20.md`.

- **2026-05-21** — `pre-registration/configs/headline.yaml`
  - prior SHA: `04119526b7fad0e00639eac67f2c5a223e467aca22f1393ed483f10f1b6c112e`
  - new SHA: `9b7c8b6433a3d250e2fc38163a2fad37a0b43ba1376d419c5dcd1e633a1fc0e1`
  - justification: Added `jepa.warm_start_after_step: 4000` and `jepa.head_learning_rate: 0.001`. No other field changed.

- **2026-05-21** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, E4 anti-collapse recipe ported to headline)

- **2026-05-20** — `headline.yaml` adopts the E4 anti-collapse recipe. The systematic latent-collapse study (Experiments E1-E4, logged in `_bmad-output/implementation-artifacts/phase0-collapse-research-2026-05-20.md` with raw per-step trajectories under `collapse-research-logs/`) found the first fully-healthy JEPA run: jepa loss 3194 -> 23, latent `std_mean` -> 0.36, `std_min` -> 0.17 (no dead dimensions), `cov_offdiag` controlled ~96. The recipe ports six `jepa:` settings into the headline config: `predictor_layers: 6` + `predictor_heads: 8` (a real cross-attention predictor — the default 1-layer predictor was pinned at the predict-the-mean floor); `warmup_steps: 50` (linear lr warmup); `standardize_target: false` (standard I-JEPA / V-JEPA do not standardize the target, and standardizing a collapsing EMA target divides by a vanishing std); `variance_reg_weight` 50 -> 25 (VicReg's mu — 50 over-constrained the geometry and stalled the predictor); `covariance_reg_weight: 0.0 -> 0.2` (VicReg's covariance term — decorrelates the latent dims; variance_reg alone let them collapse onto a correlated subspace). Code support for these knobs (`predictor_layers`/`heads`, `warmup_steps`, `standardize_target`, `covariance_reg_weight`) shipped earlier in the session; all default to the pre-recipe behaviour. The headline's pre-registered warm-start heads remain enabled — E1-E4 stripped them to isolate the collapse variable; verifying the recipe holds with the heads re-enabled is the next step.

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `aee7eeb27301d9d87468bf008865d22ef9834b94b109b6cfb9f52e1a23dbbe0d`
  - new SHA: `04119526b7fad0e00639eac67f2c5a223e467aca22f1393ed483f10f1b6c112e`
  - justification: Added `predictor_layers: 6`, `predictor_heads: 8`, `warmup_steps: 50`, `standardize_target: false`, `covariance_reg_weight: 0.2`; changed `variance_reg_weight` 50 -> 25. No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, frozen_target true -> false / EMA target)

- **2026-05-20** — `headline.yaml` `jepa.frozen_target` reverted `true` -> `false`: the target encoder becomes an EMA copy of the online encoder instead of a separate, permanently-frozen V-JEPA 2.1 encoder. The frozen-foreign-target posture was diagnosed to have no stable non-collapsed equilibrium — a predictor can satisfy a fixed foreign target degenerately. Eight successive collapse fixes (batched loop, target-latent standardization, pose/neural gate loosening, realistic worm renderer, `online_init` random->vjepa, predictor positional encoding, cross-attention predictor) got the JEPA loss to drop from a dead-flat ~2000 to ~520 but never stopped the latent std from collapsing. `frozen_target=false` adopts the standard I-JEPA / V-JEPA training posture: target = EMA of the online encoder, predictor forecasts in the shared co-evolving representation space, and the EMA + stop-gradient is the structural anti-collapse mechanism. `vjepa_variant` stays set — the runner's EMA path still initialises the online encoder from V-JEPA 2.1 public weights (`online_init`), so the transfer-learning posture is preserved; only the target's freeze is dropped. Code support: `EMATarget` widened to wrap any encoder (it already duck-typed); `runner._build_state` gains a `frozen_target=false + vjepa_variant set` branch building `online = TrainableVJEPAEncoder`, `target = EMATarget(online)`. This supersedes the headline's `frozen_target: true` commitment; the PRD's frozen-vs-end-to-end note should be read in light of this entry.

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `91f157b6e097b2e6fb13b7b59306c2b5af728c9649bf182184593576dc89015d`
  - new SHA: `aee7eeb27301d9d87468bf008865d22ef9834b94b109b6cfb9f52e1a23dbbe0d`
  - justification: `jepa.frozen_target` `true` -> `false`; the `ema_decay` inline comment updated (it is no longer irrelevant). No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, online_init random -> vjepa)

- **2026-05-20** — `headline.yaml` `jepa.online_init` reverted `random` -> `vjepa`. F1 (2026-05-19) set `online_init: random` to break a supposed `online ≡ target` collapse. Direct measurement on 2026-05-20 showed `random` is itself the root cause of the headline latent collapse: a random-initialised ViT-L, after the V-JEPA wrapper mean-pools its ~576 spatial tokens per frame, produces near-constant latents *by construction* — central-limit washout of unstructured tokens. Measured on six distinct realistic-worm clips: per-dim latent std 0.006 against abs-mean 0.80, i.e. the encoder's output is ~99% identical across distinct inputs *before any training*. The predictor then receives a constant-output encoder and the JEPA loss never leaves the predict-the-mean floor (~2000 across every 2026-05-19/20 headline run). A *trained* V-JEPA encoder's tokens carry input-dependent structure that survives mean-pooling, so `online_init: vjepa` gives the predictor real signal from step 1. The `online ≡ target` collapse F1 feared is not fatal: with `masking_ratio` 0.75 the predictor still has the genuine masked-prediction task — the standard I-JEPA / V-JEPA training posture. This supersedes the F1 entry below.

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `b867dc76e976b35d225174b7ef10cd98609d421ca31da3a703c4a67a31b3eb73`
  - new SHA: `91f157b6e097b2e6fb13b7b59306c2b5af728c9649bf182184593576dc89015d`
  - justification: `jepa.online_init` `random` -> `vjepa`. No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, headline batch_size 32 -> 8)

- **2026-05-20** — `headline.yaml` `jepa.batch_size` 32 -> 8. The run pod's GPU is an RTX 4000 Ada with 20 GB VRAM (the 80 GB A100 of the earlier sweeps was not available). ViT-L/16 at 384px with T=16 frames produces ~4608 tokens; the backprop activation footprint at batch_size=32 is ~58 GB — far over 20 GB. batch_size=8 fits 20 GB, and 8x16=128 cross-sample rows still make the VicReg variance regularizer a meaningful collapse guard. No methodological commitment changes — batch_size is a compute-fit knob, not a pre-registered threshold or protocol choice.

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `6869c2600d2f988e3437ed7ebe255004e22cd03f1286bcb28e7ea37594b29ffa`
  - new SHA: `b867dc76e976b35d225174b7ef10cd98609d421ca31da3a703c4a67a31b3eb73`
  - justification: `jepa.batch_size` 32 -> 8 (VRAM fit). No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, baaiworm image_size 384)

- **2026-05-20** — `headline.yaml` baaiworm loader spec gains `image_size: 384`. baaiworm previously had no `image_size` field, so `composition.py` fell back to its 64x64 default — the ViT-L/16 encoder received baaiworm clips as a 4x4 patch grid while the WormBehaviorDB / OpenWormMovementDB loaders (`image_size: 384`) gave it a 24x24 grid. After the 2026-05-20 gate loosening (`src/wormjepa/training/loop.py` — the JEPA loss now trains on every sample's video, not only pose+neural-bearing baaiworm), this inconsistency means the encoder sees two different input scales across the loader chain. Setting baaiworm to 384 makes the input scale uniform. No methodological commitment changes — `image_size` is an ingestion-pipeline detail, not a pre-registered threshold or protocol choice.

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `d7e93c87cea529504ab0afd579f713a55a1e51b8fbb29d9e588ebd79835cd322`
  - new SHA: `6869c2600d2f988e3437ed7ebe255004e22cd03f1286bcb28e7ea37594b29ffa`
  - justification: Added `image_size: 384` to the baaiworm loader spec. No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-20, batched training loop + headline batch_size)

- **2026-05-20** — Batched training loop (code) + headline config gains `jepa.batch_size: 32` (config). The first headline run (2026-05-19, seed 42) collapsed: `std_mean` 0.069 → 0.0035 over 74 gradient steps. Root cause: `train_jepa` (`src/wormjepa/training/loop.py`) processed one clip per gradient step (`batch_size=1`), so the F3 VicReg variance regularizer measured per-dim std across the 16 frames of a single clip rather than across distinct samples — intra-clip temporal variance is not a collapse guard, so F3 was inert by construction. The loop now accumulates `batch_size` clips of matching pose/neural shape into a `(B, T, ...)` batch per gradient step (mixed-loader corpora defer shape-mismatched samples to the next batch; no sample dropped) and epochs the dataset so `n_steps` runs in full regardless of corpus size. `variance_reg` now measures std across `B*T` distinct rows — the intended VicReg cross-sample collapse guard. Schema gains `batch_size: int = 1` (`ge=1`; default reproduces the pre-batching loop exactly). The headline config opts in at `batch_size=32` (A100-80GB-validated for ViT-L @ 384px, T=16).

- **2026-05-20** — `pre-registration/configs/headline.yaml`
  - prior SHA: `8fa282e435b1d97c2cc56e6343c7bb7fc3aaf43ef5f8cad93012076801bd6647`
  - new SHA: `d7e93c87cea529504ab0afd579f713a55a1e51b8fbb29d9e588ebd79835cd322`
  - justification: Added `jepa.batch_size: 32`. No other field changed.

- **2026-05-20** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-19, post-lock modifications; pose_decoder Phase 0 v0 caveat acknowledgement)

- **2026-05-19** — Phase 0 v0 future_pose probe caveat acknowledged in pre-registration. The orchestrator's `_run_future_pose_probe` (`src/wormjepa/eval/orchestrator.py`) feeds predicted-future JEPA latents through `PoseDecoderHead` (`src/wormjepa/models/pose_decoder.py`) to produce the `future_pose` MetricEntry that the `kill_criterion` gate consumes. `PoseDecoderHead` was originally added as a "dev-loop visualisation" (per its module docstring) and was not part of the pre-registered FR16/FR17 warm-start head set at v2 lock time, so the Phase 0 v0 future_pose probe relies on a non-pre-registered head. This is an administrative disclosure (no methodological commitment changed); the runtime `MetricEntry.notes` field already carries the same caveat for downstream readers of metrics.json. A Phase 0 Growth follow-up may train a pre-registered pose-decoder head specifically wired for the eval pipeline.

- **2026-05-19** — `pre-registration/PRE-REGISTRATION.md`
  - prior SHA: `5ccdd8418d0f683848088b8e460a888467174c5695e0206e2953f5fa1289fcb2`
  - new SHA: `eee88639111cd06209be479ab00c863c77bd5eb32eb0bb4fb41d8a700bc26442`
  - justification: Added an "Implementation caveats" section with a "Pose-decoder head used as future-pose probe (Phase 0 v0 caveat, 2026-05-19)" subsection (1-2 paragraphs) documenting the orchestrator's reliance on `PoseDecoderHead` for the v0 future_pose probe. No other content changed.

- **2026-05-19** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the new PRE-REGISTRATION.md SHA. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-19, F3: variance regularizer added to headline loss)

- **2026-05-19** — F3 VicReg-style variance regularizer wired (code) and headline config gains `jepa.variance_reg_weight: 50.0` (config). Mechanism: `v_loss = mean_d max(0, 1.0 - std(z_d))` over the online encoder's latent flattened over batch+time, added to the JEPA loss total at runtime. Default `variance_reg_weight=0.0` preserves pre-R3-followup loss shape; the R3-followup headline.yaml opts in at 50.0 (chosen per the original VicReg paper's λ=μ=25 invariance/variance balance, scaled up because the JEPA loss is unnormalised by latent dim and dominates raw-value-wise). A local 100-step MPS verification at variance_reg_weight=25.0 did not visibly slow collapse — the right weight is an open hyperparameter that tomorrow's full-scale sweep will inform. A follow-up jepa-loss-rescale-by-D commit may land if 50.0 also proves insufficient. The variance_reg term uses its own knob (`config.variance_reg_weight`) rather than the loss_weights map, keeping the loss_weights namespace reserved for per-head composition.

- **2026-05-19** — `pre-registration/configs/headline.yaml`
  - prior SHA: `82344e0c3e2411e889ba988e6b7b49912f11cf913eebaa783fbdf0a5ae080ebc`
  - new SHA: `8fa282e435b1d97c2cc56e6343c7bb7fc3aaf43ef5f8cad93012076801bd6647`
  - justification: Added `jepa.variance_reg_weight: 50.0`. No other field changed.

- **2026-05-19** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the headline.yaml SHA flip. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-19: R3-followup F1 + baaiworm scale)

- **2026-05-19** — R3-followup headline config update: applies two fixes informed by the 2026-05-18 R3 sweep outcome (see `_bmad-output/implementation-artifacts/phase0-r3-vitl-sweep-2026-05-18.md`):
    1. `jepa.online_init`: `vjepa` → `random`. The R3 ViT-L sweep confirmed the latent-collapse hypothesis was regime-driven, not capacity-driven (`std_min` 0.033 → ~0.0006 over 1000 grad steps across both seeds). Initialising the online encoder from random weights while keeping the target frozen V-JEPA 2.1 breaks the trivial `online ≡ target` solution and restores non-trivial JEPA gradient from step 1.
    2. `dataset.loaders[name=baaiworm]`: `n_worms 50 → 250` (clips_per_worm unchanged at 20). The R3 sweep's `log.jsonl` had exactly 1000 lines despite `n_steps=5000` — the loop's pose+neural gate skipped real-loader samples, baaiworm exhausted at 50×20=1000, and StopIteration ended training. Scaling to 250×20=5000 baaiworm samples means the nominal `n_steps` and the effective gradient-step count converge.
   Code support (commit landing in this same session): schema gains `online_init: Literal['vjepa','random'] = 'vjepa'` field; `build_vjepa_encoder` / `build_trainable_vjepa_encoder` gain a `random_init: bool = False` parameter that skips the V-JEPA 2.1 state-dict load when True; runner branches on `cfg.jepa.online_init`.

- **2026-05-19** — `pre-registration/configs/headline.yaml`
  - prior SHA: `6124bd561faaa3799ba608002d07f04b50cb9393819f289a93897e59c619c93c`
  - new SHA: `82344e0c3e2411e889ba988e6b7b49912f11cf913eebaa783fbdf0a5ae080ebc`
  - justification: Added `online_init: random` to the `jepa:` block; bumped `dataset.loaders[name=baaiworm].n_worms` from 50 to 250. Inline comment blocks in the file document both changes by reference to the R3-outcome doc. Schema_version remains 1.

- **2026-05-19** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the new headline.yaml SHA. No other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-18: R3 headline ViT-L bump)

- **2026-05-18** — R3 headline config upgrade: V-JEPA 2.1 ViT-B → ViT-L + n_steps 1000 → 5000 + learning_rate 5e-5 → 1e-4. Triggered by access to a remote A100-SXM4-80GB box (Runpod, 112 TB scratch, ~51 MB/s DANDI dl) which lifts the disk + bandwidth constraints that bounded the original ViT-B headline. ViT-L has 300M params + 1024-dim embed (vs ViT-B's 86M + 768-dim); the upgrade follows the V-JEPA 2 paper's transfer-learning recipe rung. The combined n_steps + LR bump targets the latent-collapse signal observed in the Story 8.11c ViT-B/1k-step sweep (`std_min ~ 0.0007 at 1000 steps`) — bigger model + more gradient steps + larger updates should give the online encoder room to differentiate from the frozen V-JEPA 2.1 target instead of falling into the trivial-constant attractor. The 8.11c / 8.12b ViT-B sweep results stay on disk as the "ViT-B reference" / "pre-R3 baseline"; the R3 ViT-L sweep is the new substantive headline submission. The dataset chain (post-γ + Story 9.1: baaiworm + WBDB + OWMD) is unchanged.

- **2026-05-18** — `pre-registration/configs/headline.yaml`
  - prior SHA: `857d314bd3001213197130a9ad3513dd63be1c39bf87c29231b9c2e0143fe7fc`
  - new SHA: `6124bd561faaa3799ba608002d07f04b50cb9393819f289a93897e59c619c93c`
  - justification: ViT-B → ViT-L bump per above. Specific field changes:
    `model_name: vjepa2_1_vit_base_384` → `vjepa2_1_vit_large_384`;
    `vjepa_variant: vjepa2_1_vit_base_384` → `vjepa2_1_vit_large_384`;
    `latent_dim: 768` → `1024`; `n_steps: 1000` → `5000`;
    `learning_rate: 0.00005` → `0.0001`;
    `pretrained_checkpoint_sha:` set to `null` (the ViT-L checkpoint SHA is currently unknown — it will be observed on the first remote download via the loader's logger.warning path, then pinned in a follow-up frozen-artifact entry). The schema_version remains 1 (the existing additive fields cover the upgrade with no migration required).

- **2026-05-18** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force`. Only the headline.yaml SHA flipped; no other artifact substance changed.

### Frozen-artifact changes (earlier 2026-05-18: Story 9.1 Flavell-2023 deferral on redundancy grounds)

- **2026-05-18** — Flavell-2023 deferred to Phase 0 Growth on redundancy grounds (Story 9.1). Investigation closed the question of where the Atanas et al. 2023 *Cell* raw paired-data corpus lives: the answer is **DANDI dandiset 000776**, the same dandiset already pinned in the WormID `dandi_federation` SPEC under the "SF" lab label. The current `flavell_2023` SPEC's pinned Zenodo record `10.5281/zenodo.8150515` is the analysis companion (CePNEM intermediates + ANTSUN NN weights, 32 GB JLD2), not the raw deposit. The WormID 000776 entry was already γ-deferred earlier this session, so `flavell_2023` raw data is transitively already on the Phase 0 Growth roadmap. The `flavell_2023` SPEC's URL / DOI / SHA fields are intentionally left untouched in this commit — a future Phase 0 Growth materialization will need to decide whether to re-pin to DANDI 000776 raw, keep the Zenodo record as a separate "analysis companion" lock, or collapse the SPEC entirely. See `_bmad-output/implementation-artifacts/phase0-flavell-redundancy-discovery-2026-05-18.md`.

- **2026-05-18** — `pre-registration/configs/headline.yaml`
  - prior SHA: `6b3e8f60e635b13536556c8cf05fa7d5a1c449ed8b04d5bb465961f02dc35449`
  - new SHA: `857d314bd3001213197130a9ad3513dd63be1c39bf87c29231b9c2e0143fe7fc`
  - justification: Removed the `flavell_2023` loader entry from `dataset.loaders`. Inline comment block in its place documents the redundancy finding and points at the discovery doc + the source SPEC docstring. No other field changed. Phase 0 active dataset chain shrinks to baaiworm + wormbehavior_db + openworm_movement.

- **2026-05-18** — `pre-registration/PRE-REGISTRATION.md`
  - prior SHA: `a05937980c8227e7549d83246866664bd729133b4e0c97c2d9342ac5fe9a7aa8`
  - new SHA: `5ccdd8418d0f683848088b8e460a888467174c5695e0206e2953f5fa1289fcb2`
  - justification: Extended the document-header "Materialization deviation 2026-05-18" line to cover Flavell-2023 alongside the WormID federation, recording the redundancy rationale (Flavell raw = WormID/SF = DANDI 000776).

- **2026-05-18** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` (twice, once per artefact edit). No other artifact substance changed; the `flavell_2023` `doi_manifest` entry is byte-identical to its 2026-05-14 v2 state (URL + DOI + SHA placeholder all preserved; only the conceptual scope is now documented as deferred via PRE-REGISTRATION.md).

### Frozen-artifact changes (earlier 2026-05-18: entire WormID federation deferral)

- **2026-05-18** — Entire WormID DANDI federation deferred to Phase 0 Growth (Option γ). The 2026-05-18 WormID content-shape discovery (see `_bmad-output/implementation-artifacts/phase0-wormid-content-discovery-2026-05-18.md`) established that the locked WormID federation contains NeuroPAL volumetric snapshots (HL + NP, confirmed by direct NWB inspection) + calcium-imaging time-series (SK1/SK2/EY/KK, confirmed by DANDI asset session descriptions like "NeuroPAL+ Calcium Imaging without Stimulation"), not the behavioral video the vision-only JEPA encoder (FR17) requires. SF (000776) is the only dandiset that likely contains behavioral video (freely-moving lab), and SF was already deferred earlier in the session on disk-capacity grounds. With no remaining WormID dandiset that provides behavioral video, the entire federation is materialization-deferred at the Phase 0 layer. The `wormid` `dandi_federation` lock entry, the 7 pinned dandiset versions, and the lab-level cohort split are all unchanged at the pre-reg layer — only the on-disk presence in Phase 0 runs is deferred. Phase 0 active dataset chain becomes baaiworm + Flavell-2023 + WormBehaviorDB + OpenWormMovementDB.

- **2026-05-18** — `pre-registration/configs/headline.yaml`
  - prior SHA: `f567c64a96964e71d78f9a19d37bb62349d7b627335fdffa80465a7f4a46b35f`
  - new SHA: `6b3e8f60e635b13536556c8cf05fa7d5a1c449ed8b04d5bb465961f02dc35449`
  - justification: Removed the `wormid` loader entry from `dataset.loaders`. Inline comment block in its place documents the deferral and points at the discovery doc. No other field changed.

- **2026-05-18** — `pre-registration/splits/wormid_train_eval.yaml`
  - prior SHA: `6fb9d335c21675155dfaece7d26a147514b98c72df85547363bba9ea4bfe3c6a`
  - new SHA: `2625fb29e07211c83e1b85c086cd4b6fa041f6c7b624bea39d5d4c099afb2a8b`
  - justification: Marked every lab in `train_labs` and `eval_labs[HL]` as `phase-0-growth-deferred` (SF was already marked deferred earlier in the session). Added a `materialization_deferral` block on the HL entry mirroring the SF format with the content-shape rationale. Added `phase_0_active_train_*` + `phase_0_active_eval_*_after_hl_deferral` rows to `totals` (all zero — no active WormID worms in Phase 0). The lab-level cohort policy and the zero-overlap invariant are unchanged.

- **2026-05-18** — `pre-registration/PRE-REGISTRATION.md`
  - prior SHA: `5599727525c903684cd0981e6d04548280548ec16bd691d447d363497a03fff4`
  - new SHA: `a05937980c8227e7549d83246866664bd729133b4e0c97c2d9342ac5fe9a7aa8`
  - justification: Rewrote the document-header "Materialization deviation 2026-05-18" line to cover the entire WormID federation rather than just SF, recording the content-shape rationale alongside the prior disk-capacity rationale.

- **2026-05-18** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the three new SHAs above. No other artifact substance changed; the `wormid` `dandi_federation` entry (still pinning all 7 dandisets at their canonical DANDI versions) is byte-identical to its 2026-05-14 v2 state.

### Frozen-artifact changes (earlier 2026-05-18: WormID SF materialization deferral)

- **2026-05-18** — WormID SF (000776) materialization deferred to Phase 0 Growth. The lab-level cohort policy (eval = HL + SF) remains the pre-registered commitment; SF stays in `pre-registration/MANIFEST.lock` (its `dandi_federation` entry is unchanged) and in `pre-registration/splits/wormid_train_eval.yaml`. What changes is which subset of the locked corpus is **materialized on local disk during Phase 0**: SF is 1.01 TB and does not fit in the 245 GB Phase 0 dev-disk capacity, so SF is marked `materialization_status: phase-0-growth-deferred` while train cohort (5 dandisets, 177 GB) plus HL eval cohort (000714, 0.48 GB) are `phase-0-active`. Phase 0 gate evaluation therefore reads only the HL eval signal (9 worms, within-distribution generalization). The SF out-of-distribution signal that motivated SF's eval inclusion is not testable in Phase 0; it lands at the Phase 0 Growth materialization of SF.

- **2026-05-18** — `pre-registration/splits/wormid_train_eval.yaml`
  - prior SHA: `88ea90dc19b58d78dab30f6ddb05c8b9d793fc3afa8507dcc4b9bf2cf4c1941a`
  - new SHA: `6fb9d335c21675155dfaece7d26a147514b98c72df85547363bba9ea4bfe3c6a`
  - justification: Added `materialization_status` per eval lab; added the `materialization_deferral` block on the SF entry recording the 2026-05-18 decision, reason (245 GB local disk vs 1.01 TB SF size), pre-reg impact (within-distribution-only eval signal in Phase 0), and the linked records. Added a `phase_0_active_*` row to the `totals` block (9 eval worms / 1 lab / 80 total active in Phase 0 vs the intended 47 / 2 / 118). The lab-level cohort policy and the zero-overlap invariant are unchanged; only the materialization timeline shifts.

- **2026-05-18** — `pre-registration/PRE-REGISTRATION.md`
  - prior SHA: `d204590a4be2eab792bca723f260770cef28ca8aa89faa2c26c0dd0967f3529d`
  - new SHA: `5599727525c903684cd0981e6d04548280548ec16bd691d447d363497a03fff4`
  - justification: Added a one-line "Materialization deviation 2026-05-18" disclosure under the document header pointing readers at the split file's `materialization_deferral` block and at this CHANGELOG entry. No methodological commitment changed; the document continues to commit to the HL+SF eval cohort at pre-reg level.

- **2026-05-18** — `pre-registration/MANIFEST.lock` re-issued via `wormjepa preregister --force` to pick up the two new SHAs above. No other artifact substance changed; the wormid `dandi_federation` entry (which still pins all 7 dandisets including SF) is byte-identical to its 2026-05-14 v2 state.

### Frozen-artifact changes (Story 8.11b — earlier this session)

- **2026-05-18** — Phase 0 headline configuration substantively populated (Story 8.11b). Story 7.9 had shipped `pre-registration/configs/headline.yaml` as a Kalman-baseline stub (no `jepa:` section) because Epic 5's JEPA training loop was being wired in parallel. Pre-flight for Story 8.11 on 2026-05-18 surfaced that the stub blocked the headline run from launching, and that the frozen V-JEPA 2.1 transfer-learning posture the PRD/architecture had committed to (NFR2-budget-load-bearing per architecture §"Frozen vs end-to-end" and prd.md §"Technical risks") was never wired into the code. Story 8.11a landed the loader (commits `9d12e39` + `71ceb69`); this Story 8.11b commit substantively populates the headline config on top of it. See `_bmad-output/implementation-artifacts/phase0-vjepa21-debt-discovery-2026-05-18.md` for the discovery audit trail.

- **2026-05-18** — `pre-registration/configs/headline.yaml`
  - prior SHA: `34137b36dac11bfd0e7c9ccc1bcbedf6907164b54f359af58ad1b9dd6e238f6a`
  - new SHA: `f567c64a96964e71d78f9a19d37bb62349d7b627335fdffa80465a7f4a46b35f`
  - justification: Replaced the Story 7.9 Kalman-baseline stub with a substantive JEPA-run configuration. New content commits to: frozen V-JEPA 2.1 ViT-B/16 transfer learning (`frozen_target: true`, `vjepa_variant: vjepa2_1_vit_base_384`, `pretrained_checkpoint_sha: 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`); 384×384 input, 768-dim latent, masking ratio 0.75, learning rate 5e-5, 1000 training steps, seed 42 (first of three pre-committed seeds); all four warm-start heads enabled (eigenworm, graph_prior, neural, behavioral); dataset chain led by `baaiworm` (50 worms × 20 clips, 49 keypoints) with the four real loaders (wormid cohort=train, flavell_2023, wormbehavior_db, openworm_movement) chained in for FR8 contract exercise on pre-registered bytes. The schema_version remains `1`; the Story 8.11a additive `JEPASection` fields (`frozen_target`, `vjepa_variant`, `pretrained_checkpoint_sha`) all have defaults so no migration is required.

- **2026-05-18** — `pre-registration/MANIFEST.lock`
  - prior `git_sha_at_lock`: `594b0feb747eb2ac940d002a205b5b38df0d5572` (Story 8.1 v2 re-lock)
  - new `git_sha_at_lock`: (this commit's SHA, recorded post-commit by `git_sha_at_lock` self-update)
  - justification: Re-issued via `wormjepa preregister --force` to pick up the new `pre-registration/configs/headline.yaml` SHA. No other artifact substance changed; the headline.yaml entry SHA flipped from `34137b36...` to `f567c64a...` as documented above. All 11 listed artifacts re-canonicalised; only the headline.yaml entry changed.

### GPU-hour ledger (NFR2: < 200 GPU-hours total for Phase 0)

Cumulative Phase 0 GPU-hours, recorded here so NFR2 is monitored from the first material spend onward. The Story 8.11 sweep is the first material spend; everything before it (8.7 BAAIWorm local synth, 8.9 GPU smoke at 0.55 s, 8.10 baselines on smoke fixtures) was sub-second to seconds and is treated as 0.0 GPU-hr for ledger purposes.

| Event | Wall-clock | Cumulative GPU-hr |
|---|---|---|
| Story 8.11a loader smoke (V-JEPA 2.1 ViT-B download + 1 forward, MPS) | ~6 min | 0.10 |
| Story 8.11b integration smoke (2 grad steps, MPS) | ~45 s | 0.11 |
| Story 8.11c seed=42 headline (1000 steps, MPS, run-id `20260518T074213Z__4f5388e7-dirty__headline`; superseded by 8.12b for lacking checkpoint) | 192.7 s | 0.16 |
| Story 8.11c seed=1337 headline (1000 steps, MPS, run-id `20260518T074534Z__4f5388e7-dirty__headline`; superseded by 8.12b) | 186.7 s | 0.21 |
| Story 8.11c seed=8675309 headline (1000 steps, MPS, run-id `20260518T074846Z__4f5388e7-dirty__headline`; superseded by 8.12b) | 186.8 s | 0.27 |
| Story 8.12b seed=42 headline + checkpoint (run-id `20260518T082244Z__f62120da-dirty__headline`) | 190.1 s | 0.32 |
| Story 8.12b seed=1337 headline + checkpoint (run-id `20260518T082601Z__f62120da-dirty__headline`) | 199.1 s | 0.38 |
| Story 8.12b seed=8675309 headline + checkpoint (run-id `20260518T083300Z__f62120da-dirty__headline`) | 217.8 s | 0.44 |
| Post-8.12b cumulative | — | **0.44 / 200** |

- **2026-05-14** — Phase 0 pre-registration v1 → v2 rollback and re-lock (Story 8.1, Option B). The v1 lock (committed 2026-05-12, git SHA `cdce1ad`) was discovered post-retrospective to encode placeholder content for 4 of 5 datasets and to use a schema that did not fit the actual data shapes. The v1 lock is acknowledged in `pre-registration/PRE-REGISTRATION.md` §"Lock History and Rollback (2026-05-14)" as a format rehearsal. The v2 lock supersedes it with substantive substance. Per-artifact changes:

- **2026-05-14** — `pre-registration/PRE-REGISTRATION.md`
  - prior SHA: `6f69c11652518fa5607538c9c1df884b695a920f04643c7af0fff92cc3d179b5`
  - new SHA: `d204590a4be2eab792bca723f260770cef28ca8aa89faa2c26c0dd0967f3529d`
  - justification: Added §"Lock History and Rollback (2026-05-14)" disclosing the v1 → v2 schema mismatch, listing the substantive substance committed at v2 (WormID federation, Flavell-2023 Zenodo DOI, BAAIWorm v1.0.0 GitHub pin, WormID lab-level cohort split), and recording the WormBehaviorDB / OpenWormMovementDB stratification policy that locks substance per record at Stories 8.5 / 8.6.

- **2026-05-14** — `pre-registration/splits/wormid_train_eval.yaml`
  - prior SHA: `e47eac44df4c61268819dd879e22ef95f815a5f0b174fc867ce928c9cef71368`
  - new SHA: `88ea90dc19b58d78dab30f6ddb05c8b9d793fc3afa8507dcc4b9bf2cf4c1941a`
  - justification: Replaced placeholder per-worm enumeration with schema v2 lab-level cohort policy. Eval cohort: HL (000714, 9 worms, immobilized) + SF (000776, 38 worms, freely moving) = 47 worms. Train cohort: NP, EY, SK (000565+000472 kept together to avoid within-lab leakage), KK = 71 worms across 4 labs. Zero-overlap invariant enforced at the dandiset partition.

- **2026-05-14** — `pre-registration/MANIFEST.lock`
  - prior schema_version: `1`; new schema_version: `2`
  - prior `git_sha_at_lock`: `cdce1ad6ead231d8649652061c2e486363c8f598`
  - new `git_sha_at_lock`: `594b0feb747eb2ac940d002a205b5b38df0d5572`
  - justification: Re-locked under schema v2 with substantive substance for 5 datasets:
    * `wormid` — `dandi_federation` canonicalization over 7 dandisets (000472, 000541, 000565, 000692, 000714, 000715, 000776) at their canonical DANDI versions per Sprague et al. 2025; entry SHA `58643d5cd86a942788de7d04d1ebc56c8d54b953e101c39ac2ed7b73a2812d5f`.
    * `flavell_2023` — `doi_manifest` canonicalization over Zenodo DOI `10.5281/zenodo.8150515` (not DANDI, contrary to the v1 placeholder); entry SHA `dad038aefec789db8e5bf7b978e2ce1eab35b564fe569b6b05c3ac78c25f15f6`.
    * `baaiworm` — `github_commit_pin` canonicalization over repo `github.com/Jessie940611/BAAIWorm`, commit `699d3ddb4c2d6a7dc2ac72c3574aee01673fc934` (tag `v1.0.0`), config `Metaworm/args/run_worm_swim_args.txt` at SHA `bbc9fa9bdd08982cd6506a7148fbb70e321eb04ad774b026fe2a0d877cc2845b`; entry SHA `bf5eeb35c73f3f6842559d82556e1ca2f95ffe3580ede6c5b0f5ed84367325c4`.
    * `wormbehavior_db` — `zenodo_subset` canonicalization over 2 anchor records (N2 Schafer + MT1078 egl-13 mutant). Remaining 98 records to land at Story 8.5 under the stratification policy in PRE-REGISTRATION.md; entry SHA `0608bf36050a45e33f937514e3ff8f3e690c6e554ff39a4e643966a4ab4dfa74`.
    * `openworm_movement` — `zenodo_subset` canonicalization over 2 anchor records (N2 Schafer + RB1883 mutant; distinct mutants from `wormbehavior_db` so the union covers more genotype diversity). Remaining 98 records to land at Story 8.6; entry SHA `da06ac357dbe8aa86098a6966cb0893c20772ff606519d6fb2173a77021876cc`.

### Added

- Phase 0 Story 1.1: Bootstrap Python package and development loop (uv, ruff, pyright, pre-commit).
- Phase 0 Story 1.2: Foundation modules (`paths.py` + `WormJEPAError` hierarchy).
- Phase 0 Story 1.3: Config schema infrastructure (`configs/models.py`, `configs/migrations.py`).
- Phase 0 Story 1.4: Typer CLI skeleton with four placeholder subcommands (`run`, `eval`, `report`, `preregister`).
- Phase 0 Story 1.5: Run-id generator (`cli/run_ids.py`).
- Phase 0 Story 1.6: Results directory contract writer (`reporting/results_writer.py`).
- Phase 0 Story 1.7: Project-discipline file skeletons.
- Phase 0 Story 1.8: Skeleton smoke test green (`tests/smoke/test_run_smoke.py`).
- Phase 0 Story 2.1: Unified iterator contract (`DatasetSample`, `WormID`, `SessionID`, `SourceDataset`).
- Phase 0 Story 2.2: DOI-pinned dataset downloader with retry/backoff, Range-header resume, SHA-256 verification.
- Phase 0 Stories 2.3-2.9 (skeletons): five dataset SPEC modules and loader skeletons under `src/wormjepa/data/`; `pre-registration/splits/wormid_train_eval.yaml` format placeholder. Real values populate in their respective stories.
- Phase 0 Story 3.1: Baseline interface (`src/wormjepa/baselines/base.py`) with `Baseline`, `BaselinePredictions`, `FuturePoseHorizon`.
- Phase 0 Story 3.2: Metrics schema (`src/wormjepa/eval/metrics_schema.py`) with `BootstrapCI`, `MetricEntry`, `SubEntry`, `MetricsOutput`. Canonical JSON I/O.
- Phase 0 Story 3.3: Worm-level bootstrap-CI API (`src/wormjepa/eval/bootstrap.py`). `WormGrouping` pydantic model is mandatory at the type level (defends NFR16 / FR28). Both percentile and BCa methods supported.
- Phase 0 Story 3.4: KalmanBaseline (persistence model for future-pose floor). End-to-end via `wormjepa run --config configs/baselines/kalman.yaml`.
- Phase 0 Story 3.5: TransformerEigenwormsBaseline (kill-criterion comparator). PCA-fit eigenworm basis + tiny autoregressive causal Transformer over eigen coefficients.
- Phase 0 Story 3.6: PoseOnlyTCNBaseline (headline neural-decoding comparator). Causal dilated TCN + linear pose head; exposes per-worm latent matrices.
- Phase 0 Story 3.7: RandomFeaturesBaseline (JEPA-literature sanity check). Frozen random TCN encoder + trainable linear pose head; seeded RNG for deterministic features.
- Phase 0 Story 3.8: Integration test (`tests/integration/test_all_baselines.py`) runs all four baselines back-to-back, verifies contract compliance + `MetricsOutput` validity + worm-level grouping for every produced row.
- Phase 0 Stories 4.1-4.4: pre-registration infrastructure. Canonicalization module (`yaml_sorted_keys_lf`, `python_ast_normalized`, `doi_manifest`, `text_lf`), MANIFEST.lock pydantic schema + I/O, lock-check module, `wormjepa preregister` CLI.
- Phase 0 Story 4.5: pre-commit hooks (`scripts/check_manifest.py`, `scripts/check_changelog_frozen.py`) wired in `.pre-commit-config.yaml`. Initial-lock commits are detected and exempt from the Frozen-artifact-changes requirement.
- Phase 0 Story 4.7: `pre-registration/PRE-REGISTRATION.md` authored with thresholds, statistical protocol (BCa, ≥1000 bootstrap samples, worm-level grouping, Holm correction, α=0.05), sample-size commitments, three pre-committed seeds (42, 1337, 8675309), and the 2026-05-12 literature-search timestamp.
- Phase 0 Story 4.8: **One-way door crossed.** Initial frozen artifacts committed and locked via `wormjepa preregister`: PRE-REGISTRATION.md, headline.yaml, WormID train/eval split, neural_decoding.py probe stub, neuron_subset.yaml, tierpsy_pin.yaml, plus five dataset DOI entries. `pre-registration/MANIFEST.lock` records 11 artifacts.
- Phase 0 Story 4.9: `cli/run.py` copies `pre-registration/MANIFEST.lock` to `results/<run-id>/manifest_at_run.lock` per reportable run (byte-identical). Baseline-driven reportable runs abort with `PreRegistrationViolation` if no lock exists.
- Phase 0 Stories 5.1-5.3: JEPA model components. `WormJEPAEncoder` (timm-wrapped ViT) with type-level FR17 enforcement (`forward(video) -> latent`, no neural input); `EMATarget` (stop-grad target with EMA-updated weights); `JEPAPredictor` (small Transformer with mask token); `random_temporal_mask` for masked spatiotemporal prediction.
- Phase 0 Stories 5.4-5.7: Four warm-start auxiliary heads. `EigenwormHead` (Stephens 2008 regularizer on a kinematic subspace), `GraphPriorHead` (Cook 2019-style edge-prediction side-task), `NeuralAuxiliaryHead` (head-neuron MLP, training-time only), `BehavioralHead` (Flavell classifier). All toggleable per-config.
- Phase 0 Stories 5.8-5.10: Training loop (`training/loop.py`), seed management (`training/seeds.py`), determinism (`training/determinism.py`), checkpoint save/resume (`training/checkpointing.py`).
- Phase 0 Story 5.11: JEPA writes to results contract. `training/runner.py` orchestrates a JEPA run end-to-end; `configs/jepa_smoke.yaml` runs through `wormjepa run` and produces a contract-valid `metrics.json` with `jepa_training_loss` entry (loss sub-rows per active head). Smoke test covers the path.
- Phase 0 Stories 6.1-6.3: Three metric implementations. `eval/residualization.py` (partial-R² via leave-one-worm-out ridge regression on JEPA + kinematic), `eval/future_pose.py` (per-horizon future-pose error with worm-level bootstrap), `eval/motif_ari.py` (k-means + Hungarian-matched ARI vs Flavell labels).
- Phase 0 Stories 6.4-6.7: Four diagnostic gates + neuron subset. `eval/neural_decoding.py` wraps the frozen probe + adds bootstrap CI; `eval/session_classifier.py` runs a leave-one-worm-out logistic regression on session-IDs and reports CI vs chance baseline; `eval/within_state.py` stratifies partial-R² per Flavell behavioral state; `eval/neuron_subsets.py` restricts the probe to the pre-committed non-trivial neurons.
- Phase 0 Stories 6.8-6.11: Ablation runner, multiple-comparison, gate evaluation, STATUS writer. `eval/ablations.py` produces neural-prior and BAAIWorm-augmentation ΔR² rows; `eval/multiple_comparison.py` wraps Holm/BH via statsmodels; `eval/gates.py` evaluates the four primary gates (kill-criterion, headline, neural-prior, session-ID) using CI-lower-bound thresholds (NFR18) and returns an outcome category; `manifest/status_writer.py` renders + writes STATUS.md after every reportable run.
- New runtime deps: `scikit-learn` (ridge, k-means, logistic regression, ARI), `statsmodels` (Holm/BH correction).
- Phase 0 Stories 7.1-7.6: reporting infrastructure. `reporting/compute_provenance.py` (GPU/CUDA/wall-time/peak-mem per run, canonical JSON), three outcome-aware jinja2 templates (`cleared.md.j2`, `kill_criterion_fired.md.j2`, `reframed.md.j2`), `reporting/template_selector.py` (mapping GateStatus->template), `reporting/render.py` (render_report + compare_metrics CI-aware diff).
- Phase 0 Story 7.5: `wormjepa report --run` renders the outcome-aware template; `--compare <path>` runs CI-aware comparison against published metrics; `--gate <name>` prints the verdict of one gate.
- Phase 0 Story 7.7: `REPRODUCE.md` finalized with the canonical reproduction sequence (clone -> uv sync -> preregister --verify -> download -> 3-seed sweep -> report --compare). Acceptance is CI-aware match against `published_results/metrics.json`.
- Phase 0 Story 7.8: CITATIONS.bib referenced in every outcome-aware template; reports point to it.
- Phase 0 Story 7.9: `configs/headline.yaml` committed as working-tree mirror of frozen `pre-registration/configs/headline.yaml`.
- Phase 0 Story 7.10: three-seed sweep documented in REPRODUCE.md (seeds 42, 1337, 8675309 from PRE-REGISTRATION.md); procedural — invoke `wormjepa run --config configs/headline.yaml` per seed.
- Phase 0 Story 7.11: `wormjepa preregister --verify` warns when LIT_WATCH.md's most-recent entry is > 35 days old (NFR15).
- **Epic 7 complete — Phase 0 plan stack complete (E1-E7 all done, modulo E2's loader skeletons awaiting real-data wiring).**
