# JEPA-WORM Phase 0 — Pre-Registration Document

**Locked: 2026-05-12 — Sami Rabinovitch**
**Re-locked at schema v2: 2026-05-14 — Sami Rabinovitch (see §"Lock History and Rollback" below)**

**Materialization deviation 2026-05-18 (entire WormID federation + Flavell-2023):** SF (000776, 1.01 TB) deferred to Phase 0 Growth on disk-capacity grounds; HL + NP + SK + EY + KK subsequently deferred to Phase 0 Growth on content-shape grounds (the dandisets contain NeuroPAL volumetric snapshots + calcium-imaging time-series, not the behavioral video the vision-only JEPA encoder requires per FR17). Flavell-2023 additionally deferred (Story 9.1, 2026-05-18) on redundancy grounds: its pinned Zenodo record `10.5281/zenodo.8150515` is CePNEM analysis artefacts + ANTSUN NN weights rather than raw video, and the corpus's actual raw deposit is DANDI dandiset 000776 — the same dandiset already in the WormID federation under the "SF" label. The entire WormID federation **and** Flavell-2023 are therefore materialization-deferred at the Phase 0 layer; Phase 0 active chain shrinks to baaiworm synthetic + WormBehaviorDB + OpenWormMovementDB. See `pre-registration/splits/wormid_train_eval.yaml` §`materialization_deferral` blocks and the 2026-05-18 entries in `CHANGELOG.md` `### Frozen-artifact changes`.

This document is the source of truth for Phase 0's methodological commitments. Every value below was committed *before* the first reportable training run. Modifications after the lock require a `### Frozen-artifact changes` entry in `CHANGELOG.md` (enforced by the pre-commit hook chain).

The thresholds, seeds, and protocol choices below correspond directly to rows in the PRD's measurable-outcomes table (`_bmad-output/planning-artifacts/prd.md`, section "Success Criteria → Measurable Outcomes").

## Lock History and Rollback (2026-05-14)

The initial v1 lock was committed on 2026-05-12 (Story 4.8, git SHA
`cdce1ad`). Post-retrospective research on 2026-05-12 surfaced that the
v1 lock schema (single DOI per dataset) did not fit the actual data
shapes of four of the five Phase 0 datasets:

| Dataset | v1 lock state | Reality |
|---|---|---|
| WormID | single placeholder DOI | Federation of 7 DANDI dandisets (Sprague et al. 2025) |
| Atanas/Flavell-2023 | DANDI-format placeholder | Single Zenodo DOI `10.5281/zenodo.8150515` (not DANDI) |
| WormBehaviorDB | single placeholder DOI | Per-experiment Zenodo records (thousands) |
| OpenWormMovementDB | single placeholder DOI | Per-experiment Zenodo records (thousands) |
| BAAIWorm | single placeholder DOI | GitHub generator (Zhao et al. 2024), no corpus DOI |

The v1 lock therefore recorded **format only**, with placeholder content
in `wormid_train_eval.yaml` and placeholder DOIs in 3 of 5 dataset
entries. **The substance of the pre-registration was not actually
committed at v1.** This document acknowledges that, treats the v1 lock
as a methodological rehearsal, and supersedes it with the v2 lock at
the date in the header.

**Schema v2 extensions** (introduced in commit `9895a0c`,
non-destructive code-only change pre-rollback):

- `dandi_federation` canonicalization for multi-dandiset corpora (WormID)
- `zenodo_subset` canonicalization for pre-committed per-experiment subsets (WormBehaviorDB + OWMD)
- `github_commit_pin` canonicalization for code-only generator repos (BAAIWorm)

**Substance committed at the v2 lock (2026-05-14):**

- **WormID corpus (`dandi_federation`):** the 7 federated dandisets are
  pinned at their canonical DANDI versions (see
  `src/wormjepa/data/sources/wormid.py`). The federation membership and
  per-dandiset version are the substantive commitment.
- **WormID train/eval split (`yaml_sorted_keys_lf`):** new lab-level
  cohort policy (see `pre-registration/splits/wormid_train_eval.yaml`).
  - **Eval cohort:** HL (000714, 9 worms, standard immobilized) and SF
    (000776, 38 worms, the only freely-moving lab). 47 worms across 2
    labs and 2 setup types.
  - **Train cohort:** NP, EY, SK (SK1 + SK2 kept together to avoid
    within-lab leakage), KK. 71 worms across 4 labs and 2 setup types
    (immobilized + semi-restricted).
  - **Zero-overlap invariant:** worm-level by construction — every
    worm in eval dandisets is excluded from train, and vice versa.
    Per-worm enumeration is implementation detail, not a methodological
    commitment.
- **Atanas/Flavell-2023 (`doi_manifest`):** Zenodo DOI
  `10.5281/zenodo.8150515`, registry is Zenodo (not DANDI).
- **BAAIWorm (`github_commit_pin`):** repository
  `github.com/Jessie940611/BAAIWorm` at commit
  `699d3ddb4c2d6a7dc2ac72c3574aee01673fc934` (tag `v1.0.0`); generator
  configuration `Metaworm/args/run_worm_swim_args.txt` at canonical
  SHA-256 `bbc9fa9bdd08982cd6506a7148fbb70e321eb04ad774b026fe2a0d877cc2845b`
  (text_lf canonicalization).

**Forward commitments to be substance-locked at downstream stories:**

The following commitments are recorded methodologically here at v2;
their per-record enumeration is locked at the time of the corresponding
loader-wiring story in Epic 8 (with a `### Frozen-artifact changes`
CHANGELOG entry for each):

- **WormBehaviorDB stratification (locks at Story 8.5):** 100 records
  total, stratified 20 records per strain × 5 strains (N2 wild-type +
  4 representative mutants). The v2 SPEC includes two anchor records
  (one N2, one mutant) as the minimum viable substantive lock; the
  remaining 98 land via a substantive frozen-artifact change in Story
  8.5 under this stratification policy.
- **OpenWormMovementDB stratification (locks at Story 8.6):** 100
  records total, stratified 20 records per strain × 5 strains, with
  the 4 mutant strains **distinct from** the WormBehaviorDB subset so
  the union of the two datasets covers more genotype diversity. Two
  anchor records in v2; the remaining 98 at Story 8.6.

The stratification policy is the binding commitment; any deviation
from "20 × 5 strains per dataset, mutants distinct between datasets" at
Stories 8.5 / 8.6 requires a documented justification in the
corresponding `### Frozen-artifact changes` entry.

## Literature-search timestamp

**2026-05-12** — Initial Phase 0 lit-search. No published JEPA application to *C. elegans* or to any real biological agent at this scale was identified. Nearest analog: Bhaskar et al., eLife 2025 (contrastive pose embeddings; different objective). See `LIT_WATCH.md` for monthly updates and the pre-committed pivot plan.

## Pre-registered thresholds (PRD measurable-outcomes rows 1–8)

All thresholds evaluated using the **lower bound of the worm-level 95% bootstrap CI** (NFR18), not the point estimate. A threshold "clears" only if the CI lower bound exceeds the threshold.

| Row | Metric | Threshold |
|---|---|---|
| 1 | Future-pose at 1 s | JEPA beats Transformer-on-eigenworms (worm-level 95% CI excluding overlap) |
| 2 | Motif ARI vs Flavell labels | ≥ VAME/Tierpsy parity on held-out worms |
| 3 | Neural probe partial R² | ≥ 0.05 over Tierpsy-256 + temporal-derivatives + pose-only TCN baseline |
| 4 | Neural-prior ablation ΔR² | ≥ 0.02 (full vs kinematic-warm-start-only) |
| 5 | BAAIWorm-augmentation ablation ΔR² | Reported; no fixed threshold (the value is the answer) |
| 6 | Session-ID classifier | At chance (95% CI of accuracy contains the chance baseline; NFR19) |
| 7 | Within-state stratified R² | Reported per behavioral state (forward / reversal / Omega-turn / pause / quiescence) |
| 8 | Non-trivial neuron subset R² | Reported on the pre-committed subset in `pre-registration/neuron_subset.yaml` |

**Kill-criterion**: Row 1 must clear by month 6 of model-training experiments. If it does not, Phase B/C work stops and the negative-result paper is written.

## Statistical protocol

- **Bootstrap method**: BCa (bias-corrected and accelerated), with jackknife-over-worms acceleration. Fall back to percentile if the bootstrap distribution is degenerate (per `wormjepa.eval.bootstrap`).
- **Bootstrap sample count**: ≥ 1000 (NFR16); 10000 for headline reporting.
- **Grouping**: worm-level (NFR16 / FR28). Frame-level bootstrap is forbidden architecturally (`wormjepa.eval.bootstrap.WormGrouping` is a mandatory pydantic parameter).
- **CV scheme**: leave-one-worm-out.
- **Multiple-comparison correction**: Holm method (`statsmodels.stats.multitest.multipletests`), α = 0.05, applied whenever multiple metrics are reported on the same data (within-state stratified R²s, per-neuron decoding on the non-trivial subset).

## Sample-size commitments

- **WormID-eval cohort**: see `pre-registration/splits/wormid_train_eval.yaml`. The eval cohort, lab/rig provenance, and dataset version are committed there.
- **Bootstrap samples per CI**: 1000 (default) or 10000 (headline). Both pre-registered.
- **Seeds for the headline sweep**: 42, 1337, 8675309 (three seeds; per NFR9). Reported as point estimate + seed-spread CI.

## Frozen artifacts (`MANIFEST.lock`)

Locked via `wormjepa preregister`. Every file or dataset in the manifest is hashed via the documented canonicalization method; any modification post-lock requires a `### Frozen-artifact changes` CHANGELOG entry and a re-lock with `--force`.

The frozen set includes:

- `pre-registration/PRE-REGISTRATION.md` (this document)
- `pre-registration/configs/headline.yaml` (the locked headline-run config)
- `pre-registration/splits/wormid_train_eval.yaml` (train/eval cohort)
- `pre-registration/probes/neural_decoding.py` (frozen neural-decoding probe code)
- `pre-registration/neuron_subset.yaml` (non-trivial neuron subset)
- `pre-registration/baseline_features/tierpsy_pin.yaml` (Tierpsy version + feature list)
- Dataset entries for the five public datasets (WormID, Atanas/Flavell-2023, WormBehavior DB, Open Worm Movement DB, BAAIWorm)

## Implementation caveats

### Pose-decoder head used as future-pose probe (Phase 0 v0 caveat, 2026-05-19)

The Phase 0 v0 future_pose probe at `_run_future_pose_probe` in `src/wormjepa/eval/orchestrator.py` converts predicted-future JEPA latents to predicted-future pose by feeding them through `PoseDecoderHead` (`src/wormjepa/models/pose_decoder.py`). The `kill_criterion` gate consumes the resulting `future_pose` MetricEntry. `PoseDecoderHead` was originally added as a "dev-loop visualisation" (per its module docstring — the GUI overlays predicted keypoints on frame strips so a developer can watch them converge on the ground truth) and was **not** part of the pre-registered warm-start head set (FR16/FR17) at v2 lock time.

The Phase 0 v0 future_pose probe therefore relies on a non-pre-registered head. The orchestrator's `_run_future_pose_probe` records the same caveat in the `MetricEntry.notes` field at runtime, so any downstream reader of the metrics.json sees the disclosure alongside the value. A Phase 0 Growth follow-up may train a pre-registered pose-decoder head specifically wired for the eval pipeline; any such follow-up will land its own `### Frozen-artifact changes` entry.

## Pivot plan (if first-of-kind status is lost)

If a concurrent Meta-FAIR-scale release (V-JEPA 3, etc.) subsumes the existence claim before Phase 0 publishes, the contribution is re-framed as **the falsifiable-evaluation protocol** (this document + the WormGrouping-enforced bootstrap API + the outcome-aware report templates) rather than the first-of-kind result. The PRD's Innovation Risk Mitigation row covers this.

## Signed

`v1 locked_at`: 2026-05-12 — git SHA `cdce1ad` (superseded; recorded for audit only per §"Lock History and Rollback")
`v2 locked_at`: 2026-05-14
`locked_by`: Sami Rabinovitch
`git_sha_at_lock`: (recorded in `MANIFEST.lock`)
