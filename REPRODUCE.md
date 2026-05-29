# Reproducing JEPA-WORM Phase 0 Results

This document records the canonical reproduction sequence for the Phase 0 headline. Reproduction tolerance is the **worm-level bootstrap CI** of each reported number (NFR7) — not bit-exact. The CI-aware comparison script flags any local result outside the published CI.

## Compute requirements

- For the local smoke and the eval orchestrator: any single CUDA GPU, Apple MPS, or CPU (the smoke runs in seconds; eval reads cached encoder features).
- For the headline run: a single ~20 GB+ CUDA card (the recipe was developed on an H200 144 GB; ViT-L/16 @ 384px backprop activations fit in 20 GB at batch_size 8). MPS / CPU is unusably slow for ViT-L.
- ≤ 64 GB host RAM.
- Internet access for dataset + V-JEPA 2.1 checkpoint download.
- Python 3.11+.

## Bootstrap

```bash
# 1. Clone at the published commit SHA recorded in pre-registration/MANIFEST.lock
git clone https://github.com/samirobbinsss/JEPA-WORM.git
cd JEPA-WORM
git checkout <commit-sha-from-published/MANIFEST.lock>

# 2. Install the pinned environment (uv resolves uv.lock deterministically)
uv sync

# 3. Verify pre-registration commitments
uv run wormjepa preregister --verify
```

## 1. Local smoke (end-to-end wiring check)

```bash
# Train: 2 gradient steps on the synthetic loader, ViT-tiny @ 64px.
uv run wormjepa run --config configs/jepa_smoke.yaml

# Evaluate the resulting run-id.
uv run wormjepa eval --run <run-id>
```

`wormjepa run` writes `results/<run-id>/{checkpoints,log.jsonl,metrics.json,compute.json,manifest_at_run.lock}`. `wormjepa eval` calls the orchestrator in `src/wormjepa/eval/orchestrator.py`, which materialises encoder features through the on-disk cache at `.cache/encoder_cache/` (built once per encoder × dataset, reused across probes), runs the pre-registered probes, and writes `results/<run-id>/metrics_eval.json` + an additive update to `STATUS.md`. Pass `--run` multiple times for cross-seed sweep mode.

`configs/jepa_smoke_realdata.yaml` is the variant that exercises the chained real-data + synthetic loader composition against the committed `tests/fixtures/` bytes; use it for a wiring check that touches the real-data path without leaving the laptop.

## 2. The collapse-research experiments (E1–E8)

The 2026-05-20/21 latent-collapse study is captured as a sequence of per-experiment configs, `configs/jepa_debug{,_e2..e8}.yaml`. Each is a single-seed sandbox run that isolates one variable in the anti-collapse recipe; the working recipe was identified at E4 and the two-phase head curriculum at E7/E8. Re-run any experiment with:

```bash
uv run wormjepa run --config configs/jepa_debug_e4.yaml
uv run wormjepa eval --run <run-id>
```

The healthy run signature is a falling `jepa` loss with a per-dim latent std that holds (no collapse to a constant latent) — visible in `results/<run-id>/log.jsonl` under `extra.latent.*`.

## 3. The headline run (remote 3-seed sweep)

`scripts/dev/remote_sweep.sh` provisions a Runpod-style ephemeral CUDA pod, rsyncs the repo, installs the env, fetches WBDB + OWMD anchors, and runs the seed sweep:

```bash
REMOTE_HOST=root@<ip> \
REMOTE_PORT=<port> \
SSH_KEY=~/.ssh/id_ed25519 \
  bash scripts/dev/remote_sweep.sh
```

Step [4/8] is the GPU preflight gate (`scripts/dev/preflight_gpu.py`) — a mis-provisioned pod (driver too old for the installed torch build, no GPU attached, CPU-only image) aborts here before any compute is spent. Defaults reproduce the 3-seed headline sweep (`configs/headline.yaml`, seeds `42 1337 8675309`); override with:

- `SWEEP_CONFIG=configs/jepa_debug.yaml` to re-run a research experiment.
- `SWEEP_SEEDS="42"` for a single-seed sandbox run.
- `SKIP_ZENODO=1` for baaiworm-only research configs that never touch the WBDB/OWMD loaders.
- `WORMJEPA_RESULTS_ON_WORKSPACE=1` to keep results on the network volume (survives pod close at a FUSE checkpoint-save penalty).

Results rsync back to `results/<run-id>/` on the local repo by default; evaluate each with `wormjepa eval --run <run-id>`. The CI-aware comparison against the published table:

```bash
uv run wormjepa report --run <run-id> --compare published_results/metrics.json
```

## 4. The current recipe at a glance

The canonical record is `configs/headline.yaml` (its inline comments document every knob and the rationale); the short version: ViT-L/16 @ 384px online encoder initialised from V-JEPA 2.1 weights with an EMA target (not a frozen V-JEPA target); a 6-layer / 8-head cross-attention predictor with positional encoding; 50-step linear lr warmup; `standardize_target=false`; VicReg's variance + covariance terms at `variance_reg_weight=25` and `covariance_reg_weight=0.2`; a two-phase curriculum at `warm_start_after_step=4000` (pure-JEPA encoder for steps 0–4000, then encoder frozen and the four warm-start heads switch on for the final 1000); and `head_learning_rate=1e-3` so the heads get their own optimizer param-group at 10x the encoder lr. Batched loop at `batch_size=8`. The recipe held std ~0.29 across the freeze with jepa loss 3201 -> 20 on synthetic data (E8 re-run, 2026-05-21).

## Acceptance

Reproduction is considered successful when every published row in `published_results/metrics.json` passes the CI-aware comparison:

```
$ uv run wormjepa report --run <run-id> --compare published_results/metrics.json
... within_ci ...
```

Rows flagged `outside_ci` indicate a genuine reproduction failure; check the CHANGELOG history for `### Frozen-artifact changes` entries that might explain the divergence.

## Reproducing a negative result

If the published outcome is `kill_criterion_fired` or `reframed`, the expected reproduction is the same numerical table — the only difference is that the published `report.md` reflects the alternate template (`kill_criterion_fired.md.j2` or `reframed.md.j2`). The CI-aware comparison makes no distinction.

## Known limitations / open work

- **Headline corpus is not yet real.** The current `headline.yaml` chain materialises baaiworm (synthetic-renderer) clips; WBDB and OWMD are present in the chain but only the two pre-registered anchor records each, not the full 100-record stratified sets. Substantively reproducing the headline on real video is gated on Story 9-5 (WBDB full extension) and Story 9-6 (OWMD full extension) under Epic 9 "Dataset Pipeline Substance".
- **Warm-start heads under-converge.** Under the curriculum + `head_learning_rate=1e-3`, the `neural` head only partially fits (2.2M -> ~460k) and `eigenworm` + `pose_decoder` bounce without a clean downtrend over the 1000 phase-2 steps. The collapse — the actual headline blocker — is solved; the head fits are open tuning.
- **`pose_decoder` is dev-loop-only.** The cross-attention `PoseDecoderHead` over `forward_tokens(video) -> (B,T,S,D)` is a visualisation head; the pre-registered pose evaluation is the orchestrator's `future_pose` probe (independent).

## How to inspect Phase 0 verdicts (Streamlit dashboard)

```bash
uv run streamlit run scripts/dev/gui_verdict_dashboard.py
```

Opens a read-only browser dashboard for the Phase 0 gate-verdict review
workflow. The page surfaces the recomputed outcome category (cleared /
kill_criterion_fired / reframed) for the selected run, the per-gate
verdict table parsed from `STATUS.md`, a selectable inspector over
each `MetricEntry` (point estimate + 95% CI bounds + producer + the
proxy-cohort / Holm-caveat notes from the orchestrator), an optional
cross-seed sweep panel for comparing 2+ runs, and a pre-rendered Holm
correction table extracted from the run's gate notes.

The dashboard reads `results/<run-id>/metrics_eval.json` files (produced
by `wormjepa eval`) plus the repo's `STATUS.md`. It never writes to
disk and never re-runs probes; pair it with `wormjepa eval` upstream
to materialise the inputs.
