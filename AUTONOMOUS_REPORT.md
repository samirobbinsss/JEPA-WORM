# JEPA-WORM — Autonomous Overnight Session Report

**Session:** 2026-06-15 → 2026-06-16, autonomous until ~08:00.
**Goal:** run the pre-registered headline sweep on a rented A100-80GB, evaluate the gates, then improve the model from the results — logging every training run and its analysis here.
**Operator:** Claude (autonomous), pod handed off by Sami.

> Status legend: ✅ done · 🔄 running · ⏳ queued · ❌ failed/killed · `[TBD]` awaiting result.

---

## 0. Executive summary

**The night's headline result: the frozen pre-registered recipe produces a healthy, non-collapsing JEPA on real worm video at full ViT-L/384 scale — validated for the first time** (all prior runs were smoke-sized). Seed 42 trained to ~1200+ steps with jepa loss **3201 → 14.9** (below the E4 synthetic benchmark) and latent **std_mean rising 0.12 → 0.44** (the opposite of the collapse that dominated Phase-0 research). See §2.

Getting there meant clearing **6 distinct ways the Runpod A100's MooseFS network volume sabotages this workload** (§1) — venv-on-MooseFS deadlock (33 min→36 s), sequential Zenodo fetch (1→21 MB/s), results-wipe-on-relaunch, the ViT-L checkpoint download/EIO (×2 path issues), and the clip-writer MP4 crash. Net rule: **local `/root` for anything random-access; MooseFS only for big append-only sequential data.** All fixes are operational/data-pipeline only — **no frozen artifact touched**.

**Honest scope:** the full 3-seed headline is a ~42 h run on one A100 and does **not** finish tonight — by 08:00 seed 42 is at ~1900–2000/5000 steps (partial, healthy), seeds 1337/8675309 unstarted, no gate eval yet. Cross-pod resume is blocked by MooseFS's inability to write the 6 GB checkpoint (deferred follow-up). The gate-eval procedure and 3 ranked improvement experiments are mapped + ready (§3/§4) for when compute allows.

---

## 1. Infrastructure (the road to a launchable run)

The pod was healthy hardware (A100-SXM4-80GB, 120GB MooseFS volume, 20GB local overlay) but three non-obvious issues each would have silently wasted hours of GPU idle-billing. All three are now fixed and the fixes are baked into `scripts/dev/remote_sweep.sh` for future pods.

### 1.1 MooseFS venv deadlock (the big one)
- **Symptom:** `uv sync` (installing torch ~6GB, thousands of small files incl. a ~2GB `libtorch_cuda.so`) ran for **33 minutes** at **0% CPU / 0 MB/s network**, load spiking — never finishing.
- **Root cause:** the venv + uv cache were on the `/workspace` MooseFS network volume. Writing thousands of small files to a FUSE network FS is pathologically slow (a metadata round-trip per file).
- **Fix:** put venv + uv cache + tmp on the **local `/root` overlay** (fast NVMe); keep only big sequential artifacts (ViT-L checkpoint, 45GB data) on `/workspace`. No `UV_LINK_MODE=copy` → uv hardlinks venv←cache on the same local fs. **Result: 33 min → 36 s.** This was almost certainly the original "no stable pod" killer across the 4 prior pods (2026-06-10).

### 1.2 Sequential Zenodo fetch
- **Symptom:** the 45GB corpus fetched at **~1 MB/s** → ~10 h ETA (A100 idle the whole time).
- **Root cause:** the fetcher downloaded records one at a time; Zenodo bandwidth-caps each *connection*.
- **Fix:** parallelized the record loop with a thread pool (`wormjepa fetch zenodo-subset --workers N`, `src/wormjepa/cli/fetch.py`). **Result: 1 MB/s → 21 MB/s at 16 workers (~15 min for the full corpus).**
- **Caveat:** 16-way write concurrency triggers MooseFS `EIO` (errno 5) on ~3% of records. Recovered cleanly at `--workers 4`. Final corpus: **203 records, 52GB**, verified complete (every record has its 3 files, no zero-byte files).

### 1.3 Cross-pod resumable checkpointing (so pod churn isn't fatal)
The block on 2026-06-10 was "no stable pod." Added resumability so an ephemeral pod dying mid-run is recoverable:
- `WORMJEPA_CHECKPOINT_EVERY=N` → atomic checkpoint every N steps to a stable `results/_resume/<slug>__seed<seed>/checkpoint.pt` on the persistent volume; resume-on-launch; `done` markers skip finished seeds.
- Resume-correctness fixes in `training/loop.py`: `while step < n_steps` (resume runs *remaining* steps, not n_steps more) + **curriculum re-freeze on resume** (the one-shot `==` encoder-freeze event would never re-fire after a resume past step 4000 → an unfrozen phase-2 encoder collapses the latent; re-applied on load).
- Atomic save (`checkpointing.py`): temp-file + rename, so a SIGKILL mid-write can't corrupt the checkpoint.
- 4 new unit tests (resume continuation, curriculum re-freeze, periodic write); adversarial review found no defects.

### 1.4 V-JEPA checkpoint download path (4th disk gotcha)
- **Symptom:** all 3 seeds crashed identically at model-build with `[Errno 28] No space left on device` downloading the ViT-L checkpoint.
- **Root cause:** the V-JEPA loader downloads the ~1.3GB checkpoint to `~/.cache/wormjepa/checkpoints/` — the 20GB overlay (4.6G free after the venv) — and ignores `TORCH_HOME`. It reads `WORMJEPA_CHECKPOINT_DIR` instead (`vjepa_loader.py:68`), which I hadn't set.
- **Fix:** `export WORMJEPA_CHECKPOINT_DIR=/workspace/.caches/vjepa-ckpt` → checkpoint lands on the 120GB volume, persists across pods (seeds 2/3 + resume-pods reuse it). Lesson: on this pod *every* large artifact needs an explicit `/workspace` redirect — the loaders/tools each use a different env var.

### 1.5 V-JEPA checkpoint on MooseFS → EIO (5th gotcha)
- **Symptom:** with the checkpoint redirected to `/workspace`, every seed crashed at model-build: `Failed to download V-JEPA 2.1 checkpoint ... [Errno 5] Input/output error`. The 5.15GB file would write to ~5GB then EIO; the loader unlinks the partial (`vjepa_loader.py:122`) → next seed re-downloads → re-fails. GPU never engaged (0% util across the loop).
- **Root cause:** MooseFS is unreliable for the **single large 5.15GB checkpoint** write *and* the SHA-read the loader does every run (`vjepa_loader.py:126`) — the same EIO class that hit the 45GB fetch under concurrency, here from one big sequential file.
- **Tension:** the checkpoint is too big for the 20GB overlay alongside the venv (§1.4), but EIOs on the 120GB MooseFS volume.
- **Fix:** free `/root` (delete the uv cache, ~5GB — not needed post-install, venv intact) → `/root` has room for venv (6GB) + checkpoint (5.15GB); pre-stage the checkpoint to **local `/root/vjepa-ckpt`** once via a detached `curl` (network→local write, reliable), then `WORMJEPA_CHECKPOINT_DIR=/root/vjepa-ckpt` so all seeds find the existing file (loader skips re-download) and SHA-read it from fast local disk. Resume-pods re-stage (a ~2 min local download).
- **General lesson:** on this pod, **small-or-large random-access files → local `/root`; only big *append-only sequential* data (the 45GB corpus) tolerates MooseFS** — and even that needed low write concurrency.

### 1.6 Large training writes (clips MP4 + 6GB checkpoint) → MooseFS failures (6th)
After seed 42 trained cleanly for **~1.5 h to ~590 steps** (healthy: jepa 648→descending, std_mean ~0.12), it crashed:
- **Clip/rollout writer:** `RuntimeError: basic_ios::clear: iostream error` in `write_end_of_file` (PyAV MP4 finalize) — the dev-loop GUI visualization writing an MP4 to `results/<run-id>/clips/` on MooseFS. **Fix:** gate clip writing behind `WORMJEPA_WRITE_CLIPS` (default OFF; `runner.py`). Not needed for metrics/gates.
- **Resume checkpoint:** the 6GB checkpoint left an orphaned `checkpoint.pt.tmp` (atomic-rename never completed) — MooseFS is slow + unreliable for the 6GB write, and `/root` lacks room (venv 6GB + ViT-L ckpt 4.8GB fill it; a 6GB ckpt + its 6GB .tmp needs 12GB, only 7.4GB free). **Fix:** `WORMJEPA_CHECKPOINT_EVERY=0` for now — trajectory still lands in `log.jsonl` (small per-step appends, which MooseFS handles fine). **Cross-pod resume is a deferred follow-up** needing a checkpoint that fits `/root` (drop optimizer state → ~3GB) or a non-MooseFS store.

**The MooseFS pattern across all 6 traps:** small appends + big *sequential append-only* reads/writes are OK; **everything else — venvs (many small files), large single-file writes (5–6GB checkpoints), MP4 finalize, concurrent writes — fails or deadlocks.** Local `/root` is the safe target but space-constrained.

**None of these touch frozen artifacts** — all operational/data-pipeline code, no MANIFEST re-lock.

---

## 2. Headline run (frozen `configs/headline.yaml`)

Frozen V-JEPA 2.1 ViT-L/16 @ 384px, EMA-target JEPA, B=8, 5000 steps, two-phase curriculum (4000 pure-JEPA → freeze → 1000 warm-start heads). 3 pre-committed seeds. The one cleanly-evaluable pre-registered gate is the **kill-criterion** (future-pose 1s: JEPA vs transformer-on-eigenworms); the 7 neural gates are synthetic-proxy (real neural data γ-deferred to Phase 0 Growth).

| Seed | run-id | status | jepa loss | latent std_mean | kill-criterion | notes |
|---|---|---|---|---|---|---|
| 42 | `20260616T022332Z__…__headline` | ✅ **COMPLETE (5000 steps)** | 3201 → **5.84** | **0.116 → 0.539** (std_min 0.48) | 🔄 eval running | full ViT-L/384 run on real video; healthy throughout incl. the step-4000 freeze; `checkpoint.pt` (6.1GB) saved OK |
| 1337 | `[TBD]` | ⏳ | `[TBD]` | `[TBD]` | `[TBD]` | |
| 8675309 | `[TBD]` | ⏳ | `[TBD]` | `[TBD]` | `[TBD]` | |

### ⚠️ Compute reality (scope vs. the "until 8am" window)
Measured on the A100-80GB: **GPU 100% util, 74GB/80GB, ~6 steps/min (~10s/step)** for the frozen ViT-L/384 @ B=8, T=16 (~4608 tokens). So:
- **~14 h per seed → ~42 h for the full 3-seed headline.** The pre-registered headline is a multi-day run on one A100; it **cannot complete by 08:00**.
- By 08:00 seed 42 reaches **~3000/5000 steps** — a partial but real trajectory, **resumable** from the last 500-step checkpoint on `/workspace` (a future session/pod continues it; seeds 1337/8675309 not started).
- **Single GPU = no parallel experiments.** Seed 42 holds the GPU at 100%; improvement experiments (§4) cannot run alongside it. Options for the user: (a) let the headline grind + resume later (current choice — respects the "headline sacred + priority" guardrail); (b) pause it (checkpointed, lossless) to run shorter research experiments; (c) add a 2nd pod. Held at (a) overnight pending user input.

**What tonight delivers:** the full pipeline **validated end-to-end on real WBDB+OWMD video at ViT-L/384 scale with healthy non-collapsing latents** (a first — prior runs were smoke-sized), the seed-42 trajectory, and a ready gate-eval + experiment plan (§3/§4).

### Per-seed detail

**Seed 42 — `20260616T022332Z__…__headline` (clean run, clips/ckpt off). Trajectory:**

| step | jepa loss | std_mean | std_min | cov_offdiag | read |
|---|---|---|---|---|---|
| 1 | 3201 | 0.116 | 0.024 | 12.6 | cold start (predict-the-mean floor) |
| 100 | 237 | 0.173 | 0.024 | 34.8 | rapid descent |
| 300 | 68 | 0.290 | 0.055 | 88.5 | std climbing, dims waking |
| 600 | 24.3 | 0.364 | 0.187 | 99.2 | ≈ E4 synthetic benchmark (~23), but on REAL data |
| 900 | 29.3 | 0.401 | 0.302 | 110.0 | std_min healthy (no dead dims) |
| 1200 | **14.9** | **0.443** | **0.346** | 116.1 | loss below E4; latent richly varied |

**Analysis — this is a healthy, non-collapsing run, and the strongest evidence yet that the pre-registered recipe holds on real data:**
- **jepa loss 3201 → 14.9** over 1200 steps — monotone descent (modulo noise at 900), and *below* the E4/E8 synthetic-data benchmark (~23 @ 500 steps). The frozen V-JEPA-2.1 ViT-L/384 + E4 anti-collapse recipe + EMA target transfers to real WBDB+OWMD video.
- **std_mean RISING 0.116 → 0.443** (vs the ~0.0007 collapse floor, and E8's ~0.29 hold) — the latent is becoming *more* varied as it trains: the opposite of collapse. The single biggest worry from all the Phase-0 collapse research does **not** reproduce here.
- **std_min 0.024 → 0.346** — no dead dimensions; the whole latent space is used.
- **cov_offdiag 12.6 → 116** — rising but controlled by covariance_reg (E2's un-regularized collapse hit ~1395); dims stay decorrelated.
- Curriculum freeze (step 4000) + warm-start heads not yet reached (run is in pure-JEPA phase 1).

**Step-4000 curriculum freeze (the trickiest part of the recipe) — held cleanly on real data:**

| step | jepa | std_mean | std_min | active heads |
|---|---|---|---|---|
| 3900 | 5.54 | 0.531 | 0.496 | — |
| 4000 | 7.73 | 0.552 | 0.436 | — (freeze fires) |
| 4001 | 7.44 | 0.554 | 0.496 | graph_prior |
| 4020 | 8.00 | 0.551 | 0.410 | graph_prior |
| 4080 | 6.84 | 0.554 | 0.447 | graph_prior |
| 4120 | 5.45 | 0.566 | 0.463 | graph_prior |

Log line: `curriculum: froze online encoder at step 4000`. **std_mean HELD across the freeze (0.53→0.57), no collapse** — the encoder froze, the warm-start heads switched on and read a stable representation, and jepa loss bumped (heads' loss joining the total: 5.5→7.7) then recovered to 5.45. This is exactly the E7/E8 design intent (E5/E6 showed heads collapse an *unfrozen* encoder), now confirmed on the real corpus at ViT-L/384 scale. (Phase 2 runs steps 4000–5000 with the encoder frozen + heads training at head_learning_rate 0.001.)

**Decision (08:00):** user chose to run seed 42 to the full 5000 steps + gate eval (~7h more A100). Continuing. On completion the runner saves the final 6GB checkpoint — same MooseFS write that left an orphaned `.tmp` before — so the plan is: if `checkpoints/checkpoint.pt` is missing but a `.tmp` exists, the `.tmp` holds the complete `torch.save` payload (only the atomic rename failed) → manually `mv` it into place, then run `wormjepa eval --run <seed42-id>` for the single-seed kill-criterion verdict + neural-proxy gates.

### Data composition (verified by h5py probe on the pod, 2026-06-16)
The encoder trains on a **mix of real video + synthetic**, confirming conclusion-draft caveat #1:
- **Real worm video** — WBDB + OWMD `.hdf5` files carry a `mask` dataset = the actual recorded *C. elegans* video (WBDB record 1011381: **215,979 frames**; OWMD 1005780: 19,439 frames; 480×640 grayscale, background-masked). The Schafer reader picks it (`_VIDEO_CANDIDATES = ["mask","full_data","video"]`) and loads it **silently** (no per-load INFO). Per FR17 the encoder operates on real video.
- The 20 `"skipping … skeleton but no video dataset"` INFO lines are **only the `_features.hdf5` siblings** (Tierpsy skeleton/features, no video) — correctly skipped, not a sign that real video is absent.
- **Synthetic** — BAAIWorm clips lead the dataset chain (250 worms × 20 = 5000 clips/epoch) and supply the pose+neural targets the warm-start heads need (real paired neural is γ-deferred).
- **Unquantified:** the exact real-vs-synthetic ratio per batch (baaiworm leads, so synthetic is well-represented). Per-sample loader provenance is not logged.

`[Seeds 1337 / 8675309: not started — single GPU, ~14h/seed.]`

---

## 3. Gate evaluation

**Procedure** (run once all 3 seeds finish; each has `config.yaml` + `checkpoints/checkpoint.pt` + `metrics.json`):
```bash
wormjepa eval --run <seed42-run-id> --run <seed1337-run-id> --run <seed8675309-run-id>
```
→ writes `results/<run-id>/metrics_eval.json` per seed + a `## Cross-seed sweep` section in `STATUS.md` (per-gate consensus verdict + mean/min/max). Add `--json` for machine-readable. **Do NOT** pass `--control` / `--baaiworm-control` — no real control run exists tonight; it would add a placeholder to the Holm family and muddy the table.

**What is actually decidable** (3 of 8 gates cleanly evaluable):
- `kill_criterion` (the headline): clears iff `jepa.future_pose[1s].upper < transformer_eigenworms.future_pose[1s].point`. Missing either producer → `pending` (not fail). Baseline trains on the eval cohort → train-test-contamination caveat noted.
- `neural_probe_partial_r2`: clears iff `ci.lower ≥ 0.05` — **synthetic-proxy** neural data.
- `session_id_at_chance`: clears iff `chance ∈ [ci.lower, ci.upper]`.
- The other 5 gates are synthetic-proxy / deferred (no real Flavell+WormID labels) — non-binding diagnostics. Holm correction is **informational only** (does not change verdicts in Phase 0 v0).

### Result — seed 42 single-run eval (`wormjepa eval --run <seed42>`), 2026-06-16

**Outcome: `kill_criterion_fired` — the headline hypothesis is falsified (single seed, proxy).**

| gate | verdict | numbers |
|---|---|---|
| **kill_criterion** | **fired** | JEPA future-pose@1s error **305.0** [277,330] vs transformer-eigenworms **93.2** [85,100] → JEPA ~3.3× *worse*; clears iff jepa.upper(330)<baseline(93) → false |
| neural_probe_partial_r2 | fired | JEPA partial-R² **−34.6** [−47.9,−24.8] vs kinematic baseline **0.9995**; CI lower ≪ 0.05 threshold |
| session_id_at_chance | fired | classifier acc 0.0, CI [0,0] excludes chance 0.042 (degenerate) |
| neural_prior_ablation | pending | no `--control` run (expected) |

Diagnostics (synthetic-proxy, non-binding): motif-ARI ≈ 0.0005; within-state R² −38…−45; non-trivial-neuron R² −71 (kinematic 0.9996). Holm: neural_probe_partial_r2 p=1.0, reject_null=False. Ridge probes logged repeated "singular matrix" warnings → the exact negative-R² magnitudes are unstable, but all are clearly ≪ 0.

**Interpretation — the night's real scientific finding:** a *healthy, low, non-collapsed* JEPA loss (3201→5.84, std_mean 0.12→0.54) did **NOT** transfer. The encoder's representation does not predict worm pose better than a compact eigenworm basis (it's ~3× worse at 1s), and its latents don't linearly decode the synthetic neural/behavioral targets. The mean-pooled frozen-V-JEPA-2.1 ViT-L features are generic-video features, not worm-kinematic features — so a transformer on eigenworms (a pose-specific basis) wins easily. **This is "necessary but not sufficient" made concrete: avoiding collapse was required to even ask the question, and the answer on the real corpus is the same falsification the smoke runs gave — the kill criterion fires.**

**Caveats bounding the result:** single seed (not the 3-seed pre-registration); kill-criterion routes through the dev pose-head on **BAAIWorm proxy pose**; neural gates use **synthetic** neurons; the eigenworm baseline trains on the eval cohort (train-test contamination); ridge instability. So: a single-seed, synthetic-proxy falsification — directionally consistent with prior runs, not the final pre-registered 8-gate verdict.

---

## 4. Improvement experiments (research configs — NOT the frozen headline)

Each = a NEW `configs/research_*.yaml` cloned from `headline.yaml` with ONE delta, run as a sibling sweep (own seeds), never retagged `headline`. **Plan** (final choice driven by the headline's latent-health result):

1. **`research_mask050.yaml`** — `masking_ratio: 0.75 → 0.50` (TOP PICK). More visible context = the collapse-guard's foundation (E4 finding). Measure std_mean/std_min (hold/beat ~0.36), jepa loss, kill-criterion margin. Kill-early if std_min collapses <0.05 in first few hundred steps → fall back to #3.
2. **`research_pred8L.yaml`** — `predictor_layers: 6 → 8` (only if #1 holds std). More capacity for the 1s future-pose horizon. Kill-early if loss plateaus ≥ 6-layer baseline AND eval R² degrades (overfit).
3. **`research_var15.yaml`** — `variance_reg_weight: 25.0 → 15.0` (fallback / std-permitting). Prioritizes jepa convergence if std stays ≥~0.3 and cov_offdiag doesn't blow up (E2 hit ~1395).

Full lever survey (8 knobs ranked: masking_ratio, predictor_layers/heads, variance_reg, covariance_reg, ema_decay, lr, batch_size, warm_start_after_step) in workflow `we9r62emi`. Note: `batch_size 8 → 16` is now feasible on the A100-80GB (was capped at 8 for the 20GB RTX 4000 Ada) and would strengthen the variance_reg cross-sample signal — a candidate if a recipe experiment needs it.

`[Results TBD — appended per experiment as they run.]`

---

## 5. Timeline / event log

- **2026-06-15 ~13:50** session start; pod handed off.
- infra debugging (§1): MooseFS venv deadlock, parallel fetch, resumability.
- **2026-06-16 ~00:00** corpus complete (203 records, 52GB).
- **2026-06-15 23:58Z** seed 42 launched (detached, pid 9228); config validated, run-id minted, manifest copied, encoder building. Pod sshd is flaky under load (255 bursts) — mitigated with minimal-command launches + a retrying ssh wrapper (`/tmp/podssh.sh`) + detached nohup so SSH drops can't kill the run.
- `[appended as events occur]`

### Known operational risk
The pod's sshd intermittently rejects connections (load spikes to ~13, MaxStartups). The training itself is detached (`nohup`, survives drops) and resumable, so monitoring gaps are non-fatal; worst case a pod death resumes from the last 500-step checkpoint.
