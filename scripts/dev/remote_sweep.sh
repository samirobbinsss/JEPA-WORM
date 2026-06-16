#!/usr/bin/env bash
# Remote 3-seed sweep helper. Designed for the Runpod-style ephemeral
# A100 pod workflow established 2026-05-18 / 2026-05-19.
#
# Required env / args:
#   REMOTE_HOST      e.g. root@<pod-ip>
#   REMOTE_PORT      e.g. <pod-ssh-port>
#   SSH_KEY          e.g. ~/.ssh/id_ed25519
#   LOCAL_REPO       project root (auto-detect via script dir if omitted)
#   REMOTE_REPO      default: /workspace/jepa-worm
#
# Usage:
#   REMOTE_HOST=root@<pod-ip> REMOTE_PORT=<port> SSH_KEY=~/.ssh/id_ed25519 \
#     bash scripts/dev/remote_sweep.sh
#
# WORMJEPA_RESULTS_ON_WORKSPACE=1 keeps results on the network volume
# (survives pod close, accepts the FUSE checkpoint-save penalty); the
# default symlinks to /root/results (faster, lost on pod close).
#
# Every step is idempotent.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?must set REMOTE_HOST (e.g. root@1.2.3.4)}"
REMOTE_PORT="${REMOTE_PORT:?must set REMOTE_PORT}"
SSH_KEY="${SSH_KEY:?must set SSH_KEY (path to ssh private key)}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/jepa-worm}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SKIP_RSYNC_BACK="${SKIP_RSYNC_BACK:-0}"
# Which config + seeds to run. Defaults reproduce the 3-seed headline
# sweep; override SWEEP_CONFIG / SWEEP_SEEDS for research experiments
# (e.g. SWEEP_CONFIG=configs/jepa_debug.yaml SWEEP_SEEDS=42).
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/headline.yaml}"
SWEEP_SEEDS="${SWEEP_SEEDS:-42 1337 8675309}"
CONFIG_STEM="$(basename "$SWEEP_CONFIG" .yaml)"
# FIX 1: opt-in background prefetch (single-producer, runs one step ahead) to
# overlap pure-Python BAAIWorm clip synthesis with GPU compute. Default ON for
# the headline sweep — it's a scheduling-only change, byte-identical results.
# Set WORMJEPA_PREFETCH=0 to fall back to the synchronous loader.
WORMJEPA_PREFETCH="${WORMJEPA_PREFETCH:-1}"
WORMJEPA_PREFETCH_DEPTH="${WORMJEPA_PREFETCH_DEPTH:-8}"

# Cross-pod resumable checkpointing. WORMJEPA_CHECKPOINT_EVERY>0 makes the runner
# atomic-save a checkpoint every N steps to a stable per-(config,seed) path under
# results/_resume/ on the /workspace network volume, and resume from it on
# relaunch. Combined with the per-seed `done` marker the sweep loop checks below,
# a fresh pod re-running this script skips finished seeds and continues the
# interrupted one — so an ephemeral pod dying mid-run is no longer fatal. Default
# 0 (off) preserves prior single-shot behaviour; the headline launch sets it.
WORMJEPA_CHECKPOINT_EVERY="${WORMJEPA_CHECKPOINT_EVERY:-0}"
# config-slug for the resume/done path; mirrors the runner's run-id slug rule
# (lowercase, any run of non-alphanumerics -> a single underscore).
SWEEP_SLUG="$(printf '%s' "$CONFIG_STEM" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"

# CRITICAL placement split (learned 2026-06-15 the hard way):
#   - The venv + uv cache + tmp go on the LOCAL overlay disk (/root). Installing
#     torch (thousands of small files, a ~2 GB libtorch_cuda.so) into a venv on
#     the MooseFS /workspace network volume deadlocks: uv pins at 0% CPU / 0 net,
#     load spikes, and `uv sync` never finishes. Local-disk small-file writes are
#     fast. No UV_LINK_MODE=copy -> uv hardlinks venv<-cache on the same local fs
#     (instant, space-efficient: ~10 GB peak on the 20 GB overlay).
#   - Only big SEQUENTIAL artifacts stay on /workspace: the torch-hub ViT-L
#     checkpoint (TORCH_HOME, one ~1.3 GB file) and the 45 GB Zenodo data — these
#     are fine on the network volume and benefit from surviving pod close.
_REMOTE_CACHE_ENV="mkdir -p /root/uvcache /root/uvtmp /root/xdg $(dirname "$REMOTE_REPO")/.caches/torch $(dirname "$REMOTE_REPO")/.caches/hf && export UV_PROJECT_ENVIRONMENT=/root/jepa-venv UV_CACHE_DIR=/root/uvcache TMPDIR=/root/uvtmp XDG_CACHE_HOME=/root/xdg TORCH_HOME=$(dirname "$REMOTE_REPO")/.caches/torch HF_HOME=$(dirname "$REMOTE_REPO")/.caches/hf WORMJEPA_CHECKPOINT_DIR=$(dirname "$REMOTE_REPO")/.caches/vjepa-ckpt"

# Keepalive options so long-running steps (uv sync, 45 GB fetch, multi-hour
# training) survive an idle/slow link instead of dropping with "Operation timed
# out / Broken pipe". ServerAliveInterval 30s x CountMax 20 tolerates ~10 min of
# silence before giving up.
_SSH_KEEPALIVE="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes"
SSH_CMD="ssh $_SSH_KEEPALIVE -i $SSH_KEY -p $REMOTE_PORT $REMOTE_HOST"
RSYNC_SSH="ssh $_SSH_KEEPALIVE -i $SSH_KEY -p $REMOTE_PORT"

echo "remote_sweep: LOCAL=$LOCAL_REPO REMOTE=$REMOTE_HOST:$REMOTE_REPO"

echo "[1/8] ensure rsync on remote"
$SSH_CMD 'which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -qq -y rsync)' >/dev/null

echo "[2/8] rsync repo"
# --no-owner --no-group: the /workspace network volume (MooseFS) rejects
# chown, which makes a plain `rsync -a` exit 23. Code files don't need
# owner/group preservation on an ephemeral pod anyway.
rsync -avz --no-owner --no-group \
    --exclude='.venv' --exclude='results' --exclude='data/downloads' \
    --exclude='third_party' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='*.pyc' --exclude='.ruff_cache' --exclude='*.egg-info' \
    --exclude='.cache' \
    --exclude='.claude' --exclude='_bmad' --exclude='.git/objects/pack' \
    -e "$RSYNC_SSH" \
    "$LOCAL_REPO/" "$REMOTE_HOST:$REMOTE_REPO/"

echo "[3/8] ensure uv + venv + package install"
$SSH_CMD "cd $REMOTE_REPO && $_REMOTE_CACHE_ENV && \
    (which uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)) && \
    export PATH=\$HOME/.local/bin:\$PATH && \
    uv sync --quiet"

echo "[4/8] GPU preflight (CUDA driver check)"
# Fail fast on a mis-provisioned pod (driver too old, no GPU) before the
# sweep — a CPU fallback runs ViT-L unusably slow and wastes the run.
# set -e aborts the script on a non-zero exit here.
$SSH_CMD "cd $REMOTE_REPO && export PATH=\$HOME/.local/bin:\$PATH && \
    uv run python scripts/dev/preflight_gpu.py"

echo "[5/8] ensure V-JEPA 2.1 submodule pinned"
$SSH_CMD "cd $REMOTE_REPO && \
    if [ ! -d third_party/vjepa2/.git ]; then \
      git clone -q https://github.com/facebookresearch/vjepa2.git third_party/vjepa2 && \
      cd third_party/vjepa2 && git checkout -q 204698b; \
    fi"

echo "[6/8] ensure WBDB + OWMD data"
# SKIP_ZENODO=1 for baaiworm-only research configs that never touch the
# WBDB/OWMD loaders — no point fetching data they will not read.
# FULL_SUBSET=1 fetches the full pre-committed 100-record subsets (~45 GB,
# the real Phase 0 headline corpus, Stories 9.5/9.6); default fetches only
# the 3 anchors (fast smoke / research configs). The headline sweep MUST
# set FULL_SUBSET=1 — training on anchors alone is not the pre-registered
# corpus. zenodo-subset is idempotent + resumable, so a re-run continues.
if [ "${SKIP_ZENODO:-0}" = "1" ]; then
  echo "  SKIP_ZENODO=1 — skipping Zenodo fetch"
elif [ "${FULL_SUBSET:-0}" = "1" ]; then
  echo "  FULL_SUBSET=1 — fetching full 100-record WBDB + OWMD subsets (~45 GB)"
  $SSH_CMD "cd $REMOTE_REPO && $_REMOTE_CACHE_ENV && export PATH=\$HOME/.local/bin:\$PATH && \
      uv run wormjepa fetch zenodo-subset --dataset both --workers 16"
else
  $SSH_CMD "cd $REMOTE_REPO && $_REMOTE_CACHE_ENV && export PATH=\$HOME/.local/bin:\$PATH && \
      uv run wormjepa fetch zenodo-anchors"
fi

echo "[7/8] symlink results"
if [ "${WORMJEPA_RESULTS_ON_WORKSPACE:-0}" = "1" ]; then
  echo "  results on /workspace (FUSE; survives pod close)"
  # PRESERVE existing results — do NOT rm. On a resume-pod the per-seed
  # checkpoints + done markers under results/_resume/ are exactly what lets the
  # run continue instead of restarting from step 0; wiping them defeats the
  # cross-pod resume. mkdir -p is enough (first launch has nothing to keep).
  $SSH_CMD "cd $REMOTE_REPO && mkdir -p results"
else
  echo "  results -> /root (container disk; LOST on pod close; faster torch.save)"
  $SSH_CMD "cd $REMOTE_REPO && rm -rf results && mkdir -p /root/results && ln -sf /root/results results"
fi

echo "[8/8] run sweep: $SWEEP_CONFIG seeds=[$SWEEP_SEEDS] prefetch=$WORMJEPA_PREFETCH depth=$WORMJEPA_PREFETCH_DEPTH"
# Write the seed loop to a pod-side script, then run it. Writing a script (vs a
# one-line inline SSH command) avoids fragile nested-quote escaping and lets the
# sweep run detached. WORMJEPA_PREFETCH gates the background prefetch loader in
# the runner (src/wormjepa/training/runner.py).
$SSH_CMD "cat > $REMOTE_REPO/run_sweep.sh" <<REMOTE_SWEEP_SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd $REMOTE_REPO
export PATH=\$HOME/.local/bin:\$PATH
$_REMOTE_CACHE_ENV
export WORMJEPA_PREFETCH=$WORMJEPA_PREFETCH
export WORMJEPA_PREFETCH_DEPTH=$WORMJEPA_PREFETCH_DEPTH
export WORMJEPA_CHECKPOINT_EVERY=$WORMJEPA_CHECKPOINT_EVERY
for seed in $SWEEP_SEEDS; do
  DONE_MARKER="$REMOTE_REPO/results/_resume/${SWEEP_SLUG}__seed\${seed}/done"
  if [ -f "\$DONE_MARKER" ]; then
    echo "=== seed=\$seed already complete — skipping (\$DONE_MARKER) ==="
    continue
  fi
  echo "=== seed=\$seed ==="
  time uv run wormjepa run --config $SWEEP_CONFIG --seed \$seed
done
echo "=== sweep complete ==="
REMOTE_SWEEP_SCRIPT

if [ "${SWEEP_DETACH:-0}" = "1" ]; then
  # Detach on the pod so a multi-hour sweep survives SSH drops. Progress -> the
  # pod-side log; poll it (SSH tail) for per-seed metrics, collect results after.
  REMOTE_SWEEP_LOG="$REMOTE_REPO/sweep.log"
  $SSH_CMD "nohup bash $REMOTE_REPO/run_sweep.sh > $REMOTE_SWEEP_LOG 2>&1 < /dev/null & echo DETACHED pid \$! log $REMOTE_SWEEP_LOG"
  echo "remote_sweep: sweep launched DETACHED on pod -> $REMOTE_SWEEP_LOG"
  echo "remote_sweep: poll that log for progress; results NOT rsynced (collect after completion)."
  exit 0
fi
$SSH_CMD "bash $REMOTE_REPO/run_sweep.sh"

if [ "$SKIP_RSYNC_BACK" != "1" ]; then
  echo "[post] rsync results back"
  # Results live where step 7 put them: /root/results (container disk,
  # default) or $REMOTE_REPO/results (the /workspace network volume).
  # run-id ends in the config stem, so the glob keys off CONFIG_STEM.
  if [ "${WORMJEPA_RESULTS_ON_WORKSPACE:-0}" = "1" ]; then
    REMOTE_RESULTS_GLOB="$REMOTE_REPO/results/*${CONFIG_STEM}"
  else
    REMOTE_RESULTS_GLOB="/root/results/*${CONFIG_STEM}"
  fi
  $SSH_CMD "ls -d $REMOTE_RESULTS_GLOB 2>/dev/null || true" | while read -r d; do
    [ -z "$d" ] && continue
    echo "  rsync $d"
    rsync -avz --no-owner --no-group -e "$RSYNC_SSH" \
        "$REMOTE_HOST:$d" "$LOCAL_REPO/results/" || true
  done
fi

echo "remote_sweep: done"
