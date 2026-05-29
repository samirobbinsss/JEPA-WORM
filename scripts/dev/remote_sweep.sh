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

SSH_CMD="ssh -i $SSH_KEY -p $REMOTE_PORT $REMOTE_HOST"
RSYNC_SSH="ssh -i $SSH_KEY -p $REMOTE_PORT"

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
$SSH_CMD "cd $REMOTE_REPO && \
    (which uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)) && \
    export PATH=\$HOME/.local/bin:\$PATH && \
    export UV_LINK_MODE=copy && \
    uv sync --quiet && \
    uv pip install -e . --quiet"

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

echo "[6/8] ensure WBDB + OWMD anchor data"
# SKIP_ZENODO=1 for baaiworm-only research configs that never touch the
# WBDB/OWMD loaders — no point fetching anchors they will not read.
if [ "${SKIP_ZENODO:-0}" = "1" ]; then
  echo "  SKIP_ZENODO=1 — skipping anchor fetch"
else
  $SSH_CMD "cd $REMOTE_REPO && export PATH=\$HOME/.local/bin:\$PATH && \
      uv run wormjepa fetch zenodo-anchors"
fi

echo "[7/8] symlink results"
if [ "${WORMJEPA_RESULTS_ON_WORKSPACE:-0}" = "1" ]; then
  echo "  results on /workspace (FUSE; survives pod close)"
  $SSH_CMD "cd $REMOTE_REPO && rm -rf results && mkdir -p results"
else
  echo "  results -> /root (container disk; LOST on pod close; faster torch.save)"
  $SSH_CMD "cd $REMOTE_REPO && rm -rf results && mkdir -p /root/results && ln -sf /root/results results"
fi

echo "[8/8] run sweep: $SWEEP_CONFIG seeds=[$SWEEP_SEEDS]"
$SSH_CMD "cd $REMOTE_REPO && export PATH=\$HOME/.local/bin:\$PATH && export UV_LINK_MODE=copy && \
    for seed in $SWEEP_SEEDS; do \
      echo \"=== seed=\$seed ===\" && \
      time uv run wormjepa run --config $SWEEP_CONFIG --seed \$seed; \
    done"

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
