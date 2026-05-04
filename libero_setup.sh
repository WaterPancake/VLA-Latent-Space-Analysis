#!/usr/bin/env bash
# setup_libero.sh
# -----------------------------------------------------------------------------
# Installs LIBERO into the existing MiniVLA rollout environment on a RunPod pod.
#
# Assumes you've already run setup_rollout.sh (so torch 2.4.0+cu124,
# transformers 4.40.1, numpy 2.3.5+, flash-attn etc. are installed).
#
# What this does:
#   1. Install the apt-level system libs MuJoCo needs for headless rendering
#      (both EGL and OSMesa, so you can fall back if one fails).
#   2. Install LIBERO's actual runtime deps at versions compatible with our
#      current env. We deliberately ignore most of LIBERO's requirements.txt
#      because half of it is fossilized (numpy==1.22.4, transformers==4.21.1).
#   3. Install LIBERO itself with --no-deps and pip install -e .
#   4. Set up MuJoCo to use EGL by default.
#   5. Configure robosuite to cache its assets in /workspace (so they survive
#      pod restarts on a Network Volume).
#   6. Smoke-test that an env actually creates and renders.
#
# Usage:
#   bash libero_setup.sh
#
# After this completes, run setup_libero_data.sh separately to download the
# task datasets (~14GB) — kept as a separate script so you can skip it during
# debugging iterations.
# -----------------------------------------------------------------------------

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
LIBERO_DIR="${LIBERO_DIR:-${WORKSPACE}/LIBERO}"
ROBOSUITE_ASSETS="${WORKSPACE}/robosuite_assets"

log()     { printf '\033[1;36m[libero] %s\033[0m\n' "$*"; }
warn()    { printf '\033[1;33m[libero WARN] %s\033[0m\n' "$*" >&2; }
fail()    { printf '\033[1;31m[libero FAIL] %s\033[0m\n' "$*" >&2; exit 1; }
section() { printf '\n\033[1;35m===== %s =====\033[0m\n' "$*"; }

# ----- Step 1: apt-level system libs ----------------------------------------
section "Step 1 — Install system libraries for headless MuJoCo rendering"

# EGL for GPU-accelerated offscreen rendering (preferred):
#   libegl1, libgles2, libglvnd0
# OSMesa for software CPU fallback (slower but bulletproof):
#   libosmesa6, libosmesa6-dev, patchelf
# X11 stubs that some MuJoCo paths still touch even without a display:
#   libgl1, libxext6, libxrandr2, libxcursor1, libxinerama1, libxi6
# Build deps for any source-built Python packages (h5py, etc.):
#   build-essential


export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    libegl1 libgles2 libglvnd0 \
    libosmesa6 libosmesa6-dev patchelf \
    libgl1 libxext6 libxrandr2 libxcursor1 libxinerama1 libxi6 \
    build-essential \
    cmake \
    ffmpeg \
    > /tmp/apt.log 2>&1 || warn "apt install had warnings"

log "System libs installed."

# ----- Step 2: Python deps that LIBERO actually needs -----------------------
section "Step 2 — Install LIBERO's runtime Python deps"

# We install these at versions COMPATIBLE with our existing rollout env, not
# at LIBERO's pinned versions in requirements.txt (which are fossils).
#
# Why each is here:
#   robosuite==1.4.1   - LIBERO's env backend (1.4.0 has a known mujoco bug)
#   bddl==1.0.1        - Behavior Definition Language, used by robosuite tasks
#   robomimic==0.3.0   - Demo loading utilities (newer than LIBERO's pin to
#                        avoid numpy<2 conflicts; APIs LIBERO uses haven't changed)
#   mujoco>=3.0.0      - The MuJoCo Python bindings (NOT the old mujoco_py)
#   gym==0.25.2        - LIBERO uses old gym API; this version is small and
#                        has no compiled deps so the pin doesn't conflict
#   hydra-core         - Config system; used by training code, not rollout,
#                        but imported eagerly so we need it
#   easydict, h5py, omegaconf, imageio, termcolor, hidet
#                      - Misc utilities LIBERO imports at top level
#
# Skipped from LIBERO's requirements.txt because we already have them at
# better versions: numpy, torch, torchvision, transformers, einops, opencv, PIL
#
# Skipped because training-only and we're doing rollout: wandb, thop, matplotlib

uv pip install --system \
    "robosuite==1.4.1" \
    "bddl==1.0.1" \
    "robomimic==0.3.0" \
    "mujoco>=3.0.0" \
    "gym==0.25.2" \
    "hydra-core>=1.2.0" \
    "easydict" \
    "h5py" \
    "omegaconf" \
    "imageio[ffmpeg]" \
    "termcolor" \
    "cloudpickle" \
    "future" \
    "matplotlib"

log "Python deps installed."

# ----- Step 3: Clone LIBERO and install with --no-deps ----------------------
section "Step 3 — Clone and install LIBERO"

if [[ -d "${LIBERO_DIR}" ]]; then
    log "${LIBERO_DIR} already exists — pulling latest"
    git -C "${LIBERO_DIR}" pull --ff-only || warn "git pull failed (maybe local changes); continuing"
else
    log "Cloning LIBERO to ${LIBERO_DIR}"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

# --no-deps because LIBERO's setup.py has install_requires=[] but its
# requirements.txt would re-introduce numpy==1.22.4 if we used it (we don't).
uv pip install --system --no-deps -e "${LIBERO_DIR}"

log "LIBERO installed (editable)."

# ----- Step 4: Configure MuJoCo for headless rendering ----------------------
section "Step 4 — Configure MuJoCo headless rendering (EGL)"

# Set MUJOCO_GL=egl as default. Persist it via /etc/environment so it
# survives shell restarts; also export it in the current shell.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Persist for future shells. Use a marker so re-running the script doesn't
# duplicate the lines.
PROFILE_MARKER="# === setup_libero.sh env vars ==="
PROFILE_FILE="/etc/profile.d/libero.sh"
if [[ ! -f "${PROFILE_FILE}" ]] || ! grep -q "${PROFILE_MARKER}" "${PROFILE_FILE}" 2>/dev/null; then
    if [[ "$(id -u)" -eq 0 ]]; then WRITE_CMD="tee"; else WRITE_CMD="sudo tee"; fi
    ${WRITE_CMD} "${PROFILE_FILE}" > /dev/null <<EOF
${PROFILE_MARKER}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOSUITE_DEFAULT_ASSET_PATH=${ROBOSUITE_ASSETS}
EOF
    log "Wrote ${PROFILE_FILE} — env vars will persist for new shells."
else
    log "${PROFILE_FILE} already configured."
fi

# Robosuite asset cache on the persistent volume
mkdir -p "${ROBOSUITE_ASSETS}"
export ROBOSUITE_DEFAULT_ASSET_PATH="${ROBOSUITE_ASSETS}"
log "Robosuite asset dir: ${ROBOSUITE_ASSETS}"

# ----- Step 5: Smoke-test imports first -------------------------------------
section "Step 5 — Smoke-test LIBERO imports"

python <<'PY'
import os
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")

# Order matters: import mujoco first to get a clear error if EGL libs missing
import mujoco
print(f"mujoco: {mujoco.__version__}")

import robosuite
print(f"robosuite: {robosuite.__version__}")

import bddl
print(f"bddl: imported")

import libero
from libero.libero import benchmark, get_libero_path
print(f"libero: {libero.__file__}")

# This is the canonical LIBERO benchmark accessor
benchmark_dict = benchmark.get_benchmark_dict()
print(f"Available benchmarks: {list(benchmark_dict.keys())}")
PY

# ----- Step 6: Smoke-test env creation + rendering --------------------------
section "Step 6 — Smoke-test env creation and offscreen rendering"

# This is the real test — actually instantiate a LIBERO env and render a frame.
# If MUJOCO_GL or the system libs are misconfigured, this fails here, not
# during your first rollout an hour into a debugging session.

python <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

# Pick the libero_90 task suite to match your converted MiniVLA model
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_90"
task_suite = benchmark_dict[task_suite_name]()

# Use task 0 as a smoke test
task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"Task: {task.language}")
print(f"BDDL: {task_bddl_file}")

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 224,
    "camera_widths": 224,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()

# Verify obs has the camera images we expect
print(f"obs keys: {sorted(obs.keys())}")
agentview = obs.get("agentview_image")
if agentview is None:
    raise RuntimeError("agentview_image missing from obs — render failed silently")
print(f"agentview_image: shape={agentview.shape}, dtype={agentview.dtype}")
print(f"  mean pixel value: {agentview.mean():.1f} (should NOT be 0 — that means a black render)")

# Step the env once with a zero action
action = np.zeros(env.env.action_dim)
obs, reward, done, info = env.step(action)
print(f"step OK: reward={reward}, done={done}")

env.close()
print("Env smoke test passed.")
PY

# ----- Done ------------------------------------------------------------------
section "LIBERO install complete"

log "Next step: download the LIBERO datasets (~14GB) with:"
log "  bash setup_libero_data.sh"
log ""
log "Or skip the dataset download if you only need to run policy rollouts (the"
log "BDDL task definitions ship with the LIBERO repo; only demo data needs the"
log "separate download)."
log ""
log "To use LIBERO in a script, set MUJOCO_GL=egl before importing mujoco/libero."
log "The /etc/profile.d/libero.sh file does this for new shells automatically."
