#!/usr/bin/env bash
# setup_pod.sh
# -----------------------------------------------------------------------------
# Combined RunPod setup for MiniVLA + LIBERO rollout environment.
# Idempotent — safe to re-run on a pod that already has some pieces installed.
#
# Target template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Recommended GPU: RTX 4090 (sm_89, native bf16, 24GB)
#
# What this script does, in order:
#   1. Sanity-check the template (Python 3.11, torch 2.4.0+cu124 with CXX11_ABI=False)
#   2. Install uv
#   3. Install apt-level system libs for headless MuJoCo + cmake for egl-probe
#   4. Write requirements.in to /workspace
#   5. Compile a hash-pinned lock with the cu124 torch index
#   6. Install from lock with the cu124 torch index again (uv refuses to bake
#      index URLs into locks, so it has to be specified at install time too)
#   7. Verify torch survived as cu124+CUDA-capable, not as the CPU PyPI wheel
#   8. Smoke-test core packages with a real GPU matmul
#   9. Install the matching flash-attn prebuilt wheel
#  10. Smoke-test flash-attn with a real CUDA kernel call
#  11. Clone LIBERO + the __init__.py touch fix + editable install with --no-deps
#  12. Configure MuJoCo for headless EGL rendering, persist via /etc/profile.d/
#  13. Smoke-test that mujoco/robosuite/bddl/libero all import
#  14. Smoke-test an actual LIBERO env reset + render + step
#
# Usage:
#   chmod +x setup_pod.sh
#   ./setup_pod.sh
#
# After this completes:
#   - source /etc/profile.d/libero.sh      (or open a new shell)
#   - python -c "from libero.libero import benchmark; print('ok')"
# -----------------------------------------------------------------------------

set -euo pipefail

# ----- Config ----------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
REQ_IN="${WORKSPACE}/requirements.in"
REQ_LOCK="${WORKSPACE}/requirements.lock"
LIBERO_DIR="${LIBERO_DIR:-${WORKSPACE}/LIBERO}"
ROBOSUITE_ASSETS="${WORKSPACE}/robosuite_assets"


VENV_DIR="${WORKSPACE}/venvs/libero"

export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TORCH_HOME="${WORKSPACE}/.cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export TMPDIR="${WORKSPACE}/tmp"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

    

# Expected template state — bail loudly if mismatch
EXPECTED_TORCH_VERSION="2.4.0"
EXPECTED_TORCH_CUDA="12.4"
EXPECTED_TORCH_CXX11_ABI="False"
EXPECTED_PYTHON_MAJORMINOR="3.11"

# PyTorch's cu124 index — torch==2.4.0 from default PyPI is the CPU build,
# so we need this index to get the GPU wheels.
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Flash-attn wheel matching torch 2.4.x + cu12 + cp311 + abiFALSE + Linux x86
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# ----- Helpers ---------------------------------------------------------------
log()     { printf '\033[1;36m[setup] %s\033[0m\n' "$*"; }
warn()    { printf '\033[1;33m[setup WARN] %s\033[0m\n' "$*" >&2; }
fail()    { printf '\033[1;31m[setup FAIL] %s\033[0m\n' "$*" >&2; exit 1; }
section() { printf '\n\033[1;35m===== %s =====\033[0m\n' "$*"; }

# ----- Step 0.5: Setup virtual env ------------------------------------------
VENV_DIR="${WORKSPACE}/venvs/libero"

export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TORCH_HOME="${WORKSPACE}/.cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export TMPDIR="${WORKSPACE}/tmp"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

# ----- Step 1: Verify preinstalled torch -------------------------------------
section "Step 1 — Verify RunPod template matches expectations"


TORCH_PROBE="$(python -c '
import sys
try:
    import torch
except ImportError as e:
    print(f"NOTORCH:{e}")
    sys.exit(0)
print(f"{torch.__version__}|{torch.version.cuda}|{torch._C._GLIBCXX_USE_CXX11_ABI}")
' 2>&1)"

PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

log "Python:       ${PY_MM}"
log "Torch probe:  ${TORCH_PROBE}"

if [[ "${TORCH_PROBE}" == NOTORCH:* ]]; then
    fail "torch is not installed. Wrong RunPod template? Expected runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04."
fi

IFS='|' read -r TORCH_VER TORCH_CUDA TORCH_ABI <<< "${TORCH_PROBE}"
TORCH_VER_BASE="${TORCH_VER%%+*}"

if [[ "${PY_MM}" != "${EXPECTED_PYTHON_MAJORMINOR}" ]]; then
    fail "Expected Python ${EXPECTED_PYTHON_MAJORMINOR}, got ${PY_MM}."
fi

if [[ "${TORCH_VER_BASE}" != "${EXPECTED_TORCH_VERSION}" ]]; then
    warn "Preinstalled torch is ${TORCH_VER_BASE}, expected ${EXPECTED_TORCH_VERSION}."
    warn "Lockfile pin will replace it during install."
fi

if [[ "${TORCH_CUDA}" != "${EXPECTED_TORCH_CUDA}" ]]; then
    warn "Preinstalled torch CUDA is ${TORCH_CUDA}, expected ${EXPECTED_TORCH_CUDA}."
fi

if [[ "${TORCH_ABI}" != "${EXPECTED_TORCH_CXX11_ABI}" ]]; then
    fail "torch CXX11_ABI is ${TORCH_ABI}, expected ${EXPECTED_TORCH_CXX11_ABI}. flash-attn wheel won't link. Bail."
fi

log "Template sanity check passed."

# ----- Step 2: Install uv ----------------------------------------------------
section "Step 2 — Install uv"

if command -v uv >/dev/null 2>&1; then
    log "uv already present: $(uv --version)"
else
    pip install -q uv
    log "Installed: $(uv --version)"
fi

# ----- Step 3: System-level apt packages -------------------------------------
section "Step 3 — Install apt packages (MuJoCo headless rendering deps + cmake)"

# EGL: GPU-accelerated offscreen rendering (preferred)
# OSMesa: software CPU fallback if EGL fails on a particular pod
# X11 stubs: some MuJoCo paths still touch them even without a display
# build-essential + cmake: build deps for source-only packages like egl-probe
# ffmpeg: imageio uses it for video encoding (rollout videos)

if [[ "$(id -u)" -eq 0 ]]; then
    APT="apt-get"
else
    APT="sudo apt-get"
fi

export DEBIAN_FRONTEND=noninteractive
${APT} update -qq
${APT} install -y --no-install-recommends \
    libegl1 libgles2 libglvnd0 \
    libosmesa6 libosmesa6-dev patchelf \
    libgl1 libxext6 libxrandr2 libxcursor1 libxinerama1 libxi6 \
    build-essential cmake \
    ffmpeg \
    > /tmp/apt.log 2>&1 || warn "apt install had warnings — see /tmp/apt.log"

${APT} clean || true
rm -rf /var/lib/apt/lists/* /tmp/apt.log || true

log "System libs installed."

# ----- Step 4: Write requirements.in ----------------------------------------
section "Step 4 — Write requirements.in to ${WORKSPACE}"

mkdir -p "${WORKSPACE}"

cat > "${REQ_IN}" <<'EOF'
# RunPod env for MiniVLA + LIBERO rollout / hidden-state capture.
# Target: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# --- Pin torch to RunPod's preinstalled version. Without this, uv resolves
#     transitively to whatever's newest on PyPI (currently torch 2.11+cu13),
#     which has no prebuilt flash-attn wheel and breaks the whole setup.
torch==2.4.0
torchvision==0.19.0

# --- VLA / model-loading deps (versions match openvla-mini's tested set)
transformers==4.40.1
tokenizers==0.19.1
timm==0.9.10
huggingface_hub
accelerate>=0.25.0
sentencepiece==0.1.99
protobuf
draccus==0.8.0

# --- Used by flash-attn at runtime
einops

# --- Numerics. numpy 2.3.5+ for _blas_supports_fpe symbol; <2 breaks TF/JAX ABI
numpy>=2.3.5

# --- Pillow 12.0.0 has a broken _Ink import; 12.1.0 fixed it
Pillow>=12.1.0

# --- LIBERO runtime stack (versions chosen for compat with our env, NOT what
#     LIBERO's requirements.txt pins — that's a fossil with numpy==1.22.4)
robosuite==1.4.1
bddl==1.0.1
robomimic==0.3.0
mujoco>=3.0.0
gym==0.25.2
hydra-core>=1.2.0
easydict
h5py
omegaconf
imageio[ffmpeg]
termcolor
cloudpickle
future
matplotlib
EOF

log "Wrote ${REQ_IN} ($(wc -l < "${REQ_IN}") lines)"

# ----- Step 5: Compile lock --------------------------------------------------
section "Step 5 — Compile requirements.lock"

uv pip compile "${REQ_IN}" \
    -o "${REQ_LOCK}" \
    --generate-hashes \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match # something something 

log "Lock compiled. Showing key lines:"
grep -E "^(torch|torchvision|numpy|Pillow|pillow|robosuite|mujoco)==" "${REQ_LOCK}" | head -10 || true

# ----- Step 6: Install from lock --------------------------------------------
section "Step 6 — Install from lock"

# --extra-index-url is REQUIRED at install time too — uv refuses to bake
# index URLs into lock files for security reasons.
uv pip install \
    -r "${REQ_LOCK}" \
    --system \
    --require-hashes \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match # something something

# ----- Step 7: Verify torch wasn't replaced with the CPU wheel --------------
section "Step 7 — Verify torch is still cu124 + CUDA-capable"

TORCH_POST="$(python -c '
import torch, torchvision
print(f"{torch.__version__}|{torch.version.cuda}|{torch._C._GLIBCXX_USE_CXX11_ABI}|{torchvision.__version__}|{torch.cuda.is_available()}")
')"

IFS='|' read -r T_VER T_CUDA T_ABI TV_VER T_CUDA_OK <<< "${TORCH_POST}"

log "Post-install torch:        ${T_VER}"
log "Post-install torch CUDA:   ${T_CUDA}"
log "Post-install torch ABI:    ${T_ABI}"
log "Post-install torchvision:  ${TV_VER}"
log "torch.cuda.is_available(): ${T_CUDA_OK}"

if [[ "${T_VER}" != *"+cu124"* ]]; then
    fail "torch is not the cu124 build (got '${T_VER}'). The cu124 index URL didn't take effect."
fi
if [[ "${T_CUDA_OK}" != "True" ]]; then
    fail "torch.cuda.is_available() == False. Check nvidia-smi and pod GPU assignment."
fi
if [[ "${T_ABI}" != "${EXPECTED_TORCH_CXX11_ABI}" ]]; then
    fail "torch ABI changed to '${T_ABI}'. flash-attn wheel won't link."
fi

log "Torch survived intact."

# ----- Step 8: Core smoke test -----------------------------------------------
section "Step 8 — Core smoke test"

python <<'PY'
import sys, torch, transformers, timm, numpy, PIL
print(f"Python:       {sys.version.split()[0]}")
print(f"torch:        {torch.__version__} | CUDA: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"transformers: {transformers.__version__}")
print(f"timm:         {timm.__version__}")
print(f"numpy:        {numpy.__version__}")
print(f"Pillow:       {PIL.__version__}")

# Real GPU op — forces CUDA init, surfaces driver/runtime issues
x = torch.randn(64, 64, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print(f"GPU matmul OK: shape={tuple(y.shape)}, dtype={y.dtype}")
PY

log "Core smoke test passed."

# ----- Step 9: Install flash-attn -------------------------------------------
section "Step 9 — Install flash-attn"

if python -c "import flash_attn" 2>/dev/null; then
    EXISTING_FA="$(python -c 'import flash_attn; print(flash_attn.__version__)')"
    log "flash-attn already installed (version ${EXISTING_FA}). Skipping."
    log "To force-reinstall: uv pip uninstall flash-attn --system && re-run this script"
else
    log "Installing flash-attn from prebuilt wheel..."
    log "URL: ${FLASH_ATTN_WHEEL_URL}"
    uv pip install --system "${FLASH_ATTN_WHEEL_URL}"
fi

# ----- Step 10: Flash-attn smoke test ---------------------------------------
section "Step 10 — Flash-attn smoke test (real CUDA kernel)"

python <<'PY'
import torch
from flash_attn import flash_attn_func
import flash_attn

assert torch.cuda.is_available(), "CUDA not available"

q = k = v = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
out = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()

assert out.shape == q.shape, f"Unexpected shape: {out.shape}"
assert out.dtype == torch.bfloat16, f"Unexpected dtype: {out.dtype}"
print(f"flash-attn OK: shape={tuple(out.shape)}, dtype={out.dtype}")
print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: {torch.cuda.get_device_capability()}")
print(f"flash-attn version: {flash_attn.__version__}")
PY

# ----- Step 11: Clone and install LIBERO ------------------------------------
section "Step 11 — Clone and install LIBERO"

if [[ -d "${LIBERO_DIR}" ]]; then
    log "${LIBERO_DIR} already exists — pulling latest"
    git -C "${LIBERO_DIR}" pull --ff-only || warn "git pull failed (maybe local changes); continuing"
else
    log "Cloning LIBERO to ${LIBERO_DIR}"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

# CRITICAL: LIBERO's repo is missing __init__.py at LIBERO/libero/. Without
# this file, find_packages() in setup.py returns no packages, and the
# editable install registers nothing — `import libero` fails silently.
# Must be done BEFORE pip install -e because find_packages runs at install time.
if [[ ! -f "${LIBERO_DIR}/libero/__init__.py" ]]; then
    echo "# Placeholder so find_packages() discovers libero.* (LIBERO repo is missing this)." \
        > "${LIBERO_DIR}/libero/__init__.py"
    log "Created ${LIBERO_DIR}/libero/__init__.py"
fi

# Also create the datasets dir to silence LIBERO's "datasets path does not exist"
# warning at import time. The dir stays empty since we don't need demo data
# for VLA rollouts.
mkdir -p "${LIBERO_DIR}/libero/datasets"

# --no-deps because LIBERO's setup.py has install_requires=[] but its
# requirements.txt would re-introduce numpy==1.22.4 etc. We installed the
# real deps already in step 6.
uv pip install --system --no-deps -e "${LIBERO_DIR}"

log "LIBERO installed (editable)."

# ----- Step 12: Configure MuJoCo headless rendering --------------------------
section "Step 12 — Configure MuJoCo headless rendering (EGL)"

mkdir -p "${ROBOSUITE_ASSETS}"

PROFILE_FILE="/etc/profile.d/libero.sh"
PROFILE_MARKER="# === setup_pod.sh env vars ==="

if [[ ! -f "${PROFILE_FILE}" ]] || ! grep -q "${PROFILE_MARKER}" "${PROFILE_FILE}" 2>/dev/null; then
    if [[ "$(id -u)" -eq 0 ]]; then WRITE_CMD="tee"; else WRITE_CMD="sudo tee"; fi
    ${WRITE_CMD} "${PROFILE_FILE}" > /dev/null <<EOF
${PROFILE_MARKER}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOSUITE_DEFAULT_ASSET_PATH=${ROBOSUITE_ASSETS}
export LIBERO_CONFIG_PATH=${WORKSPACE}/.libero
EOF
    log "Wrote ${PROFILE_FILE}"
else
    log "${PROFILE_FILE} already configured."
fi

# Apply to the current shell so the smoke tests below work
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOSUITE_DEFAULT_ASSET_PATH="${ROBOSUITE_ASSETS}"
export LIBERO_CONFIG_PATH="${WORKSPACE}/.libero"

# ----- Step 12b: Pre-write LIBERO config to skip the interactive prompt -----
section "Step 12b — Pre-write LIBERO config to skip first-import prompt"

# LIBERO's libero/libero/__init__.py prompts via input() on first import if
# ~/.libero/config.yaml (or $LIBERO_CONFIG_PATH/config.yaml) doesn't exist.
# That breaks any non-interactive context: bash heredocs, Jupyter kernels,
# CI, etc. Pre-write the config so the prompt block is never reached.
# The values match what LIBERO would write if you answered "N" to the prompt.

mkdir -p "${LIBERO_CONFIG_PATH}"

python <<PY
import os, yaml

LIBERO_PKG = "${LIBERO_DIR}/libero/libero"
config = {
    "benchmark_root": LIBERO_PKG,
    "bddl_files":     os.path.join(LIBERO_PKG, "bddl_files"),
    "init_states":    os.path.join(LIBERO_PKG, "init_files"),
    "datasets":       os.path.join(LIBERO_PKG, "../datasets"),
    "assets":         os.path.join(LIBERO_PKG, "assets"),
}

config_file = os.path.join("${LIBERO_CONFIG_PATH}", "config.yaml")
with open(config_file, "w") as f:
    yaml.dump(config, f)

print(f"Wrote {config_file}")
for k, v in config.items():
    print(f"  {k}: {v}")
PY

log "LIBERO config pre-written; first import won't prompt."

# ----- Step 13: LIBERO import smoke test ------------------------------------
section "Step 13 — LIBERO import smoke test"

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

# Note: LIBERO's importable name is libero.libero (not just libero). The
# top-level libero/ dir is a namespace; libero/libero/ is the real package.
from libero.libero import benchmark, get_libero_path
print(f"libero loaded from: {benchmark.__file__}")

benchmark_dict = benchmark.get_benchmark_dict()
print(f"Available benchmarks: {list(benchmark_dict.keys())}")
PY

# ----- Step 14: LIBERO env smoke test ---------------------------------------
section "Step 14 — LIBERO env smoke test (reset + render + step)"

# Real test: actually instantiate an env and render. If EGL is misconfigured,
# this fails here, not during your first rollout an hour into a debugging session.

python <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

# libero_10 to match the SAFE paper / our planned eval suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()

task = task_suite.get_task(0)
bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"Task 0: {task.language}")

env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224)
env.seed(0)
obs = env.reset()

# Verify obs has the expected camera images
print(f"obs keys: {sorted(obs.keys())}")
agentview = obs.get("agentview_image")
if agentview is None:
    raise RuntimeError("agentview_image missing from obs — render failed silently")

mean_pixel = float(agentview.mean())
print(f"agentview_image: shape={agentview.shape}, dtype={agentview.dtype}, mean={mean_pixel:.1f}")

if mean_pixel < 5.0:
    raise RuntimeError(
        f"Mean pixel value is {mean_pixel:.1f} — render is black, EGL is not actually rendering. "
        f"Try MUJOCO_GL=osmesa as fallback."
    )

# Step once with zero action to verify env.step works
action = np.zeros(env.env.action_dim)
obs, reward, done, info = env.step(action)
print(f"step OK: reward={reward}, done={done}")

env.close()
print("LIBERO env smoke test passed.")
PY

# ----- Done ------------------------------------------------------------------
section "All done"

log "Environment ready. Saved artifacts:"
log "  ${REQ_IN}"
log "  ${REQ_LOCK}"
log "  ${LIBERO_DIR}/"
log "  ${PROFILE_FILE}  (env vars for new shells)"
log ""
log "For new shells in this pod, env vars auto-load via /etc/profile.d/libero.sh."
log "If using JupyterLab, restart the kernel to pick up MUJOCO_GL=egl."
log ""
log "If a future pod has EGL trouble (mean pixel value 0 from MuJoCo render),"
log "switch to OSMesa: edit ${PROFILE_FILE} and change egl -> osmesa."
log ""
log "To rebuild this exact env on a fresh pod:"
log "  1. Open new pod, attach this script + requirements.lock from /workspace"
log "  2. Run ./setup_pod.sh — it'll detect existing artifacts and skip work"
