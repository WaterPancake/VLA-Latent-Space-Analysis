#!/usr/bin/env bash
# One-shot Google Colab setup for reproducing SAFE's original OpenVLA LIBERO
# rollout feature extraction path.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VLA_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE="${WORKSPACE:-/content}"
OPENVLA_DIR="${SAFE_OPENVLA_DIR:-${WORKSPACE}/vla-safe-openvla}"
LIBERO_DIR="${LIBERO_DIR:-${WORKSPACE}/LIBERO}"
SAFE_DIR="${SAFE_DIR:-${VLA_REPO_DIR}/SAFE}"
ROLLOUT_ROOT="${SAFE_OPENVLA_ROLLOUT_ROOT:-${OPENVLA_DIR}/rollouts}"
REQ_IN="${SCRIPT_DIR}/requirements-colab.in"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl}"

log() { printf '\033[1;36m[safe-openvla-colab] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[safe-openvla-colab WARN] %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[1;31m[safe-openvla-colab FAIL] %s\033[0m\n' "$*" >&2; exit 1; }
section() { printf '\n\033[1;35m===== %s =====\033[0m\n' "$*"; }

section "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. In Colab, set Runtime > Change runtime type > GPU."
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | xargs)"
log "GPU: ${GPU_NAME} (compute capability ${GPU_CC})"
case "${GPU_CC}" in
    8.0|8.6|8.9|9.0|9.0a|10.0)
        log "Flash Attention 2 compatible GPU detected."
        ;;
    7.5)
        fail "T4 detected. Use L4 or A100; T4 lacks Flash Attention 2 support and native bf16."
        ;;
    *)
        warn "Unrecognized compute capability ${GPU_CC}; continuing, but flash-attn may fail."
        ;;
esac

section "System packages for headless LIBERO"
sudo apt-get -qq update
sudo apt-get -qq install -y --no-install-recommends \
    git git-lfs ffmpeg cmake patchelf ninja-build \
    libgl1-mesa-glx libglib2.0-0 libosmesa6-dev libegl1-mesa-dev \
    libgl1-mesa-dev libglfw3 libglew-dev xvfb

section "Python dependencies"
python -m pip install --quiet --upgrade pip setuptools wheel packaging ninja
python -m pip install --quiet "cmake<4" scikit-build
python -m pip install --no-build-isolation --no-cache-dir egl_probe
python -m pip install --quiet -r "${REQ_IN}" --extra-index-url "${TORCH_INDEX_URL}"

section "dlimp OpenVLA fork"
python -m pip uninstall -y dlimp >/dev/null 2>&1 || true
python -m pip install --no-deps --no-cache-dir "dlimp @ git+https://github.com/moojink/dlimp_openvla"
python - <<'PY'
import dlimp
if not hasattr(dlimp, "DLataset"):
    raise SystemExit(f"Wrong dlimp installed at {dlimp.__file__}; expected dlimp_openvla with DLataset")
print(f"dlimp OK: {dlimp.__file__}")
PY

section "Flash Attention"
python -m pip install --quiet --no-build-isolation "${FLASH_ATTN_WHEEL_URL}"
python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
_ = flash_attn_func(q, k, v)
print(f"torch={torch.__version__} cuda={torch.version.cuda} flash_attn={flash_attn.__version__}")
PY

section "Clone/install SAFE-adapted OpenVLA"
if [[ ! -d "${OPENVLA_DIR}/.git" ]]; then
    git clone --depth 1 https://github.com/vla-safe/openvla.git "${OPENVLA_DIR}"
else
    log "Using existing ${OPENVLA_DIR}"
fi
python "${SCRIPT_DIR}/patch_openvla_layer_capture.py" "${OPENVLA_DIR}"
# Do not pip-install OpenVLA here. Its pyproject pins tensorflow==2.15.0,
# which is unavailable on current Colab Python 3.12. The runpy wrapper adds
# this checkout to sys.path before launching the eval script, which is enough
# for the repo-local `experiments` and `prismatic` imports.

section "Clone/install LIBERO"
if [[ ! -d "${LIBERO_DIR}/.git" ]]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
else
    log "Using existing ${LIBERO_DIR}"
fi
touch "${LIBERO_DIR}/libero/__init__.py"
mkdir -p "${LIBERO_DIR}/libero/datasets"
python -m pip install --quiet --no-deps -e "${LIBERO_DIR}"

section "Install SAFE detector package"
if [[ ! -d "${SAFE_DIR}" ]]; then
    git clone --depth 1 https://github.com/vla-safe/SAFE.git "${SAFE_DIR}"
fi
python -m pip install --quiet -e "${SAFE_DIR}"

section "Persist runtime env vars"
mkdir -p "${ROLLOUT_ROOT}" "${WORKSPACE}/robosuite_assets" "${WORKSPACE}/.libero"
sudo tee /etc/profile.d/safe_openvla_colab.sh >/dev/null <<EOF
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOSUITE_DEFAULT_ASSET_PATH="${WORKSPACE}/robosuite_assets"
export LIBERO_CONFIG_PATH="${WORKSPACE}/.libero"
export SAFE_OPENVLA_DIR="${OPENVLA_DIR}"
export SAFE_OPENVLA_ROLLOUT_ROOT="${ROLLOUT_ROOT}"
EOF
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOSUITE_DEFAULT_ASSET_PATH="${WORKSPACE}/robosuite_assets"
export LIBERO_CONFIG_PATH="${WORKSPACE}/.libero"
export SAFE_OPENVLA_DIR="${OPENVLA_DIR}"
export SAFE_OPENVLA_ROLLOUT_ROOT="${ROLLOUT_ROOT}"

section "Smoke imports"
python - <<'PY'
import os
import sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, os.environ["SAFE_OPENVLA_DIR"])
import torch
# from libero.libero import benchmark
from transformers import AutoProcessor
import failure_prob
import prismatic
print("CUDA available:", torch.cuda.is_available())
print("LIBERO suites include:", sorted(benchmark.get_benchmark_dict().keys())[:5], "...")
print("SAFE import OK:", failure_prob.__file__)
print("OpenVLA prismatic import OK:", prismatic.__file__)
print("Transformers import OK:", AutoProcessor.__name__)
PY

section "Done"
log "Run a smoke rollout:"
log "python ${SCRIPT_DIR}/run_openvla_libero_eval.py --suite 10 --num_trials_per_task 1 --checkpoint openvla/openvla-7b-finetuned-libero-10"
log "Rollouts will be under: ${ROLLOUT_ROOT}"
