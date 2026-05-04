#!/bin/bash
# One-shot setup script for the RunPod PyTorch 2.4 + CUDA 12.4 + Py3.11 template.
# Run this ONCE in /workspace on a fresh pod after attaching network volume.
# Idempotent — safe to re-run.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "[*] Sanity check: torch + CUDA"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.is_available()}')"

# ---- 1. uv ----
echo "[*] Installing uv"
pip install --quiet uv

# ---- 2. System libs MuJoCo + EGL need ----
# RunPod's torch image is missing these. EGL rendering will silently fall back
# to CPU OSMesa or just crash without them.
echo "[*] Installing system libs for headless EGL rendering"
apt-get update -qq
apt-get install -y --no-install-recommends \
    libegl1 libegl-dev libgles2-mesa-dev \
    libglew-dev libglfw3 libosmesa6-dev \
    patchelf ffmpeg \
    >/dev/null
    

# ---- 3. Python deps via uv ----
echo "[*] Writing rollout_requirements.in"
cat > rollout_requirements.in <<'REQEOF'
# Model-side deps (mirror the colab conversion env, minus torch which is preinstalled)
transformers==4.40.1
tokenizers==0.19.1
timm==0.9.10
huggingface_hub
draccus==0.8.0
accelerate>=0.25.0
sentencepiece==0.1.99
protobuf
Pillow>=12.1.0
numpy>=2.3.5

# Rollout / viz
imageio
imageio-ffmpeg
matplotlib
jupyterlab
ipywidgets
tqdm

# Sim stack — robosuite 1.4.1 is what LIBERO is built against; mujoco>=3 has
# the modern Python bindings (do NOT use mujoco-py, it's dead).
robosuite==1.4.1
mujoco>=3.1.0
hydra-core
easydict
bddl
cloudpickle
REQEOF

echo "[*] Compiling lock"
uv pip compile rollout_requirements.in -o rollout_requirements.lock --generate-hashes

echo "[*] Installing into system Python (preserves preinstalled torch)"
uv pip install -r rollout_requirements.lock --system --require-hashes \
    --reinstall-package numpy \
    --reinstall-package Pillow

# ---- 4. Editable installs of openvla-mini and LIBERO ----
echo "[*] Cloning openvla-mini (the conversion-export branch)"
if [ ! -d openvla-mini ]; then
    git clone --depth 1 --branch feat/support-hf-checkpoint-export \
        https://github.com/360ZMEM/openvla-mini.git
fi
uv pip install -e ./openvla-mini --no-deps --system

# dlimp + tensorflow-graphics (eager imports prismatic drags in)
DLIMP_REPO="https://github.com/moojink/dlimp_openvla.git"
DLIMP_SHA="040105d256bd28866cc6620621a3d5f7b6b91b46"
uv pip install --no-deps --system "dlimp @ git+${DLIMP_REPO}@${DLIMP_SHA}"
uv pip install --no-deps --system tensorflow-graphics
# tensorflow_graphics needs TF at import time; install CPU-only TF
uv pip install --system "tensorflow-cpu"

echo "[*] Cloning LIBERO"
if [ ! -d LIBERO ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
uv pip install -e ./LIBERO --no-deps --system

# ---- 5. Sanity check the import chain ----
echo "[*] Verifying everything imports"
MUJOCO_GL=egl python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
import torch, transformers, timm, huggingface_hub
import mujoco, robosuite
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import dlimp, tensorflow_graphics
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
print('[*] all imports OK')
print(f'    torch={torch.__version__}')
print(f'    mujoco={mujoco.__version__}')
print(f'    robosuite={robosuite.__version__}')
print(f'    GPU={torch.cuda.get_device_name() if torch.cuda.is_available() else \"NONE\"}')
"

echo "Jobs done :^)"

