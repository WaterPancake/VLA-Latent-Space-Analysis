# Colab Reproduction Harness: SAFE + Original OpenVLA

This folder is for the first-pass reproduction path:

1. Generate LIBERO rollouts with SAFE's adapted original OpenVLA fork.
2. Save hidden-state features during rollout.
3. Point the public `SAFE/` detector training code at those rollout files.

SAFE's public repo does not generate rollouts itself. Its README points to
`vla-safe/openvla` for OpenVLA-on-LIBERO rollouts; that fork modifies
`experiments/robot/libero/run_libero_eval.py` to save OpenVLA internal features.

## Files

- `setup.sh`: one-shot Google Colab setup for an L4 runtime.
- `requirements-colab.in`: pinned top-level Python dependencies.
- `run_openvla_libero_eval.py`: wrapper that runs SAFE's adapted OpenVLA LIBERO
  eval script through Python `runpy`, so a Colab notebook can launch eval
  without relying on `%cd` state.

## Colab Quick Start

Use `Runtime > Change runtime type > GPU > L4`.

```python
!git clone https://github.com/<YOUR_ORG_OR_USER>/VLA.git /content/VLA
!bash /content/VLA/colab_safe_openvla/setup.sh
```

If the OpenVLA checkpoint requires Hugging Face auth, expose `HF_TOKEN` before
running eval:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Smoke-test with one rollout per task before attempting the paper-sized run:

```python
!python /content/VLA/colab_safe_openvla/run_openvla_libero_eval.py \
  --suite 10 \
  --num_trials_per_task 1 \
  --checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --center_crop True \
  --output_hidden_states True \
  --hidden_layer_stride 4 \
  --use_wandb False \
  --save_logs True
```

Paper-style LIBERO-10 generation is 50 rollouts per task:

```python
!python /content/VLA/colab_safe_openvla/run_openvla_libero_eval.py \
  --suite 10 \
  --num_trials_per_task 50 \
  --checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --center_crop True \
  --output_hidden_states True \
  --hidden_layer_stride 4 \
  --run_id_note single-foward \
  --use_wandb False \
  --save_logs True
```

Multi-sample uncertainty rollouts are much heavier:

```python
!python /content/VLA/colab_safe_openvla/run_openvla_libero_eval.py \
  --suite 10 \
  --num_trials_per_task 50 \
  --checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --center_crop True \
  --output_hidden_states True \
  --hidden_layer_stride 4 \
  --n_samples 10 \
  --run_id_note multi-forward-10 \
  --use_wandb False \
  --save_logs True
```

After rollouts are generated, SAFE can be trained from the same Colab runtime:

```python
%cd /content/VLA/SAFE
!python -m failure_prob.train \
  --multirun \
  train.wandb_group_name=openvla_libero_repro \
  dataset=openvla_libero_10 \
  dataset.data_path_prefix=/content/vla-safe-openvla/rollouts \
  dataset.token_idx_rel=mean,0.0,1.0 \
  dataset.load_to_cuda=False \
  model=lstm \
  model.batch_size=64 \
  model.lr=1e-4 \
  model.lambda_reg=1e-2 \
  train.seed=0 \
  train.exp_suffix=lstm_smoke
```

For the full sweep, use `SAFE/scripts/batch_training/submit_openvla_libero.bash`
after setting:

```bash
export SAFE_OPENVLA_ROLLOUT_ROOT=/content/vla-safe-openvla/rollouts
```

Note the spelling: SAFE's `openvla_libero_10` config expects
`single-foward/libero_10/`, matching the adapted rollout repo's README typo.

## Hidden Layer Capture

`setup.sh` patches `vla-safe/openvla` so each rollout pickle keeps the original
SAFE-compatible field and adds a selected-layer analysis field:

- `hidden_states`: final layer only, shape `(T, action_tokens, hidden_dim)`.
- `hidden_states_layers`: selected layers, shape
  `(T, action_tokens, selected_layers, hidden_dim)`.
- `hidden_layer_indices`: hidden-state tuple indices that were saved.

With the default OpenVLA Llama backbone, `--hidden_layer_stride 4
--hidden_layer_start 4` captures layers `4, 8, 12, 16, 20, 24, 28, 32`.
The final layer is still included when `--include_final_hidden_layer True`.

## Expected Challenges on Colab L4

- Original OpenVLA is a 7B Llama-derived model. L4 has 24 GB VRAM, so
  single-sample bf16 inference may work, but `n_samples=10` can OOM or become
  too slow. Start with `--num_trials_per_task 1` and `--n_samples 1`.
- Flash Attention wheels are sensitive to the Torch, CUDA, Python, and CXX ABI
  combination. This harness pins Torch 2.4.0 CUDA 12.4 and installs a matching
  prebuilt Flash Attention wheel for Colab Python.
- LIBERO/robosuite rendering on headless Colab is fragile. `setup.sh` forces
  `MUJOCO_GL=osmesa`, trading render speed for fewer CUDA/EGL deadlocks.
- Full reproduction is rollout-expensive: LIBERO-10 at 50 trials/task is 500
  episodes before SAFE training starts.
- OpenVLA checkpoints may require Hugging Face license acceptance or an
  authenticated `HF_TOKEN`, depending on the exact checkpoint and account state.

## Later: Adapting the Pipeline to OpenVLA-Mini

The rollout adapter needs to change at the model boundary, not in SAFE's
detector trainer:

- Load miniVLA through the native `prismatic` loader instead of HF
  `AutoModelForVision2Seq`.
- Route `model_family=prismatic` through `get_model()` and `get_action()`.
- Forward generation kwargs such as `return_dict_in_generate`,
  `output_hidden_states`, `output_logits`, and `num_return_sequences`.
- Make `predict_action()` accept Hugging Face `Generate*Output` objects, decode
  actions from `.sequences`, and return `(actions, generated_outputs)` when
  feature extraction is enabled.
- Save miniVLA hidden states in a SAFE-readable rollout layout, or add a SAFE
  dataset loader for the miniVLA pickle layout.
- Account for hidden dimension changes. Original OpenVLA Llama features are not
  the same dimensionality as miniVLA Qwen features, so SAFE heads must be
  retrained or given a projection layer.
