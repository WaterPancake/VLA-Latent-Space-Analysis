import h5py
import numpy as np
from pathlib import Path
from typing import Optional


COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4


# e.g. for layers_to_save: ["layer_24", "layer_1, "pre_logits"]
def create_feature_store(
    path: str,
    model_name: str,
    hidden_dim: int,
    num_layers: int,
    layers_to_save: list[str] = ["pre_logits"],
) -> h5py.File:
    outfile = h5py.File(path, "w")

    meta = outfile.create_group("metadata")
    meta.attrs["model_name"] = model_name
    meta.attrs["hidden_dim"] = hidden_dim
    meta.attrs["num_layers"] = num_layers
    meta.attrs["layers_saved"] = ["layer_"]

    return outfile


def save_rollout(
    file: h5py.File,
    task_id: int,
    rollout_id: int,
    instruction: str,
    scene: str,
    success: bool,
    hidden_states: dict[str, np.ndarray],
    actions: Optional[np.ndarray] = None,
    seed: int = 0,
):
    task_key = f"task_{task_id}"

    if task_key not in file:
        task_grp = file.create_group(task_key)
        task_grp.attrs["task_id"] = task_id
        task_grp.attrs["instruction"] = instruction
        task_grp.attrs["scene"] = scene

    else:
        task_grp = file[task_key]

    # rollout groups
    rollout_key = f"rollout_{rollout_id}"

    rollout_grp = task_grp.create_group(rollout_key)
    rollout_grp.attrs["success"] = success
    rollout_grp.attrs["seed"] = seed

    # saving hidden state and action
    for layer_name, features in hidden_states.items():
        rollout_grp.create_dataset(
            layer_name,
            data=features.astype(np.float32),
            compression=COMPRESSION,
            compression_opts=COMPRESSION_LEVEL if COMPRESSION == "gzip" else None,
        )

    if actions is not None:
        rollout_grp.create_dataset(
            "actions",
            data=actions.astype(np.float32),
            compression=COMPRESSION,
            compression_opts=COMPRESSION_LEVEL if COMPRESSION == "gzip" else None,
        )
