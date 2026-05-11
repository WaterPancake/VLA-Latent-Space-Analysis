#!/usr/bin/env python3
"""Patch vla-safe/openvla to save every-N-layer hidden states.

SAFE's adapted OpenVLA rollout script saves only the final transformer layer in
the existing ``hidden_states`` field. This patch keeps that compatibility field
and adds ``hidden_states_layers`` with shape:

    (timesteps, action_tokens, selected_layers, hidden_dim)

For Llama-2-7B OpenVLA, default selected layers are transformer layers
4, 8, 12, 16, 20, 24, 28, and 32.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPER = '''

def _get_selected_hidden_layer_indices(num_hidden_state_entries: int, start: int, stride: int, include_final: bool) -> list[int]:
    """Return hidden-state tuple indices for transformer layers.

    Hugging Face decoder hidden states normally contain the embedding output at
    index 0, then transformer layer outputs at indices 1..N. Therefore tuple
    index 4 corresponds to transformer layer 4 for OpenVLA's Llama backbone.
    """
    if stride <= 0:
        return []
    selected = list(range(max(1, start), num_hidden_state_entries, stride))
    final_idx = num_hidden_state_entries - 1
    if include_final and final_idx not in selected:
        selected.append(final_idx)
    return selected


def _extract_selected_hidden_states(generated_outputs, cfg):
    """Extract final-layer and selected-layer hidden states for action tokens."""
    all_hidden_states = generated_outputs["hidden_states"]
    if not all_hidden_states:
        return None, None, []

    selected_indices = _get_selected_hidden_layer_indices(
        len(all_hidden_states[0]),
        cfg.hidden_layer_start,
        cfg.hidden_layer_stride,
        cfg.include_final_hidden_layer,
    )

    final_layer_by_token = []
    selected_layers_by_token = []
    for step_hidden_states in all_hidden_states:
        final_layer_by_token.append(step_hidden_states[-1][0, -1, :])
        if selected_indices:
            selected_layers_by_token.append(
                torch.stack([step_hidden_states[layer_idx][0, -1, :] for layer_idx in selected_indices], dim=0)
            )

    final_hidden_states = torch.stack(final_layer_by_token, dim=0)
    selected_hidden_states = (
        torch.stack(selected_layers_by_token, dim=0)
        if selected_layers_by_token
        else None
    )
    return final_hidden_states, selected_hidden_states, selected_indices
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match for {label}, found {count}")
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    text_new, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected one regex match for {label}, found {count}")
    return text_new


def patch_file(path: Path) -> None:
    text = path.read_text()
    if "_extract_selected_hidden_states" in text and "hidden_states_layers" in text:
        print(f"Already patched: {path}")
        return

    text = replace_once(
        text,
        "\n@dataclass\nclass GenerateConfig:",
        HELPER + "\n\n@dataclass\nclass GenerateConfig:",
        "helper insertion",
    )

    text = regex_replace_once(
        text,
        r"(\n\s*output_hidden_states:\s*bool\s*=\s*False\s*# Whether to output hidden states from the model)",
        (
            r"\1\n"
            "    hidden_layer_stride: int = 4               # Save every N transformer layers in hidden_states_layers\n"
            "    hidden_layer_start: int = 4                # First transformer layer to save when selecting earlier layers\n"
            "    include_final_hidden_layer: bool = True    # Always include the final layer even if stride misses it"
        ),
        "dataclass hidden-layer config insertion",
    )

    text = replace_once(
        text,
        "        hidden_states_episode = []\n",
        "        hidden_states_episode = []\n        hidden_states_layers_episode = []\n        hidden_layer_indices = None\n",
        "episode hidden-state buffers",
    )

    text = regex_replace_once(
        text,
        r"""                    if cfg\.output_hidden_states:\n\s*# If output_hidden_states is True, "hidden_states" exists in generated_outputs\n\s*# generated_outputs\['hidden_states'\] is a tuple of length 7 \(number of generated tokens\)\n\s*# Within which each element is a tuple of length 33 \(number of layers\)\n\s*# Each element in the inner tuple is a tensor of shape \(bs=1, 1 or N, 4096\)\n\s*# The final hidden states before decoding is generated_outputs\['hidden_states'\]\[0\]\[-1\]\[0, -1, :\]\n\s*all_hidden_states = generated_outputs\['hidden_states'\]\n\s*hidden_states_last_layer = \[s\[-1\]\[0, -1, :\] for s in all_hidden_states\]\n\s*hidden_states_last_layer = torch\.stack\(hidden_states_last_layer, dim=0\) # \(7, 4096\)\n\s*# Save the hidden states for further analysis\n\s*hidden_states_episode\.append\(hidden_states_last_layer\.detach\(\)\.cpu\(\)\)""",
        """                    if cfg.output_hidden_states:
                        hidden_states_last_layer, hidden_states_selected_layers, hidden_layer_indices = (
                            _extract_selected_hidden_states(generated_outputs, cfg)
                        )
                        # Keep the original SAFE-compatible field as final-layer-only: (action_tokens, hidden_dim).
                        hidden_states_episode.append(hidden_states_last_layer.detach().cpu())
                        # Add an analysis field with selected earlier layers: (action_tokens, selected_layers, hidden_dim).
                        if hidden_states_selected_layers is not None:
                            hidden_states_layers_episode.append(hidden_states_selected_layers.detach().cpu())""",
        "per-step hidden-state extraction",
    )

    text = replace_once(
        text,
        """                    "mp4_path": str(mp4_path),
                }
                pickle.dump(save_dict, open(hidden_states_path, "wb"))""",
        """                    "mp4_path": str(mp4_path),
                }
                if hidden_states_layers_episode:
                    save_dict["hidden_states_layers"] = torch.stack(hidden_states_layers_episode, dim=0)
                    save_dict["hidden_layer_indices"] = hidden_layer_indices
                    save_dict["hidden_layer_stride"] = cfg.hidden_layer_stride
                    save_dict["hidden_layer_start"] = cfg.hidden_layer_start
                    save_dict["include_final_hidden_layer"] = cfg.include_final_hidden_layer
                pickle.dump(save_dict, open(hidden_states_path, "wb"))""",
        "save_dict selected-layer fields",
    )

    path.write_text(text)
    print(f"Patched selected-layer hidden-state capture in: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("openvla_repo", type=Path)
    args = parser.parse_args()
    patch_file(args.openvla_repo / "experiments" / "robot" / "libero" / "run_libero_eval.py")


if __name__ == "__main__":
    main()

