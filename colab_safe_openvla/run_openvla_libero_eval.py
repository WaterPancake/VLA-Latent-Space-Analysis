#!/usr/bin/env python3
"""Run SAFE's adapted OpenVLA LIBERO eval script via runpy.

The target script uses draccus CLI parsing. This wrapper builds the expected
``sys.argv`` and executes the script with ``runpy.run_path`` so Colab notebooks
do not depend on fragile `%cd` state.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def str_bool(value: bool | str) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y"}:
        return "True"
    if value_lower in {"false", "0", "no", "n"}:
        return "False"
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch vla-safe/openvla LIBERO rollout eval via runpy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--openvla_repo",
        default=os.environ.get("SAFE_OPENVLA_DIR", "/content/vla-safe-openvla"),
        help="Path to the SAFE-adapted OpenVLA checkout.",
    )
    parser.add_argument(
        "--rollout_root",
        default=os.environ.get("SAFE_OPENVLA_ROLLOUT_ROOT", ""),
        help="Expected rollout root. Defaults to <openvla_repo>/rollouts.",
    )
    parser.add_argument("--suite", default="10", help="LIBERO suite suffix, e.g. 10, 90, spatial, object, goal.")
    parser.add_argument(
        "--checkpoint",
        default="openvla/openvla-7b-finetuned-libero-10",
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--task_start_index", type=int, default=None)
    parser.add_argument("--task_end_index", type=int, default=None)
    parser.add_argument("--resume", type=str_bool, default="False")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--hidden_layer_stride", type=int, default=4)
    parser.add_argument("--hidden_layer_start", type=int, default=4)
    parser.add_argument("--include_final_hidden_layer", type=str_bool, default="True")
    parser.add_argument("--center_crop", type=str_bool, default="True")
    parser.add_argument("--output_hidden_states", type=str_bool, default="True")
    parser.add_argument("--output_attentions", type=str_bool, default="False")
    parser.add_argument("--output_logits", type=str_bool, default="True")
    parser.add_argument("--load_in_8bit", type=str_bool, default="False")
    parser.add_argument("--load_in_4bit", type=str_bool, default="False")
    parser.add_argument("--use_wandb", type=str_bool, default="False")
    parser.add_argument("--save_logs", type=str_bool, default="True")
    parser.add_argument(
        "--run_id_note",
        default="single-foward",
        help="Output subfolder name. SAFE's config intentionally expects the paper repo typo: single-foward.",
    )
    parser.add_argument("--wandb_dir", default=None)
    parser.add_argument("--wandb_project", default="safe-openvla-colab")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--attn_avg_token", type=str_bool, default="True")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()

    repo = Path(args.openvla_repo).expanduser().resolve()
    script = repo / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    if not script.exists():
        raise FileNotFoundError(f"Could not find SAFE OpenVLA eval script at {script}")

    rollout_root = Path(args.rollout_root).expanduser().resolve() if args.rollout_root else repo / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("SAFE_OPENVLA_ROLLOUT_ROOT", str(rollout_root))

    suite_name = args.suite if args.suite.startswith("libero_") else f"libero_{args.suite}"
    eval_argv = [
        str(script),
        "--model_family",
        "openvla",
        "--pretrained_checkpoint",
        args.checkpoint,
        "--task_suite_name",
        suite_name,
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--resume",
        str_bool(args.resume),
        "--attn_implementation",
        args.attn_implementation,
        "--hidden_layer_stride",
        str(args.hidden_layer_stride),
        "--hidden_layer_start",
        str(args.hidden_layer_start),
        "--include_final_hidden_layer",
        str_bool(args.include_final_hidden_layer),
        "--center_crop",
        str_bool(args.center_crop),
        "--output_hidden_states",
        str_bool(args.output_hidden_states),
        "--output_attentions",
        str_bool(args.output_attentions),
        "--output_logits",
        str_bool(args.output_logits),
        "--n_samples",
        str(args.n_samples),
        "--load_in_8bit",
        str_bool(args.load_in_8bit),
        "--load_in_4bit",
        str_bool(args.load_in_4bit),
        "--use_wandb",
        str_bool(args.use_wandb),
        "--save_logs",
        str_bool(args.save_logs),
        "--run_id_note",
        args.run_id_note,
        "--save_root",
        str(rollout_root),
        "--wandb_project",
        args.wandb_project,
        "--attn_avg_token",
        str_bool(args.attn_avg_token),
        "--seed",
        str(args.seed),
    ]
    if args.task_start_index is not None:
        eval_argv.extend(["--task_start_index", str(args.task_start_index)])
    if args.task_end_index is not None:
        eval_argv.extend(["--task_end_index", str(args.task_end_index)])
    if args.wandb_entity:
        eval_argv.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_dir:
        eval_argv.extend(["--wandb_dir", args.wandb_dir])
    eval_argv.extend(passthrough)

    sys.path.insert(0, str(repo))
    sys.argv = eval_argv
    os.chdir(repo)
    print("[runpy-openvla] Running:", " ".join(eval_argv), flush=True)
    print("[runpy-openvla] Rollout root:", rollout_root, flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
