#!/usr/bin/env python3
"""View a policy trained by ``train_anymal.py`` in the Newton viewer.

With no checkpoint argument, the newest checkpoint under
``logs/rsl_rl/anymal_gravel`` is selected automatically::

    uv run python scripts/view_anymal_policy.py
    uv run python scripts/view_anymal_policy.py --checkpoint path/to/model_1500.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from train_anymal import (
    DEFAULT_GRAVEL_DEPTH,
    DEFAULT_GRAVEL_LENGTH,
    DEFAULT_GRAVEL_WIDTH,
    DEFAULT_VOXEL_SIZE,
    AnymalGravelEnv,
    make_agent_cfg,
    make_env_cfg,
    resolve_pretrained_policy,
)


DEFAULT_LOG_ROOT = Path("logs/rsl_rl/anymal_gravel")


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(log_root: Path) -> Path:
    """Find the newest run, then its highest-numbered checkpoint."""
    root = log_root.expanduser().resolve()
    candidates = [path for path in root.glob("*/model_*.pt") if path.is_file()]
    candidates.extend(path for path in root.glob("*/model_interrupt.pt") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found below {root}. Train one first or pass --checkpoint explicitly."
        )
    return max(
        candidates,
        key=lambda path: (path.parent.stat().st_mtime, path.stat().st_mtime, _checkpoint_iteration(path)),
    )


def _load_task_args(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "task_args.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def _saved_or_cli(args: argparse.Namespace, saved: dict[str, Any], name: str, fallback: Any) -> Any:
    cli_value = getattr(args, name)
    if cli_value is not None:
        return cli_value
    return saved.get(name, fallback)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="RSL-RL model checkpoint.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Play Newton's rigid-floor policy directly instead of an RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--pretrained-policy",
        type=Path,
        default=None,
        help="Override the rigid-floor TorchScript policy path.",
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root used for auto-selection.")
    parser.add_argument("--device", default="cuda:0", help="Simulation and inference device.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after N policy steps; 0 runs until close.")
    parser.add_argument(
        "--episodes", type=int, default=0, help="Stop after N completed episodes; 0 runs until close."
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (useful for a playback smoke test).")
    parser.add_argument("--stop-on-success", action="store_true", help="Exit after the first successful stop.")
    # None means use the values saved beside the checkpoint, preserving training geometry.
    parser.add_argument("--gravel-length", type=float, default=None, help="Override pavement length [m].")
    parser.add_argument("--gravel-width", type=float, default=None, help="Override pavement width [m].")
    parser.add_argument("--gravel-depth", type=float, default=None, help="Override gravel depth [m].")
    parser.add_argument("--voxel-size", type=float, default=None, help="Override MPM voxel size [m].")
    parser.add_argument("--particles-per-cell", type=float, default=None, help="Override MPM particle density.")
    parser.add_argument("--grid-type", choices=("sparse", "dense", "fixed"), default=None)
    parser.add_argument("--dt", type=float, default=None, help="Override physics time step [s].")
    parser.add_argument("--decimation", type=int, default=None, help="Override physics steps per policy step.")
    parser.add_argument("--episode-length", type=float, default=None, help="Override episode duration [s].")
    parser.add_argument("--max-speed", type=float, default=None, help="Override cruising speed [m/s].")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.pretrained and args.checkpoint is not None:
        raise ValueError("--pretrained and --checkpoint are mutually exclusive.")
    if args.pretrained_policy is not None and not args.pretrained:
        raise ValueError("--pretrained-policy requires --pretrained.")
    checkpoint = None
    saved: dict[str, Any] = {}
    if not args.pretrained:
        checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint is not None
            else find_latest_checkpoint(args.log_root)
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        saved = _load_task_args(checkpoint)
    env_cfg = make_env_cfg(
        num_envs=1,
        device=args.device,
        gravel_length=float(_saved_or_cli(args, saved, "gravel_length", DEFAULT_GRAVEL_LENGTH)),
        gravel_width=float(_saved_or_cli(args, saved, "gravel_width", DEFAULT_GRAVEL_WIDTH)),
        gravel_depth=float(_saved_or_cli(args, saved, "gravel_depth", DEFAULT_GRAVEL_DEPTH)),
        voxel_size=float(_saved_or_cli(args, saved, "voxel_size", DEFAULT_VOXEL_SIZE)),
        particles_per_cell=float(_saved_or_cli(args, saved, "particles_per_cell", 3.0)),
        grid_type=str(_saved_or_cli(args, saved, "grid_type", "sparse")),
        dt=float(_saved_or_cli(args, saved, "dt", 1.0 / 200.0)),
        decimation=int(_saved_or_cli(args, saved, "decimation", 4)),
        max_forward_speed=float(_saved_or_cli(args, saved, "max_speed", 0.70)),
        episode_length_s=float(_saved_or_cli(args, saved, "episode_length", 12.0)),
        viewer=not args.no_viewer,
        reset_noise=False,
    )
    env_cfg.seed = int(saved.get("seed", 42))
    agent_cfg = make_agent_cfg(device=args.device)

    raw_env = AnymalGravelEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    if args.pretrained:
        policy_path = resolve_pretrained_policy(args.pretrained_policy)
        print(f"[INFO]: Loading Newton rigid-floor policy from {policy_path}")
        scripted_policy = torch.jit.load(str(policy_path), map_location=args.device).eval()

        def policy(obs):
            return scripted_policy(obs["policy"])
    else:
        assert checkpoint is not None
        print(f"[INFO]: Loading policy from {checkpoint}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    observations, _ = env.reset()

    step = 0
    episode_count = 0
    try:
        while (
            (args.steps <= 0 or step < args.steps)
            and (args.episodes <= 0 or episode_count < args.episodes)
        ):
            if raw_env.sim.visualizers and not any(
                visualizer.is_running() and not visualizer.is_closed for visualizer in raw_env.sim.visualizers
            ):
                break
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, extras = env.step(actions)
            step += 1

            log = extras.get("log", {})
            if dones.any() and log:
                episode_count += 1
                success_rate = float(log.get("Metrics/success_rate", 0.0))
                goal_error = float(log.get("Metrics/final_abs_goal_error", float("nan")))
                distance = float(log.get("Metrics/net_forward_distance", float("nan")))
                base_height = float(log.get("Metrics/final_base_height", float("nan")))
                upright = float(log.get("Metrics/final_upright", float("nan")))
                reasons = [
                    key.removeprefix("Termination/")
                    for key, value in log.items()
                    if key.startswith("Termination/") and float(value) > 0.5
                ]
                reason = "+".join(reasons) if reasons else "unknown"
                print(
                    f"[INFO]: Episode {episode_count} ended at step {step}: reason={reason}, "
                    f"success={success_rate > 0.5}, distance={distance:.3f} m, "
                    f"absolute goal error={goal_error:.3f} m, base height={base_height:.3f} m, "
                    f"upright={upright:.3f}"
                )
                if args.stop_on_success and success_rate > 0.5:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print(f"[INFO]: Playback stopped after {step} policy steps.")


if __name__ == "__main__":
    main()
