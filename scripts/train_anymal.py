#!/usr/bin/env python3
"""Fine-tune Newton's pretrained ANYmal-C policy on an MPM gravel pavement.

This is a kit-less Isaac Lab direct-workflow task using the Newton backend and
RSL-RL PPO.  The gravel/robot interaction reuses ``gravel_coupling.py``.

Run from the repository root::

    uv run python scripts/train_anymal.py

Checkpoints are written below ``logs/rsl_rl/anymal_gravel/<timestamp>``.  Pass a
checkpoint to ``--resume`` to continue an interrupted run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.io.yaml import dump_yaml
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_newton.assets.mpm_object import MPMObject, MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_visualizers.newton import NewtonVisualizerCfg
from rsl_rl.runners import OnPolicyRunner

from gravel_coupling import gravel_material, gravel_physics_cfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


PARTICLE_COLOR = (0.55, 0.50, 0.45)
FLOOR_THICKNESS = 0.10
BORDER_THICKNESS = 0.05
SPAWN_CLEARANCE_X = 0.60
DEFAULT_GRAVEL_LENGTH = 3.0
DEFAULT_GRAVEL_WIDTH = 0.9
DEFAULT_GRAVEL_DEPTH = 0.05
DEFAULT_VOXEL_SIZE = 0.03
PRETRAINED_POLICY_FILENAME = "anymal_walking_policy_physx.pt"
PRETRAINED_JOINT_ORDER = (
    "LF_HAA",
    "LH_HAA",
    "RF_HAA",
    "RH_HAA",
    "LF_HFE",
    "LH_HFE",
    "RF_HFE",
    "RH_HFE",
    "LF_KFE",
    "LH_KFE",
    "RF_KFE",
    "RH_KFE",
)


@configclass
class AnymalGravelEnvCfg(DirectRLEnvCfg):
    """Direct RL environment settings for the point-to-point gravel task."""

    episode_length_s = 12.0
    decimation = 4
    action_space = 12
    observation_space = 48
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics=gravel_physics_cfg(voxel_size=DEFAULT_VOXEL_SIZE),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=DEFAULT_GRAVEL_LENGTH + 1.0,
        replicate_physics=True,
    )

    action_scale = 0.50
    max_forward_speed = 0.70
    stop_distance = 0.65
    goal_tolerance = 0.12
    goal_lateral_tolerance = 0.18
    success_speed = 0.12
    success_angular_speed = 0.25
    success_hold_s = 0.40
    gravel_length = DEFAULT_GRAVEL_LENGTH
    gravel_width = DEFAULT_GRAVEL_WIDTH
    gravel_depth = DEFAULT_GRAVEL_DEPTH
    spawn_x = -0.5 * DEFAULT_GRAVEL_LENGTH + SPAWN_CLEARANCE_X
    goal_x = 0.5 * DEFAULT_GRAVEL_LENGTH - 0.40
    reset_joint_noise = 0.05
    reset_position_noise = 0.03


@configclass
class AnymalGravelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO settings used by both training and policy playback."""

    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 50
    experiment_name = "anymal_gravel"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.25),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """Create one cloned static Newton collider."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                static_friction=0.9,
                dynamic_friction=0.9,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def make_env_cfg(
    *,
    num_envs: int,
    device: str,
    gravel_length: float = DEFAULT_GRAVEL_LENGTH,
    gravel_width: float = DEFAULT_GRAVEL_WIDTH,
    gravel_depth: float = DEFAULT_GRAVEL_DEPTH,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    particles_per_cell: float = 3.0,
    grid_type: str = "sparse",
    dt: float = 1.0 / 200.0,
    decimation: int = 4,
    max_forward_speed: float = 0.70,
    episode_length_s: float = 12.0,
    viewer: bool = False,
    reset_noise: bool = True,
) -> AnymalGravelEnvCfg:
    """Build a complete scene/task config for training or playback."""
    if num_envs < 1:
        raise ValueError("num_envs must be at least one.")
    if gravel_length <= SPAWN_CLEARANCE_X + 0.9:
        raise ValueError("gravel_length is too short to contain separate start and stop regions.")
    if gravel_width <= 0.6:
        raise ValueError("gravel_width must be greater than 0.6 m for ANYmal-C.")
    if gravel_depth <= 0.0 or voxel_size <= 0.0:
        raise ValueError("gravel_depth and voxel_size must be positive.")

    cfg = AnymalGravelEnvCfg()
    cfg.decimation = decimation
    cfg.episode_length_s = episode_length_s
    cfg.gravel_length = gravel_length
    cfg.gravel_width = gravel_width
    cfg.gravel_depth = gravel_depth
    cfg.spawn_x = -0.5 * gravel_length + SPAWN_CLEARANCE_X
    cfg.goal_x = 0.5 * gravel_length - 0.40
    cfg.max_forward_speed = max_forward_speed
    cfg.reset_joint_noise = 0.05 if reset_noise else 0.0
    cfg.reset_position_noise = 0.03 if reset_noise else 0.0

    visualizer_cfgs = []
    if viewer:
        visualizer_cfgs = [
            NewtonVisualizerCfg(
                eye=(0.0, -1.35 * gravel_length, 1.8 + gravel_depth),
                lookat=(0.0, 0.0, 0.45 + gravel_depth),
                show_particles=True,
                particle_color=PARTICLE_COLOR,
            )
        ]
    cfg.sim = SimulationCfg(
        dt=dt,
        device=device,
        render_interval=decimation,
        physics=gravel_physics_cfg(
            voxel_size=voxel_size,
            grid_type=grid_type,
            num_substeps=1,
            rigid_bodies=r"/World/envs/env_.*/Robot",
            rigid_shapes=r"/World/envs/env_.*/Floor.*",
            proxy_bodies=r"/World/envs/env_.*/Robot/.*(SHANK|FOOT)",
            proxy_mode="lagged",
            rigid_substeps=1,
        ),
        visualizer_cfgs=visualizer_cfgs,
    )

    env_spacing = max(gravel_length + 1.0, gravel_width + 1.0)
    scene = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
    )

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        # Match the controller used to train Newton's rigid-floor policy.
        stiffness={".*": 150.0},
        damping={".*": 5.0},
        armature={".*": 0.06},
    )
    scene.robot = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={"legs": actuator_cfg},
        init_state=ANYMAL_C_CFG.init_state.replace(
            pos=(cfg.spawn_x, 0.0, ANYMAL_C_CFG.init_state.pos[2] + gravel_depth)
        ),
    )

    half_x, half_y = 0.5 * gravel_length, 0.5 * gravel_width
    scene.gravel = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Gravel",
        spawn=MPMGridCfg(
            lower=(-half_x, -half_y, 0.0),
            upper=(half_x, half_y, gravel_depth),
            voxel_size=voxel_size,
            particles_per_cell=particles_per_cell,
            jitter=0.25 * voxel_size,
            material=gravel_material(
                young_modulus=1.0e10,
                yield_stress=3.0e4,
                yield_pressure=3.0e4,
                viscosity=0.0,
                damping=0.02,
                friction=0.8,
            ),
            visual_color=PARTICLE_COLOR,
        ),
    )

    scene.floor = _static_cuboid(
        "{ENV_REGEX_NS}/Floor",
        (gravel_length + 0.4, gravel_width + 0.4, FLOOR_THICKNESS),
        (0.0, 0.0, -0.5 * FLOOR_THICKNESS),
        (0.25, 0.25, 0.28),
    )
    wall_z = 0.5 * gravel_depth
    scene.border_pos_x = _static_cuboid(
        "{ENV_REGEX_NS}/BorderPosX",
        (BORDER_THICKNESS, gravel_width + 2.0 * BORDER_THICKNESS, gravel_depth),
        (half_x + 0.5 * BORDER_THICKNESS, 0.0, wall_z),
        (0.35, 0.35, 0.38),
    )
    scene.border_neg_x = _static_cuboid(
        "{ENV_REGEX_NS}/BorderNegX",
        (BORDER_THICKNESS, gravel_width + 2.0 * BORDER_THICKNESS, gravel_depth),
        (-half_x - 0.5 * BORDER_THICKNESS, 0.0, wall_z),
        (0.35, 0.35, 0.38),
    )
    scene.border_pos_y = _static_cuboid(
        "{ENV_REGEX_NS}/BorderPosY",
        (gravel_length, BORDER_THICKNESS, gravel_depth),
        (0.0, half_y + 0.5 * BORDER_THICKNESS, wall_z),
        (0.35, 0.35, 0.38),
    )
    scene.border_neg_y = _static_cuboid(
        "{ENV_REGEX_NS}/BorderNegY",
        (gravel_length, BORDER_THICKNESS, gravel_depth),
        (0.0, -half_y - 0.5 * BORDER_THICKNESS, wall_z),
        (0.35, 0.35, 0.38),
    )

    # A thin, non-colliding green stripe makes the stopping point obvious in playback.
    scene.goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CuboidCfg(
            size=(0.035, gravel_width, 0.008),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cfg.goal_x, 0.0, gravel_depth + 0.004)),
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    cfg.scene = scene
    return cfg


class AnymalGravelEnv(DirectRLEnv):
    """ANYmal point-to-point task on a coupled rigid/MPM gravel scene."""

    cfg: AnymalGravelEnvCfg

    def __init__(self, cfg: AnymalGravelEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros((self.num_envs, action_dim), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_x = self._robot.data.root_pos_w.torch[:, 0].clone()
        joint_names = tuple(name.rsplit("/", 1)[-1] for name in self._robot.joint_names)
        missing = set(PRETRAINED_JOINT_ORDER) - set(joint_names)
        if missing:
            raise ValueError(f"ANYmal joint mapping is incomplete; missing {sorted(missing)}")
        self._policy_to_env = torch.tensor(
            [joint_names.index(name) for name in PRETRAINED_JOINT_ORDER],
            dtype=torch.long,
            device=self.device,
        )
        self._env_to_policy = torch.argsort(self._policy_to_env)
        self._termination_flags = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in (
                "success",
                "low_base",
                "excessive_tilt",
                "left_pavement",
                "overshot",
                "non_finite",
                "timeout",
            )
        }
        self._success_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._entered_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        reward_names = (
            "world_progress",
            "velocity_error",
            "orientation_error",
            "goal_stop",
            "lateral_error",
            "heading_error",
            "vertical_velocity",
            "angular_velocity",
            "torque",
            "joint_acceleration",
            "action_rate",
            "overshoot",
            "failure_penalty",
            "success_bonus",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in reward_names
        }

    def _setup_scene(self) -> None:
        self._robot: Articulation = self.scene.articulations["robot"]
        self._gravel: MPMObject = self.scene.deformable_objects["gravel"]

    def _goal_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return remaining X, lateral error, and the goal vector in the base frame."""
        root_pos = self._robot.data.root_pos_w.torch
        remaining_x = self.scene.env_origins[:, 0] + self.cfg.goal_x - root_pos[:, 0]
        lateral_error = root_pos[:, 1] - self.scene.env_origins[:, 1]
        goal_delta_w = torch.zeros_like(root_pos)
        goal_delta_w[:, 0] = remaining_x
        goal_delta_w[:, 1] = -lateral_error
        goal_b = quat_apply_inverse(self._robot.data.root_quat_w.torch, goal_delta_w)
        return remaining_x, lateral_error, goal_b

    def _target_speed(self, remaining_x: torch.Tensor) -> torch.Tensor:
        return self.cfg.max_forward_speed * torch.clamp(remaining_x / self.cfg.stop_distance, 0.0, 1.0)

    def _velocity_command(self) -> torch.Tensor:
        """Build the [forward, lateral, yaw-rate] command used by the rigid policy."""
        remaining_x, lateral_error, goal_b = self._goal_state()
        heading_error = torch.atan2(goal_b[:, 1], goal_b[:, 0].clamp_min(1.0e-6))
        return torch.stack(
            (
                self._target_speed(remaining_x),
                torch.clamp(-0.6 * lateral_error, -0.25, 0.25),
                torch.clamp(1.2 * heading_error, -0.50, 0.50),
            ),
            dim=-1,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)
        actions_env_order = self._actions[:, self._env_to_policy]
        self._processed_actions = (
            self._robot.data.default_joint_pos.torch + self.cfg.action_scale * actions_env_order
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos_delta = (
            self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch
        )[:, self._policy_to_env]
        joint_vel = self._robot.data.joint_vel.torch[:, self._policy_to_env]
        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b.torch,
                self._robot.data.root_ang_vel_b.torch,
                self._robot.data.projected_gravity_b.torch,
                self._velocity_command(),
                joint_pos_delta,
                joint_vel,
                self._actions,
            ),
            dim=-1,
        )
        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        remaining_x, lateral_error, goal_b = self._goal_state()
        target_speed = self._target_speed(remaining_x)
        lin_vel_w = self._robot.data.root_lin_vel_w.torch
        ang_vel_b = self._robot.data.root_ang_vel_b.torch
        projected_gravity = self._robot.data.projected_gravity_b.torch

        # This is a penalty, not a positive body-frame velocity reward: oscillating
        # in place can no longer harvest return without net world displacement.
        velocity_error = torch.square(lin_vel_w[:, 0] - target_speed) + torch.square(lin_vel_w[:, 1])

        current_x = self._robot.data.root_pos_w.torch[:, 0]
        delta_x = torch.clamp(current_x - self._previous_x, -0.05, 0.05)
        self._previous_x = current_x.clone()
        # This potential-based term telescopes to net world-frame displacement.
        # Crossing the two meters from start to goal earns about 24 reward.
        world_progress = 12.0 * delta_x

        upright = -projected_gravity[:, 2]
        orientation_error = torch.square(1.0 - torch.clamp(upright, -1.0, 1.0))
        in_goal_region = (torch.abs(remaining_x) < 0.25) & (
            torch.abs(lateral_error) < self.cfg.goal_lateral_tolerance
        )
        self._entered_goal |= in_goal_region
        total_speed_sq = torch.sum(torch.square(lin_vel_w), dim=1) + 0.2 * torch.sum(
            torch.square(ang_vel_b), dim=1
        )
        goal_stop = 2.0 * in_goal_region * torch.exp(-total_speed_sq / 0.05) * self.step_dt

        goal_norm = torch.linalg.norm(goal_b[:, :2], dim=1).clamp_min(0.1)
        failed = self.reset_terminated & ~self._success
        rewards = {
            "world_progress": world_progress,
            "velocity_error": -1.0 * velocity_error * self.step_dt,
            "orientation_error": -1.0 * orientation_error * self.step_dt,
            "goal_stop": goal_stop,
            "lateral_error": -0.6 * torch.square(lateral_error) * self.step_dt,
            "heading_error": -0.3 * torch.square(goal_b[:, 1] / goal_norm) * self.step_dt,
            "vertical_velocity": -1.5 * torch.square(lin_vel_w[:, 2]) * self.step_dt,
            "angular_velocity": -0.05 * torch.sum(torch.square(ang_vel_b[:, :2]), dim=1) * self.step_dt,
            "torque": -2.5e-5
            * torch.sum(torch.square(self._robot.data.applied_torque.torch), dim=1)
            * self.step_dt,
            "joint_acceleration": -2.5e-7
            * torch.sum(torch.square(self._robot.data.joint_acc.torch), dim=1)
            * self.step_dt,
            "action_rate": -0.01
            * torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            * self.step_dt,
            "overshoot": -6.0 * torch.square(torch.clamp(-remaining_x, min=0.0)) * self.step_dt,
            "failure_penalty": -10.0 * failed,
            "success_bonus": 20.0 * self._success,
        }
        for name, value in rewards.items():
            # A terminated world is reset immediately. Keep the terminal reward
            # finite if an unstable contact produced a non-finite rigid state.
            rewards[name] = torch.nan_to_num(value, nan=-10.0, posinf=-10.0, neginf=-10.0)
            self._episode_sums[name] += rewards[name]
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        remaining_x, lateral_error, _ = self._goal_state()
        lin_speed = torch.linalg.norm(self._robot.data.root_lin_vel_b.torch, dim=1)
        ang_speed = torch.linalg.norm(self._robot.data.root_ang_vel_b.torch, dim=1)
        upright = -self._robot.data.projected_gravity_b.torch[:, 2]

        stopped_at_goal = (
            (torch.abs(remaining_x) < self.cfg.goal_tolerance)
            & (torch.abs(lateral_error) < self.cfg.goal_lateral_tolerance)
            & (lin_speed < self.cfg.success_speed)
            & (ang_speed < self.cfg.success_angular_speed)
            & (upright > 0.8)
        )
        self._success_hold = torch.where(stopped_at_goal, self._success_hold + 1, 0)
        required_hold_steps = max(1, math.ceil(self.cfg.success_hold_s / self.step_dt))
        self._success = self._success_hold >= required_hold_steps

        root_pos = self._robot.data.root_pos_w.torch
        finite = (
            torch.isfinite(root_pos).all(dim=1)
            & torch.isfinite(self._robot.data.root_quat_w.torch).all(dim=1)
            & torch.isfinite(self._robot.data.root_lin_vel_b.torch).all(dim=1)
            & torch.isfinite(self._robot.data.root_ang_vel_b.torch).all(dim=1)
            & torch.isfinite(self._robot.data.joint_pos.torch).all(dim=1)
            & torch.isfinite(self._robot.data.joint_vel.torch).all(dim=1)
        )
        low_base = root_pos[:, 2] < self.cfg.gravel_depth + 0.30
        excessive_tilt = upright < 0.45
        fallen = low_base | excessive_tilt
        left_pavement = torch.abs(lateral_error) > 0.5 * self.cfg.gravel_width + 0.25
        overshot = remaining_x < -0.35
        terminated = self._success | fallen | left_pavement | overshot | ~finite
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._termination_flags = {
            "success": self._success.clone(),
            "low_base": low_base,
            "excessive_tilt": excessive_tilt,
            "left_pavement": left_pavement,
            "overshot": overshot,
            "non_finite": ~finite,
            "timeout": time_out,
        }
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if hasattr(self, "_episode_sums"):
            log: dict[str, float] = {}
            episode_seconds = self.episode_length_buf[ids].clamp_min(1).float() * self.step_dt
            for name, episodic_sum in self._episode_sums.items():
                # Log reward rates so longer-lived episodes do not look better merely
                # because they accumulated the same per-step term for more time.
                log[f"Episode_Reward/{name}"] = (episodic_sum[ids] / episode_seconds).mean().item()
                episodic_sum[ids] = 0.0

            log["Metrics/success_rate"] = self._success[ids].float().mean().item()
            remaining_x, lateral_error, _ = self._goal_state()
            current_x = self._robot.data.root_pos_w.torch[ids, 0]
            start_x = self.scene.env_origins[ids, 0] + self.cfg.spawn_x
            forward_distance = torch.nan_to_num(current_x - start_x)
            log["Metrics/net_forward_distance"] = forward_distance.mean().item()
            log["Metrics/mean_world_forward_velocity"] = (forward_distance / episode_seconds).mean().item()
            log["Metrics/goal_region_entry_rate"] = self._entered_goal[ids].float().mean().item()
            log["Metrics/failure_rate"] = (
                self.reset_terminated[ids] & ~self._success[ids]
            ).float().mean().item()
            log["Metrics/final_abs_goal_error"] = torch.abs(remaining_x[ids]).mean().item()
            log["Metrics/final_abs_lateral_error"] = torch.abs(lateral_error[ids]).mean().item()
            root_pos = self._robot.data.root_pos_w.torch[ids]
            upright = -self._robot.data.projected_gravity_b.torch[ids, 2]
            log["Metrics/final_base_height"] = torch.nan_to_num(root_pos[:, 2]).mean().item()
            log["Metrics/final_upright"] = torch.nan_to_num(upright).mean().item()
            for name, mask in self._termination_flags.items():
                log[f"Termination/{name}"] = mask[ids].float().mean().item()
            self.extras["log"] = log

        super()._reset_idx(ids)

        joint_pos = self._robot.data.default_joint_pos.torch[ids].clone()
        if self.cfg.reset_joint_noise > 0.0:
            joint_pos += torch.empty_like(joint_pos).uniform_(
                -self.cfg.reset_joint_noise, self.cfg.reset_joint_noise
            )
        joint_vel = torch.zeros_like(joint_pos)
        root_pose = self._robot.data.default_root_pose.torch[ids].clone()
        root_pose[:, :3] += self.scene.env_origins[ids]
        if self.cfg.reset_position_noise > 0.0:
            root_pose[:, :2] += torch.empty_like(root_pose[:, :2]).uniform_(
                -self.cfg.reset_position_noise, self.cfg.reset_position_noise
            )
        root_vel = torch.zeros_like(self._robot.data.default_root_vel.torch[ids])

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=ids)
        self._robot.set_joint_velocity_target_index(target=joint_vel, env_ids=ids)

        if hasattr(self, "_actions"):
            self._actions[ids] = 0.0
            self._previous_actions[ids] = 0.0
            self._previous_x[ids] = root_pose[:, 0]
            self._success_hold[ids] = 0
            self._success[ids] = False
            self._entered_goal[ids] = False


def make_agent_cfg(
    *,
    device: str,
    max_iterations: int = 10000,
    save_interval: int = 50,
    num_steps_per_env: int = 96,
):
    """Return the version-normalized PPO config used to create an RSL-RL runner."""
    cfg = AnymalGravelPPORunnerCfg()
    cfg.device = device
    cfg.max_iterations = max_iterations
    cfg.save_interval = save_interval
    cfg.num_steps_per_env = num_steps_per_env
    return handle_deprecated_rsl_rl_cfg(cfg, importlib.metadata.version("rsl-rl-lib"))


def resolve_pretrained_policy(path: Path | None = None) -> Path:
    """Resolve Newton's cached ANYmal-C rigid-floor TorchScript policy."""
    if path is not None:
        resolved = path.expanduser().resolve()
    else:
        import newton

        asset_root = Path(newton.utils.download_asset("anybotics_anymal_c"))
        resolved = asset_root / "rl_policies" / PRETRAINED_POLICY_FILENAME
    if not resolved.is_file():
        raise FileNotFoundError(f"Pretrained ANYmal policy does not exist: {resolved}")
    return resolved


def initialize_actor_from_pretrained(runner: OnPolicyRunner, policy_path: Path) -> float:
    """Copy the rigid-floor actor into PPO and return its maximum parity error."""
    source = torch.jit.load(str(policy_path), map_location=runner.device).eval()
    target_actor = runner.alg._raw_actor
    source_state = source.state_dict()
    target_linears = [module for module in target_actor.mlp if isinstance(module, torch.nn.Linear)]
    source_indices = (0, 2, 4, 6)
    if len(target_linears) != len(source_indices):
        raise ValueError(
            "The pretrained actor requires a 48-128-128-128-12 MLP; "
            f"target has {len(target_linears)} linear layers."
        )
    with torch.no_grad():
        for target, source_index in zip(target_linears, source_indices, strict=True):
            weight = source_state[f"actor.{source_index}.weight"].to(target.weight)
            bias = source_state[f"actor.{source_index}.bias"].to(target.bias)
            if target.weight.shape != weight.shape or target.bias.shape != bias.shape:
                raise ValueError(
                    f"Pretrained layer {source_index} has shapes {tuple(weight.shape)}, {tuple(bias.shape)}; "
                    f"target has {tuple(target.weight.shape)}, {tuple(target.bias.shape)}."
                )
            target.weight.copy_(weight)
            target.bias.copy_(bias)

        test_obs = torch.linspace(-1.0, 1.0, 48, device=runner.device).unsqueeze(0)
        source_actions = source(test_obs)
        target_actions = target_actor.mlp(test_obs)
        parity_error = torch.max(torch.abs(source_actions - target_actions)).item()
    if parity_error > 1.0e-5:
        raise RuntimeError(f"Pretrained actor copy failed parity check: max error {parity_error:.3e}")
    return parity_error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Simulation and learning device.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel gravel pavements.")
    parser.add_argument("--max-iterations", type=int, default=10000, help="PPO learning iterations.")
    parser.add_argument(
        "--num-steps-per-env",
        type=int,
        default=96,
        help="Policy steps collected from each environment per PPO update.",
    )
    parser.add_argument("--save-interval", type=int, default=50, help="Checkpoint interval in iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--run-name", default="", help="Optional suffix for the timestamped run directory.")
    parser.add_argument("--log-root", default="logs/rsl_rl/anymal_gravel", help="Root checkpoint directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--pretrained-policy",
        type=Path,
        default=None,
        help="Override the rigid-floor TorchScript policy used for a fresh run.",
    )
    parser.add_argument("--gravel-length", type=float, default=DEFAULT_GRAVEL_LENGTH, help="Pavement length [m].")
    parser.add_argument("--gravel-width", type=float, default=DEFAULT_GRAVEL_WIDTH, help="Pavement width [m].")
    parser.add_argument("--gravel-depth", type=float, default=DEFAULT_GRAVEL_DEPTH, help="Gravel depth [m].")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, help="MPM voxel size [m].")
    parser.add_argument("--particles-per-cell", type=float, default=3.0, help="MPM particle density.")
    parser.add_argument("--grid-type", choices=("sparse", "dense", "fixed"), default="sparse")
    parser.add_argument("--dt", type=float, default=1.0 / 200.0, help="Physics time step [s].")
    parser.add_argument("--decimation", type=int, default=4, help="Physics steps per policy step.")
    parser.add_argument("--episode-length", type=float, default=12.0, help="Maximum episode duration [s].")
    parser.add_argument("--max-speed", type=float, default=0.70, help="Cruising speed target [m/s].")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    if args.resume is not None and args.pretrained_policy is not None:
        raise ValueError("--pretrained-policy only applies to a fresh run, not --resume.")
    checkpoint = args.resume.expanduser().resolve() if args.resume is not None else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    env_cfg = make_env_cfg(
        num_envs=args.num_envs,
        device=args.device,
        gravel_length=args.gravel_length,
        gravel_width=args.gravel_width,
        gravel_depth=args.gravel_depth,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        grid_type=args.grid_type,
        dt=args.dt,
        decimation=args.decimation,
        max_forward_speed=args.max_speed,
        episode_length_s=args.episode_length,
        viewer=False,
        reset_noise=True,
    )
    env_cfg.seed = args.seed
    agent_cfg = make_agent_cfg(
        device=args.device,
        max_iterations=args.max_iterations,
        save_interval=args.save_interval,
        num_steps_per_env=args.num_steps_per_env,
    )
    agent_cfg.seed = args.seed
    agent_cfg.run_name = args.run_name
    agent_cfg.resume = args.resume is not None

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_name += f"_{args.run_name}"
    log_dir = Path(args.log_root).expanduser().resolve() / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    dump_yaml(str(log_dir / "task_args.yaml"), serialized_args)
    dump_yaml(str(log_dir / "env_cfg.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent_cfg.yaml"), agent_cfg)
    print(f"[INFO]: Logging training run to {log_dir}")

    raw_env = AnymalGravelEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if checkpoint is None:
        policy_path = resolve_pretrained_policy(args.pretrained_policy)
        parity_error = initialize_actor_from_pretrained(runner, policy_path)
        print(
            f"[INFO]: Initialized PPO actor from {policy_path} "
            f"(maximum output error {parity_error:.2e})"
        )
    else:
        print(f"[INFO]: Resuming from {checkpoint}")
        runner.load(str(checkpoint))

    try:
        # Every rollout begins at the pavement entrance; random episode lengths would
        # erase the cross-and-stop structure of this finite traversal task.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    except KeyboardInterrupt:
        interrupt_path = log_dir / "model_interrupt.pt"
        runner.save(str(interrupt_path))
        print(f"\n[INFO]: Saved interrupted run to {interrupt_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
