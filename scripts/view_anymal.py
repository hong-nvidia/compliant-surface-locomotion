#!/usr/bin/env python3
"""Stand ANYmal-C in Isaac Lab (kit-less / Newton) under a joint-space PD controller.

The robot holds its default standing pose: every step the ANYdrive actuators are
commanded to the default joint positions, so the only motion is the settling from
gravity and contact.

Run from the repo root:

    uv run python scripts/view_anymal.py                          # Newton viewer window
    uv run python scripts/view_anymal.py --no-viewer --steps 200  # headless smoke run
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to simulate on.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn.")
parser.add_argument("--no-viewer", action="store_true", help="Run without the Newton viewer.")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Stop after this many physics steps. 0 runs until the viewer is closed (or Ctrl-C).",
)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, ANYMAL_C_CFG  # isort: skip

PHYSICS_DT = 1.0 / 200.0

# The analytic ANYdrive model replaces the LSTM actuator net of ANYMAL_C_CFG: a plain
# PD-with-torque-limit controller, which is what a conventional (non-learned) stack uses.
# The nominal RL gains (40/5) only make sense with a policy supplying target offsets; held
# at a fixed target they droop ~0.6 rad under the robot's weight, so stance gains are stiffer.
STAND_ACTUATOR_CFG = ANYDRIVE_3_SIMPLE_ACTUATOR_CFG.replace(
    stiffness={".*": 150.0},
    damping={".*": 5.0},
)


def design_scene() -> Articulation:
    """Spawn a ground plane, a light, and the ANYmal-C robots."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 2.0, 0.0, 0.0))

    robot_cfg = ANYMAL_C_CFG.replace(
        prim_path="/World/Env_.*/Robot",
        actuators={"legs": STAND_ACTUATOR_CFG},
    )
    return Articulation(robot_cfg)


def run_simulator(sim: SimulationContext, robot: Articulation) -> None:
    """Hold the default standing pose for as long as the viewer (or step budget) lasts."""
    stand_pose = robot.data.default_joint_pos.torch.clone()
    stand_velocity = torch.zeros_like(stand_pose)

    step = 0
    while True:
        if args_cli.steps > 0 and step >= args_cli.steps:
            break
        if sim.visualizers and not any(viz.is_running() and not viz.is_closed for viz in sim.visualizers):
            break

        robot.set_joint_position_target_index(target=stand_pose)
        robot.set_joint_velocity_target_index(target=stand_velocity)
        robot.write_data_to_sim()

        sim.step()
        robot.update(PHYSICS_DT)
        step += 1

    print(f"[INFO]: Stopped after {step} steps.")


def main() -> None:
    viewer_cfgs = [] if args_cli.no_viewer else [NewtonVisualizerCfg(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.5))]
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=args_cli.device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=200,
                nconmax=500,
                ls_iterations=20,
                cone="elliptic",
                # At the default impratio=1 the friction constraints are as soft as the normal
                # ones, so the feet creep outward and the robot slowly does the splits even
                # though the joint targets never move. Stiffening friction pins the stance.
                impratio=100.0,
                integrator="implicitfast",
            ),
            num_substeps=1,
        ),
        visualizer_cfgs=viewer_cfgs,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, -3.0, 2.0], target=[0.0, 0.0, 0.5])

    robot = design_scene()
    sim.reset()

    if not args_cli.no_viewer and not sim.visualizers:
        print("[WARN]: Newton viewer could not be started (is a display available?). Running blind.")

    print("[INFO]: ANYmal-C standing under joint-space PD control.")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
