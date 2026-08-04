#!/usr/bin/env python3
"""Stand ANYmal-C on a bed of MPM gravel in Isaac Lab (kit-less / Newton).

The robot holds its default standing pose under a joint-space PD controller while
MuJoCo Warp (robot) and Newton's implicit MPM solver (gravel) are stepped together
through the virtual-proxy coupler in ``scripts/gravel_coupling.py``.

The defaults track Newton's own ANYmal/MPM example. Three of them are load-bearing,
and changing any one on its own is enough to drop the robot onto its belly: the proxy
mode must be ``lagged``, the MPM collider velocities must be ``forward``, and the bed
needs real shear strength (see :func:`gravel_coupling.gravel_material`). Those three
were verified by bisection against ``scripts/repro_anymal_mpm_newton.py``, which is the
same scene in pure Newton and is the faster place to test this kind of behaviour.

Known issue: this coupled scene diverges to NaN within roughly fifty steps once the feet
reach the gravel, while the same scene in pure Newton is stable indefinitely. The fault
is on the Isaac Lab side of the coupling, not in Newton, the material or the controller:

* The bed on its own is fine; it holds its surface while the robot falls past it
  (``--spawn-clearance 0.6``), and the robot is what blows up first, at contact.
* ``--passive`` diverges too, so the PD controller is not involved at all. The same
  passive case in pure Newton merely sags onto its belly and stays finite.
* Initial foot burial is not the trigger either: pure Newton stands with the feet
  embedded twice as deep as they are here.
* Ruled out by experiment: grid type, ``njmax``, the proxy body set, both contact
  sources, and ``--no-collider-refresh``.

That leaves the remaining two :class:`gravel_coupling.NewtonMPMCouplerManager`
overrides, and how Isaac Lab drives the coupled step, as the places left to look.

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
parser.add_argument("--gravel-depth", type=float, default=0.1, help="Thickness of the gravel bed [m].")
parser.add_argument(
    "--gravel-size",
    type=float,
    nargs=2,
    default=(1.4, 0.9),
    metavar=("X", "Y"),
    help="Horizontal extent of the gravel bed [m].",
)
parser.add_argument("--voxel-size", type=float, default=0.03, help="MPM grid voxel size [m].")
parser.add_argument(
    "--grid-type",
    type=str,
    default="sparse",
    choices=["sparse", "dense", "fixed"],
    help="MPM background grid. Only 'fixed' allows the coupled step to be CUDA-graph captured.",
)
parser.add_argument("--particles-per-cell", type=float, default=3.0, help="Particle density per MPM cell.")
parser.add_argument("--young-modulus", type=float, default=1.0e15, help="Elastic modulus of the gravel [Pa].")
parser.add_argument("--yield-stress", type=float, default=1.0e6, help="Deviatoric yield stress of the gravel [Pa].")
parser.add_argument(
    "--yield-pressure", type=float, default=1.0e7, help="Compressive yield pressure of the gravel [Pa]."
)
parser.add_argument("--viscosity", type=float, default=1.0e4, help="Plastic viscosity of the gravel [Pa*s].")
parser.add_argument("--damping", type=float, default=0.02, help="Elastic damping relaxation time of the gravel [s].")
parser.add_argument("--friction", type=float, default=0.8, help="Internal friction coefficient of the gravel.")
parser.add_argument(
    "--proxy-bodies",
    type=str,
    default=None,
    help="Regex over Newton body labels selecting the robot bodies that collide with the gravel.",
)
parser.add_argument(
    "--proxy-mode",
    type=str,
    default="lagged",
    choices=["lagged", "staggered"],
    help="Force-transfer mode between the rigid and MPM solvers.",
)
parser.add_argument(
    "--mass-scale", type=float, default=1.0, help="Mass scale applied to the proxy feet in the MPM view."
)
parser.add_argument("--coupler-iterations", type=int, default=1, help="Proxy relaxation passes per coupled step.")
parser.add_argument("--dt", type=float, default=1.0 / 200.0, help="Coupled physics timestep [s].")
parser.add_argument("--substeps", type=int, default=1, help="Physics substeps per coupled step.")
parser.add_argument(
    "--joint-armature",
    type=float,
    default=0.06,
    help="Joint armature [kg m^2], matching Newton's ANYmal/MPM example. 0 reproduces the NaN blow-up.",
)
parser.add_argument(
    "--actuator",
    choices=["pd", "dc"],
    default="pd",
    help="'pd' is a plain PD drive matching Newton's example; 'dc' is the ANYdrive DC-motor model.",
)
parser.add_argument("--rigid-substeps", type=int, default=1, help="Robot solver substeps per coupled step.")
parser.add_argument(
    "--mujoco-contacts",
    action="store_true",
    help="Resolve robot contacts with MuJoCo's internal pipeline instead of Newton's.",
)
parser.add_argument(
    "--passive",
    action="store_true",
    help="Drop the robot with no joint actuation, to separate the PD controller from the contact.",
)
parser.add_argument(
    "--no-collider-refresh",
    action="store_true",
    help="Skip re-rasterizing the MPM colliders after the coupler installs the proxy bodies.",
)
parser.add_argument(
    "--spawn-clearance",
    type=float,
    default=0.0,
    help="Extra clearance between the feet and the gravel surface at spawn [m].",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="Report base height and gravel surface height every N steps. 0 disables reporting.",
)
parser.add_argument(
    "--print-labels",
    action="store_true",
    help="Print the Newton body and shape labels before the solver is built, then keep running.",
)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext
from isaaclab_newton.assets.mpm_object import MPMObject, MPMObjectCfg
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from gravel_coupling import gravel_material, gravel_physics_cfg

from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, ANYMAL_C_CFG  # isort: skip

PHYSICS_DT = args_cli.dt
PARTICLE_COLOR = (0.55, 0.50, 0.45)
FLOOR_THICKNESS = 0.1

# Base height ANYmal-C spawns at over flat ground; the gravel bed raises it.
FLAT_GROUND_BASE_HEIGHT = ANYMAL_C_CFG.init_state.pos[2]
SPAWN_HEIGHT = FLAT_GROUND_BASE_HEIGHT + args_cli.gravel_depth + args_cli.spawn_clearance

# The analytic ANYdrive model replaces the LSTM actuator net of ANYMAL_C_CFG: a plain
# PD-with-torque-limit controller, which is what a conventional (non-learned) stack uses.
# The nominal RL gains (40/5) only make sense with a policy supplying target offsets; held
# at a fixed target they droop ~0.6 rad under the robot's weight, so stance gains are stiffer.
# Armature is the drive-side rotor inertia reflected through the gearbox. Newton's own
# ANYmal/MPM example sets 0.06 and the MPM coupling needs it: with zero armature the feet
# have too little effective inertia for the one-step-lagged proxy feedback from the stiff
# bed, and the robot is launched to NaN within ~30 steps.
#
# The DC-motor model is what a real ANYdrive does, but its torque falls off with joint
# speed and reaches zero at velocity_limit=7.5 rad/s, so a foot kicked by the bed loses
# the very torque that would arrest it. Newton's example drives plain PD with no such
# saturation, so 'pd' is the setting that actually matches the reference.
if args_cli.actuator == "pd":
    STAND_ACTUATOR_CFG = ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness={".*": 150.0},
        damping={".*": 5.0},
        armature={".*": args_cli.joint_armature},
    )
else:
    STAND_ACTUATOR_CFG = ANYDRIVE_3_SIMPLE_ACTUATOR_CFG.replace(
        stiffness={".*": 150.0},
        damping={".*": 5.0},
        armature={".*": args_cli.joint_armature},
    )


def design_scene() -> tuple[Articulation, MPMObject]:
    """Spawn a light, the floors, the gravel beds, and the ANYmal-C robots."""
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 2.0, 0.0, 0.0))

    half_x, half_y = 0.5 * args_cli.gravel_size[0], 0.5 * args_cli.gravel_size[1]

    # One explicit floor per bed, in place of Isaac Lab's infinite ground plane. Its top
    # face is at z=0, which is where the bed's lowest particles sit.
    #
    # A single floor serves both solvers even though a shape can belong to only one
    # coupler entry: entry ownership decides which solver treats it as a rigid collider,
    # while the MPM solver picks up every shape in the model as a particle collider
    # regardless. So this slab is listed on the robot's entry, and the bed rests on it
    # too. Without it the bed has nothing underneath and free-falls, taking the robot
    # standing on it along.
    floor_cfg = sim_utils.CuboidCfg(
        size=(2.0 * half_x + 0.4, 2.0 * half_y + 0.4, FLOOR_THICKNESS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.NewtonMaterialPropertiesCfg(static_friction=0.9, dynamic_friction=0.9),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
    )
    for i in range(args_cli.num_envs):
        floor_cfg.func(f"/World/Env_{i}/Floor", floor_cfg, translation=(0.0, 0.0, -0.5 * FLOOR_THICKNESS))

    gravel_cfg = MPMObjectCfg(
        prim_path="/World/Env_.*/Gravel",
        spawn=MPMGridCfg(
            lower=(-half_x, -half_y, 0.0),
            upper=(half_x, half_y, args_cli.gravel_depth),
            voxel_size=args_cli.voxel_size,
            particles_per_cell=args_cli.particles_per_cell,
            jitter=0.25 * args_cli.voxel_size,
            material=gravel_material(
                young_modulus=args_cli.young_modulus,
                yield_stress=args_cli.yield_stress,
                yield_pressure=args_cli.yield_pressure,
                viscosity=args_cli.viscosity,
                damping=args_cli.damping,
                friction=args_cli.friction,
            ),
            visual_color=PARTICLE_COLOR,
        ),
    )

    robot_cfg = ANYMAL_C_CFG.replace(
        prim_path="/World/Env_.*/Robot",
        actuators={"legs": STAND_ACTUATOR_CFG},
        init_state=ANYMAL_C_CFG.init_state.replace(pos=(0.0, 0.0, SPAWN_HEIGHT)),
    )
    return Articulation(robot_cfg), MPMObject(gravel_cfg)


def print_labels(_payload) -> None:
    """Dump Newton body and shape labels while the builder still exists.

    This runs before the coupler resolves its selectors, so the labels are
    visible even when a selector regex matches nothing and the build fails.
    """
    for label in NewtonManager._builder.body_label:
        print(f"[BODY]: {label}")
    for label, body in zip(NewtonManager._builder.shape_label, NewtonManager._builder.shape_body):
        if body < 0:
            print(f"[STATIC SHAPE]: {label}")


def report(step: int, robot: Articulation, gravel: MPMObject) -> None:
    """Print the robot base height, the foot heights, and the top of the gravel bed."""
    base_height = robot.data.root_pos_w.torch[:, 2].mean()
    foot_ids = [i for i, name in enumerate(robot.body_names) if name.endswith("FOOT")]
    foot_height = robot.data.body_link_pos_w.torch[:, foot_ids, 2]
    gravel_top = gravel.data.particle_pos_w.torch[..., 2].max()
    quat = robot.data.root_quat_w.torch
    upright = (1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)).mean()
    joint_error = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().max()
    print(
        f"[INFO]: step {step:5d} | base z {base_height.item():.4f} m"
        f" | feet z {foot_height.min().item():.4f}-{foot_height.max().item():.4f} m"
        f" | gravel top {gravel_top.item():.4f} m"
        f" | upright {upright.item():+.3f} | max joint err {joint_error.item():.3f} rad"
    )


def run_simulator(sim: SimulationContext, robot: Articulation, gravel: MPMObject) -> None:
    """Hold the default standing pose for as long as the viewer (or step budget) lasts."""
    stand_pose = robot.data.default_joint_pos.torch.clone()
    stand_velocity = torch.zeros_like(stand_pose)

    step = 0
    while True:
        if args_cli.steps > 0 and step >= args_cli.steps:
            break
        if sim.visualizers and not any(viz.is_running() and not viz.is_closed for viz in sim.visualizers):
            break

        if not args_cli.passive:
            robot.set_joint_position_target_index(target=stand_pose)
            robot.set_joint_velocity_target_index(target=stand_velocity)
            robot.write_data_to_sim()

        sim.step()
        robot.update(PHYSICS_DT)
        gravel.update(PHYSICS_DT)
        step += 1

        if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
            report(step, robot, gravel)

    print(f"[INFO]: Stopped after {step} steps.")


def main() -> None:
    viewer_cfgs = (
        []
        if args_cli.no_viewer
        else [
            NewtonVisualizerCfg(
                eye=(3.0, -3.0, 2.0 + args_cli.gravel_depth),
                lookat=(0.0, 0.0, 0.5 + args_cli.gravel_depth),
                show_particles=True,
                particle_color=PARTICLE_COLOR,
            )
        ]
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=args_cli.device,
        physics=gravel_physics_cfg(
            voxel_size=args_cli.voxel_size,
            grid_type=args_cli.grid_type,
            num_substeps=args_cli.substeps,
            proxy_mode=args_cli.proxy_mode,
            mass_scale=args_cli.mass_scale,
            iterations=args_cli.coupler_iterations,
            rigid_substeps=args_cli.rigid_substeps,
            mujoco_contacts=args_cli.mujoco_contacts,
            refresh_collider=not args_cli.no_collider_refresh,
            **({} if args_cli.proxy_bodies is None else {"proxy_bodies": args_cli.proxy_bodies}),
        ),
        visualizer_cfgs=viewer_cfgs,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[3.0, -3.0, 2.0 + args_cli.gravel_depth],
        target=[0.0, 0.0, 0.5 + args_cli.gravel_depth],
    )

    if args_cli.print_labels:
        NewtonManager.register_callback(print_labels, PhysicsEvent.MODEL_INIT, wrap_weak_ref=False)

    robot, gravel = design_scene()
    sim.reset()

    print(
        f"[INFO]: ANYmal-C standing under joint-space PD control on"
        f" {gravel.num_instances * gravel.particles_per_object} MPM gravel particles."
    )
    report(0, robot, gravel)
    run_simulator(sim, robot, gravel)


if __name__ == "__main__":
    main()
