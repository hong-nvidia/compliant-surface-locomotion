"""Coupled MJWarp + implicit-MPM physics for locomotion on a granular surface.

The robot is simulated by MuJoCo Warp and the granular bed by Newton's implicit
MPM solver; the two exchange forces through Newton's virtual-proxy coupler
(``SolverCoupledProxy``), configured here via Isaac Lab's
:class:`~isaaclab_contrib.coupling.CouplerProxyCfg`.

Import this after Isaac Lab has been imported by the calling script.
"""

from __future__ import annotations

from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
)
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

ROBOT_BODY_PATTERN = r"/World/Env_.*/Robot"
"""Regex over full Newton body labels selecting the bodies owned by the rigid solver."""

GROUND_SHAPE_PATTERN = r"/World/defaultGroundPlane/.*"
"""Static shapes owned by the rigid solver, so the robot cannot fall through the world."""

GRAVEL_FLOOR_SHAPE_PATTERN = r"/World/Env_.*/GravelFloor.*"
"""Static shapes owned by the MPM solver, so the bed has something to rest on.

The two solvers need separate floors because an entry owns a shape exclusively,
and ``include_static_shapes`` is all-or-nothing.
"""

FOOT_BODY_PATTERN = r"/World/Env_.*/Robot/.*(SHANK|FOOT)"
"""Regex selecting the robot bodies exposed to the MPM solver as virtual proxies.

Newton's standalone ANYmal/MPM example achieves the same restriction by clearing
``ShapeFlags.COLLIDE_PARTICLES`` on every non-shank body. Isaac Lab has no
declarative shape-flag hook, so the proxy selection is the equivalent lever.
"""


class NewtonMPMCouplerManager(NewtonCouplerManager):
    """:class:`NewtonCouplerManager` with the pieces an MPM entry needs.

    Isaac Lab's coupler was written against VBD deformables and misses three things
    that :class:`~isaaclab_newton.physics.NewtonMPMManager` does for a standalone MPM
    solver. Each override below restores one of them.

    Two of these work around upstream Isaac Lab bugs, unfixed as of `develop`
    @ `05f68ac3` against Newton `1.5.0.dev0`. Each has a toggle so it can be switched
    off to check whether upstream has since fixed it; the symptoms to expect are:

    * ``patch_builder_hooks``: off, the MPM bed is silently empty. No error is raised
      -- the particle count is simply zero and the robot falls through to the floor.
    * ``patch_reset``: off, ``sim.reset()`` raises ``ValueError: world_mask has shape
      (1,), expected (2,)`` from ``SolverImplicitMPM._validate_reset_inputs``.

    The third (``refresh_collider``) is ordering, not a bug.

    The ``Newton`` prefix on the class name is load-bearing: Isaac Lab picks the asset
    backend from the physics manager's class name (``FactoryBase._get_backend``).
    """

    patch_builder_hooks: bool = True
    """Whether to run the per-world builder hooks that emit MPM particles."""

    patch_reset: bool = True
    """Whether to skip the solver-internal reset that rejects Isaac Lab's world mask."""

    refresh_collider: bool = True
    """Whether to re-rasterize the MPM colliders once the proxy bodies are installed."""

    @classmethod
    def instantiate_builder_from_stage(cls):
        # NewtonCouplerManager inherits the VBD builder path, which walks only the
        # deformable registry and never runs _per_world_builder_hooks. MPMObject
        # registers its particle emitter in exactly that hook list, so under a stock
        # coupler the particles silently never reach the model. The base path runs the
        # hooks, and nothing in the VBD path is needed here: the registry is empty.
        if not cls.patch_builder_hooks:
            super().instantiate_builder_from_stage()
            return
        NewtonManager.instantiate_builder_from_stage.__func__(cls)

    @classmethod
    def _build_solver(cls, model, solver_cfg) -> None:
        super()._build_solver(model, solver_cfg)
        # SolverImplicitMPM rasterizes its colliders when it is constructed, which
        # happens before the coupler installs the proxy bodies and their articulated
        # effective inertia into the MPM view. Without this refresh the robot's feet
        # are missing from (or wrongly weighted in) the MPM collider set.
        if not cls.refresh_collider:
            return
        coupled = NewtonManager._solver
        for entry in solver_cfg.entries:
            if isinstance(entry.solver_cfg, MPMSolverCfg):
                coupled.solver(entry.name).setup_collider(model=coupled.view(entry.name))

    @classmethod
    def _reset_solver_internals(cls, world_mask) -> None:
        """Skip the solver-internal reset, as :class:`NewtonMPMManager` does.

        ``SolverImplicitMPM.reset`` honors a per-world mask only when the solver
        runs one FEM environment per world, and rejects Isaac Lab's mask outright
        otherwise. ``SolverCoupled.reset`` resets every entry together, so there
        is no way to reset only the rigid entry; the coupler does not carry over
        the MPM manager's specialization, so it has to be reapplied here.
        """
        if not cls.patch_reset:
            super()._reset_solver_internals(world_mask)


def gravel_material(
    *,
    young_modulus: float = 1.0e15,
    yield_stress: float = 1.0e6,
    yield_pressure: float = 1.0e7,
    viscosity: float = 1.0e4,
    damping: float = 0.02,
    friction: float = 0.8,
    density: float = 2500.0,
) -> MPMParticleMaterialCfg:
    """Granular bed material, at the values Newton's own ANYmal/MPM example uses.

    What holds a standing robot up is the bed's resistance to *shear*, so
    ``yield_stress`` and ``viscosity`` are what matter; a bed with zero shear
    strength flows like a liquid and the robot sinks into it no matter how
    incompressible it is. ``yield_pressure`` is deliberately far below Newton's
    1e15 default, which only caps compression and does nothing to carry a load.
    """
    return MPMParticleMaterialCfg(
        density=density,
        friction=friction,
        young_modulus=young_modulus,
        viscosity=viscosity,
        damping=damping,
        yield_stress=yield_stress,
        yield_pressure=yield_pressure,
    )


def gravel_physics_cfg(
    *,
    voxel_size: float,
    num_substeps: int = 1,
    rigid_bodies: str = ROBOT_BODY_PATTERN,
    proxy_bodies: str = FOOT_BODY_PATTERN,
    proxy_mode: str = "lagged",
    mass_scale: float = 1.0,
    iterations: int = 1,
    rigid_substeps: int = 1,
    mujoco_contacts: bool = False,
    grid_type: str = "sparse",
    grid_padding: int = 50,
    max_active_cell_count: int = 1 << 16,
    refresh_collider: bool = True,
    patch_builder_hooks: bool = True,
    patch_reset: bool = True,
    manager: type[NewtonCouplerManager] = NewtonMPMCouplerManager,
) -> NewtonCfg:
    """Build the coupled MJWarp (robot) + implicit-MPM (gravel) physics config.

    Pass ``manager=NewtonCouplerManager`` to run against Isaac Lab's coupler unpatched,
    or clear one ``patch_*`` flag to isolate a single missing piece.
    """
    NewtonMPMCouplerManager.refresh_collider = refresh_collider
    NewtonMPMCouplerManager.patch_builder_hooks = patch_builder_hooks
    NewtonMPMCouplerManager.patch_reset = patch_reset
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="robot",
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    # The Isaac Lab scene carries more contacts than Newton's
                    # standalone example, which overflows nefc at the reference's 50.
                    njmax=400,
                    nconmax=1000,
                    ls_iterations=50,
                    use_mujoco_contacts=mujoco_contacts,
                ),
                bodies=[rigid_bodies],
                shape_label_patterns=[GROUND_SHAPE_PATTERN],
                substeps=rigid_substeps,
            ),
            CouplerEntryCfg(
                name="gravel",
                solver_cfg=MPMSolverCfg(
                    voxel_size=voxel_size,
                    # A fixed grid has static topology, which is what lets the coupled
                    # step be captured as a CUDA graph, but it only holds
                    # ``max_active_cell_count`` cells and silently misbehaves past that.
                    grid_type=grid_type,
                    grid_padding=grid_padding if grid_type == "fixed" else 0,
                    max_active_cell_count=max_active_cell_count if grid_type == "fixed" else -1,
                    transfer_scheme="pic",
                    strain_basis="P0",
                    # "backward" pumps energy into the bed through the collider
                    # velocities: the surface heaves and the stance collapses.
                    collider_velocity_mode="forward",
                    max_iterations=50,
                    tolerance=1.0e-6,
                    critical_fraction=0.0,
                ),
                all_particles=True,
                # The coupler rejects MPM entries that do not step in place.
                in_place=True,
                shape_label_patterns=[GRAVEL_FLOOR_SHAPE_PATTERN],
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="robot",
                destination="gravel",
                bodies=[proxy_bodies],
                mode=proxy_mode,
                mass_scale=mass_scale,
                # MPM resolves collider contact internally, so the destination needs
                # no rigid collision pipeline of its own.
                collision_pipeline=None,
            )
        ],
        iterations=iterations,
    )
    solver_cfg.class_type = manager
    return NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps)
