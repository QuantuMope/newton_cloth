"""Minimal Isaac Sim table-top cloth pickup test.

Run with:

    OMNI_KIT_ACCEPT_EULA=YES /home/horizon/isaacsim_env/bin/python \
        /home/horizon/newton_cloth/toy_isaacsim_cloth_grasp.py \
        --headless --record

The scene intentionally avoids the dual-arm robot. It creates the same table as
``isaacsim_newton_scene.py``, places the same square PhysX cloth on the table,
then drives two simple kinematic box fingers down, closed, and upward. A small
explicit adhesive patch uses the same surface-pair pressure test as the full
scene.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cloth_utils
from isaacsim_newton_scene import (
    ADHESIVE_CONTACT_MARGIN,
    CLOTH_ADHESION_OFFSET_SCALE,
    CLOTH_PARTICLE_ADHESION,
    CLOTH_PARTICLE_ADHESION_SCALE,
    CLOTH_PARTICLE_DAMPING,
    CLOTH_PARTICLE_FRICTION,
    CLOTH_PARTICLE_FRICTION_SCALE,
    GRIPPER_FINGER_COLLISION_BOX_SIZE,
    GRIPPER_FRICTION_COMBINE_MODE,
    OBJECT_ADHESIVE_SURFACE_FRICTION,
    TABLE_CENTER,
    TABLE_FRICTION_COMBINE_MODE,
    TABLE_SCALE,
    _active_adhesive_surfaces,
    _bind_material,
    _bind_physics_material,
    _cloth_particle_view,
    _create_camera,
    _create_material,
    _create_physics_material,
    _create_table,
    _enable_extensions,
    _grid_mesh,
    _look_at_quat_wxyz,
    _set_xform,
    _start_simulation_app,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "recordings" / "toy_isaacsim_cloth_grasp"
FINGER_SIZE = (0.026, 0.003, GRIPPER_FINGER_COLLISION_BOX_SIZE[1])
TOY_CLOTH_CENTER = (-0.15, TABLE_CENTER[1], TABLE_CENTER[2] + 0.075)
FINGER_CENTER_X = TOY_CLOTH_CENTER[0]
FINGER_START_Y_OFFSET = 0.070
# Keep a physical gap for folded cloth while preserving finger normal force.
FINGER_CLOSED_Y_OFFSET = 0.004
FINGER_PRESS_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.043
FINGER_START_Z = FINGER_PRESS_Z + 0.030
FINGER_CLOSE_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.067
FINGER_LIFT_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.220
FINGER_INNER_ATTACH_NORMAL_OFFSET = 0.006
FINGER_PARK_Y_OFFSET = 0.300
FINGER_PARK_Z = FINGER_START_Z + 0.050
ADHESIVE_PRESS_GAP = 0.024
TOY_CLOTH_REST_OFFSET = 0.003
TOY_CLOTH_CONTACT_OFFSET = 0.003
ADHESIVE_SIGNED_NORMAL_SLOP = TOY_CLOTH_CONTACT_OFFSET
TOY_GRIPPER_PHYSICS_FRICTION = (0.1, 0.08)
TOY_TABLE_PHYSICS_FRICTION = (0.02, 0.01)
TOY_TABLE_SLIDE_DISTANCE = 0.160
TOY_TABLE_SLIDE_APPROACH_Z = FINGER_PARK_Z
TOY_TABLE_SLIDE_FINGER_Z = FINGER_PRESS_Z - 0.002
TOY_TABLE_SLIDE_SETTLE_STEPS = 60
TOY_CLOTH_TOTAL_MASS = 0.050
TOY_CLOTH_WIDTH = 0.33
TOY_CLOTH_HEIGHT = 0.22
TOY_TABLE_SLIDE_START_X = (
    TOY_CLOTH_CENTER[0] - 0.5 * TOY_CLOTH_WIDTH - 0.5 * FINGER_SIZE[0] + 0.006
)
TOY_CLOTH_GRID_X = 165
TOY_CLOTH_GRID_Y = 110
TOY_CLOTH_GRID_COLUMNS = TOY_CLOTH_GRID_X + 1
TOY_CLOTH_GRID_ROWS = TOY_CLOTH_GRID_Y + 1
TOY_CLOTH_PARTICLE_COUNT = TOY_CLOTH_GRID_COLUMNS * TOY_CLOTH_GRID_ROWS
TOY_CLOTH_PARTICLE_MASS = TOY_CLOTH_TOTAL_MASS / TOY_CLOTH_PARTICLE_COUNT
TOY_CLOTH_PARTICLE_SPACING = (
    TOY_CLOTH_WIDTH / TOY_CLOTH_GRID_X,
    TOY_CLOTH_HEIGHT / TOY_CLOTH_GRID_Y,
)
TOY_CLOTH_STRETCH_STIFFNESS = 4500.0
TOY_CLOTH_BEND_STIFFNESS = 10.0
TOY_CLOTH_SHEAR_STIFFNESS = 1500.0
TOY_CLOTH_SPRING_DAMPING = 8.0
TOY_CLOTH_SOLVER_POSITION_ITERATIONS = 96
TOY_CLOTH_MAX_VELOCITY = 1.0
TOY_NONANCHOR_VELOCITY_DAMPING = 0.94
TOY_FOLD_PAIR_STRENGTH = 0.60
TOY_PHYSICS_DT = 1.0 / 120.0
TOY_STICKING_DEBUG = False


def _log(message: str):
    if not TOY_STICKING_DEBUG:
        return
    print(f"[toy_isaacsim_cloth_grasp] {message}", file=sys.stderr, flush=True)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="Keep the Isaac Sim window open after the scripted pinch test.",
    )
    parser.add_argument(
        "--live-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render every scripted physics step to the viewport for VNC playback.",
    )
    parser.add_argument(
        "--viewport-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set the active viewport to the toy scene camera.",
    )
    parser.add_argument(
        "--live-step-seconds",
        type=float,
        default=0.05,
        help="Sleep after live-rendered steps so the scripted motion is watchable.",
    )
    parser.add_argument(
        "--sticking-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print adhesive sticking diagnostics and final result JSON.",
    )
    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=15,
        help="Step interval for result JSON diagnostic snapshots.",
    )
    parser.add_argument(
        "--demo-mode",
        choices=("grasp", "table-slide", "gravity-fold"),
        default="grasp",
        help="Scripted toy action sequence to run.",
    )
    parser.add_argument("--steps", type=int, default=476)
    parser.add_argument(
        "--start-closed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Spawn the fingers already closed on the cloth before physics starts.",
    )
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--lower-steps", type=int, default=20)
    parser.add_argument(
        "--preclose-lift-steps",
        type=int,
        default=0,
        help="Lift the pressed patch off the table before closing the fingers.",
    )
    parser.add_argument("--close-steps", type=int, default=110)
    parser.add_argument(
        "--finger-closed-y-offset",
        type=float,
        default=FINGER_CLOSED_Y_OFFSET,
        help="Closed finger centerline Y offset from the cloth center.",
    )
    parser.add_argument(
        "--pinch-settle-steps",
        type=int,
        default=0,
        help="Hold closed fingers still before lifting so the fold can settle.",
    )
    parser.add_argument("--lift-steps", type=int, default=180)
    parser.add_argument("--slide-steps", type=int, default=240)
    parser.add_argument("--slide-hold-steps", type=int, default=30)
    parser.add_argument(
        "--table-slide-settle-steps",
        type=int,
        default=TOY_TABLE_SLIDE_SETTLE_STEPS,
        help="Initial table-slide-only settle steps before lowering fingers.",
    )
    parser.add_argument(
        "--slide-distance",
        type=float,
        default=TOY_TABLE_SLIDE_DISTANCE,
        help="World-X distance for the table-slide demo.",
    )
    parser.add_argument(
        "--slide-finger-z",
        type=float,
        default=TOY_TABLE_SLIDE_FINGER_Z,
        help="Pressed finger center z for the table-slide demo.",
    )
    parser.add_argument(
        "--finger-lift-z",
        type=float,
        default=FINGER_LIFT_Z,
        help="Absolute world z target for the scripted lift.",
    )
    parser.add_argument("--release-steps", type=int, default=60)
    parser.add_argument(
        "--explicit-sticking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive a detected cloth particle patch with the gripper surface.",
    )
    parser.add_argument(
        "--expand-adhesive-patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand contact seeds to a larger driven patch.",
    )
    parser.add_argument(
        "--adhesive-pair-mode",
        choices=("any", "finger-pinch"),
        default="any",
        help="Use any compressed finger/object pair or only opposed finger inner faces.",
    )
    parser.add_argument(
        "--adhesive-max-patches",
        type=int,
        default=10,
        help="Maximum active adhesive patches selected from all valid surface pairs.",
    )
    parser.add_argument(
        "--adhesive-components-per-pair",
        type=int,
        default=10,
        help="Maximum disjoint cloth contact components retained per surface pair.",
    )
    parser.add_argument(
        "--adhesive-min-component-particles",
        type=int,
        default=8,
        help="Minimum grid-connected contact particles needed to create a patch.",
    )
    parser.add_argument(
        "--adhesive-patch-max-particles",
        type=int,
        default=128,
        help=(
            "Maximum compressed contact particles to attach before optional "
            "expansion; nonpositive keeps the full contact component. The "
            "default keeps dense contact patches from becoming rigid plates."
        ),
    )
    parser.add_argument(
        "--adhesive-press-gap",
        type=float,
        default=ADHESIVE_PRESS_GAP,
        help="Maximum gap between opposing surfaces for a compressed adhesive contact.",
    )
    parser.add_argument(
        "--cloth-rest-offset",
        type=float,
        default=TOY_CLOTH_REST_OFFSET,
        help="PhysX particle-system rest offset for the toy cloth.",
    )
    parser.add_argument(
        "--cloth-contact-offset",
        type=float,
        default=TOY_CLOTH_CONTACT_OFFSET,
        help="PhysX particle-system contact and particle-contact offset.",
    )
    parser.add_argument(
        "--cloth-particle-damping",
        type=float,
        default=CLOTH_PARTICLE_DAMPING,
        help="PhysX PBD particle material damping for the toy cloth.",
    )
    parser.add_argument(
        "--cloth-spring-damping",
        type=float,
        default=TOY_CLOTH_SPRING_DAMPING,
        help="Spring damping for the toy cloth.",
    )
    parser.add_argument(
        "--cloth-max-velocity",
        type=float,
        default=TOY_CLOTH_MAX_VELOCITY,
        help="PhysX particle-system max velocity for toy cloth particles.",
    )
    parser.add_argument(
        "--cloth-self-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PhysX particle cloth self-collision.",
    )
    parser.add_argument(
        "--cloth-self-collision-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter adjacent particle self-collisions for the toy cloth.",
    )
    parser.add_argument(
        "--adhesive-local-patch-radius",
        type=float,
        default=0.0,
        help=(
            "Optional tangent-plane radius around the strongest contact. "
            "Use 0 to keep the physical contact candidate patch."
        ),
    )
    parser.add_argument(
        "--adhesive-expanded-patch-radius",
        type=float,
        default=0.045,
        help="Tangent-plane radius for optional patch expansion.",
    )
    parser.add_argument(
        "--adhesive-expanded-patch-max-particles",
        type=int,
        default=160,
        help="Maximum particles retained after optional patch expansion.",
    )
    parser.add_argument(
        "--pregrasp-patch-count",
        choices=("one", "two"),
        default="two",
        help="Attach one full-scene-style pregrasp patch or one patch per finger bottom.",
    )
    parser.add_argument(
        "--pinch-patch-count",
        choices=("one", "two"),
        default="one",
        help="Attach one full-scene-style inner pinch patch or one patch per finger.",
    )
    parser.add_argument(
        "--attached-velocity-mode",
        choices=("target", "zero"),
        default="target",
        help="Velocity assigned to explicitly driven adhesive particles.",
    )
    parser.add_argument(
        "--sticking-drive-mode",
        choices=("teleport", "pd"),
        default=None,
        help=(
            "Drive adhesive particles by position projection or by PD velocity "
            "updates. Defaults to teleport."
        ),
    )
    parser.add_argument(
        "--sticking-update-phase",
        choices=("before-step", "after-step", "both"),
        default="before-step",
        help="Apply explicit sticking before, after, or around the PhysX world step.",
    )
    parser.add_argument(
        "--switch-sticking-on-contact-change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation: replace stored adhesive patches when the pressed contact "
            "pair changes."
        ),
    )
    parser.add_argument(
        "--refresh-sticking-patches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep adhesive particles matched to the current compressed contact patch.",
    )
    parser.add_argument(
        "--validate-sticking-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only active adhesive particles that remain compressed by the "
            "same surface pair that created their patch."
        ),
    )
    parser.add_argument(
        "--add-new-contact-patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add fresh disjoint adhesive patches for newly compressed contacts "
            "without replacing existing patches."
        ),
    )
    parser.add_argument(
        "--sticking-pd-kp",
        type=float,
        default=80.0,
        help="Position correction rate for PD adhesive sticking.",
    )
    parser.add_argument(
        "--sticking-pd-kd",
        type=float,
        default=20.0,
        help="Velocity-error damping rate for PD adhesive sticking.",
    )
    parser.add_argument(
        "--sticking-pd-max-speed",
        type=float,
        default=1.5,
        help="Maximum correction speed for PD adhesive sticking.",
    )
    parser.add_argument(
        "--finger-static-friction",
        type=float,
        default=TOY_GRIPPER_PHYSICS_FRICTION[0],
        help="Static friction for the physical toy finger contact material.",
    )
    parser.add_argument(
        "--finger-dynamic-friction",
        type=float,
        default=TOY_GRIPPER_PHYSICS_FRICTION[1],
        help="Dynamic friction for the physical toy finger contact material.",
    )
    parser.add_argument(
        "--nonanchor-velocity-damping",
        type=float,
        default=TOY_NONANCHOR_VELOCITY_DAMPING,
        help="Velocity multiplier for particles outside explicitly driven patches.",
    )
    parser.add_argument(
        "--adhesive-finger-friction",
        type=float,
        default=TOY_GRIPPER_PHYSICS_FRICTION[0],
        help="Material-ranking friction score for finger adhesive surfaces.",
    )
    parser.add_argument(
        "--adhesive-object-friction",
        type=float,
        default=OBJECT_ADHESIVE_SURFACE_FRICTION,
        help="Material-ranking friction score for object adhesive surfaces.",
    )
    parser.add_argument(
        "--adhesive-finger-adhesion",
        type=float,
        default=0.0,
        help="Material-ranking adhesion score for finger adhesive surfaces.",
    )
    parser.add_argument(
        "--adhesive-object-adhesion",
        type=float,
        default=0.0,
        help="Material-ranking adhesion score for object adhesive surfaces.",
    )
    parser.add_argument(
        "--attach-when-closed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not create an adhesive patch until the scripted fingers finish closing.",
    )
    parser.add_argument(
        "--table-press-pregrasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a finger-object compressed contact before the fingers are fully closed.",
    )
    parser.add_argument(
        "--upgrade-pregrasp-to-pinch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace a table-press pregrasp with an opposed inner-finger pinch when available.",
    )
    parser.add_argument(
        "--handoff-pregrasp-to-inner-patches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After finger closure, move the two pregrasp patches onto the finger inner faces.",
    )
    parser.add_argument(
        "--fold-cloth-sticking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lock nearby cloth layers inside the closed jaw until the gripper opens.",
    )
    parser.add_argument(
        "--fold-cloth-pair-distance",
        type=float,
        default=0.018,
        help="Maximum particle-particle distance for compressed fold sticking.",
    )
    parser.add_argument(
        "--fold-cloth-max-pairs",
        type=int,
        default=512,
        help="Maximum cloth-cloth particle pairs to lock inside the closed jaw.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--table-static-friction",
        type=float,
        default=TOY_TABLE_PHYSICS_FRICTION[0],
        help="Toy table static friction used by the physics material.",
    )
    parser.add_argument(
        "--table-dynamic-friction",
        type=float,
        default=TOY_TABLE_PHYSICS_FRICTION[1],
        help="Toy table dynamic friction used by the physics material.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def _lerp(start: float, end: float, amount: float) -> float:
    return float(start + (end - start) * min(max(amount, 0.0), 1.0))


def _create_toy_physx_world():
    import omni.usd
    from isaacsim.core.api import World
    from pxr import Sdf, UsdGeom, UsdLux

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(
        physics_dt=TOY_PHYSICS_DT,
        rendering_dt=1.0 / 30.0,
        stage_units_in_meters=1.0,
        physics_prim_path="/World/PhysicsScene",
        backend="torch",
        device="cuda",
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world_prim.GetPrim())
    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")
    physics_context.set_solver_type("TGS")

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    light.CreateIntensityAttr(800.0)
    _set_xform(
        light.GetPrim(),
        (0.0, 0.0, 3.0),
        _look_at_quat_wxyz((0, 0, 3), (0, 0, 0), up=(0.0, 1.0, 0.0)),
    )
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(250.0)
    return stage, world


def _create_finger(
    stage,
    world,
    path: str,
    name: str,
    material,
    physics_material,
    position,
):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import UsdGeom, UsdPhysics

    finger = UsdGeom.Cube.Define(stage, path)
    finger.CreateSizeAttr(1.0)
    _set_xform(finger.GetPrim(), position, scale=FINGER_SIZE)
    UsdPhysics.CollisionAPI.Apply(finger.GetPrim())
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(finger.GetPrim())
    rigid_body_api.CreateKinematicEnabledAttr(True)
    _bind_material(finger.GetPrim(), material)
    _bind_physics_material(finger.GetPrim(), physics_material)
    return world.scene.add(
        SingleRigidPrim(
            prim_path=path,
            name=name,
            reset_xform_properties=False,
        )
    )


def _set_finger_pose(prim, position):
    # Keep USD surface-frame diagnostics synchronized with the PhysX rigid pose.
    _set_xform(prim.prim, position, scale=FINGER_SIZE)
    prim.set_world_pose(position=np.asarray(position, dtype=np.float32))


def _create_tabletop_cloth(
    stage,
    world,
    material,
    initial_fold: bool = False,
    self_collision: bool = True,
    self_collision_filter: bool = True,
    particle_damping: float = CLOTH_PARTICLE_DAMPING,
    spring_damping: float = TOY_CLOTH_SPRING_DAMPING,
    max_velocity: float = TOY_CLOTH_MAX_VELOCITY,
):
    from isaacsim.core.api.materials.particle_material import ParticleMaterial
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, "/World/ToyCloth")
    points, counts, indices = _grid_mesh(
        TOY_CLOTH_WIDTH,
        TOY_CLOTH_HEIGHT,
        TOY_CLOTH_GRID_X,
        TOY_CLOTH_GRID_Y,
        0.002,
    )
    translated_points = []
    for x, y, z in points:
        folded_x = -abs(x) if initial_fold and x > 0.0 else x
        folded_z = z + (0.010 if initial_fold and x > 0.0 else 0.0)
        translated_points.append(
            Gf.Vec3f(
                float(folded_x + TOY_CLOTH_CENTER[0]),
                float(y + TOY_CLOTH_CENTER[1]),
                float(folded_z + TOY_CLOTH_CENTER[2]),
            )
        )
    mesh.CreatePointsAttr(translated_points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    _bind_material(mesh.GetPrim(), material)

    particle_material = ParticleMaterial(
        prim_path="/World/ToyParticleMaterial",
        drag=0.1,
        lift=0.0,
        friction=CLOTH_PARTICLE_FRICTION,
        particle_friction_scale=CLOTH_PARTICLE_FRICTION_SCALE,
        damping=particle_damping,
        adhesion=CLOTH_PARTICLE_ADHESION,
        particle_adhesion_scale=CLOTH_PARTICLE_ADHESION_SCALE,
        adhesion_offset_scale=CLOTH_ADHESION_OFFSET_SCALE,
    )
    _log(
        "created toy cloth PBD material: "
        f"friction={CLOTH_PARTICLE_FRICTION} "
        f"particle_friction_scale={CLOTH_PARTICLE_FRICTION_SCALE} "
        f"damping={particle_damping} "
        f"adhesion={CLOTH_PARTICLE_ADHESION} "
        f"particle_adhesion_scale={CLOTH_PARTICLE_ADHESION_SCALE} "
        f"adhesion_offset_scale={CLOTH_ADHESION_OFFSET_SCALE}"
    )
    particle_system = SingleParticleSystem(
        prim_path="/World/ToyParticleSystem",
        name="toy_particle_system",
        simulation_owner=world.get_physics_context().prim_path,
        rest_offset=TOY_CLOTH_REST_OFFSET,
        contact_offset=TOY_CLOTH_CONTACT_OFFSET,
        solid_rest_offset=TOY_CLOTH_REST_OFFSET,
        fluid_rest_offset=TOY_CLOTH_REST_OFFSET,
        particle_contact_offset=TOY_CLOTH_CONTACT_OFFSET,
        solver_position_iteration_count=TOY_CLOTH_SOLVER_POSITION_ITERATIONS,
        max_velocity=max_velocity,
        global_self_collision_enabled=self_collision,
        non_particle_collision_enabled=True,
    )
    particle_system.set_simulation_owner(world.get_physics_context().prim_path)
    cloth = SingleClothPrim(
        prim_path="/World/ToyCloth",
        particle_system=particle_system,
        particle_material=particle_material,
        name="toy_cloth",
        particle_mass=TOY_CLOTH_PARTICLE_MASS,
        self_collision=self_collision,
        self_collision_filter=self_collision_filter,
        stretch_stiffness=TOY_CLOTH_STRETCH_STIFFNESS,
        bend_stiffness=TOY_CLOTH_BEND_STIFFNESS,
        shear_stiffness=TOY_CLOTH_SHEAR_STIFFNESS,
        spring_damping=spring_damping,
    )
    world.scene.add(cloth)
    return cloth


def _adhesive_surfaces(args):
    yz_radius = float(
        math.hypot(0.5 * FINGER_SIZE[1], 0.5 * FINGER_SIZE[2])
        + ADHESIVE_CONTACT_MARGIN
    )
    xz_radius = float(
        math.hypot(0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[2])
        + ADHESIVE_CONTACT_MARGIN
    )
    xy_radius = float(
        math.hypot(0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[1])
        + ADHESIVE_CONTACT_MARGIN
    )
    table_radius = float(
        math.hypot(0.5 * TABLE_SCALE[0], 0.5 * TABLE_SCALE[1])
        + ADHESIVE_CONTACT_MARGIN
    )

    def finger_surface(
        name,
        prim_path,
        local_origin,
        local_u,
        local_v,
        local_normal,
        half_extents,
        radius,
    ):
        return {
            "name": name,
            "kind": "finger",
            "friction": args.adhesive_finger_friction,
            "adhesion": args.adhesive_finger_adhesion,
            "prim_path": prim_path,
            "local_origin": local_origin,
            "local_u": local_u,
            "local_v": local_v,
            "local_normal": local_normal,
            "half_extents": half_extents,
            "radius": radius,
            "contact_margin": ADHESIVE_CONTACT_MARGIN,
            "requires_closed_side": "pinch",
        }

    finger_surfaces = [
        finger_surface(
            "left_inner_face",
            "/World/LeftFinger",
            (0.0, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[2]),
            xz_radius,
        ),
        finger_surface(
            "right_inner_face",
            "/World/RightFinger",
            (0.0, -0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[2]),
            xz_radius,
        ),
        finger_surface(
            "left_outer_face",
            "/World/LeftFinger",
            (0.0, -0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[2]),
            xz_radius,
        ),
        finger_surface(
            "right_outer_face",
            "/World/RightFinger",
            (0.0, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[2]),
            xz_radius,
        ),
        finger_surface(
            "left_bottom_face",
            "/World/LeftFinger",
            (0.0, 0.0, -0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[1]),
            xy_radius,
        ),
        finger_surface(
            "right_bottom_face",
            "/World/RightFinger",
            (0.0, 0.0, -0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[1]),
            xy_radius,
        ),
        finger_surface(
            "left_top_face",
            "/World/LeftFinger",
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[1]),
            xy_radius,
        ),
        finger_surface(
            "right_top_face",
            "/World/RightFinger",
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5 * FINGER_SIZE[0], 0.5 * FINGER_SIZE[1]),
            xy_radius,
        ),
        finger_surface(
            "left_xneg_face",
            "/World/LeftFinger",
            (-0.5, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.5 * FINGER_SIZE[1], 0.5 * FINGER_SIZE[2]),
            yz_radius,
        ),
        finger_surface(
            "right_xneg_face",
            "/World/RightFinger",
            (-0.5, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.5 * FINGER_SIZE[1], 0.5 * FINGER_SIZE[2]),
            yz_radius,
        ),
        finger_surface(
            "left_xpos_face",
            "/World/LeftFinger",
            (0.5, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.5 * FINGER_SIZE[1], 0.5 * FINGER_SIZE[2]),
            yz_radius,
        ),
        finger_surface(
            "right_xpos_face",
            "/World/RightFinger",
            (0.5, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.5 * FINGER_SIZE[1], 0.5 * FINGER_SIZE[2]),
            yz_radius,
        ),
    ]
    return [
        *finger_surfaces,
        {
            "name": "table_top",
            "kind": "object",
            "friction": args.adhesive_object_friction,
            "adhesion": args.adhesive_object_adhesion,
            "prim_path": "/World/Table",
            "local_origin": (0.0, 0.0, 0.5),
            "local_u": (1.0, 0.0, 0.0),
            "local_v": (0.0, 1.0, 0.0),
            "local_normal": (0.0, 0.0, 1.0),
            "half_extents": (0.5 * TABLE_SCALE[0], 0.5 * TABLE_SCALE[1]),
            "radius": table_radius,
            "contact_margin": ADHESIVE_CONTACT_MARGIN,
        },
    ]


def _torch_cloth_state(cloth):
    cloth_view = _cloth_particle_view(cloth)
    positions = cloth_view.get_world_positions()
    velocities = cloth_view.get_velocities()
    particle_positions = positions[0] if positions.ndim == 3 else positions
    particle_velocities = velocities[0] if velocities.ndim == 3 else velocities
    return cloth_view, positions, velocities, particle_positions, particle_velocities


def _surface_frame_tensors(surface, particle_positions):
    return cloth_utils.surface_frame_tensors(surface, particle_positions)


def _surface_contact_torch(surface, particle_positions):
    return cloth_utils.surface_contact_torch(surface, particle_positions)


def _surface_contacts_torch(active_surfaces, particle_positions):
    return cloth_utils.surface_contacts_torch(active_surfaces, particle_positions)


def _surfaces_are_opposed(first_surface, second_surface, particle_positions):
    return cloth_utils.surfaces_are_opposed(first_surface, second_surface)


def _pressed_between_surfaces_torch(first_contact, second_contact):
    return cloth_utils.pressed_between_surfaces_torch(
        first_contact,
        second_contact,
        ADHESIVE_PRESS_GAP,
        ADHESIVE_SIGNED_NORMAL_SLOP,
    )


def _surface_candidate_distance(surface, candidate_indices):
    return cloth_utils.surface_candidate_distance(surface, candidate_indices)


def _choose_adhesive_anchor(first_surface, second_surface, candidate_indices):
    return cloth_utils.choose_adhesive_anchor(
        first_surface,
        second_surface,
        candidate_indices,
    )


def _is_finger_pinch_mode(mode: str) -> bool:
    return cloth_utils.is_finger_pinch_mode(mode)


def _expand_adhesive_patch_indices(
    particle_positions,
    seed_indices,
    anchor_frame,
    expanded_patch_radius: float,
    expanded_patch_max_particles: int,
    eligible_indices,
):
    return cloth_utils.expand_adhesive_patch_indices(
        particle_positions,
        seed_indices,
        anchor_frame,
        expanded_patch_radius,
        expanded_patch_max_particles,
        eligible_indices,
    )


def _connected_component_indices_grid(mask, max_components: int):
    return cloth_utils.connected_component_indices_grid(
        mask,
        max_components,
        1,
        (TOY_CLOTH_GRID_ROWS, TOY_CLOTH_GRID_COLUMNS),
    )


def _make_adhesive_patch(
    first_surface,
    second_surface,
    particle_positions,
    candidate_indices,
    pair_score,
    expand_adhesive_patch: bool,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    required_anchor_name: str | None = None,
):
    return cloth_utils.make_adhesive_patch(
        first_surface,
        second_surface,
        particle_positions,
        candidate_indices,
        pair_score,
        expand_adhesive_patch=expand_adhesive_patch,
        adhesive_patch_max_particles=adhesive_patch_max_particles,
        adhesive_local_patch_radius=adhesive_local_patch_radius,
        adhesive_expanded_patch_radius=adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles=adhesive_expanded_patch_max_particles,
        required_anchor_name=required_anchor_name,
    )


def _choose_adhesive_patches_torch(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_max_patches: int,
    adhesive_components_per_pair: int,
    adhesive_min_component_particles: int,
    required_anchor_name: str | None = None,
    excluded_indices=None,
):
    return cloth_utils.choose_adhesive_patches_torch(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch=expand_adhesive_patch,
        adhesive_pair_mode=adhesive_pair_mode,
        adhesive_patch_max_particles=adhesive_patch_max_particles,
        adhesive_local_patch_radius=adhesive_local_patch_radius,
        adhesive_expanded_patch_radius=adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles=adhesive_expanded_patch_max_particles,
        adhesive_max_patches=adhesive_max_patches,
        adhesive_components_per_pair=adhesive_components_per_pair,
        adhesive_min_component_particles=adhesive_min_component_particles,
        press_gap=ADHESIVE_PRESS_GAP,
        signed_normal_slop=ADHESIVE_SIGNED_NORMAL_SLOP,
        grid_shape=(TOY_CLOTH_GRID_ROWS, TOY_CLOTH_GRID_COLUMNS),
        required_anchor_name=required_anchor_name,
        excluded_indices=excluded_indices,
    )


def _choose_adhesive_patch_torch(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_min_component_particles: int,
    required_anchor_name: str | None = None,
    excluded_indices=None,
):
    return cloth_utils.choose_adhesive_patch_torch(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch=expand_adhesive_patch,
        adhesive_pair_mode=adhesive_pair_mode,
        adhesive_patch_max_particles=adhesive_patch_max_particles,
        adhesive_local_patch_radius=adhesive_local_patch_radius,
        adhesive_expanded_patch_radius=adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles=adhesive_expanded_patch_max_particles,
        adhesive_min_component_particles=adhesive_min_component_particles,
        press_gap=ADHESIVE_PRESS_GAP,
        signed_normal_slop=ADHESIVE_SIGNED_NORMAL_SLOP,
        grid_shape=(TOY_CLOTH_GRID_ROWS, TOY_CLOTH_GRID_COLUMNS),
        required_anchor_name=required_anchor_name,
        excluded_indices=excluded_indices,
    )


def _choose_two_anchor_patches(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_min_component_particles: int,
    anchor_names,
    excluded_indices=None,
):
    patches = []
    if excluded_indices is None:
        excluded_indices = torch.empty(
            0,
            dtype=torch.long,
            device=particle_positions.device,
        )
    for anchor_name in anchor_names:
        patch = _choose_adhesive_patch_torch(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            adhesive_pair_mode,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_min_component_particles,
            required_anchor_name=anchor_name,
            excluded_indices=excluded_indices,
        )
        if patch is None:
            continue
        patches.append(patch)
        excluded_indices = torch.unique(
            torch.cat((excluded_indices, patch["indices"]))
        )
    return patches


def _choose_two_finger_pregrasp_patches(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_min_component_particles: int,
    excluded_indices=None,
):
    return _choose_two_anchor_patches(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch,
        "any",
        adhesive_patch_max_particles,
        adhesive_local_patch_radius,
        adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles,
        adhesive_min_component_particles,
        ("left_bottom_face", "right_bottom_face"),
        excluded_indices,
    )


def _choose_two_finger_inner_patches(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_min_component_particles: int,
    excluded_indices=None,
):
    return _choose_two_anchor_patches(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch,
        "finger-pinch",
        adhesive_patch_max_particles,
        adhesive_local_patch_radius,
        adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles,
        adhesive_min_component_particles,
        ("left_inner_face", "right_inner_face"),
        excluded_indices,
    )


def _project_attached_patches(
    cloth_view,
    positions,
    velocities,
    particle_positions,
    surfaces_by_name,
    patches,
    fold_pairs,
    physics_dt: float,
    nonanchor_velocity_damping: float,
    attached_velocity_mode: str,
    sticking_drive_mode: str,
    sticking_pd_kp: float,
    sticking_pd_kd: float,
    sticking_pd_max_speed: float,
):
    def project_fold_pairs(position_targets, velocity_targets):
        _project_fold_pairs(position_targets, velocity_targets, fold_pairs, physics_dt)

    return cloth_utils.project_attached_patches(
        cloth_view,
        positions,
        velocities,
        particle_positions,
        surfaces_by_name,
        patches,
        physics_dt=physics_dt,
        nonanchor_velocity_damping=nonanchor_velocity_damping,
        attached_velocity_mode=attached_velocity_mode,
        sticking_drive_mode=sticking_drive_mode,
        sticking_pd_kp=sticking_pd_kp,
        sticking_pd_kd=sticking_pd_kd,
        sticking_pd_max_speed=sticking_pd_max_speed,
        fold_pair_projector=project_fold_pairs,
    )


def _attached_patch_summaries(patches, particle_positions):
    return cloth_utils.attached_patch_summaries(patches, particle_positions)


def _free_neighbor_indices(attached_indices, particle_count):
    attached_flat = attached_indices.detach().flatten().to(torch.long)
    attached_mask = torch.zeros(
        particle_count,
        dtype=torch.bool,
        device=attached_flat.device,
    )
    attached_mask[attached_flat] = True
    row = attached_flat // TOY_CLOTH_GRID_COLUMNS
    col = attached_flat % TOY_CLOTH_GRID_COLUMNS
    neighbor_candidates = []
    if torch.count_nonzero(col > 0).item() > 0:
        neighbor_candidates.append(attached_flat[col > 0] - 1)
    if torch.count_nonzero(col < TOY_CLOTH_GRID_COLUMNS - 1).item() > 0:
        neighbor_candidates.append(attached_flat[col < TOY_CLOTH_GRID_COLUMNS - 1] + 1)
    if torch.count_nonzero(row > 0).item() > 0:
        neighbor_candidates.append(attached_flat[row > 0] - TOY_CLOTH_GRID_COLUMNS)
    if torch.count_nonzero(row < TOY_CLOTH_GRID_ROWS - 1).item() > 0:
        neighbor_candidates.append(attached_flat[row < TOY_CLOTH_GRID_ROWS - 1] + TOY_CLOTH_GRID_COLUMNS)
    if not neighbor_candidates:
        return torch.empty(0, dtype=torch.long, device=attached_flat.device)
    neighbors = torch.unique(torch.cat(neighbor_candidates))
    return neighbors[~attached_mask[neighbors]]


def _attachment_centroid_diagnostics(patches, particle_positions):
    diagnostics = {
        "cloth_centroid_z": float(torch.mean(particle_positions[:, 2]).item()),
        "free_cloth_centroid_z": float(torch.mean(particle_positions[:, 2]).item()),
        "attached_centroid_z": None,
        "bottom_attached_centroid_z": None,
        "inner_attached_centroid_z": None,
        "boundary_free_centroid_z": None,
        "bottom_boundary_free_centroid_z": None,
        "inner_boundary_free_centroid_z": None,
        "bottom_attached_particles": 0,
        "inner_attached_particles": 0,
        "boundary_free_particles": 0,
        "bottom_boundary_free_particles": 0,
        "inner_boundary_free_particles": 0,
    }
    if not patches:
        return diagnostics
    attached_indices = torch.unique(torch.cat([patch["indices"] for patch in patches]))
    free_mask = torch.ones(
        particle_positions.shape[0],
        dtype=torch.bool,
        device=particle_positions.device,
    )
    free_mask[attached_indices] = False
    free_positions = particle_positions[free_mask]
    if free_positions.numel() > 0:
        diagnostics["free_cloth_centroid_z"] = float(
            torch.mean(free_positions[:, 2]).item()
        )
    diagnostics["attached_centroid_z"] = float(
        torch.mean(particle_positions[attached_indices, 2]).item()
    )
    boundary_indices = _free_neighbor_indices(
        attached_indices,
        particle_positions.shape[0],
    )
    if boundary_indices.numel() > 0:
        diagnostics["boundary_free_particles"] = int(boundary_indices.numel())
        diagnostics["boundary_free_centroid_z"] = float(
            torch.mean(particle_positions[boundary_indices, 2]).item()
        )
    bottom_indices = [
        patch["indices"]
        for patch in patches
        if patch["anchor_name"] in ("left_bottom_face", "right_bottom_face")
    ]
    inner_indices = [
        patch["indices"]
        for patch in patches
        if patch["mode"] == "left_inner_face+right_inner_face"
    ]
    if bottom_indices:
        unique_bottom_indices = torch.unique(torch.cat(bottom_indices))
        diagnostics["bottom_attached_particles"] = int(unique_bottom_indices.numel())
        diagnostics["bottom_attached_centroid_z"] = float(
            torch.mean(particle_positions[unique_bottom_indices, 2]).item()
        )
        bottom_boundary_indices = _free_neighbor_indices(
            unique_bottom_indices,
            particle_positions.shape[0],
        )
        if bottom_boundary_indices.numel() > 0:
            diagnostics["bottom_boundary_free_particles"] = int(
                bottom_boundary_indices.numel()
            )
            diagnostics["bottom_boundary_free_centroid_z"] = float(
                torch.mean(particle_positions[bottom_boundary_indices, 2]).item()
            )
    if inner_indices:
        unique_inner_indices = torch.unique(torch.cat(inner_indices))
        diagnostics["inner_attached_particles"] = int(unique_inner_indices.numel())
        diagnostics["inner_attached_centroid_z"] = float(
            torch.mean(particle_positions[unique_inner_indices, 2]).item()
        )
        inner_boundary_indices = _free_neighbor_indices(
            unique_inner_indices,
            particle_positions.shape[0],
        )
        if inner_boundary_indices.numel() > 0:
            diagnostics["inner_boundary_free_particles"] = int(
                inner_boundary_indices.numel()
            )
            diagnostics["inner_boundary_free_centroid_z"] = float(
                torch.mean(particle_positions[inner_boundary_indices, 2]).item()
            )
    return diagnostics


def _cloth_shape_diagnostics(particle_positions):
    centroid = torch.mean(particle_positions, dim=0)
    minimum = torch.min(particle_positions, dim=0).values
    maximum = torch.max(particle_positions, dim=0).values
    span = maximum - minimum
    columns = torch.arange(
        TOY_CLOTH_GRID_COLUMNS,
        device=particle_positions.device,
    ).repeat(TOY_CLOTH_GRID_ROWS)
    front_mask = columns >= (TOY_CLOTH_GRID_COLUMNS // 2)
    front_centroid = torch.mean(particle_positions[front_mask], dim=0)
    return {
        "cloth_centroid_m": [float(value.item()) for value in centroid],
        "cloth_min_m": [float(value.item()) for value in minimum],
        "cloth_max_m": [float(value.item()) for value in maximum],
        "cloth_span_m": [float(value.item()) for value in span],
        "front_cloth_centroid_m": [
            float(value.item()) for value in front_centroid
        ],
    }


def _extreme_particle_diagnostics(particle_positions, particle_velocities, patches):
    min_z_index = int(torch.argmin(particle_positions[:, 2]).item())
    max_z_index = int(torch.argmax(particle_positions[:, 2]).item())
    attached_modes_by_index = {}
    for patch in patches:
        for index in patch["indices"].detach().cpu().tolist():
            attached_modes_by_index[int(index)] = patch["mode"]

    def particle_summary(index):
        return {
            "index": index,
            "position_m": [
                float(value.item()) for value in particle_positions[index]
            ],
            "velocity_mps": [
                float(value.item()) for value in particle_velocities[index]
            ],
            "attached_mode": attached_modes_by_index.get(index),
        }

    return {
        "min_z_particle": particle_summary(min_z_index),
        "max_z_particle": particle_summary(max_z_index),
    }


def _surface_selected_contact_diagnostic(surface, selected_indices):
    return cloth_utils.surface_selected_contact_diagnostic(surface, selected_indices)


def _active_patch_contact_diagnostics(active_surfaces, particle_positions, patches):
    return cloth_utils.active_patch_contact_diagnostics(
        active_surfaces,
        particle_positions,
        patches,
        press_gap=ADHESIVE_PRESS_GAP,
        signed_normal_slop=ADHESIVE_SIGNED_NORMAL_SLOP,
    )


def _clamp_inner_patch_targets_to_gap(patches):
    clamped_patches = []
    for patch in patches:
        if patch["anchor_name"] not in ("left_inner_face", "right_inner_face"):
            clamped_patches.append(patch)
            continue
        local_positions = patch["local_positions"].clone()
        local_positions[:, 2] = FINGER_INNER_ATTACH_NORMAL_OFFSET
        clamped_patches.append(
            {
                **patch,
                "local_positions": local_positions,
            }
        )
    return clamped_patches


def _patches_are_bottom_pregrasp(patches):
    return any(
        patch["anchor_name"] in ("left_bottom_face", "right_bottom_face")
        for patch in patches
    )


def _jaw_region_mask(particle_positions, surfaces_by_name):
    left_frame = _surface_frame_tensors(
        surfaces_by_name["left_inner_face"],
        particle_positions,
    )
    right_frame = _surface_frame_tensors(
        surfaces_by_name["right_inner_face"],
        particle_positions,
    )
    left_local = (particle_positions - left_frame["origin"]) @ left_frame["rotation"]
    right_local = (
        particle_positions - right_frame["origin"]
    ) @ right_frame["rotation"]
    half_x = 0.5 * FINGER_SIZE[0] + 0.012
    half_z = 0.5 * FINGER_SIZE[2] + 0.012
    lower_normal_bound = -TOY_CLOTH_CONTACT_OFFSET
    return (
        (left_local[:, 2] >= lower_normal_bound)
        & (right_local[:, 2] >= lower_normal_bound)
        & (torch.abs(left_local[:, 0]) <= half_x)
        & (torch.abs(left_local[:, 1]) <= half_z)
        & (torch.abs(right_local[:, 0]) <= half_x)
        & (torch.abs(right_local[:, 1]) <= half_z)
    )


def _compressed_inner_contact_mask(particle_positions, surfaces_by_name):
    left_surface = surfaces_by_name["left_inner_face"]
    right_surface = surfaces_by_name["right_inner_face"]
    left_contact = _surface_contact_torch(left_surface, particle_positions)
    right_contact = _surface_contact_torch(right_surface, particle_positions)
    return (
        left_contact["mask"]
        & right_contact["mask"]
        & _pressed_between_surfaces_torch(left_contact, right_contact)
    )


def _choose_compressed_fold_pairs(
    particle_positions,
    surfaces_by_name,
    attached_indices,
    max_pairs: int,
    pair_distance: float,
):
    jaw_indices = torch.nonzero(
        _jaw_region_mask(particle_positions, surfaces_by_name),
        as_tuple=False,
    ).flatten()
    if jaw_indices.numel() < 2:
        return None

    jaw_positions = particle_positions[jaw_indices]
    pair_distances = torch.cdist(jaw_positions, jaw_positions)
    upper_triangle = torch.triu(
        torch.ones_like(pair_distances, dtype=torch.bool),
        diagonal=1,
    )
    pair_mask = upper_triangle & (pair_distances <= float(pair_distance))
    pair_rows, pair_cols = torch.nonzero(pair_mask, as_tuple=True)
    if pair_rows.numel() == 0:
        return None

    pair_scores = pair_distances[pair_rows, pair_cols]
    if attached_indices is not None and attached_indices.numel() > 0:
        attached_mask = torch.zeros(
            particle_positions.shape[0],
            dtype=torch.bool,
            device=particle_positions.device,
        )
        attached_mask[attached_indices] = True
        jaw_attached_mask = attached_mask[jaw_indices]
        pair_has_attached = jaw_attached_mask[pair_rows] | jaw_attached_mask[pair_cols]
        pair_scores = pair_scores - pair_has_attached.to(pair_scores.dtype) * 0.05
    pair_order = torch.argsort(pair_scores)[:max_pairs]
    first_indices = jaw_indices[pair_rows[pair_order]]
    second_indices = jaw_indices[pair_cols[pair_order]]
    pair_indices = torch.stack((first_indices, second_indices), dim=1)
    rest_offsets = (
        particle_positions[second_indices] - particle_positions[first_indices]
    )
    return {
        "indices": pair_indices,
        "rest_offsets": rest_offsets.detach().clone(),
        "previous_first_targets": None,
        "previous_second_targets": None,
    }


def _project_fold_pairs(position_targets, velocity_targets, fold_pairs, physics_dt):
    if fold_pairs is None or fold_pairs["indices"].numel() == 0:
        return
    first_indices = fold_pairs["indices"][:, 0]
    second_indices = fold_pairs["indices"][:, 1]
    first_positions = position_targets[0, first_indices]
    second_positions = position_targets[0, second_indices]
    current_offsets = second_positions - first_positions
    corrections = 0.5 * TOY_FOLD_PAIR_STRENGTH * (
        current_offsets - fold_pairs["rest_offsets"]
    )
    correction_sums = torch.zeros_like(position_targets[0])
    correction_counts = torch.zeros(
        (position_targets.shape[1], 1),
        dtype=position_targets.dtype,
        device=position_targets.device,
    )
    correction_sums.index_add_(0, first_indices, corrections)
    correction_sums.index_add_(0, second_indices, -corrections)
    one_counts = torch.ones(
        (first_indices.shape[0], 1),
        dtype=position_targets.dtype,
        device=position_targets.device,
    )
    correction_counts.index_add_(0, first_indices, one_counts)
    correction_counts.index_add_(0, second_indices, one_counts)
    constrained_mask = correction_counts[:, 0] > 0
    previous_targets = position_targets[0, constrained_mask].clone()
    position_targets[0, constrained_mask] = (
        previous_targets
        + correction_sums[constrained_mask]
        / torch.clamp(correction_counts[constrained_mask], min=1.0)
    )

    constrained_indices = torch.nonzero(constrained_mask, as_tuple=False).flatten()
    next_targets = position_targets[0, constrained_indices]
    previous_targets = fold_pairs.get("previous_targets")
    previous_indices = fold_pairs.get("previous_indices")
    if (
        previous_targets is not None
        and previous_indices is not None
        and previous_targets.shape == next_targets.shape
        and torch.equal(previous_indices, constrained_indices)
    ):
        velocity_targets[0, constrained_indices] = (
            next_targets - previous_targets
        ) / float(physics_dt)
    fold_pairs["previous_indices"] = constrained_indices.detach().clone()
    fold_pairs["previous_targets"] = next_targets.detach().clone()


def _choose_current_adhesive_patches(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_max_patches: int,
    adhesive_components_per_pair: int,
    adhesive_min_component_particles: int,
    pregrasp_patch_count: str,
    pinch_patch_count: str,
    excluded_indices=None,
):
    if adhesive_pair_mode == "finger-pinch" and pinch_patch_count == "two":
        patches = _choose_two_finger_inner_patches(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_min_component_particles,
            excluded_indices,
        )
        return patches if len(patches) == 2 else []
    return _choose_adhesive_patches_torch(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch,
        adhesive_pair_mode,
        adhesive_patch_max_particles,
        adhesive_local_patch_radius,
        adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles,
        adhesive_max_patches,
        adhesive_components_per_pair,
        adhesive_min_component_particles,
        excluded_indices=excluded_indices,
    )


def _patch_contact_keys(patches):
    return cloth_utils.patch_contact_keys(patches)


def _copy_patch_velocity_history(candidate_patches, active_patches):
    cloth_utils.copy_patch_velocity_history(candidate_patches, active_patches)


def _patch_current_pressed_mask(patch, contacts_by_name):
    return cloth_utils.patch_current_pressed_mask(
        patch,
        contacts_by_name,
        press_gap=ADHESIVE_PRESS_GAP,
        signed_normal_slop=ADHESIVE_SIGNED_NORMAL_SLOP,
    )


def _filter_active_patches_by_current_contact(
    active_surfaces,
    particle_positions,
    active_patches,
    adhesive_min_component_particles: int,
):
    return cloth_utils.filter_active_patches_by_current_contact(
        active_surfaces,
        particle_positions,
        active_patches,
        press_gap=ADHESIVE_PRESS_GAP,
        min_component_particles=adhesive_min_component_particles,
        signed_normal_slop=ADHESIVE_SIGNED_NORMAL_SLOP,
    )


def _new_contact_patches(
    active_surfaces,
    particle_positions,
    active_patches,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_max_patches: int,
    adhesive_components_per_pair: int,
    adhesive_min_component_particles: int,
    pregrasp_patch_count: str,
    pinch_patch_count: str,
):
    if active_patches:
        excluded_indices = torch.unique(
            torch.cat([patch["indices"] for patch in active_patches])
        )
    else:
        excluded_indices = None
    candidate_patches = _choose_current_adhesive_patches(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch,
        adhesive_pair_mode,
        adhesive_patch_max_particles,
        adhesive_local_patch_radius,
        adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles,
        adhesive_max_patches,
        adhesive_components_per_pair,
        adhesive_min_component_particles,
        pregrasp_patch_count,
        pinch_patch_count,
        excluded_indices,
    )
    return candidate_patches


def _drive_attached_patch(
    stage,
    cloth,
    surfaces,
    grasp_state,
    log_state,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_max_patches: int,
    adhesive_components_per_pair: int,
    adhesive_min_component_particles: int,
    allow_new_attachment: bool,
    upgrade_to_finger_pinch: bool,
    handoff_to_inner_patches: bool,
    release_stale_pregrasp: bool,
    fold_cloth_sticking: bool,
    fold_cloth_max_pairs: int,
    fold_cloth_pair_distance: float,
    pregrasp_patch_count: str,
    pinch_patch_count: str,
    attached_velocity_mode: str,
    nonanchor_velocity_damping: float,
    sticking_drive_mode: str,
    sticking_pd_kp: float,
    sticking_pd_kd: float,
    sticking_pd_max_speed: float,
    switch_sticking_on_contact_change: bool,
    refresh_sticking_patches: bool,
    validate_sticking_contact: bool,
    add_new_contact_patches: bool,
):
    cloth_view, positions, velocities, particle_positions, _particle_velocities = (
        _torch_cloth_state(cloth)
    )
    active_surfaces = _active_adhesive_surfaces(stage, surfaces, {"pinch": True})
    surfaces_by_name = {surface["name"]: surface for surface in active_surfaces}
    active_patches = grasp_state.get("patches")
    if (
        release_stale_pregrasp
        and active_patches
        and not grasp_state.get("inner_handoff_done", False)
        and _patches_are_bottom_pregrasp(active_patches)
    ):
        grasp_state["patches"] = []
        grasp_state["fold_pairs"] = None
        _log("released stale bottom-face pregrasp before lift")
        return False
    if active_patches and validate_sticking_contact:
        valid_patches, release_events = _filter_active_patches_by_current_contact(
            active_surfaces,
            particle_positions,
            active_patches,
            adhesive_min_component_particles,
        )
        if release_events:
            for patch, pressed_count, original_count in release_events:
                _log(
                    "validated adhesive patch "
                    f"{patch['mode']} anchored to {patch['anchor_name']}: "
                    f"{pressed_count}/{original_count} particles remain compressed"
                )
        if len(valid_patches) != len(active_patches):
            grasp_state["fold_pairs"] = None
        grasp_state["patches"] = valid_patches
        active_patches = valid_patches
        log_state["attached_particles"] = int(
            sum(patch["indices"].numel() for patch in active_patches)
        )
    if not active_patches:
        if not allow_new_attachment:
            return False
        active_patches = _choose_current_adhesive_patches(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            adhesive_pair_mode,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_max_patches,
            adhesive_components_per_pair,
            adhesive_min_component_particles,
            pregrasp_patch_count,
            pinch_patch_count,
        )
        if not active_patches:
            return False
        grasp_state["patches"] = active_patches
        log_state["attached_step"] = log_state["step"]
        log_state["attached_particles"] = int(
            sum(patch["indices"].numel() for patch in active_patches)
        )
        for patch in active_patches:
            patch_summary = _attached_patch_summaries(
                [patch],
                particle_positions,
            )[0]
            _log(
                "attached "
                f"{int(patch['indices'].numel())} particles with "
                f"{patch['mode']} anchored to {patch['anchor_name']} "
                f"from {patch.get('seed_count', 'unknown')} contact seeds "
                f"span={patch_summary['span_m']}"
            )
    elif add_new_contact_patches and allow_new_attachment:
        remaining_patch_slots = max(adhesive_max_patches - len(active_patches), 0)
        new_patches = _new_contact_patches(
            active_surfaces,
            particle_positions,
            active_patches,
            expand_adhesive_patch,
            adhesive_pair_mode,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            remaining_patch_slots,
            adhesive_components_per_pair,
            adhesive_min_component_particles,
            pregrasp_patch_count,
            pinch_patch_count,
        )
        if new_patches:
            active_patches = [*active_patches, *new_patches]
            grasp_state["patches"] = active_patches
            log_state["attached_particles"] = int(
                sum(patch["indices"].numel() for patch in active_patches)
            )
            for patch in new_patches:
                patch_summary = _attached_patch_summaries(
                    [patch],
                    particle_positions,
                )[0]
                _log(
                    "added fresh adhesive patch "
                    f"{patch['mode']} anchored to {patch['anchor_name']} "
                    f"with {int(patch['indices'].numel())} cloth particles "
                    f"from {patch.get('seed_count', 'unknown')} contact seeds "
                    f"span={patch_summary['span_m']}"
                )
    elif (
        (switch_sticking_on_contact_change or refresh_sticking_patches)
        and allow_new_attachment
    ):
        candidate_patches = _choose_current_adhesive_patches(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            adhesive_pair_mode,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_max_patches,
            adhesive_components_per_pair,
            adhesive_min_component_particles,
            pregrasp_patch_count,
            pinch_patch_count,
        )
        if not candidate_patches and refresh_sticking_patches:
            grasp_state["patches"] = []
            grasp_state["fold_pairs"] = None
            log_state["attached_particles"] = 0
            _log("released adhesive patches after compressed contact was lost")
            return False
        old_keys = _patch_contact_keys(active_patches)
        new_keys = _patch_contact_keys(candidate_patches)
        should_replace = bool(candidate_patches) and (
            refresh_sticking_patches or new_keys != old_keys
        )
        if should_replace:
            _copy_patch_velocity_history(candidate_patches, active_patches)
            grasp_state["patches"] = candidate_patches
            grasp_state["fold_pairs"] = None
            active_patches = candidate_patches
            log_state["attached_particles"] = int(
                sum(patch["indices"].numel() for patch in active_patches)
            )
            if new_keys != old_keys:
                _log(
                    "switched adhesive patch to current pressed contact "
                    f"{_patch_contact_keys(active_patches)}"
                )
    elif (
        upgrade_to_finger_pinch
        and len(active_patches) == 1
        and not _is_finger_pinch_mode(active_patches[0]["mode"])
    ):
        active_grasp = active_patches[0]
        upgraded_grasp = _choose_adhesive_patch_torch(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            "finger-pinch",
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_min_component_particles,
        )
        min_upgrade_particles = max(
            adhesive_patch_max_particles // 2,
            int(0.75 * active_grasp["indices"].numel()),
        )
        if (
            upgraded_grasp is not None
            and upgraded_grasp["indices"].numel() >= min_upgrade_particles
        ):
            grasp_state["patches"] = [upgraded_grasp]
            active_grasp = upgraded_grasp
            log_state["attached_particles"] = int(active_grasp["indices"].numel())
            _log(
                "upgraded adhesive patch to "
                f"{active_grasp['mode']} anchored to {active_grasp['anchor_name']} "
                f"with {int(active_grasp['indices'].numel())} cloth particles "
                f"from {active_grasp.get('seed_count', 'unknown')} contact seeds"
            )
        elif upgraded_grasp is not None:
            skip_key = (
                upgraded_grasp["mode"],
                int(upgraded_grasp["indices"].numel()),
                min_upgrade_particles,
            )
            if log_state.get("last_skipped_upgrade") != skip_key:
                log_state["last_skipped_upgrade"] = skip_key
                _log(
                    "skipped early adhesive patch upgrade to "
                    f"{upgraded_grasp['mode']}: "
                    f"{skip_key[1]} particles is below "
                    f"the {min_upgrade_particles} particle minimum"
                )

    elif handoff_to_inner_patches and not grasp_state.get("inner_handoff_done", False):
        inner_patches = _choose_two_finger_inner_patches(
            active_surfaces,
            particle_positions,
            expand_adhesive_patch,
            adhesive_patch_max_particles,
            adhesive_local_patch_radius,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            adhesive_min_component_particles,
        )
        if len(inner_patches) == 2:
            inner_patches = _clamp_inner_patch_targets_to_gap(inner_patches)
            for patch in inner_patches:
                patch.pop("previous_target_positions", None)
            grasp_state["patches"] = inner_patches
            grasp_state["inner_handoff_done"] = True
            active_patches = inner_patches
            log_state["attached_particles"] = int(
                sum(patch["indices"].numel() for patch in active_patches)
            )
            for patch in active_patches:
                patch_summary = _attached_patch_summaries(
                    [patch],
                    particle_positions,
                )[0]
                _log(
                    "handed off adhesive patch to "
                    f"{patch['anchor_name']} with "
                    f"{int(patch['indices'].numel())} cloth particles "
                    f"span={patch_summary['span_m']}"
                )

    if fold_cloth_sticking and grasp_state.get("inner_handoff_done", False):
        if grasp_state.get("fold_pairs") is None:
            attached_indices = torch.unique(
                torch.cat([patch["indices"] for patch in active_patches])
            )
            fold_pairs = _choose_compressed_fold_pairs(
                particle_positions,
                surfaces_by_name,
                attached_indices,
                fold_cloth_max_pairs,
                fold_cloth_pair_distance,
            )
            if fold_pairs is not None:
                grasp_state["fold_pairs"] = fold_pairs
                _log(
                    "created compressed fold cloth-cloth lock with "
                    f"{int(fold_pairs['indices'].shape[0])} particle pairs"
                )

    project_metrics = _project_attached_patches(
        cloth_view,
        positions,
        velocities,
        particle_positions,
        surfaces_by_name,
        active_patches,
        grasp_state.get("fold_pairs"),
        TOY_PHYSICS_DT,
        nonanchor_velocity_damping,
        attached_velocity_mode,
        sticking_drive_mode,
        sticking_pd_kp,
        sticking_pd_kd,
        sticking_pd_max_speed,
    )
    if project_metrics:
        log_state.setdefault("pd_metrics", []).append(
            {
                "step": log_state["step"],
                "patches": project_metrics,
            }
        )
    return True


def _diagnose_surface_contacts(active_surfaces, particle_positions):
    import torch

    contacts = _surface_contacts_torch(active_surfaces, particle_positions)
    best_single = None
    for surface in contacts:
        contact = surface["contact"]
        min_normal_index = torch.argmin(contact["normal_distance"])
        candidate = {
            "name": surface["name"],
            "min_normal": float(
                contact["normal_distance"][min_normal_index].item()
            ),
            "tangent_at_min_normal": float(
                contact["tangent_distance"][min_normal_index].item()
            ),
            "radius": float(surface["radius"]),
            "mask_count": int(torch.count_nonzero(contact["mask"]).item()),
        }
        if best_single is None or candidate["min_normal"] < best_single["min_normal"]:
            best_single = candidate

    best_pair = None
    for first_index, first_surface in enumerate(contacts):
        for second_surface in contacts[first_index + 1:]:
            if first_surface["kind"] != "finger" and second_surface["kind"] != "finger":
                continue
            if first_surface["prim_path"] == second_surface["prim_path"]:
                continue
            first_pair_group = first_surface.get("pair_group")
            second_pair_group = second_surface.get("pair_group")
            if first_pair_group is not None and first_pair_group == second_pair_group:
                continue
            if not _surfaces_are_opposed(
                first_surface,
                second_surface,
                particle_positions,
            ):
                continue
            contact_mask = (
                first_surface["contact"]["mask"]
                & second_surface["contact"]["mask"]
            )
            pressed_mask = contact_mask & _pressed_between_surfaces_torch(
                first_surface["contact"],
                second_surface["contact"],
            )
            surface_gap = torch.linalg.norm(
                first_surface["contact"]["closest_points"]
                - second_surface["contact"]["closest_points"],
                dim=1,
            )
            contact_count = int(torch.count_nonzero(contact_mask).item())
            pressed_count = int(torch.count_nonzero(pressed_mask).item())
            gap_min = (
                float(torch.min(surface_gap[contact_mask]).item())
                if contact_count > 0
                else None
            )
            candidate = {
                "pair": f"{first_surface['name']}+{second_surface['name']}",
                "contact_count": contact_count,
                "pressed_count": pressed_count,
                "surface_gap_min": gap_min,
                "score": -pressed_count * 100000 - contact_count,
            }
            if best_pair is None or candidate["score"] < best_pair["score"]:
                best_pair = candidate

    return {
        "contact_count": int(best_pair["contact_count"]) if best_pair else 0,
        "pressed_count": int(best_pair["pressed_count"]) if best_pair else 0,
        "surface_gap_min": best_pair["surface_gap_min"] if best_pair else None,
        "best_pair": best_pair["pair"] if best_pair else None,
        "best_single": best_single,
    }


def _diagnose_named_surface_contacts(
    active_surfaces,
    particle_positions,
    particle_velocities,
    surface_names,
):
    import torch

    summaries = {}
    for surface in _surface_contacts_torch(active_surfaces, particle_positions):
        name = surface["name"]
        if name not in surface_names:
            continue
        contact = surface["contact"]
        mask = contact["mask"]
        mask_count = int(torch.count_nonzero(mask).item())
        summary = {
            "mask_count": mask_count,
            "min_signed_normal_m": float(
                torch.min(contact["signed_normal_distance"]).item()
            ),
            "min_abs_normal_m": float(torch.min(contact["normal_distance"]).item()),
            "contact_margin_m": float(surface["contact_margin"]),
            "half_extents_m": (
                [float(surface["half_extents"][0]), float(surface["half_extents"][1])]
                if surface.get("half_extents") is not None
                else None
            ),
        }
        if mask_count > 0:
            selected_velocities = particle_velocities[mask]
            local_velocities = selected_velocities @ contact["frame"]["rotation"]
            selected_signed = contact["signed_normal_distance"][mask]
            selected_tangent = contact["tangent_distance"][mask]
            summary.update(
                {
                    "contact_inside_count": int(
                        torch.count_nonzero(selected_signed < 0.0).item()
                    ),
                    "contact_signed_normal_min_m": float(
                        torch.min(selected_signed).item()
                    ),
                    "contact_signed_normal_max_m": float(
                        torch.max(selected_signed).item()
                    ),
                    "contact_tangent_max_m": float(torch.max(selected_tangent).item()),
                    "mean_normal_velocity_mps": float(
                        torch.mean(local_velocities[:, 2]).item()
                    ),
                    "mean_tangent_speed_mps": float(
                        torch.mean(torch.linalg.norm(local_velocities[:, :2], dim=1)).item()
                    ),
                    "max_tangent_speed_mps": float(
                        torch.max(torch.linalg.norm(local_velocities[:, :2], dim=1)).item()
                    ),
                }
            )
        summaries[name] = summary
    return summaries


def _summarize_pd_metrics(pd_metrics):
    if not pd_metrics:
        return None

    patch_metrics = [
        patch
        for step_metrics in pd_metrics
        for patch in step_metrics["patches"]
        if "position_error_mean_m" in patch
    ]
    if not patch_metrics:
        return None

    def _max_value(key):
        return max(float(patch[key]) for patch in patch_metrics)

    def _mean_value(key):
        return sum(float(patch[key]) for patch in patch_metrics) / len(patch_metrics)

    return {
        "samples": len(patch_metrics),
        "position_error_mean_m": _mean_value("position_error_mean_m"),
        "position_error_max_m": _max_value("position_error_max_m"),
        "correction_speed_mean_mps": _mean_value("correction_speed_mean_mps"),
        "correction_speed_max_mps": _max_value("correction_speed_max_mps"),
        "velocity_mean_mps": _mean_value("velocity_mean_mps"),
        "velocity_max_mps": _max_value("velocity_max_mps"),
        "speed_limited_fraction": (
            sum(
                int(patch["speed_limited_particles"])
                for patch in patch_metrics
            )
            / max(sum(int(patch["particles"]) for patch in patch_metrics), 1)
        ),
    }


def _setup_writer(camera_path: str, args):
    if not args.record:
        return None, None

    import omni.replicator.core as rep

    frame_dir = args.output_root / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(camera_path, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach([render_product])
    return rep, writer


def _compose_video(frame_dir: Path, video_path: Path, fps: int):
    import cv2

    rgb_paths = sorted(frame_dir.glob("rgb_*.png"))
    if not rgb_paths:
        rgb_paths = sorted(frame_dir.glob("**/rgb_*.png"))
    if not rgb_paths:
        return None

    video_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = cv2.imread(str(rgb_paths[0]))
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for rgb_path in rgb_paths:
            writer.write(cv2.imread(str(rgb_path)))
    finally:
        writer.release()
    return video_path


def _grasp_script_state(args, step: int) -> dict:
    lower_amount = (
        1.0
        if args.start_closed
        else (step - args.settle_steps) / max(args.lower_steps, 1)
    )
    close_amount = (
        1.0
        if args.start_closed
        else (
            step - args.settle_steps - args.lower_steps - args.preclose_lift_steps
        ) / max(args.close_steps, 1)
    )
    lift_amount = (
        step
        - args.settle_steps
        - args.lower_steps
        - args.preclose_lift_steps
        - args.close_steps
        - args.pinch_settle_steps
    ) / max(args.lift_steps, 1)
    release_amount = (
        step
        - args.settle_steps
        - args.lower_steps
        - args.preclose_lift_steps
        - args.close_steps
        - args.pinch_settle_steps
        - args.lift_steps
    ) / max(args.release_steps, 1)
    close_fraction = min(max(close_amount, 0.0), 1.0)
    release_fraction = min(max(release_amount, 0.0), 1.0)
    lower_fraction = min(max(lower_amount, 0.0), 1.0)
    closed_finger_y_offset = _lerp(
        FINGER_START_Y_OFFSET,
        args.finger_closed_y_offset,
        close_amount,
    )
    finger_y_offset = _lerp(
        closed_finger_y_offset,
        FINGER_START_Y_OFFSET,
        release_fraction,
    )
    press_z = _lerp(FINGER_START_Z, FINGER_PRESS_Z, lower_amount)
    if args.preclose_lift_steps > 0 and lower_fraction >= 1.0:
        preclose_lift_amount = (
            step - args.settle_steps - args.lower_steps
        ) / max(args.preclose_lift_steps, 1)
        close_z = _lerp(FINGER_PRESS_Z, FINGER_CLOSE_Z, preclose_lift_amount)
    else:
        close_z = FINGER_PRESS_Z if lower_fraction >= 1.0 else press_z
    prelift_finger_z = close_z
    lifted_finger_z = _lerp(prelift_finger_z, args.finger_lift_z, lift_amount)
    finger_z = _lerp(lifted_finger_z, args.finger_lift_z, release_fraction)
    pregrasp_active = (
        args.table_press_pregrasp
        and lower_fraction >= 1.0
        and close_fraction < 1.0
    )
    allow_new_attachment = (
        release_fraction <= 0.0
        and (
            pregrasp_active
            or (not args.attach_when_closed)
            or args.start_closed
            or close_fraction >= 1.0
        )
    )
    return {
        "finger_x": FINGER_CENTER_X,
        "finger_y_offset": finger_y_offset,
        "finger_z": finger_z,
        "lower_fraction": lower_fraction,
        "close_fraction": close_fraction,
        "release_fraction": release_fraction,
        "allow_new_attachment": allow_new_attachment,
        "upgrade_to_finger_pinch": (
            args.upgrade_pregrasp_to_pinch and close_fraction > 0.0
        ),
        "handoff_to_inner_patches": (
            args.handoff_pregrasp_to_inner_patches and close_fraction >= 0.85
        ),
        "fold_cloth_sticking": args.fold_cloth_sticking and close_fraction >= 1.0,
    }


def _table_slide_script_state(args, step: int) -> dict:
    settle_steps = max(args.settle_steps, args.table_slide_settle_steps)
    lower_amount = (step - settle_steps) / max(args.lower_steps, 1)
    slide_amount = (
        step - settle_steps - args.lower_steps
    ) / max(args.slide_steps, 1)
    release_amount = (
        step
        - settle_steps
        - args.lower_steps
        - args.slide_steps
        - args.slide_hold_steps
    ) / max(args.release_steps, 1)
    lower_fraction = min(max(lower_amount, 0.0), 1.0)
    slide_fraction = min(max(slide_amount, 0.0), 1.0)
    release_fraction = min(max(release_amount, 0.0), 1.0)
    pressed_finger_z = _lerp(
        TOY_TABLE_SLIDE_APPROACH_Z,
        args.slide_finger_z,
        lower_amount,
    )
    return {
        "finger_x": TOY_TABLE_SLIDE_START_X + args.slide_distance * slide_fraction,
        "finger_y_offset": FINGER_START_Y_OFFSET,
        "finger_z": _lerp(
            pressed_finger_z,
            TOY_TABLE_SLIDE_APPROACH_Z,
            release_fraction,
        ),
        "lower_fraction": lower_fraction,
        "close_fraction": 0.0,
        "release_fraction": release_fraction,
        "allow_new_attachment": (
            args.table_press_pregrasp
            and lower_fraction >= 1.0
            and release_fraction <= 0.0
        ),
        "upgrade_to_finger_pinch": False,
        "handoff_to_inner_patches": False,
        "fold_cloth_sticking": False,
    }


def _gravity_fold_script_state(args, step: int) -> dict:
    return {
        "finger_x": FINGER_CENTER_X,
        "finger_y_offset": FINGER_PARK_Y_OFFSET,
        "finger_z": FINGER_PARK_Z,
        "lower_fraction": 0.0,
        "close_fraction": 0.0,
        "release_fraction": 0.0,
        "allow_new_attachment": False,
        "upgrade_to_finger_pinch": False,
        "handoff_to_inner_patches": False,
        "fold_cloth_sticking": False,
    }


def _script_state(args, step: int) -> dict:
    if args.demo_mode == "table-slide":
        return _table_slide_script_state(args, step)
    if args.demo_mode == "gravity-fold":
        return _gravity_fold_script_state(args, step)
    return _grasp_script_state(args, step)


def main():
    args = _parse_args()
    global ADHESIVE_PRESS_GAP, ADHESIVE_SIGNED_NORMAL_SLOP
    global TOY_CLOTH_CONTACT_OFFSET, TOY_CLOTH_REST_OFFSET, TOY_STICKING_DEBUG
    ADHESIVE_PRESS_GAP = args.adhesive_press_gap
    TOY_CLOTH_REST_OFFSET = args.cloth_rest_offset
    TOY_CLOTH_CONTACT_OFFSET = args.cloth_contact_offset
    ADHESIVE_SIGNED_NORMAL_SLOP = args.cloth_contact_offset
    if args.sticking_drive_mode is None:
        args.sticking_drive_mode = "teleport"
    TOY_STICKING_DEBUG = args.sticking_debug
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_json = args.result_json or args.output_root / "result.json"

    simulation_app = _start_simulation_app(args)
    try:
        _enable_extensions("physx")
        stage, world = _create_toy_physx_world()

        left_finger_material = _create_material(
            stage,
            "/World/Materials/LeftFinger",
            (0.95, 0.05, 0.04),
        )
        right_finger_material = _create_material(
            stage,
            "/World/Materials/RightFinger",
            (0.95, 0.85, 0.05),
        )
        cloth_material = _create_material(stage, "/World/Materials/Cloth", (0.1, 0.45, 0.95))
        table_material = _create_material(stage, "/World/Materials/Table", (0.55, 0.40, 0.26))
        finger_physics_material = _create_physics_material(
            stage,
            "/World/Materials/FingerPhysics",
            args.finger_static_friction,
            args.finger_dynamic_friction,
            GRIPPER_FRICTION_COMBINE_MODE,
        )
        table_physics_material = _create_physics_material(
            stage,
            "/World/Materials/TablePhysics",
            args.table_static_friction,
            args.table_dynamic_friction,
            TABLE_FRICTION_COMBINE_MODE,
        )
        _create_table(stage, table_material, table_physics_material)

        initial_y_offset = (
            args.finger_closed_y_offset
            if args.start_closed
            else FINGER_START_Y_OFFSET
        )
        initial_finger_z = FINGER_PRESS_Z if args.start_closed else FINGER_START_Z
        left_finger = _create_finger(
            stage,
            world,
            "/World/LeftFinger",
            "left_finger",
            left_finger_material,
            finger_physics_material,
            (FINGER_CENTER_X, TABLE_CENTER[1] - initial_y_offset, initial_finger_z),
        )
        right_finger = _create_finger(
            stage,
            world,
            "/World/RightFinger",
            "right_finger",
            right_finger_material,
            finger_physics_material,
            (FINGER_CENTER_X, TABLE_CENTER[1] + initial_y_offset, initial_finger_z),
        )
        cloth = _create_tabletop_cloth(
            stage,
            world,
            cloth_material,
            initial_fold=args.demo_mode in ("gravity-fold", "table-slide"),
            self_collision=args.cloth_self_collision,
            self_collision_filter=args.cloth_self_collision_filter,
            particle_damping=args.cloth_particle_damping,
            spring_damping=args.cloth_spring_damping,
            max_velocity=args.cloth_max_velocity,
        )
        camera_path = _create_camera(
            stage,
            "/World/Camera",
            (0.34, TABLE_CENTER[1] - 0.12, 0.36),
            (TOY_CLOTH_CENTER[0], TOY_CLOTH_CENTER[1], 0.17),
            focal_length=22.0,
        )
        if args.viewport_camera or args.live_render:
            try:
                import omni.kit.viewport.utility

                viewport = omni.kit.viewport.utility.get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = camera_path
            except Exception as exc:
                _log(f"failed to set toy viewport camera: {exc}")

        world.reset()
        rep, writer = _setup_writer(camera_path, args)
        surfaces = _adhesive_surfaces(args)
        grasp_state = {"patches": []}
        log_state = {
            "step": 0,
            "attached_step": None,
            "attached_particles": 0,
            "released_step": None,
        }
        attached_centroid_z = []
        diagnostic_snapshots = []
        initial_cloth_centroid_z = None
        initial_cloth_centroid_m = None
        initial_front_cloth_centroid_m = None
        initial_cloth_span_m = None
        max_cloth_centroid_z = None
        max_cloth_span_z = None
        final_positions = None

        for step in range(args.steps):
            script_state = _script_state(args, step)
            finger_x = script_state["finger_x"]
            finger_y_offset = script_state["finger_y_offset"]
            finger_z = script_state["finger_z"]
            close_fraction = script_state["close_fraction"]
            release_fraction = script_state["release_fraction"]
            allow_new_attachment = script_state["allow_new_attachment"]
            adhesive_pair_mode = args.adhesive_pair_mode
            upgrade_to_finger_pinch = script_state["upgrade_to_finger_pinch"]
            handoff_to_inner_patches = script_state["handoff_to_inner_patches"]
            # Active patches are validated against the surface pair that created
            # them, so bottom/table pregrasp patches release when that contact
            # separates instead of being handed to inner-finger contact.
            release_stale_pregrasp = False
            _set_finger_pose(
                left_finger,
                (finger_x, TABLE_CENTER[1] - finger_y_offset, finger_z),
            )
            _set_finger_pose(
                right_finger,
                (finger_x, TABLE_CENTER[1] + finger_y_offset, finger_z),
            )

            log_state["step"] = step
            attached = False
            def _run_sticking_update():
                # Existing patches have a contact-based lifecycle: keep driving
                # them only while the same compressed surface-pair proxy that
                # created them still validates. Isaac's cloth tensor path does
                # not expose per-particle physical contact impulses here, so
                # this geometric proxy is the release condition until a PhysX
                # contact-force backend is available. Script phase only gates
                # creation of new patches.
                return (
                    _drive_attached_patch(
                        stage,
                        cloth,
                        surfaces,
                        grasp_state,
                        log_state,
                        args.expand_adhesive_patch,
                        adhesive_pair_mode,
                        args.adhesive_patch_max_particles,
                        args.adhesive_local_patch_radius,
                        args.adhesive_expanded_patch_radius,
                        args.adhesive_expanded_patch_max_particles,
                        args.adhesive_max_patches,
                        args.adhesive_components_per_pair,
                        args.adhesive_min_component_particles,
                        allow_new_attachment,
                        upgrade_to_finger_pinch,
                        handoff_to_inner_patches,
                        release_stale_pregrasp,
                        script_state["fold_cloth_sticking"],
                        args.fold_cloth_max_pairs,
                        args.fold_cloth_pair_distance,
                        args.pregrasp_patch_count,
                        args.pinch_patch_count,
                        args.attached_velocity_mode,
                        args.nonanchor_velocity_damping,
                        args.sticking_drive_mode,
                        args.sticking_pd_kp,
                        args.sticking_pd_kd,
                        args.sticking_pd_max_speed,
                        args.switch_sticking_on_contact_change,
                        args.refresh_sticking_patches,
                        args.validate_sticking_contact,
                        args.add_new_contact_patches,
                    )
                    if args.explicit_sticking
                    and (release_fraction <= 0.0 or grasp_state["patches"])
                    else False
                )

            if args.sticking_update_phase in ("before-step", "both"):
                had_patches = bool(grasp_state["patches"])
                attached = _run_sticking_update()
                if (
                    release_fraction > 0.0
                    and had_patches
                    and not grasp_state["patches"]
                    and log_state["released_step"] is None
                ):
                    grasp_state["fold_pairs"] = None
                    log_state["released_step"] = step
                    _log(f"released adhesive patches at step {step}")
            # Keep scripted control at the physics tick. world.step(render=True)
            # advances through the app/render cadence, so with rendering_dt >
            # physics_dt it can advance several physics ticks under one
            # kinematic finger target and destabilize cloth contact. A separate
            # render() call refreshes Fabric/render buffers without advancing
            # physics, so recorded frames still follow the particle state.
            should_render = args.live_render or args.record
            world.step(render=False, update_fabric=should_render)
            if should_render:
                world.render()
            if args.sticking_update_phase in ("after-step", "both"):
                had_patches = bool(grasp_state["patches"])
                attached = _run_sticking_update() or attached
                if (
                    release_fraction > 0.0
                    and had_patches
                    and not grasp_state["patches"]
                    and log_state["released_step"] is None
                ):
                    grasp_state["fold_pairs"] = None
                    log_state["released_step"] = step
                    _log(f"released adhesive patches at step {step}")
            _cloth_view, _positions, _velocities, particle_positions, _particle_velocities = (
                _torch_cloth_state(cloth)
            )
            final_positions = particle_positions.detach().clone()
            cloth_shape = _cloth_shape_diagnostics(particle_positions)
            cloth_centroid_z = float(torch.mean(particle_positions[:, 2]).item())
            cloth_span_z = cloth_shape["cloth_span_m"][2]
            max_cloth_centroid_z = (
                cloth_centroid_z
                if max_cloth_centroid_z is None
                else max(max_cloth_centroid_z, cloth_centroid_z)
            )
            max_cloth_span_z = (
                cloth_span_z
                if max_cloth_span_z is None
                else max(max_cloth_span_z, cloth_span_z)
            )
            diagnostic_interval = max(args.diagnostic_interval, 1)
            if step % diagnostic_interval == 0 or step == args.steps - 1:
                active_surfaces = _active_adhesive_surfaces(
                    stage,
                    surfaces,
                    {"pinch": True},
                )
                diagnostic = {
                    "step": step,
                    "finger_x": finger_x,
                    "finger_y_offset": finger_y_offset,
                    "finger_z": finger_z,
                    **cloth_shape,
                    "active_patch_summaries": _attached_patch_summaries(
                        grasp_state["patches"],
                        particle_positions,
                    )
                    if grasp_state["patches"]
                    else [],
                    "active_patch_contact_diagnostics": (
                        _active_patch_contact_diagnostics(
                            active_surfaces,
                            particle_positions,
                            grasp_state["patches"],
                        )
                        if grasp_state["patches"]
                        else []
                    ),
                    **_attachment_centroid_diagnostics(
                        grasp_state["patches"],
                        particle_positions,
                    ),
                    **_extreme_particle_diagnostics(
                        particle_positions,
                        _particle_velocities,
                        grasp_state["patches"],
                    ),
                    **_diagnose_surface_contacts(active_surfaces, particle_positions),
                    "named_surface_contacts": _diagnose_named_surface_contacts(
                        active_surfaces,
                        particle_positions,
                        _particle_velocities,
                        {
                            "left_inner_face",
                            "right_inner_face",
                            "left_outer_face",
                            "right_outer_face",
                        },
                    ),
                }
                diagnostic_snapshots.append(diagnostic)
                _log("diagnostic " + json.dumps(diagnostic, sort_keys=True))
            if initial_cloth_centroid_z is None:
                initial_cloth_centroid_z = cloth_centroid_z
                initial_cloth_centroid_m = cloth_shape["cloth_centroid_m"]
                initial_front_cloth_centroid_m = cloth_shape[
                    "front_cloth_centroid_m"
                ]
                initial_cloth_span_m = cloth_shape["cloth_span_m"]
            if attached and grasp_state["patches"]:
                indices = torch.unique(
                    torch.cat([patch["indices"] for patch in grasp_state["patches"]])
                )
                attached_centroid_z.append(
                    float(torch.mean(particle_positions[indices, 2]).item())
                )

            if rep is not None:
                # Capture the state produced by world.step() without letting
                # Replicator advance the physics timeline a second time.
                rep.orchestrator.step(
                    rt_subframes=args.rt_subframes,
                    pause_timeline=False,
                    delta_time=0.0,
                )
            if args.live_render and args.live_step_seconds > 0.0:
                time.sleep(args.live_step_seconds)

        if rep is not None:
            rep.orchestrator.wait_until_complete()
        if writer is not None:
            writer.detach()

        if final_positions is None:
            _cloth_view, _positions, _velocities, final_positions, _particle_velocities = (
                _torch_cloth_state(cloth)
            )
        final_cloth_shape = _cloth_shape_diagnostics(final_positions)
        final_cloth_centroid_z = float(torch.mean(final_positions[:, 2]).item())
        final_attached_centroid_z = (
            attached_centroid_z[-1] if attached_centroid_z else None
        )
        initial_attached_centroid_z = (
            attached_centroid_z[0] if attached_centroid_z else None
        )
        attached_patch_span_m = None
        attached_patch_summaries = []
        if grasp_state["patches"]:
            attached_indices = torch.unique(
                torch.cat([patch["indices"] for patch in grasp_state["patches"]])
            )
            final_attached_positions = final_positions[
                attached_indices
            ]
            attached_patch_span = (
                torch.max(final_attached_positions, dim=0).values
                - torch.min(final_attached_positions, dim=0).values
            )
            attached_patch_span_m = [float(value.item()) for value in attached_patch_span]
            attached_patch_summaries = _attached_patch_summaries(
                grasp_state["patches"],
                final_positions,
            )
        attached_lift_m = (
            final_attached_centroid_z - initial_attached_centroid_z
            if final_attached_centroid_z is not None
            else 0.0
        )
        cloth_lift_m = final_cloth_centroid_z - initial_cloth_centroid_z
        cloth_slide_m = (
            final_cloth_shape["cloth_centroid_m"][0] - initial_cloth_centroid_m[0]
        )
        front_cloth_slide_m = (
            final_cloth_shape["front_cloth_centroid_m"][0]
            - initial_front_cloth_centroid_m[0]
        )
        cloth_lateral_drift_m = (
            final_cloth_shape["cloth_centroid_m"][1] - initial_cloth_centroid_m[1]
        )
        cloth_span_z_growth_m = (
            max_cloth_span_z - initial_cloth_span_m[2]
            if initial_cloth_span_m is not None and max_cloth_span_z is not None
            else None
        )
        max_cloth_lift_m = max_cloth_centroid_z - initial_cloth_centroid_z
        success_cloth_lift_m = (
            max_cloth_lift_m
            if log_state["released_step"] is not None
            else cloth_lift_m
        )
        attached_to_cloth_slip_m = attached_lift_m - cloth_lift_m
        whole_cloth_follow_ratio = (
            cloth_lift_m / attached_lift_m
            if abs(attached_lift_m) > 1e-9
            else None
        )
        if args.demo_mode == "table-slide":
            success = bool(
                cloth_slide_m > 0.050
                and abs(cloth_lateral_drift_m) < 0.020
                and cloth_span_z_growth_m is not None
                and cloth_span_z_growth_m < 0.080
            )
        elif args.demo_mode == "gravity-fold":
            success = bool(
                max_cloth_span_z is not None
                and max_cloth_span_z < 0.050
                and final_cloth_shape["cloth_span_m"][2] < 0.040
            )
        else:
            success = bool(
                (
                    log_state["attached_step"] is not None
                    and attached_lift_m > 0.075
                    and success_cloth_lift_m > 0.025
                    and max_cloth_span_z < 1.0
                )
                if args.explicit_sticking
                else (
                    max_cloth_lift_m > 0.075
                    and success_cloth_lift_m > 0.025
                    and max_cloth_span_z < 1.0
                )
            )

        video_path = None
        if args.record:
            video_path = _compose_video(
                args.output_root / "frames",
                args.output_root / "toy_cloth_grasp.mp4",
                args.fps,
            )

        result = {
            "success": success,
            "demo_mode": args.demo_mode,
            "steps": args.steps,
            "start_closed": args.start_closed,
            "explicit_sticking": args.explicit_sticking,
            "adhesive_pair_mode": args.adhesive_pair_mode,
            "adhesive_max_patches": args.adhesive_max_patches,
            "adhesive_components_per_pair": args.adhesive_components_per_pair,
            "adhesive_min_component_particles": (
                args.adhesive_min_component_particles
            ),
            "pregrasp_patch_count": args.pregrasp_patch_count,
            "pinch_patch_count": args.pinch_patch_count,
            "attached_velocity_mode": args.attached_velocity_mode,
            "nonanchor_velocity_damping": args.nonanchor_velocity_damping,
            "sticking_update_phase": args.sticking_update_phase,
            "switch_sticking_on_contact_change": (
                args.switch_sticking_on_contact_change
            ),
            "refresh_sticking_patches": args.refresh_sticking_patches,
            "validate_sticking_contact": args.validate_sticking_contact,
            "add_new_contact_patches": args.add_new_contact_patches,
            "sticking_drive_mode": args.sticking_drive_mode,
            "sticking_pd_kp": args.sticking_pd_kp,
            "sticking_pd_kd": args.sticking_pd_kd,
            "sticking_pd_max_speed": args.sticking_pd_max_speed,
            "finger_lift_z": args.finger_lift_z,
            "slide_distance": args.slide_distance,
            "slide_steps": args.slide_steps,
            "slide_hold_steps": args.slide_hold_steps,
            "slide_finger_z": args.slide_finger_z,
            "finger_size_m": list(FINGER_SIZE),
            "finger_closed_y_offset": args.finger_closed_y_offset,
            "expand_adhesive_patch": args.expand_adhesive_patch,
            "adhesive_patch_max_particles": args.adhesive_patch_max_particles,
            "adhesive_press_gap": args.adhesive_press_gap,
            "adhesive_local_patch_radius": args.adhesive_local_patch_radius,
            "adhesive_expanded_patch_radius": args.adhesive_expanded_patch_radius,
            "adhesive_expanded_patch_max_particles": (
                args.adhesive_expanded_patch_max_particles
            ),
            "cloth_rest_offset": TOY_CLOTH_REST_OFFSET,
            "cloth_contact_offset": TOY_CLOTH_CONTACT_OFFSET,
            "cloth_self_collision": args.cloth_self_collision,
            "cloth_self_collision_filter": args.cloth_self_collision_filter,
            "cloth_particle_damping": args.cloth_particle_damping,
            "cloth_spring_damping": args.cloth_spring_damping,
            "cloth_max_velocity": args.cloth_max_velocity,
            "cloth_grid_x": TOY_CLOTH_GRID_X,
            "cloth_grid_y": TOY_CLOTH_GRID_Y,
            "cloth_particle_spacing_m": list(TOY_CLOTH_PARTICLE_SPACING),
            "cloth_stretch_stiffness": TOY_CLOTH_STRETCH_STIFFNESS,
            "cloth_shear_stiffness": TOY_CLOTH_SHEAR_STIFFNESS,
            "cloth_bend_stiffness": TOY_CLOTH_BEND_STIFFNESS,
            "finger_static_friction": args.finger_static_friction,
            "finger_dynamic_friction": args.finger_dynamic_friction,
            "table_static_friction": args.table_static_friction,
            "table_dynamic_friction": args.table_dynamic_friction,
            "attached_step": log_state["attached_step"],
            "attached_particles": log_state["attached_particles"],
            "released_step": log_state["released_step"],
            "initial_cloth_centroid_z": initial_cloth_centroid_z,
            "initial_cloth_centroid_m": initial_cloth_centroid_m,
            "initial_front_cloth_centroid_m": initial_front_cloth_centroid_m,
            "initial_cloth_span_m": initial_cloth_span_m,
            "final_cloth_centroid_z": final_cloth_centroid_z,
            "final_cloth_centroid_m": final_cloth_shape["cloth_centroid_m"],
            "final_front_cloth_centroid_m": final_cloth_shape[
                "front_cloth_centroid_m"
            ],
            "final_cloth_span_m": final_cloth_shape["cloth_span_m"],
            "max_cloth_centroid_z": max_cloth_centroid_z,
            "max_cloth_span_z_m": max_cloth_span_z,
            "cloth_lift_m": cloth_lift_m,
            "cloth_slide_m": cloth_slide_m,
            "front_cloth_slide_m": front_cloth_slide_m,
            "cloth_lateral_drift_m": cloth_lateral_drift_m,
            "cloth_span_z_growth_m": cloth_span_z_growth_m,
            "max_cloth_lift_m": max_cloth_lift_m,
            "success_cloth_lift_m": success_cloth_lift_m,
            "initial_attached_centroid_z": initial_attached_centroid_z,
            "final_attached_centroid_z": final_attached_centroid_z,
            "attached_lift_m": attached_lift_m,
            "attached_to_cloth_slip_m": attached_to_cloth_slip_m,
            "whole_cloth_follow_ratio": whole_cloth_follow_ratio,
            "pd_metric_summary": _summarize_pd_metrics(
                log_state.get("pd_metrics", [])
            ),
            "pd_metrics_tail": log_state.get("pd_metrics", [])[-20:],
            "attached_patch_span_m": attached_patch_span_m,
            "attached_patch_summaries": attached_patch_summaries,
            "diagnostics": diagnostic_snapshots,
            "video_path": str(video_path) if video_path is not None else None,
            "result_json": str(result_json),
        }
        result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _log(json.dumps(result, indent=2))
        while args.hold_open and simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
