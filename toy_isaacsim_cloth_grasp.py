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

from isaacsim_newton_scene import (
    ADHESIVE_CONTACT_MARGIN,
    CLOTH_ADHESION_OFFSET_SCALE,
    CLOTH_CONTACT_OFFSET,
    CLOTH_PARTICLE_ADHESION,
    CLOTH_PARTICLE_ADHESION_SCALE,
    CLOTH_PARTICLE_DAMPING,
    CLOTH_PARTICLE_FRICTION,
    CLOTH_PARTICLE_FRICTION_SCALE,
    CLOTH_REST_OFFSET,
    GRIPPER_FINGER_COLLISION_BOX_SIZE,
    GRIPPER_PHYSICS_FRICTION,
    GRIPPER_FRICTION_COMBINE_MODE,
    OBJECT_ADHESIVE_SURFACE_FRICTION,
    TABLE_CENTER,
    TABLE_FRICTION_COMBINE_MODE,
    TABLE_PHYSICS_FRICTION,
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
FINGER_SIZE = (
    GRIPPER_FINGER_COLLISION_BOX_SIZE[0],
    GRIPPER_FINGER_COLLISION_BOX_SIZE[2],
    GRIPPER_FINGER_COLLISION_BOX_SIZE[1],
)
TOY_CLOTH_CENTER = (-0.15, TABLE_CENTER[1], TABLE_CENTER[2] + 0.075)
FINGER_CENTER_X = TOY_CLOTH_CENTER[0]
FINGER_START_Y_OFFSET = 0.070
# Keep a physical gap for folded cloth while preserving finger normal force.
FINGER_CLOSED_Y_OFFSET = 0.019
FINGER_PRESS_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.046
FINGER_START_Z = FINGER_PRESS_Z + 0.030
FINGER_CLOSE_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.070
FINGER_LIFT_Z = TABLE_CENTER[2] + 0.5 * TABLE_SCALE[2] + 0.220
FINGER_INNER_ATTACH_NORMAL_OFFSET = 0.006
ADHESIVE_PRESS_GAP = 0.024
TOY_CLOTH_REST_OFFSET = 0.005
TOY_CLOTH_CONTACT_OFFSET = 0.005
ADHESIVE_SIGNED_NORMAL_SLOP = TOY_CLOTH_CONTACT_OFFSET
TOY_CLOTH_TOTAL_MASS = 0.050
TOY_CLOTH_GRID_COLUMNS = 49 + 1
TOY_CLOTH_GRID_ROWS = 33 + 1
TOY_CLOTH_PARTICLE_COUNT = TOY_CLOTH_GRID_COLUMNS * TOY_CLOTH_GRID_ROWS
TOY_CLOTH_PARTICLE_MASS = TOY_CLOTH_TOTAL_MASS / TOY_CLOTH_PARTICLE_COUNT
TOY_CLOTH_STRETCH_STIFFNESS = 6000.0
TOY_CLOTH_BEND_STIFFNESS = 35.0
TOY_CLOTH_SHEAR_STIFFNESS = 3000.0
TOY_CLOTH_SPRING_DAMPING = 8.0
TOY_CLOTH_SOLVER_POSITION_ITERATIONS = 96
TOY_NONANCHOR_VELOCITY_DAMPING = 0.94
TOY_FOLD_PAIR_STRENGTH = 0.60
TOY_PHYSICS_DT = 1.0 / 120.0


def _log(message: str):
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
        "--adhesive-patch-max-particles",
        type=int,
        default=64,
        help="Maximum compressed contact particles to attach before optional expansion.",
    )
    parser.add_argument(
        "--adhesive-press-gap",
        type=float,
        default=ADHESIVE_PRESS_GAP,
        help="Maximum gap between opposing surfaces for a compressed adhesive contact.",
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
        default="teleport",
        help="Drive adhesive particles by position projection or by PD velocity updates.",
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
        "--nonanchor-velocity-damping",
        type=float,
        default=TOY_NONANCHOR_VELOCITY_DAMPING,
        help="Velocity multiplier for particles outside explicitly driven patches.",
    )
    parser.add_argument(
        "--adhesive-finger-friction",
        type=float,
        default=GRIPPER_PHYSICS_FRICTION[0],
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
        default=TABLE_PHYSICS_FRICTION[0],
        help="Toy table static friction used by the physics material.",
    )
    parser.add_argument(
        "--table-dynamic-friction",
        type=float,
        default=TABLE_PHYSICS_FRICTION[1],
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


def _create_tabletop_cloth(stage, world, material):
    from isaacsim.core.api.materials.particle_material import ParticleMaterial
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, "/World/ToyCloth")
    points, counts, indices = _grid_mesh(0.33, 0.22, 49, 33, 0.002)
    translated_points = [
        Gf.Vec3f(
            float(x + TOY_CLOTH_CENTER[0]),
            float(y + TOY_CLOTH_CENTER[1]),
            float(z + TOY_CLOTH_CENTER[2]),
        )
        for x, y, z in points
    ]
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
        damping=CLOTH_PARTICLE_DAMPING,
        adhesion=CLOTH_PARTICLE_ADHESION,
        particle_adhesion_scale=CLOTH_PARTICLE_ADHESION_SCALE,
        adhesion_offset_scale=CLOTH_ADHESION_OFFSET_SCALE,
    )
    _log(
        "created toy cloth PBD material: "
        f"friction={CLOTH_PARTICLE_FRICTION} "
        f"particle_friction_scale={CLOTH_PARTICLE_FRICTION_SCALE} "
        f"damping={CLOTH_PARTICLE_DAMPING} "
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
        max_velocity=10.0,
        global_self_collision_enabled=True,
        non_particle_collision_enabled=True,
    )
    particle_system.set_simulation_owner(world.get_physics_context().prim_path)
    cloth = SingleClothPrim(
        prim_path="/World/ToyCloth",
        particle_system=particle_system,
        particle_material=particle_material,
        name="toy_cloth",
        particle_mass=TOY_CLOTH_PARTICLE_MASS,
        self_collision=True,
        self_collision_filter=True,
        stretch_stiffness=TOY_CLOTH_STRETCH_STIFFNESS,
        bend_stiffness=TOY_CLOTH_BEND_STIFFNESS,
        shear_stiffness=TOY_CLOTH_SHEAR_STIFFNESS,
        spring_damping=TOY_CLOTH_SPRING_DAMPING,
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
    import torch

    device = particle_positions.device
    dtype = particle_positions.dtype
    frame = surface["frame"]
    return {
        "origin": torch.as_tensor(frame["origin"], dtype=dtype, device=device),
        "rotation": torch.as_tensor(frame["rotation"], dtype=dtype, device=device),
    }


def _surface_contact_torch(surface, particle_positions):
    frame = _surface_frame_tensors(surface, particle_positions)
    local_positions = (particle_positions - frame["origin"]) @ frame["rotation"]
    tangent_distance = torch.linalg.norm(local_positions[:, :2], dim=1)
    normal_distance = torch.abs(local_positions[:, 2])
    closest_points = (
        particle_positions
        - frame["rotation"][:, 2] * local_positions[:, 2:3]
    )
    half_extents = surface.get("half_extents")
    if half_extents is None:
        tangent_mask = tangent_distance <= float(surface["radius"])
    else:
        tangent_mask = (
            (torch.abs(local_positions[:, 0]) <= float(half_extents[0]))
            & (torch.abs(local_positions[:, 1]) <= float(half_extents[1]))
        )
    mask = (normal_distance <= float(surface["contact_margin"])) & tangent_mask
    score = normal_distance + 0.25 * tangent_distance
    return {
        "frame": frame,
        "local_positions": local_positions,
        "tangent_distance": tangent_distance,
        "signed_normal_distance": local_positions[:, 2],
        "normal_distance": normal_distance,
        "closest_points": closest_points,
        "mask": mask,
        "score": score,
    }


def _surface_contacts_torch(active_surfaces, particle_positions):
    return [
        {
            **surface,
            "torch_frame": _surface_frame_tensors(surface, particle_positions),
            "contact": _surface_contact_torch(surface, particle_positions),
        }
        for surface in active_surfaces
    ]


def _surfaces_are_opposed(first_surface, second_surface, particle_positions):
    first_normal = first_surface["torch_frame"]["rotation"][:, 2]
    second_normal = second_surface["torch_frame"]["rotation"][:, 2]
    return bool(torch.dot(first_normal, second_normal).item() < -0.35)


def _pressed_between_surfaces_torch(first_contact, second_contact):
    import torch

    surface_gap = torch.linalg.norm(
        first_contact["closest_points"] - second_contact["closest_points"],
        dim=1,
    )
    between_normals = (
        first_contact["signed_normal_distance"] >= -ADHESIVE_SIGNED_NORMAL_SLOP
    ) & (
        second_contact["signed_normal_distance"] >= -ADHESIVE_SIGNED_NORMAL_SLOP
    )
    return (surface_gap <= ADHESIVE_PRESS_GAP) & between_normals


def _surface_candidate_distance(surface, candidate_indices):
    if candidate_indices is None or candidate_indices.numel() == 0:
        return float("inf")
    return float(
        torch.mean(surface["contact"]["normal_distance"][candidate_indices]).item()
    )


def _choose_adhesive_anchor(first_surface, second_surface, candidate_indices):
    return max(
        (first_surface, second_surface),
        key=lambda surface: (
            surface.get("adhesion", 0.0),
            surface.get("friction", 0.0),
            -_surface_candidate_distance(surface, candidate_indices),
        ),
    )


def _is_finger_pinch_mode(mode: str) -> bool:
    return set(mode.split("+")) == {"left_inner_face", "right_inner_face"}


def _expand_adhesive_patch_indices(
    particle_positions,
    seed_indices,
    anchor_frame,
    expanded_patch_radius: float,
    expanded_patch_max_particles: int,
    eligible_indices,
):
    if seed_indices.numel() == 0:
        return seed_indices

    local_positions = (
        particle_positions - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    seed_local_positions = local_positions[seed_indices]
    tangent_center = torch.mean(seed_local_positions[:, :2], dim=0)
    tangent_distance = torch.linalg.norm(
        local_positions[:, :2] - tangent_center,
        dim=1,
    )
    eligible_mask = torch.zeros(
        particle_positions.shape[0],
        dtype=torch.bool,
        device=particle_positions.device,
    )
    eligible_mask[eligible_indices] = True
    expanded_mask = (
        tangent_distance <= float(expanded_patch_radius)
    ) & eligible_mask
    expanded_indices = torch.nonzero(expanded_mask, as_tuple=False).flatten()
    if expanded_indices.numel() <= seed_indices.numel():
        return seed_indices

    if expanded_indices.numel() > int(expanded_patch_max_particles):
        order = torch.argsort(tangent_distance[expanded_indices])[
            :int(expanded_patch_max_particles)
        ]
        expanded_indices = expanded_indices[order]
    return torch.unique(torch.cat((seed_indices, expanded_indices)))


def _connected_component_indices_grid(mask, max_components: int):
    if torch.count_nonzero(mask).item() == 0:
        return []
    if mask.numel() != TOY_CLOTH_PARTICLE_COUNT:
        return [torch.nonzero(mask, as_tuple=False).flatten()]

    rows = TOY_CLOTH_GRID_ROWS
    columns = TOY_CLOTH_GRID_COLUMNS
    inactive_label = mask.numel()
    active_grid = mask.reshape(rows, columns)
    labels = torch.arange(mask.numel(), dtype=torch.long, device=mask.device).reshape(
        rows,
        columns,
    )
    inactive_labels = torch.full_like(labels, inactive_label)
    labels = torch.where(active_grid, labels, inactive_labels)
    for _iteration in range(rows + columns):
        next_labels = labels.clone()
        next_labels[1:, :] = torch.minimum(next_labels[1:, :], labels[:-1, :])
        next_labels[:-1, :] = torch.minimum(next_labels[:-1, :], labels[1:, :])
        next_labels[:, 1:] = torch.minimum(next_labels[:, 1:], labels[:, :-1])
        next_labels[:, :-1] = torch.minimum(next_labels[:, :-1], labels[:, 1:])
        labels = torch.where(active_grid, next_labels, inactive_labels)

    flat_labels = labels.flatten()
    active_labels = flat_labels[mask]
    component_labels, component_counts = torch.unique(
        active_labels,
        sorted=False,
        return_counts=True,
    )
    count_order = torch.argsort(component_counts, descending=True)
    retained_labels = component_labels[count_order[:max_components]]
    return [
        torch.nonzero(flat_labels == component_label, as_tuple=False).flatten()
        for component_label in retained_labels
    ]


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
    if candidate_indices.numel() == 0:
        return None
    anchor = _choose_adhesive_anchor(
        first_surface,
        second_surface,
        candidate_indices,
    )
    if required_anchor_name is not None:
        surfaces_by_name = {
            first_surface["name"]: first_surface,
            second_surface["name"]: second_surface,
        }
        anchor = surfaces_by_name.get(required_anchor_name)
        if anchor is None:
            return None
    anchor_frame = anchor["torch_frame"]
    component_pair_score = pair_score[candidate_indices]
    candidate_local_positions = (
        particle_positions[candidate_indices] - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    if adhesive_local_patch_radius > 0.0:
        best_pair_order = torch.argsort(component_pair_score)
        best_local_position = candidate_local_positions[best_pair_order[0]]
        local_distances = torch.linalg.norm(
            candidate_local_positions[:, :2] - best_local_position[:2],
            dim=1,
        )
        local_mask = local_distances <= adhesive_local_patch_radius
        local_candidate_indices = candidate_indices[local_mask]
        local_pair_score = component_pair_score[local_mask]
    else:
        local_candidate_indices = candidate_indices
        local_pair_score = component_pair_score
    if local_candidate_indices.numel() == 0:
        return None
    local_order = torch.argsort(local_pair_score)[:adhesive_patch_max_particles]
    seed_indices = local_candidate_indices[local_order]
    selected_indices = (
        _expand_adhesive_patch_indices(
            particle_positions,
            seed_indices,
            anchor_frame,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            candidate_indices,
        )
        if expand_adhesive_patch
        else seed_indices
    )
    local_positions = (
        particle_positions[selected_indices] - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    required_side = (
        first_surface.get("requires_closed_side")
        or second_surface.get("requires_closed_side")
    )
    return {
        "mode": f"{first_surface['name']}+{second_surface['name']}",
        "anchor_name": anchor["name"],
        "anchor_kind": anchor["kind"],
        "indices": selected_indices,
        "local_positions": local_positions,
        "requires_closed_side": required_side,
        "score": torch.mean(local_pair_score[local_order]),
        "seed_count": int(seed_indices.numel()),
        "candidate_count": int(candidate_indices.numel()),
    }


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
    required_anchor_name: str | None = None,
    excluded_indices=None,
):
    import torch

    contacts = _surface_contacts_torch(active_surfaces, particle_positions)
    excluded_mask = torch.zeros(
        particle_positions.shape[0],
        dtype=torch.bool,
        device=particle_positions.device,
    )
    if excluded_indices is not None and excluded_indices.numel() > 0:
        excluded_mask[excluded_indices] = True
    patches = []
    for first_index, first_surface in enumerate(contacts):
        for second_surface in contacts[first_index + 1:]:
            if first_surface["kind"] != "finger" and second_surface["kind"] != "finger":
                continue
            if adhesive_pair_mode == "finger-pinch":
                surface_names = {first_surface["name"], second_surface["name"]}
                if surface_names != {"left_inner_face", "right_inner_face"}:
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
            required_side = (
                first_surface.get("requires_closed_side")
                or second_surface.get("requires_closed_side")
            )
            if first_surface.get("requires_closed_side") not in (None, "pinch"):
                continue
            if second_surface.get("requires_closed_side") not in (None, "pinch"):
                continue
            combined_mask = (
                first_surface["contact"]["mask"]
                & second_surface["contact"]["mask"]
                & _pressed_between_surfaces_torch(
                    first_surface["contact"],
                    second_surface["contact"],
                )
                & ~excluded_mask
            )
            pair_score = (
                first_surface["contact"]["score"]
                + second_surface["contact"]["score"]
            )
            component_indices = _connected_component_indices_grid(
                combined_mask,
                adhesive_components_per_pair,
            )
            for candidate_indices in component_indices:
                patch = _make_adhesive_patch(
                    first_surface,
                    second_surface,
                    particle_positions,
                    candidate_indices,
                    pair_score,
                    expand_adhesive_patch,
                    adhesive_patch_max_particles,
                    adhesive_local_patch_radius,
                    adhesive_expanded_patch_radius,
                    adhesive_expanded_patch_max_particles,
                    required_anchor_name,
                )
                if patch is not None:
                    patches.append(patch)
    patches.sort(
        key=lambda patch: (
            -int(patch["indices"].numel()),
            float(patch["score"].item()),
            patch["mode"],
            patch["anchor_name"],
        )
    )
    return patches[:adhesive_max_patches]


def _choose_adhesive_patch_torch(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    required_anchor_name: str | None = None,
    excluded_indices=None,
):
    patches = _choose_adhesive_patches_torch(
        active_surfaces,
        particle_positions,
        expand_adhesive_patch,
        adhesive_pair_mode,
        adhesive_patch_max_particles,
        adhesive_local_patch_radius,
        adhesive_expanded_patch_radius,
        adhesive_expanded_patch_max_particles,
        1,
        1,
        required_anchor_name,
        excluded_indices,
    )
    return patches[0] if patches else None


def _choose_two_anchor_patches(
    active_surfaces,
    particle_positions,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
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
    position_targets = positions.clone()
    velocity_targets = velocities.clone() * float(nonanchor_velocity_damping)
    metrics = []
    for patch in patches:
        anchor_surface = surfaces_by_name[patch["anchor_name"]]
        anchor_frame = _surface_frame_tensors(anchor_surface, particle_positions)
        target_positions = (
            patch["local_positions"] @ anchor_frame["rotation"].T
        ) + anchor_frame["origin"]
        selected_indices = patch["indices"]
        previous_target_positions = patch.get("previous_target_positions")
        if attached_velocity_mode == "zero":
            anchor_velocities = torch.zeros_like(target_positions)
        elif (
            previous_target_positions is not None
            and previous_target_positions.shape == target_positions.shape
        ):
            anchor_velocities = (
                target_positions - previous_target_positions
            ) / float(physics_dt)
        else:
            anchor_velocities = torch.zeros_like(target_positions)
        if sticking_drive_mode == "teleport":
            position_targets[0, selected_indices] = target_positions
            velocity_targets[0, selected_indices] = anchor_velocities
            metrics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                }
            )
        else:
            selected_positions = particle_positions[selected_indices]
            selected_velocities = velocities[0, selected_indices]
            position_errors = target_positions - selected_positions
            velocity_errors = anchor_velocities - selected_velocities
            position_response = min(
                max(float(sticking_pd_kp) * float(physics_dt), 0.0),
                1.0,
            )
            velocity_response = min(
                max(float(sticking_pd_kd) * float(physics_dt), 0.0),
                1.0,
            )
            raw_corrections = position_response * position_errors
            correction_speed = torch.linalg.norm(
                raw_corrections,
                dim=1,
                keepdim=True,
            ) / float(physics_dt)
            speed_scale = torch.clamp(
                float(sticking_pd_max_speed)
                / torch.clamp(correction_speed, min=1e-6),
                max=1.0,
            )
            position_corrections = raw_corrections * speed_scale
            next_positions = selected_positions + position_corrections
            correction_velocities = position_corrections / float(physics_dt)
            damping_velocities = velocity_response * velocity_errors
            next_velocities = (
                anchor_velocities
                + correction_velocities
                + damping_velocities
            )
            next_speed = torch.linalg.norm(next_velocities, dim=1, keepdim=True)
            next_speed_scale = torch.clamp(
                float(sticking_pd_max_speed)
                / torch.clamp(next_speed, min=1e-6),
                max=1.0,
            )
            position_targets[0, selected_indices] = next_positions
            velocity_targets[0, selected_indices] = next_velocities * next_speed_scale
            metrics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                    "position_error_mean_m": float(
                        torch.mean(torch.linalg.norm(position_errors, dim=1)).item()
                    ),
                    "position_error_max_m": float(
                        torch.max(torch.linalg.norm(position_errors, dim=1)).item()
                    ),
                    "correction_speed_mean_mps": float(
                        torch.mean(correction_speed).item()
                    ),
                    "correction_speed_max_mps": float(
                        torch.max(correction_speed).item()
                    ),
                    "velocity_mean_mps": float(torch.mean(next_speed).item()),
                    "velocity_max_mps": float(torch.max(next_speed).item()),
                    "speed_limited_particles": int(
                        torch.count_nonzero(next_speed_scale[:, 0] < 0.999).item()
                    ),
                }
            )
        patch["previous_target_positions"] = target_positions.detach().clone()
    _project_fold_pairs(position_targets, velocity_targets, fold_pairs, physics_dt)
    cloth_view.set_world_positions(position_targets)
    cloth_view.set_velocities(velocity_targets)
    return metrics


def _attached_patch_summaries(patches, particle_positions):
    summaries = []
    for patch in patches:
        patch_positions = particle_positions[patch["indices"]]
        patch_span = (
            torch.max(patch_positions, dim=0).values
            - torch.min(patch_positions, dim=0).values
        )
        summaries.append(
            {
                "mode": patch["mode"],
                "anchor_name": patch["anchor_name"],
                "particles": int(patch["indices"].numel()),
                "seed_count": patch.get("seed_count"),
                "candidate_count": patch.get("candidate_count"),
                "span_m": [float(value.item()) for value in patch_span],
            }
        )
    return summaries


def _surface_selected_contact_diagnostic(surface, selected_indices):
    contact = surface["contact"]
    local_positions = contact["local_positions"][selected_indices]
    half_extents = surface.get("half_extents")
    normal_mask = (
        contact["normal_distance"][selected_indices]
        <= float(surface["contact_margin"])
    )
    if half_extents is None:
        tangent_mask = (
            contact["tangent_distance"][selected_indices]
            <= float(surface["radius"])
        )
    else:
        tangent_mask = (
            (torch.abs(local_positions[:, 0]) <= float(half_extents[0]))
            & (torch.abs(local_positions[:, 1]) <= float(half_extents[1]))
        )
    return {
        "name": surface["name"],
        "mask_count": int(
            torch.count_nonzero(contact["mask"][selected_indices]).item()
        ),
        "normal_count": int(torch.count_nonzero(normal_mask).item()),
        "tangent_count": int(torch.count_nonzero(tangent_mask).item()),
        "normal_abs_max_m": float(
            torch.max(contact["normal_distance"][selected_indices]).item()
        ),
        "signed_normal_min_m": float(
            torch.min(contact["signed_normal_distance"][selected_indices]).item()
        ),
        "signed_normal_max_m": float(
            torch.max(contact["signed_normal_distance"][selected_indices]).item()
        ),
        "tangent_abs_max_m": [
            float(torch.max(torch.abs(local_positions[:, axis])).item())
            for axis in range(2)
        ],
        "local_min_m": [
            float(value.item())
            for value in torch.min(local_positions, dim=0).values
        ],
        "local_max_m": [
            float(value.item())
            for value in torch.max(local_positions, dim=0).values
        ],
        "half_extents_m": list(half_extents) if half_extents is not None else None,
        "contact_margin_m": float(surface["contact_margin"]),
    }


def _active_patch_contact_diagnostics(active_surfaces, particle_positions, patches):
    if not patches:
        return []
    contacts_by_name = {
        surface["name"]: surface
        for surface in _surface_contacts_torch(active_surfaces, particle_positions)
    }
    diagnostics = []
    for patch in patches:
        selected_indices = patch["indices"]
        surface_names = patch["mode"].split("+")
        patch_contacts = [
            contacts_by_name[name]
            for name in surface_names
            if name in contacts_by_name
        ]
        if len(patch_contacts) != 2 or selected_indices.numel() == 0:
            diagnostics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                    "pressed_count": 0,
                    "surfaces": [],
                }
            )
            continue
        first_contact = patch_contacts[0]["contact"]
        second_contact = patch_contacts[1]["contact"]
        pair_pressed_mask = (
            first_contact["mask"]
            & second_contact["mask"]
            & _pressed_between_surfaces_torch(first_contact, second_contact)
        )
        selected_pressed = pair_pressed_mask[selected_indices]
        surface_gap = torch.linalg.norm(
            first_contact["closest_points"][selected_indices]
            - second_contact["closest_points"][selected_indices],
            dim=1,
        )
        diagnostics.append(
            {
                "mode": patch["mode"],
                "anchor_name": patch["anchor_name"],
                "particles": int(selected_indices.numel()),
                "pressed_count": int(torch.count_nonzero(selected_pressed).item()),
                "surface_gap_min_m": float(torch.min(surface_gap).item()),
                "surface_gap_max_m": float(torch.max(surface_gap).item()),
                "press_gap_m": float(ADHESIVE_PRESS_GAP),
                "surfaces": [
                    _surface_selected_contact_diagnostic(surface, selected_indices)
                    for surface in patch_contacts
                ],
            }
        )
    return diagnostics


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
        excluded_indices=excluded_indices,
    )


def _patch_contact_keys(patches):
    return sorted(
        (patch["mode"], patch["anchor_name"])
        for patch in patches
    )


def _copy_patch_velocity_history(candidate_patches, active_patches):
    for candidate_patch in candidate_patches:
        active_patch = next(
            (
                patch
                for patch in active_patches
                if patch["mode"] == candidate_patch["mode"]
                and patch["anchor_name"] == candidate_patch["anchor_name"]
                and torch.equal(patch["indices"], candidate_patch["indices"])
            ),
            None,
        )
        if active_patch is None:
            continue
        previous_targets = active_patch.get("previous_target_positions")
        if previous_targets is not None:
            candidate_patch["previous_target_positions"] = previous_targets


def _patch_current_pressed_mask(patch, contacts_by_name):
    surface_names = patch["mode"].split("+")
    if len(surface_names) != 2:
        return None
    first_surface = contacts_by_name.get(surface_names[0])
    second_surface = contacts_by_name.get(surface_names[1])
    if first_surface is None or second_surface is None:
        return None
    selected_indices = patch["indices"]
    pair_pressed_mask = (
        first_surface["contact"]["mask"]
        & second_surface["contact"]["mask"]
        & _pressed_between_surfaces_torch(
            first_surface["contact"],
            second_surface["contact"],
        )
    )
    return pair_pressed_mask[selected_indices]


def _filter_active_patches_by_current_contact(
    active_surfaces,
    particle_positions,
    active_patches,
):
    contacts_by_name = {
        surface["name"]: surface
        for surface in _surface_contacts_torch(active_surfaces, particle_positions)
    }
    valid_patches = []
    release_events = []
    for patch in active_patches:
        pressed_mask = _patch_current_pressed_mask(patch, contacts_by_name)
        if pressed_mask is None:
            release_events.append((patch, 0, int(patch["indices"].numel())))
            continue
        pressed_count = int(torch.count_nonzero(pressed_mask).item())
        original_count = int(patch["indices"].numel())
        if pressed_count == 0:
            release_events.append((patch, pressed_count, original_count))
            continue
        if pressed_count == original_count:
            valid_patches.append(patch)
            continue

        pruned_patch = {
            **patch,
            "indices": patch["indices"][pressed_mask],
            "local_positions": patch["local_positions"][pressed_mask],
            "candidate_count": pressed_count,
        }
        previous_targets = patch.get("previous_target_positions")
        if previous_targets is not None and previous_targets.shape[0] == original_count:
            pruned_patch["previous_target_positions"] = previous_targets[pressed_mask]
        valid_patches.append(pruned_patch)
        release_events.append((patch, pressed_count, original_count))
    return valid_patches, release_events


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


def main():
    args = _parse_args()
    global ADHESIVE_PRESS_GAP
    ADHESIVE_PRESS_GAP = args.adhesive_press_gap
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
            GRIPPER_PHYSICS_FRICTION[0],
            GRIPPER_PHYSICS_FRICTION[1],
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
        cloth = _create_tabletop_cloth(stage, world, cloth_material)
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
        max_cloth_centroid_z = None

        for step in range(args.steps):
            lower_amount = (
                1.0
                if args.start_closed
                else (step - args.settle_steps) / max(args.lower_steps, 1)
            )
            close_amount = (
                1.0
                if args.start_closed
                else (
                    step
                    - args.settle_steps
                    - args.lower_steps
                    - args.preclose_lift_steps
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
            adhesive_pair_mode = args.adhesive_pair_mode
            upgrade_to_finger_pinch = (
                args.upgrade_pregrasp_to_pinch
                and close_fraction > 0.0
            )
            handoff_to_inner_patches = (
                args.handoff_pregrasp_to_inner_patches
                and close_fraction >= 0.85
            )
            # Active patches are validated against the surface pair that created
            # them, so bottom/table pregrasp patches release when that contact
            # separates instead of being handed to inner-finger contact.
            release_stale_pregrasp = False
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
            _set_finger_pose(
                left_finger,
                (FINGER_CENTER_X, TABLE_CENTER[1] - finger_y_offset, finger_z),
            )
            _set_finger_pose(
                right_finger,
                (FINGER_CENTER_X, TABLE_CENTER[1] + finger_y_offset, finger_z),
            )

            log_state["step"] = step
            attached = False
            if release_fraction > 0.0 and grasp_state["patches"]:
                grasp_state["patches"] = []
                grasp_state["fold_pairs"] = None
                log_state["released_step"] = step
                _log(f"released adhesive patches at step {step}")

            def _run_sticking_update():
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
                        allow_new_attachment,
                        upgrade_to_finger_pinch,
                        handoff_to_inner_patches,
                        release_stale_pregrasp,
                        args.fold_cloth_sticking and close_fraction >= 1.0,
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
                    if args.explicit_sticking and release_fraction <= 0.0
                    else False
                )

            if args.sticking_update_phase in ("before-step", "both"):
                attached = _run_sticking_update()
            world.step(render=args.live_render)
            if args.sticking_update_phase in ("after-step", "both"):
                attached = _run_sticking_update() or attached
            _cloth_view, _positions, _velocities, particle_positions, _particle_velocities = (
                _torch_cloth_state(cloth)
            )
            cloth_centroid_z = float(torch.mean(particle_positions[:, 2]).item())
            max_cloth_centroid_z = (
                cloth_centroid_z
                if max_cloth_centroid_z is None
                else max(max_cloth_centroid_z, cloth_centroid_z)
            )
            if step % 15 == 0 or step == args.steps - 1:
                active_surfaces = _active_adhesive_surfaces(
                    stage,
                    surfaces,
                    {"pinch": True},
                )
                diagnostic = {
                    "step": step,
                    "finger_y_offset": finger_y_offset,
                    "finger_z": finger_z,
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
                    **_diagnose_surface_contacts(active_surfaces, particle_positions),
                }
                diagnostic_snapshots.append(diagnostic)
                _log("diagnostic " + json.dumps(diagnostic, sort_keys=True))
            if initial_cloth_centroid_z is None:
                initial_cloth_centroid_z = cloth_centroid_z
            if attached and grasp_state["patches"]:
                indices = torch.unique(
                    torch.cat([patch["indices"] for patch in grasp_state["patches"]])
                )
                attached_centroid_z.append(
                    float(torch.mean(particle_positions[indices, 2]).item())
                )

            if rep is not None:
                rep.orchestrator.step(rt_subframes=args.rt_subframes)
            if args.live_render and args.live_step_seconds > 0.0:
                time.sleep(args.live_step_seconds)

        if rep is not None:
            rep.orchestrator.wait_until_complete()
        if writer is not None:
            writer.detach()

        _cloth_view, _positions, _velocities, final_positions, _particle_velocities = (
            _torch_cloth_state(cloth)
        )
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
        success = bool(
            (
                log_state["attached_step"] is not None
                and attached_lift_m > 0.075
                and success_cloth_lift_m > 0.025
            )
            if args.explicit_sticking
            else (max_cloth_lift_m > 0.075 and success_cloth_lift_m > 0.025)
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
            "steps": args.steps,
            "start_closed": args.start_closed,
            "explicit_sticking": args.explicit_sticking,
            "adhesive_pair_mode": args.adhesive_pair_mode,
            "adhesive_max_patches": args.adhesive_max_patches,
            "adhesive_components_per_pair": args.adhesive_components_per_pair,
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
            "finger_closed_y_offset": args.finger_closed_y_offset,
            "expand_adhesive_patch": args.expand_adhesive_patch,
            "adhesive_patch_max_particles": args.adhesive_patch_max_particles,
            "adhesive_press_gap": args.adhesive_press_gap,
            "adhesive_local_patch_radius": args.adhesive_local_patch_radius,
            "adhesive_expanded_patch_radius": args.adhesive_expanded_patch_radius,
            "adhesive_expanded_patch_max_particles": (
                args.adhesive_expanded_patch_max_particles
            ),
            "table_static_friction": args.table_static_friction,
            "table_dynamic_friction": args.table_dynamic_friction,
            "attached_step": log_state["attached_step"],
            "attached_particles": log_state["attached_particles"],
            "released_step": log_state["released_step"],
            "initial_cloth_centroid_z": initial_cloth_centroid_z,
            "final_cloth_centroid_z": final_cloth_centroid_z,
            "max_cloth_centroid_z": max_cloth_centroid_z,
            "cloth_lift_m": cloth_lift_m,
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
