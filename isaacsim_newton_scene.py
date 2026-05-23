"""Replicate the cloth demo scene in Isaac Sim.

This is an Isaac Sim 6 standalone script.  Run it with Isaac Sim's Python, not
the system Python, for example:

    /path/to/IsaacSim/_build/linux-x86_64/release/python.sh \
        isaacsim_newton_scene.py --record --steps 10

The script recreates the dual-arm Piper-X scene from ``cloth_teleop.py``:

* PhysX mode creates a real Isaac Sim particle cloth.
* Newton mode selects Isaac Sim's Newton backend.
* The dual-arm URDF is converted/imported into USD.
* The table, cloth mesh, overview camera, and two D435 wrist cameras are
  created in the USD stage.

Isaac Sim's documented Newton backend does not currently expose raw Newton VBD
cloth through USD.  Use ``--physics-backend physx`` for a physically simulated
Isaac Sim cloth, or ``--physics-backend newton --cloth-mode visual`` for the
Newton rigid-body scene with a visual cloth mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
ROBOT_URDF_PATH = ROOT_DIR / "piper_x_description_dualarm.urdf"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "recordings" / "isaacsim_newton_scene"
ISAAC_SIM_GRIPPER_TARGET_RANGE = (-0.05, 0.0)
ISAAC_SIM_GRIPPER_OPEN_SCALE = 0.8
TABLE_CENTER = (0.10, -0.70, 0.10)
TABLE_SCALE = (0.84, 0.68, 0.08)
TABLE_PHYSICS_FRICTION = (0.08, 0.05)
TABLE_FRICTION_COMBINE_MODE = "min"
OBJECT_ADHESIVE_SURFACE_FRICTION = 0.02
GRIPPER_PHYSICS_FRICTION = (1.0, 0.8)
GRIPPER_FRICTION_COMBINE_MODE = "max"
CLOTH_PARTICLE_FRICTION = 1.2
CLOTH_PARTICLE_FRICTION_SCALE = 0.6
CLOTH_PARTICLE_DAMPING = 0.5
CLOTH_PARTICLE_ADHESION = 0.0
CLOTH_PARTICLE_ADHESION_SCALE = 0.0
CLOTH_ADHESION_OFFSET_SCALE = 0.0
CLOTH_REST_OFFSET = 0.008
CLOTH_CONTACT_OFFSET = 0.009
GRIPPER_OPEN_MAX_FORCE = 1200.0
GRIPPER_CLOSED_MAX_FORCE = 3000.0
GRIPPER_FINGER_COLLISION_BOX_SIZE = (0.056, 0.076, 0.024)
ADHESIVE_PATCH_MAX_PARTICLES = 64
ADHESIVE_LOCAL_PATCH_RADIUS = 0.030
ADHESIVE_PATCH_NEIGHBORHOOD_RADIUS = 0.030
ADHESIVE_PATCH_MAX_EXPANDED_PARTICLES = 96
ADHESIVE_CONTACT_MARGIN = 0.045
ADHESIVE_PRESS_GAP = 0.090
ADHESIVE_FINGER_SURFACE_RADIUS = 0.30
ADHESIVE_OBJECT_SURFACE_RADIUS = 0.60
ADHESIVE_DIAGNOSTIC_INTERVAL = 1.0
TELEOP_EE_LINEAR_STEP = 0.005
TELEOP_EE_LINEAR_SPEED = 0.48
TELEOP_EE_PITCH_STEP_DEG = 1.0
TELEOP_EE_PITCH_SPEED_DEG = 120.0
TELEOP_EE_ROLL_STEP_DEG = 1.0
TELEOP_EE_ROLL_SPEED_DEG = 120.0


def _log(message: str):
    print(f"[isaacsim_newton_scene] {message}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--teleop", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--cloth-asset", choices=("grid", "shirt"), default="grid")
    parser.add_argument("--physics-backend", choices=("physx", "newton"), default="physx")
    parser.add_argument("--cloth-mode", choices=("physical", "visual"), default="physical")
    parser.add_argument("--cloth-rest-offset", type=float, default=CLOTH_REST_OFFSET)
    parser.add_argument("--cloth-contact-offset", type=float, default=CLOTH_CONTACT_OFFSET)
    parser.add_argument("--cloth-particle-friction", type=float, default=CLOTH_PARTICLE_FRICTION)
    parser.add_argument(
        "--cloth-particle-friction-scale",
        type=float,
        default=CLOTH_PARTICLE_FRICTION_SCALE,
    )
    parser.add_argument("--cloth-particle-damping", type=float, default=CLOTH_PARTICLE_DAMPING)
    parser.add_argument("--cloth-particle-adhesion", type=float, default=CLOTH_PARTICLE_ADHESION)
    parser.add_argument(
        "--cloth-particle-adhesion-scale",
        type=float,
        default=CLOTH_PARTICLE_ADHESION_SCALE,
    )
    parser.add_argument(
        "--cloth-adhesion-offset-scale",
        type=float,
        default=CLOTH_ADHESION_OFFSET_SCALE,
    )
    parser.add_argument(
        "--cloth-self-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cloth-self-collision-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--merge-robot-meshes", action="store_true")
    parser.add_argument("--random-arm-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gripper-actions", action="store_true")
    parser.add_argument("--action-sampling", choices=("range", "newton"), default="range")
    parser.add_argument("--arm-random-scale", type=float, default=0.12)
    parser.add_argument("--gripper-random-scale", type=float, default=0.5)
    parser.add_argument(
        "--simple-gripper-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--teleop-hold-delay-seconds", type=float, default=0.2)
    parser.add_argument("--teleop-ee-linear-step", type=float, default=TELEOP_EE_LINEAR_STEP)
    parser.add_argument("--teleop-ee-linear-speed", type=float, default=TELEOP_EE_LINEAR_SPEED)
    parser.add_argument("--teleop-ee-pitch-step-deg", type=float, default=TELEOP_EE_PITCH_STEP_DEG)
    parser.add_argument("--teleop-ee-pitch-speed-deg", type=float, default=TELEOP_EE_PITCH_SPEED_DEG)
    parser.add_argument("--teleop-ee-roll-step-deg", type=float, default=TELEOP_EE_ROLL_STEP_DEG)
    parser.add_argument("--teleop-ee-roll-speed-deg", type=float, default=TELEOP_EE_ROLL_SPEED_DEG)
    parser.add_argument(
        "--teleop-gripper-open-scale",
        type=float,
        default=ISAAC_SIM_GRIPPER_OPEN_SCALE,
    )
    parser.add_argument("--newton-random-joint-scale", type=float, default=0.18)
    parser.add_argument(
        "--fix-imported-arm-bases",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--log-random-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physics-steps-per-action", type=int, default=12)
    parser.add_argument("--camera-render-mode", choices=("separate", "tiled"), default="separate")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument(
        "--write-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write Replicator RGB/depth files. Disable for render-only profiling.",
    )
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--robot-usd-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "assets")
    parser.add_argument("--stage-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "scene.usda")
    parser.add_argument("--video-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "isaacsim_newton_rgbd.mp4")
    parser.add_argument("--compose-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--reset-every-steps", type=int, default=0)
    parser.add_argument("--record-reset-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-arm-zero", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _start_simulation_app(args: argparse.Namespace):
    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Isaac Sim is not importable. Run this script with Isaac Sim 6's "
            "python.sh or inside an Isaac Sim Python environment that includes "
            "the isaacsim.physics.newton extension."
        ) from exc

    return SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
        }
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    import numpy as np

    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    import numpy as np

    trace = np.trace(matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )

    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    if axis == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )

    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return (
        (matrix[1, 0] - matrix[0, 1]) / scale,
        (matrix[0, 2] + matrix[2, 0]) / scale,
        (matrix[1, 2] + matrix[2, 1]) / scale,
        0.25 * scale,
    )


def _look_at_quat_wxyz(position, target, up=(0.0, 0.0, 1.0)):
    import numpy as np

    position_np = np.asarray(position, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    up_np = np.asarray(up, dtype=np.float64)
    forward = _normalize(target_np - position_np)
    right = _normalize(np.cross(forward, up_np))
    camera_up = _normalize(np.cross(right, forward))
    # USD cameras use local -Z as optical forward and +Y as camera up.
    rotation = np.column_stack((right, camera_up, -forward))
    return _quat_wxyz_from_matrix(rotation)


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float):
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _set_xform(prim, translation, rotation_wxyz=(1.0, 0.0, 0.0, 0.0), scale=None):
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xformable.AddOrientOp().Set(
        Gf.Quatf(
            float(rotation_wxyz[0]),
            Gf.Vec3f(
                float(rotation_wxyz[1]),
                float(rotation_wxyz[2]),
                float(rotation_wxyz[3]),
            ),
        )
    )
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _child_path(env_path: str, child_name: str) -> str:
    if env_path == "/World":
        return f"/World/{child_name}"
    return f"{env_path}/{child_name}"


def _env_paths(batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if batch_size == 1:
        return ["/World"]
    return [f"/World/envs/env_{env_index}" for env_index in range(batch_size)]


def _create_env_root(
    stage,
    env_path: str,
    env_index: int,
    env_spacing: float,
    grid_cols: int,
):
    if env_path == "/World":
        return

    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/envs")
    env_root = UsdGeom.Xform.Define(stage, env_path)
    row = env_index // grid_cols
    col = env_index % grid_cols
    _set_xform(env_root.GetPrim(), (col * env_spacing, row * env_spacing, 0.0))


def _enable_extensions(physics_backend: str):
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension_names = [
        "isaacsim.asset.importer.urdf",
        "isaacsim.core.simulation_manager",
        "isaacsim.core.api",
        "isaacsim.core.prims",
        "omni.physx",
        "omni.replicator.core",
    ]
    if physics_backend == "newton":
        extension_names.append("isaacsim.physics.newton")

    for extension_name in extension_names:
        manager.set_extension_enabled_immediate(extension_name, True)


def _replace_link_collisions_with_box(root, link_name: str, center, size):
    link_element = root.find(f"./link[@name='{link_name}']")
    if link_element is None:
        raise RuntimeError(f"Missing URDF link for simple collision: {link_name}")
    for collision_element in list(link_element.findall("collision")):
        link_element.remove(collision_element)

    collision_element = ET.SubElement(link_element, "collision")
    ET.SubElement(
        collision_element,
        "origin",
        {
            "xyz": " ".join(str(value) for value in center),
            "rpy": "0 0 0",
        },
    )
    geometry_element = ET.SubElement(collision_element, "geometry")
    ET.SubElement(
        geometry_element,
        "box",
        {"size": " ".join(str(value) for value in size)},
    )


def _apply_simple_gripper_collisions(root):
    for side in ("left", "right"):
        _replace_link_collisions_with_box(
            root,
            f"{side}_link7",
            (0.0, -0.033, -0.012),
            GRIPPER_FINGER_COLLISION_BOX_SIZE,
        )
        _replace_link_collisions_with_box(
            root,
            f"{side}_link8",
            (0.0, 0.033, -0.012),
            GRIPPER_FINGER_COLLISION_BOX_SIZE,
        )


def _isaac_sim_robot_urdf(robot_usd_dir: Path, simple_gripper_collisions: bool) -> Path:
    robot_urdf_path = robot_usd_dir / "piper_x_dualarm_isaacsim_gripper.urdf"
    robot_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.parse(ROBOT_URDF_PATH)
    for mesh_element in root.getroot().iter("mesh"):
        mesh_filename = mesh_element.attrib.get("filename", "")
        if not mesh_filename or mesh_filename.startswith(("/", "package://")):
            continue
        mesh_element.set("filename", str((ROOT_DIR / mesh_filename).resolve()))
    for joint_element in root.getroot().findall("joint"):
        joint_name = joint_element.attrib.get("name", "")
        if not (joint_name.endswith("_joint7") or joint_name.endswith("_joint8")):
            continue
        limit_element = joint_element.find("limit")
        if limit_element is None:
            continue
        limit_element.set("lower", str(ISAAC_SIM_GRIPPER_TARGET_RANGE[0]))
        limit_element.set("upper", str(ISAAC_SIM_GRIPPER_TARGET_RANGE[1]))
    if simple_gripper_collisions:
        _apply_simple_gripper_collisions(root.getroot())
    root.write(robot_urdf_path, encoding="utf-8", xml_declaration=True)
    return robot_urdf_path


def _convert_robot_urdf(
    robot_usd_dir: Path,
    merge_meshes: bool,
    robot_urdf_path: Path,
    simple_gripper_collisions: bool,
) -> Path:
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    robot_usd_dir.mkdir(parents=True, exist_ok=True)
    suffix = "merged" if merge_meshes else "links"
    collision_suffix = "_simple_finger_colliders" if simple_gripper_collisions else ""
    robot_usd_path = (
        robot_usd_dir
        / f"piper_x_dualarm_{suffix}_gripper_same_sign{collision_suffix}.usd"
    )
    importer = URDFImporter(
        URDFImporterConfig(
            urdf_path=str(robot_urdf_path),
            usd_path=str(robot_usd_path),
            merge_mesh=merge_meshes,
            allow_self_collision=False,
        )
    )
    output_path = Path(importer.import_urdf())
    if not output_path.exists():
        raise RuntimeError(f"URDF importer returned missing USD path: {output_path}")
    return output_path


def _create_newton_stage():
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene
    from pxr import Sdf, UsdGeom, UsdLux

    omni.usd.get_context().new_stage()
    SimulationManager.switch_physics_engine("newton")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = NewtonMjcScene("/World/PhysicsScene")
    scene.set_gravity((0.0, 0.0, -9.81))
    scene.set_dt(1.0 / 600.0)
    scene.set_integrator("implicit")
    scene.set_solver("newton")
    scene.set_iterations(80)

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    light.CreateIntensityAttr(800.0)
    _set_xform(
        light.GetPrim(),
        (0.0, 0.0, 3.0),
        _look_at_quat_wxyz((0, 0, 3), (0, 0, 0), up=(0.0, 1.0, 0.0)),
    )

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(250.0)
    return stage


def _create_physx_world():
    import omni.usd
    from isaacsim.core.api import World
    from pxr import Sdf, UsdGeom, UsdLux

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(
        physics_dt=1.0 / 120.0,
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


def _create_material(stage, path, color):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _create_physics_material(
    stage,
    path,
    static_friction,
    dynamic_friction,
    friction_combine_mode="max",
):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(float(static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material_api.CreateFrictionCombineModeAttr().Set(friction_combine_mode)
    return material


def _bind_material(prim, material):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _bind_physics_material(prim, material):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )


def _create_table(stage, material, physics_material, env_path: str = "/World"):
    from pxr import UsdGeom, UsdPhysics

    table = UsdGeom.Cube.Define(stage, _child_path(env_path, "Table"))
    table.CreateSizeAttr(1.0)
    _set_xform(
        table.GetPrim(),
        TABLE_CENTER,
        scale=TABLE_SCALE,
    )
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    _bind_material(table.GetPrim(), material)
    _bind_physics_material(table.GetPrim(), physics_material)


def _apply_gripper_physics_material(stage, physics_material, env_path: str = "/World"):
    from pxr import Usd, UsdPhysics

    robot_root = _child_path(env_path, "PiperDualArm")
    gripper_links = (
        f"{robot_root}/Geometry/base_link/left_base_link/left_link1/"
        "left_link2/left_link3/left_link4/left_link5/left_link6/"
        "left_gripper_base/left_link7",
        f"{robot_root}/Geometry/base_link/left_base_link/left_link1/"
        "left_link2/left_link3/left_link4/left_link5/left_link6/"
        "left_gripper_base/left_link8",
        f"{robot_root}/Geometry/base_link/right_base_link/right_link1/"
        "right_link2/right_link3/right_link4/right_link5/right_link6/"
        "right_gripper_base/right_link7",
        f"{robot_root}/Geometry/base_link/right_base_link/right_link1/"
        "right_link2/right_link3/right_link4/right_link5/right_link6/"
        "right_gripper_base/right_link8",
    )
    bound_count = 0
    for link_path in gripper_links:
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            continue
        _bind_physics_material(link_prim, physics_material)
        bound_count += 1
        for prim in Usd.PrimRange(link_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                _bind_physics_material(prim, physics_material)
                bound_count += 1
    _log(f"bound gripper physics material to {bound_count} finger prims")


def _grid_mesh(width: float, height: float, nx: int, ny: int, wave_height: float):
    points = []
    for row in range(ny + 1):
        y = (row / ny - 0.5) * height
        for col in range(nx + 1):
            x = (col / nx - 0.5) * width
            z = wave_height * math.sin(2.0 * math.pi * col / nx)
            points.append((x, y, z))

    indices = []
    counts = []
    for row in range(ny):
        for col in range(nx):
            i0 = row * (nx + 1) + col
            i1 = i0 + 1
            i2 = i0 + nx + 1
            i3 = i2 + 1
            indices.extend((i0, i1, i3, i0, i3, i2))
            counts.extend((3, 3))
    return points, counts, indices


def _cloth_geometry(cloth_asset: str):
    if cloth_asset == "shirt":
        points, counts, indices = _shirt_mesh()
        return points, counts, indices, (-0.05, -0.70, 0.18)

    points, counts, indices = _grid_mesh(0.33, 0.22, 49, 33, 0.002)
    return points, counts, indices, (-0.15, -0.70, 0.175)


def _shirt_mask(x: float, y: float) -> bool:
    torso = abs(x) < 0.12 and -0.16 < y < 0.18
    left_sleeve = -0.27 < x < -0.08 and 0.03 < y < 0.17
    right_sleeve = 0.08 < x < 0.27 and 0.03 < y < 0.17
    neck_cut = abs(x) < 0.055 and 0.12 < y < 0.20
    return (torso or left_sleeve or right_sleeve) and not neck_cut


def _shirt_mesh(nx: int = 36, ny: int = 36):
    import numpy as np

    xs = np.linspace(-0.28, 0.28, nx + 1)
    ys = np.linspace(-0.20, 0.22, ny + 1)
    points = []
    point_index = {}
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if _shirt_mask(float(x), float(y)):
                point_index[(row, col)] = len(points)
                points.append((float(x), float(y), 0.003 * math.sin(12.0 * x)))

    indices = []
    counts = []
    for row in range(ny):
        for col in range(nx):
            corners = [
                (row, col),
                (row, col + 1),
                (row + 1, col),
                (row + 1, col + 1),
            ]
            if all(corner in point_index for corner in corners):
                i0 = point_index[(row, col)]
                i1 = point_index[(row, col + 1)]
                i2 = point_index[(row + 1, col)]
                i3 = point_index[(row + 1, col + 1)]
                indices.extend((i0, i1, i3, i0, i3, i2))
                counts.extend((3, 3))
    return points, counts, indices


def _create_cloth_visual(stage, cloth_asset: str, material, env_path: str = "/World"):
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, _child_path(env_path, "ClothVisual"))
    if cloth_asset == "shirt":
        points, counts, indices = _shirt_mesh()
        translation = (0.0, 0.20, 0.18)
        rotation = _look_at_quat_wxyz((0.0, 0.20, 1.0), (0.0, 0.20, 0.0), up=(0.0, 1.0, 0.0))
    else:
        points, counts, indices = _grid_mesh(0.33, 0.22, 49, 33, 0.002)
        translation = (-0.15, -0.70, 0.175)
        rotation = (1.0, 0.0, 0.0, 0.0)

    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    _set_xform(mesh.GetPrim(), translation, rotation)
    _bind_material(mesh.GetPrim(), material)


def _create_physical_cloth(
    stage,
    world,
    cloth_asset: str,
    material,
    args: argparse.Namespace,
    env_path: str = "/World",
    env_index: int = 0,
):
    from isaacsim.core.api.materials.particle_material import ParticleMaterial
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from pxr import Gf, UsdGeom

    points, counts, indices, translation = _cloth_geometry(cloth_asset)
    point_array = np.asarray(points, dtype=np.float64) + np.asarray(
        translation,
        dtype=np.float64,
    )
    point_min = point_array.min(axis=0)
    point_max = point_array.max(axis=0)
    _log(
        f"authored cloth mesh bounds env={env_index}: "
        f"min=({point_min[0]:.3f},{point_min[1]:.3f},{point_min[2]:.3f}) "
        f"max=({point_max[0]:.3f},{point_max[1]:.3f},{point_max[2]:.3f}) "
        f"size=({point_max[0] - point_min[0]:.3f},"
        f"{point_max[1] - point_min[1]:.3f},"
        f"{point_max[2] - point_min[2]:.3f})"
    )
    mesh_path = _child_path(env_path, "ClothPhysical")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    _set_xform(mesh.GetPrim(), translation)
    _bind_material(mesh.GetPrim(), material)

    particle_material = ParticleMaterial(
        prim_path="/World/ParticleMaterial",
        drag=0.1,
        lift=0.0,
        friction=args.cloth_particle_friction,
        particle_friction_scale=args.cloth_particle_friction_scale,
        damping=args.cloth_particle_damping,
        adhesion=args.cloth_particle_adhesion,
        particle_adhesion_scale=args.cloth_particle_adhesion_scale,
        adhesion_offset_scale=args.cloth_adhesion_offset_scale,
    )
    _log(
        "created cloth PBD material: "
        f"friction={args.cloth_particle_friction} "
        f"particle_friction_scale={args.cloth_particle_friction_scale} "
        f"damping={args.cloth_particle_damping} "
        f"adhesion={args.cloth_particle_adhesion} "
        f"particle_adhesion_scale={args.cloth_particle_adhesion_scale} "
        f"adhesion_offset_scale={args.cloth_adhesion_offset_scale}"
    )
    _log(
        "created cloth particle system settings: "
        f"rest_offset={args.cloth_rest_offset} "
        f"contact_offset={args.cloth_contact_offset} "
        f"self_collision={args.cloth_self_collision} "
        f"self_collision_filter={args.cloth_self_collision_filter}"
    )
    particle_system = SingleParticleSystem(
        prim_path=_child_path(env_path, "ParticleSystem"),
        name=f"particle_system_{env_index}",
        simulation_owner=world.get_physics_context().prim_path,
        rest_offset=args.cloth_rest_offset,
        contact_offset=args.cloth_contact_offset,
        solid_rest_offset=args.cloth_rest_offset,
        fluid_rest_offset=args.cloth_rest_offset,
        particle_contact_offset=args.cloth_contact_offset,
        solver_position_iteration_count=16,
        max_velocity=10.0,
        global_self_collision_enabled=args.cloth_self_collision,
        non_particle_collision_enabled=True,
    )
    particle_system.set_simulation_owner(world.get_physics_context().prim_path)
    cloth = SingleClothPrim(
        prim_path=mesh_path,
        particle_system=particle_system,
        particle_material=particle_material,
        name=f"cloth_{env_index}",
        particle_mass=0.002,
        self_collision=args.cloth_self_collision,
        self_collision_filter=args.cloth_self_collision_filter,
        stretch_stiffness=5000.0,
        bend_stiffness=50.0,
        shear_stiffness=2500.0,
        spring_damping=0.2,
    )
    world.scene.add(cloth)
    return cloth


def _reference_robot(stage, robot_usd_path: Path, env_path: str = "/World"):
    from pxr import Sdf, UsdGeom

    robot_root = UsdGeom.Xform.Define(stage, Sdf.Path(_child_path(env_path, "PiperDualArm")))
    robot_root.GetPrim().GetReferences().AddReference(str(robot_usd_path))
    _set_xform(robot_root.GetPrim(), (-0.50, -0.35, 0.0))


def _fix_imported_arm_bases(stage, physics_backend: str, env_path: str = "/World"):
    from pxr import Gf, Sdf, UsdPhysics

    if physics_backend != "physx":
        return

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    fixed_joint_paths = []
    for side in ("left", "right"):
        base_path = Sdf.Path(
            f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link/{side}_base_link"
        )
        base_prim = stage.GetPrimAtPath(base_path)
        if not base_prim.IsValid():
            raise RuntimeError(f"Missing imported {side} arm base prim: {base_path}")

        if base_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            base_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None and base_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            base_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        fixed_joint_path = Sdf.Path(
            f"{_child_path(env_path, 'PiperDualArm')}/Physics/{side}_world_fixed_base"
        )
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
        fixed_joint.CreateBody1Rel().SetTargets([base_path])
        fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        UsdPhysics.ArticulationRootAPI.Apply(fixed_joint.GetPrim())
        if PhysxSchema is not None:
            PhysxSchema.PhysxArticulationAPI.Apply(fixed_joint.GetPrim())
            PhysxSchema.PhysxArticulationAPI(
                fixed_joint.GetPrim()
            ).CreateEnabledSelfCollisionsAttr(False)
        fixed_joint_paths.append(str(fixed_joint_path))

    _log("fixed imported arm bases with world joints: " + ", ".join(fixed_joint_paths))


def _joint_elements_by_name(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    return {
        joint_element.attrib["name"]: joint_element
        for joint_element in root.findall("joint")
    }


def _imported_link_paths(stage, urdf_path: Path, env_path: str = "/World"):
    root = ET.parse(urdf_path).getroot()
    child_links_by_parent = {}
    for joint_element in root.findall("joint"):
        parent_link = joint_element.find("parent").attrib["link"]
        child_link = joint_element.find("child").attrib["link"]
        child_links_by_parent.setdefault(parent_link, []).append(child_link)

    root_path = f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link"
    link_paths = {"base_link": root_path}
    pending_links = ["base_link"]
    while pending_links:
        parent_link = pending_links.pop()
        parent_path = link_paths[parent_link]
        for child_link in child_links_by_parent.get(parent_link, []):
            child_path = f"{parent_path}/{child_link}"
            if stage.GetPrimAtPath(child_path).IsValid():
                link_paths[child_link] = child_path
                pending_links.append(child_link)

    return link_paths


def _repair_imported_joint_frames(
    stage,
    urdf_path: Path,
    physics_backend: str,
    env_path: str = "/World",
):
    from pxr import Gf, Sdf

    if physics_backend != "physx":
        return

    joint_elements = _joint_elements_by_name(urdf_path)
    link_paths = _imported_link_paths(stage, urdf_path, env_path)
    skipped_joints = {"base_link_to_left_base", "base_link_to_right_base"}
    repaired_joints = []
    for joint_name, joint_element in joint_elements.items():
        if joint_name in skipped_joints:
            continue

        joint_prim = stage.GetPrimAtPath(
            f"{_child_path(env_path, 'PiperDualArm')}/Physics/{joint_name}"
        )
        parent_link = joint_element.find("parent").attrib["link"]
        child_link = joint_element.find("child").attrib["link"]
        parent_path = link_paths.get(parent_link)
        child_path = link_paths.get(child_link)
        if not joint_prim.IsValid() or parent_path is None or child_path is None:
            continue

        origin_element = joint_element.find("origin")
        origin_xyz = (
            tuple(float(value) for value in origin_element.attrib.get("xyz", "0 0 0").split())
            if origin_element is not None
            else (0.0, 0.0, 0.0)
        )
        origin_rpy = (
            tuple(float(value) for value in origin_element.attrib.get("rpy", "0 0 0").split())
            if origin_element is not None
            else (0.0, 0.0, 0.0)
        )
        origin_quat = _quat_wxyz_from_rpy(*origin_rpy)

        joint_prim.GetRelationship("physics:body0").SetTargets([Sdf.Path(parent_path)])
        joint_prim.GetRelationship("physics:body1").SetTargets([Sdf.Path(child_path)])
        joint_prim.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*origin_xyz))
        joint_prim.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(
            float(origin_quat[0]),
            Gf.Vec3f(
                float(origin_quat[1]),
                float(origin_quat[2]),
                float(origin_quat[3]),
            ),
        ))
        joint_prim.GetAttribute("physics:localRot1").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        repaired_joints.append(joint_name)

    _log(f"repaired imported joint parent frames: {len(repaired_joints)} joints")


def _find_prim_by_suffix(stage, suffix: str):
    for prim in stage.Traverse():
        if prim.GetName().endswith(suffix):
            return prim
    return None


def _find_prims_by_suffix(stage, suffix: str):
    return [prim for prim in stage.Traverse() if prim.GetName().endswith(suffix)]


def _log_dual_arm_status(stage, env_path: str = "/World"):
    required_link_paths = [
        f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link/left_base_link/left_link1/"
        "left_link2/left_link3/left_link4/left_link5/left_link6",
        f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link/right_base_link/right_link1/"
        "right_link2/right_link3/right_link4/right_link5/right_link6",
        f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link/left_base_link/left_link1/"
        "left_link2/left_link3/left_link4/left_link5/left_link6/"
        "left_gripper_base",
        f"{_child_path(env_path, 'PiperDualArm')}/Geometry/base_link/right_base_link/right_link1/"
        "right_link2/right_link3/right_link4/right_link5/right_link6/"
        "right_gripper_base",
    ]
    missing_link_paths = [
        link_path for link_path in required_link_paths
        if not stage.GetPrimAtPath(link_path).IsValid()
    ]
    for link_path in required_link_paths:
        _log(f"required robot link exists: {link_path}")
    if missing_link_paths:
        raise RuntimeError(
            "The imported dual-arm robot is missing wrist or gripper links: "
            f"{missing_link_paths}"
        )


def _joint_drive_token(joint_prim) -> str:
    if joint_prim.GetTypeName() == "PhysicsPrismaticJoint":
        return "linear"
    return "angular"


def _joint_target_range(
    joint_prim,
    drive_token: str,
    arm_random_scale: float,
    gripper_random_scale: float,
) -> tuple[float, float]:
    lower = joint_prim.GetAttribute("physics:lowerLimit").Get()
    upper = joint_prim.GetAttribute("physics:upperLimit").Get()
    if lower is None or upper is None:
        return (-0.03, 0.03) if drive_token == "linear" else (-30.0, 30.0)

    lower_float = float(lower)
    upper_float = float(upper)
    if drive_token == "linear":
        max_abs = gripper_random_scale * max(abs(lower_float), abs(upper_float))
        if lower_float >= 0.0:
            return 0.0, min(upper_float, max_abs)
        if upper_float <= 0.0:
            return max(lower_float, -max_abs), 0.0
        center = 0.5 * (lower_float + upper_float)
        radius = gripper_random_scale * 0.5 * (upper_float - lower_float)
        return center - radius, center + radius

    center = min(max(0.0, lower_float), upper_float)
    radius = min(arm_random_scale * 0.5 * (upper_float - lower_float), 25.0)
    return max(center - radius, lower_float), min(center + radius, upper_float)


def _joint_full_target_range(joint_prim, drive_token: str) -> tuple[float, float]:
    lower = joint_prim.GetAttribute("physics:lowerLimit").Get()
    upper = joint_prim.GetAttribute("physics:upperLimit").Get()
    if lower is None or upper is None:
        return (-0.03, 0.03) if drive_token == "linear" else (-30.0, 30.0)
    return float(lower), float(upper)


def _configure_robot_drives(
    stage,
    include_grippers: bool,
    arm_random_scale: float,
    gripper_random_scale: float,
    env_path: str = "/World",
):
    from pxr import Usd, UsdPhysics

    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    drive_specs = []
    missing_joints = []
    joint_indices = range(1, 9) if include_grippers else range(1, 7)
    for side in ("left", "right"):
        for joint_index in joint_indices:
            joint_name = f"{side}_joint{joint_index}"
            joint_path = f"{_child_path(env_path, 'PiperDualArm')}/Physics/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                missing_joints.append(joint_path)
                continue

            drive_token = _joint_drive_token(joint_prim)
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_token)
            target_min, target_max = _joint_target_range(
                joint_prim,
                drive_token,
                arm_random_scale,
                gripper_random_scale,
            )
            if drive_token == "linear":
                drive.CreateStiffnessAttr(2500.0)
                drive.CreateDampingAttr(180.0)
                drive.CreateMaxForceAttr(GRIPPER_CLOSED_MAX_FORCE)
            else:
                drive.CreateStiffnessAttr(8500.0)
                drive.CreateDampingAttr(450.0)
                drive.CreateMaxForceAttr(350.0)
            drive.CreateTargetPositionAttr(0.0)
            drive_specs.append((joint_name, drive, target_min, target_max))

    if missing_joints:
        raise RuntimeError(f"Missing robot joints for random actions: {missing_joints}")

    _log(f"configured {len(drive_specs)} random-action joint drives")
    return drive_specs


def _create_robot_articulations(world, env_path: str = "/World", env_index: int = 0):
    from isaacsim.core.prims import SingleArticulation

    articulations = {
        side: world.scene.add(
            SingleArticulation(
                prim_path=(
                    f"{_child_path(env_path, 'PiperDualArm')}/Physics/"
                    f"{side}_world_fixed_base"
                ),
                name=f"env_{env_index}_{side}_piper_arm",
            )
        )
        for side in ("left", "right")
    }
    _log("registered fixed-base arm articulations for tensor control")
    return articulations


def _get_articulation_dof_index(articulation, joint_name: str) -> int:
    dof_index = articulation.get_dof_index(joint_name)
    if dof_index < 0:
        articulation_name = getattr(articulation, "name", "<unnamed>")
        raise RuntimeError(
            f"Imported articulation {articulation_name} has no DOF named "
            f"{joint_name}. Available DOFs: {articulation.dof_names}"
        )
    return int(dof_index)


def _articulation_action_items(robot_articulations, drive_specs, target_by_joint):
    action_items = {"left": [], "right": []}
    for joint_name, drive, _target_min, _target_max in drive_specs:
        side = joint_name.split("_", maxsplit=1)[0]
        dof_index = _get_articulation_dof_index(robot_articulations[side], joint_name)
        action_target = _articulation_target_from_drive_target(
            drive,
            target_by_joint[joint_name],
        )
        action_items[side].append((dof_index, action_target))
    return action_items


def _articulation_target_from_drive_target(drive, target: float) -> float:
    # USD angular drive targets/limits are authored in degrees; Isaac tensor
    # articulation actions expect radians. Prismatic gripper targets are meters.
    drive_token = _joint_drive_token(drive.GetPrim())
    if drive_token == "angular":
        return math.radians(target)
    return target


def _drive_target_from_articulation_position(drive, joint_position: float) -> float:
    # Convert Isaac tensor articulation positions back to the USD drive units
    # used by target_by_joint: degrees for angular joints, meters otherwise.
    drive_token = _joint_drive_token(drive.GetPrim())
    if drive_token == "angular":
        return math.degrees(float(joint_position))
    return float(joint_position)


def _joint_positions_as_numpy(joint_positions):
    if hasattr(joint_positions, "detach"):
        return joint_positions.detach().cpu().numpy().reshape(-1)
    if hasattr(joint_positions, "cpu"):
        return joint_positions.cpu().numpy().reshape(-1)
    return np.asarray(joint_positions, dtype=np.float64).reshape(-1)


def _sample_action_targets(drive_specs, rng: random.Random) -> dict[str, float]:
    specs_by_joint = {
        joint_name: _joint_full_target_range(
            drive.GetPrim(),
            _joint_drive_token(drive.GetPrim()),
        )
        for joint_name, drive, _target_min, _target_max in drive_specs
    }
    target_by_joint = {}
    for joint_name, _drive, target_min, target_max in drive_specs:
        if joint_name.endswith("_joint7") or joint_name.endswith("_joint8"):
            continue
        target_by_joint[joint_name] = rng.uniform(target_min, target_max)

    for side in ("left", "right"):
        first_joint = f"{side}_joint7"
        second_joint = f"{side}_joint8"
        if first_joint not in specs_by_joint or second_joint not in specs_by_joint:
            continue

        first_min, _first_max = specs_by_joint[first_joint]
        second_min, _second_max = specs_by_joint[second_joint]
        max_open = min(abs(min(0.0, first_min)), abs(min(0.0, second_min)))
        amount = -rng.uniform(0.0, max_open)
        target_by_joint[first_joint] = amount
        target_by_joint[second_joint] = amount

    return target_by_joint


def _sample_newton_style_action_targets(
    drive_specs,
    rng: np.random.Generator,
    action_index: int,
    random_joint_scale: float,
) -> dict[str, float]:
    phase = 0.65 * action_index
    noise = rng.uniform(-1.0, 1.0, size=len(drive_specs))
    scale_degrees = math.degrees(random_joint_scale)
    target_by_joint = {}
    for noise_index, (joint_name, _drive, target_min, target_max) in enumerate(drive_specs):
        target = scale_degrees * math.sin(phase + float(noise[noise_index]))
        target_by_joint[joint_name] = min(max(target, target_min), target_max)
    return target_by_joint


def _apply_random_arm_action(
    drive_specs,
    rng,
    action_index: int,
    robot_articulations=None,
    log_actions: bool = True,
    log_prefix: str = "",
    action_sampling: str = "range",
    newton_random_joint_scale: float = 0.18,
):
    if action_sampling == "newton":
        target_by_joint = _sample_newton_style_action_targets(
            drive_specs,
            rng,
            action_index,
            newton_random_joint_scale,
        )
    else:
        target_by_joint = _sample_action_targets(drive_specs, rng)
    targets = []
    articulation_items = (
        _articulation_action_items(robot_articulations, drive_specs, target_by_joint)
        if robot_articulations is not None
        else None
    )
    for joint_name, drive, _target_min, _target_max in drive_specs:
        target = target_by_joint[joint_name]
        if robot_articulations is None:
            drive.GetTargetPositionAttr().Set(float(target))
        targets.append(f"{joint_name}={target:.3f}")

    if articulation_items is not None:
        import torch
        from isaacsim.core.utils.types import ArticulationAction

        for side, side_items in articulation_items.items():
            device = robot_articulations[side]._device
            target_array = torch.tensor(
                [target for _dof_index, target in side_items],
                dtype=torch.float32,
                device=device,
            )
            joint_indices = torch.tensor(
                [dof_index for dof_index, _target in side_items],
                dtype=torch.int32,
                device=device,
            )
            robot_articulations[side].apply_action(
                ArticulationAction(
                    joint_positions=target_array,
                    joint_indices=joint_indices,
                )
            )

    if log_actions:
        _log(f"{log_prefix}random arm action {action_index + 1}: " + ", ".join(targets))


def _apply_random_arm_action_batch(
    env_specs,
    rng,
    action_index: int,
    log_actions: bool,
    action_sampling: str = "range",
    newton_random_joint_scale: float = 0.18,
):
    for env_spec in env_specs:
        drive_specs = env_spec.get("drive_specs")
        if drive_specs is None:
            continue
        _apply_random_arm_action(
            drive_specs,
            rng,
            action_index,
            robot_articulations=env_spec.get("robot_articulations"),
            log_actions=log_actions,
            log_prefix=f"env {env_spec['env_index']} ",
            action_sampling=action_sampling,
            newton_random_joint_scale=newton_random_joint_scale,
        )


def _reset_robot_targets(env_specs, reset_arm_zero: bool):
    if not reset_arm_zero:
        return

    import torch
    from isaacsim.core.utils.types import ArticulationAction

    for env_spec in env_specs:
        drive_specs = env_spec.get("drive_specs") or []
        robot_articulations = env_spec.get("robot_articulations")
        if robot_articulations is None:
            for _joint_name, drive, _target_min, _target_max in drive_specs:
                drive.GetTargetPositionAttr().Set(0.0)
            continue

        for side, articulation in robot_articulations.items():
            side_items = [
                _get_articulation_dof_index(articulation, joint_name)
                for joint_name, _drive, _target_min, _target_max in drive_specs
                if joint_name.startswith(f"{side}_")
            ]
            if not side_items:
                continue
            device = articulation._device
            joint_indices = torch.tensor(side_items, dtype=torch.int32, device=device)
            zeros = torch.zeros(len(side_items), dtype=torch.float32, device=device)
            articulation.set_joint_positions(zeros, joint_indices=joint_indices)
            articulation.set_joint_velocities(zeros, joint_indices=joint_indices)
            articulation.apply_action(
                ArticulationAction(
                    joint_positions=zeros,
                    joint_indices=joint_indices,
                )
            )


def _apply_joint_targets(drive_specs, target_by_joint, robot_articulations=None):
    if robot_articulations is None:
        for joint_name, drive, _target_min, _target_max in drive_specs:
            drive.GetTargetPositionAttr().Set(float(target_by_joint[joint_name]))
        return

    import torch
    from isaacsim.core.utils.types import ArticulationAction

    articulation_items = _articulation_action_items(
        robot_articulations,
        drive_specs,
        target_by_joint,
    )
    for side, side_items in articulation_items.items():
        if not side_items:
            continue
        device = robot_articulations[side]._device
        target_array = torch.tensor(
            [target for _dof_index, target in side_items],
            dtype=torch.float32,
            device=device,
        )
        joint_indices = torch.tensor(
            [dof_index for dof_index, _target in side_items],
            dtype=torch.int32,
            device=device,
        )
        robot_articulations[side].apply_action(
            ArticulationAction(
                joint_positions=target_array,
                joint_indices=joint_indices,
            )
        )


def _set_gripper_drive_force(drive_specs, gripper_closed):
    for joint_name, drive, _target_min, _target_max in drive_specs:
        if not (joint_name.endswith("_joint7") or joint_name.endswith("_joint8")):
            continue
        side = joint_name.split("_", 1)[0]
        max_force = (
            GRIPPER_CLOSED_MAX_FORCE
            if gripper_closed.get(side, False)
            else GRIPPER_OPEN_MAX_FORCE
        )
        drive.GetMaxForceAttr().Set(float(max_force))


def _transform_matrix(xyz, rpy):
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def _axis_angle_matrix(axis, angle):
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * float(angle)).as_matrix()
    return matrix


def _load_arm_ik_chains():
    root = ET.parse(ROBOT_URDF_PATH).getroot()
    joints_by_name = {
        joint.attrib["name"]: joint
        for joint in root.findall("joint")
        if "name" in joint.attrib
    }
    chains = {}
    for side in ("left", "right"):
        fixed_joint = joints_by_name[f"base_link_to_{side}_base"]
        fixed_origin = fixed_joint.find("origin")
        fixed_xyz = tuple(float(v) for v in fixed_origin.attrib.get("xyz", "0 0 0").split())
        fixed_rpy = tuple(float(v) for v in fixed_origin.attrib.get("rpy", "0 0 0").split())
        joint_specs = []
        for joint_index in range(1, 7):
            joint_element = joints_by_name[f"{side}_joint{joint_index}"]
            origin_element = joint_element.find("origin")
            axis_element = joint_element.find("axis")
            joint_xyz = tuple(float(v) for v in origin_element.attrib.get("xyz", "0 0 0").split())
            joint_rpy = tuple(float(v) for v in origin_element.attrib.get("rpy", "0 0 0").split())
            joint_axis = tuple(float(v) for v in axis_element.attrib.get("xyz", "0 0 1").split())
            joint_specs.append(
                {
                    "origin": _transform_matrix(joint_xyz, joint_rpy),
                    "axis": np.asarray(joint_axis, dtype=np.float64),
                }
            )
        chains[side] = {
            "base": _transform_matrix((-0.50, -0.35, 0.0), (0.0, 0.0, 0.0))
            @ _transform_matrix(fixed_xyz, fixed_rpy),
            "joints": joint_specs,
            "ee_offset": np.asarray((0.0, 0.0, 0.13503), dtype=np.float64),
        }
    return chains


def _fk_arm_chain(chain, joint_q):
    transform = chain["base"].copy()
    joint_positions = []
    joint_axes = []
    for joint_spec, joint_value in zip(chain["joints"], joint_q):
        transform = transform @ joint_spec["origin"]
        joint_positions.append(transform[:3, 3].copy())
        joint_axes.append(transform[:3, :3] @ joint_spec["axis"])
        transform = transform @ _axis_angle_matrix(joint_spec["axis"], joint_value)
    ee_pos = transform[:3, 3] + transform[:3, :3] @ chain["ee_offset"]
    return ee_pos, joint_positions, joint_axes


def _solve_ee_delta(chain, joint_q, joint_limits, position_delta, pitch_delta, roll_delta):
    solved_q = np.asarray(joint_q, dtype=np.float64).copy()
    task_delta = np.asarray(
        (
            float(position_delta[0]),
            float(position_delta[1]),
            float(position_delta[2]),
            float(pitch_delta),
            float(roll_delta),
        ),
        dtype=np.float64,
    )
    if np.linalg.norm(task_delta) == 0.0:
        return solved_q

    remaining_delta = task_delta.copy()
    damping = 0.08
    global_pitch_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    global_roll_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    for _iteration in range(4):
        ee_pos, joint_positions, joint_axes = _fk_arm_chain(chain, solved_q)
        jacobian = np.zeros((5, 6), dtype=np.float64)
        for column, (joint_pos, joint_axis) in enumerate(zip(joint_positions, joint_axes)):
            jacobian[:3, column] = np.cross(joint_axis, ee_pos - joint_pos)
            jacobian[3, column] = np.dot(joint_axis, global_pitch_axis)
            jacobian[4, column] = np.dot(joint_axis, global_roll_axis)
        normal_matrix = jacobian @ jacobian.T + (damping ** 2) * np.eye(5)
        delta_q = jacobian.T @ np.linalg.solve(normal_matrix, remaining_delta)
        delta_q = np.clip(delta_q, -0.12, 0.12)
        solved_q = np.asarray(
            [
                min(max(q + dq, lower), upper)
                for q, dq, (lower, upper) in zip(solved_q, delta_q, joint_limits)
            ],
            dtype=np.float64,
        )
        remaining_delta *= 0.35
    return solved_q


def _keyboard_event_key_name(event_input) -> str:
    raw_name = event_input if isinstance(event_input, str) else getattr(
        event_input,
        "name",
        str(event_input),
    )
    key_name = raw_name.rsplit(".", 1)[-1].lower()
    return {
        ";": "semicolon",
        "semi": "semicolon",
        "semi_colon": "semicolon",
    }.get(key_name, key_name)


def _world_point_from_prim(stage, prim_path: str, local_point):
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim for grasp patch: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_point = matrix.Transform(Gf.Vec3d(*local_point))
    return np.asarray(world_point, dtype=np.float64)


def _world_direction_from_prim(stage, prim_path: str, local_direction):
    origin = _world_point_from_prim(stage, prim_path, (0.0, 0.0, 0.0))
    endpoint = _world_point_from_prim(stage, prim_path, local_direction)
    direction = endpoint - origin
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.asarray(local_direction, dtype=np.float64)
    return direction / norm


def _cloth_particle_view(cloth):
    return getattr(cloth, "_cloth_prim_view", cloth)


def _array_like_to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    if hasattr(values, "cpu"):
        return values.cpu().numpy()
    return np.asarray(values)


def _axis_vector(axis_index: int, sign: float = 1.0):
    vector = [0.0, 0.0, 0.0]
    vector[axis_index] = float(sign)
    return tuple(vector)


def _box_surface_descriptors(
    name_prefix,
    prim_path,
    local_center,
    local_size,
    kind,
    friction,
    contact_margin,
    adhesion=0.0,
    requires_closed_side=None,
    surface_radii=None,
    pair_group=None,
):
    center = np.asarray(local_center, dtype=np.float64)
    half_size = 0.5 * np.asarray(local_size, dtype=np.float64)
    radii_by_axis = (
        None
        if surface_radii is None
        else np.asarray(surface_radii, dtype=np.float64)
    )
    surfaces = []
    for normal_axis in range(3):
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        radius = float(
            radii_by_axis[normal_axis]
            if radii_by_axis is not None
            else np.linalg.norm(half_size[tangent_axes]) + contact_margin
        )
        if kind == "finger":
            radius = max(radius, ADHESIVE_FINGER_SURFACE_RADIUS)
        for normal_sign in (-1.0, 1.0):
            origin = center.copy()
            origin[normal_axis] += normal_sign * half_size[normal_axis]
            surfaces.append(
                {
                    "name": f"{name_prefix}_axis{normal_axis}_{int(normal_sign):+d}",
                    "kind": kind,
                    "friction": float(friction),
                    "adhesion": float(adhesion),
                    "prim_path": prim_path,
                    "local_origin": tuple(origin.tolist()),
                    "local_u": _axis_vector(tangent_axes[0]),
                    "local_v": _axis_vector(tangent_axes[1]),
                    "local_normal": _axis_vector(normal_axis, normal_sign),
                    "radius": radius,
                    "contact_margin": float(contact_margin),
                    "requires_closed_side": requires_closed_side,
                    "pair_group": pair_group or prim_path,
                }
            )
    return surfaces


def _cube_world_axis_radii(stage, prim_path, local_size, contact_margin):
    radii = []
    for normal_axis in range(3):
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        tangent_lengths = []
        for tangent_axis in tangent_axes:
            direction = _world_direction_vector_with_scale(
                stage,
                prim_path,
                _axis_vector(tangent_axis),
            )
            tangent_lengths.append(
                0.5 * local_size[tangent_axis] * np.linalg.norm(direction)
            )
        radii.append(float(np.linalg.norm(tangent_lengths) + contact_margin))
    return tuple(radii)


def _world_direction_vector_with_scale(stage, prim_path: str, local_direction):
    origin = _world_point_from_prim(stage, prim_path, (0.0, 0.0, 0.0))
    endpoint = _world_point_from_prim(stage, prim_path, local_direction)
    return endpoint - origin


def _collision_prim_box_descriptor(
    stage,
    prim,
    name_prefix,
    kind,
    friction,
    contact_margin,
    requires_closed_side=None,
    pair_group=None,
):
    from pxr import UsdGeom

    prim_path = prim.GetPath().pathString
    if prim.IsA(UsdGeom.Cube):
        cube_size = UsdGeom.Cube(prim).GetSizeAttr().Get() or 1.0
        local_center = (0.0, 0.0, 0.0)
        local_size = (float(cube_size), float(cube_size), float(cube_size))
        surface_radii = _cube_world_axis_radii(
            stage,
            prim_path,
            local_size,
            contact_margin,
        )
    else:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.GetExtentAttr().Get() if boundable else None
        if extent is None:
            local_center = (0.0, 0.0, 0.0)
            local_size = GRIPPER_FINGER_COLLISION_BOX_SIZE
        else:
            extent_array = np.asarray(extent, dtype=np.float64)
            local_min = extent_array[0]
            local_max = extent_array[1]
            local_center = tuple((0.5 * (local_min + local_max)).tolist())
            local_size = tuple((local_max - local_min).tolist())
        surface_radii = None

    return _box_surface_descriptors(
        name_prefix,
        prim_path,
        local_center,
        local_size,
        kind,
        friction,
        contact_margin,
        requires_closed_side=requires_closed_side,
        surface_radii=surface_radii,
        pair_group=pair_group,
    )


def _finger_adhesive_surfaces(stage, finger_link_paths):
    from pxr import Usd, UsdPhysics

    surfaces = []
    finger_specs = (
        ("link7", (0.0, -0.033, -0.012)),
        ("link8", (0.0, 0.033, -0.012)),
    )
    collision_counts = {}
    collision_types = {}
    for side, side_paths in finger_link_paths.items():
        for link_name, local_center in finger_specs:
            link_path = side_paths[link_name]
            link_prim = stage.GetPrimAtPath(link_path)
            pair_group = f"{side}_{link_name}"
            collision_prims = [
                prim
                for prim in Usd.PrimRange(link_prim)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            collision_key = f"{side}_{link_name}"
            collision_counts[collision_key] = len(collision_prims)
            collision_types[collision_key] = [
                prim.GetTypeName() for prim in collision_prims
            ]
            if collision_prims:
                for collision_index, collision_prim in enumerate(collision_prims):
                    surfaces.extend(
                        _collision_prim_box_descriptor(
                            stage,
                            collision_prim,
                            f"{side}_{link_name}_collision{collision_index}",
                            "finger",
                            GRIPPER_PHYSICS_FRICTION[0],
                            ADHESIVE_CONTACT_MARGIN,
                            requires_closed_side=side,
                            pair_group=pair_group,
                        )
                    )

            surfaces.extend(
                _box_surface_descriptors(
                    f"{side}_{link_name}_fallback",
                    link_path,
                    local_center,
                    GRIPPER_FINGER_COLLISION_BOX_SIZE,
                    "finger",
                    GRIPPER_PHYSICS_FRICTION[0],
                    ADHESIVE_CONTACT_MARGIN,
                    requires_closed_side=side,
                    pair_group=pair_group,
                )
            )
    _log(f"finger adhesive collision prim counts: {collision_counts}")
    _log(f"finger adhesive collision prim types: {collision_types}")
    return surfaces


def _object_collision_surfaces(stage, env_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    env_root = stage.GetPrimAtPath(env_path)
    surfaces = []
    for prim in Usd.PrimRange(env_root):
        prim_path = prim.GetPath().pathString
        if "/PiperDualArm/" in prim_path or prim_path.endswith("/ClothPhysical"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        boundable = UsdGeom.Boundable(prim)
        extent = boundable.GetExtentAttr().Get() if boundable else None
        if extent is None:
            local_center = (0.0, 0.0, 0.0)
            local_size = (1.0, 1.0, 1.0)
        else:
            extent_array = np.asarray(extent, dtype=np.float64)
            local_min = extent_array[0]
            local_max = extent_array[1]
            local_center = tuple((0.5 * (local_min + local_max)).tolist())
            local_size = tuple((local_max - local_min).tolist())

        surfaces.extend(
            _box_surface_descriptors(
                f"{prim.GetName()}_object",
                prim_path,
                local_center,
                local_size,
                "object",
                OBJECT_ADHESIVE_SURFACE_FRICTION,
                ADHESIVE_CONTACT_MARGIN,
                pair_group=prim_path,
            )
        )
    return surfaces


def _create_adhesive_surfaces(stage, env_path, finger_link_paths):
    surfaces = _finger_adhesive_surfaces(stage, finger_link_paths) + _object_collision_surfaces(
        stage,
        env_path,
    )
    kind_counts = {}
    for surface in surfaces:
        kind_counts[surface["kind"]] = kind_counts.get(surface["kind"], 0) + 1
    _log(
        f"created adhesive surfaces for {env_path}: "
        f"count={len(surfaces)} kinds={kind_counts}"
    )
    return surfaces


def _surface_frame(stage, surface):
    origin = _world_point_from_prim(stage, surface["prim_path"], surface["local_origin"])
    tangent_u = _world_direction_from_prim(stage, surface["prim_path"], surface["local_u"])
    normal = _world_direction_from_prim(stage, surface["prim_path"], surface["local_normal"])
    tangent_u = tangent_u - normal * np.dot(tangent_u, normal)
    tangent_u_norm = np.linalg.norm(tangent_u)
    if tangent_u_norm < 1e-9:
        tangent_u = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        tangent_u = tangent_u / tangent_u_norm
    tangent_v = np.cross(normal, tangent_u)
    tangent_v_norm = np.linalg.norm(tangent_v)
    if tangent_v_norm < 1e-9:
        tangent_v = _world_direction_from_prim(stage, surface["prim_path"], surface["local_v"])
    else:
        tangent_v = tangent_v / tangent_v_norm
    return {"origin": origin, "rotation": np.column_stack((tangent_u, tangent_v, normal))}


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
    import torch

    frame = _surface_frame_tensors(surface, particle_positions)
    local_positions = (particle_positions - frame["origin"]) @ frame["rotation"]
    tangent_distance = torch.linalg.norm(local_positions[:, :2], dim=1)
    normal_distance = torch.abs(local_positions[:, 2])
    closest_points = (
        particle_positions
        - frame["rotation"][:, 2] * local_positions[:, 2:3]
    )
    mask = (
        (normal_distance <= float(surface["contact_margin"]))
        & (tangent_distance <= float(surface["radius"]))
    )
    score = normal_distance + 0.25 * tangent_distance
    return {
        "frame": frame,
        "local_positions": local_positions,
        "tangent_distance": tangent_distance,
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


def _pressed_between_surfaces_torch(first_contact, second_contact):
    import torch

    # The cloth API used here exposes positions but not per-particle contact
    # impulse. Approximate pressure by requiring the nearest points on the two
    # contact surfaces around each particle to nearly coincide.
    surface_gap = torch.linalg.norm(
        first_contact["closest_points"] - second_contact["closest_points"],
        dim=1,
    )
    return surface_gap <= ADHESIVE_PRESS_GAP


def _adhesive_surfaces_are_opposed(first_surface, second_surface):
    import torch

    first_normal = first_surface["torch_frame"]["rotation"][:, 2]
    second_normal = second_surface["torch_frame"]["rotation"][:, 2]
    return bool(torch.dot(first_normal, second_normal).item() < -0.35)


def _adhesive_surfaces_can_pair(first_surface, second_surface, side):
    if first_surface["kind"] != "finger" and second_surface["kind"] != "finger":
        return False
    if first_surface["prim_path"] == second_surface["prim_path"]:
        return False
    first_pair_group = first_surface.get("pair_group")
    second_pair_group = second_surface.get("pair_group")
    if first_pair_group is not None and first_pair_group == second_pair_group:
        return False
    if not _adhesive_surfaces_are_opposed(first_surface, second_surface):
        return False
    if first_surface.get("requires_closed_side") not in (None, side):
        return False
    if second_surface.get("requires_closed_side") not in (None, side):
        return False
    return True


def _adhesive_pair_stats(first_surface, second_surface):
    import torch

    contact_mask = first_surface["contact"]["mask"] & second_surface["contact"]["mask"]
    contact_count = int(torch.count_nonzero(contact_mask).item())
    pair_score = first_surface["contact"]["score"] + second_surface["contact"]["score"]
    min_pair_score = float(torch.min(pair_score).item())
    if contact_count == 0:
        return {
            "contact_count": 0,
            "pressed_count": 0,
            "min_gap": None,
            "min_pair_score": min_pair_score,
            "first_min_normal": float(
                torch.min(first_surface["contact"]["normal_distance"]).item()
            ),
            "second_min_normal": float(
                torch.min(second_surface["contact"]["normal_distance"]).item()
            ),
            "first_min_tangent": float(
                torch.min(first_surface["contact"]["tangent_distance"]).item()
            ),
            "second_min_tangent": float(
                torch.min(second_surface["contact"]["tangent_distance"]).item()
            ),
            "first_radius": float(first_surface["radius"]),
            "second_radius": float(second_surface["radius"]),
        }
    surface_gap = torch.linalg.norm(
        first_surface["contact"]["closest_points"]
        - second_surface["contact"]["closest_points"],
        dim=1,
    )
    pressed_count = int(
        torch.count_nonzero(contact_mask & (surface_gap <= ADHESIVE_PRESS_GAP)).item()
    )
    return {
        "contact_count": contact_count,
        "pressed_count": pressed_count,
        "min_gap": float(torch.min(surface_gap[contact_mask]).item()),
        "min_pair_score": min_pair_score,
        "first_min_normal": float(
            torch.min(first_surface["contact"]["normal_distance"]).item()
        ),
        "second_min_normal": float(
            torch.min(second_surface["contact"]["normal_distance"]).item()
        ),
        "first_min_tangent": float(
            torch.min(first_surface["contact"]["tangent_distance"]).item()
        ),
        "second_min_tangent": float(
            torch.min(second_surface["contact"]["tangent_distance"]).item()
        ),
        "first_radius": float(first_surface["radius"]),
        "second_radius": float(second_surface["radius"]),
    }


def _surface_candidate_distance(surface, candidate_indices):
    if candidate_indices is None or candidate_indices.numel() == 0:
        return float("inf")
    return float(
        surface["contact"]["normal_distance"][candidate_indices].mean().item()
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


def _log_adhesive_diagnostic(env_spec, side, contacts):
    import torch

    now = time.perf_counter()
    diagnostic_times = env_spec.setdefault("adhesive_diagnostic_times", {})
    if now - diagnostic_times.get(side, 0.0) < ADHESIVE_DIAGNOSTIC_INTERVAL:
        return
    diagnostic_times[side] = now

    kind_counts = {}
    for surface in contacts:
        kind_counts[surface["kind"]] = kind_counts.get(surface["kind"], 0) + 1

    best_single = None
    for surface in contacts:
        contact = surface["contact"]
        contact_count = int(torch.count_nonzero(contact["mask"]).item())
        min_score = float(torch.min(contact["score"]).item())
        stats = {
            "name": surface["name"],
            "kind": surface["kind"],
            "contact_count": contact_count,
            "min_score": min_score,
            "min_normal": float(torch.min(contact["normal_distance"]).item()),
            "min_tangent": float(torch.min(contact["tangent_distance"]).item()),
            "radius": float(surface["radius"]),
        }
        if best_single is None:
            should_replace = True
        elif stats["contact_count"] > best_single["contact_count"]:
            should_replace = True
        elif (
            stats["contact_count"] == best_single["contact_count"]
            and stats["min_score"] < best_single["min_score"]
        ):
            should_replace = True
        else:
            should_replace = False
        if should_replace:
            best_single = stats

    best_stats = None
    for first_index, first_surface in enumerate(contacts):
        for second_surface in contacts[first_index + 1:]:
            if not _adhesive_surfaces_can_pair(first_surface, second_surface, side):
                continue
            stats = _adhesive_pair_stats(first_surface, second_surface)
            if best_stats is None:
                should_replace = True
            elif stats["contact_count"] > best_stats["contact_count"]:
                should_replace = True
            elif (
                stats["contact_count"] == best_stats["contact_count"]
                and stats["pressed_count"] > best_stats["pressed_count"]
            ):
                should_replace = True
            elif (
                stats["contact_count"] == best_stats["contact_count"]
                and stats["pressed_count"] == best_stats["pressed_count"]
                and stats["min_pair_score"] < best_stats["min_pair_score"]
            ):
                should_replace = True
            else:
                should_replace = False
            if should_replace:
                best_stats = {
                    **stats,
                    "first": first_surface["name"],
                    "second": second_surface["name"],
                }

    if best_stats is None:
        _log(
            f"teleop {side} adhesive diagnostic: no active surface pairs "
            f"kinds={kind_counts} best_single={best_single}"
        )
        return
    _log(
        f"teleop {side} adhesive diagnostic: best_pair="
        f"{best_stats['first']}+{best_stats['second']} "
        f"kinds={kind_counts} "
        f"contact_count={best_stats['contact_count']} "
        f"pressed_count={best_stats['pressed_count']} "
        f"min_gap={best_stats['min_gap']} "
        f"min_pair_score={best_stats['min_pair_score']:.4f} "
        f"normal=({best_stats['first_min_normal']:.4f},"
        f"{best_stats['second_min_normal']:.4f}) "
        f"tangent=({best_stats['first_min_tangent']:.4f},"
        f"{best_stats['second_min_tangent']:.4f}) "
        f"radius=({best_stats['first_radius']:.4f},"
        f"{best_stats['second_radius']:.4f}) "
        f"best_single={best_single}"
    )


def _active_adhesive_surfaces(stage, surfaces, gripper_closed):
    active_surfaces = []
    for surface in surfaces:
        required_side = surface.get("requires_closed_side")
        if required_side is not None and not gripper_closed.get(required_side, False):
            continue
        surface_frame = _surface_frame(stage, surface)
        active_surfaces.append({**surface, "frame": surface_frame})
    return active_surfaces


def _expand_adhesive_patch_indices(particle_positions, seed_indices, anchor_frame):
    import torch

    if seed_indices.numel() == 0:
        return seed_indices

    local_positions = (particle_positions - anchor_frame["origin"]) @ anchor_frame["rotation"]
    seed_local_positions = local_positions[seed_indices]
    tangent_center = torch.mean(seed_local_positions[:, :2], dim=0)
    tangent_distance = torch.linalg.norm(
        local_positions[:, :2] - tangent_center,
        dim=1,
    )
    expanded_mask = tangent_distance <= ADHESIVE_PATCH_NEIGHBORHOOD_RADIUS
    expanded_indices = torch.nonzero(expanded_mask, as_tuple=False).flatten()
    if expanded_indices.numel() <= seed_indices.numel():
        return seed_indices

    if expanded_indices.numel() > ADHESIVE_PATCH_MAX_EXPANDED_PARTICLES:
        order = torch.argsort(tangent_distance[expanded_indices])[
            :ADHESIVE_PATCH_MAX_EXPANDED_PARTICLES
        ]
        expanded_indices = expanded_indices[order]
    return torch.unique(torch.cat((seed_indices, expanded_indices)))


def _choose_adhesive_patch_from_contacts(contacts, particle_positions):
    import torch

    best_patch = None
    for first_index, first_surface in enumerate(contacts):
        for second_surface in contacts[first_index + 1:]:
            required_side = (
                first_surface.get("requires_closed_side")
                or second_surface.get("requires_closed_side")
            )
            if not _adhesive_surfaces_can_pair(first_surface, second_surface, required_side):
                continue
            combined_mask = (
                first_surface["contact"]["mask"]
                & second_surface["contact"]["mask"]
                & _pressed_between_surfaces_torch(
                    first_surface["contact"],
                    second_surface["contact"],
                )
            )
            candidate_indices = torch.nonzero(combined_mask, as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            pair_score = (
                first_surface["contact"]["score"][candidate_indices]
                + second_surface["contact"]["score"][candidate_indices]
            )
            anchor = _choose_adhesive_anchor(
                first_surface,
                second_surface,
                candidate_indices,
            )
            anchor_frame = anchor["torch_frame"]
            candidate_local_positions = (
                particle_positions[candidate_indices] - anchor_frame["origin"]
            ) @ anchor_frame["rotation"]
            best_pair_order = torch.argsort(pair_score)
            best_local_position = candidate_local_positions[best_pair_order[0]]
            local_distances = torch.linalg.norm(
                candidate_local_positions[:, :2] - best_local_position[:2],
                dim=1,
            )
            local_mask = local_distances <= ADHESIVE_LOCAL_PATCH_RADIUS
            local_candidate_indices = candidate_indices[local_mask]
            local_pair_score = pair_score[local_mask]
            local_order = torch.argsort(local_pair_score)[:ADHESIVE_PATCH_MAX_PARTICLES]
            seed_indices = local_candidate_indices[local_order]
            selected_indices = _expand_adhesive_patch_indices(
                particle_positions,
                seed_indices,
                anchor_frame,
            )
            local_positions = (
                particle_positions[selected_indices] - anchor_frame["origin"]
            ) @ anchor_frame["rotation"]
            patch = {
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
            if best_patch is None:
                best_patch = patch
            elif selected_indices.numel() > best_patch["indices"].numel():
                best_patch = patch
            elif (
                selected_indices.numel() == best_patch["indices"].numel()
                and patch["score"].item() < best_patch["score"].item()
            ):
                best_patch = patch
    return best_patch


def _choose_adhesive_patch(active_surfaces, particle_positions):
    contacts = _surface_contacts_torch(active_surfaces, particle_positions)
    return _choose_adhesive_patch_from_contacts(contacts, particle_positions)


def _cloth_state_tensors(cloth):
    cloth_view = _cloth_particle_view(cloth)
    positions = cloth_view.get_world_positions()
    velocities = cloth_view.get_velocities()
    if not hasattr(positions, "detach") or not hasattr(velocities, "detach"):
        raise RuntimeError(
            "Isaac Sim cloth sticking requires the CUDA/Torch cloth view. "
            "Do not convert cloth particles to NumPy for adhesive updates."
        )
    particle_positions = positions[0] if positions.ndim == 3 else positions
    particle_velocities = velocities[0] if velocities.ndim == 3 else velocities
    return cloth_view, positions, velocities, particle_positions, particle_velocities


def _log_cloth_bounds(env_specs, label):
    import torch

    for env_spec in env_specs:
        cloth = env_spec.get("cloth")
        if cloth is None:
            continue
        try:
            _cloth_view, _positions, _velocities, particle_positions, _particle_velocities = (
                _cloth_state_tensors(cloth)
            )
            finite_mask = torch.isfinite(particle_positions).all(dim=1)
            finite_count = int(torch.count_nonzero(finite_mask).item())
            if finite_count == 0:
                _log(
                    f"{label} env={env_spec['env_index']} cloth bounds: "
                    "no finite particles"
                )
                continue
            finite_positions = particle_positions[finite_mask]
            min_position = torch.min(finite_positions, dim=0).values
            max_position = torch.max(finite_positions, dim=0).values
            centroid = torch.mean(finite_positions, dim=0)
            _log(
                f"{label} env={env_spec['env_index']} cloth bounds: "
                f"finite={finite_count}/{particle_positions.shape[0]} "
                f"min=({min_position[0].item():.3f},"
                f"{min_position[1].item():.3f},{min_position[2].item():.3f}) "
                f"max=({max_position[0].item():.3f},"
                f"{max_position[1].item():.3f},{max_position[2].item():.3f}) "
                f"centroid=({centroid[0].item():.3f},"
                f"{centroid[1].item():.3f},{centroid[2].item():.3f})"
            )
        except Exception as exc:
            _log(
                f"{label} env={env_spec['env_index']} cloth bounds failed: {exc}"
            )


def _drive_grasped_cloth_particles(env_spec, gripper_closed):
    cloth = env_spec.get("cloth")
    adhesive_surfaces = env_spec.get("adhesive_surfaces") or []
    if cloth is None or not adhesive_surfaces:
        return

    stage = cloth.prim.GetStage()
    cloth_view, positions, velocities, particle_positions, _particle_velocities = (
        _cloth_state_tensors(cloth)
    )
    grasp_state = env_spec.setdefault("grasp_state", {"left": None, "right": None})
    updated = False

    active_surfaces = _active_adhesive_surfaces(stage, adhesive_surfaces, gripper_closed)
    surfaces_by_name = {surface["name"]: surface for surface in active_surfaces}
    sides = sorted(
        {
            surface["requires_closed_side"]
            for surface in adhesive_surfaces
            if surface.get("requires_closed_side") is not None
        }
    )
    for side in sides:
        active_grasp = grasp_state.get(side)
        if not gripper_closed.get(side, False):
            grasp_state[side] = None
            continue

        if active_grasp is None:
            side_surfaces = [
                surface
                for surface in active_surfaces
                if surface.get("requires_closed_side") in (None, side)
            ]
            contacts = _surface_contacts_torch(side_surfaces, particle_positions)
            active_grasp = _choose_adhesive_patch_from_contacts(
                contacts,
                particle_positions,
            )
            if active_grasp is None:
                _log_adhesive_diagnostic(env_spec, side, contacts)
                continue
            grasp_state[side] = active_grasp
            _log(
                f"teleop {side} adhesive patch {active_grasp['mode']} "
                f"anchored to {active_grasp['anchor_name']} with "
                f"{int(active_grasp['indices'].numel())} cloth particles "
                f"from {active_grasp.get('seed_count', 'unknown')} contact seeds"
            )

        selected_indices = active_grasp["indices"]
        anchor_surface = surfaces_by_name.get(active_grasp["anchor_name"])
        if anchor_surface is None:
            grasp_state[side] = None
            continue
        drive_frame = _surface_frame_tensors(anchor_surface, particle_positions)
        target_positions = (
            active_grasp["local_positions"] @ drive_frame["rotation"].T
        ) + drive_frame["origin"]
        positions[0, selected_indices] = target_positions
        velocities[0, selected_indices] = 0.0
        updated = True

    if updated:
        cloth_view.set_world_positions(positions)
        cloth_view.set_velocities(velocities)


def _run_teleop(world, simulation_app, env_spec, args):
    if args.headless:
        raise ValueError("Isaac Sim teleop requires a non-headless app window.")
    if args.batch_size != 1:
        raise ValueError("Minimal Isaac Sim teleop currently supports batch size 1.")
    if args.physics_backend != "physx":
        raise ValueError("Minimal Isaac Sim teleop currently requires PhysX.")

    import carb.input
    import omni.appwindow

    drive_specs = env_spec.get("drive_specs")
    robot_articulations = env_spec.get("robot_articulations")
    if drive_specs is None or robot_articulations is None:
        raise ValueError("Teleop requires PhysX articulations and configured drives.")

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    pressed_keys = set()
    pressed_key_times = {}
    active_arm = "left"
    gripper_closed = {"left": True, "right": True}
    target_by_joint = {
        joint_name: 0.0
        for joint_name, _drive, _target_min, _target_max in drive_specs
    }
    specs_by_joint = {
        joint_name: _joint_full_target_range(
            drive.GetPrim(),
            _joint_drive_token(drive.GetPrim()),
        )
        for joint_name, drive, _target_min, _target_max in drive_specs
    }
    drives_by_joint = {
        joint_name: drive
        for joint_name, drive, _target_min, _target_max in drive_specs
    }
    arm_ik_chains = _load_arm_ik_chains()
    ee_key_bindings = (
        ("u", "o", np.asarray((1.0, 0.0, 0.0), dtype=np.float64), 0.0, 0.0),
        ("j", "l", np.asarray((0.0, 1.0, 0.0), dtype=np.float64), 0.0, 0.0),
        ("i", "k", np.asarray((0.0, 0.0, 1.0), dtype=np.float64), 0.0, 0.0),
        ("y", "h", np.zeros(3, dtype=np.float64), 1.0, 0.0),
        ("n", "m", np.zeros(3, dtype=np.float64), 0.0, 1.0),
    )
    ee_key_actions = {
        positive_key: (axis, pitch_axis, roll_axis)
        for positive_key, _negative_key, axis, pitch_axis, roll_axis in ee_key_bindings
    }
    ee_key_actions.update(
        {
            negative_key: (-axis, -pitch_axis, -roll_axis)
            for _positive_key, negative_key, axis, pitch_axis, roll_axis in ee_key_bindings
        }
    )

    def active_arm_actual_q(joint_names):
        articulation = robot_articulations[active_arm]
        actual_positions = _joint_positions_as_numpy(articulation.get_joint_positions())
        actual_q = []
        for joint_name in joint_names:
            dof_index = _get_articulation_dof_index(articulation, joint_name)
            drive_target = _drive_target_from_articulation_position(
                drives_by_joint[joint_name],
                actual_positions[dof_index],
            )
            target_min, target_max = specs_by_joint[joint_name]
            clamped_target = min(max(drive_target, target_min), target_max)
            target_by_joint[joint_name] = clamped_target
            actual_q.append(math.radians(clamped_target))
        return np.asarray(actual_q, dtype=np.float64)

    def apply_active_ee_delta(position_delta, pitch_delta, roll_delta, log_update=False):
        joint_names = [f"{active_arm}_joint{joint_index}" for joint_index in range(1, 7)]
        current_q = active_arm_actual_q(joint_names)
        joint_limits = [
            (
                math.radians(specs_by_joint[joint_name][0]),
                math.radians(specs_by_joint[joint_name][1]),
            )
            for joint_name in joint_names
        ]
        solved_q = _solve_ee_delta(
            arm_ik_chains[active_arm],
            current_q,
            joint_limits,
            position_delta,
            pitch_delta,
            roll_delta,
        )
        for joint_name, joint_value in zip(joint_names, solved_q):
            target_by_joint[joint_name] = math.degrees(float(joint_value))
        if log_update:
            _log(
                "teleop "
                f"{active_arm} ee world delta xyz="
                f"({position_delta[0]:.3f}, {position_delta[1]:.3f}, "
                f"{position_delta[2]:.3f}) global pitch="
                f"{math.degrees(pitch_delta):.2f} deg global roll="
                f"{math.degrees(roll_delta):.2f} deg"
            )

    def on_keyboard_event(event):
        nonlocal active_arm
        key_name = _keyboard_event_key_name(event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            was_pressed = key_name in pressed_keys
            pressed_keys.add(key_name)
            if not was_pressed:
                pressed_key_times[key_name] = time.perf_counter()
                _log(f"teleop key pressed: {key_name}")
            if key_name == "t" and not was_pressed:
                active_arm = "right" if active_arm == "left" else "left"
                _log(f"teleop active arm: {active_arm}")
            elif key_name == "g" and not was_pressed:
                gripper_closed[active_arm] = not gripper_closed[active_arm]
                _log(
                    f"teleop {active_arm} gripper "
                    f"{'closed' if gripper_closed[active_arm] else 'open'}"
                )
            elif key_name in ee_key_actions:
                position_axis, pitch_axis, roll_axis = ee_key_actions[key_name]
                apply_active_ee_delta(
                    position_axis * args.teleop_ee_linear_step,
                    math.radians(args.teleop_ee_pitch_step_deg) * pitch_axis,
                    math.radians(args.teleop_ee_roll_step_deg) * roll_axis,
                    log_update=True,
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed_keys.discard(key_name)
            pressed_key_times.pop(key_name, None)
        return True

    subscription = input_interface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard_event,
    )
    camera_paths = env_spec.get("camera_paths") or []
    if camera_paths:
        try:
            import omni.kit.viewport.utility

            viewport = omni.kit.viewport.utility.get_active_viewport()
            if viewport is not None:
                viewport.camera_path = camera_paths[0]
        except Exception as exc:
            _log(f"failed to set teleop viewport camera: {exc}")
    print(
        "Isaac Sim teleop controls: T switch arm, "
        "U/O world X, J/L world Y, I/K world Z, Y/H pitch, N/M roll, "
        "G toggle gripper, Esc exits.",
        flush=True,
    )

    linear_step = args.teleop_ee_linear_speed / max(float(args.fps), 1.0)
    pitch_step = math.radians(args.teleop_ee_pitch_speed_deg) / max(float(args.fps), 1.0)
    roll_step = math.radians(args.teleop_ee_roll_speed_deg) / max(float(args.fps), 1.0)
    try:
        while simulation_app.is_running():
            if "escape" in pressed_keys:
                break

            # Keep supporting held-key motion for local desktops; VNC key
            # repeat also applies discrete steps in the key callback above.
            for positive_key, negative_key, position_axis, pitch_axis, roll_axis in ee_key_bindings:
                now = time.perf_counter()
                positive_held = (
                    positive_key in pressed_keys
                    and now - pressed_key_times[positive_key]
                    >= args.teleop_hold_delay_seconds
                )
                negative_held = (
                    negative_key in pressed_keys
                    and now - pressed_key_times[negative_key]
                    >= args.teleop_hold_delay_seconds
                )
                axis = float(positive_held) - float(negative_held)
                if axis == 0.0:
                    continue
                apply_active_ee_delta(
                    position_axis * (axis * linear_step),
                    pitch_axis * (axis * pitch_step),
                    roll_axis * (axis * roll_step),
                )

            for side in ("left", "right"):
                first_joint = f"{side}_joint7"
                second_joint = f"{side}_joint8"
                if first_joint not in specs_by_joint:
                    continue
                first_min, first_max = specs_by_joint[first_joint]
                second_min, second_max = specs_by_joint[second_joint]
                max_open = min(
                    abs(min(0.0, first_min)),
                    abs(min(0.0, second_min)),
                )
                open_amount = (
                    0.0
                    if gripper_closed[side]
                    else -args.teleop_gripper_open_scale * max_open
                )
                first_amount = min(max(open_amount, first_min), first_max)
                second_amount = min(max(open_amount, second_min), second_max)
                target_by_joint[first_joint] = first_amount
                target_by_joint[second_joint] = second_amount

            _set_gripper_drive_force(drive_specs, gripper_closed)
            _apply_joint_targets(drive_specs, target_by_joint, robot_articulations)
            _step_world(
                world,
                simulation_app,
                args.physics_steps_per_action,
                post_step_callback=lambda: _drive_grasped_cloth_particles(
                    env_spec,
                    gripper_closed,
                ),
            )
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)


def _step_world(
    world,
    simulation_app,
    substeps: int,
    render: bool = True,
    post_step_callback=None,
):
    for _ in range(substeps):
        if world is not None:
            world.step(render=render)
        elif simulation_app is not None:
            simulation_app.update()
        if post_step_callback is not None:
            post_step_callback()


def _create_camera(stage, path, position, target, focal_length=18.0, parent_prim=None):
    from pxr import Gf, Sdf, UsdGeom

    camera_name = Path(path).name
    camera_path = Sdf.Path(path)
    if parent_prim is not None:
        camera_path = parent_prim.GetPath().AppendChild(camera_name)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(float(focal_length))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 5.0))
    _set_xform(camera.GetPrim(), position, _look_at_quat_wxyz(position, target, up=(0.0, 1.0, 0.0)))
    camera.GetPrim().CreateAttribute("d435:rgbd", Sdf.ValueTypeNames.Bool).Set(True)
    camera.GetPrim().CreateAttribute("d435:depthUnits", Sdf.ValueTypeNames.String).Set("meters")
    return str(camera.GetPath())


def _create_cameras(stage, env_path: str = "/World") -> list[str]:
    camera_paths = [
        _create_camera(
            stage,
            _child_path(env_path, "OverviewCamera"),
            (1.45, -2.25, 1.15),
            (-0.25, -0.68, 0.28),
            focal_length=16.0,
        )
    ]

    d435_position = (0.029, -0.069, 0.022)
    d435_target = (0.0, 0.0, 0.13503)
    robot_root = _child_path(env_path, "PiperDualArm")
    for gripper_path, camera_name in (
        (
            f"{robot_root}/Geometry/base_link/left_base_link/left_link1/"
            "left_link2/left_link3/left_link4/left_link5/left_link6/"
            "left_gripper_base",
            "LeftD435",
        ),
        (
            f"{robot_root}/Geometry/base_link/right_base_link/right_link1/"
            "right_link2/right_link3/right_link4/right_link5/right_link6/"
            "right_gripper_base",
            "RightD435",
        ),
    ):
        link_prim = stage.GetPrimAtPath(gripper_path)
        if not link_prim.IsValid():
            camera_paths.append(
                _create_camera(
                    stage,
                    _child_path(env_path, camera_name),
                    (-0.15, -0.52 if "Left" in camera_name else -0.88, 0.62),
                    (0.0, -0.70, 0.16),
                    focal_length=18.0,
                )
            )
        else:
            camera_paths.append(
                _create_camera(
                    stage,
                    camera_name,
                    d435_position,
                    d435_target,
                    focal_length=18.0,
                    parent_prim=link_prim,
                )
            )
    return camera_paths


def _record_frames(
    camera_paths: list[str],
    args: argparse.Namespace,
    world=None,
    simulation_app=None,
    drive_specs=None,
    rng=None,
    robot_articulations=None,
    env_specs=None,
    profile=None,
):
    import omni.replicator.core as rep

    output_dir = (args.output_root / "replicator_frames").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera_render_mode == "tiled":
        render_products = [
            rep.create.render_product_tiled(
                cameras=camera_paths,
                tile_resolution=(args.width, args.height),
            )
        ]
    else:
        render_products = [
            rep.create.render_product(camera_path, (args.width, args.height))
            for camera_path in camera_paths
        ]
    writer = None
    annotators = []
    if args.write_frames:
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(output_dir),
            rgb=True,
            distance_to_camera=True,
        )
        writer.attach(render_products)
    else:
        annotators = [
            rep.AnnotatorRegistry.get_annotator(
                "rgb",
                device="cuda",
                do_array_copy=False,
            ),
            rep.AnnotatorRegistry.get_annotator(
                "distance_to_camera",
                device="cuda",
                do_array_copy=False,
            ),
        ]
        for annotator in annotators:
            for render_product in render_products:
                annotator.attach(render_product)

    env_step_seconds = []
    render_seconds = []
    reset_seconds = []
    action_index = 0
    episode_action_count = 0
    frame_index = 0
    while frame_index < args.steps:
        should_reset = (
            args.reset_every_steps > 0
            and episode_action_count >= args.reset_every_steps
        )
        if should_reset:
            reset_start = time.perf_counter()
            if writer is not None:
                writer.detach()
            _reset_scene(world, simulation_app, args.physics_backend)
            _reset_robot_targets(env_specs or [], args.reset_arm_zero)
            if writer is not None:
                writer.attach(render_products)
            reset_seconds.append(time.perf_counter() - reset_start)
            episode_action_count = 0
            if args.record_reset_frames:
                render_start = time.perf_counter()
                rep.orchestrator.step(rt_subframes=args.rt_subframes)
                render_seconds.append(time.perf_counter() - render_start)
                frame_index += 1
                continue

        env_step_start = time.perf_counter()
        if env_specs is not None and rng is not None:
            _apply_random_arm_action_batch(
                env_specs,
                rng,
                action_index,
                log_actions=args.log_random_actions,
                action_sampling=args.action_sampling,
                newton_random_joint_scale=args.newton_random_joint_scale,
            )
        elif drive_specs is not None and rng is not None:
            _apply_random_arm_action(
                drive_specs,
                rng,
                action_index,
                robot_articulations=robot_articulations,
                log_actions=args.log_random_actions,
                action_sampling=args.action_sampling,
                newton_random_joint_scale=args.newton_random_joint_scale,
            )
        _step_world(world, simulation_app, args.physics_steps_per_action, render=False)
        env_step_seconds.append(time.perf_counter() - env_step_start)
        render_start = time.perf_counter()
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        render_seconds.append(time.perf_counter() - render_start)
        action_index += 1
        episode_action_count += 1
        frame_index += 1

    wait_start = time.perf_counter()
    rep.orchestrator.wait_until_complete()
    wait_seconds = time.perf_counter() - wait_start
    if writer is not None:
        writer.detach()
    for annotator in annotators:
        annotator.detach()
    if profile is not None:
        profile["steps"] = args.steps
        profile["camera_count"] = len(camera_paths)
        profile["camera_render_mode"] = args.camera_render_mode
        profile["rt_subframes"] = args.rt_subframes
        profile["write_frames"] = args.write_frames
        profile["env_step_total_seconds"] = sum(env_step_seconds)
        profile["env_step_mean_seconds"] = (
            sum(env_step_seconds) / len(env_step_seconds)
            if env_step_seconds
            else 0.0
        )
        profile["render_total_seconds"] = sum(render_seconds)
        profile["render_mean_seconds"] = (
            sum(render_seconds) / len(render_seconds) if render_seconds else 0.0
        )
        profile["replicator_wait_seconds"] = wait_seconds
        profile["in_record_reset_total_seconds"] = sum(reset_seconds)
        profile["in_record_reset_count"] = len(reset_seconds)
        profile["action_count"] = len(env_step_seconds)
    return output_dir


def _reset_scene(world, simulation_app, physics_backend: str):
    if world is not None:
        world.reset(soft=False)
        return

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_current_time(0.0)
    timeline.play()
    _step_world(None, simulation_app, 1)


def _depth_to_rgb(depth_image):
    import numpy as np
    from PIL import Image

    finite_depth = np.asarray(depth_image, dtype=np.float32)
    valid_mask = np.isfinite(finite_depth)
    if not valid_mask.any():
        return Image.fromarray(np.zeros((*finite_depth.shape, 3), dtype=np.uint8))

    clipped_depth = np.clip(finite_depth, 0.0, 2.5)
    normalized_depth = 1.0 - clipped_depth / 2.5
    depth_uint8 = (255.0 * np.where(valid_mask, normalized_depth, 0.0)).astype(np.uint8)
    return Image.fromarray(np.stack((depth_uint8, depth_uint8, depth_uint8), axis=-1))


def _compose_rgbd_video(frame_dir: Path, video_path: Path, fps: int):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    camera_dirs = sorted(path for path in frame_dir.iterdir() if path.name.startswith("Replicator"))
    flat_rgb_paths = sorted(frame_dir.glob("rgb_*.png"))
    if not camera_dirs:
        if not flat_rgb_paths:
            raise RuntimeError(f"No Replicator RGB frames found in {frame_dir}")
        frame_count = len(flat_rgb_paths)
        composed_frames_dir = video_path.parent / "composed_frames"
        if composed_frames_dir.exists():
            shutil.rmtree(composed_frames_dir)
        composed_frames_dir.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        video_writer = None
        try:
            for frame_index in range(frame_count):
                rgb_path = frame_dir / f"rgb_{frame_index:04d}.png"
                depth_path = frame_dir / "distance_to_camera" / f"distance_to_camera_{frame_index:04d}.npy"
                if not depth_path.exists():
                    depth_path = frame_dir / f"distance_to_camera_{frame_index:04d}.npy"
                rgb_image = Image.open(rgb_path).convert("RGB")
                depth_image = _depth_to_rgb(np.load(depth_path))
                row = Image.new("RGB", (rgb_image.width * 2, rgb_image.height), color=(0, 0, 0))
                row.paste(rgb_image, (0, 0))
                row.paste(depth_image.resize(rgb_image.size), (rgb_image.width, 0))
                draw = ImageDraw.Draw(row)
                draw.rectangle((0, 0, 240, 22), fill=(0, 0, 0))
                draw.text((6, 4), "tiled batch RGB | depth", fill=(255, 255, 255))

                composed_path = composed_frames_dir / f"frame_{frame_index:04d}.png"
                row.save(composed_path)
                composed_rgb = np.asarray(row)
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        float(fps),
                        (composed_rgb.shape[1], composed_rgb.shape[0]),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open OpenCV VideoWriter for {video_path}")
                video_writer.write(cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
        finally:
            if video_writer is not None:
                video_writer.release()
        return video_path

    frame_count = min(len(list((camera_dir / "rgb").glob("rgb_*.png"))) for camera_dir in camera_dirs)
    if frame_count <= 0:
        raise RuntimeError(f"No RGB frames found in {frame_dir}")

    labels = ("overview", "left_d435", "right_d435")
    composed_frames_dir = video_path.parent / "composed_frames"
    if composed_frames_dir.exists():
        shutil.rmtree(composed_frames_dir)
    composed_frames_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    try:
        for frame_index in range(frame_count):
            rows = []
            for camera_index, camera_dir in enumerate(camera_dirs):
                rgb_path = camera_dir / "rgb" / f"rgb_{frame_index:04d}.png"
                depth_path = camera_dir / "distance_to_camera" / f"distance_to_camera_{frame_index:04d}.npy"
                rgb_image = Image.open(rgb_path).convert("RGB")
                depth_image = _depth_to_rgb(np.load(depth_path))
                row = Image.new("RGB", (rgb_image.width * 2, rgb_image.height), color=(0, 0, 0))
                row.paste(rgb_image, (0, 0))
                row.paste(depth_image.resize(rgb_image.size), (rgb_image.width, 0))
                draw = ImageDraw.Draw(row)
                label = labels[camera_index] if camera_index < len(labels) else camera_dir.name
                draw.rectangle((0, 0, 190, 22), fill=(0, 0, 0))
                draw.text((6, 4), f"{label} RGB | depth", fill=(255, 255, 255))
                rows.append(row)

            composed = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
            for row_index, row in enumerate(rows):
                composed.paste(row, (0, row.height * row_index))

            composed_path = composed_frames_dir / f"frame_{frame_index:04d}.png"
            composed.save(composed_path)
            composed_rgb = np.asarray(composed)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    float(fps),
                    (composed_rgb.shape[1], composed_rgb.shape[0]),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open OpenCV VideoWriter for {video_path}")
            video_writer.write(cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
    finally:
        if video_writer is not None:
            video_writer.release()

    return video_path


def _save_stage(stage, stage_path: Path):
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(stage_path))


def _print_optional_video_hint(frame_dir: Path):
    if shutil.which("ffmpeg") is None:
        return

    # BasicWriter names files by render product.  Leave MP4 composition to a
    # deliberate follow-up command because there are three RGB-D streams.
    command = [
        "find",
        str(frame_dir),
        "-maxdepth",
        "1",
        "-type",
        "f",
        "-name",
        "'rgb_*.png'",
    ]
    print("RGB/depth frames written. Example discovery command:")
    print(" ".join(command))


def main():
    args = _parse_args()
    total_start = time.perf_counter()
    profile = {
        "script": "isaacsim_newton_scene.py",
        "physics_backend": args.physics_backend,
        "cloth_mode": args.cloth_mode,
        "cloth_asset": args.cloth_asset,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "camera_render_mode": args.camera_render_mode,
        "physics_steps_per_action": args.physics_steps_per_action,
        "rt_subframes": args.rt_subframes,
        "write_frames": args.write_frames,
        "action_sampling": args.action_sampling,
        "include_gripper_actions": args.include_gripper_actions,
        "newton_random_joint_scale": args.newton_random_joint_scale,
        "profile_repeats": args.profile_repeats,
    }
    if args.teleop and args.headless:
        raise ValueError("Isaac Sim teleop requires a non-headless app window.")
    if args.teleop and args.batch_size != 1:
        raise ValueError("Minimal Isaac Sim teleop currently supports batch size 1.")
    if args.teleop and args.physics_backend != "physx":
        raise ValueError("Minimal Isaac Sim teleop currently requires PhysX.")
    if args.physics_backend == "newton" and args.cloth_mode == "physical":
        raise ValueError(
            "Isaac Sim particle cloth is a PhysX feature. Use "
            "--physics-backend physx --cloth-mode physical, or use "
            "--physics-backend newton --cloth-mode visual."
        )
    # SimulationApp forwards unknown sys.argv entries to Kit.  Keep the custom
    # recorder arguments out of Kit's parser.
    sys.argv = [sys.argv[0]]
    _log("starting SimulationApp")
    app_start = time.perf_counter()
    simulation_app = _start_simulation_app(args)
    profile["simulation_app_start_seconds"] = time.perf_counter() - app_start
    world = None
    rng = (
        np.random.default_rng(args.random_seed)
        if args.action_sampling == "newton"
        else random.Random(args.random_seed)
    )
    try:
        _log("enabling extensions")
        _enable_extensions(args.physics_backend)
        if args.physics_backend == "physx":
            _log("creating PhysX world")
            stage, world = _create_physx_world()
        else:
            _log("creating Newton stage")
            stage = _create_newton_stage()

        _log("creating materials")
        materials = {
            "table": _create_material(stage, "/World/Materials/Table", (0.78, 0.78, 0.74)),
            "cloth": _create_material(stage, "/World/Materials/Cloth", (0.10, 0.35, 0.82)),
            "table_physics": _create_physics_material(
                stage,
                "/World/PhysicsMaterials/TableLowFriction",
                TABLE_PHYSICS_FRICTION[0],
                TABLE_PHYSICS_FRICTION[1],
                TABLE_FRICTION_COMBINE_MODE,
            ),
            "gripper_physics": _create_physics_material(
                stage,
                "/World/PhysicsMaterials/GripperHighFriction",
                GRIPPER_PHYSICS_FRICTION[0],
                GRIPPER_PHYSICS_FRICTION[1],
                GRIPPER_FRICTION_COMBINE_MODE,
            ),
        }

        _log("converting/importing robot URDF")
        robot_urdf_path = _isaac_sim_robot_urdf(
            args.robot_usd_dir,
            args.simple_gripper_collisions,
        )
        robot_usd_path = _convert_robot_urdf(
            args.robot_usd_dir,
            args.merge_robot_meshes,
            robot_urdf_path,
            args.simple_gripper_collisions,
        )
        _log(f"referencing robot USD: {robot_usd_path}")
        env_paths = _env_paths(args.batch_size)
        grid_cols = int(math.ceil(math.sqrt(args.batch_size)))
        env_specs = []
        camera_paths = []
        for env_index, env_path in enumerate(env_paths):
            _log(f"creating env {env_index + 1}/{args.batch_size}: {env_path}")
            _create_env_root(stage, env_path, env_index, args.env_spacing, grid_cols)
            _reference_robot(stage, robot_usd_path, env_path)
            if args.fix_imported_arm_bases:
                _fix_imported_arm_bases(stage, args.physics_backend, env_path)
                _repair_imported_joint_frames(
                    stage,
                    robot_urdf_path,
                    args.physics_backend,
                    env_path,
                )
            if env_index == 0:
                _log_dual_arm_status(stage, env_path)
            drive_specs = (
                _configure_robot_drives(
                    stage,
                    args.include_gripper_actions or args.teleop,
                    args.arm_random_scale,
                    args.gripper_random_scale,
                    env_path,
                )
                if args.random_arm_actions or args.teleop
                else None
            )
            robot_articulations = (
                _create_robot_articulations(world, env_path, env_index)
                if world is not None and drive_specs is not None
                else None
            )
            _apply_gripper_physics_material(
                stage,
                materials["gripper_physics"],
                env_path,
            )
            _create_table(
                stage,
                materials["table"],
                materials["table_physics"],
                env_path,
            )
            if args.cloth_mode == "physical":
                cloth = _create_physical_cloth(
                    stage,
                    world,
                    args.cloth_asset,
                    materials["cloth"],
                    args,
                    env_path,
                    env_index,
                )
            else:
                cloth = None
                _create_cloth_visual(stage, args.cloth_asset, materials["cloth"], env_path)
            env_camera_paths = _create_cameras(stage, env_path)
            camera_paths.extend(env_camera_paths)
            link_paths = _imported_link_paths(stage, robot_urdf_path, env_path)
            finger_link_paths = {
                side: {
                    "link7": link_paths[f"{side}_link7"],
                    "link8": link_paths[f"{side}_link8"],
                }
                for side in ("left", "right")
            }
            adhesive_surfaces = _create_adhesive_surfaces(
                stage,
                env_path,
                finger_link_paths,
            )
            env_specs.append(
                {
                    "env_index": env_index,
                    "env_path": env_path,
                    "cloth": cloth,
                    "adhesive_surfaces": adhesive_surfaces,
                    "drive_specs": drive_specs,
                    "robot_articulations": robot_articulations,
                    "camera_paths": env_camera_paths,
                }
            )
        profile["camera_count"] = len(camera_paths)
        _log(f"saving stage: {args.stage_path}")
        _save_stage(stage, args.stage_path)

        if world is None:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

        profile["scene_init_seconds"] = time.perf_counter() - total_start
        print(f"Saved Isaac Sim scene to: {args.stage_path}")
        print(f"Physics backend: {args.physics_backend}")
        print(f"Cloth mode: {args.cloth_mode}")
        print(f"Robot USD asset: {robot_usd_path}")
        print("Camera prims:")
        for camera_path in camera_paths:
            print(f"  {camera_path}")

        if args.teleop:
            _log("resetting scene for teleop")
            _reset_scene(world, simulation_app, args.physics_backend)
            _reset_robot_targets(env_specs, True)
            _log_cloth_bounds(env_specs, "after teleop reset")
            _run_teleop(world, simulation_app, env_specs[0], args)
            profile["total_seconds"] = time.perf_counter() - total_start
            if args.profile_json is not None:
                args.profile_json.parent.mkdir(parents=True, exist_ok=True)
                args.profile_json.write_text(json.dumps(profile, indent=2) + "\n")
            return

        repeat_profiles = []
        for repeat_index in range(args.profile_repeats):
            repeat_profile = {"repeat_index": repeat_index}
            _log(f"resetting scene for repeat {repeat_index + 1}")
            reset_start = time.perf_counter()
            _reset_scene(world, simulation_app, args.physics_backend)
            _reset_robot_targets(env_specs, args.reset_arm_zero)
            _log_cloth_bounds(env_specs, f"after reset {repeat_index + 1}")
            repeat_profile["reset_seconds"] = time.perf_counter() - reset_start

            if args.record:
                _log("recording Replicator frames")
                record_start = time.perf_counter()
                frame_dir = _record_frames(
                    camera_paths,
                    args,
                    world=world,
                    simulation_app=simulation_app,
                    drive_specs=None if args.batch_size > 1 else env_specs[0]["drive_specs"],
                    rng=rng,
                    robot_articulations=(
                        None
                        if args.batch_size > 1
                        else env_specs[0]["robot_articulations"]
                    ),
                    env_specs=env_specs if args.batch_size > 1 else None,
                    profile=repeat_profile,
                )
                repeat_profile["record_total_seconds"] = (
                    time.perf_counter() - record_start
                )
                if args.compose_video:
                    _log("composing RGB-D video")
                    compose_start = time.perf_counter()
                    video_path = _compose_rgbd_video(frame_dir, args.video_path, args.fps)
                    repeat_profile["video_compose_seconds"] = (
                        time.perf_counter() - compose_start
                    )
                    print(f"RGB-D video written to: {video_path}")
                print(f"Replicator frames written to: {frame_dir}")
            else:
                env_step_seconds = []
                for action_index in range(args.steps):
                    env_step_start = time.perf_counter()
                    if args.batch_size > 1:
                        _apply_random_arm_action_batch(
                            env_specs,
                            rng,
                            action_index,
                            log_actions=args.log_random_actions,
                            action_sampling=args.action_sampling,
                            newton_random_joint_scale=args.newton_random_joint_scale,
                        )
                    elif env_specs[0]["drive_specs"] is not None:
                        _apply_random_arm_action(
                            env_specs[0]["drive_specs"],
                            rng,
                            action_index,
                            robot_articulations=env_specs[0]["robot_articulations"],
                            log_actions=args.log_random_actions,
                            action_sampling=args.action_sampling,
                            newton_random_joint_scale=args.newton_random_joint_scale,
                        )
                    _step_world(
                        world,
                        simulation_app if world is None else None,
                        args.physics_steps_per_action,
                    )
                    env_step_seconds.append(time.perf_counter() - env_step_start)
                _log_cloth_bounds(env_specs, f"after steps {repeat_index + 1}")
                repeat_profile["steps"] = args.steps
                repeat_profile["env_step_total_seconds"] = sum(env_step_seconds)
                repeat_profile["env_step_mean_seconds"] = (
                    sum(env_step_seconds) / len(env_step_seconds)
                    if env_step_seconds
                    else 0.0
                )
            repeat_profiles.append(repeat_profile)
        if repeat_profiles:
            profile.update(repeat_profiles[-1])
        profile["repeat_profiles"] = repeat_profiles
        profile["total_seconds"] = time.perf_counter() - total_start
        if args.profile_json is not None:
            args.profile_json.parent.mkdir(parents=True, exist_ok=True)
            args.profile_json.write_text(json.dumps(profile, indent=2) + "\n")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        _log("closing SimulationApp")
        simulation_app.close()


if __name__ == "__main__":
    main()
