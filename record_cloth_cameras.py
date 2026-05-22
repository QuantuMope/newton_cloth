"""Record RGB/depth views from the Newton cloth scene.

The scene itself is still the standalone Newton demo from ``cloth_teleop.py``.
This script adds three virtual RGB-D cameras for offline recording:

* overview
* left wrist D435 pose, attached to left_link6
* right wrist D435 pose, attached to right_link6
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

import cloth_teleop


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "recordings" / "cloth_cameras"
_ORIGINAL_WP_MESH = wp.Mesh


def _install_warp_mesh_compatibility_patch():
    """Drop the removed ``device=`` kwarg used by this Newton checkout."""

    def mesh_compat(*args, **kwargs):
        kwargs.pop("device", None)
        return _ORIGINAL_WP_MESH(*args, **kwargs)

    wp.Mesh = mesh_compat


class DummyViewer:
    """Minimal viewer shim needed by cloth_teleop.Example."""

    def set_model(self, model):
        pass

    def set_camera(self, **kwargs):
        pass

    def apply_forces(self, state):
        pass


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def _quat_xyzw_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float32,
        )

    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] -
                          matrix[2, 2]) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ],
            dtype=np.float32,
        )
    if axis == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] -
                          matrix[2, 2]) * 2.0
        return np.array(
            [
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ],
            dtype=np.float32,
        )

    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] -
                      matrix[1, 1]) * 2.0
    return np.array(
        [
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ],
        dtype=np.float32,
    )


def _look_at_transform(position, target, up=(0.0, 0.0, 1.0)) -> wp.transformf:
    position_np = np.asarray(position, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    up_np = np.asarray(up, dtype=np.float32)
    forward = _normalize(target_np - position_np)
    right = _normalize(np.cross(forward, up_np))
    camera_up = _normalize(np.cross(right, forward))
    # SensorTiledCamera uses camera-space -Z as the optical forward axis.
    rotation = np.column_stack((right, camera_up, -forward))
    quat_xyzw = _quat_xyzw_from_matrix(rotation)
    return wp.transformf(wp.vec3f(*position_np), wp.quatf(*quat_xyzw))


def _as_transformf(transform_like) -> wp.transformf:
    transform_np = np.asarray(transform_like, dtype=np.float32)
    return wp.transformf(
        wp.vec3f(*transform_np[:3]),
        wp.quatf(*transform_np[3:]),
    )


@wp.kernel
def _update_camera_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    left_link6_indices: wp.array(dtype=wp.int32),
    right_link6_indices: wp.array(dtype=wp.int32),
    overview_tfs: wp.array(dtype=wp.transformf),
    d435_local_tf: wp.array(dtype=wp.transformf),
    camera_tfs: wp.array2d(dtype=wp.transformf),
):
    world_index = wp.tid()
    camera_tfs[0, world_index] = overview_tfs[world_index]
    camera_tfs[1, world_index] = wp.transform_multiply(
        body_q[left_link6_indices[world_index]],
        d435_local_tf[0],
    )
    camera_tfs[2, world_index] = wp.transform_multiply(
        body_q[right_link6_indices[world_index]],
        d435_local_tf[0],
    )


class ClothCameraRecorder:
    def __init__(self, args: argparse.Namespace):
        example_args = SimpleNamespace(
            cloth_asset=args.cloth_asset,
            grasp_mode="normal",
            batch_size=args.batch_size,
            env_spacing=args.env_spacing,
            skip_ik_setup=args.initial_pose == "isaac" or args.batch_size > 1,
            frame_dt=args.action_dt,
            sim_substeps=args.sim_substeps,
            use_joint_position_targets=args.use_joint_position_targets,
            joint_target_ke=args.joint_target_ke,
            joint_target_kd=args.joint_target_kd,
            max_triangle_pairs=args.max_triangle_pairs,
            collision_broad_phase=args.collision_broad_phase,
            cloth_iterations=args.cloth_iterations,
        )
        self.example = cloth_teleop.Example(DummyViewer(), example_args)
        self.model = self.example.model
        self.state = self.example.state_0
        self.width = args.width
        self.height = args.height
        self.world_count = self.model.world_count
        self.camera_names = ("overview", "left_d435", "right_d435")
        self.camera_count = len(self.camera_names)
        self.tile_count = self.world_count * self.camera_count
        self.reset_every_steps = args.reset_every_steps
        self.record_reset_frames = args.record_reset_frames
        self.reset_arm_zero = args.reset_arm_zero
        self.use_joint_position_targets = args.use_joint_position_targets
        self.sync_timing = args.sync_timing
        self.rng = np.random.default_rng(args.seed)
        self.random_joint_scale = args.random_joint_scale

        render_order_by_name = {
            "pixel": SensorTiledCamera.RenderOrder.PIXEL_PRIORITY,
            "view": SensorTiledCamera.RenderOrder.VIEW_PRIORITY,
            "tiled": SensorTiledCamera.RenderOrder.TILED,
        }
        render_config = SensorTiledCamera.RenderConfig(
            enable_textures=True,
            enable_shadows=True,
            render_order=render_order_by_name[args.render_order],
            max_distance=args.max_depth,
        )
        self.sensor = SensorTiledCamera(self.model, config=render_config)
        self.sensor.utils.create_default_light(enable_shadows=True)

        fovs = np.deg2rad(
            np.array([args.overview_fov_deg, args.d435_fov_deg,
                      args.d435_fov_deg],
                     dtype=np.float32)
        )
        self.camera_rays = self.sensor.utils.compute_pinhole_camera_rays(
            self.width, self.height, fovs
        )
        self.color_image = self.sensor.utils.create_color_image_output(
            self.width, self.height, self.camera_count
        )
        self.depth_image = self.sensor.utils.create_depth_image_output(
            self.width, self.height, self.camera_count
        )
        self.depth_rgba = wp.empty(
            (self.tile_count, self.height, self.width, 4),
            dtype=wp.uint8,
            device=self.color_image.device,
        )

        first_left_link6_index = self.model.body_label.index(
            "piper_x_description/left_link6"
        )
        first_right_link6_index = self.model.body_label.index(
            "piper_x_description/right_link6"
        )
        body_start = self.model.body_world_start.numpy()
        first_world_body_start = body_start[0]
        left_link6_offset = first_left_link6_index - first_world_body_start
        right_link6_offset = first_right_link6_index - first_world_body_start
        self.left_link6_indices = (
            body_start[:self.world_count] + left_link6_offset
        )
        self.right_link6_indices = (
            body_start[:self.world_count] + right_link6_offset
        )
        self.left_link6_indices_wp = wp.array(
            self.left_link6_indices.astype(np.int32),
            dtype=wp.int32,
            device=self.state.body_q.device,
        )
        self.right_link6_indices_wp = wp.array(
            self.right_link6_indices.astype(np.int32),
            dtype=wp.int32,
            device=self.state.body_q.device,
        )
        # Use the D435 wrist mount position from piper_x_description_d435.urdf.
        # The URDF link frame describes the camera body, not an optical frame.
        # Aim the virtual RGB-D optical axis through the gripper workspace.
        self.d435_local_tf = _look_at_transform(
            position=(0.029, -0.069, 0.022),
            target=(0.0, 0.0, 0.13503),
            up=(0.0, 1.0, 0.0),
        )
        self.overview_tf = _look_at_transform(
            position=(1.8, -1.8, 1.2),
            target=(0.05, -0.65, 0.16),
        )
        self.overview_position = np.array((1.8, -1.8, 1.2), dtype=np.float32)
        self.overview_target = np.array((0.05, -0.65, 0.16), dtype=np.float32)
        self.overview_position_from_target = (
            self.overview_position - self.overview_target
        )
        if args.initial_pose == "isaac":
            self._set_zero_joint_pose()
        self.initial_overview_targets = self._compute_initial_overview_targets()
        self.overview_tfs = wp.array(
            [
                _look_at_transform(
                    overview_target + self.overview_position_from_target,
                    overview_target,
                )
                for overview_target in self.initial_overview_targets
            ],
            dtype=wp.transformf,
            device=self.state.body_q.device,
        )
        self.d435_local_tf_wp = wp.array(
            [self.d435_local_tf],
            dtype=wp.transformf,
            device=self.state.body_q.device,
        )
        self.camera_transforms = wp.empty(
            (self.camera_count, self.world_count),
            dtype=wp.transformf,
            device=self.state.body_q.device,
        )

        self.initial_joint_q = self.state.joint_q.numpy().copy()
        local_arm_joint_indices = np.array(
            [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
            dtype=np.int32,
        )
        joint_coord_start = self.model.joint_coord_world_start.numpy()
        self.arm_joint_indices = np.concatenate(
            [
                joint_coord_start[world_index] + local_arm_joint_indices
                for world_index in range(self.world_count)
            ]
        ).astype(np.int32)
        self.initial_state_0 = self._snapshot_state(self.example.state_0)
        self.initial_state_1 = self._snapshot_state(self.example.state_1)
        self.initial_model_joint_q = self.model.joint_q.numpy().copy()
        self.initial_model_joint_qd = self.model.joint_qd.numpy().copy()
        self.initial_control_joint_target_pos = (
            self.example.control.joint_target_pos.numpy().copy()
        )
        self.initial_particle_flags = self.model.particle_flags.numpy().copy()
        self.joint_limit_lower = self.model.joint_limit_lower.numpy().copy()
        self.joint_limit_upper = self.model.joint_limit_upper.numpy().copy()

        newton.geometry.build_bvh_shape(self.model, self.state)
        newton.geometry.build_bvh_particle(self.model, self.state)

    def _set_zero_joint_pose(self):
        zero_joint_q = np.zeros(
            self.example.state_0.joint_q.shape[0],
            dtype=np.float32,
        )
        zero_joint_qd = np.zeros(
            self.example.state_0.joint_qd.shape[0],
            dtype=np.float32,
        )
        self.example.state_0.joint_q.assign(zero_joint_q)
        self.example.state_1.joint_q.assign(zero_joint_q)
        self.example.state_0.joint_qd.assign(zero_joint_qd)
        self.example.state_1.joint_qd.assign(zero_joint_qd)
        self.model.joint_q.assign(zero_joint_q)
        self.model.joint_qd.assign(zero_joint_qd)
        self.example.control.joint_target_pos.assign(zero_joint_q)
        self.example.target_joint_qd.zero_()
        newton.eval_fk(
            self.model,
            self.example.state_0.joint_q,
            self.example.state_0.joint_qd,
            self.example.state_0,
        )
        newton.eval_fk(
            self.model,
            self.example.state_1.joint_q,
            self.example.state_1.joint_qd,
            self.example.state_1,
        )
        self.state = self.example.state_0

    @staticmethod
    def _snapshot_state(state):
        field_names = (
            "body_q",
            "body_qd",
            "particle_q",
            "particle_qd",
            "particle_f",
            "joint_q",
            "joint_qd",
        )
        return {
            field_name: getattr(state, field_name).numpy().copy()
            for field_name in field_names
        }

    @staticmethod
    def _restore_state(state, snapshot):
        for field_name, field_value in snapshot.items():
            getattr(state, field_name).assign(field_value)

    def reset_scene(self):
        self._restore_state(self.example.state_0, self.initial_state_0)
        self._restore_state(self.example.state_1, self.initial_state_1)
        self.model.joint_q.assign(self.initial_model_joint_q)
        self.model.joint_qd.assign(self.initial_model_joint_qd)
        self.example.control.joint_target_pos.assign(
            self.initial_control_joint_target_pos
        )
        self.example.target_joint_qd.zero_()
        self.model.particle_flags.assign(self.initial_particle_flags)
        for grasp in self.example.grasps.values():
            grasp["count"].zero_()
            grasp["indices"].zero_()
            grasp["local_offsets"].zero_()
            grasp["active"] = False
        self.example.sim_time = 0.0
        self.state = self.example.state_0
        if self.reset_arm_zero:
            self._set_zero_joint_pose()
        newton.geometry.refit_bvh_shape(self.model, self.state)
        newton.geometry.refit_bvh_particle(self.model, self.state)

    def _particle_ranges_by_world(self) -> list[tuple[int, int]]:
        particle_count = self.state.particle_q.shape[0]
        if self.world_count == 1:
            return [(0, particle_count)]

        particle_world_start = self.model.particle_world_start.numpy()
        valid_starts = (
            particle_world_start.size >= self.world_count + 1
            and particle_world_start[0] == 0
        )
        if not valid_starts:
            return [(0, particle_count)]

        return [
            (int(particle_world_start[world_index]),
             int(particle_world_start[world_index + 1]))
            for world_index in range(self.world_count)
        ]

    def _compute_initial_overview_targets(self) -> np.ndarray:
        particle_q = self.state.particle_q.numpy()
        targets = []
        for start, end in self._particle_ranges_by_world():
            if end <= start:
                targets.append(self.overview_target)
                continue
            particle_centroid = particle_q[start:end, :3].mean(axis=0)
            targets.append(
                np.array(
                    [
                        particle_centroid[0],
                        particle_centroid[1],
                        self.overview_target[2],
                    ],
                    dtype=np.float32,
                )
            )
        return np.asarray(targets, dtype=np.float32)

    def _update_camera_transforms(self):
        wp.launch(
            _update_camera_transforms_kernel,
            dim=self.world_count,
            inputs=[
                self.state.body_q,
                self.left_link6_indices_wp,
                self.right_link6_indices_wp,
                self.overview_tfs,
                self.d435_local_tf_wp,
            ],
            outputs=[self.camera_transforms],
        )

    def _apply_random_arm_motion(self, step_index: int):
        target_q = self.initial_joint_q.copy()
        phase = 0.65 * step_index
        noise = self.rng.uniform(-1.0, 1.0, size=self.arm_joint_indices.size)
        offsets = self.random_joint_scale * np.sin(phase + noise)
        target_q[self.arm_joint_indices] += offsets

        target_q = np.clip(target_q, self.joint_limit_lower, self.joint_limit_upper)

        if self.use_joint_position_targets:
            self.example.control.joint_target_pos.assign(target_q.astype(np.float32))
            self.example.target_joint_qd.zero_()
        else:
            current_q = self.state.joint_q.numpy()
            joint_qd = (target_q - current_q) / self.example.frame_dt
            self.example.target_joint_qd.assign(joint_qd.astype(np.float32))
        self.example.simulate()
        if self.sync_timing:
            wp.synchronize()
        self.state = self.example.state_0

    def _render_rgbd(self, readback: bool):
        _install_warp_mesh_compatibility_patch()
        newton.geometry.refit_bvh_shape(self.model, self.state)
        newton.geometry.refit_bvh_particle(self.model, self.state)
        self._update_camera_transforms()
        self.sensor.update(
            self.state,
            self.camera_transforms,
            self.camera_rays,
            color_image=self.color_image,
            depth_image=self.depth_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        if not readback:
            if self.sync_timing:
                wp.synchronize()
            return None, None

        color_rgba = self.sensor.utils.to_rgba_from_color(self.color_image)
        self.sensor.utils.to_rgba_from_depth(
            self.depth_image,
            depth_range=(0.0, 2.5),
            out_buffer=self.depth_rgba,
        )
        color_rgba_np = color_rgba.numpy()
        depth_rgba_np = self.depth_rgba.numpy()
        return color_rgba_np, depth_rgba_np

    def _compose_frame(
        self,
        step_index: int,
        color_rgba: np.ndarray,
        depth_rgba: np.ndarray,
    ) -> Image.Image:
        tile_w = self.width
        tile_h = self.height
        header_h = 24
        grid_cols = math.ceil(math.sqrt(self.tile_count))
        grid_rows = math.ceil(self.tile_count / grid_cols)
        canvas = Image.new(
            "RGB",
            (grid_cols * 2 * tile_w, grid_rows * (tile_h + header_h)),
            color=(20, 20, 20),
        )
        draw = ImageDraw.Draw(canvas)

        for tile_index in range(self.tile_count):
            world_index = tile_index // self.camera_count
            camera_index = tile_index % self.camera_count
            camera_name = f"env_{world_index} {self.camera_names[camera_index]}"
            grid_x = tile_index % grid_cols
            grid_y = tile_index // grid_cols
            x0 = grid_x * 2 * tile_w
            y0 = grid_y * (tile_h + header_h)
            draw.text((x0 + 8, y0 + 5), f"{camera_name} rgb", fill=(235, 235, 235))
            draw.text(
                (x0 + tile_w + 8, y0 + 5),
                f"{camera_name} depth",
                fill=(235, 235, 235),
            )
            rgb = Image.fromarray(color_rgba[tile_index, :, :, :3], "RGB")
            depth = Image.fromarray(depth_rgba[tile_index, :, :, :3], "RGB")
            canvas.paste(rgb, (x0, y0 + header_h))
            canvas.paste(depth, (x0 + tile_w, y0 + header_h))

        draw.text((8, canvas.height - 18), f"step {step_index}", fill=(255, 255, 0))
        return canvas

    def record(
        self,
        output_dir: Path,
        steps: int,
        fps: int,
        compose_video: bool,
        profile: dict | None = None,
    ) -> Path | None:
        frames_dir = output_dir / "frames"
        if compose_video:
            frames_dir.mkdir(parents=True, exist_ok=True)
            for old_frame in frames_dir.glob("frame_*.png"):
                old_frame.unlink()

        step_seconds = []
        render_seconds = []
        frame_save_seconds = []
        reset_seconds = []
        action_index = 0
        episode_action_count = 0
        frame_index = 0
        while frame_index < steps:
            should_reset = (
                self.reset_every_steps > 0
                and episode_action_count >= self.reset_every_steps
            )
            if should_reset:
                reset_start = time.perf_counter()
                self.reset_scene()
                reset_seconds.append(time.perf_counter() - reset_start)
                episode_action_count = 0
                if self.record_reset_frames:
                    render_start = time.perf_counter()
                    color_rgba, depth_rgba = self._render_rgbd(
                        readback=compose_video
                    )
                    render_seconds.append(time.perf_counter() - render_start)
                    if compose_video:
                        save_start = time.perf_counter()
                        frame = self._compose_frame(frame_index, color_rgba, depth_rgba)
                        frame.save(frames_dir / f"frame_{frame_index:04d}.png")
                        frame_save_seconds.append(time.perf_counter() - save_start)
                    frame_index += 1
                    continue

            step_start = time.perf_counter()
            self._apply_random_arm_motion(action_index)
            step_seconds.append(time.perf_counter() - step_start)
            render_start = time.perf_counter()
            color_rgba, depth_rgba = self._render_rgbd(readback=compose_video)
            render_seconds.append(time.perf_counter() - render_start)
            if compose_video:
                save_start = time.perf_counter()
                frame = self._compose_frame(frame_index, color_rgba, depth_rgba)
                frame.save(frames_dir / f"frame_{frame_index:04d}.png")
                frame_save_seconds.append(time.perf_counter() - save_start)
            action_index += 1
            episode_action_count += 1
            frame_index += 1

        video_path = output_dir / "cloth_cameras_rgbd.mp4"
        encode_seconds = 0.0
        if compose_video:
            encode_start = time.perf_counter()
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required to encode the MP4 video.")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames_dir / "frame_%04d.png"),
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
            )
            encode_seconds = time.perf_counter() - encode_start
        if profile is not None:
            profile["steps"] = steps
            profile["env_step_total_seconds"] = sum(step_seconds)
            profile["env_step_mean_seconds"] = (
                sum(step_seconds) / len(step_seconds) if step_seconds else 0.0
            )
            profile["render_total_seconds"] = sum(render_seconds)
            profile["render_mean_seconds"] = (
                sum(render_seconds) / len(render_seconds)
                if render_seconds
                else 0.0
            )
            profile["frame_save_total_seconds"] = sum(frame_save_seconds)
            profile["frame_save_mean_seconds"] = (
                sum(frame_save_seconds) / len(frame_save_seconds)
                if frame_save_seconds
                else 0.0
            )
            profile["video_encode_seconds"] = encode_seconds
            profile["in_record_reset_total_seconds"] = sum(reset_seconds)
            profile["in_record_reset_count"] = len(reset_seconds)
            profile["action_count"] = len(step_seconds)
        return video_path if compose_video else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record overview and wrist-D435 RGB-D views."
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cloth-asset", choices=("grid", "shirt"), default="grid")
    parser.add_argument(
        "--initial-pose",
        choices=("isaac", "teleop"),
        default="isaac",
        help="Use Isaac-style zero joints, or cloth_teleop's IK-start pose.",
    )
    parser.add_argument(
        "--action-dt",
        type=float,
        default=0.1,
        help="Simulated seconds advanced for each random action.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=12,
        help="Newton physics substeps per action.",
    )
    parser.add_argument(
        "--use-joint-position-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive robot joints through Newton joint position targets.",
    )
    parser.add_argument("--joint-target-ke", type=float, default=8500.0)
    parser.add_argument("--joint-target-kd", type=float, default=450.0)
    parser.add_argument("--max-triangle-pairs", type=int, default=1000000)
    parser.add_argument(
        "--collision-broad-phase",
        choices=("nxn", "sap", "explicit"),
        default="nxn",
    )
    parser.add_argument("--cloth-iterations", type=int, default=10)
    parser.add_argument(
        "--render-order",
        choices=("pixel", "view", "tiled"),
        default="pixel",
    )
    parser.add_argument(
        "--sync-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize Warp after step/render phases for isolated timings.",
    )
    parser.add_argument("--random-joint-scale", type=float, default=0.18)
    parser.add_argument("--overview-fov-deg", type=float, default=45.0)
    parser.add_argument("--d435-fov-deg", type=float, default=58.0)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--reset-every-steps", type=int, default=0)
    parser.add_argument("--record-reset-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-arm-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compose-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "script": "record_cloth_cameras.py",
        "cloth_asset": args.cloth_asset,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "camera_count": 3 * args.batch_size,
        "profile_repeats": args.profile_repeats,
        "initial_pose": args.initial_pose,
        "action_dt": args.action_dt,
        "sim_substeps": args.sim_substeps,
        "use_joint_position_targets": args.use_joint_position_targets,
        "joint_target_ke": args.joint_target_ke,
        "joint_target_kd": args.joint_target_kd,
        "max_triangle_pairs": args.max_triangle_pairs,
        "collision_broad_phase": args.collision_broad_phase,
        "cloth_iterations": args.cloth_iterations,
        "render_order": args.render_order,
        "sync_timing": args.sync_timing,
        "random_joint_scale": args.random_joint_scale,
    }
    total_start = time.perf_counter()
    init_start = time.perf_counter()
    recorder = ClothCameraRecorder(args)
    profile["scene_init_seconds"] = time.perf_counter() - init_start
    repeat_profiles = []
    video_path = None
    for repeat_index in range(args.profile_repeats):
        repeat_profile = {"repeat_index": repeat_index}
        reset_start = time.perf_counter()
        recorder.reset_scene()
        repeat_profile["reset_seconds"] = time.perf_counter() - reset_start
        record_start = time.perf_counter()
        video_path = recorder.record(
            args.output_dir,
            args.steps,
            args.fps,
            args.compose_video,
            profile=repeat_profile,
        )
        repeat_profile["record_total_seconds"] = time.perf_counter() - record_start
        repeat_profiles.append(repeat_profile)
    if repeat_profiles:
        profile.update(repeat_profiles[-1])
    profile["repeat_profiles"] = repeat_profiles
    profile["total_seconds"] = time.perf_counter() - total_start
    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile, indent=2) + "\n")
    print(video_path)


if __name__ == "__main__":
    main()
