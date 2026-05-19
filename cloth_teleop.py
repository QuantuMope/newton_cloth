"""Dual-arm teleop scene with table cloth.

Controls:
    T: switch teleop between the left and right arm.
    I/K: apply a delta-IK step along world X from the current active end-effector pose.
    J/L: apply a delta-IK step along world Y from the current active end-effector pose.
    U/O: apply a delta-IK step along world Z from the current active end-effector pose.
    R/F: apply a delta-IK pitch step from the current active end-effector orientation.
    G: toggle the active gripper open/closed.

Options:
    --cloth-asset grid: use Newton's square cloth mesh from example_cloth_twist.py.
    --cloth-asset shirt: use Newton's unisex shirt mesh as the cloth.
    --grasp-mode constraint: attach nearby cloth particles when the gripper closes.
    --grasp-mode normal: only actuate the gripper, with no explicit cloth attachment.
"""

import os
from pathlib import Path

import numpy as np
import warp as wp
import warp.examples

import newton.usd
from pxr import Usd
import newton.usd

import newton
import newton.examples
import newton.ik as ik
from newton import ParticleFlags


ROOT_DIR = Path(__file__).resolve().parent
ROBOT_URDF_PATH = ROOT_DIR / "piper_x_description_dualarm.urdf"


@wp.kernel
def compute_delta_joint_qd(
    current_q: wp.array[float],
    target_q: wp.array[float],
    inv_dt: float,
    joint_qd: wp.array[float],
):
    i = wp.tid()
    joint_qd[i] = (target_q[i] - current_q[i]) * inv_dt


@wp.kernel
def compute_gripper_qd(
    current_q: wp.array[float],
    gripper_target_q: wp.array[float],
    gripper_joint_start: int,
    inv_dt: float,
    joint_qd: wp.array[float],
):
    i = wp.tid()
    joint_index = gripper_joint_start + i
    joint_qd[joint_index] = (gripper_target_q[i] - current_q[joint_index]) * inv_dt


@wp.kernel
def drive_grasped_particles(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    gripper_offset: wp.vec3,
    grasp_count: wp.array(dtype=wp.int32),
    grasp_indices: wp.array(dtype=wp.int32),
    grasp_local_offsets: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= grasp_count[0]:
        return

    particle_index = grasp_indices[i]
    gripper_tf = body_q[body_index]
    gripper_pos = wp.transform_point(gripper_tf, gripper_offset)
    gripper_rot = wp.transform_get_rotation(gripper_tf)
    target_pos = gripper_pos + wp.quat_rotate(gripper_rot, grasp_local_offsets[i])
    particle_q[particle_index] = target_pos
    particle_qd[particle_index] = wp.vec3(0.0, 0.0, 0.0)


def scale_viewer_ui(viewer, font_scale):
    ui = getattr(viewer, "ui", None)
    io = getattr(ui, "io", None)
    if io is not None:
        if hasattr(io, "font_global_scale"):
            io.font_global_scale = font_scale
        elif getattr(ui, "imgui", None) is not None:
            style = ui.imgui.get_style()
            style.font_scale_main = font_scale
            if not getattr(ui, "_example_ui_sizes_scaled", False):
                style.scale_all_sizes(font_scale)
                ui._example_ui_sizes_scaled = True


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.teleop_linear_speed = 0.25
        self.teleop_pitch_speed = 1.0
        self.initial_left_ee_forward_offset = 0.35
        self.initial_left_ee_pitch = 0.5 * wp.pi
        self.gripper_open = (0.05, -0.05)
        self.gripper_closed = (0.0, 0.0)
        self.active_arm = "left"
        self.arm_switch_was_down = False
        self.gripper_is_closed = {"left": True, "right": True}
        self.gripper_toggle_was_down = False
        self.cloth_iterations = 10
        self.cloth_asset = getattr(args, "cloth_asset", "grid") if args is not None else "grid"
        self.grasp_mode = getattr(args, "grasp_mode", "constraint") if args is not None else "constraint"
        self.table_workspace_margin = 0.04
        self.gripper_table_clearance = 0.00
        self.max_grasp_particles = 48
        self.grasp_radius = 0.045
        self.grasp_half_width = 0.035
        self.grasp_depth = 0.060

        # Shape contact/friction parameters for robot/table surfaces.
        self.shape_contact_ke = 5.0e4
        self.shape_contact_kd = 1.0e-3
        self.shape_contact_mu = 1.5

        self.viewer = viewer

        table_hx = 0.42
        table_hy = 0.34
        table_hz = 0.04
        table_pos = wp.vec3(0.10, -0.70, 0.10)
        table_top_z = table_pos[2] + table_hz
        self.table_pos = table_pos
        self.table_hx = table_hx
        self.table_hy = table_hy
        self.table_top_z = float(table_top_z)
        robot_builder = newton.ModelBuilder()
        robot_builder.default_joint_cfg.armature = 0.01
        robot_builder.default_joint_cfg.target_ke = 0.0
        robot_builder.default_joint_cfg.target_kd = 0.0
        robot_builder.default_shape_cfg.ke = self.shape_contact_ke
        robot_builder.default_shape_cfg.kd = self.shape_contact_kd
        robot_builder.default_shape_cfg.mu = self.shape_contact_mu

        robot_builder.add_urdf(
            ROBOT_URDF_PATH,
            xform=wp.transform(wp.vec3(-0.50, -0.35, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        robot_builder.joint_target_pos[:] = robot_builder.joint_q[:]

        scene = newton.ModelBuilder(gravity=-981.0)
        scene.add_world(robot_builder)

        scene.add_shape_box(
            body=-1,
            xform=wp.transform(table_pos, wp.quat_identity()),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
            color=wp.vec3(1.0, 1.0, 1.0),
            label="left_arm_table",
        )
        if self.cloth_asset == "shirt":
            self.cloth_body_contact_margin = 0.008
            self.cloth_soft_contact_ke = 1.0e4
            self.cloth_soft_contact_kd = 1.0e-2
            self.cloth_self_contact_mu = 0.25
            self.cloth_self_contact_radius = 0.002
            self.cloth_self_contact_margin = 0.002
            self._add_shirt_cloth(scene, table_pos, table_top_z)
        else:
            self.cloth_body_contact_margin = 0.0018
            self.cloth_soft_contact_ke = 1.0e4
            self.cloth_soft_contact_kd = 1.0e-2
            self.cloth_self_contact_mu = 0.25
            self.cloth_self_contact_radius = 0.00045
            self.cloth_self_contact_margin = 0.00045
            self._add_twist_cloth(scene, table_pos, table_top_z)
        scene.color(include_bending=True)

        self.model = scene.finalize()
        self.model.soft_contact_ke = self.cloth_soft_contact_ke
        self.model.soft_contact_kd = self.cloth_soft_contact_kd
        self.model.soft_contact_mu = self.cloth_self_contact_mu
        self._set_shape_contact_parameters()

        self.robot_solver = newton.solvers.SolverFeatherstone(
            self.model,
            update_mass_matrix_interval=self.sim_substeps,
        )
        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.cloth_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=self.cloth_self_contact_radius,
            particle_self_contact_margin=self.cloth_self_contact_margin,
            # particle_topological_contact_filter_threshold=1,
            # particle_rest_shape_contact_exclusion_radius=0.005,
            # particle_vertex_contact_buffer_size=16,
            # particle_edge_contact_buffer_size=20,
            # particle_collision_detection_interval=-1,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=self.cloth_body_contact_margin,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.zeros_like(self.state_0.joint_qd)
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)
        self.original_particle_flags = self.model.particle_flags.numpy().copy()
        self.grasps = {
            arm_name: {
                "count": wp.zeros(1, dtype=wp.int32),
                "indices": wp.zeros(self.max_grasp_particles, dtype=wp.int32),
                "local_offsets": wp.zeros(self.max_grasp_particles, dtype=wp.vec3),
                "active": False,
            }
            for arm_name in ("left", "right")
        }

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._setup_arm_ik()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.8, -1.8, 1.2),
            pitch=-25.0,
            yaw=135.0,
        )

        self.capture()

    def _set_shape_contact_parameters(self):
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.shape_contact_ke
        shape_kd[...] = self.shape_contact_kd
        shape_mu[...] = self.shape_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

    def _add_twist_cloth(self, scene, table_pos, table_top_z):


        stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        cloth_prim = stage.GetPrimAtPath("/root/cloth/cloth")
        cloth_mesh = newton.usd.get_mesh(cloth_prim)

        vertices = [wp.vec3(vertex) for vertex in cloth_mesh.vertices]
        scene.add_cloth_mesh(
            pos=wp.vec3(table_pos[0] -0.25, table_pos[1], table_top_z + 0.035),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi),
            scale=0.0022,
            vertices=vertices,
            indices=cloth_mesh.indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=5.0e4,
            tri_ka=5.0e4,
            tri_kd=1.5e-6,
            edge_ke=5.0,
            edge_kd=1.0e-2,
            particle_radius=0.0018,
            label="left_arm_twist_cloth",
        )

    def _add_shirt_cloth(self, scene, table_pos, table_top_z):

        stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_prim = stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(shirt_prim)

        vertices = [wp.vec3(vertex) for vertex in shirt_mesh.vertices]

        scene.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            pos=wp.vec3(table_pos[0]-0.1, table_pos[1]+0.9, table_top_z + 0.20),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=0.0075,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.5e-6,
            edge_ke=5.0,
            edge_kd=1.0e-2,
            particle_radius=0.008,
        )

    def _setup_arm_ik(self):
        self.arm_configs = {
            "left": {
                "ee_index": 5,
                "ee_offset": wp.vec3(0.0, 0.0, 0.13503),
                "arm_slice": slice(0, 6),
                "gripper_slice": slice(6, 8),
                "gripper_joint_start": 6,
            },
            "right": {
                "ee_index": 13,
                "ee_offset": wp.vec3(0.0, 0.0, 0.13503),
                "arm_slice": slice(8, 14),
                "gripper_slice": slice(14, 16),
                "gripper_joint_start": 14,
            },
        }
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        body_q_np = self.state_0.body_q.numpy()
        self.arm_ik = {}
        self.ik_iters = 12
        for arm_name, config in self.arm_configs.items():
            ee_tf = wp.transform(*body_q_np[config["ee_index"]])
            ee_pos = wp.transform_point(ee_tf, config["ee_offset"])
            ee_rot = wp.transform_get_rotation(ee_tf)
            pos_obj = ik.IKObjectivePosition(
                link_index=config["ee_index"],
                link_offset=config["ee_offset"],
                target_positions=wp.array([ee_pos], dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=config["ee_index"],
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([self._quat_to_vec4(ee_rot)], dtype=wp.vec4),
            )
            ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
            solver = ik.IKSolver(
                model=self.model,
                n_problems=1,
                objectives=[pos_obj, rot_obj, self.joint_limits_obj],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )
            self.arm_ik[arm_name] = {
                "pos_obj": pos_obj,
                "rot_obj": rot_obj,
                "joint_q": ik_joint_q,
                "solver": solver,
            }
        self.gripper_open_wp = wp.array(self.gripper_open, dtype=wp.float32)
        self.gripper_closed_wp = wp.array(self.gripper_closed, dtype=wp.float32)
        self.gripper_joint_target_wp = wp.array(self.gripper_closed, dtype=wp.float32)
        self._set_initial_left_arm_pose()

    def _set_initial_left_arm_pose(self):
        config = self.arm_configs["left"]
        arm_ik = self.arm_ik["left"]
        body_q_np = self.state_0.body_q.numpy()
        left_ee_tf = wp.transform(*body_q_np[config["ee_index"]])
        initial_pos = wp.transform_point(left_ee_tf, config["ee_offset"])
        initial_rot = wp.transform_get_rotation(left_ee_tf)
        target_pos = wp.vec3(
            initial_pos[0] + self.initial_left_ee_forward_offset,
            initial_pos[1],
            initial_pos[2],
        )
        target_rot = wp.normalize(
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.initial_left_ee_pitch) * initial_rot
        )

        arm_ik["pos_obj"].set_target_position(0, target_pos)
        arm_ik["rot_obj"].set_target_rotation(0, self._quat_to_vec4(target_rot))
        arm_ik["solver"].step(arm_ik["joint_q"], arm_ik["joint_q"], iterations=64)

        joint_q = arm_ik["joint_q"].flatten()
        wp.copy(self.model.joint_q, joint_q)
        wp.copy(self.state_0.joint_q, joint_q)
        wp.copy(self.state_1.joint_q, joint_q)
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()
        self.target_joint_qd.zero_()
        wp.copy(self.control.joint_target_pos, joint_q)
        self._set_gripper_targets("left", self.gripper_closed)
        self._set_gripper_targets("right", self.gripper_closed)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    @staticmethod
    def _quat_to_vec4(q):
        return wp.vec4(q[0], q[1], q[2], q[3])

    def _clamp_teleop_target_above_table(self, target_pos):
        over_table_x = abs(float(target_pos[0]) - float(self.table_pos[0])) <= self.table_hx + self.table_workspace_margin
        over_table_y = abs(float(target_pos[1]) - float(self.table_pos[1])) <= self.table_hy + self.table_workspace_margin
        if not (over_table_x and over_table_y):
            return target_pos

        min_z = self.table_top_z + self.gripper_table_clearance
        if target_pos[2] >= min_z:
            return target_pos
        return wp.vec3(target_pos[0], target_pos[1], min_z)

    def _key_axis(self, positive_key, negative_key):
        if not hasattr(self.viewer, "is_key_down"):
            return 0.0
        return float(self.viewer.is_key_down(positive_key)) - float(self.viewer.is_key_down(negative_key))

    def _active_config(self):
        return self.arm_configs[self.active_arm]

    def _update_active_arm_switch(self):
        is_down = bool(hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down("t"))
        if is_down and not self.arm_switch_was_down:
            self.active_arm = "right" if self.active_arm == "left" else "left"
        self.arm_switch_was_down = is_down

    def _set_gripper_targets(self, arm_name, targets):
        config = self.arm_configs[arm_name]
        target_array = self.gripper_closed_wp if targets == self.gripper_closed else self.gripper_open_wp
        wp.copy(dest=self.gripper_joint_target_wp, src=target_array)
        wp.copy(dest=self.control.joint_target_pos[config["gripper_slice"]], src=target_array)

    def _current_gripper_pose(self, arm_name=None):
        config = self.arm_configs[arm_name or self.active_arm]
        body_q_np = self.state_0.body_q.numpy()
        gripper_tf = wp.transform(*body_q_np[config["ee_index"]])
        gripper_pos = wp.transform_point(gripper_tf, config["ee_offset"])
        gripper_rot = wp.transform_get_rotation(gripper_tf)
        return gripper_pos, gripper_rot

    def _refresh_particle_flags_from_grasps(self):
        flags = self.original_particle_flags.copy()
        for grasp in self.grasps.values():
            if not grasp["active"]:
                continue
            count = int(grasp["count"].numpy()[0])
            for particle_index in grasp["indices"].numpy()[:count]:
                flags[int(particle_index)] = flags[int(particle_index)] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags, dtype=self.model.particle_flags.dtype, device=self.model.particle_flags.device)

    def _attach_nearby_cloth_particles(self):
        if self.model.particle_count == 0:
            return

        arm_name = self.active_arm
        self._release_grasped_cloth_particles(arm_name)
        already_grasped = set()
        for other_arm, grasp in self.grasps.items():
            if other_arm == arm_name or not grasp["active"]:
                continue
            count = int(grasp["count"].numpy()[0])
            already_grasped.update(int(index) for index in grasp["indices"].numpy()[:count])

        gripper_pos, gripper_rot = self._current_gripper_pose(arm_name)
        particle_q = self.state_0.particle_q.numpy()
        candidates = []
        for particle_index, particle_pos in enumerate(particle_q):
            if particle_index in already_grasped:
                continue
            delta = wp.vec3(
                float(particle_pos[0]) - gripper_pos[0],
                float(particle_pos[1]) - gripper_pos[1],
                float(particle_pos[2]) - gripper_pos[2],
            )
            local = wp.quat_rotate_inv(gripper_rot, delta)
            dist_sq = float(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
            inside_box = (
                abs(float(local[0])) <= self.grasp_half_width
                and abs(float(local[1])) <= self.grasp_half_width
                and abs(float(local[2])) <= self.grasp_depth
            )
            if inside_box or dist_sq <= self.grasp_radius * self.grasp_radius:
                candidates.append((dist_sq, particle_index, local))

        if not candidates:
            return

        candidates.sort(key=lambda item: item[0])
        selected = candidates[: self.max_grasp_particles]
        selected_indices = [particle_index for _, particle_index, _ in selected]
        selected_offsets = [local for _, _, local in selected]

        self.grasps[arm_name] = {
            "count": wp.array([len(selected_indices)], dtype=wp.int32),
            "indices": wp.array(selected_indices, dtype=wp.int32),
            "local_offsets": wp.array(selected_offsets, dtype=wp.vec3),
            "active": True,
        }
        self._refresh_particle_flags_from_grasps()
        self._drive_grasped_particles(self.state_0.body_q)

    def _release_grasped_cloth_particles(self, arm_name=None):
        arm_name = arm_name or self.active_arm
        if not self.grasps[arm_name]["active"]:
            return
        self.grasps[arm_name] = {
            "count": wp.zeros(1, dtype=wp.int32),
            "indices": wp.zeros(self.max_grasp_particles, dtype=wp.int32),
            "local_offsets": wp.zeros(self.max_grasp_particles, dtype=wp.vec3),
            "active": False,
        }
        self._refresh_particle_flags_from_grasps()

    def _drive_grasped_particles(self, body_q):
        for arm_name, grasp in self.grasps.items():
            if not grasp["active"]:
                continue
            config = self.arm_configs[arm_name]
            for particle_q, particle_qd in (
                (self.state_0.particle_q, self.state_0.particle_qd),
                (self.state_1.particle_q, self.state_1.particle_qd),
            ):
                wp.launch(
                    drive_grasped_particles,
                    dim=self.max_grasp_particles,
                    inputs=[
                        body_q,
                        config["ee_index"],
                        config["ee_offset"],
                        grasp["count"],
                        grasp["indices"],
                        grasp["local_offsets"],
                    ],
                    outputs=[particle_q, particle_qd],
                )

    def _update_gripper_teleop(self):
        is_down = bool(hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down("g"))
        if is_down and not self.gripper_toggle_was_down:
            self.gripper_is_closed[self.active_arm] = not self.gripper_is_closed[self.active_arm]
            if self.gripper_is_closed[self.active_arm] and self.grasp_mode == "constraint":
                self._attach_nearby_cloth_particles()
            else:
                self._release_grasped_cloth_particles()
        self.gripper_toggle_was_down = is_down

        if self.gripper_is_closed[self.active_arm]:
            self._set_gripper_targets(self.active_arm, self.gripper_closed)
        else:
            self._set_gripper_targets(self.active_arm, self.gripper_open)

        wp.launch(
            compute_gripper_qd,
            dim=2,
            inputs=[
                self.state_0.joint_q,
                self.gripper_joint_target_wp,
                self._active_config()["gripper_joint_start"],
                1.0 / self.frame_dt,
            ],
            outputs=[self.target_joint_qd],
        )

    def _update_arm_teleop(self):
        dx = self._key_axis("i", "k")
        dy = self._key_axis("j", "l")
        dz = self._key_axis("u", "o")
        dpitch = self._key_axis("r", "f")
        if dx == 0.0 and dy == 0.0 and dz == 0.0 and dpitch == 0.0:
            return

        config = self._active_config()
        arm_ik = self.arm_ik[self.active_arm]
        body_q_np = self.state_0.body_q.numpy()
        current_ee_tf = wp.transform(*body_q_np[config["ee_index"]])
        current_ee_pos = wp.transform_point(current_ee_tf, config["ee_offset"])
        current_ee_rot = wp.transform_get_rotation(current_ee_tf)

        step = self.teleop_linear_speed * self.frame_dt
        target_ee_pos = wp.vec3(
            current_ee_pos[0] + dx * step,
            current_ee_pos[1] + dy * step,
            current_ee_pos[2] + dz * step,
        )
        target_ee_pos = self._clamp_teleop_target_above_table(target_ee_pos)
        pitch_step = self.teleop_pitch_speed * self.frame_dt
        pitch_delta = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), dpitch * pitch_step)
        target_ee_rot = wp.normalize(current_ee_rot * pitch_delta)

        arm_ik["pos_obj"].set_target_position(0, target_ee_pos)
        arm_ik["rot_obj"].set_target_rotation(0, self._quat_to_vec4(target_ee_rot))

        wp.copy(dest=arm_ik["joint_q"].flatten(), src=self.state_0.joint_q)
        arm_ik["solver"].step(arm_ik["joint_q"], arm_ik["joint_q"], iterations=self.ik_iters)
        wp.launch(
            compute_delta_joint_qd,
            dim=self.model.joint_dof_count,
            inputs=[self.state_0.joint_q, arm_ik["joint_q"].flatten(), 1.0 / self.frame_dt],
            outputs=[self.target_joint_qd],
        )
        inactive_arm = "right" if self.active_arm == "left" else "left"
        self.target_joint_qd[self.arm_configs[inactive_arm]["arm_slice"]].zero_()
        self.target_joint_qd[self.arm_configs[inactive_arm]["gripper_slice"]].zero_()
        self.target_joint_qd[config["gripper_slice"]].zero_()

    def capture(self):
        self.graph = None

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self._drive_grasped_particles(self.state_1.body_q)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.target_joint_qd.zero_()
        self._update_active_arm_switch()
        self._update_arm_teleop()
        self._update_gripper_teleop()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--cloth-asset",
            choices=("grid", "shirt"),
            default="grid",
            help="Cloth asset to place on the table.",
        )
        parser.add_argument(
            "--grasp-mode",
            choices=("constraint", "normal"),
            default="constraint",
            help="Use explicit cloth particle attachment on gripper close, or normal gripper actuation only.",
        )
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    scale_viewer_ui(viewer, 2.0)
    newton.examples.run(Example(viewer, args), args)
