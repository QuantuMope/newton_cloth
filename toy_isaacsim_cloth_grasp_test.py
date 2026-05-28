"""Integration tests for the toy Isaac Sim cloth grasp scene.

The test runs the standalone toy script once in headless mode and validates the
result JSON. Keeping Isaac Sim in a subprocess avoids importing Kit into the
unittest runner and makes failures easier to inspect from the saved result file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
ISAAC_PYTHON = Path("/home/horizon/isaacsim_env/bin/python")
TOY_SCRIPT = ROOT_DIR / "toy_isaacsim_cloth_grasp.py"
RUN_TIMEOUT_SECONDS = 180
DEFAULT_STEPS = 390
SLIDE_STEPS = 420
GRAVITY_FOLD_STEPS = 180
RECORD_SYNC_STEPS = 45
RECORD_SYNC_SETTLED_STEP = RECORD_SYNC_STEPS - 1
DIAGNOSTIC_INTERVAL = 1
OLD_CLOTH_REST_OFFSET = 0.008
OLD_CLOTH_CONTACT_OFFSET = 0.009


class ToyIsaacSimClothGraspTest(unittest.TestCase):
    """Validate the default toy cloth grasp behavior end to end."""

    @classmethod
    def setUpClass(cls):
        if not ISAAC_PYTHON.exists():
            raise unittest.SkipTest(f"Isaac Sim Python not found: {ISAAC_PYTHON}")

        cls._tmpdir = tempfile.TemporaryDirectory()
        output_root = Path(cls._tmpdir.name)
        cls.result_json = output_root / "toy_cloth_grasp_result.json"
        cls.result, cls.stdout, cls.stderr = cls._run_toy(
            output_root,
            cls.result_json,
        )
        slide_output_root = output_root / "table_slide"
        cls.slide_result, cls.slide_stdout, cls.slide_stderr = cls._run_toy(
            slide_output_root,
            slide_output_root / "result.json",
            [
                "--demo-mode",
                "table-slide",
                "--steps",
                str(SLIDE_STEPS),
            ],
        )
        gravity_fold_output_root = output_root / "gravity_fold"
        cls.gravity_fold_result, cls.gravity_fold_stdout, cls.gravity_fold_stderr = (
            cls._run_toy(
                gravity_fold_output_root,
                gravity_fold_output_root / "result.json",
                [
                    "--demo-mode",
                    "gravity-fold",
                    "--steps",
                    str(GRAVITY_FOLD_STEPS),
                ],
            )
        )
        cls.record_sync_output_root = output_root / "record_sync"
        (
            cls.record_sync_result,
            cls.record_sync_stdout,
            cls.record_sync_stderr,
        ) = cls._run_toy(
            cls.record_sync_output_root,
            cls.record_sync_output_root / "result.json",
            [
                "--demo-mode",
                "gravity-fold",
                "--steps",
                str(RECORD_SYNC_STEPS),
                "--width",
                "320",
                "--height",
                "240",
            ],
            record=True,
        )

    @classmethod
    def _run_toy(
        cls,
        output_root: Path,
        result_json: Path,
        extra_args: list[str] | None = None,
        record: bool = False,
    ) -> tuple[dict, str, str]:
        command = [
            str(ISAAC_PYTHON),
            str(TOY_SCRIPT),
            "--headless",
            "--record" if record else "--no-record",
            "--steps",
            str(DEFAULT_STEPS),
            "--diagnostic-interval",
            str(DIAGNOSTIC_INTERVAL),
            "--output-root",
            str(output_root),
            "--result-json",
            str(result_json),
        ]
        if extra_args is not None:
            command.extend(extra_args)
        env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-80:])
            raise AssertionError(
                f"toy cloth grasp run failed with code {completed.returncode}\n{tail}"
            )
        with result_json.open(encoding="utf-8") as result_file:
            return json.load(result_file), completed.stdout, completed.stderr

    @classmethod
    def tearDownClass(cls):
        tmpdir = getattr(cls, "_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()

    def _snapshot(self, step: int) -> dict:
        for snapshot in self.result["diagnostics"]:
            if snapshot["step"] == step:
                return snapshot
        self.fail(f"missing diagnostic snapshot for step {step}")

    @staticmethod
    def _patches(snapshot: dict, mode: str | None = None) -> list[dict]:
        patches = snapshot.get("active_patch_summaries", [])
        if mode is None:
            return patches
        return [patch for patch in patches if patch["mode"] == mode]

    @staticmethod
    def _contact_diagnostics(snapshot: dict, mode: str | None = None) -> list[dict]:
        diagnostics = snapshot.get("active_patch_contact_diagnostics", [])
        if mode is None:
            return diagnostics
        return [diagnostic for diagnostic in diagnostics if diagnostic["mode"] == mode]

    @staticmethod
    def _snapshots_with_inner_attachment(result: dict) -> list[dict]:
        return [
            snapshot
            for snapshot in result["diagnostics"]
            if snapshot["inner_attached_particles"] > 0
            and snapshot["inner_boundary_free_particles"] > 0
            and snapshot["inner_attached_centroid_z"] is not None
            and snapshot["inner_boundary_free_centroid_z"] is not None
        ]

    @staticmethod
    def _inner_inside_contact_count(snapshot: dict) -> int:
        named_contacts = snapshot["named_surface_contacts"]
        return sum(
            int(named_contacts[name].get("contact_inside_count") or 0)
            for name in ("left_inner_face", "right_inner_face")
        )

    @staticmethod
    def _first_open_snapshot(result: dict) -> dict:
        release_step = result["released_step"]
        return next(
            snapshot
            for snapshot in result["diagnostics"]
            if snapshot["step"] >= release_step and snapshot["finger_y_offset"] > 0.010
        )

    @staticmethod
    def _open_snapshots_after_release(result: dict) -> list[dict]:
        release_step = result["released_step"]
        return [
            snapshot
            for snapshot in result["diagnostics"]
            if snapshot["step"] >= release_step and snapshot["finger_y_offset"] >= 0.027
        ]

    @staticmethod
    def _fully_open_snapshots_after_release(result: dict) -> list[dict]:
        release_step = result["released_step"]
        return [
            snapshot
            for snapshot in result["diagnostics"]
            if snapshot["step"] >= release_step and snapshot["finger_y_offset"] >= 0.069
        ]

    @staticmethod
    def _recorded_rgb_frames(output_root: Path) -> list[Path]:
        frame_dir = output_root / "frames"
        rgb_frames = sorted(frame_dir.glob("rgb_*.png"))
        if not rgb_frames:
            rgb_frames = sorted(frame_dir.glob("**/rgb_*.png"))
        return rgb_frames

    @staticmethod
    def _blue_cloth_image_stats(image_path: Path) -> dict:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        pixels = image.load()
        xs = []
        ys = []
        for y in range(height):
            for x in range(width):
                red, green, blue = pixels[x, y]
                if blue > 110 and blue > 1.4 * red and blue > 1.08 * green:
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return {"count": 0, "centroid_x": None, "centroid_y": None}
        return {
            "count": len(xs),
            "centroid_x": sum(xs) / len(xs),
            "centroid_y": sum(ys) / len(ys),
        }

    @staticmethod
    def _mean_abs_frame_difference(first_path: Path, second_path: Path) -> float:
        from PIL import Image, ImageChops, ImageStat

        first = Image.open(first_path).convert("RGB")
        second = Image.open(second_path).convert("RGB")
        difference = ImageChops.difference(first, second)
        channel_means = ImageStat.Stat(difference).mean
        return sum(channel_means) / len(channel_means)

    def test_default_run_does_not_print_sticking_diagnostics(self):
        combined_output = self.stdout + self.stderr
        self.assertNotIn("[toy_isaacsim_cloth_grasp] diagnostic ", combined_output)
        self.assertNotIn('"diagnostics": [', combined_output)

    def test_default_physical_resolution_and_finger_size(self):
        self.assertEqual([0.026, 0.003, 0.076], self.result["finger_size_m"])
        self.assertEqual([0.002, 0.002], self.result["cloth_particle_spacing_m"])
        self.assertEqual(8, self.result["adhesive_min_component_particles"])
        self.assertEqual(128, self.result["adhesive_patch_max_particles"])

    def test_finger_cloth_tabletop_sticking_happens(self):
        snapshot = self._snapshot(30)
        bottom_patches = [
            patch
            for patch in snapshot["active_patch_summaries"]
            if patch["mode"] in (
                "left_bottom_face+table_top",
                "right_bottom_face+table_top",
            )
        ]
        modes = {patch["mode"] for patch in bottom_patches}
        self.assertEqual(
            {"left_bottom_face+table_top", "right_bottom_face+table_top"},
            modes,
        )
        self.assertTrue(
            all(
                patch["particles"] >= self.result["adhesive_min_component_particles"]
                for patch in bottom_patches
            )
        )

    def test_cloth_creases_when_fingers_close(self):
        close_window_patches = [
            patch
            for step in (120, 135, 150)
            for patch in self._patches(
                self._snapshot(step),
                "left_inner_face+right_inner_face",
            )
        ]
        self.assertTrue(close_window_patches, "no inner-finger patch during close")
        largest_inner = max(close_window_patches, key=lambda patch: patch["particles"])
        self.assertGreaterEqual(largest_inner["particles"], 20)
        self.assertGreater(
            largest_inner["span_m"][2],
            0.009,
            "folded cloth strip should have visible vertical span",
        )

    def test_all_folded_layers_are_sticked_between_fingers(self):
        for step in (180, 240, 300):
            snapshot = self._snapshot(step)
            inner_patches = self._patches(snapshot, "left_inner_face+right_inner_face")
            inner_contacts = self._contact_diagnostics(
                snapshot,
                "left_inner_face+right_inner_face",
            )
            attached_particles = sum(patch["particles"] for patch in inner_patches)
            pressed_particles = int(snapshot["pressed_count"])
            active_pressed = sum(contact["pressed_count"] for contact in inner_contacts)
            largest_inner = max(inner_patches, key=lambda patch: patch["particles"])
            self.assertGreaterEqual(
                largest_inner["particles"],
                self.result["adhesive_min_component_particles"],
            )
            self.assertGreater(attached_particles, 25)
            self.assertGreaterEqual(attached_particles, 0.85 * pressed_particles)
            self.assertGreaterEqual(active_pressed, 0.95 * attached_particles)

    def test_sticked_patch_shift_is_minimal_during_lift(self):
        start_contact = self._largest_inner_contact(self._snapshot(150))
        start_center = self._anchor_local_center(start_contact)
        matching_end_centers = [
            self._anchor_local_center(contact)
            for contact in self._contact_diagnostics(
                self._snapshot(300),
                "left_inner_face+right_inner_face",
            )
            if contact["anchor_name"] == start_contact["anchor_name"]
        ]
        self.assertTrue(matching_end_centers)
        end_center = min(
            matching_end_centers,
            key=lambda center: self._distance_m(start_center, center),
        )
        shift = self._distance_m(start_center, end_center)
        self.assertLess(shift, 0.003)

    @staticmethod
    def _distance_m(start: list[float], end: list[float]) -> float:
        return sum(
            (end_value - start_value) ** 2
            for start_value, end_value in zip(start, end)
        ) ** 0.5

    def _largest_inner_contact(self, snapshot: dict) -> dict:
        inner_contacts = self._contact_diagnostics(
            snapshot,
            "left_inner_face+right_inner_face",
        )
        return max(inner_contacts, key=lambda contact: contact["particles"])

    @staticmethod
    def _anchor_local_center(contact: dict) -> list[float]:
        anchor_surface = next(
            surface
            for surface in contact["surfaces"]
            if surface["name"] == contact["anchor_name"]
        )
        return [
            0.5 * (minimum + maximum)
            for minimum, maximum in zip(
                anchor_surface["local_min_m"],
                anchor_surface["local_max_m"],
            )
        ]

    def test_whole_cloth_lifts(self):
        self.assertGreater(self.result["attached_lift_m"], 0.15)
        self.assertGreater(self.result["max_cloth_lift_m"], 0.030)
        self.assertGreater(self.result["success_cloth_lift_m"], 0.030)
        self.assertLess(self.result["max_cloth_span_z_m"], 0.5)
        self.assertIsNotNone(self.result["released_step"])

    def test_no_one_or_two_particle_sticking_components(self):
        min_particles = self.result["adhesive_min_component_particles"]
        self.assertGreaterEqual(min_particles, 3)
        active_patches = [
            patch
            for snapshot in self.result["diagnostics"]
            for patch in snapshot["active_patch_summaries"]
        ]
        self.assertTrue(active_patches, "no active adhesive patches were diagnosed")
        for patch in active_patches:
            self.assertGreaterEqual(
                patch["particles"],
                min_particles,
                f"{patch['mode']} at diagnostic patch size {patch['particles']}",
            )

        contact_diagnostics = [
            diagnostic
            for snapshot in self.result["diagnostics"]
            for diagnostic in snapshot["active_patch_contact_diagnostics"]
        ]
        for diagnostic in contact_diagnostics:
            self.assertGreaterEqual(
                diagnostic["particles"],
                min_particles,
                f"{diagnostic['mode']} contact diagnostic below minimum size",
            )

    def test_cloth_releases_quickly_after_gripper_open(self):
        release_step = self.result["released_step"]
        self.assertIsNotNone(release_step)
        first_open_snapshot = self._first_open_snapshot(self.result)
        self.assertLessEqual(first_open_snapshot["step"] - release_step, 5)
        self.assertFalse(first_open_snapshot["active_patch_summaries"])

        later_open_snapshots = [
            snapshot
            for snapshot in self.result["diagnostics"]
            if snapshot["step"] >= first_open_snapshot["step"]
        ]
        self.assertTrue(later_open_snapshots)
        for snapshot in later_open_snapshots:
            self.assertFalse(snapshot["active_patch_summaries"])

    def test_reduced_contact_shell_releases_after_opening(self):
        output_root = Path(self._tmpdir.name) / "old_contact_shell"
        old_result, _stdout, _stderr = self._run_toy(
            output_root,
            output_root / "result.json",
            [
                "--cloth-rest-offset",
                str(OLD_CLOTH_REST_OFFSET),
                "--cloth-contact-offset",
                str(OLD_CLOTH_CONTACT_OFFSET),
            ],
        )

        min_component_particles = self.result["adhesive_min_component_particles"]
        old_open_snapshots = self._open_snapshots_after_release(old_result)
        current_open_snapshots = self._fully_open_snapshots_after_release(self.result)
        self.assertTrue(old_open_snapshots)
        self.assertTrue(current_open_snapshots)
        self.assertEqual(OLD_CLOTH_REST_OFFSET, old_result["cloth_rest_offset"])
        self.assertEqual(OLD_CLOTH_CONTACT_OFFSET, old_result["cloth_contact_offset"])
        self.assertEqual(0.003, self.result["cloth_rest_offset"])
        self.assertEqual(0.003, self.result["cloth_contact_offset"])

        old_lingering_contacts = [
            self._inner_inside_contact_count(snapshot)
            for snapshot in old_open_snapshots
        ]
        current_lingering_contacts = [
            self._inner_inside_contact_count(snapshot)
            for snapshot in current_open_snapshots
        ]
        self.assertGreaterEqual(max(old_lingering_contacts), min_component_particles)
        self.assertLess(
            max(current_lingering_contacts),
            min_component_particles,
        )
        for snapshot in current_open_snapshots:
            self.assertFalse(snapshot["active_patch_summaries"])
            self.assertEqual(0, int(snapshot["pressed_count"]))

    def test_free_cloth_near_attached_patches_follows_lift(self):
        inner_snapshots = self._snapshots_with_inner_attachment(self.result)
        self.assertGreaterEqual(len(inner_snapshots), 4)

        baseline = inner_snapshots[0]
        baseline_attached_z = baseline["inner_attached_centroid_z"]
        baseline_boundary_z = baseline["inner_boundary_free_centroid_z"]
        lift_snapshots = [
            snapshot
            for snapshot in inner_snapshots
            if snapshot["inner_attached_centroid_z"] - baseline_attached_z > 0.040
        ]
        self.assertTrue(lift_snapshots, "no meaningful inner-patch lift diagnosed")
        for snapshot in lift_snapshots:
            attached_lift = snapshot["inner_attached_centroid_z"] - baseline_attached_z
            boundary_lift = (
                snapshot["inner_boundary_free_centroid_z"] - baseline_boundary_z
            )
            self.assertGreaterEqual(boundary_lift + 0.002, 0.80 * attached_lift)
            self.assertLessEqual(
                abs(
                    snapshot["inner_attached_centroid_z"]
                    - snapshot["inner_boundary_free_centroid_z"]
                ),
                0.016,
            )

    def test_table_slide_moves_cloth_sideways_on_table(self):
        self.assertTrue(self.slide_result["success"])
        self.assertEqual("table-slide", self.slide_result["demo_mode"])
        self.assertEqual("teleport", self.slide_result["sticking_drive_mode"])
        self.assertEqual(0.160, self.slide_result["slide_distance"])
        self.assertEqual(240, self.slide_result["slide_steps"])
        self.assertGreater(self.slide_result["cloth_slide_m"], 0.050)
        self.assertGreater(self.slide_result["front_cloth_slide_m"], 0.050)
        self.assertLess(abs(self.slide_result["cloth_lateral_drift_m"]), 0.012)
        self.assertLess(abs(self.slide_result["cloth_lift_m"]), 0.035)

    def test_table_slide_folds_cloth_without_gripper_grasp(self):
        patch_modes = {
            patch["mode"]
            for snapshot in self.slide_result["diagnostics"]
            for patch in snapshot["active_patch_summaries"]
        }
        self.assertTrue(patch_modes)
        self.assertLessEqual(
            patch_modes,
            {"left_bottom_face+table_top", "right_bottom_face+table_top"},
        )
        self.assertNotIn("left_inner_face+right_inner_face", patch_modes)
        self.assertLess(self.slide_result["max_cloth_span_z_m"], 0.100)
        self.assertLess(self.slide_result["cloth_span_z_growth_m"], 0.080)
        self.assertFalse(self.slide_result["diagnostics"][-1]["active_patch_summaries"])
        for snapshot in self.slide_result["diagnostics"]:
            self.assertFalse(
                self._patches(snapshot, "left_inner_face+right_inner_face")
            )

    def test_gravity_fold_stays_clean_without_gripper_grasp(self):
        self.assertTrue(self.gravity_fold_result["success"])
        self.assertEqual("gravity-fold", self.gravity_fold_result["demo_mode"])
        self.assertIsNone(self.gravity_fold_result["attached_step"])
        self.assertEqual(0, self.gravity_fold_result["attached_particles"])
        self.assertEqual(
            [0.002, 0.002],
            self.gravity_fold_result["cloth_particle_spacing_m"],
        )
        self.assertLess(self.gravity_fold_result["max_cloth_span_z_m"], 0.025)
        self.assertLess(self.gravity_fold_result["final_cloth_span_m"][2], 0.020)

    def test_recorded_video_frames_follow_physics_drop(self):
        result = self.record_sync_result
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["video_path"])
        self.assertTrue(Path(result["video_path"]).exists())

        diagnostics = result["diagnostics"]
        initial_z = diagnostics[0]["cloth_centroid_m"][2]
        settled_z = diagnostics[RECORD_SYNC_SETTLED_STEP]["cloth_centroid_m"][2]
        self.assertGreater(initial_z - settled_z, 0.020)

        frames = self._recorded_rgb_frames(self.record_sync_output_root)
        self.assertGreaterEqual(len(frames), RECORD_SYNC_STEPS)
        initial_frame = frames[0]
        settled_frame = frames[RECORD_SYNC_SETTLED_STEP]
        initial_stats = self._blue_cloth_image_stats(initial_frame)
        settled_stats = self._blue_cloth_image_stats(settled_frame)
        self.assertGreater(initial_stats["count"], 1000)
        self.assertGreater(settled_stats["count"], 1000)

        # The gravity-fold sync run parks the fingers, so this visual motion is
        # the rendered particle cloth following the same fall measured above.
        rendered_x_shift = (
            settled_stats["centroid_x"] - initial_stats["centroid_x"]
        )
        frame_difference = self._mean_abs_frame_difference(
            initial_frame,
            settled_frame,
        )
        self.assertGreater(rendered_x_shift, 8.0)
        self.assertGreater(frame_difference, 4.0)


if __name__ == "__main__":
    unittest.main()
