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


class ToyIsaacSimClothGraspTest(unittest.TestCase):
    """Validate the default toy cloth grasp behavior end to end."""

    @classmethod
    def setUpClass(cls):
        if not ISAAC_PYTHON.exists():
            raise unittest.SkipTest(f"Isaac Sim Python not found: {ISAAC_PYTHON}")

        cls._tmpdir = tempfile.TemporaryDirectory()
        output_root = Path(cls._tmpdir.name)
        cls.result_json = output_root / "toy_cloth_grasp_result.json"
        command = [
            str(ISAAC_PYTHON),
            str(TOY_SCRIPT),
            "--headless",
            "--no-record",
            "--steps",
            "340",
            "--output-root",
            str(output_root),
            "--result-json",
            str(cls.result_json),
        ]
        env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        cls.stdout = completed.stdout
        cls.stderr = completed.stderr
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-80:])
            raise AssertionError(
                f"toy cloth grasp run failed with code {completed.returncode}\n{tail}"
            )
        with cls.result_json.open(encoding="utf-8") as result_file:
            cls.result = json.load(result_file)

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

    def test_default_run_does_not_print_sticking_diagnostics(self):
        combined_output = self.stdout + self.stderr
        self.assertNotIn("[toy_isaacsim_cloth_grasp] diagnostic ", combined_output)
        self.assertNotIn('"diagnostics": [', combined_output)

    def test_finger_cloth_tabletop_sticking_happens(self):
        self.assertLessEqual(self.result["attached_step"], 30)
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
        self.assertGreaterEqual(sum(patch["particles"] for patch in bottom_patches), 60)

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
        self.assertGreaterEqual(largest_inner["particles"], 50)
        self.assertGreater(
            largest_inner["span_m"][2],
            0.020,
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
            self.assertGreaterEqual(largest_inner["particles"], 45)
            self.assertGreater(largest_inner["span_m"][2], 0.020)
            self.assertGreaterEqual(attached_particles, 0.90 * pressed_particles)
            self.assertGreaterEqual(active_pressed, 0.95 * attached_particles)

    def test_sticked_patch_shift_is_minimal_during_lift(self):
        start_center = self._largest_anchor_local_center(self._snapshot(150))
        end_center = self._largest_anchor_local_center(self._snapshot(300))
        shift = sum(
            (end_value - start_value) ** 2
            for start_value, end_value in zip(start_center, end_center)
        ) ** 0.5
        self.assertLess(shift, 0.006)

    def _largest_anchor_local_center(self, snapshot: dict) -> list[float]:
        inner_contacts = self._contact_diagnostics(
            snapshot,
            "left_inner_face+right_inner_face",
        )
        largest_contact = max(inner_contacts, key=lambda contact: contact["particles"])
        anchor_surface = next(
            surface
            for surface in largest_contact["surfaces"]
            if surface["name"] == largest_contact["anchor_name"]
        )
        return [
            0.5 * (minimum + maximum)
            for minimum, maximum in zip(
                anchor_surface["local_min_m"],
                anchor_surface["local_max_m"],
            )
        ]

    def test_whole_cloth_lifts(self):
        self.assertTrue(self.result["success"])
        self.assertGreater(self.result["attached_lift_m"], 0.15)
        self.assertGreater(self.result["max_cloth_lift_m"], 0.070)
        self.assertGreater(self.result["whole_cloth_follow_ratio"], 0.30)
        self.assertIsNotNone(self.result["released_step"])


if __name__ == "__main__":
    unittest.main()
