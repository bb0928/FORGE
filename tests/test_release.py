"""Portable checks for release launchers and dependency/resource closure."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "rl")]
from common import PROFILE, environment, run


class ReleaseTests(unittest.TestCase):
    def test_failed_preflight_does_not_create_a_run(self):
        from unittest.mock import Mock
        record = Mock()
        with patch("common.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["preflight"])):
            with self.assertRaises(subprocess.CalledProcessError):
                run(["training"], environment(), before_run=record)
        record.assert_not_called()

    def test_environment_does_not_resume_or_inherit_other_experiments(self):
        with patch.dict(os.environ, {"RESUME_CKPT": "old.pt", "SEED": "99",
                                    "FORGE_CURRICULUM_MAX": "1.0", "RANK": "7",
                                    "FORGE_POSE_GUARD_WEIGHT": "-9",
                                    "WRIST_TRACKING_TERMINATION": "1"}):
            env = environment()
        self.assertNotIn("RESUME_CKPT", env)
        self.assertNotIn("RANK", env)
        self.assertEqual(env["SEED"], "7")
        self.assertEqual(env["FORGE_CURRICULUM_MAX"], "0.6")
        self.assertEqual(env["FORGE_POSE_GUARD_WEIGHT"], "-0.25")
        self.assertEqual(env["WRIST_TRACKING_TERMINATION"], "0")

    def test_dry_run_works_outside_repo_and_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="forge release ") as temp:
            log = Path(temp) / "logs with spaces"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"),
                                     "--log-root", str(log), "--dry-run"],
                                    cwd=temp, text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertFalse(log.exists())
            self.assertIn("--nproc-per-node=4", plan["command"])
            self.assertEqual(plan["environment"]["NUM_ENVS"], "4096")
            self.assertEqual(plan["environment"]["FORGE_LOG_ROOT"], str(log.resolve()))

    def test_train_rejects_existing_run_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "taken"; log.mkdir(); (log / "checkpoint.pt").touch()
            for name in ["taken", "../escape"]:
                result = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"),
                                         "--log-root", temp, "--run-name", name],
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            self.assertTrue((log / "checkpoint.pt").exists())

    def test_play_checkpoint_argument_is_literal(self):
        with tempfile.TemporaryDirectory(prefix="forge play ") as temp:
            checkpoint = Path(temp) / "model ; literal.pt"; checkpoint.touch()
            result = subprocess.run([sys.executable, str(ROOT / "scripts/play.py"),
                                     "--checkpoint", str(checkpoint), "--headless", "--dry-run"],
                                    cwd=temp, text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertIn(str(checkpoint.resolve()), plan["command"])
            self.assertEqual(plan["environment"]["ENABLE_DOMAIN_RAND"], "0")

    def test_xml_references_are_packaged(self):
        path = ROOT / "assets/g1_xhand/xmls/g1_dex31_payload.xml"
        xml = ET.parse(path).getroot()
        meshes = xml.findall(".//mesh")
        self.assertEqual(len(meshes), 49)
        meshdir = xml.find("compiler").get("meshdir", "")
        for node in meshes:
            self.assertTrue((path.parent / meshdir / node.attrib["file"]).is_file())

    def test_milestone_hashes_and_release_manifest(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(),
                             item["release_sha256"], item["path"])
        for path in ["rewards.py", "rl/forge_runner.py", "assets/g1_xhand/xmls/g1_dex31_payload.xml"]:
            item = next(i for i in manifest["files"] if i["path"] == path)
            self.assertEqual(item["original_sha256"], item["release_sha256"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install CPU test dependencies from README")
class NeuralContractTests(unittest.TestCase):
    def test_actor_critic_forward_and_checkpoint_roundtrip(self):
        import torch
        from rl.base_actor_critic import BaseActorCritic
        from obs_spec import ACTOR_OBS_GROUPS, CRITIC_EXTRA_GROUPS
        groups = [d for _, d in ACTOR_OBS_GROUPS]
        kwargs = dict(actor_group_dims=groups, critic_group_dims=groups + [d for _, d in CRITIC_EXTRA_GROUPS],
                      num_actions=29, lower_dof_indices=list(range(12)) + [13, 14],
                      upper_dof_indices=[12] + list(range(15, 29)))
        torch.manual_seed(7)
        model = BaseActorCritic(**kwargs).eval()
        obs = torch.cat([torch.randn(2, 786), torch.ones(2, 66)], dim=-1)
        critic = torch.cat([torch.randn(2, 156), torch.ones(2, 17)], dim=-1)
        with torch.no_grad():
            action, prediction = model.act_inference(obs)
            value = model.evaluate(critic)
        self.assertEqual(tuple(action.shape), (2, 29))
        self.assertEqual(tuple(prediction.shape), (2, 15))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertTrue(torch.isfinite(action).all())
        restored = BaseActorCritic(**kwargs).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(action, restored.act_inference(obs)[0], rtol=0, atol=0)

    def test_symmetry_is_involution_and_preserves_masks(self):
        import torch
        from forge_symmetry_func import _flip_actor_block, _flip_critic_block
        from symmetry_v3 import flip_joint_vector
        for width, flip in [(852, _flip_actor_block), (173, _flip_critic_block), (29, flip_joint_vector)]:
            value = torch.randn(4, width)
            torch.testing.assert_close(flip(flip(value)), value, rtol=0, atol=0)
            if width in (852, 173):
                start = 786 if width == 852 else 156
                torch.testing.assert_close(flip(value)[:, start:], value[:, start:])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install MuJoCo for XML compilation")
class RobotTests(unittest.TestCase):
    def test_robot_compiles_with_29_joints_and_payload_bodies(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1_xhand/xmls/g1_dex31_payload.xml"))
        self.assertEqual((model.nq, model.nv, model.njnt), (36, 35, 30))
        self.assertEqual(model.nu, 0)  # mjlab adds the configured PD actuators.
        for side in ("left", "right"):
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                    side + "_hand_payload"), 0)


if __name__ == "__main__":
    unittest.main()
