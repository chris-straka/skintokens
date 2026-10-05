"""Fast tests (no model): GLB IO, humanoid naming, weight mapping, joints doc, CLI contract.

    .venv/bin/python -m unittest discover -s tests      (from the repo root, inside the SkinTokens venv)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fixture import HUMANOID, humanoid, write_glb  # noqa: E402
from skintokens import cli  # noqa: E402
from skintokens.glb import Glb  # noqa: E402
from skintokens.naming import name_humanoid  # noqa: E402

EXPECT = {
    "hips": "Hips", "spine": "Spine", "spine1": "Spine1", "chest": "Spine2", "neck": "Neck", "head": "Head",
    "l_shoulder": "LeftShoulder", "l_arm": "LeftArm", "l_forearm": "LeftForeArm", "l_hand": "LeftHand",
    "r_shoulder": "RightShoulder", "r_arm": "RightArm", "r_forearm": "RightForeArm", "r_hand": "RightHand",
    "l_thigh": "LeftUpLeg", "l_shin": "LeftLeg", "l_foot": "LeftFoot", "l_toe": "LeftToeBase",
    "r_thigh": "RightUpLeg", "r_shin": "RightLeg", "r_foot": "RightFoot", "r_toe": "RightToeBase",
}


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def glb(self, name, **kw):
        p = os.path.join(self.tmp, name)
        write_glb(p, humanoid(**{k: v for k, v in kw.items() if k in ("mirror_z", "names")}),
                  **{k: v for k, v in kw.items() if k in ("jitter", "weight_shift")})
        return p

    def test_skeleton_roundtrip(self):
        g = Glb.read(self.glb("a.glb"))
        names, parents, heads = g.skeleton()
        self.assertEqual(names, [n for n, _, _ in HUMANOID])
        self.assertEqual(parents, [p for _, p, _ in HUMANOID])
        np.testing.assert_allclose(heads[9], HUMANOID[9][2], atol=1e-6)
        out = os.path.join(self.tmp, "b.glb")
        g.write(out)
        self.assertEqual(Glb.read(out).skeleton()[0], names)

    def test_naming_facing_plus_z(self):
        g = Glb.read(self.glb("a.glb"))
        mapping, missing = name_humanoid(*g.skeleton())
        self.assertEqual(missing, [])
        self.assertEqual({k: v.removeprefix("mixamorig:") for k, v in mapping.items()}, EXPECT)

    def test_naming_facing_minus_z_keeps_character_left(self):
        g = Glb.read(self.glb("a.glb", mirror_z=True))
        mapping, missing = name_humanoid(*g.skeleton())
        self.assertEqual(missing, [])
        self.assertEqual(mapping["l_arm"], "mixamorig:LeftArm")
        self.assertEqual(mapping["r_thigh"], "mixamorig:RightUpLeg")

    def test_naming_rejects_non_humanoid(self):
        names, parents, heads = ["a", "b", "c"], [None, 0, 1], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2.0]])
        mapping, missing = name_humanoid(names, parents, heads)
        self.assertTrue(missing)

    def test_map_weights_keeps_skeleton_changes_only_weights(self):
        orig_p = self.glb("orig.glb")
        # "model output": same mesh, joints renamed bone_N, heads quantized, weights moved to the parent joint
        st_p = self.glb("st.glb", names=[f"bone_{i}" for i in range(len(HUMANOID))], jitter=0.5,
                        weight_shift=lambda i: HUMANOID[i][1] if HUMANOID[i][1] is not None else i)
        orig = Glb.read(orig_p)
        before = orig.skeleton()
        info = cli.map_weights(orig, Glb.read(st_p))
        self.assertLess(info["joint_match_max_m"], 0.05)
        out = os.path.join(self.tmp, "out.glb")
        orig.write(out)
        g = Glb.read(out)
        after = g.skeleton()
        self.assertEqual(after[0], before[0])
        np.testing.assert_allclose(after[2], before[2])
        _, J, W = g.skinned_vertices()
        self.assertEqual(int(J[36, 0]), HUMANOID[9][1])  # l_hand's quad now on l_forearm
        np.testing.assert_allclose(W.sum(1), 1.0, atol=1e-6)

    def test_joints_doc(self):
        doc = cli.joints_doc(Glb.read(self.glb("a.glb")), subject="t")
        self.assertEqual(doc["format"], "skintokens-joints/1")
        self.assertEqual(len(doc["joints"]), len(HUMANOID))
        n = np.array([j["normalized"] for j in doc["joints"]])
        self.assertLessEqual(np.abs(n).max(), 1.0 + 1e-6)

    def test_cli_contract_usage_errors_write_no_report(self):
        rep = os.path.join(self.tmp, "r.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "skintokens", "cli.py"), "rig",
                            os.path.join(self.tmp, "missing.glb"), os.path.join(self.tmp, "o.glb"), "--report", rep],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(os.path.exists(rep))

    def test_report_schema(self):
        rep = os.path.join(self.tmp, "step", "result.json")
        out = os.path.join(self.tmp, "step", "out.glb")
        cli.write_report(rep, [out], "rig", True, {"joints_raw": 3})
        d = json.load(open(rep))
        self.assertEqual(d["outputs"], ["out.glb"])
        self.assertEqual((d["ok"], d["tool"], d["stage"]), (True, "skintokens", "rig"))
        self.assertIn("rig", d)


if __name__ == "__main__":
    unittest.main()
