#!/usr/bin/env python3
"""Unit tests for `tools/bench/run_skeleton_rivals.py` (stdlib only, synthetic fixtures)."""
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bench"))
import run_skeleton_rivals as rsr  # noqa: E402


def _make_synthetic_bare_glb(path: str) -> None:
    """Create a minimal valid glTF 2.0 GLB with a 2-triangle quad mesh."""
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]
    tris = [0, 1, 2, 0, 2, 3]
    bin_buf = bytearray()
    pos_raw = b"".join(struct.pack("<fff", *v) for v in verts)
    bin_buf.extend(pos_raw)
    idx_off = len(bin_buf)
    idx_raw = b"".join(struct.pack("<H", i) for i in tris)
    bin_buf.extend(idx_raw)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "body", "mesh": 0}],
        "meshes": [
            {
                "name": "body",
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}],
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_raw)},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_raw)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 2.0, 0.0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(tris),
                "type": "SCALAR",
            },
        ],
        "buffers": [{"byteLength": len(bin_buf)}],
    }
    rsr.write_glb(path, gltf, bin_buf)


class TestSkeletonRivals(unittest.TestCase):
    def test_parse_rig_txt_tree_and_disconnected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "pred.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(
                    "joints j0 0.0 0.5 0.0\n"
                    "joints j1 0.0 1.0 0.0\n"
                    "joints j2 0.0 1.5 0.0\n"
                    "joints j3 0.2 1.6 0.0\n"
                    "root j0\n"
                    "hier j0 j1\n"
                    "hier j1 j2\n"
                    "hier j2 j0\n"  # cycle back to root
                    # j3 disconnected -> should attach to nearest visited (j2)
                )
            names, positions, parents = rsr.parse_rig_txt(p)
            self.assertEqual(len(names), 4)
            self.assertEqual(names[0], "j0")
            self.assertEqual(parents[0], -1)
            self.assertEqual(parents.count(-1), 1)
            # Every non-root parent must precede its child
            for i in range(1, len(parents)):
                self.assertGreaterEqual(parents[i], 0)
                self.assertLess(parents[i], i)
            idx_j2 = names.index("j2")
            idx_j3 = names.index("j3")
            self.assertEqual(parents[idx_j3], idx_j2)

    def test_glb_to_obj_and_attach_skeleton_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = os.path.join(tmp, "bare.glb")
            obj = os.path.join(tmp, "mesh.obj")
            out_rigid = os.path.join(tmp, "rigid.glb")
            out_skin_in = os.path.join(tmp, "skin_in.glb")
            _make_synthetic_bare_glb(bare)

            nv, nf = rsr.glb_to_obj(bare, obj)
            self.assertEqual((nv, nf), (4, 2))
            lines = [l.strip() for l in open(obj, encoding="utf-8") if l.strip()]
            self.assertEqual(sum(1 for l in lines if l.startswith("v ")), 4)
            self.assertEqual(sum(1 for l in lines if l.startswith("f ")), 2)

            names = ["root", "mid", "tip", "far_leaf"]
            positions = [(0.5, 0.2, 0.0), (0.5, 1.0, 0.0), (0.5, 1.9, 0.0), (0.5, 5.0, 0.0)]
            parents = [-1, 0, 1, 2]

            rsr.attach_skeleton_to_glb(bare, out_rigid, names, positions, parents, min_joint_weight=0.0)
            gltf, bin_buf = rsr.read_glb(out_rigid)
            self.assertEqual(len(gltf["skins"]), 1)
            self.assertEqual(len(gltf["skins"][0]["joints"]), 4)
            prim = gltf["meshes"][0]["primitives"][0]
            self.assertIn("JOINTS_0", prim["attributes"])
            self.assertIn("WEIGHTS_0", prim["attributes"])
            ibms = rsr._read_accessor(gltf, bin_buf, gltf["skins"][0]["inverseBindMatrices"])
            self.assertEqual(len(ibms), 4)
            # Column-major translation in IBM[0] is (-0.5, -0.2, 0.0) at indices 12, 13, 14
            self.assertAlmostEqual(ibms[0][12], -0.5, places=5)
            self.assertAlmostEqual(ibms[0][13], -0.2, places=5)

            # With min_joint_weight > 0, even far_leaf (joint 3) must hold > 1e-6 weight
            rsr.attach_skeleton_to_glb(bare, out_skin_in, names, positions, parents, min_joint_weight=1e-4)
            gltf2, bin_buf2 = rsr.read_glb(out_skin_in)
            prim2 = gltf2["meshes"][0]["primitives"][0]
            js = rsr._read_accessor(gltf2, bin_buf2, prim2["attributes"]["JOINTS_0"])
            ws = rsr._read_accessor(gltf2, bin_buf2, prim2["attributes"]["WEIGHTS_0"])
            wsum = [0.0] * 4
            for j4, w4 in zip(js, ws):
                self.assertAlmostEqual(sum(w4), 1.0, places=5)
                for j, w in zip(j4, w4):
                    wsum[j] += w
            for j in range(4):
                self.assertGreater(wsum[j], 1e-6, f"joint {j} got zero weight")

    def test_patch_puppeteer_sdpa_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "skeleton_models"))
            os.makedirs(os.path.join(tmp, "utils"))
            with open(os.path.join(tmp, "skeleton_models", "skeletongen.py"), "w", encoding="utf-8") as f:
                f.write('cfg = dict(_attn_implementation="flash_attention_2")\n')
            with open(os.path.join(tmp, "skeleton_models", "skeleton_opt.py"), "w", encoding="utf-8") as f:
                f.write(
                    "from transformers.models.opt.modeling_opt import OPTDecoder\n"
                    'self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"\n'
                    'raise ValueError("Only flash_attention_2 is supported")\n'
                )
            with open(os.path.join(tmp, "utils", "save_utils.py"), "w", encoding="utf-8") as f:
                f.write("from data_utils.pyrender_wrapper import PyRenderWrapper\n")
            with open(os.path.join(tmp, "utils", "mesh_to_pc.py"), "w", encoding="utf-8") as f:
                f.write("import mesh2sdf.core\n")

            c1 = rsr.patch_puppeteer_sdpa(tmp)
            self.assertEqual(len(c1), 4)
            c2 = rsr.patch_puppeteer_sdpa(tmp)
            self.assertEqual(len(c2), 0)
            opt_txt = open(os.path.join(tmp, "skeleton_models", "skeleton_opt.py"), encoding="utf-8").read()
            self.assertIn("_prepare_4d_causal_attention_mask_for_sdpa", opt_txt)
            self.assertNotIn("Only flash_attention_2 is supported", opt_txt)


if __name__ == "__main__":
    unittest.main()
