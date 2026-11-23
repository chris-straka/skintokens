"""Gate-first tests for unified `autorig` CLI (`bin/autorig` / `tools/autorig.py`)."""

import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest


def _find_built_creature_rig() -> str | None:
    target_dir = os.environ.get("CARGO_TARGET_DIR") or os.path.join(REPO, "target")
    cands = [
        os.environ.get("CREATURE_RIG_BIN", ""),
        os.path.join(target_dir, "release", "creature-rig"),
        os.path.join(target_dir, "debug", "creature-rig"),
        os.path.join(REPO, "target", "release", "creature-rig"),
        os.path.join(REPO, "target", "debug", "creature-rig"),
    ]
    for p in cands:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _write_rigged_fixture_glb(path: str) -> None:
    verts, tris, joints, weights = [], [], [], []

    def add_box(lo, hi, j):
        base = len(verts)
        for x, y, z in (
            (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
        ):
            verts.append((x, y, z))
            joints.append((j, 0, 0, 0))
            weights.append((1.0, 0.0, 0.0, 0.0))
        for a, b, c in (
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ):
            tris.append((base + a, base + b, base + c))

    joint_specs = [
        ("pelvis", None, (0.0, 0.45, 0.0)),
        ("torso", 0, (0.0, 0.70, 0.0)),
        ("head", 1, (0.0, 1.00, 0.0)),
        ("hand.L", 1, (0.32, 0.72, 0.0)),
        ("finger1.L", 3, (0.38, 0.72, 0.03)),
        ("finger2.L", 3, (0.39, 0.72, 0.015)),
        ("finger3.L", 3, (0.40, 0.72, 0.0)),
        ("finger4.L", 3, (0.39, 0.72, -0.015)),
        ("finger5.L", 3, (0.38, 0.72, -0.03)),
        ("ear_spike_out", 2, (1.40, 1.65, 0.0)),
    ]
    add_box((-0.14, 0.0, -0.10), (0.14, 0.90, 0.10), 0)
    add_box((-0.10, 0.90, -0.09), (0.10, 1.12, 0.09), 2)
    add_box((0.08, 0.96, -0.04), (0.12, 1.04, 0.04), 9)
    add_box((0.12, 0.66, -0.05), (0.36, 0.78, 0.05), 3)
    for fi in range(5):
        z0 = -0.035 + fi * 0.015
        add_box((0.36, 0.69, z0), (0.42, 0.75, z0 + 0.01), 4 + fi)

    lo = [min(p[a] for p in verts) for a in range(3)]
    hi = [max(p[a] for p in verts) for a in range(3)]
    buf = bytearray()

    def push(data: bytes) -> tuple[int, int]:
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(data)
        return off, len(data)

    pos_off, pos_len = push(b"".join(struct.pack("<fff", *v) for v in verts))
    idx_off, idx_len = push(b"".join(struct.pack("<III", *t) for t in tris))
    jnt_off, jnt_len = push(b"".join(struct.pack("<HHHH", *j) for j in joints))
    wgt_off, wgt_len = push(b"".join(struct.pack("<ffff", *w) for w in weights))
    ibms = bytearray()
    for _, _, h in joint_specs:
        ibms.extend(struct.pack(
            "<16f",
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -h[0], -h[1], -h[2], 1.0,
        ))
    ibm_off, ibm_len = push(bytes(ibms))

    nodes = [{"name": "body", "mesh": 0, "skin": 0}]
    for name, pi, h in joint_specs:
        if pi is None:
            local = list(h)
        else:
            ph = joint_specs[pi][2]
            local = [h[0] - ph[0], h[1] - ph[1], h[2] - ph[2]]
        nodes.append({"name": name, "translation": local})
    for idx, (_, pi, _) in enumerate(joint_specs):
        if pi is not None:
            nodes[1 + pi].setdefault("children", []).append(1 + idx)

    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_len, "target": 34963},
            {"buffer": 0, "byteOffset": jnt_off, "byteLength": jnt_len, "target": 34962},
            {"buffer": 0, "byteOffset": wgt_off, "byteLength": wgt_len, "target": 34962},
            {"buffer": 0, "byteOffset": ibm_off, "byteLength": ibm_len},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": 1, "componentType": 5125, "count": len(tris) * 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5123, "count": len(joints), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": len(weights), "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": len(joint_specs), "type": "MAT4"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1, "mode": 4}]}],
        "nodes": nodes,
        "skins": [{"joints": [1 + i for i in range(len(joint_specs))], "skeleton": 1, "inverseBindMatrices": 4}],
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0,
    }
    jb = json.dumps(doc).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = bytes(buf) + b"\x00" * ((4 - len(buf) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(bb), 0x004E4942) + bb)

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
BIN = os.path.join(REPO, "bin", "autorig")


def _write_exe(path: str, body: str) -> None:
    with open(path, "w") as f:
        f.write("#!/usr/bin/env python3\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def _pass_stubs(d: str) -> tuple[str, str, dict[str, str]]:
    fake_cb = os.path.join(d, "stub_creature_bench")
    _write_exe(fake_cb, """
import json, sys
jpath = sys.argv[sys.argv.index("--json") + 1]
with open(jpath, "w") as f:
    json.dump({"pass": True, "findings": []}, f)
sys.exit(0)
""")
    fake_wf = os.path.join(d, "stub_weights")
    _write_exe(fake_wf, """
import json, shutil, sys
if sys.argv[1] == "fix":
    shutil.copyfile(sys.argv[2], sys.argv[sys.argv.index("--out") + 1])
    sys.exit(0)
print(json.dumps({"pass": True, "score": 90.0, "tears_pct": 0.0, "headline": [{"name": "poses", "tears_pct": 0.0, "findings": []}], "findings": []}))
""")
    env = dict(os.environ, HOME=d)
    return fake_cb, fake_wf, env


class AutorigContractTest(unittest.TestCase):
    def test_help_lists_all_methods_and_unknown_method_exits_2(self):
        self.assertTrue(os.path.isfile(BIN), f"missing {BIN}")
        r = subprocess.run([BIN, "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for m in ("auto", "skintokens", "gamerig", "rigforge", "body", "creature-rig", "motion", "unirig", "puppeteer", "magicarticulate"):
            self.assertIn(m, r.stdout, f"--help missing method {m}")

        with tempfile.TemporaryDirectory() as d:
            in_glb = os.path.join(d, "in.glb")
            out_glb = os.path.join(d, "out.glb")
            with open(in_glb, "wb") as f:
                f.write(b"glTF")
            bad = subprocess.run([BIN, in_glb, out_glb, "--method", "bogus"], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertFalse(os.path.exists(out_glb))

    def test_gamerig_and_rigforge_aliases_route_presets(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "calls.log")
            fake_gr = os.path.join(d, "auto_rig.sh")
            _write_exe(fake_gr, f"""
import sys
with open({log!r}, "a") as lf:
    lf.write(" ".join(sys.argv[1:]) + "\\n")
out = sys.argv[sys.argv.index("--out") + 1]
with open(out, "wb") as f:
    f.write(b"glTF-gamerig")
sys.exit(0)
""")
            fake_cr = os.path.join(d, "creature-rig")
            _write_exe(fake_cr, """
import json, shutil, sys
if sys.argv[1] == "prune":
    shutil.copyfile(sys.argv[2], sys.argv[3])
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({"fingers_folded": 0, "outside_cut": 0}, f)
    sys.exit(0)
print(json.dumps({"pass": True, "score": 90.0}))
""")
            fake_cb, fake_wf, env = _pass_stubs(d)
            in_glb = os.path.join(d, "mesh.glb")
            with open(in_glb, "wb") as f:
                f.write(b"glTF")
            for method, klass, kind, want_preset in [
                ("gamerig", "hero", "humanoid", "biped_hero"),
                ("rigforge", "monster", "quadruped", "quadruped_stalker"),
            ]:
                out_glb = os.path.join(d, f"{method}.glb")
                r = subprocess.run(
                    [BIN, in_glb, out_glb, "--method", method, "--class", klass, "--kind", kind,
                     "--gamerig", fake_gr, "--creature-rig", fake_cr,
                     "--creature-bench", fake_cb, "--weights", fake_wf],
                    capture_output=True, text=True, env=env,
                )
                self.assertEqual(r.returncode, 0, f"{method} failed: {r.stderr}\n{r.stdout}")
                with open(out_glb, "rb") as f:
                    self.assertEqual(f.read(), b"glTF-gamerig")
            with open(log) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2, lines)
            self.assertIn("--preset biped_hero", lines[0])
            self.assertIn("--preset quadruped_stalker", lines[1])

    def test_non_skinning_skeleton_gets_skinned_and_pruned(self):
        with tempfile.TemporaryDirectory() as d:
            events = os.path.join(d, "events.log")
            fake_sk = os.path.join(d, "skintokens")
            _write_exe(fake_sk, f"""
import shutil, sys
with open({events!r}, "a") as lf:
    lf.write("sk:" + sys.argv[1] + "\\n")
shutil.copyfile(sys.argv[2], sys.argv[3])
sys.exit(0)
""")
            fake_cr = os.path.join(d, "creature-rig")
            _write_exe(fake_cr, f"""
import json, shutil, sys
with open({events!r}, "a") as lf:
    lf.write("cr:" + " ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "prune":
    shutil.copyfile(sys.argv[2], sys.argv[3])
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({{"fingers_folded": 1, "outside_cut": 1}}, f)
    sys.exit(0)
sys.exit(0)
""")
            fake_rival = os.path.join(d, "puppeteer_pred.py")
            # Emit a minimal unskinned GLB (only nodes, no skin/JOINTS_0).
            _write_exe(fake_rival, f"""
import json, struct, sys
with open({events!r}, "a") as lf:
    lf.write("rival:puppeteer\\n")
out = sys.argv[sys.argv.index("--out") + 1]
j = json.dumps({{"asset": {{"version": "2.0"}}, "nodes": [{{"name": "root"}}], "meshes": []}}).encode()
j += b" " * ((4 - len(j) % 4) % 4)
total = 12 + 8 + len(j)
glb = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(j), 0x4E4F534A) + j
with open(out, "wb") as f:
    f.write(glb)
""")
            fake_cb, fake_wf, env = _pass_stubs(d)
            in_glb = os.path.join(d, "mesh.glb")
            with open(in_glb, "wb") as f:
                f.write(b"glTF")
            out_glb = os.path.join(d, "out.glb")
            r = subprocess.run(
                [BIN, in_glb, out_glb, "--method", "puppeteer",
                 "--rival-cmd", f"{sys.executable} {fake_rival} --in {{in}} --out {{out}}",
                 "--skintokens", fake_sk, "--creature-rig", fake_cr,
                 "--creature-bench", fake_cb, "--weights", fake_wf, "--no-fix"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(r.returncode, 0, r.stderr + "\n" + r.stdout)
            with open(events) as f:
                evs = f.read().splitlines()
            self.assertIn("rival:puppeteer", evs)
            self.assertIn("sk:skin", evs, f"expected skintokens skin on skeleton-only output: {evs}")
            self.assertTrue(any(e.startswith("cr:prune") for e in evs), f"expected creature-rig prune: {evs}")

    def test_auto_prunes_respecting_class_hero_and_picks_cleaner_passing_seed(self):
        with tempfile.TemporaryDirectory() as d:
            events = os.path.join(d, "events.log")
            fake_sk = os.path.join(d, "skintokens")
            _write_exe(fake_sk, f"""
import sys
seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else "0"
with open({events!r}, "a") as lf:
    lf.write(f"rig:{{seed}}\\n")
with open(sys.argv[3], "wb") as f:
    f.write(f"GLB-SEED-{{seed}}".encode())
""")
            fake_cr = os.path.join(d, "creature-rig")
            _write_exe(fake_cr, f"""
import json, shutil, sys
cmd = sys.argv[1]
with open({events!r}, "a") as lf:
    lf.write(cmd + ":" + " ".join(sys.argv[2:]) + "\\n")
if cmd == "prune":
    shutil.copyfile(sys.argv[2], sys.argv[3])
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({{"fingers_folded": 0, "outside_cut": 1}}, f)
    sys.exit(0)
if cmd == "check":
    print(json.dumps({{"pass": True, "score": 90.0, "bones_along_nothing": 0}}))
    sys.exit(0)
""")
            fake_cb = os.path.join(d, "creature-bench")
            _write_exe(fake_cb, """
import json, sys
jpath = sys.argv[sys.argv.index("--json") + 1]
with open(sys.argv[2], "rb") as f:
    data = f.read()
ok = b"SEED-2" in data or b"SEED-1" in data
with open(jpath, "w") as f:
    json.dump({"pass": ok, "findings": []}, f)
sys.exit(0 if ok else 1)
""")
            fake_wf = os.path.join(d, "weights")
            _write_exe(fake_wf, """
import json, sys
with open(sys.argv[2], "rb") as f:
    data = f.read()
# Seed 1 tears 8.0%, seed 2 tears 1.0% -> auto must choose seed 2.
tears = 8.0 if b"SEED-1" in data else 1.0
print(json.dumps({"pass": tears < 5.0, "score": 70.0, "tears_pct": tears, "headline": [{"name": "poses", "tears_pct": tears, "findings": []}], "findings": []}))
sys.exit(0 if tears < 5.0 else 1)
""")
            in_glb = os.path.join(d, "hero.glb")
            with open(in_glb, "wb") as f:
                f.write(b"glTF")
            out_glb = os.path.join(d, "hero_rig.glb")
            rep = os.path.join(d, "hero_rig.json")
            r = subprocess.run(
                [BIN, in_glb, out_glb, "--method", "auto", "--class", "hero", "--seeds", "1,2",
                 "--skintokens", fake_sk, "--creature-rig", fake_cr,
                 "--creature-bench", fake_cb, "--weights", fake_wf, "--no-fix", "--report", rep],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + "\n" + r.stdout)
            with open(out_glb, "rb") as f:
                self.assertEqual(f.read(), b"GLB-SEED-2")
            with open(events) as f:
                evs = f.read()
            self.assertIn("--keep-digits", evs, f"--class hero must pass --keep-digits to prune: {evs}")

            # Nonzero tool error exits 2 when all candidates fail.
            bad_sk = os.path.join(d, "bad_skintokens")
            _write_exe(bad_sk, "import sys; sys.stderr.write('boom\\n'); sys.exit(2)\n")
            out_fail = os.path.join(d, "fail.glb")
            rf = subprocess.run(
                [BIN, in_glb, out_fail, "--method", "skintokens", "--skintokens", bad_sk, "--no-prune", "--no-fix"],
                capture_output=True, text=True,
            )
            self.assertEqual(rf.returncode, 2)
            self.assertFalse(os.path.exists(out_fail))

    def test_bin_autorig_real_creature_rig_prune_hero_vs_monster(self):
        cr_bin = _find_built_creature_rig()
        self.assertIsNotNone(cr_bin, "built creature-rig binary not found (build with cargo build --release -p creature-rig)")
        with tempfile.TemporaryDirectory() as d:
            fixture = os.path.join(d, "rigged_fixture.glb")
            _write_rigged_fixture_glb(fixture)
            fake_sk = os.path.join(d, "skintokens")
            _write_exe(fake_sk, f"import shutil, sys\nshutil.copyfile({fixture!r}, sys.argv[3])\n")
            fake_cb, fake_wf, env = _pass_stubs(d)

            out_monster = os.path.join(d, "monster.glb")
            rep_monster = os.path.join(d, "monster.json")
            rm = subprocess.run(
                [BIN, fixture, out_monster, "--method", "skintokens", "--class", "monster",
                 "--skintokens", fake_sk, "--creature-rig", cr_bin,
                 "--creature-bench", fake_cb, "--weights", fake_wf, "--no-fix", "--report", rep_monster],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(rm.returncode, 0, rm.stderr + "\n" + rm.stdout)
            with open(rep_monster) as f:
                pm = json.load(f)["winner"]["prune"]
            self.assertGreaterEqual(pm["outside_cut"], 1, pm)
            self.assertGreaterEqual(pm["fingers_folded"], 5, pm)

            out_hero = os.path.join(d, "hero.glb")
            rep_hero = os.path.join(d, "hero.json")
            rh = subprocess.run(
                [BIN, fixture, out_hero, "--method", "skintokens", "--class", "hero",
                 "--skintokens", fake_sk, "--creature-rig", cr_bin,
                 "--creature-bench", fake_cb, "--weights", fake_wf, "--no-fix", "--report", rep_hero],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(rh.returncode, 0, rh.stderr + "\n" + rh.stdout)
            with open(rep_hero) as f:
                ph = json.load(f)["winner"]["prune"]
            self.assertGreaterEqual(ph["outside_cut"], 1, ph)
            self.assertEqual(ph["fingers_folded"], 0, ph)


if __name__ == "__main__":
    unittest.main()
