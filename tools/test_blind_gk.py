"""Gate-first tests for `creature_blind.py` -> `gk blind` delegation and `unirig-mac` cleanup."""

import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

TOOLS = os.path.dirname(os.path.abspath(__file__))


def _make_png(path: str, rgb: tuple[int, int, int]) -> None:
    w, h = 16, 16
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_fake_gk(path: str, log_path: str) -> None:
    code = f"""#!/usr/bin/env python3
import json, os, shutil, sys
with open({log_path!r}, "a") as lf:
    lf.write(" ".join(sys.argv[1:]) + "\\n")
assert sys.argv[1] == "blind", sys.argv
sub = sys.argv[2]
if sub == "make":
    a_dir = sys.argv[sys.argv.index("--a") + 1]
    b_dir = sys.argv[sys.argv.index("--b") + 1]
    out_dir = sys.argv[sys.argv.index("--out") + 1]
    planted = sys.argv[sys.argv.index("--planted") + 1] if "--planted" in sys.argv else None
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    q = sys.argv[sys.argv.index("--question") + 1] if "--question" in sys.argv else "Which side is better?"
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted(f for f in os.listdir(a_dir) if f.endswith(".png"))
    pairs_pub, pairs_key = [], []
    n = 0
    for idx, name in enumerate(imgs):
        n += 1
        pid = f"pair-{{n:02d}}"
        flip = ((seed + idx) % 2) == 1
        left_side = "b" if flip else "a"
        src_l = os.path.join(b_dir if flip else a_dir, name)
        src_r = os.path.join(a_dir if flip else b_dir, name)
        shutil.copyfile(src_l, os.path.join(out_dir, f"{{pid}}-left.png"))
        shutil.copyfile(src_r, os.path.join(out_dir, f"{{pid}}-right.png"))
        shutil.copyfile(src_l, os.path.join(out_dir, f"{{pid}}.png"))
        pairs_pub.append({{"id": pid, "left": f"{{pid}}-left.png", "right": f"{{pid}}-right.png", "sheet": f"{{pid}}.png"}})
        pairs_key.append({{"id": pid, "kind": "claim", "claim": os.path.splitext(name)[0], "left": left_side, "rmse": 10.0}})
    n += 1
    pid = f"pair-{{n:02d}}"
    shutil.copyfile(os.path.join(a_dir, imgs[0]), os.path.join(out_dir, f"{{pid}}-left.png"))
    shutil.copyfile(os.path.join(a_dir, imgs[0]), os.path.join(out_dir, f"{{pid}}-right.png"))
    shutil.copyfile(os.path.join(a_dir, imgs[0]), os.path.join(out_dir, f"{{pid}}.png"))
    pairs_pub.append({{"id": pid, "left": f"{{pid}}-left.png", "right": f"{{pid}}-right.png", "sheet": f"{{pid}}.png"}})
    pairs_key.append({{"id": pid, "kind": "control", "claim": os.path.splitext(imgs[0])[0] + " (control)", "left": "a", "rmse": 0.0}})
    if planted:
        n += 1
        pid = f"pair-{{n:02d}}"
        flip_p = (seed % 2) == 1
        left_p = "planted" if flip_p else "a"
        src_l = planted if flip_p else os.path.join(a_dir, imgs[0])
        src_r = os.path.join(a_dir, imgs[0]) if flip_p else planted
        shutil.copyfile(src_l, os.path.join(out_dir, f"{{pid}}-left.png"))
        shutil.copyfile(src_r, os.path.join(out_dir, f"{{pid}}-right.png"))
        shutil.copyfile(src_l, os.path.join(out_dir, f"{{pid}}.png"))
        pairs_pub.append({{"id": pid, "left": f"{{pid}}-left.png", "right": f"{{pid}}-right.png", "sheet": f"{{pid}}.png"}})
        pairs_key.append({{"id": pid, "kind": "planted", "claim": os.path.splitext(imgs[0])[0] + " (planted)", "left": left_p, "rmse": 50.0}})
    with open(os.path.join(out_dir, "judge.json"), "w") as f:
        json.dump({{"version": 1, "question": q, "answers": "one of left | right | cant_tell per pair", "pairs": pairs_pub}}, f)
    with open(out_dir + ".key.json", "w") as f:
        json.dump({{"version": 1, "seed": seed, "question": q, "pairs": pairs_key}}, f)
    sys.exit(0)
if sub == "score":
    out_dir = sys.argv[3]
    ans_path = sys.argv[sys.argv.index("--answers") + 1]
    key_path = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else out_dir + ".key.json"
    with open(ans_path) as f:
        ans = json.load(f)
    with open(key_path) as f:
        key = json.load(f)
    ctrl_ok, plant_ok = True, True
    a_wins, b_wins, cant = 0, 0, 0
    rows = []
    for p in key["pairs"]:
        a = ans.get(p["id"], "cant_tell")
        if p["kind"] == "control":
            ctrl_ok = ctrl_ok and (a == "cant_tell")
            w = "cant_tell" if a == "cant_tell" else "a"
        elif p["kind"] == "planted":
            want = "left" if p["left"] == "a" else "right"
            plant_ok = plant_ok and (a == want)
            w = "a" if a == want else "b"
        else:
            if a == "cant_tell":
                w = "cant_tell"
                cant += 1
            else:
                picked = p["left"] if a == "left" else ("b" if p["left"] == "a" else "a")
                w = picked
                if w == "a": a_wins += 1
                else: b_wins += 1
        rows.append({{"pair": p["id"], "kind": p["kind"], "claim": p["claim"], "answer": a, "winner": w}})
    v = {{"version": 1, "pass": ctrl_ok and plant_ok, "control_ok": ctrl_ok, "planted_ok": plant_ok, "claims": {{"a_wins": a_wins, "b_wins": b_wins, "cant_tell": cant}}, "pairs": rows}}
    with open(os.path.join(out_dir, "verdict.json"), "w") as f:
        json.dump(v, f)
    print(json.dumps(v))
    sys.exit(0 if v["pass"] else 1)
sys.exit(2)
"""
    with open(path, "w") as f:
        f.write(code)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class CreatureBlindGkTest(unittest.TestCase):
    def test_make_and_score_delegate_to_gk_blind(self):
        sys.path.insert(0, TOOLS)
        import creature_blind as cb  # noqa: E402

        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "gk.log")
            fake_gk = os.path.join(d, "gk")
            write_fake_gk(fake_gk, log)

            a1 = os.path.join(d, "a1.png")
            b1 = os.path.join(d, "b1.png")
            _make_png(a1, (200, 30, 30))
            _make_png(b1, (30, 200, 30))
            raw_items = [
                {
                    "category": "claim",
                    "subject": "fox",
                    "a_name": "predicted",
                    "b_name": "body",
                    "a_pngs": [a1],
                    "b_pngs": [b1],
                    "expected": "body",
                    "truth_winner": "body",
                    "pick_winner": "body",
                    "old_winner": "predicted",
                },
                {
                    "category": "control",
                    "subject": "fox",
                    "a_name": "same_a",
                    "b_name": "same_b",
                    "a_pngs": [a1],
                    "b_pngs": [a1],
                    "expected": "cant_tell",
                },
                {
                    "category": "planted",
                    "subject": "fox",
                    "a_name": "clean",
                    "b_name": "planted_swapped_limbs",
                    "a_pngs": [a1],
                    "b_pngs": [b1],
                    "expected": "clean",
                    "planted_metric_ok": True,
                },
            ]
            out_dir = os.path.join(d, "pack")
            old_gk = os.environ.get("GK_BIN")
            os.environ["GK_BIN"] = fake_gk
            try:
                key_path = cb.write_blind_pack(raw_items, out_dir, seed=99)
                self.assertEqual(os.path.abspath(key_path), os.path.abspath(out_dir + ".key.json"))
                self.assertTrue(os.path.isfile(os.path.join(out_dir, "judge.json")), os.listdir(out_dir))
                self.assertTrue(os.path.isfile(key_path))
                self.assertFalse(os.path.exists(os.path.join(out_dir, os.path.basename(out_dir) + ".key.json")))
                with open(log) as lf:
                    self.assertIn("blind make", lf.read())

                with open(key_path) as kf:
                    key = json.load(kf)
                ans = {}
                for p in key["pairs"]:
                    if p["category"] == "control":
                        ans[p["id"]] = "cant_tell"
                    elif p["category"] == "planted":
                        ans[p["id"]] = "left" if p["left_rig"] == p["expected"] else "right"
                    else:
                        ans[p["id"]] = "left" if p["left_rig"] == p["truth_winner"] else "right"
                ans_path = os.path.join(d, "ans.json")
                with open(ans_path, "w") as af:
                    json.dump(ans, af)
                r = subprocess.run(
                    [sys.executable, os.path.join(TOOLS, "creature_blind.py"), "score", out_dir, "--answers", ans_path],
                    capture_output=True, text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + "\n" + r.stdout)
                with open(log) as lf:
                    self.assertIn("blind score", lf.read())
                with open(os.path.join(out_dir, "verdict.json")) as vf:
                    verdict = json.load(vf)
                self.assertTrue(verdict["pass"])
                self.assertTrue(verdict["control_ok"])
                self.assertTrue(verdict["planted_ok"])
                self.assertEqual(verdict["old_score_retired_pairs"], [p["pair"] for p in key["pairs"] if p["category"] == "claim"])
            finally:
                if old_gk is None:
                    os.environ.pop("GK_BIN", None)
                else:
                    os.environ["GK_BIN"] = old_gk

    def test_missing_gk_exits_nonzero(self):
        sys.path.insert(0, TOOLS)
        import creature_blind as cb  # noqa: E402

        with tempfile.TemporaryDirectory() as d:
            old_gk, old_path = os.environ.get("GK_BIN"), os.environ.get("PATH")
            os.environ["GK_BIN"] = os.path.join(d, "no_such_gk")
            os.environ["PATH"] = d
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    cb.find_gk_bin()
                self.assertIn("gk", str(ctx.exception).lower())
            finally:
                if old_gk is None:
                    os.environ.pop("GK_BIN", None)
                else:
                    os.environ["GK_BIN"] = old_gk
                if old_path is not None:
                    os.environ["PATH"] = old_path

    def test_run_unirig_rival_has_no_unirig_mac_default(self):
        script = os.path.join(TOOLS, "bench", "run_unirig_rival.py")
        rh = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        self.assertEqual(rh.returncode, 0, rh.stderr)
        self.assertNotIn("unirig-mac-5", rh.stdout)
        self.assertNotIn("/data/f/Games/_blender/unirig-mac", rh.stdout)
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, script, d],
                env={k: v for k, v in os.environ.items() if k not in ("UNIRIG_DIR", "UNIRIG_PYTHON")},
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("unirig-mac", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
