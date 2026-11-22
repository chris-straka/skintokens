#!/usr/bin/env python3
"""Regression tests for exit-code handling in `creature_route.py` and `tools/bench/heldout.sh`."""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TOOLS_DIR)


def _write_exe(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class ExitCodeHandlingTest(unittest.TestCase):
    def test_creature_route_does_not_swallow_tool_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(bin_dir)
            os.makedirs(out_dir)

            mesh = os.path.join(tmp, "mesh.glb")
            with open(mesh, "wb") as f:
                f.write(b"glTF")

            cr = os.path.join(bin_dir, "creature-rig")
            cb = os.path.join(bin_dir, "creature-bench")
            wf = os.path.join(bin_dir, "weights")
            mf = os.path.join(bin_dir, "motionforge")
            rf = os.path.join(bin_dir, "rfcheck")

            _write_exe(
                cr,
                """#!/usr/bin/env python3
import json, os, sys
cmd = sys.argv[1]
if cmd == "rig":
    out = sys.argv[3]
    with open(out, "wb") as f: f.write(b"rig")
    for i, a in enumerate(sys.argv):
        if a == "--rom":
            with open(sys.argv[i+1], "w") as f: f.write("()")
        if a == "--json":
            with open(sys.argv[i+1], "w") as f:
                json.dump({"kind": "four-legged", "upright": False, "bones": ["root"], "limbs": []}, f)
        if a == "--sheet":
            with open(sys.argv[i+1], "wb") as f: f.write(b"PNG")
    sys.exit(0)
if cmd == "check":
    for i, a in enumerate(sys.argv):
        if a == "--json":
            with open(sys.argv[i+1], "w") as f:
                json.dump({"score": 100.0, "pass": True, "findings": []}, f)
        if a == "--rom":
            with open(sys.argv[i+1], "w") as f: f.write("()")
    sys.exit(0)
""",
            )
            # Plant a stale valid-looking anatomy JSON on disk and have creature-bench anatomy exit 2!
            stale_anat = os.path.join(out_dir, "try.anatomy.json")
            with open(stale_anat, "w", encoding="utf-8") as f:
                json.dump({"pass": True, "findings": [], "limbs": []}, f)

            _write_exe(
                cb,
                """#!/usr/bin/env python3
import sys
sys.stderr.write("anatomy crashed\\n")
sys.exit(2)
""",
            )
            _write_exe(
                wf,
                """#!/usr/bin/env python3
import json, sys
if sys.argv[1] == "check":
    print(json.dumps({"score": 95.0, "pass": True, "findings": []}))
    sys.exit(0)
if sys.argv[1] == "sheet":
    sys.stderr.write("sheet failed\\n")
    sys.exit(2)
sys.exit(0)
""",
            )
            _write_exe(mf, "#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
            _write_exe(
                rf,
                '#!/usr/bin/env python3\nimport json, sys; print(json.dumps({"diags": []})); sys.exit(0)\n',
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(TOOLS_DIR, "creature_route.py"),
                    mesh,
                    out_dir,
                    "--name",
                    "testcritter",
                    "--clean-mesh",
                    mesh,
                    "--no-clips",
                    "--no-predicted",
                    "--creature-rig",
                    cr,
                    "--creature-bench",
                    cb,
                    "--weights",
                    wf,
                    "--motionforge",
                    mf,
                    "--rfcheck",
                    rf,
                ],
                capture_output=True,
                text=True,
            )
            # creature-bench anatomy exited 2 (crash/error): anatomy_of MUST NOT trust stale JSON!
            with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
                rep = json.load(f)
            anat = rep["tries"][0]["anatomy"]
            self.assertFalse(anat.get("pass"), f"ignored exit 2 from creature-bench anatomy: {anat}")
            self.assertEqual(
                [x.get("code") for x in anat.get("findings", [])],
                ["ANATOMY_DID_NOT_RUN"],
                f"expected ANATOMY_DID_NOT_RUN when anatomy exits 2, got {anat}; route exit={proc.returncode}",
            )

            # Now let anatomy pass, but have weights sheet (fallback when no sheet exists) exit 2 -> route must fail exit 2!
            os.remove(os.path.join(out_dir, "try.sheet.png")) if os.path.exists(os.path.join(out_dir, "try.sheet.png")) else None
            _write_exe(
                cb,
                """#!/usr/bin/env python3
import json, sys
for i, a in enumerate(sys.argv):
    if a == "--json":
        with open(sys.argv[i+1], "w") as f:
            json.dump({"pass": True, "findings": [], "limbs": []}, f)
sys.exit(0)
""",
            )
            proc2 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(TOOLS_DIR, "creature_route.py"),
                    mesh,
                    out_dir,
                    "--name",
                    "testcritter",
                    "--clean-mesh",
                    mesh,
                    "--no-clips",
                    "--no-predicted",
                    "--creature-rig",
                    cr,
                    "--creature-bench",
                    cb,
                    "--weights",
                    wf,
                    "--motionforge",
                    mf,
                    "--rfcheck",
                    rf,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc2.returncode,
                2,
                f"expected exit 2 when weights sheet fails with exit 2, got {proc2.returncode}: {proc2.stdout} {proc2.stderr}",
            )

    def test_heldout_sh_fails_when_score_artists_or_route_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            import hashlib

            def sha(b):
                return hashlib.sha256(b).hexdigest()

            bench = os.path.join(tmp, "bench")
            os.makedirs(bench)
            kinds = [
                "winged",
                "hunched",
                "long-armed",
                "many-legged",
                "legless",
                "quadruped",
                "quadruped",
                "quadruped",
                "quadruped",
                "quadruped",
                "quadruped",
                "quadruped",
            ]
            animals = []
            tsv_lines = []
            for idx, kind in enumerate(kinds):
                aid = f"test_animal_{idx:02d}"
                adir = os.path.join(bench, "animals", aid)
                os.makedirs(adir)
                bare_b = f"bare-{idx}".encode()
                truth_b = f"truth-{idx}".encode()
                with open(os.path.join(adir, "bare.glb"), "wb") as f:
                    f.write(bare_b)
                with open(os.path.join(adir, "truth.glb"), "wb") as f:
                    f.write(truth_b)
                animals.append(
                    {
                        "id": aid,
                        "license": "CC0-1.0",
                        "author": "TestAuthor",
                        "kind": kind,
                        "triangles": 16000 if idx == 0 else 2000,
                        "bare_sha256": sha(bare_b),
                        "truth_sha256": sha(truth_b),
                    }
                )
                tsv_lines.append(f"{aid}\t1\n")

            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"in_sample_overlap": 0, "animals": animals}, f)
            with open(os.path.join(bench, "animals.tsv"), "w", encoding="utf-8") as f:
                f.writelines(tsv_lines)

            # Missing creature-rig / weights binaries causes score_artists.py to fail (exit 2).
            # heldout.sh MUST fail non-zero rather than swallowing score_artists failure via `|| true`.
            fake_cb = os.path.join(tmp, "creature-bench")
            _write_exe(fake_cb, "#!/usr/bin/env python3\nimport sys; sys.exit(2)\n")
            env = dict(os.environ)
            env["CREATURE_BENCH_BIN"] = fake_cb
            env["CREATURE_RIG_BIN"] = os.path.join(tmp, "missing-creature-rig")
            env["WEIGHTS_BIN"] = os.path.join(tmp, "missing-weights")

            proc = subprocess.run(
                [
                    "bash",
                    os.path.join(REPO_DIR, "tools", "bench", "heldout.sh"),
                    "--bench",
                    bench,
                    "--manifest",
                    manifest,
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(
                proc.returncode,
                0,
                f"heldout.sh exited 0 despite score_artists/creature-bench failures!\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
