#!/usr/bin/env python3
"""Run upstream UniRig (SDPA, no flash-attention) as an outside rival on `creature-bench`
(`skintokens-8` / `unirig-mac-5`, stdlib only).

    python3 tools/bench/run_unirig_rival.py BENCH_DIR \\
        --unirig-dir /data/f/Games/_blender/unirig-mac \\
        --unirig-python /data/rigwork/venv-unirig/bin/python \\
        --creature-bench /home/f/rigwork/bin_current/creature-bench \\
        [--manifest PATH] [--seed 42] [--out-json OUT.json] [--label insample]

Runs UniRig skeleton + skin + merge on each animal's `bare.glb` (caching output at
`BENCH_DIR/animals/<id>/unirig/<id>.glb`), scores every rigged GLB against `truth.glb`
with `creature-bench score`, and prints aggregate medians + per-set JSON.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

EXTRACT_TIME = "2026-11-20"


def find_one(root: str, filename: str):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def write_extract_config(work: str, out_dir: str) -> str:
    cfg = os.path.join(work, "extract_cfg.yaml")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(f"input_dataset_dir: {work}\noutput_dataset_dir: {out_dir}\n")
    return cfg


def write_gpu_sdpa_tasks(unirig_dir: str, tmp_dir: str) -> tuple:
    """Create GPU-SDPA skeleton task and CPU-compat skin task under unirig_dir/configs/task/.
    Wait: run.py loads `--task=configs/task/<name>` relative to unirig_dir, or an absolute/relative yaml path."""
    skel_cfg = os.path.join(tmp_dir, "skel_gpu_sdpa.yaml")
    with open(skel_cfg, "w", encoding="utf-8") as f:
        f.write(
            "mode: predict\n"
            "debug: False\n"
            "experiment_name: quick_inference_skeleton_articulationxl_ar_256_gpu_sdpa\n"
            "resume_from_checkpoint: experiments/skeleton/articulation-xl_quantization_256/model.ckpt\n"
            "components:\n"
            "  data: quick_inference\n"
            "  tokenizer: tokenizer_parts_articulationxl_256\n"
            "  transform: inference_ar_transform\n"
            "  model: unirig_ar_350m_1024_81920_float32_mac\n"
            "  system: ar_inference_articulationxl\n"
            "  data_name: raw_data.npz\n"
            "writer:\n"
            "  __target__: ar\n"
            "  output_dir: ~\n"
            "  add_num: False\n"
            "  repeat: 1\n"
            "  export_npz: predict_skeleton\n"
            "  export_obj: skeleton\n"
            "  export_fbx: skeleton\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "  num_nodes: 1\n"
            "  devices: 1\n"
            "  precision: 16-mixed\n"
            "  accelerator: gpu\n"
            "  strategy: auto\n"
        )
    skin_cfg = os.path.join(tmp_dir, "skin_compat.yaml")
    with open(skin_cfg, "w", encoding="utf-8") as f:
        f.write(
            "mode: predict\n"
            "debug: False\n"
            "experiment_name: quick_inference_skin_compat\n"
            "resume_from_checkpoint: experiments/skin/articulation-xl/model.ckpt\n"
            "components:\n"
            "  data: quick_inference\n"
            "  transform: inference_skin_transform_mac\n"
            "  model: unirig_skin_mac\n"
            "  system: skin\n"
            "  data_name: predict_skeleton.npz\n"
            "writer:\n"
            "  __target__: skin\n"
            "  output_dir: results\n"
            "  add_num: False\n"
            "  repeat: 1\n"
            "  save_name: predict\n"
            "  export_npz: predict_skin\n"
            "  export_fbx: result_fbx\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "  num_nodes: 1\n"
            "  devices: 1\n"
            "  precision: 32-true\n"
            "  accelerator: cpu\n"
            "  strategy: auto\n"
            "  inference_mode: True\n"
        )
    return skel_cfg, skin_cfg


def run_stage(cmd: list, cwd: str, log_path: str, timeout: int = 1200) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            r = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def run_unirig_one(py: str, unirig_dir: str, bare_glb: str, out_dir: str, aid: str,
                   skel_task: str, skin_task: str, seed: int, redo: bool = False) -> tuple:
    rigged_glb = os.path.join(out_dir, f"{aid}.glb")
    if not redo and os.path.isfile(rigged_glb):
        return True, rigged_glb, "cached"

    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "work")
    mesh_npz = os.path.join(work, "mesh_npz")
    skin_npz = os.path.join(work, "skin_npz")
    os.makedirs(work, exist_ok=True)
    skel_fbx = os.path.join(out_dir, f"{aid}_skeleton.fbx")
    skin_fbx = os.path.join(out_dir, f"{aid}_skin.fbx")

    t0 = time.time()
    # 1. Extract mesh -> npz
    rc = run_stage(
        [py, "-m", "src.data.extract", "--config", write_extract_config(work, mesh_npz),
         "--require_suffix", "glb,gltf,fbx,obj,dae", "--faces_target_count", "50000",
         "--num_runs", "1", "--force_override", "True", "--id", "0", "--time", EXTRACT_TIME,
         "--input", bare_glb, "--output_dir", mesh_npz],
        unirig_dir, os.path.join(work, "1_extract_mesh.log"),
    )
    if rc != 0 or find_one(mesh_npz, "raw_data.npz") is None:
        return False, rigged_glb, f"extract_mesh failed (exit {rc})"

    # 2. Skeleton prediction (GPU SDPA, fallback to CPU if GPU fails)
    rc = run_stage(
        [py, "run.py", f"--task={skel_task}", "--input", bare_glb,
         "--npz_dir", mesh_npz, "--seed", str(seed), "--output", skel_fbx],
        unirig_dir, os.path.join(work, "2_skeleton.log"),
    )
    if rc != 0 or not os.path.isfile(skel_fbx):
        # Fallback to mac CPU config if CUDA hits precision/OOM issue
        rc = run_stage(
            [py, "run.py", "--task=configs/task/quick_inference_skeleton_articulationxl_ar_256_mac",
             "--input", bare_glb, "--npz_dir", mesh_npz, "--seed", str(seed), "--output", skel_fbx],
            unirig_dir, os.path.join(work, "2b_skeleton_cpu.log"),
        )
    if rc != 0 or not os.path.isfile(skel_fbx):
        return False, rigged_glb, f"skeleton failed (exit {rc})"

    # 3. Extract skeleton FBX -> npz
    rc = run_stage(
        [py, "-m", "src.data.extract", "--config", write_extract_config(work, skin_npz),
         "--require_suffix", "fbx", "--faces_target_count", "50000",
         "--num_runs", "1", "--force_override", "True", "--id", "0", "--time", EXTRACT_TIME,
         "--input", skel_fbx, "--output_dir", skin_npz],
        unirig_dir, os.path.join(work, "3_extract_skel.log"),
    )
    if rc != 0 or find_one(skin_npz, "raw_data.npz") is None:
        return False, rigged_glb, f"extract_skel failed (exit {rc})"

    # 4. Skin prediction
    rc = run_stage(
        [py, "run.py", f"--task={skin_task}", "--input", skel_fbx,
         "--npz_dir", skin_npz, "--data_name", "raw_data.npz", "--output", skin_fbx],
        unirig_dir, os.path.join(work, "4_skin.log"),
    )
    if rc != 0 or not os.path.isfile(skin_fbx):
        return False, rigged_glb, f"skin failed (exit {rc})"

    # 5. Merge into rigged GLB
    rc = run_stage(
        [py, "-m", "src.inference.merge", "--require_suffix", "fbx",
         "--num_runs", "1", "--id", "0", "--source", skin_fbx,
         "--target", bare_glb, "--output", rigged_glb],
        unirig_dir, os.path.join(work, "5_merge.log"),
    )
    if rc != 0 or not os.path.isfile(rigged_glb):
        return False, rigged_glb, f"merge failed (exit {rc})"

    return True, rigged_glb, f"ok in {time.time() - t0:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench_dir")
    ap.add_argument("--unirig-dir", default="/data/f/Games/_blender/unirig-mac")
    ap.add_argument("--unirig-python", default="/data/rigwork/venv-unirig/bin/python")
    ap.add_argument("--creature-bench", default=os.environ.get("CREATURE_BENCH_BIN", "/home/f/rigwork/bin_current/creature-bench"))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    if a.manifest:
        ids = [x["id"] for x in json.load(open(a.manifest))["animals"]]
    else:
        ids = [l.split("\t")[0] for l in open(os.path.join(a.bench_dir, "animals.tsv")) if l.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        skel_task, skin_task = write_gpu_sdpa_tasks(a.unirig_dir, tmp)
        # Strip .yaml suffix because run.py appends .yaml if missing (or handles both)
        skel_arg = skel_task[:-5] if skel_task.endswith(".yaml") else skel_task
        skin_arg = skin_task[:-5] if skin_task.endswith(".yaml") else skin_task

        scored = []
        failed = []

        for idx, aid in enumerate(ids, 1):
            adir = os.path.join(a.bench_dir, "animals", aid)
            bare = os.path.join(adir, "bare.glb")
            truth = os.path.join(adir, "truth.glb")
            udir = os.path.join(adir, "unirig")
            ok, glb, msg = run_unirig_one(
                a.unirig_python, a.unirig_dir, bare, udir, aid, skel_arg, skin_arg, a.seed, a.redo
            )
            tag_label = f"animal_{idx}" if a.manifest else aid
            if not ok:
                print(f"[{a.label}] {tag_label}: FAIL ({msg})", flush=True)
                failed.append({"id": tag_label, "reason": msg})
                continue

            sj = os.path.join(udir, "score.json")
            cmd = [a.creature_bench, "score", truth, glb, "--json", sj]
            roles = os.path.join(adir, "roles.json")
            if os.path.isfile(roles):
                cmd += ["--roles", roles]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0 or not os.path.isfile(sj):
                err = (r.stderr or r.stdout or "").strip()[-200:]
                print(f"[{a.label}] {tag_label}: score FAIL ({err})", flush=True)
                failed.append({"id": tag_label, "reason": f"score failed: {err}"})
                continue
            d = json.load(open(sj))
            rec = {
                "id": tag_label,
                "joints_pct": d["skeleton"]["joint_error_pct"],
                "missing_chains": len(d["skeleton"]["missing_chains"]),
                "skin_mean_pct": d["skin"]["mean_pct"],
                "skin_p95_pct": d["skin"]["p95_pct"],
                "weights_only_pct": (d.get("skin_weights_only") or d.get("weights_only") or {}).get("mean_pct"),
            }
            scored.append(rec)
            print(
                f"[{a.label}] {tag_label}: {msg} | joints={rec['joints_pct']:.2f}% "
                f"miss={rec['missing_chains']} skin={rec['skin_mean_pct']:.2f}% "
                f"p95={rec['skin_p95_pct']:.2f}%",
                flush=True,
            )

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None

    summary = {
        "label": a.label,
        "rival": "UniRig (VAST-AI/UniRig, SDPA)",
        "seed": a.seed,
        "total": len(ids),
        "succeeded": len(scored),
        "failed_count": len(failed),
        "medians": {
            "joints_median_pct": med([x["joints_pct"] for x in scored]),
            "missing_chain_animals": sum(1 for x in scored if x["missing_chains"] > 0),
            "skin_mean_median_pct": med([x["skin_mean_pct"] for x in scored]),
            "skin_p95_median_pct": med([x["skin_p95_pct"] for x in scored]),
            "weights_only_median_pct": med([x["weights_only_pct"] for x in scored]),
        },
        "failed": failed,
    }
    if not a.manifest:
        summary["per_animal"] = scored

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"=== {a.label} UniRig summary ===")
    print(json.dumps(summary["medians"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
