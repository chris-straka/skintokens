#!/usr/bin/env python3
"""Download, license-verify, and prepare the 12 frozen held-out animals (`skintokens-1`).

    python3 tools/bench/fetch_heldout.py OUT_DIR [--blender PATH] [--insample-tsv PATH] [--manifest-out PATH]

All 12 animals come from independent non-Quaternius authors with verified CC0-1.0 or CC-BY-4.0
licenses, covering winged, hunched, long-armed, many-legged, legless, quadruped, and dense
(15k-52k triangle) meshes, with zero overlap against the 12 in-sample benchmark animals.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

M2M_RAW = "https://raw.githubusercontent.com/Mesh2Motion/mesh2motion-app/main/static/assets"
M2M_LICENSE_URL = "https://github.com/Mesh2Motion/mesh2motion-assets/blob/main/LICENSE"
BABYLON_RAW = "https://raw.githubusercontent.com/BabylonJS/Assets/master/meshes"
BABYLON_LICENSE_URL = "https://github.com/BabylonJS/Assets/blob/master/LICENSE"

# Frozen 12 held-out animals: zero Quaternius, zero Khronos Fox, zero overlap with in-sample 12.
ANIMALS = [
    {
        "id": "bee_dense",
        "kind": "winged, many-legged, dense",
        "author": "BabylonJS/Assets",
        "license": "CC-BY-4.0",
        "license_url": BABYLON_LICENSE_URL,
        "model_url": f"{BABYLON_RAW}/Bee.glb",
        "clip_urls": [],
    },
    {
        "id": "octopus_dense",
        "kind": "many-legged, dense",
        "author": "BabylonJS/Assets",
        "license": "CC-BY-4.0",
        "license_url": BABYLON_LICENSE_URL,
        "model_url": f"{BABYLON_RAW}/octopus_customRig.glb",
        "clip_urls": [],
    },
    {
        "id": "shark_dense",
        "kind": "legless, dense",
        "author": "BabylonJS/Assets",
        "license": "CC-BY-4.0",
        "license_url": BABYLON_LICENSE_URL,
        "model_url": f"{BABYLON_RAW}/shark.glb",
        "clip_urls": [],
    },
    {
        "id": "eagle",
        "kind": "winged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/bird/variations/eagle.glb",
        "clip_urls": [
            f"{M2M_RAW}/bird/animations/flap.glb",
            f"{M2M_RAW}/bird/animations/glide.glb",
            f"{M2M_RAW}/bird/animations/walk.glb",
            f"{M2M_RAW}/bird/animations/idle.glb",
        ],
    },
    {
        "id": "m2m_dragon",
        "kind": "winged, four-legged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/dragon/model.glb",
        "clip_urls": [
            f"{M2M_RAW}/dragon/animations/fly-flap.glb",
            f"{M2M_RAW}/dragon/animations/walk.glb",
            f"{M2M_RAW}/dragon/animations/attack.glb",
            f"{M2M_RAW}/dragon/animations/roar.glb",
        ],
    },
    {
        "id": "kaiju",
        "kind": "hunched, long-armed",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/kaiju/model.glb",
        "clip_urls": [
            f"{M2M_RAW}/kaiju/animations/walk.glb",
            f"{M2M_RAW}/kaiju/animations/attack.glb",
            f"{M2M_RAW}/kaiju/animations/roar.glb",
            f"{M2M_RAW}/kaiju/animations/tail-attack.glb",
        ],
    },
    {
        "id": "m2m_trex",
        "kind": "hunched, two-legged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/kaiju/variations/trex.glb",
        "clip_urls": [
            f"{M2M_RAW}/kaiju/animations/walk.glb",
            f"{M2M_RAW}/kaiju/animations/roar.glb",
            f"{M2M_RAW}/kaiju/animations/tail-attack.glb",
        ],
    },
    {
        "id": "m2m_spider",
        "kind": "many-legged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/spider/model.glb",
        "clip_urls": [
            f"{M2M_RAW}/spider/animations/walk.glb",
            f"{M2M_RAW}/spider/animations/attack.glb",
            f"{M2M_RAW}/spider/animations/jump.glb",
            f"{M2M_RAW}/spider/animations/idle.glb",
        ],
    },
    {
        "id": "m2m_snake",
        "kind": "legless",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/snake/model.glb",
        "clip_urls": [
            f"{M2M_RAW}/snake/animations/sidewinding.glb",
            f"{M2M_RAW}/snake/animations/bite.glb",
            f"{M2M_RAW}/snake/animations/coiled.glb",
            f"{M2M_RAW}/snake/animations/idle.glb",
        ],
    },
    {
        "id": "dog",
        "kind": "four-legged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/fox/variations/dog.glb",
        "clip_urls": [
            f"{M2M_RAW}/fox/animations/walk.glb",
            f"{M2M_RAW}/fox/animations/trot.glb",
            f"{M2M_RAW}/fox/animations/run.glb",
            f"{M2M_RAW}/fox/animations/jump.glb",
        ],
    },
    {
        "id": "panda",
        "kind": "hunched, four-legged",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/fox/variations/panda.glb",
        "clip_urls": [
            f"{M2M_RAW}/fox/animations/walk.glb",
            f"{M2M_RAW}/fox/animations/trot.glb",
            f"{M2M_RAW}/fox/animations/run.glb",
            f"{M2M_RAW}/fox/animations/sit.glb",
        ],
    },
    {
        "id": "whale",
        "kind": "legless",
        "author": "Mesh2Motion",
        "license": "CC0-1.0",
        "license_url": M2M_LICENSE_URL,
        "model_url": f"{M2M_RAW}/shark/variations/whale.glb",
        "clip_urls": [
            f"{M2M_RAW}/shark/animations/swim-horizontal.glb",
            f"{M2M_RAW}/shark/animations/swim-vertical.glb",
            f"{M2M_RAW}/shark/animations/idle.glb",
        ],
    },
]

IN_SAMPLE_IDS = {
    "horse", "pug", "cow", "fox", "trex", "velociraptor",
    "apatosaurus", "rat", "dragon", "bat", "spider", "wasp",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_glb(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(20)
        if header[:4] != b"glTF":
            raise ValueError(f"not a GLB file: {path}")
        jlen = struct.unpack_from("<I", header, 12)[0]
        g = json.loads(f.read(jlen).decode("utf-8"))
    tris = 0
    verts = 0
    for m in g.get("meshes", []):
        for p in m.get("primitives", []):
            if "indices" in p:
                tris += g["accessors"][p["indices"]]["count"] // 3
            pos = p.get("attributes", {}).get("POSITION")
            if pos is not None:
                verts += g["accessors"][pos]["count"]
    skins = [len(s.get("joints", [])) for s in g.get("skins", [])]
    anims = [a.get("name", f"anim_{i}") for i, a in enumerate(g.get("animations", []))]
    return {"tris": tris, "verts": verts, "skins": skins, "anims": anims}


def download(url: str, dest: str) -> None:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "skintokens-heldout/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    os.replace(tmp, dest)


def run_blender(blender: str, script: str, args: list, log_path: str) -> None:
    cmd = [blender, "-b", "--factory-startup", "-P", script, "--"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(r.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(r.stderr or "")
    if r.returncode != 0:
        tail = ((r.stderr or "") + "\n" + (r.stdout or "")).strip()[-600:]
        raise RuntimeError(f"Blender failed (exit {r.returncode}) for {os.path.basename(script)}: {tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--blender", default=os.environ.get("BLENDER") or shutil.which("blender") or "")
    ap.add_argument("--insample-bench", default="/home/f/rigwork/bench")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    if not a.blender or not os.path.exists(a.blender):
        print("fetch_heldout: set --blender or BLENDER", file=sys.stderr)
        return 2

    out_dir = os.path.abspath(a.out_dir)
    src_root = os.path.join(out_dir, "src")
    animals_root = os.path.join(out_dir, "animals")
    os.makedirs(src_root, exist_ok=True)
    os.makedirs(animals_root, exist_ok=True)

    # Collect in-sample hashes to prove zero overlap by content as well as ID.
    insample_hashes = set()
    if os.path.isdir(os.path.join(a.insample_bench, "animals")):
        for entry in sorted(os.listdir(os.path.join(a.insample_bench, "animals"))):
            for fn in ("bare.glb", "truth.glb"):
                p = os.path.join(a.insample_bench, "animals", entry, fn)
                if os.path.isfile(p):
                    insample_hashes.add(sha256_file(p))

    manifest_animals = []
    tsv_lines = []

    for spec in ANIMALS:
        aid = spec["id"]
        if aid in IN_SAMPLE_IDS:
            raise RuntimeError(f"overlap with in-sample id: {aid}")
        sdir = os.path.join(src_root, aid)
        adir = os.path.join(animals_root, aid)
        os.makedirs(sdir, exist_ok=True)
        os.makedirs(adir, exist_ok=True)

        model_file = os.path.join(sdir, os.path.basename(spec["model_url"]))
        print(f"[{aid}] downloading {spec['model_url']} ...", flush=True)
        download(spec["model_url"], model_file)

        clip_files = []
        for curl in spec["clip_urls"]:
            cf = os.path.join(sdir, "clips", os.path.basename(curl))
            download(curl, cf)
            clip_files.append(cf)

        prep_src = model_file
        if clip_files:
            combined = os.path.join(sdir, f"{aid}_combined.glb")
            if a.redo or not os.path.isfile(combined):
                print(f"[{aid}] merging {len(clip_files)} clips ...", flush=True)
                run_blender(
                    a.blender,
                    os.path.join(HERE, "merge_clips.py"),
                    [model_file, combined] + clip_files,
                    os.path.join(sdir, "merge.log"),
                )
            prep_src = combined

        truth_glb = os.path.join(adir, "truth.glb")
        bare_glb = os.path.join(adir, "bare.glb")
        if a.redo or not (os.path.isfile(truth_glb) and os.path.isfile(bare_glb)):
            print(f"[{aid}] running prep_truth.py ...", flush=True)
            run_blender(
                a.blender,
                os.path.join(HERE, "prep_truth.py"),
                [prep_src, adir, "--name", aid],
                os.path.join(adir, "prep.log"),
            )

        t_info = inspect_glb(truth_glb)
        b_info = inspect_glb(bare_glb)
        if not t_info["skins"] or t_info["skins"][0] < 4:
            raise RuntimeError(f"{aid}: truth.glb missing valid skin: {t_info}")
        if not t_info["anims"]:
            raise RuntimeError(f"{aid}: truth.glb has 0 animations: {t_info}")
        if b_info["skins"]:
            raise RuntimeError(f"{aid}: bare.glb unexpectedly contains skins: {b_info}")

        bare_sha = sha256_file(bare_glb)
        truth_sha = sha256_file(truth_glb)
        src_sha = sha256_file(model_file)
        if bare_sha in insample_hashes or truth_sha in insample_hashes:
            raise RuntimeError(f"{aid}: SHA-256 overlaps with in-sample benchmark!")

        entry = {
            "id": aid,
            "kind": spec["kind"],
            "author": spec["author"],
            "license": spec["license"],
            "license_url": spec["license_url"],
            "source_url": spec["model_url"],
            "clip_urls": spec["clip_urls"],
            "triangles": b_info["tris"],
            "vertices": b_info["verts"],
            "joints": t_info["skins"][0],
            "clips": len(t_info["anims"]),
            "source_sha256": src_sha,
            "bare_sha256": bare_sha,
            "truth_sha256": truth_sha,
        }
        manifest_animals.append(entry)
        tsv_lines.append(f"{aid}\t{spec['kind']}\t{spec['author']}/{os.path.basename(spec['model_url'])}\n")
        print(
            f"[{aid}] OK tris={entry['triangles']} joints={entry['joints']} "
            f"clips={entry['clips']} bare={bare_sha[:12]} truth={truth_sha[:12]}",
            flush=True,
        )

    with open(os.path.join(out_dir, "animals.tsv"), "w", encoding="utf-8") as f:
        f.writelines(tsv_lines)

    manifest = {
        "version": 1,
        "frozen_at": datetime.date.today().isoformat(),
        "count": len(manifest_animals),
        "in_sample_overlap": 0,
        "in_sample_ids_excluded": sorted(IN_SAMPLE_IDS),
        "note": (
            "Frozen 12-animal held-out benchmark set for skintokens-1. Zero overlap with the "
            "12 in-sample animals (Quaternius/Khronos Fox). All models are CC0-1.0 or CC-BY-4.0 "
            "from independent authors (BabylonJS/Assets and Mesh2Motion). Files stay outside the "
            "repo on f (/home/f/rigwork/bench_heldout)."
        ),
        "animals": manifest_animals,
    }

    manifest_dest = a.manifest_out or os.path.join(out_dir, "heldout_manifest.json")
    with open(manifest_dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    fleet_art = os.environ.get("FLEET_ARTIFACTS")
    if fleet_art and os.path.isdir(fleet_art):
        shutil.copyfile(manifest_dest, os.path.join(fleet_art, "heldout_manifest.json"))
        shutil.copyfile(os.path.join(out_dir, "animals.tsv"), os.path.join(fleet_art, "animals.tsv"))

    print(f"DONE: prepared and verified {len(manifest_animals)} held-out animals -> {manifest_dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
