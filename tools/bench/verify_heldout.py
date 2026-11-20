#!/usr/bin/env python3
"""Verify the frozen `skintokens-1` held-out animal manifest and workspace (stdlib only).

    python3 tools/bench/verify_heldout.py [--manifest PATH] [--bench DIR]

Checks:
- manifest contains >= 12 unique animals;
- every animal has a verified CC0 or CC-BY license and 64-hex SHA-256 checksums;
- zero ID overlap with the 12 in-sample benchmark animals;
- body-plan coverage includes winged, hunched, long-armed, many-legged, legless,
  and dense (>= 15,000 triangles) meshes;
- when `--bench DIR` is passed, every `animals/<id>/bare.glb` and `truth.glb`
  matches its frozen SHA-256 exactly.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "heldout_manifest.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY"}
REQUIRED_KINDS = ("winged", "hunched", "long-armed", "many-legged", "legless")
MIN_ANIMALS = 12
DENSE_TRIS = 15000

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


def validate_manifest(path: str) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"held-out manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    animals = doc.get("animals")
    if not isinstance(animals, list) or len(animals) < MIN_ANIMALS:
        raise ValueError(
            f"held-out manifest must list at least {MIN_ANIMALS} animals, "
            f"got {len(animals) if isinstance(animals, list) else type(animals).__name__}"
        )

    seen_ids = set()
    seen_bare = set()
    seen_truth = set()
    covered_kinds = set()
    dense_count = 0

    for a in animals:
        aid = a.get("id", "")
        if not aid or aid in seen_ids:
            raise ValueError(f"missing or duplicate animal id: {aid!r}")
        if aid in IN_SAMPLE_IDS:
            raise ValueError(f"held-out animal overlaps in-sample set: {aid!r}")
        seen_ids.add(aid)

        lic = a.get("license", "")
        if lic not in ALLOWED_LICENSES:
            raise ValueError(f"{aid}: unsupported license {lic!r} (allowed: {sorted(ALLOWED_LICENSES)})")
        if not a.get("author") or a["author"].lower() == "quaternius":
            raise ValueError(f"{aid}: held-out animals must come from non-Quaternius authors, got {a.get('author')!r}")

        for field, seen_set in (("bare_sha256", seen_bare), ("truth_sha256", seen_truth)):
            h = (a.get(field) or "").lower()
            if not SHA256_RE.match(h):
                raise ValueError(f"{aid}: invalid {field}: {h!r}")
            if h in seen_set:
                raise ValueError(f"{aid}: duplicate {field}: {h[:12]}")
            seen_set.add(h)

        kind = (a.get("kind") or "").lower()
        for req in REQUIRED_KINDS:
            if req in kind:
                covered_kinds.add(req)
        if int(a.get("triangles") or 0) >= DENSE_TRIS:
            dense_count += 1

    missing_kinds = [k for k in REQUIRED_KINDS if k not in covered_kinds]
    if missing_kinds:
        raise ValueError(f"held-out manifest missing required body plans: {missing_kinds}")
    if dense_count < 1:
        raise ValueError(f"held-out manifest must include at least one dense (>={DENSE_TRIS} tris) mesh")

    return doc


def verify_workspace(bench_dir: str, manifest_path: str) -> dict:
    doc = validate_manifest(manifest_path)
    for a in doc["animals"]:
        aid = a["id"]
        adir = os.path.join(bench_dir, "animals", aid)
        bare = os.path.join(adir, "bare.glb")
        truth = os.path.join(adir, "truth.glb")
        for label, p, exp in (("bare.glb", bare, a["bare_sha256"]), ("truth.glb", truth, a["truth_sha256"])):
            if not os.path.isfile(p):
                raise ValueError(f"{aid}: missing {label} at {p}")
            got = sha256_file(p)
            if got != exp.lower():
                raise ValueError(f"{aid}: {label} sha256 mismatch (got {got[:12]}, expected {exp[:12]})")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--bench", default="")
    args = ap.parse_args()
    try:
        if args.bench:
            doc = verify_workspace(args.bench, args.manifest)
            print(f"OK verified {len(doc['animals'])} held-out animals in {args.bench} against {args.manifest}")
        else:
            doc = validate_manifest(args.manifest)
            print(f"OK validated held-out manifest ({len(doc['animals'])} animals): {args.manifest}")
    except ValueError as e:
        print(f"verify_heldout: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
