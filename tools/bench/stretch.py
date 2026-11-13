#!/usr/bin/env python3
"""A bench animal with other proportions (stdlib only): a probe for
whether a learned rigger reads the body or remembers the model.

    python3 tools/bench/stretch.py ANIMAL_DIR OUT_DIR [--scale 1.0,1.35,0.8]

Scales `bare.glb` and `truth.glb` along the glTF axes (x across, y up, z
along the body) by a node scale on the mesh and on the rig's top node.
The stretched artist's joints are where a rig of the stretched mesh
should have its joints, so `creature-bench score` still reads the
skeleton. The clips are no longer a truth for the skin (a rotation under
an uneven scale shears): only the joint numbers of such a run mean
anything.
"""
import json
import os
import struct
import sys


def read(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n:]


def write(path, j, rest):
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + len(rest)) + struct.pack("<II", len(js), 0x4E4F534A) + js + rest
    open(path, "wb").write(out)


def main():
    a = sys.argv[1:]
    scale = [1.0, 1.35, 0.8]
    if "--scale" in a:
        k = a.index("--scale")
        scale = [float(x) for x in a[k + 1].split(",")]
        del a[k:k + 2]
    src, dst = a
    os.makedirs(dst, exist_ok=True)
    for name in ("bare.glb", "truth.glb"):
        j, rest = read(os.path.join(src, name))
        tops = j["scenes"][j.get("scene", 0)]["nodes"]
        for t in tops:
            nd = j["nodes"][t]
            if "matrix" in nd:
                raise SystemExit(f"{name}: top node is a matrix")
            r = nd.get("rotation", [0, 0, 0, 1])
            if abs(abs(r[3]) - 1.0) > 1e-4:
                raise SystemExit(f"{name}: top node is turned ({r}); this probe wants an animal that was not turned")
            s = nd.get("scale", [1, 1, 1])
            nd["scale"] = [s[i] * scale[i] for i in range(3)]
            tr = nd.get("translation")
            if tr:
                nd["translation"] = [tr[i] * scale[i] for i in range(3)]
        write(os.path.join(dst, name), j, rest)
    with open(os.path.join(dst, "stretch.json"), "w") as f:
        json.dump({"from": os.path.abspath(src), "scale_xyz": scale}, f)
    print(f"{dst}: stretched by {scale}")


main()
