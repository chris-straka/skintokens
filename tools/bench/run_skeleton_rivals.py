#!/usr/bin/env python3
"""Run Apache-2.0 outside skeleton rivals (Puppeteer and/or MagicArticulate, SDPA without
flash-attention) on `creature-bench` (`skintokens-8`, stdlib only for GLB/scoring logic).

    python3 tools/bench/run_skeleton_rivals.py BENCH_DIR \\
        --rival puppeteer|magicarticulate|both \\
        --puppeteer-dir /data/rigwork/puppeteer \\
        --rival-python /data/rigwork/venv-puppeteer/bin/python \\
        --skintokens-bin /home/f/rigwork/bin_current/skintokens \\
        --creature-bench /home/f/rigwork/bin_current/creature-bench \\
        [--manifest PATH] [--seed 42] [--out-json OUT.json] [--label insample]

Licences & exclusions:
  - Puppeteer skeleton stage (`Seed3D/Puppeteer`, `puppeteer_skeleton_w_diverse_pose.pth`):
    Apache-2.0. Puppeteer's skinning stage requires NVIDIA PartField (non-commercial) and is
    explicitly excluded; never downloaded or invoked.
  - MagicArticulate skeleton stage (`Seed3D/MagicArticulate`, `checkpoint_trainonv2_hier.pth`):
    Apache-2.0. MagicArticulate has no released skinning stage.
  - Shape encoder (`Maikou/Michelangelo`, `shapevae-256.ckpt`): LGPL-3.0 weights / GPL-3.0 code
    (tool-side only; model outputs / rigged GLBs are unaffected).

For each animal's `bare.glb`, this script:
  1. Exports world-space glTF (+Y up) triangles to `.obj`.
  2. Runs the rival skeleton generator (batched once per set so the 4.6 GB checkpoint loads
     once) to produce `<id>_pred.txt`.
  3. Attaches the predicted skeleton directly to `bare.glb`:
     - `<id>_rigid.glb`: pure nearest-bone rigid weights (skeleton baseline).
     - `<id>_skin_in.glb`: same skeleton with a tiny 1e-4 weight floor on each joint's
       nearest vertex so `skintokens skin` keeps 100% of predicted joints.
     - `<id>.glb`: `<id>_skin_in.glb` re-skinned by `skintokens skin --seed SEED`
       (rival skeleton + SkinTokens ML skinning weights).
  4. Scores both `<id>.glb` (rival skeleton + SkinTokens skin) and `<id>_rigid.glb`
     (rival skeleton + rigid weights) against `truth.glb` with `creature-bench score`.
"""
import argparse
import collections
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

RIVAL_SPECS = {
    "puppeteer": {
        "display": "Puppeteer skeleton (Seed3D/Puppeteer, Apache-2.0, joint-token SDPA)",
        "ckpt_rel": "skeleton/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
        "demo_flags": ["--joint_token", "--seq_shuffle"],
        "subdir": "puppeteer",
    },
    "magicarticulate": {
        "display": "MagicArticulate skeleton (Seed3D/MagicArticulate, Apache-2.0, hier SDPA)",
        "ckpt_rel": "skeleton/skeleton_ckpts/checkpoint_trainonv2_hier.pth",
        "demo_flags": ["--hier_order"],
        "subdir": "magicarticulate",
    },
}


# ---------------------------------------------------------------------------
# Pure-Python 4x4 matrix & glTF 2.0 GLB helpers
# ---------------------------------------------------------------------------

def _mat4_identity() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def _mat4_apply_point(m: List[List[float]], p: Sequence[float]) -> Tuple[float, float, float]:
    x, y, z = p
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def _node_local_matrix(nd: dict) -> List[List[float]]:
    if "matrix" in nd:
        m = nd["matrix"]  # glTF column-major
        return [
            [float(m[0]), float(m[4]), float(m[8]), float(m[12])],
            [float(m[1]), float(m[5]), float(m[9]), float(m[13])],
            [float(m[2]), float(m[6]), float(m[10]), float(m[14])],
            [float(m[3]), float(m[7]), float(m[11]), float(m[15])],
        ]
    tx, ty, tz = nd.get("translation", [0.0, 0.0, 0.0])
    qx, qy, qz, qw = nd.get("rotation", [0.0, 0.0, 0.0, 1.0])
    sx, sy, sz = nd.get("scale", [1.0, 1.0, 1.0])
    # Rotation matrix from unit quaternion
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return [
        [r00 * sx, r01 * sy, r02 * sz, float(tx)],
        [r10 * sx, r11 * sy, r12 * sz, float(ty)],
        [r20 * sx, r21 * sy, r22 * sz, float(tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _compute_world_matrices(nodes: List[dict]) -> List[List[List[float]]]:
    n = len(nodes)
    parent = [None] * n
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parent[c] = i
    locals_m = [_node_local_matrix(nd) for nd in nodes]
    worlds: List[Optional[List[List[float]]]] = [None] * n

    def resolve(i: int) -> List[List[float]]:
        if worlds[i] is not None:
            return worlds[i]  # type: ignore[return-value]
        if parent[i] is None:
            worlds[i] = locals_m[i]
        else:
            worlds[i] = _mat4_mul(resolve(parent[i]), locals_m[i])  # type: ignore[arg-type]
        return worlds[i]  # type: ignore[return-value]

    for i in range(n):
        resolve(i)
    return worlds  # type: ignore[return-value]


def read_glb(path: str) -> Tuple[dict, bytearray]:
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError(f"not a glTF 2.0 GLB: {path}")
    off = 12
    gltf_json = None
    bin_buf = bytearray()
    while off + 8 <= length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == CHUNK_JSON:
            gltf_json = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN:
            bin_buf = bytearray(chunk)
    if gltf_json is None:
        raise ValueError(f"missing JSON chunk in {path}")
    return gltf_json, bin_buf


def write_glb(path: str, gltf_json: dict, bin_buf: bytearray) -> None:
    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)
    if bin_buf:
        if not gltf_json.get("buffers"):
            gltf_json["buffers"] = [{"byteLength": len(bin_buf)}]
        else:
            gltf_json["buffers"][0]["byteLength"] = len(bin_buf)
    jbytes = json.dumps(gltf_json, separators=(",", ":")).encode("utf-8")
    while len(jbytes) % 4 != 0:
        jbytes += b" "
    total = 12 + 8 + len(jbytes) + (8 + len(bin_buf) if bin_buf else 0)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(jbytes), CHUNK_JSON)
    out += jbytes
    if bin_buf:
        out += struct.pack("<II", len(bin_buf), CHUNK_BIN)
        out += bin_buf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)


_COMP_FMT = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_accessor(gltf: dict, bin_buf: bytearray, acc_idx: int) -> list:
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    fmt_char, csize = _COMP_FMT[acc["componentType"]]
    ncomp = _TYPE_N[acc["type"]]
    count = acc["count"]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", csize * ncomp)
    s = struct.Struct("<" + fmt_char * ncomp)
    out = []
    for i in range(count):
        vals = s.unpack_from(bin_buf, start + i * stride)
        out.append(vals[0] if ncomp == 1 else vals)
    return out


def _push_accessor(
    gltf: dict,
    bin_buf: bytearray,
    raw: bytes,
    comp_type: int,
    kind: str,
    count: int,
) -> int:
    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)
    offset = len(bin_buf)
    bin_buf.extend(raw)
    bvs = gltf.setdefault("bufferViews", [])
    bvs.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
    bv_idx = len(bvs) - 1
    accs = gltf.setdefault("accessors", [])
    accs.append({
        "bufferView": bv_idx,
        "componentType": comp_type,
        "count": count,
        "type": kind,
    })
    return len(accs) - 1


def glb_to_obj(bare_glb: str, out_obj: str) -> Tuple[int, int]:
    """Export world-space glTF (+Y up) triangles from `bare.glb` to `.obj`."""
    gltf, bin_buf = read_glb(bare_glb)
    nodes = gltf.get("nodes", [])
    worlds = _compute_world_matrices(nodes)
    meshes = gltf.get("meshes", [])

    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    for ni, nd in enumerate(nodes):
        mi = nd.get("mesh")
        if mi is None:
            continue
        mw = worlds[ni]
        for prim in meshes[mi].get("primitives", []):
            mode = prim.get("mode", 4)
            if mode != 4:
                continue
            pos_acc = prim["attributes"].get("POSITION")
            if pos_acc is None:
                continue
            base = len(verts)
            pos_list = _read_accessor(gltf, bin_buf, pos_acc)
            for p in pos_list:
                verts.append(_mat4_apply_point(mw, p))
            if "indices" in prim:
                idxs = _read_accessor(gltf, bin_buf, prim["indices"])
            else:
                idxs = list(range(len(pos_list)))
            for k in range(0, len(idxs) - 2, 3):
                a, b, c = idxs[k], idxs[k + 1], idxs[k + 2]
                if a != b and b != c and a != c:
                    faces.append((base + a + 1, base + b + 1, base + c + 1))

    os.makedirs(os.path.dirname(os.path.abspath(out_obj)), exist_ok=True)
    with open(out_obj, "w", encoding="utf-8") as f:
        for x, y, z in verts:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")
    return len(verts), len(faces)


# ---------------------------------------------------------------------------
# RigNet / Puppeteer / MagicArticulate `_pred.txt` parser & tree sanitizer
# ---------------------------------------------------------------------------

def parse_rig_txt(path: str) -> Tuple[List[str], List[Tuple[float, float, float]], List[int]]:
    """Parse RigNet-format `_pred.txt` (`joints`, `root`, `hier`) into a valid single-root tree.

    Returns `(names, positions, parents)` ordered in BFS/DFS topological order with `parents[0] == -1`
    and every `parents[i] < i` for `i > 0`.
    """
    raw_names: List[str] = []
    raw_pos: List[Tuple[float, float, float]] = []
    name_to_idx: Dict[str, int] = {}
    root_name: Optional[str] = None
    edges: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "joints" and len(parts) >= 5:
                name = parts[1]
                pos = (float(parts[2]), float(parts[3]), float(parts[4]))
                if name not in name_to_idx:
                    name_to_idx[name] = len(raw_names)
                    raw_names.append(name)
                    raw_pos.append(pos)
            elif tag == "root" and len(parts) >= 2:
                root_name = parts[1]
            elif tag == "hier" and len(parts) >= 3:
                p, c = parts[1], parts[2]
                if p in name_to_idx and c in name_to_idx and p != c:
                    edges.append((name_to_idx[p], name_to_idx[c]))

    n = len(raw_names)
    if n == 0:
        raise ValueError(f"no joints found in {path}")

    # Build directed + undirected adjacency
    directed_children: Dict[int, List[int]] = collections.defaultdict(list)
    undirected: Dict[int, List[int]] = collections.defaultdict(list)
    in_deg = [0] * n
    for p, c in edges:
        if c not in directed_children[p]:
            directed_children[p].append(c)
            in_deg[c] += 1
        if c not in undirected[p]:
            undirected[p].append(c)
        if p not in undirected[c]:
            undirected[c].append(p)

    if root_name is not None and root_name in name_to_idx:
        root_idx = name_to_idx[root_name]
    else:
        zeros = [i for i in range(n) if in_deg[i] == 0]
        if zeros:
            root_idx = zeros[0]
        else:
            cx = sum(p[0] for p in raw_pos) / n
            cy = sum(p[1] for p in raw_pos) / n
            cz = sum(p[2] for p in raw_pos) / n
            root_idx = min(
                range(n),
                key=lambda i: (raw_pos[i][0] - cx) ** 2 + (raw_pos[i][1] - cy) ** 2 + (raw_pos[i][2] - cz) ** 2,
            )

    # Traverse from root_idx using directed edges first, then undirected edges to break cycles cleanly
    order: List[int] = [root_idx]
    parent_old: Dict[int, int] = {root_idx: -1}
    visited = {root_idx}
    q = collections.deque([root_idx])

    while q:
        u = q.popleft()
        for v in directed_children[u]:
            if v not in visited:
                visited.add(v)
                parent_old[v] = u
                order.append(v)
                q.append(v)
        for v in undirected[u]:
            if v not in visited:
                visited.add(v)
                parent_old[v] = u
                order.append(v)
                q.append(v)

    # Attach any disconnected components to their nearest already-visited joint
    while len(visited) < n:
        best_d2 = float("inf")
        best_u, best_v = root_idx, -1
        for v in range(n):
            if v in visited:
                continue
            vx, vy, vz = raw_pos[v]
            for u in visited:
                ux, uy, uz = raw_pos[u]
                d2 = (vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_u, best_v = u, v
        visited.add(best_v)
        parent_old[best_v] = best_u
        order.append(best_v)
        q2 = collections.deque([best_v])
        while q2:
            u = q2.popleft()
            for v in directed_children[u] + undirected[u]:
                if v not in visited:
                    visited.add(v)
                    parent_old[v] = u
                    order.append(v)
                    q2.append(v)

    old_to_new = {old: new for new, old in enumerate(order)}
    names = [raw_names[old] for old in order]
    positions = [raw_pos[old] for old in order]
    parents = [-1 if parent_old[old] < 0 else old_to_new[parent_old[old]] for old in order]
    return names, positions, parents


def _dist2_point_seg(
    p: Tuple[float, float, float],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 <= 1e-18:
        return apx * apx + apy * apy + apz * apz
    t = max(0.0, min(1.0, (apx * abx + apy * aby + apz * abz) / ab2))
    dx = apx - t * abx
    dy = apy - t * aby
    dz = apz - t * abz
    return dx * dx + dy * dy + dz * dz


def _compute_rigid_weights(
    verts: Sequence[Tuple[float, float, float]],
    positions: Sequence[Tuple[float, float, float]],
    parents: Sequence[int],
    min_joint_weight: float = 0.0,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[float, float, float, float]]]:
    """Nearest-bone rigid skin weights (matching `creature-bench rigid` bone-segment proximity).

    When `min_joint_weight > 0`, ensures every joint holds at least `min_joint_weight` on its
    closest vertex so `skintokens skin` keeps 100% of predicted joints without trimming.
    """
    nj = len(positions)
    children: List[List[int]] = [[] for _ in range(nj)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    # Each joint `j` is represented by its outgoing segments `(j -> child)`, or its head point for leaves.
    segs: List[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = []
    for j in range(nj):
        if children[j]:
            segs.append([(positions[j], positions[c]) for c in children[j]])
        else:
            segs.append([(positions[j], positions[j])])

    joints_out: List[List[int]] = []
    weights_out: List[List[float]] = []
    held = [0.0] * nj

    for v in verts:
        best_j = 0
        best_d2 = float("inf")
        for j in range(nj):
            for a, b in segs[j]:
                d2 = _dist2_point_seg(v, a, b)
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
        joints_out.append([best_j, 0, 0, 0])
        weights_out.append([1.0, 0.0, 0.0, 0.0])
        held[best_j] += 1.0

    if min_joint_weight > 0.0 and verts:
        eps = min(0.05, max(1e-4, min_joint_weight))
        for j in range(nj):
            if held[j] > 1e-6:
                continue
            jx, jy, jz = positions[j]
            nearest_v = min(
                range(len(verts)),
                key=lambda vi: (verts[vi][0] - jx) ** 2 + (verts[vi][1] - jy) ** 2 + (verts[vi][2] - jz) ** 2,
            )
            # Add j in slot 1 (or overwrite smallest non-dominant slot)
            js = joints_out[nearest_v]
            ws = weights_out[nearest_v]
            if j in js:
                continue
            placed = False
            for s in range(1, 4):
                if ws[s] == 0.0:
                    js[s] = j
                    ws[s] = eps
                    ws[0] = max(0.0, ws[0] - eps)
                    placed = True
                    break
            if not placed:
                js[3] = j
                ws[3] = eps
                tot = sum(ws)
                for s in range(4):
                    ws[s] /= tot
            held[j] += eps

    return (
        [(a, b, c, d) for a, b, c, d in joints_out],
        [(a, b, c, d) for a, b, c, d in weights_out],
    )


def attach_skeleton_to_glb(
    bare_glb: str,
    out_glb: str,
    names: Sequence[str],
    positions: Sequence[Tuple[float, float, float]],
    parents: Sequence[int],
    min_joint_weight: float = 0.0,
) -> None:
    """Attach predicted world-space skeleton + rigid weights onto `bare.glb` and write `out_glb`."""
    gltf, bin_buf = read_glb(bare_glb)
    # Strip any pre-existing skin/animation keys defensively
    gltf.pop("skins", None)
    gltf.pop("animations", None)

    nodes = gltf.setdefault("nodes", [])
    worlds = _compute_world_matrices(nodes)
    meshes = gltf.get("meshes", [])

    # Collect mesh nodes & compute per-primitive skin attributes
    mesh_nodes = [i for i, nd in enumerate(nodes) if "mesh" in nd]
    if not mesh_nodes:
        raise ValueError(f"no mesh nodes in {bare_glb}")

    # Build joint nodes (pure translation in world space)
    nj = len(names)
    base_node = len(nodes)
    children: List[List[int]] = [[] for _ in range(nj)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    for i in range(nj):
        p = parents[i]
        if p >= 0:
            tx = positions[i][0] - positions[p][0]
            ty = positions[i][1] - positions[p][1]
            tz = positions[i][2] - positions[p][2]
        else:
            tx, ty, tz = positions[i]
        nd: dict = {
            "name": names[i],
            "translation": [float(tx), float(ty), float(tz)],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        if children[i]:
            nd["children"] = [base_node + c for c in children[i]]
        nodes.append(nd)

    roots = [base_node + i for i, p in enumerate(parents) if p < 0]
    arm_node = len(nodes)
    nodes.append({"name": "Armature", "children": roots})

    scene_idx = gltf.get("scene", 0)
    scenes = gltf.setdefault("scenes", [{"nodes": []}])
    scenes[scene_idx].setdefault("nodes", []).append(arm_node)

    # Write one skin per mesh node (IBM = inv(joint_world) * mesh_world)
    skins = gltf.setdefault("skins", [])
    joint_node_indices = [base_node + i for i in range(nj)]

    # Track per-mesh primitives so we only compute weights once per mesh
    mesh_done: Dict[int, bool] = {}
    for ni in mesh_nodes:
        mi = nodes[ni]["mesh"]
        mw = worlds[ni]
        # Build IBMs: inv(T(pos_j)) @ mw = T(-pos_j) @ mw
        ibm_raw = bytearray()
        for j in range(nj):
            jx, jy, jz = positions[j]
            m = [
                [mw[0][0], mw[0][1], mw[0][2], mw[0][3] - jx],
                [mw[1][0], mw[1][1], mw[1][2], mw[1][3] - jy],
                [mw[2][0], mw[2][1], mw[2][2], mw[2][3] - jz],
                [0.0, 0.0, 0.0, 1.0],
            ]
            # glTF column-major
            for col in range(4):
                for row in range(4):
                    ibm_raw += struct.pack("<f", m[row][col])
        ibm_acc = _push_accessor(gltf, bin_buf, bytes(ibm_raw), 5126, "MAT4", nj)
        skin_idx = len(skins)
        skins.append({
            "name": f"Skin_{skin_idx}",
            "joints": joint_node_indices,
            "skeleton": arm_node,
            "inverseBindMatrices": ibm_acc,
        })
        nodes[ni]["skin"] = skin_idx

        if mesh_done.get(mi):
            continue
        mesh_done[mi] = True

        # Gather all primitive vertices in world space so `min_joint_weight` spans the whole mesh
        prims = meshes[mi].get("primitives", [])
        prim_verts: List[List[Tuple[float, float, float]]] = []
        flat_verts: List[Tuple[float, float, float]] = []
        for prim in prims:
            pos_acc = prim["attributes"].get("POSITION")
            if pos_acc is None:
                prim_verts.append([])
                continue
            ps = [_mat4_apply_point(mw, p) for p in _read_accessor(gltf, bin_buf, pos_acc)]
            prim_verts.append(ps)
            flat_verts.extend(ps)

        all_js, all_ws = _compute_rigid_weights(flat_verts, positions, parents, min_joint_weight=min_joint_weight)
        cursor = 0
        for prim, ps in zip(prims, prim_verts):
            cnt = len(ps)
            if cnt == 0:
                continue
            p_js = all_js[cursor:cursor + cnt]
            p_ws = all_ws[cursor:cursor + cnt]
            cursor += cnt
            j_raw = bytearray()
            w_raw = bytearray()
            for j4, w4 in zip(p_js, p_ws):
                j_raw += struct.pack("<HHHH", j4[0], j4[1], j4[2], j4[3])
                w_raw += struct.pack("<ffff", w4[0], w4[1], w4[2], w4[3])
            j_acc = _push_accessor(gltf, bin_buf, bytes(j_raw), 5123, "VEC4", cnt)
            w_acc = _push_accessor(gltf, bin_buf, bytes(w_raw), 5126, "VEC4", cnt)
            prim["attributes"]["JOINTS_0"] = j_acc
            prim["attributes"]["WEIGHTS_0"] = w_acc

    write_glb(out_glb, gltf, bin_buf)


# ---------------------------------------------------------------------------
# Idempotent SDPA patch for Seed3D/Puppeteer/skeleton (no flash-attn needed!)
# ---------------------------------------------------------------------------

def patch_puppeteer_sdpa(skel_dir: str) -> List[str]:
    """Patch `skeletongen.py`, `skeleton_opt.py`, `demo.py`, `save_utils.py`, and `mesh_to_pc.py`
    to run with PyTorch SDPA on non-Ampere GPUs (e.g. RTX 2060 SUPER sm_75) without `flash_attn`
    or `pyrender`, and with PyTorch 2.6 `weights_only=False` + resilient per-mesh error handling.
    """
    changed: List[str] = []

    def _read(p: str) -> str:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def _write(p: str, s: str) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)

    # 1. skeleton_models/skeletongen.py: switch flash_attention_2 -> sdpa
    sg_path = os.path.join(skel_dir, "skeleton_models", "skeletongen.py")
    if os.path.isfile(sg_path):
        txt = _read(sg_path)
        new_txt = txt.replace('"flash_attention_2"', '"sdpa"').replace("'flash_attention_2'", "'sdpa'")
        if new_txt != txt:
            _write(sg_path, new_txt)
            changed.append(sg_path)

    # 2. skeleton_models/skeleton_opt.py: add SDPA + eager causal mask support
    opt_path = os.path.join(skel_dir, "skeleton_models", "skeleton_opt.py")
    if os.path.isfile(opt_path):
        txt = _read(opt_path)
        new_txt = txt
        if "_prepare_4d_causal_attention_mask_for_sdpa" not in new_txt:
            imp = (
                "from transformers.modeling_attn_mask_utils import (\n"
                "    _prepare_4d_causal_attention_mask,\n"
                "    _prepare_4d_causal_attention_mask_for_sdpa,\n"
                ")\n"
            )
            anchor = "from transformers.models.opt.modeling_opt import"
            if anchor in new_txt:
                new_txt = new_txt.replace(anchor, imp + anchor, 1)
        old_init = 'self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"'
        new_init = (
            'self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"\n'
            '        self._use_sdpa = config._attn_implementation == "sdpa"'
        )
        if old_init in new_txt and "self._use_sdpa" not in new_txt:
            new_txt = new_txt.replace(old_init, new_init, 1)

        old_raise = 'raise ValueError("Only flash_attention_2 is supported")'
        new_mask = (
            "if getattr(self, '_use_sdpa', False) and not output_attentions:\n"
            "                causal_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(\n"
            "                    attention_mask, inputs_embeds.size()[:-1], inputs_embeds, past_key_values_length\n"
            "                )\n"
            "            else:\n"
            "                causal_attention_mask = _prepare_4d_causal_attention_mask(\n"
            "                    attention_mask, inputs_embeds.size()[:-1], inputs_embeds, past_key_values_length\n"
            "                )"
        )
        if old_raise in new_txt:
            new_txt = new_txt.replace(old_raise, new_mask, 1)

        if new_txt != txt:
            _write(opt_path, new_txt)
            changed.append(opt_path)

    # 3. utils/save_utils.py: make PyRenderWrapper import optional (not needed without --save_render)
    su_path = os.path.join(skel_dir, "utils", "save_utils.py")
    if os.path.isfile(su_path):
        txt = _read(su_path)
        old_imp = "from data_utils.pyrender_wrapper import PyRenderWrapper"
        new_imp = (
            "try:\n"
            "    from data_utils.pyrender_wrapper import PyRenderWrapper\n"
            "except Exception:\n"
            "    PyRenderWrapper = None"
        )
        if old_imp in txt and "PyRenderWrapper = None" not in txt:
            new_txt = txt.replace(old_imp, new_imp, 1)
            _write(su_path, new_txt)
            changed.append(su_path)

    # 4. utils/mesh_to_pc.py: make mesh2sdf import lazy so --no-marching-cubes works if mesh2sdf is absent
    mp_path = os.path.join(skel_dir, "utils", "mesh_to_pc.py")
    if os.path.isfile(mp_path):
        txt = _read(mp_path)
        old_m2s = "import mesh2sdf.core"
        new_m2s = (
            "try:\n"
            "    import mesh2sdf.core\n"
            "except Exception:\n"
            "    mesh2sdf = None"
        )
        if old_m2s in txt and "mesh2sdf = None" not in txt:
            new_txt = txt.replace(old_m2s, new_m2s, 1)
            _write(mp_path, new_txt)
            changed.append(mp_path)

    # 5. demo.py: PyTorch 2.6 weights_only=False and skip None detokenize instead of crashing batch
    demo_path = os.path.join(skel_dir, "demo.py")
    if os.path.isfile(demo_path):
        txt = _read(demo_path)
        new_txt = txt
        old_load = 'pkg = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))'
        new_load = 'pkg = torch.load(args.pretrained_weights, map_location=torch.device("cpu"), weights_only=False)'
        if old_load in new_txt:
            new_txt = new_txt.replace(old_load, new_load, 1)
        old_gen = "        skeleton = pred_bone_coords[0].cpu().numpy()"
        new_gen = (
            "        if pred_bone_coords is None:\n"
            "            print(f'Warning: generation returned None for {file_name}', flush=True)\n"
            "            continue\n"
            "        skeleton = pred_bone_coords[0].cpu().numpy()"
        )
        if old_gen in new_txt and "if pred_bone_coords is None:" not in new_txt:
            new_txt = new_txt.replace(old_gen, new_gen, 1)
        if new_txt != txt:
            _write(demo_path, new_txt)
            changed.append(demo_path)

    # 6. third_partys/Michelangelo: PyTorch 2.6 weights_only=False in asl_pl_module.py / sal_perceiver.py
    for rel in (
        os.path.join("third_partys", "Michelangelo", "michelangelo", "models", "tsal", "asl_pl_module.py"),
        os.path.join("third_partys", "Michelangelo", "michelangelo", "models", "tsal", "sal_perceiver.py"),
    ):
        m_init = os.path.join(skel_dir, rel)
        if os.path.isfile(m_init):
            txt = _read(m_init)
            new_txt = txt.replace(
                'torch.load(path, map_location="cpu")',
                'torch.load(path, map_location="cpu", weights_only=False)',
            ).replace(
                'torch.load(ckpt_path, map_location="cpu")',
                'torch.load(ckpt_path, map_location="cpu", weights_only=False)',
            )
            if new_txt != txt:
                _write(m_init, new_txt)
                changed.append(m_init)

    return changed


# ---------------------------------------------------------------------------
# Runner & scoring
# ---------------------------------------------------------------------------

def score_one(cb_bin: str, truth_glb: str, cand_glb: str, score_json: str, roles_json: str = "") -> Optional[dict]:
    cmd = [cb_bin, "score", truth_glb, cand_glb, "--json", score_json]
    if roles_json and os.path.isfile(roles_json):
        cmd += ["--roles", roles_json]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(score_json):
        return None
    d = json.load(open(score_json, encoding="utf-8"))
    return {
        "joints_pct": d["skeleton"]["joint_error_pct"],
        "missing_chains": len(d["skeleton"]["missing_chains"]),
        "skin_mean_pct": d["skin"]["mean_pct"],
        "skin_p95_pct": d["skin"]["p95_pct"],
        "weights_only_pct": (d.get("skin_weights_only") or d.get("weights_only") or {}).get("mean_pct"),
    }


def med(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return round(statistics.median(vals), 3) if vals else None


def summarize(records: Sequence[dict]) -> dict:
    return {
        "joints_median_pct": med([x["joints_pct"] for x in records]),
        "missing_chain_animals": sum(1 for x in records if x["missing_chains"] > 0),
        "skin_mean_median_pct": med([x["skin_mean_pct"] for x in records]),
        "skin_p95_median_pct": med([x["skin_p95_pct"] for x in records]),
        "weights_only_median_pct": med([x["weights_only_pct"] for x in records]),
    }


def run_one_rival(
    rival_key: str,
    bench_dir: str,
    ids: List[str],
    puppeteer_dir: str,
    rival_python: str,
    skintokens_bin: str,
    creature_bench: str,
    seed: int,
    label: str,
    masked: bool,
    apply_mc: bool,
    redo: bool,
) -> dict:
    spec = RIVAL_SPECS[rival_key]
    skel_dir = os.path.join(puppeteer_dir, "skeleton")
    ckpt_path = os.path.join(puppeteer_dir, spec["ckpt_rel"])
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"missing checkpoint for {rival_key}: {ckpt_path}")

    patch_puppeteer_sdpa(skel_dir)

    subdir = spec["subdir"]
    batch_work = os.path.join(bench_dir, f"_rival_work_{rival_key}")
    obj_dir = os.path.join(batch_work, "objs")
    out_dir = os.path.join(batch_work, "preds")
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Stage .obj for every animal that needs skeleton prediction
    need_ids: List[str] = []
    # Clean obj_dir so demo.py only processes animals missing cached _pred.txt
    for f in os.listdir(obj_dir):
        if f.endswith(".obj"):
            os.remove(os.path.join(obj_dir, f))

    for aid in ids:
        adir = os.path.join(bench_dir, "animals", aid)
        rdir = os.path.join(adir, subdir)
        os.makedirs(rdir, exist_ok=True)
        pred_txt = os.path.join(rdir, f"{aid}_pred.txt")
        if redo or not os.path.isfile(pred_txt):
            bare = os.path.join(adir, "bare.glb")
            glb_to_obj(bare, os.path.join(obj_dir, f"{aid}.obj"))
            need_ids.append(aid)

    # 2. Run rival demo.py once over all needed .obj files (4.6 GB model loads once!)
    t_infer = 0.0
    if need_ids:
        save_name = f"batch_{label}_{rival_key}"
        save_out = os.path.join(out_dir, save_name)
        if os.path.isdir(save_out):
            shutil.rmtree(save_out)
        cmd = [
            rival_python,
            "demo.py",
            "--input_dir", obj_dir,
            "--output_dir", out_dir,
            "--save_name", save_name,
            "--pretrained_weights", ckpt_path,
            "--input_pc_num", "8192",
            "--num_beams", "1",
            "--seed", str(seed),
            *spec["demo_flags"],
        ]
        if apply_mc:
            cmd.append("--apply_marching_cubes")

        log_path = os.path.join(batch_work, f"demo_{label}_{rival_key}.log")
        t0 = time.time()
        demo_env = os.environ.copy()
        demo_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            r = subprocess.run(cmd, cwd=skel_dir, env=demo_env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        t_infer = time.time() - t0
        print(
            f"[{label}/{rival_key}] demo.py on {len(need_ids)} meshes finished (rc={r.returncode}) in {t_infer:.1f}s",
            flush=True,
        )
        if r.returncode != 0:
            tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
            print(f"[{label}/{rival_key}] demo.py tail:\n{tail}", flush=True)

        for aid in need_ids:
            src_txt = os.path.join(save_out, f"{aid}_pred.txt")
            dst_txt = os.path.join(bench_dir, "animals", aid, subdir, f"{aid}_pred.txt")
            if os.path.isfile(src_txt):
                shutil.copyfile(src_txt, dst_txt)

    # 3. Attach skeleton -> rigid GLB + SkinTokens-skinned GLB, and score both
    scored_skinned: List[dict] = []
    scored_rigid: List[dict] = []
    failed: List[dict] = []

    for idx, aid in enumerate(ids, 1):
        tag_label = f"animal_{idx}" if masked else aid
        adir = os.path.join(bench_dir, "animals", aid)
        bare = os.path.join(adir, "bare.glb")
        truth = os.path.join(adir, "truth.glb")
        roles = os.path.join(adir, "roles.json")
        rdir = os.path.join(adir, subdir)
        pred_txt = os.path.join(rdir, f"{aid}_pred.txt")

        if not os.path.isfile(pred_txt):
            msg = "missing _pred.txt from skeleton inference"
            print(f"[{label}/{rival_key}] {tag_label}: FAIL ({msg})", flush=True)
            failed.append({"id": tag_label, "reason": msg})
            continue

        try:
            names, positions, parents = parse_rig_txt(pred_txt)
        except Exception as e:
            msg = f"parse_rig_txt failed: {e}"
            print(f"[{label}/{rival_key}] {tag_label}: FAIL ({msg})", flush=True)
            failed.append({"id": tag_label, "reason": msg})
            continue

        rigid_glb = os.path.join(rdir, f"{aid}_rigid.glb")
        skin_in_glb = os.path.join(rdir, f"{aid}_skin_in.glb")
        skinned_glb = os.path.join(rdir, f"{aid}.glb")

        if redo or not os.path.isfile(rigid_glb):
            attach_skeleton_to_glb(bare, rigid_glb, names, positions, parents, min_joint_weight=0.0)
        if redo or not os.path.isfile(skin_in_glb):
            attach_skeleton_to_glb(bare, skin_in_glb, names, positions, parents, min_joint_weight=1e-4)

        # Skin with SkinTokens ML skin stage (keeping rival skeleton untouched)
        skin_ok = os.path.isfile(skinned_glb) and not redo
        if not skin_ok:
            skin_log = os.path.join(rdir, "skintokens_skin.log")
            with open(skin_log, "w", encoding="utf-8") as lf:
                sr = subprocess.run(
                    [skintokens_bin, "skin", skin_in_glb, skinned_glb, "--seed", str(seed)],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            skin_ok = sr.returncode == 0 and os.path.isfile(skinned_glb)

        # Score rigid baseline
        rec_rigid = score_one(creature_bench, truth, rigid_glb, os.path.join(rdir, "score_rigid.json"), roles)
        if rec_rigid:
            rec_rigid["id"] = tag_label
            rec_rigid["n_joints"] = len(names)
            scored_rigid.append(rec_rigid)

        # Score rival skeleton + SkinTokens skin
        if not skin_ok:
            msg = "skintokens skin failed"
            print(f"[{label}/{rival_key}] {tag_label}: FAIL ({msg})", flush=True)
            failed.append({"id": tag_label, "reason": msg})
            continue

        rec_skin = score_one(creature_bench, truth, skinned_glb, os.path.join(rdir, "score.json"), roles)
        if not rec_skin:
            msg = "creature-bench score failed on skinned GLB"
            print(f"[{label}/{rival_key}] {tag_label}: FAIL ({msg})", flush=True)
            failed.append({"id": tag_label, "reason": msg})
            continue

        rec_skin["id"] = tag_label
        rec_skin["n_joints"] = len(names)
        scored_skinned.append(rec_skin)
        r_j = f"{rec_rigid['joints_pct']:.2f}%" if rec_rigid else "-"
        print(
            f"[{label}/{rival_key}] {tag_label}: J={len(names)} | "
            f"joints={rec_skin['joints_pct']:.2f}% (rigid {r_j}) miss={rec_skin['missing_chains']} "
            f"skin={rec_skin['skin_mean_pct']:.2f}% p95={rec_skin['skin_p95_pct']:.2f}% "
            f"wo={rec_skin['weights_only_pct']:.2f}%",
            flush=True,
        )

    summary = {
        "label": label,
        "rival": spec["display"],
        "rival_key": rival_key,
        "skin_stage": "skintokens skin (ML weights on rival skeleton)",
        "apply_marching_cubes": apply_mc,
        "seed": seed,
        "total": len(ids),
        "succeeded": len(scored_skinned),
        "failed_count": len(failed),
        "medians": summarize(scored_skinned),
        "rigid_baseline": {
            "succeeded": len(scored_rigid),
            "medians": summarize(scored_rigid),
        },
        "failed": failed,
    }
    if not masked:
        summary["per_animal"] = scored_skinned
        summary["per_animal_rigid"] = scored_rigid
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench_dir")
    ap.add_argument("--rival", choices=["puppeteer", "magicarticulate", "both"], default="both")
    ap.add_argument("--puppeteer-dir", default="/data/rigwork/puppeteer")
    ap.add_argument("--rival-python", default="/data/rigwork/venv-puppeteer/bin/python")
    ap.add_argument(
        "--skintokens-bin",
        default=os.environ.get("SKINTOKENS_BIN", "/home/f/rigwork/bin_current/skintokens"),
    )
    ap.add_argument(
        "--creature-bench",
        default=os.environ.get("CREATURE_BENCH_BIN", "/home/f/rigwork/bin_current/creature-bench"),
    )
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-marching-cubes", action="store_true")
    ap.add_argument("--only-ids", default="", help="Comma-separated animal IDs (in-sample smoke test only)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)

    masked = bool(a.manifest)
    if a.manifest:
        ids = [x["id"] for x in json.load(open(a.manifest, encoding="utf-8"))["animals"]]
    else:
        ids = [l.split("\t")[0] for l in open(os.path.join(a.bench_dir, "animals.tsv"), encoding="utf-8") if l.strip()]

    if a.only_ids:
        if masked:
            raise SystemExit("refusing --only-ids on held-out manifest")
        want = set(x.strip() for x in a.only_ids.split(",") if x.strip())
        ids = [x for x in ids if x in want]

    rivals = ["puppeteer", "magicarticulate"] if a.rival == "both" else [a.rival]
    results = {}
    for rk in rivals:
        res = run_one_rival(
            rival_key=rk,
            bench_dir=a.bench_dir,
            ids=ids,
            puppeteer_dir=a.puppeteer_dir,
            rival_python=a.rival_python,
            skintokens_bin=a.skintokens_bin,
            creature_bench=a.creature_bench,
            seed=a.seed,
            label=a.label,
            masked=masked,
            apply_mc=not a.no_marching_cubes,
            redo=a.redo,
        )
        results[rk] = res
        print(f"=== {a.label} {rk} summary (skel + skintokens skin) ===", flush=True)
        print(json.dumps(res["medians"], indent=2), flush=True)
        print(f"=== {a.label} {rk} rigid baseline ===", flush=True)
        print(json.dumps(res["rigid_baseline"]["medians"], indent=2), flush=True)

    payload = results[rivals[0]] if len(rivals) == 1 else {"label": a.label, "rivals": results}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
