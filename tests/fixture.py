"""Synthetic skinned GLB for tests: a stick humanoid (TokenRig-like bone_N
names, 22 joints) with one quad strip per bone, weights fully on that bone."""

from __future__ import annotations

import json
import struct

import numpy as np

# name, parent, head (x, y, z); +Y up, faces +Z, character left = +X
HUMANOID = [
    ("hips", None, (0.0, 1.0, 0.0)),
    ("spine", 0, (0.0, 1.12, 0.0)),
    ("spine1", 1, (0.0, 1.26, 0.0)),
    ("chest", 2, (0.0, 1.40, 0.0)),
    ("neck", 3, (0.0, 1.56, 0.0)),
    ("head", 4, (0.0, 1.66, 0.0)),
    ("l_shoulder", 3, (0.07, 1.54, 0.0)),
    ("l_arm", 6, (0.20, 1.46, 0.0)),
    ("l_forearm", 7, (0.31, 1.20, 0.0)),
    ("l_hand", 8, (0.40, 0.97, 0.0)),
    ("r_shoulder", 3, (-0.07, 1.54, 0.0)),
    ("r_arm", 10, (-0.20, 1.46, 0.0)),
    ("r_forearm", 11, (-0.31, 1.20, 0.0)),
    ("r_hand", 12, (-0.40, 0.97, 0.0)),
    ("l_thigh", 0, (0.12, 0.97, 0.0)),
    ("l_shin", 14, (0.14, 0.53, 0.0)),
    ("l_foot", 15, (0.18, 0.12, 0.0)),
    ("l_toe", 16, (0.23, 0.02, 0.10)),
    ("r_thigh", 0, (-0.12, 0.97, 0.0)),
    ("r_shin", 18, (-0.14, 0.53, 0.0)),
    ("r_foot", 19, (-0.18, 0.12, 0.0)),
    ("r_toe", 20, (-0.23, 0.02, 0.10)),
]


def humanoid(mirror_z=False, names=None):
    joints = [(n, p, np.array(h) * (1, 1, -1 if mirror_z else 1)) for n, p, h in HUMANOID]
    if mirror_z:  # face -Z: turn around (x and z flip) so left stays the character's left
        joints = [(n, p, h * (-1, 1, 1)) for n, p, h in joints]
    if names:
        joints = [(names[i], p, h) for i, (_, p, h) in enumerate(joints)]
    return joints


def write_glb(path, joints, jitter=0.0, weight_shift=None):
    """One quad per joint around its head; JOINTS_0 = that joint (u8), WEIGHTS_0 float."""
    P, F, J, W = [], [], [], []
    for i, (_, _, h) in enumerate(joints):
        b = len(P)
        for dx, dy in ((-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)):
            P.append(h + (dx, dy, 0.0) + jitter * i * 1e-3)
            k = i if weight_shift is None else weight_shift(i)
            J.append((k, 0, 0, 0))
            W.append((1.0, 0, 0, 0))
        F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    P = np.asarray(P, "<f4"); F = np.asarray(F, "<u2"); J = np.asarray(J, "u1"); W = np.asarray(W, "<f4")
    ibm = np.stack([np.linalg.inv(np.block([[np.eye(3), h.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]))
                    for _, _, h in joints]).transpose(0, 2, 1).astype("<f4")
    blobs = [P.tobytes(), F.tobytes(), J.tobytes(), W.tobytes(), ibm.tobytes()]
    views, off, buf = [], 0, b""
    for bl in blobs:
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(bl)})
        buf += bl + b"\0" * ((4 - len(bl) % 4) % 4)
        off = len(buf)
    acc = [
        {"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3",
         "min": P.min(0).tolist(), "max": P.max(0).tolist()},
        {"bufferView": 1, "componentType": 5123, "count": F.size, "type": "SCALAR"},
        {"bufferView": 2, "componentType": 5121, "count": len(J), "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": len(W), "type": "VEC4"},
        {"bufferView": 4, "componentType": 5126, "count": len(joints), "type": "MAT4"},
    ]
    nodes = [{"name": "mesh", "mesh": 0, "skin": 0}]
    for i, (n, p, h) in enumerate(joints):
        local = h - (joints[p][2] if p is not None else 0)
        nodes.append({"name": n, "translation": [float(v) for v in local]})
    for i, (_, p, _) in enumerate(joints):
        if p is not None:
            nodes[p + 1].setdefault("children", []).append(i + 1)
    root = [i + 1 for i, (_, p, _) in enumerate(joints) if p is None]
    js = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0] + root}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
        "skins": [{"joints": [i + 1 for i in range(len(joints))], "inverseBindMatrices": 4}],
        "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": len(buf)}],
    }
    j = json.dumps(js).encode(); j += b" " * ((4 - len(j) % 4) % 4)
    body = struct.pack("<II", len(j), 0x4E4F534A) + j + struct.pack("<II", len(buf), 0x004E4942) + buf
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
