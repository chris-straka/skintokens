"""Minimal GLB access: skeleton (first skin), skinned vertices, in-place weight rewrite.

Only what the wrapper needs, numpy only. Positions and joint heads are read in
the mesh's bind space (POSITION as stored; heads from the inverse bind
matrices), so two exports of the same mesh compare directly.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

_JSON, _BIN = 0x4E4F534A, 0x004E4942
_CT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Glb:
    js: dict
    bin: bytearray

    @classmethod
    def read(cls, path: str) -> "Glb":
        b = open(path, "rb").read()
        magic, _ver, total = struct.unpack("<III", b[:12])
        if magic != 0x46546C67:
            raise ValueError(f"{path}: not a GLB")
        off, js, bin_ = 12, None, bytearray()
        while off < total:
            ln, typ = struct.unpack("<II", b[off : off + 8])
            chunk = b[off + 8 : off + 8 + ln]
            if typ == _JSON:
                js = json.loads(chunk)
            elif typ == _BIN:
                bin_ = bytearray(chunk)
            off += 8 + ln
        if js is None:
            raise ValueError(f"{path}: no JSON chunk")
        return cls(js, bin_)

    def write(self, path: str) -> None:
        j = json.dumps(self.js, separators=(",", ":")).encode()
        j += b" " * ((4 - len(j) % 4) % 4)
        b = bytes(self.bin) + b"\0" * ((4 - len(self.bin) % 4) % 4)
        body = struct.pack("<II", len(j), _JSON) + j
        if b:
            body += struct.pack("<II", len(b), _BIN) + b
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)

    # ------------------------------------------------------------ accessors

    def _layout(self, i: int):
        a = self.js["accessors"][i]
        bv = self.js["bufferViews"][a["bufferView"]]
        dt = np.dtype(_CT[a["componentType"]])
        n = _NC[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * n
        return a, dt, n, start, stride

    def accessor(self, i: int) -> np.ndarray:
        a, dt, n, start, stride = self._layout(i)
        count = a["count"]
        raw = np.frombuffer(self.bin, dtype=np.uint8, count=(count - 1) * stride + dt.itemsize * n, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dt.itemsize * n), strides=(stride, 1))
        out = rows.copy().view(dt).reshape(count, n).astype(np.float64)
        if a.get("normalized") and dt.kind in "ui":
            out /= np.iinfo(dt).max
        return out

    def set_accessor(self, i: int, arr: np.ndarray) -> None:
        a, dt, n, start, stride = self._layout(i)
        arr = np.asarray(arr, dtype=np.float64).reshape(a["count"], n)
        if a.get("normalized") and dt.kind in "ui":
            arr = np.round(arr * np.iinfo(dt).max)
        data = arr.astype(dt)
        for k in range(a["count"]):
            p = start + k * stride
            self.bin[p : p + dt.itemsize * n] = data[k].tobytes()
        if "min" in a or "max" in a:  # keep bounds valid for the new values
            a["min"] = data.min(0).tolist()
            a["max"] = data.max(0).tolist()

    # ------------------------------------------------------------- rigging

    def skinned_primitives(self):
        for m in self.js.get("meshes", []):
            for p in m["primitives"]:
                if "JOINTS_0" in p["attributes"] and "WEIGHTS_0" in p["attributes"]:
                    yield p

    def skeleton(self):
        """(names, parents, heads) of the first skin; parents index into the joint list."""
        skins = self.js.get("skins") or []
        if not skins:
            raise ValueError("no skin")
        joints = skins[0]["joints"]
        ibm = self.accessor(skins[0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        nodes = self.js["nodes"]
        names = [nodes[j].get("name", f"joint_{j}") for j in joints]
        parent_of = {c: i for i, nd in enumerate(nodes) for c in nd.get("children", [])}
        index = {j: k for k, j in enumerate(joints)}
        parents = []
        for j in joints:
            p = parent_of.get(j)
            while p is not None and p not in index:
                p = parent_of.get(p)
            parents.append(None if p is None else index[p])
        heads = np.array([np.linalg.inv(m)[:3, 3] for m in ibm])
        return names, parents, heads

    def skinned_vertices(self):
        """Concatenated (positions, joints, weights) over skinned primitives."""
        P, J, W = [], [], []
        for p in self.skinned_primitives():
            a = p["attributes"]
            P.append(self.accessor(a["POSITION"]))
            J.append(self.accessor(a["JOINTS_0"]).astype(int))
            W.append(self.accessor(a["WEIGHTS_0"]))
        if not P:
            raise ValueError("no skinned primitive")
        return np.concatenate(P), np.concatenate(J), np.concatenate(W)

    def all_positions(self) -> np.ndarray:
        P = [self.accessor(p["attributes"]["POSITION"]) for m in self.js.get("meshes", []) for p in m["primitives"]]
        return np.concatenate(P)

    def helper_joints(self) -> dict[int, int]:
        """{helper: driver} as first-skin joint indices.

        Twist/helper bones (motionforge's HLL skeleton: DEF-upper_arm_twist.L,
        DEF-thigh_twist.R, ...) sit on their driver's joint and turn by a share
        of it. Marked by node extras.hll_helper.driver, else by the name
        `<driver>_twist.<side>`.
        """
        skins = self.js.get("skins") or []
        if not skins:
            return {}
        joints = skins[0]["joints"]
        nodes = self.js["nodes"]
        names = [nodes[j].get("name", "") for j in joints]
        out = {}
        for k, j in enumerate(joints):
            extra = (nodes[j].get("extras") or {}).get("hll_helper") or {}
            driver = extra.get("driver")
            if driver is None:
                base, _, side = names[k].rpartition(".")
                if side in ("L", "R") and base.endswith("_twist"):
                    driver = f"{base[: -len('_twist')]}.{side}"
            if driver in names and names.index(driver) != k:
                out[k] = names.index(driver)
        return out

    def without_helpers(self) -> tuple["Glb", list[int]]:
        """A copy whose skin leaves out the helper joints (their weight goes
        to the driver; the helper nodes are detached from the hierarchy).
        Returns (copy, kept): kept[i] is the original index of joint i."""
        import copy

        helpers = self.helper_joints()
        g = Glb(copy.deepcopy(self.js), bytearray(self.bin))
        if not helpers:
            return g, list(range(len(self.js["skins"][0]["joints"])))
        skin = g.js["skins"][0]
        joints = skin["joints"]
        kept = [k for k in range(len(joints)) if k not in helpers]
        new_of = {old: new for new, old in enumerate(kept)}
        for h, d in helpers.items():
            new_of[h] = new_of[d]
        ibm = self.accessor(skin["inverseBindMatrices"])[kept]
        skin["inverseBindMatrices"] = g.push_accessor(ibm, "MAT4")
        dropped = {joints[h] for h in helpers}
        skin["joints"] = [joints[k] for k in kept]
        for nd in g.js["nodes"]:
            if "children" in nd:
                nd["children"] = [c for c in nd["children"] if c not in dropped]
                if not nd["children"]:
                    del nd["children"]
        for p in g.skinned_primitives():
            a = p["attributes"]
            J = g.accessor(a["JOINTS_0"]).astype(int)
            g.set_accessor(a["JOINTS_0"], np.vectorize(new_of.get)(J))
        return g, kept

    def push_accessor(self, arr: np.ndarray, kind: str) -> int:
        """Append float32 rows as a new accessor; returns its index."""
        data = np.asarray(arr, dtype="<f4").tobytes()
        self.bin += b"\0" * ((4 - len(self.bin) % 4) % 4)
        views = self.js.setdefault("bufferViews", [])
        views.append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)})
        self.bin += data
        self.js["buffers"][0]["byteLength"] = len(self.bin)
        accs = self.js.setdefault("accessors", [])
        accs.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(arr), "type": kind})
        return len(accs) - 1

    def rename_joints(self, mapping: dict[str, str]) -> None:
        skins = self.js.get("skins") or []
        if not skins:
            return
        for j in skins[0]["joints"]:
            nd = self.js["nodes"][j]
            if nd.get("name") in mapping:
                nd["name"] = mapping[nd["name"]]
