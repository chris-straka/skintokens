"""Structural humanoid naming for TokenRig skeletons.

TokenRig's general ("articulation") class emits unnamed joints (bone_0..).
For humanoids we name them by structure so motionforge's standardize
adapter can map them onto the HLL skeleton: Mixamo names, which it already
reads (it also reads Tripo's `spec: mixamo` rigs). glTF frame: +Y up; the
character's left is +X when it faces +Z, and the toes decide the facing.

Joints the walk cannot place keep a non-canonical name (`extra_*`);
standardize then folds their weights into the nearest kept ancestor.
"""

from __future__ import annotations

import numpy as np

CORE = [
    "Hips", "Head",
    "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
]
_ARM = ["Shoulder", "Arm", "ForeArm", "Hand"]
_LEG = ["UpLeg", "Leg", "Foot", "ToeBase"]
_SPINE = ["Spine", "Spine1", "Spine2"]
PREFIX = "mixamorig:"


def name_humanoid(names, parents, heads) -> tuple[dict[str, str], list[str]]:
    """Return ({old: new}, missing core names). New names carry the mixamorig: prefix."""
    heads = np.asarray(heads, dtype=float)
    n = len(names)
    kids: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p is not None:
            kids[p].append(i)
    size = [0] * n

    def count(i):
        size[i] = 1 + sum(count(k) for k in kids[i])
        return size[i]

    roots = [i for i, p in enumerate(parents) if p is None]
    if len(roots) != 1:
        return {}, list(CORE)
    root = roots[0]
    count(root)
    height = float(np.ptp(heads[:, 1])) or 1.0
    mid_x = heads[root][0]
    out: dict[int, str] = {root: "Hips"}

    # Spine: from the root, climb through the most central upward child.
    def dx(k):
        return abs(heads[k][0] - mid_x)

    def step(j, exclude=()):
        c = [k for k in kids[j] if k not in exclude and heads[k][1] >= heads[j][1] - 0.02 * height
             and dx(k) < 0.04 * height]
        return min(c, key=lambda k: (round(dx(k), 3), -size[k])) if c else None

    chain, cur = [], step(root)
    while cur is not None:
        chain.append(cur)
        cur = step(cur)

    def lowest(i):
        return min([heads[i][1]] + [lowest(k) for k in kids[i]])

    def pick_pair(cands):
        # One limb per side: the largest subtree on each side; the rest are extras.
        best = {}
        for k in cands:
            s = heads[k][0] > mid_x
            if s not in best or size[k] > size[best[s]]:
                best[s] = k
        return list(best.values()), [k for k in cands if k not in best.values()]

    legs, odd = pick_pair([k for k in kids[root] if k not in chain and lowest(k) < heads[root][1] - 0.3 * height])
    odd += [k for k in kids[root] if k not in chain and k not in legs and k not in odd]

    # The chest is the highest chain joint with children on both sides (the arms).
    chest = None
    for j in chain:
        side = [k for k in kids[j] if k not in chain]
        if any(heads[k][0] > mid_x for k in side) and any(heads[k][0] < mid_x for k in side):
            chest = j
    if chest is None:
        return {}, [c for c in CORE if c != "Hips"]
    ci = chain.index(chest)
    spine, above = chain[: ci + 1], chain[ci + 1 :]
    for k, j in enumerate(spine):
        out[j] = _SPINE[k] if k < len(_SPINE) else f"extra_spine{k}"
    if len(above) == 1:
        out[above[0]] = "Head"
    elif above:
        out[above[0]] = "Neck"
        out[above[1]] = "Head"
        for k, j in enumerate(above[2:]):
            out[j] = f"extra_head{k}"

    def walk(start, labels, side):
        j, k = start, 0
        while j is not None:
            out[j] = f"{side}{labels[k]}" if k < len(labels) else f"extra_{side}{labels[-1]}{k}"
            k += 1
            j = max(kids[j], key=lambda c: size[c]) if kids[j] else None

    arms, odd_arms = pick_pair([k for k in kids[chest] if k not in chain])
    for k, j in enumerate(odd + odd_arms):
        walk(j, [f"extra_{k}_"], "")
    for a in arms:
        walk(a, _ARM, "Left" if heads[a][0] > mid_x else "Right")
    for leg in legs:
        walk(leg, _LEG, "Left" if heads[leg][0] > mid_x else "Right")

    # Facing: toes in front of the ankles. Facing -Z mirrors left/right.
    by_name = {v: k for k, v in out.items()}
    dz = [heads[by_name[f"{s}ToeBase"]][2] - heads[by_name[f"{s}Foot"]][2]
          for s in ("Left", "Right") if f"{s}ToeBase" in by_name and f"{s}Foot" in by_name]
    if dz and np.mean(dz) < 0:
        swap = {"Left": "Right", "Right": "Left"}
        for k, v in out.items():
            for a, b in swap.items():
                if a in v:
                    out[k] = v.replace(a, b)
                    break

    got = set(out.values())
    missing = [c for c in CORE if c not in got]
    mapping = {names[i]: (PREFIX + v if not v.startswith("extra") else v) for i, v in out.items()}
    return mapping, missing
