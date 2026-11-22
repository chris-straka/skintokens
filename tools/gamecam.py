"""Shared HLH follow-camera helpers for creature_blind.py and render_gamecam.py."""
import importlib.util
import json
import os
import re

DEFAULT_FOLLOW_FOV_DEG = 75.0


def _load_limits_module():
    """Import `_tools/blender/scale/limits.py` when a Games workspace sits beside us."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "_tools", "blender", "scale", "limits.py"),
        os.path.join(os.path.dirname(os.path.dirname(here)), "_tools", "blender", "scale", "limits.py"),
    ]
    tools_dir = os.environ.get("GAMES_TOOLS_DIR")
    if tools_dir:
        candidates.insert(0, os.path.join(tools_dir, "blender", "scale", "limits.py"))
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_scale_limits", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    return None


def _parse_ron_fallback(text):
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    body = re.sub(r",(\s*[}\]])", r"\1", "".join(out))
    return json.loads(body)


def _read_camera_fov(metrics_path):
    limits = _load_limits_module()
    if limits is not None:
        doc = limits.load(metrics_path)
    else:
        with open(metrics_path, "r", encoding="utf-8") as f:
            doc = _parse_ron_fallback(f.read())
    cam = doc.get("camera")
    if not isinstance(cam, dict):
        return None
    fov = cam.get("fov")
    if isinstance(fov, (int, float)) and not isinstance(fov, bool) and 1.0 < float(fov) < 179.0:
        return float(fov)
    return None


def resolve_follow_fov_deg(metrics_path=None, fov_deg=None):
    """Resolve follow-camera vertical FOV in degrees.

    Precedence:
    1. Explicit `fov_deg` (CLI `--fov`).
    2. Explicit `metrics_path` (CLI `--metrics`), else `$METRICS_FILE` / `$HLH_METRICS` (or `$GAME_DIR` via `_tools/blender/scale/limits.py`).
    3. Canonical 75 deg follow-camera default.
    """
    if fov_deg is not None and str(fov_deg).strip() != "":
        val = float(fov_deg)
        if not (1.0 < val < 179.0):
            raise ValueError(f"follow-camera FOV out of range: {fov_deg!r}")
        return val

    mpath = metrics_path or os.environ.get("METRICS_FILE") or os.environ.get("HLH_METRICS")
    if not mpath and os.environ.get("GAME_DIR"):
        limits = _load_limits_module()
        if limits is not None:
            mpath = limits.find(os.environ["GAME_DIR"])

    if mpath:
        if metrics_path and not os.path.isfile(mpath):
            raise FileNotFoundError(f"metrics file not found: {mpath}")
        if os.path.isfile(mpath):
            val = _read_camera_fov(mpath)
            if val is not None:
                return val

    return DEFAULT_FOLLOW_FOV_DEG
