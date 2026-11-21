#!/usr/bin/env python3
"""Shared binary lookup and subprocess runner for skintokens tools (stdlib only)."""

import os
import shutil
import subprocess
import sys
import time


def find(name, explicit, env, fallbacks, first=(), required=True):
    for p in [explicit, os.environ.get(env)] + list(first) + [shutil.which(name)] + list(fallbacks):
        if p and os.path.exists(p):
            return p
    if not required:
        return None
    print(f"{name} not found: pass --{name}, set {env}, or put it on PATH", file=sys.stderr)
    raise SystemExit(2)


def run(cmd, timeout=None, env=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 124, e.stdout or "", f"timed out after {timeout} s"
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
    return {"code": code, "secs": round(time.time() - t, 1), "out": out, "err": err}
