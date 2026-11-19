#!/usr/bin/env python3
"""The creature rig benchmark, part B: a rig with no hand-made truth made
to follow a reference video (stdlib only).

    python3 tools/creature_follow.py DIR [clips...] [--creature-bench BIN] [--weights BIN]
        [--blender BIN] [--step 2]

DIR holds one creature: `rig.glb` (the route's rig), `camera.json` and
`start.png` (from `tools/bench/render_side.py`: the picture the video
was started from, and its camera) and one `<clip>.mp4` per reference
video (a still camera, a plain background, the creature seen from its
side). For each clip:

1. ffmpeg cuts the video into frames at the camera's size.
2. `creature-bench follow` lines the camera up on the first frame, then
   turns the rig's joints frame by frame until its outline lies on the
   video's; it writes the rig with the followed motion as a clip
   (`<clip>.followed.glb`) and the overlap of the two outlines per frame.
3. `weights check` (meshweights) on that file with no pose set: tears
   while following = vertices in failing stretch, volume and bleed
   findings, as a share of all vertices.
4. Blender renders the followed clip, painted, through the lined-up
   camera, and ffmpeg sets it beside the video and the two outlines
   (`<clip>.loop.mp4`: video, rig, outlines; red = video only, green =
   rig only, yellow = both).

Writes `follow.json` in DIR. What the numbers mean: docs/creature-bench.md.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toolpaths  # noqa: E402

TEARS = ("D_STRETCH", "D_VOLUME", "D_BLEED")


def follow_clip(cb, wf, d, c, video, rig, cam, camera, fps, step, empty, followed, js, lined, overlays):
    """Frames, the follow itself, and the tears while following."""
    frames = os.path.join(d, c + "_frames")
    if os.path.isdir(frames):
        shutil.rmtree(frames)
    os.makedirs(frames)
    res = toolpaths.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", f"scale={camera['width']}:{camera['height']}",
                         os.path.join(frames, "%04d.png")])
    if res["code"] != 0:
        raise SystemExit(f"{c}: ffmpeg: {res['err'].strip()[-300:]}")
    if os.path.isdir(overlays):
        shutil.rmtree(overlays)
    res = toolpaths.run([cb, "follow", rig, "--camera", cam, "--frames", frames, "--out", followed, "--json", js, "--name", c,
                         "--fps", str(fps), "--step", str(step), "--overlays", overlays, "--camera-out", lined])
    if res["code"] != 0:
        raise SystemExit(f"{c}: follow: {res['err'].strip()[-300:]}")
    print(" ", res["out"].strip())
    with open(js) as f:
        rep = json.load(f)
    res = toolpaths.run([wf, "check", followed, "--json", "--poses", empty])
    try:
        chk = json.loads(res["out"])
        fails = [x for x in chk.get("findings", []) if x.get("severity") == "fail" and x["code"] in TEARS]
        rep["tears_pct"] = round(min(100.0, 100.0 * sum(x["verts"] for x in fails) / max(chk.get("verts", 0), 1)), 2)
        rep["tears_findings"] = [f"{x['code']} {x['region']} ({x['verts']}): {x['detail']}" for x in fails]
        rep["frames_checked_for_tears"] = chk.get("poses")
    except ValueError:
        rep["tears_error"] = (res["err"] or res["out"]).strip()[-200:]
    shutil.rmtree(frames)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--creature-bench", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--blender", default="")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--keep", action="store_true", help="a clip already followed is not followed again: only its pictures are made")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cb = toolpaths.find("creature-bench", a.creature_bench, "CREATURE_BENCH_BIN", [os.path.join(repo, "target/release/creature-bench")])
    wf = toolpaths.find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(os.path.dirname(repo), "meshweights/rust/target/release/weights")])
    d = os.path.abspath(a.dir)
    rig, cam = os.path.join(d, "rig.glb"), os.path.join(d, "camera.json")
    with open(cam) as f:
        camera = json.load(f)
    clips = a.clips or sorted(f[:-4] for f in os.listdir(d) if f.endswith(".mp4") and not f.endswith(".loop.mp4"))
    empty = os.path.join(d, "clips_only.ron")
    with open(empty, "w") as f:
        f.write('PoseSet(\n    name: "clips only",\n    poses: [],\n)\n')
    out_path = os.path.join(d, "follow.json")
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
    for c in clips:
        video = os.path.join(d, c + ".mp4")
        followed = os.path.join(d, c + ".followed.glb")
        js = os.path.join(d, c + ".follow.json")
        lined = os.path.join(d, c + ".camera.json")
        overlays = os.path.join(d, c + "_overlays")
        fps_text = toolpaths.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video])["out"]
        num, _, den = fps_text.strip().partition("/")
        fps = float(num) / float(den or 1) if num else 24.0
        if a.keep and all(os.path.exists(p) for p in (followed, js, lined, overlays)):
            with open(js) as f:
                rep = json.load(f)
            for k in ("render_error", "loop_error"):
                rep.pop(k, None)
        else:
            rep = follow_clip(cb, wf, d, c, video, rig, cam, camera, fps, a.step, empty, followed, js, lined, overlays)
        if not a.no_render:
            blender = toolpaths.find("blender", a.blender, "BLENDER", ["/Applications/Blender.app/Contents/MacOS/Blender"])
            shots = os.path.join(d, c + "_rig_frames")
            if os.path.isdir(shots):
                shutil.rmtree(shots)
            # the clip holds every `step`-th video frame: render it at that rate
            rate = fps / a.step
            res = toolpaths.run([blender, "-b", "--factory-startup", "-P", os.path.join(repo, "tools/bench/render_side.py"), "--", followed,
                                 "--camera", lined, "--clip", c, "--frames", shots, "--fps", str(rate)])
            if res["code"] != 0 or "RENDER_SIDE" not in res["out"]:
                rep["render_error"] = (res["err"] or res["out"]).strip()[-300:]
            else:
                loop = os.path.join(d, c + ".loop.mp4")
                w, h = camera["width"] // 2, camera["height"] // 2
                flt = (f"[0:v]select='not(mod(n\\,{a.step}))',setpts=N/({rate}*TB),scale={w}:{h}[v];[1:v]scale={w}:{h}[r];"
                       f"[2:v]scale={w}:{h}:flags=neighbor[o];[v][r][o]hstack=inputs=3,pad=ceil(iw/2)*2:ceil(ih/2)*2[out]")
                res = toolpaths.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-framerate", str(rate), "-i", os.path.join(shots, "%04d.png"),
                                     "-framerate", str(rate), "-i", os.path.join(overlays, "%04d.png"), "-filter_complex", flt, "-map", "[out]",
                                     "-r", str(rate), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", loop])
                if res["code"] != 0:
                    rep["loop_error"] = res["err"].strip()[-300:]
                else:
                    rep["loop"] = loop
                    # a still of the worst frame, for a report
                    k = rep["overlap_per_frame"].index(min(rep["overlap_per_frame"]))
                    toolpaths.run(["ffmpeg", "-y", "-loglevel", "error", "-i", loop, "-vf", f"select='eq(n\\,{k})'", "-frames:v", "1", os.path.join(d, c + ".worst.png")])
        with open(js, "w") as f:
            json.dump(rep, f, indent=1)
        out[c] = {k: v for k, v in rep.items() if k not in ("overlap_per_frame", "video_outline_size")}
        print(f"  {c}: overlap {rep['overlap_mean']} mean, {rep['overlap_worst']} worst; tears {rep.get('tears_pct', '?')}%")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
