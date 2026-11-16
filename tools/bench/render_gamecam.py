"""Headless Blender: render one creature rig at synchronized frames of one
clip through the game camera, with no text or diagnostic marks in the frame.

    blender -b --factory-startup -P tools/bench/render_gamecam.py -- \
        --rig PATH.glb --clip Walk [--out PREFIX] [--at 0.20,0.50,0.80] \
        [--times s1,s2,s3] [--paint-from REF.glb] \
        [--cam-spec u0,u1,v0,v1,dist,ground_v] [--measure-only no] \
        [--width 560] [--height 360] [--fps 24]

Why one rig per Blender invocation:
- Importing two glTF armatures into one Blender scene can share animation
  action slots across armatures. Rendering LEFT and RIGHT in separate clean
  headless scenes through one shared `--cam-spec` guarantees zero cross-rig
  contamination and pixel-identical (`RMSE == 0`) control renders.
- Camera matches the game's follow camera (`45 deg` vertical FOV, `-18.3 deg`
  downward pitch, `55 deg` front-three-quarter orbit so all limbs, head and
  tail silhouette read cleanly), scaled to the creature's stride bounds so
  the whole body fills ~68% of the frame height on a plain ground plane.
"""
import json
import math
import os
import struct
import sys

import bpy
from mathutils import Matrix, Vector


def options():
    argv = sys.argv[sys.argv.index("--") + 1:]
    o = {
        "at": "0.20,0.50,0.80",
        "width": "560",
        "height": "360",
        "fps": "24",
        "measure-only": "no",
    }
    k = 0
    while k < len(argv):
        key = argv[k][2:]
        o[key] = argv[k + 1]
        k += 2
    if "out" in o:
        o["out"] = os.path.abspath(o["out"])
    return o


def glb_materials(path):
    with open(path, "rb") as f:
        b = f.read()
    n = struct.unpack("<I", b[12:16])[0]
    doc = json.loads(b[20:20 + n])
    by_name = {m.get("name"): m for m in doc.get("materials", []) if m.get("name")}
    order = []
    if len(doc.get("meshes", [])) == 1:
        for pr in doc["meshes"][0].get("primitives", []):
            mi = pr.get("material")
            if mi is not None and mi not in order and mi < len(doc.get("materials", [])):
                order.append(doc["materials"][mi])
    return by_name, order


def opaque_paint(path, ref_path=None):
    """Restores flat baseColorFactor colours for Workbench (including alpha-0
    FBX exports and routes whose clean step dropped flat material factors)."""
    ref_by_name, ref_order = glb_materials(ref_path) if ref_path and os.path.exists(ref_path) else ({}, [])
    by_name, order = glb_materials(path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    slot_src = {}
    if len(meshes) == 1:
        for idx, slot in enumerate(meshes[0].material_slots):
            if slot.material:
                if idx < len(order):
                    slot_src[slot.material.name] = order[idx]
                if idx < len(ref_order):
                    slot_src.setdefault(slot.material.name + "#ref", ref_order[idx])
    for m in bpy.data.materials:
        stem = m.name if m.name in by_name else m.name.rsplit(".", 1)[0]
        src = slot_src.get(m.name) or by_name.get(stem)
        ref = slot_src.get(m.name + "#ref") or ref_by_name.get((src or {}).get("name", stem)) or ref_by_name.get(stem)
        c = list(m.diffuse_color)
        f4 = (src or {}).get("pbrMetallicRoughness", {}).get("baseColorFactor")
        has_tex = bool((src or {}).get("pbrMetallicRoughness", {}).get("baseColorTexture"))
        if f4 and (f4[:3] != [1.0, 1.0, 1.0] or not ref):
            c = [x ** (1.0 / 2.2) for x in f4[:3]] + [1.0]
        elif ref and not has_tex:
            rf4 = ref.get("pbrMetallicRoughness", {}).get("baseColorFactor")
            if rf4:
                c = [x ** (1.0 / 2.2) for x in rf4[:3]] + [1.0]
        elif f4:
            c = [x ** (1.0 / 2.2) for x in f4[:3]] + [1.0]
        m.diffuse_color = (c[0], c[1], c[2], 1.0)


def pick_clip(acts, wanted):
    if not acts:
        return None
    if wanted:
        low_w = wanted.lower()
        exact = [
            a for a in acts
            if a.name == wanted or a.name.startswith(wanted + "_") or a.name.split(".")[0] == wanted
        ]
        if exact:
            return min(exact, key=lambda a: len(a.name))
        sub = [a for a in acts if low_w in a.name.lower()]
        if sub:
            return min(sub, key=lambda a: len(a.name))
    for kw in ("walk", "trot", "run", "fly"):
        cand = [a for a in acts if kw in a.name.lower() and "slow" not in a.name.lower()]
        if cand:
            return min(cand, key=lambda a: len(a.name))
    return acts[0]


def bind_clip(arms, act):
    for arm in arms:
        if arm.animation_data is None:
            arm.animation_data_create()
        ad = arm.animation_data
        ad.action = act
        if act is not None and hasattr(ad, "action_suitable_slots") and len(ad.action_suitable_slots):
            slots = list(ad.action_suitable_slots)
            pick = next(
                (
                    s for s in slots
                    if getattr(s, "name_display", "") in (arm.name, arm.data.name)
                    or getattr(s, "identifier", "") in (arm.name, arm.data.name)
                ),
                None,
            )
            ad.action_slot = pick if pick is not None else slots[0]


def eval_bounds(objs, right, up, view):
    dg = bpy.context.evaluated_depsgraph_get()
    b = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    z_min = 1e9
    for o in objs:
        if o.type != "MESH":
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        for v in me.vertices:
            p = e.matrix_world @ v.co
            u, w, dd = p.dot(right), p.dot(up), p.dot(view)
            b = [min(b[0], u), max(b[1], u), min(b[2], w), max(b[3], w), min(b[4], dd), max(b[5], dd)]
            z_min = min(z_min, p.z)
        e.to_mesh_clear()
    return b, z_min


def set_exact_frame(scene, f_float):
    fi = int(math.floor(f_float))
    sub = max(0.0, min(0.9999, f_float - fi))
    scene.frame_set(fi, subframe=sub)


def main():
    o = options()
    path = os.path.abspath(o["rig"])
    fps = float(o["fps"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))

    known = set(bpy.ops.import_scene.gltf.get_rna_type().properties.keys())
    kw = {"disable_bone_shape": True} if "disable_bone_shape" in known else {}
    bpy.ops.import_scene.gltf(filepath=path, **kw)
    for x in [x for x in bpy.data.objects if x.type == "MESH" and x.name.startswith("Icosphere")]:
        bpy.data.objects.remove(x)
    opaque_paint(path, o.get("paint-from"))

    objs = [x for x in bpy.data.objects if x.type in ("MESH", "ARMATURE", "EMPTY")]
    arms = [x for x in objs if x.type == "ARMATURE"]
    acts = list(bpy.data.actions)
    for arm in arms:
        if arm.animation_data:
            for tr in list(arm.animation_data.nla_tracks):
                arm.animation_data.nla_tracks.remove(tr)
            arm.animation_data.action = None

    act = pick_clip(acts, o.get("clip", ""))
    bind_clip(arms, act)
    if act is not None:
        f0, f1 = float(act.frame_range[0]), float(act.frame_range[1])
    else:
        f0, f1 = 0.0, 1.0

    if "times" in o and o["times"]:
        f_list = [float(t) * fps for t in o["times"].split(",")]
    else:
        span = max(f1 - f0, 0.0)
        f_list = [f0 + span * float(x) for x in o["at"].split(",")]

    # HLH follow-camera geometry: 45 deg vertical FOV, -18.3 deg (-0.32 rad)
    # downward pitch, 55 deg front-three-quarter orbit.
    fov = math.radians(45.0)
    yaw, pitch = math.radians(55.0), math.radians(18.3)
    d = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    right = Vector((0.0, 0.0, 1.0)).cross(d).normalized() * -1.0
    up = d.cross(right).normalized() * -1.0

    box = [1e9, -1e9, 1e9, -1e9]
    ground_z = 1e9
    for ff in f_list:
        set_exact_frame(scene, ff)
        b, zm = eval_bounds(objs, right, up, d)
        box = [min(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), max(box[3], b[3])]
        ground_z = min(ground_z, zm)

    cell_w, cell_h = float(o["width"]), float(o["height"])
    if "cam-spec" in o and o["cam-spec"]:
        vals = [float(x) for x in o["cam-spec"].split(",")]
        box = vals[:4]
        dist, ground_z = vals[4], vals[5]
    else:
        # Whole figure fills ~68% of vertical frame height (matching 1.75m at 3.0m / 45deg).
        span_u = max(box[1] - box[0], 1e-3) / 0.72
        span_v = max(box[3] - box[2], 1e-3) / 0.68
        fit_w = span_u / (2.0 * math.tan(fov * 0.5) * (cell_w / cell_h))
        fit_h = span_v / (2.0 * math.tan(fov * 0.5))
        dist = max(fit_w, fit_h)

    spec = {
        "clip": act.name if act else "",
        "frame_range": [round(f0, 4), round(f1, 4)],
        "sampled_frames": [round(x, 4) for x in f_list],
        "sampled_times": [round(x / fps, 4) for x in f_list],
        "cam_spec": [round(box[0], 5), round(box[1], 5), round(box[2], 5), round(box[3], 5), round(dist, 5), round(ground_z, 5)],
    }
    print("RENDER_GAMECAM_SPEC " + json.dumps(spec))
    if o.get("measure-only") == "yes":
        return

    centre_u = 0.5 * (box[0] + box[1])
    centre_v = 0.5 * (box[2] + box[3])

    # World-horizontal ground plane at z = ground_z so foot planting and hover read honestly.
    span = max(box[1] - box[0], box[3] - box[2], 1.0) * 6.0
    ground = bpy.data.meshes.new("ground")
    ground.from_pydata(
        [(-span, -span, ground_z), (span, -span, ground_z), (span, span, ground_z), (-span, span, ground_z)],
        [],
        [(0, 1, 2, 3)],
    )
    gm = bpy.data.materials.new("ground_mat")
    gm.diffuse_color = (0.60, 0.62, 0.64, 1.0)
    ground.materials.append(gm)
    gobj = bpy.data.objects.new("ground", ground)
    scene.collection.objects.link(gobj)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = fov
    cam_data.clip_start = max(0.01, dist * 0.02)
    cam_data.clip_end = max(200.0, dist * 20.0)
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    eye = right * centre_u + up * centre_v + d * dist
    cam.matrix_world = Matrix.Translation(eye) @ Matrix((right, up, d)).transposed().to_4x4()
    scene.camera = cam

    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = True
    scene.display.render_aa = "8"
    world = bpy.data.worlds.new("plain")
    world.color = (0.85, 0.86, 0.88)
    scene.world = world
    scene.render.film_transparent = False
    scene.render.resolution_x = int(cell_w)
    scene.render.resolution_y = int(cell_h)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"

    for i, ff in enumerate(f_list):
        set_exact_frame(scene, ff)
        out_png = f"{o['out']}_{i + 1}.png"
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        print("RENDER_GAMECAM_FRAME", out_png, round(ff / fps, 4))


if __name__ == "__main__":
    main()
