"""Blender: several rigs of one mesh side by side on the same frame of
the same clip, painted, from one camera.

    blender -b --factory-startup -P render_compare.py -- --out SHEET.png
        --rig "LABEL=path.glb" [--rig ...] --clip NAME --time SECONDS
        [--loop OUT.mp4 [--loop-clip NAME] [--loop-frames N]] [--view side|front34|back34]
        [--width 640] [--height 560] [--fps 24]

Every rig file carries the clip under the same name (the artist's file
as it is; a route's file as `creature-bench score --clips` wrote it).
The sheet is one picture at --time. --loop also renders the whole of a
clip as a video (needs ffmpeg), the rigs in the same order.
The paint is each file's own materials and textures, lit flat
(Workbench), on a plain background.
"""
import math
import os
import subprocess
import sys

import bpy
from mathutils import Matrix, Vector


def options():
    argv = sys.argv[sys.argv.index("--") + 1:]
    o = {"rig": [], "view": "front34", "width": "640", "height": "560", "fps": "24", "loop-frames": "0"}
    k = 0
    while k < len(argv):
        key = argv[k][2:]
        if key == "rig":
            o["rig"].append(argv[k + 1])
        else:
            o[key] = argv[k + 1]
        k += 2
    for key in ("out", "loop"):
        if key in o:
            o[key] = os.path.abspath(o[key])
    return o


PAINT = {}


def opaque_paint(path, before):
    """A file's flat material colours as Workbench shows them. Old FBX
    files carry an alpha of 0 on every material, which a GLB keeps: the
    colour is right and the model is invisible. The colour is read from
    the file (the importer hides it behind nodes when the mesh also has
    vertex colours) and made opaque."""
    import json as _json
    import struct as _struct
    with open(path, "rb") as f:
        b = f.read()
    n = _struct.unpack("<I", b[12:16])[0]
    doc = _json.loads(b[20:20 + n])
    mats = {m.get("name"): m for m in doc.get("materials", [])}
    # Which of the file's materials a Blender material is: by its slot on
    # the mesh when the file has one mesh (Blender renames a material
    # whose name is taken: "Material.003" comes in as "Material.004"),
    # else by its name.
    by_slot = {}
    new_meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(s.material and s.material not in before for s in o.material_slots)]
    if len(doc.get("meshes", [])) == 1 and len(new_meshes) == 1:
        order = []
        for pr in doc["meshes"][0].get("primitives", []):
            k = pr.get("material")
            if k is not None and k not in order:
                order.append(k)
        for slot, k in zip(new_meshes[0].material_slots, order):
            if slot.material:
                by_slot[slot.material.name] = doc["materials"][k]
    for m in bpy.data.materials:
        if m in before:
            continue
        name = m.name if m.name in mats else m.name.rsplit(".", 1)[0]
        src = by_slot.get(m.name) or mats.get(name)
        if m.name in by_slot:
            name = src.get("name", name)
        c = list(m.diffuse_color)
        f4 = src.get("pbrMetallicRoughness", {}).get("baseColorFactor") if src else None
        if f4:
            # the file's colour is linear; the viewport colour is shown as it is
            c = [x ** (1 / 2.2) for x in f4[:3]] + [1.0]
            PAINT.setdefault(name, c)
        elif name in PAINT and not (src or {}).get("pbrMetallicRoughness", {}).get("baseColorTexture"):
            # a route's clean step can drop a flat colour: the first
            # file's material of the same name is the same paint
            c = PAINT[name]
        m.diffuse_color = (c[0], c[1], c[2], 1.0)


def load(path, clip_names):
    before_o = set(bpy.data.objects)
    before_a = set(bpy.data.actions)
    before_m = set(bpy.data.materials)
    known = set(bpy.ops.import_scene.gltf.get_rna_type().properties.keys())
    kw = {"disable_bone_shape": True} if "disable_bone_shape" in known else {}
    bpy.ops.import_scene.gltf(filepath=path, **kw)
    # the importer's own bone-shape helper is not part of the model
    for x in [x for x in bpy.data.objects if x not in before_o and x.type == "MESH" and x.name.startswith("Icosphere")]:
        bpy.data.objects.remove(x)
    opaque_paint(path, before_m)
    objs = [x for x in bpy.data.objects if x not in before_o]
    acts = [x for x in bpy.data.actions if x not in before_a]
    arms = [x for x in objs if x.type == "ARMATURE"]
    for arm in arms:
        if arm.animation_data:
            for tr in list(arm.animation_data.nla_tracks):
                arm.animation_data.nla_tracks.remove(tr)
            arm.animation_data.action = None
    return {"objs": objs, "arms": arms, "acts": acts}


def set_clip(rig, name):
    """The rig's action for a clip of that name (the importer may add the
    object's name to it)."""
    best = None
    for a in rig["acts"]:
        if a.name == name or a.name.startswith(name + "_") or a.name.split(".")[0] == name:
            if best is None or len(a.name) < len(best.name):
                best = a
    for arm in rig["arms"]:
        if arm.animation_data is None:
            arm.animation_data_create()
        ad = arm.animation_data
        ad.action = best
        if best is not None and hasattr(ad, "action_suitable_slots") and len(ad.action_suitable_slots):
            ad.action_slot = ad.action_suitable_slots[0]
    return best


def bounds(objs, right, up):
    """How far the meshes reach across and up the picture."""
    dg = bpy.context.evaluated_depsgraph_get()
    b = [1e9, -1e9, 1e9, -1e9]
    for o in objs:
        if o.type != "MESH":
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        for v in me.vertices:
            p = e.matrix_world @ v.co
            u, w = p.dot(right), p.dot(up)
            b = [min(b[0], u), max(b[1], u), min(b[2], w), max(b[3], w)]
        e.to_mesh_clear()
    return b


def main():
    o = options()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = int(o["fps"])
    scene.render.fps = fps
    rigs = []
    for spec in o["rig"]:
        label, path = spec.split("=", 1)
        r = load(path, None)
        r["label"] = label
        rigs.append(r)
    n = len(rigs)

    # the camera: orthographic, a little above, turned toward the front
    yaw = {"side": 90.0, "front34": 55.0, "back34": 125.0}[o["view"]]
    pitch = 12.0
    d = Vector((math.sin(math.radians(yaw)) * math.cos(math.radians(pitch)),
                -math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)),
                math.sin(math.radians(pitch))))
    right = Vector((0, 0, 1)).cross(d).normalized() * -1.0
    up = d.cross(right).normalized() * -1.0

    # how much room one rig needs: its bounds over the frames to be shown
    clip = o["clip"]
    frame = int(round(float(o["time"]) * fps))
    shown = [(clip, frame)]
    loop_clip = o.get("loop-clip", clip)
    loop_frames = []
    if "loop" in o:
        for r in rigs:
            a = set_clip(r, loop_clip)
            if a is not None:
                f0, f1 = a.frame_range
                loop_frames = list(range(int(round(f0)), int(round(f1)) + 1))
                break
        if int(o["loop-frames"]) and len(loop_frames) > int(o["loop-frames"]):
            loop_frames = loop_frames[:int(o["loop-frames"])]
        shown += [(loop_clip, f) for f in loop_frames[::max(1, len(loop_frames) // 8)]]
    box = [1e9, -1e9, 1e9, -1e9]
    for c, f in shown:
        set_clip(rigs[0], c)
        scene.frame_set(f)
        b = bounds(rigs[0]["objs"], right, up)
        box = [min(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), max(box[3], b[3])]
    w = (box[1] - box[0]) * 1.1
    h = (box[3] - box[2]) * 1.1
    cell_w, cell_h = float(o["width"]), float(o["height"])
    label_h = 0.05 * max(w, h)
    scale_per_cell = max(w, (h + 3.2 * label_h) * cell_w / cell_h)
    centre_u = (box[0] + box[1]) / 2
    low = box[2] - (h - (box[3] - box[2])) / 2
    centre_v = (low - 3.2 * label_h + box[3] + (h - (box[3] - box[2])) / 2) / 2

    black = bpy.data.materials.new("label")
    black.diffuse_color = (0.05, 0.05, 0.05, 1.0)
    for k, r in enumerate(rigs):
        off = right * ((k - (n - 1) / 2) * scale_per_cell)
        for ob in r["objs"]:
            if ob.parent is None:
                ob.matrix_world = Matrix.Translation(off) @ ob.matrix_world
        cu = bpy.data.curves.new(r["label"], "FONT")
        cu.body = r["label"]
        cu.align_x = "CENTER"
        cu.size = label_h
        t = bpy.data.objects.new("label_" + r["label"], cu)
        scene.collection.objects.link(t)
        cu.materials.append(black)
        rot = Matrix((right, up, d)).transposed().to_4x4()
        pos = off + right * centre_u + up * (low - 2.2 * label_h) + d * 3.0
        t.matrix_world = Matrix.Translation(pos) @ rot

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = scale_per_cell * n
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    at = right * centre_u + up * centre_v + d * 50.0
    cam.matrix_world = Matrix.Translation(at) @ Matrix((right, up, d)).transposed().to_4x4()
    cam_data.clip_end = 200.0
    scene.camera = cam

    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = True
    scene.display.render_aa = "8"
    world = bpy.data.worlds.new("plain")
    world.color = (0.86, 0.87, 0.88)
    scene.world = world
    scene.render.film_transparent = False
    scene.render.resolution_x = int(cell_w) * n
    scene.render.resolution_y = int(cell_h)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"

    for r in rigs:
        set_clip(r, clip)
    scene.frame_set(frame)
    scene.render.filepath = o["out"]
    bpy.ops.render.render(write_still=True)

    if "loop" in o and loop_frames:
        for r in rigs:
            set_clip(r, loop_clip)
        tmp = o["loop"] + ".frames"
        os.makedirs(tmp, exist_ok=True)
        for k, f in enumerate(loop_frames):
            scene.frame_set(f)
            scene.render.filepath = os.path.join(tmp, f"{k:04d}.png")
            bpy.ops.render.render(write_still=True)
        # twice round, so a short clip is still long enough to read
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "2", "-framerate", str(fps), "-i", os.path.join(tmp, "%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", o["loop"]]
        try:
            subprocess.run(cmd, check=True)
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)
        except (OSError, subprocess.CalledProcessError) as e:
            print("RENDER_COMPARE no video:", e, "(frames kept in", tmp + ")")
    print("RENDER_COMPARE", o["out"])


main()
