"""Blender: a rigged creature from the side, painted, on a plain
background, through a camera that is written down so other tools can
look through the same one.

    blender -b --factory-startup -P render_side.py -- RIG.glb --out START.png --camera-out CAM.json
        [--width 1280] [--height 720] [--fill 0.42] [--at 0.42] [--ground 0.82] [--grey 0.62]
    blender -b --factory-startup -P render_side.py -- RIG.glb --camera CAM.json --clip NAME --frames DIR [--fps 24]

The first form frames the rest pose (the body `--fill` of the picture's
width at most, its middle `--at` across, the ground `--ground` down) and
writes the picture and the camera. The second renders every frame of a
clip of the file through a camera written earlier, as DIR/0000.png ...

The camera is orthographic and looks along glTF +X from the creature's
right, so a creature that faces glTF +Z faces the right of the picture:
pixel u = (z - centre_z) * px_per_m + width / 2,
pixel v = height / 2 - (y - centre_y) * px_per_m   (glTF y up, z forward).
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector


def options():
    argv = sys.argv[sys.argv.index("--") + 1:]
    o = {"rig": argv[0], "width": "1280", "height": "720", "fill": "0.42", "at": "0.42", "ground": "0.82", "grey": "0.62", "fps": "24"}
    k = 1
    while k < len(argv):
        o[argv[k][2:]] = argv[k + 1]
        k += 2
    for key in ("out", "camera-out", "camera", "frames"):
        if key in o:
            o[key] = os.path.abspath(o[key])
    return o


def main():
    o = options()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(float(o["fps"]))))
    known = set(bpy.ops.import_scene.gltf.get_rna_type().properties.keys())
    kw = {"disable_bone_shape": True} if "disable_bone_shape" in known else {}
    bpy.ops.import_scene.gltf(filepath=o["rig"], **kw)
    for x in [x for x in bpy.data.objects if x.type == "MESH" and x.name.startswith("Icosphere")]:
        bpy.data.objects.remove(x)
    arms = [x for x in bpy.data.objects if x.type == "ARMATURE"]
    for arm in arms:
        if arm.animation_data:
            for tr in list(arm.animation_data.nla_tracks):
                arm.animation_data.nla_tracks.remove(tr)
            arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()

    if "camera" in o:
        with open(o["camera"]) as f:
            cam = json.load(f)
    else:
        # frame the rest pose; Blender (x, y, z) = glTF (x, -z, y)
        dg = bpy.context.evaluated_depsgraph_get()
        zs, ys = [], []
        for ob in bpy.data.objects:
            if ob.type != "MESH":
                continue
            e = ob.evaluated_get(dg)
            me = e.to_mesh()
            for v in me.vertices:
                p = e.matrix_world @ v.co
                zs.append(-p.y)
                ys.append(p.z)
            e.to_mesh_clear()
        w, h = int(o["width"]), int(o["height"])
        length, height = max(zs) - min(zs), max(ys) - min(ys)
        ppm = min(float(o["fill"]) * w / length, 0.66 * h / height)
        cam = {
            "width": w, "height": h, "px_per_m": ppm,
            "centre_z": (max(zs) + min(zs)) / 2 + (0.5 - float(o["at"])) * w / ppm,
            "centre_y": min(ys) + (float(o["ground"]) - 0.5) * h / ppm,
            "ground_y": min(ys), "grey": float(o["grey"]),
            "looks_along": "+x", "right": "+z", "up": "+y",
        }
        with open(o["camera-out"], "w") as f:
            json.dump(cam, f, indent=1)

    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = max(cam["width"], cam["height"]) / cam["px_per_m"]
    cd.clip_end = 500.0
    co = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(co)
    right, up, back = Vector((0, -1, 0)), Vector((0, 0, 1)), Vector((-1, 0, 0))
    co.matrix_world = Matrix.Translation(Vector((-100.0, -cam["centre_z"], cam["centre_y"]))) @ Matrix((right, up, back)).transposed().to_4x4()
    scene.camera = co

    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = True
    scene.display.render_aa = "8"
    world = bpy.data.worlds.new("plain")
    g = cam.get("grey", 0.62)
    world.color = (g, g, g)
    scene.world = world
    scene.render.film_transparent = False
    scene.render.resolution_x = cam["width"]
    scene.render.resolution_y = cam["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"

    if "frames" in o:
        act = None
        for a in bpy.data.actions:
            if a.name == o["clip"] or a.name.startswith(o["clip"] + "_"):
                if act is None or len(a.name) < len(act.name):
                    act = a
        if act is None:
            raise SystemExit(f"no clip {o['clip']} in {o['rig']}: {[a.name for a in bpy.data.actions]}")
        for arm in arms:
            if arm.animation_data is None:
                arm.animation_data_create()
            arm.animation_data.action = act
            ad = arm.animation_data
            if hasattr(ad, "action_suitable_slots") and len(ad.action_suitable_slots):
                ad.action_slot = ad.action_suitable_slots[0]
        f0, f1 = act.frame_range
        os.makedirs(o["frames"], exist_ok=True)
        for k, f in enumerate(range(int(round(f0)), int(round(f1)) + 1)):
            scene.frame_set(f)
            scene.render.filepath = os.path.join(o["frames"], f"{k:04d}.png")
            bpy.ops.render.render(write_still=True)
        print("RENDER_SIDE", o["frames"], int(round(f1)) - int(round(f0)) + 1, "frames")
    else:
        scene.render.filepath = o["out"]
        bpy.ops.render.render(write_still=True)
        print("RENDER_SIDE", o["out"], json.dumps(cam))


main()
