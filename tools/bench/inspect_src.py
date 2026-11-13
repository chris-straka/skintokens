"""Blender: what a hand-made rigged model holds (bones, clips, size).
    blender -b --factory-startup -P inspect_src.py -- SRC
"""
import bpy, sys, json, os
from mathutils import Vector
src = sys.argv[sys.argv.index("--") + 1]
ext = os.path.splitext(src)[1].lower()
bpy.ops.wm.read_factory_settings(use_empty=True)
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext == ".blend":
    bpy.ops.wm.open_mainfile(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src)
out = {"src": os.path.basename(src), "objects": [], "actions": []}
for o in bpy.data.objects:
    d = {"name": o.name, "type": o.type, "parent": o.parent.name if o.parent else None,
         "scale": [round(x, 4) for x in o.matrix_world.to_scale()]}
    if o.type == "MESH":
        d["verts"] = len(o.data.vertices); d["tris"] = sum(len(p.vertices) - 2 for p in o.data.polygons)
        d["mats"] = [m.name if m else None for m in o.data.materials]
        d["images"] = sorted({n.image.name for m in o.data.materials if m and m.use_nodes for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image})
        cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
        d["lo"] = [round(min(c[i] for c in cs), 3) for i in range(3)]; d["hi"] = [round(max(c[i] for c in cs), 3) for i in range(3)]
        d["mods"] = [m.type for m in o.modifiers]
    if o.type == "ARMATURE":
        d["bones"] = [[b.name, b.parent.name if b.parent else None, b.use_deform,
                       [round(x, 3) for x in (o.matrix_world @ b.head_local)]] for b in o.data.bones]
    out["objects"].append(d)
for a in bpy.data.actions:
    out["actions"].append([a.name, [round(x, 1) for x in a.frame_range]])
out["fps"] = bpy.context.scene.render.fps
print("INSPECT " + json.dumps(out))
