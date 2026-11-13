"""Blender: a hand-made rigged, animated animal -> the benchmark's truth.

    blender -b --factory-startup -P prep_truth.py -- SRC OUT_DIR --name NAME
        [--armature NAME] [--turn DEG] [--length 1.5] [--max-frames 48]

Writes into OUT_DIR:

  truth.glb   the artist's rig as it is (every bone, every weight, every
              clip sampled on each frame), turned to face +Z (glTF), feet
              on the ground, its longest side scaled to --length metres.
  bare.glb    the same mesh in its rest pose with nothing else: no bones,
              no weights, no clips. Its materials and UVs are kept (the
              paint). This is what a rig route is given.
  truth.json  what was found: bones, clips, sizes, how it was turned.

Nothing of the artist's rig is changed: the turn and the scale sit on a
parent node above it. SRC is an .fbx, .blend, .glb or .gltf. A file with
more than one armature keeps the one that carries the most skin (or
--armature); meshes skinned to another armature are left out and named
in truth.json.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    a = {"src": argv[0], "out": argv[1], "name": "", "armature": "", "turn": None, "length": 1.5, "max_frames": 48}
    k = 2
    while k < len(argv):
        key = argv[k][2:].replace("-", "_")
        a[key] = argv[k + 1]
        k += 2
    a["length"] = float(a["length"])
    a["max_frames"] = int(a["max_frames"])
    if a["turn"] is not None:
        a["turn"] = float(a["turn"])
    return a


def skinned_to(arm):
    return [o for o in bpy.data.objects if o.type == "MESH"
            and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]


def export(path, **kw):
    """The glTF exporter's options differ a little between versions: pass
    only the ones this Blender has."""
    known = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    bpy.ops.export_scene.gltf(filepath=path, **{k: v for k, v in kw.items() if k in known})


def main():
    a = args()
    src = a["src"]
    ext = os.path.splitext(src)[1].lower()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=src)
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=src)
    else:
        bpy.ops.import_scene.gltf(filepath=src)
    scene = bpy.context.scene
    name = a["name"] or os.path.splitext(os.path.basename(src))[0].lower()
    os.makedirs(a["out"], exist_ok=True)

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature in " + src)
    arm = max(arms, key=lambda o: (o.name == a["armature"], sum(len(m.data.vertices) for m in skinned_to(o))))
    meshes = skinned_to(arm)
    if not meshes:
        raise SystemExit("the armature carries no skin")
    left_out = [o.name for o in bpy.data.objects if o.type == "MESH" and o not in meshes]

    # One mesh, nothing on it but the armature.
    for o in bpy.data.objects:
        o.select_set(False)
    for m in meshes:
        m.hide_set(False)
        m.select_set(True)
        for md in list(m.modifiers):
            if md.type != "ARMATURE":
                m.modifiers.remove(md)
        if m.data.shape_keys:
            m.shape_key_clear()
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active

    # The rest pose, where the body is and which way it faces.
    actions = [x for x in bpy.data.actions]
    if arm.animation_data:
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    pts = [ev.matrix_world @ v.co for v in ev.data.vertices]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    centre = (lo + hi) / 2
    # Facing: the head is in front. Blender's "front" is -Y, which glTF
    # writes as +Z.
    head = None
    for b in arm.data.bones:
        n = b.name.lower()
        if "head" in n and "_end" not in n:
            head = arm.matrix_world @ b.head_local
            break
    turn = a["turn"]
    facing = "given"
    if turn is None:
        turn = 0.0
        facing = "no head bone: left as it is"
        if head is not None:
            d = head - centre
            if abs(d.y) >= abs(d.x):
                turn = 180.0 if d.y > 0 else 0.0
            else:
                turn = -90.0 if d.x > 0 else 90.0
            facing = "head bone"
    longest = max(hi - lo)
    s = a["length"] / longest
    rot = Matrix.Rotation(math.radians(turn), 4, "Z")
    # ground and centre after the turn
    tp = [rot @ (p * s) for p in pts]
    tlo = Vector([min(p[i] for p in tp) for i in range(3)])
    thi = Vector([max(p[i] for p in tp) for i in range(3)])
    move = Matrix.Translation(Vector((-(tlo.x + thi.x) / 2, -(tlo.y + thi.y) / 2, -tlo.z)))
    norm = move @ rot @ Matrix.Scale(s, 4)

    root = bpy.data.objects.new(name, None)
    scene.collection.objects.link(root)
    for o in [arm] + ([mesh] if mesh.parent is None else []):
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_world = mw
    root.matrix_world = norm
    bpy.context.view_layer.update()

    # The bare mesh: the rest pose, baked, alone.
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    bare_me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    bare_me.transform(ev.matrix_world)
    bare = bpy.data.objects.new(name + "_bare", bare_me)
    scene.collection.objects.link(bare)
    for g in list(bare.vertex_groups):
        bare.vertex_groups.remove(g)
    for o in bpy.data.objects:
        o.select_set(False)
    bare.select_set(True)
    bpy.context.view_layer.objects.active = bare
    common = dict(export_format="GLB", use_selection=True, export_yup=True, export_apply=False,
                  export_cameras=False, export_lights=False, export_morph=False, export_extras=False)
    export(os.path.join(a["out"], "bare.glb"), export_skins=False, export_animations=False, **common)
    bare_stats = {"verts": len(bare_me.vertices), "tris": sum(len(p.vertices) - 2 for p in bare_me.polygons)}
    bpy.data.objects.remove(bare)

    # The truth: the rig and every clip, sampled on each frame.
    arm.data.pose_position = "POSE"
    clips = []
    for act in actions:
        f0, f1 = act.frame_range
        clips.append({"name": act.name.split("|")[-1], "action": act.name, "frames": int(round(f1 - f0)) + 1})
        act.name = act.name.split("|")[-1]
    if actions and arm.animation_data is None:
        arm.animation_data_create()
    for o in bpy.data.objects:
        o.select_set(False)
    for o in (root, arm, mesh):
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    export(os.path.join(a["out"], "truth.glb"), export_skins=True, export_all_influences=True,
           export_def_bones=False, export_animations=True, export_animation_mode="ACTIONS",
           export_force_sampling=True, export_frame_step=1, export_optimize_animation_size=False,
           export_optimize_animation_keep_anim_armature=True, export_anim_single_armature=True,
           export_reset_pose_bones=True, export_rest_position_armature=True, export_bake_animation=False,
           export_anim_slide_to_zero=True, export_negative_frame="SLIDE", **common)

    info = {
        "name": name, "source": os.path.basename(src), "armature": arm.name, "left_out": left_out,
        "bones": len(arm.data.bones), "verts": bare_stats["verts"], "tris": bare_stats["tris"],
        "materials": [m.name for m in mesh.data.materials if m],
        "images": sorted({n.image.name for m in mesh.data.materials if m and m.use_nodes
                          for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}),
        "scale": round(s, 6), "turned_deg": turn, "facing_from": facing, "length_m": a["length"],
        "size_m": [round(x * s, 3) for x in (hi - lo)], "fps": scene.render.fps, "clips": clips,
    }
    with open(os.path.join(a["out"], "truth.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("PREP_TRUTH " + json.dumps(info))


main()
