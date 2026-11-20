"""Blender: merge a Mesh2Motion rigged model.glb with one or more animation clip GLBs.

    blender -b --factory-startup -P merge_clips.py -- MODEL.glb OUT.glb CLIP1.glb [CLIP2.glb ...]

Keeps the armature and skinned meshes from MODEL.glb, imports each CLIP*.glb
to extract its animation Action, binds it to the model's armature (including
Blender 4.4+/5.x action slots), and exports one combined rigged+animated GLB.
"""
import os
import sys
import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) < 3:
        raise SystemExit(__doc__)
    model_path = argv[0]
    out_path = argv[1]
    clip_paths = argv[2:]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=model_path)

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit(f"no armature in {model_path}")
    arm = arms[0]
    if arm.animation_data is None:
        arm.animation_data_create()

    keep_objs = set(bpy.data.objects)
    kept_actions = []

    for clip_path in clip_paths:
        before_objs = set(bpy.data.objects)
        before_acts = set(bpy.data.actions)
        bpy.ops.import_scene.gltf(filepath=clip_path)
        new_objs = [o for o in bpy.data.objects if o not in before_objs]
        new_acts = [a for a in bpy.data.actions if a not in before_acts]

        stem = os.path.splitext(os.path.basename(clip_path))[0]
        if new_acts:
            act = new_acts[0]
            act.name = stem
            act.use_fake_user = True
            # Bind action + slot (Blender 4.4+/5.x compatibility)
            arm.animation_data.action = act
            if hasattr(act, "slots") and len(act.slots) > 0:
                try:
                    arm.animation_data.action_slot = act.slots[0]
                except Exception:
                    pass
            track = arm.animation_data.nla_tracks.new()
            track.name = stem
            f0 = int(round(act.frame_range[0]))
            strip = track.strips.new(stem, f0, act)
            if hasattr(strip, "action_slot") and hasattr(act, "slots") and len(act.slots) > 0:
                try:
                    strip.action_slot = act.slots[0]
                except Exception:
                    pass
            kept_actions.append(act)

        for o in new_objs:
            if o not in keep_objs:
                bpy.data.objects.remove(o, do_unlink=True)

    if kept_actions:
        arm.animation_data.action = kept_actions[0]
        if hasattr(kept_actions[0], "slots") and len(kept_actions[0].slots) > 0:
            try:
                arm.animation_data.action_slot = kept_actions[0].slots[0]
            except Exception:
                pass

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for o in bpy.data.objects:
        o.select_set(o in keep_objs)
    bpy.context.view_layer.objects.active = arm

    known = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    kw = dict(
        filepath=out_path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_skins=True,
        export_all_influences=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_frame_step=1,
        export_anim_single_armature=True,
        export_reset_pose_bones=True,
        export_rest_position_armature=True,
    )
    bpy.ops.export_scene.gltf(**{k: v for k, v in kw.items() if k in known})
    print(f"MERGED {os.path.basename(out_path)} actions={[a.name for a in kept_actions]}")


if __name__ == "__main__":
    main()
