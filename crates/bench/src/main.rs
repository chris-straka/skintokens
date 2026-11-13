//! creature-bench: a creature rig measured against a hand-made rig of
//! the same mesh.

mod anatomy;
mod bench;
mod follow;
mod rig;

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use skintokens::geom::{self, KdTree, V3};

use bench::{Drive, Truth};
use rig::Rig;

const HELP: &str =
    "creature-bench: a creature rig measured against a hand-made rig of the same mesh.

    creature-bench score TRUTH.glb ROUTE.glb [--json OUT.json] [--clips OUT.glb]
                         [--max-frames 48] [--sweeps 6] [--roles ROLES.json] [--label NAME]
    creature-bench rigid TRUTH.glb OUT.glb
    creature-bench info  RIG.glb [--roles ROLES.json]
    creature-bench anatomy RIG.glb [--json OUT.json]
    creature-bench joints TRUTH.glb OTHER.glb [--clip NAME] [--other-clip NAME] [--rest yes] [--json OUT.json]
    creature-bench follow RIG.glb --camera CAM.json --frames DIR --out FOLLOWED.glb
                         [--json OUT.json] [--name CLIP] [--fps 24] [--step 2] [--down 4]
                         [--steady 0.001] [--overlays DIR] [--camera-out CAM.json]

score  TRUTH is the artist's rig with its clips; ROUTE is a rig of the same
       mesh made by a rig route (it may have been turned by quarter turns,
       scaled or moved; its own clips are ignored). Prints one line and
       writes the full report as JSON:
       skeleton  for each of the artist's joints that hold skin, the
                 distance to the route's joint there, as a share of the
                 body's length (the mean is over the legs, neck, head,
                 jaw, tail and wings; the trunk's joints apart); the
                 chains the artist has and the route lacks; the route's
                 bones that lie along nothing.
       skin      the artist's clips are played on the artist's rig, then
                 followed by the route's rig as well as its bones allow.
                 First each of its bones is turned as the artist's bone
                 it lies along turns (`rotations carried`: a plain
                 retarget). Then, a few times over, each joint in turn
                 is rotated about its own place, everything below it
                 turning with it, to where the whole skin lands nearest
                 to the artist's; a step is kept only if the skin gets
                 nearer, and joints stay on their parents. What is left
                 is how far each vertex lands from the artist's: per
                 frame the mean and the 95th percentile over the
                 vertices, as shares of the body's length; then the mean
                 over a clip's frames and over the clips. `weights only`
                 carries each bone bodily with the artist's bone, so
                 joint places stop counting.
       feet      while one of the artist's feet is on the ground, how far
                 the route's foot is from it.
       --clips writes ROUTE with those clips on it (for the bend checker
       and for pictures), standing where TRUTH stands. --roles: {\"BoneName\": \"trunk|neck|head|jaw|leg|
       toe|tail|wing|other\"} for bones whose names don't say what they are.
rigid  a control: the artist's skeleton and clips with each vertex held
       wholly by its nearest bone. What crude weights on a perfect
       skeleton score.
info   the joints of a rig (role, parent, skin held) and its clips.
anatomy a skeleton judged with nothing to hold it against but itself: no
       artist's rig, and no reading of the body by the tool that made it.
       From the rig's own joints, the skin they hold and the mirror plane:
       every limb that branches off the centre line has a partner on the
       other side (joined at the mirrored place, as many joints give or
       take one, about as long); bones on the centre line stay near the
       mirror plane; a bone past the last limbs holds a head's worth of
       skin; a body whose trunk stands taller than it is long has a pair
       of limbs on the upper part of the trunk (arms); where the body
       stands on the ground off the mirror plane, the skin there is held
       by a side limb and not by the trunk; and a limb's name
       agrees with where it joins the trunk (a limb is a leg by joining
       at the hips, not by reaching the ground). A body that is itself
       lopsided (more than 2% of its length from its mirror image) gets
       its left-and-right findings as warnings. Exit 1 when any fails.
joints two files of one skeleton, each with a clip: how far OTHER's joints
       are from TRUTH's on every frame of TRUTH's clip (joints matched by
       name; millimetres, and shares of the body's height). For a motion
       read back from pictures of a known motion: the reader's error,
       joint by joint. --rest yes holds OTHER in its rest pose instead: what
       a reader that sees no motion at all would score.
follow a rig with no hand-made truth, judged against a video of the same
       creature filmed from the side by a still camera (CAM.json: the
       camera `tools/bench/render_side.py` wrote for the video's first
       picture; DIR: the video's frames as PNGs of that size, in order).
       Per frame the video's outline is what differs from its plain
       background. The camera is first lined up on the video's first
       frame (a video model zooms and moves the first picture a little,
       and the creature is already in motion there; --camera-out writes
       the camera as found). Then the rig's pose (each joint's turn in the picture's
       plane, and where the body is) starts at the frame before and each
       number is nudged, by a step that halves, for as long as the two
       outlines get nearer; turning far from the frame before and going
       under the ground are charged. The poses are then smoothed over
       time and lifted onto the ground. Reports the overlap of the two
       outlines per frame (pixels in both over pixels in either) and
       writes the rig with the followed motion as a clip. One camera
       sees no depth: a left leg and a right leg are told apart only by
       which fits, and nothing turns out of the picture's plane.";

struct Args {
    pos: Vec<String>,
    flags: HashMap<String, String>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut pos = vec![];
    let mut flags = HashMap::new();
    let mut k = 0;
    while k < args.len() {
        if let Some(name) = args[k].strip_prefix("--") {
            let v = args
                .get(k + 1)
                .ok_or_else(|| anyhow!("--{name} needs a value"))?;
            flags.insert(name.to_string(), v.clone());
            k += 2;
        } else {
            pos.push(args[k].clone());
            k += 1;
        }
    }
    Ok(Args { pos, flags })
}

fn roles_file(a: &Args) -> Result<Value> {
    match a.flags.get("roles") {
        Some(p) => Ok(serde_json::from_str(
            &std::fs::read_to_string(p).with_context(|| format!("reading {p}"))?,
        )?),
        None => Ok(json!({})),
    }
}

fn nearest(of: &[V3], among: &[V3]) -> Vec<usize> {
    let tree = KdTree::new(among.to_vec());
    of.iter().map(|p| tree.knn(*p, 1)[0].1).collect()
}

fn score(a: &Args) -> Result<()> {
    if a.pos.len() != 2 {
        bail!("score TRUTH.glb ROUTE.glb");
    }
    let truth_rig = Rig::read(Path::new(&a.pos[0]))?;
    let route = Rig::read(Path::new(&a.pos[1]))?;
    if truth_rig.clips.is_empty() {
        bail!("{} has no clips", a.pos[0]);
    }
    if route.joints.is_empty() {
        bail!("{} has no skin", a.pos[1]);
    }
    let max_frames: usize = a
        .flags
        .get("max-frames")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48);
    let al = bench::align(&truth_rig.rest_pos, &route.rest_pos);
    let truth = Truth::new(&truth_rig, &al, &roles_file(a)?);
    let t2r = nearest(&truth.rest_pos, &route.rest_pos);
    let r2t = nearest(&route.rest_pos, &truth.rest_pos);
    let map = bench::map_bones(&truth, &route, &r2t);
    let skeleton = bench::skeleton(&truth, &route, &map);
    let fr = bench::frames(&truth, max_frames);
    // the headline: the rig's best effort at the artist's motion
    let sweeps: usize = a
        .flags
        .get("sweeps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(6);
    let fz = bench::Fitter::new(&route, sweeps);
    let targets: Vec<Vec<V3>> = fr
        .truth_skin
        .iter()
        .map(|skin| r2t.iter().map(|&t| skin[t]).collect())
        .collect();
    // a plain retarget: each bone turned as the artist's bone it lies along
    let mapped = bench::skin_error(&truth, &route, &fr, &t2r, |f| {
        bench::drive(
            &truth,
            &route,
            &map.by_line,
            &fr.truth_joints[f],
            Drive::Turn,
        )
    });
    // the headline: from there, the rig's best effort at the artist's skin
    let turn = bench::skin_error(&truth, &route, &fr, &t2r, |f| {
        bench::fit(&route, &fz, &mapped.joints[f], &targets[f])
    });
    let carry = bench::skin_error(&truth, &route, &fr, &t2r, |f| {
        bench::drive(
            &truth,
            &route,
            &map.by_line,
            &fr.truth_joints[f],
            Drive::Carry,
        )
    });
    let feet = bench::feet(&truth, &fr, &turn, &t2r);

    let mut clips_note = Value::Null;
    if let Some(out) = a.flags.get("clips") {
        let mut clips = vec![];
        for (ci, clip) in truth_rig.clips.iter().enumerate() {
            let idx: Vec<usize> = (0..fr.time.len()).filter(|&f| fr.clip[f] == ci).collect();
            clips.push((
                clip.name.clone(),
                idx.iter().map(|&f| fr.time[f]).collect::<Vec<_>>(),
                idx.iter()
                    .map(|&f| turn.joints[f].clone())
                    .collect::<Vec<_>>(),
            ));
        }
        let mut g = route.with_clips(&clips)?;
        // The file stands where the artist's file stands: a route may have
        // turned, scaled or moved the mesh, and a node above everything
        // undoes that, so both files can be looked at through one camera.
        let back_to_artist = geom::inverse(&al.m);
        if !geom::is_identity(&al.m) {
            let scene = g.json["scene"].as_u64().unwrap_or(0) as usize;
            let roots = g.json["scenes"][scene]["nodes"].clone();
            let flat: Vec<f64> = back_to_artist
                .iter()
                .flat_map(|c| c.iter().copied())
                .collect();
            let nodes = g.json["nodes"]
                .as_array_mut()
                .ok_or_else(|| anyhow!("no nodes"))?;
            nodes.push(json!({"name": "as_the_artist_stands", "children": roots, "matrix": flat}));
            let top = nodes.len() - 1;
            g.json["scenes"][scene]["nodes"] = json!([top]);
        }
        g.write(Path::new(out))?;
        // read it back: the file must put the skin where the measurement did
        let back = Rig::read(Path::new(out))?;
        let turn_back = |p: V3| geom::apply(&al.m, p);
        let mut worst = 0.0f64;
        for (ci, clip) in back.clips.iter().enumerate() {
            for f in (0..fr.time.len()).filter(|&f| fr.clip[f] == ci) {
                let skin = back.skin_nodes(&back.pose_nodes(clip, fr.time[f]));
                for (v, p) in skin.iter().enumerate() {
                    worst = worst
                        .max(geom::norm(geom::sub(turn_back(*p), turn.skins[f][v])) / truth.len);
                }
            }
        }
        clips_note = json!({"file": out, "clips": back.clips.len(), "read_back_off_pct": (worst * 1e6).round() / 1e4});
        if worst > 1e-3 {
            eprintln!("warning: the written clips put the skin {:.3}% of the body's length from the measured one", worst * 100.0);
        }
    }

    let differ = (0..route.joints.len())
        .filter(|&j| map.by_line[j] != map.by_skin[j])
        .count();
    let bones: Vec<Value> = (0..route.joints.len())
        .map(|j| {
            json!({"bone": route.names[j], "follows": truth_rig.names[map.by_line[j]], "off_pct": (map.off[j] * 1e4).round() / 1e2,
                "by_skin": truth_rig.names[map.by_skin[j]], "skin_held": (route.mass[j] * 10.0).round() / 10.0})
        })
        .collect();
    let report = json!({
        "format": "creature-bench/1",
        "label": a.flags.get("label"),
        "truth": a.pos[0],
        "route": a.pos[1],
        "verts": {"artist": truth_rig.rest_pos.len(), "route": route.rest_pos.len()},
        "align": {"turned_deg": al.yaw_deg, "scale": (al.scale * 1e5).round() / 1e5,
            "mesh_off_mean_pct": (al.fit * 1e6).round() / 1e4, "mesh_off_max_pct": (al.fit_max * 1e6).round() / 1e4},
        "frames": fr.time.len(),
        "skeleton": skeleton,
        "skin": turn.json,
        "skin_weights_only": {"mean_pct": carry.json["mean_pct"], "p95_pct": carry.json["p95_pct"],
            "note": "each bone carried bodily by the artist's bone it follows: joint places no longer count"},
        "skin_rotations_carried": {"mean_pct": mapped.json["mean_pct"], "p95_pct": mapped.json["p95_pct"], "worst_frame": mapped.json["worst_frame"],
            "bones_that_hold_other_skin": differ,
            "note": "each bone turned as the artist's bone it lies along turns (a plain retarget): a rig that splits a limb differently is charged for it"},
        "feet": feet,
        "written": clips_note,
        "mapping": bones,
    });
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {p}"))?;
    }
    let s = &report["skeleton"];
    println!(
        "{}: limb joints off {}% (worst {}%, {} of {} without a joint; trunk {}%), missing {}, extra {}; skin off {}% mean, {}% p95 (rotations carried {}% / {}%; weights only {}% / {}%); worst {} at {} s under {}; feet off {}%, sink {}%; mesh fit {}%",
        a.flags.get("label").cloned().unwrap_or_else(|| a.pos[1].clone()),
        s["joint_error_pct"], s["joint_error_max_pct"], s["limb_joints_without_a_joint"], s["limb_joints"], s["trunk_joint_error_pct"],
        s["missing_chains"].as_array().map(|x| x.len()).unwrap_or(0), s["extra_chains"],
        report["skin"]["mean_pct"], report["skin"]["p95_pct"],
        report["skin_rotations_carried"]["mean_pct"], report["skin_rotations_carried"]["p95_pct"],
        report["skin_weights_only"]["mean_pct"], report["skin_weights_only"]["p95_pct"],
        report["skin"]["worst_frame"]["clip"], report["skin"]["worst_frame"]["time"], report["skin"]["worst_frame"]["where"],
        report["feet"]["off_mean_pct"], report["feet"]["sink_max_pct"], report["align"]["mesh_off_mean_pct"],
    );
    Ok(())
}

/// The artist's file with each vertex wholly on the bone it is nearest.
fn rigid(a: &Args) -> Result<()> {
    if a.pos.len() != 2 {
        bail!("rigid TRUTH.glb OUT.glb");
    }
    let rig = Rig::read(Path::new(&a.pos[0]))?;
    let al = bench::align(&rig.rest_pos, &rig.rest_pos);
    let truth = Truth::new(&rig, &al, &json!({}));
    let mut g = skintokens::glb::Glb {
        json: rig.glb.json.clone(),
        bin: rig.glb.bin.clone(),
    };
    let doc = rig.glb.doc()?;
    let mut v0 = 0usize;
    let mut seen = std::collections::HashSet::new();
    for ni in rig.glb.scene_nodes() {
        let node = doc.nodes().nth(ni).unwrap();
        let Some(mesh) = node.mesh() else { continue };
        if !seen.insert((mesh.index(), node.skin().map(|s| s.index()))) {
            continue;
        }
        let slots: Vec<usize> = node
            .skin()
            .map(|s| s.joints().map(|j| rig.joint_of[&j.index()]).collect())
            .unwrap_or_default();
        for (pi, prim) in mesh.primitives().enumerate() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let Some(n) = prim
                .reader(rig.glb.buffer())
                .read_positions()
                .map(|p| p.count())
            else {
                continue;
            };
            if !slots.is_empty() {
                let mut js = vec![[0u16; 4]; n];
                let ws = vec![[1.0f32, 0.0, 0.0, 0.0]; n];
                for v in 0..n {
                    let p = truth.rest_pos[v0 + v];
                    let mut best = (f64::INFINITY, 0usize);
                    for (slot, &j) in slots.iter().enumerate() {
                        if !truth.skin_bone[j] {
                            continue;
                        }
                        for &(la, lb) in &truth.lines[j] {
                            let ab = geom::sub(lb, la);
                            let l2 = geom::dot(ab, ab).max(1e-18);
                            let u = (geom::dot(geom::sub(p, la), ab) / l2).clamp(0.0, 1.0);
                            let d = geom::norm(geom::sub(p, geom::add(la, geom::scale(ab, u))));
                            if d < best.0 {
                                best = (d, slot);
                            }
                        }
                    }
                    js[v][0] = best.1 as u16;
                }
                let (ja, wa) = g.push_skin_attrs(&js, &ws);
                let attrs = g.json["meshes"][mesh.index()]["primitives"][pi]["attributes"]
                    .as_object_mut()
                    .unwrap();
                attrs.retain(|k, _| !k.starts_with("JOINTS_") && !k.starts_with("WEIGHTS_"));
                attrs.insert("JOINTS_0".into(), json!(ja));
                attrs.insert("WEIGHTS_0".into(), json!(wa));
            }
            v0 += n;
        }
    }
    g.write(Path::new(&a.pos[1]))?;
    println!("{}: every vertex on its nearest bone", a.pos[1]);
    Ok(())
}

fn info(a: &Args) -> Result<()> {
    if a.pos.len() != 1 {
        bail!("info RIG.glb");
    }
    let rig = Rig::read(Path::new(&a.pos[0]))?;
    let al = bench::align(&rig.rest_pos, &rig.rest_pos);
    let truth = Truth::new(&rig, &al, &roles_file(a)?);
    println!(
        "{}: {} vertices, {} triangles, {} joints, body length {:.3}",
        a.pos[0],
        rig.rest_pos.len(),
        rig.tris.len(),
        rig.joints.len(),
        truth.len
    );
    for &j in &rig.joint_order {
        println!(
            "  {:28} {:6} under {:24} holds {:7.1}",
            rig.names[j],
            truth.roles[j].name(),
            rig.parent[j].map(|p| rig.names[p].as_str()).unwrap_or("-"),
            rig.mass[j]
        );
    }
    for c in &rig.clips {
        let t = Rig::frame_times(c, usize::MAX);
        println!(
            "  clip {:24} {} frames, {:.2} s",
            c.name,
            t.len(),
            t.last().copied().unwrap_or(0.0)
        );
    }
    Ok(())
}

fn follow_cmd(a: &Args) -> Result<()> {
    if a.pos.len() != 1 {
        bail!("follow RIG.glb --camera CAM.json --frames DIR --out FOLLOWED.glb");
    }
    let need = |k: &str| a.flags.get(k).ok_or_else(|| anyhow!("--{k} is needed"));
    let rig = Rig::read(Path::new(&a.pos[0]))?;
    let cam = follow::Camera::read(Path::new(need("camera")?))?;
    let mut frames: Vec<std::path::PathBuf> = std::fs::read_dir(need("frames")?)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    frames.sort();
    if frames.len() < 2 {
        bail!("{}: fewer than two PNG frames", need("frames")?);
    }
    let num = |k: &str, d: f64| -> Result<f64> {
        Ok(a.flags.get(k).map(|s| s.parse()).transpose()?.unwrap_or(d))
    };
    let (fps, step, down, steady) = (
        num("fps", 24.0)?,
        num("step", 2.0)? as usize,
        num("down", 4.0)? as usize,
        num("steady", 0.001)?,
    );
    let name = a
        .flags
        .get("name")
        .cloned()
        .unwrap_or_else(|| "followed".into());
    let got = follow::follow(
        &rig,
        &cam,
        &frames,
        fps,
        step,
        down,
        steady,
        a.flags.get("overlays").map(Path::new),
    )?;
    let g = rig.with_clips(&[(name.clone(), got.times.clone(), got.joints.clone())])?;
    g.write(Path::new(need("out")?))?;
    if let Some(p) = a.flags.get("camera-out") {
        got.camera.write(Path::new(p), 0.62)?;
    }
    let mut report = got.json.clone();
    report["rig"] = json!(a.pos[0]);
    report["clip"] = json!(name);
    report["out"] = json!(need("out")?);
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {p}"))?;
    }
    println!(
        "{}: {} frames followed, outlines overlap {} mean, {} at worst ({} s); rest pose on the first frame {}; widest turn {}",
        name, report["frames"], report["overlap_mean"], report["overlap_worst"], report["overlap_worst_at_s"],
        report["overlap_rest_on_first_frame"], report["widest_turns_deg"][0]
    );
    Ok(())
}

fn joints_cmd(a: &Args) -> Result<()> {
    if a.pos.len() != 2 {
        bail!("joints TRUTH.glb OTHER.glb");
    }
    let truth = Rig::read(Path::new(&a.pos[0]))?;
    let other = Rig::read(Path::new(&a.pos[1]))?;
    let pick = |rig: &Rig, flag: &str, path: &str| -> Result<usize> {
        match a.flags.get(flag) {
            Some(n) => rig
                .clips
                .iter()
                .position(|c| &c.name == n)
                .ok_or_else(|| anyhow!("{path} has no clip {n}")),
            None if rig.clips.is_empty() => bail!("{path} has no clips"),
            None => Ok(0),
        }
    };
    let ct = pick(&truth, "clip", &a.pos[0])?;
    let rest = a.flags.get("rest").map(|s| s == "yes").unwrap_or(false);
    let co = if rest {
        0
    } else {
        pick(&other, "other-clip", &a.pos[1])?
    };
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &truth.rest_pos {
        lo = lo.min(p[1]);
        hi = hi.max(p[1]);
    }
    let height = (hi - lo).max(1e-9);
    let pairs: Vec<(usize, usize)> = (0..truth.joints.len())
        .filter_map(|j| {
            other
                .names
                .iter()
                .position(|n| *n == truth.names[j])
                .map(|k| (j, k))
        })
        .collect();
    if pairs.is_empty() {
        bail!("the two files share no joint names");
    }
    let times = Rig::frame_times(&truth.clips[ct], usize::MAX);
    let mut sum = vec![0.0f64; pairs.len()];
    let mut worst = vec![0.0f64; pairs.len()];
    let mut per_frame = vec![];
    for &t in &times {
        let wt = truth.pose_nodes(&truth.clips[ct], t);
        let wo = if rest {
            other.rest_world.clone()
        } else {
            other.pose_nodes(&other.clips[co], t)
        };
        let mut frame = 0.0;
        for (i, &(j, k)) in pairs.iter().enumerate() {
            let d = geom::norm(geom::sub(
                geom::translation(&wt[truth.joints[j]]),
                geom::translation(&wo[other.joints[k]]),
            ));
            sum[i] += d;
            worst[i] = worst[i].max(d);
            frame += d;
        }
        per_frame.push(frame / pairs.len() as f64);
    }
    let n = times.len().max(1) as f64;
    let mm = |x: f64| (x * 1e4).round() / 10.0;
    let mean = sum.iter().sum::<f64>() / n / pairs.len() as f64;
    let max = worst.iter().cloned().fold(0.0, f64::max);
    let mut joints: Vec<Value> = pairs
        .iter()
        .enumerate()
        .map(|(i, &(j, _))| json!({"joint": truth.names[j], "mean_mm": mm(sum[i] / n), "max_mm": mm(worst[i]), "mean_pct_of_height": ((sum[i] / n / height) * 1e4).round() / 1e2}))
        .collect();
    joints.sort_by(|x, y| {
        y["mean_mm"]
            .as_f64()
            .partial_cmp(&x["mean_mm"].as_f64())
            .unwrap()
    });
    let report = json!({
        "format": "creature-bench-joints/1",
        "truth": a.pos[0], "other": if rest { format!("{} at rest", a.pos[1]) } else { a.pos[1].clone() },
        "frames": times.len(), "joints_compared": pairs.len(), "height_m": (height * 1e4).round() / 1e4,
        "mean_mm": mm(mean), "max_mm": mm(max), "mean_pct_of_height": ((mean / height) * 1e4).round() / 1e2,
        "worst_frame_mean_mm": mm(per_frame.iter().cloned().fold(0.0, f64::max)),
        "joints": joints,
    });
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {p}"))?;
    }
    println!(
        "{}: joints {} mm from the truth on average ({}% of the height), {} mm at worst; farthest {} ({} mm)",
        report["other"].as_str().unwrap_or(""),
        report["mean_mm"],
        report["mean_pct_of_height"],
        report["max_mm"],
        report["joints"][0]["joint"].as_str().unwrap_or(""),
        report["joints"][0]["mean_mm"]
    );
    Ok(())
}

fn anatomy_cmd(a: &Args) -> Result<()> {
    if a.pos.len() != 1 {
        bail!("anatomy RIG.glb");
    }
    let rig = Rig::read(Path::new(&a.pos[0]))?;
    let mut report = anatomy::check(&rig);
    report["rig"] = json!(a.pos[0]);
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {p}"))?;
    }
    let fails: Vec<String> = report["findings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["severity"] == json!("fail"))
        .map(|f| {
            format!(
                "{} {}",
                f["code"].as_str().unwrap_or(""),
                f["detail"].as_str().unwrap_or("")
            )
        })
        .collect();
    println!(
        "{}: stands {}, {} ground patches off the plane, {} limb pairs, {} limbs without a partner, head {}: {}",
        a.pos[0],
        report["stands"].as_str().unwrap_or(""),
        report["ground_patches_off_the_plane"],
        report["limb_pairs"],
        report["limbs_without_a_partner"],
        if report["head"] == json!(true) { "yes" } else { "no" },
        if fails.is_empty() { "PASSES".to_string() } else { format!("FAILS: {}", fails.join("; ")) }
    );
    if !fails.is_empty() {
        FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let run = || -> Result<()> {
        let a = parse(&args[1..])?;
        match args[0].as_str() {
            "score" => score(&a),
            "rigid" => rigid(&a),
            "info" => info(&a),
            "follow" => follow_cmd(&a),
            "anatomy" => anatomy_cmd(&a),
            "joints" => joints_cmd(&a),
            other => bail!("unknown command {other}\n\n{HELP}"),
        }
    };
    match run() {
        Ok(()) if FAILED.load(std::sync::atomic::Ordering::Relaxed) => ExitCode::from(1),
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("creature-bench: {e:#}");
            ExitCode::from(2)
        }
    }
}
