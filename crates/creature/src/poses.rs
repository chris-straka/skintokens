//! A creature's range of motion as a weightforge pose set (`weights check
//! --poses`): what each part of this kind of body does, written on this
//! skeleton's bones. A leg swings, folds and spreads; a spine arches and
//! twists; a neck looks around; a tail swings; a wing beats and folds.
//! The four-legged moves are the ones `motionforge creature-gait --rom`
//! writes, so a rig is judged on the same motions whoever made it.

use crate::roles::{Anatomy, Kind, Role, Side};
use crate::skeleton::Skeleton;

type Pose = (String, Vec<(usize, String)>);

/// The pose set as RON. weightforge's frame is glTF's (+Y up, +Z forward).
pub fn rom(an: &Anatomy, sk: &Skeleton) -> String {
    let mut poses: Vec<Pose> = vec![];
    // a body the clean step left facing -Z bends "forward" toward -Z
    let dir = |d: &'static str| -> &'static str {
        if an.forward > 0.0 {
            return d;
        }
        match d {
            "Forward" => "Back",
            "Back" => "Forward",
            "Left" => "Right",
            "Right" => "Left",
            o => o,
        }
    };
    let bend = |d: &'static str, deg: f64| format!("Bend({}, {deg})", dir(d));
    let shared = |bones: &[usize], d: &'static str, total: f64| -> Vec<(usize, String)> {
        let each = (total / bones.len().max(1) as f64 * 10.0).round() / 10.0;
        bones.iter().map(|&b| (b, bend(d, each))).collect()
    };
    let chain = |limb: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..sk.bones.len()).filter(|&b| sk.bones[b].limb == Some(limb)).collect();
        v.sort_by_key(|&b| sk.bones[b].index);
        v
    };
    let level = !an.upright;
    let mut crouch: Vec<(usize, String)> = vec![];
    for (li, l) in an.limbs.iter().enumerate() {
        let c = chain(li);
        if c.is_empty() {
            continue;
        }
        let tag = l.name.replace('.', "_");
        let out = match l.side {
            Side::Left => "Left",
            Side::Right => "Right",
            Side::Centre => "Left",
        };
        match l.role {
            Role::Leg if level => {
                poses.push((format!("{tag}_forward"), vec![(c[0], bend("Forward", 35.0))]));
                poses.push((format!("{tag}_back"), vec![(c[0], bend("Back", 35.0))]));
                poses.push((format!("{tag}_out"), vec![(c[0], bend(out, 20.0))]));
                if c.len() > 2 {
                    let (a, b) = if l.front { ("Back", "Forward") } else { ("Forward", "Back") };
                    poses.push((format!("{tag}_fold"), vec![(c[1], bend(a, 45.0)), (c[2], bend(b, 60.0))]));
                    crouch.push((c[1], bend(a, 27.0)));
                    crouch.push((c[2], bend(b, 36.0)));
                } else if c.len() == 2 {
                    let a = if l.front { "Back" } else { "Forward" };
                    poses.push((format!("{tag}_fold"), vec![(c[1], bend(a, 45.0))]));
                    crouch.push((c[1], bend(a, 27.0)));
                }
                if c.len() > 3 {
                    poses.push((format!("{tag}_paw"), vec![(c[c.len() - 2], bend("Forward", 30.0))]));
                }
            }
            Role::Leg => {
                poses.push((format!("{tag}_forward"), vec![(c[0], bend("Forward", 60.0))]));
                poses.push((format!("{tag}_back"), vec![(c[0], bend("Back", 30.0))]));
                poses.push((format!("{tag}_out"), vec![(c[0], bend(out, 35.0))]));
                if c.len() > 1 {
                    poses.push((format!("{tag}_knee"), vec![(c[1], bend("Back", 90.0))]));
                    crouch.push((c[0], bend("Forward", 40.0)));
                    crouch.push((c[1], bend("Back", 70.0)));
                }
                if c.len() > 2 {
                    poses.push((format!("{tag}_foot"), vec![(c[c.len() - 1], bend("Up", 25.0))]));
                }
            }
            Role::Arm => {
                poses.push((format!("{tag}_forward"), vec![(c[0], bend("Forward", 60.0))]));
                poses.push((format!("{tag}_back"), vec![(c[0], bend("Back", 35.0))]));
                poses.push((format!("{tag}_out"), vec![(c[0], bend(out, 50.0))]));
                if c.len() > 1 {
                    poses.push((format!("{tag}_elbow"), vec![(c[1], bend("Forward", 80.0))]));
                }
                if c.len() > 2 {
                    poses.push((format!("{tag}_hand"), vec![(c[c.len() - 1], bend("Forward", 40.0))]));
                }
            }
            Role::Head => {
                let moves: [(&str, &'static str, f64); 4] = if level {
                    [("up", "Up", 45.0), ("down", "Down", 50.0), ("left", "Left", 45.0), ("right", "Right", 45.0)]
                } else {
                    [("forward", "Forward", 40.0), ("back", "Back", 30.0), ("left", "Left", 35.0), ("right", "Right", 35.0)]
                };
                for (what, d, total) in moves {
                    poses.push((format!("neck_{what}"), shared(&c, d, total)));
                }
                poses.push(("head_twist".into(), vec![(*c.last().unwrap(), "Twist(30.0)".into())]));
            }
            Role::Tail => {
                // a tail that hangs straight down lifts backward, not "up"
                let b = &sk.bones[c[0]];
                let d = skintokens::geom::sub(b.tail, b.head);
                let hangs = d[1].abs() > 0.85 * skintokens::geom::norm(d);
                let lift: [(&str, &'static str, f64); 2] = if hangs { [("up", "Back", 60.0), ("down", "Forward", 30.0)] } else { [("up", "Up", 60.0), ("down", "Down", 30.0)] };
                for (what, d, total) in lift.into_iter().chain([("left", "Left", 60.0), ("right", "Right", 60.0)]) {
                    poses.push((format!("{tag}_{what}"), shared(&c, d, total)));
                }
            }
            Role::Wing => {
                poses.push((format!("{tag}_up"), vec![(c[0], bend("Up", 45.0))]));
                poses.push((format!("{tag}_down"), vec![(c[0], bend("Down", 35.0))]));
                poses.push((format!("{tag}_forward"), vec![(c[0], bend("Forward", 25.0))]));
                poses.push((format!("{tag}_back"), vec![(c[0], bend("Back", 25.0))]));
                if c.len() > 2 {
                    poses.push((format!("{tag}_fold"), vec![(c[1], bend("Down", 40.0)), (c[2], bend("Down", 40.0))]));
                }
            }
            Role::Extra => {
                for (what, d) in [("forward", "Forward"), ("back", "Back"), ("left", "Left"), ("right", "Right")] {
                    poses.push((format!("{tag}_{what}"), shared(&c, d, 30.0)));
                }
            }
        }
    }
    // the spine: every trunk bone but the hips (turning them turns everything)
    let mut spine: Vec<usize> = (0..sk.bones.len()).filter(|&b| sk.bones[b].trunk && sk.bones[b].parent.is_some()).collect();
    spine.sort_by_key(|&b| sk.bones[b].index);
    if !spine.is_empty() {
        let moves: [(&str, &'static str, f64); 4] = if level {
            [("arch", "Up", 30.0), ("hollow", "Down", 20.0), ("left", "Left", 30.0), ("right", "Right", 30.0)]
        } else {
            [("forward", "Forward", 40.0), ("back", "Back", 25.0), ("left", "Left", 25.0), ("right", "Right", 25.0)]
        };
        for (what, d, total) in moves {
            poses.push((format!("spine_{what}"), shared(&spine, d, total)));
        }
        let each = (25.0 / spine.len() as f64 * 10.0).round() / 10.0;
        poses.push(("spine_twist".into(), spine.iter().map(|&b| (b, format!("Twist({each})"))).collect()));
    }
    if crouch.len() >= 4 {
        poses.push(("crouch".into(), crouch));
    }
    let mut s = String::new();
    s.push_str("// A creature's range of motion, written by `creature-rig` from the\n// parts of its body.\n");
    s.push_str(&format!("PoseSet(\n    name: \"{}\",\n    poses: [\n", an.kind.name()));
    for (pose, moves) in &poses {
        s.push_str(&format!("        PoseSpec(name: \"{pose}\", mirror: false, moves: [\n"));
        for (bone, motion) in moves {
            s.push_str(&format!("            Move(bone: \"{}\", motion: {motion}),\n", sk.bones[*bone].name));
        }
        s.push_str("        ]),\n");
    }
    s.push_str("    ],\n)\n");
    let _ = Kind::Legless;
    s
}
