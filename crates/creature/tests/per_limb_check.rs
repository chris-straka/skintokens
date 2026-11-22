//! Per-limb skeleton check tests (skintokens-6).
//!
//! Verifies that `creature-rig check` scores each limb explicitly (`score`,
//! `role`, `pass`, top-level `min_limb_score`), catches a one-limb falloff via
//! per-limb score + symmetry even when bone counts differ by only 1, and still
//! passes a healthy 3-vs-2 bone pair whose coverage is symmetric.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use skintokens_geom::geom;
use skintokens_geom::glb::{Frame, Glb};
use skintokens_geom::rig_edit::attach_skeleton;

fn rig_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_creature-rig"))
}

/// Same watertight 8-box quadruped proportions as `crates/creature/src/tests.rs`.
fn box_dog_glb() -> Glb {
    let boxes: [([f32; 3], [f32; 3]); 8] = [
        ([-0.15, 0.45, -0.45], [0.15, 0.75, 0.45]),  // body
        ([0.07, 0.0, 0.30], [0.15, 0.50, 0.40]),     // front left leg
        ([-0.15, 0.0, 0.30], [-0.07, 0.50, 0.40]),   // front right leg
        ([0.07, 0.0, -0.40], [0.15, 0.50, -0.30]),   // hind left leg
        ([-0.15, 0.0, -0.40], [-0.07, 0.50, -0.30]), // hind right leg
        ([-0.06, 0.65, 0.40], [0.06, 0.95, 0.52]),   // neck
        ([-0.08, 0.85, 0.48], [0.08, 1.0, 0.75]),    // head
        ([-0.03, 0.62, -0.85], [0.03, 0.68, -0.40]), // tail
    ];
    let (mut pos, mut idx) = (Vec::<f32>::new(), Vec::<u16>::new());
    let n = 2usize;
    let faces = [
        (0, false, 2, 1),
        (0, true, 1, 2),
        (1, false, 0, 2),
        (1, true, 2, 0),
        (2, false, 1, 0),
        (2, true, 0, 1),
    ];
    for (lo, hi) in boxes {
        for (fa, is_hi, ua, va) in faces {
            let base = (pos.len() / 3) as u16;
            let fv = if is_hi { hi[fa] } else { lo[fa] };
            for iu in 0..=n {
                for iv in 0..=n {
                    let mut p = [0.0f32; 3];
                    p[fa] = fv;
                    p[ua] = lo[ua] + (hi[ua] - lo[ua]) * (iu as f32 / n as f32);
                    p[va] = lo[va] + (hi[va] - lo[va]) * (iv as f32 / n as f32);
                    pos.extend(p);
                }
            }
            let stride = (n + 1) as u16;
            for iu in 0..n as u16 {
                for iv in 0..n as u16 {
                    let a = base + iu * stride + iv;
                    let b = base + (iu + 1) * stride + iv;
                    let c = base + (iu + 1) * stride + iv + 1;
                    let d = base + iu * stride + iv + 1;
                    idx.extend([a, b, c, a, c, d]);
                }
            }
        }
    }
    let nv = pos.len() / 3;
    let mut mn = vec![f64::MAX; 3];
    let mut mx = vec![f64::MIN; 3];
    for k in 0..nv {
        for a in 0..3 {
            mn[a] = mn[a].min(pos[k * 3 + a] as f64);
            mx[a] = mx[a].max(pos[k * 3 + a] as f64);
        }
    }
    let mut g = Glb {
        json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
        bin: vec![],
    };
    let pb: Vec<u8> = pos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let pa = g
        .push_accessor(&pb, 5126, "VEC3", nv, Some((mn, mx)), false)
        .unwrap();
    let ia = g
        .push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false)
        .unwrap();
    g.json["meshes"] =
        json!([{"name": "dog", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "dog", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);
    g
}

fn run_check(glb: &Path, json_out: &Path) -> (i32, Value) {
    let out = Command::new(rig_bin())
        .args([
            "check",
            glb.to_str().unwrap(),
            "--res",
            "96",
            "--json",
            json_out.to_str().unwrap(),
        ])
        .output()
        .expect("run creature-rig check");
    let code = out.status.code().unwrap_or(-1);
    let text = std::fs::read_to_string(json_out).unwrap_or_else(|e| {
        panic!(
            "check wrote no JSON ({e}); exit {code}; stderr={}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (code, serde_json::from_str(&text).expect("parse check json"))
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skintokens6-check-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rig_box_dog(d: &Path) -> PathBuf {
    let bare = d.join("bare.glb");
    let rigged = d.join("rigged.glb");
    box_dog_glb().write(&bare).unwrap();
    let out = Command::new(rig_bin())
        .args([
            "rig",
            bare.to_str().unwrap(),
            rigged.to_str().unwrap(),
            "--res",
            "96",
            "--turn",
            "no",
        ])
        .output()
        .expect("run creature-rig rig");
    assert!(
        out.status.success(),
        "creature-rig rig failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    rigged
}

/// Strip skin weights from distal joint(s) of `leg_front.L` onto joint 0 (hips),
/// leaving only its single proximal bone deforming while `leg_front.R` keeps >=2.
fn strip_front_left_distal(src: &Path, dst: &Path) {
    let mut g = Glb::read(src).unwrap();
    let info = g.skin_info().unwrap();
    let mut drop_slots = std::collections::HashSet::new();
    let mut front_l: Vec<usize> = info
        .names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.contains("leg_front") && n.ends_with(".L"))
        .map(|(i, _)| i)
        .collect();
    front_l.sort();
    assert!(
        front_l.len() >= 2,
        "expected >=2 bones on leg_front.L, got {:?}",
        info.names
    );
    // Keep only the top bone (whose parent is not also in front_l); strip all descendants.
    let set: std::collections::HashSet<usize> = front_l.iter().copied().collect();
    let root_fl = front_l
        .iter()
        .copied()
        .find(|&i| info.parents[i] < 0 || !set.contains(&(info.parents[i] as usize)))
        .unwrap_or(front_l[0]);
    for &ji in &front_l {
        if ji != root_fl {
            drop_slots.insert(info.skin_slot[ji] as u16);
        }
    }
    let root_slot = info.skin_slot[0] as u16;
    let doc = g.doc().unwrap();
    let mut replacements = vec![];
    for mesh in doc.meshes() {
        for (pi, prim) in mesh.primitives().enumerate() {
            let r = prim.reader(g.buffer());
            let (Some(jr), Some(wr)) = (r.read_joints(0), r.read_weights(0)) else {
                continue;
            };
            let mut js: Vec<[u16; 4]> = jr.into_u16().collect();
            let ws: Vec<[f32; 4]> = wr.into_f32().collect();
            for j4 in &mut js {
                for slot in j4.iter_mut() {
                    if drop_slots.contains(slot) {
                        *slot = root_slot;
                    }
                }
            }
            replacements.push((mesh.index(), pi, js, ws));
        }
    }
    for (mi, pi, js, ws) in replacements {
        let (ja, wa) = g.push_skin_attrs(&js, &ws).unwrap();
        let attrs = g.json["meshes"][mi]["primitives"][pi]["attributes"]
            .as_object_mut()
            .unwrap();
        attrs.retain(|k, _| !k.starts_with("JOINTS_") && !k.starts_with("WEIGHTS_"));
        attrs.insert("JOINTS_0".into(), json!(ja));
        attrs.insert("WEIGHTS_0".into(), json!(wa));
    }
    g.write(dst).unwrap();
}

/// Insert one collinear mid-bone inside `leg_front.L` so `.L` has +1 deforming bone
/// over `.R` while both cover the full leg cleanly (healthy artist-rig pattern).
fn split_front_left_bone(src: &Path, dst: &Path) {
    let g_in = Glb::read(src).unwrap();
    let info = g_in.skin_info().unwrap();
    let parts = g_in.parts(Frame::Rest).unwrap();
    let mut names = info.names.clone();
    let mut parents: Vec<i64> = info.parents.clone();
    let mut worlds: Vec<geom::M4> = info
        .heads
        .iter()
        .map(|&h| geom::from_trs(h, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]))
        .collect();

    let mut front_l: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.contains("leg_front") && n.ends_with(".L"))
        .map(|(i, _)| i)
        .collect();
    front_l.sort();
    assert!(
        front_l.len() >= 2,
        "expected >=2 leg_front.L bones: {names:?}"
    );
    let set: std::collections::HashSet<usize> = front_l.iter().copied().collect();
    let p_idx = front_l
        .iter()
        .copied()
        .find(|&i| parents[i] < 0 || !set.contains(&(parents[i] as usize)))
        .unwrap_or(front_l[0]);
    let c_idx = front_l
        .iter()
        .copied()
        .find(|&i| parents[i] == p_idx as i64)
        .unwrap_or(front_l[1]);
    let mid = geom::scale(geom::add(info.heads[p_idx], info.heads[c_idx]), 0.5);
    let new_idx = names.len();
    names.push("DEF-leg_front.mid.L".to_string());
    parents.push(p_idx as i64);
    worlds.push(geom::from_trs(mid, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]));
    parents[c_idx] = new_idx as i64;

    let mut js = vec![];
    let mut ws = vec![];
    let p_slot = info.skin_slot[p_idx] as u16;
    let mut slot_to_joint = vec![0u16; info.skin_slot.len().max(names.len())];
    for (ji, &s) in info.skin_slot.iter().enumerate() {
        if s < slot_to_joint.len() {
            slot_to_joint[s] = ji as u16;
        }
    }
    let mid_y = mid[1];
    for part in &parts {
        let pjs = part.joints.as_ref().unwrap();
        let pws = part.weights.as_ref().unwrap();
        for (vi, (j4, w4)) in pjs.iter().zip(pws.iter()).enumerate() {
            let mut nj = [0u16; 4];
            for k in 0..4 {
                if w4[k] > 0.0 {
                    if j4[k] == p_slot && part.positions[vi][1] < mid_y {
                        nj[k] = new_idx as u16;
                    } else {
                        nj[k] = slot_to_joint[j4[k] as usize];
                    }
                }
            }
            js.push(nj);
            ws.push(*w4);
        }
    }

    let mut g_out = box_dog_glb();
    let bare_parts = g_out.parts(Frame::World).unwrap();
    attach_skeleton(&mut g_out, &bare_parts, &names, &parents, &worlds, &js, &ws).unwrap();
    g_out.write(dst).unwrap();
}

#[test]
fn clean_box_dog_reports_per_limb_scores_and_passes() {
    let d = tmp_dir("clean");
    let rigged = rig_box_dog(&d);
    let (code, rep) = run_check(&rigged, &d.join("check.json"));
    assert_eq!(code, 0, "clean box dog failed check: {rep}");
    assert_eq!(rep["pass"], json!(true), "{rep}");
    assert!(rep["score"].as_f64().unwrap() >= 90.0, "{rep}");
    let min_limb = rep["min_limb_score"]
        .as_f64()
        .unwrap_or_else(|| panic!("missing min_limb_score in check report: {rep}"));
    assert!(
        min_limb >= 0.8,
        "expected min_limb_score >= 0.8, got {min_limb}: {rep}"
    );
    let limbs = rep["limbs"].as_array().expect("limbs array");
    assert!(limbs.len() >= 5, "expected quadruped limbs: {rep}");
    for row in limbs {
        let s = row["score"]
            .as_f64()
            .unwrap_or_else(|| panic!("limb row missing numeric score: {row}"));
        assert!((0.0..=1.0).contains(&s), "score out of [0,1]: {row}");
        assert!(
            row["role"].as_str().is_some(),
            "limb row missing role: {row}"
        );
        assert!(
            row["pass"].as_bool().is_some(),
            "limb row missing pass bool: {row}"
        );
        if row["role"] != json!("extra") {
            assert_eq!(row["pass"], json!(true), "clean limb should pass: {row}");
            assert!(s >= 0.8, "clean limb score too low: {row}");
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn one_limb_falloff_drops_per_limb_score_and_fails_symmetry() {
    let d = tmp_dir("falloff");
    let rigged = rig_box_dog(&d);
    let falloff = d.join("falloff.glb");
    strip_front_left_distal(&rigged, &falloff);

    let (code, rep) = run_check(&falloff, &d.join("check.json"));
    assert_eq!(code, 1, "one-limb falloff must fail check: {rep}");
    assert_eq!(rep["pass"], json!(false), "{rep}");
    let min_limb = rep["min_limb_score"]
        .as_f64()
        .unwrap_or_else(|| panic!("missing min_limb_score: {rep}"));
    assert!(
        min_limb <= 0.5,
        "min_limb_score should drop <= 0.5 on falloff: {rep}"
    );
    let fl = rep["limbs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["limb"] == json!("leg_front.L"))
        .unwrap_or_else(|| panic!("no leg_front.L in limbs: {rep}"));
    assert_eq!(fl["pass"], json!(false), "leg_front.L must fail: {fl}");
    assert!(
        fl["score"].as_f64().unwrap() <= 0.5,
        "leg_front.L score too high: {fl}"
    );
    assert!(
        rep["sides_match"].as_f64().unwrap() < 1.0,
        "broken 1-vs-2 falloff leg pair must fail symmetry: {rep}"
    );
    let lopsided_fail = rep["findings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f["code"] == json!("S_LOPSIDED") && f["severity"] == json!("fail"));
    assert!(
        lopsided_fail,
        "broken 1-vs-2 falloff leg pair must emit S_LOPSIDED fail: {rep}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn healthy_three_vs_two_bone_leg_pair_still_passes_symmetry() {
    let d = tmp_dir("three_vs_two");
    let rigged = rig_box_dog(&d);
    let split = d.join("split.glb");
    split_front_left_bone(&rigged, &split);

    let (code, rep) = run_check(&split, &d.join("check.json"));
    assert_eq!(code, 0, "healthy 3-vs-2 leg pair should pass: {rep}");
    assert_eq!(rep["pass"], json!(true), "{rep}");
    assert_eq!(
        rep["sides_match"],
        json!(1.0),
        "healthy 3-vs-2 symmetry should be 1.0: {rep}"
    );
    assert!(
        rep["min_limb_score"].as_f64().unwrap() >= 0.8,
        "healthy 3-vs-2 min_limb_score should stay >= 0.8: {rep}"
    );
    let _ = std::fs::remove_dir_all(&d);
}
