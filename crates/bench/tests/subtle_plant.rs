//! Subtle planted rigs and per-limb anatomy checks (skintokens-6).
//!
//! Tests `creature-bench plant` (`--mode diagonal|mirror|falloff`) and
//! `creature-bench anatomy` on self-contained synthetic rigged quadruped and
//! upright biped GLBs.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use skintokens_geom::geom;
use skintokens_geom::glb::{Frame, Glb};
use skintokens_geom::rig_edit::attach_skeleton;

fn bench_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_creature-bench"))
}

struct BoxPart {
    lo: [f32; 3],
    hi: [f32; 3],
    joint: u16,
}

/// Append a 2x2-subdivided closed box (54 verts, 48 outward triangles) assigned
/// to `joint` (with optional per-vertex override).
fn push_box(
    pos: &mut Vec<f32>,
    idx: &mut Vec<u16>,
    js: &mut Vec<[u16; 4]>,
    ws: &mut Vec<[f32; 4]>,
    lo: [f32; 3],
    hi: [f32; 3],
    joint_for_vert: impl Fn(usize, [f32; 3]) -> u16,
) {
    let n = 2usize;
    let mut local_v = 0usize;
    // 6 faces: (fixed_axis, fixed_val, u_axis, v_axis, flip_winding)
    let faces = [
        (0, lo[0], 2, 1, false),
        (0, hi[0], 1, 2, false),
        (1, lo[1], 0, 2, false),
        (1, hi[1], 2, 0, false),
        (2, lo[2], 1, 0, false),
        (2, hi[2], 0, 1, false),
    ];
    for (fa, fv, ua, va, _) in faces {
        let base = (pos.len() / 3) as u16;
        for iu in 0..=n {
            for iv in 0..=n {
                let mut p = [0.0f32; 3];
                p[fa] = fv;
                p[ua] = lo[ua] + (hi[ua] - lo[ua]) * (iu as f32 / n as f32);
                p[va] = lo[va] + (hi[va] - lo[va]) * (iv as f32 / n as f32);
                pos.extend(p);
                let j = joint_for_vert(local_v, p);
                local_v += 1;
                js.push([j, 0, 0, 0]);
                ws.push([1.0, 0.0, 0.0, 0.0]);
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

fn build_skinned_glb(
    boxes: &[BoxPart],
    overrides: &[(usize, usize, u16)], // (box_idx, keep_first_n_on_box_joint, fallback_joint)
    joints: &[(&str, i64, [f64; 3])],
) -> Glb {
    let (mut pos, mut idx, mut js, mut ws) = (vec![], vec![], vec![], vec![]);
    for (bi, bp) in boxes.iter().enumerate() {
        let ov = overrides.iter().find(|x| x.0 == bi).copied();
        push_box(
            &mut pos,
            &mut idx,
            &mut js,
            &mut ws,
            bp.lo,
            bp.hi,
            |lv, _| {
                if let Some((_, keep, fb)) = ov {
                    if lv < keep {
                        bp.joint
                    } else {
                        fb
                    }
                } else {
                    bp.joint
                }
            },
        );
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
        json!([{"name": "body", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "body", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);

    let parts = g.parts(Frame::World).expect("mesh parts");
    let names: Vec<String> = joints.iter().map(|j| j.0.to_string()).collect();
    let parents: Vec<i64> = joints.iter().map(|j| j.1).collect();
    let worlds: Vec<geom::M4> = joints
        .iter()
        .map(|j| geom::from_trs(j.2, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]))
        .collect();
    attach_skeleton(&mut g, &parts, &names, &parents, &worlds, &js, &ws).expect("attach skeleton");
    g
}

fn make_quadruped() -> Glb {
    let joints = [
        ("hips", -1, [0.0, 0.55, -0.10]),
        ("spine", 0, [0.0, 0.55, 0.25]),
        ("head", 1, [0.0, 0.78, 0.56]),
        ("leg_front_0.L", 1, [0.14, 0.45, 0.32]),
        ("leg_front_1.L", 3, [0.14, 0.12, 0.32]),
        ("leg_front_0.R", 1, [-0.14, 0.45, 0.32]),
        ("leg_front_1.R", 5, [-0.14, 0.12, 0.32]),
        ("leg_hind_0.L", 0, [0.14, 0.45, -0.32]),
        ("leg_hind_1.L", 7, [0.14, 0.12, -0.32]),
        ("leg_hind_0.R", 0, [-0.14, 0.45, -0.32]),
        ("leg_hind_1.R", 9, [-0.14, 0.12, -0.32]),
    ];
    let boxes = [
        BoxPart {
            lo: [-0.14, 0.45, -0.45],
            hi: [0.14, 0.72, 0.05],
            joint: 0,
        },
        BoxPart {
            lo: [-0.14, 0.45, 0.05],
            hi: [0.14, 0.72, 0.45],
            joint: 1,
        },
        BoxPart {
            lo: [-0.08, 0.68, 0.44],
            hi: [0.08, 0.90, 0.72],
            joint: 2,
        },
        BoxPart {
            lo: [0.10, 0.24, 0.28],
            hi: [0.18, 0.48, 0.36],
            joint: 3,
        },
        BoxPart {
            lo: [0.10, 0.00, 0.28],
            hi: [0.18, 0.24, 0.36],
            joint: 4,
        },
        BoxPart {
            lo: [-0.18, 0.24, 0.28],
            hi: [-0.10, 0.48, 0.36],
            joint: 5,
        },
        BoxPart {
            lo: [-0.18, 0.00, 0.28],
            hi: [-0.10, 0.24, 0.36],
            joint: 6,
        },
        BoxPart {
            lo: [0.10, 0.24, -0.36],
            hi: [0.18, 0.48, -0.28],
            joint: 7,
        },
        BoxPart {
            lo: [0.10, 0.00, -0.36],
            hi: [0.18, 0.24, -0.28],
            joint: 8,
        },
        BoxPart {
            lo: [-0.18, 0.24, -0.36],
            hi: [-0.10, 0.48, -0.28],
            joint: 9,
        },
        BoxPart {
            lo: [-0.18, 0.00, -0.36],
            hi: [-0.10, 0.24, -0.28],
            joint: 10,
        },
    ];
    build_skinned_glb(&boxes, &[], &joints)
}

enum ArmMode {
    Clean,
    DistalStripped,
    PartialMass,
}

fn make_biped(arm_mode: ArmMode) -> Glb {
    let joints = [
        ("hips", -1, [0.0, 0.95, 0.0]),
        ("spine", 0, [0.0, 1.38, 0.0]),
        ("head", 1, [0.0, 1.66, 0.0]),
        ("arm_0.L", 1, [0.22, 1.42, 0.0]),
        ("arm_1.L", 3, [0.42, 1.42, 0.0]),
        ("arm_0.R", 1, [-0.22, 1.42, 0.0]),
        ("arm_1.R", 5, [-0.42, 1.42, 0.0]),
        ("leg_0.L", 0, [0.12, 0.85, 0.0]),
        ("leg_1.L", 7, [0.12, 0.25, 0.0]),
        ("leg_0.R", 0, [-0.12, 0.85, 0.0]),
        ("leg_1.R", 9, [-0.12, 0.25, 0.0]),
    ];
    let boxes = [
        BoxPart {
            lo: [-0.15, 0.90, -0.10],
            hi: [0.15, 1.25, 0.10],
            joint: 0,
        },
        BoxPart {
            lo: [-0.16, 1.25, -0.10],
            hi: [0.16, 1.52, 0.10],
            joint: 1,
        },
        BoxPart {
            lo: [-0.09, 1.54, -0.10],
            hi: [0.09, 1.80, 0.10],
            joint: 2,
        },
        BoxPart {
            lo: [0.18, 1.36, -0.05],
            hi: [0.36, 1.48, 0.05],
            joint: 3,
        }, // box 3: armL0
        BoxPart {
            lo: [0.36, 1.36, -0.05],
            hi: [0.54, 1.48, 0.05],
            joint: 4,
        }, // box 4: armL1
        BoxPart {
            lo: [-0.36, 1.36, -0.05],
            hi: [-0.18, 1.48, 0.05],
            joint: 5,
        },
        BoxPart {
            lo: [-0.54, 1.36, -0.05],
            hi: [-0.36, 1.48, 0.05],
            joint: 6,
        },
        BoxPart {
            lo: [0.06, 0.45, -0.06],
            hi: [0.18, 0.90, 0.06],
            joint: 7,
        },
        BoxPart {
            lo: [0.06, 0.00, -0.06],
            hi: [0.18, 0.45, 0.06],
            joint: 8,
        },
        BoxPart {
            lo: [-0.18, 0.45, -0.06],
            hi: [-0.06, 0.90, 0.06],
            joint: 9,
        },
        BoxPart {
            lo: [-0.18, 0.00, -0.06],
            hi: [-0.06, 0.45, 0.06],
            joint: 10,
        },
    ];
    let overrides: Vec<(usize, usize, u16)> = match arm_mode {
        ArmMode::Clean => vec![],
        // Distal arm bone loses all skin onto spine while proximal arm keeps 12 verts:
        // arm_0.L stays live with 1 joint and length 0.0, neither arm touches ground.
        ArmMode::DistalStripped => vec![(3, 12, 1), (4, 0, 1)],
        // Both arm bones stay live (2 verts each -> same joint count & length as arm.R),
        // while >95% of arm.L skin fell off onto spine.
        ArmMode::PartialMass => vec![(3, 2, 1), (4, 2, 1)],
    };
    build_skinned_glb(&boxes, &overrides, &joints)
}

fn run_anatomy(glb: &Path, json_out: &Path) -> (i32, Value) {
    let out = Command::new(bench_bin())
        .args([
            "anatomy",
            glb.to_str().unwrap(),
            "--json",
            json_out.to_str().unwrap(),
        ])
        .output()
        .expect("run creature-bench anatomy");
    let code = out.status.code().unwrap_or(-1);
    let text = std::fs::read_to_string(json_out).unwrap_or_else(|e| {
        panic!(
            "anatomy wrote no JSON ({e}); exit {code}; stderr={}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (
        code,
        serde_json::from_str(&text).expect("parse anatomy json"),
    )
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skintokens6-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn clean_quadruped_and_biped_pass_anatomy_with_skin_mass() {
    let d = tmp_dir("clean");
    let quad = d.join("quad.glb");
    let biped = d.join("biped.glb");
    make_quadruped().write(&quad).unwrap();
    make_biped(ArmMode::Clean).write(&biped).unwrap();

    for (name, path) in [("quad", &quad), ("biped", &biped)] {
        let (code, rep) = run_anatomy(path, &d.join(format!("{name}.json")));
        assert_eq!(code, 0, "{name} anatomy failed: {rep}");
        assert_eq!(rep["pass"], json!(true), "{name}: {rep}");
        let limbs = rep["limbs"].as_array().expect("limbs array");
        assert_eq!(limbs.len(), 4, "{name} limbs: {rep}");
        for l in limbs {
            let m = l["skin_mass"]
                .as_f64()
                .unwrap_or_else(|| panic!("{name} limb missing numeric skin_mass: {l}"));
            assert!(m > 10.0, "{name} limb skin_mass too small: {l}");
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn plant_diagonal_and_mirror_modes_are_distinct_and_fail_anatomy() {
    let d = tmp_dir("swap");
    let quad = d.join("quad.glb");
    make_quadruped().write(&quad).unwrap();

    // Default (omitted --mode) matches diagonal.
    let def_glb = d.join("def.glb");
    let def_json = d.join("def.json");
    let out = Command::new(bench_bin())
        .args([
            "plant",
            quad.to_str().unwrap(),
            def_glb.to_str().unwrap(),
            "--json",
            def_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "default plant failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let def_rep: Value =
        serde_json::from_str(&std::fs::read_to_string(&def_json).unwrap()).unwrap();
    assert_eq!(def_rep["mode"], json!("diagonal"), "{def_rep}");

    // Explicit --mode diagonal.
    let diag_glb = d.join("diag.glb");
    let diag_json = d.join("diag.json");
    let out = Command::new(bench_bin())
        .args([
            "plant",
            quad.to_str().unwrap(),
            diag_glb.to_str().unwrap(),
            "--mode",
            "diagonal",
            "--json",
            diag_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "diagonal plant failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let diag_rep: Value =
        serde_json::from_str(&std::fs::read_to_string(&diag_json).unwrap()).unwrap();
    assert_eq!(diag_rep["mode"], json!("diagonal"), "{diag_rep}");
    assert_eq!(diag_rep["pairs"], def_rep["pairs"]);

    // Plain left/right mirror swap (--mode mirror).
    let mir_glb = d.join("mirror.glb");
    let mir_json = d.join("mirror.json");
    let out = Command::new(bench_bin())
        .args([
            "plant",
            quad.to_str().unwrap(),
            mir_glb.to_str().unwrap(),
            "--mode",
            "mirror",
            "--json",
            mir_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mirror plant failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mir_rep: Value =
        serde_json::from_str(&std::fs::read_to_string(&mir_json).unwrap()).unwrap();
    assert_eq!(mir_rep["mode"], json!("mirror"), "{mir_rep}");
    assert_ne!(
        mir_rep["pairs"], diag_rep["pairs"],
        "mirror must differ from diagonal on a quadruped"
    );
    let pairs = mir_rep["pairs"].as_array().expect("mirror pairs");
    assert_eq!(pairs.len(), 4, "all 4 limb joint pairs swapped: {mir_rep}");
    for p in pairs {
        let a = p[0].as_str().unwrap();
        let b = p[1].as_str().unwrap();
        assert_eq!(
            a.trim_end_matches(".L"),
            b.trim_end_matches(".R"),
            "mirror swap must pair exact L/R counterparts, got [{a}, {b}]"
        );
    }

    for (label, glb) in [("diagonal", &diag_glb), ("mirror", &mir_glb)] {
        let (code, an) = run_anatomy(glb, &d.join(format!("{label}_an.json")));
        assert_eq!(code, 1, "{label} should fail anatomy: {an}");
        assert_eq!(an["pass"], json!(false), "{label}: {an}");
        assert!(
            an["findings"].to_string().contains("B_SWAPPED"),
            "{label} should report B_SWAPPED: {an}"
        );
    }

    // Unknown mode must error cleanly.
    let bad = Command::new(bench_bin())
        .args([
            "plant",
            quad.to_str().unwrap(),
            d.join("bad.glb").to_str().unwrap(),
            "--mode",
            "bogus",
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success(), "unknown mode must fail");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn plant_falloff_keeps_bones_and_fails_anatomy_on_quadruped_and_biped_arm() {
    let d = tmp_dir("falloff");
    let quad = d.join("quad.glb");
    make_quadruped().write(&quad).unwrap();

    // 1) Quadruped --mode falloff via CLI: skeleton keeps all 11 joints, skin falls off one limb.
    let fall_glb = d.join("quad_falloff.glb");
    let fall_json = d.join("quad_falloff.json");
    let out = Command::new(bench_bin())
        .args([
            "plant",
            quad.to_str().unwrap(),
            fall_glb.to_str().unwrap(),
            "--mode",
            "falloff",
            "--json",
            fall_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "falloff plant failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rep: Value = serde_json::from_str(&std::fs::read_to_string(&fall_json).unwrap()).unwrap();
    assert_eq!(rep["mode"], json!("falloff"), "{rep}");
    assert!(
        rep["reassigned_vertices"].as_u64().unwrap_or(0) > 0,
        "falloff must reassign limb vertices: {rep}"
    );
    assert!(
        !rep["cleared_joints"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true),
        "falloff must report cleared_joints: {rep}"
    );
    let orig_info = Glb::read(&quad).unwrap().skin_info().unwrap();
    let fall_info = Glb::read(&fall_glb).unwrap().skin_info().unwrap();
    assert_eq!(
        orig_info.names.len(),
        fall_info.names.len(),
        "falloff keeps the limb bones in the skeleton"
    );
    let (code, an) = run_anatomy(&fall_glb, &d.join("quad_falloff_an.json"));
    assert_eq!(code, 1, "quadruped falloff must fail anatomy: {an}");
    assert_eq!(an["pass"], json!(false), "{an}");

    // 2) Biped --mode falloff via CLI also fails anatomy.
    let biped = d.join("biped.glb");
    make_biped(ArmMode::Clean).write(&biped).unwrap();
    let bfall_glb = d.join("biped_falloff.glb");
    let bfall_json = d.join("biped_falloff.json");
    let out = Command::new(bench_bin())
        .args([
            "plant",
            biped.to_str().unwrap(),
            bfall_glb.to_str().unwrap(),
            "--mode",
            "falloff",
            "--json",
            bfall_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "biped falloff plant failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (code, an) = run_anatomy(&bfall_glb, &d.join("biped_falloff_an.json"));
    assert_eq!(code, 1, "biped falloff must fail anatomy: {an}");
    assert_eq!(an["pass"], json!(false), "{an}");

    // 3) Subtle non-ground arm falloff: distal arm stripped (arm_0.L stays live with 1 joint,
    // length 0.0) must fail anatomy rather than slipping past the length==0 ratio loophole.
    let arm_distal = d.join("arm_distal.glb");
    make_biped(ArmMode::DistalStripped)
        .write(&arm_distal)
        .unwrap();
    let (code, an) = run_anatomy(&arm_distal, &d.join("arm_distal_an.json"));
    assert_eq!(code, 1, "distal arm falloff must fail anatomy: {an}");
    assert_eq!(an["pass"], json!(false), "{an}");

    // 4) Subtle partial mass falloff: both arm joints stay live (same joint count & bone length),
    // only skin mass drops below 20% of its mirrored partner -> must fail with B_FALLOFF.
    let arm_partial = d.join("arm_partial.glb");
    make_biped(ArmMode::PartialMass)
        .write(&arm_partial)
        .unwrap();
    let (code, an) = run_anatomy(&arm_partial, &d.join("arm_partial_an.json"));
    assert_eq!(code, 1, "partial arm mass falloff must fail anatomy: {an}");
    assert_eq!(an["pass"], json!(false), "{an}");
    assert!(
        an["findings"].to_string().contains("B_FALLOFF"),
        "partial arm mass falloff should emit B_FALLOFF: {an}"
    );

    let _ = std::fs::remove_dir_all(&d);
}
