//! Fast tests (no model): GLB I/O, humanoid naming, helper bones, joints
//! doc, CLI contract. Port of the former tests/test_unit.py.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use skintokens::glb::{Frame, Glb};
use skintokens::naming::name_humanoid;
use skintokens::pipeline;

/// name, parent, head (x, y, z); +Y up, faces +Z, character left = +X
const HUMANOID: [(&str, i64, [f64; 3]); 22] = [
    ("hips", -1, [0.0, 1.0, 0.0]),
    ("spine", 0, [0.0, 1.12, 0.0]),
    ("spine1", 1, [0.0, 1.26, 0.0]),
    ("chest", 2, [0.0, 1.40, 0.0]),
    ("neck", 3, [0.0, 1.56, 0.0]),
    ("head", 4, [0.0, 1.66, 0.0]),
    ("l_shoulder", 3, [0.07, 1.54, 0.0]),
    ("l_arm", 6, [0.20, 1.46, 0.0]),
    ("l_forearm", 7, [0.31, 1.20, 0.0]),
    ("l_hand", 8, [0.40, 0.97, 0.0]),
    ("r_shoulder", 3, [-0.07, 1.54, 0.0]),
    ("r_arm", 10, [-0.20, 1.46, 0.0]),
    ("r_forearm", 11, [-0.31, 1.20, 0.0]),
    ("r_hand", 12, [-0.40, 0.97, 0.0]),
    ("l_thigh", 0, [0.12, 0.97, 0.0]),
    ("l_shin", 14, [0.14, 0.53, 0.0]),
    ("l_foot", 15, [0.18, 0.12, 0.0]),
    ("l_toe", 16, [0.23, 0.02, 0.10]),
    ("r_thigh", 0, [-0.12, 0.97, 0.0]),
    ("r_shin", 18, [-0.14, 0.53, 0.0]),
    ("r_foot", 19, [-0.18, 0.12, 0.0]),
    ("r_toe", 20, [-0.23, 0.02, 0.10]),
];

const EXPECT: [(&str, &str); 22] = [
    ("hips", "Hips"), ("spine", "Spine"), ("spine1", "Spine1"), ("chest", "Spine2"), ("neck", "Neck"), ("head", "Head"),
    ("l_shoulder", "LeftShoulder"), ("l_arm", "LeftArm"), ("l_forearm", "LeftForeArm"), ("l_hand", "LeftHand"),
    ("r_shoulder", "RightShoulder"), ("r_arm", "RightArm"), ("r_forearm", "RightForeArm"), ("r_hand", "RightHand"),
    ("l_thigh", "LeftUpLeg"), ("l_shin", "LeftLeg"), ("l_foot", "LeftFoot"), ("l_toe", "LeftToeBase"),
    ("r_thigh", "RightUpLeg"), ("r_shin", "RightLeg"), ("r_foot", "RightFoot"), ("r_toe", "RightToeBase"),
];

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skintokens_test_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One quad per joint around its head, fully weighted to that joint.
fn humanoid_glb(mirror_z: bool, with_helper: bool) -> Glb {
    let joints: Vec<(&str, i64, [f64; 3])> =
        HUMANOID.iter().map(|&(n, p, h)| (n, p, if mirror_z { [-h[0], h[1], -h[2]] } else { h })).collect();
    let (mut pos, mut idx, mut jo, mut we) = (vec![], vec![], vec![], vec![]);
    for (i, (_, _, h)) in joints.iter().enumerate() {
        let b = (pos.len() / 3) as u16;
        for (dx, dy) in [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)] {
            pos.extend([(h[0] + dx) as f32, (h[1] + dy) as f32, h[2] as f32]);
            jo.extend([i as u8, 0, 0, 0]);
            we.extend([1.0f32, 0.0, 0.0, 0.0]);
        }
        idx.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    let mut g = Glb { json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}), bin: vec![] };
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let n = pos.len() / 3;
    let (mn, mx) = (0..3).fold((vec![f64::MAX; 3], vec![f64::MIN; 3]), |(mut mn, mut mx), a| {
        for k in 0..n {
            mn[a] = mn[a].min(pos[k * 3 + a] as f64);
            mx[a] = mx[a].max(pos[k * 3 + a] as f64);
        }
        (mn, mx)
    });
    let pa = g.push_accessor(&f32b(&pos), 5126, "VEC3", n, Some((mn, mx)), false);
    let ia = g.push_accessor(&idx.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), 5123, "SCALAR", idx.len(), None, false);
    let ja = g.push_accessor(&jo, 5121, "VEC4", n, None, false);
    let wa = g.push_accessor(&f32b(&we), 5126, "VEC4", n, None, false);
    let mut nodes = vec![json!({"name": "mesh", "mesh": 0, "skin": 0})];
    for (n, p, h) in &joints {
        let base = if *p >= 0 { joints[*p as usize].2 } else { [0.0; 3] };
        nodes.push(json!({"name": n, "translation": [h[0] - base[0], h[1] - base[1], h[2] - base[2]]}));
    }
    for (i, (_, p, _)) in joints.iter().enumerate() {
        if *p >= 0 {
            let c = nodes[*p as usize + 1].as_object_mut().unwrap().entry("children").or_insert(json!([])).as_array_mut().unwrap().push(json!(i + 1));
            let _ = c;
        }
    }
    let mut ibm: Vec<[[f64; 4]; 4]> = joints
        .iter()
        .map(|(_, _, h)| [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-h[0], -h[1], -h[2], 1.0]])
        .collect();
    let mut skin_joints: Vec<usize> = (1..=joints.len()).collect();
    if with_helper {
        // a twist helper on l_arm, as motionforge's standardize adds them
        let arm = 1 + 7;
        nodes.push(json!({"name": "helper", "translation": [0.0, 0.0, 0.0], "extras": {"hll_helper": {"driver": "l_arm", "share": 0.5}}}));
        let hi = nodes.len() - 1;
        nodes[arm].as_object_mut().unwrap().entry("children").or_insert(json!([])).as_array_mut().unwrap().push(json!(hi));
        skin_joints.push(hi);
        ibm.push(ibm[7]);
    }
    let ib = g.push_mat4s(&ibm);
    g.json["nodes"] = Value::Array(nodes);
    g.json["meshes"] = json!([{"primitives": [{"attributes": {"POSITION": pa, "JOINTS_0": ja, "WEIGHTS_0": wa}, "indices": ia}]}]);
    g.json["skins"] = json!([{"joints": skin_joints, "inverseBindMatrices": ib}]);
    g.json["scene"] = json!(0);
    g.json["scenes"] = json!([{"nodes": [0, 1]}]);
    g.doc().unwrap();
    g
}

#[test]
fn glb_roundtrip_and_skeleton() {
    let d = tmp("rt");
    let p = d.join("a.glb");
    humanoid_glb(false, false).write(&p).unwrap();
    let g = Glb::read(&p).unwrap();
    let info = g.skin_info().unwrap();
    assert_eq!(info.names, HUMANOID.iter().map(|x| x.0.to_string()).collect::<Vec<_>>());
    assert_eq!(info.parents, HUMANOID.iter().map(|x| x.1).collect::<Vec<_>>());
    for (h, e) in info.heads.iter().zip(HUMANOID.iter()) {
        for a in 0..3 {
            assert!((h[a] - e.2[a]).abs() < 1e-5);
        }
    }
    let parts = g.parts(Frame::SkinBind).unwrap();
    assert_eq!(parts[0].positions.len(), 88);
    assert_eq!(parts[0].triangles.len(), 44);
}

fn names_and_heads(g: &Glb) -> (Vec<String>, Vec<Option<usize>>, Vec<[f64; 3]>) {
    let i = g.skin_info().unwrap();
    (i.names, i.parents.iter().map(|&p| if p < 0 { None } else { Some(p as usize) }).collect(), i.heads)
}

#[test]
fn naming_facing_plus_z() {
    let (n, p, h) = names_and_heads(&humanoid_glb(false, false));
    let (mapping, missing) = name_humanoid(&n, &p, &h);
    assert!(missing.is_empty(), "{missing:?}");
    for (old, new) in EXPECT {
        assert_eq!(mapping[old], format!("mixamorig:{new}"), "{old}");
    }
}

#[test]
fn naming_facing_minus_z_keeps_character_left() {
    let (n, p, h) = names_and_heads(&humanoid_glb(true, false));
    let (mapping, missing) = name_humanoid(&n, &p, &h);
    assert!(missing.is_empty());
    assert_eq!(mapping["l_arm"], "mixamorig:LeftArm");
    assert_eq!(mapping["r_thigh"], "mixamorig:RightUpLeg");
}

#[test]
fn naming_rejects_non_humanoid() {
    let n: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let (_, missing) = name_humanoid(&n, &[None, Some(0), Some(1)], &[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]);
    assert!(!missing.is_empty());
}

#[test]
fn helpers_found_by_extras_and_by_name() {
    let g = humanoid_glb(false, true);
    let joints: Vec<usize> = g.json["skins"][0]["joints"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let h = pipeline::helper_joints(&g, &joints);
    assert_eq!(h.len(), 1);
    assert_eq!(h[joints.last().unwrap()], 8); // l_arm node
    // by name: <driver>_twist.<side>
    let mut g = humanoid_glb(false, true);
    let hi = *joints.last().unwrap();
    g.json["nodes"][hi].as_object_mut().unwrap().remove("extras");
    g.json["nodes"][hi]["name"] = json!("DEF-l_arm_twist.L");
    g.json["nodes"][8]["name"] = json!("DEF-l_arm.L");
    assert_eq!(pipeline::helper_joints(&g, &joints).len(), 1);
}

#[test]
fn joints_doc_schema() {
    let doc = pipeline::joints_doc(&humanoid_glb(false, false), Some("t"), "x.glb").unwrap();
    assert_eq!(doc["format"], "skintokens-joints/1");
    let js = doc["joints"].as_array().unwrap();
    assert_eq!(js.len(), 22);
    for j in js {
        for v in j["normalized"].as_array().unwrap() {
            assert!(v.as_f64().unwrap().abs() <= 1.0 + 1e-6);
        }
    }
    assert_eq!(js[1]["parent"], 0);
    assert!(js[5]["tail_world"].is_null()); // head: leaf
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_skintokens"))
}

#[test]
fn cli_usage_errors_write_no_report() {
    let d = tmp("cli");
    let rep = d.join("r.json");
    let out = Command::new(bin()).args(["rig", d.join("missing.glb").to_str().unwrap(), d.join("o.glb").to_str().unwrap(), "--report"]).arg(&rep).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(!rep.exists());
    // skin needs a rigged GLB
    let mut g = humanoid_glb(false, false);
    g.json.as_object_mut().unwrap().remove("skins");
    g.json["nodes"][0].as_object_mut().unwrap().remove("skin");
    let p = d.join("unrigged.glb");
    g.write(&p).unwrap();
    let out = Command::new(bin()).args(["skin", p.to_str().unwrap(), d.join("o.glb").to_str().unwrap(), "--report"]).arg(&rep).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(!rep.exists());
    let out = Command::new(bin()).args(["rig", "a.glb", "b.glb", "--class", "dragon"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn cli_joints_writes_doc() {
    let d = tmp("joints");
    let p = d.join("a.glb");
    humanoid_glb(false, false).write(&p).unwrap();
    let o = d.join("j.json");
    let out = Command::new(bin()).args(["joints", p.to_str().unwrap(), o.to_str().unwrap(), "--subject", "t"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&o).unwrap()).unwrap();
    assert_eq!(doc["subject"], "t");
    assert_eq!(doc["joints"].as_array().unwrap().len(), 22);
    let _ = Path::new("");
}
