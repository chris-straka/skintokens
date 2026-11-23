use serde_json::{json, Value};
use skintokens_geom::geom::{self, V3};
use skintokens_geom::glb::Glb;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_creature-rig"))
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cr_prune_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn add_box(
    lo: V3,
    hi: V3,
    joint: u16,
    verts: &mut Vec<V3>,
    tris: &mut Vec<[u32; 3]>,
    joints: &mut Vec<[u16; 4]>,
    weights: &mut Vec<[f32; 4]>,
) {
    let base = verts.len() as u32;
    let corners = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ];
    for p in corners {
        verts.push(p);
        joints.push([joint, 0, 0, 0]);
        weights.push([1.0, 0.0, 0.0, 0.0]);
    }
    let faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    for f in faces {
        tris.push([base + f[0], base + f[1], base + f[2]]);
    }
}

/// Build a rigged humanoid-like box fixture with:
/// 0: pelvis, 1: torso, 2: head, 3: hand.L, 4..8: finger1.L..finger5.L,
/// 9: ear_spike_out (leaf far outside mesh).
fn build_prune_fixture(path: &Path) {
    let joint_specs: Vec<(&str, Option<usize>, V3)> = vec![
        ("pelvis", None, [0.0, 0.45, 0.0]),
        ("torso", Some(0), [0.0, 0.70, 0.0]),
        ("head", Some(1), [0.0, 1.00, 0.0]),
        ("hand.L", Some(1), [0.32, 0.72, 0.0]),
        ("finger1.L", Some(3), [0.38, 0.72, 0.03]),
        ("finger2.L", Some(3), [0.39, 0.72, 0.015]),
        ("finger3.L", Some(3), [0.40, 0.72, 0.0]),
        ("finger4.L", Some(3), [0.39, 0.72, -0.015]),
        ("finger5.L", Some(3), [0.38, 0.72, -0.03]),
        ("ear_spike_out", Some(2), [1.40, 1.65, 0.0]),
    ];

    let mut verts = Vec::new();
    let mut tris = Vec::new();
    let mut v_joints = Vec::new();
    let mut v_weights = Vec::new();
    // Main trunk + limbs + hand mesh
    add_box(
        [-0.14, 0.0, -0.10],
        [0.14, 0.90, 0.10],
        0,
        &mut verts,
        &mut tris,
        &mut v_joints,
        &mut v_weights,
    );
    add_box(
        [-0.10, 0.90, -0.09],
        [0.10, 1.12, 0.09],
        2,
        &mut verts,
        &mut tris,
        &mut v_joints,
        &mut v_weights,
    );
    // Attach some head vertices weighted to the out-of-body spike joint (9)
    add_box(
        [0.08, 0.96, -0.04],
        [0.12, 1.04, 0.04],
        9,
        &mut verts,
        &mut tris,
        &mut v_joints,
        &mut v_weights,
    );
    // Left arm + hand box including vertices weighted to finger1..5
    add_box(
        [0.12, 0.66, -0.05],
        [0.36, 0.78, 0.05],
        3,
        &mut verts,
        &mut tris,
        &mut v_joints,
        &mut v_weights,
    );
    for fi in 0..5u16 {
        let z0 = -0.035 + fi as f64 * 0.015;
        add_box(
            [0.36, 0.69, z0],
            [0.42, 0.75, z0 + 0.01],
            4 + fi,
            &mut verts,
            &mut tris,
            &mut v_joints,
            &mut v_weights,
        );
    }

    let mut g = Glb {
        json: json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 0}],
            "nodes": [],
            "meshes": [{"primitives": [{}]}],
            "scenes": [{"nodes": []}],
            "scene": 0,
        }),
        bin: vec![],
    };

    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let mut pos_bytes = Vec::with_capacity(verts.len() * 12);
    for p in &verts {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
            pos_bytes.extend_from_slice(&(p[a] as f32).to_le_bytes());
        }
    }
    let pos_acc = g
        .push_accessor(
            &pos_bytes,
            5126,
            "VEC3",
            verts.len(),
            Some((lo.to_vec(), hi.to_vec())),
            false,
        )
        .unwrap();
    let mut idx_bytes = Vec::with_capacity(tris.len() * 12);
    for t in &tris {
        for i in t {
            idx_bytes.extend_from_slice(&i.to_le_bytes());
        }
    }
    let idx_acc = g
        .push_accessor(&idx_bytes, 5125, "SCALAR", tris.len() * 3, None, false)
        .unwrap();
    let (ja, wa) = g.push_skin_attrs(&v_joints, &v_weights).unwrap();

    // Create mesh node (node 0) and joint nodes (1..=N)
    let mut nodes: Vec<Value> = vec![json!({"name": "body", "mesh": 0, "skin": 0})];
    let mut joint_node_ids = Vec::with_capacity(joint_specs.len());
    for (idx, (name, parent, head)) in joint_specs.iter().enumerate() {
        let local = if let Some(pi) = parent {
            let ph = joint_specs[*pi].2;
            [head[0] - ph[0], head[1] - ph[1], head[2] - ph[2]]
        } else {
            *head
        };
        joint_node_ids.push(1 + idx);
        nodes.push(json!({
            "name": name,
            "translation": local,
        }));
    }
    for (idx, (_, parent, _)) in joint_specs.iter().enumerate() {
        if let Some(pi) = parent {
            let pnode = 1 + *pi;
            let children = nodes[pnode]
                .as_object_mut()
                .unwrap()
                .entry("children")
                .or_insert_with(|| json!([]));
            children.as_array_mut().unwrap().push(json!(1 + idx));
        }
    }
    let ibms: Vec<_> = joint_specs
        .iter()
        .map(|(_, _, h)| {
            geom::from_trs([-h[0], -h[1], -h[2]], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        })
        .collect();
    let ibm_acc = g.push_mat4s(&ibms).unwrap();

    g.json["meshes"][0]["primitives"][0] = json!({
        "attributes": {"POSITION": pos_acc, "JOINTS_0": ja, "WEIGHTS_0": wa},
        "indices": idx_acc,
        "mode": 4
    });
    g.json["nodes"] = json!(nodes);
    g.json["skins"] = json!([{
        "joints": joint_node_ids,
        "skeleton": 1,
        "inverseBindMatrices": ibm_acc,
    }]);
    g.json["scenes"][0]["nodes"] = json!([0, 1]);
    g.write(path).unwrap();
}

#[test]
fn prune_cuts_outside_spike_folds_creature_digits_and_keeps_hero_fingers() {
    let d = tmp_dir("hero_vs_monster");
    let src = d.join("fixture.glb");
    build_prune_fixture(&src);

    // 1. Creature / monster class: folds 5 finger leaves AND cuts outside ear spike.
    let out_m = d.join("monster.glb");
    let rep_m = d.join("monster_prune.json");
    let o = Command::new(bin())
        .args([
            "prune",
            src.to_str().unwrap(),
            out_m.to_str().unwrap(),
            "--class",
            "monster",
            "--json",
            rep_m.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        o.status.code(),
        Some(0),
        "creature-rig prune --class monster failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let r_m: Value = serde_json::from_slice(&std::fs::read(&rep_m).unwrap()).unwrap();
    assert!(
        r_m["outside_cut"].as_u64().unwrap() >= 1,
        "expected outside spike cut: {r_m}"
    );
    assert!(
        r_m["fingers_folded"].as_u64().unwrap() >= 5,
        "expected 5 fingers folded for monster: {r_m}"
    );
    let gm = Glb::read(&out_m).unwrap();
    let skim = gm.skin_info().unwrap();
    assert!(
        !skim.names.iter().any(|n| n == "ear_spike_out"),
        "{:?}",
        skim.names
    );
    assert!(
        !skim.names.iter().any(|n| n.starts_with("finger")),
        "{:?}",
        skim.names
    );
    let head_idx = skim.names.iter().position(|n| n == "head").unwrap();
    let hand_idx = skim.names.iter().position(|n| n == "hand.L").unwrap();
    let parts = gm.parts(skintokens_geom::glb::Frame::Rest).unwrap();
    let pj = parts[0].joints.as_ref().unwrap();
    let pw = parts[0].weights.as_ref().unwrap();
    for (_jrow, wrow) in pj.iter().zip(pw.iter()) {
        let sum: f32 = wrow.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weight sum off: {sum}");
    }
    assert!(
        pj.iter().any(|j| j[0] as usize == head_idx),
        "transferred spike weights should reach head"
    );
    assert!(
        pj.iter().any(|j| j[0] as usize == hand_idx),
        "transferred finger weights should reach hand.L"
    );

    // 2. Hero class: keeps all 5 finger bones intact, STILL cuts outside ear spike!
    let out_h = d.join("hero.glb");
    let rep_h = d.join("hero_prune.json");
    let oh = Command::new(bin())
        .args([
            "prune",
            src.to_str().unwrap(),
            out_h.to_str().unwrap(),
            "--class",
            "hero",
            "--json",
            rep_h.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        oh.status.code(),
        Some(0),
        "creature-rig prune --class hero failed:\nstderr: {}",
        String::from_utf8_lossy(&oh.stderr)
    );
    let r_h: Value = serde_json::from_slice(&std::fs::read(&rep_h).unwrap()).unwrap();
    assert!(
        r_h["outside_cut"].as_u64().unwrap() >= 1,
        "hero should still prune outside spike: {r_h}"
    );
    assert_eq!(
        r_h["fingers_folded"].as_u64().unwrap(),
        0,
        "hero must preserve 5 fingers: {r_h}"
    );
    let gh = Glb::read(&out_h).unwrap();
    let skih = gh.skin_info().unwrap();
    assert!(
        !skih.names.iter().any(|n| n == "ear_spike_out"),
        "{:?}",
        skih.names
    );
    for f in 1..=5 {
        let want = format!("finger{f}.L");
        assert!(
            skih.names.contains(&want),
            "missing {want} in {:?}",
            skih.names
        );
    }

    // 3. Missing input exits 2 without creating output.
    let bad_out = d.join("missing_out.glb");
    let om = Command::new(bin())
        .args([
            "prune",
            d.join("nonexistent.glb").to_str().unwrap(),
            bad_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(om.status.code(), Some(2));
    assert!(!bad_out.exists());

    let _ = std::fs::remove_dir_all(&d);
}
