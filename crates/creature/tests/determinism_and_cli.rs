//! CLI validation, degenerate mesh handling, and byte determinism tests (Wave 4G).

use std::path::PathBuf;
use std::process::Command;

use serde_json::json;
use skintokens_geom::glb::Glb;

fn rig_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_creature-rig"))
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("creature_det_cli_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn box_dog_glb() -> Glb {
    let boxes: [([f32; 3], [f32; 3]); 8] = [
        ([-0.15, 0.45, -0.45], [0.15, 0.75, 0.45]),
        ([0.07, 0.0, 0.30], [0.15, 0.50, 0.40]),
        ([-0.15, 0.0, 0.30], [-0.07, 0.50, 0.40]),
        ([0.07, 0.0, -0.40], [0.15, 0.50, -0.30]),
        ([-0.15, 0.0, -0.40], [-0.07, 0.50, -0.30]),
        ([-0.06, 0.65, 0.40], [0.06, 0.95, 0.52]),
        ([-0.08, 0.85, 0.48], [0.08, 1.0, 0.75]),
        ([-0.03, 0.62, -0.85], [0.03, 0.68, -0.40]),
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

fn empty_mesh_glb() -> Glb {
    Glb {
        json: json!({
            "asset": {"version": "2.0"},
            "buffers": [],
            "meshes": [{"name": "empty", "primitives": []}],
            "nodes": [{"name": "empty", "mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0
        }),
        bin: vec![],
    }
}

#[test]
fn cli_rejects_invalid_flags_and_degenerate_inputs() {
    let d = tmp_dir("invalid_cli");
    let dog = d.join("dog.glb");
    box_dog_glb().write(&dog).unwrap();

    // Unknown flag must exit 2 instead of silently succeeding.
    let out_bogus = d.join("bogus.glb");
    let r = Command::new(rig_bin())
        .args([
            "rig",
            dog.to_str().unwrap(),
            out_bogus.to_str().unwrap(),
            "--bogus",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(
        r.status.code(),
        Some(2),
        "unknown flag must fail: stdout={} stderr={}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(!out_bogus.exists());

    // --res 0 must fail cleanly with exit 2.
    let out_res0 = d.join("res0.glb");
    let r = Command::new(rig_bin())
        .args([
            "rig",
            dog.to_str().unwrap(),
            out_res0.to_str().unwrap(),
            "--res",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(2));
    assert!(!out_res0.exists());

    // --blend 0 and --blend nan must fail cleanly with exit 2.
    for bad_blend in ["0", "-1", "nan", "inf"] {
        let out_b = d.join(format!("blend_{bad_blend}.glb"));
        let r = Command::new(rig_bin())
            .args([
                "rig",
                dog.to_str().unwrap(),
                out_b.to_str().unwrap(),
                "--blend",
                bad_blend,
            ])
            .output()
            .unwrap();
        assert_eq!(
            r.status.code(),
            Some(2),
            "--blend {bad_blend} should exit 2"
        );
        assert!(!out_b.exists());
    }

    // --turn maybe on parts must fail cleanly with exit 2.
    let r = Command::new(rig_bin())
        .args(["parts", dog.to_str().unwrap(), "--turn", "maybe"])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(2));

    // --class dragon on prune must fail cleanly with exit 2.
    let rigged = d.join("rigged.glb");
    let ok = Command::new(rig_bin())
        .args(["rig", dog.to_str().unwrap(), rigged.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(ok.status.code(), Some(0));

    let pruned = d.join("pruned.glb");
    let r = Command::new(rig_bin())
        .args([
            "prune",
            rigged.to_str().unwrap(),
            pruned.to_str().unwrap(),
            "--class",
            "dragon",
        ])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(2));
    assert!(!pruned.exists());

    // Empty / 0-triangle mesh must exit 2 cleanly (no panic).
    let empty = d.join("empty.glb");
    empty_mesh_glb().write(&empty).unwrap();
    let r = Command::new(rig_bin())
        .args(["parts", empty.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(2));

    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn cli_rig_and_parts_are_byte_deterministic() {
    let d = tmp_dir("det");
    let dog = d.join("dog.glb");
    box_dog_glb().write(&dog).unwrap();

    let parts1 = d.join("parts1.json");
    let parts2 = d.join("parts2.json");
    for jp in [&parts1, &parts2] {
        let r = Command::new(rig_bin())
            .args([
                "parts",
                dog.to_str().unwrap(),
                "--json",
                jp.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(r.status.code(), Some(0));
    }
    assert_eq!(
        std::fs::read(&parts1).unwrap(),
        std::fs::read(&parts2).unwrap()
    );

    let rig1 = d.join("rig1.glb");
    let rig2 = d.join("rig2.glb");
    let j1 = d.join("rig1.json");
    let j2 = d.join("rig2.json");
    for (op, jp) in [(&rig1, &j1), (&rig2, &j2)] {
        let r = Command::new(rig_bin())
            .args([
                "rig",
                dog.to_str().unwrap(),
                op.to_str().unwrap(),
                "--json",
                jp.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(r.status.code(), Some(0));
    }
    assert_eq!(std::fs::read(&rig1).unwrap(), std::fs::read(&rig2).unwrap());
    assert_eq!(std::fs::read(&j1).unwrap(), std::fs::read(&j2).unwrap());

    let _ = std::fs::remove_dir_all(&d);
}
