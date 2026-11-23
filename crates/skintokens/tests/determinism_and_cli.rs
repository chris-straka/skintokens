//! CLI validation & edge-case contracts for skintokens without needing a GPU/checkpoint (Wave 4G).

use std::path::PathBuf;
use std::process::Command;

use serde_json::json;
use skintokens::glb::Glb;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_skintokens"))
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skintokens_cli_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
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

fn nan_mesh_glb() -> Glb {
    let pos: Vec<f32> = vec![f32::NAN, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let idx: Vec<u16> = vec![0, 1, 2];
    let mut g = Glb {
        json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
        bin: vec![],
    };
    let pb: Vec<u8> = pos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let pa = g
        .push_accessor(
            &pb,
            5126,
            "VEC3",
            3,
            Some((vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 0.0])),
            false,
        )
        .unwrap();
    let ia = g
        .push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false)
        .unwrap();
    g.json["meshes"] =
        json!([{"name": "nan", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "nan", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);
    g
}

#[test]
fn cli_rig_rejects_empty_and_nan_meshes_before_checkpoint() {
    let d = tmp_dir("mesh_check");
    let empty = d.join("empty.glb");
    let nan = d.join("nan.glb");
    let out = d.join("out.glb");
    let rep = d.join("rep.json");
    empty_mesh_glb().write(&empty).unwrap();
    nan_mesh_glb().write(&nan).unwrap();

    for (path, label) in [(&empty, "empty"), (&nan, "nan")] {
        let r = Command::new(bin())
            .env("SKINTOKENS_CKPT", d.join("missing.ckpt"))
            .args([
                "rig",
                path.to_str().unwrap(),
                out.to_str().unwrap(),
                "--report",
                rep.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&r.stderr);
        assert_eq!(r.status.code(), Some(2), "{label}: {err}");
        assert!(
            !rep.exists(),
            "{label} should not write report on usage error"
        );
        assert!(
            !err.contains("checkpoint") && !err.contains("missing"),
            "{label} should validate geometry before loading checkpoint: {err}"
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn cli_rejects_unknown_short_flags_and_bad_seed() {
    let d = tmp_dir("flags");
    let empty = d.join("in.glb");
    empty_mesh_glb().write(&empty).unwrap();

    let r = Command::new(bin())
        .args([
            "rig",
            empty.to_str().unwrap(),
            d.join("o.glb").to_str().unwrap(),
            "-x",
        ])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(2));
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(
        err.contains("unknown option"),
        "expected unknown option message for -x, got: {err}"
    );

    for bad_seed in ["-1", "NaN", "1.5"] {
        let rs = Command::new(bin())
            .args([
                "rig",
                empty.to_str().unwrap(),
                d.join("o.glb").to_str().unwrap(),
                "--seed",
                bad_seed,
            ])
            .output()
            .unwrap();
        assert_eq!(rs.status.code(), Some(2));
    }

    let _ = std::fs::remove_dir_all(&d);
}

fn valid_tri_glb() -> Glb {
    let pos: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let idx: Vec<u16> = vec![0, 1, 2];
    let mut g = Glb {
        json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
        bin: vec![],
    };
    let pb: Vec<u8> = pos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let pa = g
        .push_accessor(
            &pb,
            5126,
            "VEC3",
            3,
            Some((vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 0.0])),
            false,
        )
        .unwrap();
    let ia = g
        .push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false)
        .unwrap();
    g.json["meshes"] =
        json!([{"name": "tri", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "tri", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);
    g
}

#[test]
fn cli_rig_rejects_zero_num_beams_env_before_checkpoint() {
    let d = tmp_dir("zero_beams");
    let tri = d.join("tri.glb");
    let out = d.join("out.glb");
    let rep = d.join("rep.json");
    valid_tri_glb().write(&tri).unwrap();

    for bad_beams in ["0", "abc"] {
        let r = Command::new(bin())
            .env("SKINTOKENS_CKPT", d.join("missing.ckpt"))
            .env("SKINTOKENS_NUM_BEAMS", bad_beams)
            .args([
                "rig",
                tri.to_str().unwrap(),
                out.to_str().unwrap(),
                "--report",
                rep.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&r.stderr);
        assert_eq!(r.status.code(), Some(2), "{bad_beams}: {err}");
        assert!(!rep.exists());
        assert!(
            err.contains("SKINTOKENS_NUM_BEAMS"),
            "must validate SKINTOKENS_NUM_BEAMS={bad_beams} before checkpoint: {err}"
        );
    }

    let _ = std::fs::remove_dir_all(&d);
}
