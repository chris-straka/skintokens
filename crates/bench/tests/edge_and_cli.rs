//! Edge-case hardening and CLI validation tests for creature-bench (Wave 4G).

use std::path::PathBuf;
use std::process::Command;

use serde_json::json;
use skintokens_geom::glb::Glb;

fn bench_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_creature-bench"))
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bench_edge_cli_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn unrigged_box_glb() -> Glb {
    let pos: Vec<f32> = vec![
        -0.5, 0.0, -0.5, 0.5, 0.0, -0.5, 0.5, 1.0, -0.5, -0.5, 1.0, -0.5,
    ];
    let idx: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
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
            4,
            Some((vec![-0.5, 0.0, -0.5], vec![0.5, 1.0, -0.5])),
            false,
        )
        .unwrap();
    let ia = g
        .push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false)
        .unwrap();
    g.json["meshes"] =
        json!([{"name": "quad", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "quad", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);
    g
}

fn zero_extent_rigged_glb() -> Glb {
    let pos: Vec<f32> = vec![0.0; 12];
    let idx: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
    let js = [0u8; 16];
    let ws = [1.0f32, 0.0, 0.0, 0.0].repeat(4);
    let ibm: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut g = Glb {
        json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
        bin: vec![],
    };
    let pb: Vec<u8> = pos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let wb: Vec<u8> = ws.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ibmb: Vec<u8> = ibm.iter().flat_map(|x| x.to_le_bytes()).collect();
    let pa = g
        .push_accessor(
            &pb,
            5126,
            "VEC3",
            4,
            Some((vec![0.0; 3], vec![0.0; 3])),
            false,
        )
        .unwrap();
    let ia = g
        .push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false)
        .unwrap();
    let ja = g.push_accessor(&js, 5121, "VEC4", 4, None, false).unwrap();
    let wa = g.push_accessor(&wb, 5126, "VEC4", 4, None, false).unwrap();
    let ma = g
        .push_accessor(&ibmb, 5126, "MAT4", 1, None, false)
        .unwrap();
    g.json["meshes"] = json!([{
        "name": "point",
        "primitives": [{"attributes": {"POSITION": pa, "JOINTS_0": ja, "WEIGHTS_0": wa}, "indices": ia}]
    }]);
    g.json["nodes"] = json!([
        {"name": "point", "mesh": 0, "skin": 0},
        {"name": "hips", "translation": [0.0, 0.0, 0.0]}
    ]);
    g.json["skins"] = json!([{"joints": [1], "skeleton": 1, "inverseBindMatrices": ma}]);
    g.json["scenes"] = json!([{"nodes": [0, 1]}]);
    g.json["scene"] = json!(0);
    g
}

#[test]
fn anatomy_rejects_unrigged_glb_cleanly() {
    let d = tmp_dir("unrigged");
    let p = d.join("bare.glb");
    unrigged_box_glb().write(&p).unwrap();

    let r = Command::new(bench_bin())
        .args(["anatomy", p.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(
        r.status.code(),
        Some(2),
        "anatomy on unrigged GLB must fail exit 2, got {:?}: {}",
        r.status.code(),
        String::from_utf8_lossy(&r.stdout)
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn cli_rejects_unknown_flags() {
    let d = tmp_dir("unknown_flag");
    let p = d.join("rigged.glb");
    zero_extent_rigged_glb().write(&p).unwrap();

    let r = Command::new(bench_bin())
        .args(["anatomy", p.to_str().unwrap(), "--bogus", "1"])
        .output()
        .unwrap();
    assert_eq!(
        r.status.code(),
        Some(2),
        "unknown flag must exit 2: stdout={} stderr={}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn anatomy_handles_zero_extent_rig_without_nan_or_panic() {
    let d = tmp_dir("zero_ext");
    let p = d.join("zero.glb");
    let j = d.join("rep.json");
    zero_extent_rigged_glb().write(&p).unwrap();

    let r = Command::new(bench_bin())
        .args([
            "anatomy",
            p.to_str().unwrap(),
            "--json",
            j.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        matches!(r.status.code(), Some(0) | Some(1)),
        "zero-extent rig must not panic: {:?} stderr={}",
        r.status.code(),
        String::from_utf8_lossy(&r.stderr)
    );
    let txt = std::fs::read_to_string(&j).unwrap();
    assert!(!txt.contains("NaN") && !txt.contains("null"), "{txt}");
    let _ = std::fs::remove_dir_all(&d);
}
