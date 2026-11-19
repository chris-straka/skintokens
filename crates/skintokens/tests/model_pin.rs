//! A pin on the model's output (skintokens-11). Parity with the Python original was proven
//! once (2026-10-05); nothing would show drift after a candle upgrade, a weights change or a
//! port edit. This rigs one small made-up figure on the CPU with a fixed seed (class
//! `custom`: the model's own skeleton, no humanoid naming on top) and compares
//! what comes out with what was recorded:
//!
//! - the skeleton: every joint's name and its bind-pose head, within 1 mm;
//! - the skin: each joint's share of the weight over all vertices, within 0.002, and the
//!   largest change of any one vertex's weight on any joint, at most 0.02.
//!
//! It needs the checkpoint, so it is ignored by default:
//!
//! ```sh
//! fleet run --snapshot --cores 6 --data skintokens-weights -- \
//!   cargo test --release -p skintokens --test model_pin -- --ignored model_pin
//! ```
//!
//! (on Linux add `--no-default-features`: the default `metal` feature is the Mac's)
//!
//! The checkpoint: `$SKINTOKENS_PIN_CKPT`, else `$FLEET_DATA_SKINTOKENS_WEIGHTS/grpo_1400.ckpt`,
//! else the binary's own default (`$SKINTOKENS_HOME/weights`). A missing checkpoint fails the
//! test (it never passes without running). The recorded output is per platform
//! (`tests/model_pin/<os>-<arch>.json`; floats differ in the last bits between machines);
//! `SKINTOKENS_PIN_RECORD=1` writes it, from main only. `tests/model_pin_planted.sh` proves
//! the pin fails on a checkpoint with one weight tensor scaled by 1%.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use skintokens::glb::Glb;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_skintokens"))
}

fn ckpt() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SKINTOKENS_PIN_CKPT") {
        return Some(PathBuf::from(p));
    }
    if let Ok(d) = std::env::var("FLEET_DATA_SKINTOKENS_WEIGHTS") {
        return Some(Path::new(&d).join("grpo_1400.ckpt"));
    }
    None
}

/// A standing figure of boxes (+Y up, facing +Z, left = +X, 1.8 m): torso, head, two arms
/// out in an A pose, two legs. Closed boxes, 8 vertices and 12 triangles each.
fn figure() -> Glb {
    let boxes: [([f32; 3], [f32; 3]); 6] = [
        ([-0.17, 0.95, -0.10], [0.17, 1.50, 0.10]),  // torso
        ([-0.10, 1.52, -0.11], [0.10, 1.80, 0.11]),  // head
        ([0.18, 0.95, -0.05], [0.28, 1.46, 0.05]),   // left arm (down beside the body)
        ([-0.28, 0.95, -0.05], [-0.18, 1.46, 0.05]), // right arm
        ([0.03, 0.00, -0.07], [0.15, 0.93, 0.07]),   // left leg
        ([-0.15, 0.00, -0.07], [-0.03, 0.93, 0.07]), // right leg
    ];
    let (mut pos, mut idx) = (Vec::<f32>::new(), Vec::<u16>::new());
    for (lo, hi) in boxes {
        let b = (pos.len() / 3) as u16;
        for k in 0..8u16 {
            pos.extend([
                if k & 1 == 0 { lo[0] } else { hi[0] },
                if k & 2 == 0 { lo[1] } else { hi[1] },
                if k & 4 == 0 { lo[2] } else { hi[2] },
            ]);
        }
        // outward-facing quads: -x, +x, -y, +y, -z, +z
        for q in [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ] {
            idx.extend([b + q[0], b + q[1], b + q[2], b + q[0], b + q[2], b + q[3]]);
        }
    }
    let n = pos.len() / 3;
    let mut mn = vec![f64::MAX; 3];
    let mut mx = vec![f64::MIN; 3];
    for k in 0..n {
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
    let pa = g.push_accessor(&pb, 5126, "VEC3", n, Some((mn, mx)), false);
    let ia = g.push_accessor(&ib, 5123, "SCALAR", idx.len(), None, false);
    g.json["meshes"] = json!([{"name": "figure", "primitives": [{"attributes": {"POSITION": pa}, "indices": ia}]}]);
    g.json["nodes"] = json!([{"name": "figure", "mesh": 0}]);
    g.json["scenes"] = json!([{"nodes": [0]}]);
    g.json["scene"] = json!(0);
    g
}

/// What the pin compares: joint names and heads, each joint's weight share, and every
/// vertex's weights (for the largest per-vertex change).
fn summary(out: &Path) -> Value {
    let g = Glb::read(out).expect("read the rigged GLB");
    let info = g.skin_info().expect("the rig has a skin");
    let doc = g.doc().expect("gltf document");
    let nj = info.names.len();
    let mut per_vertex: Vec<Vec<f64>> = vec![];
    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let r = prim.reader(g.buffer());
            let (Some(js), Some(ws)) = (r.read_joints(0), r.read_weights(0)) else {
                continue;
            };
            for (j, w) in js.into_u16().zip(ws.into_f32()) {
                let mut row = vec![0.0; nj];
                for k in 0..4 {
                    // skin.joints slot -> our joint order
                    if let Some(o) = info.skin_slot.iter().position(|&s| s == j[k] as usize) {
                        row[o] += w[k] as f64;
                    }
                }
                per_vertex.push(row);
            }
        }
    }
    let nv = per_vertex.len().max(1) as f64;
    let share: Vec<f64> = (0..nj)
        .map(|o| per_vertex.iter().map(|r| r[o]).sum::<f64>() / nv)
        .collect();
    let r4 = |x: f64| (x * 1e4).round() / 1e4;
    json!({
        "joints": info.names,
        "heads": info.heads.iter().map(|h| h.iter().map(|&x| r4(x)).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "share": share.iter().map(|&x| r4(x)).collect::<Vec<_>>(),
        "vertices": per_vertex.iter().map(|r| r.iter().map(|&x| r4(x)).collect::<Vec<_>>()).collect::<Vec<_>>(),
    })
}

fn golden_path() -> PathBuf {
    let os = if cfg!(target_os = "macos") {
        "darwin"
    } else {
        std::env::consts::OS
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/model_pin")
        .join(format!("{os}-{}.json", std::env::consts::ARCH))
}

/// The differences between a run and the pin, empty when it holds.
fn compare(want: &Value, got: &Value) -> Vec<String> {
    let mut d = vec![];
    if want["joints"] != got["joints"] {
        d.push(format!(
            "joints: want {}, got {}",
            want["joints"], got["joints"]
        ));
        return d;
    }
    let f = |v: &Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<f64>>()
    };
    let names: Vec<String> = want["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    for (i, n) in names.iter().enumerate() {
        let (a, b) = (f(&want["heads"][i]), f(&got["heads"][i]));
        let dist = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        if dist > 0.001 {
            d.push(format!("joint {n}: head moved {:.1} mm", dist * 1000.0));
        }
        let (sa, sb) = (
            want["share"][i].as_f64().unwrap(),
            got["share"][i].as_f64().unwrap(),
        );
        if (sa - sb).abs() > 0.002 {
            d.push(format!("joint {n}: weight share {sa:.4} -> {sb:.4}"));
        }
    }
    let (wv, gv) = (
        want["vertices"].as_array().unwrap(),
        got["vertices"].as_array().unwrap(),
    );
    if wv.len() != gv.len() {
        d.push(format!("vertices: want {}, got {}", wv.len(), gv.len()));
    } else {
        let worst = wv
            .iter()
            .zip(gv)
            .flat_map(|(a, b)| f(a).into_iter().zip(f(b)).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f64::max);
        if worst > 0.02 {
            d.push(format!(
                "a vertex's weight on a joint moved by {worst:.3} (at most 0.02)"
            ));
        }
    }
    d
}

#[test]
#[ignore = "needs the SkinTokens checkpoint (--data skintokens-weights); see the file's header"]
fn model_pin() {
    let d = std::env::temp_dir().join(format!("skintokens_pin_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let (inp, out, rep) = (
        d.join("figure.glb"),
        d.join("rigged.glb"),
        d.join("report.json"),
    );
    figure().write(&inp).unwrap();
    let mut cmd = Command::new(bin());
    cmd.args([
        "rig",
        inp.to_str().unwrap(),
        out.to_str().unwrap(),
        "--report",
        rep.to_str().unwrap(),
    ])
    .args(["--seed", "1", "--class", "custom"])
    .env("SKINTOKENS_DEVICE", "cpu")
    .env("SKINTOKENS_DTYPE", "f32");
    if let Some(c) = ckpt() {
        assert!(c.exists(), "no checkpoint at {}", c.display());
        cmd.env("SKINTOKENS_CKPT", &c);
    }
    let o = cmd.output().unwrap();
    let report = std::fs::read_to_string(&rep).unwrap_or_default();
    assert_eq!(
        o.status.code(),
        Some(0),
        "skintokens rig failed:\n{}\n{}\nreport: {report}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let got = summary(&out);
    let gp = golden_path();
    if std::env::var("SKINTOKENS_PIN_RECORD").is_ok() {
        std::fs::create_dir_all(gp.parent().unwrap()).unwrap();
        std::fs::write(&gp, serde_json::to_string_pretty(&got).unwrap()).unwrap();
        if let Ok(a) = std::env::var("FLEET_ARTIFACTS") {
            std::fs::copy(&gp, Path::new(&a).join(gp.file_name().unwrap())).unwrap();
        }
        eprintln!(
            "recorded {} ({} joints)",
            gp.display(),
            got["joints"].as_array().unwrap().len()
        );
        return;
    }
    let want: Value = serde_json::from_str(
        &std::fs::read_to_string(&gp)
            .unwrap_or_else(|_| panic!("no pin recorded for this platform: {}", gp.display())),
    )
    .unwrap();
    let diffs = compare(&want, &got);
    let _ = std::fs::remove_dir_all(&d);
    assert!(
        diffs.is_empty(),
        "the model's output drifted from the pin ({}):\n{}",
        gp.display(),
        diffs.join("\n")
    );
}

/// The comparison itself, without the model: identical passes; a moved joint, a changed share
/// and a moved vertex weight each fail.
#[test]
fn pin_comparison_catches_drift() {
    let base = json!({"joints": ["Hips", "Spine"], "heads": [[0.0, 1.0, 0.0], [0.0, 1.2, 0.0]],
                      "share": [0.6, 0.4], "vertices": [[1.0, 0.0], [0.2, 0.8]]});
    assert!(compare(&base, &base).is_empty());
    let mut moved = base.clone();
    moved["heads"][1][1] = json!(1.202);
    assert_eq!(compare(&base, &moved).len(), 1);
    let mut share = base.clone();
    share["share"] = json!([0.597, 0.403]);
    assert_eq!(compare(&base, &share).len(), 2);
    let mut vert = base.clone();
    vert["vertices"][1] = json!([0.25, 0.75]);
    assert_eq!(compare(&base, &vert).len(), 1);
    let mut renamed = base.clone();
    renamed["joints"][1] = json!("Spine1");
    assert_eq!(compare(&base, &renamed).len(), 1);
}
