//! Multi-seed determinism and geometry/GLB edge hardening tests (Wave 4G).

use serde_json::json;
use skintokens_geom::geom::{self, KdTree, V3};
use skintokens_geom::glb::Glb;
use skintokens_geom::mesh::{
    reparent, sample_with_faces, transfer, weld, Mesh, Normalize, Transfer,
};
use skintokens_geom::rng::Rng;

#[test]
fn kdtree_knn_breaks_ties_deterministically_by_index() {
    let pts = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let tree = KdTree::new(pts.clone());
    let got = tree.knn([0.0, 0.0, 0.0], 3);
    let mut brute: Vec<(f64, usize)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| (geom::dist2(*p, [0.0, 0.0, 0.0]).sqrt(), i))
        .collect();
    brute.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    brute.truncate(3);
    assert_eq!(got, brute);
}

#[test]
fn kdtree_knn_does_not_let_nan_points_hide_finite_neighbors() {
    let pts = vec![[0.1, 0.0, 0.0], [f64::NAN, 0.0, 0.0], [f64::NAN, 1.0, 0.0]];
    let tree = KdTree::new(pts);
    let res = tree.knn([0.0, 0.0, 0.0], 2);
    assert!(
        !res.is_empty(),
        "finite neighbor must not be hidden by NaN points"
    );
    assert_eq!(res[0].1, 0);
    assert!((res[0].0 - 0.1).abs() < 1e-12);
    assert!(res.iter().all(|(d, _)| d.is_finite()));
}

#[test]
fn quat_from_cols_handles_zero_and_nan_matrices() {
    let q_zero = geom::quat_from_cols([[0.0; 3]; 3]);
    assert_eq!(q_zero, [0.0, 0.0, 0.0, 1.0]);

    let q_nan = geom::quat_from_cols([[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(q_nan, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn normalize_fit_and_invert_handle_empty_nan_and_huge_bounds() {
    let empty: [V3; 0] = [];
    let n_empty = Normalize::fit(empty.iter());
    assert!(n_empty.center.iter().all(|x| x.is_finite()));
    assert!(n_empty.scale.is_finite() && n_empty.scale > 0.0);
    let back_empty = n_empty.invert(n_empty.apply([0.0, 0.0, 0.0]));
    assert!(back_empty.iter().all(|x| x.is_finite()));

    let nan_pts = [[f64::NAN, 1.0, 2.0]];
    let n_nan = Normalize::fit(nan_pts.iter());
    assert!(n_nan.center.iter().all(|x| x.is_finite()));
    assert!(n_nan.scale.is_finite() && n_nan.scale > 0.0);
}

#[test]
fn mesh_sampling_weld_and_reparent_edges() {
    // Sampling with NaN coordinate must never leak NaN into sample cloud.
    let verts = vec![[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let faces = vec![[0, 1, 2]];
    let m = Mesh {
        vertices: verts.clone(),
        faces: faces.clone(),
        offsets: vec![0],
    };
    let fnorm = m.face_normals();
    let vnorm = m.vertex_normals(&fnorm);
    let mut rng = Rng::new(7);
    let (samples, _) = sample_with_faces(&verts, &faces, &vnorm, &fnorm, &mut rng);
    assert!(
        samples.iter().flatten().all(|x| x.is_finite()),
        "NaN leaked into samples"
    );

    // Non-positive eps must not weld distant points (e.g. (-1.0)^2 == 1.0).
    let (ids_neg, reps_neg) = weld(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], -1.0);
    assert_eq!(ids_neg, vec![0, 1]);
    assert_eq!(reps_neg.len(), 2);

    // Out-of-bounds parent index must not panic in reparent.
    let rp = reparent(&[-1, 99], &[0, 1]);
    assert_eq!(rp, vec![-1, -1]);
}

#[test]
fn glb_from_bytes_rejects_truncated_declared_total_and_bad_version() {
    let mut g = Glb {
        json: json!({"asset": {"version": "2.0"}, "buffers": []}),
        bin: vec![],
    };
    let raw = g.to_bytes().unwrap();

    // Header total larger than actual byte buffer -> truncated GLB.
    let mut bad_total = raw.clone();
    let bogus_len = (raw.len() as u32) + 64;
    bad_total[8..12].copy_from_slice(&bogus_len.to_le_bytes());
    assert!(
        Glb::from_bytes(&bad_total).is_err(),
        "must reject GLB whose header length exceeds file length"
    );

    // Unsupported glTF version 1 -> reject.
    let mut bad_ver = raw;
    bad_ver[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert!(
        Glb::from_bytes(&bad_ver).is_err(),
        "must reject glTF version != 2"
    );
    let _ = &mut g;
}

#[test]
fn transfer_surface_and_facing_are_bitwise_deterministic() {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.2],
    ];
    let faces = vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
    let m = Mesh {
        vertices: verts.clone(),
        faces: faces.clone(),
        offsets: vec![0],
    };
    let fnorm = m.face_normals();
    let vnorm = m.vertex_normals(&fnorm);
    let mut rng = Rng::new(42);
    let (samples, sample_faces) = sample_with_faces(&verts, &faces, &vnorm, &fnorm, &mut rng);
    let skin: Vec<Vec<f32>> = samples
        .iter()
        .map(|s| {
            let a = (s[0] + 0.5).clamp(0.05, 0.95);
            vec![a, 1.0 - a]
        })
        .collect();

    for method in [Transfer::Facing, Transfer::Surface] {
        let (base, _) = transfer(method, &samples, &sample_faces, &skin, &verts, &faces);
        for _ in 0..4 {
            let (again, _) = transfer(method, &samples, &sample_faces, &skin, &verts, &faces);
            assert_eq!(base.len(), again.len());
            for (r0, r1) in base.iter().zip(&again) {
                let b0: Vec<u32> = r0.iter().map(|x| x.to_bits()).collect();
                let b1: Vec<u32> = r1.iter().map(|x| x.to_bits()).collect();
                assert_eq!(b0, b1);
            }
        }
    }
}
