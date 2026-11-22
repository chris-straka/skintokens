//! NaN-safe float sorting tests (skintokens-10 Rust half): bad/NaN floats
//! in mesh geometry or joint coordinates must never panic in `partial_cmp().unwrap()`.

use skintokens::geom::KdTree;
use skintokens::naming::name_humanoid;

#[test]
fn kdtree_does_not_panic_on_nan_coordinates() {
    let pts = vec![
        [0.0, 0.0, 0.0],
        [f64::NAN, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, f64::NAN, 0.5],
        [-1.0, 0.2, 0.3],
    ];
    let tree = KdTree::new(pts);
    let _ = tree.knn([0.0, 0.0, 0.0], 3);
    let _ = tree.knn([f64::NAN, 0.0, 0.0], 2);
}

#[test]
fn name_humanoid_does_not_panic_on_nan_joint_heads() {
    let names: Vec<String> = (0..5).map(|i| format!("b{i}")).collect();
    let parents = vec![None, Some(0), Some(0), Some(0), Some(1)];
    let heads = vec![
        [0.0, 1.0, 0.0],
        [0.0, 1.4, 0.0],
        [f64::NAN, 0.9, 0.0],
        [-0.2, 0.9, 0.0],
        [0.0, 1.7, 0.0],
    ];
    let (map, _missing) = name_humanoid(&names, &parents, &heads);
    assert_eq!(map.len(), names.len());
}

#[test]
fn local_matrix_does_not_panic_on_short_trs_arrays() {
    let node = serde_json::json!({"translation": [1.0], "rotation": [], "scale": [1.0, 2.0]});
    let m = skintokens::glb::Glb::local_matrix(&node);
    assert_eq!(m, skintokens::geom::IDENTITY);
}
