//! A creature made of boxes, through the whole tool.

use skintokens::geom::V3;

use crate::check;
use crate::parts::Body;
use crate::roles::{Anatomy, Kind, Role};
use crate::skeleton::Skeleton;
use crate::solid::tests::cuboid;
use crate::weights;

/// A body along z on four legs, a neck and head up and forward, a tail
/// out behind: +Y up, +Z forward, standing on y = 0. Boxes pushed into
/// each other, as a generated mesh's parts are.
fn box_dog() -> (Vec<V3>, Vec<[u32; 3]>) {
    let boxes: [(V3, V3); 8] = [
        ([-0.15, 0.45, -0.45], [0.15, 0.75, 0.45]),   // body
        ([0.07, 0.0, 0.30], [0.15, 0.50, 0.40]),      // front left leg
        ([-0.15, 0.0, 0.30], [-0.07, 0.50, 0.40]),    // front right leg
        ([0.07, 0.0, -0.40], [0.15, 0.50, -0.30]),    // hind left leg
        ([-0.15, 0.0, -0.40], [-0.07, 0.50, -0.30]),  // hind right leg
        ([-0.06, 0.65, 0.40], [0.06, 0.95, 0.52]),    // neck
        ([-0.08, 0.85, 0.48], [0.08, 1.0, 0.75]),     // head
        ([-0.03, 0.62, -0.85], [0.03, 0.68, -0.40]),  // tail
    ];
    let (mut v, mut f) = (vec![], vec![]);
    for (lo, hi) in boxes {
        let (bv, bf) = cuboid(lo, hi, v.len() as u32);
        v.extend(bv);
        f.extend(bf);
    }
    (v, f)
}

#[test]
fn a_box_dog_is_four_legged_with_a_head_and_a_tail() {
    let (v, f) = box_dog();
    let body = Body::build(&v, &f, 96);
    let an = Anatomy::read(&body);
    let names: Vec<&str> = an.limbs.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(an.kind, Kind::FourLegged, "{names:?}");
    assert!(!an.upright);
    assert!(an.forward > 0.0);
    for want in ["leg_front.L", "leg_front.R", "leg_hind.L", "leg_hind.R", "neck", "tail"] {
        assert!(names.contains(&want), "no {want} among {names:?}");
    }
    let tail = an.limbs.iter().find(|l| l.role == Role::Tail).unwrap();
    assert!(tail.tip()[2] < -0.7, "the tail's line ends at its tip: {:?}", tail.tip());
    // the spine runs from over the hind legs to the neck
    assert!(an.spine.first().unwrap()[2] < -0.2 && an.spine.last().unwrap()[2] > 0.2, "{:?}", an.spine);
}

#[test]
fn its_own_skeleton_passes_the_check_and_one_without_a_tail_does_not() {
    let (v, f) = box_dog();
    let body = Body::build(&v, &f, 96);
    let an = Anatomy::read(&body);
    let sk = Skeleton::fit(&body, &an);
    // every bone inside, both sides alike
    let own = check::check(&body, &an, sk.bones.clone());
    assert_eq!(own.failing, 0, "{}", own.report);
    assert!(own.score > 90.0, "{}", own.report);
    for side in ["leg_front", "leg_hind"] {
        let n = |s: &str| sk.bones.iter().filter(|b| b.name.contains(side) && b.name.ends_with(s)).count();
        assert_eq!(n(".L"), n(".R"));
        assert!(n(".L") >= 3);
    }
    // a pair of legs hangs from one bone, and each leg ends in a foot whose
    // joint is near the ground (what a gait needs to plant it)
    for stem in ["leg_front", "leg_hind"] {
        let first = |side: &str| sk.bones.iter().find(|b| b.name.contains(stem) && b.name.ends_with(side) && b.index == 0).unwrap();
        assert_eq!(first(".L").parent, first(".R").parent, "{stem}");
        let last = sk.bones.iter().filter(|b| b.name.contains(stem) && b.name.ends_with(".L")).max_by_key(|b| b.index).unwrap();
        assert!(last.head[1] < 0.2, "{stem}'s foot joint is {} up", last.head[1]);
    }
    // the same skeleton with the tail's bones taken out
    let tail = an.limbs.iter().position(|l| l.role == Role::Tail).unwrap();
    let keep: Vec<usize> = (0..sk.bones.len()).filter(|&b| sk.bones[b].limb != Some(tail)).collect();
    let mut cut: Vec<_> = keep.iter().map(|&b| sk.bones[b].clone()).collect();
    for b in cut.iter_mut() {
        b.parent = b.parent.map(|p| keep.iter().position(|&k| k == p).unwrap());
    }
    let v2 = check::check(&body, &an, cut);
    assert!(v2.failing >= 1 && v2.score < own.score);
    assert!(v2.report["findings"].to_string().contains("S_NO_CHAIN"), "{}", v2.report);
}

#[test]
fn weights_keep_a_leg_to_itself() {
    let (v, f) = box_dog();
    let body = Body::build(&v, &f, 96);
    let an = Anatomy::read(&body);
    let sk = Skeleton::fit(&body, &an);
    let skin = weights::skin(&body, &sk, &v, &f, weights::BLEND);
    for (i, p) in v.iter().enumerate() {
        let sum: f32 = skin.weights[i].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // a foot corner of the front left leg: all of it on that leg's bones
        if p[1] == 0.0 && p[0] > 0.0 && p[2] > 0.0 {
            for k in 0..4 {
                if skin.weights[i][k] > 0.01 {
                    let name = &sk.bones[skin.joints[i][k] as usize].name;
                    assert!(name.contains("leg_front") && name.ends_with(".L"), "foot vertex on {name}");
                }
            }
        }
    }
}
