//! Structural humanoid naming for TokenRig skeletons (port of the former
//! `skintokens/naming.py`).
//!
//! TokenRig's general ("articulation") class emits unnamed joints (bone_0..).
//! For humanoids they are named by structure so motionforge's standardize
//! adapter maps them onto the HLL skeleton: Mixamo names, which it already
//! reads. glTF frame: +Y up; the character's left is +X when it faces +Z, and
//! the toes decide the facing. Joints the walk cannot place keep a
//! non-canonical name (`extra_*`); standardize folds their weights into the
//! nearest kept ancestor.

use std::collections::HashMap;

pub const CORE: [&str; 14] = [
    "Hips", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg",
    "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
];
const ARM: [&str; 4] = ["Shoulder", "Arm", "ForeArm", "Hand"];
const LEG: [&str; 4] = ["UpLeg", "Leg", "Foot", "ToeBase"];
const SPINE: [&str; 3] = ["Spine", "Spine1", "Spine2"];
pub const PREFIX: &str = "mixamorig:";

fn round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}

/// Returns ({old name: new name}, missing core names). New names carry the
/// `mixamorig:` prefix (extras do not).
pub fn name_humanoid(names: &[String], parents: &[Option<usize>], heads: &[[f64; 3]]) -> (HashMap<String, String>, Vec<String>) {
    let n = names.len();
    let all_missing = || (HashMap::new(), CORE.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let mut kids: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, p) in parents.iter().enumerate() {
        if let Some(p) = p {
            kids[*p].push(i);
        }
    }
    let roots: Vec<usize> = (0..n).filter(|&i| parents[i].is_none()).collect();
    if roots.len() != 1 {
        return all_missing();
    }
    let root = roots[0];
    let mut size = vec![0usize; n];
    fn count(i: usize, kids: &[Vec<usize>], size: &mut [usize]) -> usize {
        let s = 1 + kids[i].iter().map(|&k| count(k, kids, size)).sum::<usize>();
        size[i] = s;
        s
    }
    count(root, &kids, &mut size);
    let ys: Vec<f64> = heads.iter().map(|h| h[1]).collect();
    let ptp = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let height = if ptp != 0.0 { ptp } else { 1.0 };
    let mid_x = heads[root][0];
    let mut out: HashMap<usize, String> = HashMap::new();
    out.insert(root, "Hips".into());

    let dx = |k: usize| (heads[k][0] - mid_x).abs();
    let step = |j: usize| -> Option<usize> {
        kids[j]
            .iter()
            .copied()
            .filter(|&k| heads[k][1] >= heads[j][1] - 0.02 * height && dx(k) < 0.04 * height)
            .min_by(|&a, &b| {
                round3(dx(a)).partial_cmp(&round3(dx(b))).unwrap().then(size[b].cmp(&size[a])).then(std::cmp::Ordering::Equal)
            })
    };
    let mut chain = vec![];
    let mut cur = step(root);
    while let Some(c) = cur {
        if chain.contains(&c) {
            break;
        }
        chain.push(c);
        cur = step(c);
    }
    fn lowest(i: usize, kids: &[Vec<usize>], heads: &[[f64; 3]]) -> f64 {
        kids[i].iter().map(|&k| lowest(k, kids, heads)).fold(heads[i][1], f64::min)
    }
    // one limb per side: the largest subtree on each side; the rest are extras
    let pick_pair = |cands: &[usize]| -> (Vec<usize>, Vec<usize>) {
        let mut best: Vec<(bool, usize)> = vec![]; // insertion-ordered dict
        for &k in cands {
            let s = heads[k][0] > mid_x;
            match best.iter_mut().find(|(side, _)| *side == s) {
                Some(e) => {
                    if size[k] > size[e.1] {
                        e.1 = k
                    }
                }
                None => best.push((s, k)),
            }
        }
        let chosen: Vec<usize> = best.iter().map(|x| x.1).collect();
        let rest = cands.iter().copied().filter(|k| !chosen.contains(k)).collect();
        (chosen, rest)
    };
    let leg_cands: Vec<usize> =
        kids[root].iter().copied().filter(|k| !chain.contains(k) && lowest(*k, &kids, heads) < heads[root][1] - 0.3 * height).collect();
    let (legs, mut odd) = pick_pair(&leg_cands);
    let extra: Vec<usize> = kids[root].iter().copied().filter(|k| !chain.contains(k) && !legs.contains(k) && !odd.contains(k)).collect();
    odd.extend(extra);

    // the chest is the highest chain joint with children on both sides (the arms)
    let mut chest = None;
    for &j in &chain {
        let side: Vec<usize> = kids[j].iter().copied().filter(|k| !chain.contains(k)).collect();
        if side.iter().any(|&k| heads[k][0] > mid_x) && side.iter().any(|&k| heads[k][0] < mid_x) {
            chest = Some(j);
        }
    }
    let Some(chest) = chest else {
        return (HashMap::new(), CORE.iter().filter(|&&c| c != "Hips").map(|s| s.to_string()).collect());
    };
    let ci = chain.iter().position(|&c| c == chest).unwrap();
    let (spine, above) = (chain[..=ci].to_vec(), chain[ci + 1..].to_vec());
    for (k, &j) in spine.iter().enumerate() {
        out.insert(j, if k < SPINE.len() { SPINE[k].to_string() } else { format!("extra_spine{k}") });
    }
    if above.len() == 1 {
        out.insert(above[0], "Head".into());
    } else if !above.is_empty() {
        out.insert(above[0], "Neck".into());
        out.insert(above[1], "Head".into());
        for (k, &j) in above[2..].iter().enumerate() {
            out.insert(j, format!("extra_head{k}"));
        }
    }
    let walk = |start: usize, labels: &[String], side: &str, out: &mut HashMap<usize, String>| {
        let mut j = Some(start);
        let mut k = 0;
        while let Some(x) = j {
            let name = if k < labels.len() { format!("{side}{}", labels[k]) } else { format!("extra_{side}{}{k}", labels[labels.len() - 1]) };
            out.insert(x, name);
            k += 1;
            // python max(): first of the largest
            j = kids[x].iter().copied().fold(None, |acc: Option<usize>, c| match acc {
                Some(a) if size[c] <= size[a] => Some(a),
                _ => Some(c),
            });
        }
    };
    let arm_cands: Vec<usize> = kids[chest].iter().copied().filter(|k| !chain.contains(k)).collect();
    let (arms, odd_arms) = pick_pair(&arm_cands);
    for (k, &j) in odd.iter().chain(odd_arms.iter()).enumerate() {
        walk(j, &[format!("extra_{k}_")], "", &mut out);
    }
    let arm_labels: Vec<String> = ARM.iter().map(|s| s.to_string()).collect();
    let leg_labels: Vec<String> = LEG.iter().map(|s| s.to_string()).collect();
    for &a in &arms {
        walk(a, &arm_labels, if heads[a][0] > mid_x { "Left" } else { "Right" }, &mut out);
    }
    for &l in &legs {
        walk(l, &leg_labels, if heads[l][0] > mid_x { "Left" } else { "Right" }, &mut out);
    }
    // facing: toes in front of the ankles; facing -Z mirrors left/right
    let by_name: HashMap<String, usize> = out.iter().map(|(k, v)| (v.clone(), *k)).collect();
    let dz: Vec<f64> = ["Left", "Right"]
        .iter()
        .filter_map(|s| {
            let (t, f) = (by_name.get(&format!("{s}ToeBase"))?, by_name.get(&format!("{s}Foot"))?);
            Some(heads[*t][2] - heads[*f][2])
        })
        .collect();
    if !dz.is_empty() && dz.iter().sum::<f64>() / (dz.len() as f64) < 0.0 {
        for v in out.values_mut() {
            if v.contains("Left") {
                *v = v.replacen("Left", "Right", 1);
            } else if v.contains("Right") {
                *v = v.replacen("Right", "Left", 1);
            }
        }
    }
    let got: std::collections::HashSet<&String> = out.values().collect();
    let missing: Vec<String> = CORE.iter().filter(|c| !got.contains(&c.to_string())).map(|s| s.to_string()).collect();
    let mapping = out
        .iter()
        .map(|(&i, v)| (names[i].clone(), if v.starts_with("extra") { v.clone() } else { format!("{PREFIX}{v}") }))
        .collect();
    (mapping, missing)
}
