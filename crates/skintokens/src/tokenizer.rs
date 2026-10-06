//! TokenRig's skeleton tokenizer (upstream `src/tokenizer/tokenizer_part.py`
//! and `make_skeleton` from `spec.py`, MIT) with the checkpoint's config:
//! 256 coordinate bins over [-1, 1], classes rignet/vroid/articulation,
//! parts body/hand, no part order (joint names are `bone_N`).

pub const NUM_DISCRETE: u32 = 256;
pub const BRANCH: u32 = 256;
pub const BOS: u32 = 257;
pub const EOS: u32 = 258;
pub const PAD: u32 = 259;
pub const SPRING: u32 = 260;
pub const PART_BODY: u32 = 261;
pub const PART_HAND: u32 = 262;
pub const CLS_NONE: u32 = 263;
pub const CLS_RIGNET: u32 = 264;
pub const CLS_VROID: u32 = 265;
pub const CLS_ARTICULATION: u32 = 266;
/// Skeleton vocabulary; SkinTokens follow at VOCAB.., then the LM's EOS.
pub const VOCAB: u32 = 267;

pub fn cls_token(name: &str) -> u32 {
    match name {
        "rignet" => CLS_RIGNET,
        "vroid" => CLS_VROID,
        "articulation" => CLS_ARTICULATION,
        _ => CLS_NONE,
    }
}

fn is_cls(t: u32) -> bool {
    t == CLS_NONE || (CLS_RIGNET..=CLS_ARTICULATION).contains(&t)
}

fn is_part(t: u32) -> bool {
    t == SPRING || t == PART_BODY || t == PART_HAND
}

pub fn discretize(x: f64) -> u32 {
    let t = (x - -1.0) / 2.0 * NUM_DISCRETE as f64;
    // numpy round: half to even
    let r = t.round_ties_even();
    r.clamp(0.0, (NUM_DISCRETE - 1) as f64) as u32
}

pub fn undiscretize(t: u32) -> f32 {
    ((t as f32) + 0.5) / NUM_DISCRETE as f32 * 2.0 + -1.0
}

/// Tokens for a skeleton (joints in [-1, 1], root at index 0, parents
/// before children): bos, class, then per joint either its xyz or, when
/// its parent is not the previous joint, branch + parent xyz + xyz; eos.
pub fn tokenize(joints: &[[f64; 3]], parents: &[i64], cls: u32) -> Vec<u32> {
    let mut t = vec![BOS, cls];
    for i in 0..joints.len() {
        let branch = i != 0 && parents[i] != i as i64 - 1;
        let p = if i == 0 { 0 } else { parents[i].max(0) as usize };
        if branch {
            t.push(BRANCH);
            t.extend(joints[p].iter().map(|&v| discretize(v)));
        }
        t.extend(joints[i].iter().map(|&v| discretize(v)));
    }
    t.push(EOS);
    t
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Bos,
    ClsPartJoint,
    PartJoint,
    Joint2,
    Joint3,
    BranchPartJoint,
    Joint,
}

fn walk(ids: &[u32], mut on_joint3: impl FnMut()) -> State {
    let mut s = State::Bos;
    for &id in ids {
        s = match s {
            State::Bos => State::ClsPartJoint,
            State::ClsPartJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else if is_cls(id) {
                    State::PartJoint
                } else {
                    State::Joint
                }
            }
            State::PartJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::PartJoint
                }
            }
            State::Joint2 => State::Joint3,
            State::Joint3 => {
                on_joint3();
                State::BranchPartJoint
            }
            State::BranchPartJoint => {
                if id == BRANCH || id >= NUM_DISCRETE {
                    State::Joint
                } else {
                    State::Joint2
                }
            }
            State::Joint => State::Joint2,
        };
    }
    s
}

/// Grammar mask before the skeleton's eos: which ids may come next.
pub fn next_possible(ids: &[u32]) -> Vec<u32> {
    if ids.is_empty() {
        return vec![BOS];
    }
    let joints = || 0..NUM_DISCRETE;
    let parts = || [SPRING, PART_BODY, PART_HAND].into_iter();
    let classes = || [CLS_NONE, CLS_RIGNET, CLS_VROID, CLS_ARTICULATION].into_iter();
    match walk(ids, || {}) {
        State::Bos => vec![BOS],
        State::ClsPartJoint => classes().chain(parts()).chain(joints()).collect(),
        State::PartJoint => parts().chain(joints()).chain([EOS]).collect(),
        State::Joint2 | State::Joint3 | State::Joint => joints().collect(),
        State::BranchPartJoint => joints().chain(parts()).chain([BRANCH, EOS]).collect(),
    }
}

/// Number of joints in a token sequence (upstream `bones_in_sequence`,
/// including its counting rule: a joint is counted on the token after its
/// third coordinate, and a branch's parent triple is not a joint).
pub fn bones_in_sequence(ids: &[u32]) -> usize {
    let mut s = State::Bos;
    let mut n = 0usize;
    let mut is_branch = false;
    for &id in ids {
        s = match s {
            State::Bos => State::ClsPartJoint,
            State::ClsPartJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else if is_cls(id) {
                    State::PartJoint
                } else {
                    State::Joint
                }
            }
            State::PartJoint => {
                if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::PartJoint
                }
            }
            State::Joint2 => State::Joint3,
            State::Joint3 => {
                if !is_branch {
                    n += 1;
                }
                is_branch = false;
                State::BranchPartJoint
            }
            State::BranchPartJoint => {
                if id == BRANCH {
                    is_branch = true;
                    State::Joint
                } else if id < NUM_DISCRETE {
                    State::Joint2
                } else {
                    State::Joint
                }
            }
            State::Joint => State::Joint2,
        };
        if id == EOS {
            break;
        }
    }
    n
}

pub struct Skeleton {
    /// Joint heads in [-1, 1] (model frame).
    pub joints: Vec<[f32; 3]>,
    pub parents: Vec<i64>,
    pub names: Vec<String>,
}

fn d2(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Skeleton tokens (bos .. eos, trailing pads allowed) -> joints and
/// parents. Each joint's parent is the earlier joint nearest to its
/// parent position (upstream `make_skeleton`).
pub fn detokenize(ids: &[u32]) -> anyhow::Result<Skeleton> {
    use anyhow::bail;
    if ids.first() != Some(&BOS) {
        bail!("skeleton tokens do not start with bos");
    }
    let mut end = ids.len();
    while end > 0 && ids[end - 1] == PAD {
        end -= 1;
    }
    if end == 0 || ids[end - 1] != EOS {
        bail!("skeleton tokens do not end with eos");
    }
    let ids = &ids[1..end - 1];
    let un = |s: &[u32]| -> [f32; 3] { [undiscretize(s[0]), undiscretize(s[1]), undiscretize(s[2])] };
    let mut joints: Vec<[f32; 3]> = vec![];
    let mut pj: Vec<[f32; 3]> = vec![];
    let mut i = 0;
    let mut is_branch = false;
    let mut last: Option<[f32; 3]> = None;
    while i < ids.len() {
        let id = ids[i];
        if id < NUM_DISCRETE {
            if is_branch {
                if i + 6 > ids.len() {
                    bail!("truncated branch");
                }
                pj.push(un(&ids[i..i + 3]));
                joints.push(un(&ids[i + 3..i + 6]));
                i += 6;
            } else {
                if i + 3 > ids.len() {
                    bail!("truncated joint");
                }
                let cur = un(&ids[i..i + 3]);
                joints.push(cur);
                pj.push(if pj.is_empty() { cur } else { last.ok_or_else(|| anyhow::anyhow!("joint without parent"))? });
                i += 3;
            }
            last = Some(*joints.last().unwrap());
            is_branch = false;
        } else if id == BRANCH {
            is_branch = true;
            last = None;
            i += 1;
        } else if is_part(id) || is_cls(id) {
            i += 1;
        } else {
            bail!("unexpected token {id} in skeleton");
        }
    }
    if joints.is_empty() {
        bail!("no joints");
    }
    let mut parents = vec![-1i64];
    for k in 1..joints.len() {
        let mut best = (f32::INFINITY, -1i64);
        // upstream: reversed scan with strict '<' keeps the latest of equal distances
        for j in (0..k).rev() {
            let d = d2(&joints[j], &pj[k]);
            if d < best.0 {
                best = (d, j as i64);
            }
        }
        // upstream starts from dis = 999999
        parents.push(if best.0 < 999999.0 { best.1 } else { -1 });
    }
    let names = (0..joints.len()).map(|i| format!("bone_{i}")).collect();
    Ok(Skeleton { joints, parents, names })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_branch() {
        let j = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.3, 0.0, 0.2], [0.5, 0.0, 0.2]];
        let p = [-1, 0, 0, 2];
        let t = tokenize(&j, &p, CLS_ARTICULATION);
        assert_eq!(t[0], BOS);
        assert_eq!(*t.last().unwrap(), EOS);
        assert_eq!(bones_in_sequence(&t), 4);
        let s = detokenize(&t).unwrap();
        assert_eq!(s.parents, vec![-1, 0, 0, 2]);
        for (a, b) in s.joints.iter().zip(j.iter()) {
            for k in 0..3 {
                assert!((a[k] as f64 - b[k]).abs() <= 1.0 / 128.0 + 1e-6);
            }
        }
    }

    #[test]
    fn grammar() {
        assert_eq!(next_possible(&[BOS, CLS_ARTICULATION]).len(), 3 + 256 + 1);
        assert!(next_possible(&[BOS, CLS_ARTICULATION, 1, 2, 3]).contains(&BRANCH));
        assert!(!next_possible(&[BOS, CLS_ARTICULATION, 1, 2]).contains(&EOS));
    }
}
