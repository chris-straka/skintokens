//! A rigged GLB as the benchmark needs it: joints, weights, clips, and
//! the skin posed by a set of joint matrices.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use skintokens::geom::{self, M4, V3};
use skintokens::glb::Glb;

pub type M3 = [V3; 3];

#[derive(Clone, Copy, Debug)]
pub struct Trs {
    pub t: V3,
    pub r: [f64; 4],
    pub s: V3,
}

impl Trs {
    pub fn matrix(&self) -> M4 {
        geom::from_trs(self.t, self.r, self.s)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Path3 {
    Translation,
    Rotation,
    Scale,
}

pub struct Channel {
    pub node: usize,
    pub path: Path3,
    pub times: Vec<f64>,
    pub values: Vec<[f64; 4]>,
    pub step: bool,
}

pub struct Clip {
    pub name: String,
    pub channels: Vec<Channel>,
}

/// One bone's hold on a vertex: the weight, and the vertex in that
/// joint's own space (inverse bind matrix times the stored position).
#[derive(Clone, Copy)]
pub struct Influence {
    pub joint: usize,
    pub w: f64,
    pub q: V3,
}

pub struct Rig {
    pub glb: Glb,
    pub node_parent: Vec<Option<usize>>,
    /// Nodes, a parent before its children.
    pub order: Vec<usize>,
    pub rest: Vec<Trs>,
    pub rest_world: Vec<M4>,
    /// Node of each joint (every joint of every skin, once).
    pub joints: Vec<usize>,
    pub joint_of: HashMap<usize, usize>,
    pub names: Vec<String>,
    /// Nearest joint above each joint.
    pub parent: Vec<Option<usize>>,
    pub children: Vec<Vec<usize>>,
    /// Joints, a parent before its children.
    pub joint_order: Vec<usize>,
    pub inf: Vec<Vec<Influence>>,
    pub tris: Vec<[u32; 3]>,
    pub rest_pos: Vec<V3>,
    /// Summed weight per joint.
    pub mass: Vec<f64>,
    /// The joint that holds each vertex most (usize::MAX: unskinned).
    pub dominant: Vec<usize>,
    pub clips: Vec<Clip>,
}

pub fn m3_apply(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

pub fn m3_mul(a: &M3, b: &M3) -> M3 {
    [m3_apply(a, b[0]), m3_apply(a, b[1]), m3_apply(a, b[2])]
}

pub fn m3_t(a: &M3) -> M3 {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}

fn unit(v: V3) -> V3 {
    let n = geom::norm(v);
    if n > 1e-12 {
        geom::scale(v, 1.0 / n)
    } else {
        v
    }
}

/// The rotation in a matrix: its axes made unit and square to each other
/// (a mirrored matrix keeps its handedness, so the turn between two poses
/// of one bone is always a proper rotation).
pub fn rotation_of(m: &M4) -> M3 {
    let c = geom::cols(m);
    let x = unit(c[0]);
    let y = unit(geom::sub(c[1], geom::scale(x, geom::dot(c[1], x))));
    let mut z = geom::cross(x, y);
    if geom::dot(z, c[2]) < 0.0 {
        z = geom::scale(z, -1.0);
    }
    [x, y, z]
}

fn decompose(m: &M4) -> Trs {
    let c = geom::cols(m);
    let mut s = [geom::norm(c[0]), geom::norm(c[1]), geom::norm(c[2])];
    let mut r = rotation_of(m);
    if geom::dot(geom::cross(r[0], r[1]), r[2]) < 0.0 {
        r[2] = geom::scale(r[2], -1.0);
        s[2] = -s[2];
    }
    Trs { t: geom::translation(m), r: geom::quat_from_cols(r), s }
}

fn node_trs(node: &Value) -> Trs {
    if node["matrix"].is_array() {
        return decompose(&Glb::local_matrix(node));
    }
    let a = |k: &str, d: &[f64]| -> Vec<f64> {
        node[k]
            .as_array()
            .map(|v| v.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
            .unwrap_or_else(|| d.to_vec())
    };
    let t = a("translation", &[0.0, 0.0, 0.0]);
    let r = a("rotation", &[0.0, 0.0, 0.0, 1.0]);
    let s = a("scale", &[1.0, 1.0, 1.0]);
    Trs { t: [t[0], t[1], t[2]], r: [r[0], r[1], r[2], r[3]], s: [s[0], s[1], s[2]] }
}

fn m4_from_f32(m: [[f32; 4]; 4]) -> M4 {
    std::array::from_fn(|c| std::array::from_fn(|r| m[c][r] as f64))
}

impl Rig {
    pub fn read(path: &Path) -> Result<Rig> {
        Rig::from_glb(Glb::read(path)?)
    }

    pub fn from_glb(glb: Glb) -> Result<Rig> {
        let doc = glb.doc()?;
        let n = glb.nodes().len();
        let node_parent = glb.parents();
        // parents before children
        let mut order = Vec::with_capacity(n);
        let mut depth = vec![0usize; n];
        for i in 0..n {
            let (mut d, mut p) = (0, node_parent[i]);
            while let Some(x) = p {
                d += 1;
                p = node_parent[x];
                if d > 10_000 {
                    bail!("node cycle");
                }
            }
            depth[i] = d;
        }
        order.extend(0..n);
        order.sort_by_key(|&i| depth[i]);
        let rest: Vec<Trs> = glb.nodes().iter().map(node_trs).collect();
        let rest_world = glb.world_matrices();

        let mut joints: Vec<usize> = vec![];
        let mut joint_of: HashMap<usize, usize> = HashMap::new();
        let mut inf: Vec<Vec<Influence>> = vec![];
        let mut rest_pos: Vec<V3> = vec![];
        let mut tris: Vec<[u32; 3]> = vec![];
        let mut seen_mesh = std::collections::HashSet::new();
        for ni in glb.scene_nodes() {
            let node = doc.nodes().nth(ni).unwrap();
            let Some(mesh) = node.mesh() else { continue };
            if !seen_mesh.insert((mesh.index(), node.skin().map(|s| s.index()))) {
                continue;
            }
            // this skin's slots as joints of the whole rig
            let mut slot_joint: Vec<usize> = vec![];
            let mut ibm: Vec<M4> = vec![];
            if let Some(skin) = node.skin() {
                for j in skin.joints() {
                    let next = joints.len();
                    let g = *joint_of.entry(j.index()).or_insert_with(|| {
                        joints.push(j.index());
                        next
                    });
                    slot_joint.push(g);
                }
                ibm = match skin.reader(glb.buffer()).read_inverse_bind_matrices() {
                    Some(m) => m.map(m4_from_f32).collect(),
                    None => vec![geom::IDENTITY; slot_joint.len()],
                };
            }
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let r = prim.reader(glb.buffer());
                let Some(pos) = r.read_positions() else { continue };
                let pos: Vec<V3> = pos.map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
                let base = rest_pos.len() as u32;
                let idx: Vec<u32> = match r.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..pos.len() as u32).collect(),
                };
                tris.extend(idx.chunks_exact(3).map(|c| [c[0] + base, c[1] + base, c[2] + base]));
                let mut sets: Vec<(Vec<[u16; 4]>, Vec<[f32; 4]>)> = vec![];
                if !slot_joint.is_empty() {
                    for set in 0..4 {
                        match (r.read_joints(set), r.read_weights(set)) {
                            (Some(j), Some(w)) => sets.push((j.into_u16().collect(), w.into_f32().collect())),
                            _ => break,
                        }
                    }
                }
                for (vi, p) in pos.iter().enumerate() {
                    let mut list: Vec<Influence> = vec![];
                    for (js, ws) in &sets {
                        for k in 0..4 {
                            let w = ws[vi][k] as f64;
                            let slot = js[vi][k] as usize;
                            if w > 0.0 && slot < slot_joint.len() {
                                let g = slot_joint[slot];
                                let q = geom::apply(&ibm[slot], *p);
                                match list.iter_mut().find(|x| x.joint == g) {
                                    Some(x) => x.w += w,
                                    None => list.push(Influence { joint: g, w, q }),
                                }
                            }
                        }
                    }
                    let sum: f64 = list.iter().map(|x| x.w).sum();
                    if sum > 0.0 {
                        for x in &mut list {
                            x.w /= sum;
                        }
                    }
                    inf.push(list);
                    rest_pos.push([0.0; 3]); // filled below
                }
                // unskinned vertices stay where the node puts them
                let from = rest_pos.len() - pos.len();
                for (vi, p) in pos.iter().enumerate() {
                    if inf[from + vi].is_empty() {
                        rest_pos[from + vi] = geom::apply(&rest_world[ni], *p);
                    }
                }
            }
        }
        if tris.is_empty() {
            bail!("no triangle mesh in the scene");
        }
        let names: Vec<String> = joints
            .iter()
            .map(|&nd| glb.nodes()[nd]["name"].as_str().map(String::from).unwrap_or(format!("joint_{nd}")))
            .collect();
        let parent: Vec<Option<usize>> = joints
            .iter()
            .map(|&nd| {
                let mut p = node_parent[nd];
                while let Some(x) = p {
                    if let Some(&j) = joint_of.get(&x) {
                        return Some(j);
                    }
                    p = node_parent[x];
                }
                None
            })
            .collect();
        let mut children = vec![vec![]; joints.len()];
        for (j, p) in parent.iter().enumerate() {
            if let Some(p) = p {
                children[*p].push(j);
            }
        }
        let mut joint_order: Vec<usize> = (0..joints.len()).collect();
        joint_order.sort_by_key(|&j| depth[joints[j]]);
        let mut mass = vec![0.0; joints.len()];
        let mut dominant = vec![usize::MAX; inf.len()];
        for (v, list) in inf.iter().enumerate() {
            let mut best = 0.0;
            for x in list {
                mass[x.joint] += x.w;
                if x.w > best {
                    best = x.w;
                    dominant[v] = x.joint;
                }
            }
        }

        let mut clips = vec![];
        for (ai, anim) in doc.animations().enumerate() {
            let mut channels = vec![];
            for ch in anim.channels() {
                let path = match ch.target().property() {
                    gltf::animation::Property::Translation => Path3::Translation,
                    gltf::animation::Property::Rotation => Path3::Rotation,
                    gltf::animation::Property::Scale => Path3::Scale,
                    _ => continue,
                };
                let rd = ch.reader(glb.buffer());
                let Some(times) = rd.read_inputs() else { continue };
                let times: Vec<f64> = times.map(|t| t as f64).collect();
                let mut values: Vec<[f64; 4]> = match rd.read_outputs() {
                    Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                        it.map(|v| [v[0] as f64, v[1] as f64, v[2] as f64, 0.0]).collect()
                    }
                    Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                        it.map(|v| [v[0] as f64, v[1] as f64, v[2] as f64, 0.0]).collect()
                    }
                    Some(gltf::animation::util::ReadOutputs::Rotations(it)) => {
                        it.into_f32().map(|v| [v[0] as f64, v[1] as f64, v[2] as f64, v[3] as f64]).collect()
                    }
                    _ => continue,
                };
                let interp = ch.sampler().interpolation();
                if interp == gltf::animation::Interpolation::CubicSpline {
                    // (in-tangent, value, out-tangent) per key: the values alone, joined by straight lines
                    values = values.chunks_exact(3).map(|c| c[1]).collect();
                }
                if times.is_empty() || values.len() != times.len() {
                    continue;
                }
                channels.push(Channel {
                    node: ch.target().node().index(),
                    path,
                    times,
                    values,
                    step: interp == gltf::animation::Interpolation::Step,
                });
            }
            if !channels.is_empty() {
                clips.push(Clip { name: anim.name().map(String::from).unwrap_or(format!("clip_{ai}")), channels });
            }
        }

        let mut rig = Rig {
            glb,
            node_parent,
            order,
            rest,
            rest_world,
            joints,
            joint_of,
            names,
            parent,
            children,
            joint_order,
            inf,
            tris,
            rest_pos,
            mass,
            dominant,
            clips,
        };
        let worlds = rig.rest_world.clone();
        let posed = rig.skin_nodes(&worlds);
        rig.rest_pos = posed;
        Ok(rig)
    }

    /// Position of every joint in the rest pose.
    pub fn joint_rest(&self) -> Vec<V3> {
        self.joints.iter().map(|&n| geom::translation(&self.rest_world[n])).collect()
    }

    /// World matrix of every node at a time of a clip.
    pub fn pose_nodes(&self, clip: &Clip, t: f64) -> Vec<M4> {
        let mut local = self.rest.clone();
        for ch in &clip.channels {
            let v = sample(ch, t);
            let l = &mut local[ch.node];
            match ch.path {
                Path3::Translation => l.t = [v[0], v[1], v[2]],
                Path3::Scale => l.s = [v[0], v[1], v[2]],
                Path3::Rotation => l.r = v,
            }
        }
        let mut world = vec![geom::IDENTITY; local.len()];
        for &i in &self.order {
            let m = local[i].matrix();
            world[i] = match self.node_parent[i] {
                Some(p) => geom::mul(&world[p], &m),
                None => m,
            };
        }
        world
    }

    pub fn joint_worlds(&self, nodes: &[M4]) -> Vec<M4> {
        self.joints.iter().map(|&n| nodes[n]).collect()
    }

    pub fn skin_nodes(&self, nodes: &[M4]) -> Vec<V3> {
        self.skin(&self.joint_worlds(nodes))
    }

    /// The skin under a matrix per joint (linear blend).
    pub fn skin(&self, jw: &[M4]) -> Vec<V3> {
        self.inf
            .iter()
            .enumerate()
            .map(|(v, list)| {
                if list.is_empty() {
                    return self.rest_pos[v];
                }
                let mut p = [0.0; 3];
                for x in list {
                    let q = geom::apply(&jw[x.joint], x.q);
                    p = geom::add(p, geom::scale(q, x.w));
                }
                p
            })
            .collect()
    }

    /// The times a clip is sampled at: its own keys (the channel with the
    /// most of them), thinned evenly to at most `max` frames.
    pub fn frame_times(clip: &Clip, max: usize) -> Vec<f64> {
        let ch = clip.channels.iter().max_by_key(|c| c.times.len()).unwrap();
        let n = ch.times.len();
        if n <= max || max < 2 {
            return ch.times.clone();
        }
        (0..max).map(|k| ch.times[(k * (n - 1)) / (max - 1)]).collect()
    }

    /// A copy of this file with `clips` instead of its own animations:
    /// per clip a name, its frame times and a world matrix per joint per
    /// frame. Each joint gets a rotation track; a joint whose place under
    /// its parent changes gets a translation track too. Nodes above the
    /// joints stay as they are.
    pub fn with_clips(&self, clips: &[(String, Vec<f64>, Vec<Vec<M4>>)]) -> Result<Glb> {
        let mut g = Glb { json: self.glb.json.clone(), bin: self.glb.bin.clone() };
        let mut anims = vec![];
        let span = {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for p in &self.rest_pos {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max)
        };
        for (name, times, frames) in clips {
            if times.is_empty() {
                continue;
            }
            let tb: Vec<u8> = times.iter().flat_map(|t| (*t as f32).to_le_bytes()).collect();
            let t_acc = g.push_accessor(
                &tb,
                5126,
                "SCALAR",
                times.len(),
                Some((vec![times[0] as f32 as f64], vec![*times.last().unwrap() as f32 as f64])),
                false,
            );
            let mut samplers = vec![];
            let mut channels = vec![];
            for (j, &node) in self.joints.iter().enumerate() {
                let mut rots: Vec<[f64; 4]> = vec![];
                let mut trs: Vec<V3> = vec![];
                for jw in frames {
                    let above = match self.node_parent[node] {
                        Some(p) => match self.joint_of.get(&p) {
                            Some(&pj) => jw[pj],
                            None => self.rest_world[p],
                        },
                        None => geom::IDENTITY,
                    };
                    let local = geom::mul(&geom::inverse(&above), &jw[j]);
                    let mut q = geom::quat_from_cols(rotation_of(&local));
                    if let Some(prev) = rots.last() {
                        if prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2] + prev[3] * q[3] < 0.0 {
                            q = [-q[0], -q[1], -q[2], -q[3]];
                        }
                    }
                    rots.push(q);
                    trs.push(geom::translation(&local));
                }
                let rb: Vec<u8> = rots.iter().flat_map(|q| q.iter().flat_map(|v| (*v as f32).to_le_bytes())).collect();
                let r_acc = g.push_accessor(&rb, 5126, "VEC4", rots.len(), None, false);
                samplers.push(json!({"input": t_acc, "output": r_acc, "interpolation": "LINEAR"}));
                channels.push(json!({"sampler": samplers.len() - 1, "target": {"node": node, "path": "rotation"}}));
                let rest_t = self.rest[node].t;
                let moved = trs.iter().any(|t| geom::norm(geom::sub(*t, rest_t)) > 1e-6 * span.max(1e-9));
                if moved {
                    let b: Vec<u8> = trs.iter().flat_map(|t| t.iter().flat_map(|v| (*v as f32).to_le_bytes())).collect();
                    let a = g.push_accessor(&b, 5126, "VEC3", trs.len(), None, false);
                    samplers.push(json!({"input": t_acc, "output": a, "interpolation": "LINEAR"}));
                    channels.push(json!({"sampler": samplers.len() - 1, "target": {"node": node, "path": "translation"}}));
                }
            }
            anims.push(json!({"name": name, "samplers": samplers, "channels": channels}));
        }
        let obj = g.json.as_object_mut().ok_or_else(|| anyhow!("no JSON object"))?;
        if anims.is_empty() {
            obj.remove("animations");
        } else {
            obj.insert("animations".into(), json!(anims));
        }
        // a node given as a matrix cannot be animated: write its TRS instead
        let joints = self.joints.clone();
        if let Some(nodes) = g.json["nodes"].as_array_mut() {
            for &nd in &joints {
                if nodes[nd]["matrix"].is_array() {
                    let trs = self.rest[nd];
                    let o = nodes[nd].as_object_mut().unwrap();
                    o.remove("matrix");
                    o.insert("translation".into(), json!(trs.t));
                    o.insert("rotation".into(), json!(trs.r));
                    o.insert("scale".into(), json!(trs.s));
                }
            }
        }
        Ok(g)
    }
}

fn sample(ch: &Channel, t: f64) -> [f64; 4] {
    let n = ch.times.len();
    if t <= ch.times[0] || n == 1 {
        return ch.values[0];
    }
    if t >= ch.times[n - 1] {
        return ch.values[n - 1];
    }
    let k = ch.times.partition_point(|&x| x <= t).clamp(1, n - 1);
    let (t0, t1) = (ch.times[k - 1], ch.times[k]);
    let (a, b) = (ch.values[k - 1], ch.values[k]);
    if ch.step || t1 <= t0 {
        return a;
    }
    let u = (t - t0) / (t1 - t0);
    if ch.path != Path3::Rotation {
        return std::array::from_fn(|i| a[i] + (b[i] - a[i]) * u);
    }
    // the short way round
    let mut d: f64 = (0..4).map(|i| a[i] * b[i]).sum();
    let mut b = b;
    if d < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        d = -d;
    }
    let (wa, wb) = if d > 0.9995 {
        (1.0 - u, u)
    } else {
        let th = d.clamp(-1.0, 1.0).acos();
        (((1.0 - u) * th).sin() / th.sin(), (u * th).sin() / th.sin())
    };
    let q: [f64; 4] = std::array::from_fn(|i| a[i] * wa + b[i] * wb);
    let nrm = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    [q[0] / nrm, q[1] / nrm, q[2] / nrm, q[3] / nrm]
}
