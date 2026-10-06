//! GLB access in pure Rust. Reading goes through the `gltf` crate (parsed
//! and validated document, accessor readers); edits are made on the raw
//! JSON and binary chunk so everything the tool does not touch (materials,
//! textures, extensions, extras) is written back byte for byte.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::geom::{self, M4, V3};

pub struct Glb {
    pub json: Value,
    pub bin: Vec<u8>,
}

const MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

impl Glb {
    pub fn read(path: &Path) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&b).with_context(|| format!("{}: unreadable GLB", path.display()))
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let u32at = |o: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(b.get(o..o + 4).ok_or_else(|| anyhow!("truncated"))?.try_into()?))
        };
        if b.len() < 12 || u32at(0)? != MAGIC {
            bail!("not a GLB (binary glTF)");
        }
        let total = (u32at(8)? as usize).min(b.len());
        let (mut off, mut js, mut bin) = (12usize, None, Vec::new());
        while off + 8 <= total {
            let len = u32at(off)? as usize;
            let typ = u32at(off + 4)?;
            let data = b.get(off + 8..off + 8 + len).ok_or_else(|| anyhow!("truncated chunk"))?;
            match typ {
                CHUNK_JSON => js = Some(serde_json::from_slice::<Value>(data)?),
                CHUNK_BIN => bin = data.to_vec(),
                _ => {}
            }
            off += 8 + len;
        }
        let json = js.ok_or_else(|| anyhow!("no JSON chunk"))?;
        let g = Glb { json, bin };
        g.doc()?; // validate
        Ok(g)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut j = serde_json::to_vec(&self.json)?;
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        let mut bin = self.bin.clone();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let mut out = Vec::with_capacity(28 + j.len() + bin.len());
        let total = 12 + 8 + j.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(j.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&j);
        if !bin.is_empty() {
            out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
            out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
            out.extend_from_slice(&bin);
        }
        Ok(out)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).ok();
        }
        std::fs::write(path, self.to_bytes()?).with_context(|| format!("writing {}", path.display()))
    }

    /// Parsed view of the current JSON.
    pub fn doc(&self) -> Result<gltf::Document> {
        let mut j = serde_json::to_vec(&self.json)?;
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        let g = gltf::Gltf::from_slice(&glb_bytes(&j, &self.bin)).map_err(|e| anyhow!("invalid glTF: {e}"))?;
        for b in g.document.buffers() {
            if let gltf::buffer::Source::Uri(u) = b.source() {
                bail!("external buffer {u:?} is not supported (pack the file as a self-contained .glb)");
            }
        }
        Ok(g.document)
    }

    pub fn buffer<'a>(&'a self) -> impl Fn(gltf::Buffer<'_>) -> Option<&'a [u8]> + Clone + 'a {
        move |_b| Some(self.bin.as_slice())
    }

    // ------------------------------------------------------------ nodes

    pub fn nodes(&self) -> &Vec<Value> {
        static EMPTY: Vec<Value> = Vec::new();
        self.json["nodes"].as_array().unwrap_or(&EMPTY)
    }

    pub fn local_matrix(node: &Value) -> M4 {
        if let Some(m) = node["matrix"].as_array() {
            let f: Vec<f64> = m.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
            if f.len() == 16 {
                return [[f[0], f[1], f[2], f[3]], [f[4], f[5], f[6], f[7]], [f[8], f[9], f[10], f[11]], [f[12], f[13], f[14], f[15]]];
            }
        }
        let a = |k: &str, d: &[f64]| -> Vec<f64> {
            node[k].as_array().map(|v| v.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_else(|| d.to_vec())
        };
        let t = a("translation", &[0.0, 0.0, 0.0]);
        let r = a("rotation", &[0.0, 0.0, 0.0, 1.0]);
        let s = a("scale", &[1.0, 1.0, 1.0]);
        geom::from_trs([t[0], t[1], t[2]], [r[0], r[1], r[2], r[3]], [s[0], s[1], s[2]])
    }

    /// parent[i] of every node (None for roots).
    pub fn parents(&self) -> Vec<Option<usize>> {
        let n = self.nodes().len();
        let mut p = vec![None; n];
        for (i, nd) in self.nodes().iter().enumerate() {
            for c in nd["children"].as_array().into_iter().flatten() {
                if let Some(c) = c.as_u64() {
                    if (c as usize) < n {
                        p[c as usize] = Some(i);
                    }
                }
            }
        }
        p
    }

    /// World matrix of every node.
    pub fn world_matrices(&self) -> Vec<M4> {
        let parents = self.parents();
        let n = self.nodes().len();
        let mut w: Vec<Option<M4>> = vec![None; n];
        fn get(i: usize, g: &Glb, parents: &[Option<usize>], w: &mut Vec<Option<M4>>, depth: usize) -> M4 {
            if let Some(m) = w[i] {
                return m;
            }
            let l = Glb::local_matrix(&g.nodes()[i]);
            let m = match parents[i] {
                Some(p) if depth < 1000 => geom::mul(&get(p, g, parents, w, depth + 1), &l),
                _ => l,
            };
            w[i] = Some(m);
            m
        }
        (0..n).map(|i| get(i, self, &parents, &mut w, 0)).collect()
    }

    /// Nodes reachable from the default scene (all nodes if there is no scene).
    pub fn scene_nodes(&self) -> Vec<usize> {
        let si = self.json["scene"].as_u64().unwrap_or(0) as usize;
        let roots: Vec<usize> = match self.json["scenes"].get(si) {
            Some(s) => s["nodes"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect(),
            None => return (0..self.nodes().len()).collect(),
        };
        let mut out = vec![];
        let mut stack: Vec<usize> = roots.into_iter().rev().collect();
        let mut seen = vec![false; self.nodes().len()];
        while let Some(i) = stack.pop() {
            if i >= seen.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            out.push(i);
            let kids: Vec<usize> = self.nodes()[i]["children"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
            stack.extend(kids.into_iter().rev());
        }
        out
    }

    // ---------------------------------------------------------- buffers

    /// Appends raw bytes as a new bufferView + accessor; returns the accessor index.
    pub fn push_accessor(&mut self, bytes: &[u8], component: u32, kind: &str, count: usize, minmax: Option<(Vec<f64>, Vec<f64>)>, normalized: bool) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        if self.json["buffers"].as_array().map(|a| a.is_empty()).unwrap_or(true) {
            self.json["buffers"] = json!([{"byteLength": 0}]);
        }
        self.json["buffers"][0]["byteLength"] = json!(self.bin.len());
        let views = self.json.as_object_mut().unwrap().entry("bufferViews").or_insert(json!([]));
        let views = views.as_array_mut().unwrap();
        views.push(json!({"buffer": 0, "byteOffset": offset, "byteLength": bytes.len()}));
        let view = views.len() - 1;
        let mut acc = json!({"bufferView": view, "componentType": component, "count": count, "type": kind});
        if normalized {
            acc["normalized"] = json!(true);
        }
        if let Some((mn, mx)) = minmax {
            acc["min"] = json!(mn);
            acc["max"] = json!(mx);
        }
        let accs = self.json.as_object_mut().unwrap().entry("accessors").or_insert(json!([]));
        let accs = accs.as_array_mut().unwrap();
        accs.push(acc);
        accs.len() - 1
    }

    pub fn push_mat4s(&mut self, ms: &[M4]) -> usize {
        let mut b = Vec::with_capacity(ms.len() * 64);
        for m in ms {
            for c in m {
                for v in c {
                    b.extend_from_slice(&(*v as f32).to_le_bytes());
                }
            }
        }
        self.push_accessor(&b, 5126, "MAT4", ms.len(), None, false)
    }

    /// JOINTS_0 (u8 or u16) and WEIGHTS_0 (f32) accessors for 4 influences per vertex.
    pub fn push_skin_attrs(&mut self, joints: &[[u16; 4]], weights: &[[f32; 4]]) -> (usize, usize) {
        let wide = joints.iter().flatten().any(|&j| j > 255);
        let mut jb = Vec::with_capacity(joints.len() * 8);
        for j in joints {
            for &v in j {
                if wide {
                    jb.extend_from_slice(&v.to_le_bytes());
                } else {
                    jb.push(v as u8);
                }
            }
        }
        let ja = self.push_accessor(&jb, if wide { 5123 } else { 5121 }, "VEC4", joints.len(), None, false);
        let mut wb = Vec::with_capacity(weights.len() * 16);
        for w in weights {
            for v in w {
                wb.extend_from_slice(&v.to_le_bytes());
            }
        }
        let wa = self.push_accessor(&wb, 5126, "VEC4", weights.len(), None, false);
        (ja, wa)
    }
}

fn glb_bytes(j: &[u8], bin: &[u8]) -> Vec<u8> {
    let mut b = bin.to_vec();
    while b.len() % 4 != 0 {
        b.push(0);
    }
    let total = 12 + 8 + j.len() + if b.is_empty() { 0 } else { 8 + b.len() };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(j);
    if !b.is_empty() {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&b);
    }
    out
}

/// One triangle primitive placed in the scene.
#[derive(Clone, Debug)]
pub struct Part {
    pub node: usize,
    pub mesh: usize,
    pub prim: usize,
    /// Vertex positions (frame chosen by the caller of `parts`).
    pub positions: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
    /// Existing skin data, if any.
    pub joints: Option<Vec<[u16; 4]>>,
    pub weights: Option<Vec<[f32; 4]>>,
    pub skin: Option<usize>,
}

/// Which primitives and in which frame.
pub enum Frame {
    /// Every mesh node in the scene, positions in world space (node transforms applied).
    World,
    /// Primitives of nodes using skin 0, positions as stored (the skin's bind space).
    SkinBind,
}

impl Glb {
    pub fn parts(&self, frame: Frame) -> Result<Vec<Part>> {
        let doc = self.doc()?;
        let worlds = self.world_matrices();
        let mut out = vec![];
        let mut used_mesh = std::collections::HashSet::new();
        for ni in self.scene_nodes() {
            let node = doc.nodes().nth(ni).unwrap();
            let Some(mesh) = node.mesh() else { continue };
            let skin = node.skin().map(|s| s.index());
            if matches!(frame, Frame::SkinBind) && skin != Some(0) {
                continue;
            }
            // a mesh instanced twice gets weights once (first instance)
            if !used_mesh.insert(mesh.index()) {
                continue;
            }
            for (pi, prim) in mesh.primitives().enumerate() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let r = prim.reader(self.buffer());
                let Some(pos) = r.read_positions() else { continue };
                let positions: Vec<V3> = pos
                    .map(|p| {
                        let p = [p[0] as f64, p[1] as f64, p[2] as f64];
                        match frame {
                            Frame::World => geom::apply(&worlds[ni], p),
                            Frame::SkinBind => p,
                        }
                    })
                    .collect();
                let idx: Vec<u32> = match r.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                let triangles = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                let joints = r.read_joints(0).map(|j| j.into_u16().collect());
                let weights = r.read_weights(0).map(|w| w.into_f32().collect());
                out.push(Part { node: ni, mesh: mesh.index(), prim: pi, positions, triangles, joints, weights, skin });
            }
        }
        if out.is_empty() {
            bail!("no triangle mesh in the scene");
        }
        Ok(out)
    }
}

/// The first skin's joints, as Blender's importer orders bones: depth
/// first from the topmost joints, children in node order.
pub struct SkinInfo {
    /// Node index per joint (in that order).
    pub nodes: Vec<usize>,
    /// Index into skin.joints per joint.
    pub skin_slot: Vec<usize>,
    pub names: Vec<String>,
    pub parents: Vec<i64>,
    /// Joint heads in the skin's bind space (from the inverse bind matrices).
    pub heads: Vec<V3>,
    /// Bind space -> Blender's armature space (bind-pose world, relative to
    /// the node above the top joint). Upstream's data lives there and is
    /// float32, which decides how joints on token bin edges round.
    pub to_armature: M4,
}

impl Glb {
    pub fn skin_info(&self) -> Result<SkinInfo> {
        let doc = self.doc()?;
        let skin = doc.skins().next().ok_or_else(|| anyhow!("no skin"))?;
        let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        if joints.is_empty() {
            bail!("skin has no joints");
        }
        let ibm: Vec<[[f32; 4]; 4]> = match skin.reader(self.buffer()).read_inverse_bind_matrices() {
            Some(m) => m.collect(),
            None => vec![[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]; joints.len()],
        };
        let slot_of: std::collections::HashMap<usize, usize> = joints.iter().enumerate().map(|(k, &n)| (n, k)).collect();
        let parents = self.parents();
        // nearest joint ancestor of each joint
        let joint_parent = |n: usize| -> Option<usize> {
            let mut p = parents[n];
            while let Some(x) = p {
                if slot_of.contains_key(&x) {
                    return Some(x);
                }
                p = parents[x];
            }
            None
        };
        let tops: Vec<usize> = joints.iter().copied().filter(|&n| joint_parent(n).is_none()).collect();
        // depth-first through the node tree, children in order, collecting joints
        let mut order = vec![];
        fn visit(g: &Glb, n: usize, slot_of: &std::collections::HashMap<usize, usize>, order: &mut Vec<usize>, depth: usize) {
            if depth > 1000 {
                return;
            }
            if slot_of.contains_key(&n) && !order.contains(&n) {
                order.push(n);
            }
            let kids: Vec<usize> = g.nodes()[n]["children"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
            for k in kids {
                visit(g, k, slot_of, order, depth + 1);
            }
        }
        for t in tops {
            visit(self, t, &slot_of, &mut order, 0);
        }
        let pos_of: std::collections::HashMap<usize, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let names = order.iter().map(|&n| self.nodes()[n]["name"].as_str().map(String::from).unwrap_or(format!("joint_{n}"))).collect();
        let parents_out = order.iter().map(|&n| joint_parent(n).map(|p| pos_of[&p] as i64).unwrap_or(-1)).collect();
        let heads = order
            .iter()
            .map(|&n| {
                let m = ibm[slot_of[&n]];
                let m: M4 = [
                    [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64, m[0][3] as f64],
                    [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64, m[1][3] as f64],
                    [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64, m[2][3] as f64],
                    [m[3][0] as f64, m[3][1] as f64, m[3][2] as f64, m[3][3] as f64],
                ];
                geom::translation(&geom::inverse(&m))
            })
            .collect();
        let skin_slot: Vec<usize> = order.iter().map(|n| slot_of[n]).collect();
        let worlds = self.world_matrices();
        let root = order[0];
        let ibm0: M4 = std::array::from_fn(|c| std::array::from_fn(|r| ibm[slot_of[&root]][c][r] as f64));
        let bind = geom::mul(&worlds[root], &ibm0);
        let arm = parents[root].map(|p| worlds[p]).unwrap_or(geom::IDENTITY);
        let to_armature = geom::mul(&geom::inverse(&arm), &bind);
        Ok(SkinInfo { nodes: order, skin_slot, names, parents: parents_out, heads, to_armature })
    }
}
