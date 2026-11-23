//! GLB access in pure Rust. Reading goes through the `gltf` crate (parsed
//! and validated document, accessor readers); edits are made on the raw
//! JSON and binary chunk so everything the tool does not touch (materials,
//! textures, extensions, extras) is written back byte for byte.

use std::collections::{BTreeMap, BTreeSet};
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
            Ok(u32::from_le_bytes(
                b.get(o..o + 4)
                    .ok_or_else(|| anyhow!("truncated"))?
                    .try_into()?,
            ))
        };
        if b.len() < 12 || u32at(0)? != MAGIC {
            bail!("not a GLB (binary glTF)");
        }
        let total = (u32at(8)? as usize).min(b.len());
        let (mut off, mut js, mut bin) = (12usize, None, Vec::new());
        while off + 8 <= total {
            let len = u32at(off)? as usize;
            let typ = u32at(off + 4)?;
            let start = off + 8;
            let end = start
                .checked_add(len)
                .ok_or_else(|| anyhow!("chunk length overflow"))?;
            let data = b
                .get(start..end)
                .ok_or_else(|| anyhow!("truncated chunk"))?;
            match typ {
                CHUNK_JSON => js = Some(serde_json::from_slice::<Value>(data)?),
                CHUNK_BIN => bin = data.to_vec(),
                _ => {}
            }
            off = end;
        }
        let json = js.ok_or_else(|| anyhow!("no JSON chunk"))?;
        let g = Glb { json, bin };
        g.doc()?; // validate
        Ok(g)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut j = serde_json::to_vec(&self.json)?;
        while !j.len().is_multiple_of(4) {
            j.push(b' ');
        }
        let mut bin = self.bin.clone();
        while !bin.len().is_multiple_of(4) {
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
        std::fs::write(path, self.to_bytes()?)
            .with_context(|| format!("writing {}", path.display()))
    }

    /// Parsed view of the current JSON.
    pub fn doc(&self) -> Result<gltf::Document> {
        let mut j = serde_json::to_vec(&self.json)?;
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        let g = gltf::Gltf::from_slice(&glb_bytes(&j, &self.bin))
            .map_err(|e| anyhow!("invalid glTF: {e}"))?;
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
        self.json
            .get("nodes")
            .and_then(|v| v.as_array())
            .unwrap_or(&EMPTY)
    }

    pub fn local_matrix(node: &Value) -> M4 {
        if let Some(m) = node.get("matrix").and_then(|v| v.as_array()) {
            let f: Vec<f64> = m.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
            if f.len() == 16 {
                return [
                    [f[0], f[1], f[2], f[3]],
                    [f[4], f[5], f[6], f[7]],
                    [f[8], f[9], f[10], f[11]],
                    [f[12], f[13], f[14], f[15]],
                ];
            }
        }
        let v3 = |k: &str, d: [f64; 3]| -> [f64; 3] {
            match node.get(k).and_then(|v| v.as_array()) {
                Some(a) if a.len() >= 3 => [
                    a[0].as_f64().unwrap_or(d[0]),
                    a[1].as_f64().unwrap_or(d[1]),
                    a[2].as_f64().unwrap_or(d[2]),
                ],
                _ => d,
            }
        };
        let v4 = |k: &str, d: [f64; 4]| -> [f64; 4] {
            match node.get(k).and_then(|v| v.as_array()) {
                Some(a) if a.len() >= 4 => [
                    a[0].as_f64().unwrap_or(d[0]),
                    a[1].as_f64().unwrap_or(d[1]),
                    a[2].as_f64().unwrap_or(d[2]),
                    a[3].as_f64().unwrap_or(d[3]),
                ],
                _ => d,
            }
        };
        let t = v3("translation", [0.0, 0.0, 0.0]);
        let r = v4("rotation", [0.0, 0.0, 0.0, 1.0]);
        let s = v3("scale", [1.0, 1.0, 1.0]);
        geom::from_trs(t, r, s)
    }

    /// parent[i] of every node (None for roots).
    pub fn parents(&self) -> Vec<Option<usize>> {
        let n = self.nodes().len();
        let mut p = vec![None; n];
        for (i, nd) in self.nodes().iter().enumerate() {
            for c in nd
                .get("children")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
            {
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
        fn get(
            i: usize,
            g: &Glb,
            parents: &[Option<usize>],
            w: &mut Vec<Option<M4>>,
            depth: usize,
        ) -> M4 {
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
        let si = self.json.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let roots: Vec<usize> = match self.json.get("scenes").and_then(|s| s.get(si)) {
            Some(s) => s
                .get("nodes")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect(),
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
            let kids: Vec<usize> = self.nodes()[i]
                .get("children")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect();
            stack.extend(kids.into_iter().rev());
        }
        out
    }

    // ---------------------------------------------------------- buffers

    fn check_mut_schema(&self) -> Result<()> {
        let obj = self
            .json
            .as_object()
            .ok_or_else(|| anyhow!("GLB JSON root is not an object"))?;
        if let Some(bufs) = obj.get("buffers") {
            let arr = bufs
                .as_array()
                .ok_or_else(|| anyhow!("GLB JSON 'buffers' is not an array"))?;
            if let Some(b0) = arr.first() {
                if !b0.is_object() {
                    bail!("GLB JSON 'buffers[0]' is not an object");
                }
            }
        }
        if let Some(views) = obj.get("bufferViews") {
            if !views.is_array() {
                bail!("GLB JSON 'bufferViews' is not an array");
            }
        }
        if let Some(accs) = obj.get("accessors") {
            if !accs.is_array() {
                bail!("GLB JSON 'accessors' is not an array");
            }
        }
        Ok(())
    }

    /// Appends raw bytes as a new bufferView + accessor; returns the accessor index.
    pub fn push_accessor(
        &mut self,
        bytes: &[u8],
        component: u32,
        kind: &str,
        count: usize,
        minmax: Option<(Vec<f64>, Vec<f64>)>,
        normalized: bool,
    ) -> Result<usize> {
        self.check_mut_schema()?;
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let bin_len = self.bin.len();

        let obj = self
            .json
            .as_object_mut()
            .ok_or_else(|| anyhow!("GLB JSON root is not an object"))?;
        let bufs = obj
            .entry("buffers")
            .or_insert_with(|| json!([{"byteLength": 0}]))
            .as_array_mut()
            .ok_or_else(|| anyhow!("GLB JSON 'buffers' is not an array"))?;
        if bufs.is_empty() {
            bufs.push(json!({"byteLength": bin_len}));
        } else {
            let b0 = bufs[0]
                .as_object_mut()
                .ok_or_else(|| anyhow!("GLB JSON 'buffers[0]' is not an object"))?;
            b0.insert("byteLength".into(), json!(bin_len));
        }

        let view = {
            let views = obj
                .entry("bufferViews")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .ok_or_else(|| anyhow!("GLB JSON 'bufferViews' is not an array"))?;
            views.push(json!({"buffer": 0, "byteOffset": offset, "byteLength": bytes.len()}));
            views.len() - 1
        };

        let mut acc =
            json!({"bufferView": view, "componentType": component, "count": count, "type": kind});
        if normalized {
            acc["normalized"] = json!(true);
        }
        if let Some((mn, mx)) = minmax {
            acc["min"] = json!(mn);
            acc["max"] = json!(mx);
        }
        let accs = obj
            .entry("accessors")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| anyhow!("GLB JSON 'accessors' is not an array"))?;
        accs.push(acc);
        Ok(accs.len() - 1)
    }

    pub fn push_mat4s(&mut self, ms: &[M4]) -> Result<usize> {
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
    pub fn push_skin_attrs(
        &mut self,
        joints: &[[u16; 4]],
        weights: &[[f32; 4]],
    ) -> Result<(usize, usize)> {
        if joints.len() != weights.len() {
            bail!(
                "skin attributes length mismatch: {} joints vs {} weights",
                joints.len(),
                weights.len()
            );
        }
        self.check_mut_schema()?;
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
        let ja = self.push_accessor(
            &jb,
            if wide { 5123 } else { 5121 },
            "VEC4",
            joints.len(),
            None,
            false,
        )?;
        let mut wb = Vec::with_capacity(weights.len() * 16);
        for w in weights {
            for v in w {
                wb.extend_from_slice(&v.to_le_bytes());
            }
        }
        let wa = self.push_accessor(&wb, 5126, "VEC4", weights.len(), None, false)?;
        Ok((ja, wa))
    }
}

fn glb_bytes(j: &[u8], bin: &[u8]) -> Vec<u8> {
    let mut b = bin.to_vec();
    while !b.len().is_multiple_of(4) {
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

/// Where one skinned vertex rests: the weighted sum over its influences of
/// `jointWorld * inverseBind * p`, normalised. `None` when it has no positive
/// influence, so the caller can fall back to the mesh node's world matrix.
pub fn skin_rest(p: V3, inf: &[(usize, f64)], joint_world: &[M4], ibm: &[M4]) -> Option<V3> {
    let (mut acc, mut sum) = ([0.0; 3], 0.0);
    for &(slot, w) in inf {
        if w > 0.0 && slot < joint_world.len() && slot < ibm.len() {
            let m = geom::mul(&joint_world[slot], &ibm[slot]);
            acc = geom::add(acc, geom::scale(geom::apply(&m, p), w));
            sum += w;
        }
    }
    (sum > 0.0).then(|| geom::scale(acc, 1.0 / sum))
}

/// Which primitives and in which frame.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    /// Every mesh node in the scene, positions in world space (node transforms applied).
    World,
    /// Every mesh node in the scene at rest: skinned vertices through their
    /// joints (`jointWorld * inverseBind * p`), unskinned ones by their node.
    Rest,
    /// Primitives of nodes using skin 0, positions as stored (the skin's bind space).
    SkinBind,
}

impl Glb {
    pub fn parts(&self, frame: Frame) -> Result<Vec<Part>> {
        let doc = self.doc()?;
        let worlds = self.world_matrices();
        let mut out = vec![];
        let mut used_mesh = BTreeSet::new();
        for ni in self.scene_nodes() {
            let node = doc
                .nodes()
                .nth(ni)
                .ok_or_else(|| anyhow!("scene node {ni} missing in glTF document"))?;
            let Some(mesh) = node.mesh() else { continue };
            let skin = node.skin().map(|s| s.index());
            if matches!(frame, Frame::SkinBind) && skin != Some(0) {
                continue;
            }
            // a mesh instanced twice gets weights once (first instance)
            if !used_mesh.insert(mesh.index()) {
                continue;
            }
            let skin_mats: Option<(Vec<M4>, Vec<M4>)> = if frame == Frame::Rest {
                node.skin().map(|s| {
                    let jw: Vec<M4> = s
                        .joints()
                        .map(|j| worlds.get(j.index()).copied().unwrap_or(geom::IDENTITY))
                        .collect();
                    let ibm: Vec<M4> = match s.reader(self.buffer()).read_inverse_bind_matrices() {
                        Some(m) => m
                            .map(|x| {
                                std::array::from_fn(|c| std::array::from_fn(|r| x[c][r] as f64))
                            })
                            .collect(),
                        None => vec![geom::IDENTITY; jw.len()],
                    };
                    (jw, ibm)
                })
            } else {
                None
            };
            for (pi, prim) in mesh.primitives().enumerate() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let r = prim.reader(self.buffer());
                let Some(pos) = r.read_positions() else {
                    continue;
                };
                let raw_pos: Vec<V3> = pos
                    .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                    .collect();
                let mut sets = vec![];
                if skin_mats.is_some() {
                    for set in 0..4 {
                        match (r.read_joints(set), r.read_weights(set)) {
                            (Some(j), Some(w)) => sets.push((
                                j.into_u16().collect::<Vec<_>>(),
                                w.into_f32().collect::<Vec<_>>(),
                            )),
                            _ => break,
                        }
                    }
                }
                let positions: Vec<V3> = raw_pos
                    .iter()
                    .enumerate()
                    .map(|(vi, &p)| match frame {
                        Frame::World => geom::apply(&worlds[ni], p),
                        Frame::SkinBind => p,
                        Frame::Rest => skin_mats
                            .as_ref()
                            .and_then(|(jw, ibm)| {
                                let inf: Vec<(usize, f64)> = sets
                                    .iter()
                                    .filter_map(|(js, ws)| {
                                        let jr = js.get(vi)?;
                                        let wr = ws.get(vi)?;
                                        Some((0..4).map(move |k| (jr[k] as usize, wr[k] as f64)))
                                    })
                                    .flatten()
                                    .collect();
                                skin_rest(p, &inf, jw, ibm)
                            })
                            .unwrap_or_else(|| geom::apply(&worlds[ni], p)),
                    })
                    .collect();
                let idx: Vec<u32> = match r.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                let (tri_chunks, _) = idx.as_chunks::<3>();
                let triangles = tri_chunks.to_vec();
                let joints = r.read_joints(0).map(|j| j.into_u16().collect());
                let weights = r.read_weights(0).map(|w| w.into_f32().collect());
                out.push(Part {
                    node: ni,
                    mesh: mesh.index(),
                    prim: pi,
                    positions,
                    triangles,
                    joints,
                    weights,
                    skin,
                });
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
        let n_nodes = self.nodes().len();
        let ibm: Vec<[[f32; 4]; 4]> = match skin.reader(self.buffer()).read_inverse_bind_matrices()
        {
            Some(m) => m.collect(),
            None => vec![
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
                joints.len()
            ],
        };
        let slot_of: BTreeMap<usize, usize> =
            joints.iter().enumerate().map(|(k, &n)| (n, k)).collect();
        let parents = self.parents();
        // nearest joint ancestor of each joint
        let joint_parent = |n: usize| -> Option<usize> {
            let mut p = parents.get(n).copied().flatten();
            let mut steps = 0usize;
            while let Some(x) = p {
                if steps >= parents.len() {
                    break;
                }
                steps += 1;
                if slot_of.contains_key(&x) {
                    return Some(x);
                }
                p = parents.get(x).copied().flatten();
            }
            None
        };
        let tops: Vec<usize> = joints
            .iter()
            .copied()
            .filter(|&n| joint_parent(n).is_none())
            .collect();
        // depth-first through the node tree, children in order, collecting joints
        let mut order = vec![];
        fn visit(
            g: &Glb,
            n: usize,
            n_nodes: usize,
            slot_of: &BTreeMap<usize, usize>,
            order: &mut Vec<usize>,
            depth: usize,
        ) {
            if depth > 1000 || n >= n_nodes {
                return;
            }
            if slot_of.contains_key(&n) && !order.contains(&n) {
                order.push(n);
            }
            let kids: Vec<usize> = g.nodes()[n]
                .get("children")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect();
            for k in kids {
                visit(g, k, n_nodes, slot_of, order, depth + 1);
            }
        }
        for t in tops {
            visit(self, t, n_nodes, &slot_of, &mut order, 0);
        }
        let &root = order
            .first()
            .ok_or_else(|| anyhow!("skin has no reachable joint nodes"))?;
        let pos_of: BTreeMap<usize, usize> =
            order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let names = order
            .iter()
            .map(|&n| {
                self.nodes()[n]
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| format!("joint_{n}"))
            })
            .collect();
        let parents_out = order
            .iter()
            .map(|&n| {
                joint_parent(n)
                    .and_then(|p| pos_of.get(&p))
                    .map(|&i| i as i64)
                    .unwrap_or(-1)
            })
            .collect();
        let mut heads = Vec::with_capacity(order.len());
        let mut skin_slot = Vec::with_capacity(order.len());
        for &n in &order {
            let slot = *slot_of
                .get(&n)
                .ok_or_else(|| anyhow!("missing skin slot for joint node {n}"))?;
            let m = ibm
                .get(slot)
                .copied()
                .ok_or_else(|| anyhow!("missing inverse bind matrix for slot {slot}"))?;
            let m4: M4 = std::array::from_fn(|c| std::array::from_fn(|r| m[c][r] as f64));
            heads.push(geom::translation(&geom::inverse(&m4)));
            skin_slot.push(slot);
        }
        let worlds = self.world_matrices();
        let root_slot = skin_slot[0];
        let ibm0_raw = ibm[root_slot];
        let ibm0: M4 = std::array::from_fn(|c| std::array::from_fn(|r| ibm0_raw[c][r] as f64));
        let bind = geom::mul(&worlds[root], &ibm0);
        let arm = parents[root].map(|p| worlds[p]).unwrap_or(geom::IDENTITY);
        let to_armature = geom::mul(&geom::inverse(&arm), &bind);
        Ok(SkinInfo {
            nodes: order,
            skin_slot,
            names,
            parents: parents_out,
            heads,
            to_armature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_glb_json_returns_errors_without_panicking() {
        for bad in [
            json!([]),
            json!("not-an-object"),
            json!({"buffers": "bad"}),
            json!({"buffers": [42]}),
            json!({"bufferViews": {}}),
            json!({"accessors": {}}),
        ] {
            let mut g = Glb {
                json: bad.clone(),
                bin: vec![],
            };
            let acc = g.push_accessor(&[1, 2, 3, 4], 5126, "SCALAR", 1, None, false);
            let mat = g.push_mat4s(&[geom::IDENTITY]);
            let sk = g.push_skin_attrs(&[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
            assert!(acc.is_err(), "expected error on push_accessor for {bad}");
            assert!(mat.is_err(), "expected error on push_mat4s for {bad}");
            assert!(sk.is_err(), "expected error on push_skin_attrs for {bad}");
            assert!(
                g.bin.is_empty(),
                "failed JSON schema check leaked bytes into bin for {bad}"
            );
        }
        let mut mismatch = Glb {
            json: json!({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
            bin: vec![],
        };
        assert!(mismatch
            .push_skin_attrs(
                &[[0, 0, 0, 0]],
                &[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            )
            .is_err());
        assert!(mismatch.bin.is_empty());
    }
}
