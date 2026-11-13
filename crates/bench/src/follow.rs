//! A rig made to follow a video seen from the side: its joints are
//! turned, frame by frame, until its outline from the same camera lies
//! on the video's outline.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use skintokens::geom::{self, M4, V3};

use crate::rig::{m3_apply, m3_mul, Rig, M3};

/// The camera `tools/bench/render_side.py` writes: orthographic, looking
/// along +X, +Z to the right of the picture, +Y up.
#[derive(Clone)]
pub struct Camera {
    pub width: usize,
    pub height: usize,
    pub px_per_m: f64,
    pub centre_z: f64,
    pub centre_y: f64,
    pub ground_y: f64,
}

impl Camera {
    pub fn write(&self, path: &Path, grey: f64) -> Result<()> {
        let v = json!({"width": self.width, "height": self.height, "px_per_m": self.px_per_m, "centre_z": self.centre_z,
            "centre_y": self.centre_y, "ground_y": self.ground_y, "grey": grey, "looks_along": "+x", "right": "+z", "up": "+y"});
        std::fs::write(path, serde_json::to_string_pretty(&v)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Camera> {
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )?;
        let f = |k: &str| {
            v[k].as_f64()
                .ok_or_else(|| anyhow!("{}: no {k}", path.display()))
        };
        Ok(Camera {
            width: f("width")? as usize,
            height: f("height")? as usize,
            px_per_m: f("px_per_m")?,
            centre_z: f("centre_z")?,
            centre_y: f("centre_y")?,
            ground_y: f("ground_y")?,
        })
    }
}

/// A picture of what is body and what is background.
#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub on: Vec<bool>,
}

impl Mask {
    fn new(w: usize, h: usize) -> Mask {
        Mask {
            w,
            h,
            on: vec![false; w * h],
        }
    }

    pub fn count(&self) -> usize {
        self.on.iter().filter(|x| **x).count()
    }

    /// Distance of every pixel to the nearest pixel that is on (0 on
    /// them), by two chamfer passes.
    fn distance(&self) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let far = (w + h) as f32;
        let mut d: Vec<f32> = self.on.iter().map(|&b| if b { 0.0 } else { far }).collect();
        const A: f32 = 1.0;
        const B: f32 = 1.4142;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut v = d[i];
                if x > 0 {
                    v = v.min(d[i - 1] + A);
                }
                if y > 0 {
                    v = v.min(d[i - w] + A);
                    if x > 0 {
                        v = v.min(d[i - w - 1] + B);
                    }
                    if x + 1 < w {
                        v = v.min(d[i - w + 1] + B);
                    }
                }
                d[i] = v;
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                let mut v = d[i];
                if x + 1 < w {
                    v = v.min(d[i + 1] + A);
                }
                if y + 1 < h {
                    v = v.min(d[i + w] + A);
                    if x + 1 < w {
                        v = v.min(d[i + w + 1] + B);
                    }
                    if x > 0 {
                        v = v.min(d[i + w - 1] + B);
                    }
                }
                d[i] = v;
            }
        }
        d
    }
}

/// Overlap of two outlines: the share of the pixels in either that are in both.
pub fn overlap(a: &Mask, b: &Mask) -> f64 {
    let (mut both, mut either) = (0usize, 0usize);
    for i in 0..a.on.len() {
        both += (a.on[i] && b.on[i]) as usize;
        either += (a.on[i] || b.on[i]) as usize;
    }
    if either == 0 {
        1.0
    } else {
        both as f64 / either as f64
    }
}

/// A video frame's outline: what differs from the plain background (the
/// colour of the picture's border), its largest piece, shrunk `down` times.
pub fn frame_mask(path: &Path, down: usize, cam: &Camera) -> Result<Mask> {
    let dec = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    ));
    let mut rd = dec
        .read_info()
        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; rd.output_buffer_size()];
    let info = rd
        .next_frame(&mut buf)
        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w != cam.width || h != cam.height {
        bail!(
            "{}: {w}x{h}, the camera is {}x{}",
            path.display(),
            cam.width,
            cam.height
        );
    }
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => bail!("{}: colour type {other:?} (want RGB)", path.display()),
    };
    if info.bit_depth != png::BitDepth::Eight {
        bail!("{}: not 8 bits a channel", path.display());
    }
    let px = |x: usize, y: usize| -> [i32; 3] {
        let i = (y * w + x) * ch;
        [buf[i] as i32, buf[i + 1] as i32, buf[i + 2] as i32]
    };
    // the background: the median of the border
    let mut border: [Vec<i32>; 3] = [vec![], vec![], vec![]];
    for x in 0..w {
        for y in [0, 1, h - 2, h - 1] {
            let p = px(x, y);
            for c in 0..3 {
                border[c].push(p[c]);
            }
        }
    }
    for y in 0..h {
        for x in [0, 1, w - 2, w - 1] {
            let p = px(x, y);
            for c in 0..3 {
                border[c].push(p[c]);
            }
        }
    }
    let bg: Vec<i32> = border
        .iter_mut()
        .map(|v| {
            v.sort();
            v[v.len() / 2]
        })
        .collect();
    // shrunk: a cell is body when most of it differs from the background
    let (mw, mh) = (w / down, h / down);
    let mut m = Mask::new(mw, mh);
    for my in 0..mh {
        for mx in 0..mw {
            let mut n = 0;
            for dy in 0..down {
                for dx in 0..down {
                    let p = px(mx * down + dx, my * down + dy);
                    let d = (0..3).map(|c| (p[c] - bg[c]).abs()).max().unwrap();
                    n += (d > 48) as usize;
                }
            }
            m.on[my * mw + mx] = n * 2 > down * down;
        }
    }
    // the largest piece
    let mut label = vec![0u32; mw * mh];
    let mut best = (0usize, 0u32);
    let mut next = 0u32;
    for start in 0..mw * mh {
        if !m.on[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        let mut stack = vec![start];
        label[start] = next;
        let mut size = 0;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % mw, i / mw);
            let mut go = |j: usize| {
                if m.on[j] && label[j] == 0 {
                    label[j] = next;
                    stack.push(j);
                }
            };
            if x > 0 {
                go(i - 1);
            }
            if x + 1 < mw {
                go(i + 1);
            }
            if y > 0 {
                go(i - mw);
            }
            if y + 1 < mh {
                go(i + mw);
            }
        }
        if size > best.0 {
            best = (size, next);
        }
    }
    for i in 0..mw * mh {
        m.on[i] = m.on[i] && label[i] == best.1;
    }
    Ok(m)
}

/// The pose of a rig seen from the side: where its top joints are moved
/// to in the picture's plane, and each joint's turn in that plane
/// relative to its parent (degrees).
#[derive(Clone)]
pub struct Pose {
    pub shift: [f64; 2],
    pub turn: Vec<f64>,
}

pub struct Follower<'a> {
    pub rig: &'a Rig,
    pub cam: Camera,
    pub down: usize,
    pos: Vec<V3>,
    /// Joints that move skin (they hold some, or a joint below them does).
    pub active: Vec<usize>,
}

fn rot_x(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    // columns: where x, y and z go
    [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
}

impl<'a> Follower<'a> {
    pub fn new(rig: &'a Rig, cam: &Camera, down: usize) -> Follower<'a> {
        let n = rig.joints.len();
        let mut moves = vec![false; n];
        for &j in rig.joint_order.iter().rev() {
            if rig.mass[j] > 0.5 {
                moves[j] = true;
            }
            if moves[j] {
                if let Some(p) = rig.parent[j] {
                    moves[p] = true;
                }
            }
        }
        let active = rig
            .joint_order
            .iter()
            .copied()
            .filter(|&j| moves[j])
            .collect();
        Follower {
            rig,
            cam: cam.clone(),
            down,
            pos: rig.joint_rest(),
            active,
        }
    }

    /// A world matrix per joint for a pose.
    pub fn joints(&self, pose: &Pose) -> Vec<M4> {
        let n = self.rig.joints.len();
        let mut total = vec![0.0; n];
        let mut at = self.pos.clone();
        let mut rot: Vec<M3> = vec![rot_x(0.0); n];
        let mut out = vec![geom::IDENTITY; n];
        for &j in &self.rig.joint_order {
            match self.rig.parent[j] {
                None => {
                    total[j] = pose.turn[j];
                    at[j] = geom::add(self.pos[j], [0.0, pose.shift[1], pose.shift[0]]);
                }
                Some(p) => {
                    total[j] = total[p] + pose.turn[j];
                    at[j] = geom::add(
                        at[p],
                        m3_apply(&rot[p], geom::sub(self.pos[j], self.pos[p])),
                    );
                }
            }
            rot[j] = rot_x(total[j]);
            out[j] = geom::from_rt(
                m3_mul(
                    &rot[j],
                    &geom::cols(&self.rig.rest_world[self.rig.joints[j]]),
                ),
                at[j],
            );
        }
        out
    }

    /// The rig's outline for a pose, and its lowest point (world y).
    pub fn outline(&self, pose: &Pose) -> (Mask, f64) {
        let skin = self.rig.skin(&self.joints(pose));
        let k = self.cam.px_per_m / self.down as f64;
        let (w, h) = (self.cam.width / self.down, self.cam.height / self.down);
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let uv: Vec<(f64, f64)> = skin
            .iter()
            .map(|p| {
                (
                    (p[2] - self.cam.centre_z) * k + cx,
                    cy - (p[1] - self.cam.centre_y) * k,
                )
            })
            .collect();
        let mut m = Mask::new(w, h);
        for t in &self.rig.tris {
            let (a, b, c) = (uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]);
            let x0 = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
            let x1 = (a.0.max(b.0).max(c.0).ceil() as i64).min(w as i64 - 1);
            let y0 = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
            let y1 = (a.1.max(b.1).max(c.1).ceil() as i64).min(h as i64 - 1);
            if x1 < 0 || y1 < 0 {
                continue;
            }
            let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
            if area.abs() < 1e-12 {
                continue;
            }
            for y in y0..=(y1 as usize) {
                for x in x0..=(x1 as usize) {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let e0 = (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0);
                    let e1 = (c.0 - b.0) * (py - b.1) - (c.1 - b.1) * (px - b.0);
                    let e2 = (a.0 - c.0) * (py - c.1) - (a.1 - c.1) * (px - c.0);
                    if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
                        || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                    {
                        m.on[y * w + x] = true;
                    }
                }
            }
        }
        let low = skin.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        (m, low)
    }

    /// How far the rig's outline is from the video's (mean distance, in
    /// shrunk pixels, each way), plus what holds the pose steady: a
    /// charge for turning far from the frame before and for going under
    /// the ground.
    fn cost(
        &self,
        pose: &Pose,
        video: &Mask,
        video_dist: &[f32],
        before: &Pose,
        steady: f64,
    ) -> f64 {
        let (m, low) = self.outline(pose);
        let n = m.count();
        if n == 0 {
            return 1e9;
        }
        let rig_dist = m.distance();
        let mut a = 0.0f64;
        for i in 0..m.on.len() {
            if m.on[i] {
                a += video_dist[i] as f64;
            }
        }
        let (mut b, mut nv) = (0.0f64, 0usize);
        for i in 0..video.on.len() {
            if video.on[i] {
                b += rig_dist[i] as f64;
                nv += 1;
            }
        }
        let mut c = a / n as f64 + b / nv.max(1) as f64;
        for &j in &self.active {
            let d = pose.turn[j] - before.turn[j];
            c += steady * d * d;
        }
        let under = (self.cam.ground_y - low).max(0.0) * self.cam.px_per_m / self.down as f64;
        c + 2.0 * under
    }

    /// Lines the camera up on the video's first frame. A video model
    /// does not hold the first picture exactly: it may zoom or move it
    /// a little, and the creature is already in motion. So with the rig
    /// standing where it stands, the camera's zoom and place and the
    /// joints' turns are nudged together until the outlines are nearest.
    /// Returns the pose found; the camera stays as found for every
    /// later frame.
    pub fn line_up(&mut self, video: &Mask) -> Pose {
        let dist = video.distance();
        let n = self.rig.joints.len();
        let mut pose = Pose {
            shift: [0.0, 0.0],
            turn: vec![0.0; n],
        };
        let rest = pose.clone();
        let mut best = self.cost(&pose, video, &dist, &rest, 0.0);
        let px = self.down as f64 / self.cam.px_per_m;
        for (deg, shift, zoom) in [
            (24.0, 16.0, 1.06),
            (16.0, 10.0, 1.04),
            (8.0, 6.0, 1.02),
            (16.0, 10.0, 1.02),
            (8.0, 6.0, 1.01),
            (4.0, 3.0, 1.01),
            (2.0, 1.5, 1.005),
            (1.0, 0.75, 1.0025),
        ] {
            for _ in 0..2 {
                for k in 0..3 {
                    for sign in [1.0, -1.0] {
                        let keep = self.cam.clone();
                        match k {
                            0 => self.cam.centre_z += sign * shift * px,
                            1 => self.cam.centre_y += sign * shift * px,
                            _ => self.cam.px_per_m *= if sign > 0.0 { zoom } else { 1.0 / zoom },
                        }
                        let c = self.cost(&pose, video, &dist, &rest, 0.0);
                        if c < best {
                            best = c;
                        } else {
                            self.cam = keep;
                        }
                    }
                }
                for j in self.active.clone() {
                    for sign in [1.0, -1.0] {
                        let mut p = pose.clone();
                        p.turn[j] += sign * deg;
                        let c = self.cost(&p, video, &dist, &rest, 0.0);
                        if c < best {
                            best = c;
                            pose = p;
                        }
                    }
                }
            }
        }
        pose
    }

    /// One frame: from `start`, each of the pose's numbers in turn is
    /// nudged up and down by a step that halves, and a nudge is kept when
    /// the outlines get nearer.
    pub fn fit_frame(&self, start: &Pose, video: &Mask, before: &Pose, steady: f64) -> Pose {
        let dist = video.distance();
        let mut pose = start.clone();
        let mut best = self.cost(&pose, video, &dist, before, steady);
        let px = self.down as f64 / self.cam.px_per_m;
        for (deg, shift) in [(12.0, 6.0), (6.0, 3.0), (3.0, 1.5), (1.5, 0.75)] {
            for _ in 0..2 {
                for k in 0..2 {
                    for sign in [1.0, -1.0] {
                        let mut p = pose.clone();
                        p.shift[k] += sign * shift * px;
                        let c = self.cost(&p, video, &dist, before, steady);
                        if c < best {
                            best = c;
                            pose = p;
                        }
                    }
                }
                for &j in &self.active {
                    for sign in [1.0, -1.0] {
                        let mut p = pose.clone();
                        p.turn[j] += sign * deg;
                        let c = self.cost(&p, video, &dist, before, steady);
                        if c < best {
                            best = c;
                            pose = p;
                        }
                    }
                }
            }
        }
        pose
    }
}

/// An overlay for the eye: the video's outline red, the rig's green,
/// both yellow.
pub fn write_overlay(path: &Path, video: &Mask, rig: &Mask) -> Result<()> {
    let mut data = Vec::with_capacity(video.w * video.h * 3);
    for i in 0..video.on.len() {
        let px: [u8; 3] = match (video.on[i], rig.on[i]) {
            (true, true) => [235, 215, 70],
            (true, false) => [215, 60, 50],
            (false, true) => [60, 170, 80],
            (false, false) => [28, 30, 34],
        };
        data.extend_from_slice(&px);
    }
    let f = std::fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), video.w as u32, video.h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&data)?;
    Ok(())
}

pub struct Followed {
    /// The camera as lined up on the first frame.
    pub camera: Camera,
    pub json: Value,
    pub times: Vec<f64>,
    pub joints: Vec<Vec<M4>>,
}

/// The whole clip. `frames`: the video's pictures in order, `fps` their
/// rate, `step` every how many of them is used.
pub fn follow(
    rig: &Rig,
    cam: &Camera,
    frames: &[std::path::PathBuf],
    fps: f64,
    step: usize,
    down: usize,
    steady: f64,
    overlays: Option<&Path>,
) -> Result<Followed> {
    let mut fw = Follower::new(rig, cam, down);
    let rest = Pose {
        shift: [0.0, 0.0],
        turn: vec![0.0; rig.joints.len()],
    };
    let used: Vec<usize> = (0..frames.len()).step_by(step.max(1)).collect();
    let mut masks = vec![];
    for &f in &used {
        masks.push(frame_mask(&frames[f], down, cam)?);
    }
    let (rest_outline, _) = fw.outline(&rest);
    let start_overlap = overlap(&rest_outline, &masks[0]);
    // the camera and the first pose from the first frame, then frame by
    // frame, each from the one before
    let first = fw.line_up(&masks[0]);
    let mut poses: Vec<Pose> = vec![first.clone()];
    let mut prev = first;
    for m in masks.iter().skip(1) {
        let p = fw.fit_frame(&prev, m, &prev, steady);
        prev = p.clone();
        poses.push(p);
    }
    let raw: Vec<f64> = poses
        .iter()
        .zip(&masks)
        .map(|(p, m)| overlap(&fw.outline(p).0, m))
        .collect();
    // smooth over time (weights 1 2 1), then keep the lowest point on the ground
    let n = poses.len();
    let mut smooth = poses.clone();
    for i in 0..n {
        let (a, b, c) = (
            &poses[i.saturating_sub(1)],
            &poses[i],
            &poses[(i + 1).min(n - 1)],
        );
        for k in 0..2 {
            smooth[i].shift[k] = (a.shift[k] + 2.0 * b.shift[k] + c.shift[k]) / 4.0;
        }
        for j in 0..b.turn.len() {
            smooth[i].turn[j] = (a.turn[j] + 2.0 * b.turn[j] + c.turn[j]) / 4.0;
        }
    }
    let mut lifted = 0.0f64;
    for p in &mut smooth {
        let (_, low) = fw.outline(p);
        if low < cam.ground_y {
            p.shift[1] += cam.ground_y - low;
            lifted = lifted.max(cam.ground_y - low);
        }
    }
    let mut per_frame = vec![];
    let mut joints = vec![];
    let mut widest = vec![0.0f64; rig.joints.len()];
    for (i, p) in smooth.iter().enumerate() {
        let (m, _) = fw.outline(p);
        per_frame.push(overlap(&m, &masks[i]));
        joints.push(fw.joints(p));
        for &j in &fw.active {
            widest[j] = widest[j].max(p.turn[j].abs());
        }
        if let Some(dir) = overlays {
            std::fs::create_dir_all(dir).ok();
            write_overlay(&dir.join(format!("{i:04}.png")), &masks[i], &m)?;
        }
    }
    let times: Vec<f64> = used.iter().map(|&f| f as f64 / fps).collect();
    let mean = per_frame.iter().sum::<f64>() / n as f64;
    let worst = per_frame.iter().cloned().fold(1.0, f64::min);
    let worst_at = per_frame.iter().position(|&x| x == worst).unwrap_or(0);
    let r3 = |x: f64| (x * 1e3).round() / 1e3;
    let mut turned: Vec<(String, f64)> = fw
        .active
        .iter()
        .map(|&j| (rig.names[j].clone(), widest[j]))
        .collect();
    turned.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let body = masks[0].count().max(1) as f64;
    let json = json!({
        "frames": n,
        "seconds": r3(*times.last().unwrap_or(&0.0)),
        "joints_turned": fw.active.len(),
        "overlap_rest_on_first_frame": r3(start_overlap),
        "overlap_mean": r3(mean),
        "overlap_worst": r3(worst),
        "overlap_worst_at_s": r3(times[worst_at]),
        "overlap_before_smoothing_mean": r3(raw.iter().sum::<f64>() / n as f64),
        "overlap_per_frame": per_frame.iter().map(|x| r3(*x)).collect::<Vec<_>>(),
        "video_outline_size": masks.iter().map(|m| r3(m.count() as f64 / body)).collect::<Vec<_>>(),
        "lifted_to_the_ground_m": r3(lifted),
        "widest_turns_deg": turned.iter().take(8).map(|(n, d)| json!({"joint": n, "deg": (d * 10.0).round() / 10.0})).collect::<Vec<_>>(),
        "travel_m": r3(smooth.last().map(|p| p.shift[0]).unwrap_or(0.0)),
        "camera_lined_up": {"zoom": r3(fw.cam.px_per_m / cam.px_per_m),
            "moved_px": [((cam.centre_z - fw.cam.centre_z) * fw.cam.px_per_m).round(), ((fw.cam.centre_y - cam.centre_y) * fw.cam.px_per_m).round()]},
    });
    Ok(Followed {
        json,
        times,
        joints,
        camera: fw.cam.clone(),
    })
}
