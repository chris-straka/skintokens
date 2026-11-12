//! Plain pictures for checking the analysis by eye: the solid from the
//! side, the front and the top, a colour per part, lines and dots on top.

use skintokens::geom::V3;

use crate::parts::Body;

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

pub type Rgb = [u8; 3];

pub const PALETTE: [Rgb; 14] = [
    [150, 150, 150],
    [230, 80, 70],
    [70, 140, 230],
    [80, 190, 90],
    [240, 170, 40],
    [170, 100, 220],
    [60, 200, 200],
    [235, 110, 180],
    [150, 110, 60],
    [190, 210, 60],
    [100, 100, 230],
    [230, 130, 100],
    [60, 150, 120],
    [200, 60, 140],
];

impl Canvas {
    pub fn new(w: usize, h: usize, bg: Rgb) -> Canvas {
        Canvas { w, h, rgb: (0..w * h).flat_map(|_| bg).collect() }
    }
    pub fn set(&mut self, x: i64, y: i64, c: Rgb) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let o = (y as usize * self.w + x as usize) * 3;
            self.rgb[o..o + 3].copy_from_slice(&c);
        }
    }
    pub fn rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: Rgb) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.set(xx, yy, c);
            }
        }
    }
    pub fn disc(&mut self, x: f64, y: f64, r: f64, c: Rgb) {
        let k = r.ceil() as i64;
        for dy in -k..=k {
            for dx in -k..=k {
                if (dx * dx + dy * dy) as f64 <= r * r {
                    self.set(x.round() as i64 + dx, y.round() as i64 + dy, c);
                }
            }
        }
    }
    pub fn line(&mut self, a: (f64, f64), b: (f64, f64), width: f64, c: Rgb) {
        let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil() as usize).max(1);
        for k in 0..=n {
            let t = k as f64 / n as f64;
            self.disc(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, width * 0.5, c);
        }
    }
    /// Digits and a few letters, 3x5 pixels scaled by `s`.
    pub fn text(&mut self, x: i64, y: i64, s: i64, text: &str, c: Rgb) {
        let mut cx = x;
        for ch in text.chars() {
            if let Some(g) = glyph(ch) {
                for (row, bits) in g.iter().enumerate() {
                    for col in 0..3 {
                        if bits >> (2 - col) & 1 == 1 {
                            self.rect(cx + col * s, y + row as i64 * s, s, s, c);
                        }
                    }
                }
            }
            cx += 4 * s;
        }
    }
    pub fn write_png(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut e = png::Encoder::new(f, self.w as u32, self.h as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header()?.write_image_data(&self.rgb)?;
        Ok(())
    }
}

fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c.to_ascii_uppercase() {
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [7, 1, 7, 4, 7],
        '3' => [7, 1, 7, 1, 7],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 7, 1, 7],
        '6' => [7, 4, 7, 5, 7],
        '7' => [7, 1, 2, 2, 2],
        '8' => [7, 5, 7, 5, 7],
        '9' => [7, 5, 7, 1, 7],
        'A' => [2, 5, 7, 5, 5],
        'B' => [6, 5, 6, 5, 6],
        'C' => [7, 4, 4, 4, 7],
        'D' => [6, 5, 5, 5, 6],
        'E' => [7, 4, 6, 4, 7],
        'F' => [7, 4, 6, 4, 4],
        'G' => [7, 4, 5, 5, 7],
        'H' => [5, 5, 7, 5, 5],
        'I' => [7, 2, 2, 2, 7],
        'J' => [1, 1, 1, 5, 7],
        'K' => [5, 5, 6, 5, 5],
        'L' => [4, 4, 4, 4, 7],
        'M' => [5, 7, 7, 5, 5],
        'N' => [6, 5, 5, 5, 5],
        'O' => [7, 5, 5, 5, 7],
        'P' => [7, 5, 7, 4, 4],
        'Q' => [7, 5, 5, 7, 1],
        'R' => [6, 5, 6, 5, 5],
        'S' => [7, 4, 7, 1, 7],
        'T' => [7, 2, 2, 2, 2],
        'U' => [5, 5, 5, 5, 7],
        'V' => [5, 5, 5, 5, 2],
        'W' => [5, 5, 7, 7, 5],
        'X' => [5, 5, 2, 5, 5],
        'Y' => [5, 5, 2, 2, 2],
        'Z' => [7, 1, 2, 4, 7],
        '.' => [0, 0, 0, 0, 2],
        '-' => [0, 0, 7, 0, 0],
        '_' => [0, 0, 0, 0, 7],
        ':' => [0, 2, 0, 2, 0],
        '/' => [1, 1, 2, 4, 4],
        ' ' => [0, 0, 0, 0, 0],
        _ => return None,
    })
}

/// Three views of a body side by side: from its right side (forward to
/// the right), from the front, from above.
pub struct Views {
    pub canvas: Canvas,
    scale: f64,
    lo: V3,
    hi: V3,
    /// Left edge of each panel.
    x0: [f64; 3],
    margin: f64,
}

impl Views {
    /// (horizontal axis, vertical axis, depth axis, depth sign: +1 draws
    /// larger depth on top) per panel.
    const AXES: [(usize, usize, usize, f64); 3] = [(2, 1, 0, 1.0), (0, 1, 2, 1.0), (0, 2, 1, 1.0)];

    pub fn new(lo: V3, hi: V3, height_px: usize) -> Views {
        let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let margin = 24.0;
        let tall = ext[1].max(ext[2]);
        let scale = (height_px as f64 - 2.0 * margin) / tall;
        let widths = [ext[2], ext[0], ext[0]].map(|e| e * scale + 2.0 * margin);
        let x0 = [0.0, widths[0], widths[0] + widths[1]];
        let w = (widths[0] + widths[1] + widths[2]).ceil() as usize;
        Views { canvas: Canvas::new(w, height_px, [28, 28, 32]), scale, lo, hi, x0, margin }
    }

    pub fn project(&self, panel: usize, p: V3) -> (f64, f64) {
        let (h, v, _, _) = Self::AXES[panel];
        let x = self.x0[panel] + self.margin + (p[h] - self.lo[h]) * self.scale;
        // top view: forward (z) up the page; others: y up
        let y = self.margin + (self.hi[v] - p[v]) * self.scale;
        (x, y)
    }

    /// The solid, a colour per cell from `colour` (None: not drawn).
    pub fn solid(&mut self, body: &Body, colour: impl Fn(usize) -> Option<Rgb>) {
        let s = &body.solid;
        let px = (s.cell * self.scale).ceil().max(1.0) as i64;
        for panel in 0..3 {
            let (_, _, d, _) = Self::AXES[panel];
            // paint far cells first so the near surface shows
            let mut cells: Vec<usize> = (0..s.len()).filter(|&i| s.inside[i]).collect();
            cells.sort_by_key(|&i| s.coords(i)[d]);
            let depth_span = s.dims[d].max(1) as f64;
            for i in cells {
                let Some(c) = colour(i) else { continue };
                let shade = 0.55 + 0.45 * (s.coords(i)[d] as f64 / depth_span);
                let c = c.map(|x| (x as f64 * shade) as u8);
                let p = s.center(i);
                let (x, y) = self.project(panel, p);
                self.canvas.rect((x - px as f64 / 2.0).round() as i64, (y - px as f64 / 2.0).round() as i64, px, px, c);
            }
        }
    }

    pub fn polyline(&mut self, pts: &[V3], width: f64, c: Rgb) {
        for panel in 0..3 {
            for k in 1..pts.len() {
                let (a, b) = (self.project(panel, pts[k - 1]), self.project(panel, pts[k]));
                self.canvas.line(a, b, width, c);
            }
        }
    }

    pub fn dot(&mut self, p: V3, r: f64, c: Rgb) {
        for panel in 0..3 {
            let (x, y) = self.project(panel, p);
            self.canvas.disc(x, y, r, c);
        }
    }

    pub fn label(&mut self, p: V3, text: &str, c: Rgb) {
        for panel in 0..3 {
            let (x, y) = self.project(panel, p);
            self.canvas.text(x as i64 + 4, y as i64 - 8, 2, text, c);
        }
    }
}
