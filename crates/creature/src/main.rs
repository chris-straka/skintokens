//! creature-rig: creature rigs from the body's own shape.

mod check;
mod draw;
mod parts;
mod poses;
mod roles;
mod skeleton;
mod solid;
#[cfg(test)]
mod tests;
mod weights;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Result};
use serde_json::json;
use skintokens::geom::V3;
use skintokens::glb::{Frame, Glb};

const HELP: &str = "creature-rig: creature rigs from the body's own shape.

    creature-rig parts MESH.glb [--json OUT.json] [--sheet OUT.png] [--res 128]
    creature-rig check RIG.glb [--json OUT.json] [--rom OUT.ron] [--sheet OUT.png] [--res 128]
    creature-rig rig   MESH.glb OUT.glb [--rom OUT.ron] [--json OUT.json] [--sheet OUT.png]
                       [--res 128] [--blend 0.6]

parts  finds the body's parts in its volume (the branches of the solid:
       legs, neck, tail, wings) and writes them with their centre lines.
       The sheet shows the solid from the side, the front and the top, a
       colour per part.
check  the skeleton check on a rigged GLB: every leg, arm, tail, wing and
       the neck and head of its body has a chain of bones, the bones lie
       inside the body, a left and its right match. Prints the report
       (JSON); exit 1 when something fails. --rom writes the body's range
       of motion on this rig's bones, as `rig` does for its own.
rig    lays a skeleton along those parts (hips and spine, a chain down
       every limb, the tail, the neck and head, the wings), weights the
       mesh to it by distance through the body, and writes the rigged GLB
       (MESH's own rig, if it has one, is replaced). --rom writes this
       body's range of motion as a weightforge pose set.";

/// Every mesh of the file as one triangle soup in world space (glTF: +Y up).
fn load_mesh(path: &Path) -> Result<(Vec<V3>, Vec<[u32; 3]>)> {
    let g = Glb::read(path)?;
    let mut verts = vec![];
    let mut tris = vec![];
    for p in g.parts(Frame::World)? {
        let base = verts.len() as u32;
        verts.extend_from_slice(&p.positions);
        tris.extend(p.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    }
    if tris.is_empty() {
        bail!("{} has no triangles", path.display());
    }
    Ok((verts, tris))
}

static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct Args {
    pos: Vec<String>,
    flags: std::collections::HashMap<String, String>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut pos = vec![];
    let mut flags = std::collections::HashMap::new();
    let mut k = 0;
    while k < args.len() {
        if let Some(name) = args[k].strip_prefix("--") {
            let v = args.get(k + 1).ok_or_else(|| anyhow!("--{name} needs a value"))?;
            flags.insert(name.to_string(), v.clone());
            k += 2;
        } else {
            pos.push(args[k].clone());
            k += 1;
        }
    }
    Ok(Args { pos, flags })
}

fn r3(p: V3) -> [f64; 3] {
    p.map(|x| (x * 1e4).round() / 1e4)
}

fn cmd_parts(a: &Args) -> Result<()> {
    let input = PathBuf::from(a.pos.first().ok_or_else(|| anyhow!("parts needs MESH.glb"))?);
    let res: usize = a.flags.get("res").map(|s| s.parse()).transpose()?.unwrap_or(128);
    let (verts, tris) = load_mesh(&input)?;
    let body = parts::Body::build(&verts, &tris, res);
    let doc = json!({
        "format": "creature-parts/1",
        "source": input.display().to_string(),
        "cell": body.solid.cell,
        "size": body.size,
        "seed": r3(body.solid.center(body.seed)),
        "parts": body.parts.iter().enumerate().map(|(i, p)| json!({
            "id": i,
            "parent": p.parent,
            "children": p.children,
            "length": (p.length * 1e4).round() / 1e4,
            "cells": p.cells,
            "base": r3(p.base()),
            "tip": r3(p.tip()),
            "lowest": (p.lo_y * 1e4).round() / 1e4,
            "spread": p.spread.map(|x| (x * 1e4).round() / 1e4),
            "line": p.line.iter().map(|&q| r3(q)).collect::<Vec<_>>(),
            "radius": p.radius.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&doc)?)?;
    }
    let anatomy = roles::Anatomy::read(&body);
    if let Some(p) = a.flags.get("sheet") {
        let mut v = draw::Views::new(body.lo, body.hi, 640);
        // trunk grey, each limb its role's colour (left and right shades apart)
        v.solid(&body, |i| {
            let p = body.part_of[i];
            if p == u32::MAX {
                return Some([60, 60, 60]);
            }
            Some(match anatomy.limb_of_part[p as usize] {
                None => [150, 150, 150],
                Some(l) => role_colour(&anatomy.limbs[l]),
            })
        });
        v.polyline(&anatomy.spine, 3.0, [255, 255, 255]);
        for l in &anatomy.limbs {
            v.polyline(&l.line, 2.0, [20, 20, 20]);
            v.dot(l.at(l.start).0, 3.5, [255, 255, 255]);
            v.label(l.tip(), &l.name, [255, 255, 0]);
        }
        v.canvas.text(8, 8, 3, &format!("{}{}{}", anatomy.kind.name(), if anatomy.upright { " upright" } else { "" }, if anatomy.forward < 0.0 { " faces -Z" } else { "" }), [255, 255, 255]);
        v.canvas.write_png(Path::new(p))?;
    }
    println!("{}{}{}: {}", anatomy.kind.name(), if anatomy.upright { ", upright" } else { "" }, if anatomy.forward < 0.0 { ", faces -Z" } else { "" }, anatomy.limbs.iter().map(|l| format!("{} ({:.2})", l.name, l.length)).collect::<Vec<_>>().join(", "));
    println!("{} parts, {} cells inside, cell {:.4}", body.parts.len(), body.solid.inside.iter().filter(|&&b| b).count(), body.solid.cell);
    for (i, p) in body.parts.iter().enumerate() {
        println!(
            "  part {i}: parent {:?}, length {:.3}, cells {}, base {:?}, tip {:?}, spread {:?}",
            p.parent,
            p.length,
            p.cells,
            r3(p.base()),
            r3(p.tip()),
            p.spread.map(|x| (x * 1e3).round() / 1e3)
        );
    }
    Ok(())
}

fn cmd_check(a: &Args) -> Result<bool> {
    let input = PathBuf::from(a.pos.first().ok_or_else(|| anyhow!("check needs RIG.glb"))?);
    let res: usize = a.flags.get("res").map(|s| s.parse()).transpose()?.unwrap_or(128);
    let (verts, tris) = load_mesh(&input)?;
    let body = parts::Body::build(&verts, &tris, res);
    let an = roles::Anatomy::read(&body);
    let bones = check::read_rig(&Glb::read(&input)?)?;
    let v = check::check(&body, &an, bones);
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&v.report)?)?;
    }
    if let Some(p) = a.flags.get("rom") {
        std::fs::write(p, poses::rom(&an, &v.skeleton))?;
    }
    if let Some(p) = a.flags.get("sheet") {
        let mut vw = draw::Views::new(body.lo, body.hi, 640);
        vw.solid(&body, |i| {
            let p = body.part_of[i];
            if p == u32::MAX {
                return Some([60, 60, 60]);
            }
            Some(match an.limb_of_part[p as usize] {
                None => [120, 120, 120],
                Some(l) => role_colour(&an.limbs[l]).map(|x| (x as f64 * 0.6) as u8),
            })
        });
        for b in &v.skeleton.bones {
            // white: along a part; red: along nothing
            let c = if b.trunk || b.limb.is_some() { [255, 255, 255] } else { [255, 60, 60] };
            vw.polyline(&[b.head, b.tail], 3.0, c);
        }
        for b in &v.skeleton.bones {
            vw.dot(b.head, 4.0, [255, 230, 0]);
        }
        vw.canvas.text(8, 8, 3, &format!("skeleton {:.0}  {} bones  {}", v.score, v.skeleton.bones.len(), an.kind.name()), [255, 255, 255]);
        vw.canvas.write_png(Path::new(p))?;
    }
    println!("{}", serde_json::to_string_pretty(&v.report)?);
    Ok(v.failing == 0)
}

fn cmd_rig(a: &Args) -> Result<()> {
    let input = PathBuf::from(a.pos.first().ok_or_else(|| anyhow!("rig needs MESH.glb OUT.glb"))?);
    let output = PathBuf::from(a.pos.get(1).ok_or_else(|| anyhow!("rig needs MESH.glb OUT.glb"))?);
    let res: usize = a.flags.get("res").map(|s| s.parse()).transpose()?.unwrap_or(128);
    let blend: f64 = a.flags.get("blend").map(|s| s.parse()).transpose()?.unwrap_or(weights::BLEND);
    let (verts, tris) = load_mesh(&input)?;
    let body = parts::Body::build(&verts, &tris, res);
    let an = roles::Anatomy::read(&body);
    let sk = skeleton::Skeleton::fit(&body, &an);
    let skin = weights::skin(&body, &sk, &verts, &tris, blend);
    // bones follow Blender's convention: local +Y along the bone, roll 0
    let worlds: Vec<skintokens::geom::M4> = sk
        .bones
        .iter()
        .map(|b| {
            let d = skintokens::geom::sub(b.tail, b.head);
            let l = skintokens::geom::norm(d);
            let d = if l > 0.0 { skintokens::geom::scale(d, 1.0 / l) } else { [0.0, 1.0, 0.0] };
            let cols = skintokens::geom::bone_rotation(skintokens::mesh::to_model(d));
            skintokens::geom::from_rt(cols.map(skintokens::mesh::from_model), b.head)
        })
        .collect();
    let src = Glb::read(&input)?;
    let mut g = Glb { json: src.json.clone(), bin: src.bin.clone() };
    skintokens::pipeline::strip_rig(&mut g);
    let gparts = g.parts(Frame::World)?;
    let names: Vec<String> = sk.bones.iter().map(|b| b.name.clone()).collect();
    let parents: Vec<i64> = sk.bones.iter().map(|b| b.parent.map(|p| p as i64).unwrap_or(-1)).collect();
    skintokens::pipeline::attach_skeleton(&mut g, &gparts, &names, &parents, &worlds, &skin.joints, &skin.weights)?;
    g.doc()?;
    g.write(&output)?;
    if let Some(p) = a.flags.get("rom") {
        std::fs::write(p, poses::rom(&an, &sk))?;
    }
    let doc = json!({
        "format": "creature-rig/1",
        "kind": an.kind.name(),
        "upright": an.upright,
        "faces": if an.forward > 0.0 { "+z" } else { "-z" },
        "limbs": an.limbs.iter().enumerate().map(|(i, l)| json!({
            "name": l.name,
            "role": format!("{:?}", l.role).to_lowercase(),
            "length": (l.length * 1e4).round() / 1e4,
            "bones": sk.bones.iter().filter(|b| b.limb == Some(i)).map(|b| b.name.clone()).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "bones": sk.bones.iter().map(|b| json!({"name": b.name, "parent": b.parent.map(|p| sk.bones[p].name.clone()), "head": r3(b.head), "tail": r3(b.tail)})).collect::<Vec<_>>(),
    });
    if let Some(p) = a.flags.get("json") {
        std::fs::write(p, serde_json::to_string_pretty(&doc)?)?;
    }
    if let Some(p) = a.flags.get("sheet") {
        let mut v = draw::Views::new(body.lo, body.hi, 640);
        v.solid(&body, |i| {
            let p = body.part_of[i];
            if p == u32::MAX {
                return Some([60, 60, 60]);
            }
            Some(match an.limb_of_part[p as usize] {
                None => [120, 120, 120],
                Some(l) => role_colour(&an.limbs[l]).map(|x| (x as f64 * 0.6) as u8),
            })
        });
        for b in &sk.bones {
            v.polyline(&[b.head, b.tail], 3.0, [255, 255, 255]);
        }
        for b in &sk.bones {
            v.dot(b.head, 4.0, [255, 230, 0]);
        }
        v.canvas.text(8, 8, 3, &format!("{}{}{} {} bones", an.kind.name(), if an.upright { " upright" } else { "" }, if an.forward < 0.0 { " faces -Z" } else { "" }, sk.bones.len()), [255, 255, 255]);
        v.canvas.write_png(Path::new(p))?;
    }
    println!(
        "{}{}{}: {} bones ({})",
        an.kind.name(),
        if an.upright { ", upright" } else { "" },
        if an.forward < 0.0 { ", faces -Z" } else { "" },
        sk.bones.len(),
        an.limbs.iter().enumerate().map(|(i, l)| format!("{} {}", l.name, sk.bones.iter().filter(|b| b.limb == Some(i)).count())).collect::<Vec<_>>().join(", ")
    );
    Ok(())
}

fn role_colour(l: &roles::Limb) -> draw::Rgb {
    let c: draw::Rgb = match l.role {
        roles::Role::Leg => [70, 140, 230],
        roles::Role::Arm => [80, 190, 90],
        roles::Role::Head => [240, 170, 40],
        roles::Role::Tail => [170, 100, 220],
        roles::Role::Wing => [60, 200, 200],
        roles::Role::Extra => [235, 110, 180],
    };
    if l.side == roles::Side::Right {
        c.map(|x| (x as f64 * 0.7) as u8)
    } else {
        c
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        println!("{HELP}");
        return ExitCode::from(2);
    };
    let run = || -> Result<()> {
        let a = parse(&args[1..])?;
        match cmd.as_str() {
            "parts" => cmd_parts(&a),
            "rig" => cmd_rig(&a),
            "check" => {
                if !cmd_check(&a)? {
                    FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(())
            }
            "help" | "--help" | "-h" => {
                println!("{HELP}");
                Ok(())
            }
            other => bail!("unknown command {other}\n\n{HELP}"),
        }
    };
    match run() {
        Ok(()) if FAILED.load(std::sync::atomic::Ordering::Relaxed) => ExitCode::from(1),
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("creature-rig: {e:#}");
            ExitCode::from(2)
        }
    }
}
