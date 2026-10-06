//! skintokens: SkinTokens/TokenRig auto-rigging as a file-in, file-out CLI.

use skintokens::{generate, glb, model, parity, pipeline};

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const MODEL: &str = "VAST-AI/SkinTokens@79736cad grpo_1400 (TokenRig + skin_vae_2_10_32768)";
const UPSTREAM: &str = "273b691";
const HF_REV: &str = "79736cad0fd84de384d5eede659b4ebd24effe33";
const CKPT_REL: &str = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt";

const HELP: &str = "skintokens: SkinTokens/TokenRig auto-rigging as a file-in, file-out CLI.

    skintokens rig    IN.glb OUT.glb [--report R.json] [--class humanoid|quadruped|custom] [--seed N]
    skintokens skin   IN.glb OUT.glb [--report R.json] [--seed N]
    skintokens joints IN.glb OUT.json [--subject NAME]
    skintokens doctor

rig    generates a skeleton and skin weights for an unrigged mesh. Humanoid
       joints are named by structure (Mixamo names) so motionforge's
       standardize adapter maps them onto the HLL skeleton.
skin   keeps IN's skeleton and replaces only its weights (JOINTS_0/WEIGHTS_0
       bytes): a weightforge `fix --candidate`. Twist/helper bones are hidden
       from the model and left unweighted.
joints writes skintokens-joints/1 (same schema as unirig-joints/1) from a
       rigged GLB: rigforge's joint hints.

The report follows genforge's adapter contract: {\"ok\", \"outputs\" (paths
relative to the report's folder), \"tool\", \"version\", \"stage\", \"<stage>\": {..},
\"reason\"?}. Exit 0 ok, 1 ran and failed (report written), 2 usage or
environment error (no report). Inputs are never modified. One inference runs
at a time per machine (a file lock in $SKINTOKENS_HOME).

Environment: SKINTOKENS_HOME (default ~/.local/share/skintokens),
SKINTOKENS_CKPT (path to grpo_1400.ckpt), SKINTOKENS_DEVICE=cpu|metal,
SKINTOKENS_DTYPE=f32|bf16 (default f32).";

/// Exit 2: nothing ran, no report.
struct Usage(String);

impl<E: std::fmt::Display> From<E> for Usage {
    fn from(e: E) -> Self {
        Usage(format!("{e:#}"))
    }
}

struct Args {
    pos: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(v: &[String], with_value: &[&str], switches: &[&str]) -> Result<Args, Usage> {
        let mut pos = vec![];
        let mut flags = vec![];
        let mut i = 0;
        while i < v.len() {
            let a = &v[i];
            if let Some(f) = a.strip_prefix("--") {
                let (k, inline) = match f.split_once('=') {
                    Some((k, x)) => (k.to_string(), Some(x.to_string())),
                    None => (f.to_string(), None),
                };
                if switches.contains(&k.as_str()) {
                    flags.push((k, String::new()));
                } else if with_value.contains(&k.as_str()) {
                    let val = match inline {
                        Some(x) => x,
                        None => {
                            i += 1;
                            v.get(i).cloned().ok_or_else(|| Usage(format!("--{k} needs a value")))?
                        }
                    };
                    flags.push((k, val));
                } else {
                    return Err(Usage(format!("unknown option --{k}")));
                }
            } else {
                pos.push(a.clone());
            }
            i += 1;
        }
        Ok(Args { pos, flags })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(x, _)| x == k).map(|(_, v)| v.as_str())
    }
}

fn home() -> PathBuf {
    std::env::var_os("SKINTOKENS_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share/skintokens"))
}

/// The official checkpoint: $SKINTOKENS_CKPT, $SKINTOKENS_HOME/weights, or
/// the Hugging Face cache at the pinned revision.
fn find_ckpt() -> Vec<PathBuf> {
    let mut c = vec![];
    if let Some(p) = std::env::var_os("SKINTOKENS_CKPT") {
        c.push(PathBuf::from(p));
    }
    c.push(home().join("weights/grpo_1400.ckpt"));
    let hf = std::env::var_os("HF_HUB_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HF_HOME")
            .map(|h| PathBuf::from(h).join("hub"))
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub"))
    });
    c.push(hf.join(format!("models--VAST-AI--SkinTokens/snapshots/{HF_REV}/{CKPT_REL}")));
    c
}

fn ckpt() -> Result<PathBuf, Usage> {
    find_ckpt()
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| Usage("SkinTokens weights not found (grpo_1400.ckpt); run fetch-weights.sh or set SKINTOKENS_CKPT".into()))
}

fn dtype() -> candle_core::DType {
    match std::env::var("SKINTOKENS_DTYPE").as_deref() {
        Ok("bf16") => candle_core::DType::BF16,
        _ => candle_core::DType::F32,
    }
}

fn load_model() -> Result<model::TokenRig, Usage> {
    let dev = skintokens_nn::pick_device()?;
    Ok(model::TokenRig::load(&ckpt()?, &dev, dtype())?)
}

/// Machine-wide lock: one inference at a time.
fn lock() -> Result<std::fs::File, Usage> {
    let h = home();
    std::fs::create_dir_all(&h).map_err(|e| Usage(format!("{}: {e}", h.display())))?;
    let f = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(h.join(".lock"))?;
    f.lock()?;
    Ok(f)
}

fn write_report(path: &Path, outputs: &[&Path], stage: &str, ok: bool, details: Value, reason: Option<String>) -> Result<()> {
    let dir = std::path::absolute(path)?.parent().map(Path::to_path_buf).unwrap_or_default();
    std::fs::create_dir_all(&dir)?;
    let rel = |f: &Path| -> String {
        let a = std::path::absolute(f).unwrap_or(f.to_path_buf());
        pathdiff(&a, &dir)
    };
    let mut rep = json!({
        "ok": ok,
        "outputs": if ok { outputs.iter().map(|f| rel(f)).collect::<Vec<_>>() } else { vec![] },
        "tool": "skintokens",
        "version": VERSION,
        "stage": stage,
    });
    rep[stage] = details;
    if let Some(r) = reason {
        rep["reason"] = json!(r);
    }
    std::fs::write(path, serde_json::to_string_pretty(&rep)? + "\n")?;
    Ok(())
}

/// Relative path from `base` (a directory) to `p`, both absolute.
fn pathdiff(p: &Path, base: &Path) -> String {
    let pc: Vec<_> = p.components().collect();
    let bc: Vec<_> = base.components().collect();
    let common = pc.iter().zip(&bc).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..bc.len() {
        out.push("..");
    }
    for c in &pc[common..] {
        out.push(c.as_os_str());
    }
    out.to_string_lossy().into_owned()
}

fn report_path(a: &Args, out: &Path) -> PathBuf {
    a.get("report").map(PathBuf::from).unwrap_or_else(|| {
        std::path::absolute(out).ok().and_then(|p| p.parent().map(|d| d.join("result.json"))).unwrap_or_else(|| PathBuf::from("result.json"))
    })
}

fn read_input(p: &str) -> Result<glb::Glb, Usage> {
    let path = Path::new(p);
    if !path.is_file() {
        return Err(Usage(format!("input not found: {p}")));
    }
    glb::Glb::read(path).map_err(|e| Usage(format!("{e:#}")))
}

fn seed(a: &Args) -> Result<u64, Usage> {
    a.get("seed").unwrap_or("0").parse::<u64>().map_err(|_| Usage("--seed must be a non-negative integer".into()))
}

fn base_details(seed: u64) -> Value {
    json!({"seed": seed, "model": MODEL, "upstream": UPSTREAM, "runtime": format!("rust {VERSION}")})
}

fn merge(a: &mut Value, b: Value) {
    if let (Some(a), Value::Object(b)) = (a.as_object_mut(), b) {
        for (k, v) in b {
            a.insert(k, v);
        }
    }
}

fn cmd_rig(v: &[String]) -> Result<u8, Usage> {
    let a = Args::parse(v, &["report", "class", "seed"], &[])?;
    let [inp, out] = &a.pos[..] else { return Err(Usage("rig needs IN.glb OUT.glb".into())) };
    let cls = a.get("class").unwrap_or("humanoid");
    if !["humanoid", "quadruped", "custom"].contains(&cls) {
        return Err(Usage(format!("--class must be humanoid, quadruped or custom (got {cls})")));
    }
    let seed = seed(&a)?;
    let g = read_input(inp)?;
    let out = PathBuf::from(out);
    let report = report_path(&a, &out);
    let t0 = Instant::now();
    let _lock = lock()?;
    let model = load_model()?;
    let mut details = base_details(seed);
    details["class"] = json!(cls);
    match pipeline::rig(&g, &model, &generate::GenConfig::default(), seed, cls == "humanoid") {
        Err(e) => {
            write_report(&report, &[], "rig", false, details, Some(format!("model run failed: {e:#}")))?;
            Ok(1)
        }
        Ok(r) => {
            let joints = r.details["joints_raw"].as_u64().unwrap_or(0);
            merge(&mut details, r.details);
            details["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
            if !r.missing.is_empty() {
                let why = format!("skeleton is not a humanoid TokenRig could name (missing {})", r.missing.join(", "));
                write_report(&report, &[], "rig", false, details, Some(why))?;
                return Ok(1);
            }
            r.glb.write(&out)?;
            write_report(&report, &[&out], "rig", true, details.clone(), None)?;
            println!("skintokens rig: ok ({joints} joints, {} s)", details["seconds"]);
            Ok(0)
        }
    }
}

fn cmd_skin(v: &[String]) -> Result<u8, Usage> {
    let a = Args::parse(v, &["report", "seed"], &[])?;
    let [inp, out] = &a.pos[..] else { return Err(Usage("skin needs IN.glb OUT.glb".into())) };
    let seed = seed(&a)?;
    let g = read_input(inp)?;
    if let Err(e) = g.skin_info().and_then(|_| g.parts(glb::Frame::SkinBind)) {
        return Err(Usage(format!("{inp}: skin needs a rigged GLB ({e:#})")));
    }
    let out = PathBuf::from(out);
    let report = report_path(&a, &out);
    let t0 = Instant::now();
    let _lock = lock()?;
    let model = load_model()?;
    let mut details = base_details(seed);
    match pipeline::skin(&g, &model, &generate::GenConfig::default(), seed) {
        Err(e) => {
            write_report(&report, &[], "skin", false, details, Some(format!("{e:#}")))?;
            Ok(1)
        }
        Ok(r) => {
            merge(&mut details, r.details);
            details["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
            r.glb.write(&out)?;
            write_report(&report, &[&out], "skin", true, details.clone(), None)?;
            println!("skintokens skin: ok ({} s)", details["seconds"]);
            Ok(0)
        }
    }
}

fn cmd_joints(v: &[String]) -> Result<u8, Usage> {
    let a = Args::parse(v, &["subject"], &[])?;
    let [inp, out] = &a.pos[..] else { return Err(Usage("joints needs IN.glb OUT.json".into())) };
    let g = read_input(inp)?;
    let src = std::path::absolute(inp)?.to_string_lossy().into_owned();
    let doc = pipeline::joints_doc(&g, a.get("subject"), &src).map_err(|e| Usage(format!("{inp}: {e:#}")))?;
    std::fs::write(out, serde_json::to_string_pretty(&doc)? + "\n")?;
    println!("skintokens joints: {} joints -> {out}", doc["joints"].as_array().map(|j| j.len()).unwrap_or(0));
    Ok(0)
}

fn cmd_doctor() -> Result<u8, Usage> {
    let mut ok = true;
    match find_ckpt().into_iter().find(|p| p.is_file()) {
        Some(p) => println!("ok  weights {}", p.display()),
        None => {
            ok = false;
            println!("MISSING weights grpo_1400.ckpt (looked in: {})", find_ckpt().iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "));
        }
    }
    match skintokens_nn::pick_device() {
        Ok(d) => println!("ok  device {}", if d.is_metal() { "metal (GPU)" } else { "cpu" }),
        Err(e) => {
            ok = false;
            println!("MISSING device ({e})");
        }
    }
    println!("ok  skintokens {VERSION} (rust), model {MODEL}, upstream {UPSTREAM}");
    if ok {
        let t = Instant::now();
        match load_model() {
            Ok(_) => println!("ok  checkpoint loads ({:.1} s)", t.elapsed().as_secs_f64()),
            Err(Usage(e)) => {
                ok = false;
                println!("FAIL checkpoint: {e}");
            }
        }
    }
    Ok(if ok { 0 } else { 2 })
}

fn run(args: &[String]) -> Result<u8, Usage> {
    let Some(cmd) = args.first() else {
        eprintln!("{HELP}");
        return Ok(2);
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "rig" => cmd_rig(rest),
        "skin" => cmd_skin(rest),
        "joints" => cmd_joints(rest),
        "doctor" => cmd_doctor(),
        "-V" | "--version" => {
            println!("skintokens {VERSION}");
            Ok(0)
        }
        "-h" | "--help" | "help" => {
            println!("{HELP}");
            Ok(0)
        }
        // developer checks against the Python reference (parity/)
        "parity-net" => {
            let m = load_model()?;
            parity::run(&m, Path::new(rest.first().ok_or_else(|| Usage("parity-net REFDIR [gen]".into()))?), rest.get(1).map(|s| s == "gen").unwrap_or(false))?;
            Ok(0)
        }
        "parity-input" => {
            let [inp, dir, mode] = rest else { return Err(Usage("parity-input IN.glb DIR rig|skin".into())) };
            parity::dump_input(&read_input(inp)?, Path::new(dir), mode == "skin")?;
            Ok(0)
        }
        other => Err(Usage(format!("unknown command {other:?} (see --help)"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    match run(&args) {
        Ok(c) => ExitCode::from(c),
        Err(Usage(e)) => {
            eprintln!("skintokens {cmd}: {e}");
            ExitCode::from(2)
        }
    }
}
