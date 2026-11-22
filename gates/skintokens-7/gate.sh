#!/usr/bin/env bash
# Gate for skintokens-7 (`_tools/docs/plans/skintokens.md`, 2026-11-18 decision):
#   Five hero fingers come from the base body via the humanoid route;
#   motion-1's grip_gate.sh (>= 0.90) and meshweights `weights check`
#   hand/finger findings no worse judge the result.
#
# Needs: --needs blender --needs cargo --data hands-corpus --data skintokens-weights
# Prints PASS/FAIL per check and ends with GATE PASS or GATE FAIL.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 2
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }

C="${HANDS_CORPUS:-${FLEET_DATA_HANDS_CORPUS:-}}"
SK_WEIGHTS="${SKINTOKENS_WEIGHTS:-${FLEET_DATA_SKINTOKENS_WEIGHTS:-}}"
B="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
[ -x "$B" ] || B="$(command -v blender || true)"
SIB="$(cd "$ROOT/.." && pwd)"
BODYGRAFT="${BODYGRAFT:-$SIB/bodygraft}"
MOTION="${MOTION:-$SIB/motion}"
MESHWEIGHTS="${MESHWEIGHTS:-$SIB/meshweights}"

[ -n "$C" ] && [ -f "$C/base/base_body.glb" ] || { echo "FAIL  missing hands-corpus"; echo "GATE FAIL"; exit 2; }
[ -n "$SK_WEIGHTS" ] && [ -d "$SK_WEIGHTS" ] || { echo "FAIL  missing skintokens-weights"; echo "GATE FAIL"; exit 2; }
[ -n "$B" ] && [ -x "$B" ] || { echo "FAIL  missing Blender"; echo "GATE FAIL"; exit 2; }
for d in "$BODYGRAFT" "$MOTION" "$MESHWEIGHTS"; do
  [ -d "$d" ] || { echo "FAIL  missing sibling $d"; echo "GATE FAIL"; exit 2; }
done

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

echo "== 1. Python & Rust tests"
python3 -m unittest discover -s tools -p "test_*.py" > "$T/py.out" 2>&1
pc=$?
if [ $pc -eq 0 ]; then
  pass "python3 -m unittest ($(grep -E '^Ran [0-9]+ test' "$T/py.out" | awk '{print $2}') tests)"
else
  fail "python3 -m unittest failed (exit $pc)"
  show < "$T/py.out"
fi

feat=()
[ "$(uname -s)" = Darwin ] || feat=(--no-default-features)
cargo test --release --workspace --no-fail-fast ${feat[@]+"${feat[@]}"} > "$T/rs.out" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  rn=$(grep -E '^test result: ok' "$T/rs.out" | awk '{s+=$4} END {print s+0}')
  pass "cargo test --release --workspace: $rn passed"
else
  fail "cargo test --release --workspace failed (exit $rc)"
  tail -n 15 "$T/rs.out" | show
fi

echo "== 2. skintokens baseline vs humanoid-route five-finger hands on a generated human"
GEN_HUMAN="$(ls "$C/mixes/"*.glb 2>/dev/null | sort | head -n 1)"
[ -n "$GEN_HUMAN" ] && [ -f "$GEN_HUMAN" ] || { echo "FAIL  no generated human in $C/mixes"; echo "GATE FAIL"; exit 2; }

build_bin() {
  local dir="$1" pkg="$2" bin="$3"
  shift 3
  (cd "$dir" && cargo build --release --locked -p "$pkg" "$@" --message-format=json 2>/dev/null \
    | python3 -c 'import json,sys
bin_name = sys.argv[1]
exes = []
for line in sys.stdin:
    try:
        m = json.loads(line)
    except Exception:
        continue
    if m.get("executable") and m.get("target", {}).get("name") == bin_name:
        exes.append(m["executable"])
print(exes[-1] if exes else "")' "$bin")
}

SK_BIN="$(build_bin "$ROOT" skintokens skintokens ${feat[@]+"${feat[@]}"})"
[ -x "$SK_BIN" ] || { echo "FAIL  could not build skintokens binary"; echo "GATE FAIL"; exit 2; }
MF_BIN="${MOTIONFORGE_BIN:-$(build_bin "$MOTION/rust" motionforge motionforge)}"
[ -x "$MF_BIN" ] || { echo "FAIL  could not build motionforge binary"; echo "GATE FAIL"; exit 2; }
WF_BIN="${WEIGHTS_BIN:-$(build_bin "$MESHWEIGHTS/rust" weights weights)}"
[ -x "$WF_BIN" ] || { echo "FAIL  could not build weights binary"; echo "GATE FAIL"; exit 2; }
export MOTIONFORGE_BIN="$MF_BIN" WEIGHTS_BIN="$WF_BIN" SKINTOKENS_BIN="$SK_BIN"

python3 - "$GEN_HUMAN" "$T/bare_human.glb" <<'PY'
import json, struct, sys
b = open(sys.argv[1], "rb").read()
jl = struct.unpack("<I", b[12:16])[0]
j = json.loads(b[20:20+jl])
boff = 20 + jl
bl = struct.unpack("<I", b[boff:boff+4])[0]
binb = b[boff+8:boff+8+bl]
j.pop("skins", None)
j.pop("animations", None)
nodes = j.get("nodes", [])
mesh_nodes = [i for i, n in enumerate(nodes) if "mesh" in n]
j["nodes"] = [dict(nodes[i]) for i in mesh_nodes]
for n in j["nodes"]:
    n.pop("skin", None)
    n.pop("children", None)
j["scenes"] = [{"nodes": list(range(len(j["nodes"])))}]
for m in j.get("meshes", []):
    for p in m.get("primitives", []):
        p["attributes"].pop("JOINTS_0", None)
        p["attributes"].pop("WEIGHTS_0", None)
jb = json.dumps(j, separators=(",", ":")).encode()
jb += b" " * ((4 - len(jb) % 4) % 4)
binb += b"\x00" * ((4 - len(binb) % 4) % 4)
open(sys.argv[2], "wb").write(
    b"glTF" + struct.pack("<II", 2, 20 + len(jb) + 8 + len(binb))
    + struct.pack("<I", len(jb)) + b"JSON" + jb
    + struct.pack("<I", len(binb)) + b"BIN\x00" + binb
)
PY

CKPT="$SK_WEIGHTS/grpo_1400.ckpt"
[ -f "$CKPT" ] || CKPT="$(find "$SK_WEIGHTS" -name '*.ckpt' | head -n 1)"
SKINTOKENS_CKPT="$CKPT" "$SK_BIN" rig "$T/bare_human.glb" "$T/sk_raw.glb" --class humanoid --seed 1 > "$T/sk_rig.log" 2>&1
rc_sk=$?
"$MF_BIN" standardize --input "$T/sk_raw.glb" --output "$T/sk_std.glb" > "$T/sk_std.log" 2>&1
rc_std=$?
if [ $rc_sk -eq 0 ] && [ $rc_std -eq 0 ] && [ -f "$T/sk_std.glb" ]; then
  pass "skintokens baseline humanoid rig built"
else
  fail "skintokens baseline rig failed"
  cat "$T/sk_rig.log" "$T/sk_std.log" | show
fi

if [ -f "$ROOT/tools/humanoid_hands.py" ]; then
  # Verify both: (a) wrapped humanoid-route generated human vs skintokens baseline,
  # and (b) grafting base_body five-finger hands onto a welded humanoid rig.
  python3 "$ROOT/tools/humanoid_hands.py" \
    --input "$GEN_HUMAN" \
    --baseline "$T/sk_std.glb" \
    --base-body "$C/base/base_body.glb" \
    --motion-clip-rig "$C/hlh/andras_rig.glb" \
    --out "$T/human_wrap_5f.glb" \
    --report "$T/hh_wrap.json" > "$T/hh_wrap.log" 2>&1
  rc_hw=$?
  cat "$T/hh_wrap.log"
  if [ $rc_hw -eq 0 ] && [ -f "$T/human_wrap_5f.glb" ]; then
    pass "tools/humanoid_hands.py verified wrapped humanoid-route generated human"
  else
    fail "tools/humanoid_hands.py failed on wrapped generated human (exit $rc_hw)"
  fi

  python3 "$ROOT/tools/humanoid_hands.py" \
    --input "$C/hlh/andras_rig.glb" \
    --baseline "$C/hlh/andras_rig.glb" \
    --base-body "$C/base/base_body.glb" \
    --motion-clip-rig "$C/hlh/andras_rig.glb" \
    --out "$T/human_graft_5f.glb" \
    --report "$T/hh_graft.json" > "$T/hh_graft.log" 2>&1
  rc_hg=$?
  cat "$T/hh_graft.log"
  if [ $rc_hg -eq 0 ] && [ -f "$T/human_graft_5f.glb" ]; then
    pass "tools/humanoid_hands.py grafted and verified 5-finger welded human"
  else
    fail "tools/humanoid_hands.py failed on welded human graft (exit $rc_hg)"
  fi
else
  fail "tools/humanoid_hands.py missing (skintokens has no humanoid five-finger hand integration)"
  echo "== baseline skintokens rig fails motion-1 grip_gate.sh --five-finger (expected RED)"
  (cd "$MOTION" && bash tools/bench/grip_gate.sh --five-finger "$T/sk_std.glb") > "$T/grip_red.log" 2>&1 || true
  grep -E "five_finger|GRIP GATE|ERROR|failed" "$T/grip_red.log" | tail -n 10
fi

echo "== 3. Direct motion-1 grip_gate.sh --five-finger (>= 0.90) & weights check hand parity"
if [ -f "$T/human_wrap_5f.glb" ] && [ -f "$T/human_graft_5f.glb" ]; then
  (cd "$MOTION" && bash tools/bench/grip_gate.sh --five-finger "$T/human_wrap_5f.glb") > "$T/grip_wrap.log" 2>&1
  rc_g1=$?
  grep -E "five_finger|GRIP GATE|ERROR" "$T/grip_wrap.log" || tail -n 15 "$T/grip_wrap.log"
  if [ $rc_g1 -eq 0 ]; then
    pass "wrapped generated human passes grip_gate.sh --five-finger (>= 0.90)"
  else
    fail "wrapped generated human failed grip_gate.sh --five-finger (exit $rc_g1)"
  fi

  (cd "$MOTION" && bash tools/bench/grip_gate.sh --five-finger "$T/human_graft_5f.glb") > "$T/grip_graft.log" 2>&1
  rc_g2=$?
  grep -E "five_finger|GRIP GATE|ERROR" "$T/grip_graft.log" || tail -n 15 "$T/grip_graft.log"
  if [ $rc_g2 -eq 0 ]; then
    pass "grafted five-finger human passes grip_gate.sh --five-finger (>= 0.90)"
  else
    fail "grafted five-finger human failed grip_gate.sh --five-finger (exit $rc_g2)"
  fi

  "$WF_BIN" check "$T/sk_std.glb" --motion "$C/hlh/andras_rig.glb" --json > "$T/wc_base.json" 2>"$T/wc_base.err" || true
  "$WF_BIN" check "$T/human_wrap_5f.glb" --motion "$C/hlh/andras_rig.glb" --json > "$T/wc_wrap.json" 2>"$T/wc_wrap.err" || true
  "$WF_BIN" check "$C/hlh/andras_rig.glb" --motion "$C/hlh/andras_rig.glb" --json > "$T/wc_orig.json" 2>"$T/wc_orig.err" || true
  "$WF_BIN" check "$T/human_graft_5f.glb" --motion "$C/hlh/andras_rig.glb" --json > "$T/wc_graft.json" 2>"$T/wc_graft.err" || true
  python3 - "$T/wc_base.json" "$T/wc_wrap.json" "$T/wc_orig.json" "$T/wc_graft.json" <<'PY'
import json, os, sys

def hand_metrics(path):
    if not os.path.exists(path) or not os.path.getsize(path):
        raise RuntimeError(f"missing/empty weights check output: {path}")
    d = json.load(open(path))
    fails, total, area = 0, 0, 0.0
    for sec in d.get("headline", []):
        for f in sec.get("findings", []):
            txt = (f.get("part", "") + " " + f.get("detail", "")).lower()
            if any(w in txt for w in ("hand", "finger", "thumb", "wrist")):
                total += 1
                if f.get("severity") == "fail":
                    fails += 1
                area += float(f.get("area_cm2", 0.0))
    return fails, total, round(area, 2)

bf, bt, ba = hand_metrics(sys.argv[1])
wf, wt, wa = hand_metrics(sys.argv[2])
of, ot, oa = hand_metrics(sys.argv[3])
gf, gt, ga = hand_metrics(sys.argv[4])
print(f"WEIGHTS_HANDS sk_base=(fails={bf},total={bt},cm2={ba}) wrap_5f=(fails={wf},total={wt},cm2={wa})")
print(f"WEIGHTS_HANDS welded_orig=(fails={of},total={ot},cm2={oa}) graft_5f=(fails={gf},total={gt},cm2={ga})")
ok = (wf <= bf and wt <= bt and wa <= ba + 1e-3 and gf <= of and gt <= ot and ga <= oa + 1e-3)
sys.exit(0 if ok else 1)
PY
  rc_wc=$?
  if [ $rc_wc -eq 0 ]; then
    pass "meshweights weights check hand findings no worse than baseline"
  else
    fail "meshweights weights check hand findings worse than baseline"
  fi
else
  fail "no five-finger generated human output to test"
fi

echo
if [ "$FAILS" -eq 0 ]; then
  echo "GATE PASS"
  exit 0
fi
echo "GATE FAIL ($FAILS)"
exit 1
