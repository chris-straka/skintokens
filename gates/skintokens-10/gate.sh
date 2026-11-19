#!/usr/bin/env bash
# Gate for skintokens-10 (game-tools docs/plans/skintokens.md), its Python part: plumbing in one
# place, and blind packs can't delete a folder.
#  - tools/toolpaths.py defines find() and run(); creature_route.py, creature_bench.py,
#    creature_follow.py and creature_blind.py define none of their own (find, find_bin, run,
#    run_cmd) and use it;
#  - tools/pngfont.py holds the PNG reader and writer and the bitmap font (FONT_5X7,
#    read_png_rgb, write_png_rgb, draw_text); creature_blind.py defines none of them (it may
#    import them back) and is at most 760 lines;
#  - a blind pack made by this tree is byte-identical to one made by the base's
#    creature_blind.py (same items, same seed): every pair_NN.png, pairs.json and the key;
#  - tools/gate/carbon.sh passes: judges and tests untouched; the Python tests (the two
#    write_blind_pack refusal tests among them) and the Rust workspace green.
# Builds the Rust workspace (through fleet, about 8 cores). Prints PASS/FAIL per check, ends
# with GATE PASS or GATE FAIL.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
SLUG=skintokens-10
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
BASE=$(git log -1 --format=%H -- gates/$SLUG)
[ -n "$BASE" ] || { fail "cannot find the commit that added the gate"; BASE=HEAD; }

echo "== one place for the plumbing"
python3 - > "$T/ast.out" 2>&1 <<'PY'
import ast, os, sys
bad = []
def defs(f):
    if not os.path.exists(f):
        bad.append(f"{f} is missing"); return {}
    t = ast.parse(open(f).read())
    return {n.name: n for n in t.body if isinstance(n, (ast.FunctionDef, ast.Assign, ast.ClassDef)) and hasattr(n, "name")} | \
           {x.id: n for n in t.body if isinstance(n, ast.Assign) for x in n.targets if isinstance(x, ast.Name)}
tp = defs("tools/toolpaths.py")
for need in ("find", "run"):
    if need not in tp: bad.append(f"tools/toolpaths.py defines no {need}()")
pf = defs("tools/pngfont.py")
for need in ("FONT_5X7", "read_png_rgb", "write_png_rgb", "draw_text"):
    if need not in pf: bad.append(f"tools/pngfont.py has no {need}")
for f in ("tools/creature_route.py", "tools/creature_bench.py", "tools/creature_follow.py", "tools/creature_blind.py"):
    d = defs(f)
    for own in ("find", "find_bin", "run", "run_cmd"):
        if own in d and isinstance(d[own], ast.FunctionDef): bad.append(f"{f} still defines {own}()")
    if "toolpaths" not in open(f).read(): bad.append(f"{f} doesn't use toolpaths")
d = defs("tools/creature_blind.py")
for own in ("FONT_5X7", "read_png_rgb", "write_png_rgb", "draw_text"):
    if own in d and not isinstance(d[own], ast.ImportFrom): bad.append(f"tools/creature_blind.py still defines {own}")
n = sum(1 for _ in open("tools/creature_blind.py"))
if n > 760: bad.append(f"tools/creature_blind.py is {n} lines (at most 760)")
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY
if [ $? -eq 0 ]; then pass "find/run in toolpaths.py, PNG and font in pngfont.py, no copies left"; else fail "plumbing:"; show < "$T/ast.out"; fi

echo "== blind packs byte-identical to the base's"
mkdir -p "$T/base" && git archive "$BASE" tools | tar -x -C "$T/base"
cat > "$T/pack.py" <<'PY'
import os, sys
tools, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
import creature_blind as cb
os.makedirs(out + "_in", exist_ok=True)
pngs = []
for k, col in enumerate(([80, 120, 160], [180, 70, 60], [20, 200, 90])):
    p = os.path.join(out + "_in", f"p{k}.png")
    cb.write_png_rgb(p, 32, 24, bytearray(col * (32 * 24)))
    pngs.append(p)
items = [
    {"category": "claim", "subject": "fox", "a_name": "predicted", "b_name": "body", "a_pngs": [pngs[0], pngs[2]],
     "b_pngs": [pngs[1], pngs[1]], "truth_winner": "predicted", "pick_winner": "predicted", "old_winner": "body"},
    {"category": "control", "subject": "fox", "a_name": "same_a", "b_name": "same_b", "a_pngs": [pngs[0]],
     "b_pngs": [pngs[0]], "expected": "cant_tell"},
    {"category": "planted", "subject": "fox", "a_name": "clean", "b_name": "planted_swapped_limbs", "a_pngs": [pngs[2]],
     "b_pngs": [pngs[1]], "expected": "clean", "planted_metric_ok": True},
]
cb.write_blind_pack(items, os.path.join(out, "pack"), 20261119)
PY
(cd "$T" && python3 "$T/pack.py" "$T/base/tools" "$T/a") > "$T/a.log" 2>&1
R=$(pwd)
(cd "$T" && python3 "$T/pack.py" "$R/tools" "$T/b") > "$T/b.log" 2>&1
if [ ! -f "$T/a/pack/pairs.json" ]; then fail "the base made no pack"; show < "$T/a.log"
elif [ ! -f "$T/b/pack/pairs.json" ]; then fail "this tree made no pack"; show < "$T/b.log"
else
  diffs=""
  for f in $(cd "$T/a/pack" && ls); do cmp -s "$T/a/pack/$f" "$T/b/pack/$f" || diffs="$diffs $f"; done
  [ "$(ls "$T/a/pack" | wc -l)" = "$(ls "$T/b/pack" | wc -l)" ] || diffs="$diffs (file count)"
  python3 - "$T/a/pack.key.json" "$T/b/pack.key.json" "$T/a" "$T/b" <<'PY' || diffs="$diffs pack.key.json"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ra, rb = sys.argv[3], sys.argv[4]
s = lambda o, r: json.loads(json.dumps(o).replace(r, "ROOT"))
sys.exit(0 if s(a, ra) == s(b, rb) else 1)
PY
  if [ -z "$diffs" ]; then pass "pack identical: $(ls "$T/a/pack" | tr '\n' ' ')+ key"; else fail "pack differs:$diffs"; fi
fi

echo "== skintokens' gate"
GATE_BASE=$BASE bash tools/gate/carbon.sh
[ $? -eq 0 ] || FAILS=$((FAILS + 1))

if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
