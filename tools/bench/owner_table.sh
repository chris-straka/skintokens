#!/bin/bash
# The owner's creatures after a route run: tears, the skeleton checks and a
# sheet per creature, then one table.
#
#   BENCH=/path/to/bench tools/bench/owner_table.sh TAG SET.tsv [OLD_PATTERN]
#
# For each creature in SET.tsv (name, kind, model) with a rig in
# $BENCH/owner/TAG/<name>/<name>.glb:
#   - the body's range of motion written on that rig's own bones
#     (`creature-rig check --rom`), and `weights check` with that pose set,
#     once without the rig's clips (like for like across rigs) and once
#     with them: tears = vertices in failing stretch, volume and bleed
#     findings as a share of all vertices;
#   - the route's skeleton check and the no-artist check;
#   - a sheet: the rig painted from the side and its skeleton over the
#     body's parts (three views), above the rig bent through its range of
#     motion, failing skin red.
# OLD_PATTERN (with %n for the name) adds the same tears numbers for an
# earlier rig of each creature, where there is one.
# Writes $BENCH/owner/TAG.table.md, TAG.table.json and TAG.sheets/<name>.jpg.
set -uo pipefail
BENCH=${BENCH:?set BENCH}; TAG=$1; SET=$2; OLD=${3:-}
TOOLS=${TOOLS:-/home/f/rigwork/creatures}
CR=${CREATURE_RIG_BIN:-$BENCH/bin/creature-rig}; CB=${CREATURE_BENCH_BIN:-$BENCH/bin/creature-bench}
WF=${WEIGHTS_BIN:-$TOOLS/bin/weights}
O=$BENCH/owner/$TAG; S=$O.sheets; mkdir -p "$S"
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH
one() {
  n=$1; rig=$2; d=$3
  mkdir -p "$d"
  "$CR" check "$rig" --json "$d/skeleton.json" --rom "$d/rom.ron" --sheet "$d/skeleton.png" > /dev/null 2>&1
  "$WF" check "$rig" --json --no-clips --poses "$d/rom.ron" > "$d/bend_noclips.json" 2>/dev/null
  "$WF" check "$rig" --json --poses "$d/rom.ron" > "$d/bend.json" 2>/dev/null
  "$CB" anatomy "$rig" --json "$d/anatomy.json" > /dev/null
}
export -f one; export CR CB WF
while IFS=$'\t' read -r n kind model; do
  [ -z "$n" ] && continue
  rig=$O/$n/$n.glb
  [ -f "$rig" ] || { echo "$n: no rig"; continue; }
  one "$n" "$rig" "$O/$n/check"
  "$WF" sheet "$rig" --out "$O/$n/check/posed.png" --poses "$O/$n/check/rom.ron" --cols 6 > /dev/null 2>&1
  blender -b --factory-startup -P "$HERE/render_side.py" -- "$rig" --out "$O/$n/check/painted.png" --camera-out "$O/$n/check/painted_camera.json" \
    --width 900 --height 640 --fill 0.8 --at 0.5 --ground 0.88 > /dev/null 2>&1
  if [ -f "$O/$n/check/skeleton.png" ] && [ -f "$O/$n/check/posed.png" ]; then
    top=$O/$n/check/skeleton.png
    if [ -f "$O/$n/check/painted.png" ]; then
      convert "$O/$n/check/painted.png" -resize x500 "$O/$n/check/painted_s.png"
      convert "$O/$n/check/skeleton.png" -resize x500 "$O/$n/check/skeleton_s.png"
      convert "$O/$n/check/painted_s.png" "$O/$n/check/skeleton_s.png" +append "$O/$n/check/top.png"
      top=$O/$n/check/top.png
    fi
    convert "$top" "$O/$n/check/posed.png" -resize 1600x -background "#1c1e22" -gravity center -extent 1600x \
      -append -quality 80 "$S/$n.jpg" 2>/dev/null
  fi
  if [ -n "$OLD" ]; then
    old=${OLD//%n/$n}
    [ -f "$old" ] && one "$n" "$old" "$O/$n/check_old"
  fi
  echo "$n done"
done < "$SET"
python3 - "$O" "$SET" "$TAG" <<'PY'
import json, os, sys
O, SET, TAG = sys.argv[1:4]
TEARS = ("D_STRETCH", "D_VOLUME", "D_BLEED")
def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None
def tears(d):
    if not d:
        return None
    f = [x for x in d.get("findings", []) if x.get("severity") == "fail" and x["code"] in TEARS]
    return round(100.0 * sum(x["verts"] for x in f) / max(d.get("verts", 1), 1), 2)
WORD = {"B_LOPSIDED": "left/right", "B_NO_HEAD": "no head", "B_NO_ARMS": "no arms", "B_NO_LEG": "foot without leg", "B_KIND": "misnamed limb", "B_OFF_CENTRE": "centre bone off"}
rows, js = [], []
for line in open(SET):
    p = line.rstrip("\n").split("\t")
    if len(p) < 2 or not p[0]:
        continue
    n, kind = p[0], p[1]
    c = os.path.join(O, n, "check")
    rep = load(os.path.join(O, n, "report.json"))
    if not rep:
        rows.append(f"| {n} | {kind} | no rig | | | | | |")
        continue
    t = rep["tries"][0]
    a = load(os.path.join(c, "anatomy.json")) or {}
    sk = load(os.path.join(c, "skeleton.json")) or {}
    nc, wc = load(os.path.join(c, "bend_noclips.json")), load(os.path.join(c, "bend.json"))
    old_nc = load(os.path.join(O, n, "check_old", "bend_noclips.json"))
    old_a = load(os.path.join(O, n, "check_old", "anatomy.json"))
    af = sorted({WORD.get(f["code"], f["code"]) for f in a.get("findings", []) if f["severity"] == "fail"})
    old_af = sorted({WORD.get(f["code"], f["code"]) for f in (old_a or {}).get("findings", []) if f["severity"] == "fail"}) if old_a else None
    limbs = ", ".join(f"{k} {v}" for k, v in (t.get("limbs") or {}).items())
    row = {"name": n, "kind": kind, "read_as": t.get("body"), "turned_deg": None, "kept": t.get("picked"), "limbs": limbs,
           "tears_noclips_pct": tears(nc), "tears_with_clips_pct": tears(wc), "clips": bool(t.get("clips")) if t.get("picked") == "body" else bool((t.get("predicted") or {}).get("clips")),
           "skeleton_check": sk.get("score"), "no_artist_check": "passes" if not af else ", ".join(af),
           "old_tears_noclips_pct": tears(old_nc), "old_no_artist_check": (("passes" if not old_af else ", ".join(old_af)) if old_a is not None else None)}
    rig_json = load(os.path.join(O, n, "try.rig.json"))
    if rig_json:
        row["turned_deg"] = rig_json.get("turned_deg")
    js.append(row)
    old = "" if row["old_tears_noclips_pct"] is None else f"{row['old_tears_noclips_pct']}%, {row['old_no_artist_check']}"
    turned = f", turned {row['turned_deg']:.0f} degrees" if row["turned_deg"] not in (None, 0, 0.0) else ""
    rows.append(f"| {n} | {kind} | {row['read_as']}{turned}; kept the {'body-read' if row['kept'] == 'body' else 'model'} rig | "
                f"{row['tears_noclips_pct']}% | {row['tears_with_clips_pct']}%{' (has clips)' if row['clips'] else ''} | {row['skeleton_check']} | {row['no_artist_check']} | {old} |")
ok = [r for r in js if r["tears_noclips_pct"] is not None]
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
head = [f"# The owner's creatures, route {TAG}", "",
        "Tears = vertices in failing stretch, volume and bleed findings of the bend checker, as a share of all vertices, on the body's range of motion written on each rig's own bones "
        "(without the rig's clips: like for like; with them: what ships). Hand-made plain game rigs sit at 0.3 to 0.7% on a range of motion; the twelve hand-made animals of the benchmark "
        "sit at 0 to 6.7% (median 0.8%) in their own clips. Skeleton check = the route's own (0 to 100). No-artist check = `creature-bench anatomy`. "
        "Before = this morning's route, where it rigged that creature.", "",
        "| Creature | Kind | Read as; rig kept | Tears, no clips | Tears, with clips | Skeleton check | No-artist check | Before: tears, no-artist check |",
        "|---|---|---|---|---|---|---|---|"]
foot = ["", f"Over {len(ok)} rigs: tears without clips median {med([r['tears_noclips_pct'] for r in ok])}%, at or under 0.7% on "
        f"{sum(r['tears_noclips_pct'] <= 0.7 for r in ok)}; no-artist check passes on {sum(r['no_artist_check'] == 'passes' for r in js)} of {len(js)}.", ""]
open(f"{O}.table.md", "w").write("\n".join(head + rows + foot))
json.dump(js, open(f"{O}.table.json", "w"), indent=1)
print("\n".join(head[-2:] + rows + foot))
PY
