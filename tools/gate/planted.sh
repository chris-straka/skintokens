#!/usr/bin/env bash
# Proof that tools/gate/carbon.sh can fail (skintokens-4): in a throwaway clone, one worker
# commit ("carbon planted: round 1") per guarded path must turn its scope check red, and a
# commit to an unguarded file must leave it green. Builds nothing.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
fails=0
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
git clone -q --no-hardlinks . "$T/r" 2>/dev/null || { echo "FAIL  clone"; exit 1; }
# the working tree's gate, in case it isn't committed yet
mkdir -p "$T/r/tools/gate" && cp tools/gate/carbon.sh "$T/r/tools/gate/carbon.sh"
cd "$T/r" || exit 2
git add -A && git -c user.name=t -c user.email=t@t commit -q -m "gate under test" --allow-empty
case_() { # name, file to touch (or a command), want 0|1
  git reset -q --hard HEAD && git clean -qfd
  base=$(git rev-parse HEAD)
  eval "$2"
  git add -A && git -c user.name=t -c user.email=t@t commit -q -m "carbon planted: round 1"
  GATE_BASE=$base bash tools/gate/carbon.sh --scope-only > out.txt 2>&1
  got=$?
  if [ "$got" = "$3" ]; then echo "PASS  $1 (exit $got)"; else echo "FAIL  $1 (exit $got, want $3)"; sed 's/^/      /' out.txt; fails=$((fails + 1)); fi
  git reset -q --hard "$base"
}
case_ "control: route code" 'echo "# planted" >> tools/creature_route.py' 0
case_ "pick_score.py" 'echo "# planted" >> tools/pick_score.py' 1
case_ "tools/bench/" 'echo "# planted" >> tools/bench/score_artists.py' 1
case_ "held-out manifest" 'echo "{}" > tools/bench/heldout_manifest.json' 1
case_ "crates/bench/" 'echo "// planted" >> crates/bench/src/main.rs' 1
case_ "creature check.rs" 'echo "// planted" >> crates/creature/src/check.rs' 1
case_ "a Python test" 'echo "# planted" >> tools/test_pick_score.py' 1
case_ "a Rust test file" 'echo "// planted" >> crates/skintokens/tests/fast.rs' 1
case_ "the gate" 'echo "# planted" >> tools/gate/carbon.sh' 1
case_ "a #[test] removed" 'python3 -c "
import re,glob
for f in sorted(glob.glob(\"crates/creature/src/*.rs\")):
    s=open(f).read()
    if \"#[test]\" in s:
        open(f,\"w\").write(s.replace(\"#[test]\",\"\",1)); break"' 1
case_ "an #[ignore] added" 'python3 -c "
import glob
for f in sorted(glob.glob(\"crates/creature/src/*.rs\")):
    s=open(f).read()
    if \"#[test]\" in s:
        open(f,\"w\").write(s.replace(\"#[test]\",\"#[test]\n#[ignore]\",1)); break"' 1
if [ "$fails" -eq 0 ]; then echo "PLANTED PASS"; exit 0; else echo "PLANTED FAIL ($fails)"; exit 1; fi
