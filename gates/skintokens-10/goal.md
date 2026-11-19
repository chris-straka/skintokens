# skintokens-10: plumbing in one place; blind packs can't delete a folder

Repo: skintokens (`~/Games/_blender/skintokens`). One goal, this repo only. Python only.

## Why

The same helpers are written four times: `find` / `find_bin` (look a program up by flag,
environment variable, PATH and fallbacks) and `run` / `run_cmd` (run a command, return its exit
code and output) in `tools/creature_route.py:59-75`, `tools/creature_bench.py:45-55`,
`tools/creature_follow.py:40-50` and `tools/creature_blind.py:447-457`. `creature_blind.py`
(917 lines) also carries its own PNG reader and writer and a bitmap font (lines 71-260:
`FONT_5X7`, `read_png_rgb`, `write_png_rgb`, `draw_text` and their helpers). And
`write_blind_pack` (`creature_blind.py:532`) deletes whatever folder `--out` names
(`shutil.rmtree`), with no check that it is a blind pack.

The test that says what "done" means for the last point is written:
`tools/test_creature_blind.py`, class `PackRefusesStrangeFoldersTest` (read it). It fails today.

## What to build

1. `tools/toolpaths.py`: one `find(name, explicit, env, fallbacks, first=())` and one
   `run(cmd, timeout=None, env=None)`, taken from `creature_route.py`'s (it already stops with
   exit 2 and a message on a missing program; its `run` returns a dict with code, secs, out,
   err). The other three files use them; where a caller wants the old shapes (`run_cmd`
   returned a tuple `(code, stdout, stderr)`), adapt at the call site or keep a one-line
   wrapper that calls `toolpaths.run`, but no file defines its own lookup or runner.
2. `tools/pngfont.py`: the PNG reader and writer and the bitmap font moved out of
   `creature_blind.py` unchanged (cut and paste: same code, same constants), imported back by
   name so `from creature_blind import write_png_rgb` (the tests do this) still works.
   `creature_blind.py` ends at most 760 lines.
3. `write_blind_pack` replaces only a pack it made: an `out_dir` that exists must hold its own
   `pairs.json` and have its key beside it (`<out_dir>.key.json`), or be empty; anything else
   raises `ValueError` naming the folder, and nothing is deleted.
4. Nothing else changes: same packs byte for byte (pair sheets, pairs.json, key), same exit
   codes, same messages.

## Evidence

`bash gates/skintokens-10/gate.sh` (through fleet; it builds the Rust workspace) prints
`GATE PASS`. It checks the two new modules and that no copies are left (by parsing the files),
creature_blind.py's length, that a pack made by your tree is byte-identical to one the base
makes from the same inputs and seed, and then runs `tools/gate/carbon.sh` (guarded paths
untouched; Python tests, the refusal tests among them; the Rust workspace).

## Out of scope; don't

- Don't edit `tools/pick_score.py`, `tools/bench/` (its `score_artists.judge` stays where it
  is), `crates/` (the Rust sorts are a separate change), `tools/gate/`, `gates/` or any test.
- Don't change any number, threshold, picture layout or output format.
- Don't start Blender or build anything on this Mac.
