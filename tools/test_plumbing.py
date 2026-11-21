#!/usr/bin/env python3
"""Unit tests for skintokens-10 consolidated plumbing (`toolpaths`, `rigjudge`, `pngfont`)."""
import ast
import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(TOOLS_DIR, "bench")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, BENCH_DIR)


def _top_defs(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


class PlumbingConsolidationTest(unittest.TestCase):
    def test_toolpaths_optional_find_returns_none_when_missing(self):
        import toolpaths

        found = toolpaths.find(
            "definitely-missing-binary-xyz",
            "",
            "SKINTOKENS_TEST_MISSING_ENV_XYZ",
            ["/nonexistent/path/definitely-missing-binary-xyz"],
            required=False,
        )
        self.assertIsNone(found)

    def test_single_rigjudge_module_shared_by_blind_and_artists(self):
        import creature_blind
        import rigjudge
        import score_artists
        import toolpaths

        self.assertTrue(callable(rigjudge.judge))
        self.assertIs(creature_blind.judge_rig_metrics, rigjudge.judge)
        self.assertIs(score_artists.judge, rigjudge.judge)
        self.assertIs(score_artists.find, toolpaths.find)

    def test_no_duplicate_find_run_or_judge_definitions(self):
        sa_defs = _top_defs(os.path.join(BENCH_DIR, "score_artists.py"))
        for dup in ("find", "run", "judge"):
            self.assertNotIn(dup, sa_defs, f"score_artists.py still defines {dup}()")

        cb_defs = _top_defs(os.path.join(TOOLS_DIR, "creature_blind.py"))
        self.assertNotIn("judge_rig_metrics", cb_defs, "creature_blind.py still defines judge_rig_metrics()")

        with open(os.path.join(TOOLS_DIR, "creature_route.py"), encoding="utf-8") as f:
            route_src = f.read()
        self.assertNotIn('shutil.which("skintokens")', route_src)
        self.assertIn("required=False", route_src)


if __name__ == "__main__":
    unittest.main()
