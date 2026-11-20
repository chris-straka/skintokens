#!/usr/bin/env python3
"""Unit tests for the frozen `skintokens-1` held-out manifest & verifier (stdlib only)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(TOOLS_DIR, "bench")
sys.path.insert(0, BENCH_DIR)
import verify_heldout as vh  # noqa: E402


class HeldoutManifestTest(unittest.TestCase):
    def test_committed_manifest_passes_all_invariants(self):
        doc = vh.validate_manifest(vh.DEFAULT_MANIFEST)
        self.assertGreaterEqual(len(doc["animals"]), 12)
        self.assertEqual(doc.get("in_sample_overlap"), 0)

    def test_rejects_overlap_and_bad_checksum(self):
        with open(vh.DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
            doc = json.load(f)
        with tempfile.TemporaryDirectory() as tmp:
            # Overlap with in-sample id refused.
            bad = json.loads(json.dumps(doc))
            bad["animals"][0]["id"] = "fox"
            p = os.path.join(tmp, "overlap.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(bad, f)
            with self.assertRaises(ValueError):
                vh.validate_manifest(p)

            # Workspace SHA mismatch refused.
            bench = os.path.join(tmp, "bench")
            for a in doc["animals"]:
                adir = os.path.join(bench, "animals", a["id"])
                os.makedirs(adir, exist_ok=True)
                with open(os.path.join(adir, "bare.glb"), "wb") as f:
                    f.write(b"fake")
                with open(os.path.join(adir, "truth.glb"), "wb") as f:
                    f.write(b"fake")
            with self.assertRaises(ValueError):
                vh.verify_workspace(bench, vh.DEFAULT_MANIFEST)


if __name__ == "__main__":
    unittest.main()
