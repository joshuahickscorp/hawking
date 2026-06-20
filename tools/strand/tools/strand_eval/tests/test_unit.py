# strand_eval unit tests — naming, harness_key, ledger tells. Torch-free.
# Run: /usr/local/bin/python3 -m unittest discover -s tools/strand_eval/tests -v
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from strand_eval import HARNESS_VERSION, locate_repo_root
from strand_eval.core import (build_record, harness_key, model_id_from_dir,
                              output_path, sanitize)
from strand_eval.ledger import append_record, check, ingest, read_ledger


class TestNaming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="se_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk(self, *parts):
        p = os.path.join(self.tmp, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def test_sanitize(self):
        self.assertEqual(sanitize("qwen2.5-0.5B"), "qwen2.5-0.5B")
        self.assertEqual(sanitize("a b/c"), "a-b-c")
        self.assertEqual(sanitize(""), "unnamed")

    def test_model_id_plain_dir(self):
        d = self._mk("qwen-05b")
        self.assertEqual(model_id_from_dir(d), "qwen-05b")

    def test_model_id_skips_generic_leaf(self):
        d = self._mk("qwen-7b", "reopen", "q2_l12_out1", "recon")
        self.assertEqual(model_id_from_dir(d), "q2_l12_out1")

    def test_output_name_shape(self):
        d = self._mk("qwen-05b")
        p = output_path(self.tmp, model_id_from_dir(d), "q2_l12_out1", d)
        self.assertEqual(os.path.basename(p), "ppl_qwen-05b_q2_l12_out1.json")

    def test_collision_impossible(self):
        # the llama2-overwrote-qwen incident: same derived name, DIFFERENT model dir
        d1 = self._mk("qwen-7b", "x", "recon")     # both reduce to model_id "x"
        d2 = self._mk("llama2-7b", "x", "recon")
        out = self._mk("results")
        p1 = output_path(out, model_id_from_dir(d1), "q2", d1)
        with open(p1, "w") as f:
            json.dump({"model_path": os.path.realpath(d1), "ppl": 1.0}, f)
        p2 = output_path(out, model_id_from_dir(d2), "q2", d2)
        self.assertNotEqual(p1, p2, "second model MUST NOT overwrite the first")
        # same dir again -> same name (resume-friendly overwrite)
        p1b = output_path(out, model_id_from_dir(d1), "q2", d1)
        self.assertEqual(p1, p1b)


class TestHarnessKey(unittest.TestCase):
    def test_fields_and_stability(self):
        hk, k8 = harness_key("mps", "bfloat16", 2048, 64, "wikitext")
        self.assertEqual(hk["harness_version"], HARNESS_VERSION)
        self.assertEqual(len(k8), 8)
        _, k8b = harness_key("mps", "bfloat16", 2048, 64, "wikitext")
        self.assertEqual(k8, k8b, "key must be deterministic")

    def test_any_field_changes_key(self):
        base = ("mps", "bfloat16", 2048, 64, "wikitext")
        _, k0 = harness_key(*base)
        for variant in [("cpu", "bfloat16", 2048, 64, "wikitext"),
                        ("mps", "float32", 2048, 64, "wikitext"),
                        ("mps", "bfloat16", 1024, 64, "wikitext"),
                        ("mps", "bfloat16", 2048, 146, "wikitext"),  # the 64w/146w guard
                        ("mps", "bfloat16", 2048, 64, "Salesforce/wikitext")]:
            _, kv = harness_key(*variant)
            self.assertNotEqual(k0, kv, f"variant {variant} must change the key")

    def test_record_carries_key(self):
        rec = build_record(model_path="/tmp", tag="t", ppl=3.14, ctx=2048, chunks=64,
                           tokens=1, device_resolved="cpu", device_mode="auto",
                           dtype="bfloat16", dataset_id="wikitext", dataset_fp="ab")
        self.assertIn("harness_key8", rec)
        self.assertEqual(rec["harness_key"]["chunks"], 64)
        self.assertIn("timestamp", rec)


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="se_ledger_")
        self.lp = os.path.join(self.tmp, "ledger.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rec(self, model, tag, ppl, device="cpu", chunks=64):
        return build_record(model_path=os.path.join(self.tmp, model), tag=tag, ppl=ppl,
                            ctx=2048, chunks=chunks, tokens=131008,
                            device_resolved=device, device_mode=device,
                            dtype="bfloat16", dataset_id="wikitext", dataset_fp="x")

    def test_fifteen_digit_tell_fires(self):
        # the mp-fallback bug shape: two DIFFERENT configs, bit-identical ppl
        os.makedirs(os.path.join(self.tmp, "qwen-7b"), exist_ok=True)
        append_record(self.lp, self._rec("qwen-7b", "q4", 7.811234567890123))
        append_record(self.lp, self._rec("qwen-7b", "mp_light", 7.811234567890123))
        errors, _ = check(self.lp)
        self.assertEqual(len(errors), 1)
        self.assertIn("15-DIGIT TELL", errors[0])

    def test_close_but_not_identical_ppl_ok(self):
        append_record(self.lp, self._rec("m", "a", 7.8112345678901230))
        append_record(self.lp, self._rec("m", "b", 7.8112345678901239))
        errors, _ = check(self.lp)
        self.assertEqual(errors, [])

    def test_same_config_duplicate_is_warning_not_error(self):
        r = self._rec("m", "a", 9.42)
        append_record(self.lp, r)
        append_record(self.lp, r)
        errors, warnings = check(self.lp)
        self.assertEqual(errors, [])
        self.assertTrue(any("duplicate result" in w for w in warnings))

    def test_harness_mismatch_warns(self):
        # same model+tag measured at 64w and 146w -> NOT the same measurement
        append_record(self.lp, self._rec("qwen-7b", "baseline", 6.629, chunks=64))
        append_record(self.lp, self._rec("qwen-7b", "baseline", 7.7362, chunks=146))
        errors, warnings = check(self.lp)
        self.assertEqual(errors, [])
        self.assertTrue(any("HARNESS MISMATCH (same model+tag)" in w for w in warnings))
        self.assertTrue(any("do NOT compare across keys" in w for w in warnings))

    def test_unprovenanced_warns(self):
        append_record(self.lp, {"model": "m", "tag": "t", "ppl": 5.0,
                                "harness_key": None, "harness_key8": None,
                                "provenance": "ingested"})
        _, warnings = check(self.lp)
        self.assertTrue(any("UN-PROVENANCED" in w for w in warnings))

    def test_ingest_idempotent_and_legacy_shape(self):
        src = os.path.join(self.tmp, "pod-results")
        os.makedirs(src)
        legacy = {"tag": "q2_l12_out1", "ppl": 10.538037609591512, "ctx": 2048,
                  "chunks": 64, "tokens": 131008, "device": "mps",
                  "dtype": "bfloat16", "model": "scratch/qwen-7b/reopen/q2_l12_out1/recon"}
        with open(os.path.join(src, "ppl_qwen-7b_q2_l12_out1.json"), "w") as f:
            json.dump(legacy, f)
        self.assertEqual(ingest(self.lp, src, quiet=True), 1)
        self.assertEqual(ingest(self.lp, src, quiet=True), 0, "second ingest must no-op")
        recs = [r for _, r in read_ledger(self.lp)]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["provenance"], "ingested")
        self.assertEqual(recs[0]["model"], "qwen-7b",
                         "model must come from the filename prefix, not the 'recon' leaf")
        self.assertIsNone(recs[0]["harness_key8"])
        self.assertEqual(recs[0]["ppl"], legacy["ppl"])

    def test_empty_ledger_clean(self):
        open(self.lp, "w").close()
        errors, warnings = check(self.lp)
        self.assertEqual((errors, warnings), ([], []))


class TestRepoLocation(unittest.TestCase):
    def test_locates_this_repo(self):
        root = locate_repo_root()
        self.assertTrue(os.path.isfile(os.path.join(root, "Cargo.toml")))
        with open(os.path.join(root, "Cargo.toml")) as f:
            self.assertIn("strand-quant", f.read())


if __name__ == "__main__":
    unittest.main()
