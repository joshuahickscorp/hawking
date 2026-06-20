# 2-chunk SMOKE on scratch/qwen-05b — the canon 64-ch protocol SHAPE at 2 windows.
# Loads torch + the 0.5B: gated behind STRAND_EVAL_SMOKE=1 so the unit suite stays
# seconds. Run explicitly:
#   STRAND_EVAL_SMOKE=1 /usr/local/bin/python3 -m unittest discover \
#       -s tools/strand_eval/tests -p test_smoke.py -v     (from the repo root)
# Box note: this is a measurement-adjacent run — stamp machine state in any
# number you quote from it (device, idle/contended).
import json
import os
import shutil
import sys
import tempfile
import unittest

_TESTS = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.dirname(_TESTS)
REPO = os.path.dirname(os.path.dirname(PKG_DIR))
sys.path.insert(0, os.path.dirname(PKG_DIR))

MODEL = os.path.join(REPO, "scratch", "qwen-05b")
GATED = os.environ.get("STRAND_EVAL_SMOKE") != "1"


@unittest.skipIf(GATED, "set STRAND_EVAL_SMOKE=1 to run the 0.5B smoke")
@unittest.skipIf(not os.path.isdir(MODEL), "scratch/qwen-05b not present")
class TestSmoke05B(unittest.TestCase):
    def test_two_chunk_canon_shape(self):
        from strand_eval import HARNESS_VERSION
        from strand_eval.core import run_eval
        from strand_eval.ledger import check, read_ledger

        device = os.environ.get("STRAND_EVAL_SMOKE_DEVICE", "auto")
        tmp = tempfile.mkdtemp(prefix="se_smoke_")
        try:
            ledger = os.path.join(tmp, "ledger.jsonl")
            rec, out_json = run_eval(MODEL, "smoke2", ctx=2048, limit_chunks=2,
                                     device=device, dtype="bfloat16",
                                     out_dir=tmp, ledger_path=ledger)
            # canon protocol shape: same fields/derivations as a real 64-ch run
            self.assertEqual(os.path.basename(out_json), "ppl_qwen-05b_smoke2.json")
            self.assertEqual(rec["chunks"], 2)
            self.assertEqual(rec["ctx"], 2048)
            self.assertEqual(rec["tokens"], 2 * 2047)   # shift-labels per window
            self.assertEqual(rec["dtype"], "bfloat16")
            self.assertEqual(rec["harness_version"], HARNESS_VERSION)
            self.assertEqual(len(rec["harness_key8"]), 8)
            self.assertEqual(rec["harness_key"]["chunks"], 2)
            self.assertIn(rec["dataset_id"], ("wikitext", "Salesforce/wikitext"))
            self.assertEqual(len(rec["dataset_fp"]), 16)
            self.assertNotIn(rec["device"], ("auto", "offload"),
                             "record must carry the RESOLVED device")
            # sanity: a working 0.5B bf16 sits near the 12.55 canon, never NaN/collapse
            self.assertGreater(rec["ppl"], 1.0)
            self.assertLess(rec["ppl"], 100.0)
            # written json == returned record
            with open(out_json) as f:
                disk = json.load(f)
            self.assertEqual(disk["ppl"], rec["ppl"])
            self.assertEqual(disk["harness_key8"], rec["harness_key8"])
            # ledger got exactly one line and checks clean
            self.assertEqual(len(read_ledger(ledger)), 1)
            errors, _ = check(ledger)
            self.assertEqual(errors, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
