# Canonical strand_eval tests: naming, ledger, self-location, and gated smoke.
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

PKG_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(PKG_DIR))

from strand_eval import HARNESS_VERSION, locate_repo_root
from strand_eval.core import (build_record, harness_key, model_id_from_dir,
                              output_path, sanitize)
from strand_eval.ledger import append_record, check, ingest, read_ledger

REPO = locate_repo_root()
CLI_SH = os.path.join(REPO, "tools", "strand", "scripts", "strand-eval")
PY = sys.executable


def run_command(command, cwd=None, env=None):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        command, cwd=cwd, env=merged, capture_output=True, text=True, check=False
    )


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


class TestSelfLocation(unittest.TestCase):
    def test_invoked_from_root_cwd(self):
        result = run_command([CLI_SH, "where"], cwd="/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"repo_root  : {os.path.realpath(REPO)}", result.stdout)

    def test_module_cli_from_root_cwd(self):
        result = run_command([PY, os.path.join(PKG_DIR, "cli.py"), "where"], cwd="/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("repo_root", result.stdout)

    def test_via_symlink(self):
        with tempfile.TemporaryDirectory(prefix="se_sym_") as tmp:
            link = os.path.join(tmp, "strand-eval-link")
            os.symlink(CLI_SH, link)
            result = run_command([link, "where"], cwd="/")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(os.path.realpath(REPO), result.stdout)

    def test_copied_package_with_strand_root(self):
        with tempfile.TemporaryDirectory(prefix="se_copy_") as tmp:
            copied = os.path.join(tmp, "strand_eval")
            shutil.copytree(
                PKG_DIR,
                copied,
                ignore=shutil.ignore_patterns("test_*.py", "__pycache__"),
            )
            result = run_command(
                [PY, os.path.join(copied, "cli.py"), "where"],
                cwd="/",
                env={"STRAND_ROOT": REPO},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(os.path.realpath(REPO), result.stdout)

    def test_copied_package_without_root_dies_loudly(self):
        with tempfile.TemporaryDirectory(prefix="se_lost_") as tmp:
            copied = os.path.join(tmp, "strand_eval")
            shutil.copytree(
                PKG_DIR,
                copied,
                ignore=shutil.ignore_patterns("test_*.py", "__pycache__"),
            )
            result = run_command(
                [PY, os.path.join(copied, "cli.py"), "where"],
                cwd="/",
                env={"STRAND_ROOT": ""},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("STRAND_ROOT", result.stdout + result.stderr)

    def test_shell_wrapper_copied_offrepo_dies_loudly(self):
        with tempfile.TemporaryDirectory(prefix="se_ship_") as tmp:
            copied = os.path.join(tmp, "strand-eval")
            shutil.copy2(CLI_SH, copied)
            result = run_command([copied, "where"], cwd="/", env={"STRAND_ROOT": ""})
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("not the strand repo root", result.stderr)

    def test_ledger_check_cwd_independent(self):
        result = run_command([CLI_SH, "ledger", "check"], cwd="/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record(s)", result.stdout)


@unittest.skipIf(
    os.environ.get("STRAND_EVAL_SMOKE") != "1",
    "set STRAND_EVAL_SMOKE=1 to run the 0.5B smoke",
)
@unittest.skipIf(
    not os.path.isdir(os.path.join(REPO, "scratch", "qwen-05b")),
    "scratch/qwen-05b not present",
)
class TestSmoke05B(unittest.TestCase):
    def test_two_chunk_canon_shape(self):
        from strand_eval.core import run_eval

        model = os.path.join(REPO, "scratch", "qwen-05b")
        device = os.environ.get("STRAND_EVAL_SMOKE_DEVICE", "auto")
        tmp = tempfile.mkdtemp(prefix="se_smoke_")
        try:
            ledger = os.path.join(tmp, "ledger.jsonl")
            rec, out_json = run_eval(
                model,
                "smoke2",
                ctx=2048,
                limit_chunks=2,
                device=device,
                dtype="bfloat16",
                out_dir=tmp,
                ledger_path=ledger,
            )
            self.assertEqual(os.path.basename(out_json), "ppl_qwen-05b_smoke2.json")
            self.assertEqual(rec["chunks"], 2)
            self.assertEqual(rec["ctx"], 2048)
            self.assertEqual(rec["tokens"], 2 * 2047)
            self.assertEqual(rec["dtype"], "bfloat16")
            self.assertEqual(rec["harness_version"], HARNESS_VERSION)
            self.assertEqual(len(rec["harness_key8"]), 8)
            self.assertEqual(rec["harness_key"]["chunks"], 2)
            self.assertIn(rec["dataset_id"], ("wikitext", "Salesforce/wikitext"))
            self.assertEqual(len(rec["dataset_fp"]), 16)
            self.assertNotIn(rec["device"], ("auto", "offload"))
            self.assertGreater(rec["ppl"], 1.0)
            self.assertLess(rec["ppl"], 100.0)
            with open(out_json) as handle:
                disk = json.load(handle)
            self.assertEqual(disk["ppl"], rec["ppl"])
            self.assertEqual(disk["harness_key8"], rec["harness_key8"])
            self.assertEqual(len(read_ledger(ledger)), 1)
            errors, _ = check(ledger)
            self.assertEqual(errors, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
