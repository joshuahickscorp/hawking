#!/usr/bin/env python3.12
"""Adversarial unit tests for tools/verify/case_extract.py (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "tools" / "verify" / "case_extract.py"


def run_extract(cwd: Path, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(EXTRACT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def init_mini_repo(tmp: Path) -> str:
    """Tiny git repo exercising extractor branches; returns HEAD sha."""
    subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "f1@test.local"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "f1"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    (tmp / "crates" / "demo" / "src").mkdir(parents=True)
    (tmp / "crates" / "demo" / "tests").mkdir(parents=True)
    (tmp / "tools" / "demo" / "tests").mkdir(parents=True)
    (tmp / "app" / "src").mkdir(parents=True)

    (tmp / "crates" / "demo" / "src" / "lib.rs").write_text(
        textwrap.dedent(
            """
            pub fn f() {}

            #[cfg(test)]
            mod tests {
                #[test]
                fn unit_one() { assert_eq!(1, 1); }

                #[tokio::test]
                async fn unit_async() { assert!(true); }
            }
            """
        ),
        encoding="utf-8",
    )
    (tmp / "crates" / "demo" / "tests" / "int.rs").write_text(
        textwrap.dedent(
            """
            #[test]
            fn integration_one() { assert!(true); }
            """
        ),
        encoding="utf-8",
    )
    (tmp / "tools" / "demo" / "tests" / "test_params.py").write_text(
        textwrap.dedent(
            '''
            import pytest

            @pytest.mark.parametrize("a,b", [(1, 2), (3, 4)])
            def test_multi(a, b):
                assert a < b

            @pytest.mark.parametrize("x", [10, 20])
            @pytest.mark.parametrize("y", ["p", "q"])
            def test_stacked(x, y):
                assert x and y

            @pytest.mark.parametrize("z", list(range(3)))
            def test_dynamic(z):
                assert z >= 0

            def test_plain():
                assert True
            '''
        ),
        encoding="utf-8",
    )
    (tmp / "app" / "src" / "widget.test.ts").write_text(
        textwrap.dedent(
            """
            describe("group", () => {
              it("literal title", () => { expect(1).toBe(1); });
              test("another literal", () => { expect(2).toBe(2); });
              it(dynamicTitle(), () => {});
            });
            """
        ),
        encoding="utf-8",
    )
    for name, doc in {
        "REBUILD_BEHAVIOUR_CONSTITUTION.json": {
            "schema": "t",
            "behaviours": [
                {"id": "BC-TEST-001", "verification_status": "documented_only"}
            ],
        },
        "REBUILD_BLACKBOX_TEST_MATRIX.json": {
            "schema": "t",
            "checks": [
                {
                    "behaviour_id": "BC-TEST-001",
                    "runnable_now": True,
                    "command": "true",
                    "assertion": "exit0",
                    "blocker": None,
                }
            ],
        },
        "REBUILD_DATA_MIGRATION_CONTRACT.json": {
            "schema": "t",
            "items": [{"id": "MIG-001", "name": "x", "sample_exists": True}],
        },
        "REBUILD_PERFORMANCE_BASELINE_MEASURED.json": {
            "schema": "t",
            "metrics": [
                {
                    "name": "base_tps.demo",
                    "status": "unavailable",
                    "family": "base_tps",
                }
            ],
        },
    }.items():
        (tmp / name).write_text(json.dumps(doc), encoding="utf-8")

    subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "mini"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    return _git(tmp, "rev-parse", "HEAD").strip()


class CaseExtractHelpers(unittest.TestCase):
    """Import-level tests against helper functions."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "tools" / "verify"))
        import case_extract as ce  # type: ignore

        cls.ce = ce

    def test_multi_arg_and_stacked_expand(self) -> None:
        ce = self.ce
        warnings: list[str] = []
        src = textwrap.dedent(
            """
            import pytest
            @pytest.mark.parametrize("a,b", [(1, 2), (3, 4)])
            def test_multi(a, b):
                pass

            @pytest.mark.parametrize("x", [10, 20])
            @pytest.mark.parametrize("y", ["p", "q"])
            def test_stacked(x, y):
                pass
            """
        )
        import ast

        tree = ast.parse(src)
        multi = tree.body[1]
        stacked = tree.body[2]
        rows_m = ce._expand_param_rows(
            list(multi.decorator_list), "t.py", "test_multi", warnings
        )
        rows_s = ce._expand_param_rows(
            list(stacked.decorator_list), "t.py", "test_stacked", warnings
        )
        self.assertEqual(len(rows_m or []), 2)
        self.assertEqual(rows_m[0]["a"], 1)
        self.assertEqual(rows_m[0]["b"], 2)
        self.assertEqual(len(rows_s or []), 4)
        self.assertEqual(len(warnings), 0)

    def test_dynamic_params_warn_single_case(self) -> None:
        ce = self.ce
        warnings: list[str] = []
        import ast

        src = textwrap.dedent(
            """
            import pytest
            @pytest.mark.parametrize("z", list(range(3)))
            def test_dynamic(z):
                pass
            """
        )
        tree = ast.parse(src)
        fn = tree.body[1]
        rows = ce._expand_param_rows(
            list(fn.decorator_list), "t.py", "test_dynamic", warnings
        )
        self.assertIsNone(rows)
        self.assertTrue(any("py_parametrize_unparsed" in w for w in warnings))

    def test_title_slug_and_param_suffix_stable(self) -> None:
        ce = self.ce
        self.assertEqual(ce.title_slug("Hello, World!"), "hello_world")
        p = {"b": 2, "a": 1}
        s1 = ce.param_suffix(p)
        s2 = ce.param_suffix({"a": 1, "b": 2})
        self.assertEqual(s1, s2)
        self.assertTrue(s1.startswith("#a="))

    def test_duplicate_case_id_fails(self) -> None:
        ce = self.ce
        e = ce.make_entry(
            kind="py_fn",
            source_path="tools/x/test_a.py",
            symbol="test_one",
            fingerprint_material="a",
        )
        orig_rust = ce.extract_rust
        orig_py = ce.extract_python
        orig_ts = ce.extract_typescript
        orig_bb = ce.extract_bb
        orig_mig = ce.extract_mig
        orig_perf = ce.extract_perf
        try:
            ce.extract_rust = lambda rev, w: [e, dict(e)]  # type: ignore
            ce.extract_python = lambda rev, w: []  # type: ignore
            ce.extract_typescript = lambda rev, w: []  # type: ignore
            ce.extract_bb = lambda rev, w: []  # type: ignore
            ce.extract_mig = lambda rev, w: []  # type: ignore
            ce.extract_perf = lambda rev, w: []  # type: ignore
            with self.assertRaises(SystemExit) as cm:
                ce.build_ledger()
            self.assertIn("collision", str(cm.exception).lower())
        finally:
            ce.extract_rust = orig_rust  # type: ignore
            ce.extract_python = orig_py  # type: ignore
            ce.extract_typescript = orig_ts  # type: ignore
            ce.extract_bb = orig_bb  # type: ignore
            ce.extract_mig = orig_mig  # type: ignore
            ce.extract_perf = orig_perf  # type: ignore

    def test_rust_body_mutation_changes_fingerprint(self) -> None:
        ce = self.ce
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_mini_repo(tmp)
            env = {"HAWKING_CASE_EXTRACT_ROOT": str(tmp)}
            # Patch extractors to only rust from this root via env.
            os.environ["HAWKING_CASE_EXTRACT_ROOT"] = str(tmp)
            try:
                # Reload-free: call extract_rust with resolved rev.
                rev = ce.resolve_rev("HEAD")
                w: list[str] = []
                before = {e["symbol"]: e["content_fingerprint"] for e in ce.extract_rust(rev, w)}
                self.assertIn("unit_one", before)
                lib = tmp / "crates" / "demo" / "src" / "lib.rs"
                text = lib.read_text(encoding="utf-8")
                lib.write_text(
                    text.replace(
                        "fn unit_one() { assert_eq!(1, 1); }",
                        "fn unit_one() { assert_eq!(1, 1); assert!(true); }",
                    ),
                    encoding="utf-8",
                )
                subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", "body mut"],
                    cwd=tmp,
                    check=True,
                    capture_output=True,
                )
                rev2 = ce.resolve_rev("HEAD")
                after = {
                    e["symbol"]: e["content_fingerprint"]
                    for e in ce.extract_rust(rev2, [])
                }
                self.assertNotEqual(before["unit_one"], after["unit_one"])
                # Name-stable sibling should match if body unchanged.
                self.assertEqual(before["unit_async"], after["unit_async"])
            finally:
                os.environ.pop("HAWKING_CASE_EXTRACT_ROOT", None)

    def test_vitest_expect_mutation_changes_fingerprint(self) -> None:
        ce = self.ce
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_mini_repo(tmp)
            os.environ["HAWKING_CASE_EXTRACT_ROOT"] = str(tmp)
            try:
                rev = ce.resolve_rev("HEAD")
                before = {
                    e["symbol"]: e["content_fingerprint"]
                    for e in ce.extract_typescript(rev, [])
                }
                self.assertIn("literal title", before)
                path = tmp / "app" / "src" / "widget.test.ts"
                path.write_text(
                    path.read_text(encoding="utf-8").replace(
                        "expect(1).toBe(1)", "expect(1).toBe(2)"
                    ),
                    encoding="utf-8",
                )
                subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", "expect mut"],
                    cwd=tmp,
                    check=True,
                    capture_output=True,
                )
                rev2 = ce.resolve_rev("HEAD")
                after = {
                    e["symbol"]: e["content_fingerprint"]
                    for e in ce.extract_typescript(rev2, [])
                }
                self.assertNotEqual(before["literal title"], after["literal title"])
                self.assertEqual(before["another literal"], after["another literal"])
            finally:
                os.environ.pop("HAWKING_CASE_EXTRACT_ROOT", None)

    def test_vitest_line_insert_preserves_ids(self) -> None:
        ce = self.ce
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_mini_repo(tmp)
            # Two describes, same title — must disambiguate by chain, not line.
            (tmp / "app" / "src" / "dup.test.ts").write_text(
                textwrap.dedent(
                    """
                    describe("alpha", () => {
                      it("shared title", () => { expect(1).toBe(1); });
                    });
                    describe("beta", () => {
                      it("shared title", () => { expect(2).toBe(2); });
                    });
                    """
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "dup titles"],
                cwd=tmp,
                check=True,
                capture_output=True,
            )
            os.environ["HAWKING_CASE_EXTRACT_ROOT"] = str(tmp)
            try:
                rev1 = ce.resolve_rev("HEAD")
                ids1 = sorted(
                    e["case_id"]
                    for e in ce.extract_typescript(rev1, [])
                    if "dup_test" in e["case_id"] or "shared_title" in e["case_id"]
                )
                self.assertEqual(len(ids1), 2)
                self.assertTrue(any("alpha" in i for i in ids1))
                self.assertTrue(any("beta" in i for i in ids1))
                self.assertFalse(any("#L" in i for i in ids1))

                path = tmp / "app" / "src" / "dup.test.ts"
                path.write_text(
                    "// inserted comment line does not shift durable ids\n"
                    + path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", "insert"],
                    cwd=tmp,
                    check=True,
                    capture_output=True,
                )
                rev2 = ce.resolve_rev("HEAD")
                ids2 = sorted(
                    e["case_id"]
                    for e in ce.extract_typescript(rev2, [])
                    if "shared_title" in e["case_id"]
                )
                self.assertEqual(ids1, ids2)
            finally:
                os.environ.pop("HAWKING_CASE_EXTRACT_ROOT", None)

    def test_vitest_comment_and_string_fake_calls_ignored(self) -> None:
        ce = self.ce
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            init_mini_repo(tmp)
            (tmp / "app" / "src" / "fake.test.ts").write_text(
                textwrap.dedent(
                    """
                    // it("comment fake", () => {});
                    /* it("block fake", () => {}); */
                    const s = 'it("string fake", () => {})';
                    const t = `test("template fake", () => {})`;
                    describe("real", () => {
                      it("only real", () => { expect(true).toBe(true); });
                    });
                    """
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-A"], cwd=tmp, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "fakes"],
                cwd=tmp,
                check=True,
                capture_output=True,
            )
            os.environ["HAWKING_CASE_EXTRACT_ROOT"] = str(tmp)
            try:
                rev = ce.resolve_rev("HEAD")
                entries = [
                    e
                    for e in ce.extract_typescript(rev, [])
                    if e["source_path"].endswith("fake.test.ts")
                ]
                symbols = [e["symbol"] for e in entries]
                self.assertEqual(symbols, ["only real"])
            finally:
                os.environ.pop("HAWKING_CASE_EXTRACT_ROOT", None)

    def test_sealed_check_survives_unrelated_later_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seal_rev = init_mini_repo(tmp)
            env = {
                "HAWKING_CASE_EXTRACT_ROOT": str(tmp),
                "PYTHONNOUSERSITE": "1",
            }
            ledger = tmp / "LEDGER.json"
            w = run_extract(tmp, "--write", str(ledger), env_extra=env)
            self.assertEqual(w.returncode, 0, w.stderr + w.stdout)
            chk1 = run_extract(tmp, "--check", str(ledger), env_extra=env)
            self.assertEqual(chk1.returncode, 0, chk1.stderr + chk1.stdout)

            # Unrelated later commit (new file; does not alter sealed tree).
            (tmp / "UNRELATED.md").write_text("later\n", encoding="utf-8")
            subprocess.run(["git", "add", "UNRELATED.md"], cwd=tmp, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "unrelated"],
                cwd=tmp,
                check=True,
                capture_output=True,
            )
            head = _git(tmp, "rev-parse", "HEAD").strip()
            self.assertNotEqual(head, seal_rev)

            chk2 = run_extract(tmp, "--check", str(ledger), env_extra=env)
            self.assertEqual(chk2.returncode, 0, chk2.stderr + chk2.stdout)
            self.assertIn("PASS", chk2.stdout)

            # Mutating the sealed ledger pin must fail.
            doc = json.loads(ledger.read_text(encoding="utf-8"))
            doc["sealed_at_commit"] = head
            ledger.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            # Pin now points at later commit whose tree differs (no change to tests,
            # but sealed pin rewrite means regenerate at head — should still match
            # for test content). Force a real fail by also flipping a fingerprint.
            doc = json.loads(ledger.read_text(encoding="utf-8"))
            doc["entries"][0]["content_fingerprint"] = "0" * 64
            ledger.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            chk3 = run_extract(tmp, "--check", str(ledger), env_extra=env)
            self.assertNotEqual(chk3.returncode, 0)


class CaseExtractRepo(unittest.TestCase):
    """Determinism and --check against the real sealed worktree."""

    def test_byte_deterministic_twice(self) -> None:
        a = run_extract(ROOT, "--json")
        b = run_extract(ROOT, "--json")
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertEqual(a.stdout, b.stdout)
        da, db = json.loads(a.stdout), json.loads(b.stdout)
        self.assertEqual(da["counts"]["total"], db["counts"]["total"])
        ha = hashlib.sha256(a.stdout.encode()).hexdigest()
        hb = hashlib.sha256(b.stdout.encode()).hexdigest()
        self.assertEqual(ha, hb)

    def test_vitest_describe_not_counted(self) -> None:
        r = run_extract(ROOT, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        kinds = set(doc["counts"]["by_kind"])
        self.assertNotIn("describe", kinds)
        for e in doc["entries"]:
            if e["kind"] == "ts_it":
                self.assertFalse(e["case_id"].startswith("CASE.describe."))
                self.assertNotIn("#L", e["case_id"])

    def test_ledger_check_detects_fingerprint_mutation(self) -> None:
        r = run_extract(ROOT, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertGreater(len(doc["entries"]), 100)
        doc["entries"][0]["content_fingerprint"] = "0" * 64
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.json"
            path.write_text(
                json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            chk = run_extract(ROOT, "--check", str(path))
            self.assertNotEqual(chk.returncode, 0)
            self.assertIn("FAIL", chk.stderr + chk.stdout)


if __name__ == "__main__":
    unittest.main()
