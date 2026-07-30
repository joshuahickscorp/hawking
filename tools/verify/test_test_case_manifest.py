#!/usr/bin/env python3.12
"""Adversarial unit tests for tools/verify/test_case_manifest.py (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "verify" / "test_case_manifest.py"


def run_tool(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mini_ledger(ids: list[str]) -> dict:
    entries = []
    for cid in ids:
        entries.append(
            {
                "case_id": cid,
                "kind": "py_fn",
                "source_path": "t.py",
                "symbol": cid.split("::")[-1] if "::" in cid else cid,
                "param": None,
                "seed": None,
                "bc": None,
                "content_fingerprint": "a" * 64,
                "status": None,
                "notes": None,
            }
        )
    return {
        "schema": "hawking.assertion_ledger.v1",
        "extractor_version": "hawking.case_extract.v2",
        "sealed_at_commit": "deadbeef",
        "authority": "controller",
        "source_tools": {},
        "counts": {"total": len(entries), "by_kind": {"py_fn": len(entries)}},
        "warnings": [],
        "entries": entries,
    }


def _hash_ledger(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestAntiGaming(unittest.TestCase):
    def test_empty_manifest_seal_check_pass_gate_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "ASSERTION_LEDGER.json"
            man_path = td_path / "TEST_CASE_MANIFEST.json"
            ids = [f"CASE.py_fn.t::test_{i}" for i in range(5)]
            write(ledger_path, mini_ledger(ids))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f1_scaffold",
                    "status": "no_rewrite_accounting",
                    "ledger_ref": {
                        "path": "control/ASSERTION_LEDGER.json",
                        "sha256": h,
                    },
                    "entries": [],
                },
            )
            seal = run_tool("--seal-check", str(ledger_path), str(man_path))
            self.assertEqual(seal.returncode, 0, seal.stdout + seal.stderr)
            self.assertIn("PASS", seal.stdout)

            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
            )
            self.assertNotEqual(gate.returncode, 0, gate.stdout)
            self.assertIn("unaccounted", gate.stdout.lower() + gate.stderr.lower())
            for cid in ids:
                self.assertIn(cid, gate.stdout)

    def test_n_replacements_without_mapping_fail(self) -> None:
        """N replaces + receipt with only one row and no identity map must fail."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            old = [f"CASE.py_fn.t::old_{i}" for i in range(3)]
            write(ledger_path, mini_ledger(old))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::new_table",
                            "disposition": "rewrite",
                            "replaces": old,
                            "executor": {"runtime": "pytest", "command": "true"},
                            "expected_status": "pass",
                        }
                    ],
                },
            )
            # Only the new id in results — no N-entry identity map.
            receipt = {
                "schema": "hawking.test_case_execution_receipt.v1",
                "results": [
                    {"case_id": "CASE.py_fn.t::new_table", "status": "pass"},
                ],
            }
            rpath = td_path / "receipt.json"
            write(rpath, receipt)
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--receipt",
                str(rpath),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0, gate.stdout)
            payload = json.loads(gate.stdout)
            err_text = " ".join(payload.get("errors") or [])
            self.assertTrue(
                "identity map" in err_text.lower() or "receipt" in err_text.lower(),
                err_text,
            )

    def test_complete_one_to_one_receipt_map_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            old = [f"CASE.py_fn.t::old_{i}" for i in range(3)]
            write(ledger_path, mini_ledger(old))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::new_table",
                            "disposition": "rewrite",
                            "replaces": old,
                            "receipt_map": {oid: f"row-{i}" for i, oid in enumerate(old)},
                            "executor": {"runtime": "pytest", "command": "true"},
                            "expected_status": "pass",
                        }
                    ],
                },
            )
            receipt = {
                "schema": "hawking.test_case_execution_receipt.v1",
                "results": [
                    {"case_id": "CASE.py_fn.t::new_table", "status": "pass"},
                ],
            }
            rpath = td_path / "receipt.json"
            write(rpath, receipt)
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--receipt",
                str(rpath),
                "--json",
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            payload = json.loads(gate.stdout)
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["unaccounted_count"], 0)

    def test_unknown_manifest_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            ids = ["CASE.py_fn.t::a"]
            write(ledger_path, mini_ledger(ids))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::a",
                            "disposition": "execute",
                            "executor": {"command": "true"},
                        },
                        {
                            "case_id": "CASE.py_fn.t::ghost",
                            "disposition": "execute",
                            "executor": {"command": "true"},
                        },
                    ],
                },
            )
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0)
            payload = json.loads(gate.stdout)
            err = " ".join(payload.get("errors") or [])
            self.assertIn("unknown", err.lower())

    def test_overlapping_replacement_ownership_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            old = "CASE.py_fn.t::shared"
            write(ledger_path, mini_ledger([old]))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::new_a",
                            "disposition": "rewrite",
                            "replaces": [old],
                        },
                        {
                            "case_id": "CASE.py_fn.t::new_b",
                            "disposition": "rewrite",
                            "replaces": [old],
                        },
                    ],
                },
            )
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0)
            payload = json.loads(gate.stdout)
            err = " ".join(payload.get("errors") or [])
            self.assertIn("multiple accounting owners", err)

    def test_duplicate_and_circular_supersession_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            ids = ["CASE.py_fn.t::a", "CASE.py_fn.t::b"]
            write(ledger_path, mini_ledger(ids))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::a",
                            "disposition": "superseded",
                            "superseded_by": "CASE.py_fn.t::b",
                        },
                        {
                            "case_id": "CASE.py_fn.t::b",
                            "disposition": "superseded",
                            "superseded_by": "CASE.py_fn.t::a",
                        },
                        {
                            "case_id": "CASE.py_fn.t::a",
                            "disposition": "execute",
                            "executor": {"command": "true"},
                        },
                    ],
                },
            )
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0)
            out = gate.stdout + gate.stderr
            self.assertTrue(
                "duplicate" in out.lower() or "circular" in out.lower(),
                out,
            )

    def test_blocked_cannot_be_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            cid = "CASE.bb.BC-X"
            write(ledger_path, mini_ledger([cid]))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": cid,
                            "disposition": "blocked_model",
                            "expected_status": "pass",
                        }
                    ],
                },
            )
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0)
            self.assertIn("blocked", gate.stdout.lower())

            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": cid,
                            "disposition": "blocked_fixture",
                            "expected_status": "unavailable_env",
                        }
                    ],
                },
            )
            rpath = td_path / "r.json"
            write(
                rpath,
                {
                    "schema": "hawking.test_case_execution_receipt.v1",
                    "results": [{"case_id": cid, "status": "pass"}],
                },
            )
            gate2 = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--receipt",
                str(rpath),
                "--json",
            )
            self.assertNotEqual(gate2.returncode, 0)
            self.assertIn("pass", gate2.stdout.lower())

    def test_rewrite_without_replaces_fails_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            write(ledger_path, mini_ledger(["CASE.py_fn.t::x"]))
            h = _hash_ledger(ledger_path)
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f2",
                    "status": "accounting",
                    "ledger_ref": {"path": "control/ASSERTION_LEDGER.json", "sha256": h},
                    "entries": [
                        {
                            "case_id": "CASE.py_fn.t::x",
                            "disposition": "rewrite",
                            "replaces": [],
                        }
                    ],
                },
            )
            gate = run_tool(
                "--gate",
                "--before",
                str(ledger_path),
                "--after",
                str(man_path),
                "--json",
            )
            self.assertNotEqual(gate.returncode, 0)
            self.assertIn("replaces", gate.stdout.lower())

    def test_forged_ledger_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "L.json"
            man_path = td_path / "M.json"
            write(ledger_path, mini_ledger(["CASE.py_fn.t::a"]))
            write(
                man_path,
                {
                    "schema": "hawking.test_case_manifest.v1",
                    "version": 1,
                    "phase": "f1_scaffold",
                    "status": "no_rewrite_accounting",
                    "ledger_ref": {
                        "path": "control/ASSERTION_LEDGER.json",
                        "sha256": "0" * 64,
                    },
                    "entries": [],
                },
            )
            seal = run_tool("--seal-check", str(ledger_path), str(man_path))
            self.assertNotEqual(seal.returncode, 0)
            self.assertIn("sha256", seal.stdout.lower() + seal.stderr.lower())


if __name__ == "__main__":
    unittest.main()
