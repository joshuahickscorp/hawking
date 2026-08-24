from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.grok_bridge import (  # noqa: E402
    GrokBridge,
    GrokContractError,
    GrokNotAvailable,
    GrokRunError,
    GrokRunHandle,
    NO_MUTATION_LOCK_WARNING,
    extract_task_id,
    find_grok_run,
    parse_grok_status,
    validate_contract_text,
)

VALID_CONTRACT = """# WRITE
create tools/haider/hcli/grok_bridge.py

# VERIFY
`pytest tools/haider/hcli/tests -q` must exit 0

# ACCEPTANCE
pytest exits 0
"""

FAKE_BIN = "/fake/grok-run"


def _completed(argv, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        list(argv), returncode, stdout=stdout, stderr=stderr
    )


def _dry_stdout(task_id="slug-20260101-000000", cwd="/tmp/ws", extra=""):
    task_dir = f"/tmp/fake-tasks/{task_id}"
    return (
        "DRY RUN — would execute:\n"
        f"grok --prompt-file {task_dir}/task.md --cwd {cwd} "
        "-m grok-4.6 --effort xhigh --sandbox grokdev\n"
        f"{extra}"
        f"task dir: {task_dir}\n"
    )


class TestParseGrokStatus(unittest.TestCase):
    def test_running(self):
        got = parse_grok_status("status: running (exit -)")
        self.assertEqual(got, {"state": "running", "exit_code": None})

    def test_done_zero(self):
        got = parse_grok_status("status: done (exit 0)")
        self.assertEqual(got, {"state": "done", "exit_code": 0})

    def test_done_nonzero_is_failed(self):
        got = parse_grok_status("status: done (exit 1)")
        self.assertEqual(got, {"state": "failed", "exit_code": 1})

    def test_garbage_is_unknown_not_raise(self):
        got = parse_grok_status("this is not status output at all")
        self.assertEqual(got["state"], "unknown")
        self.assertIsNone(got["exit_code"])

    def test_empty_and_none(self):
        self.assertEqual(parse_grok_status(""), {"state": "unknown", "exit_code": None})
        self.assertEqual(parse_grok_status(None), {"state": "unknown", "exit_code": None})

    def test_embedded_in_full_status_dump(self):
        blob = '{\n  "task_id": "x"\n}\nstatus: running (exit -)\ntotal 40\n'
        self.assertEqual(parse_grok_status(blob)["state"], "running")


class TestValidateContract(unittest.TestCase):
    def test_empty_refused(self):
        with self.assertRaises(GrokContractError) as ctx:
            validate_contract_text("")
        self.assertIn("will not invent a contract", str(ctx.exception))

    def test_none_refused(self):
        with self.assertRaises(GrokContractError):
            validate_contract_text(None)

    def test_missing_verify(self):
        with self.assertRaises(GrokContractError) as ctx:
            validate_contract_text("# WRITE\ncreate foo.py\n")
        self.assertIn("VERIFY", str(ctx.exception))

    def test_missing_write(self):
        with self.assertRaises(GrokContractError) as ctx:
            validate_contract_text("# VERIFY\n`pytest` must exit 0\n")
        self.assertIn("WRITE", str(ctx.exception))

    def test_valid(self):
        self.assertIn("WRITE", validate_contract_text(VALID_CONTRACT))


class TestExtractTaskId(unittest.TestCase):
    def test_from_task_dir_line(self):
        self.assertEqual(
            extract_task_id("task dir: /tmp/x/slug-20260101-000000\n"),
            "slug-20260101-000000",
        )

    def test_from_bare_id(self):
        self.assertEqual(
            extract_task_id("slug-20260101-000000\n"),
            "slug-20260101-000000",
        )

    def test_refuses_to_invent(self):
        with self.assertRaises(GrokRunError) as ctx:
            extract_task_id("DRY RUN — would execute:\ngrok --cwd /tmp\n")
        self.assertIn("refusing to invent", str(ctx.exception))


class _BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.bridge = GrokBridge(self.root)
        self.which_patch = patch(
            "hcli.grok_bridge.shutil.which",
            return_value=FAKE_BIN,
        )
        self.which_patch.start()
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("GROK_RUN", None)

    def tearDown(self):
        self._env.stop()
        self.which_patch.stop()
        self.tmpdir.cleanup()

    def _patch_run(self, stdout="", stderr="", returncode=0, collector=None):
        def fake_run(argv, **kwargs):
            if collector is not None:
                collector.append((list(argv), kwargs))
            return _completed(argv, stdout=stdout, stderr=stderr, returncode=returncode)

        return patch(
            "hcli.grok_bridge.subprocess.run",
            side_effect=fake_run,
        )


class TestMissingContractDoesNotSpawn(_BridgeTest):
    def test_empty_contract_does_not_call_subprocess(self):
        spawned = []

        def boom(*a, **k):
            spawned.append((a, k))
            raise AssertionError("subprocess spawned")

        with patch("hcli.grok_bridge.subprocess.run", side_effect=boom):
            with patch(
                "hcli.grok_bridge.subprocess.Popen", side_effect=boom
            ):
                with self.assertRaises(GrokContractError):
                    self.bridge.delegate("t", "")
                with self.assertRaises(GrokContractError):
                    self.bridge.audit("t", None)  # type: ignore[arg-type]
        self.assertEqual(spawned, [])

    def test_whitespace_contract_does_not_spawn(self):
        spawned = []

        def boom(*a, **k):
            spawned.append((a, k))
            raise AssertionError("subprocess spawned")

        with patch("hcli.grok_bridge.subprocess.run", side_effect=boom):
            with patch(
                "hcli.grok_bridge.subprocess.Popen", side_effect=boom
            ):
                with self.assertRaises(GrokContractError):
                    self.bridge.delegate("t", "   \n")
        self.assertEqual(spawned, [])


class TestGrokNotAvailable(_BridgeTest):
    # find_grok_run falls back to the canonical install location after PATH, so
    # "absent" now means not on PATH AND not installed. Stubbing shutil.which
    # alone no longer simulates absence on any machine where grok-run is
    # actually installed, which is every machine that would run these tests.
    ABSENT_HINT = Path("/nonexistent/grok-run-definitely-absent")

    def test_which_none_raises_specific_error(self):
        self.which_patch.stop()
        with patch("hcli.grok_bridge.shutil.which", return_value=None), patch(
            "hcli.grok_bridge.DEFAULT_GROK_RUN_HINT", self.ABSENT_HINT
        ):
            with self.assertRaises(GrokNotAvailable) as ctx:
                self.bridge.delegate("t", VALID_CONTRACT, dry_run=True)
        self.assertIn("not on PATH", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, GrokRunError)

    def test_find_grok_run_none(self):
        self.which_patch.stop()
        with patch("hcli.grok_bridge.shutil.which", return_value=None), patch(
            "hcli.grok_bridge.DEFAULT_GROK_RUN_HINT", self.ABSENT_HINT
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GROK_RUN", None)
                with self.assertRaises(GrokNotAvailable):
                    find_grok_run()

    def test_installed_binary_is_found_without_being_on_path(self):
        """PATH is not the only authority: the documented install path counts."""
        self.which_patch.stop()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "grok-run"
            fake.write_text("#!/bin/sh\nexit 0\n")
            fake.chmod(0o755)
            with patch("hcli.grok_bridge.shutil.which", return_value=None), patch(
                "hcli.grok_bridge.DEFAULT_GROK_RUN_HINT", fake
            ):
                os.environ.pop("GROK_RUN", None)
                self.assertEqual(find_grok_run(), str(fake))


class TestArgvConstruction(_BridgeTest):
    def test_delegate_flags(self):
        calls = []
        stdout = _dry_stdout("hcli-unit-20260101-000000", cwd=str(self.root))
        with self._patch_run(stdout=stdout, collector=calls):
            handle = self.bridge.delegate(
                "hcli-unit",
                VALID_CONTRACT,
                profile="power",
                background=True,
                no_worktree=True,
                dry_run=True,
                mutation_lock=lambda: nullcontext(),
            )
        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(argv[0], FAKE_BIN)
        self.assertIn("delegate", argv)
        self.assertEqual(argv[argv.index("--task") + 1], "hcli-unit")
        contract = Path(argv[argv.index("--contract") + 1])
        self.assertTrue(contract.is_file(), contract)
        self.assertIn("WRITE", contract.read_text())
        self.assertEqual(argv[argv.index("--profile") + 1], "power")
        self.assertIn("--background", argv)
        self.assertIn("--no-worktree", argv)
        self.assertEqual(argv[argv.index("--repo") + 1], str(self.root))
        self.assertEqual(kwargs["env"]["GROK_DRYRUN"], "1")
        self.assertEqual(handle.task_id, "hcli-unit-20260101-000000")
        self.assertEqual(handle.command_run, argv)
        self.assertTrue(handle.dry_run)

    def test_audit_flags(self):
        calls = []
        stdout = _dry_stdout("hcli-aud-20260101-000000")
        with self._patch_run(stdout=stdout, collector=calls):
            handle = self.bridge.audit(
                "hcli-aud", VALID_CONTRACT, background=True, dry_run=True
            )
        argv = calls[0][0]
        self.assertIn("audit", argv)
        self.assertEqual(argv[argv.index("--task") + 1], "hcli-aud")
        self.assertTrue(Path(argv[argv.index("--contract") + 1]).is_file())
        self.assertIn("--background", argv)
        self.assertNotIn("--no-worktree", argv)
        self.assertNotIn("--profile", argv)
        self.assertEqual(handle.task_id, "hcli-aud-20260101-000000")

    def test_consult_flags(self):
        calls = []
        stdout = _dry_stdout("consult-20260101-000000")
        with self._patch_run(stdout=stdout, collector=calls):
            handle = self.bridge.consult("say ping", background=True, dry_run=True)
        argv = calls[0][0]
        self.assertIn("consult", argv)
        self.assertEqual(argv[argv.index("--prompt") + 1], "say ping")
        self.assertIn("--background", argv)
        self.assertEqual(handle.task_id, "consult-20260101-000000")


class TestReceipts(_BridgeTest):
    def test_delegate_and_audit_write_receipts(self):
        stdout = _dry_stdout("recv-20260101-000000", cwd=str(self.root))
        with self._patch_run(stdout=stdout):
            handle = self.bridge.delegate(
                "recv",
                VALID_CONTRACT,
                dry_run=True,
                mutation_lock=lambda: nullcontext(),
            )
        path = self.root / ".hcli" / "grok" / f"{handle.task_id}.json"
        self.assertTrue(path.is_file(), path)
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["task_id"], handle.task_id)
        self.assertEqual(receipt["command_run"], handle.command_run)
        self.assertIn("started_at", receipt["timestamps"])
        self.assertIn("status", receipt)
        self.assertIn("report_path", receipt)
        self.assertTrue(receipt["dry_run"])
        self.assertEqual(str(path), handle.receipt_path)

        stdout2 = _dry_stdout("audr-20260101-000000")
        with self._patch_run(stdout=stdout2):
            audit = self.bridge.audit("audr", VALID_CONTRACT, dry_run=True)
        apath = self.root / ".hcli" / "grok" / f"{audit.task_id}.json"
        self.assertTrue(apath.is_file(), apath)
        areceipt = json.loads(apath.read_text())
        self.assertEqual(areceipt["mode"], "audit")
        self.assertEqual(areceipt["command_run"], audit.command_run)


class TestReceiptCompletenessAndSuccess(_BridgeTest):
    """Directive fields must be observed, never fabricated. done+nonzero is not success."""

    def test_status_refreshes_receipt_completeness(self):
        task_id = "recv-20260101-000000"
        task_dir = self.root / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "grok-report.md").write_text("raw report body\n", encoding="utf-8")
        (task_dir / "telemetry.json").write_text(
            json.dumps({"task_id": task_id, "retries": 2, "result": "pass"}),
            encoding="utf-8",
        )
        (task_dir / "grok-stderr.log").write_text("", encoding="utf-8")
        launch_stdout = (
            "started in background (pid 4242)\n"
            f"task dir: {task_dir}\n"
        )

        def fake_run(argv, **kwargs):
            if len(argv) >= 2 and argv[1] == "status":
                return _completed(argv, stdout="status: done (exit 0)\n")
            return _completed(argv, stdout=launch_stdout)

        with patch("hcli.grok_bridge.subprocess.run", side_effect=fake_run):
            handle = self.bridge.consult("say ping", background=True, dry_run=False)
            launch_receipt = json.loads(Path(handle.receipt_path).read_text(encoding="utf-8"))
            self.assertEqual((launch_receipt.get("status") or {}).get("state"), "running")
            compact = self.bridge.receipts_dir / f"{handle.task_id}.compact.json"
            compact.write_text(json.dumps({"task_id": handle.task_id}), encoding="utf-8")
            got = self.bridge.status(handle.task_id)

        self.assertTrue(got.get("successful"))
        self.assertEqual(got.get("grok_state"), "done")
        receipt = json.loads(Path(handle.receipt_path).read_text(encoding="utf-8"))
        self.assertEqual(receipt["task_id"], handle.task_id)
        self.assertEqual(receipt["command_run"], handle.command_run)
        self.assertTrue(receipt.get("executable"))
        self.assertEqual(receipt["executable"], handle.command_run[0])
        self.assertIn("started_at", receipt["timestamps"])
        self.assertIn("finished_at", receipt["timestamps"])
        self.assertIn("launch_pid", receipt)
        self.assertEqual(receipt.get("grok_state"), "done")
        self.assertEqual((receipt.get("status") or {}).get("state"), "done")
        self.assertTrue((receipt.get("status") or {}).get("successful"))
        self.assertEqual(receipt.get("retries"), 2)
        self.assertNotIn("retries_reason", receipt)
        self.assertEqual(receipt.get("compact_path"), str(compact))
        self.assertTrue(receipt.get("report_path"))
        self.assertIn("verifier_command", receipt)
        self.assertIn("verifier_outcome", receipt)
        if receipt.get("verifier_command") is None:
            self.assertTrue(receipt.get("verifier_command_reason"))
        if receipt.get("verifier_outcome") is None:
            self.assertTrue(receipt.get("verifier_outcome_reason"))
        self.assertIn("throttle_evidence", receipt)
        if receipt.get("throttle_evidence") is None:
            self.assertTrue(receipt.get("throttle_evidence_reason"))

    def test_retries_null_when_telemetry_absent_never_fabricated_zero(self):
        task_id = "notele-20260101-000000"
        task_dir = self.root / "tasks" / task_id
        task_dir.mkdir(parents=True)
        launch_stdout = f"task dir: {task_dir}\n"

        def fake_run(argv, **kwargs):
            if len(argv) >= 2 and argv[1] == "status":
                return _completed(argv, stdout="status: done (exit 0)\n")
            return _completed(argv, stdout=launch_stdout)

        with patch("hcli.grok_bridge.subprocess.run", side_effect=fake_run):
            handle = self.bridge.consult("say ping", background=True, dry_run=False)
            self.bridge.status(handle.task_id)
        receipt = json.loads(Path(handle.receipt_path).read_text(encoding="utf-8"))
        self.assertIsNone(receipt.get("retries"))
        self.assertNotEqual(receipt.get("retries"), 0)
        self.assertTrue(receipt.get("retries_reason"))
        self.assertIn("telemetry", str(receipt.get("retries_reason")).lower())

    def test_done_exit_1_is_not_successful_consult_223811_shape(self):
        # On-disk consult-20260822-223811: status file "done", exit_code file "1".
        parsed = parse_grok_status("status: done (exit 1)")
        self.assertEqual(parsed["exit_code"], 1)
        with self._patch_run(stdout="status: done (exit 1)\n"):
            got = self.bridge.status("consult-20260822-223811")
        self.assertEqual(got["exit_code"], 1)
        self.assertIn("successful", got)
        self.assertFalse(got["successful"])
        self.assertFalse(got.get("successful"))
        # Raw grok-run state "done" with a nonzero exit is not success either.
        from hcli.grok_bridge import grok_succeeded

        self.assertFalse(grok_succeeded({"state": "done", "exit_code": 1}))
        self.assertFalse(grok_succeeded(parsed))
        self.assertTrue(grok_succeeded({"state": "done", "exit_code": 0}))

    def test_status_refreshes_stale_running_receipt_grok_state(self):
        task_id = "consult-20260822-215115"
        path = self.bridge.receipt_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "command_run": [FAKE_BIN, "consult"],
                    "timestamps": {"started_at": "2026-08-23T01:51:15+00:00", "finished_at": None},
                    "status": {"state": "running", "exit_code": None},
                    "grok_state": "running",
                    "task_dir": str(self.root / "missing-task-dir"),
                }
            ),
            encoding="utf-8",
        )
        with self._patch_run(stdout="status: done (exit 0)\n"):
            got = self.bridge.status(task_id)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(got.get("grok_state"), "done")
        self.assertEqual(receipt.get("grok_state"), "done")
        self.assertNotEqual(receipt.get("grok_state"), "running")
        self.assertEqual((receipt.get("status") or {}).get("state"), "done")


class TestMutationLock(_BridgeTest):
    def test_missing_lock_warns_and_still_returns_handle(self):
        records: list[str] = []

        class Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        log = logging.getLogger("hcli.grok_bridge")
        handler = Handler()
        log.addHandler(handler)
        old_level = log.level
        log.setLevel(logging.WARNING)
        stdout = _dry_stdout("warn-20260101-000000")
        try:
            with self._patch_run(stdout=stdout):
                handle = self.bridge.delegate(
                    "warn", VALID_CONTRACT, dry_run=True
                )
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)
        self.assertIsInstance(handle, GrokRunHandle)
        self.assertTrue(handle.task_id)
        blob = "\n".join(records)
        self.assertIn("mutation-serialization", blob)
        self.assertIn("no mutation_lock provided", blob)
        self.assertIn(NO_MUTATION_LOCK_WARNING.split(";")[0], blob)

    def test_lock_is_entered(self):
        order: list[str] = []

        @contextmanager
        def lock():
            order.append("enter")
            yield
            order.append("exit")

        def fake_run(argv, **kwargs):
            order.append("run")
            return _completed(
                argv, stdout=_dry_stdout("lock-20260101-000000")
            )

        with patch("hcli.grok_bridge.subprocess.run", side_effect=fake_run):
            self.bridge.delegate(
                "lock", VALID_CONTRACT, dry_run=True, mutation_lock=lock
            )
        self.assertEqual(order, ["enter", "run", "exit"])


class TestLintRejection(_BridgeTest):
    def test_lint_becomes_contract_error(self):
        stderr = (
            "grok-run: contract rejected before launch:\n"
            "NO_VERIFICATION: no test path and no runnable command — "
            "the result cannot be checked | NO_ACCEPTANCE: no acceptance "
            "criterion — nothing can mark this task done\n"
            "Fix the contract, or set SG_OFF=1 to launch anyway.\n"
        )
        with self._patch_run(stderr=stderr, returncode=1):
            with self.assertRaises(GrokContractError) as ctx:
                self.bridge.delegate(
                    "lint",
                    VALID_CONTRACT,
                    dry_run=True,
                    mutation_lock=lambda: nullcontext(),
                )
        self.assertIn("NO_VERIFICATION", ctx.exception.codes)
        self.assertIn("NO_ACCEPTANCE", ctx.exception.codes)
        self.assertIn("rejected the contract", str(ctx.exception))


class TestStatusParserIndependentOfSubprocess(unittest.TestCase):
    def test_parser_is_pure(self):
        spawned = []

        def boom(*a, **k):
            spawned.append(1)
            raise AssertionError("subprocess spawned")

        with patch("hcli.grok_bridge.subprocess.run", side_effect=boom):
            self.assertEqual(
                parse_grok_status("status: running (exit -)")["state"],
                "running",
            )
            self.assertEqual(
                parse_grok_status("status: done (exit 0)")["state"],
                "done",
            )
        self.assertEqual(spawned, [])


@unittest.skip(
    "live grok-run audit spends a real Grok session; this lane uses "
    "GROK_DRYRUN=1. See tools/headless/hcli_grokbridge_test.py."
)
def test_live_audit_skipped():
    raise AssertionError("this test must stay skipped")


if __name__ == "__main__":
    unittest.main()
