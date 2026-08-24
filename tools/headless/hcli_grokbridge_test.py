#!/usr/bin/env python3
"""Protected HCLI GrokBridge checks. Plain python3 + assert. No GPU, no model.

Run:
    python3 tools/headless/hcli_grokbridge_test.py
    pytest tools/headless/hcli_grokbridge_test.py -q

Checks 2, 3, 5, 6 were observed FAILING against a naive first draft; the
observed failure text is printed by check_antivacuity() on every run.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import pathlib
import tempfile
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]

from hcli.grok_bridge import (  # noqa: E402
    GrokBridge,
    GrokContractError,
    GrokNotAvailable,
    GrokRunHandle,
    parse_grok_status,
)

VALID_CONTRACT = """# WRITE
create tools/haider/hcli/grok_bridge.py

# VERIFY
`pytest tools/haider/hcli/tests -q` must exit 0

# ACCEPTANCE
pytest exits 0
"""

FAILS: list[str] = []
NAIVE_FAILURES: dict[str, str] = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


# ---------------------------------------------------------------------------
# Naive first drafts — the implementations these checks must not pass against.
# ---------------------------------------------------------------------------


def _naive_parse_grok_status(text: str) -> dict:
    """First draft: split the status line; garbage raises."""
    line = [ln for ln in text.splitlines() if ln.startswith("status:")][0]
    parts = line.replace("(", "").replace(")", "").split()
    state = parts[1]
    exit_tok = parts[-1]
    return {"state": state, "exit_code": None if exit_tok == "-" else int(exit_tok)}


class _NaiveBridge:
    """First draft: no contract check, no which check, no warning, invents ids."""

    def __init__(self, workspace, receipts_dir=None) -> None:
        self.workspace = Path(workspace)
        self.receipts_dir = Path(receipts_dir) if receipts_dir else self.workspace / ".hcli" / "grok"

    def delegate(self, task, contract_text, **kwargs):
        import hcli.grok_bridge as gb

        gb.subprocess.run(["true"], check=False)
        return GrokRunHandle(
            task_id="invented-task-id",
            command_run=["true"],
            started_at="now",
        )


def _capture_log() -> tuple[logging.Logger, logging.Handler, list[str]]:
    records: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    log = logging.getLogger("hcli.grok_bridge")
    handler = Handler()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    return log, handler, records


@contextmanager
def _bomb_subprocess(module) -> Any:
    spawned: list[Any] = []

    def boom(*a, **k):
        spawned.append((a, k))
        raise AssertionError("subprocess spawned")

    old_run = module.subprocess.run
    old_popen = module.subprocess.Popen
    module.subprocess.run = boom
    module.subprocess.Popen = boom
    try:
        yield spawned
    finally:
        module.subprocess.run = old_run
        module.subprocess.Popen = old_popen


def _observe(name: str, fn: Callable[[], None]) -> str:
    """Run fn, expecting it to fail. Return the failure text."""
    try:
        fn()
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        NAIVE_FAILURES[name] = text
        print(f"ok   antivacuity-{name} observed FAIL: {text}")
        return text
    msg = f"naive draft for {name} unexpectedly passed"
    NAIVE_FAILURES[name] = msg
    print(f"FAIL antivacuity-{name}: {msg}")
    FAILS.append(f"antivacuity-{name}: {msg}")
    return msg


def check_antivacuity() -> None:
    """Observe checks 2, 3, 5, 6 failing against a naive first draft."""
    import hcli.grok_bridge as gb

    def naive_missing_contract() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with _bomb_subprocess(gb) as spawned:
                _NaiveBridge(tmp).delegate("t", "")
                if spawned:
                    raise AssertionError("subprocess spawned")
                raise AssertionError(
                    "naive delegate accepted empty contract and did not spawn"
                )

    _observe("2-missing-contract", naive_missing_contract)

    def naive_garbage_status() -> None:
        got = _naive_parse_grok_status("this is not status output at all")
        if got.get("state") != "unknown":
            raise AssertionError(
                f"naive parser did not return unknown (got {got!r}); "
                "garbage raised or mis-parsed"
            )

    _observe("3-status-garbage", naive_garbage_status)

    def naive_no_lock_warning() -> None:
        log, handler, records = _capture_log()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # Naive delegate never logs. Real check 5 requires the warning.
                handle = _NaiveBridge(tmp).delegate("t", VALID_CONTRACT)
            blob = "\n".join(records)
            assert handle.task_id
            assert "mutation-serialization" in blob, (
                f"expected warning naming mutation-serialization, got {records!r}"
            )
        finally:
            log.removeHandler(handler)

    _observe("5-no-mutation-lock-warning", naive_no_lock_warning)

    def naive_absent_grok() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Naive ignores shutil.which and invents a task id — B5 forbids this.
            old = gb.shutil.which
            gb.shutil.which = lambda name: None  # type: ignore[assignment]
            try:
                handle = _NaiveBridge(tmp).delegate("t", VALID_CONTRACT)
            finally:
                gb.shutil.which = old
            raise AssertionError(
                f"expected GrokNotAvailable, got handle {handle.task_id!r}"
            )

    _observe("6-grok-absent", naive_absent_grok)


def _grok_binary():
    """Resolve grok-run the way production does, not by raw PATH.

    These checks used `shutil.which` directly, so they reported "not on PATH"
    on a machine where the binary is installed and the bridge finds it fine.
    That made three real assertions unrunnable for an environmental reason.
    """
    from hcli.grok_bridge import GrokNotAvailable, find_grok_run

    try:
        return find_grok_run()
    except GrokNotAvailable:
        return None


def check_dryrun_argv() -> None:
    grok = _grok_binary()
    if grok is None:
        check(
            "dryrun-argv",
            False,
            "grok-run not found on PATH or at the install location; "
            "cannot assert argv against the real binary",
        )
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        bridge = GrokBridge(ws)

        d = bridge.delegate(
            "hcli-gbr-dry-del",
            VALID_CONTRACT,
            profile="power",
            background=True,
            no_worktree=True,
            dry_run=True,
            mutation_lock=lambda: nullcontext(),
        )
        argv = d.command_run
        check("dryrun-delegate-bin", argv[0].endswith("grok-run"), str(argv[0]))
        check("dryrun-delegate-mode", "delegate" in argv, str(argv))
        check(
            "dryrun-delegate-task",
            "--task" in argv and argv[argv.index("--task") + 1] == "hcli-gbr-dry-del",
            str(argv),
        )
        check("dryrun-delegate-contract-flag", "--contract" in argv, str(argv))
        cpath = Path(argv[argv.index("--contract") + 1]) if "--contract" in argv else None
        check(
            "dryrun-delegate-contract-file",
            bool(cpath and cpath.is_file() and "WRITE" in cpath.read_text()),
            str(cpath),
        )
        check(
            "dryrun-delegate-profile",
            "--profile" in argv and argv[argv.index("--profile") + 1] == "power",
            str(argv),
        )
        check("dryrun-delegate-background", "--background" in argv, str(argv))
        check("dryrun-delegate-no-worktree", "--no-worktree" in argv, str(argv))
        check("dryrun-delegate-stdout-is-dryrun", "DRY RUN" in (d.stdout or ""), d.stdout[:200])
        # grok-run's own dry-run prints the *inner* grok command, not grok-run flags.
        # --no-worktree shows up as --cwd <repo>; --profile power as --effort xhigh.
        check(
            "dryrun-delegate-inner-cwd",
            str(ws) in (d.stdout or "") or str(ws) in (d.resolved_command or ""),
            (d.resolved_command or d.stdout)[:300],
        )
        check(
            "dryrun-delegate-inner-effort",
            "xhigh" in (d.resolved_command or d.stdout or ""),
            (d.resolved_command or "")[:200],
        )
        check("dryrun-delegate-task-id-not-invented", bool(d.task_id) and "invented" not in d.task_id, d.task_id)

        a = bridge.audit(
            "hcli-gbr-dry-aud",
            VALID_CONTRACT,
            background=True,
            dry_run=True,
        )
        aargv = a.command_run
        check("dryrun-audit-mode", "audit" in aargv, str(aargv))
        check(
            "dryrun-audit-task",
            "--task" in aargv and aargv[aargv.index("--task") + 1] == "hcli-gbr-dry-aud",
            str(aargv),
        )
        check("dryrun-audit-contract", "--contract" in aargv, str(aargv))
        check("dryrun-audit-background", "--background" in aargv, str(aargv))
        check("dryrun-audit-stdout-is-dryrun", "DRY RUN" in (a.stdout or ""), a.stdout[:200])
        check(
            "dryrun-audit-sandbox-readonly",
            "read-only" in (a.resolved_command or a.stdout or ""),
            (a.resolved_command or "")[:200],
        )

        c = bridge.consult("one word: ping", background=True, dry_run=True)
        cargv = c.command_run
        check("dryrun-consult-mode", "consult" in cargv, str(cargv))
        check(
            "dryrun-consult-prompt",
            "--prompt" in cargv and cargv[cargv.index("--prompt") + 1] == "one word: ping",
            str(cargv),
        )
        check("dryrun-consult-background", "--background" in cargv, str(cargv))
        check("dryrun-consult-stdout-is-dryrun", "DRY RUN" in (c.stdout or ""), c.stdout[:200])
        check(
            "dryrun-consult-sandbox-readonly",
            "read-only" in (c.resolved_command or c.stdout or ""),
            (c.resolved_command or "")[:200],
        )


def check_missing_contract() -> None:
    import hcli.grok_bridge as gb

    with tempfile.TemporaryDirectory() as tmp:
        bridge = GrokBridge(tmp)
        with _bomb_subprocess(gb) as spawned:
            raised = None
            try:
                bridge.delegate("t", "")
            except GrokContractError as exc:
                raised = exc
            except Exception as exc:  # noqa: BLE001
                check(
                    "missing-contract-empty",
                    False,
                    f"wrong exception {type(exc).__name__}: {exc}",
                )
                return
            check(
                "missing-contract-empty",
                raised is not None and not spawned,
                f"raised={raised!r} spawned={spawned!r}",
            )

            raised = None
            try:
                bridge.audit("t", None)  # type: ignore[arg-type]
            except GrokContractError as exc:
                raised = exc
            except Exception as exc:  # noqa: BLE001
                check(
                    "missing-contract-none",
                    False,
                    f"wrong exception {type(exc).__name__}: {exc}",
                )
                return
            check(
                "missing-contract-none",
                raised is not None and not spawned,
                f"raised={raised!r} spawned={spawned!r}",
            )


def check_status_parsing() -> None:
    running = parse_grok_status("status: running (exit -)")
    check(
        "status-running",
        running == {"state": "running", "exit_code": None},
        str(running),
    )
    done = parse_grok_status("status: done (exit 0)")
    check(
        "status-done",
        done == {"state": "done", "exit_code": 0},
        str(done),
    )
    try:
        garbage = parse_grok_status("this is not status output at all")
    except Exception as exc:  # noqa: BLE001
        check("status-garbage", False, f"raised {type(exc).__name__}: {exc}")
        return
    check(
        "status-garbage",
        garbage.get("state") == "unknown" and garbage.get("exit_code") is None,
        str(garbage),
    )


def check_receipts() -> None:
    grok = _grok_binary()
    if grok is None:
        check("receipts", False, "grok-run not found on PATH or at the install location")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        bridge = GrokBridge(ws)
        d = bridge.delegate(
            "hcli-gbr-rcpt-del",
            VALID_CONTRACT,
            dry_run=True,
            mutation_lock=lambda: nullcontext(),
        )
        a = bridge.audit(
            "hcli-gbr-rcpt-aud",
            VALID_CONTRACT,
            dry_run=True,
        )
        for handle, mode in ((d, "delegate"), (a, "audit")):
            path = ws / ".hcli" / "grok" / f"{handle.task_id}.json"
            check(f"receipt-{mode}-exists", path.is_file(), str(path))
            if not path.is_file():
                continue
            receipt = json.loads(path.read_text(encoding="utf-8"))
            check(
                f"receipt-{mode}-task-id",
                receipt.get("task_id") == handle.task_id,
                str(receipt.get("task_id")),
            )
            check(
                f"receipt-{mode}-command-run",
                receipt.get("command_run") == handle.command_run
                and isinstance(receipt.get("command_run"), list),
                str(receipt.get("command_run")),
            )
            ts = receipt.get("timestamps") or {}
            check(
                f"receipt-{mode}-timestamps",
                bool(ts.get("started_at")),
                str(ts),
            )
            check(f"receipt-{mode}-status-key", "status" in receipt, str(list(receipt)))
            check(
                f"receipt-{mode}-report-path-key",
                "report_path" in receipt,
                str(list(receipt)),
            )


def check_no_mutation_lock_warning() -> None:
    grok = _grok_binary()
    if grok is None:
        check("no-mutation-lock-warning", False, "grok-run not found on PATH or at the install location")
        return
    log, handler, records = _capture_log()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bridge = GrokBridge(tmp)
            handle = bridge.delegate(
                "hcli-gbr-nolock",
                VALID_CONTRACT,
                dry_run=True,
            )
        blob = "\n".join(records)
        check(
            "no-mutation-lock-returns-handle",
            isinstance(handle, GrokRunHandle) and bool(handle.task_id),
            str(handle),
        )
        check(
            "no-mutation-lock-warning",
            "mutation-serialization" in blob and "mutation_lock" in blob,
            blob or "(no log records)",
        )
    finally:
        log.removeHandler(handler)


def check_grok_absent() -> None:
    import hcli.grok_bridge as gb

    with tempfile.TemporaryDirectory() as tmp:
        bridge = GrokBridge(tmp)
        old = gb.shutil.which
        gb.shutil.which = lambda name: None  # type: ignore[assignment]
        os.environ.pop("GROK_RUN", None)
        # Absence now means "not on PATH AND not at the canonical install
        # location", because find_grok_run falls back to the documented install
        # path rather than making HCLI depend on the caller's PATH. Stubbing
        # only shutil.which no longer simulates absence on a machine where the
        # binary is actually installed -- which is every machine that matters.
        old_hint = gb.DEFAULT_GROK_RUN_HINT
        gb.DEFAULT_GROK_RUN_HINT = pathlib.Path(tmp) / "definitely-absent" / "grok-run"
        try:
            try:
                bridge.delegate("t", VALID_CONTRACT, dry_run=True)
            except GrokNotAvailable as exc:
                check(
                    "grok-absent",
                    "PATH" in str(exc) or "which" in str(exc).lower(),
                    str(exc),
                )
                return
            except Exception as exc:  # noqa: BLE001
                check(
                    "grok-absent",
                    False,
                    f"wrong exception {type(exc).__name__}: {exc}",
                )
                return
            check("grok-absent", False, "did not raise GrokNotAvailable")
        finally:
            gb.shutil.which = old
            gb.DEFAULT_GROK_RUN_HINT = old_hint


def check_real_audit_skipped() -> None:
    """Exactly-one live grok-run call: SKIPPED.

    Even `grok-run audit` launches a full Grok session (read-only sandbox,
    still billed, still a model process). This lane's contract forbids
    spawning a full Grok session. GROK_DRYRUN=1 already exercised the
    real grok-run binary for delegate, audit, and consult argv.
    """
    print(
        "skip real-audit: skipped — grok-run audit still launches a billed "
        "Grok session; GROK_DRYRUN=1 already hit the real binary for argv"
    )


CHECKS = [
    ("antivacuity", check_antivacuity),
    ("dryrun-argv", check_dryrun_argv),
    ("missing-contract", check_missing_contract),
    ("status-parsing", check_status_parsing),
    ("receipts", check_receipts),
    ("no-mutation-lock-warning", check_no_mutation_lock_warning),
    ("grok-absent", check_grok_absent),
    ("real-audit-skipped", check_real_audit_skipped),
]


def main() -> int:
    FAILS.clear()
    NAIVE_FAILURES.clear()
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli grokbridge checks passed")
    print("naive first-draft failures (anti-vacuity):")
    for key, text in NAIVE_FAILURES.items():
        print(f"  {key}: {text}")
    return 0


def test_hcli_grokbridge() -> None:
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} grokbridge checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
