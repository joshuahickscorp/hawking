#!/usr/bin/env python3
"""Protected HCLI /grok dispatch checks. Plain python3 + assert. No GPU, no model.

Run:
    python3 tools/headless/hcli_grok_dispatch_test.py
    pytest tools/headless/hcli_grok_dispatch_test.py -q

Assertion style for checks 1-3 and 5: monkeypatched ``GrokBridge`` methods
(record the call, return a fake handle / dict / string). ``GROK_DRYRUN=1``
is set for the process so a missed patch cannot launch a billed session.
Check 4 bombs ``subprocess.run`` / ``Popen`` on the grok_bridge module.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

# Anything that would launch grok-run must not spend a session.
os.environ["GROK_DRYRUN"] = "1"

REPO = Path(__file__).resolve().parents[2]

from hcli.commands import CommandHandler  # noqa: E402
from hcli.grok_bridge import GrokBridge, GrokRunHandle  # noqa: E402
import hcli.grok_bridge as grok_bridge_mod  # noqa: E402

VALID_CONTRACT = """# WRITE
create tools/haider/hcli/commands.py

# VERIFY
`pytest tools/haider/hcli/tests -q` must exit 0

# ACCEPTANCE
pytest exits 0
"""

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


class FakeSession:
    def __init__(self) -> None:
        self.id = "sess-1"
        self.goal = "test goal"
        self.runtime_count = 1
        self.model = "model.gguf"
        self.messages: list = []


class FakeSteer:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeController:
    """Minimal controller: no MutationLock, no live mission."""

    def __init__(self, workspace: Path) -> None:
        self.workspace_root = str(workspace)
        self.session = FakeSession()
        self.mission = None
        self._selected: Optional[str] = None
        self._steers: List[str] = []
        self._missions: List[str] = []

    def select_model(self, name: str) -> str:
        self._selected = name
        return name

    def queue_steer(self, text: str) -> FakeSteer:
        self._steers.append(text)
        return FakeSteer(text)

    def run_mission(self, goal: str) -> Dict[str, str]:
        self._missions.append(goal)
        return {"mission_id": "m1", "status": "ok", "reason": "started"}

    def list_models(self) -> list:
        return []


def _handler(tmp: Path) -> CommandHandler:
    return CommandHandler(FakeController(tmp))


def _handle_ret(mode: str) -> Any:
    if mode in ("delegate", "audit", "consult"):

        def launch(self, *args, **kwargs):
            return GrokRunHandle(
                task_id=f"{mode}-rec",
                command_run=["grok-run", mode],
                started_at="now",
                mode=mode,
                dry_run=True,
            )

        return launch

    if mode in ("status", "wait"):

        def statusy(self, task_id, *args, **kwargs):
            return {"state": "done", "exit_code": 0, "task_id": task_id}

        return statusy

    if mode == "report":

        def report(self, task_id, *args, **kwargs):
            return f"report-for-{task_id}"

        return report

    if mode == "cleanup":

        def cleanup(self, task_id, *args, **kwargs):
            return {
                "task_id": task_id,
                "ok": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            }

        return cleanup

    raise ValueError(mode)


@contextmanager
def record_bridge(methods: Optional[List[str]] = None):
    """Patch GrokBridge methods; yield the call log."""
    names = methods or [
        "delegate",
        "audit",
        "consult",
        "status",
        "wait",
        "report",
        "cleanup",
    ]
    calls: List[Dict[str, Any]] = []
    patches = []

    def wrap(name: str) -> Callable:
        ret = _handle_ret(name)

        def impl(self, *args, **kwargs):
            calls.append(
                {
                    "method": name,
                    "args": args,
                    "kwargs": kwargs,
                    "task": args[0] if args else kwargs.get("task") or kwargs.get(
                        "task_id"
                    ) or kwargs.get("prompt"),
                    "contract_text": (
                        args[1]
                        if name in ("delegate", "audit") and len(args) > 1
                        else kwargs.get("contract_text")
                    ),
                    "prompt": args[0] if name == "consult" and args else kwargs.get(
                        "prompt"
                    ),
                    "mutation_lock": kwargs.get("mutation_lock", "UNSET"),
                }
            )
            return ret(self, *args, **kwargs)

        return impl

    try:
        for name in names:
            p = patch.object(GrokBridge, name, wrap(name))
            p.start()
            patches.append(p)
        yield calls
    finally:
        for p in reversed(patches):
            p.stop()


@contextmanager
def bomb_subprocess():
    spawned: list = []

    def boom(*a, **k):
        spawned.append((a, k))
        raise AssertionError("subprocess spawned")

    old_run = grok_bridge_mod.subprocess.run
    old_popen = grok_bridge_mod.subprocess.Popen
    grok_bridge_mod.subprocess.run = boom  # type: ignore[assignment]
    grok_bridge_mod.subprocess.Popen = boom  # type: ignore[assignment]
    try:
        yield spawned
    finally:
        grok_bridge_mod.subprocess.run = old_run
        grok_bridge_mod.subprocess.Popen = old_popen


def check_delegate_reaches_bridge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        contract = ws / "contract.md"
        contract.write_text(VALID_CONTRACT, encoding="utf-8")
        handler = _handler(ws)
        with record_bridge() as calls:
            result = handler.handle(f"/grok delegate g027-dispatch {contract}")
        check(
            "delegate-no-raise",
            result is not None and not isinstance(result, BaseException),
            repr(result),
        )
        hit = [c for c in calls if c["method"] == "delegate"]
        check("delegate-reached", len(hit) == 1, f"calls={calls!r} result={result!r}")
        if not hit:
            return
        rec = hit[0]
        check(
            "delegate-task",
            rec["task"] == "g027-dispatch",
            f"task={rec['task']!r}",
        )
        check(
            "delegate-contract",
            isinstance(rec["contract_text"], str)
            and "WRITE" in rec["contract_text"]
            and "VERIFY" in rec["contract_text"],
            f"contract_text={rec['contract_text']!r}"[:300],
        )


def check_audit_and_consult_reach_bridge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        contract = ws / "contract.md"
        contract.write_text(VALID_CONTRACT, encoding="utf-8")
        handler = _handler(ws)
        with record_bridge() as calls:
            audit_out = handler.handle(f"/grok audit g027-audit {contract}")
            consult_out = handler.handle("/grok consult one word: ping")
        audit_hits = [c for c in calls if c["method"] == "audit"]
        consult_hits = [c for c in calls if c["method"] == "consult"]
        check(
            "audit-reached",
            len(audit_hits) == 1,
            f"calls={calls!r} out={audit_out!r}",
        )
        if audit_hits:
            rec = audit_hits[0]
            check("audit-task", rec["task"] == "g027-audit", f"task={rec['task']!r}")
            check(
                "audit-contract",
                isinstance(rec["contract_text"], str) and "WRITE" in rec["contract_text"],
                f"contract_text={rec['contract_text']!r}"[:200],
            )
        check(
            "consult-reached",
            len(consult_hits) == 1,
            f"calls={calls!r} out={consult_out!r}",
        )
        if consult_hits:
            rec = consult_hits[0]
            check(
                "consult-prompt",
                rec["prompt"] == "one word: ping",
                f"prompt={rec['prompt']!r}",
            )


def check_status_wait_report_cleanup() -> None:
    task_id = "slug-20260101-000000"
    with tempfile.TemporaryDirectory() as tmp:
        handler = _handler(Path(tmp))
        with record_bridge() as calls:
            handler.handle(f"/grok status {task_id}")
            handler.handle(f"/grok wait {task_id}")
            handler.handle(f"/grok report {task_id}")
            handler.handle(f"/grok cleanup {task_id}")
        by_method = {c["method"]: c for c in calls}
        for name in ("status", "wait", "report", "cleanup"):
            rec = by_method.get(name)
            check(
                f"{name}-reached",
                rec is not None,
                f"calls={calls!r}",
            )
            if rec is not None:
                got = rec["args"][0] if rec["args"] else rec["kwargs"].get("task_id")
                check(
                    f"{name}-id",
                    got == task_id,
                    f"got={got!r} rec={rec!r}",
                )


def check_unrecognized_usage_no_spawn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        handler = _handler(Path(tmp))
        with bomb_subprocess() as spawned:
            raised = None
            result = None
            try:
                result = handler.handle("/grok frobnicate")
            except Exception as exc:  # noqa: BLE001
                raised = exc
        check(
            "unrecognized-no-raise",
            raised is None,
            f"raised={type(raised).__name__}: {raised}" if raised else "",
        )
        check(
            "unrecognized-no-subprocess",
            spawned == [],
            f"spawned={spawned!r}",
        )
        text = result or ""
        check(
            "unrecognized-usage",
            "delegate" in text
            and "audit" in text
            and "consult" in text
            and ("Commands:" in text or "Usage" in text or "/grok" in text),
            f"result={result!r}",
        )


def check_delegate_without_lock_returns_handle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        contract = ws / "contract.md"
        contract.write_text(VALID_CONTRACT, encoding="utf-8")
        handler = _handler(ws)
        # FakeController has no mutation_lock and mission is None.
        with record_bridge() as calls:
            raised = None
            result = None
            try:
                result = handler.handle(f"/grok delegate nolock-task {contract}")
            except Exception as exc:  # noqa: BLE001
                raised = exc
        check(
            "nolock-no-raise",
            raised is None,
            f"raised={type(raised).__name__}: {raised}" if raised else "",
        )
        hit = [c for c in calls if c["method"] == "delegate"]
        check("nolock-reached", len(hit) == 1, f"calls={calls!r} result={result!r}")
        if hit:
            rec = hit[0]
            lock = rec["mutation_lock"]
            check(
                "nolock-lock-omitted-or-none",
                lock is None or lock == "UNSET",
                f"mutation_lock={lock!r}",
            )
        check(
            "nolock-human-string",
            isinstance(result, str) and bool(result) and "Unknown command" not in result,
            f"result={result!r}",
        )


def check_no_regression_existing_commands() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        handler = _handler(Path(tmp))
        model_out = handler.handle("/model foo-gguf")
        mission_out = handler.handle("/mission do the thing")
        steer_out = handler.handle("/steer remember this")
        status_out = handler.handle("/status")
        for name, out in (
            ("model", model_out),
            ("mission", mission_out),
            ("steer", steer_out),
            ("status", status_out),
        ):
            check(
                f"regression-{name}-str",
                isinstance(out, str) and bool(out),
                repr(out),
            )
            check(
                f"regression-{name}-not-unknown",
                isinstance(out, str) and not out.startswith("Unknown command"),
                repr(out),
            )
        check(
            "regression-model-switched",
            isinstance(model_out, str) and "Switched" in model_out,
            repr(model_out),
        )
        check(
            "regression-mission-id",
            isinstance(mission_out, str) and "m1" in mission_out,
            repr(mission_out),
        )
        check(
            "regression-steer-queued",
            isinstance(steer_out, str) and "Steer queued" in steer_out,
            repr(steer_out),
        )
        check(
            "regression-status-session",
            isinstance(status_out, str) and "sess-1" in status_out,
            repr(status_out),
        )


CHECKS = [
    ("delegate-reaches-bridge", check_delegate_reaches_bridge),
    ("audit-and-consult-reach-bridge", check_audit_and_consult_reach_bridge),
    ("status-wait-report-cleanup", check_status_wait_report_cleanup),
    ("unrecognized-usage-no-spawn", check_unrecognized_usage_no_spawn),
    ("delegate-without-lock-returns-handle", check_delegate_without_lock_returns_handle),
    ("no-regression-existing-commands", check_no_regression_existing_commands),
]


def main() -> int:
    FAILS.clear()
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
    print("\nall hcli grok dispatch checks passed")
    return 0


def test_hcli_grok_dispatch() -> None:
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} grok dispatch checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
