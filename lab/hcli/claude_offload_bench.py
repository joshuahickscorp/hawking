"""CLAUDE_OFFLOAD_BENCH — routine Claude work actually done in this repo.

A harness that cannot displace real work has not met G005. Every task below
cites a file, receipt, or grok-run artifact this campaign already produced.

FRACTION_OF_ROUTINE_CLAUDE_WORK_DISPLACED is the share of attempted routine
tasks whose result was produced by the native Qwen3.8 model. Harness-executed
passes (tools, fences, scripted say) are reported separately; they are not
model displacement. Environment-missing and GPU-lock refusals are SKIP and
are excluded from both sides. A skip is never a pass.

A model-driven task counts only when the native backend emitted the text
and a strict tool-call parse succeeded. Malformed calls are refused and
FAIL. The harness never rewrites the model's call into something that works.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lab.hcli.special_unit import (
    GROK_TASKS,
    GPU_LOCK,
    ActResult,
    NativeDecodeError,
    NativeDecodeRefused,
    NativeQwen38Backend,
    ResourceClass,
    ResourceGate,
    ScriptedBackend,
    SpecialUnit,
    StepStatus,
    looks_gpu_command,
)
from lab.layout import REPO_ROOT
from lab.receipts import seal
from lab.verification_authority import AuthorityPrincipal, SelfPromotionError

SCHEMA = "hawking.special_unit.claude_offload_bench.v3"
DEFAULT_RECEIPT = REPO_ROOT / "receipts" / "ascent-2026-08-16" / "G005_MODEL_DRIVES_TOOLS.json"
G015 = REPO_ROOT / "receipts" / "ascent-2026-08-16" / "G015_NATIVE_LEG_VERIFY_ON_MAIN.json"
REAL_GROK_TASK = "q80-host-facets-20260816-143023"
PRODUCER_MODEL = "model"
PRODUCER_HARNESS = "harness"
TaskRun = tuple[bool, str] | tuple[bool, str, str | None]


@dataclass
class OffloadTask:
    id: str
    title: str
    source: str
    routine_claude: bool
    run: Callable[[Path], TaskRun]
    producer: str = PRODUCER_HARNESS


def _unpack_run(result: TaskRun) -> tuple[bool, str, str | None]:
    if len(result) == 3:
        ok, detail, produced_by = result
        return bool(ok), str(detail), produced_by
    ok, detail = result
    return bool(ok), str(detail), None


def _unit(repo: Path, tmp: Path, **kwargs: Any) -> SpecialUnit:
    return SpecialUnit(
        repo=repo,
        session_root=tmp / "sessions",
        owned_worktree=tmp / "wt",
        backend=ScriptedBackend(["ok"]),
        **kwargs,
    )


def _begin_native_unit(
    tmp: Path,
    *,
    repo: Path = REPO_ROOT,
    owned_worktree: Path | None = None,
) -> tuple[SpecialUnit | None, str]:
    """Open a native-backed unit, or return a SKIP detail."""

    gate = ResourceGate(lock_path=GPU_LOCK)
    live, why = gate.protected_bench_live()
    if live:
        return None, f"SKIP {why}"
    owner = gate.lock_owner()
    if owner:
        return None, f"SKIP gpu lock held by {owner}; will not contend"
    backend = NativeQwen38Backend(repo=REPO_ROOT, gate=gate)
    unit = SpecialUnit(
        repo=repo,
        session_root=tmp / "sessions",
        owned_worktree=owned_worktree or (tmp / "wt"),
        backend=backend,
        gate=gate,
    )
    return unit, ""


def _act_or_skip(
    tmp: Path,
    user_text: str,
    *,
    repo: Path = REPO_ROOT,
    owned_worktree: Path | None = None,
    known_tools: list[str] | None = None,
    max_rounds: int = 1,
) -> tuple[ActResult | None, str, str | None]:
    unit, skip = _begin_native_unit(tmp, repo=repo, owned_worktree=owned_worktree)
    if unit is None:
        return None, skip, None
    try:
        result = unit.act(user_text, known_tools=known_tools, max_rounds=max_rounds)
    except NativeDecodeRefused as exc:
        return None, exc.bench_detail(), None
    except NativeDecodeError as exc:
        return None, f"native generate failed: {exc}", PRODUCER_MODEL
    return result, "", result.produced_by


def _require_model_calls(
    result: ActResult,
    *,
    tool: str | None = None,
) -> tuple[bool, str]:
    if result.malformed:
        return False, f"malformed tool call refused: {result.malformed}"
    if not result.calls:
        excerpt = result.text.replace("\n", " ")[:240]
        return False, f"model produced no tool call: {excerpt!r}"
    if result.produced_by != PRODUCER_MODEL:
        return False, f"not model-produced: {result.produced_by}"
    if tool is not None and not any(c.name == tool for c in result.calls):
        names = [c.name for c in result.calls]
        return False, f"model did not call {tool}: {names}"
    return True, ""


def _t_read_g015(tmp: Path) -> tuple[bool, str, str | None]:
    result, skip, produced = _act_or_skip(
        tmp,
        "Read receipts/ascent-2026-08-16/G015_NATIVE_LEG_VERIFY_ON_MAIN.json "
        "(limit 80 lines) so the still-open G015 harness legs can be extracted.",
        known_tools=["read"],
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result, tool="read")
    if not ok:
        return False, why, produced
    call = next(c for c in result.calls or [] if c.name == "read")
    path = str(call.arguments.get("path") or "")
    if "G015_NATIVE_LEG_VERIFY_ON_MAIN.json" not in path:
        return False, f"model read the wrong path: {path!r}", produced
    blob = "\n".join(r.output for r in result.results if r.name == "read")
    found = "legs_STILL_OPEN_for_G015" in blob and "HCLI conversation" in blob
    return found, "model read G015 open legs" if found else "G015 payload missing expected legs", produced


def _t_grep_proposed(tmp: Path) -> tuple[bool, str, str | None]:
    result, skip, produced = _act_or_skip(
        tmp,
        "Search the HCLI tree at lab/hcli for the symbol proposed_complete.",
        known_tools=["grep"],
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result, tool="grep")
    if not ok:
        return False, why, produced
    call = next(c for c in result.calls or [] if c.name == "grep")
    pattern = str(call.arguments.get("pattern") or "")
    if "proposed_complete" not in pattern:
        return False, f"model grep pattern missed proposed_complete: {pattern!r}", produced
    hits = 0
    for row in result.results:
        if row.name == "grep":
            hits = int(row.detail.get("hits") or 0)
    return hits > 0, f"model grep hits={hits}", produced


def _t_pytest_option_c(tmp: Path) -> tuple[bool, str, str | None]:
    result, skip, produced = _act_or_skip(
        tmp,
        "Run the Option-C unit tests at lab/tests/test_option_c.py.",
        known_tools=["pytest"],
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result, tool="pytest")
    if not ok:
        return False, why, produced
    call = next(c for c in result.calls or [] if c.name == "pytest")
    target = str(call.arguments.get("target") or "")
    if "test_option_c.py" not in target:
        return False, f"model pytest target missed test_option_c.py: {target!r}", produced
    ran = next((r for r in result.results if r.name == "pytest"), None)
    if ran is None:
        return False, "pytest tool did not run", produced
    return ran.ok, (ran.output or "")[-500:], produced


def _t_machine_state(tmp: Path) -> tuple[bool, str]:
    script = REPO_ROOT / "tools" / "agentos" / "machine_state.py"
    if not script.is_file():
        return False, f"SKIP missing {script}"
    unit = _unit(REPO_ROOT, tmp)
    result = unit.tool(
        "bash",
        {
            "argv": [__import__("sys").executable, str(script)],
            "timeout": 20,
        },
    )
    if not result.ok:
        return False, result.output
    try:
        snap = json.loads(result.output)
    except json.JSONDecodeError:
        return False, "machine_state did not emit JSON"
    return "clean_box_ok" in snap and "active_grok_lanes" in snap, "machine_state snapshot ok"


def _t_seal_receipt(tmp: Path) -> tuple[bool, str]:
    from lab.receipts import seal as _seal
    from lab.receipts import verify

    doc = _seal({"schema": "hawking.special_unit.bench_probe.v1", "ok": True})
    verify(doc)
    path = tmp / "wt" / "probe.json"
    unit = _unit(REPO_ROOT, tmp)
    (tmp / "wt").mkdir(parents=True, exist_ok=True)
    result = unit.tool("write", {"path": str(path), "content": json.dumps(doc, indent=2)})
    ok = result.ok and path.is_file() and "seal_sha256" in doc
    return ok, "sealed + wrote receipt" if ok else result.output


def _t_consume_real_grok(tmp: Path) -> tuple[bool, str]:
    task_dir = GROK_TASKS / REAL_GROK_TASK
    if not task_dir.is_dir():
        return False, f"SKIP missing {task_dir}"
    unit = _unit(REPO_ROOT, tmp, grok_tasks=GROK_TASKS)
    consumption = unit.grok_consume(REAL_GROK_TASK)
    ok = (
        consumption.get("consumed") is True
        and consumption.get("verified_complete") is False
        and consumption.get("dispatched") is True
        and bool(consumption.get("claim_excerpt"))
    )
    return ok, f"consumed {REAL_GROK_TASK}; verified_complete={consumption.get('verified_complete')}"


def _t_gpu_lock_inspect(tmp: Path) -> tuple[bool, str]:
    if looks_gpu_command("./tools/gpu_lane_lock.sh q80-mixed cargo bench"):
        # Must inspect, never take, the real lock.
        gate = ResourceGate(lock_path=GPU_LOCK)
        live, why = gate.protected_bench_live()
        _ = live
        unit = _unit(REPO_ROOT, tmp, gate=gate)
        ctx = unit.refresh_context()
        return "gpu_lock_owner" in ctx, why
    return False, "gpu pattern detector failed"


def _t_session_persist(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    unit.say("Qwen3.8 native leg is done; harness next.")
    sid = unit.session.session_id
    root = unit.store.root
    reopened = SpecialUnit.open(sid, repo=REPO_ROOT, session_root=root, owned_worktree=tmp / "wt")
    ok = (
        len(reopened.session.transcript) >= 2
        and reopened.session.transcript[0].text.startswith("Qwen3.8")
        and reopened.session.context_digest
    )
    return ok, f"resumed {sid} with {len(reopened.session.transcript)} turns"


def _t_plan_g015(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    plan = unit.plan("close the G015 harness legs")
    ids = [s["id"] for s in plan["steps"]]
    need = {"session_context", "grok_delegate", "grok_consume", "agentos_verify", "resource_pause"}
    return need.issubset(ids), f"steps={ids}"


def _t_proposed_vs_verified(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    unit.plan("boundary", steps=[{"id": "a", "title": "A", "dependencies": [], "oracle": {"kind": "predicate"}}])
    unit.propose("a", author="qwen38")
    step = unit.session.plan["steps"][0]
    if step["status"] != StepStatus.PROPOSED_COMPLETE.value:
        return False, f"propose set {step['status']}"
    if step["status"] == StepStatus.VERIFIED_COMPLETE.value:
        return False, "propose leaked verified_complete"
    try:
        unit.verify("a", principal=AuthorityPrincipal.SANDBOX_MODEL, certifier_id="qwen38")
        return False, "sandbox was allowed to verified_complete"
    except SelfPromotionError:
        pass
    unit.verify("a", principal=AuthorityPrincipal.PROTECTED_CONTROLLER, certifier_id="protected_controller")
    step = unit.session.plan["steps"][0]
    return step["status"] == StepStatus.VERIFIED_COMPLETE.value, step["status"]


def _t_resource_refuse(tmp: Path) -> tuple[bool, str]:
    lock = tmp / "fake-gpu.lock"
    lock.mkdir()
    (lock / "owner").write_text("q80-mixed-bench\n", encoding="utf-8")
    gate = ResourceGate(lock_path=lock, allow_gpu=True)
    unit = _unit(REPO_ROOT, tmp, gate=gate)
    admitted, why = unit.pause_for_resources(ResourceClass.GPU_HEAVY)
    if admitted:
        return False, f"admitted GPU during protected bench: {why}"
    if unit.session.status != "paused":
        return False, f"status={unit.session.status}"
    # CPU work must still run.
    cpu_ok, cpu_why = unit.gate.admit(ResourceClass.CPU)
    if not cpu_ok:
        return False, cpu_why
    # Resume stays paused while the lock is held.
    unit.resume()
    if unit.session.status != "paused":
        return False, f"resume cleared pause while lock live: {unit.session.status}"
    (lock / "owner").unlink()
    lock.rmdir()
    unit.resume()
    return unit.session.status == "idle", f"final={unit.session.status} pause_reason={unit.session.pause_reason}"


def _t_interrupt_resume(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    unit.say("start a long read")
    unit.interrupt("user")
    if unit.session.status != "interrupted":
        return False, unit.session.status
    unit.resume()
    return unit.session.status == "idle" and unit.session.pending_interrupt is None, unit.session.status


def _t_write_and_pytest(tmp: Path) -> tuple[bool, str, str | None]:
    wt = tmp / "wt"
    wt.mkdir(parents=True, exist_ok=True)
    result, skip, produced = _act_or_skip(
        tmp,
        "In the current directory write tiny.py implementing add(a, b) that returns "
        "a + b, write test_tiny.py that asserts add(2, 3) == 5, then run pytest on "
        "test_tiny.py. Emit one tool_call per action, in that order.",
        repo=wt,
        owned_worktree=wt,
        known_tools=["write", "pytest"],
        max_rounds=4,
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result)
    if not ok:
        return False, why, produced
    writes = [c for c in (result.calls or []) if c.name == "write"]
    pytests = [c for c in (result.calls or []) if c.name == "pytest"]
    if len(writes) < 2:
        return False, f"model wrote {len(writes)} file(s), need tiny.py and test_tiny.py", produced
    if not pytests:
        return False, "model did not call pytest", produced
    paths = [str(c.arguments.get("path") or "") for c in writes]
    if not any(p.endswith("tiny.py") and not p.endswith("test_tiny.py") for p in paths):
        return False, f"model did not write tiny.py: {paths}", produced
    if not any("test_tiny.py" in p for p in paths):
        return False, f"model did not write test_tiny.py: {paths}", produced
    ran = next((r for r in result.results if r.name == "pytest"), None)
    if ran is None or not ran.ok:
        return False, (ran.output if ran else "pytest did not run")[-300:], produced
    return True, ran.output[-300:], produced


def _t_model_read_native_wired(tmp: Path) -> tuple[bool, str, str | None]:
    result, skip, produced = _act_or_skip(
        tmp,
        "Read receipts/ascent-2026-08-16/G005_NATIVE_WIRED.json (limit 60 lines).",
        known_tools=["read"],
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result, tool="read")
    if not ok:
        return False, why, produced
    call = next(c for c in result.calls or [] if c.name == "read")
    path = str(call.arguments.get("path") or "")
    if "G005_NATIVE_WIRED.json" not in path:
        return False, f"model read the wrong path: {path!r}", produced
    blob = "\n".join(r.output for r in result.results if r.name == "read")
    found = "NativeQwen38Backend" in blob or "FRACTION_OF_ROUTINE_CLAUDE_WORK_DISPLACED" in blob
    return found, "model read G005_NATIVE_WIRED" if found else "native-wired receipt missing expected keys", produced


def _t_model_grep_verified(tmp: Path) -> tuple[bool, str, str | None]:
    result, skip, produced = _act_or_skip(
        tmp,
        "Search lab/hcli for the symbol verified_complete.",
        known_tools=["grep"],
    )
    if result is None:
        return False, skip, produced
    ok, why = _require_model_calls(result, tool="grep")
    if not ok:
        return False, why, produced
    call = next(c for c in result.calls or [] if c.name == "grep")
    pattern = str(call.arguments.get("pattern") or "")
    if "verified_complete" not in pattern:
        return False, f"model grep pattern missed verified_complete: {pattern!r}", produced
    hits = 0
    for row in result.results:
        if row.name == "grep":
            hits = int(row.detail.get("hits") or 0)
    return hits > 0, f"model grep verified_complete hits={hits}", produced


def _t_delegate_then_consume(tmp: Path) -> tuple[bool, str]:
    tasks = tmp / "grok-tasks"
    task_id = "bench-delegate-1"

    class _Runner:
        def delegate(self, **kwargs: Any) -> dict[str, Any]:
            tdir = tasks / task_id
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "status").write_text("done\n", encoding="utf-8")
            (tdir / "exit_code").write_text("0\n", encoding="utf-8")
            (tdir / "metadata.json").write_text(
                json.dumps({"task_id": task_id, "mode": "delegate", "workdir": str(tmp)}),
                encoding="utf-8",
            )
            (tdir / "grok-report.md").write_text(
                "CLAIM\nWrote tiny helper.\nSTATUS: SHIPPED\nEVIDENCE: test_tiny passed\n",
                encoding="utf-8",
            )
            (tdir / "diff.patch").write_text(
                "diff --git a/tiny.py b/tiny.py\n--- a/tiny.py\n+++ b/tiny.py\n",
                encoding="utf-8",
            )
            return {"task_id": task_id}

    unit = _unit(REPO_ROOT, tmp, grok_runner=_Runner(), grok_tasks=tasks)
    contract = tmp / "wt" / "contract.md"
    (tmp / "wt").mkdir(parents=True, exist_ok=True)
    contract.write_text("do the thing\n", encoding="utf-8")
    handle = unit.grok_delegate(slug="bench-delegate", contract=contract)
    if handle.get("consumed"):
        return False, "delegate must not consume"
    if not unit.unconsumed_grok():
        return False, "dispatch without consume should list unconsumed"
    consumption = unit.grok_consume(task_id)
    if unit.unconsumed_grok():
        return False, "consume did not clear the handle"
    ok = (
        consumption.get("consumed")
        and consumption.get("verified_complete") is False
        and consumption.get("files_in_diff") == ["tiny.py"]
    )
    return ok, f"delegate+consume; verified_complete={consumption.get('verified_complete')}"


def _t_authority_fence(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    unit.plan("fence", steps=[{"id": "z", "title": "Z", "dependencies": [], "oracle": {"kind": "predicate"}}])
    unit.propose("z")
    try:
        unit.verify("z", principal="sandbox_model", certifier_id="qwen38")
    except SelfPromotionError:
        return True, "sandbox cannot certify verified_complete"
    return False, "fence failed"


def _t_gpu_cmd_refuse(tmp: Path) -> tuple[bool, str]:
    unit = _unit(REPO_ROOT, tmp)
    result = unit.tool("bash", {"command": "./tools/gpu_lane_lock.sh q80-mixed ./workspace/ops/build/rust/release/examples/ascension_qwen80_hybrid_greedy"})
    return (not result.ok) and "refused" in result.output, result.output


def _native_unit(repo: Path, tmp: Path, **kwargs: Any) -> SpecialUnit:
    gate = kwargs.pop("gate", None) or ResourceGate()
    backend = kwargs.pop("backend", None) or NativeQwen38Backend(repo=repo, gate=gate)
    return SpecialUnit(
        repo=repo,
        session_root=tmp / "sessions",
        owned_worktree=tmp / "wt",
        backend=backend,
        gate=gate,
        **kwargs,
    )


def _t_native_qwen38_say(tmp: Path) -> tuple[bool, str]:
    """say() answered by the verified native decode, under gpu_lane_lock.sh."""

    gate = ResourceGate(lock_path=GPU_LOCK)
    live, why = gate.protected_bench_live()
    if live:
        return False, f"SKIP {why}"
    owner = gate.lock_owner()
    if owner:
        return False, f"SKIP gpu lock held by {owner}; will not contend"
    unit = _native_unit(REPO_ROOT, tmp, gate=gate)
    try:
        turn = unit.say("Say hi.")
    except NativeDecodeRefused as exc:
        return False, exc.bench_detail()
    except NativeDecodeError as exc:
        return False, f"native generate failed: {exc}"
    native = dict(turn.meta.get("native") or {})
    sid = unit.session.session_id
    reopened = SpecialUnit.open(
        sid,
        repo=REPO_ROOT,
        session_root=unit.store.root,
        owned_worktree=tmp / "wt",
    )
    persisted = (
        len(reopened.session.transcript) >= 2
        and reopened.session.transcript[1].text == turn.text
    )
    ok = (
        turn.role == "assistant"
        and bool(turn.text.strip())
        and turn.meta.get("produced_by") == PRODUCER_MODEL
        and native.get("fallbacks") == 0
        and persisted
    )
    excerpt = turn.text.replace("\n", " ")[:160]
    return (
        ok,
        (
            f"model generated {len(turn.text)} chars fallbacks={native.get('fallbacks')} "
            f"used_gpu_lane_lock={native.get('used_gpu_lane_lock')} persisted={persisted} "
            f"excerpt={excerpt!r}"
        ),
    )


def _t_native_refuse_protected(tmp: Path) -> tuple[bool, str]:
    """Native generate must inspect and refuse a protected lock, never take it."""

    lock = tmp / "fake-gpu.lock"
    lock.mkdir()
    (lock / "owner").write_text("q80-mixed-bench\n", encoding="utf-8")
    called: list[list[str]] = []

    def _runner(cmd: list[str]) -> Any:
        called.append(cmd)
        raise AssertionError("native backend invoked generate under a protected lock")

    gate = ResourceGate(lock_path=lock, allow_gpu=True)
    backend = NativeQwen38Backend(
        repo=REPO_ROOT,
        gate=gate,
        binary=Path("/nonexistent/ascension_qwen38_hybrid_greedy"),
        runner=_runner,
    )
    try:
        backend.complete("Say hi.", {})
    except NativeDecodeRefused as exc:
        if called:
            return False, f"invoked generate after refuse: {called[0][:4]}"
        ok = "protected" in str(exc).lower() or "q80" in str(exc).lower()
        return ok, f"refused protected lock without invoke: {exc}"
    return False, "native backend generated while a protected q80 lock was held"


TASKS: list[OffloadTask] = [
    OffloadTask(
        "native_qwen38_say",
        "say() answered by native Qwen3.8 decode (gpu_lane_lock.sh)",
        "crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs + G015_NATIVE_LEG_VERIFY_ON_MAIN.json",
        True,
        _t_native_qwen38_say,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "read_g015_open_legs",
        "Read G015 and extract still-open harness legs",
        "receipts/ascent-2026-08-16/G015_NATIVE_LEG_VERIFY_ON_MAIN.json",
        True,
        _t_read_g015,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "grep_proposed_complete",
        "Search the HCLI tree for proposed_complete",
        "lab/hcli/special_unit.py (this lane's first recon move, same shape as every Claude grep)",
        True,
        _t_grep_proposed,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "run_option_c_tests",
        "Run lab/tests/test_option_c.py",
        "lab/tests/test_option_c.py — standing Option-C suite Claude already maintains",
        True,
        _t_pytest_option_c,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "model_read_native_wired",
        "Model-driven read of G005_NATIVE_WIRED.json",
        "receipts/ascent-2026-08-16/G005_NATIVE_WIRED.json",
        True,
        _t_model_read_native_wired,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "model_grep_verified_complete",
        "Model-driven search of lab/hcli for verified_complete",
        "lab/hcli/special_unit.py",
        True,
        _t_model_grep_verified,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "machine_state_clean_box",
        "Query tools/agentos/machine_state.py",
        "tools/agentos/machine_state.py + receipts/agentos/PROGRAM_PLAN.md W0.T10",
        True,
        _t_machine_state,
    ),
    OffloadTask(
        "seal_json_receipt",
        "Seal a JSON receipt and write it in an owned worktree",
        "lab/receipts.py — every ascent lane's last action",
        True,
        _t_seal_receipt,
    ),
    OffloadTask(
        "consume_q80_host_facets_report",
        "Consume a real grok-run result (q80-host-facets)",
        f"~/.claude-grok/tasks/{REAL_GROK_TASK}/grok-report.md",
        True,
        _t_consume_real_grok,
    ),
    OffloadTask(
        "gpu_lock_inspect_without_take",
        "Inspect the GPU lock without creating or taking it",
        "tools/gpu_lane_lock.sh + S002 anti-scope (GPU belongs to Q80/DSV)",
        True,
        _t_gpu_lock_inspect,
    ),
    OffloadTask(
        "session_persist_qwen38_context",
        "Persist a Qwen3.8 conversation and resume it in a new process object",
        "G015 open leg: persistent session + project context",
        True,
        _t_session_persist,
    ),
    OffloadTask(
        "plan_g015_remaining_legs",
        "Plan the remaining G015 harness legs as a DAG",
        "G015_NATIVE_LEG_VERIFY_ON_MAIN.json legs_STILL_OPEN_for_G015",
        True,
        _t_plan_g015,
    ),
    OffloadTask(
        "proposed_vs_verified",
        "proposed_complete must not equal verified_complete",
        "G015 AgentOS verification leg + lab/verification_authority.py",
        True,
        _t_proposed_vs_verified,
    ),
    OffloadTask(
        "resource_refuse_gpu_during_protected",
        "Pause GPU work while a fake q80 lock is held; CPU still runs",
        "S002 anti-scope; tools/gpu_lane_lock.sh owner protocol",
        True,
        _t_resource_refuse,
    ),
    OffloadTask(
        "interrupt_and_resume",
        "Interrupt a live session and resume it",
        "G015 HCLI interrupt + restart/resume",
        True,
        _t_interrupt_resume,
    ),
    OffloadTask(
        "write_and_pytest_small",
        "Write a tiny module in an owned worktree and pytest it",
        "routine Claude test-authoring (same motion as lab/tests/*)",
        True,
        _t_write_and_pytest,
        PRODUCER_MODEL,
    ),
    OffloadTask(
        "delegate_then_consume_roundtrip",
        "Dispatch a Grok handle, refuse to treat it as done, then consume artifacts",
        "S002 §16-18; ~/.claude-grok/bin/grok-run artifact layout",
        True,
        _t_delegate_then_consume,
    ),
    OffloadTask(
        "verification_authority_no_self_promote",
        "Sandbox principal cannot certify verified_complete",
        "lab/verification_authority.py + bible §2 / §22",
        True,
        _t_authority_fence,
    ),
    OffloadTask(
        "refuse_protected_gpu_command",
        "Refuse a gpu_lane_lock + Q80 greedy command",
        "tools/gpu_lane_lock.sh + ascension_qwen80_hybrid_greedy",
        True,
        _t_gpu_cmd_refuse,
    ),
    OffloadTask(
        "native_refuses_protected_lock",
        "Native generate inspects a protected q80 lock and refuses without taking it",
        "tools/gpu_lane_lock.sh owner protocol + NativeQwen38Backend",
        True,
        _t_native_refuse_protected,
    ),
]


def _is_skip(detail: str) -> bool:
    return detail.startswith("SKIP ")


def run_bench(*, repo: Path = REPO_ROOT, receipt_path: Path | None = None) -> dict[str, Any]:
    _ = repo
    NativeQwen38Backend.reset_counters()
    rows: list[dict[str, Any]] = []
    attempted = 0
    passed = 0
    skipped = 0
    model_attempted = 0
    model_passed = 0
    harness_attempted = 0
    harness_passed = 0
    with tempfile.TemporaryDirectory(prefix="su-offload-") as td:
        tmp = Path(td)
        for task in TASKS:
            actual_producer: str | None = None
            try:
                ok, detail, actual_producer = _unpack_run(task.run(tmp))
            except NativeDecodeRefused as exc:
                ok, detail = False, exc.bench_detail()
            except Exception as exc:  # noqa: BLE001 — bench must record, not crash
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            if _is_skip(detail):
                skipped += 1
                rows.append(
                    {
                        "id": task.id,
                        "title": task.title,
                        "source": task.source,
                        "routine_claude": task.routine_claude,
                        "produced_by": None,
                        "expected_producer": task.producer,
                        "status": "SKIP",
                        "detail": detail,
                    }
                )
                continue
            produced_by = actual_producer if actual_producer is not None else task.producer
            if task.routine_claude:
                attempted += 1
                if ok:
                    passed += 1
                # Displacement counts only measured model production.
                # A scripted or harness result on a model-tagged task is not
                # model work. A model result on a harness-tagged task is.
                if produced_by == PRODUCER_MODEL:
                    model_attempted += 1
                    if ok:
                        model_passed += 1
                else:
                    harness_attempted += 1
                    if ok:
                        harness_passed += 1
            rows.append(
                {
                    "id": task.id,
                    "title": task.title,
                    "source": task.source,
                    "routine_claude": task.routine_claude,
                    "produced_by": produced_by,
                    "expected_producer": task.producer,
                    "status": "PASS" if ok else "FAIL",
                    "detail": detail[:800],
                }
            )
    # Honest displacement: only model-produced passes count. Harness passes
    # are capability, not displacement. Skips are excluded from both sides.
    fraction = (model_passed / attempted) if attempted else 0.0
    native_untouched = NativeQwen38Backend.generates_completed == 0
    gpu_untouched = NativeQwen38Backend.lock_acquisitions == 0
    doc = seal(
        {
            "schema": SCHEMA,
            "date": "2026-08-16",
            "claim": (
                "CLAUDE_OFFLOAD_BENCH with native Qwen3.8 driving the tool plane "
                "via strict <tool_call> parse (no repair)"
            ),
            "metric_definition": (
                "FRACTION_OF_ROUTINE_CLAUDE_WORK_DISPLACED = "
                "measured model-produced routine passes / all attempted routine tasks. "
                "produced_by is taken from the turn (native backend = model; "
                "scripted = scripted; harness tools = harness). "
                "A static expected_producer label is not displacement. "
                "Malformed tool calls are refused and FAIL. "
                "SKIP (missing env or GPU-lock refuse) is excluded from both sides "
                "and is never counted as a pass."
            ),
            "FRACTION_OF_ROUTINE_CLAUDE_WORK_DISPLACED": round(fraction, 4),
            "passed": passed,
            "attempted": attempted,
            "skipped": skipped,
            "model_produced_pass": model_passed,
            "model_produced_attempted": model_attempted,
            "harness_produced_pass": harness_passed,
            "harness_produced_attempted": harness_attempted,
            "status": "PASS" if attempted and passed == attempted else "PARTIAL" if passed else "FAIL",
            "gpu_lock_untouched": gpu_untouched,
            "native_model_untouched": native_untouched,
            "native_lock_acquisitions": NativeQwen38Backend.lock_acquisitions,
            "native_generates_completed": NativeQwen38Backend.generates_completed,
            "tasks": rows,
        }
    )
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main() -> int:
    doc = run_bench(receipt_path=DEFAULT_RECEIPT)
    print(json.dumps({k: doc[k] for k in doc if k != "tasks"}, indent=2))
    return 0 if doc.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
