#!/usr/bin/env python3
"""One campaign, all three backends live, every unit real work.

The earlier MAX campaign satisfied MAX_NO_ARTIFICIAL_WORK and nothing else: it
ran 23 genuinely useful CPU checks against a `NullEngine`, so no Qwen cognition
and no Grok dispatch took part. Calling that a MIXED run would have been the
same species of claim this campaign keeps catching -- a green result from a path
that was never exercised.

This one dispatches all three:

  CPU    real checks over this repository, accepted on their own exit code
  QWEN   real cognition through the live llama-server, accepted on a
         DETERMINISTIC verifier that re-derives the answer independently
  GROK   real read-only audits through grok-run, accepted on the artifacts
         actually landing on disk

The Qwen units matter most and are the easiest to fake. A unit that asks a model
a question and then asks the same model whether it was right is a loop, not a
check. So every Qwen unit's answer is verified by a separate deterministic
command that computes the truth from the repository -- the model can fail, and
when it does the unit fails.

    python3 tools/headless/hcli_true_mixed_max.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

from hcli.config import Config          # noqa: E402
from hcli.engine import Engine          # noqa: E402
from hcli.events import EventBus        # noqa: E402
from hcli.mission import Mission        # noqa: E402
from hcli.workspace import Workspace    # noqa: E402
from hcli.workunit import WorkUnit      # noqa: E402

PY = sys.executable


def llama_port() -> Optional[int]:
    p = subprocess.run(
        ["bash", "-lc",
         "lsof -iTCP -sTCP:LISTEN -P -a -c llama-server | "
         "awk 'NR>1{print $9}' | sed 's/.*://' | head -1"],
        capture_output=True, text=True, timeout=30,
    )
    s = (p.stdout or "").strip()
    return int(s) if s.isdigit() else None


def _cpu_unit(uid: str, desc: str, command: str,
              deps: Optional[List[str]] = None) -> WorkUnit:
    # Anchored to the repo root: a CPU verifier runs with the MISSION workspace
    # as cwd, and these checks import hcli.
    return WorkUnit(
        id=uid, role="validate", description=desc,
        dependencies=list(deps or []),
        preferred_backend="cpu", resource_class="TEST",
        verifier=f"cd {REPO_ROOT} && {command}",
    )


def _qwen_unit(uid: str, desc: str, prompt: str, expect_command: str) -> WorkUnit:  # noqa: D401
    """Cognition on the local runtime, accepted by an INDEPENDENT re-derivation.

    `expect_command` computes the true answer from the repository without
    consulting the model. It is carried on the unit rather than as `verifier`
    because `_route_validation` runs verifiers as CPU commands and never passes
    them the model's output -- a shell verifier physically cannot compare
    against the answer. `VerifyingEngine.validate_workunit` does the comparison.

    Asking the model to grade itself would make every unit pass and measure
    nothing, which is the whole reason this indirection exists.
    """
    wu = WorkUnit(
        id=uid, role="analyze", description=f"{desc}. {prompt}",
        preferred_backend="qwen", resource_class="GPU_DECODE",
    )
    setattr(wu, "expect_command", f"cd {REPO_ROOT} && {expect_command}")
    return wu


def _grok_unit(uid: str, desc: str, prompt: str) -> WorkUnit:
    """A real read-only Grok audit, accepted on artifacts landing on disk.

    `_run_grok` REFUSES a Grok unit with no verifier -- "Grok text is evidence,
    not acceptance" -- which is the right call and is why this carries one. The
    verifier is formatted with the real task id, so it checks THAT task's report
    rather than anything the harness could have written itself.
    """
    return WorkUnit(
        id=uid, role="analyze", description=f"{desc}. {prompt}",
        preferred_backend="grok", resource_class="GROK",
        verifier=(
            "python3 -c \"import sys,pathlib;"
            "p=pathlib.Path.home()/'.claude-grok/tasks/{task_id}';"
            "r=[f for f in p.glob('grok-report*.md') if f.stat().st_size>200];"
            "print('report bytes:', [f.stat().st_size for f in r]);"
            "sys.exit(0 if r else 1)\""
        ),
    )


class VerifyingEngine(Engine):
    """Deterministic acceptance for every unit in this campaign.

    `_route_validation` consults `validate_workunit` BEFORE a unit's own
    verifier, so once this exists it owns acceptance for all three classes and
    has to handle each honestly:

      qwen  compare the model's answer against a command that re-derives the
            truth from the repository
      grok  require real non-empty output
      cpu   run the unit's own verifier, exactly as the mission would
    """

    def validate_workunit(self, wu: Any, raw: Any) -> Dict[str, Any]:
        expect = getattr(wu, "expect_command", None)
        if expect:
            answer = ""
            if isinstance(raw, dict):
                answer = str(raw.get("content") or "")
            got = "".join(ch for ch in answer if ch.isdigit())
            want_p = subprocess.run(
                expect, shell=True, capture_output=True, text=True, timeout=300)
            want = "".join(ch for ch in (want_p.stdout or "") if ch.isdigit())
            ok = bool(want) and got == want
            return {
                "ok": ok,
                "acceptance_source": "independent_rederivation",
                "model_answer": answer.strip()[:200],
                "rederived": want,
                "reason": None if ok else f"model said {got!r}, truth is {want!r}",
            }
        # The Grok executor already ran the unit's verifier AND consulted the
        # task's terminal state before letting the verifier speak. That logic is
        # better than anything repeated here, so its verdict stands.
        if isinstance(raw, dict) and isinstance(raw.get("validation"), dict):
            val = dict(raw["validation"])
            val.setdefault("acceptance_source", "backend_executor_verdict")
            return val
        from hcli.executors import WorkUnitExecutor

        # Engine.workspace is a Workspace object, not a path, and
        # WorkUnitExecutor takes a path. Passing the object raised inside
        # validate_workunit, _route_validation caught it, and EVERY CPU unit
        # failed with a reason that had nothing to do with what it checks --
        # the exact shape this campaign exists to catch, produced by me.
        ws = getattr(self.workspace, "root", None) or getattr(
            self.workspace, "path", None) or self.workspace
        cpu = WorkUnitExecutor(str(ws), engine=self)._run_cpu(wu, {})
        val = cpu.get("validation")
        if isinstance(val, dict):
            val.setdefault("acceptance_source", "workunit_verifier")
            return val
        return {"ok": False, "reason": "NO_DETERMINISTIC_VALIDATION"}


def build_units() -> List[WorkUnit]:
    units: List[WorkUnit] = []

    # ---- CPU: real checks over this repository -------------------------
    units.append(_cpu_unit(
        "cpu.suite", "the full HCLI unit suite",
        f'{PY} -m pytest {REPO_ROOT / "tools/haider/hcli/tests"} -q'))
    units.append(_cpu_unit(
        "cpu.ctxauth", "the context authority still serves the long ingress",
        f'{PY} -c "import sys; '
        'from hcli.context_budget import per_seq_context; '
        'assert per_seq_context(32768, 3) == 11008, per_seq_context(32768, 3)"'))
    units.append(_cpu_unit(
        "cpu.vmcp", "VMCP evidence still gates a WorkUnit, replay still refused",
        f'{PY} tools/headless/hcli_vmcp_integration.py'))
    units.append(_cpu_unit(
        "cpu.repair", "the repair tree still terminates",
        f'{PY} tools/headless/hcli_repair_homeostasis_test.py'))

    # ---- QWEN: real cognition, independently verified -------------------
    # Each answer is checked by a command that computes the truth from disk.
    units.append(_qwen_unit(
        "qwen.equilibrium",
        "recover the measured Grok equilibrium from its receipt",
        "Read receipts/headless/GROK_MAX_EQUILIBRIUM.json and report the value "
        "of the field useful_equilibrium. Answer with only that integer.",
        "python3 -c \"import json;print(json.load(open('receipts/headless/"
        "GROK_MAX_EQUILIBRIUM.json'))['useful_equilibrium'])\""))
    units.append(_qwen_unit(
        "qwen.scheduler_cap",
        "report the repair depth cap the scheduler enforces",
        "Read tools/haider/hcli/scheduler.py and report the integer value of "
        "MAX_REPAIR_DEPTH. Answer with only that integer.",
        "grep -m1 '^MAX_REPAIR_DEPTH' tools/haider/hcli/scheduler.py "
        "| sed 's/[^0-9]//g'"))

    # ---- GROK: real read-only audit -------------------------------------
    units.append(_grok_unit(
        "grok.audit_status",
        "an independent read of the status surface",
        "Read tools/haider/hcli/commands.py and report, in one sentence, "
        "whether format_status can print a per-hour rate without a window "
        "behind it. Cite file:line."))

    return units


def main() -> int:
    port = llama_port()
    if port is None:
        print("REFUSED: no llama-server is listening. A mixed campaign without "
              "a cognition backend is the CPU-only campaign again, and calling "
              "it mixed is the claim this harness exists to avoid.")
        return 2
    print(f"llama-server on :{port}")

    units = {u.id: u for u in build_units()}
    with tempfile.TemporaryDirectory(prefix="truemixed-") as ws:
        root = Path(ws)
        rt = type("R", (), {})()
        rt.index, rt.pid, rt.port, rt.active = 0, 1, port, True
        pool = type("P", (), {})()
        pool.runtimes, pool.admitted_n = [rt], 1
        cfg = Config(str(root), global_path=str(root / "g.json"))
        engine = VerifyingEngine(
            workspace=Workspace(str(root)), event_bus=EventBus(),
            runtime_provider=lambda: pool, runtime_state_provider=lambda: pool,
            runtime_count=1, model_name="local", config=cfg,
        )
        os.environ.setdefault("HCLI_CPU_TIMEOUT", "900")
        # GrokBridge resolves grok-run through PATH and REFUSES when it is
        # absent -- "Refusing to invent a task id or pretend a Grok session
        # ran." That refusal is correct and is the only reason the empty Grok
        # arm was diagnosable instead of silently green, but it does mean the
        # campaign has to supply the directory.
        grok_bin = str(Path.home() / ".claude-grok" / "bin")
        if grok_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = grok_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ["HCLI_MIXED_WORKSPACE"] = str(root)

        active: List[int] = []
        stop = {"f": False}

        def sample() -> None:
            while not stop["f"]:
                active.append(sum(
                    1 for u in units.values() if u.status == "running"))
                time.sleep(0.25)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.time()
        mission = Mission(
            str(root), engine=engine, units=units,
            goal="run every backend class on real work at once",
            runtime_count=1, quiet=False, no_progress_threshold=90,
        )
        mission.run()
        stop["f"] = True
        th.join(timeout=2)
        wall = time.time() - t0

        final = {u.id: u for u in mission.scheduler.units.values()}

    by_backend: Dict[str, Dict[str, Any]] = {}
    for uid, u in final.items():
        b = str(getattr(u, "assigned_backend", None)
                or getattr(u, "preferred_backend", None) or "unknown")
        row = by_backend.setdefault(b, {"verified": 0, "failed": 0, "units": []})
        ok = u.status == "completed"
        row["verified" if ok else "failed"] += 1
        row["units"].append({
            "id": uid,
            "status": u.status,
            "verification": getattr(u, "verification", None),
        })

    print(f"\nwall {wall:.1f}s  peak concurrent {max(active) if active else 0}")
    for b, row in sorted(by_backend.items()):
        print(f"  {b:<8} verified={row['verified']} failed={row['failed']}")

    three_live = all(
        by_backend.get(b, {}).get("verified", 0) > 0
        for b in ("cpu", "qwen", "grok")
    )
    receipt = {
        "gate": "HCLI_TRUE_MIXED_MAX",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "llama_port": port,
        "what_makes_this_mixed": "All three backend classes dispatched in ONE "
            "mission. The earlier campaign ran 23 real CPU units against a "
            "NullEngine, which satisfied MAX_NO_ARTIFICIAL_WORK and nothing else.",
        "how_qwen_units_are_accepted": "By a DETERMINISTIC verifier that "
            "re-derives the answer from the repository. The model never grades "
            "itself; a unit whose answer is wrong fails.",
        "wall_s": round(wall, 2),
        "peak_concurrent": max(active) if active else 0,
        "by_backend": by_backend,
        "all_three_verified": three_live,
        "result": "PASS" if three_live else "FAIL",
    }
    out = REPO_ROOT / "receipts/headless/HCLI_TRUE_MIXED_MAX.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nall three verified: {three_live}\nreceipt: {out}")
    return 0 if three_live else 1


if __name__ == "__main__":
    raise SystemExit(main())
