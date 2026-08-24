#!/usr/bin/env python3
"""Demonstrate HCLI driving the Noetic loop. Do not describe it — run it.

Chain (each step records the command that produced it):

    HCLI -> AgentOS -> resident -> Doctor -> Gravity experiment -> tools
         -> verifier -> accepted or rejected -> updated science

A REJECTED science candidate is the intended outcome of this campaign's
honest negative. WorkUnits themselves complete: the gate is exercised by a
verifier that would have to FAIL in order to promote.

Does not load a second 27B. Does not modify hcli/. HEAD ``hcli/`` is extracted
via ``git archive`` into a temp PYTHONPATH (sparse-checkout add is blocked).

    python3 tools/headless/noetic_ascension_loop.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPTS = REPO / "receipts" / "headless"
LOOP_RECEIPT = RECEIPTS / "NOETIC_ASCENSION_LOOP.json"
SCIENCE_UPDATE = RECEIPTS / "NOETIC_ASCENSION_SCIENCE_UPDATE.json"

STAGES = (
    "HCLI",
    "AgentOS",
    "resident",
    "Doctor",
    "Gravity experiment",
    "tools",
    "verifier",
    "accepted or rejected",
    "updated science",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").strip() or "unknown"


def atomic_write(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def run_cmd(
    argv: List[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": argv,
            "command_pretty": subprocess.list2cmdline(argv),
            "cwd": str(cwd),
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "wall_s": round(time.perf_counter() - t0, 4),
            "status": "RUN",
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": argv,
            "command_pretty": subprocess.list2cmdline(argv),
            "cwd": str(cwd),
            "exit_code": 124,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else str(exc),
            "wall_s": round(time.perf_counter() - t0, 4),
            "status": "RUN",
            "timeout": True,
        }
    except OSError as exc:
        return {
            "command": argv,
            "command_pretty": subprocess.list2cmdline(argv),
            "cwd": str(cwd),
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
            "wall_s": round(time.perf_counter() - t0, 4),
            "status": "NOT_RUN",
            "reason": f"OSError: {exc}",
            "timeout": False,
        }


def extract_hcli(dest: Path) -> Dict[str, Any]:
    """Materialize HEAD hcli/ without touching the worktree or sparse-checkout."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "archive", "--format=tar", "HEAD", "hcli"]
    t0 = time.perf_counter()
    arch = subprocess.run(cmd, cwd=str(REPO), capture_output=True)
    if arch.returncode != 0:
        return {
            "ok": False,
            "command": cmd,
            "stderr": (arch.stderr or b"").decode("utf-8", "replace"),
            "wall_s": round(time.perf_counter() - t0, 4),
        }
    unt = subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=arch.stdout,
        capture_output=True,
    )
    pkg = dest / "hcli" / "__main__.py"
    return {
        "ok": unt.returncode == 0 and pkg.is_file(),
        "command": cmd + ["|", "tar", "-x", "-C", str(dest)],
        "dest": str(dest),
        "package": str(dest / "hcli"),
        "stderr": (unt.stderr or b"").decode("utf-8", "replace"),
        "wall_s": round(time.perf_counter() - t0, 4),
        "verified_against": "HEAD:hcli (git archive, not the working tree)",
    }


def write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def hcli_env(extract: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(extract) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["NOETIC_REPO"] = str(REPO)
    # Never let a model path from the operator environment spawn a 27B.
    env.pop("HCLI_MODEL_PATH", None)
    env.pop("HAIDER_MODEL_PATH", None)
    if extra:
        env.update(extra)
    return env


def step(
    *,
    stage: str,
    name: str,
    command: List[str],
    result: Dict[str, Any],
    produced: Any = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    out = {
        "stage": stage,
        "name": name,
        "command": command,
        "command_pretty": result.get("command_pretty")
        or subprocess.list2cmdline(command),
        "exit_code": result.get("exit_code"),
        "status": result.get("status", "RUN"),
        "wall_s": result.get("wall_s"),
        "stdout_tail": (result.get("stdout") or "")[-4000:],
        "stderr_tail": (result.get("stderr") or "")[-2000:],
        "produced": produced,
    }
    if result.get("reason"):
        out["reason"] = result["reason"]
    if result.get("timeout"):
        out["timeout"] = True
    if notes:
        out["notes"] = notes
    return out


SCRIPT_AGENTOS = r'''
"""Prove AgentOS is the importable ownership surface, then write a GOAL ledger."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

extract = Path(os.environ["HCLI_EXTRACT"])
repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
sys.path.insert(0, str(extract))

from hcli.agentos import (  # noqa: E402
    DagStore, Ledger, Mission, Scheduler, WorkUnit, command_is_admissible,
)
from hcli import doctor, gravity  # noqa: E402

goal_md = ws / ".hcli" / "GOAL.md"
goal_md.parent.mkdir(parents=True, exist_ok=True)
ledger = Ledger()
ledger._preamble = (
    "# Ultragoal\n\n"
    "No Noetic candidate may be promoted unless it beats the parent on "
    "capability AND performance. An honest REJECTED is the correct gate "
    "output when none does.\n\n"
)
ob = ledger.add(
    "No Noetic candidate currently beats the qualified parent "
    "(qwen38-gravity-uniform-q4-v1).",
    acceptance=(
        "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json has "
        "verdict == NO_CANDIDATE_YET_BEATS_PARENT and no candidate is marked promoted"
    ),
    verify_command=(
        f"python3 {repo / 'tools/headless/test_no_candidate_beats_parent.py'}"
    ),
    tier="V1",
    risk="high",
)
ledger.save(goal_md)

ok_cmd, why = command_is_admissible("true")
assert ok_cmd is False, why

doc = {
    "ok": True,
    "agentos_exports": ["DagStore", "Ledger", "Mission", "Scheduler", "WorkUnit"],
    "mission_cls": Mission.__module__ + "." + Mission.__name__,
    "workunit_cls": WorkUnit.__module__ + "." + WorkUnit.__name__,
    "scheduler_cls": Scheduler.__module__ + "." + Scheduler.__name__,
    "ledger_cls": Ledger.__module__ + "." + Ledger.__name__,
    "dagstore_cls": DagStore.__module__ + "." + DagStore.__name__,
    "doctor_owned_paths": list(doctor.OWNED_PATHS),
    "gravity_owned_prefixes": list(gravity.OWNED_PREFIXES),
    "vacuous_true_admissible": False,
    "vacuous_true_reason": why,
    "obligation_id": ob.id,
    "goal_md": str(goal_md),
}
(ws / "steps" / "agentos.boot.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({"ok": True, "obligation_id": ob.id}))
'''

SCRIPT_RESIDENT = r'''
"""Observe a resident without spawning one. Never load a second 27B."""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path

extract = Path(os.environ["HCLI_EXTRACT"])
repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
sys.path.insert(0, str(extract))

from hcli.machine import resolve_runtime_limits  # noqa: E402
from hcli.runtime import RuntimePool  # noqa: E402

genome = json.loads((repo / "receipts/headless/MACHINE_GENOME.json").read_text())
metrics = json.loads((repo / "receipts/headless/NOETIC_METRICS.json").read_text())
cited = (
    ((metrics.get("time") or {}).get("COMPLETE_DECODE_TPS") or {}).get("not_re_run")
    or ""
)

limits = resolve_runtime_limits(repo_root=str(repo), start_dir=str(repo))
# Instantiate the pool object. Do NOT call start() / ensure — that would spawn.
pool_cls = RuntimePool.__name__

probes = []
for port, label in ((52484, "cited_llama_server_from_NOETIC_METRICS"), (11434, "ollama")):
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            body = resp.read()[:500].decode("utf-8", "replace")
            probes.append({
                "port": port, "label": label, "url": url,
                "status": "RUN", "http": resp.status, "body_head": body,
            })
    except Exception as exc:
        probes.append({
            "port": port, "label": label, "url": url,
            "status": "RUN", "http": None,
            "error": f"{type(exc).__name__}: {exc}",
        })

ps = {
    "status": "NOT_RUN",
    "reason": "ps is blocked in this sandbox (operation not permitted); HTTP probes ran instead",
}

# Did we spawn? The pool was constructed, never started.
doc = {
    "ok": True,
    "did_not_spawn_resident": True,
    "did_not_load_second_27b": True,
    "pool_class": "hcli.runtime.RuntimePool",
    "pool_started": False,
    "recorded_RESIDENT_RUNTIME_LIMIT": genome.get("RESIDENT_RUNTIME_LIMIT"),
    "recorded_ACTIVE_DECODE_LIMIT": genome.get("ACTIVE_DECODE_LIMIT"),
    "resolve_runtime_limits": {
        "resident_limit": getattr(limits, "resident_limit", None),
        "resident_source": getattr(limits, "resident_source", None),
        "gpu_decode": getattr(limits, "gpu_decode", None),
    },
    "cited_prior_resident": cited,
    "http_probes": probes,
    "process_table": ps,
}
(ws / "steps" / "resident.observe.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({
    "ok": True,
    "did_not_spawn_resident": True,
    "did_not_load_second_27b": True,
    "probes": [
        {"port": p["port"], "http": p.get("http"), "error": p.get("error")}
        for p in probes
    ],
}))
'''

SCRIPT_DOCTOR = r'''
"""Doctor: consume the v2 prescription. Do not re-prescribe (that streams parent shards)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

extract = Path(os.environ["HCLI_EXTRACT"])
repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
sys.path.insert(0, str(extract))

from hcli import doctor  # noqa: E402

path = repo / "receipts/headless/DOCTOR_V2_PRESCRIPTION.json"
d = json.loads(path.read_text())
assert d["schema"] == "hawking.headless.doctor_v2_prescription.v1"
assert d["organs_need_different_prescriptions"] is True
assert d["live_27b_policy"]["did_not_load_second_27b"] is True
organs = []
for o in d.get("organs") or []:
    presc = o.get("prescription") or {}
    organs.append({
        "organ_id": o.get("organ_id"),
        "kind": o.get("kind"),
        "layer": o.get("layer"),
        "physical_computation": presc.get("physical_computation"),
        "candidate_id": presc.get("candidate_id"),
        "real_organ_of_qualified_parent": o.get("real_organ_of_qualified_parent"),
    })
mlp = next(x for x in organs if x["kind"] == "mlp_swiglu")
gqa = next(x for x in organs if x["kind"] == "attention_gqa")
assert mlp["physical_computation"] != gqa["physical_computation"]
doc = {
    "ok": True,
    "did_not_re_prescribe": True,
    "did_not_load_second_27b": True,
    "doctor_owned_paths": list(doctor.OWNED_PATHS),
    "question": d.get("question"),
    "rejected_question": d.get("rejected_question"),
    "qualified_parent": d.get("qualified_parent"),
    "organs_need_different_prescriptions": True,
    "organs": organs,
    "source_receipt": "receipts/headless/DOCTOR_V2_PRESCRIPTION.json",
}
(ws / "steps" / "doctor.read.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({
    "ok": True,
    "mlp": mlp["candidate_id"],
    "gqa": gqa["candidate_id"],
    "differ": True,
}))
'''

SCRIPT_GRAVITY = r'''
"""Gravity experiment: scoring without a kernel is refused; a kernel win on reconstruct is refused."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
sys.path.insert(0, str(repo / "tools" / "headless"))

from gravity_compiler_search import (  # noqa: E402
    ScoringRefused, KernelWinRefused,
    candidate_without_kernel, candidate_q4_reconstruct_gemv, candidate_q4_geo_tpr64,
    try_score, try_credit_kernel_win, kernel_catalog, score, credit_kernel_win,
)

catalog = kernel_catalog()
no_kernel = try_score(candidate_without_kernel(), catalog=catalog)
assert no_kernel["refused"] is True
assert no_kernel["scored"] is False
assert no_kernel["defaulted_kernel"] in (None, False)

recon = candidate_q4_reconstruct_gemv()
fused = candidate_q4_geo_tpr64()
# score() requires a kernel; both of these name one.
score(recon, catalog=catalog)
score(fused, catalog=catalog)
win = try_credit_kernel_win(recon, fused, catalog=catalog)
assert win["refused"] is True
assert win["credited"] is False

raised_score = False
try:
    score(candidate_without_kernel(), catalog=catalog)
except ScoringRefused:
    raised_score = True
raised_win = False
try:
    credit_kernel_win(recon, fused, catalog=catalog)
except KernelWinRefused:
    raised_win = True
assert raised_score and raised_win

doc = {
    "ok": True,
    "did_not_load_second_27b": True,
    "scoring_without_kernel": {
        "refused": True,
        "scored": False,
        "exception": no_kernel.get("exception"),
        "reason": no_kernel.get("reason"),
    },
    "kernel_win_on_reconstruct": {
        "refused": True,
        "credited": False,
        "decision": (win.get("payload") or {}).get("decision") or win.get("decision"),
        "exception": win.get("exception"),
    },
    "ScoringRefused_raised": raised_score,
    "KernelWinRefused_raised": raised_win,
    "all_required_kernels_declared": catalog.get("all_required_declared"),
    "source_module": "tools/headless/gravity_compiler_search.py",
    "source_receipt": "receipts/headless/GRAVITY_COMPILER_SEARCH.json",
}
(ws / "steps" / "gravity.experiment.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({
    "ok": True,
    "scoring_without_kernel_refused": True,
    "kernel_win_refused": True,
}))
'''

SCRIPT_TOOLS = r'''
"""Tools: the honest-negative scoreboard still holds."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
cmd = [sys.executable, str(repo / "tools/headless/test_no_candidate_beats_parent.py")]
proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
parent = json.loads((repo / "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json").read_text())
assert proc.returncode == 0, proc.stdout + proc.stderr
assert parent["verdict"] == "NO_CANDIDATE_YET_BEATS_PARENT"
measured = parent["blocker"]["measured"]
doc = {
    "ok": True,
    "command": cmd,
    "exit_code": proc.returncode,
    "stdout": proc.stdout,
    "verdict": parent["verdict"],
    "incumbent": parent["incumbent"],
    "best_coherent_candidate_id": parent["best_coherent_candidate"]["id"],
    "bytes_reduction_fraction": measured["bytes_reduction_fraction"],
    "throughput_gain_fraction": measured["throughput_gain_fraction"],
    "reading": measured["reading"],
    "next_family": parent["next_representation_family"]["family"],
    "source_receipt": "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
}
(ws / "steps" / "tools.scoreboard.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({
    "ok": True,
    "verdict": parent["verdict"],
    "reading": measured["reading"],
}))
'''

SCRIPT_VERIFIER = r'''
"""Mechanical settle: 'a Noetic candidate beats the parent' is FALSE."""
from __future__ import annotations
import json, os
from pathlib import Path

repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
extract = Path(os.environ["HCLI_EXTRACT"])
sys_path_insert = str(extract)
import sys
sys.path.insert(0, sys_path_insert)
from hcli.verifier_pipeline import command_is_admissible  # noqa: E402

parent = json.loads((repo / "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json").read_text())
claim = "A Noetic candidate beats the qualified parent on capability AND performance."
cmd = (
    f"python3 {repo / 'tools/headless/test_no_candidate_beats_parent.py'}"
)
admissible, why = command_is_admissible(cmd)
assert admissible is True, why

# The discriminator exits 0 when the NEGATIVE holds. That settles the
# promotion claim as FALSE — not as a vacuous pass.
import subprocess
proc = subprocess.run(cmd, shell=True, cwd=str(repo), capture_output=True, text=True)
assert proc.returncode == 0
assert parent["verdict"] == "NO_CANDIDATE_YET_BEATS_PARENT"
promoted = "promoted" in json.dumps(parent).lower() and "no promotion is claimed" not in json.dumps(parent).lower()
assert not promoted

doc = {
    "ok": True,
    "claim": claim,
    "settled": "FALSE",
    "discriminator": cmd,
    "discriminator_exit": proc.returncode,
    "command_is_admissible": True,
    "vacuous_true_rejected_earlier": True,
    "receipt_verdict": parent["verdict"],
    "faking_the_shift": parent.get("faking_the_shift"),
    "source_receipt": "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
}
(ws / "steps" / "verifier.settle.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({"ok": True, "settled": "FALSE", "verdict": parent["verdict"]}))
'''

SCRIPT_GATE = r'''
"""Accept-or-reject the science candidate. REJECTED is a real gate output."""
from __future__ import annotations
import json, os
from pathlib import Path

repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
parent = json.loads((repo / "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json").read_text())
verifier = json.loads((ws / "steps" / "verifier.settle.json").read_text())
assert verifier["settled"] == "FALSE"
gate = "REJECTED"
reason = parent["blocker"]["statement"]
measured = parent["blocker"]["measured"]
doc = {
    "ok": True,
    "gate": gate,
    "science_candidate": parent.get("best_coherent_candidate", {}).get("id"),
    "why": reason,
    "measured": measured,
    "would_have_accepted_if": (
        "a candidate won protected qualification on capability AND performance, "
        "which none of the recorded mixes did"
    ),
    "did_not_engineer_an_acceptance": True,
    "source_receipt": "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
}
(ws / "steps" / "gate.reject.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps({"ok": True, "gate": gate, "candidate": doc["science_candidate"]}))
'''

SCRIPT_SCIENCE = r'''
"""Update recorded science with the REJECTED outcome of this loop."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

repo = Path(os.environ["NOETIC_REPO"])
ws = Path(os.environ["NOETIC_WS"])
receipts = repo / "receipts" / "headless"
parent = json.loads((receipts / "NO_CANDIDATE_YET_BEATS_PARENT.json").read_text())
gate = json.loads((ws / "steps" / "gate.reject.json").read_text())
head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
).stdout.strip()
doc = {
    "schema": "hawking.headless.noetic_ascension_science_update.v1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_head": head,
    "updated_by": "HCLI AgentOS WorkUnit science.update",
    "gate": gate["gate"],
    "verdict": parent["verdict"],
    "did_not_promote": True,
    "strongest_candidate_still": parent["best_coherent_candidate"]["id"],
    "blocker_id": parent["blocker"]["id"],
    "blocker_reading": parent["blocker"]["measured"]["reading"],
    "next_family": parent["next_representation_family"],
    "cites": [
        "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        "receipts/headless/NOETIC_ASCENSION_LOOP.json",
    ],
    "workspace_gate_artifact": str(ws / "steps" / "gate.reject.json"),
}
out = receipts / "NOETIC_ASCENSION_SCIENCE_UPDATE.json"
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n")
os.replace(tmp, out)
(ws / "steps" / "science.update.json").write_text(json.dumps({
    "ok": True,
    "written": str(out),
    "gate": doc["gate"],
    "verdict": doc["verdict"],
}, indent=2) + "\n")
print(json.dumps({"ok": True, "written": str(out), "gate": doc["gate"]}))
'''


UNITS = (
    {
        "id": "agentos.boot",
        "stage": "AgentOS",
        "script_name": "agentos_boot.py",
        "body": SCRIPT_AGENTOS,
        "deps": [],
        "step_file": "agentos.boot.json",
    },
    {
        "id": "resident.observe",
        "stage": "resident",
        "script_name": "resident_observe.py",
        "body": SCRIPT_RESIDENT,
        "deps": ["agentos.boot"],
        "step_file": "resident.observe.json",
    },
    {
        "id": "doctor.read",
        "stage": "Doctor",
        "script_name": "doctor_read.py",
        "body": SCRIPT_DOCTOR,
        "deps": ["resident.observe"],
        "step_file": "doctor.read.json",
    },
    {
        "id": "gravity.experiment",
        "stage": "Gravity experiment",
        "script_name": "gravity_experiment.py",
        "body": SCRIPT_GRAVITY,
        "deps": ["doctor.read"],
        "step_file": "gravity.experiment.json",
    },
    {
        "id": "tools.scoreboard",
        "stage": "tools",
        "script_name": "tools_scoreboard.py",
        "body": SCRIPT_TOOLS,
        "deps": ["gravity.experiment"],
        "step_file": "tools.scoreboard.json",
    },
    {
        "id": "verifier.settle",
        "stage": "verifier",
        "script_name": "verifier_settle.py",
        "body": SCRIPT_VERIFIER,
        "deps": ["tools.scoreboard"],
        "step_file": "verifier.settle.json",
    },
    {
        "id": "gate.reject",
        "stage": "accepted or rejected",
        "script_name": "gate_reject.py",
        "body": SCRIPT_GATE,
        "deps": ["verifier.settle"],
        "step_file": "gate.reject.json",
    },
    {
        "id": "science.update",
        "stage": "updated science",
        "script_name": "science_update.py",
        "body": SCRIPT_SCIENCE,
        "deps": ["gate.reject"],
        "step_file": "science.update.json",
    },
)


def _tail_file(path: Path, n: int = 80) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def run_loop(*, keep_scratch: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix="noetic_ascension_"))
    extract = scratch / "extract"
    workspace = scratch / "ws"
    workspace.mkdir(parents=True)
    (workspace / "steps").mkdir()
    (workspace / "scripts").mkdir()
    steps: List[Dict[str, Any]] = []
    head = git_head()

    extracted = extract_hcli(extract)
    steps.append(
        step(
            stage="HCLI",
            name="extract_hcli_from_HEAD",
            command=["git", "archive", "--format=tar", "HEAD", "hcli"],
            result={
                "command_pretty": "git archive --format=tar HEAD hcli | tar -x -C <scratch>/extract",
                "exit_code": 0 if extracted.get("ok") else 1,
                "stdout": json.dumps({k: extracted.get(k) for k in ("dest", "package", "ok")}),
                "stderr": extracted.get("stderr") or "",
                "wall_s": extracted.get("wall_s"),
                "status": "RUN" if extracted.get("ok") else "NOT_RUN",
                "reason": None if extracted.get("ok") else (extracted.get("stderr") or "extract failed"),
            },
            produced=extracted,
            notes=extracted.get("verified_against"),
        )
    )
    if not extracted.get("ok"):
        doc = {
            "schema": "hawking.headless.noetic_ascension_loop.v1",
            "generated_at": utc_now(),
            "git_head": head,
            "status": "NOT_RUN",
            "reason": "HEAD hcli/ could not be extracted via git archive",
            "steps": steps,
        }
        atomic_write(LOOP_RECEIPT, doc)
        return doc

    env = hcli_env(extract)
    env["HCLI_EXTRACT"] = str(extract)
    env["NOETIC_WS"] = str(workspace)
    env["HCLI_CPU_TIMEOUT"] = "180"
    # WorkUnitExecutor.run uses shell=True with inherited environ, not `env`.
    os.environ["HCLI_EXTRACT"] = str(extract)
    os.environ["NOETIC_REPO"] = str(REPO)
    os.environ["NOETIC_WS"] = str(workspace)
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    os.environ["HCLI_CPU_TIMEOUT"] = "180"
    os.environ.pop("HCLI_MODEL_PATH", None)
    os.environ.pop("HAIDER_MODEL_PATH", None)

    launch = ["python3", "-m", "hcli", "--workspace", str(workspace)]
    help_run = run_cmd(launch + ["/help"], cwd=workspace, env=env, timeout=60)
    steps.append(
        step(
            stage="HCLI",
            name="python3 -m hcli /help",
            command=launch + ["/help"],
            result=help_run,
            notes="canonical product entry from docs/CURRENT_ARCHITECTURE.md",
        )
    )
    status_run = run_cmd(launch + ["/status"], cwd=workspace, env=env, timeout=60)
    steps.append(
        step(
            stage="HCLI",
            name="python3 -m hcli /status",
            command=launch + ["/status"],
            result=status_run,
            notes="observes this process's pool; does not spawn a resident",
        )
    )

    sys.path.insert(0, str(extract))
    from hcli.controller import Controller  # noqa: WPS433
    from hcli.workunit import WorkUnit  # noqa: WPS433

    workunits = []
    for spec in UNITS:
        script = write_script(workspace / "scripts" / spec["script_name"], spec["body"])
        workunits.append(
            WorkUnit(
                id=spec["id"],
                role="tool",
                description=f"{spec['stage']}: {spec['id']}",
                dependencies=list(spec["deps"]),
                preferred_backend="tool",
                resource_class="LIGHT_CONTROL",
                verifier=f"{sys.executable} {script}",
            )
        )

    n = {"i": 0}

    def fingerprint() -> str:
        n["i"] += 1
        parts = [f"n={n['i']}"]
        steps_dir = workspace / "steps"
        if steps_dir.is_dir():
            for p in sorted(steps_dir.glob("*.json")):
                parts.append(f"{p.name}:{p.stat().st_size}")
        return "|".join(parts)

    ctrl = Controller(workspace=str(workspace), runtime_count=1, model=None)
    mission_result: Dict[str, Any]
    try:
        mission_result = ctrl.run_mission(
            goal=(
                "Demonstrate HCLI -> AgentOS -> resident -> Doctor -> Gravity "
                "-> tools -> verifier -> reject -> science. Do not promote. "
                "Do not load a second 27B."
            ),
            units=workunits,
            quiet=True,
            fingerprint_fn=fingerprint,
            repo_root=str(REPO),
            install_signals=False,
        )
    finally:
        try:
            ctrl.shutdown()
        except Exception:
            pass

    unit_rows = []
    if ctrl.mission is not None:
        for uid, wu in ctrl.mission.scheduler.units.items():
            unit_rows.append(
                {
                    "id": uid,
                    "status": wu.status,
                    "backend": wu.assigned_backend,
                    "verifier": wu.verifier,
                    "verification": wu.verification,
                    "attempts": wu.attempts,
                    "classification": wu.classification,
                }
            )
            spec = next((s for s in UNITS if s["id"] == uid), None)
            produced = None
            step_path = workspace / "steps" / (spec["step_file"] if spec else "")
            if spec and step_path.is_file():
                try:
                    produced = json.loads(step_path.read_text())
                except json.JSONDecodeError:
                    produced = {"raw": step_path.read_text()[-1000:]}
            ver = wu.verification if isinstance(wu.verification, dict) else {}
            steps.append(
                {
                    "stage": spec["stage"] if spec else "AgentOS",
                    "name": uid,
                    "command": [wu.verifier] if wu.verifier else [],
                    "command_pretty": wu.verifier,
                    "durable_reproduce": [
                        "python3",
                        "tools/headless/noetic_ascension_loop.py",
                    ],
                    "script_name": spec["script_name"] if spec else None,
                    "script_body": spec["body"] if spec else None,
                    "exit_code": ver.get("exit_code"),
                    "status": "RUN" if wu.status in ("completed", "failed") else "NOT_RUN",
                    "workunit_status": wu.status,
                    "stdout_tail": (ver.get("output") or "")[-4000:],
                    "stderr_tail": "",
                    "produced": produced,
                    "verification_ok": ver.get("ok"),
                    "driven_by": "hcli.mission.Mission via WorkUnitExecutor backend=tool",
                    "note": (
                        "The absolute verifier path is the command that ran in this "
                        "process. Scratch is reaped; re-run the durable_reproduce "
                        "command to regenerate the chain."
                    ),
                }
            )

    log_path = workspace / ".hcli" / "mission" / "mission.log"
    if not log_path.is_file():
        # mission.py uses mission_log_path
        for cand in (
            workspace / ".hcli" / "mission.log",
            workspace / ".hcli" / "mission" / "log.jsonl",
        ):
            if cand.is_file():
                log_path = cand
                break

    doc = {
        "schema": "hawking.headless.noetic_ascension_loop.v1",
        "generated_at": utc_now(),
        "git_head": head,
        "builder": "tools/headless/noetic_ascension_loop.py",
        "canonical_hcli_launch": "python3 -m hcli",
        "durable_reproduce": [
            "python3",
            "tools/headless/noetic_ascension_loop.py",
        ],
        "hcli_source": extracted,
        "workspace": str(workspace),
        "scratch": str(scratch),
        "did_not_load_second_27b": True,
        "did_not_modify_hcli_tree": True,
        "sparse_note": (
            "hcli/ is not materialized in this worktree. HEAD:hcli was extracted "
            "with git archive. git sparse-checkout add is blocked in this sandbox."
        ),
        "mission": mission_result,
        "units": unit_rows,
        "mission_log_tail": _tail_file(log_path, 60),
        "stages_required": list(STAGES),
        "steps": steps,
        "science_update_path": str(SCIENCE_UPDATE),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    # Gate outcome from the WorkUnit artifact, not from hope.
    gate_art = workspace / "steps" / "gate.reject.json"
    if gate_art.is_file():
        doc["gate"] = json.loads(gate_art.read_text()).get("gate")
    else:
        doc["gate"] = "UNKNOWN"
        doc["gate_reason"] = "gate.reject.json was not written"

    atomic_write(LOOP_RECEIPT, doc)

    if not keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        doc["scratch_reaped"] = True
        atomic_write(LOOP_RECEIPT, doc)
    else:
        doc["scratch_reaped"] = False
        atomic_write(LOOP_RECEIPT, doc)
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    keep = "--keep-scratch" in args
    doc = run_loop(keep_scratch=keep)
    print(f"wrote {LOOP_RECEIPT}")
    print(f"gate={doc.get('gate')} mission={((doc.get('mission') or {}).get('status'))}")
    print(f"steps={len(doc.get('steps') or [])} elapsed_s={doc.get('elapsed_s')}")
    # Assemble the handoff from receipts + this transcript.
    # extract/ was inserted at sys.path[0] for the HCLI import; put this
    # directory back in front so the builder resolves as a sibling module.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    else:
        sys.path.remove(str(HERE))
        sys.path.insert(0, str(HERE))
    from noetic_ascension_handoff import assemble_and_write  # noqa: WPS433

    handoff_path = assemble_and_write(loop_doc=doc)
    print(f"wrote {handoff_path}")
    failed = [
        s
        for s in (doc.get("steps") or [])
        if s.get("stage") != "HCLI"
        and s.get("name") in {u["id"] for u in UNITS}
        and s.get("workunit_status") not in ("completed", None)
        and s.get("verification_ok") is not True
    ]
    # HCLI launch steps must have exited 0.
    for s in doc.get("steps") or []:
        if s.get("name", "").startswith("python3 -m hcli") and s.get("exit_code") not in (0, None):
            failed.append(s)
    if doc.get("gate") != "REJECTED":
        print("FAIL: demonstration did not record REJECTED", file=sys.stderr)
        return 1
    if (doc.get("mission") or {}).get("status") != "completed":
        print(
            f"FAIL: mission status {(doc.get('mission') or {}).get('status')}",
            file=sys.stderr,
        )
        return 1
    if failed:
        print(f"FAIL: {len(failed)} steps did not complete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
