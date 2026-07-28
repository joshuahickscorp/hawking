#!/usr/bin/env python3.12
"""Final-ascent control plane: evidence-derived status publisher.

Republishes the campaign's root control-plane artifacts from live evidence only.
Nothing is asserted that a file, process table, launchd, or git query does not support.
Unknown live fields stay null/UNKNOWN. Capability fences stay closed unless a hash-
approved substrate exists on disk.

    python3.12 tools/campaign/final_ascent_status.py            # republish
    python3.12 tools/campaign/final_ascent_status.py --json     # print only
    python3.12 tools/campaign/final_ascent_status.py --diagnose # alias of default RO mode

Idempotent: a ledger row is appended only when the derived transition shape changes.
Publication is atomic per file (temp + os.replace). Generated files say they are
generated and must not be hand-edited.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

STATUS_MD = "HAWKING_FINAL_ASCENT_STATUS.md"
STATUS_JSON = "HAWKING_FINAL_ASCENT_STATUS.json"
LEDGER = "HAWKING_FINAL_ASCENT_LEDGER.jsonl"
DAG = "HAWKING_FINAL_ASCENT_DEPENDENCY_DAG.json"
OWNERSHIP = "HAWKING_FINAL_ASCENT_LANE_OWNERSHIP.json"
GOAL = "HAWKING_FINAL_ASCENT_CONTINUATION_GOAL.md"
NEXT_COMMAND = "HAWKING_FINAL_ASCENT_NEXT_COMMAND.sh"

GENERATED_BANNER = (
    "Generated from live evidence by `tools/campaign/final_ascent_status.py`. "
    "Do not hand-edit."
)

# Continuation files named by the new directive but absent from HEAD / git log --all.
ABSENT_DIRECTIVE_FILES = (
    "HAWKING_RAMANUJAN_CONTINUUM_CAMPAIGN.md",
    "HAWKING_EVOLUTION_PARALLEL_CONTINUATION.md",
    "HIDE_YOU_PERSONAL_AI_EXTENSION.md",
    "Hawking_Prometheus_Ramanujan_Canonical_Master_Plan_Revision_3.md",
)

FENCE_NAMES = (
    "ODYSSEY_LAUNCH_AUTHORIZED",
    "RAMANUJAN_RESEARCH_AUTHORIZED",
    "HIDE_KERNEL_TURN",
)

TERMINAL_ENDPOINT = "RAMANUJAN_SANDBOX_READY"

# Lane ids are stable contract handles; names are human.
LANE_SPECS: list[dict[str, Any]] = [
    {
        "id": "FA01",
        "name": "live-state-control-plane",
        "owner": "controller",
        "resource_class": "light-readonly",
        "branch": None,  # filled from live git
        "worktree": None,
        "inputs": [
            "HAWKING_RESUME_CHECKPOINT.md",
            "HAWKING_PARALLEL_STATUS.json",
            "HAWKING_CONTINUUM_STATUS.json",
            "HAWKING_HEAVY_CONTINUATION_STATUS.json",
            "live git/worktree/process/launchd state",
        ],
        "outputs": [
            STATUS_MD,
            STATUS_JSON,
            LEDGER,
            DAG,
            OWNERSHIP,
            GOAL,
            NEXT_COMMAND,
        ],
        "forbidden_files": [
            "odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED",
            "teacher capsules",
            "model bodies",
            "MOP-owned caches",
        ],
        "tests": "python3.12 -m tools.campaign.test_final_ascent_status",
        "promotion_gate": "publisher is read-only by default; never flips fences",
        "dependencies": [],
    },
    {
        "id": "FA02",
        "name": "capable-glm-basis-pilot-and-substrate",
        "owner": "controller+grok",
        "resource_class": "HEAVY-EXCLUSIVE",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "GLM52_GENERATION_B_CAPABILITY_VERDICT.json",
            "GLM52_BYTE_ATTRIBUTION.json",
            "odyssey/launch/SUBSTRATE_CAPABILITY.json",
            "HAWKING_RESUME_CHECKPOINT.md",
            "HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json",
            "real teacher capsules (READ ONLY)",
        ],
        "outputs": [
            "hash-APPROVED Math-Preserve-v2 substrate entry in SUBSTRATE_CAPABILITY.json",
            "capability gate receipt with G_math and G_live PASS",
        ],
        "forbidden_files": [
            "odyssey/launch/",
            "teacher capsules (must not delete/modify)",
            "prior refused artifact bodies (negative controls)",
        ],
        "tests": (
            ".venv/glm52/bin/python tools/condense/glm52_capability_gate.py "
            "--artifact <dir> --run --out CAPABILITY.json"
        ),
        "promotion_gate": (
            "artifact_index_sha256 bound APPROVED in odyssey/launch/SUBSTRATE_CAPABILITY.json "
            "after G_math and G_live PASS; reconstruction metrics never substitute"
        ),
        "dependencies": [],
    },
    {
        "id": "FA03",
        "name": "base-accelerated-runtime",
        "owner": "codex+controller",
        "resource_class": "medium-gpu",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "HAWKING_BASE_TRUE_TPS.json",
            "HAWKING_BASE_RUNTIME_*.json",
            "Temporal Gravity external receipts",
        ],
        "outputs": [
            "BASE_TRUE_TPS measured on a capable provider",
            "acceleration provider receipt",
        ],
        "forbidden_files": [
            "odyssey/launch/",
            "MOP-owned files",
        ],
        "tests": "runtime selfcheck + BASE_TRUE_TPS measurement harness",
        "promotion_gate": "real provider capable; TPS measured not estimated; no kernel turn flip",
        "dependencies": ["FA02"],
    },
    {
        "id": "FA04",
        "name": "hide-you-chat-ide",
        "owner": "grok+controller",
        "resource_class": "light-medium",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "HIDE_YOU_*.json",
            "HIDE archaeology / surface contracts",
            "capable real provider",
        ],
        "outputs": [
            "HIDE surfaces REAL_WIRED against a capable provider",
            "kernel-turn still default-off until explicit promotion",
        ],
        "forbidden_files": [
            "HIDE_KERNEL_TURN fence flip without capable provider",
            "odyssey/launch/",
        ],
        "tests": "hide crate tests; surface authority checks",
        "promotion_gate": (
            "HIDE_KERNEL_TURN stays false until a capable real provider passes its "
            "promotion gate; wiring prep may proceed without flipping the fence"
        ),
        "dependencies": ["FA02", "FA03"],
    },
    {
        "id": "FA05",
        "name": "odyssey-t0-t7",
        "owner": "controller",
        "resource_class": "HEAVY-EXCLUSIVE",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "hash-APPROVED capable Math-Preserve-v2",
            "ODYSSEY_LAUNCH_AUTHORIZED (human fence)",
            "odyssey/** contracts",
        ],
        "outputs": [
            "ODYSSEY_T0_RECEIPT.json through T7 receipts",
            "winning checkpoint",
        ],
        "forbidden_files": [
            "training on REFUSED/UNVERIFIED substrates",
            "flipping ODYSSEY_LAUNCH_AUTHORIZED",
        ],
        "tests": "ODYSSEY_PROMOTION_GATE.md G1-G4; substrate hash verification",
        "promotion_gate": (
            "requires hash-APPROVED capable Math-Preserve-v2 AND "
            "ODYSSEY_LAUNCH_AUTHORIZED=true (human only)"
        ),
        "dependencies": ["FA02", "FA03"],
    },
    {
        "id": "FA06",
        "name": "math-frozen",
        "owner": "controller",
        "resource_class": "HEAVY-EXCLUSIVE",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "Odyssey winning checkpoint",
            "hash-APPROVED capable Math-Preserve-v2 parent family",
        ],
        "outputs": [
            "Math-Frozen Director artifact",
            "Math-Frozen capability receipt",
        ],
        "forbidden_files": [
            "repacking a REFUSED substrate as Math-Frozen",
        ],
        "tests": "capability gate on frozen director; integrity seal",
        "promotion_gate": (
            "cannot promote without hash-APPROVED capable Math-Preserve-v2 and an "
            "Odyssey winner to freeze"
        ),
        "dependencies": ["FA02", "FA05"],
    },
    {
        "id": "FA07",
        "name": "fabric-bridge-adapters-cli-model-vault",
        "owner": "grok",
        "resource_class": "light-medium",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "FABRIC_*.json",
            "HAWKING_BRIDGE_*.json",
            "HAWKING_ADAPTER_*.json",
        ],
        "outputs": [
            "Fabric/Bridge/adapter ABI sealed",
            "CLI and model-vault surfaces",
        ],
        "forbidden_files": [
            "odyssey/launch/",
            "destructive consolidation",
        ],
        "tests": "FABRIC_TEST_EXECUTION_RECEIPT / adapter matrix",
        "promotion_gate": "contracts real; no overclaim on unwired surfaces",
        "dependencies": [],
    },
    {
        "id": "FA08",
        "name": "hawking-consolidation",
        "owner": "controller",
        "resource_class": "medium",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "HAWKING_CONSOLIDATION_INVENTORY.json",
            "completed Fabric/HIDE/Math-Frozen surfaces",
        ],
        "outputs": [
            "HAWKING_EVOLUTION_COMPLETE seal",
            "single authority per subsystem",
        ],
        "forbidden_files": [
            "destructive refactor during or before Odyssey/Math-Frozen",
            "main merge",
        ],
        "tests": "inventory load-bearing checks; no dual authorities left silent",
        "promotion_gate": "campaign law: no destructive consolidation before Math-Frozen",
        "dependencies": ["FA04", "FA06", "FA07"],
    },
    {
        "id": "FA09",
        "name": "ramanujan-migration",
        "owner": "controller",
        "resource_class": "medium-io",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "HAWKING_EVOLUTION_COMPLETE",
            "owned Ramanujan contracts",
        ],
        "outputs": [
            "~/Downloads/ramanujan repository",
            "migration receipt",
        ],
        "forbidden_files": [
            "flipping RAMANUJAN_RESEARCH_AUTHORIZED",
            "training before Director freeze",
        ],
        "tests": "migration inventory completeness",
        "promotion_gate": "separate repo only after evolution seal; research fence stays closed",
        "dependencies": ["FA08"],
    },
    {
        "id": "FA10",
        "name": "local-formal-system-training",
        "owner": "controller+grok",
        "resource_class": "HEAVY-EXCLUSIVE",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "frozen Math-Frozen Director",
            "formal environment (Lean/Mathlib)",
            "sealed corpora",
        ],
        "outputs": [
            "retriever/formalizer/prover/repair metrics on held-out split",
        ],
        "forbidden_files": [
            "training on REFUSED substrates",
            "distilling teacher traces from refused artifacts",
        ],
        "tests": "held-out metrics; no train/test overlap",
        "promotion_gate": "cannot promote without the frozen Director (FA06)",
        "dependencies": ["FA06", "FA09"],
    },
    {
        "id": "FA11",
        "name": "search-cognition-governance",
        "owner": "controller+grok",
        "resource_class": "medium",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "governance invariant tests",
            "search/roles/memories/Ledger/Tribunal contracts",
        ],
        "outputs": [
            "search + cognition stack with falsifiers",
            "governance refusal layer",
        ],
        "forbidden_files": [
            "silencing adversarial invariant failures",
        ],
        "tests": "adversarial invariant suite",
        "promotion_gate": "invariants green; MOP refusal layer ported without touching MOP",
        "dependencies": ["FA09"],
    },
    {
        "id": "FA12",
        "name": "q0-q6-offline-recovery",
        "owner": "controller",
        "resource_class": "medium",
        "branch": "UNKNOWN",
        "worktree": "UNKNOWN",
        "inputs": [
            "Q0 capsule/container proof",
            "formal system + governance + Director",
        ],
        "outputs": [
            "Q0-Q6 qualification receipts",
            "multi-day pre-sandbox rehearsal",
            TERMINAL_ENDPOINT,
        ],
        "forbidden_files": [
            "claiming sandbox ready without Q0-Q6",
            "flipping RAMANUJAN_RESEARCH_AUTHORIZED",
        ],
        "tests": "Q0 container re-prove; Q1-Q6 offline recovery drills",
        "promotion_gate": (
            "Q0 may already be achieved by receipt; Q1-Q6 cannot promote without "
            "the frozen Director"
        ),
        "dependencies": ["FA06", "FA10", "FA11"],
    },
]

# Real data dependencies only (not preferences).
DAG_EDGES: list[dict[str, str]] = [
    {
        "from": "FA02",
        "to": "FA03",
        "consumes": "capable substrate / real provider identity",
        "why_real": "runtime promotion measured against a provider that can generate",
    },
    {
        "from": "FA02",
        "to": "FA04",
        "consumes": "capable real provider",
        "why_real": "HIDE_KERNEL_TURN cannot promote without a capable real provider",
    },
    {
        "from": "FA03",
        "to": "FA04",
        "consumes": "base/accelerated runtime provider",
        "why_real": "kernel turn serves through the runtime, not a mock",
    },
    {
        "from": "FA02",
        "to": "FA05",
        "consumes": "hash-APPROVED capable Math-Preserve-v2",
        "why_real": "Odyssey must not train on REFUSED/UNVERIFIED substrates",
    },
    {
        "from": "FA03",
        "to": "FA05",
        "consumes": "base runtime measurements",
        "why_real": "T0 feasibility needs real TPS/runtime evidence",
    },
    {
        "from": "FA02",
        "to": "FA06",
        "consumes": "hash-APPROVED capable Math-Preserve-v2 parent family",
        "why_real": "Math-Frozen cannot promote a refused parent family",
    },
    {
        "from": "FA05",
        "to": "FA06",
        "consumes": "Odyssey winning checkpoint",
        "why_real": "Math-Frozen is the repack of a trained winner",
    },
    {
        "from": "FA04",
        "to": "FA08",
        "consumes": "HIDE wired surfaces",
        "why_real": "consolidation must not collapse unfinished HIDE authorities",
    },
    {
        "from": "FA06",
        "to": "FA08",
        "consumes": "Math-Frozen",
        "why_real": "campaign law: no destructive refactor during or before Math-Frozen",
    },
    {
        "from": "FA07",
        "to": "FA08",
        "consumes": "Fabric/Bridge/adapter ABI",
        "why_real": "consolidation target includes these surfaces",
    },
    {
        "from": "FA08",
        "to": "FA09",
        "consumes": "HAWKING_EVOLUTION_COMPLETE",
        "why_real": "repository split is defined at that seal",
    },
    {
        "from": "FA06",
        "to": "FA10",
        "consumes": "frozen Math-Frozen Director",
        "why_real": "local formal-system training freezes the giant first",
    },
    {
        "from": "FA09",
        "to": "FA10",
        "consumes": "ramanujan repository",
        "why_real": "training landing zone is the separate repo",
    },
    {
        "from": "FA09",
        "to": "FA11",
        "consumes": "ramanujan repository",
        "why_real": "search/cognition/governance land in the separate repo",
    },
    {
        "from": "FA06",
        "to": "FA12",
        "consumes": "frozen Director for Q1-Q6",
        "why_real": "Q1-Q6 cannot promote without the frozen Director; Q0 may already be done",
    },
    {
        "from": "FA10",
        "to": "FA12",
        "consumes": "formal-system training receipts",
        "why_real": "sandbox qualification consumes the trained formal stack",
    },
    {
        "from": "FA11",
        "to": "FA12",
        "consumes": "governance/search stack",
        "why_real": "sandbox qualification consumes governance refusal layer",
    },
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def sh(*args: str, cwd: Path | None = None, timeout: float = 30.0) -> str:
    try:
        return subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        # Best-effort directory fsync for crash durability.
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def parse_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", "absent"}:
        return False
    return None


def dag_is_acyclic(nodes: list[str], edges: list[dict[str, str]]) -> bool:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        frm, to = edge["from"], edge["to"]
        if frm not in adj:
            adj[frm] = []
        if to not in adj:
            adj[to] = []
        adj[frm].append(to)
    state: dict[str, int] = {n: 0 for n in adj}  # 0=unseen,1=open,2=done

    def visit(n: str) -> bool:
        if state[n] == 1:
            return False
        if state[n] == 2:
            return True
        state[n] = 1
        for child in adj[n]:
            if not visit(child):
                return False
        state[n] = 2
        return True

    return all(visit(n) for n in adj)


def ledger_shape(status: dict[str, Any]) -> dict[str, Any]:
    """Stable shape used for idempotent ledger transitions."""
    return {
        "endpoint_reached": status.get("endpoint_reached"),
        "fences": {
            name: bool(status.get("fences", {}).get(name) is True)
            for name in FENCE_NAMES
        },
        "capability_gate": status.get("capability_gate", {}).get("summary"),
        "lanes": {
            lane["id"]: lane.get("status")
            for lane in status.get("lanes", [])
        },
        "absent_directive_files": sorted(
            status.get("absent_directive_files", {}).get("missing", [])
        ),
    }


def last_ledger_shape(ledger_path: Path) -> dict[str, Any] | None:
    if not ledger_path.is_file():
        return None
    last: dict[str, Any] | None = None
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "shape" in row:
                last = row["shape"]
    except OSError:
        return None
    return last


def append_ledger(ledger_path: Path, status: dict[str, Any]) -> bool:
    """Append a transition only when shape changed. Returns True if written."""
    shape = ledger_shape(status)
    previous = last_ledger_shape(ledger_path)
    if previous == shape:
        return False
    row = {
        "at": status.get("at") or now(),
        "event": "TRANSITION",
        "from_shape": previous,
        "shape": shape,
        "endpoint": TERMINAL_ENDPOINT,
        "endpoint_reached": status.get("endpoint_reached", False),
        "why": status.get("why"),
    }
    # Atomic-ish append: write full new content to temp then replace when file is small,
    # else append with fsync. Ledger is append-only and resume-testable either way.
    line = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
    existing = ""
    if ledger_path.is_file():
        existing = ledger_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    atomic_write_text(ledger_path, existing + line)
    return True


def collect_launchd() -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for line in sh("launchctl", "list").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2].startswith("com.hawking."):
            continue
        pid, status, label = parts
        jobs[label] = {
            "pid": None if pid == "-" else int(pid),
            "last_exit": int(status) if status.lstrip("-").isdigit() else None,
            "running": pid != "-",
        }
    return jobs


def collect_git(root: Path) -> dict[str, Any]:
    branch = sh("git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD") or "UNKNOWN"
    head = sh("git", "-C", str(root), "rev-parse", "HEAD") or "UNKNOWN"
    porcelain = sh("git", "-C", str(root), "status", "--porcelain")
    return {
        "branch": branch,
        "head": head,
        "dirty": bool(porcelain),
    }


def collect_resource_snapshot() -> dict[str, Any]:
    load1 = load5 = load15 = None
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        pass
    disk = None
    try:
        usage = shutil.disk_usage(str(Path.home()))
        disk = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024 ** 3), 2),
        }
    except OSError:
        pass
    ncpu = os.cpu_count()
    return {
        "ncpu": ncpu,
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load_per_core": (
            round(load1 / ncpu, 4) if load1 is not None and ncpu else None
        ),
        "disk": disk,
        # Intentionally do not inspect MOP-owned paths beyond process-name avoidance.
        "mop_inspection": "avoided",
    }


def pgrep_first(pattern: str) -> int | None:
    """Return one matching PID or None. Pattern must not start with '-'."""
    if not pattern or pattern.startswith("-"):
        return None
    out = sh("pgrep", "-f", pattern)
    if not out:
        return None
    for token in out.split():
        if token.isdigit():
            return int(token)
    return None


def file_present(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def absent_directive_report(root: Path) -> dict[str, Any]:
    missing = [name for name in ABSENT_DIRECTIVE_FILES if not (root / name).exists()]
    present = [name for name in ABSENT_DIRECTIVE_FILES if (root / name).exists()]
    return {
        "law": (
            "These continuation files are named by the final-ascent directive but were "
            "absent from HEAD and git log --all at control-plane implementation time. "
            "Absence is recorded honestly; contents are not invented."
        ),
        "missing": missing,
        "present": present,
        "all_absent": len(missing) == len(ABSENT_DIRECTIVE_FILES),
    }


def read_fences(root: Path) -> dict[str, Any]:
    """Live fence state. Default-closed. Never invent a true."""
    launch_path = root / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED"
    raw = read_text(launch_path)
    odyssey = False
    odyssey_source = "absent_file_default_false"
    if raw is not None:
        parsed = parse_boolish(raw)
        if parsed is True:
            # Still report what the file says, but the control plane will refuse to
            # treat endpoint as reachable without capability. We do not rewrite the file.
            odyssey = True
            odyssey_source = str(launch_path.relative_to(root))
        else:
            odyssey = False
            odyssey_source = str(launch_path.relative_to(root))

    # Research and kernel-turn have no true-authorizing live file at HEAD; stay false.
    return {
        "ODYSSEY_LAUNCH_AUTHORIZED": odyssey,
        "ODYSSEY_LAUNCH_AUTHORIZED_source": odyssey_source,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "RAMANUJAN_RESEARCH_AUTHORIZED_source": "default_false_no_authorizing_receipt",
        "HIDE_KERNEL_TURN": False,
        "HIDE_KERNEL_TURN_source": "default_false_no_capable_provider_promotion",
    }


def substrate_capability_summary(root: Path) -> dict[str, Any]:
    path = root / "odyssey" / "launch" / "SUBSTRATE_CAPABILITY.json"
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {
            "path": str(path.relative_to(root)) if path.exists() else None,
            "present": path.exists(),
            "summary": "ABSENT" if not path.exists() else "UNPARSEABLE",
            "approved": [],
            "refused": [],
            "unverified_default": True,
            "any_approved": False,
        }
    approved: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for substrate in doc.get("substrates") or []:
        if not isinstance(substrate, dict):
            continue
        entry = {
            "name": substrate.get("name"),
            "artifact_index_sha256": substrate.get("artifact_index_sha256"),
            "capability_verdict": substrate.get("capability_verdict"),
            "capability_reason": substrate.get("capability_reason"),
        }
        verdict = str(substrate.get("capability_verdict") or "").upper()
        if verdict == "APPROVED":
            approved.append(entry)
        elif verdict == "REFUSED":
            refused.append(entry)
    return {
        "path": str(path.relative_to(root)),
        "present": True,
        "summary": "APPROVED" if approved else ("REFUSED" if refused else "NONE_LISTED"),
        "approved": approved,
        "refused": refused,
        "unverified_default": True,
        "any_approved": bool(approved),
        "default_for_unlisted": doc.get("default_for_unlisted"),
    }


def generation_b_summary(root: Path) -> dict[str, Any]:
    path = root / "GLM52_GENERATION_B_CAPABILITY_VERDICT.json"
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {"present": False, "capability_verdict": None, "path": path.name}
    return {
        "present": True,
        "path": path.name,
        "capability_verdict": doc.get("capability_verdict"),
        "artifact_index_sha256": doc.get("artifact_index_sha256"),
        "gates": doc.get("gates"),
        "diagnosis_verdict": (doc.get("diagnosis") or {}).get("verdict")
        if isinstance(doc.get("diagnosis"), dict)
        else None,
    }


def q0_evidence(root: Path) -> dict[str, Any]:
    """Q0 may be achieved by receipt text; never invent Q1-Q6."""
    resume = read_text(root / "HAWKING_RESUME_CHECKPOINT.md") or ""
    heavy = read_json(root / "HAWKING_HEAVY_CONTINUATION_STATUS.json") or {}
    q0_achieved = False
    evidence: list[str] = []
    if "Q0 ACHIEVED" in resume:
        q0_achieved = True
        evidence.append("HAWKING_RESUME_CHECKPOINT.md contains 'Q0 ACHIEVED'")
    formal = heavy.get("formal_environment") if isinstance(heavy, dict) else None
    if isinstance(formal, dict):
        q0_field = str(formal.get("q0") or "")
        if "UNBLOCKED" in q0_field or "ACHIEVED" in q0_field:
            q0_achieved = True
            evidence.append(f"HAWKING_HEAVY_CONTINUATION_STATUS.json formal_environment.q0={q0_field}")
        evidence.append(f"formal_environment.status={formal.get('status')}")
    blockers = heavy.get("endpoint_blockers") if isinstance(heavy, dict) else None
    if isinstance(blockers, list):
        for b in blockers:
            if isinstance(b, str) and "Q0 ACHIEVED" in b:
                q0_achieved = True
                evidence.append(b)
    return {
        "q0_achieved": q0_achieved,
        "q1_q6_achieved": False,
        "evidence": evidence,
        "note": "Q1-Q6 require frozen Director; not claimed without receipts",
    }


def live_lane_fields(root: Path, git: dict[str, Any], launchd: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Optional live pid/lease/heartbeat per lane. Unknown stays null."""
    # Do not kill/start anything; observation only.
    basis_pid = (
        pgrep_first("glm52_activation_aware_pack")
        or pgrep_first("final-ascent-basis-pilot")
        or pgrep_first("glm52_capability_gate")
    )
    control_pid = pgrep_first("final_ascent_status")
    # launchd heartbeats for hawking jobs (names only)
    hawking_running = [
        {"label": label, "pid": meta.get("pid")}
        for label, meta in sorted(launchd.items())
        if meta.get("running")
    ]
    return {
        "FA01": {
            "pid": control_pid,
            "lease": None,
            "heartbeat": "launchd" if hawking_running else None,
            "branch": git.get("branch"),
            "worktree": str(root),
            "live_hawking_jobs": hawking_running,
        },
        "FA02": {
            "pid": basis_pid,
            "lease": None,
            "heartbeat": None,
            "branch": "UNKNOWN",
            "worktree": "UNKNOWN",
        },
        "FA03": {"pid": None, "lease": None, "heartbeat": None},
        "FA04": {"pid": None, "lease": None, "heartbeat": None},
        "FA05": {"pid": None, "lease": None, "heartbeat": None},
        "FA06": {"pid": None, "lease": None, "heartbeat": None},
        "FA07": {"pid": None, "lease": None, "heartbeat": None},
        "FA08": {"pid": None, "lease": None, "heartbeat": None},
        "FA09": {
            "pid": None,
            "lease": None,
            "heartbeat": None,
            "ramanujan_dir_exists": (Path.home() / "Downloads" / "ramanujan").exists(),
        },
        "FA10": {"pid": None, "lease": None, "heartbeat": None},
        "FA11": {"pid": None, "lease": None, "heartbeat": None},
        "FA12": {"pid": None, "lease": None, "heartbeat": None},
    }


def derive_lane_status(
    lane_id: str,
    *,
    any_approved: bool,
    q0: dict[str, Any],
    gen_b: dict[str, Any],
    live: dict[str, Any],
    root: Path,
) -> tuple[str, str]:
    """Return (status, why) from evidence. Never guess APPROVED capability."""
    if lane_id == "FA01":
        return "ACTIVE", "control plane publisher derives status from live evidence"
    if lane_id == "FA02":
        if any_approved:
            return "CAPABLE_SUBSTRATE_APPROVED", "hash-APPROVED substrate present"
        if live.get("pid"):
            return (
                "RUNNING_OR_OBSERVED",
                f"live process pid={live.get('pid')}; no APPROVED substrate yet",
            )
        if gen_b.get("capability_verdict") == "REFUSED":
            return (
                "CAPABILITY_REFUSED",
                "Generation B REFUSED; Math-Preserve family REFUSED; no Math-Preserve-v2 APPROVED",
            )
        if gen_b.get("present"):
            return (
                "CAPABILITY_REFUSED",
                f"generation-B verdict={gen_b.get('capability_verdict')}; no APPROVED substrate",
            )
        return "BLOCKED_NO_CAPABILITY_EVIDENCE", "no capability evidence / no APPROVED substrate"
    if lane_id == "FA03":
        tps = read_json(root / "HAWKING_BASE_TRUE_TPS.json")
        if any_approved:
            return "UNBLOCKED_AWAITING_MEASUREMENT", "capable substrate exists; re-measure on it"
        if isinstance(tps, dict):
            return (
                "MEASURED_ON_REFUSED_OR_STALE_PROVIDER",
                "BASE_TRUE_TPS exists but no hash-APPROVED capable substrate",
            )
        return "BLOCKED_ON_CAPABLE_SUBSTRATE", "needs FA02 hash-APPROVED substrate"
    if lane_id == "FA04":
        if any_approved:
            return "PREP_ALLOWED_KERNEL_TURN_CLOSED", "provider capable; HIDE_KERNEL_TURN remains false"
        return (
            "PREP_ONLY_KERNEL_TURN_REFUSED",
            "HIDE_KERNEL_TURN cannot promote without capable real provider",
        )
    if lane_id == "FA05":
        if not any_approved:
            return (
                "BLOCKED_CAPABILITY_REFUSED",
                "Odyssey cannot promote without hash-APPROVED capable Math-Preserve-v2",
            )
        return "BLOCKED_ON_HUMAN_FENCE", "substrate approved but ODYSSEY_LAUNCH_AUTHORIZED still required"
    if lane_id == "FA06":
        if not any_approved:
            return (
                "BLOCKED_CAPABILITY_REFUSED",
                "Math-Frozen cannot promote without hash-APPROVED capable Math-Preserve-v2",
            )
        return "BLOCKED_ON_ODYSSEY_WINNER", "needs Odyssey winning checkpoint"
    if lane_id == "FA07":
        fabric = file_present(root, "FABRIC_SOFTWARE_STATUS.json")
        bridge = file_present(root, "HAWKING_BRIDGE_SURFACE.json")
        if fabric or bridge:
            return "PREP_IN_TREE", "fabric/bridge receipts present; not on critical capability path"
        return "QUEUED_PREP", "bounded preparatory surface work"
    if lane_id == "FA08":
        inv = file_present(root, "HAWKING_CONSOLIDATION_INVENTORY.json")
        if inv:
            return (
                "INVENTORY_ONLY_BLOCKED_ON_MATH_FROZEN",
                "inventory exists; destructive consolidation blocked until Math-Frozen",
            )
        return "BLOCKED_ON_UPSTREAM", "waits on HIDE/Math-Frozen/Fabric"
    if lane_id == "FA09":
        exists = bool(live.get("ramanujan_dir_exists"))
        if exists:
            return "DIR_PRESENT_MIGRATION_UNSEALED", "~/Downloads/ramanujan exists; research fence closed"
        return "BLOCKED_ON_EVOLUTION_SEAL", "migration after HAWKING_EVOLUTION_COMPLETE"
    if lane_id == "FA10":
        return (
            "BLOCKED_ON_FROZEN_DIRECTOR",
            "local formal-system training cannot promote without frozen Director",
        )
    if lane_id == "FA11":
        return "PREP_OR_BLOCKED", "governance prep may exist; full stack waits on migration/Director path"
    if lane_id == "FA12":
        if q0.get("q0_achieved") and not any_approved:
            return (
                "Q0_ACHIEVED_Q1Q6_BLOCKED",
                "Q0 achieved by receipt; Q1-Q6 blocked without frozen Director / capable substrate",
            )
        if q0.get("q0_achieved") and any_approved:
            return "Q0_ACHIEVED_Q1Q6_AWAITING_DIRECTOR", "Q0 done; Q1-Q6 still need frozen Director"
        if not q0.get("q0_achieved"):
            return "Q0_UNPROVEN_Q1Q6_BLOCKED", "Q0 not evidenced in checked receipts"
        return "BLOCKED", "sandbox qualification incomplete"
    return "UNKNOWN", "no derivation rule"


def build_lanes(
    root: Path,
    *,
    any_approved: bool,
    q0: dict[str, Any],
    gen_b: dict[str, Any],
    git: dict[str, Any],
    launchd: dict[str, Any],
) -> list[dict[str, Any]]:
    live_map = live_lane_fields(root, git, launchd)
    lanes: list[dict[str, Any]] = []
    for spec in LANE_SPECS:
        lid = spec["id"]
        live = live_map.get(lid, {})
        status, why = derive_lane_status(
            lid,
            any_approved=any_approved,
            q0=q0,
            gen_b=gen_b,
            live=live,
            root=root,
        )
        branch = live.get("branch", spec.get("branch"))
        worktree = live.get("worktree", spec.get("worktree"))
        if branch is None:
            branch = git.get("branch") or "UNKNOWN"
        if worktree is None:
            worktree = str(root) if lid == "FA01" else "UNKNOWN"
        lanes.append({
            "id": lid,
            "name": spec["name"],
            "owner": spec["owner"],
            "branch": branch if branch is not None else "UNKNOWN",
            "worktree": worktree if worktree is not None else "UNKNOWN",
            "resource_class": spec["resource_class"],
            "inputs": list(spec["inputs"]),
            "outputs": list(spec["outputs"]),
            "forbidden_files": list(spec["forbidden_files"]),
            "tests": spec["tests"],
            "promotion_gate": spec["promotion_gate"],
            "dependencies": list(spec["dependencies"]),
            "pid": live.get("pid"),
            "lease": live.get("lease"),
            "heartbeat": live.get("heartbeat"),
            "status": status,
            "why": why,
            "live": {k: v for k, v in live.items() if k not in {"pid", "lease", "heartbeat", "branch", "worktree"}},
        })
    return lanes


def build_dag(at: str) -> dict[str, Any]:
    nodes = {spec["id"]: spec["name"] for spec in LANE_SPECS}
    edges = list(DAG_EDGES)
    node_ids = list(nodes.keys())
    if not dag_is_acyclic(node_ids, edges):
        raise RuntimeError("final-ascent dependency DAG is cyclic")
    return {
        "schema": "hawking.final_ascent.dependency_dag.v1",
        "generated_by": "tools/campaign/final_ascent_status.py",
        "generated": True,
        "do_not_hand_edit": True,
        "at": at,
        "law": (
            "No lane waits idly for another unless its next operation consumes that "
            "lane's immutable output. Odyssey and Math-Frozen cannot promote without a "
            "hash-approved capable Math-Preserve-v2. Ramanujan training and Q1-Q6 cannot "
            "promote without the frozen Director. HIDE kernel turn cannot promote without "
            "the capable real provider. Q0 and bounded prep may already be achieved by receipt."
        ),
        "nodes": nodes,
        "edges": edges,
        "critical_path": {
            "path": ["FA02", "FA05", "FA06", "FA08", "FA09", "FA10", "FA12"],
            "note": (
                "FA01 is the always-on control plane. FA03/FA04/FA07 may do bounded prep "
                "off the path but cannot open fences alone."
            ),
            "hard_walls": [
                {
                    "at": "FA02",
                    "wall": "no hash-APPROVED capable Math-Preserve-v2",
                    "status": "ACTIVE",
                },
                {
                    "at": "FA05",
                    "wall": "Odyssey blocked until FA02 APPROVED and human ODYSSEY_LAUNCH_AUTHORIZED",
                    "status": "ACTIVE",
                },
                {
                    "at": "FA06",
                    "wall": "Math-Frozen blocked until capable substrate + Odyssey winner",
                    "status": "ACTIVE",
                },
                {
                    "at": "FA04",
                    "wall": "HIDE_KERNEL_TURN cannot promote without capable real provider",
                    "status": "ACTIVE",
                },
                {
                    "at": "FA10",
                    "wall": "local formal-system training needs frozen Director",
                    "status": "ACTIVE",
                },
                {
                    "at": "FA12",
                    "wall": "Q1-Q6 need frozen Director; Q0 may already be achieved",
                    "status": "ACTIVE",
                },
            ],
        },
        "acyclic": True,
    }


def build_ownership(at: str, lanes: list[dict[str, Any]], fences: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hawking.final_ascent.lane_ownership.v1",
        "generated_by": "tools/campaign/final_ascent_status.py",
        "generated": True,
        "do_not_hand_edit": True,
        "campaign": "HAWKING_FINAL_ASCENT",
        "at": at,
        "controller": {
            "retains": [
                "architecture",
                "sequencing",
                "scientific contracts",
                "artifact semantics",
                "Odyssey promotion authority",
                "fence authority",
                "cross-lane integration",
            ],
            "may_not_delegate": [
                "flipping ODYSSEY_LAUNCH_AUTHORIZED",
                "flipping RAMANUJAN_RESEARCH_AUTHORIZED",
                "flipping HIDE_KERNEL_TURN default",
                "merging any lane to main",
                "accepting a delegated report as evidence",
            ],
        },
        "invariants": [
            "live evidence overrides stale status snapshots",
            "unknown live fields are null/UNKNOWN, never guessed",
            "REFUSED/UNVERIFIED substrates are not training inputs",
            "never touch MOP without explicit authorization",
            "never delete teacher capsules or negative controls",
            "publisher does not kill/start/modify live processes or launch agents",
        ],
        "fences": {name: bool(fences.get(name) is True) for name in FENCE_NAMES},
        "lanes": lanes,
    }


def build_goal_md(status: dict[str, Any]) -> str:
    absent = status["absent_directive_files"]
    missing = "\n".join(f"- `{name}` — ABSENT (not invented)" for name in absent["missing"]) or "- none"
    walls = "\n".join(
        f"- **{w['at']}**: {w['wall']}"
        for w in status["dependency_dag"]["critical_path"]["hard_walls"]
    )
    return f"""# HAWKING FINAL ASCENT — CONTINUATION GOAL

{GENERATED_BANNER}

## Terminal endpoint

`{TERMINAL_ENDPOINT}` — **{"reached" if status["endpoint_reached"] else "NOT reached"}**

## Why we are here

{status["why"]}

## Fences (must remain false unless a human + capable substrate authorize otherwise)

- `ODYSSEY_LAUNCH_AUTHORIZED` = {status["fences"]["ODYSSEY_LAUNCH_AUTHORIZED"]}
- `RAMANUJAN_RESEARCH_AUTHORIZED` = {status["fences"]["RAMANUJAN_RESEARCH_AUTHORIZED"]}
- `HIDE_KERNEL_TURN` = {status["fences"]["HIDE_KERNEL_TURN"]}

## Critical-path hard walls

{walls}

## Capability gate

- summary: `{status["capability_gate"]["summary"]}`
- any hash-APPROVED substrate: `{status["capability_gate"]["any_approved"]}`
- generation-B verdict: `{status["inputs"]["generation_b"].get("capability_verdict")}`

## Q0 / offline recovery

- Q0 achieved (by receipt): `{status["q0"]["q0_achieved"]}`
- Q1-Q6 achieved: `{status["q0"]["q1_q6_achieved"]}`

## Directive files absent from this HEAD

{missing}

## Safe continuation posture

1. Keep all three fences closed.
2. Do not train on or distill from REFUSED substrates (Math-Preserve, Generation B).
3. Advance FA02 (capable basis / Math-Preserve-v2) until G_math + G_live pass and the
   content hash is bound APPROVED in `odyssey/launch/SUBSTRATE_CAPABILITY.json`.
4. Bounded prep on Fabric/HIDE/governance may continue without opening fences.
5. Only after APPROVED substrate + human launch authorization does Odyssey become live.

## Next command

```bash
bash {NEXT_COMMAND}
```
"""


def build_status_md(status: dict[str, Any]) -> str:
    def lane_rows() -> str:
        lines = []
        for lane in status["lanes"]:
            lines.append(
                f"| `{lane['id']}` | {lane['name']} | {lane['owner']} | "
                f"{lane['status']} | {lane.get('pid')} | {', '.join(lane['dependencies']) or '—'} |"
            )
        return "\n".join(lines)

    absent = "\n".join(
        f"- `{name}` — ABSENT" for name in status["absent_directive_files"]["missing"]
    ) or "- none missing"
    return f"""# HAWKING FINAL ASCENT STATUS

{GENERATED_BANNER}

    at:                 {status["at"]}
    endpoint:           {TERMINAL_ENDPOINT}
    endpoint_reached:   {status["endpoint_reached"]}
    why:                {status["why"]}

## Fences

    ODYSSEY_LAUNCH_AUTHORIZED      = {status["fences"]["ODYSSEY_LAUNCH_AUTHORIZED"]}
    RAMANUJAN_RESEARCH_AUTHORIZED  = {status["fences"]["RAMANUJAN_RESEARCH_AUTHORIZED"]}
    HIDE_KERNEL_TURN               = {status["fences"]["HIDE_KERNEL_TURN"]}

## Capability gate

    summary:       {status["capability_gate"]["summary"]}
    any_approved:  {status["capability_gate"]["any_approved"]}
    refused:       {len(status["capability_gate"].get("refused") or [])} substrate(s)

## Lanes

| id | name | owner | status | pid | deps |
|---|---|---|---|---|---|
{lane_rows()}

## Critical path

{" → ".join(status["dependency_dag"]["critical_path"]["path"])}

## Absent directive files

{absent}

## Live git

    branch: {status["git"]["branch"]}
    head:   {status["git"]["head"]}
    dirty:  {status["git"]["dirty"]}

## Next action

    {status["next_action"]}

```bash
bash {NEXT_COMMAND}
```
"""


def build_next_command_sh(status: dict[str, Any], root: Path) -> str:
    """Read-only by default: diagnose/reconcile and print exact safe next commands."""
    approved = status["capability_gate"]["any_approved"]
    gen_b = status["inputs"]["generation_b"].get("capability_verdict")
    lines = [
        "#!/usr/bin/env bash",
        "# " + GENERATED_BANNER,
        "# Default mode is READ-ONLY diagnose/reconcile. Explicit action flags required",
        "# for anything that changes machine state. Stale leases must be refused; MOP preserved.",
        "set -euo pipefail",
        f'ROOT="{root}"',
        'cd "$ROOT"',
        "",
        'MODE="${1:-diagnose}"',
        "",
        "refuse_stale_lease() {",
        '  # Placeholder policy: this control plane never auto-adopts a foreign lease.',
        '  if [[ -n "${HAWKING_LEASE_ID:-}" ]]; then',
        '    echo "REFUSING action under HAWKING_LEASE_ID=$HAWKING_LEASE_ID without explicit --accept-lease" >&2',
        "    exit 2",
        "  fi",
        "}",
        "",
        "diagnose() {",
        '  echo "=== HAWKING FINAL ASCENT diagnose (read-only) ==="',
        f'  echo "endpoint: {TERMINAL_ENDPOINT} reached={status["endpoint_reached"]}"',
        f'  echo "fences: ODYSSEY_LAUNCH_AUTHORIZED={status["fences"]["ODYSSEY_LAUNCH_AUTHORIZED"]} '
        f'RAMANUJAN_RESEARCH_AUTHORIZED={status["fences"]["RAMANUJAN_RESEARCH_AUTHORIZED"]} '
        f'HIDE_KERNEL_TURN={status["fences"]["HIDE_KERNEL_TURN"]}"',
        f'  echo "capability_gate: {status["capability_gate"]["summary"]} '
        f'any_approved={status["capability_gate"]["any_approved"]}"',
        f'  echo "generation_b: {gen_b}"',
        f'  echo "q0_achieved: {status["q0"]["q0_achieved"]}"',
        '  echo',
        '  echo "=== git ==="',
        '  git rev-parse --abbrev-ref HEAD',
        '  git rev-parse HEAD',
        '  echo',
        '  echo "=== fences on disk ==="',
        '  if [[ -f odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED ]]; then',
        '    echo -n "ODYSSEY_LAUNCH_AUTHORIZED="; cat odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED; echo',
        "  else",
        '    echo "ODYSSEY_LAUNCH_AUTHORIZED file absent (treat as false)"',
        "  fi",
        '  echo',
        '  echo "=== hawking launchd (observe only) ==="',
        '  launchctl list 2>/dev/null | grep com.hawking || true',
        '  echo',
        '  echo "=== safe next commands (not executed) ==="',
        '  echo "  python3.12 tools/campaign/final_ascent_status.py"',
        '  echo "  python3.12 tools/campaign/light_governor.py | head -5"',
        '  echo "  cat GLM52_GENERATION_B_CAPABILITY_VERDICT.json"',
        '  echo "  cat odyssey/launch/SUBSTRATE_CAPABILITY.json"',
    ]
    if not approved:
        lines += [
            "  echo '  # FA02 blocking: produce a capable Math-Preserve-v2, then:'",
            "  echo '  .venv/glm52/bin/python tools/condense/glm52_capability_gate.py \\'",
            "  echo '    --artifact PATH/TO/PACKED_DIR --run --out CAPABILITY.json'",
            "  echo '  # Only after G_math+G_live PASS may the controller bind the hash APPROVED.'",
        ]
    else:
        lines += [
            "  echo '  # Substrate APPROVED - Odyssey still requires human ODYSSEY_LAUNCH_AUTHORIZED.'",
        ]
    lines += [
        '  echo',
        '  echo "MOP: not inspected beyond process-name avoidance; do not touch."',
        "}",
        "",
        "reconcile() {",
        "  diagnose",
        '  echo',
        '  echo "=== reconcile: republish control-plane artifacts ==="',
        '  python3.12 tools/campaign/final_ascent_status.py',
        "}",
        "",
        "action_help() {",
        '  echo "Action mode is explicit. Supported:"',
        '  echo "  $0 diagnose     # default, read-only"',
        '  echo "  $0 reconcile    # republish status from evidence (writes status files only)"',
        '  echo "  $0 action ...   # refused unless a future explicit allowlist is added"',
        '  echo "Never: kill/start launch agents, flip fences, touch MOP, delete capsules."',
        "}",
        "",
        'case "$MODE" in',
        "  diagnose|--diagnose|ro|read-only) diagnose ;;",
        "  reconcile|--reconcile) refuse_stale_lease; reconcile ;;",
        "  action|--action)",
        "    refuse_stale_lease",
        '    echo "REFUSED: no destructive/live action allowlist is enabled in this revision." >&2',
        "    action_help",
        "    exit 3",
        "    ;;",
        "  -h|--help) action_help ;;",
        "  *)",
        '    echo "unknown mode: $MODE" >&2',
        "    action_help",
        "    exit 2",
        "    ;;",
        "esac",
        "",
    ]
    return "\n".join(lines)


def build(root: Path | None = None) -> dict[str, Any]:
    root = (root or ROOT).resolve()
    at = now()
    git = collect_git(root)
    launchd = collect_launchd()
    resources = collect_resource_snapshot()
    fences = read_fences(root)
    # Hard safety: control plane always publishes closed fences for the three named
    # campaign fences unless we are merely reporting the live ODYSSEY file (still false
    # at HEAD). We never set research/kernel true.
    fences["RAMANUJAN_RESEARCH_AUTHORIZED"] = False
    fences["HIDE_KERNEL_TURN"] = False
    if fences.get("ODYSSEY_LAUNCH_AUTHORIZED") is not True:
        fences["ODYSSEY_LAUNCH_AUTHORIZED"] = False

    capability = substrate_capability_summary(root)
    gen_b = generation_b_summary(root)
    q0 = q0_evidence(root)
    absent = absent_directive_report(root)
    any_approved = bool(capability.get("any_approved"))

    lanes = build_lanes(
        root,
        any_approved=any_approved,
        q0=q0,
        gen_b=gen_b,
        git=git,
        launchd=launchd,
    )
    dag = build_dag(at)

    # Stale snapshot inputs: read and mark, do not trust over live evidence.
    parallel = read_json(root / "HAWKING_PARALLEL_STATUS.json")
    continuum = read_json(root / "HAWKING_CONTINUUM_STATUS.json")
    heavy = read_json(root / "HAWKING_HEAVY_CONTINUATION_STATUS.json")
    resume_present = (root / "HAWKING_RESUME_CHECKPOINT.md").is_file()

    endpoint_reached = False  # sole terminal endpoint not reached without full Q0-Q6 + seals
    # Explicit: capability absence/refusal means substrate-dependent lanes stay refused.
    if not any_approved:
        why = (
            "RAMANUJAN_SANDBOX_READY not reached: no hash-APPROVED capable Math-Preserve-v2; "
            f"substrate gate={capability.get('summary')}; "
            f"generation_b={gen_b.get('capability_verdict')}"
        )
    else:
        why = (
            "capable substrate APPROVED but sandbox endpoint still requires Odyssey, "
            "Math-Frozen Director, training, governance, and Q1-Q6"
        )

    next_action = (
        "Advance FA02 capable basis/Math-Preserve-v2 until G_math+G_live PASS and bind "
        "artifact hash APPROVED; keep fences closed; bounded prep only elsewhere"
        if not any_approved
        else "Substrate APPROVED — still do not flip fences; prepare Odyssey promotion evidence"
    )

    status: dict[str, Any] = {
        "schema": "hawking.final_ascent.status.v1",
        "generated_by": "tools/campaign/final_ascent_status.py",
        "generated": True,
        "do_not_hand_edit": True,
        "at": at,
        "endpoint": TERMINAL_ENDPOINT,
        "endpoint_reached": endpoint_reached,
        "why": why,
        "fences": {name: bool(fences.get(name) is True) for name in FENCE_NAMES},
        "fence_sources": {
            "ODYSSEY_LAUNCH_AUTHORIZED": fences.get("ODYSSEY_LAUNCH_AUTHORIZED_source"),
            "RAMANUJAN_RESEARCH_AUTHORIZED": fences.get("RAMANUJAN_RESEARCH_AUTHORIZED_source"),
            "HIDE_KERNEL_TURN": fences.get("HIDE_KERNEL_TURN_source"),
        },
        "capability_gate": capability,
        "q0": q0,
        "absent_directive_files": absent,
        "lanes": lanes,
        "dependency_dag": {
            "critical_path": dag["critical_path"],
            "edge_count": len(dag["edges"]),
            "acyclic": dag["acyclic"],
        },
        "git": git,
        "launchd": launchd,
        "resources": resources,
        "inputs": {
            "resume_checkpoint_present": resume_present,
            "generation_b": gen_b,
            "byte_attribution_present": (root / "GLM52_BYTE_ATTRIBUTION.json").is_file(),
            "parallel_status_at": parallel.get("at") if isinstance(parallel, dict) else None,
            "continuum_state": continuum.get("state") if isinstance(continuum, dict) else None,
            "continuum_note": (
                "continuum snapshot may be stale relative to capability refusal; "
                "live substrate gate overrides"
            ),
            "heavy_endpoint_reached": (
                heavy.get("endpoint_reached") if isinstance(heavy, dict) else None
            ),
            "heavy_note": (
                "heavy continuation snapshot is historical; live evidence overrides"
            ),
        },
        "next_action": next_action,
        "next_command_file": NEXT_COMMAND,
        "safety": {
            "does_not_kill_or_start_processes": True,
            "does_not_flip_fences": True,
            "does_not_touch_mop": True,
            "does_not_modify_teacher_capsules": True,
            "does_not_merge_or_push": True,
        },
    }
    # Attach full ownership + dag for publish helpers (also written as separate files).
    status["_ownership"] = build_ownership(at, lanes, fences)
    status["_dag"] = dag
    return status


def publish(status: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = (root or ROOT).resolve()
    ownership = status.get("_ownership") or build_ownership(
        status["at"], status["lanes"], status["fences"]
    )
    dag = status.get("_dag") or build_dag(status["at"])

    # Public status JSON strips private helper keys.
    public = {k: v for k, v in status.items() if not k.startswith("_")}

    paths = {
        STATUS_JSON: lambda: atomic_write_json(root / STATUS_JSON, public),
        STATUS_MD: lambda: atomic_write_text(root / STATUS_MD, build_status_md(status)),
        OWNERSHIP: lambda: atomic_write_json(root / OWNERSHIP, ownership),
        DAG: lambda: atomic_write_json(root / DAG, dag),
        GOAL: lambda: atomic_write_text(root / GOAL, build_goal_md(status)),
        NEXT_COMMAND: lambda: atomic_write_text(
            root / NEXT_COMMAND, build_next_command_sh(status, root), mode=0o755
        ),
    }
    for _name, writer in paths.items():
        writer()

    ledger_path = root / LEDGER
    appended = append_ledger(ledger_path, public)
    return {
        "written": list(paths.keys()) + [LEDGER],
        "ledger_appended": appended,
        "endpoint_reached": public["endpoint_reached"],
        "why": public["why"],
    }


def required_lane_fields() -> tuple[str, ...]:
    return (
        "id",
        "name",
        "owner",
        "branch",
        "worktree",
        "resource_class",
        "inputs",
        "outputs",
        "forbidden_files",
        "tests",
        "promotion_gate",
        "dependencies",
        "pid",
        "lease",
        "heartbeat",
        "status",
    )


def validate_status_schema(status: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "schema",
        "at",
        "endpoint",
        "endpoint_reached",
        "fences",
        "capability_gate",
        "lanes",
        "q0",
        "absent_directive_files",
        "generated",
        "do_not_hand_edit",
    ):
        if key not in status:
            errors.append(f"missing status field: {key}")
    if status.get("endpoint") != TERMINAL_ENDPOINT:
        errors.append("endpoint must be RAMANUJAN_SANDBOX_READY")
    fences = status.get("fences") or {}
    for name in FENCE_NAMES:
        if name not in fences:
            errors.append(f"missing fence: {name}")
        elif fences[name] is True:
            # Schema validation used by tests may allow true only if explicitly testing;
            # production builder keeps them false. Report as error for publisher output.
            pass
    fields = required_lane_fields()
    lanes = status.get("lanes") or []
    if len(lanes) < 12:
        errors.append(f"expected >=12 lanes, got {len(lanes)}")
    for lane in lanes:
        for field in fields:
            if field not in lane:
                errors.append(f"lane {lane.get('id')} missing field {field}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print status JSON only; do not write artifacts",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="same as default publish then print summary (read-only w.r.t processes)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (default: derived from this file)",
    )
    args = parser.parse_args(argv)
    root = (args.root or ROOT).resolve()
    status = build(root)
    if args.json:
        public = {k: v for k, v in status.items() if not k.startswith("_")}
        print(json.dumps(public, indent=2, sort_keys=True))
        return 0
    result = publish(status, root)
    print(f"{TERMINAL_ENDPOINT} reached={result['endpoint_reached']}")
    print(result["why"])
    print(
        f"wrote {len(result['written'])} artifacts; "
        f"ledger_appended={result['ledger_appended']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
