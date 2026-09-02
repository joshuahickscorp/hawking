"""Shared receipt + import helpers for ModelLake/Qwen27 acceptance.

The lake at /Volumes/corpdrive/hawking-modellake is read-only to this lane.
Never retire, move, or rewrite a specimen. Never signal the live HCLI daemon.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

RECEIPT_SCHEMA = "hawking.acceptance.gate.v1"
WORKTREE = Path(__file__).resolve().parents[3]
PRIMARY = Path("/Users/scammermike/Downloads/hawking")
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMENS = LAKE / "specimens"
PARTIAL = LAKE / "partial"
LAKE_MANIFESTS = LAKE / "manifests"
RECEIPTS = WORKTREE / "receipts" / "acceptance"
ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")

# Catalog implementing symbols (civilization/CAPABILITY_GRAPH.json).
GATES: dict[str, dict[str, Any]] = {
    "MODELLAKE_IDENTITY_RESOLVED": {
        "symbol": "hcli.agentos.modellake_gate.run_modellake_census",
        "call_name": "run_modellake_census",
        "acceptance_span": (531, 553),
        "ledger_line": 9487,
        "operational": (
            "H-ROADMAP.md §14: DISCOVERED → IDENTITY_RESOLVED → MANIFEST_READY → … "
            "Identity is resolved when repo and revision are known. Every sealed "
            "specimen must be at IDENTITY_RESOLVED or later, with repo+revision."
        ),
    },
    "MODELLAKE_HASH_VERIFIED": {
        "symbol": "tools.odyssey.modellake_watch.reconcile",
        "call_name": "reconcile",
        "acceptance_span": (531, 553),
        "ledger_line": 9488,
        "operational": (
            "MODE-003 hash verification. acquire() checks each file against the hub "
            "LFS oid (sha256); files without an oid are size-only and recorded as "
            "such. A specimen is only visible as complete once every declared file's "
            "sha256 matches the upstream LFS oid. Catalog symbol reconcile() is the "
            "size-completeness second look (not cryptographic)."
        ),
    },
    "MODELLAKE_ATOMIC_PROMOTION": {
        "symbol": "tools.odyssey.modellake_promote.promote",
        "call_name": "promote",
        "acceptance_span": (531, 553),
        "ledger_line": 9489,
        "operational": (
            "MODE-004 atomic promotion. os.rename of a verified-complete tree from "
            "partial/ into specimens/. Partial files never masquerade as final "
            "artifacts. Dry-run is default; an existing destination is never overwritten."
        ),
    },
    "QWEN27_RUNTIME_IDENTITY_FROZEN": {
        "symbol": "hcli.agentos.qwen27_runtime_identity.run_runtime_archaeology",
        "call_name": "run_runtime_archaeology",
        "acceptance_span": (506, 530),
        "ledger_line": 9490,
        "operational": (
            "§11.4.1 / QWEN-001: freeze exact historical best runtime identity — "
            "source commit, binary hash, Cargo profile/rustflags, model artifact, "
            "tokenizer, representation census, env, graph, dispatch count, "
            "capability contract and benchmark command. Unknowns stay UNKNOWN."
        ),
    },
    "QWEN27_PROTECTED_BASELINE": {
        "symbol": "hcli.agentos.protected_accelerator_benchmark.run_protected_accelerator_benchmark",
        "call_name": "run_protected_accelerator_benchmark",
        "acceptance_span": (506, 530),
        "ledger_line": 9491,
        "operational": (
            "§11.4.4 / Q27-03 / QWEN-003: watch for a protected quiescent window "
            "and run a current release-fast protected baseline. Qualification "
            "requires a quiet machine; a contended or unobservable machine is not "
            "a baseline."
        ),
    },
    "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED": {
        "symbol": "hcli.agentos.qwen27_mlp_diagnostic.run_qwen27_mlp_diagnostic_ab",
        "call_name": "run_qwen27_mlp_diagnostic_ab",
        "acceptance_span": (506, 530),
        "ledger_line": 9492,
        "operational": (
            "§11.4 / Q27-02 / QWEN-002: inspect MLP fusion parser (HAWKING_QWEN38_FUSE_MLP "
            "`swiglu` and `1` are the same strongest arm in source); diagnostic A/B "
            "only for graph identity; the gate passes if the current-vs-historical "
            "regression is explained or bounded. No performance promotion from a "
            "contaminated window."
        ),
    },
}


def receipts_dir() -> Path:
    """Look up RECEIPTS at call time so tests can monkeypatch it."""
    return RECEIPTS


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_roadmap(start: int, end: int) -> str:
    try:
        lines = ROADMAP.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    chunk = lines[start - 1 : end]
    return "\n".join(chunk)


def ensure_tools_path() -> None:
    """Keep this worktree first so `import tools` is ours, not the primary checkout's."""
    wt = str(WORKTREE)
    if sys.path and sys.path[0] == wt:
        return
    try:
        sys.path.remove(wt)
    except ValueError:
        pass
    sys.path.insert(0, wt)


def ensure_hcli_path() -> Path:
    """hcli is not in this sparse cone. Import the primary checkout read-only."""
    ensure_tools_path()
    hcli_here = WORKTREE / "hcli" / "agentos"
    if hcli_here.is_dir():
        return WORKTREE
    if (PRIMARY / "hcli" / "agentos").is_dir():
        pr = str(PRIMARY)
        if pr not in sys.path:
            sys.path.insert(1, pr)
        return PRIMARY
    raise FileNotFoundError(
        "hcli/ is not materialized in this sparse worktree and "
        f"{PRIMARY} has no hcli/agentos either"
    )


def load_symbol(module: str, name: str) -> Any:
    ensure_hcli_path()
    import importlib

    mod = importlib.import_module(module)
    ensure_tools_path()
    fn = getattr(mod, name)
    if not callable(fn):
        raise TypeError(f"{module}.{name} is not callable")
    return fn


def lake_mounted() -> bool:
    return SPECIMENS.is_dir()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_receipt(
    gate: str,
    *,
    verdict: str,
    command: list[str],
    output: Mapping[str, Any],
    measured: Mapping[str, Any],
    checks: Mapping[str, Any],
    evidence_tier: str,
    symbol_invoked: bool,
    blocker: Optional[Mapping[str, Any]] = None,
    elapsed_s: float = 0.0,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if verdict not in {"ACCEPTED", "BLOCKED"}:
        raise ValueError(f"verdict must be ACCEPTED or BLOCKED, got {verdict!r}")
    meta = GATES[gate]
    start, end = meta["acceptance_span"]
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "gate": gate,
        "verdict": verdict,
        "criterion": {
            "source": str(ROADMAP),
            "start_line": start,
            "end_line": end,
            "ledger_line": meta["ledger_line"],
            "operational": meta["operational"],
            "quote": quote_roadmap(start, end),
        },
        "implementing_symbol": meta["symbol"],
        "symbol_invoked": bool(symbol_invoked),
        "command": list(command),
        "evidence_tier": evidence_tier,
        "measured": dict(measured),
        "output": dict(output),
        "checks": dict(checks),
        "blocker": dict(blocker) if blocker else None,
        "criterion_altered": False,
        "lake_readonly": True,
        "hcli_unmodified": True,
        "generated_at": utc_now(),
        "elapsed_s": round(float(elapsed_s), 3),
        "worktree": str(WORKTREE),
    }
    if extra:
        body.update(extra)
    dest = receipts_dir() / f"{gate}.json"
    body["receipt_path"] = str(dest)
    atomic_write_json(dest, body)
    return body


def jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


class timed:
    def __init__(self) -> None:
        self.elapsed_s = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "timed":
        self._t0 = time.time()
        return self

    def snap(self) -> float:
        self.elapsed_s = time.time() - self._t0
        return self.elapsed_s

    def __exit__(self, *exc: object) -> None:
        self.snap()
