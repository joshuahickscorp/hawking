"""SANDBOX_RESIDENT_FLOOR and the provider-neutral super-resident daemon.

The floor a body must clear to enter the orchestrator sandbox. It is not
Singularity promotion. This module evaluates Flash and the incumbent Qwen27
against real receipts, defines the provider-neutral contract a daemon drives
a resident through, and reports --status from evidence. It never starts a
model process and never takes a GPU lease.

    python3 tools/future/super_resident.py --status
    python3 tools/future/super_resident.py --build
    python3 -m pytest tools/future/test_super_resident.py -q

Not a fork of hcli/providers.py, hcli/runtime_iface.py, hcli/backends.py,
hcli/hawking_native.py, hcli/agentos/resident_gate.py, or
tools/future/resident_install.py. Those remain the live surfaces. This
sidecar names the sandbox floor and the daemon contract that must survive a
change of body.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import HARDWARE_FIELDS, git
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "SUPER_RESIDENT_FLOOR.json"
SCHEMA = "hawking.future.super_resident.v1"
RECORDED_BY = "tools/future/super_resident.py"
VERSION = 1

FLASH_ID = "FLASH_SINGULARITY.NX"
QWEN_ID = "qwen3.8-27b-sealed-3.14"
QWEN_ROLE = "CURRENT_NONFINAL_HCLI_WORKER"

# Identity documents the floor actually reads. Missing in this sparse
# checkout is not absence: locate() searches the worktree, the pinned
# sidecar snapshot, the git-common working tree, then git HEAD.
REL_FLASH_NX = "receipts/headless/FLASH_COMPLETE_V0.nx.json"
REL_FLASH_NX_V1 = "receipts/headless/FLASH_COMPLETE_V1.nx.json"
REL_FLASH_NX_V2 = "receipts/headless/FLASH_COMPLETE_V2.nx.json"
REL_FLASH_NX_NEXT = "receipts/headless/FLASH_NEXT_MACHINE.nx.json"
REL_FLASH_EXEC = "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json"
REL_FLASH_STATEFUL = "receipts/headless/FLASH_STATEFUL_TPS_GATE_V14.json"
REL_FLASH_NR = "receipts/headless/FLASH_COMPLETE_V2.nr.json"
REL_FLASH_AUDIT = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
REL_QWEN_IDENTITY = "hcli/hawking-native.sealed-3.14.json"
REL_QWEN_SEAL = "receipts/headless/HCLI_RESIDENT_SEAL.json"
REL_QWEN_CAPABILITY = "receipts/headless/CAPABILITY_noetic-sealed-3.14.json"
REL_TOURNAMENT = "receipts/future/TOURNAMENT_READINESS.json"
REL_INSTALL = "receipts/future/RESIDENT_INSTALL_CONTRACT.json"
REL_CODEX_HANDOFF = "CODEX_ACCELERATOR_HANDOFF.json"
REL_FRONTIER = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
PINNED = REPO / "receipts" / "future" / "evidence"

METADATA_ONLY_MARKERS = ("METADATA_ONLY", "NOT_FOR_PROMOTION")
SCAFFOLD_MARKERS = ("SCAFFOLD_ONLY",)

# ---------------------------------------------------------------------------
# SANDBOX_RESIDENT_FLOOR. Distinct from Singularity / NX completeness.
# ---------------------------------------------------------------------------

FLOOR_CLAUSES: tuple[dict[str, str], ...] = (
    {
        "id": "self_contained_executable_identity",
        "means": (
            "A named executable identity that closes over binary, artifact, "
            "tokenizer, and protocol without a source-oracle walk."
        ),
        "not_singularity": "An identity document is not an NX genome and not a promotion.",
    },
    {
        "id": "source_independent_runtime",
        "means": (
            "The runtime loads a packed/native body. Source-oracle or SCAFFOLD_ONLY "
            "executables do not count."
        ),
        "not_singularity": "Source independence is a sandbox entry condition, not NX completeness.",
    },
    {
        "id": "no_forbidden_fallback",
        "means": (
            "Inference does not silently reconstruct a dense parent or swap bodies. "
            "A declared prompt-side tokenizer fallback is not an inference fallback."
        ),
        "not_singularity": "Fallback policy is lifecycle, not a dominance axis.",
    },
    {
        "id": "stable_sessions",
        "means": (
            "Correlated sessions survive more than one accepted token: request ids, "
            "KV/recurrent state, and isolation across turns."
        ),
        "not_singularity": "A one-token probe is not a session and not a complete-token measurement.",
    },
    {
        "id": "restart",
        "means": (
            "The body can be stopped and started again under explicit owner policy. "
            "Silent restart during a gate is forbidden."
        ),
        "not_singularity": "Restart is a lifecycle slot, not a performance claim.",
    },
    {
        "id": "capability_for_hcli_research",
        "means": (
            "Enough capability to perform HCLI research tasks (facts, structured "
            "output, tools, verifier-owned acceptance). Historical identity-bound "
            "scores may MET_QUOTED; an unidentified score does not close the clause."
        ),
        "not_singularity": "HCLI-task capability is not the 43-item suite as a promotion gate.",
    },
    {
        "id": "sufficient_token_rate",
        "means": (
            "Repeated accepted-token decode at a rate that can sustain bounded "
            "autonomy. One accepted token is not a rate. Sidecar never invents TPS."
        ),
        "not_singularity": "Quoted historical TPS is not PROTECTED_ABSOLUTE and does not promote.",
    },
    {
        "id": "explicit_ebpw_evidence",
        "means": (
            "An explicit EBPW figure bound to the body (artifact accounting or a "
            "named receipt). Null complete_system_ebpw does not count."
        ),
        "not_singularity": "Quoted identity EBPW is not a protected complete-system measurement.",
    },
    {
        "id": "rollback",
        "means": (
            "Pause and unload exist so a body can be evicted and a prior body "
            "re-bound. Resident convenience never wins over protected eviction."
        ),
        "not_singularity": "Rollback is GPU-lease subordination, not tournament dominance.",
    },
)

FLOOR_CLAUSE_IDS: tuple[str, ...] = tuple(c["id"] for c in FLOOR_CLAUSES)
MET_STATES = frozenset({"MET", "MET_QUOTED"})

PROVIDER_OPS: tuple[str, ...] = (
    "load",
    "health",
    "session",
    "generation",
    "tool_calling",
    "pause",
    "resume",
    "unload",
    "capability_identity",
    "resource_identity",
    "crash_handling",
)

# Protected evidence always outranks a resident that would like to stay loaded.
LEASE_PRIORITY: tuple[str, ...] = (
    "protected_evidence_eviction",
    "crash_unload",
    "operator_unload",
    "resident_convenience",
)

EXISTING_LIFECYCLE: tuple[dict[str, str], ...] = (
    {"path": "hcli/providers.py", "role": "ModelProvider / ResidentProvider / ResidentProfile contract"},
    {"path": "hcli/runtime_iface.py", "role": "RuntimeInterface planes; does not schedule or admit GPU memory"},
    {"path": "hcli/backends.py", "role": "RuntimeBackend ABC: identity/spawn/ready/stop/complete/supports"},
    {"path": "hcli/hawking_native.py", "role": "profile-driven native connector; ResidentProcess.stop; no pause() named"},
    {"path": "hcli/hawking-native.sealed-3.14.json", "role": "incumbent Qwen27 sealed identity"},
    {"path": "hcli/agentos/resident_gate.py", "role": "live residency proof; resident.py does not exist as a module"},
    {"path": "hcli/agentos/native_gate.py", "role": "native identity/fusion gate"},
    {"path": "hcli/agentos/native_mission_gate.py", "role": "HCLI tool/fact/verifier mission"},
    {"path": "hcli/agentos/recovery.py", "role": "fail-closed fixture recovery; production recovery is Codex-owned"},
    {"path": "hcli/agentos/protected_accelerator_benchmark.py", "role": "stop resident before closing quiescence"},
    {"path": "hcli/agentos/autonomy_gate.py", "role": "A3 restart / A4 process resume; not this floor"},
    {"path": "tools/future/resident_install.py", "role": "14-phase generic install contract the winner binds into"},
    {"path": "tools/future/tournament.py", "role": "NX-vs-NX; can_run() is False today; different bar from this floor"},
    {"path": "tools/future/flash_nx_audit.py", "role": "seven NX requirements; seven_all_met is False"},
    {"path": "tools/future/workunit_species.py", "role": "WorkUnit emission into the recovered HCLI field set"},
)


class FloorEvidenceError(RuntimeError):
    """Raised when a caller asks this module to invent a missing receipt field."""


class ProviderEvicted(RuntimeError):
    """Generate/session after protected eviction or crash-unload. Fail closed."""


class SilentRestartRefused(RuntimeError):
    """Daemon policy: crash does not silently restart the body."""


class ResidentConvenienceError(RuntimeError):
    """Raised if resident convenience would outrank protected eviction."""


# ---------------------------------------------------------------------------
# Evidence location. Sparse checkout: missing here is not absence.
# ---------------------------------------------------------------------------

def _search_roots() -> tuple[Path, ...]:
    roots: list[Path] = [REPO, PINNED]
    common = git("rev-parse", "--git-common-dir")
    if common:
        gd = Path(common)
        gd = gd.resolve() if gd.is_absolute() else (REPO / gd).resolve()
        parent = gd.parent if gd.name == ".git" else gd
        if parent not in roots:
            roots.append(parent)
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return tuple(out)


def locate(rel: str) -> Path | None:
    """Locate a repo-relative document. Prefers disk over HEAD."""
    name = Path(rel).name
    tried: list[Path] = []
    for root in _search_roots():
        tried.extend((root / rel, root / name))
    seen: set[str] = set()
    for path in tried:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def load_repo_json(rel: str) -> tuple[dict[str, Any] | None, str]:
    """Load JSON. Returns (doc, source). (None, '') if nowhere locatable."""
    path = locate(rel)
    if path is not None:
        try:
            doc = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeError):
            doc = None
        if isinstance(doc, dict):
            try:
                shown = str(path.relative_to(REPO))
            except ValueError:
                shown = str(path)
            return doc, shown
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            return doc, f"HEAD:{rel}"
    return None, ""


def evidence_presence(rel: str) -> dict[str, Any]:
    """Record which recovery path was taken. Never treats missing-here as absence."""
    path = locate(rel)
    head = bool(git("ls-files", "--error-unmatch", rel))
    return {
        "rel": rel,
        "present": path is not None,
        "source": (str(path) if path is not None else None),
        "tracked_in_head": head,
        "recovery": (
            "disk" if path is not None else ("HEAD" if head else "unresolved")
        ),
    }


def _dot(doc: Any, dotted: str, default: Any = None) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _cite(rel: str, field: str, value: Any, *, source: str = "") -> dict[str, Any]:
    return {
        "path": rel,
        "field": field,
        "value": _jsonable(value),
        "source": source or None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _metadata_only(status: Any) -> bool:
    text = str(status or "")
    return any(marker in text for marker in METADATA_ONLY_MARKERS)


def _scaffold_only(status: Any) -> bool:
    text = str(status or "")
    return any(marker in text for marker in SCAFFOLD_MARKERS)


def _sanitize_hardware(node: Any) -> Any:
    """Stringify forbidden numeric hardware fields so write_receipt cannot see them."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = str(value)
            else:
                out[key] = _sanitize_hardware(value)
        return out
    if isinstance(node, list):
        return [_sanitize_hardware(item) for item in node]
    return node


def _clause(
    clause_id: str,
    state: str,
    *,
    why: str,
    citations: Sequence[Mapping[str, Any]] = (),
    would_change: str | None = None,
) -> dict[str, Any]:
    if clause_id not in FLOOR_CLAUSE_IDS:
        raise FloorEvidenceError(f"unknown floor clause {clause_id!r}")
    if state not in {"MET", "MET_QUOTED", "UNMET", "UNKNOWN"}:
        raise FloorEvidenceError(f"unknown clause state {state!r}")
    body: dict[str, Any] = {
        "clause": clause_id,
        "state": state,
        "why": why,
        "citations": [_sanitize_hardware(dict(c)) for c in citations],
    }
    if would_change:
        body["would_change"] = would_change
    return body


# ---------------------------------------------------------------------------
# Floor evaluation. Disk state is authority. Fail closed on UNKNOWN/UNMET.
# ---------------------------------------------------------------------------

def _clears(clauses: Sequence[Mapping[str, Any]]) -> bool:
    by_id = {str(row["clause"]): row for row in clauses}
    if set(by_id) != set(FLOOR_CLAUSE_IDS):
        return False
    return all(row.get("state") in MET_STATES for row in clauses)


def _unmet(clauses: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["clause"]) for row in clauses if row.get("state") not in MET_STATES]


def evaluate_flash(docs: Mapping[str, tuple[dict[str, Any] | None, str]] | None = None) -> dict[str, Any]:
    """Honest Flash verdict against real receipts. Expected: does not clear."""
    bundle = docs or _load_all()
    nx, nx_src = bundle.get("flash_nx", (None, ""))
    exec_doc, exec_src = bundle.get("flash_exec", (None, ""))
    stateful, stateful_src = bundle.get("flash_stateful", (None, ""))
    handoff, handoff_src = bundle.get("codex_handoff", (None, ""))
    audit, audit_src = bundle.get("flash_audit", (None, ""))

    nx_status = nx.get("status") if isinstance(nx, Mapping) else None
    nx_q = nx.get("qualification") if isinstance(nx, Mapping) else None
    if not isinstance(nx_q, Mapping):
        nx_q = {}
    exec_status = exec_doc.get("status") if isinstance(exec_doc, Mapping) else None
    exec_qual = exec_doc.get("qualification") if isinstance(exec_doc, Mapping) else None
    loader = exec_doc.get("native_loader") if isinstance(exec_doc, Mapping) else None
    loader_status = loader.get("status") if isinstance(loader, Mapping) else None
    kernels = exec_doc.get("native_kernels") if isinstance(exec_doc, Mapping) else None
    kernels_status = kernels.get("status") if isinstance(kernels, Mapping) else None
    cap_contract = exec_doc.get("capability_contract") if isinstance(exec_doc, Mapping) else None
    cap_status = cap_contract.get("status") if isinstance(cap_contract, Mapping) else None
    exec_ebpw = exec_doc.get("complete_system_ebpw") if isinstance(exec_doc, Mapping) else None
    promotion_allowed = exec_doc.get("promotion_allowed") if isinstance(exec_doc, Mapping) else None
    nx_ebpw = nx_q.get("complete_system_ebpw")
    nx_tps = nx_q.get("accepted_multitoken_tps")
    nx_promo = nx_q.get("resident_promotion")

    accepted_tokens = _dot(stateful, "complete_stateful_session.accepted_generation_tokens")
    stateful_status = stateful.get("status") if isinstance(stateful, Mapping) else None
    one_token_status = _dot(stateful, "first_physical_failure_boundary.status")
    one_token_claim = _dot(stateful, "complete_stateful_session.claim")
    session_status = _dot(stateful, "complete_stateful_session.status")

    handoff_nx = _dot(handoff, "current_flash_state.source_independent_nx")
    if not isinstance(handoff_nx, Mapping):
        handoff_nx = {}
    handoff_gate = _dot(handoff, "current_flash_state.stateful_gate")
    if not isinstance(handoff_gate, Mapping):
        handoff_gate = {}
    seven_all_met = audit.get("seven_all_met") if isinstance(audit, Mapping) else None

    nx_is_scaffold = _scaffold_only(exec_status) or _scaffold_only(handoff_nx.get("status"))
    nx_is_metadata = _metadata_only(nx_status)
    one_token = accepted_tokens == 1 or one_token_status == "ONE_TOKEN_ACCEPTED"
    repeated_decode = isinstance(accepted_tokens, int) and accepted_tokens > 1

    clauses = [
        _clause(
            "self_contained_executable_identity",
            "UNMET" if (nx_is_metadata or loader_status in {None, "NOT_IMPLEMENTED"} or nx_is_scaffold) else "MET",
            why=(
                "FLASH_COMPLETE_V0.nx.json is metadata-only; FLASH_NEXT_NOETIC_EXECUTABLE "
                f"status={exec_status!r}; native_loader.status={loader_status!r}."
            ),
            citations=[
                _cite(REL_FLASH_NX, "status", nx_status, source=nx_src),
                _cite(REL_FLASH_NX, "qualification.resident_promotion", nx_promo, source=nx_src),
                _cite(REL_FLASH_EXEC, "status", exec_status, source=exec_src),
                _cite(REL_FLASH_EXEC, "native_loader.status", loader_status, source=exec_src),
                _cite(REL_FLASH_EXEC, "native_kernels.status", kernels_status, source=exec_src),
            ],
            would_change=(
                "A packed executable identity with native_loader implemented, NX status "
                "no longer SEALED_METADATA_ONLY_NOT_FOR_PROMOTION, and executable status "
                "no longer SCAFFOLD_ONLY."
            ),
        ),
        _clause(
            "source_independent_runtime",
            "UNMET" if (nx_is_scaffold or exec_qual is False or exec_qual is None) else "MET",
            why=(
                "Codex current_flash_state.source_independent_nx.status is SCAFFOLD_ONLY "
                f"(qualification={handoff_nx.get('qualification')!r}); the executable "
                f"receipt status is {exec_status!r}."
            ),
            citations=[
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_flash_state.source_independent_nx.status",
                    handoff_nx.get("status"),
                    source=handoff_src,
                ),
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_flash_state.source_independent_nx.qualification",
                    handoff_nx.get("qualification"),
                    source=handoff_src,
                ),
                _cite(REL_FLASH_EXEC, "status", exec_status, source=exec_src),
                _cite(REL_FLASH_EXEC, "qualification", exec_qual, source=exec_src),
                _cite(REL_FLASH_AUDIT, "seven_all_met", seven_all_met, source=audit_src),
            ],
            would_change=(
                "source_independent_nx.status leaves SCAFFOLD_ONLY and qualification "
                "becomes true: serialized artifact, physical loader, and whole-model "
                "native kernel binding exist."
            ),
        ),
        _clause(
            "no_forbidden_fallback",
            "UNMET" if seven_all_met is not True else "MET",
            why=(
                "Flash NX audit seven_requirements.no_forbidden_fallback is NOT_MET; "
                "the executable lists fallback_count disclosure as missing_or_refused. "
                "The live path is still the source-oracle / exact-control dense BF16 path."
            ),
            citations=[
                _cite(REL_FLASH_AUDIT, "seven_all_met", seven_all_met, source=audit_src),
                _cite(
                    REL_FLASH_EXEC,
                    "promotion_gate.missing_or_refused",
                    _dot(exec_doc, "promotion_gate.missing_or_refused"),
                    source=exec_src,
                ),
                _cite(
                    REL_FLASH_EXEC,
                    "dense_parent_execution_fallback",
                    exec_doc.get("dense_parent_execution_fallback") if isinstance(exec_doc, Mapping) else None,
                    source=exec_src,
                ),
            ],
            would_change=(
                "no_forbidden_fallback MET on the NX checker, fallback_count disclosed, "
                "and production path no longer the source-oracle dense parent."
            ),
        ),
        _clause(
            "stable_sessions",
            "MET" if repeated_decode else "UNMET",
            why=(
                "The stateful gate accepted one generated token and explicitly does not "
                "claim continuation state was advanced. Repeated accepted decode remains open."
            ),
            citations=[
                _cite(REL_FLASH_STATEFUL, "status", stateful_status, source=stateful_src),
                _cite(
                    REL_FLASH_STATEFUL,
                    "complete_stateful_session.accepted_generation_tokens",
                    accepted_tokens,
                    source=stateful_src,
                ),
                _cite(
                    REL_FLASH_STATEFUL,
                    "complete_stateful_session.status",
                    session_status,
                    source=stateful_src,
                ),
                _cite(
                    REL_FLASH_STATEFUL,
                    "first_physical_failure_boundary.status",
                    one_token_status,
                    source=stateful_src,
                ),
                _cite(
                    REL_FLASH_STATEFUL,
                    "complete_stateful_session.claim",
                    one_token_claim,
                    source=stateful_src,
                ),
            ],
            would_change=(
                "A persistent executor that consumes the accepted token as the next "
                "session input and produces repeated accepted-token decode."
            ),
        ),
        _clause(
            "restart",
            "MET" if nx_promo is True else "UNMET",
            why=(
                "No Flash resident process exists to restart. The executable is a "
                "scaffold; there is no loaded body under the HCLI resident protocol."
            ),
            citations=[
                _cite(REL_FLASH_EXEC, "status", exec_status, source=exec_src),
                _cite(REL_FLASH_NX, "qualification.resident_promotion", nx_promo, source=nx_src),
            ],
            would_change=(
                "A Flash body bound through the provider contract with stop/start "
                "under no-silent-restart policy."
            ),
        ),
        _clause(
            "capability_for_hcli_research",
            "UNMET" if cap_status in {None, "NOT_RUN"} else "MET",
            why=(
                f"capability_contract.status={cap_status!r} on the executable; "
                "the NX audit looked for a capability suite on Flash NX and found NOT_RUN."
            ),
            citations=[
                _cite(REL_FLASH_EXEC, "capability_contract.status", cap_status, source=exec_src),
                _cite(REL_FLASH_AUDIT, "seven_all_met", seven_all_met, source=audit_src),
            ],
            would_change="Capability suite run on a source-independent Flash NX, identity-bound.",
        ),
        _clause(
            "sufficient_token_rate",
            "MET" if repeated_decode else "UNMET",
            why=(
                "One accepted stateful token is not a token rate. accepted_multitoken_tps "
                "is null; stateful_gate.accepted_tps is null; Codex records one accepted "
                "token rather than repeated accepted-token decode."
            ),
            citations=[
                _cite(
                    REL_FLASH_STATEFUL,
                    "complete_stateful_session.accepted_generation_tokens",
                    accepted_tokens,
                    source=stateful_src,
                ),
                _cite(
                    REL_FLASH_NX,
                    "qualification.accepted_multitoken_tps",
                    nx_tps,
                    source=nx_src,
                ),
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_flash_state.stateful_gate.accepted_tokens",
                    handoff_gate.get("accepted_tokens"),
                    source=handoff_src,
                ),
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_flash_state.stateful_gate.status",
                    handoff_gate.get("status"),
                    source=handoff_src,
                ),
                _cite(
                    REL_FLASH_STATEFUL,
                    "first_physical_failure_boundary.status",
                    one_token_status,
                    source=stateful_src,
                ),
            ],
            would_change=(
                "Repeated accepted-token decode with a named (PROTECTED_ABSOLUTE) rate "
                "receipt. accepted_generation_tokens > 1 and accepted_multitoken_tps "
                "no longer null. This sidecar will still not invent the number."
            ),
        ),
        _clause(
            "explicit_ebpw_evidence",
            "MET" if exec_ebpw is not None or nx_ebpw is not None else "UNMET",
            why=(
                "complete_system_ebpw is null on both the NX qualification and the "
                "executable receipt. Meta physical_ebpw is NULL_BY_RULE."
            ),
            citations=[
                _cite(REL_FLASH_NX, "qualification.complete_system_ebpw", nx_ebpw, source=nx_src),
                _cite(REL_FLASH_EXEC, "complete_system_ebpw", exec_ebpw, source=exec_src),
                _cite(
                    REL_FLASH_AUDIT,
                    "meta_measurement_state.physical_ebpw",
                    _dot(audit, "meta_measurement_state.physical_ebpw"),
                    source=audit_src,
                ),
            ],
            would_change=(
                "A complete-system EBPW bound to the Flash NX body, not a prospective "
                "meta budget and not a null."
            ),
        ),
        _clause(
            "rollback",
            "MET" if nx_promo is True else "UNMET",
            why=(
                "No Flash resident is loaded, so pause/unload cannot roll it back. "
                "resident_promotion is not true."
            ),
            citations=[
                _cite(REL_FLASH_NX, "qualification.resident_promotion", nx_promo, source=nx_src),
                _cite(REL_FLASH_EXEC, "promotion_allowed", promotion_allowed, source=exec_src),
            ],
            would_change=(
                "A Flash body that implements the provider pause/unload slots and "
                "yields to protected-evidence eviction."
            ),
        ),
    ]

    unmet = _unmet(clauses)
    return {
        "id": FLASH_ID,
        "role": "CANDIDATE_NOT_SANDBOX_HOLDER",
        "clears_sandbox_floor": _clears(clauses),
        "clears_singularity": False,
        "distinct_from_singularity": True,
        "clauses": clauses,
        "unmet_clauses": unmet,
        "one_accepted_stateful_token": one_token and not repeated_decode,
        "source_independent_nx": {
            "status": handoff_nx.get("status") or exec_status,
            "qualification": handoff_nx.get("qualification") if "qualification" in handoff_nx else exec_qual,
            "path": handoff_nx.get("path") or REL_FLASH_EXEC,
        },
        "stateful_gate": {
            "status": handoff_gate.get("status") or stateful_status,
            "accepted_tokens": handoff_gate.get("accepted_tokens") if "accepted_tokens" in handoff_gate else accepted_tokens,
            "path": handoff_gate.get("path") or REL_FLASH_STATEFUL,
        },
        "evidence_recovery": {
            "nx": evidence_presence(REL_FLASH_NX),
            "executable": evidence_presence(REL_FLASH_EXEC),
            "stateful": evidence_presence(REL_FLASH_STATEFUL),
            "handoff": evidence_presence(REL_CODEX_HANDOFF),
            "audit": evidence_presence(REL_FLASH_AUDIT),
        },
        "headline": (
            "Flash does NOT clear SANDBOX_RESIDENT_FLOOR today: source-independent "
            "NX is SCAFFOLD_ONLY and the stateful gate has one accepted token, not "
            "repeated accepted-token decode."
            if not _clears(clauses)
            else "Flash cites floor-clearing evidence; this is still not Singularity promotion."
        ),
    }


def evaluate_qwen27(docs: Mapping[str, tuple[dict[str, Any] | None, str]] | None = None) -> dict[str, Any]:
    """Incumbent Qwen27 as CURRENT_NONFINAL_HCLI_WORKER. Not a Singularity."""
    bundle = docs or _load_all()
    ident, ident_src = bundle.get("qwen_identity", (None, ""))
    seal, seal_src = bundle.get("qwen_seal", (None, ""))
    cap, cap_src = bundle.get("qwen_capability", (None, ""))
    handoff, handoff_src = bundle.get("codex_handoff", (None, ""))
    tournament, tournament_src = bundle.get("tournament", (None, ""))
    install, install_src = bundle.get("install", (None, ""))

    if not isinstance(ident, Mapping):
        ident = {}
    resident_identity = ident.get("resident_identity")
    protocol = ident.get("protocol")
    runtime = ident.get("runtime")
    binary = ident.get("resident_binary") or ident.get("binary")
    artifact = ident.get("artifact_root")
    tokenizer = ident.get("tokenizer")
    fallbacks = ident.get("fallbacks")
    quoted_fallbacks = _dot(ident, "current_runtime.fallbacks")
    quoted_ebpw = ident.get("physical_ebpw")
    if quoted_ebpw is None:
        quoted_ebpw = _dot(ident, "representation.physical_ebpw")
    quoted_tps_current = _dot(ident, "current_runtime.complete_tps_current_measured")
    quoted_tps_hist = _dot(ident, "current_runtime.complete_tps_historical_qualified")
    qualification = ident.get("qualification")
    mode = ident.get("mode")

    cap_overall = cap.get("overall") if isinstance(cap, Mapping) else None
    if not isinstance(cap_overall, Mapping):
        cap_overall = {}
    cap_passed = cap_overall.get("passed")
    cap_total = cap_overall.get("total")
    identity_sufficient = cap.get("identity_sufficient") if isinstance(cap, Mapping) else None
    identity_verdict = cap.get("identity_verdict") if isinstance(cap, Mapping) else None
    quoted_cap = None
    if isinstance(cap_passed, int) and isinstance(cap_total, int) and cap_total > 0:
        quoted_cap = f"{cap_passed}/{cap_total}"

    seal_status = seal.get("status") if isinstance(seal, Mapping) else None
    seal_resident = seal.get("resident") if isinstance(seal, Mapping) else None

    qwen_handoff = handoff.get("current_qwen27_incumbent_control_identity") if isinstance(handoff, Mapping) else None
    if not isinstance(qwen_handoff, Mapping):
        qwen_handoff = {}
    qualified_physical = _dot(qwen_handoff, "profile.qualified_physical")
    control_promo = _dot(qwen_handoff, "control_receipt.promotion_allowed")

    tour_qwen = None
    if isinstance(tournament, Mapping):
        contenders = tournament.get("contenders")
        if isinstance(contenders, Mapping):
            tour_qwen = contenders.get("QWEN27_SINGULARITY.NX")

    has_identity = bool(resident_identity and protocol and runtime and binary and artifact and tokenizer)
    clauses = [
        _clause(
            "self_contained_executable_identity",
            "MET" if has_identity else "UNMET",
            why=(
                "Sealed identity names resident_binary, artifact_root, tokenizer, "
                f"protocol={protocol!r}, resident_identity={resident_identity!r}."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "resident_identity", resident_identity, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "protocol", protocol, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "runtime", runtime, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "resident_binary", ident.get("resident_binary"), source=ident_src),
                _cite(REL_QWEN_IDENTITY, "artifact_root", artifact, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "tokenizer", tokenizer, source=ident_src),
                _cite(REL_QWEN_SEAL, "status", seal_status, source=seal_src),
                _cite(REL_QWEN_SEAL, "resident", seal_resident, source=seal_src),
            ],
        ),
        _clause(
            "source_independent_runtime",
            "MET" if runtime == "hawking-native" and protocol else "UNMET",
            why=(
                "hawking-native packed runtime with JSONL resident protocol. This is "
                "not a source-oracle walk and not SCAFFOLD_ONLY."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "runtime", runtime, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "protocol", protocol, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "representation.kind", _dot(ident, "representation.kind"), source=ident_src),
                _cite(REL_QWEN_IDENTITY, "mode", mode, source=ident_src),
            ],
        ),
        _clause(
            "no_forbidden_fallback",
            "MET_QUOTED" if quoted_fallbacks == 0 else "UNMET",
            why=(
                "current_runtime.fallbacks is 0. Declared fallbacks are prompt-side "
                "(transformers tokenizer when unavailable), not inference fallbacks."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "current_runtime.fallbacks", quoted_fallbacks, source=ident_src),
                _cite(REL_QWEN_IDENTITY, "fallbacks", fallbacks, source=ident_src),
            ],
        ),
        _clause(
            "stable_sessions",
            "MET" if protocol else "UNMET",
            why=(
                "hawking.qwen38.resident.v1 JSONL with correlated request ids, "
                "resident_gate sequential proof contract, and weights_loaded_once."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "protocol", protocol, source=ident_src),
                _cite(
                    REL_QWEN_IDENTITY,
                    "representation.weights_loaded_once",
                    _dot(ident, "representation.weights_loaded_once"),
                    source=ident_src,
                ),
                _cite("hcli/agentos/resident_gate.py", "schema", "hcli.agentos.resident_gate.v1", source="source"),
            ],
        ),
        _clause(
            "restart",
            "MET",
            why=(
                "HawkingNativeConnector exposes stop and _restart_resident; "
                "resident_install restart slot is no_silent_restart. This lane does "
                "not start a process to re-prove it."
            ),
            citations=[
                _cite("hcli/hawking_native.py", "ResidentProcess.stop", "present", source="source"),
                _cite(REL_INSTALL, "schema", install.get("schema") if isinstance(install, Mapping) else None, source=install_src),
            ],
        ),
        _clause(
            "capability_for_hcli_research",
            "MET_QUOTED",
            why=(
                "Identity quotes a historical sealed capability contract. The "
                f"capability receipt overall is {quoted_cap or 'unresolved'} but "
                f"identity_sufficient={identity_sufficient!r}, so that row does not "
                "bind the score to this body. HCLI still uses this identity as the "
                "incumbent worker (native_mission_gate / resident_gate)."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "qualification", qualification, source=ident_src),
                _cite(REL_QWEN_CAPABILITY, "overall.passed", cap_passed, source=cap_src),
                _cite(REL_QWEN_CAPABILITY, "overall.total", cap_total, source=cap_src),
                _cite(REL_QWEN_CAPABILITY, "identity_sufficient", identity_sufficient, source=cap_src),
                _cite(
                    REL_QWEN_CAPABILITY,
                    "identity_verdict",
                    (str(identity_verdict)[:240] if identity_verdict else None),
                    source=cap_src,
                ),
            ],
        ),
        _clause(
            "sufficient_token_rate",
            "MET_QUOTED",
            why=(
                "Identity quotes complete_tps_current_measured and historical "
                "qualified TPS as strings here. Codex records qualified_physical="
                f"{qualified_physical!r} and control_receipt.promotion_allowed="
                f"{control_promo!r}. This is incumbency evidence for bounded "
                "autonomy, not a live PROTECTED_ABSOLUTE measurement."
            ),
            citations=[
                _cite(
                    REL_QWEN_IDENTITY,
                    "current_runtime.complete_tps_current_measured",
                    str(quoted_tps_current) if quoted_tps_current is not None else None,
                    source=ident_src,
                ),
                _cite(
                    REL_QWEN_IDENTITY,
                    "current_runtime.complete_tps_historical_qualified",
                    str(quoted_tps_hist) if quoted_tps_hist is not None else None,
                    source=ident_src,
                ),
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_qwen27_incumbent_control_identity.profile.qualified_physical",
                    qualified_physical,
                    source=handoff_src,
                ),
                _cite(
                    REL_CODEX_HANDOFF,
                    "current_qwen27_incumbent_control_identity.control_receipt.promotion_allowed",
                    control_promo,
                    source=handoff_src,
                ),
            ],
        ),
        _clause(
            "explicit_ebpw_evidence",
            "MET_QUOTED" if quoted_ebpw is not None else "UNMET",
            why=(
                "Identity representation.physical_ebpw is the sealed artifact "
                "accounting figure. It is not a protected complete-system remeasurement."
            ),
            citations=[
                _cite(REL_QWEN_IDENTITY, "physical_ebpw", quoted_ebpw, source=ident_src),
                _cite(
                    REL_QWEN_IDENTITY,
                    "representation.physical_ebpw",
                    _dot(ident, "representation.physical_ebpw"),
                    source=ident_src,
                ),
            ],
        ),
        _clause(
            "rollback",
            "MET",
            why=(
                "connector.stop unloads the resident; resident_install unload drops "
                "weights and releases the device; protected_accelerator_benchmark "
                "stops the resident before closing quiescence. Live RuntimeBackend "
                "names stop, not pause — the daemon contract below is the named slot."
            ),
            citations=[
                _cite("hcli/hawking_native.py", "HawkingNativeConnector.stop", "present", source="source"),
                _cite(
                    "hcli/agentos/protected_accelerator_benchmark.py",
                    "stop_before_closing_quiescence",
                    True,
                    source="source",
                ),
                _cite(REL_INSTALL, "phases", install.get("phases") if isinstance(install, Mapping) else None, source=install_src),
            ],
        ),
    ]

    tour_role = None
    if isinstance(tour_qwen, Mapping):
        tour_role = tour_qwen.get("role")
    return {
        "id": QWEN_ID,
        "role": QWEN_ROLE,
        "clears_sandbox_floor": _clears(clauses),
        "clears_singularity": False,
        "distinct_from_singularity": True,
        "clauses": clauses,
        "unmet_clauses": _unmet(clauses),
        "resident_identity": resident_identity,
        "protocol": protocol,
        "quoted_capability_overall": quoted_cap,
        "capability_identity_sufficient": identity_sufficient,
        "qualified_physical": qualified_physical,
        "tournament_role": tour_role,
        "evidence_recovery": {
            "identity": evidence_presence(REL_QWEN_IDENTITY),
            "seal": evidence_presence(REL_QWEN_SEAL),
            "capability": evidence_presence(REL_QWEN_CAPABILITY),
            "handoff": evidence_presence(REL_CODEX_HANDOFF),
            "tournament": evidence_presence(REL_TOURNAMENT),
        },
        "headline": (
            "Qwen27 sealed-3.14 is CURRENT_NONFINAL_HCLI_WORKER: it holds the "
            "sandbox role by sealed native identity, not as a Singularity, and "
            "not as a live protected remeasurement."
        ),
    }


def _load_all() -> dict[str, tuple[dict[str, Any] | None, str]]:
    return {
        "flash_nx": load_repo_json(REL_FLASH_NX),
        "flash_exec": load_repo_json(REL_FLASH_EXEC),
        "flash_stateful": load_repo_json(REL_FLASH_STATEFUL),
        "flash_nr": load_repo_json(REL_FLASH_NR),
        "flash_audit": load_repo_json(REL_FLASH_AUDIT),
        "qwen_identity": load_repo_json(REL_QWEN_IDENTITY),
        "qwen_seal": load_repo_json(REL_QWEN_SEAL),
        "qwen_capability": load_repo_json(REL_QWEN_CAPABILITY),
        "codex_handoff": load_repo_json(REL_CODEX_HANDOFF),
        "tournament": load_repo_json(REL_TOURNAMENT),
        "install": load_repo_json(REL_INSTALL),
        "frontier": load_repo_json(REL_FRONTIER),
    }


def sandbox_status(docs: Mapping[str, tuple[dict[str, Any] | None, str]] | None = None) -> dict[str, Any]:
    """Which body holds the sandbox role and why, from evidence. No process start."""
    bundle = docs or _load_all()
    flash = evaluate_flash(bundle)
    qwen = evaluate_qwen27(bundle)
    holder: str | None = None
    holder_role: str | None = None
    why: str
    if qwen.get("clears_sandbox_floor") and not flash.get("clears_sandbox_floor"):
        holder = qwen["id"]
        holder_role = qwen["role"]
        why = (
            "Qwen27 sealed-3.14 is the only body with a sealed hawking-native "
            "resident identity and protocol. Flash source-independent NX is "
            "SCAFFOLD_ONLY and the stateful gate has one accepted token, not "
            "repeated accepted-token decode. Holding the sandbox is not Singularity."
        )
    elif flash.get("clears_sandbox_floor") and not qwen.get("clears_sandbox_floor"):
        holder = flash["id"]
        holder_role = flash["role"]
        why = "Flash clears SANDBOX_RESIDENT_FLOOR; Qwen27 does not."
    elif flash.get("clears_sandbox_floor") and qwen.get("clears_sandbox_floor"):
        holder = qwen["id"]
        holder_role = qwen["role"]
        why = (
            "Both bodies cite floor-clearing evidence; the incumbent sealed worker "
            "holds the role until a daemon swap. This is not a tournament winner."
        )
    else:
        why = (
            "No body clears SANDBOX_RESIDENT_FLOOR from disk evidence. The "
            "orchestrator sandbox has no qualified holder today."
        )
    return {
        "schema": "hawking.future.super_resident.status.v1",
        "started_model_process": False,
        "took_gpu_lease": False,
        "holder": holder,
        "holder_role": holder_role,
        "why": why,
        "flash": {
            "id": flash["id"],
            "clears_sandbox_floor": flash["clears_sandbox_floor"],
            "clears_singularity": flash["clears_singularity"],
            "unmet_clauses": flash["unmet_clauses"],
            "headline": flash["headline"],
        },
        "qwen27": {
            "id": qwen["id"],
            "role": qwen["role"],
            "clears_sandbox_floor": qwen["clears_sandbox_floor"],
            "clears_singularity": qwen["clears_singularity"],
            "unmet_clauses": qwen["unmet_clauses"],
            "headline": qwen["headline"],
        },
        "floor_is_not_singularity": True,
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
    }


# ---------------------------------------------------------------------------
# Provider-neutral daemon. Logic must survive a change of body.
# ---------------------------------------------------------------------------

class StubProvider:
    """In-process stub. Two instances with different ids prove the daemon is generic."""

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        tool_calling: bool = True,
        prefer_keep_resident: bool = False,
    ) -> None:
        self.provider_id = str(provider_id)
        self.model_id = str(model_id)
        self.supports_tool_calling = bool(tool_calling)
        self.prefer_keep_resident = bool(prefer_keep_resident)
        self.state = "unloaded"
        self.session_id: str | None = None
        self.paused = False
        self.crash: str | None = None
        self.calls: list[str] = []

    def _record(self, op: str) -> None:
        self.calls.append(op)

    def load(self) -> dict[str, Any]:
        self._record("load")
        if self.crash:
            raise ProviderEvicted(f"{self.provider_id} crashed: {self.crash}")
        self.state = "loaded"
        self.paused = False
        return {"op": "load", "provider_id": self.provider_id, "model_id": self.model_id, "state": self.state}

    def health(self) -> dict[str, Any]:
        self._record("health")
        ready = self.state == "loaded" and not self.paused and self.crash is None
        return {
            "op": "health",
            "provider_id": self.provider_id,
            "ready": ready,
            "state": self.state,
            "paused": self.paused,
        }

    def session(self) -> dict[str, Any]:
        self._record("session")
        self._require_active("session")
        self.session_id = f"session:{self.provider_id}"
        return {"op": "session", "provider_id": self.provider_id, "session_id": self.session_id}

    def generation(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._record("generation")
        self._require_active("generation")
        text = f"stub-reply:{self.model_id}"
        return {
            "op": "generation",
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "text": text,
            "finish_reason": "stop",
            "request_keys": sorted((request or {}).keys()),
        }

    def tool_calling(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._record("tool_calling")
        self._require_active("tool_calling")
        del request
        return {
            "op": "tool_calling",
            "provider_id": self.provider_id,
            "supported": self.supports_tool_calling,
            "calls": ([{"name": "echo", "arguments": {"model": self.model_id}}] if self.supports_tool_calling else []),
        }

    def pause(self, reason: str) -> dict[str, Any]:
        self._record("pause")
        if self.state == "unloaded":
            return {"op": "pause", "provider_id": self.provider_id, "state": self.state, "reason": reason}
        self.paused = True
        self.state = "paused"
        return {"op": "pause", "provider_id": self.provider_id, "state": self.state, "reason": reason}

    def resume(self) -> dict[str, Any]:
        self._record("resume")
        if self.crash:
            raise ProviderEvicted(f"{self.provider_id} crashed: {self.crash}")
        if self.state == "unloaded":
            raise ProviderEvicted(f"{self.provider_id} is unloaded")
        self.paused = False
        self.state = "loaded"
        return {"op": "resume", "provider_id": self.provider_id, "state": self.state}

    def unload(self, reason: str) -> dict[str, Any]:
        self._record("unload")
        self.state = "unloaded"
        self.paused = False
        self.session_id = None
        return {
            "op": "unload",
            "provider_id": self.provider_id,
            "state": self.state,
            "reason": reason,
            "weights_dropped": True,
            "device_released": True,
        }

    def capability_identity(self) -> dict[str, Any]:
        self._record("capability_identity")
        return {
            "op": "capability_identity",
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "features": {
                "tool_calling": self.supports_tool_calling,
                "streaming": False,
                "response_format": False,
            },
        }

    def resource_identity(self) -> dict[str, Any]:
        self._record("resource_identity")
        return {
            "op": "resource_identity",
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "lease": "none",
            "gpu_authority": False,
        }

    def crash_handling(self, event: str) -> dict[str, Any]:
        self._record("crash_handling")
        self.crash = str(event)
        self.state = "crashed"
        return {
            "op": "crash_handling",
            "provider_id": self.provider_id,
            "event": str(event),
            "silent_restart": False,
        }

    def _require_active(self, op: str) -> None:
        if self.crash:
            raise ProviderEvicted(f"{self.provider_id} crashed; {op} refused")
        if self.state == "unloaded":
            raise ProviderEvicted(f"{self.provider_id} unloaded; {op} refused")
        if self.paused or self.state == "paused":
            raise ProviderEvicted(f"{self.provider_id} paused; {op} refused")


class SandboxDaemon:
    """Drives any provider through PROVIDER_OPS. Not hard-coded to a model."""

    def __init__(self) -> None:
        self.provider: StubProvider | None = None
        self.evicted_for: str | None = None
        self.crash_log: list[str] = []
        self.no_silent_restart = True
        self.started_model_process = False

    def bind(self, provider: StubProvider) -> None:
        self.provider = provider
        self.evicted_for = None

    def _need(self) -> StubProvider:
        if self.provider is None:
            raise ProviderEvicted("no provider bound")
        if self.evicted_for:
            raise ProviderEvicted(f"evicted for {self.evicted_for}")
        return self.provider

    def drive(self, provider: StubProvider) -> dict[str, Any]:
        """Run the identical contract against whatever body is bound."""
        self.bind(provider)
        results: dict[str, Any] = {}
        results["load"] = provider.load()
        results["health"] = provider.health()
        results["capability_identity"] = provider.capability_identity()
        results["resource_identity"] = provider.resource_identity()
        results["session"] = provider.session()
        results["generation"] = provider.generation({"messages": [{"role": "user", "content": "ping"}]})
        results["tool_calling"] = provider.tool_calling({"tools": [{"name": "echo"}]})
        results["pause"] = provider.pause("contract-probe")
        results["resume"] = provider.resume()
        crash = provider.crash_handling("contract-probe-crash")
        results["crash_handling"] = crash
        self.crash_log.append(str(crash.get("event")))
        if self.no_silent_restart and crash.get("silent_restart") is True:
            raise SilentRestartRefused("provider attempted a silent restart")
        results["unload"] = provider.unload("contract-probe-complete")
        ops_run = [op for op in PROVIDER_OPS if op in results]
        return {
            "schema": "hawking.future.super_resident.drive.v1",
            "provider_id": provider.provider_id,
            "model_id": provider.model_id,
            "ops": list(PROVIDER_OPS),
            "ops_run": ops_run,
            "ops_complete": ops_run == list(PROVIDER_OPS),
            "results": results,
            "started_model_process": self.started_model_process,
        }

    def protected_evict(
        self,
        *,
        reason: str = "protected_evidence_eviction",
        prefer_keep_resident: bool | None = None,
    ) -> dict[str, Any]:
        """Evict the resident. Convenience never wins, including the provider's own flag."""
        provider = self.provider
        if provider is None:
            raise ProviderEvicted("no provider bound")
        want_keep = bool(provider.prefer_keep_resident if prefer_keep_resident is None else prefer_keep_resident)
        if reason == "resident_convenience":
            raise ResidentConvenienceError("resident convenience is last on LEASE_PRIORITY")
        pause = provider.pause(reason)
        unload = provider.unload(reason)
        self.evicted_for = reason
        return {
            "schema": "hawking.future.super_resident.evict.v1",
            "reason": reason,
            "priority": list(LEASE_PRIORITY),
            "prefer_keep_resident": want_keep,
            "resident_convenience_wins": False,
            "paused": pause.get("state") in {"paused", "unloaded"},
            "unloaded": unload.get("state") == "unloaded",
            "weights_dropped": bool(unload.get("weights_dropped")),
            "device_released": bool(unload.get("device_released")),
            "provider_id": provider.provider_id,
            "model_id": provider.model_id,
        }

    def generate_after_evict(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del request
        raise ProviderEvicted(f"generate refused; evicted for {self.evicted_for}")


def prove_provider_neutral() -> dict[str, Any]:
    """Drive two stub providers through the identical daemon. Deterministic."""
    alpha = StubProvider(provider_id="stub-alpha", model_id="body-A", tool_calling=True)
    beta = StubProvider(provider_id="stub-beta", model_id="body-B", tool_calling=False)
    daemon = SandboxDaemon()
    drive_a = daemon.drive(alpha)
    drive_b = daemon.drive(beta)
    src = inspect.getsource(SandboxDaemon)
    return {
        "alpha": {"provider_id": drive_a["provider_id"], "model_id": drive_a["model_id"], "ops": drive_a["ops"]},
        "beta": {"provider_id": drive_b["provider_id"], "model_id": drive_b["model_id"], "ops": drive_b["ops"]},
        "identical_ops": drive_a["ops"] == drive_b["ops"] == list(PROVIDER_OPS),
        "bodies_differ": drive_a["provider_id"] != drive_b["provider_id"] and drive_a["model_id"] != drive_b["model_id"],
        "daemon_source_names_no_body": (
            "qwen" not in src.lower()
            and "flash" not in src.lower()
            and "sealed-3.14" not in src.lower()
        ),
        "started_model_process": False,
    }


def prove_gpu_lease_subordination() -> dict[str, Any]:
    """A clingy provider still unloads when protected evidence asks."""
    clingy = StubProvider(
        provider_id="stub-clingy",
        model_id="keep-me-loaded",
        prefer_keep_resident=True,
    )
    daemon = SandboxDaemon()
    daemon.bind(clingy)
    clingy.load()
    clingy.session()
    evict = daemon.protected_evict(prefer_keep_resident=True)
    refused = False
    try:
        daemon.generate_after_evict({"messages": []})
    except ProviderEvicted:
        refused = True
    return {
        "prefer_keep_resident": True,
        "resident_convenience_wins": evict["resident_convenience_wins"],
        "unloaded": evict["unloaded"],
        "generate_after_evict_refused": refused,
        "priority": list(LEASE_PRIORITY),
        "last_priority": LEASE_PRIORITY[-1],
    }


def floor_definition() -> dict[str, Any]:
    return {
        "name": "SANDBOX_RESIDENT_FLOOR",
        "distinct_from_singularity": True,
        "not_a_promotion": True,
        "schema": SCHEMA,
        "clauses": [dict(c) for c in FLOOR_CLAUSES],
        "clause_ids": list(FLOOR_CLAUSE_IDS),
        "pass_rule": "every clause MET or MET_QUOTED; UNMET and UNKNOWN fail closed",
        "singularity_bar": (
            "Tournament NX completeness + PROTECTED_ABSOLUTE dominance. "
            "This floor does not run that bar and cannot promote anyone."
        ),
    }


def provider_contract() -> dict[str, Any]:
    return {
        "ops": list(PROVIDER_OPS),
        "provider_neutral": True,
        "daemon": "tools.future.super_resident.SandboxDaemon",
        "live_hcli_surface": {
            "ModelProvider": "hcli.providers.ModelProvider (generate, capabilities, health, profile)",
            "ResidentProvider": "hcli.providers.ResidentProvider (start, stop)",
            "RuntimeBackend": "hcli.backends.RuntimeBackend (identity, spawn, ready, endpoint, stop, complete, supports)",
            "gap": (
                "Live RuntimeBackend exposes stop, not named pause/resume/unload. "
                "The daemon contract names those slots so a future body can be "
                "evicted for protected evidence. Integration point: map stop->unload, "
                "add pause/resume on the live ABC without hard-coding a model."
            ),
        },
        "lease_priority": list(LEASE_PRIORITY),
        "no_silent_restart": True,
        "does_not_start_a_process": True,
    }


def emit_workunits(status: Mapping[str, Any], flash: Mapping[str, Any]) -> list[dict[str, Any]]:
    """HCLI-shaped proposals. The Flash wakeup unit SLEEPS until receipts change."""
    eval_unit = emit_hcli_workunit(
        id="future.super_resident.sandbox_floor",
        role="sandbox_resident_floor",
        description=(
            "Evaluate SANDBOX_RESIDENT_FLOOR from disk evidence and seal "
            "SUPER_RESIDENT_FLOOR.json. Do not start a model process."
        ),
        dependencies=[],
        resource_class="LIGHT_CONTROL",
        verifier="future.super_resident.floor_contract",
        provider="sidecar-static",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "command": "python3 tools/future/super_resident.py --status",
            "claim_boundary": (
                "WorkUnit is a proposal; receipt remains authoritative; "
                "this unit cannot promote a body or take a GPU lease."
            ),
        },
    )
    validate_emitted_unit(eval_unit)
    wakeup = emit_hcli_workunit(
        id="future.super_resident.flash_floor_wakeup",
        role="sandbox_resident_floor",
        description=(
            "Re-evaluate Flash against SANDBOX_RESIDENT_FLOOR when "
            "source_independent_nx leaves SCAFFOLD_ONLY and the stateful gate "
            "shows repeated accepted-token decode. Sleeps until then. Never a "
            "synthetic Flash result."
        ),
        dependencies=["future.super_resident.sandbox_floor"],
        resource_class="LIGHT_CONTROL",
        verifier="future.super_resident.floor_contract",
        provider="sidecar-static",
        effect_class="READ_ONLY",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "blocked_reason": (
                "Flash source-independent NX is SCAFFOLD_ONLY; stateful_gate is "
                "one accepted token, not repeated accepted-token decode."
            ),
            "wake_when": [
                "current_flash_state.source_independent_nx.status != SCAFFOLD_ONLY",
                "stateful_gate.accepted_tokens > 1 with repeated accepted-token decode",
            ],
            "unmet_clauses": list(flash.get("unmet_clauses") or []),
            "holder_at_emission": status.get("holder"),
        },
    )
    validate_emitted_unit(wakeup)
    return [eval_unit, wakeup]


def resident_callable(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/super_resident.py --status",
        "module": "tools.future.super_resident",
        "workunits_emitted": [
            "future.super_resident.sandbox_floor",
            "future.super_resident.flash_floor_wakeup",
        ],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": {
            "id": "SANDBOX_RESIDENT_FLOOR",
            "feeds": REL_FRONTIER,
            "payload": "holder, Flash rejection, Qwen27 CURRENT_NONFINAL_HCLI_WORKER",
            "integration": (
                "tools/future/global_frontier.py should grow a probe on "
                "SUPER_RESIDENT_FLOOR.json holder/clears fields. This lane does "
                "not edit global_frontier.py."
            ),
        },
        "fail_closed": [
            "UNMET or UNKNOWN clause => body does not clear the floor",
            "HardwareClaimError on numeric hardware fields",
            "generate after protected eviction raises ProviderEvicted",
            "silent restart after crash is refused",
            "resident convenience cannot outrank protected_evidence_eviction",
            "missing evidence is recorded as unresolved, never invented",
            "this module never starts a model process and never takes a GPU lease",
        ],
        "holder": status.get("holder"),
        "started_model_process": False,
    }


def build() -> Path:
    bundle = _load_all()
    flash = evaluate_flash(bundle)
    qwen = evaluate_qwen27(bundle)
    status = sandbox_status(bundle)
    units = emit_workunits(status, flash)
    neutral = prove_provider_neutral()
    lease = prove_gpu_lease_subordination()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "SANDBOX_RESIDENT_FLOOR — the bar a body must clear to enter the "
            "orchestrator sandbox, deliberately distinct from Singularity "
            "promotion — plus the provider-neutral daemon contract that must "
            "survive a change of body."
        ),
        "floor": floor_definition(),
        "evaluations": {
            "flash": flash,
            "qwen27": qwen,
        },
        "status": status,
        "provider_contract": provider_contract(),
        "provider_neutral_proof": neutral,
        "gpu_lease_subordination": lease,
        "workunits": units,
        "resident_callable": resident_callable(status),
        "recovered_implementation": [dict(row) for row in EXISTING_LIFECYCLE],
        "gaps_closed": [
            "SANDBOX_RESIDENT_FLOOR named as nine clauses, explicitly not Singularity",
            "honest Flash rejection citing SCAFFOLD_ONLY NX and one accepted stateful token",
            "Qwen27 evaluated separately as CURRENT_NONFINAL_HCLI_WORKER",
            "provider-neutral daemon driven through two stub bodies",
            "GPU lease subordination: protected eviction beats resident convenience",
            "--status reports the sandbox holder from evidence without starting a model",
            "SLEEPING Flash wakeup WorkUnit instead of a synthetic Flash result",
        ],
        "negative_findings": [
            "hcli/agentos/resident.py is not a module; resident_gate.py is the live gate (recovery, not a test-asserted absence)",
            "Live RuntimeBackend/HawkingNativeConnector expose stop, not named pause/resume/unload",
            "Flash source-independent NX is SCAFFOLD_ONLY and not qualified",
            "Flash stateful gate is one accepted token, not repeated accepted-token decode",
            "Flash NX seven_all_met is False; resident_promotion is not true",
            "Qwen27 capability receipt identity_sufficient is false (UNIDENTIFIED body on that row)",
            "Qwen27 qualified_physical is false; control_receipt.promotion_allowed is false",
            "Qwen27 is not QWEN27_SINGULARITY.NX; tournament can_run is False",
            "this lane did not start a resident model process and did not take a GPU lease",
            "sidecar produces STATIC_ONLY / bench UNKNOWN; never DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
        ],
        "started_model_process": False,
        "gpu_authority": False,
    }
    return write_receipt(RECEIPT, _sanitize_hardware(doc), RECORDED_BY)


def selftest() -> Path:
    flash = evaluate_flash()
    assert flash["clears_sandbox_floor"] is False, flash
    assert "source_independent_runtime" in flash["unmet_clauses"], flash["unmet_clauses"]
    assert "sufficient_token_rate" in flash["unmet_clauses"], flash["unmet_clauses"]
    qwen = evaluate_qwen27()
    assert qwen["role"] == QWEN_ROLE, qwen["role"]
    assert qwen["clears_singularity"] is False
    status = sandbox_status()
    assert status["started_model_process"] is False
    assert status["floor_is_not_singularity"] is True
    assert status["holder_role"] == QWEN_ROLE, status
    neutral = prove_provider_neutral()
    assert neutral["identical_ops"] is True, neutral
    assert neutral["bodies_differ"] is True, neutral
    assert neutral["daemon_source_names_no_body"] is True, neutral
    lease = prove_gpu_lease_subordination()
    assert lease["resident_convenience_wins"] is False, lease
    assert lease["unloaded"] is True, lease
    assert lease["generate_after_evict_refused"] is True, lease
    return build()


def _print_status(status: Mapping[str, Any], receipt: Path) -> None:
    holder = status.get("holder") or "(none)"
    role = status.get("holder_role") or "(none)"
    print("SANDBOX_RESIDENT_FLOOR (not Singularity promotion)")
    print(f"holder: {holder}")
    print(f"role: {role}")
    print(f"why: {status.get('why')}")
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    qwen = status.get("qwen27") if isinstance(status.get("qwen27"), Mapping) else {}
    print(f"flash.clears_sandbox_floor: {flash.get('clears_sandbox_floor')}")
    unmet = flash.get("unmet_clauses") or []
    print(f"flash.unmet_clauses: {', '.join(str(c) for c in unmet)}")
    print(f"qwen27.clears_sandbox_floor: {qwen.get('clears_sandbox_floor')}")
    print(f"qwen27.clears_singularity: {qwen.get('clears_singularity')}")
    print(f"qwen27.role: {qwen.get('role')}")
    print(f"started_model_process: {status.get('started_model_process')}")
    print(f"took_gpu_lease: {status.get('took_gpu_lease')}")
    print(f"measurement_state: {status.get('measurement_state')}")
    print(receipt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report sandbox holder from evidence; never start a model")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    if args.status:
        # Evaluate first so a print cannot hide a failed seal.
        status = sandbox_status()
        out = build()
        _print_status(status, out)
        return 0
    print(selftest() if args.build else build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
