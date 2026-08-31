"""FLASH_NX_AUDIT — what stands between Flash NR V2 and a promotable NX.

The physical qualification queue's BLOCKED Flash rows collapse, except for
two specialized extras, to one missing dependency: a source-independent Flash
NX with a protected complete-token measurement. This module audits disk state
only. It does not compile, load, or time anything. Disk counts win over any
stale 12-of-14 wording in the frontier.

    python3 tools/future/flash_nx_audit.py --audit
    python3 tools/future/flash_nx_audit.py --check-nx receipts/headless/FLASH_COMPLETE_V0.nx.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from pathlib import Path
from typing import Any

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "FLASH_NX_COMPLETENESS_AUDIT.json"
SCHEMA = "hawking.future.flash_nx_audit.v1"

# Directive complete-executable authority rule. Order is the rule's order.
SEVEN_REQUIREMENTS: tuple[str, ...] = (
    "complete_byte_ledger",
    "self_contained_dependencies",
    "accepted_generation",
    "capability",
    "reproducibility",
    "protected_performance",
    "no_forbidden_fallback",
)

REL_QUEUE = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
REL_NR_V2 = "receipts/headless/FLASH_COMPLETE_V2.nr.json"
REL_NX_V0 = "receipts/headless/FLASH_COMPLETE_V0.nx.json"
REL_NX_V1 = "receipts/headless/FLASH_COMPLETE_V1.nx.json"
REL_NX_V2 = "receipts/headless/FLASH_COMPLETE_V2.nx.json"
REL_NX_NEXT = "receipts/headless/FLASH_NEXT_MACHINE.nx.json"
REL_META = "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
REL_LEDGER = "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json"
REL_CENSUS = "receipts/headless/FLASH_ORGAN_CENSUS.json"
REL_EXECUTABLE = "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json"
REL_TOKEN_ATTEMPT = "receipts/headless/FLASH_COMPLETE_TOKEN_NATIVE_ATTEMPT.json"
REL_TOKEN_DEVICE = "receipts/headless/FLASH_COMPLETE_TOKEN_DEVICE_RESIDENT_V1.json"
REL_TOKEN_ACCEPTED = "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json"
REL_TOKEN_SESSION = "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json"
REL_TOKEN_NS = "receipts/headless/FLASH_TOKEN_NS_BUDGET.json"
REL_EBPW = "receipts/headless/FLASH_EBPW_BUDGET.json"
REL_FRONTIER = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"

METADATA_ONLY = "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"
NR_NOT_FOR_PROMOTION = "COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION"

# Extra missing objects named by blocked_reason that an exact-control NX
# would not itself provide. Everyone else collapses to NX + complete-token.
_ADDITIONAL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "serialized_functional_artifact",
        (
            "serialized functional artifact",
            "meta budget is only a prospective",
        ),
    ),
    (
        "compact_expert_consumer",
        ("compact expert consumer",),
    ),
)


# ---------------------------------------------------------------------------
# Evidence root. Untracked Codex receipts live on the git-common working tree;
# this sparse worktree only materializes tracked HEAD paths.
# ---------------------------------------------------------------------------

def evidence_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        p = Path(common)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        else:
            p = p.resolve()
        parent = p.parent if p.name == ".git" else p
        if parent not in roots and parent.is_dir():
            roots.append(parent)
    return roots


def evidence_path(rel: str) -> Path | None:
    for root in evidence_roots():
        p = root / rel
        if p.is_file():
            return p
    return None


def load_evidence(rel: str) -> dict[str, Any]:
    p = evidence_path(rel)
    if p is None:
        searched = [str(r / rel) for r in evidence_roots()]
        raise FileNotFoundError(f"missing evidence {rel!r}; searched {searched}")
    return load_json(p)


def evidence_location(rel: str) -> dict[str, Any]:
    p = evidence_path(rel)
    in_tree = False
    if p is not None:
        resolved = p.resolve()
        repo = REPO.resolve()
        in_tree = resolved == repo / rel or repo in resolved.parents
    return {
        "rel": rel,
        "present": p is not None,
        "resolved": None if p is None else str(p),
        "in_this_worktree": in_tree,
        "tracked_in_head": bool(git("ls-files", "--error-unmatch", rel)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    return json.loads(json.dumps(value))


def _dot(doc: Any, dotted: str, default: Any = None) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _cite(rel: str, field: str, value: Any) -> dict[str, Any]:
    return {"path": rel, "field": field, "value": _jsonable(value)}


# ---------------------------------------------------------------------------
# NX completeness checker (reusable).
# ---------------------------------------------------------------------------

def _reason(req: str, ok: bool, state: str, cited_field: str, cited_value: Any, why: str) -> dict[str, Any]:
    return {
        "requirement": req,
        "ok": ok,
        "state": state,
        "cited_field": cited_field,
        "cited_value": _jsonable(cited_value),
        "why": why,
    }


def _status_is_metadata_only(nx: MappingLike) -> bool:
    status = str(nx.get("status") or "")
    return status == METADATA_ONLY or "METADATA_ONLY" in status or "NOT_FOR_PROMOTION" in status


def _serialized_artifact(nx: MappingLike) -> Any:
    return (
        nx.get("serialized_artifact")
        or _dot(nx, "physical_program.serialized_artifact")
        or _dot(nx, "artifact")
    )


def _loader(nx: MappingLike) -> Any:
    return nx.get("physical_loader") or nx.get("loader") or _dot(nx, "native_loader")


def _kernel(nx: MappingLike) -> Any:
    return nx.get("native_kernel") or nx.get("kernel_catalog") or _dot(nx, "native_kernels")


def _bench_state(nx: MappingLike) -> Any:
    return _dot(nx, "bench.state") or nx.get("bench_state") or _dot(nx, "protected_performance.bench_state")


def _measurement_class(nx: MappingLike) -> Any:
    return (
        nx.get("measurement_class")
        or _dot(nx, "protected_performance.measurement_class")
        or nx.get("benchmark_class")
    )


def _fallback_count(nx: MappingLike) -> Any:
    if "fallback_count" in nx:
        return nx.get("fallback_count")
    fb = nx.get("fallback")
    if isinstance(fb, dict) and "fallback_count" in fb:
        return fb.get("fallback_count")
    return _dot(nx, "qualification.fallback_count")


def _dense_flag(nx: MappingLike) -> Any:
    if "dense_rematerialization" in nx:
        return nx.get("dense_rematerialization")
    fb = nx.get("fallback")
    if isinstance(fb, dict) and "dense_rematerialization" in fb:
        return fb.get("dense_rematerialization")
    return _dot(nx, "accelerator_contract.dense_rematerialization")


def _byte_ledger_closed(nx: MappingLike, context: MappingLike | None) -> tuple[bool, str, str, Any, str]:
    """Complete-system ledger of the NX, not the exact-control 16.0 EBPW book."""
    ledger = nx.get("byte_ledger")
    if isinstance(ledger, dict):
        closed = (
            ledger.get("status") in {"CLOSED", "COMPLETE", "COMPLETE_SYSTEM_CLOSED"}
            and ledger.get("all_required_bytes_included") is True
            and ledger.get("complete_system") is True
        )
        field = "byte_ledger.status"
        value = {
            "status": ledger.get("status"),
            "all_required_bytes_included": ledger.get("all_required_bytes_included"),
            "complete_system": ledger.get("complete_system"),
        }
        if closed:
            return True, "MET", field, value, "NX carries a closed complete-system byte ledger"
        return False, "NOT_MET", field, value, "NX byte_ledger is present but not a closed complete-system ledger"
    ledger_path = _dot(nx, "qualification.byte_ledger")
    related = (context or {}).get("byte_ledger") if isinstance(context, dict) else None
    if isinstance(related, dict):
        allowed = related.get("promotion_allowed")
        compact = _dot(related, "routed_expert_sensitivity.complete_storage_bytes")
        status = related.get("status")
        field = "qualification.byte_ledger -> status/promotion_allowed"
        value = {
            "path": ledger_path,
            "status": status,
            "promotion_allowed": allowed,
            "routed_expert_sensitivity.complete_storage_bytes": compact,
        }
        if allowed is True and compact is not None and status not in (None, "MEASURED_EXACT_CONTROL_WITH_ROUTE_IO_PROFILE"):
            return True, "MET", field, value, "bound ledger is closed and promotion-allowed"
        return (
            False,
            "NOT_MET",
            field,
            value,
            "bound ledger is exact-control / I/O profile only; complete candidate ledger remains open",
        )
    return False, "NOT_MET", "byte_ledger", ledger, "no complete-system byte ledger on the NX"


def _self_contained(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    status = nx.get("status")
    art = _serialized_artifact(nx)
    loader = _loader(nx)
    kernel = _kernel(nx)
    if _status_is_metadata_only(nx):
        return (
            False,
            "NOT_MET",
            "status",
            status,
            "NX is a metadata seal, not a self-contained executable",
        )
    art_ok = False
    if isinstance(art, dict):
        art_ok = (
            art.get("status") not in {None, "NOT_BUILT", "ABSENT"}
            and bool(art.get("sha256") or art.get("digest"))
            and art.get("self_contained") is True
        )
    elif isinstance(art, str) and art not in {"", "NOT_BUILT"}:
        art_ok = True
    loader_ok = False
    if isinstance(loader, dict):
        loader_ok = loader.get("status") in {"BUILT", "IMPLEMENTED", "PASSED"} and loader.get("source_independent") is True
    kernel_ok = False
    if isinstance(kernel, dict):
        kernel_ok = kernel.get("status") in {"BUILT", "IMPLEMENTED", "PASSED", "BOUND"}
    if art_ok and loader_ok and kernel_ok:
        return True, "MET", "serialized_artifact+physical_loader+native_kernel", {
            "serialized_artifact": _jsonable(art),
            "physical_loader": _jsonable(loader),
            "native_kernel": _jsonable(kernel),
        }, "serialized artifact, source-independent loader, and native kernel catalog are present"
    return False, "NOT_MET", "serialized_artifact", _jsonable(art), (
        "missing serialized artifact and/or source-independent loader/kernel; "
        "source-bound executors are not a self-contained NX"
    )


def _accepted_generation(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    ag = nx.get("accepted_generation")
    if isinstance(ag, dict):
        ok = ag.get("status") == "PASSED" and ag.get("multi_token") is True
        return ok, "MET" if ok else "NOT_MET", "accepted_generation", _jsonable(ag), (
            "multi-token accepted generation passed" if ok
            else "accepted_generation is present but not a passed multi-token session"
        )
    tps = _dot(nx, "qualification.accepted_multitoken_tps")
    if tps is not None:
        # A numeric TPS would be a hardware claim. Presence of a non-null
        # field without a PROTECTED measurement class is still not enough.
        return False, "NOT_MET", "qualification.accepted_multitoken_tps", tps, (
            "accepted_multitoken_tps is set without an accepted_generation PASSED block"
        )
    return False, "NOT_MET", "qualification.accepted_multitoken_tps", tps, (
        "no multi-token accepted generation; a single terminal argmax is not accepted generation"
    )


def _capability(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    cap = nx.get("capability") or nx.get("capability_contract")
    if isinstance(cap, dict):
        status = cap.get("status")
        ok = status == "PASSED"
        return ok, "MET" if ok else "NOT_MET", "capability.status" if "capability" in nx else "capability_contract.status", status, (
            "capability suite passed" if ok else "capability suite has not passed"
        )
    return False, "NOT_MET", "capability", None, "no capability suite result on the NX"


def _reproducibility(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    repro = nx.get("reproducibility")
    if isinstance(repro, dict):
        ok = repro.get("status") == "PASSED" and repro.get("byte_reproducible") is True and bool(repro.get("closure_sha256"))
        return ok, "MET" if ok else "NOT_MET", "reproducibility", _jsonable(repro), (
            "content-addressed executable closure is byte-reproducible" if ok
            else "reproducibility block is present but not a passed byte-reproducible closure"
        )
    art = _serialized_artifact(nx)
    genome = _dot(nx, "compiled_for_machine_genome.genome_digest")
    # Source hashes of .rs/.metal files are a metadata seal, not an executable closure.
    if isinstance(art, dict) and art.get("sha256") and genome and not _status_is_metadata_only(nx):
        return False, "NOT_MET", "reproducibility", None, (
            "artifact hash and machine genome exist but byte_reproducible closure is not declared"
        )
    return False, "NOT_MET", "reproducibility", None, (
        "no executable closure hash; metadata hashes of source files are not byte reproducibility"
    )


def _protected_performance(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    perf = nx.get("protected_performance") if isinstance(nx.get("protected_performance"), dict) else {}
    klass = _measurement_class(nx)
    bench = _bench_state(nx)
    measured = perf.get("complete_token_measured") if isinstance(perf, dict) else None
    window = perf.get("protected_window") if isinstance(perf, dict) else nx.get("protected_window")
    ok = (
        klass in {"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"}
        and bench == "QUIESCED"
        and measured is True
        and window is True
    )
    value = {
        "measurement_class": klass,
        "bench.state": bench,
        "complete_token_measured": measured,
        "protected_window": window,
    }
    if ok:
        return True, "MET", "protected_performance", value, "protected complete-token measurement under a QUIESCED lease"
    why = "no PROTECTED_ABSOLUTE complete-token measurement"
    if bench == "UNKNOWN":
        why = "bench.state is UNKNOWN; DIAGNOSTIC_RELATIVE and metadata seals do not decide"
    return False, "NOT_MET", "bench.state", bench, why


def _no_forbidden_fallback(nx: MappingLike) -> tuple[bool, str, str, Any, str]:
    count = _fallback_count(nx)
    dense = _dense_flag(nx)
    dense_ok = dense in {False, "forbidden", "FORBIDDEN_BY_FINAL_RUNTIME_POLICY"}
    count_ok = isinstance(count, int) and not isinstance(count, bool) and count == 0
    source_ind = None
    loader = _loader(nx)
    if isinstance(loader, dict):
        source_ind = loader.get("source_independent")
    ag = nx.get("accepted_generation")
    if isinstance(ag, dict) and source_ind is None:
        source_ind = ag.get("source_independent")
    if nx.get("source_independent") is True:
        source_ind = True
    ok = count_ok and dense_ok and source_ind is True
    value = {
        "fallback_count": count,
        "dense_rematerialization": dense,
        "source_independent": source_ind,
    }
    if ok:
        return True, "MET", "fallback_count", value, "zero disclosed fallbacks, no dense rematerialization, source-independent"
    if count is None:
        return False, "NOT_MET", "fallback_count", value, "fallback_count is undisclosed; missing is not zero"
    return False, "NOT_MET", "fallback_count", value, "forbidden fallback, dense rematerialization, or source-oracle path remains"


MappingLike = dict[str, Any]


_CHECKERS = {
    "complete_byte_ledger": lambda nx, ctx: _byte_ledger_closed(nx, ctx),
    "self_contained_dependencies": lambda nx, ctx: _self_contained(nx),
    "accepted_generation": lambda nx, ctx: _accepted_generation(nx),
    "capability": lambda nx, ctx: _capability(nx),
    "reproducibility": lambda nx, ctx: _reproducibility(nx),
    "protected_performance": lambda nx, ctx: _protected_performance(nx),
    "no_forbidden_fallback": lambda nx, ctx: _no_forbidden_fallback(nx),
}


def check_nx(nx: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return whether an NX document is promotable, and if not, exactly why.

    A guard nobody has watched fail is not a guard. The real
    FLASH_COMPLETE_V0.nx.json must be rejected as metadata-only.
    """
    if not isinstance(nx, dict):
        return {
            "promotable": False,
            "reasons": ["NX document is not an object"],
            "requirements": {},
            "status": None,
        }
    requirements: dict[str, Any] = {}
    reasons: list[str] = []
    for req in SEVEN_REQUIREMENTS:
        ok, state, field, value, why = _CHECKERS[req](nx, context)
        requirements[req] = _reason(req, ok, state, field, value, why)
        if not ok:
            reasons.append(f"{req}: {why} (cited {field}={value!r})")
    status = nx.get("status")
    if _status_is_metadata_only(nx):
        tag = f"status is {status}"
        if tag not in reasons:
            reasons.insert(0, tag)
    resident = _dot(nx, "qualification.resident_promotion")
    if resident is False:
        reasons.append("qualification.resident_promotion is false")
    promotable = all(row["ok"] for row in requirements.values()) and not _status_is_metadata_only(nx)
    return {
        "promotable": promotable,
        "status": status,
        "reasons": reasons,
        "requirements": requirements,
        "failed_requirements": [k for k, v in requirements.items() if not v["ok"]],
    }


def synthetic_promotable_nx() -> dict[str, Any]:
    """A document that genuinely satisfies all seven requirements.

    Used only as a negative-control inverse. It is not written to disk and
    makes no hardware measurement claim.
    """
    return {
        "schema": "hawking.flash.nx_genome.v1",
        "nx_kind": "hawking.nos.flash_noetic_executable_genome",
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "source_independent": True,
        "serialized_artifact": {
            "path": "synthetic/flash_complete.nxbin",
            "sha256": "0" * 64,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {
            "status": "BUILT",
            "sha256": "1" * 64,
            "source_independent": True,
        },
        "native_kernel": {
            "status": "BUILT",
            "catalog_sha256": "2" * 64,
        },
        "byte_ledger": {
            "status": "CLOSED",
            "complete_system": True,
            "all_required_bytes_included": True,
        },
        "accepted_generation": {
            "status": "PASSED",
            "multi_token": True,
            "source_independent": True,
        },
        "capability": {"status": "PASSED"},
        "reproducibility": {
            "status": "PASSED",
            "byte_reproducible": True,
            "closure_sha256": "3" * 64,
        },
        "protected_performance": {
            "measurement_class": "PROTECTED_ABSOLUTE",
            "bench_state": "QUIESCED",
            "complete_token_measured": True,
            "protected_window": True,
        },
        "fallback": {
            "fallback_count": 0,
            "dense_rematerialization": False,
        },
        "qualification": {"resident_promotion": True},
        "bench": {"state": "QUIESCED"},
        "compiled_for_machine_genome": {"genome_digest": "4" * 64},
    }


# ---------------------------------------------------------------------------
# Queue / dependency chain.
# ---------------------------------------------------------------------------

def flash_blocked_candidates(queue: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cand in queue.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        if cand.get("model") != "Flash" or cand.get("status") != "BLOCKED":
            continue
        rows.append(cand)
    rows.sort(key=lambda c: str(c.get("candidate_id") or ""))
    return rows


def extra_pieces_from_reason(reason: str) -> list[str]:
    text = (reason or "").lower()
    extra: list[str] = []
    for piece, markers in _ADDITIONAL_MARKERS:
        if any(m in text for m in markers):
            extra.append(piece)
    return extra


def cluster_of(reason: str) -> str:
    return "ADDITIONAL" if extra_pieces_from_reason(reason) else "DOMINANT"


def candidate_summary(cand: dict[str, Any]) -> dict[str, Any]:
    reason = cand.get("blocked_reason")
    extra = extra_pieces_from_reason(str(reason or ""))
    cid = cand.get("candidate_id")
    pieces = [
        "serialized_source_independent_nx_artifact",
        "physical_loader",
        "whole_model_native_kernel_binding",
        "protected_complete_token_measurement",
        "capability_suite_on_nx",
    ] + extra
    return {
        "candidate_id": cid,
        "status": cand.get("status"),
        "blocked_reason": reason,
        "capability_contract": cand.get("capability_contract"),
        "control_configuration": cand.get("control_configuration"),
        "baseline_path": cand.get("baseline_path"),
        "cluster": "ADDITIONAL" if extra else "DOMINANT",
        "extra_pieces": extra,
        "requires": pieces,
        "measurements_status": _dot(cand, "measurements.status"),
    }


def dependency_chain(blocked: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [candidate_summary(c) for c in blocked]
    dominant = [s["candidate_id"] for s in summaries if s["cluster"] == "DOMINANT"]
    additional = [s["candidate_id"] for s in summaries if s["cluster"] == "ADDITIONAL"]

    def who_needs(piece: str) -> list[str]:
        return [s["candidate_id"] for s in summaries if piece in s["requires"]]

    missing = [
        {
            "id": "serialized_source_independent_nx_artifact",
            "lane": "CPU",
            "state": "NOT_BUILT",
            "prerequisites": ["flash_nr_v2_exists"],
            "prerequisite_state": "MET",
            "unblocks_to_qualifiable": [],
            "still_required_by": who_needs("serialized_source_independent_nx_artifact"),
            "why_not_sufficient_alone": (
                "A packed artifact without a loader, kernels, and a protected "
                "complete-token receipt cannot move a BLOCKED row."
            ),
            "evidence": [
                _cite(REL_META, "measurement_state.serialized_artifact", "NOT_BUILT"),
                _cite(REL_NX_V0, "status", METADATA_ONLY),
            ],
        },
        {
            "id": "physical_loader",
            "lane": "CPU",
            "state": "NOT_IMPLEMENTED",
            "prerequisites": ["serialized_source_independent_nx_artifact"],
            "unblocks_to_qualifiable": [],
            "still_required_by": who_needs("physical_loader"),
            "evidence": [
                _cite(REL_META, "measurement_state.physical_loader", "NOT_BUILT"),
                _cite(REL_EXECUTABLE, "native_loader.status", "NOT_IMPLEMENTED"),
                _cite(REL_TOKEN_ATTEMPT, "status", "BLOCKED"),
            ],
        },
        {
            "id": "whole_model_native_kernel_binding",
            "lane": "CPU",
            "state": "PLAN_ONLY",
            "prerequisites": ["physical_loader"],
            "unblocks_to_qualifiable": [],
            "still_required_by": who_needs("whole_model_native_kernel_binding"),
            "evidence": [
                _cite(REL_META, "measurement_state.native_kernel", "NOT_BUILT"),
                _cite(REL_EXECUTABLE, "native_kernels.status", "PLAN_ONLY"),
            ],
        },
        {
            "id": "protected_complete_token_measurement",
            "lane": "GPU",
            "state": "NOT_MEASURED",
            "prerequisites": ["whole_model_native_kernel_binding"],
            "unblocks_to_qualifiable": dominant,
            "still_required_by": who_needs("protected_complete_token_measurement"),
            "why_this_is_the_gpu_window": (
                "Once artifact, loader, and kernels exist, one protected complete-token "
                "lease with capability and zero-fallback gates is the measurement that "
                "makes the largest number of BLOCKED rows qualifiable."
            ),
            "evidence": [
                _cite(REL_META, "measurement_state.complete_token", "NOT_MEASURED"),
                _cite(REL_TOKEN_NS, "status", "PLANNED_UNTIL_NATIVE_EXECUTION"),
                _cite(REL_NX_V0, "bench.state", "UNKNOWN"),
            ],
        },
        {
            "id": "capability_suite_on_nx",
            "lane": "GPU",
            "state": "NOT_RUN",
            "prerequisites": ["protected_complete_token_measurement"],
            "unblocks_to_qualifiable": dominant,
            "still_required_by": who_needs("capability_suite_on_nx"),
            "same_window_as": "protected_complete_token_measurement",
            "evidence": [
                _cite(REL_META, "measurement_state.capability", "NOT_MEASURED"),
                _cite(REL_EXECUTABLE, "capability_contract.status", "NOT_RUN"),
            ],
        },
        {
            "id": "compact_expert_consumer",
            "lane": "CPU_THEN_GPU",
            "state": "NOT_QUALIFIED",
            "prerequisites": ["protected_complete_token_measurement"],
            "unblocks_to_qualifiable": [
                s["candidate_id"] for s in summaries if "compact_expert_consumer" in s["extra_pieces"]
            ],
            "still_required_by": who_needs("compact_expert_consumer"),
            "evidence": [
                {
                    "path": REL_QUEUE,
                    "field": "candidates[flash-compact-moe-epilogue].blocked_reason",
                    "value": next(
                        (s["blocked_reason"] for s in summaries if s["candidate_id"] == "flash-compact-moe-epilogue"),
                        None,
                    ),
                }
            ],
        },
        {
            "id": "serialized_functional_artifact",
            "lane": "CPU_THEN_GPU",
            "state": "NOT_BUILT",
            "prerequisites": ["protected_complete_token_measurement"],
            "unblocks_to_qualifiable": [
                s["candidate_id"] for s in summaries if "serialized_functional_artifact" in s["extra_pieces"]
            ],
            "still_required_by": who_needs("serialized_functional_artifact"),
            "evidence": [
                _cite(REL_META, "measurement_state.serialized_artifact", "NOT_BUILT"),
                _cite(REL_META, "status", "PROSPECTIVE_META_ONLY"),
            ],
        },
    ]

    order = []
    for i, piece in enumerate(missing, start=1):
        order.append(
            {
                "rank": i,
                "id": piece["id"],
                "lane": piece["lane"],
                "state": piece["state"],
                "prerequisites": piece["prerequisites"],
                "unblocks_to_qualifiable": piece["unblocks_to_qualifiable"],
                "unblocks_count": len(piece["unblocks_to_qualifiable"]),
                "still_required_by_count": len(piece["still_required_by"]),
                "why_this_position": (
                    "Largest number of candidates become qualifiable at the first GPU "
                    "window, but that window is worthless until the CPU artifact/loader/"
                    "kernel chain exists."
                    if piece["id"] == "protected_complete_token_measurement"
                    else "Must exist before any later piece can be consumed."
                    if piece["lane"] == "CPU"
                    else "Specialized consumer; does not sit on the 12-wide path."
                    if piece["id"] in {"compact_expert_consumer", "serialized_functional_artifact"}
                    else "Same protected window as complete-token; listed after it because "
                    "the funnel promotion rule requires both."
                ),
            }
        )

    return {
        "blocked_count": len(summaries),
        "dominant_count": len(dominant),
        "additional_count": len(additional),
        "dominant_candidates": dominant,
        "additional_candidates": additional,
        "dominant_dependency": (
            "source-independent Flash NX with a protected complete-token measurement"
        ),
        "candidates": summaries,
        "missing_pieces": missing,
        "topological_order": order,
        "gpu_window_worth_the_most": {
            "piece": "protected_complete_token_measurement",
            "unblocks_to_qualifiable": dominant,
            "unblocks_count": len(dominant),
            "cpu_prerequisites_that_must_exist_first": [
                "serialized_source_independent_nx_artifact",
                "physical_loader",
                "whole_model_native_kernel_binding",
            ],
            "still_blocked_after_window": additional,
        },
    }


# ---------------------------------------------------------------------------
# Seven requirements against live receipts.
# ---------------------------------------------------------------------------

def seven_from_disk(docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    nx = docs["nx_v0"]
    nr = docs["nr_v2"]
    ledger = docs["ledger"]
    meta = docs["meta"]
    exe = docs["executable"]
    accepted = docs["token_accepted"]
    device = docs["token_device"]
    attempt = docs["token_attempt"]
    token_ns = docs["token_ns"]

    check = check_nx(nx, context={"byte_ledger": ledger, "meta": meta, "nr": nr})
    rows = []

    def row(req: str, state: str, cites: list[dict[str, Any]], note: str) -> dict[str, Any]:
        checker = check["requirements"][req]
        return {
            "requirement": req,
            "state": state,
            "cited": cites,
            "checker_on_FLASH_COMPLETE_V0_nx": checker,
            "note": note,
        }

    rows.append(row(
        "complete_byte_ledger",
        "NOT_MET",
        [
            _cite(REL_LEDGER, "status", ledger.get("status")),
            _cite(REL_LEDGER, "promotion_allowed", ledger.get("promotion_allowed")),
            _cite(REL_LEDGER, "routed_expert_sensitivity.complete_storage_bytes",
                  _dot(ledger, "routed_expert_sensitivity.complete_storage_bytes")),
            _cite(REL_LEDGER, "measured_fastpath_profile.complete_ebpw",
                  _dot(ledger, "measured_fastpath_profile.complete_ebpw")),
            _cite(REL_NR_V2, "promotion.blockers", _dot(nr, "promotion.blockers")),
            _cite(REL_EXECUTABLE, "promotion_gate.missing_or_refused",
                  _dot(exe, "promotion_gate.missing_or_refused")),
        ],
        (
            "FLASH_COMPLETE_V0.BYTE_LEDGER.json closes the exact-control 16.0 EBPW "
            "book and explicitly leaves compact/complete-candidate storage open. "
            "NR V2 still lists 'complete candidate byte ledger' as a promotion blocker. "
            "evaluate_flash_promotion reports the complete-system byte-field set missing."
        ),
    ))
    rows.append(row(
        "self_contained_dependencies",
        "NOT_MET",
        [
            _cite(REL_NX_V0, "status", nx.get("status")),
            _cite(REL_NX_V0, "physical_program.source_binding",
                  _dot(nx, "physical_program.source_binding")),
            _cite(REL_META, "measurement_state.serialized_artifact",
                  _dot(meta, "measurement_state.serialized_artifact")),
            _cite(REL_META, "measurement_state.physical_loader",
                  _dot(meta, "measurement_state.physical_loader")),
            _cite(REL_EXECUTABLE, "native_loader.status", _dot(exe, "native_loader.status")),
            _cite(REL_TOKEN_ATTEMPT, "first_physical_failure_boundary.stage",
                  _dot(attempt, "first_physical_failure_boundary.stage")),
        ],
        (
            "V0/V1/V2 NX files are metadata seals that bind source .rs executors and "
            "the Metal shader tree. There is no packed NX body. The packed Qwen38 "
            "greedy loader refuses the raw Flash specimen. native_loader.status is "
            "NOT_IMPLEMENTED. serialized_artifact and physical_loader are NOT_BUILT."
        ),
    ))
    rows.append(row(
        "accepted_generation",
        "NOT_MET",
        [
            _cite(REL_NX_V0, "qualification.accepted_multitoken_tps",
                  _dot(nx, "qualification.accepted_multitoken_tps")),
            _cite(REL_TOKEN_DEVICE, "status", device.get("status")),
            _cite(REL_TOKEN_DEVICE, "terminal_token.classification",
                  _dot(device, "terminal_token.classification")),
            _cite(REL_TOKEN_DEVICE, "terminal_token.accepted_generation_tokens",
                  _dot(device, "terminal_token.accepted_generation_tokens")),
            _cite(REL_TOKEN_ACCEPTED, "status", accepted.get("status")),
            _cite(REL_TOKEN_ACCEPTED, "accepted_generation_tokens",
                  accepted.get("accepted_generation_tokens")),
            _cite(REL_TOKEN_ACCEPTED, "promotion_allowed", accepted.get("promotion_allowed")),
            _cite(REL_TOKEN_SESSION, "status", docs["token_session"].get("status")),
        ],
        (
            "The device-resident receipt is a single terminal argmax probe, not "
            "multi-token accepted generation. The stateful acceptance receipt records "
            "one token and refuses promotion; continuation state is not advanced. "
            "The session receipt is PASSED_COMPLETE_FORWARD_CANDIDATE_REJECTED. "
            "NX accepted_multitoken_tps is null."
        ),
    ))
    rows.append(row(
        "capability",
        "NOT_MET",
        [
            _cite(REL_META, "measurement_state.capability",
                  _dot(meta, "measurement_state.capability")),
            _cite(REL_EXECUTABLE, "capability_contract.status",
                  _dot(exe, "capability_contract.status")),
            _cite(REL_NR_V2, "promotion.blockers", _dot(nr, "promotion.blockers")),
        ],
        (
            "No capability suite has been run on a source-independent Flash NX. "
            "FLASH_NEXT_NOETIC_EXECUTABLE.capability_contract.status is NOT_RUN. "
            "NR V2 lists 'capability suite' as a promotion blocker."
        ),
    ))
    rows.append(row(
        "reproducibility",
        "NOT_MET",
        [
            _cite(REL_NX_V0, "status", nx.get("status")),
            _cite(REL_EXECUTABLE, "runtime_genome.status", _dot(exe, "runtime_genome.status")),
            _cite(REL_EXECUTABLE, "runtime_genome.executable_sha256",
                  _dot(exe, "runtime_genome.executable_sha256")),
            _cite(REL_EXECUTABLE, "runtime_genome.loader_sha256",
                  _dot(exe, "runtime_genome.loader_sha256")),
            _cite(REL_EXECUTABLE, "promotion_gate.evidence.reproducible_protected_receipt",
                  _dot(exe, "promotion_gate.evidence.reproducible_protected_receipt")),
        ],
        (
            "The metadata NX hashes source files and a machine genome. There is no "
            "packed executable digest, no loader digest, and no byte-reproducible "
            "closure. runtime_genome.executable_sha256 and loader_sha256 are null. "
            "reproducible_protected_receipt is false."
        ),
    ))
    rows.append(row(
        "protected_performance",
        "NOT_MET",
        [
            _cite(REL_NX_V0, "bench.state", _dot(nx, "bench.state")),
            _cite(REL_META, "measurement_state.complete_token",
                  _dot(meta, "measurement_state.complete_token")),
            _cite(REL_EXECUTABLE, "complete_token_timing.status",
                  _dot(exe, "complete_token_timing.status")),
            _cite(REL_TOKEN_NS, "status", token_ns.get("status")),
            _cite(REL_QUEUE, "funnel.promotion_rule",
                  _dot(docs["queue"], "funnel.promotion_rule")),
            _cite(REL_TOKEN_ACCEPTED, "bench.state", _dot(accepted, "bench.state")),
            _cite(REL_TOKEN_DEVICE, "bench.state", _dot(device, "bench.state")),
        ],
        (
            "Every Flash NX/token receipt on disk has bench.state UNKNOWN. "
            "complete_token is NOT_MEASURED. The token-ns budget is "
            "PLANNED_UNTIL_NATIVE_EXECUTION. Queue promotion requires a protected "
            "complete-token receipt. Existing token receipts are not "
            "PROTECTED_ABSOLUTE and must not be read as such."
        ),
    ))
    rows.append(row(
        "no_forbidden_fallback",
        "NOT_MET",
        [
            _cite(REL_NR_V2, "representation.parts[routed_experts].representation",
                  next((p.get("representation") for p in _dot(nr, "representation.parts") or []
                        if p.get("family") == "routed_experts"), None)),
            _cite(REL_LEDGER, "complete_exact_control.representation",
                  _dot(ledger, "complete_exact_control.representation")),
            _cite(REL_EXECUTABLE, "fallback_count", exe.get("fallback_count")),
            _cite(REL_EXECUTABLE, "native_kernels.dense_rematerialization",
                  _dot(exe, "native_kernels.dense_rematerialization")),
            _cite(REL_TOKEN_ACCEPTED, "next", accepted.get("next")),
            _cite(REL_QUEUE, "queue_policy.flash_source_oracle_is_not_flash_nx",
                  _dot(docs["queue"], "queue_policy.flash_source_oracle_is_not_flash_nx")),
        ],
        (
            "fallback_count is undisclosed on the scaffold. Exact-control NR/ledger "
            "still name source_bf16_exact as the runtime fallback. The source oracle "
            "is policy-not-NX. Dense rematerialization is forbidden by policy and "
            "not evidenced as absent on a whole-model NX, because that NX is not built. "
            "The accepted-token next-step still asks to eliminate dense source-bank reloads."
        ),
    ))
    return rows


# ---------------------------------------------------------------------------
# Runtime designed vs built.
# ---------------------------------------------------------------------------

def runtime_dependency_audit(docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    exe = docs["executable"]
    meta = docs["meta"]
    nx = docs["nx_v0"]
    attempt = docs["token_attempt"]
    coverage = _dot(exe, "native_kernels.coverage") or []
    kernel_rows = []
    if isinstance(coverage, list):
        for row in coverage:
            if isinstance(row, dict):
                kernel_rows.append({
                    "organ": row.get("organ"),
                    "kernel": row.get("kernel"),
                    "status": row.get("status"),
                    "designed": True,
                    "built": row.get("status") not in {None, "NOT_IMPLEMENTED", "PLAN_ONLY"},
                    "wired_as_whole_model_nx_consumer": False,
                })
    return [
        {
            "need": "serialized_nx_artifact",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_META, "measurement_state.serialized_artifact",
                      _dot(meta, "measurement_state.serialized_artifact")),
                _cite(REL_NX_V0, "status", nx.get("status")),
            ],
            "note": "tools/flash_nx_genome.py emits a metadata seal, not a packed body.",
        },
        {
            "need": "physical_loader",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_EXECUTABLE, "native_loader.status", _dot(exe, "native_loader.status")),
                _cite(REL_EXECUTABLE, "native_loader.required", _dot(exe, "native_loader.required")),
                _cite(REL_META, "measurement_state.physical_loader",
                      _dot(meta, "measurement_state.physical_loader")),
            ],
            "note": (
                "flash_executable.py lists loader requirements. "
                "native_loader.status is NOT_IMPLEMENTED. Bounded Q4 body load exists "
                "for a 128-row expert slice; whole-model Flash load does not."
            ),
        },
        {
            "need": "native_kernel_catalog",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_EXECUTABLE, "native_kernels.status", _dot(exe, "native_kernels.status")),
                _cite(REL_META, "measurement_state.native_kernel",
                      _dot(meta, "measurement_state.native_kernel")),
            ],
            "organs": kernel_rows,
            "note": (
                "Coverage is a plan. Layer-0 selected-top-10 composition is bounded "
                "evidence, not a whole-model NX kernel catalog. Most organs are "
                "NOT_IMPLEMENTED."
            ),
        },
        {
            "need": "whole_model_nx_consumer",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_EXECUTABLE, "status", exe.get("status")),
                _cite(REL_EXECUTABLE, "runtime_genome.status", _dot(exe, "runtime_genome.status")),
            ],
            "note": "FLASH_NEXT_NOETIC_EXECUTABLE is SCAFFOLD_ONLY. Bounded multi-component is not a complete NX.",
        },
        {
            "need": "packed_qwen38_greedy_as_flash_loader",
            "designed": False,
            "built": True,
            "wired": False,
            "evidence": [
                _cite(REL_TOKEN_ATTEMPT, "status", attempt.get("status")),
                _cite(REL_TOKEN_ATTEMPT, "first_physical_failure_boundary.cause",
                      _dot(attempt, "first_physical_failure_boundary.cause")),
            ],
            "note": (
                "ascension_qwen38_hybrid_greedy is a real binary. It cannot open the "
                "pinned Flash specimen. Built for a different artifact class, not wired."
            ),
        },
        {
            "need": "source_oracle_executors",
            "designed": True,
            "built": True,
            "wired": False,
            "evidence": [
                _cite(REL_NX_V0, "physical_program.source_binding",
                      _dot(nx, "physical_program.source_binding")),
                _cite(REL_QUEUE, "queue_policy.flash_source_oracle_is_not_flash_nx",
                      _dot(docs["queue"], "queue_policy.flash_source_oracle_is_not_flash_nx")),
            ],
            "note": (
                "flash_fast_chain.rs / organ executors exist and are content-bound by "
                "the metadata NX. Queue policy forbids reading them as Flash NX."
            ),
        },
        {
            "need": "protected_complete_token_path",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_QUEUE, "measurement_contract.metric_scope",
                      _dot(docs["queue"], "measurement_contract.metric_scope")),
                _cite(REL_EXECUTABLE, "complete_token_timing.status",
                      _dot(exe, "complete_token_timing.status")),
            ],
            "note": "Contract exists. No native protected complete-token receipt exists for Flash NX.",
        },
        {
            "need": "capability_suite_on_nx",
            "designed": True,
            "built": False,
            "wired": False,
            "evidence": [
                _cite(REL_EXECUTABLE, "capability_contract.status",
                      _dot(exe, "capability_contract.status")),
            ],
            "note": "Required list is declared. Status is NOT_RUN. Declarations are not runtime evidence.",
        },
    ]


def dense_rematerialization_audit(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exe = docs["executable"]
    meta = docs["meta"]
    accepted = docs["token_accepted"]
    device = docs["token_device"]
    nr = docs["nr_v2"]
    reconstruct = _dot(exe, "chosen_representation.bounded_source_transform.independent_weight_reconstruction")
    reconstruct_present = isinstance(reconstruct, dict) and reconstruct.get("count") is not None
    return {
        "production_may_not_reconstruct_dense_weights": True,
        "verification_may_reconstruct": True,
        "policy": {
            "flash_executable_native_kernels": _dot(exe, "native_kernels.dense_rematerialization"),
            "flash_next_capability_required": _dot(exe, "capability_contract.required"),
            "meta_program.dense_weight_materialization": _dot(meta, "meta_program.dense_weight_materialization"),
            "meta_accelerator_contract.dense_rematerialization": _dot(meta, "accelerator_contract.dense_rematerialization"),
            "bounded_native_loader.dense_rematerialization": _dot(
                exe, "chosen_representation.bounded_source_transform.native_kernel_parity.native_loader.dense_rematerialization"
            ),
        },
        "verification_reconstructs": {
            "found": reconstruct_present,
            "where": (
                "FLASH_NEXT_NOETIC_EXECUTABLE.chosen_representation."
                "bounded_source_transform.independent_weight_reconstruction"
            ),
            "also": (
                "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP roundtrip.weight_reconstruction "
                "is a bounded descriptor probe, not production decode"
            ),
            "router_ab": _dot(
                exe, "source_router_representation_ab.physical_graph.representation.dense_rematerialization"
            ),
            "note": (
                "independent_weight_reconstruction is a verification metric on a "
                "bounded Q4 probe. in_memory_study_only appears on the router "
                "representation A/B. Neither is a production NX decode path."
            ),
        },
        "production_path": {
            "whole_model_nx_production": "NOT_BUILT",
            "current_runtime_representation": _dot(nr, "representation.candidate_variants[0].name"),
            "exact_control_runtime_ready": next(
                (v.get("runtime_ready") for v in _dot(nr, "representation.candidate_variants") or []
                 if v.get("name") == "exact_control"),
                None,
            ),
            "byte_ledger_exact_control_representation": _dot(
                docs["ledger"], "complete_exact_control.representation"
            ),
            "device_resident_token_representation": device.get("representation"),
            "stateful_accepted_next": accepted.get("next"),
            "finding": (
                "No whole-model Flash NX production path exists to rematerialize into. "
                "The live source-oracle / exact-control path IS dense source-BF16; it "
                "does not unpack a compact code into a dense parent as a hidden fallback "
                "because the compact NX is not built. The stateful accepted-token next "
                "step still names 'eliminate dense source-bank reloads', which is a "
                "production residual on the source path, not a compact-code rematerializer. "
                "The planned meta path forbids dense rematerialization and is NOT_BUILT."
            ),
        },
        "verdict": {
            "designed_production_rematerialization": "forbidden",
            "built_production_compact_path": "NOT_BUILT",
            "verification_reconstruction_present": reconstruct_present,
            "hidden_dense_rematerialization_on_whole_model_nx": "UNKNOWN_BECAUSE_NX_NOT_BUILT",
        },
    }


# ---------------------------------------------------------------------------
# Recovered implementation map.
# ---------------------------------------------------------------------------

def recovered_implementation(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    locations = {
        name: evidence_location(rel)
        for name, rel in {
            "queue": REL_QUEUE,
            "nr_v2": REL_NR_V2,
            "nx_v0": REL_NX_V0,
            "nx_v1": REL_NX_V1,
            "nx_v2": REL_NX_V2,
            "nx_next_machine": REL_NX_NEXT,
            "meta": REL_META,
            "ledger": REL_LEDGER,
            "census": REL_CENSUS,
            "executable": REL_EXECUTABLE,
            "token_attempt": REL_TOKEN_ATTEMPT,
            "token_device": REL_TOKEN_DEVICE,
            "token_accepted": REL_TOKEN_ACCEPTED,
            "token_session": REL_TOKEN_SESSION,
            "token_ns": REL_TOKEN_NS,
            "ebpw": REL_EBPW,
        }.items()
    }
    return {
        "existing_tools": [
            {
                "path": "tools/flash_complete_nr.py",
                "does": "composes the portable Flash NR; status COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION",
                "adequate_for_this_lane": False,
                "gap": "does not audit NX promotion or the qualification-queue fan-out",
            },
            {
                "path": "tools/flash_nx_genome.py",
                "does": "seals a machine-bound metadata NX; status SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
                "adequate_for_this_lane": False,
                "gap": "explicitly not a packed executable and not a completeness checker",
            },
            {
                "path": "tools/flash_complete_byte_ledger.py",
                "does": "closes the exact-control 16.0 EBPW book with a route I/O profile",
                "adequate_for_this_lane": False,
                "gap": "compact/complete-candidate ledger remains open; not an NX promotion checker",
            },
            {
                "path": "tools/flash_complete_token_receipt.py",
                "does": "transforms a 0..47 chain plus one terminal probe into a single-token receipt",
                "adequate_for_this_lane": False,
                "gap": "refuses to invent accepted TPS; not a protected complete-token path",
            },
            {
                "path": "tools/flash_organ_census.py",
                "does": "index/header-only organ census",
                "adequate_for_this_lane": False,
                "gap": "STRUCTURAL_METADATA_SCREEN; not an NX",
            },
            {
                "path": "tools/flash_meta_representation.py",
                "does": "prospective sub-1 meta budget; measurement_state lists NOT_BUILT/NOT_MEASURED",
                "adequate_for_this_lane": False,
                "gap": "the NOT_BUILT fields are inputs to this audit, not a substitute for it",
            },
            {
                "path": "hcli/flash_next.py",
                "does": "evaluate_flash_promotion — the hard Flash-Next joint promotion law",
                "adequate_for_this_lane": False,
                "gap": (
                    "Codex-owned; cannot be modified here. It does not name the seven "
                    "directive requirements, does not map missing pieces onto the "
                    "BLOCKED Flash candidates, and does not distinguish designed vs built."
                ),
            },
            {
                "path": "hcli/agentos/flash_executable.py",
                "does": "scaffold for FLASH_NEXT_NOETIC_EXECUTABLE",
                "adequate_for_this_lane": False,
                "gap": "status SCAFFOLD_ONLY; native_loader NOT_IMPLEMENTED; native_kernels PLAN_ONLY",
            },
            {
                "path": "tools/accelerator/physical_qualification.py",
                "does": "the candidate queue this audit reads",
                "adequate_for_this_lane": False,
                "gap": "Codex-owned queue builder; sidecar must not mutate it",
            },
            {
                "path": "tools/nr_container.py / tools/nx_genome.py",
                "does": "Qwen38 NR/NX container and machine-genome sealer",
                "adequate_for_this_lane": False,
                "gap": "not Flash-complete; Flash has its own flash_nx_genome.py",
            },
        ],
        "existing_receipts": locations,
        "untracked_on_origin_working_tree": [
            rel for rel, loc in locations.items()
            if loc["present"] and not loc["tracked_in_head"]
        ],
        "this_worktree_is_sparse": True,
        "fork_decision": (
            "Nothing on disk is an NX completeness checker that maps the seven "
            "requirements onto the BLOCKED Flash candidates. This module adds "
            "that checker and the dependency chain. It does not fork flash_nx_genome.py "
            "or evaluate_flash_promotion."
        ),
    }


def negative_findings(docs: dict[str, dict[str, Any]], recovered: dict[str, Any]) -> list[dict[str, Any]]:
    findings = [
        {
            "looked_for": "a packed source-independent Flash NX body",
            "found": "FLASH_COMPLETE_V0/V1/V2.nx.json and FLASH_NEXT_MACHINE.nx.json, all SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        },
        {
            "looked_for": "a physical loader that opens Flash as NX",
            "found": "native_loader.status NOT_IMPLEMENTED; FLASH_COMPLETE_TOKEN_NATIVE_ATTEMPT BLOCKED at whole_token_weight_loader",
        },
        {
            "looked_for": "a PROTECTED_ABSOLUTE Flash complete-token receipt",
            "found": "single-terminal and one-token stateful receipts with bench.state UNKNOWN; complete_token NOT_MEASURED",
        },
        {
            "looked_for": "capability suite on Flash NX",
            "found": "capability_contract.status NOT_RUN",
        },
        {
            "looked_for": "complete-system byte ledger with the Flash-Next accounting categories",
            "found": "exact-control 16.0 EBPW ledger; promotion_allowed false; compact storage bytes null",
        },
        {
            "looked_for": "whole-model native kernel catalog bound to an NX",
            "found": "native_kernels.status PLAN_ONLY; most organs NOT_IMPLEMENTED",
        },
        {
            "looked_for": "hardware measurement",
            "found": "sidecar has no GPU; all numbers that would require a lease remain unclaimed here",
        },
    ]
    missing = [name for name, loc in recovered["existing_receipts"].items() if not loc["present"]]
    if missing:
        findings.append({"looked_for": "required evidence files in this worktree", "found_missing": missing})
    return findings


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------

def _load_all() -> dict[str, dict[str, Any]]:
    return {
        "queue": load_evidence(REL_QUEUE),
        "nr_v2": load_evidence(REL_NR_V2),
        "nx_v0": load_evidence(REL_NX_V0),
        "nx_v1": load_evidence(REL_NX_V1),
        "nx_v2": load_evidence(REL_NX_V2),
        "nx_next": load_evidence(REL_NX_NEXT),
        "meta": load_evidence(REL_META),
        "ledger": load_evidence(REL_LEDGER),
        "census": load_evidence(REL_CENSUS),
        "executable": load_evidence(REL_EXECUTABLE),
        "token_attempt": load_evidence(REL_TOKEN_ATTEMPT),
        "token_device": load_evidence(REL_TOKEN_DEVICE),
        "token_accepted": load_evidence(REL_TOKEN_ACCEPTED),
        "token_session": load_evidence(REL_TOKEN_SESSION),
        "token_ns": load_evidence(REL_TOKEN_NS),
        "ebpw": load_evidence(REL_EBPW),
    }


def assemble(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blocked = flash_blocked_candidates(docs["queue"])
    chain = dependency_chain(blocked)
    recovered = recovered_implementation(docs)
    nx_v0_check = check_nx(
        docs["nx_v0"],
        context={"byte_ledger": docs["ledger"], "meta": docs["meta"], "nr": docs["nr_v2"]},
    )
    synth_check = check_nx(synthetic_promotable_nx())
    seven = seven_from_disk(docs)
    static = [
        c for c in (docs["queue"].get("candidates") or [])
        if isinstance(c, dict) and c.get("model") == "Flash" and c.get("status") == "STATIC_ONLY"
    ]
    return {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Static completeness audit of the path from Flash NR V2 to a "
            "promotable source-independent NX, and the queue fan-out that "
            "collapses onto that NX."
        ),
        "head": git("rev-parse", "HEAD"),
        "evidence_roots": [str(r) for r in evidence_roots()],
        "nr_v2": {
            "path": REL_NR_V2,
            "status": docs["nr_v2"].get("status"),
            "generation": docs["nr_v2"].get("generation"),
            "schema": docs["nr_v2"].get("schema"),
            "promotion": docs["nr_v2"].get("promotion"),
            "next": docs["nr_v2"].get("next"),
            "kernel_requirements": docs["nr_v2"].get("kernel_requirements"),
            "negative_science": docs["nr_v2"].get("negative_science"),
        },
        "existing_nx_seals": [
            {
                "path": rel,
                "status": docs[key].get("status"),
                "lowers_nr": _dot(docs[key], "lowers_nr.path"),
                "resident_promotion": _dot(docs[key], "qualification.resident_promotion"),
                "accepted_multitoken_tps": _dot(docs[key], "qualification.accepted_multitoken_tps"),
                "complete_system_ebpw": _dot(docs[key], "qualification.complete_system_ebpw"),
                "bench_state": _dot(docs[key], "bench.state"),
            }
            for rel, key in (
                (REL_NX_V0, "nx_v0"),
                (REL_NX_V1, "nx_v1"),
                (REL_NX_V2, "nx_v2"),
                (REL_NX_NEXT, "nx_next"),
            )
        ],
        "meta_measurement_state": docs["meta"].get("measurement_state"),
        "seven_requirements": seven,
        "seven_all_met": all(r["state"] == "MET" for r in seven),
        "frontier_vs_disk": {
            "frontier_F001_wording": "12 of 14 BLOCKED Flash candidates",
            "disk_blocked_flash": chain["blocked_count"],
            "disk_dominant": chain["dominant_count"],
            "disk_additional": chain["additional_count"],
            "added_since_frontier_wording": sorted(
                set(chain["dominant_candidates"]) - {
                    "flash-attention-gate-fusion",
                    "flash-compact-moe-bf16-vec4",
                    "flash-encoder-label-elision",
                    "flash-fullseq-catalog-cache",
                    "flash-fullseq-ordered-encoder",
                    "flash-hc-staged-threadgroup",
                    "flash-pipeline-cache-reuse",
                    "flash-qkv-gqa-rope-fusion",
                    "flash-routed-fp4-gate-up-swiglu-fused",
                    "flash-router-topk-fusion",
                    "flash-shared-fp8-gate-up-swiglu-fused",
                    "flash-source-bf16-simd",
                }
            ),
            "authority": "disk",
        },
        "blocked_flash_candidates": chain["candidates"],
        "static_only_flash_candidates": [
            {
                "candidate_id": c.get("candidate_id"),
                "status": c.get("status"),
                "blocked_reason": c.get("blocked_reason"),
            }
            for c in sorted(static, key=lambda x: str(x.get("candidate_id") or ""))
        ],
        "dependency_chain": chain,
        "runtime_dependency_audit": runtime_dependency_audit(docs),
        "dense_rematerialization": dense_rematerialization_audit(docs),
        "nx_completeness_checker": {
            "api": "check_nx(nx: dict, context: dict | None) -> {promotable, reasons, requirements}",
            "real_FLASH_COMPLETE_V0_nx": {
                "promotable": nx_v0_check["promotable"],
                "status": nx_v0_check["status"],
                "failed_requirements": nx_v0_check["failed_requirements"],
                "reasons": nx_v0_check["reasons"],
            },
            "synthetic_promotable_nx": {
                "promotable": synth_check["promotable"],
                "failed_requirements": synth_check["failed_requirements"],
                "written_to_disk": False,
            },
            "discriminator_holds": (
                nx_v0_check["promotable"] is False
                and METADATA_ONLY in str(nx_v0_check["status"])
                and synth_check["promotable"] is True
            ),
        },
        "recovered_implementation": recovered,
        "gaps_closed": [
            "reusable NX promotability checker with a watched refusal on the real V0 metadata NX",
            "seven-requirement scoring with exact receipt field citations",
            "dependency chain from missing pieces to each BLOCKED Flash candidate",
            "topological order that puts the dominant GPU window after the CPU artifact/loader/kernel chain",
            "designed-vs-built runtime audit for loader, kernel, catalog, and consumer",
            "verification-vs-production dense rematerialization split",
        ],
        "negative_findings": negative_findings(docs, recovered),
        "claim_boundary_audit": (
            "STATIC_ONLY. No hardware measurement. DIAGNOSTIC_RELATIVE token receipts "
            "are cited by status/path only and are not promoted to PROTECTED_ABSOLUTE."
        ),
    }


def build() -> Path:
    docs = _load_all()
    doc = assemble(docs)
    return write_receipt(RECEIPT, doc, "tools/future/flash_nx_audit.py")


selftest = build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="emit FLASH_NX_COMPLETENESS_AUDIT.json")
    ap.add_argument("--check-nx", metavar="PATH", help="run the completeness checker on an NX JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.check_nx:
        path = Path(a.check_nx)
        if not path.is_file():
            resolved = evidence_path(a.check_nx)
            if resolved is None:
                print(f"missing NX: {a.check_nx}", file=sys.stderr)
                return 2
            path = resolved
        nx = load_json(path)
        ctx = None
        ledger_p = evidence_path(REL_LEDGER)
        if ledger_p:
            ctx = {"byte_ledger": load_json(ledger_p)}
        result = check_nx(nx, context=ctx)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["promotable"] else 1
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
