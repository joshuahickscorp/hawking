"""NR_NX_PATH — split the seven NX misses into work-not-done vs physical.

nr_nx_path_callable is unmet because seven_all_met is False and
FLASH_COMPLETE_V0.nx is SEALED_METADATA_ONLY_NOT_FOR_PROMOTION. The launch
criterion currently reports every miss identically. A miss that needs a packed
artifact nobody has built is not the same as a miss that needs a GPU lease
this sidecar does not hold, and neither is the same as a miss labelled
"blocked on GPU" when the host was believed to have no Metal device.

This module refuses that inherited causal label. It reads each of the seven
from the audit, re-derives the blocker class from what is on disk TODAY
(Metal reachability, a 1024-row source-BF16 teacher capture, the qualification
lease check, contamination class), and advances the CPU contracts that can
move without packing an NX.

It does not pack an NX, does not claim physical EBPW, does not take a lease,
and does not lower a requirement. A Metal GPU being present is not GPU
authority, is not a QUIESCED bench, and is not a qualified NX.

    python3 tools/future/nr_nx_path.py --build
    python3 -m pytest tools/future/test_nr_nx_path.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import RECEIPTS, load_json, write_receipt
from tools.future import ebpw_categories as ebpw
from tools.future import flash_nx_audit as nx_audit
from tools.future import workunit_species as wus

RECEIPT = "NR_NX_PATH.json"
SCHEMA = "hawking.future.nr_nx_path.v1"
RECORDED_BY = "tools/future/nr_nx_path.py"
VERSION = 1

REL_AUDIT = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
REL_METAL = "receipts/future/METAL_REACHABILITY.json"
REL_SCREEN_1024 = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL1024.json"
REL_TEACHER_1024 = "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL1024.json"
REL_QUAL = "receipts/future/QUALIFICATION_PIPELINE.json"
REL_CONT = "receipts/future/CONTAMINATION_SCIENCE.json"
REL_TEACHER_CONTRACT = "receipts/future/TEACHER_CORPUS_CONTRACT.json"
REL_NR_COMPLETE = "receipts/future/FLASH_NR_COMPLETE.json"
REL_LAUNCH = "receipts/future/ODYSSEY_LAUNCH_GATE.json"

# A requirement is hardware-blocked only when a physical measurement or a
# protected lease is the thing that would satisfy it. Host-GPU absence is a
# separate, falsifiable claim and is never inferred from "not measured".
WORK_NOT_DONE = "WORK_NOT_DONE"
PRIOR_ARTIFACT = "PRIOR_ARTIFACT"
PHYSICAL_AUTHORITY = "PHYSICAL_AUTHORITY"
HOST_GPU_ABSENT_CONFIRMED = "HOST_GPU_ABSENT_CONFIRMED"
HOST_GPU_UNTESTED = "HOST_GPU_UNTESTED"

CPU_NEXT = "CPU_NEXT"
BLOCKED_ON_PRIOR = "BLOCKED_ON_PRIOR"
SLEEPING = "SLEEPING"

# Schedules a physical requirement is forbidden to wear. "pending" is the
# laundering this campaign already recorded as STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM.
FORBIDDEN_PHYSICAL_SCHEDULES = frozenset({"pending", "PENDING", "ready", "READY"})

PHYSICAL_EBPW_KEYS = frozenset(
    {
        "physical_ebpw",
        "complete_physical_ebpw",
        "qualified_complete_physical_ebpw",
    }
)

# Promotion-gate field set a closed complete-system ledger must carry.
# Copied from FLASH_NEXT_NOETIC_EXECUTABLE.promotion_gate.missing_or_refused
# (string names only). Filling any of these from exact-control BF16 bytes
# would launder source identity as compact storage.
EBPW_ACCOUNTING_FIELDS: tuple[str, ...] = (
    "ebpw_accounting.all_required_bytes_included=true",
    "ebpw_accounting.bases.bytes",
    "ebpw_accounting.dictionaries.bytes",
    "ebpw_accounting.expert_indices.bytes",
    "ebpw_accounting.generators.bytes",
    "ebpw_accounting.lookup_structures.bytes",
    "ebpw_accounting.mtp_representation.bytes",
    "ebpw_accounting.ngram_representation.bytes",
    "ebpw_accounting.required_executable_metadata.bytes",
    "ebpw_accounting.residuals.bytes",
    "ebpw_accounting.routing_metadata.bytes",
    "ebpw_accounting.scales.bytes",
    "ebpw_accounting.weight_codes.bytes",
    "ebpw_accounting.zero_points.bytes",
)

REPRO_DIGEST_FIELDS: tuple[str, ...] = (
    "runtime_genome.executable_sha256",
    "runtime_genome.loader_sha256",
    "runtime_genome.compiler_identity",
    "runtime_genome.device_identity",
    "reproducibility.closure_sha256",
)

SATISFYING_ARTIFACT: dict[str, dict[str, Any]] = {
    "complete_byte_ledger": {
        "kind": "closed complete-system byte ledger on the NX, not the exact-control 16.0 book",
        "path": "the NX itself (byte_ledger.*) or receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json with promotion_allowed and compact complete_storage_bytes",
        "predicates": (
            "byte_ledger.status in {CLOSED, COMPLETE, COMPLETE_SYSTEM_CLOSED}",
            "byte_ledger.all_required_bytes_included is True",
            "byte_ledger.complete_system is True",
            "OR bound ledger promotion_allowed is True AND complete_storage_bytes is not null AND status is not exact-control I/O profile",
        ),
    },
    "self_contained_dependencies": {
        "kind": "packed source-independent Flash NX body + physical loader + native kernel catalog",
        "path": "receipts/headless/FLASH_COMPLETE_V*.nx.json whose status is not SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "predicates": (
            "status is not METADATA_ONLY / NOT_FOR_PROMOTION",
            "serialized_artifact status not in {None, NOT_BUILT, ABSENT} and sha256 and self_contained is True",
            "physical_loader.status in {BUILT, IMPLEMENTED, PASSED} and source_independent is True",
            "native_kernel.status in {BUILT, IMPLEMENTED, PASSED, BOUND}",
        ),
    },
    "accepted_generation": {
        "kind": "multi-token accepted generation PASSED on the source-independent NX",
        "path": "the NX accepted_generation block; not FLASH_COMPLETE_TOKEN_DEVICE_RESIDENT_V1 and not a teacher mlp_input capture",
        "predicates": (
            "accepted_generation.status == PASSED",
            "accepted_generation.multi_token is True",
        ),
    },
    "capability": {
        "kind": "capability suite PASSED on the source-independent NX",
        "path": "the NX capability / capability_contract block; not FLASH_NEXT_NOETIC_EXECUTABLE.capability_contract.status=NOT_RUN",
        "predicates": ("capability.status == PASSED",),
    },
    "reproducibility": {
        "kind": "byte-reproducible executable closure of the packed NX",
        "path": "the NX reproducibility block plus runtime_genome executable_sha256 and loader_sha256",
        "predicates": (
            "reproducibility.status == PASSED",
            "reproducibility.byte_reproducible is True",
            "reproducibility.closure_sha256 is a 64-char digest",
        ),
    },
    "protected_performance": {
        "kind": "PROTECTED_ABSOLUTE complete-token measurement under a QUIESCED lease",
        "path": "the NX protected_performance / bench block; not DIAGNOSTIC_RELATIVE token receipts, not the teacher capture",
        "predicates": (
            "measurement_class in {PROTECTED_ABSOLUTE, QUALIFIED_PROTECTED}",
            "bench.state == QUIESCED",
            "complete_token_measured is True",
            "protected_window is True",
        ),
    },
    "no_forbidden_fallback": {
        "kind": "disclosed zero-fallback source-independent production NX with no dense rematerialization",
        "path": "the NX fallback / accepted_generation / physical_loader block; not a metadata seal of the source oracle",
        "predicates": (
            "fallback_count == 0 (int, disclosed; missing is not zero)",
            "dense_rematerialization in {False, forbidden, FORBIDDEN_BY_FINAL_RUNTIME_POLICY}",
            "source_independent is True",
        ),
    },
}

# Which of the seven need a protected GPU window to become MET. The rest
# become MET from CPU artifacts. This set is the schedule discriminator.
PHYSICAL_TO_SATISFY: frozenset[str] = frozenset(
    {"accepted_generation", "capability", "protected_performance"}
)

# CPU work that cannot start until a packed NX exists.
NEEDS_PACKED_NX: frozenset[str] = frozenset(
    {
        "complete_byte_ledger",
        "self_contained_dependencies",
        "reproducibility",
        "no_forbidden_fallback",
        "accepted_generation",
        "capability",
        "protected_performance",
    }
)


class ArtifactRequired(ValueError):
    """A requirement was offered as MET without an on-disk artifact."""


class PhysicalEbpwForbidden(ValueError):
    """This sidecar does not write a physical EBPW value."""


class FallbackDisclosureError(ValueError):
    """Stamping fallback_count=0 on a metadata / source-oracle NX is a lie."""


# ---------------------------------------------------------------------------
# Evidence. Missing here is not project-absence; it is a recorded refusal.
# ---------------------------------------------------------------------------


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_rel(rel: str) -> tuple[dict[str, Any] | None, str]:
    path = nx_audit.evidence_path(rel)
    if path is None:
        return None, "missing"
    try:
        doc = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{path}:{exc}"
    if not isinstance(doc, dict):
        return None, f"not_object:{path}"
    return doc, str(path)


def _xcrun_metal() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["xcrun", "-f", "metal"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        found = proc.returncode == 0 and bool((proc.stdout or "").strip())
        return {
            "probed": True,
            "found": found,
            "path": (proc.stdout or "").strip() or None,
            "returncode": proc.returncode,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probed": True, "found": False, "error": f"{type(exc).__name__}: {exc}"}


def today_host_facts() -> dict[str, Any]:
    """Metal picture as sealed today. Enumeration receipt, not a new probe.

    Re-running cargo metal probes would be a capability query we already have
    executed evidence for. Declared-not-executed is the failure this campaign
    refuses. gpu_authority stays false regardless of device presence.
    """
    rec, via = load_rel(REL_METAL)
    xcrun = _xcrun_metal()
    if rec is None:
        return {
            "metal_receipt": None,
            "metal_via": via,
            "verdict": "UNTESTED",
            "gpu_present": None,
            "gpu_present_state": "UNTESTED",
            "gpu_authority": False,
            "source_compile": None,
            "runtime_binding_device": None,
            "xcrun_metal": xcrun,
            "why": (
                "METAL_REACHABILITY.json is missing; that is not evidence the "
                "host has no GPU and is not evidence it does"
            ),
            "is_a_measurement": False,
        }
    observed = rec.get("observed") if isinstance(rec.get("observed"), Mapping) else {}
    runtime = rec.get("observed_runtime_binding") if isinstance(rec.get("observed_runtime_binding"), Mapping) else {}
    verdict_block = rec.get("verdict") if isinstance(rec.get("verdict"), Mapping) else {}
    verdict = verdict_block.get("verdict")
    device = observed.get("system_default") if isinstance(observed, Mapping) else None
    source_compile = runtime.get("runtime_source_compile") if isinstance(runtime, Mapping) else None
    gpu_present = bool(device)
    return {
        "metal_receipt": REL_METAL,
        "metal_via": via,
        "verdict": verdict,
        "gpu_present": gpu_present,
        "gpu_present_state": "PRESENT" if gpu_present else "ABSENT_IN_RECEIPT",
        "gpu_authority": False,
        "source_compile": source_compile,
        "runtime_binding_device": runtime.get("system_default") if isinstance(runtime, Mapping) else None,
        "metal_crate_version": runtime.get("metal_crate_version") if isinstance(runtime, Mapping) else None,
        "device_name": device,
        "n_devices": observed.get("n_devices") if isinstance(observed, Mapping) else None,
        "xcrun_metal": xcrun,
        "is_a_measurement": rec.get("is_a_measurement") is True,
        "why": verdict_block.get("why"),
        "what_this_does_not_establish": verdict_block.get("what_this_does_not_establish"),
    }


def today_teacher_facts() -> dict[str, Any]:
    """The 1024-row source-BF16 capture, if on disk. Not accepted generation."""
    screen, screen_via = load_rel(REL_SCREEN_1024)
    teacher, teacher_via = load_rel(REL_TEACHER_1024)
    contract, contract_via = load_rel(REL_TEACHER_CONTRACT)
    units = (contract.get("capture_workunits") or []) if isinstance(contract, Mapping) else []
    n_units = len(units) if isinstance(units, list) else 0
    n_executed = (
        sum(1 for u in units if isinstance(u, Mapping) and u.get("executed") is True)
        if isinstance(units, list)
        else 0
    )
    capture_present = teacher is not None
    rows = None
    status = None
    pipeline = None
    if isinstance(teacher, Mapping):
        status = teacher.get("status")
        rows = teacher.get("rows")
        if isinstance(rows, list):
            rows = len(rows)
        elif not isinstance(rows, int):
            rows = _dot(teacher, "teacher_trace.rows")
        pipeline = _dot(teacher, "teacher_trace.capture_binding.source_pipeline")
        if pipeline is None:
            pipeline = _dot(screen or {}, "teacher_trace.capture_binding.source_pipeline")
    screen_status = screen.get("status") if isinstance(screen, Mapping) else None
    return {
        "screen_via": screen_via,
        "teacher_via": teacher_via,
        "contract_via": contract_via,
        "capture_present": capture_present,
        "capture_status": status,
        "capture_rows": rows,
        "screen_status": screen_status,
        "source_pipeline": pipeline,
        "promotion_allowed": (
            teacher.get("promotion_allowed") if isinstance(teacher, Mapping) else None
        ),
        "physical_ebpw_on_screen": _dot(screen or {}, "representation.physical_ebpw"),
        "complete_token_on_screen": _dot(screen or {}, "measurement_state.complete_token"),
        "contract_workunits": n_units,
        "contract_executed": n_executed,
        "declared_vs_executed": (
            "TEACHER_CORPUS_CONTRACT.capture_workunits executed="
            f"{n_executed}/{n_units}; a source-BF16 L4 capture receipt "
            f"{'is' if capture_present else 'is not'} on disk. Naming the "
            "workunit is not evidence it ran; the capture receipt is."
        ),
    }


def today_lease_facts() -> dict[str, Any]:
    qual, q_via = load_rel(REL_QUAL)
    cont, c_via = load_rel(REL_CONT)
    stop = _dot(qual or {}, "dry_run_stop") if isinstance(qual, Mapping) else None
    return {
        "qualification_via": q_via,
        "contamination_via": c_via,
        "lease_proven": False,
        "dry_run_stop_reason": (stop or {}).get("reason") if isinstance(stop, Mapping) else None,
        "dry_run_stop_stage": (stop or {}).get("stage_id") if isinstance(stop, Mapping) else None,
        "contamination_class": cont.get("contamination_class") if isinstance(cont, Mapping) else None,
        "machine_quiesced": False,
        "why_lease_not_proven": (
            (stop or {}).get("reason")
            if isinstance(stop, Mapping) and stop.get("reason")
            else "qualification pipeline receipt missing or did not prove a holder pid; flock is a seizure and is never attempted"
        ),
        "why_not_quiesced": (
            f"contamination_class={None if not isinstance(cont, Mapping) else cont.get('contamination_class')!r}; "
            "HEAVY or unknown is not a protected window; sidecar will not SIGSTOP standing workers"
        ),
    }


# ---------------------------------------------------------------------------
# Guards. A validator nobody has watched reject is not a validator.
# ---------------------------------------------------------------------------


def record_physical_ebpw(value: Any) -> None:
    """There is no legitimate call. The raise is the implementation."""
    raise PhysicalEbpwForbidden(
        f"refused to write physical_ebpw={value!r}; this sidecar has no GPU "
        "authority and does not launder a description budget into a measurement"
    )


def _numeric_physical_ebpw(node: Any, path: str = "") -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in PHYSICAL_EBPW_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{here}={value!r}"
            found = _numeric_physical_ebpw(value, here)
            if found:
                return found
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found = _numeric_physical_ebpw(value, f"{path}[{i}]")
            if found:
                return found
    return None


def assert_no_physical_ebpw(doc: Mapping[str, Any]) -> None:
    found = _numeric_physical_ebpw(doc)
    if found:
        raise PhysicalEbpwForbidden(f"receipt carries a numeric physical EBPW: {found}")


def stamp_fallback_count(nx: Mapping[str, Any], count: int) -> None:
    """Refuse every stamp. Metadata zero is a lie; this lane does not write NX."""
    if nx_audit._status_is_metadata_only(nx) and count == 0:
        raise FallbackDisclosureError(
            "stamping fallback_count=0 on a metadata NX is a lie: the live "
            "runtime is the source oracle / exact-control path"
        )
    raise FallbackDisclosureError(
        "this module does not write fallback_count onto an NX; disclosure "
        "belongs on the packed executable, which is not built"
    )


def fallback_disclosure(nx: Mapping[str, Any]) -> dict[str, Any]:
    count = nx_audit._fallback_count(nx)
    metadata = nx_audit._status_is_metadata_only(nx)
    if metadata:
        return {
            "ok": False,
            "would_be_a_lie": True,
            "fallback_count": count,
            "why": (
                "metadata NX / source-oracle path is the runtime; missing "
                "fallback_count is not zero, and stamping 0 would hide the fallback"
            ),
        }
    if count is None:
        return {
            "ok": False,
            "would_be_a_lie": False,
            "fallback_count": None,
            "why": "fallback_count is undisclosed; missing is not zero",
        }
    if isinstance(count, int) and not isinstance(count, bool) and count == 0:
        source_ind = nx.get("source_independent")
        loader = nx_audit._loader(nx)
        if isinstance(loader, Mapping) and source_ind is None:
            source_ind = loader.get("source_independent")
        if source_ind is True:
            return {
                "ok": True,
                "would_be_a_lie": False,
                "fallback_count": 0,
                "why": "disclosed zero on a source-independent NX",
            }
        return {
            "ok": False,
            "would_be_a_lie": True,
            "fallback_count": 0,
            "why": "fallback_count=0 without source_independent is not the requirement",
        }
    return {
        "ok": False,
        "would_be_a_lie": False,
        "fallback_count": count,
        "why": "fallback_count is disclosed and is not zero",
    }


def judge_requirement(
    req: str,
    *,
    nx: Mapping[str, Any] | None,
    nx_path: str | Path | None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """MET only if check_nx says so AND the artifact is a real on-disk file.

    A synthetic promotable document that never touched disk is the exact
    Goodhart this guard exists to catch.
    """
    if req not in nx_audit.SEVEN_REQUIREMENTS:
        raise ValueError(f"unknown requirement {req!r}")
    path = Path(nx_path) if nx_path else None
    if path is None or not path.is_file():
        return {
            "requirement": req,
            "ok": False,
            "state": "REFUSED_WITHOUT_ARTIFACT",
            "why": "a requirement is never MET without its artifact on disk",
            "artifact": None,
        }
    if not isinstance(nx, Mapping):
        return {
            "requirement": req,
            "ok": False,
            "state": "REFUSED_WITHOUT_ARTIFACT",
            "why": "NX document is not an object",
            "artifact": str(path),
        }
    result = nx_audit.check_nx(dict(nx), context=dict(context) if isinstance(context, Mapping) else None)
    row = result["requirements"][req]
    if row["ok"] and nx_audit._status_is_metadata_only(nx):
        return {
            "requirement": req,
            "ok": False,
            "state": "REFUSED_METADATA_ONLY",
            "why": "check_nx must not satisfy a requirement on a metadata seal",
            "artifact": str(path),
            "checker": row,
        }
    out = dict(row)
    out["artifact"] = str(path)
    return out


def teacher_satisfies_accepted_generation(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The 1024-row capture is evidence the GPU ran source-BF16. It is not NX generation."""
    if not facts.get("capture_present"):
        return {
            "ok": False,
            "why": "teacher capture receipt is not on disk; absence is not accepted generation either",
        }
    return {
        "ok": False,
        "why": (
            f"teacher status={facts.get('capture_status')!r} rows={facts.get('capture_rows')!r} "
            f"pipeline={facts.get('source_pipeline')!r} is a bounded mlp_input source-BF16 "
            "surface. Queue policy: the source oracle is not Flash NX. A single organ "
            "hidden-state capture is not multi-token accepted generation on an NX."
        ),
        "capture_present": True,
        "screen_status": facts.get("screen_status"),
    }


def reclassify_causal_claim(text: str, host: Mapping[str, Any]) -> dict[str, Any]:
    """A status that names a cause this host no longer supports is laundering."""
    blob = (text or "").lower()
    host_absence = (
        "this host has no metal-capable gpu" in blob
        or "no metal-capable gpu on the host of record" in blob
        or "sidecar has no gpu" in blob
        and "authority" not in blob
    )
    if not host_absence:
        return {"verdict": "NOT_A_HOST_ABSENCE_CLAIM", "text": text}
    v = host.get("verdict")
    if v == "FALSIFIED_AS_A_HOST_PROPERTY":
        return {
            "verdict": "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
            "text": text,
            "today": (
                f"METAL_REACHABILITY verdict={v}; device={host.get('device_name')!r}; "
                "gpu_authority remains false"
            ),
            "what_remains_true": "sidecar has no GPU authority; no PROTECTED_ABSOLUTE NX measurement exists",
            "what_is_false": "this host has no Metal-capable GPU",
        }
    if v == "CONFIRMED":
        return {"verdict": "HOST_ABSENCE_CONFIRMED_BY_PROBE", "text": text, "today": v}
    return {
        "verdict": "HOST_ABSENCE_UNTESTED",
        "text": text,
        "today": v,
        "why": "the claim names host absence; the Metal receipt is not a FALSIFIED/CONFIRMED verdict",
    }


# ---------------------------------------------------------------------------
# CPU contracts that can move without packing an NX.
# ---------------------------------------------------------------------------


def complete_system_ledger_contract(
    ledger: Mapping[str, Any] | None,
    executable: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Exact field set a closed complete-system ledger must carry, scored on disk.

    Closing still requires packed compact bytes. This function does not invent
    them. Copying complete_exact_control.runtime_required_bytes into
    complete_storage_bytes would launder BF16 identity as compact storage.
    """
    if not isinstance(ledger, Mapping):
        return {
            "closed": False,
            "refused": True,
            "why": "BYTE_LEDGER is missing; an absent ledger is not a closed complete-system ledger",
            "missing_fields": list(EBPW_ACCOUNTING_FIELDS),
        }
    accounting = ledger.get("ebpw_accounting")
    missing_or_refused = _dot(executable or {}, "promotion_gate.missing_or_refused") or []
    if not isinstance(missing_or_refused, list):
        missing_or_refused = []
    needed = [s for s in missing_or_refused if str(s).startswith("ebpw_accounting")]
    if not needed:
        needed = list(EBPW_ACCOUNTING_FIELDS)
    fields = []
    for item in needed:
        fields.append(
            {
                "field": item,
                "present_on_byte_ledger": isinstance(accounting, Mapping) and item in (accounting or {}),
                "value": None,
            }
        )
    compact = _dot(ledger, "routed_expert_sensitivity.complete_storage_bytes")
    closed = (
        ledger.get("status") in {"CLOSED", "COMPLETE", "COMPLETE_SYSTEM_CLOSED"}
        and ledger.get("promotion_allowed") is True
        and compact is not None
        and isinstance(accounting, Mapping)
        and ledger.get("all_required_bytes_included") is True
        and ledger.get("complete_system") is True
    )
    return {
        "closed": False if not closed else True,
        "refused": not closed,
        "status": ledger.get("status"),
        "promotion_allowed": ledger.get("promotion_allowed"),
        "complete_storage_bytes_is_null": compact is None,
        "ebpw_accounting_present": isinstance(accounting, Mapping),
        "missing_fields": fields,
        "also_missing_from_promotion_gate": [
            s
            for s in missing_or_refused
            if s in {
                "complete-system byte ledger",
                "fallback_count disclosure",
                "native executable",
                "MTP accounting",
            }
        ],
        "refused_to_fill_from_exact_control": (
            "complete_exact_control is source_bf16_exact identity; copying "
            "runtime_required_bytes into complete_storage_bytes would launder "
            "exact-control as compact complete-system storage"
        ),
        "cpu_work_done_this_lane": (
            "field-set contract extracted from promotion_gate.missing_or_refused "
            "and scored against BYTE_LEDGER; ledger remains open"
        ),
        "why_cannot_close_here": (
            "compact/complete-candidate storage bytes are null; the packed NX "
            "body is NOT_BUILT; this lane's write list forbids packing; "
            "tools/flash_complete_byte_ledger.py is Codex-owned"
        ),
    }


def reproducibility_closure_inventory(executable: Mapping[str, Any] | None, nx: Mapping[str, Any] | None) -> dict[str, Any]:
    """Digests a byte-reproducible closure would hash. All null today."""
    rows = []
    sources = {
        "runtime_genome.executable_sha256": _dot(executable or {}, "runtime_genome.executable_sha256"),
        "runtime_genome.loader_sha256": _dot(executable or {}, "runtime_genome.loader_sha256"),
        "runtime_genome.compiler_identity": _dot(executable or {}, "runtime_genome.compiler_identity"),
        "runtime_genome.device_identity": _dot(executable or {}, "runtime_genome.device_identity"),
        "reproducibility.closure_sha256": _dot(nx or {}, "reproducibility.closure_sha256"),
    }
    for field in REPRO_DIGEST_FIELDS:
        value = sources.get(field)
        rows.append({"field": field, "present": bool(value), "value": value})
    return {
        "byte_reproducible": False,
        "all_digests_present": all(r["present"] for r in rows),
        "fields": rows,
        "why_cannot_hash_here": (
            "runtime_genome.executable_sha256 and loader_sha256 are null; "
            "metadata hashes of .rs/.metal source files are not an executable closure; "
            "this lane does not pack the NX body"
        ),
        "cpu_work_done_this_lane": "closure inventory scored against FLASH_NEXT_NOETIC_EXECUTABLE and V0 NX",
    }


def kernel_catalog_overlay(docs: Mapping[str, Any], host: Mapping[str, Any]) -> dict[str, Any]:
    """Designed-vs-built from the audit, with today's Metal compile wall overlaid.

    The audit classified whole_model_native_kernel_binding as CPU PLAN_ONLY.
    That lane is still right. What changed: source compile of the runtime
    metal crate is OK, so the absent offline compiler is a cost, not a wall.
    """
    exe = docs.get("executable") if isinstance(docs.get("executable"), Mapping) else {}
    coverage = _dot(exe, "native_kernels.coverage") or []
    compile_ok = host.get("source_compile") == "OK"
    organs = []
    if isinstance(coverage, list):
        for row in coverage:
            if not isinstance(row, Mapping):
                continue
            status = row.get("status")
            organs.append(
                {
                    "organ": row.get("organ"),
                    "kernel": row.get("kernel"),
                    "status": status,
                    "built": status not in {None, "NOT_IMPLEMENTED", "PLAN_ONLY", "UNRESOLVED"},
                    "metal_compile_is_a_host_wall": False if compile_ok else None,
                }
            )
    return {
        "native_kernels_status": _dot(exe, "native_kernels.status"),
        "dense_rematerialization_policy": _dot(exe, "native_kernels.dense_rematerialization"),
        "source_compile_today": host.get("source_compile"),
        "metal_compile_is_a_host_wall": False if compile_ok else host.get("source_compile") is None,
        "why_compile_is_not_the_wall": (
            "metal crate source compile returned OK on this host; load_or_compile_shader_library "
            "falls back to new_library_with_source; the missing xcrun precompile path is a cost"
            if compile_ok
            else "source compile was not OK in METAL_REACHABILITY; compile remains a possible wall"
        ),
        "organs": organs,
        "built_count": sum(1 for r in organs if r["built"]),
        "not_built_count": sum(1 for r in organs if not r["built"]),
        "still_plan_only": _dot(exe, "native_kernels.status") == "PLAN_ONLY",
    }


# ---------------------------------------------------------------------------
# Classification. States come from the audit; classes are re-derived today.
# ---------------------------------------------------------------------------


def seven_from_audit(audit_doc: Mapping[str, Any] | None, via: str) -> dict[str, Any]:
    if not isinstance(audit_doc, Mapping):
        return {
            "refused": True,
            "seven_all_met": None,
            "why": f"FLASH_NX_COMPLETENESS_AUDIT.json {via}; an absent audit is not seven_all_met=True",
            "requirements": [],
        }
    rows = audit_doc.get("seven_requirements") or []
    if not isinstance(rows, list) or len(rows) != len(nx_audit.SEVEN_REQUIREMENTS):
        return {
            "refused": True,
            "seven_all_met": audit_doc.get("seven_all_met"),
            "why": "audit seven_requirements is missing or the wrong length; refusing to invent states",
            "requirements": [],
        }
    out = []
    for req, row in zip(nx_audit.SEVEN_REQUIREMENTS, rows):
        if not isinstance(row, Mapping) or row.get("requirement") != req:
            return {
                "refused": True,
                "seven_all_met": audit_doc.get("seven_all_met"),
                "why": f"audit row order/name mismatch at {req!r}; refusing to restate",
                "requirements": [],
            }
        out.append(row)
    return {
        "refused": False,
        "seven_all_met": audit_doc.get("seven_all_met"),
        "nx_status": _dot(audit_doc, "nx_completeness_checker.real_FLASH_COMPLETE_V0_nx.status"),
        "requirements": out,
        "via": via,
    }


def _wake(
    *,
    id: str,
    holds: bool,
    evidence: str,
    never_pending: bool = True,
) -> dict[str, Any]:
    return {
        "id": id,
        "holds": holds,
        "evidence": evidence,
        "schedule_if_this_were_the_only_gate": SLEEPING if not holds else "MET_PRECONDITION",
        "never_pending": never_pending,
    }


def wake_conditions_for(
    req: str,
    host: Mapping[str, Any],
    teacher: Mapping[str, Any],
    lease: Mapping[str, Any],
    packed_nx: bool,
) -> list[dict[str, Any]]:
    """Physical requirements list every gate. CPU requirements list none.

    A Metal GPU being present is a precondition that HOLDS today. It does
    not wake the unit by itself.
    """
    if req not in PHYSICAL_TO_SATISFY:
        return []
    compile_ok = host.get("source_compile") == "OK"
    gpu_present = host.get("gpu_present") is True
    return [
        _wake(
            id="metal_gpu_present",
            holds=gpu_present,
            evidence=(
                f"METAL_REACHABILITY verdict={host.get('verdict')!r} "
                f"device={host.get('device_name')!r}"
            ),
        ),
        _wake(
            id="metal_source_compile",
            holds=compile_ok,
            evidence=f"runtime_source_compile={host.get('source_compile')!r}",
        ),
        _wake(
            id="source_independent_nx_on_disk",
            holds=packed_nx,
            evidence=(
                "FLASH_COMPLETE_V0.nx status is not a metadata seal"
                if packed_nx
                else "FLASH_COMPLETE_V0.nx is SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"
            ),
        ),
        _wake(
            id="gpu_lease_proven_holder",
            holds=False,
            evidence=str(lease.get("why_lease_not_proven") or "lease not proven"),
        ),
        _wake(
            id="machine_quiesced",
            holds=False,
            evidence=str(lease.get("why_not_quiesced") or "machine not quiesced"),
        ),
        _wake(
            id="teacher_capture_is_not_this_requirement",
            holds=True,
            evidence=(
                "a 1024-row source-BF16 mlp_input capture "
                f"{'exists' if teacher.get('capture_present') else 'does not exist'}; "
                "it does not satisfy accepted_generation, capability, or protected_performance"
            ),
        ),
    ]


def _schedule(req: str, packed_nx: bool) -> str:
    if req in PHYSICAL_TO_SATISFY:
        return SLEEPING
    if req in NEEDS_PACKED_NX and not packed_nx:
        # Field-set / disclosure / packing specs can run as CPU work. Closing
        # accepted_generation/capability/protected_performance cannot: those
        # already returned SLEEPING above.
        if req in {
            "complete_byte_ledger",
            "reproducibility",
            "no_forbidden_fallback",
            "self_contained_dependencies",
        }:
            return CPU_NEXT
        return BLOCKED_ON_PRIOR
    return CPU_NEXT


def _needs(req: str, host: Mapping[str, Any]) -> dict[str, Any]:
    compile_needed_for_artifact = req in {
        "self_contained_dependencies",
        "reproducibility",
        "protected_performance",
        "accepted_generation",
        "capability",
    }
    return {
        "gpu_authority": req in PHYSICAL_TO_SATISFY,
        "metal_compile": compile_needed_for_artifact,
        "metal_compile_is_a_host_wall": (
            compile_needed_for_artifact and host.get("source_compile") != "OK"
        ),
        "teacher_corpus": False,
        "cpu_work": True,
        "cpu_only_to_satisfy": req not in PHYSICAL_TO_SATISFY,
    }


def _blocker_classes(req: str, host: Mapping[str, Any], packed_nx: bool) -> list[str]:
    classes: list[str] = []
    if req == "self_contained_dependencies":
        classes.append(WORK_NOT_DONE)
    elif req in PHYSICAL_TO_SATISFY:
        if not packed_nx:
            classes.append(PRIOR_ARTIFACT)
        classes.append(PHYSICAL_AUTHORITY)
    else:
        if not packed_nx:
            classes.append(PRIOR_ARTIFACT)
        classes.append(WORK_NOT_DONE)
    v = host.get("verdict")
    if v == "CONFIRMED" and req in PHYSICAL_TO_SATISFY:
        classes.append(HOST_GPU_ABSENT_CONFIRMED)
    elif v not in {"FALSIFIED_AS_A_HOST_PROPERTY", "CONFIRMED"} and req in PHYSICAL_TO_SATISFY:
        classes.append(HOST_GPU_UNTESTED)
    seen: set[str] = set()
    out: list[str] = []
    for c in classes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def emit_sleeping_unit(req: str, wakes: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    holding = [w["id"] for w in wakes if not w["holds"]]
    unit = wus.emit_hcli_workunit(
        id=f"future.nr-nx.sleep.{req}",
        role="science",
        description=f"SLEEPING until wake conditions hold for {req}: {reason}",
        dependencies=[],
        resource_class="GPU_EXCLUSIVE",
        verifier=f"future.nr_nx_path.wakeup:{req}",
        provider="future.nr_nx_path",
        effect_class="READ_ONLY",
        preferred_backend="metal",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "sleeping": True,
            "blocked_reason": reason,
            "requires_quiescence": True,
            "synthetic_result_forbidden": True,
            "wake_unmet": holding,
            "requirement": req,
        },
    )
    wus.validate_emitted_unit(unit)
    if unit.get("status") in FORBIDDEN_PHYSICAL_SCHEDULES:
        raise ValueError(f"{req}: sleeping unit leaked status={unit.get('status')!r}")
    return {
        "id": unit["id"],
        "status": unit["status"],
        "classification": unit.get("classification"),
        "resource_class": unit.get("resource_class"),
        "verifier": unit.get("verifier"),
        "blocked_reason": unit.get("blocked_reason"),
        "wake_unmet": holding,
        "synthetic_result_forbidden": True,
        "requires_quiescence": True,
    }


def classify_requirement(
    req: str,
    *,
    audit_row: Mapping[str, Any],
    live: Mapping[str, Any],
    host: Mapping[str, Any],
    teacher: Mapping[str, Any],
    lease: Mapping[str, Any],
    packed_nx: bool,
    nx_path: str | None,
    docs: Mapping[str, Any],
) -> dict[str, Any]:
    audit_state = audit_row.get("state")
    live_state = live.get("state")
    agreement = "AGREE" if audit_state == live_state else "DISAGREE"
    wakes = wake_conditions_for(req, host, teacher, lease, packed_nx)
    schedule = _schedule(req, packed_nx)
    if req in PHYSICAL_TO_SATISFY:
        schedule = SLEEPING
        if schedule in FORBIDDEN_PHYSICAL_SCHEDULES:
            raise ValueError(f"{req}: physical requirement scheduled as {schedule}")
    sleeping = emit_sleeping_unit(req, wakes, str(audit_row.get("note") or live.get("why") or req)) if schedule == SLEEPING else None
    needs = _needs(req, host)
    blockers = _blocker_classes(req, host, packed_nx)
    cpu_done = None
    cannot = None
    if req == "complete_byte_ledger":
        contract = complete_system_ledger_contract(docs.get("ledger"), docs.get("executable"))
        cpu_done = contract.get("cpu_work_done_this_lane")
        cannot = contract.get("why_cannot_close_here")
    elif req == "reproducibility":
        inv = reproducibility_closure_inventory(docs.get("executable"), docs.get("nx_v0"))
        cpu_done = inv.get("cpu_work_done_this_lane")
        cannot = inv.get("why_cannot_hash_here")
    elif req == "no_forbidden_fallback":
        nx = docs.get("nx_v0") if isinstance(docs.get("nx_v0"), Mapping) else {}
        disc = fallback_disclosure(nx)
        cpu_done = "refused to stamp fallback_count=0 on the metadata NX"
        cannot = disc.get("why")
    elif req == "self_contained_dependencies":
        cpu_done = (
            "reclassified Metal source compile as not a host wall; packing the NX "
            "body remains Codex-owned work-not-done"
        )
        cannot = (
            "this lane's write list forbids building an NX; tools/flash_nx_genome.py "
            "and hcli/agentos/flash_executable.py are Codex-owned; the V0/V1/V2 seals "
            "are metadata; the bounded Q4 128-row expert slice is not a whole-model NX"
        )
    elif req in PHYSICAL_TO_SATISFY:
        cpu_done = (
            "wake conditions split into already-holding (Metal GPU, source compile) "
            "and unmet (packed NX, lease, QUIESCED); teacher capture refused as a substitute"
        )
        cannot = (
            "satisfying this requirement is a protected GPU run on a source-independent "
            "NX that does not exist; this sidecar has no GPU authority and will not flock"
        )
    judged = judge_requirement(req, nx=docs.get("nx_v0"), nx_path=nx_path, context={"byte_ledger": docs.get("ledger")})
    if judged.get("ok") is True and not (nx_path and Path(nx_path).is_file()):
        raise ArtifactRequired(f"{req}: checker returned ok without an on-disk artifact")
    return {
        "requirement": req,
        "audit_state": audit_state,
        "live_checker_state": live_state,
        "state_agreement": agreement,
        "ok": False if judged.get("ok") is not True else True,
        "judge": {
            "state": judged.get("state"),
            "why": judged.get("why"),
            "artifact": judged.get("artifact"),
        },
        "audit_why": (audit_row.get("checker_on_FLASH_COMPLETE_V0_nx") or {}).get("why"),
        "audit_cited": audit_row.get("cited"),
        "audit_note": audit_row.get("note"),
        "satisfying_artifact": SATISFYING_ARTIFACT[req],
        "needs": needs,
        "blocker_classes": blockers,
        "rejected_inherited_class": "HOST_HAS_NO_GPU",
        "schedule": schedule,
        "wake_conditions": wakes,
        "sleeping_workunit": sleeping,
        "cpu_work_done_this_lane": cpu_done,
        "why_cpu_cannot_close_here": cannot,
    }


def audit_disagreements(
    *,
    audit_doc: Mapping[str, Any] | None,
    classified: list[dict[str, Any]],
    host: Mapping[str, Any],
    teacher: Mapping[str, Any],
    nr_complete: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Disagreement is the finding. This module does not silently pick a winner."""
    rows: list[dict[str, Any]] = []
    for row in classified:
        if row["state_agreement"] == "DISAGREE":
            rows.append(
                {
                    "kind": "requirement_state",
                    "requirement": row["requirement"],
                    "audit_state": row["audit_state"],
                    "live_checker_state": row["live_checker_state"],
                    "resolution": "REPORTED_NOT_RESOLVED",
                    "authority": "both cited; this module does not pick a winner and does not flip MET",
                }
            )
    if isinstance(audit_doc, Mapping):
        for finding in audit_doc.get("negative_findings") or []:
            if not isinstance(finding, Mapping):
                continue
            text = str(finding.get("found") or "")
            claim = reclassify_causal_claim(text, host)
            if claim["verdict"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM":
                rows.append(
                    {
                        "kind": "causal_label",
                        "id": "audit_negative_finding_sidecar_has_no_gpu",
                        "looked_for": finding.get("looked_for"),
                        "audit_text": text,
                        "resolution": "REPORTED_NOT_RESOLVED",
                        **{k: v for k, v in claim.items() if k != "text"},
                    }
                )
    nr_evidence = _dot(nr_complete or {}, "seven_quantities.qualified_complete_physical_ebpw.evidence")
    if isinstance(nr_evidence, str):
        claim = reclassify_causal_claim(nr_evidence, host)
        if claim["verdict"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM":
            rows.append(
                {
                    "kind": "causal_label",
                    "id": "flash_nr_complete_host_of_record_has_no_gpu",
                    "audit_text": nr_evidence,
                    "resolution": "REPORTED_NOT_RESOLVED",
                    **{k: v for k, v in claim.items() if k != "text"},
                }
            )
    if teacher.get("capture_present") and teacher.get("contract_executed") == 0:
        rows.append(
            {
                "kind": "declared_vs_executed",
                "id": "teacher_contract_workunits_unexecuted_but_capture_exists",
                "contract_executed": teacher.get("contract_executed"),
                "contract_workunits": teacher.get("contract_workunits"),
                "capture_status": teacher.get("capture_status"),
                "capture_rows": teacher.get("capture_rows"),
                "resolution": "REPORTED_NOT_RESOLVED",
                "why": teacher.get("declared_vs_executed"),
                "does_not_satisfy": "accepted_generation / capability / protected_performance",
            }
        )
    return rows


def criterion_snapshot(
    audit_from: dict[str, Any],
    nx_status: str | None,
    nr_status: str | None,
) -> dict[str, Any]:
    """What odyssey_launch._eval_nr_nx checks, plus the split it currently lacks."""
    seven = audit_from.get("seven_all_met")
    callable_path = seven is True and nx_status not in {
        None,
        nx_audit.METADATA_ONLY,
        "NOT_BUILT",
        "SCAFFOLD_ONLY",
    }
    return {
        "id": "nr_nx_path_callable",
        "met": False,
        "seven_all_met": seven,
        "nx_status": nx_status,
        "nr_status": nr_status,
        "callable_path": bool(callable_path),
        "why_the_criterion_collapses_the_seven": (
            "odyssey_launch._eval_nr_nx treats seven_all_met=False identically "
            "regardless of whether a miss is WORK_NOT_DONE or PHYSICAL_AUTHORITY. "
            "That collapse is why this module exists."
        ),
        "reason": (
            "NR/NX path is source-independent, complete, and resident-callable"
            if callable_path
            else (
                f"NR/NX path is not callable. seven_all_met={seven!r} "
                f"FLASH_COMPLETE_V0.nx status={nx_status!r} nr_v2={nr_status!r}. "
                "Flash source-independent NX is SCAFFOLD_ONLY / metadata, not qualified."
            )
        ),
    }


def next_workunits(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for row in classified:
        req = row["requirement"]
        if row["schedule"] == SLEEPING:
            units.append(
                {
                    "id": f"WU.SLEEPING.nr-nx.{req}",
                    "requirement": req,
                    "schedule": SLEEPING,
                    "wake_unmet": [
                        w["id"] for w in row["wake_conditions"] if not w["holds"]
                    ],
                    "workunit": (row.get("sleeping_workunit") or {}).get("id"),
                }
            )
        elif row["schedule"] == CPU_NEXT:
            units.append(
                {
                    "id": f"WU.CPU.nr-nx.{req}",
                    "requirement": req,
                    "schedule": CPU_NEXT,
                    "owner": (
                        "this sidecar (field-set / disclosure / inventory)"
                        if req in {"complete_byte_ledger", "reproducibility", "no_forbidden_fallback"}
                        else "Codex (pack NX / loader / kernel catalog)"
                    ),
                    "done_this_lane": row.get("cpu_work_done_this_lane"),
                    "still_open": row.get("why_cpu_cannot_close_here"),
                }
            )
        else:
            units.append(
                {
                    "id": f"WU.BLOCKED_ON_PRIOR.nr-nx.{req}",
                    "requirement": req,
                    "schedule": BLOCKED_ON_PRIOR,
                    "prior": "serialized source-independent NX body",
                }
            )
    return units


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def assemble() -> dict[str, Any]:
    audit_doc, audit_via = load_rel(REL_AUDIT)
    host = today_host_facts()
    teacher = today_teacher_facts()
    lease = today_lease_facts()
    nr_complete, nr_complete_via = load_rel(REL_NR_COMPLETE)
    launch, launch_via = load_rel(REL_LAUNCH)

    audit_seven = seven_from_audit(audit_doc, audit_via)
    live_docs: dict[str, Any] | None = None
    live_error = None
    try:
        live_docs = nx_audit._load_all()
    except FileNotFoundError as exc:
        live_error = f"FileNotFoundError: {exc}"

    live_check = None
    packed_nx = False
    nx_path = None
    nx_status = None
    nr_status = None
    if live_docs is not None:
        nx = live_docs.get("nx_v0") if isinstance(live_docs.get("nx_v0"), Mapping) else {}
        nx_status = nx.get("status")
        nr_status = (live_docs.get("nr_v2") or {}).get("status") if isinstance(live_docs.get("nr_v2"), Mapping) else None
        loc = nx_audit.evidence_location(nx_audit.REL_NX_V0)
        nx_path = loc.get("resolved")
        packed_nx = not nx_audit._status_is_metadata_only(nx)
        live_check = nx_audit.check_nx(
            dict(nx),
            context={"byte_ledger": live_docs.get("ledger"), "meta": live_docs.get("meta"), "nr": live_docs.get("nr_v2")},
        )

    classified: list[dict[str, Any]] = []
    if audit_seven["refused"] or live_docs is None or live_check is None:
        why = audit_seven.get("why") or live_error or "live NX docs unavailable"
        for req in nx_audit.SEVEN_REQUIREMENTS:
            wakes = wake_conditions_for(req, host, teacher, lease, packed_nx)
            schedule = SLEEPING if req in PHYSICAL_TO_SATISFY else CPU_NEXT
            classified.append(
                {
                    "requirement": req,
                    "audit_state": None,
                    "live_checker_state": None,
                    "state_agreement": "UNCOMPARABLE",
                    "ok": False,
                    "schedule": schedule,
                    "blocker_classes": _blocker_classes(req, host, packed_nx),
                    "why": why,
                    "satisfying_artifact": SATISFYING_ARTIFACT[req],
                    "wake_conditions": wakes,
                    "sleeping_workunit": (
                        emit_sleeping_unit(req, wakes, str(why)) if schedule == SLEEPING else None
                    ),
                    "needs": _needs(req, host),
                    "rejected_inherited_class": "HOST_HAS_NO_GPU",
                }
            )
    else:
        audit_rows = {r["requirement"]: r for r in audit_seven["requirements"]}
        for req in nx_audit.SEVEN_REQUIREMENTS:
            live_row = live_check["requirements"][req]
            classified.append(
                classify_requirement(
                    req,
                    audit_row=audit_rows[req],
                    live=live_row,
                    host=host,
                    teacher=teacher,
                    lease=lease,
                    packed_nx=packed_nx,
                    nx_path=nx_path,
                    docs=live_docs,
                )
            )

    disagreements = audit_disagreements(
        audit_doc=audit_doc,
        classified=classified,
        host=host,
        teacher=teacher,
        nr_complete=nr_complete,
    )

    ledger_contract = complete_system_ledger_contract(
        None if live_docs is None else live_docs.get("ledger"),
        None if live_docs is None else live_docs.get("executable"),
    )
    repro_inv = reproducibility_closure_inventory(
        None if live_docs is None else live_docs.get("executable"),
        None if live_docs is None else live_docs.get("nx_v0"),
    )
    kernel_ov = kernel_catalog_overlay(live_docs or {}, host)
    teacher_as_ag = teacher_satisfies_accepted_generation(teacher)

    # Dense remat: reuse the landed type system rather than restating policy.
    remat = None
    if live_docs is not None and isinstance(live_docs.get("executable"), Mapping):
        remat = ebpw.judge_dense_rematerialization(live_docs["executable"]).as_dict()

    launch_unmet = _dot(launch or {}, "verdict.unmet") if isinstance(launch, Mapping) else None
    snapshot = criterion_snapshot(audit_seven, nx_status, nr_status)

    physical_rows = [r for r in classified if r.get("schedule") == SLEEPING]
    for row in physical_rows:
        if row.get("schedule") in FORBIDDEN_PHYSICAL_SCHEDULES:
            raise ValueError(f"{row['requirement']}: physical schedule leaked {row.get('schedule')!r}")
        wu = row.get("sleeping_workunit") or {}
        if wu and wu.get("status") in FORBIDDEN_PHYSICAL_SCHEDULES:
            raise ValueError(f"{row['requirement']}: sleeping unit status={wu.get('status')!r}")

    any_met_without_artifact = [
        r["requirement"] for r in classified if r.get("ok") is True and not (r.get("judge") or {}).get("artifact")
    ]
    if any_met_without_artifact:
        raise ArtifactRequired(f"MET without artifact: {any_met_without_artifact}")

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Map each of the seven Flash NX completeness requirements onto "
            "WORK_NOT_DONE vs PHYSICAL_AUTHORITY using today's Metal and teacher "
            "evidence, advance the CPU contracts that do not require packing an NX, "
            "and emit SLEEPING WorkUnits with wake conditions for the rest."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "criterion": snapshot,
        "launch_gate_unmet": launch_unmet,
        "launch_via": launch_via,
        "today": {
            "host": host,
            "teacher": teacher,
            "lease": lease,
            "teacher_is_not_accepted_generation": teacher_as_ag,
        },
        "audit": {
            "via": audit_via,
            "refused": audit_seven["refused"],
            "seven_all_met": audit_seven.get("seven_all_met"),
            "nx_status": audit_seven.get("nx_status") or nx_status,
            "why": audit_seven.get("why"),
        },
        "live_nx": {
            "error": live_error,
            "path": nx_path,
            "status": nx_status,
            "packed": packed_nx,
            "nr_status": nr_status,
            "promotable": None if live_check is None else live_check.get("promotable"),
            "failed_requirements": None if live_check is None else live_check.get("failed_requirements"),
        },
        "seven": classified,
        "seven_all_met": False,
        "audit_disagreements": disagreements,
        "cpu_contracts": {
            "complete_system_ledger": ledger_contract,
            "reproducibility_closure": repro_inv,
            "kernel_catalog": kernel_ov,
            "dense_rematerialization": remat,
        },
        "next_workunits": next_workunits(classified),
        "physical_ebpw": None,
        "physical_ebpw_written": False,
        "nr_complete_via": nr_complete_via,
        "recovered_implementation": [
            "tools/future/flash_nx_audit.py — owns SEVEN_REQUIREMENTS, check_nx, seven_from_disk, dependency_chain; EXTENDED, not forked",
            "tools/future/odyssey_launch.py _eval_nr_nx — the criterion that collapses seven_all_met; read, not rewritten",
            "tools/future/flash_nr_complete.py — continuous NR + typed EBPW; qualified_complete_physical_ebpw stays UNKNOWN",
            "tools/future/ebpw_categories.py — can_promote / judge_dense_rematerialization reused",
            "tools/future/metal_reachability.py — FALSIFIED_AS_A_HOST_PROPERTY receipt; not re-probed",
            "tools/future/workunit_species.py emit_hcli_workunit — SLEEPING units, never pending",
            "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json — source of requirement state",
            "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL1024.json — teacher capture ran; screen failed",
        ],
        "gaps_closed": [
            "the seven misses are no longer reported identically: WORK_NOT_DONE vs PHYSICAL_AUTHORITY vs PRIOR_ARTIFACT",
            "HOST_HAS_NO_GPU is rejected as an inherited causal class; Metal GPU present is a holding wake condition, not a qualified NX",
            "complete-system ledger field-set contract scored against BYTE_LEDGER without closing it",
            "reproducibility closure inventory scored against null digests",
            "fallback_count=0 refused on the metadata NX",
            "SLEEPING WorkUnits for accepted_generation, capability, protected_performance with wake conditions split into holding vs unmet",
            "1024-row source-BF16 teacher capture is on disk and is refused as accepted_generation",
        ],
        "negative_findings": [
            "seven_all_met remains False; FLASH_COMPLETE_V0.nx remains SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            "no packed source-independent Flash NX body exists to close self_contained_dependencies",
            "no GPU lease with a proven holder pid; contamination_class is HEAVY; protected_performance stays SLEEPING",
            "the L4 REAL1024 coherence screen is OFFLINE_META_SURFACE_GATE_FAILED; more rows made the family worse, which does not qualify an NX",
            "TEACHER_CORPUS_CONTRACT still records executed=0 while a capture receipt exists — declared capability is not executed capability",
            "audit negative_findings still say 'sidecar has no GPU'; that causal label disagrees with METAL_REACHABILITY and is reported, not silently rewritten",
            "this module cannot pack Flash, cannot take a lease, and cannot establish physical EBPW",
        ],
        "what_this_cannot_establish": [
            "a qualified source-independent Flash NX",
            "protected complete-token ns / accepted TPS / physical EBPW",
            "why a previous Rust process saw Device::system_default() return None",
            "that the blocked Codex capture binaries will now succeed",
            "capability preservation of any compact organ",
        ],
        "resident_callable": {
            "entry_point": "tools.future.nr_nx_path.build()",
            "workunit": "one CPU_ANALYSIS unit; classification + CPU contracts; no GPU authority",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": (
                "absent audit/NX is a recorded refusal; MET requires an on-disk artifact; "
                "physical requirements are SLEEPING never pending; physical_ebpw cannot be written"
            ),
        },
    }
    assert_no_physical_ebpw(doc)
    return doc


def build() -> Path:
    doc = assemble()
    assert_no_physical_ebpw(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
