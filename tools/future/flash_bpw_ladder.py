"""COMPLETE-EXECUTABLE-EBPW LADDER — gate every rung on qualified physical EBPW.

Search pressure from the ~2.25 coherent-class floor down through 2.0, 1.75,
1.5, 1.25, ~1.0, and sub-1. Standing on a rung is a claim about total
executable information of a complete executable, not a uniform
bits-per-value and not a description budget. This module exists to stop
the L4 REAL256 failure from being reported as a sub-1 win: a diagnostic
factor bpw of 0.0254 sat next to a held-out error of 0.5284 — a smaller
number attached to a representation that does not work.

Refuses: hardware measurement, GPU lease, marking REACHED on any quantity
other than qualified_complete_physical_ebpw, uniform-bpw budgets, and a
plan that is smaller but incoherent, dense-rematerializing, or
compute-multiplied past its byte saving.

Cannot establish: a physical EBPW number (no GPU authority), that any
rung has been reached on this host, or that a budget proposal preserves
capability.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, write_receipt
from tools.future.ebpw_categories import (
    CategoryError,
    PRODUCTION,
    PROTECTED,
    Quantity,
    VERIFICATION,
    judge_dense_rematerialization,
)
from tools.future.flash_nr_complete import (
    ProspectiveMetaBpw,
    QualifiedCompletePhysicalEbpw,
    RESEARCH_TARGET,
    SEVEN_TYPES,
    SerializedNrInformation,
    SerializedNxEbpw,
    SevenLedger,
    SourceControlEbpw,
    StaticActiveEbpwEstimate,
    StaticCompleteEbpwEstimate,
    UNKNOWN,
    can_promote,
    ebpw_organs,
    judge_production_dense_checkpoint,
    load_docs,
    resolve_evidence,
    seven_from_docs,
)

# Imported by name so a later fork cannot quietly redefine them here.
IMPORTED_SEVEN: tuple[type[Quantity], ...] = (
    SourceControlEbpw,
    StaticActiveEbpwEstimate,
    StaticCompleteEbpwEstimate,
    ProspectiveMetaBpw,
    SerializedNrInformation,
    SerializedNxEbpw,
    QualifiedCompletePhysicalEbpw,
)
from tools.future.flash_nx_audit import SEVEN_REQUIREMENTS

RECEIPT = "FLASH_BPW_LADDER.json"
SCHEMA = "hawking.future.flash_bpw_ladder.v1"
RECORDED_BY = "tools/future/flash_bpw_ladder.py"

REACHED = "REACHED"
REFUSED = "REFUSED"
UNTESTED = "UNTESTED"

DOMINATED = "DOMINATED"
ON_FRONT = "ON_FRONT"

RULE_INCOHERENT = "smaller_but_incoherent"
RULE_REMAT = "smaller_but_dense_rematerializing"
RULE_COMPUTE = "smaller_but_compute_multiplied"

REQUIRED_QUANTITY = QualifiedCompletePhysicalEbpw.category
REQUIRED_EVIDENCE_CLASS = "PROTECTED_ABSOLUTE"

# Held-out contract copied from the L4 REAL256 screen. A plan that misses
# these is incoherent even when its diagnostic factor bpw looks small.
SCREEN_REL = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL256.json"
SCREEN_NAME = "FLASH_META_COHERENCE_SCREEN_L4_REAL256.json"
MAX_HELDOUT_RELATIVE_FRO = 0.05
MIN_HELDOUT_COSINE = 0.999

COHERENT_FLOOR_BPW = 2.25
SUB1_RESEARCH_TARGET_BPW = 0.887

# Relative structural units for domination. Not bytes of a packed artifact.
FLOOR_CONTROL_STORAGE = 1000
FLOOR_CONTROL_FLOP_MILLI = 1000

# Control-plane floors (bits/value) from FLASH_META_REPRESENTATION_SUB1
# family_budget plus the router premium recovered from flash_schools.
# Bulk organs have no floor; they absorb compression so the weighted total
# can fall without crushing route/state/terminal islands.
PREMIUM_FLOOR_BPW: dict[str, float] = {
    "norm": 16.0,
    "NORMALIZATION": 16.0,
    "router": 16.0,
    "ROUTER": 16.0,
    "embedding_lm_head": 3.5,
    "embeddings": 3.5,
    "lm_head": 3.5,
    "LM_HEAD": 3.5,
    "EMBEDDING": 3.5,
    "full_attention": 3.0,
    "sparse_attention": 3.0,
    "FULL_ATTENTION": 3.0,
    "KV_STATE": 3.0,
    "linear_attention_hyperconnection": 2.5,
    "deltanet": 2.5,
    "recurrent_state": 2.5,
    "DELTANET_RECURRENT_STATE": 2.5,
    "mlp_hyperconnection": 2.5,
    "residual_hyperconnections": 2.5,
    "HC_HYPERCONNECTION": 2.5,
    "shared_expert": 2.5,
    "SHARED_EXPERTS": 2.5,
    "other": 2.5,
    "support_misc": 2.5,
    "POSITIONAL_STRUCTURE": 2.5,
}

BULK_ORGANS: frozenset[str] = frozenset(
    {
        "routed_experts",
        "ROUTED_EXPERTS",
        "ngram_embedding",
        "ngram_engine",
        "NGRAM",
        "mtp",
        "MTP_SPECULATION",
        "vision_backbone",
    }
)

NEAR_ZERO_ORGANS: frozenset[str] = frozenset({"vision_backbone"})

WRONG_CLAIM_QUANTITIES: frozenset[str] = (
    frozenset(SEVEN_TYPES) - {REQUIRED_QUANTITY}
) | {
    "diagnostic_factor_equivalent_bpw",
    "diagnostic_factor_bpw",
    "meta_bpw",
    "complete_bits_per_weight",
    "complete_exact_control.complete_ebpw",
}


class UnknownRungError(ValueError):
    """A caller named a rung that is not on the ladder."""


class LadderRefuse(ValueError):
    """The ladder refused an input rather than guessing."""


# ---------------------------------------------------------------------------
# Ladder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One search-pressure step. Reaching it requires qualified physical EBPW."""

    id: str
    target_bpw: float
    exclusive: bool
    label: str
    what_reaching_proves: str
    what_reaching_does_not_prove: str

    def as_dict(self) -> dict[str, Any]:
        bound = f"< {self.target_bpw}" if self.exclusive else f"<= {self.target_bpw}"
        return {
            "id": self.id,
            "label": self.label,
            "target_bpw": self.target_bpw,
            "exclusive": self.exclusive,
            "bound": bound,
            "required_quantity": REQUIRED_QUANTITY,
            "required_evidence_class": REQUIRED_EVIDENCE_CLASS,
            "required_artifacts": [dict(a) for a in REQUIRED_ARTIFACTS],
            "nx_completeness_requirements": list(SEVEN_REQUIREMENTS),
            "what_reaching_proves": self.what_reaching_proves,
            "what_reaching_does_not_prove": self.what_reaching_does_not_prove,
            "search_pressure": (
                "This is a gated search target, not a requirement to produce a win. "
                "UNTESTED is the honest state until qualified_complete_physical_ebpw exists."
            ),
            "prospective_meta_bpw_role": RESEARCH_TARGET,
        }


# Exact artifacts a rung needs before it will accept REACHED. Paths are
# candidates; sparse checkouts are not absences. Predicates are the rule.
REQUIRED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "id": "source_independent_nx",
        "paths": (
            "receipts/headless/FLASH_COMPLETE_V2.nx.json",
            "receipts/headless/FLASH_NEXT_MACHINE.nx.json",
            "receipts/future/evidence/FLASH_COMPLETE_V2.nx.json",
            "receipts/future/evidence/FLASH_NEXT_MACHINE.nx.json",
            "receipts/headless/FLASH_COMPLETE_V0.nx.json",
            "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json",
        ),
        "must": (
            "a source-independent complete NX whose status is not "
            "SEALED_METADATA_ONLY / NOT_FOR_PROMOTION"
        ),
    },
    {
        "id": "closed_complete_system_byte_ledger",
        "paths": (
            "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
            "receipts/future/evidence/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
        ),
        "must": (
            "self_contained and for_this_executable with positive "
            "complete_storage_bytes; exact-control 16.0 identity is not this ledger"
        ),
    },
    {
        "id": "capability_preserving_runtime",
        "paths": (
            "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json",
            "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json",
        ),
        "must": "capability preserved on an accepted complete-token runtime",
    },
    {
        "id": "protected_complete_token",
        "paths": (
            "receipts/headless/FLASH_COMPLETE_TOKEN_NATIVE_ATTEMPT.json",
            "receipts/headless/FLASH_COMPLETE_TOKEN_DEVICE_RESIDENT_V1.json",
            "receipts/headless/FLASH_TOKEN_NS_BUDGET.json",
        ),
        "must": (
            "PROTECTED_ABSOLUTE complete-token measurement; DIAGNOSTIC_RELATIVE "
            "and STATIC_ONLY cannot back a rung"
        ),
    },
    {
        "id": "qualified_complete_physical_ebpw",
        "paths": (),
        "must": (
            "typed qualified_complete_physical_ebpw backed by PROTECTED_ABSOLUTE, "
            "at or below the rung bound; never a copy of prospective_meta_bpw"
        ),
    },
    {
        "id": "direct_production_consume",
        "paths": (),
        "must": (
            "production consumes the representation directly; decompress-then-"
            "ordinary-kernels / dense-parent rematerialization is refused"
        ),
    },
    {
        "id": "coherence_held",
        "paths": (SCREEN_REL,),
        "must": (
            "held-out function within contract (relative Frobenius error "
            f"<= {MAX_HELDOUT_RELATIVE_FRO}, cosine >= {MIN_HELDOUT_COSINE}); "
            "a diagnostic factor bpw is not coherence"
        ),
    },
)


_PROVES = (
    "qualified_complete_physical_ebpw of a complete executable is at or below "
    "this bound, under PROTECTED_ABSOLUTE, without dense rematerialization, "
    "with held-out function inside contract"
)
_DOES_NOT = (
    "that a cheaper description budget (prospective_meta_bpw), a diagnostic "
    "factor bpw, serialized NR/NX bytes, or a STATIC estimate is a complete "
    "executable; that capability is preserved beyond the stated suite; that "
    "this sidecar measured hardware"
)


RUNGS: tuple[Rung, ...] = (
    Rung(
        "bpw_2_25",
        2.25,
        False,
        "2.25 coherent-class floor",
        _PROVES,
        _DOES_NOT
        + "; the 2.25 figure is the measured coherent-class floor used as "
        "search pressure, not a Flash qualified-physical result on this host",
    ),
    Rung("bpw_2_00", 2.0, False, "2.0", _PROVES, _DOES_NOT),
    Rung("bpw_1_75", 1.75, False, "1.75", _PROVES, _DOES_NOT),
    Rung("bpw_1_50", 1.5, False, "1.5", _PROVES, _DOES_NOT),
    Rung("bpw_1_25", 1.25, False, "1.25", _PROVES, _DOES_NOT),
    Rung("bpw_1_00", 1.0, False, "~1.0", _PROVES, _DOES_NOT),
    Rung(
        "bpw_sub1",
        1.0,
        True,
        "sub-1",
        _PROVES + " and strictly below 1.0 qualified_complete_physical_ebpw",
        _DOES_NOT
        + "; prospective_meta_bpw 0.887 is RESEARCH_TARGET and proves nothing; "
        "the L4 REAL256 diagnostic factor bpw of ~0.0254 proves nothing",
    ),
)
RUNGS_BY_ID = {r.id: r for r in RUNGS}


def rungs() -> tuple[Rung, ...]:
    """The ladder. Each rung names the quantity and artifacts it requires."""
    return RUNGS


def resolve_rung(rung: Rung | str | float) -> Rung:
    if isinstance(rung, Rung):
        return rung
    if isinstance(rung, (int, float)) and not isinstance(rung, bool):
        key = float(rung)
        if key < 1.0:
            return RUNGS_BY_ID["bpw_sub1"]
        for row in RUNGS:
            if not row.exclusive and row.target_bpw == key:
                return row
        raise UnknownRungError(f"no rung at target_bpw={key}: {sorted(r.id for r in RUNGS)}")
    name = str(rung).strip()
    if name in RUNGS_BY_ID:
        return RUNGS_BY_ID[name]
    lowered = name.lower().replace(" ", "").replace("bpw", "")
    aliases = {
        "2.25": "bpw_2_25",
        "2.0": "bpw_2_00",
        "2.00": "bpw_2_00",
        "1.75": "bpw_1_75",
        "1.5": "bpw_1_50",
        "1.50": "bpw_1_50",
        "1.25": "bpw_1_25",
        "1.0": "bpw_1_00",
        "1.00": "bpw_1_00",
        "~1.0": "bpw_1_00",
        "~1": "bpw_1_00",
        "sub-1": "bpw_sub1",
        "sub1": "bpw_sub1",
        "<1": "bpw_sub1",
        "floor": "bpw_2_25",
    }
    ident = aliases.get(name) or aliases.get(lowered)
    if ident:
        return RUNGS_BY_ID[ident]
    raise UnknownRungError(
        f"unknown rung {rung!r}; known ids: {sorted(RUNGS_BY_ID)}"
    )


def _meets_target(value: float, rung: Rung) -> bool:
    if rung.exclusive:
        return value < rung.target_bpw
    return value <= rung.target_bpw


# ---------------------------------------------------------------------------
# Evidence. Missing in this sparse tree is a path taken, not a project-absent.
# ---------------------------------------------------------------------------


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Quantity):
        return value.value
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _nullish(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str) and value.strip().upper() in {
        "NULL",
        "NULL_BY_RULE",
        "NOT_BUILT",
        "NOT_MEASURED",
        "NOT_TESTED",
        "UNKNOWN",
        "ABSENT",
        "NONE",
        "",
    }:
        return True
    return False


def _present(qty: Quantity | None) -> bool:
    return qty is not None and qty.value is not None


def resolve_named(filename: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Pinned / headless / sibling / git HEAD. Missing is recorded, not invented."""
    doc, via, path = resolve_evidence(filename)
    if doc is not None:
        return doc, via, path
    for rel in (
        f"receipts/future/evidence/{filename}",
        f"receipts/headless/{filename}",
    ):
        blob = git("show", f"HEAD:{rel}")
        if not blob:
            continue
        try:
            return json.loads(blob), "git:HEAD", rel
        except json.JSONDecodeError:
            continue
    return None, "missing", None


def load_screen() -> dict[str, Any]:
    """The L4 REAL256 screen is the worked example this ladder exists to keep honest."""
    doc, via, path = resolve_named(SCREEN_NAME)
    if doc is None:
        on_disk = (REPO / SCREEN_REL).is_file()
        return {
            "reachable": False,
            "resolved_via": via,
            "path": path,
            "on_disk_in_this_worktree": on_disk,
            "reason": (
                f"{SCREEN_NAME} not readable from pinned, headless, sibling, "
                "or git HEAD; trap rule still encoded, numbers not invented"
            ),
        }
    rows = _dot(doc, "surface.rows") or []
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    contract = doc.get("coherence_contract") if isinstance(doc.get("coherence_contract"), Mapping) else {}
    return {
        "reachable": True,
        "resolved_via": via,
        "path": path,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "physical_ebpw": _dot(doc, "representation.physical_ebpw"),
        "meta_bpw_target": _dot(doc, "representation.meta_bpw_target"),
        "dense_rematerialization": _dot(doc, "representation.dense_rematerialization"),
        "runtime_artifact_emitted": _dot(doc, "representation.runtime_artifact_emitted"),
        "measurement_state": doc.get("measurement_state"),
        "claim_boundary": doc.get("claim_boundary"),
        "max_heldout_relative_fro_error": _as_number(
            contract.get("max_heldout_relative_fro_error")
        )
        or MAX_HELDOUT_RELATIVE_FRO,
        "min_heldout_cosine": _as_number(contract.get("min_heldout_cosine"))
        or MIN_HELDOUT_COSINE,
        "diagnostic_factor_equivalent_bpw": _as_number(
            row.get("diagnostic_factor_equivalent_bpw")
        ),
        "heldout_relative_fro_error": _as_number(row.get("heldout_relative_fro_error")),
        "heldout_cosine": _as_number(row.get("heldout_cosine")),
        "beats_per_expert_q4_on_heldout": row.get("beats_per_expert_q4_on_heldout"),
        "surface_gate_pass": row.get("surface_gate_pass"),
        "first_surface_failure": row.get("first_surface_failure"),
        "surface_failure_gates": list(row.get("surface_failure_gates") or []),
        "diagnostic_factor_bytes": row.get("diagnostic_factor_bytes"),
        "selected_dense_source_bytes": row.get("selected_dense_source_bytes"),
        "why_this_is_not_a_win": (
            "A diagnostic factor bpw below 1 sat alongside a held-out error "
            "far above contract. Smaller number, representation that does not work."
        ),
    }


def screen_trap_plan(screen: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The L4 REAL256 row as a domination plan. Numbers come from the receipt."""
    screen = screen if screen is not None else load_screen()
    if not screen.get("reachable"):
        raise LadderRefuse(
            "cannot build the screen trap plan: "
            + str(screen.get("reason") or "screen unreachable")
        )
    factor = screen.get("diagnostic_factor_equivalent_bpw")
    error = screen.get("heldout_relative_fro_error")
    if factor is None or error is None:
        raise LadderRefuse(
            "screen is reachable but diagnostic_factor_equivalent_bpw or "
            "heldout_relative_fro_error is absent; refusing to invent them"
        )
    storage = screen.get("diagnostic_factor_bytes")
    incumbent_storage = screen.get("selected_dense_source_bytes")
    return {
        "id": "l4_real256_factor_screen",
        "quantity": "diagnostic_factor_equivalent_bpw",
        "bpw": factor,
        "storage_bytes": storage,
        "flop_milli": FLOOR_CONTROL_FLOP_MILLI,
        "coherent": False,
        "surface_gate_pass": bool(screen.get("surface_gate_pass")),
        "heldout_relative_fro_error": error,
        "heldout_cosine": screen.get("heldout_cosine"),
        "rematerializes_dense_parent": False,
        "path_kind": PRODUCTION,
        "incumbent": {
            "id": "selected_dense_source",
            "storage_bytes": incumbent_storage,
            "flop_milli": FLOOR_CONTROL_FLOP_MILLI,
            "coherent": True,
            "rematerializes_dense_parent": False,
            "path_kind": PRODUCTION,
        },
        "source": screen.get("path"),
        "not_a_hardware_measurement": True,
    }


# ---------------------------------------------------------------------------
# Artifact inventory. Fail closed: unreadable is ABSENT, not a pass.
# ---------------------------------------------------------------------------


def _status_is_metadata_only(nx: Mapping[str, Any]) -> bool:
    status = str(nx.get("status") or "")
    return "METADATA_ONLY" in status or "NOT_FOR_PROMOTION" in status


def _load_first(paths: Sequence[str]) -> tuple[dict[str, Any] | None, str | None, str]:
    for rel in paths:
        name = Path(rel).name
        doc, via, resolved = resolve_named(name)
        if doc is not None:
            return doc, rel if resolved is None else resolved, via
        # Also try the exact relative path via git, in case the filename
        # collides across headless/evidence.
        blob = git("show", f"HEAD:{rel}")
        if blob:
            try:
                return json.loads(blob), rel, "git:HEAD"
            except json.JSONDecodeError:
                continue
        live = REPO / rel
        if live.is_file():
            try:
                return json.loads(live.read_text()), rel, "local"
            except (OSError, json.JSONDecodeError):
                continue
    return None, None, "missing"


def inspect_artifacts(docs: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per-artifact state against disk. Does not invent a MET where the predicate fails."""
    docs = docs if docs is not None else load_docs()
    seven = seven_from_docs(docs)
    screen = load_screen()
    rows: list[dict[str, Any]] = []
    for art in REQUIRED_ARTIFACTS:
        ident = str(art["id"])
        if ident == "qualified_complete_physical_ebpw":
            qty = seven.qualified_complete_physical_ebpw
            present = _present(qty)
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if present else "ABSENT",
                    "path": qty.source if qty is not None else None,
                    "value": qty.value if qty is not None else None,
                    "reason": (
                        qty.evidence
                        if qty is not None
                        else "qualified_complete_physical_ebpw is UNKNOWN on this host"
                    ),
                    "must": art["must"],
                }
            )
            continue
        if ident == "direct_production_consume":
            nx = docs.get("nx_v0") if isinstance(docs.get("nx_v0"), Mapping) else None
            if nx is None:
                nx = docs.get("nx_next") if isinstance(docs.get("nx_next"), Mapping) else None
            if nx is None:
                rows.append(
                    {
                        "id": ident,
                        "state": "ABSENT",
                        "path": None,
                        "reason": "no NX document to judge production consumption",
                        "must": art["must"],
                    }
                )
                continue
            remat = judge_production_dense_checkpoint(nx)
            landed = judge_dense_rematerialization(nx)
            metadata = _status_is_metadata_only(nx)
            if metadata:
                state = "PRESENT_NOT_MET"
                reason = (
                    f"NX status={nx.get('status')!r} is not a production consumer; "
                    "direct consume is unproven"
                )
            elif remat.get("ok") is True and landed.ok:
                state = "PRESENT_MET"
                reason = remat.get("reason") or landed.reason
            else:
                state = "PRESENT_NOT_MET"
                reason = remat.get("reason") or landed.reason
            rows.append(
                {
                    "id": ident,
                    "state": state,
                    "path": (docs.get("resolution") or {}).get("nx_v0", {}).get("resolved"),
                    "reason": reason,
                    "must": art["must"],
                }
            )
            continue
        if ident == "coherence_held":
            if not screen.get("reachable"):
                rows.append(
                    {
                        "id": ident,
                        "state": "ABSENT",
                        "path": SCREEN_REL,
                        "reason": screen.get("reason"),
                        "must": art["must"],
                    }
                )
                continue
            passed = screen.get("surface_gate_pass") is True
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if passed else "PRESENT_NOT_MET",
                    "path": screen.get("path"),
                    "reason": (
                        "held-out surface gate passed"
                        if passed
                        else (
                            f"screen status={screen.get('status')}; "
                            f"heldout_relative_fro_error={screen.get('heldout_relative_fro_error')}; "
                            f"diagnostic_factor_equivalent_bpw={screen.get('diagnostic_factor_equivalent_bpw')}; "
                            "a smaller diagnostic number is not coherence"
                        )
                    ),
                    "must": art["must"],
                }
            )
            continue
        doc, path, via = _load_first(tuple(art["paths"]))
        if doc is None:
            rows.append(
                {
                    "id": ident,
                    "state": "ABSENT",
                    "path": None,
                    "searched": list(art["paths"]),
                    "reason": f"{ident} not readable in this checkout (sparse is not absence)",
                    "must": art["must"],
                }
            )
            continue
        if ident == "source_independent_nx":
            ok = not _status_is_metadata_only(doc)
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if ok else "PRESENT_NOT_MET",
                    "path": path,
                    "resolved_via": via,
                    "reason": (
                        f"NX status={doc.get('status')!r}"
                        + ("" if ok else "; metadata seal is not a complete executable")
                    ),
                    "must": art["must"],
                }
            )
            continue
        if ident == "closed_complete_system_byte_ledger":
            exact = doc.get("complete_exact_control") if isinstance(doc.get("complete_exact_control"), Mapping) else {}
            self_contained = doc.get("self_contained") is True
            for_this = doc.get("for_this_executable") is True
            storage = _as_number(
                doc.get("complete_storage_bytes") or exact.get("runtime_required_bytes")
            )
            # Exact-control 16.0 identity is source control, not a packed ledger.
            exact_identity = (
                _as_number(exact.get("complete_ebpw")) == 16.0
                or str(doc.get("status") or "").find("EXACT_CONTROL") >= 0
            )
            ok = self_contained and for_this and storage is not None and storage > 0 and not exact_identity
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if ok else "PRESENT_NOT_MET",
                    "path": path,
                    "resolved_via": via,
                    "reason": (
                        "closed self-contained ledger for this executable"
                        if ok
                        else (
                            f"ledger present but not a closed packed-NX ledger "
                            f"(self_contained={doc.get('self_contained')}, "
                            f"for_this_executable={doc.get('for_this_executable')}, "
                            f"status={doc.get('status')!r})"
                        )
                    ),
                    "must": art["must"],
                }
            )
            continue
        if ident == "capability_preserving_runtime":
            cap = doc.get("capability_preserving_runtime")
            if cap is not True:
                ms = doc.get("measurement_state")
                if isinstance(ms, Mapping):
                    label = str(ms.get("capability") or "")
                    cap = label.upper() in {"PRESERVED", "PASSED", "MATCHED"}
            ok = cap is True
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if ok else "PRESENT_NOT_MET",
                    "path": path,
                    "resolved_via": via,
                    "reason": (
                        "capability preserved"
                        if ok
                        else f"capability not preserved (status={doc.get('status')!r})"
                    ),
                    "must": art["must"],
                }
            )
            continue
        if ident == "protected_complete_token":
            bench = doc.get("bench") if isinstance(doc.get("bench"), Mapping) else {}
            state = str(bench.get("state") or doc.get("bench_state") or "").upper()
            authority = str(
                doc.get("physical_measurement_authority")
                or _dot(doc, "measurement_state.authority")
                or ""
            ).upper()
            ok = state == PROTECTED or authority == PROTECTED
            if "DIAGNOSTIC" in state or "DIAGNOSTIC" in authority or state in {"UNKNOWN", ""}:
                ok = False
            rows.append(
                {
                    "id": ident,
                    "state": "PRESENT_MET" if ok else "PRESENT_NOT_MET",
                    "path": path,
                    "resolved_via": via,
                    "reason": (
                        "PROTECTED_ABSOLUTE complete-token present"
                        if ok
                        else (
                            f"complete-token receipt is not PROTECTED_ABSOLUTE "
                            f"(bench.state={state or None!r}, authority={authority or None!r})"
                        )
                    ),
                    "must": art["must"],
                }
            )
            continue
        rows.append(
            {
                "id": ident,
                "state": "ABSENT",
                "path": path,
                "reason": f"no inspector for {ident}",
                "must": art["must"],
            }
        )
    return rows


def missing_artifacts(inventory: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row["id"])
        for row in inventory
        if row.get("state") != "PRESENT_MET"
    ]


# ---------------------------------------------------------------------------
# status(). REACHED only on qualified_complete_physical_ebpw.
# ---------------------------------------------------------------------------


def _claim_category(evidence: Any) -> tuple[str | None, float | None, str]:
    """Which quantity a caller is trying to stand on, and its numeric value."""
    if isinstance(evidence, QualifiedCompletePhysicalEbpw):
        return REQUIRED_QUANTITY, evidence.value, evidence.evidence
    if isinstance(evidence, Quantity):
        return evidence.category, evidence.value, evidence.evidence
    if isinstance(evidence, SevenLedger):
        q = evidence.qualified_complete_physical_ebpw
        if _present(q) and q is not None:
            return REQUIRED_QUANTITY, q.value, q.evidence
        p = evidence.prospective_meta_bpw
        if _present(p) and p is not None:
            return ProspectiveMetaBpw.category, p.value, p.evidence
        present = evidence.present_names()
        if present:
            qty = evidence.get(present[0])
            return present[0], qty.value if qty is not None else None, (
                qty.evidence if qty is not None else "seven ledger"
            )
        return None, None, "empty SevenLedger"
    if not isinstance(evidence, Mapping):
        return None, None, f"unsupported evidence type {type(evidence).__name__}"

    explicit = evidence.get("quantity") or evidence.get("claimed_on") or evidence.get("claimed_quantity")
    if isinstance(explicit, str) and explicit.strip():
        name = explicit.strip()
        raw = evidence.get("value", evidence.get(name))
        return name, _as_number(raw), "evidence.quantity"

    if isinstance(evidence.get(REQUIRED_QUANTITY), (int, float, Quantity)) and not isinstance(
        evidence.get(REQUIRED_QUANTITY), bool
    ):
        return REQUIRED_QUANTITY, _as_number(evidence.get(REQUIRED_QUANTITY)), REQUIRED_QUANTITY

    for name in SEVEN_TYPES:
        raw = evidence.get(name)
        if isinstance(raw, Quantity) and raw.value is not None:
            return name, raw.value, name
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return name, float(raw), name

    schema = str(evidence.get("schema") or "")
    if "meta_coherence_screen" in schema or evidence.get("diagnostic_factor_equivalent_bpw") is not None:
        row0 = {}
        rows = _dot(evidence, "surface.rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            row0 = rows[0]
        factor = _as_number(
            evidence.get("diagnostic_factor_equivalent_bpw")
            or row0.get("diagnostic_factor_equivalent_bpw")
        )
        return "diagnostic_factor_equivalent_bpw", factor, "coherence screen"
    if schema.startswith("hawking.flash.meta_representation") or evidence.get("prospective_meta_bpw") is not None:
        value = _as_number(
            evidence.get("prospective_meta_bpw")
            or _dot(evidence, "metric.prospective_target")
        )
        return ProspectiveMetaBpw.category, value, "meta representation"
    return None, None, "no quantity named"


def _evidence_flags(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capability_preserving_runtime": evidence.get("capability_preserving_runtime"),
        "physical_measurement_authority": evidence.get("physical_measurement_authority"),
        "bench_state": evidence.get("bench_state") or _dot(evidence, "bench.state"),
        "measurement_state": evidence.get("measurement_state"),
        "path_kind": evidence.get("path_kind") or PRODUCTION,
        "dense_rematerialization": evidence.get("dense_rematerialization"),
        "consumes_representation_directly": evidence.get("consumes_representation_directly"),
        "executable_byte_ledger": evidence.get("executable_byte_ledger"),
        "coherent": evidence.get("coherent"),
        "surface_gate_pass": evidence.get("surface_gate_pass"),
        "heldout_relative_fro_error": evidence.get("heldout_relative_fro_error"),
        "combinator": evidence.get("combinator") is True,
        "artifacts": evidence.get("artifacts"),
    }


def _coherence_from_evidence(evidence: Mapping[str, Any] | None) -> tuple[bool | None, str]:
    if evidence is None:
        return None, "no coherence evidence supplied"
    if evidence.get("coherent") is False or evidence.get("surface_gate_pass") is False:
        return False, "caller marked the representation incoherent / surface gate failed"
    error = _as_number(evidence.get("heldout_relative_fro_error"))
    cosine = _as_number(evidence.get("heldout_cosine"))
    max_err = _as_number(evidence.get("max_heldout_relative_fro_error")) or MAX_HELDOUT_RELATIVE_FRO
    min_cos = _as_number(evidence.get("min_heldout_cosine")) or MIN_HELDOUT_COSINE
    if error is not None and error > max_err:
        return False, (
            f"heldout_relative_fro_error={error} exceeds contract {max_err}"
        )
    if cosine is not None and cosine < min_cos:
        return False, f"heldout_cosine={cosine} below contract {min_cos}"
    if evidence.get("coherent") is True or evidence.get("surface_gate_pass") is True:
        return True, "caller marked coherence held"
    schema = str(evidence.get("schema") or "")
    if "meta_coherence_screen" in schema:
        rows = _dot(evidence, "surface.rows") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            row = rows[0]
            if row.get("surface_gate_pass") is False:
                return False, (
                    f"screen first_surface_failure={row.get('first_surface_failure')}"
                )
            if row.get("surface_gate_pass") is True:
                return True, "screen surface_gate_pass"
    return None, "coherence not measured on this evidence"


def _status_payload(
    *,
    rung: Rung,
    verdict: str,
    reason: str,
    missing: Sequence[str],
    claimed_quantity: str | None,
    claimed_value: float | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rung": rung.id,
        "label": rung.label,
        "target_bpw": rung.target_bpw,
        "exclusive": rung.exclusive,
        "verdict": verdict,
        "reason": reason,
        "required_quantity": REQUIRED_QUANTITY,
        "required_evidence_class": REQUIRED_EVIDENCE_CLASS,
        "claimed_quantity": claimed_quantity,
        "claimed_value": claimed_value,
        "missing_artifacts": list(missing),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
    }
    if extra:
        row.update(dict(extra))
    return row


def status(rung: Rung | str | float, evidence: Any = None) -> dict[str, Any]:
    """REACHED | REFUSED | UNTESTED.

    REACHED only on qualified_complete_physical_ebpw at or below the bound,
    with every required artifact MET. A claim on prospective_meta_bpw (or
    any other typed quantity, or a diagnostic factor bpw) is REFUSED and
    names the required class. Absent physical evidence is UNTESTED.
    """
    resolved = resolve_rung(rung)
    if evidence is None:
        return _status_from_disk(resolved)
    return _status_from_evidence(resolved, evidence)


def _status_from_disk(rung: Rung) -> dict[str, Any]:
    docs = load_docs()
    seven = seven_from_docs(docs)
    inventory = inspect_artifacts(docs)
    missing = missing_artifacts(inventory)
    q = seven.qualified_complete_physical_ebpw
    p = seven.prospective_meta_bpw
    claimed_q = REQUIRED_QUANTITY if _present(q) else None
    claimed_v = q.value if q is not None else None
    if _present(p) and not _present(q):
        # A research target on disk is not a claim that the rung was reached;
        # it is UNTESTED physical, recorded so nobody launders it later.
        return _status_payload(
            rung=rung,
            verdict=UNTESTED,
            reason=(
                "no qualified_complete_physical_ebpw on disk; "
                f"prospective_meta_bpw={p.value if p is not None else None} is "
                f"{RESEARCH_TARGET} and cannot mark {rung.id} REACHED; "
                f"missing_artifacts={missing}"
            ),
            missing=missing,
            claimed_quantity=ProspectiveMetaBpw.category,
            claimed_value=p.value if p is not None else None,
            extra={
                "artifacts": inventory,
                "prospective_meta_bpw_role": RESEARCH_TARGET,
                "qualified_complete_physical_ebpw_state": UNKNOWN,
            },
        )
    if not _present(q):
        return _status_payload(
            rung=rung,
            verdict=UNTESTED,
            reason=(
                "qualified_complete_physical_ebpw is UNKNOWN on this host; "
                f"rung {rung.id} is not REACHED. missing_artifacts={missing}"
            ),
            missing=missing,
            claimed_quantity=claimed_q,
            claimed_value=claimed_v,
            extra={
                "artifacts": inventory,
                "qualified_complete_physical_ebpw_state": UNKNOWN,
            },
        )
    # A numeric physical on disk still has to clear artifacts, authority, bound.
    return _status_from_evidence(
        rung,
        {
            REQUIRED_QUANTITY: q,
            "physical_measurement_authority": None,
            "disk_inventory": inventory,
            "combinator": False,
        },
    )


def _status_from_evidence(rung: Rung, evidence: Any) -> dict[str, Any]:
    category, value, cat_source = _claim_category(evidence)
    mapping = evidence if isinstance(evidence, Mapping) else {}
    flags = _evidence_flags(mapping) if mapping else {}
    inventory = mapping.get("disk_inventory")
    if not isinstance(inventory, list):
        supplied = mapping.get("artifacts")
        if flags.get("combinator") and isinstance(supplied, Mapping):
            inventory = [
                {
                    "id": art["id"],
                    "state": "PRESENT_MET" if supplied.get(art["id"]) is True else "ABSENT",
                    "reason": "combinator-supplied inventory (not a measurement)",
                    "must": art["must"],
                }
                for art in REQUIRED_ARTIFACTS
            ]
        else:
            inventory = inspect_artifacts(load_docs())
    missing = missing_artifacts(inventory)

    if category is not None and category != REQUIRED_QUANTITY:
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"rung {rung.id} claimed on {category}; required class is "
                f"{REQUIRED_QUANTITY} backed by {REQUIRED_EVIDENCE_CLASS}. "
                f"{category} is not interchangeable with {REQUIRED_QUANTITY} "
                f"(source={cat_source})"
            ),
            missing=missing if missing else [REQUIRED_QUANTITY],
            claimed_quantity=category,
            claimed_value=value,
            extra={"artifacts": inventory, "claim_source": cat_source},
        )

    if category is None and not flags.get("combinator"):
        return _status_payload(
            rung=rung,
            verdict=UNTESTED,
            reason=(
                f"no quantity named on evidence ({cat_source}); "
                f"{rung.id} stays {UNTESTED}"
            ),
            missing=missing,
            claimed_quantity=None,
            claimed_value=None,
            extra={"artifacts": inventory},
        )

    if value is None:
        return _status_payload(
            rung=rung,
            verdict=UNTESTED,
            reason=f"{REQUIRED_QUANTITY} is present as a type but value is NULL/UNKNOWN",
            missing=missing if REQUIRED_QUANTITY in missing or missing else [REQUIRED_QUANTITY],
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=None,
            extra={"artifacts": inventory},
        )

    if not _meets_target(value, rung):
        bound = f"< {rung.target_bpw}" if rung.exclusive else f"<= {rung.target_bpw}"
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"{REQUIRED_QUANTITY}={value} does not meet rung {rung.id} bound {bound}"
            ),
            missing=missing,
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=value,
            extra={"artifacts": inventory},
        )

    coherent, coh_reason = _coherence_from_evidence(mapping if mapping else None)
    if coherent is False:
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"representation is incoherent ({coh_reason}); a number at or "
                f"below {rung.target_bpw} is not a reached rung"
            ),
            missing=missing if "coherence_held" in missing else missing + ["coherence_held"],
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=value,
            extra={"artifacts": inventory, "coherence": coh_reason},
        )

    promote_obj: Any
    if isinstance(evidence, (SevenLedger, Quantity)):
        promote_obj = evidence
    else:
        promote_obj = dict(mapping)
        if REQUIRED_QUANTITY not in promote_obj:
            promote_obj[REQUIRED_QUANTITY] = QualifiedCompletePhysicalEbpw(
                value, evidence="status() claim"
            )
    ok, promote_reason = can_promote(promote_obj)
    if not ok:
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"can_promote refused: {promote_reason}; "
                f"rung {rung.id} is not REACHED"
            ),
            missing=missing,
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=value,
            extra={"artifacts": inventory, "can_promote_reason": promote_reason},
        )

    if missing and not flags.get("combinator"):
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"physical quantity named and can_promote opened, but required "
                f"artifacts are not MET: {missing}"
            ),
            missing=missing,
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=value,
            extra={"artifacts": inventory},
        )

    if flags.get("combinator") and missing:
        return _status_payload(
            rung=rung,
            verdict=REFUSED,
            reason=(
                f"combinator did not supply MET artifacts {missing}; "
                "the gate can open, this attempt did not"
            ),
            missing=missing,
            claimed_quantity=REQUIRED_QUANTITY,
            claimed_value=value,
            extra={"artifacts": inventory, "combinator": True},
        )

    return _status_payload(
        rung=rung,
        verdict=REACHED,
        reason=(
            f"{REQUIRED_QUANTITY}={value} meets {rung.id}; artifacts MET"
            + ("; combinator, not a measurement" if flags.get("combinator") else "")
        ),
        missing=[],
        claimed_quantity=REQUIRED_QUANTITY,
        claimed_value=value,
        extra={
            "artifacts": inventory,
            "combinator": flags.get("combinator"),
            "not_a_measurement": bool(flags.get("combinator")),
            "can_promote_reason": promote_reason,
            "coherence": coh_reason,
        },
    )


def ladder_status(evidence: Any = None) -> list[dict[str, Any]]:
    return [status(r, evidence) for r in RUNGS]


# ---------------------------------------------------------------------------
# budget(). Heterogeneous per-organ proposal. Uniform bpw is not the objective.
# ---------------------------------------------------------------------------


def _bit_class(organ: str) -> str:
    if organ in NEAR_ZERO_ORGANS:
        return "NEAR_ZERO"
    if organ in BULK_ORGANS:
        return "PREDICTABLE_BULK"
    if organ in PREMIUM_FLOOR_BPW:
        return "CONTROL_PREMIUM"
    return "CONTROL_PREMIUM"


def _meta_targets(docs: Mapping[str, Any]) -> dict[str, float]:
    meta = docs.get("meta") if isinstance(docs.get("meta"), Mapping) else None
    out: dict[str, float] = {}
    if not isinstance(meta, Mapping):
        doc, _, _ = resolve_named("FLASH_META_REPRESENTATION_SUB1.json")
        meta = doc
    if not isinstance(meta, Mapping):
        return out
    for row in meta.get("family_budget") or []:
        if not isinstance(row, Mapping) or not row.get("family"):
            continue
        value = _as_number(row.get("meta_bpw_target"))
        if value is not None:
            out[str(row["family"])] = value
    return out


def _organ_mass(docs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Organs with source mass when known. Mass withheld rather than invented."""
    census = docs.get("census") if isinstance(docs.get("census"), Mapping) else None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(census, Mapping):
        for fam in census.get("family_summary") or []:
            if not isinstance(fam, Mapping) or not fam.get("family"):
                continue
            name = str(fam["family"])
            bytes_ = fam.get("bytes")
            frac = fam.get("fraction")
            rows.append(
                {
                    "organ": name,
                    "source_bytes": int(bytes_) if isinstance(bytes_, (int, float)) else None,
                    "source_fraction": float(frac) if isinstance(frac, (int, float)) else None,
                    "origin": "FLASH_ORGAN_CENSUS.family_summary",
                }
            )
            seen.add(name)
    # Collapse only the slices that the census already named as a family.
    # Do NOT use flash_nr_complete.EBPW_TO_FAMILY here: that map sends router
    # into routed_experts, which would delete the control plane from the budget.
    collapse = {
        "embeddings": "embedding_lm_head",
        "lm_head": "embedding_lm_head",
        "ngram_engine": "ngram_embedding",
        "deltanet": "linear_attention_hyperconnection",
        "sparse_attention": "full_attention",
        "residual_hyperconnections": "mlp_hyperconnection",
    }
    for org in ebpw_organs(docs.get("ebpw") if isinstance(docs.get("ebpw"), Mapping) else None):
        name = str(org["organ"])
        mapped = collapse.get(name, name)
        if name in seen or mapped in seen:
            continue
        bytes_ = org.get("source_bytes")
        rows.append(
            {
                "organ": name,
                "source_bytes": int(bytes_) if isinstance(bytes_, (int, float)) else None,
                "source_fraction": None,
                "origin": "FLASH_EBPW_BUDGET.organs",
            }
        )
        seen.add(name)
    if "router" not in seen:
        rows.append(
            {
                "organ": "router",
                "source_bytes": None,
                "source_fraction": None,
                "origin": "control-plane overlay (not a census family)",
            }
        )
    return rows


def _fill_fractions(mass: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = 0
    known = [row for row in mass if isinstance(row.get("source_bytes"), int) and row["source_bytes"] > 0]
    for row in known:
        total += int(row["source_bytes"])
    out = []
    for row in mass:
        item = dict(row)
        if item.get("source_fraction") is None and total > 0 and isinstance(item.get("source_bytes"), int):
            item["source_fraction"] = item["source_bytes"] / total if total else None
        out.append(item)
    return out


def budget(rung: Rung | str | float, docs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Heterogeneous per-organ search-pressure proposal for a rung.

    Router / norm / terminal / recurrent islands keep a premium bit class.
    Predictable bulk (routed experts, n-gram table) absorbs compression.
    Uniform bits-per-value across organs is refused: that is a recorded-dead
    family (uniform_subbit_allocation) and it is not the objective.
    """
    resolved = resolve_rung(rung)
    docs = docs if docs is not None else load_docs()
    mass = _fill_fractions(_organ_mass(docs))
    meta_targets = _meta_targets(docs)

    premium = [row for row in mass if _bit_class(row["organ"]) == "CONTROL_PREMIUM"]
    bulk = [row for row in mass if _bit_class(row["organ"]) != "CONTROL_PREMIUM"]

    premium_contrib = 0.0
    premium_mass = 0.0
    for row in premium:
        frac = row.get("source_fraction")
        if frac is None:
            continue
        floor = PREMIUM_FLOOR_BPW.get(row["organ"], 2.5)
        premium_contrib += float(frac) * floor
        premium_mass += float(frac)

    bulk_mass = 0.0
    for row in bulk:
        frac = row.get("source_fraction")
        if frac is not None:
            bulk_mass += float(frac)

    target = SUB1_RESEARCH_TARGET_BPW if resolved.id == "bpw_sub1" else resolved.target_bpw
    remaining = target - premium_contrib
    bulk_bpw: float | None
    allocation_reason: str
    if bulk_mass <= 0.0:
        bulk_bpw = None
        allocation_reason = (
            "bulk source fractions unavailable or zero; bulk bpw withheld "
            "rather than invented"
        )
    elif remaining < 0:
        bulk_bpw = None
        allocation_reason = (
            f"premium floors already contribute {premium_contrib:.6f} bpw of "
            f"weighted mass, which exceeds target {target}; refusing to crush "
            "the control plane to hit a number"
        )
    else:
        bulk_bpw = remaining / bulk_mass
        allocation_reason = (
            f"premium floors held; bulk organs share remaining weighted mass "
            f"{remaining:.6f} / {bulk_mass:.6f} = {bulk_bpw:.6f} proposed bpw "
            f"so the weighted total equals search-pressure target {target}"
        )

    organs: list[dict[str, Any]] = []
    weighted = 0.0
    weighted_ok = True
    for row in mass:
        organ = row["organ"]
        bit_class = _bit_class(organ)
        cited_meta = meta_targets.get(organ)
        if bit_class == "NEAR_ZERO":
            proposed: float | None = 0.0
            why = "unused on the language decode path; near-zero storage is the proposal"
        elif bit_class == "CONTROL_PREMIUM":
            proposed = PREMIUM_FLOOR_BPW.get(organ, 2.5)
            why = (
                "control / terminal / recurrent island keeps a premium bit class; "
                "uniform crush with bulk is the dead family uniform_subbit_allocation"
            )
            if cited_meta is not None and resolved.id == "bpw_sub1":
                proposed = cited_meta
                why = (
                    "sub-1 search pressure reuses FLASH_META_REPRESENTATION_SUB1 "
                    f"family_budget meta_bpw_target={cited_meta} as a description-"
                    "budget allocation, not as qualified_complete_physical_ebpw"
                )
        else:
            if resolved.id == "bpw_sub1" and cited_meta is not None:
                proposed = cited_meta
                why = (
                    "sub-1 bulk target cited from FLASH_META_REPRESENTATION_SUB1 "
                    f"family_budget meta_bpw_target={cited_meta}; RESEARCH_TARGET, "
                    "not a physical claim"
                )
            else:
                proposed = bulk_bpw
                why = allocation_reason
        if proposed is None:
            weighted_ok = False
        elif row.get("source_fraction") is None:
            weighted_ok = False
        else:
            weighted += float(row["source_fraction"]) * float(proposed)
        organs.append(
            {
                "organ": organ,
                "bit_class": bit_class,
                "proposed_bpw": proposed,
                "premium_floor_bpw": PREMIUM_FLOOR_BPW.get(organ),
                "meta_bpw_target_cited": cited_meta,
                "source_bytes": row.get("source_bytes"),
                "source_fraction": row.get("source_fraction"),
                "origin": row.get("origin"),
                "why": why,
                "quantity": "search_pressure_allocation",
                "not_qualified_complete_physical_ebpw": True,
                "not_a_measurement": True,
            }
        )

    proposed_values = [o["proposed_bpw"] for o in organs if o["proposed_bpw"] is not None]
    distinct = sorted(set(round(v, 6) for v in proposed_values))
    uniform = bool(proposed_values) and len(distinct) == 1

    return {
        "rung": resolved.id,
        "label": resolved.label,
        "target_bpw": resolved.target_bpw,
        "exclusive": resolved.exclusive,
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": False,
        "uniform_refused": True,
        "uniform_reason": (
            "uniform bits-per-value across organs is not the objective and is "
            "the recorded-dead family uniform_subbit_allocation; router and "
            "other control structures keep premium precision while predictable "
            "bulk goes near zero"
        ),
        "allocation_kind": "SEARCH_PRESSURE_PROPOSAL",
        "allocation_reason": allocation_reason,
        "organs": organs,
        "counts": {
            "organs": len(organs),
            "premium": sum(1 for o in organs if o["bit_class"] == "CONTROL_PREMIUM"),
            "bulk": sum(1 for o in organs if o["bit_class"] == "PREDICTABLE_BULK"),
            "near_zero": sum(1 for o in organs if o["bit_class"] == "NEAR_ZERO"),
            "distinct_proposed_bpw": len(distinct),
        },
        "weighted_proposed_bpw": weighted if weighted_ok else None,
        "weighted_withheld_reason": (
            None
            if weighted_ok
            else "one or more organs lack source_fraction or proposed_bpw; total withheld"
        ),
        "emitted_uniform": uniform,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "not_a_measurement": True,
        "claim_boundary": (
            "Search-pressure allocation proposal. proposed_bpw is not "
            "qualified_complete_physical_ebpw, not a packed artifact, and not "
            "a capability result. Capability of any allocation is UNTESTED."
        ),
    }


# ---------------------------------------------------------------------------
# dominated(). Three rules, each independently able to fire.
# ---------------------------------------------------------------------------


FLOOR_CONTROL: dict[str, Any] = {
    "id": "coherent_floor_control",
    "bpw": COHERENT_FLOOR_BPW,
    "quantity": "search_pressure_floor",
    "storage_bytes": FLOOR_CONTROL_STORAGE,
    "flop_milli": FLOOR_CONTROL_FLOP_MILLI,
    "coherent": True,
    "surface_gate_pass": True,
    "rematerializes_dense_parent": False,
    "path_kind": PRODUCTION,
    "not_a_measurement": True,
    "note": (
        "Relative structural control at the ~2.25 coherent-class floor. "
        "Not a Flash qualified-physical result."
    ),
}


def _plan_storage(plan: Mapping[str, Any]) -> int | None:
    raw = plan.get("storage_bytes")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        if raw < 0:
            raise LadderRefuse("storage_bytes < 0 is not a plan")
        return int(raw)
    return None


def _plan_flop(plan: Mapping[str, Any]) -> int | None:
    raw = plan.get("flop_milli")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        if raw < 0:
            raise LadderRefuse("flop_milli < 0 is not a plan")
        return int(raw)
    return None


def _plan_bpw(plan: Mapping[str, Any]) -> float | None:
    return _as_number(plan.get("bpw"))


def _smaller(plan: Mapping[str, Any], incumbent: Mapping[str, Any]) -> tuple[bool | None, str]:
    ps, iss = _plan_storage(plan), _plan_storage(incumbent)
    if ps is not None and iss is not None:
        if ps < iss:
            return True, f"storage_bytes {ps} < incumbent {iss}"
        return False, f"storage_bytes {ps} is not < incumbent {iss}"
    pb, ib = _plan_bpw(plan), _plan_bpw(incumbent)
    if pb is not None and ib is not None:
        # Bare numbers of the plan's own claimed figure, not Quantity arithmetic.
        if pb < ib:
            return True, f"claimed bpw number {pb} < incumbent {ib} (not category arithmetic)"
        return False, f"claimed bpw number {pb} is not < incumbent {ib}"
    return None, "plan and incumbent share no comparable size (storage_bytes or bpw)"


def _incoherent(plan: Mapping[str, Any]) -> tuple[bool, str]:
    if plan.get("coherent") is False or plan.get("surface_gate_pass") is False:
        return True, "plan is marked incoherent / surface_gate_pass is false"
    error = _as_number(plan.get("heldout_relative_fro_error"))
    cosine = _as_number(plan.get("heldout_cosine"))
    max_err = _as_number(plan.get("max_heldout_relative_fro_error")) or MAX_HELDOUT_RELATIVE_FRO
    min_cos = _as_number(plan.get("min_heldout_cosine")) or MIN_HELDOUT_COSINE
    if error is not None and error > max_err:
        return True, (
            f"heldout_relative_fro_error={error} exceeds contract {max_err}; "
            "the L4 REAL256 shape (small diagnostic bpw, large held-out error) is dominated"
        )
    if cosine is not None and cosine < min_cos:
        return True, f"heldout_cosine={cosine} below contract {min_cos}"
    if plan.get("quantity") in WRONG_CLAIM_QUANTITIES and (
        error is not None or plan.get("surface_gate_pass") is False
    ):
        return True, (
            f"plan quantity {plan.get('quantity')} is not {REQUIRED_QUANTITY} "
            "and the function evidence does not hold"
        )
    return False, "no incoherence evidence on this plan"


def _rematerializes(plan: Mapping[str, Any]) -> tuple[bool, str]:
    if plan.get("rematerializes_dense_parent") is True:
        return True, "plan rematerializes a dense parent"
    if plan.get("decompresses_to_dense_weight_tensor") is True:
        return True, "plan decompresses to a dense weight tensor"
    if plan.get("runs_ordinary_kernels") is True and plan.get("decompresses_to_dense_weight_tensor") is True:
        return True, "plan decompresses to dense then runs ordinary kernels"
    path_kind = str(plan.get("path_kind") or PRODUCTION).upper()
    if path_kind == VERIFICATION:
        return False, "verification MAY reconstruct; remat-domination applies to production"
    remat = judge_production_dense_checkpoint(plan) if isinstance(plan, Mapping) else None
    if remat is not None and remat.get("reconstructs_dense_checkpoint") is True and path_kind != VERIFICATION:
        return True, remat.get("reason") or "production reconstructs a dense checkpoint"
    return False, "no proven dense-parent rematerialization"


def _compute_multiplied(plan: Mapping[str, Any], incumbent: Mapping[str, Any]) -> tuple[bool | None, str]:
    ps, iss = _plan_storage(plan), _plan_storage(incumbent)
    pf, iff = _plan_flop(plan), _plan_flop(incumbent)
    if ps is None or iss is None or pf is None or iff is None:
        return None, (
            "compute-multiplication rule withheld: need storage_bytes and "
            "flop_milli on both plan and incumbent"
        )
    if ps <= 0 or iss <= 0 or iff <= 0:
        return None, "compute-multiplication rule withheld: non-positive size/flop"
    byte_ratio = ps / iss
    flop_ratio = pf / iff
    if byte_ratio >= 1.0:
        return False, "plan is not smaller in storage; compute-multiplication rule does not apply"
    byte_saving = iss / ps
    if flop_ratio > byte_saving:
        return True, (
            f"flop_milli {pf} / {iff} = {flop_ratio:.4f}x exceeds byte saving "
            f"{byte_saving:.4f}x (storage {ps}/{iss}). Recovered trap: "
            "flash_schools 0.5x bytes / 3.0x FLOPs scores 3500 vs 2000 joint_cost."
        )
    return False, (
        f"flop_ratio {flop_ratio:.4f}x does not exceed byte saving {byte_saving:.4f}x"
    )


def dominated(plan: Mapping[str, Any], incumbent: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reject a plan that is smaller but incoherent, rematerializing, or slower.

    Three independent rules. Each has a watched-failing negative control.
    A smaller coherent direct-consume plan whose FLOP growth stays inside
    its byte saving stays on the Pareto front.
    """
    if not isinstance(plan, Mapping) or not plan:
        raise LadderRefuse("dominated() refuses an empty or non-mapping plan")
    inc = dict(incumbent) if isinstance(incumbent, Mapping) else dict(plan.get("incumbent") or FLOOR_CONTROL)
    smaller, small_reason = _smaller(plan, inc)
    if smaller is None:
        return {
            "verdict": REFUSED,
            "dominated": True,
            "fired": [],
            "withheld": [RULE_INCOHERENT, RULE_REMAT, RULE_COMPUTE],
            "reasons": [small_reason],
            "on_pareto_front": False,
            "smaller": None,
            "plan_id": plan.get("id"),
            "incumbent_id": inc.get("id"),
            "fail_closed": small_reason,
        }

    fired: list[str] = []
    reasons: list[str] = []
    withheld: list[str] = []

    if smaller:
        bad, why = _incoherent(plan)
        if bad:
            fired.append(RULE_INCOHERENT)
            reasons.append(why)
        bad, why = _rematerializes(plan)
        if bad:
            fired.append(RULE_REMAT)
            reasons.append(why)
        comp, why = _compute_multiplied(plan, inc)
        if comp is None:
            withheld.append(RULE_COMPUTE)
            reasons.append(why)
        elif comp:
            fired.append(RULE_COMPUTE)
            reasons.append(why)
        else:
            reasons.append(why)
    else:
        reasons.append(small_reason)

    is_dom = bool(fired)
    return {
        "verdict": DOMINATED if is_dom else ON_FRONT,
        "dominated": is_dom,
        "fired": fired,
        "withheld": withheld,
        "reasons": reasons,
        "on_pareto_front": (not is_dom) and smaller is not None,
        "smaller": smaller,
        "smaller_reason": small_reason,
        "plan_id": plan.get("id"),
        "incumbent_id": inc.get("id"),
        "plan_quantity": plan.get("quantity"),
        "required_quantity_for_a_rung": REQUIRED_QUANTITY,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "not_a_measurement": True,
    }


def pareto_front(
    plans: Sequence[Mapping[str, Any]],
    incumbent: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Plans that survive all three domination rules against the incumbent."""
    front: list[dict[str, Any]] = []
    for plan in plans:
        verdict = dominated(plan, incumbent)
        if verdict["verdict"] == ON_FRONT:
            front.append(dict(plan))
    return front


# ---------------------------------------------------------------------------
# Combinators. Prove the gates can open. Not measurements.
# ---------------------------------------------------------------------------


def combinator_reached_evidence(rung: Rung | str | float) -> dict[str, Any]:
    """In-memory evidence that can open REACHED. Explicitly not a measurement."""
    resolved = resolve_rung(rung)
    value = resolved.target_bpw * 0.5 if resolved.exclusive else resolved.target_bpw
    if resolved.exclusive:
        value = min(value, 0.5)
    return {
        "quantity": REQUIRED_QUANTITY,
        REQUIRED_QUANTITY: QualifiedCompletePhysicalEbpw(
            value, evidence="synthetic combinator control (not a measurement)"
        ),
        "value": value,
        "executable_byte_ledger": {
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        "capability_preserving_runtime": True,
        "physical_measurement_authority": PROTECTED,
        "bench_state": "PROTECTED",
        "measurement_state": PROTECTED,
        "path_kind": PRODUCTION,
        "dense_rematerialization": False,
        "consumes_representation_directly": True,
        "coherent": True,
        "surface_gate_pass": True,
        "heldout_relative_fro_error": 0.01,
        "heldout_cosine": 0.9995,
        "combinator": True,
        "not_a_measurement": True,
        "artifacts": {art["id"]: True for art in REQUIRED_ARTIFACTS},
    }


def remat_plan() -> dict[str, Any]:
    return {
        "id": "smaller_dense_remat",
        "storage_bytes": FLOOR_CONTROL_STORAGE // 2,
        "flop_milli": FLOOR_CONTROL_FLOP_MILLI,
        "bpw": 1.0,
        "coherent": True,
        "surface_gate_pass": True,
        "rematerializes_dense_parent": True,
        "decompresses_to_dense_weight_tensor": True,
        "runs_ordinary_kernels": True,
        "path_kind": PRODUCTION,
        "not_a_measurement": True,
    }


def compute_trap_plan() -> dict[str, Any]:
    """Half the bytes, triple the FLOPs. Recovered from flash_schools.make_trap."""
    return {
        "id": "half_bytes_triple_flops",
        "storage_bytes": FLOOR_CONTROL_STORAGE // 2,
        "flop_milli": FLOOR_CONTROL_FLOP_MILLI * 3,
        "bpw": COHERENT_FLOOR_BPW / 2.0,
        "coherent": True,
        "surface_gate_pass": True,
        "rematerializes_dense_parent": False,
        "path_kind": PRODUCTION,
        "not_a_measurement": True,
        "extends": "tools/future/flash_schools.py#make_trap",
    }


def honest_smaller_plan() -> dict[str, Any]:
    return {
        "id": "honest_smaller_direct",
        "storage_bytes": int(FLOOR_CONTROL_STORAGE * 0.8),
        "flop_milli": int(FLOOR_CONTROL_FLOP_MILLI * 1.1),
        "bpw": 2.0,
        "coherent": True,
        "surface_gate_pass": True,
        "heldout_relative_fro_error": 0.01,
        "rematerializes_dense_parent": False,
        "path_kind": PRODUCTION,
        "not_a_measurement": True,
    }


# ---------------------------------------------------------------------------
# Selftest. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def selftest() -> dict[str, Any]:
    results: dict[str, Any] = {}

    if tuple(cls.category for cls in IMPORTED_SEVEN) != tuple(SEVEN_TYPES):
        raise AssertionError("IMPORTED_SEVEN drifted from flash_nr_complete.SEVEN_TYPES")
    results["seven_imported_not_redefined"] = True

    ids = [r.id for r in rungs()]
    results["rung_ids"] = ids
    results["rung_count"] = len(ids)
    if ids != [
        "bpw_2_25",
        "bpw_2_00",
        "bpw_1_75",
        "bpw_1_50",
        "bpw_1_25",
        "bpw_1_00",
        "bpw_sub1",
    ]:
        raise AssertionError(f"ladder drifted: {ids}")

    meta_claim = ProspectiveMetaBpw(0.887, evidence=RESEARCH_TARGET)
    refused = status("bpw_sub1", meta_claim)
    results["meta_claim_verdict"] = refused["verdict"]
    results["meta_claim_reason"] = refused["reason"]
    if refused["verdict"] != REFUSED:
        raise AssertionError("prospective_meta_bpw claim was not REFUSED")
    if REQUIRED_QUANTITY not in refused["reason"] and refused["required_quantity"] != REQUIRED_QUANTITY:
        raise AssertionError("refusal did not name qualified_complete_physical_ebpw")

    for name in WRONG_CLAIM_QUANTITIES & set(SEVEN_TYPES):
        cls = SEVEN_TYPES[name]
        qty = cls(0.5, evidence="wrong-class negative control")
        row = status("bpw_1_00", qty)
        if row["verdict"] != REFUSED:
            raise AssertionError(f"claim on {name} was not REFUSED")
        if row["required_quantity"] != REQUIRED_QUANTITY:
            raise AssertionError(f"refusal for {name} did not name required class")
    results["all_six_wrong_classes_refused"] = True

    disk = ladder_status()
    reached = [row["rung"] for row in disk if row["verdict"] == REACHED]
    results["disk_reached"] = reached
    results["disk_verdicts"] = {row["rung"]: row["verdict"] for row in disk}
    if reached:
        raise AssertionError(f"disk marked REACHED without qualified physical: {reached}")
    if any(row["verdict"] not in {UNTESTED, REFUSED} for row in disk):
        raise AssertionError(f"disk verdicts drifted: {results['disk_verdicts']}")

    rem = dominated(remat_plan())
    results["remat_fired"] = rem["fired"]
    if RULE_REMAT not in rem["fired"] or rem["verdict"] != DOMINATED:
        raise AssertionError(f"dense-remat plan was not dominated: {rem}")

    trap = dominated(compute_trap_plan())
    results["compute_fired"] = trap["fired"]
    if RULE_COMPUTE not in trap["fired"] or trap["verdict"] != DOMINATED:
        raise AssertionError(f"0.5x/3.0x plan was not dominated: {trap}")

    screen = load_screen()
    results["screen_reachable"] = bool(screen.get("reachable"))
    if screen.get("reachable"):
        trap_plan = screen_trap_plan(screen)
        inc = trap_plan.get("incumbent")
        incoh = dominated(trap_plan, inc if isinstance(inc, Mapping) else None)
        results["screen_fired"] = incoh["fired"]
        results["screen_factor_bpw"] = trap_plan.get("bpw")
        results["screen_heldout_error"] = trap_plan.get("heldout_relative_fro_error")
        if RULE_INCOHERENT not in incoh["fired"] or incoh["verdict"] != DOMINATED:
            raise AssertionError(f"L4 REAL256 screen plan was not dominated: {incoh}")
        screen_status = status("bpw_sub1", {"schema": "hawking.flash.meta_coherence_screen.v1", **screen, "diagnostic_factor_equivalent_bpw": trap_plan["bpw"]})
        results["screen_status_verdict"] = screen_status["verdict"]
        if screen_status["verdict"] != REFUSED:
            raise AssertionError("sub-1 claimed on diagnostic factor bpw was not REFUSED")
    else:
        # Screen unread: still prove the rule with the contract's cited shape,
        # labeled as a structural control, not as a loaded receipt.
        synthetic_screen = {
            "id": "cited_l4_shape",
            "quantity": "diagnostic_factor_equivalent_bpw",
            "bpw": 0.0254,
            "storage_bytes": 1218560,
            "flop_milli": FLOOR_CONTROL_FLOP_MILLI,
            "heldout_relative_fro_error": 0.5284,
            "surface_gate_pass": False,
            "coherent": False,
            "rematerializes_dense_parent": False,
            "path_kind": PRODUCTION,
            "not_a_measurement": True,
        }
        incoh = dominated(
            synthetic_screen,
            {"id": "dense", "storage_bytes": 766771200, "flop_milli": FLOOR_CONTROL_FLOP_MILLI, "coherent": True},
        )
        results["screen_fired"] = incoh["fired"]
        results["screen_unreachable_used_cited_shape"] = True
        if RULE_INCOHERENT not in incoh["fired"]:
            raise AssertionError("cited L4 shape did not fire incoherent domination")

    honest = dominated(honest_smaller_plan())
    results["honest_verdict"] = honest["verdict"]
    if honest["verdict"] != ON_FRONT or honest["dominated"] is not False:
        raise AssertionError(f"honest smaller plan was dominated: {honest}")

    opened = status("bpw_2_25", combinator_reached_evidence("bpw_2_25"))
    results["combinator_verdict"] = opened["verdict"]
    if opened["verdict"] != REACHED:
        raise AssertionError(f"positive combinator failed to open REACHED: {opened}")
    results["combinator_is_not_a_measurement"] = True

    above = status(
        "bpw_2_00",
        combinator_reached_evidence("bpw_2_25"),
    )
    results["above_target_verdict"] = above["verdict"]
    if above["verdict"] != REFUSED:
        raise AssertionError("physical above the rung target was not REFUSED")

    for r in RUNGS:
        b = budget(r)
        if b["forces_uniform_bpw"] is True:
            raise AssertionError(f"{r.id} budget forces uniform bpw")
        values = [o["proposed_bpw"] for o in b["organs"] if o["proposed_bpw"] is not None]
        if values and len(set(round(v, 6) for v in values)) < 2:
            raise AssertionError(f"{r.id} budget collapsed to a single proposed_bpw")
        by_name = {o["organ"]: o for o in b["organs"]}
        router = by_name.get("router") or by_name.get("ROUTER")
        routed = by_name.get("routed_experts") or by_name.get("ROUTED_EXPERTS")
        if (
            router
            and routed
            and router["proposed_bpw"] is not None
            and routed["proposed_bpw"] is not None
            and router["proposed_bpw"] < routed["proposed_bpw"]
        ):
            raise AssertionError(
                f"{r.id}: router proposed_bpw {router['proposed_bpw']} "
                f"< routed_experts {routed['proposed_bpw']}"
            )
    results["budgets_heterogeneous"] = True

    mixed = False
    try:
        _ = ProspectiveMetaBpw(0.887) + QualifiedCompletePhysicalEbpw(0.887)
    except CategoryError:
        mixed = True
    results["cross_category_still_raises"] = mixed
    if not mixed:
        raise AssertionError("seven-type arithmetic no longer raises")

    unknown = False
    try:
        resolve_rung("ERA_VI_DOES_NOT_EXIST")
    except UnknownRungError:
        unknown = True
    results["unknown_rung_raises"] = unknown
    if not unknown:
        raise AssertionError("unknown rung was accepted")

    empty = False
    try:
        dominated({})
    except LadderRefuse:
        empty = True
    results["empty_plan_raises"] = empty
    if not empty:
        raise AssertionError("empty plan was not refused")

    return results


# ---------------------------------------------------------------------------
# Recovery / gaps / callable.
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/future/ebpw_categories.py",
            "what": (
                "Five Quantity subclasses, can_promote, production/verification "
                "remat split, PromotionLedger. The seven named quantities the "
                "lane listed are NOT here."
            ),
            "adequate": False,
            "gap": "five types, not the seven; no ladder, no dominated(), no per-rung budget",
        },
        {
            "path": "tools/future/flash_nr_complete.py",
            "what": (
                "The seven typed quantities (SourceControlEbpw, "
                "StaticActiveEbpwEstimate, StaticCompleteEbpwEstimate, "
                "ProspectiveMetaBpw, SerializedNrInformation, SerializedNxEbpw, "
                "QualifiedCompletePhysicalEbpw), SevenLedger, can_promote, "
                "continuous NR, resolve_evidence. IMPORTED, not redefined."
            ),
            "adequate": False,
            "gap": "types and promotion, not a gated 2.25→sub-1 ladder or three-rule domination",
        },
        {
            "path": "tools/future/flash_nx_audit.py",
            "what": "SEVEN_REQUIREMENTS for a promotable NX. Cited as required artifacts.",
            "adequate": False,
            "gap": "NX completeness, not EBPW rungs",
        },
        {
            "path": "tools/future/meta_funnel.py",
            "what": (
                "Nine-gate advance refusal; total executable information as the "
                "unit; uniform-bpw plans are well-formed-but-not-the-objective."
            ),
            "adequate": False,
            "gap": "funnel kills candidates; does not gate a physical-EBPW ladder",
        },
        {
            "path": "tools/future/flash_schools.py",
            "what": (
                "joint_cost / make_trap 0.5x-bytes/3.0x-FLOPs; heterogeneous "
                "router premium vs uniform crush; forbids_dense_rematerialization."
            ),
            "adequate": False,
            "gap": "organ Gravity search, not complete-executable EBPW rungs",
        },
        {
            "path": "tools/headless/composition_ladder.py",
            "what": (
                "Eight-rung QUALIFICATION ladder (local probe → capability). "
                "UNREACHED ≠ FAILED. Recovered via git show; not forked."
            ),
            "adequate": False,
            "gap": "qualification rungs, not executable-information rungs; Codex-owned",
        },
        {
            "path": SCREEN_REL,
            "what": (
                "L4 REAL256 screen: diagnostic_factor_equivalent_bpw ~0.0254 "
                "with heldout_relative_fro_error ~0.5284; status "
                "OFFLINE_META_SURFACE_GATE_FAILED. The trap this module exists to keep."
            ),
            "adequate": False,
            "gap": "a failed organ screen, not a ladder",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "Gated 2.25→2.0→1.75→1.5→1.25→~1.0→sub-1 ladder; each rung names required quantity, evidence class, and artifacts.",
        "status() is REACHED only on qualified_complete_physical_ebpw; prospective_meta_bpw and diagnostic factor bpw are REFUSED and name the required class.",
        "With no qualified physical evidence on disk, every rung is UNTESTED or REFUSED and none REACHED.",
        "budget() emits a heterogeneous per-organ search-pressure proposal; uniform bpw is refused; control keeps premium bits, bulk absorbs compression.",
        "dominated() encodes three independent rules (incoherent / dense-rematerializing / compute-multiplied) and keeps the Pareto front.",
        "Seven EBPW types imported from flash_nr_complete; mixed arithmetic still raises CategoryError; none redefined here.",
    ]


def negative_findings_from(docs: Mapping[str, Any], screen: Mapping[str, Any]) -> list[dict[str, Any]]:
    seven = seven_from_docs(docs)
    q = seven.qualified_complete_physical_ebpw
    return [
        {
            "looked_for": "qualified_complete_physical_ebpw on disk",
            "found": q.evidence if q is not None else UNKNOWN,
            "value": q.value if q is not None else None,
        },
        {
            "looked_for": "a source-independent complete NX",
            "found": (
                f"nx_v0 resolved_via={(docs.get('resolution') or {}).get('nx_v0', {}).get('resolved_via')}; "
                "metadata seals are not complete executables"
            ),
        },
        {
            "looked_for": "a coherent sub-1 representation",
            "found": (
                f"screen reachable={screen.get('reachable')} status={screen.get('status')} "
                f"diagnostic_factor_equivalent_bpw={screen.get('diagnostic_factor_equivalent_bpw')} "
                f"heldout_relative_fro_error={screen.get('heldout_relative_fro_error')}"
            ),
        },
        {
            "looked_for": "hardware measurement of any rung",
            "found": "sidecar is STATIC_ONLY / gpu_authority false; no GPU lease",
        },
        {
            "looked_for": "orchestration BINDINGS entry for this module",
            "found": (
                "tools/future/orchestration.py is outside this lane's WRITE list; "
                "resident_callable names the frontier this receipt informs, but "
                "the binding table is not mutated here"
            ),
        },
    ]


def emit_workunits() -> list[dict[str, Any]]:
    return [
        {
            "id": "future.flash_bpw_ladder.status",
            "role": "science",
            "description": (
                "Evaluate the complete-executable EBPW ladder against disk. "
                "One CPU_ANALYSIS unit. Never marks a rung REACHED on a "
                "description budget or a diagnostic factor bpw."
            ),
            "dependencies": [],
            "status": "pending",
            "resource_class": "CPU_ANALYSIS",
            "verifier": "future.flash_bpw_ladder.selftest",
            "provider": "sidecar.flash_bpw_ladder",
            "effect_class": "READ_ONLY",
            "preferred_backend": None,
            "classification": "STATIC_ONLY",
            "command": ["python3", "tools/future/flash_bpw_ladder.py", "--build"],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "independent_reproduction",
            "claim_boundary": (
                "WorkUnit is a proposal; receipt and protected capability gates remain authoritative"
            ),
        }
    ]


def resident_callable(units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.flash_bpw_ladder.status()",
        "callables": ["rungs", "status", "budget", "dominated", "pareto_front", "build"],
        "workunit": (
            "one CPU_ANALYSIS unit; evaluate every rung against disk and refuse "
            "REACHED on any quantity other than qualified_complete_physical_ebpw"
        ),
        "work_units_emitted": [u.get("id") for u in units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
        "fails_closed": (
            "unknown rung raises; empty plan raises; missing size refuses "
            "domination rather than passing; absent physical evidence is "
            "UNTESTED not REACHED; wrong-class claims are REFUSED and name "
            "qualified_complete_physical_ebpw; HardwareClaimError on numeric "
            "hardware fields"
        ),
        "can_hcli_invoke": True,
        "discoverable": True,
    }


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def assemble(docs: Mapping[str, Any]) -> dict[str, Any]:
    controls = selftest()
    screen = load_screen()
    units = emit_workunits()
    seven = seven_from_docs(docs)
    inventory = inspect_artifacts(docs)
    disk = ladder_status()
    budgets = [budget(r, docs) for r in RUNGS]
    return {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Gate the 2.25→sub-1 complete-executable EBPW search so a "
            "description budget or a diagnostic factor bpw cannot be reported "
            "as a reached rung, and so a smaller incoherent / rematerializing / "
            "compute-multiplied plan cannot sit on the Pareto front."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "measurement_class": "STATIC_ONLY",
        "required_quantity": REQUIRED_QUANTITY,
        "required_evidence_class": REQUIRED_EVIDENCE_CLASS,
        "seven_quantities_imported_from": "tools.future.flash_nr_complete.SEVEN_TYPES",
        "seven_quantities": list(SEVEN_TYPES),
        "seven_from_disk": seven.as_dict(),
        "rungs": [r.as_dict() for r in RUNGS],
        "disk_status": disk,
        "disk_reached_count": sum(1 for row in disk if row["verdict"] == REACHED),
        "artifacts": inventory,
        "budgets": budgets,
        "screen": {
            k: screen.get(k)
            for k in (
                "reachable",
                "resolved_via",
                "path",
                "status",
                "diagnostic_factor_equivalent_bpw",
                "heldout_relative_fro_error",
                "heldout_cosine",
                "surface_gate_pass",
                "first_surface_failure",
                "physical_ebpw",
                "meta_bpw_target",
                "why_this_is_not_a_win",
            )
        },
        "domination_rules": [RULE_INCOHERENT, RULE_REMAT, RULE_COMPUTE],
        "selftest": controls,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings_from(docs, screen),
        "resident_callable": resident_callable(units),
        "work_units": units,
        "next_workunits": [
            {
                "id": "future.flash_bpw_ladder.status",
                "status": "pending",
                "resource_class": "CPU_ANALYSIS",
            },
            {
                "id": "future.orchestration.bind_flash_bpw_ladder",
                "status": "blocked_write_partition",
                "reason": (
                    "orchestration.py BINDINGS is outside this lane's WRITE list; "
                    "the frontier this receipt informs is named, not bound"
                ),
            },
            {
                "id": "future.flash_nr_complete.qualified_physical_ebpw",
                "status": "sleeping",
                "resource_class": "GPU_EXCLUSIVE",
                "must_not_synthesize_result": True,
                "reason": "the only quantity that can mark a rung REACHED is still UNKNOWN",
            },
        ],
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. Search pressure, "
            "not a win. prospective_meta_bpw and diagnostic factor bpw cannot "
            "mark a rung REACHED."
        ),
    }


def build() -> Path:
    docs = load_docs()
    doc = assemble(docs)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
