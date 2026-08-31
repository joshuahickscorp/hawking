"""META_DOWNSTREAM_READY — gates 3–9 start the instant a real corpus exists.

The Flash meta funnel's binding gate is `real_teacher_fit`. Physical capture
reports BLOCKED_NO_METAL_GPU (0 of 256 rows) at
`dense_source_bf16_prefix_initialization`. This sidecar does not have a GPU
and does not simulate training rows. It prepares every downstream stage so
that the moment an admitted corpus exists, the work starts with no design
left to do.

Compounds, does not fork:
  * tools/future/meta_funnel.py — nine gates; families stall at gate 2 REFUSED
  * tools/future/teacher_corpus.py — validate_corpus anti-fabrication guard
  * tools/future/ebpw_categories.py, router_science.py, ngram_school.py,
    expert_bank_school.py, flash_nx_audit.py — already-built later screens

    python3 tools/future/meta_ready.py --build
    python3 tools/future/meta_ready.py --simulate-arrival
    python3 -m pytest tools/future/test_meta_ready.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import git
from tools.future.flash_nx_audit import evidence_path
from tools.future.meta_funnel import (
    FLASH_MODEL,
    Funnel,
    GATES,
    GATES_BY_ID,
    recover_families,
)
from tools.future.teacher_corpus import (
    AUTHORITIES,
    BOUNDED_TARGET_ROWS,
    CAPABILITY_DOMAINS,
    CorpusRefused,
    DIVERSITY_MEASURES,
    FLASH_SPECIMEN,
    MIN_DOMAINS_FOR_FIT,
    MIN_POSITIONS,
    MIN_PROMPTS_FOR_FIT,
    MIN_ROUTES_ROUTED,
    NATURAL_DUP_RATE,
    POSITION_MAX_SHARE,
    PROVENANCE_PADDING_KINDS,
    ROUTED_SURFACES,
    SURFACES,
    UNIQUE_RATIO_FLOOR,
    UNIQUE_RATIO_MIN,
    annotate_corpus,
    make_row,
    validate_corpus,
)


RECEIPT = "META_DOWNSTREAM_READY.json"
SCHEMA = "hawking.future.meta_ready.v1"
RECORDED_BY = "tools/future/meta_ready.py"
VERSION = 1

PINNED_DIR = REPO / "receipts" / "future" / "evidence"

# Live Codex receipts. Prefer the pinned snapshot when the same name exists
# there; read live_headless only when current capture state is required.
REL_CAPTURE_BOUNDARY = "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json"
REL_META_SUB1 = "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
REL_COHERENCE = "receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json"
REL_NR_V2 = "receipts/headless/FLASH_COMPLETE_V2.nr.json"
REL_NX_V0 = "receipts/headless/FLASH_COMPLETE_V0.nx.json"
REL_ROUTER_SEL = "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json"
REL_ROUTER_MAP = "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json"

PINNED_SUB1 = "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json"
PINNED_COHERENCE = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4.json"
PINNED_NR_V2 = "receipts/future/evidence/FLASH_COMPLETE_V2.nr.json"
PINNED_NX_V0 = "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json"
PINNED_ROUTER_SEL = "receipts/future/evidence/FLASH_NOETIC_ROUTER_SELECTION.json"
PINNED_ROUTER_MAP = "receipts/future/evidence/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json"

SIDECAR_FUNNEL = "receipts/future/META_EXPERIMENT_FUNNEL.json"
SIDECAR_CORPUS = "receipts/future/TEACHER_CORPUS_CONTRACT.json"
SIDECAR_EBPW = "receipts/future/EBPW_CATEGORY_VALIDATOR.json"
SIDECAR_NX_AUDIT = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
SIDECAR_ROUTER = "receipts/future/ROUTER_SENSITIVE_ALLOCATION.json"
SIDECAR_NGRAM = "receipts/future/NGRAM_SCHOOL.json"
SIDECAR_EXPERT = "receipts/future/EXPERT_BANK_SCHOOL.json"

# Label that can never be mistaken for a GPU capture. validate_corpus must
# still refuse this fixture; if it ever accepts it, the fixture is the bug.
STRUCTURAL_FIXTURE_KIND = "STRUCTURAL_NOT_REAL_CAPTURE"
STRUCTURAL_PAYLOAD_PREFIX = "STRUCTURAL_FIXTURE_NOT_A_CAPTURE"
STRUCTURAL_AUTHORITY = "STRUCTURAL_FIXTURE"  # not in teacher_corpus.AUTHORITIES
STRUCTURAL_SPECIMEN = {
    "model": "fixture/meta-ready-structural",
    "pinned_revision": "0" * 40,
    "seal_sha256": hashlib.sha256(b"fixture/meta-ready-structural").hexdigest(),
    "seal_kind": "structural_fixture_identity",
    "source": "tools/future/meta_ready.py::STRUCTURAL_SPECIMEN",
}
STRUCTURAL_N_ROWS = 4  # schema sample; never 256 fabricated training rows

# Funnel input keys, in gate order. Gate N consumes gate N-1's output.
FUNNEL_INPUT_ORDER = tuple(g.required_input for g in GATES)


# ---------------------------------------------------------------------------
# Evidence loading. Sparse checkout is not absence. Prefer pinned snapshot.
# ---------------------------------------------------------------------------


def _git_common_parent() -> Path | None:
    common = git("rev-parse", "--git-common-dir")
    if not common:
        return None
    p = Path(common)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    else:
        p = p.resolve()
    parent = p.parent if p.name == ".git" else p
    return parent if parent.is_dir() else None


def resolve_evidence(live_rel: str, *, prefer_pinned: bool = True) -> dict[str, Any]:
    """Locate a named receipt. Records which path was taken; never claims absence as proof.

    Preference: pinned snapshot (stable) then live this worktree, then the
    primary worktree via git-common-dir (untracked Codex receipts), then HEAD.
    Capture-boundary current state should pass prefer_pinned=False.
    """
    name = Path(live_rel).name
    pinned = PINNED_DIR / name
    here = REPO / live_rel
    searched = [
        str(pinned.relative_to(REPO)) if pinned.exists() or True else str(pinned),
        live_rel,
    ]

    if prefer_pinned and pinned.is_file():
        return {
            "rel": live_rel,
            "present": True,
            "evidence_source": "pinned_snapshot",
            "path": str(pinned.relative_to(REPO)),
            "resolved": str(pinned),
            "doc": load_json(pinned),
            "in_this_worktree": True,
        }

    if here.is_file():
        return {
            "rel": live_rel,
            "present": True,
            "evidence_source": "live_headless",
            "path": live_rel,
            "resolved": str(here),
            "doc": load_json(here),
            "in_this_worktree": True,
        }

    via_audit = evidence_path(live_rel)
    if via_audit is not None and Path(via_audit).is_file():
        resolved = Path(via_audit).resolve()
        in_tree = resolved == (REPO / live_rel).resolve()
        return {
            "rel": live_rel,
            "present": True,
            "evidence_source": "live_headless",
            "path": live_rel,
            "resolved": str(resolved),
            "doc": load_json(resolved),
            "in_this_worktree": in_tree,
        }

    primary = _git_common_parent()
    if primary is not None:
        alt = primary / live_rel
        searched.append(str(alt))
        if alt.is_file():
            return {
                "rel": live_rel,
                "present": True,
                "evidence_source": "live_headless",
                "path": live_rel,
                "resolved": str(alt),
                "doc": load_json(alt),
                "in_this_worktree": False,
            }

    raw = git("show", f"HEAD:{live_rel}")
    if raw:
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            doc = None
        if doc is not None:
            return {
                "rel": live_rel,
                "present": True,
                "evidence_source": "live_headless",
                "path": live_rel,
                "resolved": f"HEAD:{live_rel}",
                "doc": doc,
                "in_this_worktree": False,
            }

    return {
        "rel": live_rel,
        "present": False,
        "evidence_source": "live_headless" if not prefer_pinned else "pinned_snapshot",
        "path": live_rel,
        "resolved": None,
        "doc": None,
        "in_this_worktree": False,
        "lookup_coped": True,
        "searched": searched,
        "note": (
            "Not visible from this sparse worktree. A missing file here is not "
            "evidence it is absent from the civilization. Downstream still "
            "declares the Metal-GPU capture blocker."
        ),
    }


def sidecar_receipt(rel: str) -> dict[str, Any] | None:
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _path_state(rel: str) -> dict[str, Any]:
    """Record whether a path is visible. Never used as a negative proof."""
    here = (REPO / rel).is_file()
    return {"path": rel, "on_disk_in_this_worktree": here}


# ---------------------------------------------------------------------------
# Capture boundary + family budgets (derived, not hard-coded counts)
# ---------------------------------------------------------------------------


def lookup_capture_boundary() -> dict[str, Any]:
    """Current capture state. Live headless: this is the thing that moves."""
    hit = resolve_evidence(REL_CAPTURE_BOUNDARY, prefer_pinned=False)
    doc = hit.get("doc") if isinstance(hit.get("doc"), dict) else None
    out = {
        "lookup_copes": True,
        "present": bool(hit.get("present") and doc is not None),
        "evidence_source": hit.get("evidence_source"),
        "path": hit.get("path"),
        "resolved": hit.get("resolved"),
        "in_this_worktree": hit.get("in_this_worktree"),
        "status": None,
        "requested_rows": None,
        "minimum_rows": None,
        "teacher_rows_written": None,
        "source_authority_capture": None,
        "failure_stage": None,
        "failure_error": None,
        "promotion_allowed": None,
        "model": None,
        "pinned_revision": None,
        "claim_boundary": None,
        "schema": None,
    }
    if doc is None:
        out["note"] = hit.get("note")
        out["blocker_even_if_receipt_unseen"] = (
            "Sidecar has no Metal-capable GPU and must not seize one. Capture "
            "cannot start in this lane regardless of whether the Codex boundary "
            "receipt is materialized in this worktree."
        )
        return out
    failure = doc.get("failure") if isinstance(doc.get("failure"), dict) else {}
    out.update(
        {
            "status": doc.get("status"),
            "requested_rows": doc.get("requested_rows"),
            "minimum_rows": doc.get("minimum_rows"),
            "teacher_rows_written": doc.get("teacher_rows_written"),
            "source_authority_capture": doc.get("source_authority_capture"),
            "failure_stage": failure.get("stage"),
            "failure_error": failure.get("error"),
            "promotion_allowed": doc.get("promotion_allowed"),
            "model": doc.get("model"),
            "pinned_revision": doc.get("pinned_revision"),
            "claim_boundary": doc.get("claim_boundary"),
            "schema": doc.get("schema"),
        }
    )
    return out


def contract_min_rows(boundary: Mapping[str, Any] | None = None) -> int:
    """Admission floor. Derived from the capture boundary when present."""
    b = boundary if isinstance(boundary, dict) else lookup_capture_boundary()
    for key in ("minimum_rows", "requested_rows"):
        v = b.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return int(BOUNDED_TARGET_ROWS)


def load_family_budgets() -> dict[str, Any]:
    hit = resolve_evidence(REL_META_SUB1, prefer_pinned=True)
    doc = hit.get("doc") if isinstance(hit.get("doc"), dict) else None
    families: list[dict[str, Any]] = []
    if doc is not None:
        raw = doc.get("family_budget")
        if isinstance(raw, list):
            for spec in raw:
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("family") or spec.get("id") or "").strip()
                if not name:
                    continue
                frac = spec.get("source_fraction")
                families.append(
                    {
                        "family": name,
                        "program": spec.get("program"),
                        "source_fraction": frac if isinstance(frac, (int, float)) else None,
                        "meta_bpw_target": spec.get("meta_bpw_target"),
                        "ledger_components": (
                            sorted((spec.get("ledger") or {}).keys())
                            if isinstance(spec.get("ledger"), dict)
                            else []
                        ),
                        "runtime_shape": spec.get("runtime_shape"),
                    }
                )
        families.sort(
            key=lambda r: (
                -(r["source_fraction"] if isinstance(r["source_fraction"], (int, float)) else -1.0),
                r["family"],
            )
        )
    coherence = None
    next_gate = None
    measurement_state = None
    if doc is not None:
        coherence = doc.get("coherence_contract")
        next_gate = doc.get("next_gate")
        measurement_state = doc.get("measurement_state")
    return {
        "evidence_source": hit.get("evidence_source"),
        "present": bool(hit.get("present") and doc is not None),
        "path": hit.get("path"),
        "n_families": len(families),
        "families": families,
        "coherence_contract": coherence if isinstance(coherence, dict) else None,
        "next_gate": next_gate,
        "measurement_state": measurement_state if isinstance(measurement_state, dict) else None,
        "status": (doc or {}).get("status") if doc else None,
        "model": (doc or {}).get("model") if doc else None,
    }


def load_coherence_screen() -> dict[str, Any]:
    hit = resolve_evidence(REL_COHERENCE, prefer_pinned=True)
    doc = hit.get("doc") if isinstance(hit.get("doc"), dict) else None
    trace = (doc or {}).get("teacher_trace") if isinstance(doc, dict) else None
    contract = (doc or {}).get("coherence_contract") if isinstance(doc, dict) else None
    meas = (doc or {}).get("measurement_state") if isinstance(doc, dict) else None
    return {
        "evidence_source": hit.get("evidence_source"),
        "present": bool(hit.get("present") and doc is not None),
        "path": hit.get("path"),
        "status": (doc or {}).get("status") if doc else None,
        "min_rows_required": (trace or {}).get("min_rows_required") if isinstance(trace, dict) else None,
        "unsafe_small_probe": (trace or {}).get("unsafe_small_probe") if isinstance(trace, dict) else None,
        "teacher_trace_rows": (trace or {}).get("rows") if isinstance(trace, dict) else None,
        "coherence_contract": contract if isinstance(contract, dict) else None,
        "measurement_state": meas if isinstance(meas, dict) else None,
        "next_gate": (doc or {}).get("next_gate") if doc else None,
        "claim_boundary": (doc or {}).get("claim_boundary") if doc else None,
    }


# ---------------------------------------------------------------------------
# Per-stage readiness dossier (gates 3–9)
# ---------------------------------------------------------------------------


def _null_from_coherence(screen: Mapping[str, Any], budgets: Mapping[str, Any]) -> dict[str, Any]:
    """Stated nulls recovered from pinned coherence / meta-sub1 contracts."""
    cc = screen.get("coherence_contract") if isinstance(screen.get("coherence_contract"), dict) else {}
    sub1 = budgets.get("coherence_contract") if isinstance(budgets.get("coherence_contract"), dict) else {}
    router = sub1.get("router") if isinstance(sub1.get("router"), dict) else {}
    gen = sub1.get("generation") if isinstance(sub1.get("generation"), dict) else {}
    distill = sub1.get("teacher_distillation") if isinstance(sub1.get("teacher_distillation"), dict) else {}
    return {
        "held_out_numerical": {
            "min_heldout_cosine": cc.get("min_heldout_cosine"),
            "max_heldout_relative_fro_error": cc.get("max_heldout_relative_fro_error"),
            "must_beat_per_expert_q4": cc.get("must_beat_per_expert_q4"),
            "fit_and_holdout_split": distill.get("fit_and_holdout_split"),
            "source": "FLASH_META_COHERENCE_SCREEN_L4.coherence_contract + SUB1.teacher_distillation",
        },
        "route_stability": {
            "topk_membership_match": router.get("topk_membership_match"),
            "topk_order_match": router.get("topk_order_match"),
            "kill_on": "status MISMATCH or expert_ids_exact_match is false",
            "source": "FLASH_META_REPRESENTATION_SUB1.coherence_contract.router + funnel gate 4",
        },
        "logit_token": {
            "short_horizon_token_agreement": gen.get("short_horizon_token_agreement"),
            "long_horizon_no_collapse": gen.get("long_horizon_no_collapse"),
            "kill_on": "argmax disagreement or decode degeneration",
            "source": "SUB1.coherence_contract.generation + funnel gate 5",
        },
        "bounded_capability": {
            "capability_suite": gen.get("capability_suite"),
            "kill_on": "failure on any incumbent substantive axis; vacuous silence is not a pass",
            "source": "SUB1.coherence_contract.generation + funnel gate 6",
        },
    }


def _funnel_input_shape(gate_id: int) -> dict[str, Any]:
    """Exact dict the funnel evaluator already accepts. Not a new schema."""
    shapes = {
        2: {
            "required_input": "teacher_corpus",
            "accepted_forms": [
                {"fit_passed": True, "status": "PASSED"},
                {"passed": True, "status": "FIT_PASSED"},
                {"status": "PASSED"},
            ],
            "kill_forms": [
                {"fit_passed": False, "mechanism": "<null that failed>"},
                {"status": "FAILED"},
            ],
            "presence_without_fit": (
                "A PRESENT corpus envelope without fit_passed/status in "
                "MEASURED_PASS is REFUSED as unmeasured — presence is not a fit."
            ),
        },
        3: {
            "required_input": "held_out_numerical",
            "accepted_forms": [{"passed": True, "status": "PASSED"}],
            "kill_forms": [{"passed": False, "status": "FAILED", "mechanism": "<null>"}],
        },
        4: {
            "required_input": "route_traces",
            "accepted_forms": [
                {"status": "PASSED", "expert_ids_exact_match": True},
            ],
            "kill_forms": [
                {"status": "MISMATCH", "expert_ids_exact_match": False},
            ],
        },
        5: {
            "required_input": "logit_token",
            "accepted_forms": [{"status": "PASSED", "argmax_agree": True}],
            "kill_forms": [{"status": "FAILED", "argmax_agree": False}],
        },
        6: {
            "required_input": "bounded_capability",
            "accepted_forms": [{"status": "PASSED", "passed": True}],
            "kill_forms": [{"status": "FAILED", "passed": False}],
        },
        7: {
            "required_input": "physical_nr",
            "accepted_forms": [{"status": "PASSED"}],
            "kill_forms": [{"status": "FAILED"}],
            "refusal_tokens": ["NOT_BUILT", "PLAN_ONLY", "NOT_IMPLEMENTED"],
        },
        8: {
            "required_input": "complete_nx",
            "accepted_forms": [{"status": "PASSED"}],
            "kill_forms": [{"status": "FAILED"}],
            "refusal_tokens": [
                "NOT_BUILT",
                "SCAFFOLD_ONLY",
                "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            ],
        },
        9: {
            "required_input": "ebpw_ledger",
            "accepted_forms": [
                {
                    "status": "PASSED",
                    "all_required_bytes_included": True,
                    "complete_system_ebpw": "<counted, not a target>",
                }
            ],
            "kill_forms": [{"status": "FAILED", "all_required_bytes_included": False}],
            "refusal_tokens": ["NOT_MEASURED", "null complete_system_bytes"],
        },
    }
    return shapes[gate_id]


def stage_dossiers(
    *,
    budgets: Mapping[str, Any] | None = None,
    screen: Mapping[str, Any] | None = None,
    boundary: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-gate 3–9 dossier. Gate 2 is the corpus; everything after consumes the predecessor."""
    budgets = budgets or load_family_budgets()
    screen = screen or load_coherence_screen()
    boundary = boundary or lookup_capture_boundary()
    nulls = _null_from_coherence(screen, budgets)

    nr_hit = resolve_evidence(REL_NR_V2, prefer_pinned=True)
    nx_hit = resolve_evidence(REL_NX_V0, prefer_pinned=True)
    sel_hit = resolve_evidence(REL_ROUTER_SEL, prefer_pinned=True)
    map_hit = resolve_evidence(REL_ROUTER_MAP, prefer_pinned=True)
    nr_doc = nr_hit.get("doc") if isinstance(nr_hit.get("doc"), dict) else {}
    nx_doc = nx_hit.get("doc") if isinstance(nx_hit.get("doc"), dict) else {}
    sel_doc = sel_hit.get("doc") if isinstance(sel_hit.get("doc"), dict) else {}
    sel_parity = sel_doc.get("source_selection_parity") if isinstance(sel_doc.get("source_selection_parity"), dict) else {}

    funnel_mod = "python3 tools/future/meta_funnel.py --build"
    nx_audit = "python3 tools/future/flash_nx_audit.py --audit"
    nx_check = (
        "python3 tools/future/flash_nx_audit.py --check-nx "
        "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json"
    )
    ebpw_cmd = (
        "python3 tools/future/ebpw_categories.py --validate "
        "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json"
    )
    router_cmd = "python3 tools/future/router_science.py --build"
    ngram_cmd = "python3 tools/future/ngram_school.py --build"
    expert_cmd = "python3 tools/future/expert_bank_school.py --build"
    admit_cmd = (
        "python3 tools/future/meta_ready.py --simulate-arrival --corpus <admitted.json>"
    )

    specs: list[dict[str, Any]] = [
        {
            "gate_id": 3,
            "exact_inputs": [
                "predecessor: gate 2 teacher_fit artifact (fit_passed on the admitted corpus)",
                "held-out partition of the SAME admitted corpus (disjoint rows; not a new capture)",
            ],
            "null_it_tests": nulls["held_out_numerical"],
            "command_that_would_run_it": funnel_mod,
            "already_built": [
                "tools/future/meta_funnel.py gate 3 evaluator (_eval_heldout)",
                "FLASH_META_COHERENCE_SCREEN_L4.coherence_contract thresholds (pinned)",
                "SUB1.coherence_contract.teacher_distillation.fit_and_holdout_split",
            ],
            "still_missing": [
                "A CPU held-out operator that writes inputs.held_out_numerical from the admitted split",
                "The 4-row L4 screen is UNSAFE_SMALL_PROBE_NOT_PROMOTABLE and is not this gate",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "Contract thresholds can be read now. The measurement cannot."
            ),
        },
        {
            "gate_id": 4,
            "exact_inputs": [
                "predecessor: gate 3 held_out_numerical PASSED",
                "per-row route_ids already required on the admitted corpus",
                "student top-k on those same rows",
            ],
            "null_it_tests": nulls["route_stability"],
            "command_that_would_run_it": funnel_mod,
            "supporting_commands": [router_cmd],
            "already_built": [
                "tools/future/meta_funnel.py gate 4 evaluator (_eval_routes); kills on expert_ids_exact_match is false",
                "tools/future/router_science.py precision allocation (STATIC_ONLY; not a gate pass)",
                "FLASH_NOETIC_ROUTER_SELECTION.source_selection_parity (bounded in-memory study)",
                "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4 (DIAGNOSTIC seam map; does not skip gate 2)",
            ],
            "still_missing": [
                "Teacher-corpus-bound route traces at >= contract_min_rows; the bounded router AB is not that",
            ],
            "cited_existing_mismatch": {
                "path": sel_hit.get("path"),
                "evidence_source": sel_hit.get("evidence_source"),
                "status": sel_parity.get("status"),
                "expert_ids_exact_match": sel_parity.get("expert_ids_exact_match"),
                "note": (
                    "Cited MISMATCH cannot be reached as a funnel kill until gates 2 and 3 "
                    "pass. Funnel.advance refuses to skip."
                ),
            },
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "router_science allocation and the L3/L4 sensitivity map can be read now. "
                "They do not pass gate 4 and must not be used to skip teacher fit."
            ),
        },
        {
            "gate_id": 5,
            "exact_inputs": [
                "predecessor: gate 4 route_traces PASSED",
                "student complete-token / argmax on the admitted probe rows",
            ],
            "null_it_tests": nulls["logit_token"],
            "command_that_would_run_it": funnel_mod,
            "already_built": [
                "tools/future/meta_funnel.py gate 5 evaluator (_eval_logits); kills on argmax_agree is false",
                "SUB1.coherence_contract.generation.short_horizon_token_agreement",
            ],
            "still_missing": [
                "A CPU logit/token operator that writes inputs.logit_token from the admitted corpus",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": "Null is declared. The measurement cannot run.",
        },
        {
            "gate_id": 6,
            "exact_inputs": [
                "predecessor: gate 5 logit_token PASSED",
                "bounded capability suite axes against the incumbent (domains already on the corpus)",
            ],
            "null_it_tests": nulls["bounded_capability"],
            "command_that_would_run_it": funnel_mod,
            "supporting_commands": [ngram_cmd, expert_cmd],
            "already_built": [
                "tools/future/meta_funnel.py gate 6 evaluator (_eval_capability)",
                "tools/future/ngram_school.py five-axis analytical candidates",
                "tools/future/expert_bank_school.py generator candidates + cheapest falsifiers",
                "teacher_corpus capability_domain diversity (math/code/prose/tool/shell)",
            ],
            "still_missing": [
                "A Flash-meta bounded capability suite runner that writes inputs.bounded_capability",
                "Incumbent-axis list as an executable suite (SUB1 names the requirement, not the items)",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "ngram_school and expert_bank_school can emit candidates now. They do not pass gate 6."
            ),
        },
        {
            "gate_id": 7,
            "exact_inputs": [
                "predecessor: gate 6 bounded_capability PASSED",
                "a Physical NR lowering of THAT fitted representation (not the exact-control NR)",
            ],
            "null_it_tests": {
                "kill_on": GATES_BY_ID[7].kill_criterion,
                "existing_nr_is_not_this_gate": (
                    f"FLASH_COMPLETE_V2.nr.json status="
                    f"{nr_doc.get('status')!r} is an exact-control heterogeneous candidate "
                    "NOT_FOR_PROMOTION. Auditing it is not lowering a fitted meta program."
                ),
            },
            "command_that_would_run_it": nx_audit,
            "already_built": [
                "tools/future/flash_nx_audit.py (STATIC completeness audit)",
                PINNED_NR_V2 if nr_hit.get("present") else REL_NR_V2,
                "tools/future/meta_funnel.py gate 7 evaluator (_eval_nr)",
            ],
            "still_missing": [
                "NR lowering of a gate-2..6 survivor; the existing V2 NR is a different artifact",
                "This sidecar will not run a kernel (STATIC_ONLY)",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "flash_nx_audit of the existing exact-control NR can run now. "
                "Funnel gate 7 cannot PASS without the predecessor chain."
            ),
            "nr_status_cited": {
                "path": nr_hit.get("path"),
                "evidence_source": nr_hit.get("evidence_source"),
                "status": nr_doc.get("status"),
                "promotion_allowed": (nr_doc.get("promotion") or {}).get("allowed")
                if isinstance(nr_doc.get("promotion"), dict)
                else None,
            },
        },
        {
            "gate_id": 8,
            "exact_inputs": [
                "predecessor: gate 7 physical_nr PASSED",
                "a source-independent complete NX that is more than sealed metadata",
            ],
            "null_it_tests": {
                "kill_on": GATES_BY_ID[8].kill_criterion,
                "refusal_if": "SCAFFOLD_ONLY / SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            },
            "command_that_would_run_it": nx_check,
            "supporting_commands": [nx_audit],
            "already_built": [
                "tools/future/flash_nx_audit.py seven-requirement checker",
                PINNED_NX_V0 if nx_hit.get("present") else REL_NX_V0,
                "tools/future/meta_funnel.py gate 8 evaluator (_eval_nx)",
            ],
            "still_missing": [
                "A source-independent complete NX (current V0 is SEALED_METADATA_ONLY_NOT_FOR_PROMOTION)",
                "native loader + native kernel catalog + protected complete-token measurement",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "Completeness audit of the sealed NX can run now and currently reports NOT_MET. "
                "That is not a funnel pass."
            ),
            "nx_status_cited": {
                "path": nx_hit.get("path"),
                "evidence_source": nx_hit.get("evidence_source"),
                "status": nx_doc.get("status"),
            },
        },
        {
            "gate_id": 9,
            "exact_inputs": [
                "predecessor: gate 8 complete_nx PASSED",
                "complete-system EBPW ledger with every required executable-information field counted",
            ],
            "null_it_tests": {
                "kill_on": GATES_BY_ID[9].kill_criterion,
                "law": "A target ceiling is not a measurement. prospective_meta_bpw < 1 never promotes.",
            },
            "command_that_would_run_it": ebpw_cmd,
            "already_built": [
                "tools/future/ebpw_categories.py five-quantity type system + can_promote refusal",
                "tools/future/meta_funnel.py gate 9 evaluator (_eval_ebpw)",
                "FLASH_META_REPRESENTATION_SUB1 prospective meta budget (description, not physical EBPW)",
            ],
            "still_missing": [
                "complete_system_bytes of a fitted, capability-preserving executable (currently null / NOT_MEASURED)",
                "PROTECTED_ABSOLUTE complete_physical_ebpw — this sidecar cannot take it",
            ],
            "can_proceed_without_corpus": False,
            "analytical_screen_without_corpus": (
                "Category validation of the prospective budget can run now and must stay GREEN "
                "on honest SUB1. It cannot pass gate 9."
            ),
        },
    ]

    dossiers: list[dict[str, Any]] = []
    for spec in specs:
        gid = spec["gate_id"]
        gate = GATES_BY_ID[gid]
        pred = GATES_BY_ID[gid - 1]
        row = {
            "gate_id": gid,
            "gate_name": gate.name,
            "cost_class": gate.cost_class,
            "required_input": gate.required_input,
            "predecessor_gate_id": pred.id,
            "predecessor_gate_name": pred.name,
            "predecessor_output": pred.required_input,
            "needs_nothing_but_predecessor_output": True,
            "kill_criterion": gate.kill_criterion,
            "passing_proves": gate.passing_proves,
            "passing_does_not_prove": gate.passing_does_not_prove,
            "funnel_input_shape": _funnel_input_shape(gid),
            "wired_in_funnel": True,
            "command_when_predecessor_present": spec["command_that_would_run_it"],
            "admit_then_funnel": admit_cmd,
            **{k: spec[k] for k in spec if k != "gate_id"},
        }
        # Map-hit is used only to record whether the diagnostic map was visible.
        if gid == 4:
            row["router_map_cited"] = {
                "path": map_hit.get("path"),
                "evidence_source": map_hit.get("evidence_source"),
                "present": map_hit.get("present"),
            }
        dossiers.append(row)
    return dossiers


def pipeline_wiring(dossiers: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The pipeline is WIRED iff each gate 3–9 consumes only its predecessor's output."""
    dossiers = list(dossiers or stage_dossiers())
    edges = []
    unwired = []
    for row in dossiers:
        pred_id = row["predecessor_gate_id"]
        pred_input = GATES_BY_ID[pred_id].required_input
        this_input = row["required_input"]
        ok = (
            row.get("wired_in_funnel") is True
            and row.get("needs_nothing_but_predecessor_output") is True
            and bool(row.get("command_when_predecessor_present") or row.get("command_that_would_run_it"))
            and pred_input in FUNNEL_INPUT_ORDER
            and this_input in FUNNEL_INPUT_ORDER
            and FUNNEL_INPUT_ORDER.index(this_input) == FUNNEL_INPUT_ORDER.index(pred_input) + 1
        )
        edge = {
            "from_gate": pred_id,
            "from_output": pred_input,
            "to_gate": row["gate_id"],
            "to_input": this_input,
            "wired": ok,
            "command": row.get("command_when_predecessor_present")
            or row.get("command_that_would_run_it"),
        }
        edges.append(edge)
        if not ok:
            unwired.append(edge)
    gate2 = GATES_BY_ID[2]
    return {
        "wired": not unwired,
        "gate_2_needs_only_corpus": True,
        "gate_2_required_input": gate2.required_input,
        "gates_3_to_9_need_nothing_but_predecessor": all(
            r.get("needs_nothing_but_predecessor_output") is True for r in dossiers
        ),
        "edges": edges,
        "unwired": unwired,
        "funnel_enforces_no_skip": True,
        "n_downstream_gates": len(dossiers),
    }


# ---------------------------------------------------------------------------
# Corpus arrival contract + one-call admission
# ---------------------------------------------------------------------------


def corpus_arrival_contract(boundary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Precise shape a real capture must have to satisfy gate 2."""
    boundary = boundary or lookup_capture_boundary()
    min_rows = contract_min_rows(boundary)
    return {
        "one_call": "tools.future.meta_ready.admit_capture(rows, envelope=...)",
        "validator": "tools.future.teacher_corpus.validate_corpus",
        "min_rows": min_rows,
        "min_rows_source": (
            "capture_boundary.minimum_rows"
            if isinstance(boundary.get("minimum_rows"), int)
            else "teacher_corpus.BOUNDED_TARGET_ROWS"
        ),
        "specimen_binding": {
            "model": FLASH_SPECIMEN["model"],
            "pinned_revision": FLASH_SPECIMEN["pinned_revision"],
            "seal_sha256": FLASH_SPECIMEN["seal_sha256"],
            "seal_kind": FLASH_SPECIMEN.get("seal_kind"),
            "source": FLASH_SPECIMEN.get("source"),
            "rule": "Every row.specimen must match this Flash identity exactly.",
        },
        "layer_surface_binding": {
            "required_per_row": ["layer", "surface"],
            "surfaces": list(SURFACES),
            "routed_surfaces": sorted(ROUTED_SURFACES),
            "note": (
                "This capture attempt is FLASH_META_TEACHER_L4 (route layer 4, "
                "layer_input 3 on the pinned coherence screen). A future multi-layer "
                "capture is admissible if every row still binds layer+surface and "
                "validate_corpus accepts. Layer 4 is the current attempt, not a cap."
            ),
        },
        "per_row_route_ids": {
            "required": True,
            "rule": "route_ids is a list of int; routed surfaces need >= min_routes unique ids",
            "min_routes_on_routed_surface": MIN_ROUTES_ROUTED,
        },
        "diversity_thresholds": {
            name: {
                "threshold": spec["threshold"],
                "inadequate_below": spec["inadequate_below"],
                "formula": spec["formula"],
            }
            for name, spec in sorted(DIVERSITY_MEASURES.items())
        },
        "diversity_constants": {
            "unique_ratio_min": UNIQUE_RATIO_MIN,
            "unique_ratio_floor": UNIQUE_RATIO_FLOOR,
            "natural_dup_rate": NATURAL_DUP_RATE,
            "min_prompts": MIN_PROMPTS_FOR_FIT,
            "min_domains": MIN_DOMAINS_FOR_FIT,
            "min_positions": MIN_POSITIONS,
            "position_max_share": POSITION_MAX_SHARE,
            "min_routes_routed": MIN_ROUTES_ROUTED,
            "capability_domains": list(CAPABILITY_DOMAINS),
        },
        "source_authority_capture": {
            "required": True,
            "current": boundary.get("source_authority_capture"),
            "rule": (
                "Envelope source_authority_capture must be true. The current "
                "boundary records false because capture did not start."
            ),
        },
        "provenance": {
            "kind_must_be": "captured",
            "kind_must_not_be": sorted(PROVENANCE_PADDING_KINDS | {"structural_fixture"}),
            "authority_must_be_one_of": list(AUTHORITIES),
            "structural_fixture_authority_is_not_one_of_those": STRUCTURAL_AUTHORITY,
        },
        "anti_fabrication": {
            "sacred_guard": "THRESHOLD_MET_ONLY_BY_DUPLICATION",
            "loud_exception": "CorpusRefused",
            "rule": (
                "A corpus that only meets min_rows by copying/resampling/synthesis "
                "is REFUSED. This module does not catch CorpusRefused to convert a "
                "FAIL into a PASS. A structural fixture that would pass the guard "
                "is a bug in the fixture."
            ),
        },
        "held_out_split": {
            "required": True,
            "rule": (
                "The 256 rows must be partitionable into fit vs held-out. The pinned "
                "L4 screen used 3/1 on 4 rows and is UNSAFE_SMALL_PROBE. That split "
                "is not this contract."
            ),
        },
        "does_not_satisfy": [
            "Weight reconstruction on a bounded slice",
            "4-row L4 coherence screen",
            "STATIC_ONLY teacher_corpus.make_diverse_corpus fixtures",
            "Anything this module's --simulate-arrival can produce",
        ],
    }


def _is_structural_row(row: Mapping[str, Any]) -> bool:
    spec = row.get("specimen") or {}
    prov = row.get("provenance") or {}
    model = str(spec.get("model") or "")
    kind = str(prov.get("kind") or "")
    authority = str(prov.get("authority") or "")
    payload = str(row.get("payload") or "")
    source_path = str(prov.get("source_path") or "")
    note = str(prov.get("note") or "")
    if model.startswith("fixture/"):
        return True
    if kind in {"structural_fixture", "synthesised", "duplicated", "resampled"}:
        return True
    if authority == STRUCTURAL_AUTHORITY:
        return True
    if payload.startswith(STRUCTURAL_PAYLOAD_PREFIX):
        return True
    if "fixture" in source_path.lower() or "structural" in source_path.lower():
        return True
    if "not a GPU capture" in note or "STRUCTURAL" in note:
        return True
    if row.get("fixture_kind") == STRUCTURAL_FIXTURE_KIND:
        return True
    return False


def _specimen_mismatch(row: Mapping[str, Any]) -> list[str]:
    spec = row.get("specimen") or {}
    miss = []
    for key in ("model", "pinned_revision", "seal_sha256"):
        if spec.get(key) != FLASH_SPECIMEN[key]:
            miss.append(f"specimen.{key}")
    return miss


def admit_capture(
    rows: Sequence[Mapping[str, Any]],
    *,
    envelope: Mapping[str, Any] | None = None,
    min_rows: int | None = None,
) -> dict[str, Any]:
    """ONE call. A capture that arrives is admissible or rejected with a clear reason.

    Wraps teacher_corpus.validate_corpus and additionally refuses anything that
    is not a real Flash source-authority capture (structural fixtures, specimen
    mismatch, source_authority_capture is not true). Presence of 256 diverse
    STATIC_ONLY fixture rows is not admission.
    """
    envelope = dict(envelope or {})
    min_rows = int(min_rows) if min_rows is not None else contract_min_rows()
    annotated = annotate_corpus(rows)
    n = len(annotated)

    codes: list[str] = []
    details: dict[str, Any] = {}

    structural_rows = [r.get("row_id") for r in annotated if _is_structural_row(r)]
    if structural_rows or envelope.get("fixture_kind") == STRUCTURAL_FIXTURE_KIND:
        codes.append("STRUCTURAL_FIXTURE_NOT_REAL_CAPTURE")
        details["structural_row_ids"] = structural_rows[:12]
        details["envelope_fixture_kind"] = envelope.get("fixture_kind")

    if envelope.get("source_authority_capture") is not True:
        codes.append("SOURCE_AUTHORITY_CAPTURE_FALSE")
        details["source_authority_capture"] = envelope.get("source_authority_capture")

    mismatches = []
    for r in annotated:
        miss = _specimen_mismatch(r)
        if miss:
            mismatches.append({"row_id": r.get("row_id"), "fields": miss})
    if mismatches:
        codes.append("SPECIMEN_BINDING_MISMATCH")
        details["specimen_mismatches"] = mismatches[:12]

    routed = [r for r in annotated if r.get("surface") in ROUTED_SURFACES]
    missing_routes = [r.get("row_id") for r in routed if not (r.get("route_ids") or [])]
    if missing_routes:
        codes.append("ROUTE_IDS_MISSING_ON_ROUTED_SURFACE")
        details["missing_route_row_ids"] = missing_routes[:12]

    vc: dict[str, Any]
    try:
        vc = validate_corpus(annotated, min_rows=min_rows, raise_on_refuse=False)
    except ValueError as exc:
        vc = {
            "accepted": False,
            "refusals": ["VALIDATE_CORPUS_VALUE_ERROR"],
            "inadequacy": [],
            "n_rows": n,
            "n_unique_content": 0,
            "min_rows": min_rows,
            "details": {"error": str(exc)},
        }
    if vc.get("refusals"):
        for c in vc["refusals"]:
            if c not in codes:
                codes.append(c)
    if vc.get("inadequacy"):
        for c in vc["inadequacy"]:
            if c not in codes:
                codes.append(c)

    accepted = (
        not codes
        and bool(vc.get("accepted"))
        and envelope.get("source_authority_capture") is True
        and not structural_rows
        and not mismatches
    )
    if accepted:
        reason = "ADMITTED"
    elif codes:
        reason = codes[0]
    else:
        reason = "NOT_ADMITTED"

    return {
        "accepted": accepted,
        "reason": reason,
        "codes": codes,
        "min_rows": min_rows,
        "n_rows": n,
        "n_unique_content": vc.get("n_unique_content"),
        "structural_fixture": bool(structural_rows) or envelope.get("fixture_kind") == STRUCTURAL_FIXTURE_KIND,
        "source_authority_capture": envelope.get("source_authority_capture"),
        "validate_corpus": {
            "accepted": vc.get("accepted"),
            "refusals": list(vc.get("refusals") or []),
            "inadequacy": list(vc.get("inadequacy") or []),
            "n_rows": vc.get("n_rows"),
            "n_unique_content": vc.get("n_unique_content"),
            "min_rows": vc.get("min_rows"),
        },
        "details": details,
        "claim_boundary": (
            "STATIC_ONLY admission. No GPU capture was performed here. "
            "Admission is a structural property of the rows plus the envelope."
        ),
    }


def make_structural_fixture(n: int = STRUCTURAL_N_ROWS) -> list[dict[str, Any]]:
    """Clearly-labelled STRUCTURAL fixture. Not a capture. Never 256 training rows.

    Schema-complete so wiring can inspect field names. Authority is
    STRUCTURAL_FIXTURE, which is not in teacher_corpus.AUTHORITIES, so
    validate_corpus REFUSES with MISSING_SPECIMEN_OR_PROVENANCE_BINDING.
    If this fixture ever starts passing that guard, the fixture is the bug.
    """
    if n > 16:
        raise ValueError(
            "structural fixture refuses to emit more than 16 rows; this path "
            "must not fabricate a training corpus"
        )
    prov = {
        "kind": "structural_fixture",
        "authority": STRUCTURAL_AUTHORITY,
        "source_path": "tools/future/meta_ready.py::make_structural_fixture",
        "source_sha256": hashlib.sha256(b"meta-ready-structural-fixture-v1").hexdigest(),
        "capture_tool": "none",
        "note": "STRUCTURAL FIXTURE; not a GPU capture and not a promotion",
    }
    rows: list[dict[str, Any]] = []
    for i in range(n):
        domain = CAPABILITY_DOMAINS[i % len(CAPABILITY_DOMAINS)]
        rows.append(
            make_row(
                row_id=f"structural-{i:04d}",
                specimen=STRUCTURAL_SPECIMEN,
                layer=i % 2,
                surface="routed_expert",
                prompt_id=f"structural-prompt-{i}",
                prompt_text=f"STRUCTURAL_FIXTURE prompt {i}",
                token_position=i,
                route_ids=[i, i + 1],
                capability_domain=domain,
                payload=f"{STRUCTURAL_PAYLOAD_PREFIX}|i={i}",
                provenance=prov,
            )
        )
        rows[-1]["fixture_kind"] = STRUCTURAL_FIXTURE_KIND
    return annotate_corpus(rows)


def load_caller_corpus(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a caller-supplied corpus. Does not generate rows."""
    p = Path(path)
    doc = load_json(p)
    envelope: dict[str, Any] = {}
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        rows = doc.get("rows") or doc.get("corpus") or []
        envelope = {
            k: doc.get(k)
            for k in (
                "source_authority_capture",
                "fixture_kind",
                "status",
                "specimen",
                "min_rows",
            )
            if k in doc
        }
    else:
        rows = []
    if not isinstance(rows, list):
        rows = []
    return [r for r in rows if isinstance(r, dict)], envelope


def bind_corpus_to_families(
    families: Sequence[Mapping[str, Any]],
    admission: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Set teacher_corpus PRESENT only for an admitted real capture.

    Presence is not a fit: fit_passed is not set. Funnel gate 2 then REFUSES
    as unmeasured rather than NOT_BUILT. Structural fixtures never bind.
    """
    out = [dict(c) for c in families]
    if not admission.get("accepted"):
        return out, {
            "bound": False,
            "reason": admission.get("reason") or "NOT_ADMITTED",
            "gate2_advanced": False,
        }
    envelope = {
        "status": "PRESENT",
        "admitted": True,
        "n_rows": admission.get("n_rows"),
        "n_unique_content": admission.get("n_unique_content"),
        "note": (
            "Corpus admitted. Fit not yet measured. Funnel must REFUSE gate 2 "
            "as unmeasured until a teacher-fit operator writes fit_passed."
        ),
    }
    for cand in out:
        inputs = dict(cand.get("inputs") or {})
        inputs["teacher_corpus"] = dict(envelope)
        cand["inputs"] = inputs
    return out, {"bound": True, "reason": "ADMITTED_PRESENT_UNMEASURED", "gate2_advanced": False}


# ---------------------------------------------------------------------------
# --simulate-arrival: wiring only. Never fabricate training rows.
# ---------------------------------------------------------------------------


def simulate_arrival(
    *,
    corpus_path: str | Path | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    envelope: Mapping[str, Any] | None = None,
    dossiers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check that the pipeline is WIRED. Do not fabricate rows. Do not pass gate 2."""
    env = dict(envelope or {})
    if rows is not None:
        used_rows = annotate_corpus(rows)
        source = "caller_supplied_rows"
        structural = bool(used_rows) and all(_is_structural_row(r) for r in used_rows)
    elif corpus_path is not None:
        used_rows, file_env = load_caller_corpus(corpus_path)
        for k, v in file_env.items():
            env.setdefault(k, v)
        source = "caller_supplied_path"
        structural = False
    else:
        used_rows = make_structural_fixture()
        source = "structural_fixture"
        structural = True
        env.setdefault("fixture_kind", STRUCTURAL_FIXTURE_KIND)
        env.setdefault("source_authority_capture", False)

    admission = admit_capture(used_rows, envelope=env)
    wiring = pipeline_wiring(dossiers)

    families, provenance = recover_families()
    bound_families, bind_info = bind_corpus_to_families(families, admission)
    funnel = Funnel()
    # Always run the unbound recovered families: simulate must not claim a pass.
    recovered_runs = [funnel.run(dict(c)) for c in families]
    stall_gates = sorted({r.get("stall_gate") for r in recovered_runs})
    stall_at_2 = all(r.get("stall_gate") == 2 for r in recovered_runs)

    bound_runs = None
    bound_stall = None
    if bind_info.get("bound"):
        funnel2 = Funnel()
        bound_runs = [funnel2.run(c) for c in bound_families]
        bound_stall = sorted({r.get("stall_gate") for r in bound_runs})
        # Presence without fit_passed is still REFUSED (unmeasured), never PASSED.
        bound_passed_gate2 = any(2 in (r.get("passed_gates") or []) for r in bound_runs)
    else:
        bound_passed_gate2 = False

    # Direct negative control: validate_corpus on whatever we inspected.
    vc_refused = False
    vc_codes: list[str] = []
    try:
        validate_corpus(
            used_rows,
            min_rows=contract_min_rows(),
            raise_on_refuse=True,
        )
        vc_accepted_flag = True
    except CorpusRefused as exc:
        vc_refused = True
        vc_codes = list(exc.codes)
        vc_accepted_flag = False

    gate2_advanced = False  # invariant of this function
    return {
        "mode": "SIMULATE_ARRIVAL_WIRING_ONLY",
        "fabricated_training_rows": False,
        "row_source": source,
        "structural_fixture": structural,
        "n_rows_inspected": len(used_rows),
        "n_unique_content": len({r.get("content_sha256") for r in used_rows}),
        "admission": {
            "accepted": admission["accepted"],
            "reason": admission["reason"],
            "codes": admission["codes"],
            "validate_corpus": admission["validate_corpus"],
        },
        "validate_corpus_refused_fixture": vc_refused if structural else None,
        "validate_corpus_codes": vc_codes,
        "validate_corpus_accepted": vc_accepted_flag,
        "wiring": {
            "wired": wiring["wired"],
            "gate_2_needs_only_corpus": wiring["gate_2_needs_only_corpus"],
            "gates_3_to_9_need_nothing_but_predecessor": wiring[
                "gates_3_to_9_need_nothing_but_predecessor"
            ],
            "n_edges": len(wiring["edges"]),
            "unwired": wiring["unwired"],
        },
        "bind": bind_info,
        "gate2_advanced": gate2_advanced,
        "bound_gate2_passed": bound_passed_gate2,
        "recovered_families": {
            "n": len(recovered_runs),
            "stall_gates": stall_gates,
            "all_stall_at_gate_2": stall_at_2,
            "primary_source": provenance.get("primary_source"),
        },
        "bound_family_stall_gates": bound_stall,
        "invariant": (
            "simulate-arrival never writes training rows, never sets fit_passed, "
            "and never reports gate 2 PASSED. A structural fixture is REFUSED by "
            "validate_corpus and by admit_capture. An admitted caller corpus "
            "becomes PRESENT/unmeasured at most."
        ),
    }


# ---------------------------------------------------------------------------
# Blocking chain + ranked first experiments
# ---------------------------------------------------------------------------


def blocking_chain(
    *,
    boundary: Mapping[str, Any] | None = None,
    dossiers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    boundary = boundary or lookup_capture_boundary()
    dossiers = list(dossiers or stage_dossiers())
    status = boundary.get("status")
    without_corpus = []
    for row in dossiers:
        without_corpus.append(
            {
                "gate_id": row["gate_id"],
                "gate_name": row["gate_name"],
                "can_pass_funnel_without_corpus": False,
                "analytical_screen_without_corpus": row.get("analytical_screen_without_corpus"),
            }
        )
    return {
        "capture": {
            "blocker": (
                "Metal-capable GPU. This sidecar does not have one and must not "
                "seize Codex's protected GPU lease."
            ),
            "status": status,
            "failure_stage": boundary.get("failure_stage")
            or "dense_source_bf16_prefix_initialization",
            "failure_error": boundary.get("failure_error"),
            "requested_rows": boundary.get("requested_rows"),
            "minimum_rows": boundary.get("minimum_rows"),
            "teacher_rows_written": boundary.get("teacher_rows_written"),
            "source_authority_capture": boundary.get("source_authority_capture"),
            "promotion_allowed": boundary.get("promotion_allowed"),
            "evidence_source": boundary.get("evidence_source"),
            "present": boundary.get("present"),
            "lookup_copes": boundary.get("lookup_copes", True),
        },
        "gate_1_analytical": {
            "can_proceed_without_corpus": True,
            "already_passed": True,
            "note": (
                "meta_funnel recovered Flash families already PASS gate 1 from "
                "family_budget[].ledger allocations and stall at gate 2 REFUSED."
            ),
        },
        "gate_2_teacher_fit": {
            "can_proceed_without_corpus": False,
            "needs": "admitted real teacher corpus (256 unique bound rows)",
        },
        "downstream_funnel_passes": without_corpus,
        "analytical_screens_that_can_run_now": [
            {
                "what": "allocation / gate 1 re-run",
                "command": "python3 tools/future/meta_funnel.py --build",
            },
            {
                "what": "EBPW category type-check of the prospective budget",
                "command": (
                    "python3 tools/future/ebpw_categories.py --validate "
                    "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json"
                ),
            },
            {
                "what": "NX/NR completeness audit of sealed artifacts",
                "command": "python3 tools/future/flash_nx_audit.py --audit",
            },
            {
                "what": "router-sensitive allocation (diagnostic map; not a gate pass)",
                "command": "python3 tools/future/router_science.py --build",
            },
            {
                "what": "n-gram school analytical candidates",
                "command": "python3 tools/future/ngram_school.py --build",
            },
            {
                "what": "expert-bank school candidates + cheapest falsifiers",
                "command": "python3 tools/future/expert_bank_school.py --build",
            },
            {
                "what": "this readiness dossier / wiring check",
                "command": "python3 tools/future/meta_ready.py --simulate-arrival",
            },
        ],
        "genuinely_cannot_without_corpus": [
            "gate 2 real_teacher_fit",
            "gate 3 held-out numerical of a fitted candidate",
            "gate 4 route stability on teacher traces",
            "gate 5 logit/token identity",
            "gate 6 bounded capability of a student",
            "gate 7 NR lowering of a fitted meta representation",
            "gate 8 complete NX of that lowering",
            "gate 9 complete-system EBPW of that NX",
            "any promotion",
        ],
        "sidecar_gpu_authority": False,
        "measurement_class_produced": "STATIC_ONLY",
    }


def ranked_first_experiments(
    *,
    budgets: Mapping[str, Any] | None = None,
    boundary: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """When 256 rows land: first experiment by expected information per unit cost.

    Qualitative rank only. No invented bit-counts, no hardware numbers.
    Cost classes are the funnel's. Family order is derived from family_budget
    source_fraction (largest organ first = cheapest kill of the meta program).
    """
    budgets = budgets or load_family_budgets()
    boundary = boundary or lookup_capture_boundary()
    min_rows = contract_min_rows(boundary)
    families = list(budgets.get("families") or [])
    ranked: list[dict[str, Any]] = [
        {
            "rank": 1,
            "experiment": "admit_capture",
            "why": (
                f"One CPU call against the {min_rows}-row contract. If the rows "
                "fail validate_corpus or specimen/source-authority binding, every "
                "later gate is moot. Highest information per unit cost."
            ),
            "cost_class": "CHEAP_ANALYTICAL",
            "requires_corpus": True,
            "requires_gpu": False,
            "command": (
                "python3 tools/future/meta_ready.py --simulate-arrival "
                "--corpus <capture.json>"
            ),
            "kill_if": "any admit_capture code fires (fabrication, inadequacy, mismatch)",
        }
    ]
    rank = 2
    for fam in families:
        ranked.append(
            {
                "rank": rank,
                "experiment": f"gate_2_teacher_fit.{fam['family']}",
                "family": fam["family"],
                "why": (
                    "Teacher fit is the cheapest remaining kill of the meta program. "
                    f"Family {fam['family']!r} is next by cited source_fraction "
                    f"{fam.get('source_fraction')!r} from the pinned family_budget "
                    "(largest remaining organ first). A fail writes a funnel scar "
                    "and stops that shape."
                ),
                "cost_class": "REAL_TEACHER_CPU",
                "requires_corpus": True,
                "requires_gpu": False,
                "command": "python3 tools/future/meta_funnel.py --build",
                "cited_source_fraction": fam.get("source_fraction"),
                "program": fam.get("program"),
            }
        )
        rank += 1
    later = [
        (
            3,
            "held_out_numerical",
            "HELDOUT_NUMERICAL_CPU",
            "Held-out split of the same corpus; kills organ-local overfit the fit itself cannot see.",
        ),
        (
            4,
            "route_stability",
            "ROUTE_TRACE_CPU",
            "Top-k identity on the same rows. A bounded MISMATCH already exists on the router study; real traces are the cheap confirmation or scar.",
        ),
        (
            5,
            "logit_token_validation",
            "LOGIT_TOKEN_CPU",
            "Argmax/short decode. A cheaper kernel that flips the token is dead before capability spend.",
        ),
        (
            6,
            "bounded_capability",
            "BOUNDED_CAPABILITY_CPU",
            "Incumbent axes. Silence on vacuous tasks is not a pass. Costlier than numerical gates.",
        ),
        (
            7,
            "physical_nr_lowering",
            "PHYSICAL_NR_STATIC",
            "Only after a CPU survivor exists. Existing exact-control NR is not this experiment.",
        ),
        (
            8,
            "complete_nx",
            "COMPLETE_NX_STATIC",
            "Source-independent NX of the survivor. Sealed metadata is a refusal.",
        ),
        (
            9,
            "ebpw",
            "EBPW_ACCOUNTING_STATIC",
            "Complete-system bytes of that NX. A prospective meta-BPW < 1 is not this experiment.",
        ),
    ]
    for gid, name, cost, why in later:
        ranked.append(
            {
                "rank": rank,
                "experiment": f"gate_{gid}_{name}",
                "why": why,
                "cost_class": cost,
                "requires_corpus": True,
                "requires_gpu": gid >= 7,
                "gpu_note": (
                    "Sidecar still does not take the measurement; Codex does, after the CPU chain."
                    if gid >= 7
                    else None
                ),
                "command": (
                    "python3 tools/future/flash_nx_audit.py --audit"
                    if gid in {7, 8}
                    else (
                        "python3 tools/future/ebpw_categories.py --validate "
                        "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json"
                        if gid == 9
                        else "python3 tools/future/meta_funnel.py --build"
                    )
                ),
            }
        )
        rank += 1
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def recovered_implementation() -> dict[str, Any]:
    return {
        "meta_funnel": {
            "module": "tools/future/meta_funnel.py",
            "receipt": SIDECAR_FUNNEL,
            "what": "Nine ordered gates, advance() refusal, shape-keyed scars, family recovery from family_budget[].ledger",
            "adequate_for": "gate contracts, stall at gate 2, no-skip rule",
            "gap_this_lane_closes": "does not name per-stage exact inputs / commands / already-built vs missing, nor the arrival contract",
        },
        "teacher_corpus": {
            "module": "tools/future/teacher_corpus.py",
            "receipt": SIDECAR_CORPUS,
            "what": "manifest, five diversity measures, validate_corpus, THRESHOLD_MET_ONLY_BY_DUPLICATION",
            "adequate_for": "anti-fabrication; this guard is sacred and is not forked",
            "gap_this_lane_closes": "Flash specimen + source_authority_capture envelope around validate_corpus as one admit_capture call",
        },
        "ebpw_categories": {
            "module": "tools/future/ebpw_categories.py",
            "receipt": SIDECAR_EBPW,
            "what": "five non-interchangeable EBPW quantities; prospective_meta_bpw < 1 never promotes",
        },
        "flash_nx_audit": {
            "module": "tools/future/flash_nx_audit.py",
            "receipt": SIDECAR_NX_AUDIT,
            "what": "seven-requirement NX completeness; disk counts win over stale 12-of-14 wording",
        },
        "router_science": {
            "module": "tools/future/router_science.py",
            "receipt": SIDECAR_ROUTER,
            "what": "per-surface precision allocation from the L3/L4 sensitivity map; not a gate-4 pass",
        },
        "ngram_school": {
            "module": "tools/future/ngram_school.py",
            "receipt": SIDECAR_NGRAM,
            "what": "five-axis n-gram candidate generator; analytical, no specimen fit",
        },
        "expert_bank_school": {
            "module": "tools/future/expert_bank_school.py",
            "receipt": SIDECAR_EXPERT,
            "what": "structured expert storage/compute candidates + cheapest falsifiers; no weight fit",
        },
        "capture_boundary": {
            "path": REL_CAPTURE_BOUNDARY,
            "what": "Codex physical capture attempt: BLOCKED_NO_METAL_GPU, 0 of 256 rows",
        },
        "pinned_meta_sub1": {
            "path": PINNED_SUB1,
            "what": "nine family_budget entries, coherence_contract, next_gate",
        },
        "pinned_coherence_l4": {
            "path": PINNED_COHERENCE,
            "what": "UNSAFE_SMALL_PROBE_NOT_PROMOTABLE 4-row screen; min_rows_required 256",
        },
    }


def gaps_closed() -> list[str]:
    return [
        "Per-stage readiness dossier for funnel gates 3–9: exact inputs, stated null, kill criterion, command, already-built vs still-missing, and that each gate needs nothing but its predecessor's output.",
        "Corpus arrival contract: 256-row floor derived from the capture boundary, Flash specimen binding, layer/surface binding, per-row route ids, five diversity thresholds, source_authority_capture, one admit_capture call with a clear reason.",
        "--simulate-arrival checks wiring only. It does not fabricate training rows. The structural fixture is labelled STRUCTURAL_NOT_REAL_CAPTURE and is REFUSED by teacher_corpus.validate_corpus.",
        "Blocking chain: capture is blocked on a Metal GPU this sidecar does not have; analytical screens that can run now are named separately from funnel passes that cannot.",
        "Ranked first experiments when 256 rows land, ordered by expected information per unit cost, with family order derived from family_budget source_fraction rather than a hard-coded count.",
        "bind_corpus_to_families injects PRESENT/unmeasured only for an admitted capture and never for a structural fixture; simulate-arrival never reports gate 2 PASSED.",
    ]


def negative_findings(boundary: Mapping[str, Any], budgets: Mapping[str, Any]) -> list[str]:
    findings = [
        "This sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. Every hardware quantity remains UNKNOWN / null.",
        "No CPU teacher-fit operator lives in tools/future. tools/flash_meta_coherence_screen.py is Codex-owned; the pinned L4 screen is UNSAFE_SMALL_PROBE_NOT_PROMOTABLE (4 rows vs 256 required).",
        "No held-out / logit-token / bounded-capability runner for Flash meta families lives in tools/future. The funnel evaluators exist; the producers of their inputs do not.",
        "FLASH_COMPLETE_V2.nr.json is COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION (exact-control), not a lowering of a fitted meta program.",
        "FLASH_COMPLETE_V0.nx.json is SEALED_METADATA_ONLY_NOT_FOR_PROMOTION. flash_nx_audit reports the seven requirements NOT_MET.",
        "prospective_meta_bpw < 1 on SUB1 is a description budget and never a gate-9 pass.",
        "FLASH_ROUTE_STABILITY.json is in the snapshot WANTED list but was not among the pinned evidence files visible here; route-stability null is taken from the funnel + SUB1.router contract instead.",
        "meta_funnel.recover_families() reads receipts/headless/FLASH_META_REPRESENTATION_SUB1.json (live/HEAD), not the pinned snapshot. In this sparse worktree that live path is not visible, so the funnel recovered library/receipt families rather than the pinned family_budget entries. Ranked experiments derive from the pinned family_budget. All recovered families still stall at gate 2.",
        "No Metal-capable GPU is available to this lane. Capture cannot be unblocked here.",
    ]
    if not boundary.get("present"):
        findings.append(
            f"{REL_CAPTURE_BOUNDARY} was not visible from this sparse worktree; "
            "lookup coped and still named the Metal-GPU blocker rather than treating "
            "invisibility as absence of the campaign."
        )
    else:
        findings.append(
            f"Capture boundary present via {boundary.get('evidence_source')}: "
            f"status={boundary.get('status')!r} teacher_rows_written="
            f"{boundary.get('teacher_rows_written')!r} source_authority_capture="
            f"{boundary.get('source_authority_capture')!r} failure.stage="
            f"{boundary.get('failure_stage')!r}."
        )
    if not budgets.get("present"):
        findings.append(
            f"{REL_META_SUB1} / pinned snapshot was not readable; family ranking "
            "cannot be derived until it is."
        )
    return findings


def evidence_sources(
    boundary: Mapping[str, Any],
    budgets: Mapping[str, Any],
    screen: Mapping[str, Any],
) -> dict[str, str]:
    """Per-input evidence_source: pinned_snapshot or live_headless."""
    nr = resolve_evidence(REL_NR_V2, prefer_pinned=True)
    nx = resolve_evidence(REL_NX_V0, prefer_pinned=True)
    sel = resolve_evidence(REL_ROUTER_SEL, prefer_pinned=True)
    mp = resolve_evidence(REL_ROUTER_MAP, prefer_pinned=True)
    out = {
        REL_CAPTURE_BOUNDARY: str(boundary.get("evidence_source") or "live_headless"),
        REL_META_SUB1: str(budgets.get("evidence_source") or "pinned_snapshot"),
        REL_COHERENCE: str(screen.get("evidence_source") or "pinned_snapshot"),
        REL_NR_V2: str(nr.get("evidence_source") or "pinned_snapshot"),
        REL_NX_V0: str(nx.get("evidence_source") or "pinned_snapshot"),
        REL_ROUTER_SEL: str(sel.get("evidence_source") or "pinned_snapshot"),
        REL_ROUTER_MAP: str(mp.get("evidence_source") or "pinned_snapshot"),
    }
    return out


def build_document() -> dict[str, Any]:
    boundary = lookup_capture_boundary()
    budgets = load_family_budgets()
    screen = load_coherence_screen()
    dossiers = stage_dossiers(budgets=budgets, screen=screen, boundary=boundary)
    wiring = pipeline_wiring(dossiers)
    contract = corpus_arrival_contract(boundary)
    sim = simulate_arrival(dossiers=dossiers)
    chain = blocking_chain(boundary=boundary, dossiers=dossiers)
    ranked = ranked_first_experiments(budgets=budgets, boundary=boundary)
    families, provenance = recover_families()
    funnel = Funnel()
    runs = [funnel.run(c) for c in families]
    n_gate2 = sum(1 for r in runs if r.get("stall_gate") == 2)

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Prepare Flash meta downstream stages (funnel gates 3–9) so that the "
            "moment a real teacher corpus exists the work starts with no design "
            "left to do. Anti-fabrication is preserved. Training rows are not "
            "simulated. Odyssey I (WHAT IS TRUE?). Disk state is authority. "
            "There is no Era VI and no Odyssey IV. FPGA stays inside Accelerator "
            "/ Physical Compiler / Fusion."
        ),
        "claim_class": "STATIC_ONLY",
        "gpu_authority": False,
        "model": FLASH_MODEL,
        "stage_dossiers": dossiers,
        "pipeline_wiring": wiring,
        "corpus_arrival_contract": contract,
        "simulate_arrival": sim,
        "blocking_chain": chain,
        "ranked_first_experiments": ranked,
        "current_funnel_stall": {
            "n_recovered_families": len(runs),
            "n_stall_at_gate_2": n_gate2,
            "all_recovered_stall_at_gate_2": n_gate2 == len(runs) and len(runs) > 0,
            "primary_source": provenance.get("primary_source"),
            "n_families_from_pinned_budget": budgets.get("n_families"),
            "correct_answer": (
                f"{n_gate2} of {len(runs)} recovered Flash families stall at gate 2 "
                "because the teacher corpus is NOT_BUILT. That is the correct "
                "answer. This module does not advance them."
            ),
        },
        "family_budgets_derived": {
            "n_families": budgets.get("n_families"),
            "families": [
                {"family": f["family"], "source_fraction": f.get("source_fraction")}
                for f in (budgets.get("families") or [])
            ],
            "evidence_source": budgets.get("evidence_source"),
        },
        "coherence_screen_cited": {
            "status": screen.get("status"),
            "min_rows_required": screen.get("min_rows_required"),
            "unsafe_small_probe": screen.get("unsafe_small_probe"),
            "teacher_trace_rows": screen.get("teacher_trace_rows"),
            "evidence_source": screen.get("evidence_source"),
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(boundary, budgets),
        "evidence_source": evidence_sources(boundary, budgets, screen),
        "integration": {
            "admit_capture": "admit_capture(rows, envelope=...) -> {accepted, reason, codes}",
            "simulate_arrival": "simulate_arrival(corpus_path=...) -> wiring report; never PASSES gate 2",
            "make_structural_fixture": "labelled STRUCTURAL_NOT_REAL_CAPTURE; validate_corpus REFUSES it",
            "stage_dossiers": "stage_dossiers() -> gates 3–9",
            "ranked_first_experiments": "ranked_first_experiments() -> ordered by information/cost",
            "bind_corpus_to_families": "PRESENT/unmeasured only if admit_capture accepted",
        },
        "era_vocabulary": {
            "eras": 5,
            "odysseys": 3,
            "fpga_is": "FPGA is part of Accelerator / Physical Compiler / Fusion, not its own civilization",
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
    }
    return doc


def build() -> Path:
    doc = build_document()
    out = write_receipt(RECEIPT, doc, RECORDED_BY)
    written = load_json(out)
    if written.get("schema") != SCHEMA or not written.get("seal_sha256"):
        raise SystemExit(f"receipt {out} failed round-trip")
    if written.get("bench", {}).get("state") != "UNKNOWN":
        raise SystemExit("receipt bench.state is not UNKNOWN")
    if written.get("simulate_arrival", {}).get("gate2_advanced") is True:
        raise SystemExit("simulate-arrival illegally advanced gate 2")
    if written.get("simulate_arrival", {}).get("fabricated_training_rows") is True:
        raise SystemExit("simulate-arrival illegally fabricated training rows")
    return out


def selftest() -> dict[str, Any]:
    fixture = make_structural_fixture()
    refused = False
    codes: list[str] = []
    try:
        validate_corpus(fixture, min_rows=contract_min_rows(), raise_on_refuse=True)
    except CorpusRefused as exc:
        refused = True
        codes = list(exc.codes)
    if not refused:
        raise SystemExit(
            "selftest: structural fixture was NOT refused by validate_corpus — "
            "the fixture is the bug"
        )
    admission = admit_capture(
        fixture,
        envelope={
            "fixture_kind": STRUCTURAL_FIXTURE_KIND,
            "source_authority_capture": False,
        },
    )
    if admission["accepted"]:
        raise SystemExit("selftest: admit_capture accepted a structural fixture")
    sim = simulate_arrival()
    if sim["gate2_advanced"] or sim["fabricated_training_rows"]:
        raise SystemExit("selftest: simulate-arrival violated invariants")
    if not sim["wiring"]["wired"]:
        raise SystemExit(f"selftest: pipeline not wired: {sim['wiring']}")
    return {
        "structural_fixture_refused_by_validate_corpus": True,
        "validate_corpus_codes": codes,
        "admit_reason": admission["reason"],
        "simulate_wired": sim["wiring"]["wired"],
        "simulate_gate2_advanced": sim["gate2_advanced"],
        "n_structural_rows": len(fixture),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--simulate-arrival",
        action="store_true",
        help="wiring check only; never fabricates training rows",
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="caller-supplied corpus JSON (rows or {rows, source_authority_capture, ...})",
    )
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if a.simulate_arrival:
        report = simulate_arrival(corpus_path=a.corpus)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if a.build:
            print(build())
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
