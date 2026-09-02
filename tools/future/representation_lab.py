#!/usr/bin/env python3
"""Beyond-Dense research laboratory — plug a family in; do not start a search.

Roadmap §39: Beyond-Dense is Era III-D. This module is the environment a
future representation campaign plugs into. It does not choose a family, run
a SUB2 search, or claim a hardware number.

Recovered existing implementations this lab connects rather than rewrites:

* tools/odyssey/noetic_compiler.py — family registry, round_trip, chain_status
* tools/future/complete_ebpw.py — complete bill, refuse_unbilled_components
* tools/future/capability_eval.py — score_representation_family (same axes)
* tools/future/science_corpus.py — disk-or-git receipt loader
* tools/odyssey/families/*.py — plugin families; core does not name them

    python3 tools/future/representation_lab.py --build
    python3 -m pytest tools/future/test_representation_lab.py tools/odyssey/test_noetic_representation_chain.py -o addopts="" -q

This module MEASURES NOTHING. Corpus rows are STATIC / COST_MODEL readings
of named receipts. Family round-trips are FUNCTIONAL_SIM on a micro-site.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from tools.future import capability_eval as cap_eval
from tools.future import complete_ebpw as ce
from tools.future._common import REPO, git, write_receipt
from tools.future.science_corpus import _read_json
from tools.odyssey import noetic_compiler as nc


RECORDED_BY = "tools/future/representation_lab.py"
RECEIPT = "REPRESENTATION_LAB.json"
SCHEMA = "hawking.future.representation_lab.v1"
VERSION = 1

# Named real receipts this corpus actually loads. Absence of one source
# skips that adapter; an empty corpus after all adapters is a refusal.
NAMED_RECEIPTS: tuple[str, ...] = (
    "receipts/future/COMPLETE_EBPW.json",
    "receipts/future/REPRESENTATION_FLOOR.json",
    "receipts/future/RIVAL_CODEC_SCREEN.json",
    "receipts/future/MLP_CODE_INFORMATION.json",
    "receipts/future/MLP_AUXILIARY_INFORMATION.json",
    "receipts/future/MLP_SPARSE_RESIDUAL.json",
    "receipts/future/DELTANET_REPRESENTATION.json",
    "receipts/future/FLASH_BPW_LADDER.json",
    "receipts/future/AUX_CAPABILITY_SCREEN.json",
    "receipts/future/REPRESENTATION_DECODE_FUSION.json",
    "receipts/future/MLP_BYTE_CENSUS.json",
    "receipts/future/ECONOMICS_CALIBRATION.json",
)

RECORD_KEYS = (
    "record_id",
    "source_receipt",
    "source_schema",
    "family_or_move_id",
    "kind",
    "status",
    "axes",
    "evidence_tier",
    "source_evidence_class",
)

EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)

CORE_MODULE_RELS = nc.CORE_MODULE_RELS


class LabRefused(RuntimeError):
    """The laboratory cannot proceed without guessing."""


class FamilyClaimRefused(RuntimeError):
    """A family tried to claim without passing the verifier."""


def _record(
    *,
    record_id: str,
    source_receipt: str,
    source_schema: Any,
    family_or_move_id: str,
    kind: str,
    status: Any,
    axes: Mapping[str, Any],
    evidence_tier: str,
    source_evidence_class: Any = None,
) -> dict[str, Any]:
    if evidence_tier not in EVIDENCE_TIERS:
        raise LabRefused(f"unknown evidence_tier {evidence_tier!r}")
    if evidence_tier == "HARDWARE_MEASURED":
        raise LabRefused(
            "the laboratory reads receipts; it does not promote a file into "
            "HARDWARE_MEASURED"
        )
    if not record_id or not family_or_move_id:
        raise LabRefused("record_id and family_or_move_id must be non-empty")
    return {
        "record_id": record_id,
        "source_receipt": source_receipt,
        "source_schema": source_schema,
        "family_or_move_id": family_or_move_id,
        "kind": kind,
        "status": status,
        "axes": json.loads(json.dumps(dict(axes), default=str)),
        "evidence_tier": evidence_tier,
        "source_evidence_class": source_evidence_class,
    }


def _status_of(row: Mapping[str, Any]) -> Any:
    return row.get("status") or row.get("evidence_status") or row.get("verdict")


# ---------------------------------------------------------------------------
# Adapters. Each one names the receipt it reads. Numbers are copied, not
# recomputed.
# ---------------------------------------------------------------------------


def adapt_complete_ebpw(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    inc = doc.get("incumbent") if isinstance(doc.get("incumbent"), Mapping) else {}
    if not inc:
        return []
    return [
        _record(
            record_id=f"{source_receipt}:incumbent",
            source_receipt=source_receipt,
            source_schema=doc.get("schema"),
            family_or_move_id=str(inc.get("id") or "incumbent_sealed_3_14"),
            kind="incumbent",
            status="MEASURED" if inc.get("complete_ebpw") is not None else "UNKNOWN",
            axes={
                "complete_ebpw": inc.get("complete_ebpw"),
                "stored_bytes": inc.get("stored_bytes") or inc.get("payload_bytes"),
                "billed_ms": inc.get("billed_ms"),
                "parent_params": inc.get("parent_params"),
                "is_sub2_executable": inc.get("is_sub2_executable"),
            },
            evidence_tier="STATIC",
            source_evidence_class=doc.get("evidence_class") or "STATIC_ONLY",
        )
    ]


def adapt_representation_floor(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    floor = doc.get("floor") if isinstance(doc.get("floor"), Mapping) else {}
    if floor:
        out.append(
            _record(
                record_id=f"{source_receipt}:floor",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id="representation_floor",
                kind="floor",
                status="MEASURED",
                axes={
                    "incumbent_bpw": floor.get("incumbent_bpw"),
                    "incumbent_bytes": floor.get("incumbent_bytes"),
                    "measured_safe_bpw": floor.get("measured_safe_bpw"),
                    "measured_safe_bytes": floor.get("measured_safe_bytes"),
                    "measured_safe_ms_saved_billed": floor.get(
                        "measured_safe_ms_saved_billed"
                    ),
                    "if_every_untested_move_worked_bpw": floor.get(
                        "if_every_untested_move_worked_bpw"
                    ),
                },
                evidence_tier="STATIC",
                source_evidence_class="STATIC_ONLY",
            )
        )
    worth = doc.get("worth_it") if isinstance(doc.get("worth_it"), Mapping) else {}
    if worth:
        out.append(
            _record(
                record_id=f"{source_receipt}:worth_it",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id="conventional_compression_campaign",
                kind="verdict",
                status=worth.get("verdict"),
                axes={
                    "expected_ms_per_token": worth.get("expected_ms_per_token"),
                    "time_meets_ms_bar": worth.get("time_meets_ms_bar"),
                    "entropy_meets_size_bar": worth.get("entropy_meets_size_bar"),
                },
                evidence_tier="STATIC",
                source_evidence_class="STATIC_ONLY",
            )
        )
    cands = doc.get("candidates")
    if isinstance(cands, list):
        for raw in cands:
            if not isinstance(raw, Mapping) or not raw.get("id"):
                continue
            out.append(
                _record(
                    record_id=f"{source_receipt}:{raw['id']}",
                    source_receipt=source_receipt,
                    source_schema=doc.get("schema"),
                    family_or_move_id=str(raw["id"]),
                    kind="floor_candidate",
                    status=_status_of(raw),
                    axes={
                        "bytes_saved": raw.get("bytes_saved"),
                        "ms_saved": raw.get("ms_saved"),
                        "stream_class": raw.get("stream_class"),
                        "counts_toward_measured_safe": raw.get(
                            "counts_toward_measured_safe"
                        ),
                    },
                    evidence_tier="STATIC",
                    source_evidence_class=raw.get("evidence_status"),
                )
            )
    return out


def adapt_rival_codec_screen(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fams = doc.get("families")
    if not isinstance(fams, list):
        return out
    for raw in fams:
        if not isinstance(raw, Mapping):
            continue
        fid = raw.get("family")
        if not fid:
            continue
        passed = bool(raw.get("any_pass"))
        out.append(
            _record(
                record_id=f"{source_receipt}:{fid}",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id=str(fid),
                kind="rival",
                status="PASSED_CONTRACT" if passed else "FAILED_CONTRACT",
                axes={
                    "n_scored": raw.get("n_scored"),
                    "n_passed_contract": raw.get("n_passed_contract"),
                    "n_beats_q4": raw.get("n_beats_q4"),
                    "wins_the_screen": raw.get("wins_the_screen"),
                    "any_pass": passed,
                    "promotion_allowed": bool(doc.get("promotion_allowed")),
                },
                evidence_tier="STATIC",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    return out


def adapt_named_candidates(
    doc: Mapping[str, Any],
    *,
    source_receipt: str,
    kind: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cands = doc.get("candidates")
    if not isinstance(cands, list):
        return out
    for raw in cands:
        if not isinstance(raw, Mapping) or not raw.get("id"):
            continue
        axes: dict[str, Any] = {
            "bytes_eliminated_if_true": raw.get("bytes_eliminated_if_true"),
            "dense_rematerialization": raw.get("dense_rematerialization"),
        }
        measured = raw.get("measured")
        if isinstance(measured, Mapping) and "H_q_bits" in measured:
            axes["H_q_bits"] = measured.get("H_q_bits")
        out.append(
            _record(
                record_id=f"{source_receipt}:{raw['id']}",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id=str(raw["id"]),
                kind=kind,
                status=_status_of(raw),
                axes=axes,
                evidence_tier="STATIC",
                source_evidence_class=raw.get("evidence_class")
                or doc.get("evidence_class"),
            )
        )
    return out


def adapt_sparse_residual(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    answers = doc.get("answers") if isinstance(doc.get("answers"), Mapping) else {}
    if answers:
        out.append(
            _record(
                record_id=f"{source_receipt}:answers",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id="mlp_sparse_residual",
                kind="sparse_residual",
                status="MEASURED_NEGATIVE",
                axes={
                    "best_billed_bulk_held_out_relative_l2": answers.get(
                        "best_billed_bulk_held_out_relative_l2"
                    ),
                    "n_measured_negative": (doc.get("candidate_counts") or {}).get(
                        "measured_negative"
                    ),
                    "n_open": (doc.get("candidate_counts") or {}).get("open"),
                },
                evidence_tier="STATIC",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    bulks = doc.get("bulks")
    if isinstance(bulks, list):
        for raw in bulks:
            if not isinstance(raw, Mapping) or not raw.get("id"):
                continue
            out.append(
                _record(
                    record_id=f"{source_receipt}:bulk:{raw['id']}",
                    source_receipt=source_receipt,
                    source_schema=doc.get("schema"),
                    family_or_move_id=str(raw["id"]),
                    kind="sparse_bulk",
                    status="MEASURED_NEGATIVE",
                    axes={
                        "held_out_relative_l2": raw.get("held_out_relative_l2"),
                        "rank": raw.get("rank"),
                        "shape": raw.get("shape"),
                        "billed": raw.get("billed"),
                    },
                    evidence_tier="STATIC",
                    source_evidence_class=doc.get("evidence_class"),
                )
            )
    return out


def adapt_deltanet(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    out = adapt_named_candidates(doc, source_receipt=source_receipt, kind="deltanet")
    acc = doc.get("accounting") if isinstance(doc.get("accounting"), Mapping) else {}
    if acc:
        out.append(
            _record(
                record_id=f"{source_receipt}:accounting",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id="deltanet_representation",
                kind="organ_accounting",
                status="RECONCILED" if acc.get("reconciled") else "UNRECONCILED",
                axes={
                    "stored_bytes": acc.get("stored_bytes"),
                    "code_bytes": acc.get("code_bytes"),
                    "auxiliary_bytes": acc.get("auxiliary_bytes"),
                    "reconciled": acc.get("reconciled"),
                },
                evidence_tier="STATIC",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    return out


def adapt_flash_bpw_ladder(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rungs = doc.get("rungs")
    if not isinstance(rungs, list):
        return out
    for raw in rungs:
        if not isinstance(raw, Mapping) or not raw.get("id"):
            continue
        out.append(
            _record(
                record_id=f"{source_receipt}:{raw['id']}",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id=str(raw["id"]),
                kind="ladder_rung",
                status="TARGET",
                axes={
                    "target_bpw": raw.get("target_bpw"),
                    "required_quantity": raw.get("required_quantity"),
                    "required_evidence_class": raw.get("required_evidence_class"),
                },
                evidence_tier="STATIC",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    return out


def adapt_aux_capability_screen(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    levers = doc.get("levers")
    if not isinstance(levers, list):
        return out
    for raw in levers:
        if not isinstance(raw, Mapping) or not raw.get("id"):
            continue
        out.append(
            _record(
                record_id=f"{source_receipt}:{raw['id']}",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id=str(raw["id"]),
                kind="aux_lever",
                status=_status_of(raw),
                axes={
                    "bytes_removed": raw.get("bytes_removed"),
                    "bytes_added": raw.get("bytes_added"),
                    "dense_rematerialization": raw.get("dense_rematerialization"),
                },
                evidence_tier="STATIC",
                source_evidence_class=raw.get("evidence_tier")
                or doc.get("evidence_class"),
            )
        )
    return out


def adapt_decode_fusion(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    ranking = doc.get("ranking") if isinstance(doc.get("ranking"), Mapping) else {}
    if not ranking:
        return []
    return [
        _record(
            record_id=f"{source_receipt}:ranking",
            source_receipt=source_receipt,
            source_schema=doc.get("schema"),
            family_or_move_id=str(ranking.get("top_legal") or "decode_fusion"),
            kind="decode_fusion",
            status=ranking.get("top_status"),
            axes={
                "top_legal": ranking.get("top_legal"),
                "top_bytes_eliminated_vs_split_decode": ranking.get(
                    "top_bytes_eliminated_vs_split_decode"
                ),
                "rejected_dense_remat": ranking.get("rejected_dense_remat"),
            },
            evidence_tier="STATIC",
            source_evidence_class=doc.get("evidence_class"),
        )
    ]


def adapt_mlp_byte_census(
    doc: Mapping[str, Any], *, source_receipt: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if doc.get("mlp_active_bytes") is not None:
        out.append(
            _record(
                record_id=f"{source_receipt}:mlp_active",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id="mlp_byte_census",
                kind="census",
                status="MEASURED",
                axes={
                    "mlp_active_bytes": doc.get("mlp_active_bytes"),
                    "mlp_share_of_active": doc.get("mlp_share_of_active"),
                    "family_counts": doc.get("family_counts"),
                },
                evidence_tier="STATIC",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    fams = doc.get("families")
    if isinstance(fams, list):
        for raw in fams:
            if not isinstance(raw, Mapping) or not raw.get("id"):
                continue
            out.append(
                _record(
                    record_id=f"{source_receipt}:{raw['id']}",
                    source_receipt=source_receipt,
                    source_schema=doc.get("schema"),
                    family_or_move_id=str(raw["id"]),
                    kind="census_family",
                    status=_status_of(raw),
                    axes={
                        "bytes_eliminated_if_true": raw.get("bytes_eliminated_if_true"),
                        "dense_rematerialization": raw.get("dense_rematerialization"),
                    },
                    evidence_tier="STATIC",
                    source_evidence_class=raw.get("evidence_class")
                    or doc.get("evidence_class"),
                )
            )
    return out


def adapt_economics(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    classes = doc.get("stream_classes")
    if not isinstance(classes, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for name, row in classes.items():
        if not isinstance(row, Mapping):
            continue
        out.append(
            _record(
                record_id=f"{source_receipt}:stream:{name}",
                source_receipt=source_receipt,
                source_schema=doc.get("schema"),
                family_or_move_id=f"stream_class:{name}",
                kind="stream_rate",
                status="CATALOG",
                axes={
                    "ms_per_gb_saved": row.get("ms_per_gb_saved"),
                    "on_critical_path": row.get("on_critical_path"),
                },
                evidence_tier="COST_MODEL",
                source_evidence_class=doc.get("evidence_class"),
            )
        )
    return out


ADAPTERS: dict[str, Any] = {
    "COMPLETE_EBPW.json": adapt_complete_ebpw,
    "REPRESENTATION_FLOOR.json": adapt_representation_floor,
    "RIVAL_CODEC_SCREEN.json": adapt_rival_codec_screen,
    "MLP_CODE_INFORMATION.json": lambda d, source_receipt: adapt_named_candidates(
        d, source_receipt=source_receipt, kind="mlp_code"
    ),
    "MLP_AUXILIARY_INFORMATION.json": lambda d, source_receipt: adapt_named_candidates(
        d, source_receipt=source_receipt, kind="mlp_aux"
    ),
    "MLP_SPARSE_RESIDUAL.json": adapt_sparse_residual,
    "DELTANET_REPRESENTATION.json": adapt_deltanet,
    "FLASH_BPW_LADDER.json": adapt_flash_bpw_ladder,
    "AUX_CAPABILITY_SCREEN.json": adapt_aux_capability_screen,
    "REPRESENTATION_DECODE_FUSION.json": adapt_decode_fusion,
    "MLP_BYTE_CENSUS.json": adapt_mlp_byte_census,
    "ECONOMICS_CALIBRATION.json": adapt_economics,
}


def load_prior_results(
    sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load named real receipts into a comparable corpus.

    A new idea looks up a move_id here instead of re-measuring it.
    """
    wanted = sources if sources is not None else NAMED_RECEIPTS
    records: list[dict[str, Any]] = []
    loaded: list[str] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel in wanted:
        doc, meta = _read_json(rel)
        if doc is None:
            skipped.append({**meta, "why": "missing_or_unreadable"})
            continue
        adapter = ADAPTERS.get(Path(rel).name)
        if adapter is None:
            skipped.append({**meta, "why": "no_adapter"})
            continue
        adapted = list(adapter(doc, source_receipt=rel))
        if not adapted:
            skipped.append({**meta, "why": "adapter_emitted_nothing"})
            continue
        loaded.append(rel)
        for rec in adapted:
            missing = [k for k in RECORD_KEYS if k not in rec]
            if missing:
                raise LabRefused(f"{rel} record missing {missing}")
            rid = rec["record_id"]
            if rid in seen:
                continue
            seen.add(rid)
            records.append(rec)
    if not records:
        raise LabRefused(
            "representation corpus is empty; adapters loaded nothing from "
            f"{list(wanted)}"
        )
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for rec in records:
        by_kind[rec["kind"]] = by_kind.get(rec["kind"], 0) + 1
        st = str(rec["status"])
        by_status[st] = by_status.get(st, 0) + 1
    return {
        "schema": SCHEMA,
        "named_receipts": list(wanted),
        "named_receipts_loaded": loaded,
        "skipped": skipped,
        "n_records": len(records),
        "by_kind": by_kind,
        "by_status": by_status,
        "records": records,
        "evidence_tier": "STATIC",
        "claim_boundary": (
            "Projection of named historical representation receipts. "
            "evidence_tier is STATIC or COST_MODEL because this process is "
            "reading files, not measuring. Not a SUB2 search and not a "
            "hardware number."
        ),
    }


def lookup_prior(corpus: Mapping[str, Any], move_id: str) -> list[dict[str, Any]]:
    """Prior results for a named move/family. Empty means unmeasured, not zero."""
    needle = move_id.lower()
    if not needle:
        return []
    hits = []
    for rec in corpus.get("records") or []:
        if needle in str(rec.get("family_or_move_id", "")).lower() or needle in str(
            rec.get("record_id", "")
        ).lower():
            hits.append(rec)
    return hits


# ---------------------------------------------------------------------------
# Capability-evaluation hook + verifier.
# ---------------------------------------------------------------------------


def local_dense_f32_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Same-W dense f32 incumbent so a micro-site is scored on commensurate axes."""
    n = nc._parent_params_of(payload)
    n_bytes = int(n) * 4
    parts = ce.empty_parts()
    parts["regions"] = [
        {
            "name": "dense_f32",
            "bytes": n_bytes,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    return ce.candidate_from_parts(
        family_id="local_dense_f32",
        parent_params=max(int(n), 1),
        parts=parts,
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )


def _family_axes(result: Mapping[str, Any], billed: Mapping[str, Any]) -> dict[str, Any]:
    axes = ce.axes_of(billed)
    execute = result.get("execute") if isinstance(result.get("execute"), Mapping) else None
    chain = result.get("chain") if isinstance(result.get("chain"), Mapping) else {}
    axes["execute_match"] = bool(execute and execute.get("match_atol_1e5")) if execute else False
    axes["chain_complete"] = bool(chain.get("complete"))
    return axes


def _incumbent_axes(inc_row: Mapping[str, Any], *, execute_match: bool, chain_complete: bool) -> dict[str, Any]:
    axes = ce.axes_of(inc_row)
    axes["execute_match"] = execute_match
    axes["chain_complete"] = chain_complete
    return axes


def score_family(family_id: str) -> dict[str, Any]:
    """Score a registered family on the same axes as a local dense incumbent.

    CALL SITES (not imports):
      nc.round_trip, ce.candidate_from_parts, ce.refuse_unbilled_components,
      ce.cost, ce.compare_to_incumbent, cap_eval.score_representation_family.
    """
    spec = nc.get_family(family_id)
    result = nc.round_trip(family_id)
    payload = spec.demo_payload() if spec.demo_payload is not None else None
    if payload is None:
        raise FamilyClaimRefused(f"{family_id}: no demo_payload; cannot score")
    billed_parts = spec.bill_parts(payload) if spec.bill_parts is not None else None
    if billed_parts is None:
        raise FamilyClaimRefused(f"{family_id}: no bill_parts; cannot score")
    parts = ce.empty_parts()
    for cat, rows in billed_parts.items():
        if cat in parts:
            parts[cat] = list(rows)
    parent_params = nc._parent_params_of(payload)
    # CALL SITE of the unbilled-component gate's own symbol. Extra bill_parts
    # keys stay on the probe so a sidecar cannot skip billing by living
    # outside PART_CATEGORIES.
    probe = {
        "id": family_id,
        "parent_params": parent_params,
        "stated_total_bytes": 0,
        **parts,
        "reconstructs_dense_parent": False,
        "consumes_representation_directly": True,
    }
    for key, value in billed_parts.items():
        if key not in ce.PART_CATEGORIES:
            probe[key] = value
    ce.refuse_unbilled_components(probe)
    cand = ce.candidate_from_parts(
        family_id=family_id,
        parent_params=parent_params,
        parts=parts,
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    billed = ce.cost(cand)
    local_inc = local_dense_f32_candidate(payload)
    compared = ce.compare_to_incumbent(cand, incumbent=local_inc)
    cand_axes = _family_axes(result, billed)
    inc_axes = _incumbent_axes(
        compared["incumbent_row"],
        execute_match=True,
        chain_complete=True,
    )
    # CALL SITE of the capability-evaluation hook.
    score = cap_eval.score_representation_family(
        candidate_id=family_id,
        candidate=cand_axes,
        incumbent_id="local_dense_f32",
        incumbent=inc_axes,
    )
    return {
        "family_id": family_id,
        "result": result,
        "billed": billed,
        "compared": {
            "candidate_id": compared["candidate_id"],
            "incumbent_id": compared["incumbent_id"],
            "candidate_axes": compared["candidate_axes"],
            "incumbent_axes": compared["incumbent_axes"],
            "versus": compared["versus"],
            "same_axes": compared["same_axes"],
        },
        "score": score,
        "evidence_tier": result.get("evidence_tier") or spec.evidence_tier,
        "call_sites": (
            "tools.odyssey.noetic_compiler.round_trip",
            "tools.future.complete_ebpw.refuse_unbilled_components",
            "tools.future.complete_ebpw.candidate_from_parts",
            "tools.future.complete_ebpw.cost",
            "tools.future.complete_ebpw.compare_to_incumbent",
            "tools.future.capability_eval.score_representation_family",
        ),
    }


def verify_family(family_id: str, *, corpus: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A family must pass this before it is allowed to claim anything."""
    spec = nc.get_family(family_id)
    chain = nc.chain_status(spec)
    blockers = list(chain.get("blockers") or spec.blockers)
    checks: dict[str, Any] = {
        "registered": True,
        "evidence_tier_labeled": spec.evidence_tier in EVIDENCE_TIERS,
        "evidence_tier_not_hardware_from_this_lab": spec.evidence_tier
        != "HARDWARE_MEASURED",
        "plugin_or_source": nc._is_plugin_path(spec.source_path)
        or spec.source_path in {s.source_path for s in nc.list_families()},
        "named_in_core": False,
    }
    core_src = "\n".join((REPO / rel).read_text() for rel in CORE_MODULE_RELS)
    if family_id in core_src and nc._is_plugin_path(spec.source_path):
        checks["named_in_core"] = True
        blockers.append(f"{family_id} is named in a core module; plugins must not be")

    scored: dict[str, Any] | None = None
    if spec.demo_payload is not None and spec.bill_parts is not None:
        scored = score_family(family_id)
        checks["round_trip_verified"] = bool(scored["result"].get("verified"))
        checks["ebpw_reconciled"] = bool(scored["billed"].get("reconciled"))
        checks["scored_on_incumbent_axes"] = bool(
            scored["score"].get("same_axes_as_incumbent")
        )
        execute = scored["result"].get("execute")
        if spec.executes:
            checks["execute_matches_reconstruct"] = bool(
                execute and execute.get("match_atol_1e5")
            )
        else:
            checks["execute_matches_reconstruct"] = execute is None
    else:
        checks["round_trip_verified"] = False
        checks["ebpw_reconciled"] = False
        checks["scored_on_incumbent_axes"] = False
        if not spec.demo_payload:
            blockers.append(f"{family_id}: no demo_payload")
        if not spec.bill_parts:
            blockers.append(f"{family_id}: no bill_parts")

    priors = []
    if corpus is not None:
        priors = lookup_prior(corpus, family_id)
    checks["prior_lookup_ran"] = corpus is not None
    checks["n_priors"] = len(priors)

    failed = [
        name
        for name, held in checks.items()
        if name != "named_in_core"
        and name != "n_priors"
        and name != "prior_lookup_ran"
        and held is False
    ]
    if checks["named_in_core"]:
        failed.append("named_in_core")
    passed = not failed and not blockers if spec.executes else not failed
    # Incomplete source families may verify as "blocked, honestly named".
    if not spec.executes and blockers and "round_trip_verified" in failed:
        # They are not allowed to claim execution. They may exist.
        passed = checks["registered"] and checks["evidence_tier_labeled"]

    return {
        "family_id": family_id,
        "passed": bool(passed) and spec.executes and not blockers,
        "exists": True,
        "executes": spec.executes,
        "checks": checks,
        "failed": failed,
        "blockers": blockers,
        "chain": chain,
        "score": None if scored is None else scored["score"],
        "accounting": None
        if scored is None
        else {
            "stored_bytes": scored["billed"]["stored_bytes"],
            "complete_ebpw": scored["billed"]["complete_ebpw"],
            "billed_ms": scored["billed"]["billed_ms"],
            "is_sub2_executable": scored["billed"]["is_sub2_executable"],
            "reconciled": scored["billed"]["reconciled"],
            "parts": [
                {"name": p["name"], "category": p["category"], "bytes": p["bytes"]}
                for p in scored["billed"]["parts"]
            ],
        },
        "compared": None if scored is None else scored["compared"],
        "priors": [
            {
                "record_id": p["record_id"],
                "source_receipt": p["source_receipt"],
                "status": p["status"],
            }
            for p in priors
        ],
        "evidence_tier": spec.evidence_tier,
        "source_path": spec.source_path,
        "plugin": nc._is_plugin_path(spec.source_path),
    }


def claim(
    family_id: str,
    statement: str,
    *,
    asserted: Mapping[str, Any] | None = None,
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Allow a statement only after verify_family passes.

    A sub-2 claim, a hardware-tier claim, or a billed-number mismatch is refused.
    """
    report = verify_family(family_id, corpus=corpus)
    if not report["passed"]:
        raise FamilyClaimRefused(
            f"{family_id} has not passed the laboratory verifier; "
            f"failed={report['failed']} blockers={report['blockers']}"
        )
    acc = report.get("accounting") or {}
    asserted = dict(asserted or {})
    if asserted.get("evidence_tier") == "HARDWARE_MEASURED":
        raise FamilyClaimRefused(
            f"{family_id}: this laboratory does not measure hardware; "
            "HARDWARE_MEASURED is refused"
        )
    if "complete_ebpw" in asserted:
        actual = float(acc["complete_ebpw"])
        if abs(float(asserted["complete_ebpw"]) - actual) > 1e-9:
            raise FamilyClaimRefused(
                f"{family_id}: asserted complete_ebpw {asserted['complete_ebpw']} "
                f"does not match billed {actual}"
            )
    if asserted.get("is_sub2_executable"):
        if not acc.get("is_sub2_executable"):
            raise FamilyClaimRefused(
                f"{family_id}: sub-2 claim refused; billed complete_ebpw is "
                f"{acc.get('complete_ebpw')}"
            )
    if asserted.get("beats_sealed_incumbent"):
        raise FamilyClaimRefused(
            f"{family_id}: a micro-site cannot claim to beat the sealed-3.14 "
            "incumbent; compare against the corpus instead of re-measuring it"
        )
    return {
        "family_id": family_id,
        "allowed": True,
        "statement": statement,
        "asserted": asserted,
        "billed_complete_ebpw": acc.get("complete_ebpw"),
        "evidence_tier": report["evidence_tier"],
        "verifier": "tools.future.representation_lab.verify_family",
    }


def core_diff() -> str:
    """git diff --name-only of core modules. A plugin must not require these."""
    raw = subprocess.run(
        ["git", "--no-optional-locks", "diff", "--name-only", "--", *CORE_MODULE_RELS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return (raw.stdout or "").strip()


def plugin_not_in_core(family_id: str) -> dict[str, Any]:
    spec = nc.get_family(family_id)
    core_src = "\n".join((REPO / rel).read_text() for rel in CORE_MODULE_RELS)
    diff = core_diff()
    return {
        "family_id": family_id,
        "source_path": spec.source_path,
        "plugin": nc._is_plugin_path(spec.source_path),
        "named_in_core_source": family_id in core_src,
        "source_path_in_core_rels": spec.source_path in CORE_MODULE_RELS,
        "git_diff_name_only_core": diff or "(empty)",
    }


def run_second_family_proof(family_id: str = "toy_mean_residual") -> dict[str, Any]:
    """Register-and-chain proof for the second toy family. Zero core edits."""
    nc.ensure_families()
    spec = nc.get_family(family_id)
    identity = plugin_not_in_core(family_id)
    report = verify_family(family_id)
    allowed = claim(
        family_id,
        "FUNCTIONAL_SIM round-trip on a synthetic micro-site; not a research candidate",
        asserted={"is_sub2_executable": False},
    )
    sub2_refused = False
    try:
        claim(family_id, "this is a sub-2 executable", asserted={"is_sub2_executable": True})
    except FamilyClaimRefused:
        sub2_refused = True
    hw_refused = False
    try:
        claim(
            family_id,
            "hardware measured",
            asserted={"evidence_tier": "HARDWARE_MEASURED"},
        )
    except FamilyClaimRefused:
        hw_refused = True
    return {
        "family_id": family_id,
        "source_path": spec.source_path,
        "plugin": identity["plugin"],
        "named_in_core_source": identity["named_in_core_source"],
        "git_diff_name_only_core": identity["git_diff_name_only_core"],
        "verified": report["passed"],
        "claim_allowed": allowed["allowed"],
        "sub2_claim_refused": sub2_refused,
        "hardware_claim_refused": hw_refused,
        "accounting": report["accounting"],
        "score_evaluator": (report.get("score") or {}).get("evaluator_id"),
        "same_axes_as_incumbent": (report.get("score") or {}).get(
            "same_axes_as_incumbent"
        ),
        "call_sites": (report.get("score") or {}).get("call_site"),
    }


def build() -> dict[str, Any]:
    corpus = load_prior_results()
    nc.ensure_families()
    second = run_second_family_proof("toy_mean_residual")
    first = verify_family("toy_xor_codes", corpus=corpus)
    # Corpus incumbent is the sealed mix, named so a new idea can compare.
    sealed = lookup_prior(corpus, "incumbent_sealed_3_14")
    entropy = lookup_prior(corpus, "entropy_code_mlp_codes")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "obligation": (
            "Beyond-Dense laboratory: corpus, capability-eval hook, complete "
            "bill, verifier. Do not start a representation search."
        ),
        "question": (
            "Can a new representation family plug in, get billed, get scored "
            "on the same axes as an incumbent, and be forbidden from claiming "
            "until it passes, without editing core?"
        ),
        "named_receipts_loaded": corpus["named_receipts_loaded"],
        "corpus": {
            "n_records": corpus["n_records"],
            "by_kind": corpus["by_kind"],
            "by_status": corpus["by_status"],
            "named_receipts_loaded": corpus["named_receipts_loaded"],
            "skipped": corpus["skipped"],
        },
        "sealed_incumbent_priors": [
            {"record_id": r["record_id"], "axes": r["axes"], "source_receipt": r["source_receipt"]}
            for r in sealed
        ],
        "entropy_code_mlp_codes_priors": [
            {"record_id": r["record_id"], "status": r["status"], "axes": r["axes"]}
            for r in entropy
        ],
        "first_toy": {
            "family_id": "toy_xor_codes",
            "passed": first["passed"],
            "plugin": first["plugin"],
            "source_path": first["source_path"],
        },
        "second_toy": second,
        "recovered_implementation": [
            {
                "path": "tools/odyssey/noetic_compiler.py",
                "what": "family registry, round_trip, chain_status, plugin glob",
            },
            {
                "path": "tools/future/complete_ebpw.py",
                "what": "complete bill; refuse_unbilled_components; COMPARE_AXES",
            },
            {
                "path": "tools/future/capability_eval.py",
                "what": "score_representation_family hook; subject kind representation",
            },
            {
                "path": "tools/odyssey/families/toy_xor_codes.py",
                "what": "first plugin family; this lab adds a second",
            },
        ],
        "call_sites": (
            "representation_lab.score_family -> noetic_compiler.round_trip",
            "representation_lab.score_family -> complete_ebpw.refuse_unbilled_components",
            "representation_lab.score_family -> complete_ebpw.cost",
            "representation_lab.score_family -> complete_ebpw.compare_to_incumbent",
            "representation_lab.score_family -> capability_eval.score_representation_family",
            "representation_lab.claim -> representation_lab.verify_family",
        ),
        "not_a_sub2_search": True,
        "not_a_hardware_measurement": True,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT, doc, RECORDED_BY))
        return 0
    print(
        json.dumps(
            {
                "named_receipts_loaded": doc["named_receipts_loaded"],
                "corpus_n_records": doc["corpus"]["n_records"],
                "second_toy": {
                    k: doc["second_toy"][k]
                    for k in (
                        "family_id",
                        "source_path",
                        "verified",
                        "named_in_core_source",
                        "git_diff_name_only_core",
                        "sub2_claim_refused",
                    )
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
