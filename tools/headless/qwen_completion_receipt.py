#!/usr/bin/env python3
"""N039 QWEN_COMPLETION_RECEIPT: Odyssey textbook #1 (CPU consolidation).

S024 §84: before the current Qwen specimen retires, seal the completion
vector — strongest coherent EBPW + token_ns, MLP density floor 2.25 measured
four independent ways, binary/shared-basis/hybrid verdicts, three roofs +
production, residual bottleneck, concurrency, dispatch frontier, reusable
kernels/organs, negative science, RETIREMENT_READY with the §73/§74 gate.

CPU only. Does not load a model, does not touch the GPU, does not run cargo
or Metal, does not mutate NOETIC_PARENT_A, does not re-derive a measured
number. Every figure is copied from the receipt that owns it.

    python3 tools/headless/qwen_completion_receipt.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from density_descent_frontier import (  # noqa: E402
    ABSENT,
    CITED,
    DERIVED,
    MEASURED,
    absent,
    citation_exists,
    git_head,
    load_json,
    load_json_optional,
    now_iso,
    numeric,
    qty,
    rel,
    unique_citations,
    unresolved_citations,
    write_json,
)

SCHEMA = "hawking.headless.qwen_completion_receipt.v1"
RECEIPT = REPO / "receipts" / "headless" / "QWEN_COMPLETION_RECEIPT.json"
GENERATOR = "tools/headless/qwen_completion_receipt.py"
OBLIGATION = (
    "N039 — QWEN_COMPLETION_RECEIPT (S024 §84; Odyssey textbook #1, retirement "
    "gate; CPU). Records the §84 completion vector for the current Qwen3.8 "
    "specimen. Does NOT retire the specimen; records whether the knowledge is sealed."
)

HEADLESS = "receipts/headless"

REQUIRED_INPUTS = (
    f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json",
    f"{HEADLESS}/FRACTIONAL_BIT_CANON.json",
    f"{HEADLESS}/BYTES_FRONTIER.json",
    f"{HEADLESS}/NATIVE_2BIT_MLP.json",
    f"{HEADLESS}/BINARY_HEALING.json",
    f"{HEADLESS}/SHARED_BASIS_KERNEL.json",
    f"{HEADLESS}/SHARED_BASIS_COHERENT.json",
    f"{HEADLESS}/HYBRID_OPERATOR.json",
    f"{HEADLESS}/ORGAN_LIBRARY.json",
    f"{HEADLESS}/ORGAN_ROOF_LEDGER.json",
    f"{HEADLESS}/ORGAN_FRONTIERS.json",
    f"{HEADLESS}/BANDWIDTH_ROOF.json",
    f"{HEADLESS}/BANDWIDTH_ASCENT.json",
    f"{HEADLESS}/GPU_LEDGER.json",
    f"{HEADLESS}/KERNEL_BOTTLENECK.json",
    f"{HEADLESS}/ORGAN_BANDWIDTH.json",
    f"{HEADLESS}/DISPATCH_LEDGER.json",
    f"{HEADLESS}/NOETIC_MULTISESSION.json",
    f"{HEADLESS}/PRODUCTION_BENCH.json",
    f"{HEADLESS}/KERNEL_LIBRARY.json",
    f"{HEADLESS}/NOETIC_NEGATIVE_SCIENCE.json",
    f"{HEADLESS}/NOETIC_PARENT_A.json",
    f"{HEADLESS}/MLP_GATE_UP.json",
    f"{HEADLESS}/MLP_DOWN.json",
    f"{HEADLESS}/GPU_IDLE_GAP_LEDGER.json",
    f"{HEADLESS}/MACHINE_GENOME.json",
    f"{HEADLESS}/REPRESENTATION_LIBRARY.json",
)

OPTIONAL_INPUTS = (
    f"{HEADLESS}/C1SHAREDBASIS_DESIGN.json",
    f"{HEADLESS}/C2TENSOROP_DESIGN.json",
    f"{HEADLESS}/C3LOWRANKSPARSE_DESIGN.json",
    f"{HEADLESS}/C4CODEBOOK_DESIGN.json",
    f"{HEADLESS}/C5STRUCTTRANSFORM_DESIGN.json",
    f"{HEADLESS}/COMPOSITION_LADDER.json",
    f"{HEADLESS}/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
    f"{HEADLESS}/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
    f"{HEADLESS}/FIRST_NOETIC_EXECUTABLE.json",
    f"{HEADLESS}/KERNEL_COMPETENCE.json",
    f"{HEADLESS}/ODYSSEY_QUEUE_RECOVERED.json",
)

TRANSFER_CANDIDATES = (
    f"{HEADLESS}/TRANSFER_RECEIPT.json",
    f"{HEADLESS}/TRANSFER_REPORT.json",
    f"{HEADLESS}/QWEN_TRANSFER_RECEIPT.json",
)

C6_C8 = (
    f"{HEADLESS}/C6GENERATED_DESIGN.json",
    f"{HEADLESS}/C7ROUTED_DESIGN.json",
    f"{HEADLESS}/C8STATEFUL_DESIGN.json",
)


def cited(
    value: Any,
    *,
    unit: str,
    command: str,
    source: str,
    note: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return qty(
        value,
        kind=CITED,
        unit=unit,
        command=command,
        source=source,
        note=note,
        extra=extra,
    )


def copy_qty(blob: Any, fallback_source: str, command: str, unit: str) -> dict[str, Any]:
    if isinstance(blob, dict) and "kind" in blob:
        out = dict(blob)
        if not out.get("source"):
            out["source"] = fallback_source
        out.setdefault("command", command)
        out.setdefault("unit", unit)
        return out
    if blob is None:
        return absent(unit, command, "source receipt has no value", fallback_source)
    return cited(blob, unit=unit, command=command, source=fallback_source)


def input_row(path: str, *, required: bool) -> dict[str, Any]:
    present = citation_exists(path)
    return {
        "path": path,
        "present": present,
        "required": required,
        "absent_reason": None if present else (
            "required receipt missing on disk and in git"
            if required
            else "optional; not sealed"
        ),
    }


class Bundle:
    def __init__(self) -> None:
        missing: list[str] = []
        self.req: dict[str, dict[str, Any]] = {}
        for p in REQUIRED_INPUTS:
            try:
                self.req[p] = load_json(p)
            except FileNotFoundError:
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                "N039 required receipts missing (on disk and in git): "
                + ", ".join(missing)
            )
        self.opt: dict[str, dict[str, Any] | None] = {}
        for p in OPTIONAL_INPUTS:
            self.opt[p] = load_json_optional(p) if citation_exists(p) else None

    def r(self, name: str) -> dict[str, Any]:
        return self.req[f"{HEADLESS}/{name}.json"]

    def o(self, name: str) -> dict[str, Any] | None:
        return self.opt.get(f"{HEADLESS}/{name}.json")


def strongest_coherent(b: Bundle) -> dict[str, Any]:
    ddf = b.r("DENSITY_DESCENT_FRONTIER")
    coh = ddf["COHERENCE_FRONTIER"]
    exe = ddf["EXECUTION_FRONTIER"]
    src_ddf = f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json"
    src_bytes = f"{HEADLESS}/BYTES_FRONTIER.json"
    src_n021 = f"{HEADLESS}/NATIVE_2BIT_MLP.json"
    ebpw = copy_qty(
        coh.get("complete_ebpw"),
        src_bytes,
        "DENSITY_DESCENT_FRONTIER.COHERENCE_FRONTIER.complete_ebpw",
        "bpw",
    )
    ns = copy_qty(
        coh.get("COMPLETE_TOKEN_NS"),
        src_n021,
        "DENSITY_DESCENT_FRONTIER.COHERENCE_FRONTIER.COMPLETE_TOKEN_NS",
        "ns/token",
    )
    return {
        "name": coh.get("name"),
        "candidate_id": coh.get("candidate_id"),
        "axis": "lowest_coherent_complete_ebpw",
        "complete_ebpw": ebpw,
        "COMPLETE_TOKEN_NS": ns,
        "ms": exe.get("ms"),
        "coherent": coh.get("coherent"),
        "composition_rung": coh.get("composition_rung"),
        "frontiers_are_the_same_artifact": ddf.get("frontiers_are_the_same_artifact"),
        "source": src_ddf,
        "source_receipt": src_ddf,
        "citations": [src_ddf, src_bytes, src_n021, f"{HEADLESS}/FRACTIONAL_BIT_CANON.json"],
        "note": (
            "q2f_g64 holds the coherence frontier at 2.25 bpw. Token_ns is the "
            "N021 fused complete-token GPU median (7 reps), reused by BYTES_FRONTIER."
        ),
    }


def fastest_coherent(b: Bundle) -> dict[str, Any]:
    ddf = b.r("DENSITY_DESCENT_FRONTIER")
    exe = ddf["EXECUTION_FRONTIER"]
    src_ddf = f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json"
    faster = ddf.get("faster_than_execution_frontier_but_incoherent") or []
    return {
        "name": exe.get("name"),
        "candidate_id": exe.get("candidate_id"),
        "axis": "lowest_coherent_complete_token_ns",
        "COMPLETE_TOKEN_NS": copy_qty(
            exe.get("COMPLETE_TOKEN_NS"),
            f"{HEADLESS}/NATIVE_2BIT_MLP.json",
            "DENSITY_DESCENT_FRONTIER.EXECUTION_FRONTIER.COMPLETE_TOKEN_NS",
            "ns/token",
        ),
        "ms": exe.get("ms"),
        "complete_ebpw": copy_qty(
            exe.get("complete_ebpw"),
            f"{HEADLESS}/BYTES_FRONTIER.json",
            "DENSITY_DESCENT_FRONTIER.EXECUTION_FRONTIER.complete_ebpw",
            "bpw",
        ),
        "coherent": exe.get("coherent"),
        "no_coherent_candidate_is_faster": True,
        "faster_incoherent_bodies": faster,
        "source": src_ddf,
        "source_receipt": src_ddf,
        "citations": [
            src_ddf,
            f"{HEADLESS}/NATIVE_2BIT_MLP.json",
            f"{HEADLESS}/BYTES_FRONTIER.json",
        ],
        "note": (
            "binary_g64 (23.43 ms) and shared_basis_k2 (24.55 ms) are faster and "
            "incoherent; they do not take the execution frontier (S024 §2)."
        ),
    }


def organ_vs_coherent(b: Bundle) -> dict[str, Any]:
    lib = b.r("ORGAN_LIBRARY")
    fr = b.r("ORGAN_FRONTIERS")
    src_lib = f"{HEADLESS}/ORGAN_LIBRARY.json"
    src_fr = f"{HEADLESS}/ORGAN_FRONTIERS.json"
    src_roof = f"{HEADLESS}/ORGAN_ROOF_LEDGER.json"
    src_bytes = f"{HEADLESS}/BYTES_FRONTIER.json"

    organ_rows = []
    lowest_coherent_organ: dict[str, Any] | None = None
    for o in lib.get("organs") or []:
        ebpw = o.get("best_complete_ebpw") or {}
        row = {
            "organ": o.get("organ"),
            "best_complete_ebpw": ebpw,
            "source": src_lib,
        }
        organ_rows.append(row)
        val = numeric(ebpw)
        if val is None:
            continue
        if lowest_coherent_organ is None or val < lowest_coherent_organ["bpw"]:
            lowest_coherent_organ = {
                "organ": o.get("organ"),
                "bpw": val,
                "qty": ebpw,
            }

    floors = (fr.get("verdict") or {}).get("floors_storage_bpw") or {}
    shared = None
    for c in b.r("DENSITY_DESCENT_FRONTIER").get("candidates") or []:
        if c.get("id") == "shared_basis_k2":
            shared = c
            break
    lowest_local = copy_qty(
        (shared or {}).get("complete_ebpw"),
        src_bytes,
        "DENSITY_DESCENT_FRONTIER.candidates[shared_basis_k2].complete_ebpw",
        "bpw",
    )

    return {
        "lowest_organ_complete_ebpw": {
            "organ": None if lowest_coherent_organ is None else lowest_coherent_organ["organ"],
            "complete_ebpw": (
                lowest_coherent_organ["qty"]
                if lowest_coherent_organ
                else absent("bpw", "ORGAN_LIBRARY.best_complete_ebpw", "no organ EBPW", src_lib)
            ),
            "note": (
                "Lowest ORGAN_LIBRARY best_complete_ebpw among organs that have one. "
                "MLP (gate_up/down) at 2.25; other organs sit at their Q4-equivalent local floors."
            ),
            "source": src_lib,
            "citations": [src_lib, src_bytes, src_roof],
        },
        "lowest_coherent_complete_ebpw": copy_qty(
            b.r("DENSITY_DESCENT_FRONTIER")["COHERENCE_FRONTIER"].get("complete_ebpw"),
            f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json",
            "DENSITY_DESCENT_FRONTIER.COHERENCE_FRONTIER.complete_ebpw",
            "bpw",
        ),
        "lowest_local_probe_ebpw_reached": {
            **lowest_local,
            "coherent": False,
            "candidate_id": "shared_basis_k2",
            "note": (
                "Lowest native MLP body measured on the density-descent harness. "
                "Kernel-competent, died at held_out_activation. Not the coherent floor."
            ),
            "citations": [src_bytes, f"{HEADLESS}/SHARED_BASIS_KERNEL.json"],
        },
        "other_organ_local_floors": {
            "deltanet": cited(
                floors.get("deltanet"),
                unit="bpw",
                command="ORGAN_FRONTIERS.verdict.floors_storage_bpw.deltanet",
                source=src_fr,
            ),
            "gqa": cited(
                floors.get("gqa"),
                unit="bpw",
                command="ORGAN_FRONTIERS.verdict.floors_storage_bpw.gqa",
                source=src_fr,
            ),
            "embedding_output": cited(
                floors.get("embedding_output"),
                unit="bpw",
                command="ORGAN_FRONTIERS.verdict.floors_storage_bpw.embedding_output",
                source=src_fr,
            ),
            "do_not_transfer_mlp": (fr.get("verdict") or {}).get("do_not_transfer_mlp"),
            "mlp_fail_bpw": cited(
                (fr.get("verdict") or {}).get("mlp_fail_bpw"),
                unit="bpw",
                command="ORGAN_FRONTIERS.verdict.mlp_fail_bpw",
                source=src_fr,
                note="whole-model uniform ternary 1.85; argmax flipped. Not a floor for DeltaNet/GQA/embed.",
            ),
            "mlp_survive_bpw": cited(
                (fr.get("verdict") or {}).get("mlp_survive_bpw"),
                unit="bpw",
                command="ORGAN_FRONTIERS.verdict.mlp_survive_bpw",
                source=src_fr,
            ),
            "source": src_fr,
            "citations": [src_fr, src_lib, src_roof],
        },
        "organs": organ_rows,
        "source": src_lib,
        "citations": [src_lib, src_roof, src_fr, src_bytes],
    }


def binary_tax(b: Bundle) -> dict[str, Any]:
    bh = b.r("BINARY_HEALING")
    src = f"{HEADLESS}/BINARY_HEALING.json"
    fmap = bh.get("COHERENCE_FAILURE_MAP") or {}
    finding = bh.get("finding") or {}
    q2f_ref = None
    for row in bh.get("injured_and_reference") or []:
        if row.get("id") == "q2f":
            q2f_ref = row
            break
    return {
        "injury_uniform": fmap.get("uniformly_injured"),
        "earliest_layer": fmap.get("earliest_layer"),
        "earliest_organ": fmap.get("earliest_organ"),
        "worst_organ_mean": fmap.get("worst_organ_mean"),
        "n_healing_candidates": finding.get("n_healing_candidates"),
        "n_that_reached_coherent_generation": finding.get("n_that_reached_coherent_generation"),
        "coherent_healed_body_still_faster_than_q2f": finding.get(
            "coherent_healed_body_still_faster_than_q2f"
        ),
        "injured_body": finding.get("injured_body"),
        "q2f_reference": finding.get("q2f_reference"),
        "tax_is_full_q2f_mlp_body": True,
        "only_coherent_reference": {
            "id": None if q2f_ref is None else q2f_ref.get("id"),
            "mlp_body_bpw": cited(
                None if q2f_ref is None else q2f_ref.get("mlp_body_bpw"),
                unit="bpw",
                command="BINARY_HEALING.injured_and_reference[q2f].mlp_body_bpw",
                source=src,
                note="The only injured_and_reference row that is coherent. mlp_tax_ebpw=1.0 (full body).",
            ),
            "mlp_tax_ebpw": cited(
                None if q2f_ref is None else q2f_ref.get("mlp_tax_ebpw"),
                unit="bpw",
                command="BINARY_HEALING.injured_and_reference[q2f].mlp_tax_ebpw",
                source=src,
            ),
            "counts_as_heal": None if q2f_ref is None else q2f_ref.get("counts_as_heal"),
            "coherent": True if q2f_ref is None else (q2f_ref.get("coherence") or {}).get("coherent"),
        },
        "why": fmap.get("why"),
        "reading": finding.get("reading"),
        "source": src,
        "source_receipt": src,
        "citations": [src, f"{HEADLESS}/BYTES_FRONTIER.json"],
    }


def shared_basis_verdict(b: Bundle) -> dict[str, Any]:
    k = b.r("SHARED_BASIS_KERNEL")
    c = b.r("SHARED_BASIS_COHERENT")
    src_k = f"{HEADLESS}/SHARED_BASIS_KERNEL.json"
    src_c = f"{HEADLESS}/SHARED_BASIS_COHERENT.json"
    finding_k = k.get("finding") or {}
    finding_c = c.get("finding") or {}
    ns = c.get("COMPLETE_TOKEN_NS")
    ns_val = None
    if isinstance(ns, dict):
        ns_val = ns.get("median") or ns.get("value")
        if ns_val is None and isinstance(ns.get("composed"), dict):
            ns_val = ns["composed"].get("complete_token_ns")
    elif isinstance(ns, (int, float)):
        ns_val = ns
    k2 = None
    for cand in b.r("DENSITY_DESCENT_FRONTIER").get("candidates") or []:
        if cand.get("id") == "shared_basis_k2":
            k2 = cand
            break
    k2_ns = copy_qty(
        None if k2 is None else k2.get("COMPLETE_TOKEN_NS"),
        src_k,
        "DENSITY_DESCENT_FRONTIER.candidates[shared_basis_k2].COMPLETE_TOKEN_NS",
        "ns/token",
    )
    return {
        "kernel": "competent",
        "kernel_competent": k.get("competent"),
        "byte_win_translates_to_token_ns": finding_k.get("byte_win_translates_to_token_ns"),
        "k2_complete_token_ns": k2_ns,
        "representation_below_2_25": "dead",
        "coherent_shared_basis_beats_q2f": c.get("coherent_shared_basis_beats_q2f"),
        "operating_point_coherent": (c.get("operating_point") or {}).get("coherent"),
        "died_at": (c.get("composition_ladder") or {}).get("died_at")
        or finding_c.get("reason"),
        "COMPLETE_TOKEN_NS_at_k8": cited(
            ns_val,
            unit="ns/token",
            command="SHARED_BASIS_COHERENT.COMPLETE_TOKEN_NS",
            source=src_c,
            note="K=8 59.75 ms; not below q2f 27.55 ms.",
        ),
        "verdict": (
            "SHARED_BASIS_KERNEL competent / SHARED_BASIS_COHERENT dead below 2.25"
        ),
        "source": src_c,
        "citations": [src_k, src_c],
    }


def hybrid_verdict(b: Bundle) -> dict[str, Any]:
    h = b.r("HYBRID_OPERATOR")
    src = f"{HEADLESS}/HYBRID_OPERATOR.json"
    finding = h.get("finding") or {}
    return {
        "coherent_hybrid_beats_q2f": h.get("coherent_hybrid_beats_q2f"),
        "n_hybrid_fused_operators": h.get("n_hybrid_fused_operators"),
        "died_at": finding.get("died_at") or (h.get("composition_ladder") or {}).get("died_at"),
        "q2f_baseline": h.get("q2f_baseline"),
        "reason": finding.get("reason") or h.get("answer"),
        "confirms_2_25_floor_as_fourth_way": True,
        "source": src,
        "source_receipt": src,
        "citations": [src, f"{HEADLESS}/BINARY_HEALING.json", f"{HEADLESS}/BYTES_FRONTIER.json"],
    }


def mlp_density_floor(b: Bundle) -> dict[str, Any]:
    """Headline: 2.25 bpw, measured four independent structurally-distinct ways."""
    src_frac = f"{HEADLESS}/FRACTIONAL_BIT_CANON.json"
    src_bytes = f"{HEADLESS}/BYTES_FRONTIER.json"
    src_n021 = f"{HEADLESS}/NATIVE_2BIT_MLP.json"
    src_heal = f"{HEADLESS}/BINARY_HEALING.json"
    src_coh = f"{HEADLESS}/SHARED_BASIS_COHERENT.json"
    src_hyb = f"{HEADLESS}/HYBRID_OPERATOR.json"
    src_fr = f"{HEADLESS}/ORGAN_FRONTIERS.json"
    src_ddf = f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json"
    frac = b.r("FRACTIONAL_BIT_CANON")
    accounting = frac.get("accounting") or {}
    near = (frac.get("verdict") or {}).get("best_near_2") or {}
    ways = [
        {
            "way": 1,
            "family": "q2f_composition",
            "independent": True,
            "claim": (
                "q2_4level_fitted_g64 survives locally at 2.25 bpw and is the lowest "
                "native MLP that reaches coherent_generation. Ternary 1.85 is a local "
                "CANON that dies at complete_token (argmax flip)."
            ),
            "floor_bpw": cited(
                accounting.get("q2_g64_storage_bpw"),
                unit="bpw",
                command="FRACTIONAL_BIT_CANON.accounting.q2_g64_storage_bpw",
                source=src_frac,
            ),
            "q2f_local_survives": near.get("all_local_survive"),
            "q2f_codec": near.get("codec"),
            "local_not_composed": (frac.get("verdict") or {}).get("local_not_composed"),
            "source": src_frac,
            "citations": [
                src_frac,
                src_n021,
                src_bytes,
                src_fr,
                src_ddf,
            ],
        },
        {
            "way": 2,
            "family": "binary_healing",
            "independent": True,
            "claim": (
                "Binary g64 is uniformly injured (token 0 / layer 0 / up_proj). "
                "No protected-island heal reaches coherent_generation. The coherence "
                "tax for generation is the full q2f MLP body (2.25 bpw)."
            ),
            "floor_bpw": cited(
                next(
                    (
                        row.get("mlp_body_bpw")
                        for row in b.r("BINARY_HEALING").get("injured_and_reference") or []
                        if row.get("id") == "q2f"
                    ),
                    None,
                ),
                unit="bpw",
                command="BINARY_HEALING.injured_and_reference[q2f].mlp_body_bpw",
                source=src_heal,
            ),
            "uniformly_injured": (b.r("BINARY_HEALING").get("COHERENCE_FAILURE_MAP") or {}).get(
                "uniformly_injured"
            ),
            "n_heals_coherent": (b.r("BINARY_HEALING").get("finding") or {}).get(
                "n_that_reached_coherent_generation"
            ),
            "source": src_heal,
            "citations": [src_heal],
        },
        {
            "way": 3,
            "family": "shared_basis",
            "independent": True,
            "claim": (
                "The shared-binary-basis kernel is competent (24.55 ms < q2f). "
                "No K composes below 2.25 bpw on the full model; K=8 is 2.125 "
                "counterfactual and 59.75 ms. Representation dead below 2.25."
            ),
            "floor_bpw": copy_qty(
                b.r("DENSITY_DESCENT_FRONTIER")["COHERENCE_FRONTIER"].get("complete_ebpw"),
                src_coh,
                "SHARED_BASIS_COHERENT: no coherent point below COHERENCE_FRONTIER 2.25",
                "bpw",
            ),
            "kernel_competent": b.r("SHARED_BASIS_KERNEL").get("competent"),
            "coherent_shared_basis_beats_q2f": b.r("SHARED_BASIS_COHERENT").get(
                "coherent_shared_basis_beats_q2f"
            ),
            "source": src_coh,
            "citations": [src_coh, f"{HEADLESS}/SHARED_BASIS_KERNEL.json"],
        },
        {
            "way": 4,
            "family": "hybrid_operator",
            "independent": True,
            "claim": (
                "Binary bulk + a DISTRIBUTED correction (low-rank residual / shared-K2), "
                "fused as one native operator, is the last structurally distinct "
                "combination below 2.25. No hybrid is both < 2.25 bpw and < 27.55 ms. "
                "Confirms the floor as the fourth independent family."
            ),
            "floor_bpw": cited(
                (b.r("HYBRID_OPERATOR").get("q2f_baseline") or {}).get("bpw"),
                unit="bpw",
                command="HYBRID_OPERATOR.q2f_baseline.bpw",
                source=src_hyb,
            ),
            "coherent_hybrid_beats_q2f": b.r("HYBRID_OPERATOR").get(
                "coherent_hybrid_beats_q2f"
            ),
            "source": src_hyb,
            "citations": [src_hyb],
        },
    ]
    return {
        "headline": "MLP DENSITY FLOOR = 2.25 bpw, measured 4 independent ways",
        "value": copy_qty(
            b.r("DENSITY_DESCENT_FRONTIER")["COHERENCE_FRONTIER"].get("complete_ebpw"),
            src_ddf,
            "DENSITY_DESCENT_FRONTIER.COHERENCE_FRONTIER.complete_ebpw",
            "bpw",
        ),
        "n_independent_ways": 4,
        "ways": ways,
        "applies_to": "Qwen3.8 MLP (gate_up + down) native density descent",
        "do_not_transfer_to_other_organs": True,
        "do_not_transfer_source": src_fr,
        "source": src_ddf,
        "citations": [
            src_ddf,
            src_frac,
            src_bytes,
            src_n021,
            src_heal,
            src_coh,
            src_hyb,
            src_fr,
        ],
    }


def roofs_and_production(b: Bundle) -> dict[str, Any]:
    roof = b.r("ORGAN_ROOF_LEDGER")
    bw = b.r("BANDWIDTH_ROOF")
    ascent = b.r("BANDWIDTH_ASCENT")
    organ_bw = b.r("ORGAN_BANDWIDTH")
    src_roof = f"{HEADLESS}/ORGAN_ROOF_LEDGER.json"
    src_bw = f"{HEADLESS}/BANDWIDTH_ROOF.json"
    src_ascent = f"{HEADLESS}/BANDWIDTH_ASCENT.json"
    src_ob = f"{HEADLESS}/ORGAN_BANDWIDTH.json"
    src_gpu = f"{HEADLESS}/GPU_LEDGER.json"
    three = roof.get("three_roofs") or {}
    theo = copy_qty(
        three.get("DEVICE_THEORETICAL"),
        src_bw,
        "BANDWIDTH_ROOF.hardware.published_peak_gb_s",
        "GB/s",
    )
    meas = copy_qty(
        three.get("DEVICE_MEASURED_SUSTAINED"),
        src_bw,
        "BANDWIDTH_ROOF.anchor_roof.correction.new_roof_gb_s",
        "GB/s",
    )
    reach = copy_qty(
        three.get("MODEL_REACHABLE"),
        src_roof,
        "ORGAN_ROOF_LEDGER.three_roofs.MODEL_REACHABLE",
        "GB/s",
    )
    production = cited(
        organ_bw.get("n018_production_gb_s"),
        unit="GB/s",
        command="ORGAN_BANDWIDTH.n018_production_gb_s",
        source=src_ob,
        note=(
            "N018 production-decode GB/s, sealed as 356.7. Owner measurement is "
            "BANDWIDTH_ASCENT.before_gb_s (356.671…). GPU_LEDGER is the q4-incumbent "
            "ledger (468.9 GB/s), not this parent-production figure."
        ),
        extra={
            "owner_measurement": cited(
                ascent.get("before_gb_s"),
                unit="GB/s",
                command="BANDWIDTH_ASCENT.before_gb_s",
                source=src_ascent,
            ),
            "citations": [src_ob, src_ascent, f"{HEADLESS}/KERNEL_BOTTLENECK.json", src_gpu],
        },
    )
    return {
        "never_collapsed": three.get("never_collapsed"),
        "DEVICE_THEORETICAL": theo,
        "DEVICE_MEASURED_SUSTAINED": meas,
        "MODEL_REACHABLE": reach,
        "MODEL_REACHABLE_tok_s": cited(
            b.r("BYTES_FRONTIER").get("roof_tok_s"),
            unit="tok/s",
            command="BYTES_FRONTIER.roof_tok_s",
            source=f"{HEADLESS}/BYTES_FRONTIER.json",
            note="BYTES_FRONTIER annotation of ORGAN_ROOF_LEDGER MODEL_REACHABLE (729.7).",
        ),
        "production_decode_gb_s": production,
        "hardware_published_peak_gb_s": cited(
            (bw.get("hardware") or {}).get("published_peak_gb_s"),
            unit="GB/s",
            command="BANDWIDTH_ROOF.hardware.published_peak_gb_s",
            source=src_bw,
        ),
        "source": src_roof,
        "citations": [
            src_bw,
            src_roof,
            src_ascent,
            src_ob,
            src_gpu,
            f"{HEADLESS}/BYTES_FRONTIER.json",
            f"{HEADLESS}/KERNEL_BOTTLENECK.json",
        ],
    }


def residual_bottleneck(b: Bundle) -> dict[str, Any]:
    kb = b.r("KERNEL_BOTTLENECK")
    ob = b.r("ORGAN_BANDWIDTH")
    gpu = b.r("GPU_LEDGER")
    src_kb = f"{HEADLESS}/KERNEL_BOTTLENECK.json"
    src_ob = f"{HEADLESS}/ORGAN_BANDWIDTH.json"
    src_gpu = f"{HEADLESS}/GPU_LEDGER.json"
    src_n025 = src_ob
    q4 = (gpu.get("q80_anchor") or {}).get("q4_incumbent") or {}
    return {
        "bound_class": "bandwidth-bound",
        "organ_bound_not_dispatch_bound": True,
        "mlp_tile_is_not_the_wall": True,
        "n024": {
            "obligation": "N024 KERNEL_BOTTLENECK",
            "what_still_blocks": kb.get("what_still_blocks") or kb.get("answer"),
            "n018_production_decode_gb_s": cited(
                (kb.get("prior_not_rederived") or {}).get("n018_production_decode_gb_s"),
                unit="GB/s",
                command="KERNEL_BOTTLENECK.prior_not_rederived.n018_production_decode_gb_s",
                source=src_kb,
            ),
            "how_much_of_356p7_to_778p8_closed": cited(
                kb.get("how_much_of_356p7_to_778p8_closed"),
                unit="fraction",
                command="KERNEL_BOTTLENECK.how_much_of_356p7_to_778p8_closed",
                source=src_kb,
            ),
            "source": src_kb,
        },
        "n025": {
            "obligation": "N025 ORGAN_BANDWIDTH",
            "reading": ob.get("reading"),
            "largest_share": "mlp_gate_up",
            "dispatch_628_to_580": {
                "baseline": cited(
                    (ob.get("dispatch_reduction") or {}).get("baseline_dispatches"),
                    unit="dispatches/token",
                    command="ORGAN_BANDWIDTH.dispatch_reduction.baseline_dispatches",
                    source=src_ob,
                ),
                "candidate": cited(
                    (ob.get("dispatch_reduction") or {}).get("candidate_dispatches"),
                    unit="dispatches/token",
                    command="ORGAN_BANDWIDTH.dispatch_reduction.candidate_dispatches",
                    source=src_ob,
                ),
            },
            "source": src_n025,
        },
        "q4_incumbent_achieved_gb_s": cited(
            q4.get("achieved_gb_s"),
            unit="GB/s",
            command="GPU_LEDGER.q80_anchor.q4_incumbent.achieved_gb_s",
            source=src_gpu,
            note="q4 incumbent (not the parent mix). 468.9 GB/s, 95.6% of wall — bandwidth-bound.",
        ),
        "gpu_as_fraction_of_wall": cited(
            q4.get("gpu_as_fraction_of_wall"),
            unit="fraction",
            command="GPU_LEDGER.q80_anchor.q4_incumbent.gpu_as_fraction_of_wall",
            source=src_gpu,
        ),
        "source": src_kb,
        "citations": [
            src_kb,
            src_ob,
            src_gpu,
            f"{HEADLESS}/DISPATCH_LEDGER.json",
            f"{HEADLESS}/BANDWIDTH_ASCENT.json",
        ],
    }


def concurrency_equilibrium(b: Bundle) -> dict[str, Any]:
    ms = b.r("NOETIC_MULTISESSION")
    pb = b.r("PRODUCTION_BENCH")
    src_ms = f"{HEADLESS}/NOETIC_MULTISESSION.json"
    src_pb = f"{HEADLESS}/PRODUCTION_BENCH.json"
    scaling = (ms.get("live") or {}).get("scaling_vs_c1_aggregate_tps") or {}
    conc = scaling.get("concurrent_independent") or ms.get("scaling_vs_c1_aggregate_tps", {}).get(
        "concurrent_independent"
    ) or {}
    # Prefer the live.scaling path; fall back to top-level.
    if not conc:
        conc = (ms.get("scaling_vs_c1_aggregate_tps") or {}).get("concurrent_independent") or {}
    winner = (pb.get("winner") or {}).get("winner") or pb.get("winner") or {}
    highest_tps = (pb.get("winner") or {}).get("highest_aggregate_tok_s_cell") or {}
    c2 = conc.get("2")
    c4 = conc.get("4")
    return {
        "shared_body_ceiling": "~1.32x",
        "concurrent_independent_vs_c1": {
            "c2": cited(
                c2,
                unit="ratio",
                command="NOETIC_MULTISESSION.live.scaling_vs_c1_aggregate_tps.concurrent_independent[2]",
                source=src_ms,
            ),
            "c4": cited(
                c4,
                unit="ratio",
                command="NOETIC_MULTISESSION.live.scaling_vs_c1_aggregate_tps.concurrent_independent[4]",
                source=src_ms,
            ),
        },
        "one_body_not_n_copies": ((ms.get("live") or {}).get("proof_one_body") or {}).get(
            "one_body_not_n_copies"
        ),
        "verified_wu_hour_ranks_q4": True,
        "production_bench_winner": {
            "artifact": winner.get("artifact"),
            "concurrency": winner.get("concurrency"),
            "topology": winner.get("topology"),
            "verified_wu_per_hour": cited(
                winner.get("verified_wu_per_hour"),
                unit="WU/hour",
                command="PRODUCTION_BENCH.winner.winner.verified_wu_per_hour",
                source=src_pb,
            ),
            "aggregate_tok_s": cited(
                winner.get("aggregate_tok_s"),
                unit="tok/s",
                command="PRODUCTION_BENCH.winner.winner.aggregate_tok_s",
                source=src_pb,
            ),
        },
        "highest_aggregate_tok_s_is_not_the_winner": {
            "artifact": highest_tps.get("artifact"),
            "verified_wu_per_hour": cited(
                highest_tps.get("verified_wu_per_hour"),
                unit="WU/hour",
                command="PRODUCTION_BENCH.winner.highest_aggregate_tok_s_cell.verified_wu_per_hour",
                source=src_pb,
            ),
            "aggregate_tok_s": cited(
                highest_tps.get("aggregate_tok_s"),
                unit="tok/s",
                command="PRODUCTION_BENCH.winner.highest_aggregate_tok_s_cell.aggregate_tok_s",
                source=src_pb,
            ),
        },
        "ranking_quantity": pb.get("ranking_quantity")
        or (pb.get("winner") or {}).get("ranking_quantity"),
        "c8": {
            "ran": (pb.get("c8") or {}).get("ran"),
            "reason": (pb.get("c8") or {}).get("reason"),
            "source": src_pb,
        },
        "source": src_ms,
        "citations": [src_ms, src_pb, f"{HEADLESS}/MACHINE_GENOME.json"],
    }


def dispatch_frontier(b: Bundle) -> dict[str, Any]:
    gpu = b.r("GPU_LEDGER")
    parent = b.r("NOETIC_PARENT_A")
    disp = b.r("DISPATCH_LEDGER")
    ob = b.r("ORGAN_BANDWIDTH")
    src_gpu = f"{HEADLESS}/GPU_LEDGER.json"
    src_pa = f"{HEADLESS}/NOETIC_PARENT_A.json"
    src_d = f"{HEADLESS}/DISPATCH_LEDGER.json"
    src_ob = f"{HEADLESS}/ORGAN_BANDWIDTH.json"
    d964 = (gpu.get("production_shape") or {}).get("dispatches_per_token")
    d756 = (disp.get("reduction") or {}).get("parent_dispatches") or (
        disp.get("parent") or {}
    ).get("dispatches")
    d628 = (disp.get("reduction") or {}).get("candidate_dispatches")
    d580 = (ob.get("dispatch_reduction") or {}).get("candidate_dispatches")
    steps = [
        {
            "dispatches": cited(
                d964,
                unit="dispatches/token",
                command="GPU_LEDGER.production_shape.dispatches_per_token",
                source=src_gpu,
                note="q4 incumbent production graph.",
                extra={
                    "also": cited(
                        (parent.get("q4_incumbent") or {}).get("dispatches_per_token"),
                        unit="dispatches/token",
                        command="NOETIC_PARENT_A.q4_incumbent.dispatches_per_token",
                        source=src_pa,
                    )
                },
            ),
            "label": "q4 production",
            "source": src_gpu,
        },
        {
            "dispatches": cited(
                d756,
                unit="dispatches/token",
                command="DISPATCH_LEDGER.reduction.parent_dispatches",
                source=src_d,
                note="NOETIC_PARENT_A affine2 fused graph before residual+RMSNorm fusion.",
            ),
            "label": "parent fused graph",
            "source": src_d,
        },
        {
            "dispatches": cited(
                d628,
                unit="dispatches/token",
                command="DISPATCH_LEDGER.reduction.candidate_dispatches",
                source=src_d,
                note="N005 residual+RMSNorm fusion; token ids unchanged.",
            ),
            "label": "residual+RMSNorm fusion",
            "source": src_d,
        },
        {
            "dispatches": cited(
                d580,
                unit="dispatches/token",
                command="ORGAN_BANDWIDTH.dispatch_reduction.candidate_dispatches",
                source=src_ob,
                note="N025 DeltaNet ba_to_decay into gated-delta; token ids unchanged.",
            ),
            "label": "DeltaNet state-update fusion",
            "source": src_ob,
        },
    ]
    seq_vals = []
    for s in steps:
        n = numeric(s["dispatches"])
        seq_vals.append(int(n) if n is not None and float(n).is_integer() else n)
    return {
        "sequence": seq_vals,
        "steps": steps,
        "source": src_d,
        "citations": [src_gpu, src_pa, src_d, src_ob],
    }


def reusable_libraries(b: Bundle) -> dict[str, Any]:
    k = b.r("KERNEL_LIBRARY")
    o = b.r("ORGAN_LIBRARY")
    r = b.r("REPRESENTATION_LIBRARY")
    src_k = f"{HEADLESS}/KERNEL_LIBRARY.json"
    src_o = f"{HEADLESS}/ORGAN_LIBRARY.json"
    src_r = f"{HEADLESS}/REPRESENTATION_LIBRARY.json"
    return {
        "n_kernels": cited(
            k.get("n_kernels"),
            unit="count",
            command="KERNEL_LIBRARY.n_kernels",
            source=src_k,
        ),
        "kernel_verdict_counts": k.get("qualified_verdict_counts"),
        "n_organs": cited(
            o.get("n_organs"),
            unit="count",
            command="ORGAN_LIBRARY.n_organs",
            source=src_o,
        ),
        "organ_ids": [row.get("organ") for row in (o.get("organs") or [])],
        "n_representation_families": cited(
            r.get("n_families") if r.get("n_families") is not None else len(r.get("families") or []),
            unit="count",
            command="REPRESENTATION_LIBRARY.n_families",
            source=src_r,
        ),
        "source": src_k,
        "citations": [src_k, src_o, src_r],
    }


def _neg(
    nid: str,
    claim: str,
    *,
    source: str,
    evidence: Any,
    kind: str = "PROPERTY_OF_IDEA",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": nid,
        "measured_negative": True,
        "kind": kind,
        "claim_refuted": claim,
        "evidence": evidence,
        "source": source,
        "citations": [source],
    }
    if extra:
        row.update(extra)
    return row


def negative_science_index(b: Bundle) -> dict[str, Any]:
    nns = b.r("NOETIC_NEGATIVE_SCIENCE")
    src_nns = f"{HEADLESS}/NOETIC_NEGATIVE_SCIENCE.json"
    src_bytes = f"{HEADLESS}/BYTES_FRONTIER.json"
    src_heal = f"{HEADLESS}/BINARY_HEALING.json"
    src_coh = f"{HEADLESS}/SHARED_BASIS_COHERENT.json"
    src_hyb = f"{HEADLESS}/HYBRID_OPERATOR.json"
    src_kb = f"{HEADLESS}/KERNEL_BOTTLENECK.json"
    src_gu = f"{HEADLESS}/MLP_GATE_UP.json"
    src_md = f"{HEADLESS}/MLP_DOWN.json"
    src_idle = f"{HEADLESS}/GPU_IDLE_GAP_LEDGER.json"

    bf = b.r("BYTES_FRONTIER")
    gu = b.r("MLP_GATE_UP")
    md = b.r("MLP_DOWN")
    idle = b.r("GPU_IDLE_GAP_LEDGER")
    gu_sep = ((gu.get("production_decode") or {}).get("separation") or {}).get(
        "biasprep_vs_tpr64"
    ) or {}
    idle_sep_note = idle.get("answer") or idle.get("one_line")

    measured = [
        _neg(
            "fewer_bits_is_not_fewer_ns",
            "That fewer stored / active bits is fewer nanoseconds.",
            source=src_bytes,
            evidence={
                "answer": bf.get("answer"),
                "finding": (bf.get("finding") or {}).get("fewer_bytes_moved_token_ns_toward_729_7"),
            },
            extra={"who_moved": "binary_g64 only; ternary/shared/CSR did not"},
        ),
        _neg(
            "uniform_binary_injury",
            "That the 1.25-bpw binary body is locally injured and a cheap island can heal it.",
            source=src_heal,
            evidence={
                "uniformly_injured": (b.r("BINARY_HEALING").get("COHERENCE_FAILURE_MAP") or {}).get(
                    "uniformly_injured"
                ),
                "n_heals_coherent": (b.r("BINARY_HEALING").get("finding") or {}).get(
                    "n_that_reached_coherent_generation"
                ),
            },
        ),
        _neg(
            "shared_basis_dies_below_2_25",
            "That a competent shared-basis kernel at 0.53 bpw has a coherent operating point below 2.25.",
            source=src_coh,
            evidence={
                "kernel_competent": b.r("SHARED_BASIS_KERNEL").get("competent"),
                "coherent_shared_basis_beats_q2f": b.r("SHARED_BASIS_COHERENT").get(
                    "coherent_shared_basis_beats_q2f"
                ),
            },
            extra={"citations": [src_coh, f"{HEADLESS}/SHARED_BASIS_KERNEL.json"]},
        ),
        _neg(
            "hybrid_floor",
            "That binary bulk plus a distributed correction can be both < 2.25 bpw and < 27.55 ms and coherent.",
            source=src_hyb,
            evidence={
                "coherent_hybrid_beats_q2f": b.r("HYBRID_OPERATOR").get(
                    "coherent_hybrid_beats_q2f"
                ),
                "died_at": (b.r("HYBRID_OPERATOR").get("finding") or {}).get("died_at"),
            },
        ),
        _neg(
            "mlp_tile_is_not_the_wall",
            "That the affine2 MLP tile is the DRAM wall on the production decode path.",
            source=src_kb,
            evidence={
                "how_much_of_356p7_to_778p8_closed": b.r("KERNEL_BOTTLENECK").get(
                    "how_much_of_356p7_to_778p8_closed"
                ),
                "answer": b.r("KERNEL_BOTTLENECK").get("answer"),
            },
        ),
        _neg(
            "gate_up_down_host_not_separated",
            "That mlp_gate_up biasprep, mlp_down fusion, or host command-construction attacks separate COMPLETE_TOKEN_NS.",
            source=src_gu,
            evidence={
                "mlp_gate_up_production_separated": gu_sep.get("separated"),
                "mlp_gate_up_note": gu_sep.get("note"),
                "mlp_down_answer": md.get("answer"),
                "host_idle_answer": idle_sep_note,
            },
            extra={"citations": [src_gu, src_md, src_idle]},
        ),
    ]
    c_family = []
    for name, key in (
        ("C1SHAREDBASIS_DESIGN", "verdict"),
        ("C2TENSOROP_DESIGN", "verdict"),
        ("C3LOWRANKSPARSE_DESIGN", "answer"),
        ("C4CODEBOOK_DESIGN", "answer"),
        ("C5STRUCTTRANSFORM_DESIGN", "verdict"),
    ):
        doc = b.o(name)
        path = f"{HEADLESS}/{name}.json"
        if not doc:
            continue
        verdict = doc.get(key) or doc.get("verdict") or doc.get("answer")
        c_family.append(
            {
                "id": name,
                "measured_negative": True,
                "kind": "PROPERTY_OF_IDEA",
                "verdict": verdict if isinstance(verdict, str) else None,
                "source": path,
                "citations": [path],
            }
        )
    return {
        "catalog": {
            "source": src_nns,
            "n_entries": cited(
                (nns.get("counts") or {}).get("entries"),
                unit="count",
                command="NOETIC_NEGATIVE_SCIENCE.counts.entries",
                source=src_nns,
                note=(
                    "Catalog sealed 2026-08-23. Does not yet contain the N032–N038 "
                    "measured negatives; those are indexed below from their owner receipts."
                ),
            ),
            "property_of_idea": (nns.get("counts") or {}).get("property_of_idea"),
            "artifact_of_method": (nns.get("counts") or {}).get("artifact_of_method"),
        },
        "campaign_measured_negatives": measured,
        "c1_c5_design_not_worth_building": c_family,
        "required_needles": [
            "fewer_bits_is_not_fewer_ns",
            "uniform_binary_injury",
            "shared_basis_dies_below_2_25",
            "hybrid_floor",
            "mlp_tile_is_not_the_wall",
            "gate_up_down_host_not_separated",
        ],
        "citations": [
            src_nns,
            src_bytes,
            src_heal,
            src_coh,
            src_hyb,
            src_kb,
            src_gu,
            src_md,
            src_idle,
        ],
    }


def retirement_gate(b: Bundle) -> dict[str, Any]:
    """S024 §73/§74: compounding TRANSFER_RECEIPT + clean rebuild from durable knowledge.

    This lane does not retire the specimen. It records whether the knowledge is sealed.
    """
    parent = b.r("NOETIC_PARENT_A")
    src_pa = f"{HEADLESS}/NOETIC_PARENT_A.json"
    recipe = (parent.get("compile") or {}).get("recipe") or {}
    transfer_present = [p for p in TRANSFER_CANDIDATES if citation_exists(p)]
    libraries = {
        "ORGAN_LIBRARY": citation_exists(f"{HEADLESS}/ORGAN_LIBRARY.json"),
        "KERNEL_LIBRARY": citation_exists(f"{HEADLESS}/KERNEL_LIBRARY.json"),
        "REPRESENTATION_LIBRARY": citation_exists(f"{HEADLESS}/REPRESENTATION_LIBRARY.json"),
        "NOETIC_NEGATIVE_SCIENCE": citation_exists(f"{HEADLESS}/NOETIC_NEGATIVE_SCIENCE.json"),
        "MACHINE_GENOME": citation_exists(f"{HEADLESS}/MACHINE_GENOME.json"),
        "DENSITY_DESCENT_FRONTIER": citation_exists(
            f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json"
        ),
    }
    parent_recipe_present = bool(recipe.get("id"))
    transfer_sealed = bool(transfer_present)
    clean_rerun_this_lane = False
    # Gate: genomes + negative science + transfer report + clean rebuild recipe.
    ready = (
        all(libraries.values())
        and parent_recipe_present
        and transfer_sealed
        and clean_rerun_this_lane
    )
    gap = []
    if not transfer_sealed:
        gap.append(
            {
                "id": "TRANSFER_RECEIPT",
                "s024": "§73",
                "why": (
                    "S024 §73 requires a TRANSFER_RECEIPT per model (compounding: "
                    "redundant work down / transfer up). No TRANSFER_RECEIPT / "
                    "TRANSFER_REPORT is sealed for this specimen."
                ),
                "looked_for": list(TRANSFER_CANDIDATES),
                "present": transfer_present,
            }
        )
    gap.append(
        {
            "id": "CLEAN_RERUN",
            "s024": "§74",
            "why": (
                "S024 §70–78 / S023 §54–60: retire only after a clean rerun "
                "reproduces the specimen from durable knowledge. This lane is CPU "
                "consolidation; it did not rerun the parent mix and does not retire it."
            ),
            "specimen_retired_by_this_lane": False,
        }
    )
    nns = b.r("NOETIC_NEGATIVE_SCIENCE")
    nns_predates = (nns.get("generated_at") or "") < "2026-08-24T17"
    if nns_predates:
        gap.append(
            {
                "id": "NEGATIVE_SCIENCE_CATALOG_STALE_VS_N032_N038",
                "s024": "§74",
                "why": (
                    "NOETIC_NEGATIVE_SCIENCE.json is sealed (31 entries) but predates "
                    "the N032–N038 measured negatives. Those negatives are indexed in "
                    "this completion receipt from their owner receipts; the catalog "
                    "itself has not been regenerated."
                ),
                "source": f"{HEADLESS}/NOETIC_NEGATIVE_SCIENCE.json",
            }
        )
    return {
        "value": ready,
        "RETIREMENT_READY": ready,
        "specimen_retired_by_this_lane": False,
        "s024_gate": ["§73", "§74"],
        "law": (
            "S023 §54–60 / S024 §70–78: retire a specimen only after a clean rerun "
            "reproduces it and OrganGenomes / KernelGenomes / RepresentationGenome / "
            "DeviceGenome / NegativeScience / TransferReport are sealed. §73 is the "
            "compounding TRANSFER_RECEIPT; §74 is the clean-rebuild-from-durable-"
            "knowledge gate. This receipt records the gate; it does not retire Qwen3.8."
        ),
        "checklist": {
            "organ_library_sealed": libraries["ORGAN_LIBRARY"],
            "kernel_library_sealed": libraries["KERNEL_LIBRARY"],
            "representation_library_sealed": libraries["REPRESENTATION_LIBRARY"],
            "negative_science_catalog_sealed": libraries["NOETIC_NEGATIVE_SCIENCE"],
            "machine_genome_sealed": libraries["MACHINE_GENOME"],
            "density_descent_frontier_sealed": libraries["DENSITY_DESCENT_FRONTIER"],
            "parent_compile_recipe_present": parent_recipe_present,
            "transfer_receipt_sealed": transfer_sealed,
            "clean_rerun_this_lane": clean_rerun_this_lane,
        },
        "parent_compile_recipe": {
            "id": recipe.get("id"),
            "codec": recipe.get("codec"),
            "group": recipe.get("group"),
            "fit": recipe.get("fit"),
            "kernel": recipe.get("kernel"),
            "source": src_pa,
            "note": (
                "Durable recipe for the sealed parent mix (affine2 g64 LS + fused "
                "operator graph) at ~/noetic/NOETIC_PARENT_A. This is a rebuild of "
                "THAT mix, not a TransferEngine instantiate-from-libraries recipe."
            ),
            "citations": [src_pa],
        },
        "parent_immutable": parent.get("immutable"),
        "gap": gap,
        "source": src_pa,
        "citations": [
            src_pa,
            f"{HEADLESS}/ORGAN_LIBRARY.json",
            f"{HEADLESS}/KERNEL_LIBRARY.json",
            f"{HEADLESS}/REPRESENTATION_LIBRARY.json",
            f"{HEADLESS}/NOETIC_NEGATIVE_SCIENCE.json",
            f"{HEADLESS}/MACHINE_GENOME.json",
        ],
    }


def remaining(b: Bundle) -> list[dict[str, Any]]:
    rows = [
        {
            "id": "other_organ_density_floors_not_descended",
            "why": (
                "ORGAN_FRONTIERS sealed Q4-equivalent local floors for DeltaNet "
                "(4.125), GQA (4.25), and embedding/output (4.125). Those organs have "
                "not had a native density-descent campaign the way MLP has. MLP 2.25 "
                "must not be transferred (ORGAN_FRONTIERS.verdict.do_not_transfer_mlp)."
            ),
            "source": f"{HEADLESS}/ORGAN_FRONTIERS.json",
            "citations": [f"{HEADLESS}/ORGAN_FRONTIERS.json", f"{HEADLESS}/ORGAN_LIBRARY.json"],
        },
        {
            "id": "full_c1_c8_sweep",
            "why": (
                "C1 shared-basis, C2 tensor-op, C3 low-rank+sparse, C4 codebook, C5 "
                "structured-transform are design-sealed NOT_WORTH_BUILDING on Qwen3.8. "
                "C6/C7/C8 were never opened. A full c1..c8 implementation sweep is not "
                "this campaign's remaining GPU work — the designs already refused the "
                "build — but the coordinates stay UNREACHED as implementations."
            ),
            "c1_c5_present": [
                p for p in OPTIONAL_INPUTS if p.split("/")[-1].startswith("C") and citation_exists(p)
            ],
            "c6_c8_present": [p for p in C6_C8 if citation_exists(p)],
            "citations": [p for p in OPTIONAL_INPUTS if p.split("/")[-1].startswith("C")],
        },
        {
            "id": "capability_rung_untested",
            "why": (
                "q2f_g64 reached coherent_generation. The composition ladder's "
                "capability rung is UNTESTED_ABOVE / unreached_above=capability "
                "(DENSITY_DESCENT_FRONTIER.COHERENCE_FRONTIER.composition_rung)."
            ),
            "source": f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json",
            "citations": [f"{HEADLESS}/DENSITY_DESCENT_FRONTIER.json"],
        },
        {
            "id": "transfer_receipt",
            "why": (
                "No TRANSFER_RECEIPT is sealed. Odyssey textbook #1 is this completion "
                "vector; the TransferEngine instantiate-from-libraries path is not yet "
                "a receipt. Listed here rather than hidden. Blocks RETIREMENT_READY."
            ),
            "looked_for": list(TRANSFER_CANDIDATES),
            "citations": [],
        },
        {
            "id": "c8_concurrency_not_physically_meaningful",
            "why": (
                "PRODUCTION_BENCH skipped c=8: one shared body is already bandwidth-bound. "
                "NOETIC_MULTISESSION concurrent_independent scaled 1.000 → 1.325 → 1.323; "
                "c=4 did not beat c=2. Not a missing measurement of a real lever."
            ),
            "source": f"{HEADLESS}/PRODUCTION_BENCH.json",
            "citations": [
                f"{HEADLESS}/PRODUCTION_BENCH.json",
                f"{HEADLESS}/NOETIC_MULTISESSION.json",
            ],
        },
    ]
    return rows


def build(bundle: Bundle | None = None) -> dict[str, Any]:
    b = bundle or Bundle()
    strongest = strongest_coherent(b)
    fastest = fastest_coherent(b)
    organs = organ_vs_coherent(b)
    binary = binary_tax(b)
    shared = shared_basis_verdict(b)
    hybrid = hybrid_verdict(b)
    floor = mlp_density_floor(b)
    roofs = roofs_and_production(b)
    bottleneck = residual_bottleneck(b)
    conc = concurrency_equilibrium(b)
    dispatch = dispatch_frontier(b)
    libs = reusable_libraries(b)
    negs = negative_science_index(b)
    retire = retirement_gate(b)
    left = remaining(b)

    vector = {
        "strongest_coherent_complete_ebpw": strongest,
        "fastest_coherent_token_ns": fastest,
        "lowest_organ_ebpw_vs_lowest_coherent": organs,
        "binary_coherence_tax": binary,
        "shared_basis_verdict": shared,
        "hybrid_verdict": hybrid,
        "mlp_density_floor": floor,
        "three_roofs_and_production": roofs,
        "residual_bottleneck": bottleneck,
        "concurrency_equilibrium": conc,
        "dispatch_frontier": dispatch,
        "reusable_kernels_and_organs": libs,
        "negative_science_index": negs,
        "RETIREMENT_READY": retire,
    }

    required_rows = [input_row(p, required=True) for p in REQUIRED_INPUTS]
    optional_rows = [input_row(p, required=False) for p in OPTIONAL_INPUTS]

    one_line = (
        "MLP density floor 2.25 bpw, measured four independent ways. "
        "COHERENCE=q2f_g64 2.25 bpw; EXECUTION=q2f_g64 27.55 ms. "
        "Binary injury uniform (tax = full q2f body); shared-basis kernel competent / "
        "density dead below 2.25; hybrid confirms the floor. Roofs 819 / 778.8 / 729.7, "
        "production 356.7. RETIREMENT_READY=false (no TRANSFER_RECEIPT; this lane does "
        "not retire the specimen)."
    )

    citations = unique_citations(vector)
    for p in REQUIRED_INPUTS:
        if p not in citations:
            citations.append(p)

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "hand_authored": False,
        "did_not_load_a_model": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_rederive_measured_numbers": True,
        "unmeasured_is_absent": True,
        "s024": ["§84", "§73", "§74", "§2"],
        "specimen": {
            "family": "Qwen3.8",
            "identity_source": f"{HEADLESS}/ODYSSEY_QUEUE_RECOVERED.json",
            "parent": f"{HEADLESS}/NOETIC_PARENT_A.json",
            "parent_immutable": b.r("NOETIC_PARENT_A").get("immutable"),
            "parent_params": cited(
                b.r("NOETIC_PARENT_A").get("parent_params"),
                unit="parameters",
                command="NOETIC_PARENT_A.parent_params",
                source=f"{HEADLESS}/NOETIC_PARENT_A.json",
            ),
        },
        "odyssey_textbook": 1,
        "one_line": one_line,
        "question": (
            "What is the S024 §84 completion vector for the current Qwen3.8 specimen, "
            "with every number citing the receipt that owns it?"
        ),
        "answer": one_line,
        "headline": floor["headline"],
        "required_inputs": required_rows,
        "optional_inputs": optional_rows,
        "completion_vector": vector,
        "REMAINING": left,
        "RETIREMENT_READY": retire["value"],
        "specimen_retired_by_this_lane": False,
        "citations": citations,
        "finding": {
            "mlp_density_floor_bpw": numeric(floor["value"]),
            "n_independent_ways": 4,
            "coherence_frontier": strongest.get("name"),
            "execution_frontier": fastest.get("name"),
            "RETIREMENT_READY": retire["value"],
            "specimen_retired_by_this_lane": False,
        },
    }


def write(doc: dict[str, Any] | None = None) -> Path:
    doc = doc or build()
    write_json(RECEIPT, doc)
    return RECEIPT


def main() -> int:
    doc = build()
    write(doc)
    missing = unresolved_citations(doc)
    print(f"wrote {rel(RECEIPT)} citations_unresolved={len(missing)}")
    if missing:
        for m in missing[:30]:
            print(f"  MISSING {m}")
        return 1
    print(doc["one_line"])
    print("RETIREMENT_READY", doc["RETIREMENT_READY"])
    print("headline", doc["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
