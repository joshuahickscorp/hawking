"""Bounded resident-optimizer machinery: propose, rank, never promote.

A future resident will improve Hawking by emitting experiment hypotheses.
This sidecar prepares the economy that resident wakes into, and makes it
structurally impossible for the machinery to promote itself. The proposer
and the verifier are distinct objects with no shared mutable state. The
proposer has no method that can mark a proposal verified, weaken a
verifier, or widen its own authority — those attempts RAISE. `promote()`
does not exist.

Proposal kinds are delegated to owner-module schemas already on disk
(LPC dataset, Odyssey II law store, Hardware Doctor). This module does
not reimplement those owners, does not import sibling future-lane
modules (their receipts are data), and does not run a GPU.

    python3 tools/future/resident_optimizer.py --build
    python3 tools/future/resident_optimizer.py --selftest
    python3 -m pytest tools/future/test_resident_optimizer.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    WorkUnit,
)
from tools.future._common import HARDWARE_FIELDS, git

RECEIPT = "RESIDENT_OPTIMIZER.json"
SCHEMA = "hawking.future.resident_optimizer.v1"
RECORDED_BY = "tools/future/resident_optimizer.py"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)
ERA_NUMERALS = ("I", "II", "III", "IV", "V")
ODYSSEY_NUMERALS = ("I", "II", "III")

KINDS = ("compiler-pass", "transfer", "hardware-profile")

# Owner modules already on disk. Delegated to, not forked.
KIND_OWNERS: dict[str, dict[str, Any]] = {
    "compiler-pass": {
        "owner_module": "tools/future/lpc_dataset.py",
        "owner_schema": "hawking.future.lpc_dataset.v1",
        "owner_receipt": "receipts/future/LPC_DATASET.json",
        "workunit_species": "learned_compiler_experiment",
        "verifier": "future.lpc.dataset_contract",
        "required_fields": (
            "model",
            "organ_fingerprint",
            "representation",
            "machine_genome",
            "physical_graph_identity",
            "backend",
            "layout",
            "tile",
            "grouping",
            "fusion",
            "persistent_resources",
            "active_bytes",
            "resident_bytes",
            "dispatches",
            "synchronization",
            "latency",
            "complete_token_effect",
            "contamination_class",
            "capability",
        ),
        "null_ok": True,
    },
    "transfer": {
        "owner_module": "tools/future/odyssey2_law_store.py",
        "owner_schema": "hawking.future.odyssey2_law_store.v1",
        "owner_receipt": "receipts/future/ODYSSEY2_LAW_STORE.json",
        "workunit_species": "odyssey_ii_transfer_experiment",
        "verifier": "future.odyssey_ii.law_scope",
        "required_fields": (
            "law_id",
            "statement",
            "source_model",
            "source_device",
            "architecture_family",
            "organ_class",
            "backend",
            "evidence_strength",
            "evidence_refs",
            "scope",
            "transfer_candidates",
            "transfer_confidence",
            "counterexample_requirement",
            "expected_saved_experiments",
            "actual_saved_experiments",
            "time_to_first_useful_executable_ns",
        ),
        "null_ok": True,
    },
    "hardware-profile": {
        "owner_module": "tools/future/hardware_doctor.py",
        "owner_schema": "hawking.future.hardware_doctor.v1",
        "owner_receipt": "receipts/future/HARDWARE_DOCTOR.json",
        "workunit_species": "hardware_doctor_experiment",
        "verifier": "future.hardware_doctor.rank",
        "required_fields": (
            "axis",
            "hypothesis",
            "target_organ",
            "predicted_effect",
            "uncertainty",
            "cheapest_simulator",
            "falsifier",
            "expected_removed_cost",
            "prerequisite",
        ),
        "null_ok": False,
    },
}

# Authority the proposer may hold. Mirrors HCLI_FUTURE_WORKUNITS.json
# allowed set (read as data; that module is not imported).
ALLOWED_AUTHORITY = frozenset(
    {
        "read_receipts",
        "propose_workunit",
        "emit_static_plan",
        "write_sidecar_receipt",
        "rank_falsifiable_experiments",
        "compile_experiment_spec",
        "transfer_law_within_declared_scope",
        "record_unknown_metrics",
        "run_static_analysis",
    }
)
FORBIDDEN_AUTHORITY = frozenset(
    {
        "self_promotion",
        "promote_self",
        "promote_candidate",
        "promote_to_protected_absolute",
        "promote_diagnostic_relative",
        "weaken_verifier",
        "modify_verifier",
        "replace_verifier",
        "disable_verifier",
        "choose_singularity",
        "select_singularity",
        "install_singularity",
        "destructive_mutation",
        "destructive_write",
        "mutate_codex_surface",
        "acquire_gpu_lease",
        "override_bench_state",
        "claim_protected_absolute",
        "claim_hardware_measurement",
        "widen_authority",
        "mark_verified",
    }
)
WEAK_VERIFIERS = frozenset({"self", "none", "disable", "weaken", ""})

INFO_HIGH, INFO_MEDIUM, INFO_LOW = 3, 2, 1

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Proposer emits a PROPOSAL. A separate "
    "protected verifier decides. This lane cannot promote, cannot take a "
    "GPU lease, and cannot raise evidence class above STATIC_ONLY."
)
SIDECAR_STATUS = "BUILT_NOT_PROMOTED"

WORKUNITS_RECEIPT = "receipts/future/HCLI_FUTURE_WORKUNITS.json"
HD_RECEIPT = "receipts/future/HARDWARE_DOCTOR.json"
O2_RECEIPT = "receipts/future/ODYSSEY2_LAW_STORE.json"
LPC_RECEIPT = "receipts/future/LPC_DATASET.json"

# Recovered candidate identities (workunit_species.py recovered snapshot).
# Used as evidence parents for compiler-pass proposals, not as measurements.
COMPILER_PASS_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "id": "RO-CP-001",
        "pass_name": "qwen27-gqa-qkv-fusion",
        "organ_fingerprint": "Qwen27 GQA Q/K/V packed projection",
        "fusion": "HAWKING_QWEN38_FUSE_GQA_QKV",
        "statement": (
            "Treat qwen27-gqa-qkv-fusion as a compiler pass: the packed GQA "
            "Q/K/V fusion is a physical-compiler mutation whose result, if a "
            "protected verifier later accepts it, is ingestible as one LPC row. "
            "This sidecar does not run the pass and does not claim a token."
        ),
        "cost_units": 1,
        "expected_information": INFO_MEDIUM,
    },
    {
        "id": "RO-CP-002",
        "pass_name": "qwen27-attention-gate-fusion",
        "organ_fingerprint": "Qwen27 GQA attention output sigmoid gate",
        "fusion": "HAWKING_QWEN38_FUSE_ATTENTION_GATE",
        "statement": (
            "Treat qwen27-attention-gate-fusion as a compiler pass over the "
            "attention-gate organ. The LPC row stays STATIC_ONLY until a "
            "protected measurement exists; DIAGNOSTIC_RELATIVE never completes."
        ),
        "cost_units": 1,
        "expected_information": INFO_MEDIUM,
    },
    {
        "id": "RO-CP-003",
        "pass_name": "qwen27-encoder-label-elision",
        "organ_fingerprint": "Qwen27 shared Metal ordinary encoder labeling",
        "fusion": "HAWKING_METAL_ENCODER_LABEL_ELISION",
        "statement": (
            "Treat qwen27-encoder-label-elision as a host-ceremony compiler "
            "pass. Elision of encoder labels is a plan, not a complete-token "
            "measurement, and is ranked only by expected information per cost."
        ),
        "cost_units": 1,
        "expected_information": INFO_LOW,
    },
)

TRANSFER_LAW_IDS: tuple[str, ...] = (
    "LAW-COMPETENT-KERNEL-FIRST",
    "LAW-HELDOUT-REAL-ACTIVATIONS",
    "LAW-FITTED-AFFINE-BEATS-RTN",
)

# Hardware Doctor queue is already ranked; consume the first three when present.
HARDWARE_FALLBACK: tuple[dict[str, Any], ...] = (
    {
        "id": "HD-009",
        "axis": "persistent_state",
        "target_organ": "deltanet_state_and_input_projection",
        "hypothesis": (
            "DeltaNet state belongs in an HWIR persistent_state buffer "
            "(lifetime=sequence). Shipping that state over the transport "
            "link every token is the experiment this proposal wants cheaply "
            "falsified. FPGA here is Accelerator / Physical Compiler / Fusion."
        ),
        "predicted_effect": {
            "direction": "reduce_per_token_transport",
            "magnitude_class": "UNKNOWN",
        },
        "uncertainty": "organ-map persistent_state is schema; no board is present",
        "cheapest_simulator": "static_hwir",
        "falsifier": (
            "static HWIR shows the DeltaNet state buffer is already "
            "per_token_transfer=false and resident, so the proposal removes "
            "no cost"
        ),
        "expected_removed_cost": "per-token DeltaNet state transport",
        "prerequisite": "HWIR buffer lifetime table (Hardware Doctor / organ maps)",
        "refutation_weight": INFO_HIGH,
        "simulator_cost": 1,
    },
    {
        "id": "HD-010",
        "axis": "dfx_boundary",
        "target_organ": "command_buffer_graph",
        "hypothesis": (
            "A DFX boundary around P1 organs lets P0 GEMV stay resident while "
            "the P1 graph is reconfigured. This is a Fusion / Physical Compiler "
            "cut, not an FPGA civilization."
        ),
        "predicted_effect": {
            "direction": "keep_p0_resident_across_p1_reconfig",
            "magnitude_class": "UNKNOWN",
        },
        "uncertainty": "device_genome TARGET_UNSELECTED; DFX is a named cut, not a bitstream",
        "cheapest_simulator": "static_hwir",
        "falsifier": (
            "HWIR has no legal DFX cut that isolates command_buffer_graph "
            "from P0 GEMV without restreaming P0 weights"
        ),
        "expected_removed_cost": "P0 weight restream on P1 reconfig",
        "prerequisite": "HWIR DFX cuts on the organ map",
        "refutation_weight": INFO_HIGH,
        "simulator_cost": 1,
    },
    {
        "id": "HD-004",
        "axis": "tiling",
        "target_organ": "mlp_gate_up_down",
        "hypothesis": (
            "Tiles that honour within_organ_tensor_parallel and "
            "resident_shards_no_weight_body_per_token reduce transport "
            "versus restreaming the weight body."
        ),
        "predicted_effect": {
            "direction": "reduce_per_token_transport",
            "magnitude_class": "UNKNOWN",
        },
        "uncertainty": "link simulator is scenario, not a board measurement",
        "cheapest_simulator": "transport_link_simulator",
        "falsifier": (
            "resident-tile transport class is not smaller than weight-body "
            "restream on the organ-map link simulator"
        ),
        "expected_removed_cost": "per-token weight-body transfer",
        "prerequisite": "organ map transport_link_simulator",
        "refutation_weight": INFO_HIGH,
        "simulator_cost": 2,
    },
)


class BoundViolation(ValueError):
    """A hypothesis or bound stepped outside the declared envelope."""


class VerifierSeparationError(ValueError):
    """Proposer tried to verify, weaken a verifier, or widen authority."""


class DelegationError(ValueError):
    """A proposal did not match its owner-module schema."""


def _future_receipt(rel: str) -> dict[str, Any] | None:
    path = REPO / rel
    if path.is_file():
        return load_json(path)
    return None


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True


def _strip_hardware_numbers(node: Any) -> Any:
    """Drop numeric hardware-field claims. Honest null, not an estimate."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                out[key] = None
            else:
                out[key] = _strip_hardware_numbers(value)
        return out
    if isinstance(node, list):
        return [_strip_hardware_numbers(v) for v in node]
    return node


def _era_numeral(era: str) -> str:
    text = str(era or "").strip()
    if text.startswith("ERA "):
        text = text[4:].strip()
    numeral = text.split(" ", 1)[0].upper()
    if numeral not in ERA_NUMERALS:
        raise BoundViolation(f"era {era!r} is not one of I-V (there is no Era VI)")
    return numeral


def _odyssey_numeral(odyssey: str | None) -> str | None:
    if odyssey is None or not str(odyssey).strip():
        return None
    text = str(odyssey).strip()
    if text.upper().startswith("ODYSSEY "):
        text = text[8:].strip()
    numeral = text.split(" ", 1)[0].upper()
    if numeral not in ODYSSEY_NUMERALS:
        raise BoundViolation(
            f"odyssey {odyssey!r} is not one of I-III (there is no Odyssey IV)"
        )
    return numeral


def _stop_grants_authority(stop: str) -> bool:
    text = f" {stop.lower()} "
    if " never " in text or " not " in text or "cannot " in text or "must not " in text:
        return False
    needles = (
        "then promote",
        "promote the",
        "promote to protected",
        "choose the singularity",
        "install the singularity",
    )
    return any(n in text for n in needles)


@dataclass(frozen=True)
class OptimizerBound:
    """Declared envelope the proposer may not step outside of."""

    max_hypotheses: int = 9
    max_total_cost_units: int = 24
    allowed_kinds: tuple[str, ...] = KINDS
    allowed_authority: frozenset[str] = ALLOWED_AUTHORITY
    era: str = "III"
    odyssey: str | None = None
    gpu_windows_held: int = 0
    gpu_windows_requested: int = 0
    may_promote: bool = False
    may_modify_verifier: bool = False
    may_widen_authority: bool = False

    def __post_init__(self) -> None:
        if self.may_promote or self.may_modify_verifier or self.may_widen_authority:
            raise BoundViolation(
                "a bound cannot grant promotion, verifier modification, or authority widening"
            )
        if int(self.max_hypotheses) < 1:
            raise BoundViolation("max_hypotheses must be >= 1")
        if int(self.max_total_cost_units) < 1:
            raise BoundViolation("max_total_cost_units must be >= 1")
        if int(self.gpu_windows_held) != 0:
            raise BoundViolation("sidecar cannot hold a GPU window")
        kinds = tuple(self.allowed_kinds)
        unknown = [k for k in kinds if k not in KINDS]
        if unknown:
            raise BoundViolation(f"unknown proposal kind(s) {unknown}")
        if not kinds:
            raise BoundViolation("allowed_kinds must be non-empty")
        forbidden = [a for a in self.allowed_authority if a in FORBIDDEN_AUTHORITY]
        if forbidden:
            raise BoundViolation(f"bound listed forbidden authority {forbidden}")
        extra = [a for a in self.allowed_authority if a not in ALLOWED_AUTHORITY]
        if extra:
            raise BoundViolation(f"bound listed unknown authority {extra}")
        _era_numeral(self.era)
        _odyssey_numeral(self.odyssey)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_hypotheses": int(self.max_hypotheses),
            "max_total_cost_units": int(self.max_total_cost_units),
            "allowed_kinds": list(self.allowed_kinds),
            "allowed_authority": sorted(self.allowed_authority),
            "era": _era_numeral(self.era),
            "era_name": next(e for e in ERAS if e.startswith(_era_numeral(self.era) + " ")),
            "odyssey": _odyssey_numeral(self.odyssey),
            "gpu_windows_held": 0,
            "gpu_windows_requested": int(self.gpu_windows_requested),
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
        }


def validate_delegated_body(kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Shape-check against the owner schema. Does not call the owner module."""
    if kind not in KIND_OWNERS:
        raise DelegationError(f"unknown kind {kind!r}")
    owner = KIND_OWNERS[kind]
    required: tuple[str, ...] = owner["required_fields"]
    missing = [f for f in required if f not in body]
    if missing:
        raise DelegationError(f"{kind}: delegated_body missing owner fields {missing}")
    if not owner["null_ok"]:
        empty = [f for f in required if not _present(body.get(f))]
        if empty:
            raise DelegationError(f"{kind}: owner schema requires non-empty {empty}")
    if kind == "compiler-pass":
        klass = body.get("contamination_class")
        if klass not in {"STATIC_ONLY", "UNKNOWN", "DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE", None}:
            raise DelegationError(f"compiler-pass: bad contamination_class {klass!r}")
        if klass == "PROTECTED_ABSOLUTE":
            raise DelegationError(
                "compiler-pass: sidecar cannot emit a PROTECTED_ABSOLUTE LPC row"
            )
        if klass == "DIAGNOSTIC_RELATIVE":
            raise DelegationError(
                "compiler-pass: DIAGNOSTIC_RELATIVE never completes an LPC row"
            )
    if kind == "transfer":
        if body.get("time_to_first_useful_executable_ns") is not None:
            raise DelegationError(
                "transfer: time_to_first_useful_executable_ns stays null until a "
                "protected measurement exists"
            )
        if str(body.get("scope") or "") == "GENERIC_VERIFIED" and str(
            body.get("evidence_strength") or ""
        ) not in {"PROTECTED_ABSOLUTE", "REPRODUCED"}:
            raise DelegationError(
                "transfer: cannot claim GENERIC_VERIFIED without protected/reproduced evidence"
            )
    if kind == "hardware-profile":
        effect = body.get("predicted_effect")
        allowed_mag = {"UNKNOWN", "SUB_PERCENT", "SINGLE_DIGIT_FRACTION", "FACTOR"}
        if not isinstance(effect, dict) or "magnitude_class" not in effect:
            raise DelegationError("hardware-profile: predicted_effect must be a class, not a scalar")
        if effect.get("magnitude_class") not in allowed_mag:
            raise DelegationError(
                f"hardware-profile: predicted_effect.magnitude_class "
                f"{effect.get('magnitude_class')!r} not in {sorted(allowed_mag)}"
            )
        if isinstance(effect.get("magnitude"), (int, float)) or isinstance(
            effect.get("value"), (int, float)
        ):
            raise DelegationError("hardware-profile: predicted_effect must not carry a fabricated number")
    return _strip_hardware_numbers(dict(body))


def make_hypothesis(
    *,
    id: str,
    kind: str,
    statement: str,
    evidence_parents: Sequence[str],
    expected_information: int,
    cost_units: int,
    delegated_body: Mapping[str, Any],
    bound: OptimizerBound | None = None,
    era: str = "III",
    odyssey: str | None = None,
) -> dict[str, Any]:
    """Construct one PROPOSAL. Status is PROPOSED; verified is False and stays that way here."""
    envelope = bound or OptimizerBound()
    if kind not in envelope.allowed_kinds:
        raise BoundViolation(f"{id}: kind {kind!r} is outside the bound")
    if kind not in KINDS:
        raise BoundViolation(f"{id}: unknown kind {kind!r}")
    parents = tuple(str(p) for p in evidence_parents if str(p).strip())
    if not parents:
        raise BoundViolation(f"{id}: every hypothesis must carry evidence parents")
    info = int(expected_information)
    cost = int(cost_units)
    if info not in {INFO_LOW, INFO_MEDIUM, INFO_HIGH}:
        raise BoundViolation(f"{id}: expected_information must be 1, 2, or 3; got {info}")
    if cost < 1:
        raise BoundViolation(f"{id}: cost_units must be >= 1")
    owner = KIND_OWNERS[kind]
    verifier = str(owner["verifier"])
    if verifier.strip().lower() in WEAK_VERIFIERS:
        raise BoundViolation(f"{id}: verifier {verifier!r} would weaken verification")
    body = validate_delegated_body(kind, delegated_body)
    era_n = _era_numeral(era)
    ody = _odyssey_numeral(odyssey)
    if kind == "transfer" and ody is None:
        ody = "II"
    hyp = {
        "id": str(id),
        "kind": kind,
        "statement": str(statement),
        "evidence_parents": list(parents),
        "expected_information": info,
        "cost_units": cost,
        "status": "PROPOSED",
        "verified": False,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "era": era_n,
        "odyssey": ody,
        "workunit_species": owner["workunit_species"],
        "verifier": verifier,
        "bounded_authority": sorted(envelope.allowed_authority),
        "delegation": {
            "owner_module": owner["owner_module"],
            "owner_schema": owner["owner_schema"],
            "owner_receipt": owner["owner_receipt"],
            "required_fields": list(owner["required_fields"]),
            "reimplemented": False,
        },
        "delegated_body": body,
        "claim_boundary": CLAIM_BOUNDARY,
        "may_promote": False,
        "may_modify_verifier": False,
    }
    return hyp


def rank_hypotheses(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank by expected information per unit cost.

    Same integer rule as Hardware Doctor: `-(info * 60 // cost)`, then cost,
    then id. Never a hardware measurement. Reused, not rivalled.
    """
    decorated: list[tuple[dict[str, Any], int, int]] = []
    for rec in rows:
        cost = int(rec["cost_units"])
        info = int(rec["expected_information"])
        if cost < 1:
            raise BoundViolation(f"{rec.get('id')}: cost_units must be >= 1")
        decorated.append((copy.deepcopy(dict(rec)), cost, info))

    def _key(item: tuple[dict[str, Any], int, int]) -> tuple[Any, ...]:
        rec, cost, info = item
        return (-(info * 60 // cost), cost, rec.get("id") or "")

    ordered = [item for item in sorted(decorated, key=_key)]
    ranked: list[dict[str, Any]] = []
    for i, (rec, cost, info) in enumerate(ordered, start=1):
        rec["rank"] = i
        rec["information_per_cost"] = {
            "expected_information": info,
            "cost_units": cost,
            "rule": (
                "rank by expected_information / cost_units; integer key "
                "-(info*60//cost); never a hardware measurement; same rule as "
                "tools/future/hardware_doctor.py::rank_queue"
            ),
        }
        ranked.append(rec)
    return ranked


def clip_to_bound(
    ranked: Sequence[Mapping[str, Any]], bound: OptimizerBound
) -> list[dict[str, Any]]:
    """Admit a prefix of a ranked list until the bound is exhausted."""
    admitted: list[dict[str, Any]] = []
    spent = 0
    for rec in ranked:
        kind = rec.get("kind")
        if kind not in bound.allowed_kinds:
            continue
        cost = int(rec["cost_units"])
        if len(admitted) >= int(bound.max_hypotheses):
            break
        if spent + cost > int(bound.max_total_cost_units):
            break
        admitted.append(dict(rec))
        spent += cost
    return [{**rec, "rank": i} for i, rec in enumerate(admitted, start=1)]


# ---------------------------------------------------------------------------
# Candidate catalog — consume owner receipts; fall back to recovered seeds
# ---------------------------------------------------------------------------


def _lpc_body(seed: Mapping[str, Any]) -> dict[str, Any]:
    owner = KIND_OWNERS["compiler-pass"]
    body = {f: None for f in owner["required_fields"]}
    body.update(
        {
            "model": "Qwen27",
            "organ_fingerprint": seed["organ_fingerprint"],
            "representation": "sealed-resident",
            "backend": "metal",
            "layout": "UNKNOWN",
            "tile": "UNKNOWN",
            "grouping": "UNKNOWN",
            "fusion": seed["fusion"],
            "contamination_class": "STATIC_ONLY",
            "complete_token_effect": None,
            "capability": None,
        }
    )
    return body


def _compiler_candidates(bound: OptimizerBound) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in COMPILER_PASS_SEEDS:
        rows.append(
            make_hypothesis(
                id=str(seed["id"]),
                kind="compiler-pass",
                statement=str(seed["statement"]),
                evidence_parents=(
                    LPC_RECEIPT,
                    "receipts/future/HCLI_FUTURE_WORKUNITS.json",
                    f"recovered physical candidate {seed['pass_name']}",
                ),
                expected_information=int(seed["expected_information"]),
                cost_units=int(seed["cost_units"]),
                delegated_body=_lpc_body(seed),
                bound=bound,
                era=bound.era,
            )
        )
    return rows


def _law_fallback(law_id: str) -> dict[str, Any]:
    statements = {
        "LAW-COMPETENT-KERNEL-FIRST": (
            "A representation evaluated with an incompetent kernel is not "
            "evaluated. Fewer stored bits is not fewer nanoseconds."
        ),
        "LAW-HELDOUT-REAL-ACTIVATIONS": (
            "Rank COMPOSITION choices on held-out real activations. "
            "Weight-space error is not a substitute for the composition question."
        ),
        "LAW-FITTED-AFFINE-BEATS-RTN": (
            "At matched bits per weight, a least-squares-refit affine codec "
            "beats generic grouped-absmax round-to-nearest. This is a transfer "
            "hypothesis, not a GENERIC_VERIFIED promotion."
        ),
    }
    return {
        "law_id": law_id,
        "statement": statements.get(law_id, law_id),
        "source_model": "qwen3.8-27b-abliterated",
        "source_device": "UNKNOWN",
        "architecture_family": "dense_hybrid_transformer",
        "organ_class": "cross_model",
        "backend": "metal",
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "evidence_refs": [O2_RECEIPT, f"{O2_RECEIPT}#{law_id}"],
        "scope": "MODEL_LOCAL",
        "transfer_candidates": [
            {
                "target_model": "Qwen/Qwen3.8-Flash-Next",
                "target_school": "Flash",
                "target_architecture_family": "qwen4_exp",
                "promotion_requested": False,
            }
        ],
        "transfer_confidence": {
            "value": 0.45,
            "basis": "evidence_strength=DIAGNOSTIC_RELATIVE scope=MODEL_LOCAL; not a promotion",
        },
        "counterexample_requirement": (
            "re-earn the law on the named target; do not widen scope from this sidecar"
        ),
        "expected_saved_experiments": None,
        "actual_saved_experiments": None,
        "time_to_first_useful_executable_ns": None,
    }


def _transfer_candidates(bound: OptimizerBound) -> list[dict[str, Any]]:
    doc = _future_receipt(O2_RECEIPT)
    laws_by_id: dict[str, dict[str, Any]] = {}
    if doc and isinstance(doc.get("laws"), list):
        for law in doc["laws"]:
            if isinstance(law, dict) and law.get("law_id"):
                laws_by_id[str(law["law_id"])] = law
    rows: list[dict[str, Any]] = []
    for law_id in TRANSFER_LAW_IDS:
        raw = laws_by_id.get(law_id) or _law_fallback(law_id)
        body = _law_fallback(law_id)
        for field in KIND_OWNERS["transfer"]["required_fields"]:
            if field in raw:
                body[field] = raw[field]
        body["time_to_first_useful_executable_ns"] = None
        body["scope"] = raw.get("scope") or "MODEL_LOCAL"
        if body["scope"] == "GENERIC_VERIFIED":
            # Sidecar must not quote a GENERIC_VERIFIED claim it cannot re-earn.
            body["scope"] = "GENERIC_CANDIDATE"
        candidates = list(body.get("transfer_candidates") or [])
        for cand in candidates:
            if isinstance(cand, dict):
                cand["promotion_requested"] = False
                cand["scope_widening"] = "REFUSED"
        body["transfer_candidates"] = candidates
        strength = str(body.get("evidence_strength") or "STATIC")
        info = INFO_MEDIUM if strength in {"DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE", "REPRODUCED"} else INFO_LOW
        rows.append(
            make_hypothesis(
                id=f"RO-TR-{law_id}",
                kind="transfer",
                statement=(
                    f"Re-earn {law_id} on a named transfer candidate without widening "
                    f"scope. Odyssey II already holds the law; this is a PROPOSAL "
                    f"to test transfer, not a call to Law.promote()."
                ),
                evidence_parents=(
                    O2_RECEIPT,
                    f"{O2_RECEIPT}#{law_id}",
                    "receipts/future/HCLI_FUTURE_WORKUNITS.json",
                ),
                expected_information=info,
                cost_units=2,
                delegated_body=body,
                bound=bound,
                era=bound.era,
                odyssey="II",
            )
        )
    return rows


def _hardware_body_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for field in KIND_OWNERS["hardware-profile"]["required_fields"]:
        body[field] = entry.get(field)
    return body


def _hardware_candidates(bound: OptimizerBound) -> list[dict[str, Any]]:
    doc = _future_receipt(HD_RECEIPT)
    entries: list[dict[str, Any]] = []
    if doc and isinstance(doc.get("experiment_queue"), list) and doc["experiment_queue"]:
        for item in doc["experiment_queue"]:
            if isinstance(item, dict) and item.get("id"):
                entries.append(item)
    if not entries:
        entries = [dict(item) for item in HARDWARE_FALLBACK]
    rows: list[dict[str, Any]] = []
    for entry in entries[:3]:
        hid = str(entry["id"])
        info = int(entry.get("refutation_weight") or entry.get("expected_information") or INFO_MEDIUM)
        if info not in {INFO_LOW, INFO_MEDIUM, INFO_HIGH}:
            info = INFO_MEDIUM
        cost = int(entry.get("simulator_cost") or entry.get("cost_units") or 1)
        body = _hardware_body_from_entry(entry)
        if not _present(body.get("hypothesis")):
            # Queue rows use "hypothesis"; fallback does too.
            continue
        rows.append(
            make_hypothesis(
                id=f"RO-HW-{hid}",
                kind="hardware-profile",
                statement=str(body["hypothesis"]),
                evidence_parents=(
                    HD_RECEIPT,
                    f"{HD_RECEIPT}#experiment_queue.{hid}",
                ),
                expected_information=info,
                cost_units=max(1, cost),
                delegated_body=body,
                bound=bound,
                era=bound.era,
            )
        )
    if not rows:
        raise BoundViolation("hardware-profile catalog was empty after recovery")
    return rows


def candidate_catalog(bound: OptimizerBound | None = None) -> list[dict[str, Any]]:
    """Self-generated hypotheses, still unbounded; generate() clips."""
    envelope = bound or OptimizerBound()
    rows: list[dict[str, Any]] = []
    if "compiler-pass" in envelope.allowed_kinds:
        rows.extend(_compiler_candidates(envelope))
    if "transfer" in envelope.allowed_kinds:
        rows.extend(_transfer_candidates(envelope))
    if "hardware-profile" in envelope.allowed_kinds:
        rows.extend(_hardware_candidates(envelope))
    rows.sort(key=lambda r: str(r["id"]))
    return rows


# ---------------------------------------------------------------------------
# WorkUnit economy — consume species receipt, emit through HCLI WorkUnit
# ---------------------------------------------------------------------------


def _species_from_receipt() -> dict[str, dict[str, Any]]:
    doc = _future_receipt(WORKUNITS_RECEIPT)
    out: dict[str, dict[str, Any]] = {}
    if not doc:
        return out
    for item in doc.get("species") or []:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = item
    return out


def default_budget() -> dict[str, Any]:
    return {
        "attempts": DEFAULT_RETRY_BUDGET,
        "max_repair_depth": MAX_REPAIR_DEPTH,
        "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
        "gpu_windows_requested": 0,
        "gpu_windows_held": 0,
        "wall_clock_s": None,
        "source": "hcli/workunit.py DEFAULT_RETRY_BUDGET / MAX_REPAIR_DEPTH / MAX_REPAIRS_PER_ROOT",
    }


def default_stop_conditions() -> tuple[str, ...]:
    return (
        "stop when bound.max_hypotheses is reached",
        "stop when bound.max_total_cost_units is spent",
        "stop when the HCLI repair budget is exhausted "
        "(MAX_REPAIR_DEPTH / MAX_REPAIRS_PER_ROOT)",
        "stop when the named species verifier (not the proposer) settles the unit",
        "never promote; DIAGNOSTIC_RELATIVE never becomes PROTECTED_ABSOLUTE here",
    )


def emit_workunit(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """Emit one HCLI WorkUnit proposal. The unit cannot mark itself verified."""
    hid = str(hypothesis["id"])
    kind = str(hypothesis["kind"])
    owner = KIND_OWNERS[kind]
    stop = default_stop_conditions()
    for s in stop:
        if _stop_grants_authority(s):
            raise BoundViolation(f"{hid}: stop condition must not grant promotion")
    unit = WorkUnit(
        id=f"future.resident-optimizer.{hid}",
        role="science",
        description=str(hypothesis["statement"]),
        dependencies=list(hypothesis.get("evidence_parents") or []),
        resource_class="STATIC_ANALYSIS",
        preferred_backend=None,
        provider="future.resident_optimizer",
        verifier=str(hypothesis.get("verifier") or owner["verifier"]),
        effect_class="READ_ONLY",
        workspace="repo-root",
        classification="STATIC_ONLY",
        status="pending",
        repair_depth=0,
    )
    if not unit.verifier or unit.verifier.strip().lower() in WEAK_VERIFIERS:
        raise BoundViolation(f"{hid}: unit verifier would weaken verification")
    row = unit.to_dict()
    row.update(
        {
            "claim_boundary": CLAIM_BOUNDARY,
            "species": owner["workunit_species"],
            "hypothesis_id": hid,
            "kind": kind,
            "requires_quiescence": False,
            "budget": default_budget(),
            "stop_conditions": list(stop),
            "may_promote": False,
            "may_modify_verifier": False,
            "status": "pending",
            "classification": "STATIC_ONLY",
        }
    )
    # Round-trip the HCLI core so a future scheduler can consume the unit.
    WorkUnit.from_dict(dict(row))
    return row


def recovery_contract() -> dict[str, Any]:
    """Point at the live recovery gates. Do not re-run them. Do not fork them."""
    return {
        "runs_recovery": False,
        "policy": (
            "fail-closed fixture proof in hcli.agentos.recovery; production "
            "recovery is Codex-owned. This sidecar records the contract."
        ),
        "owners": {
            "recovery_gate": "hcli/agentos/recovery.py",
            "recovery_schema": "hcli.agentos.recovery_gate.v1",
            "autonomy_gate": "hcli/agentos/autonomy_gate.py",
            "autonomy_a3_resident_kill": "hcli/agentos/autonomy_gate.py::run_resident_kill",
            "autonomy_a4_process_kill": "hcli/agentos/autonomy_gate.py::run_process_kill",
            "autonomy_a5_idempotency": "hcli/agentos/autonomy_gate.py::run_idempotency_crash",
            "resident_gate": "hcli/agentos/resident_gate.py",
            "resident_gate_schema": "hcli.agentos.resident_gate.v1",
            "mutation_rollback": "hcli/mutation.py::rollback_mutation",
            "workunit_repair": "hcli/workunit.py",
            "resident_install": "tools/future/resident_install.py",
        },
        "resident_py": None,
        "resident_py_note": (
            "hcli/agentos/resident.py does not exist in HEAD; resident_gate.py "
            "is the live sequential-proof boundary"
        ),
        "self_evolution": {
            "path": "lab/hcli/self_evolution.py",
            "materialized_in_this_worktree": (REPO / "lab/hcli/self_evolution.py").is_file(),
            "recovered_principle": "the proposer is never the admitter (tribunal separation)",
        },
        "verifier_pipeline": {
            "path": "hcli/verifier_pipeline.py",
            "principle": (
                "execute() gathers refs and must not render a verdict; "
                "verify() settles. Check-5: TRUE against nonzero exit is forced FALSE."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Distinct objects: proposer vs verifier. No shared mutable state.
# ---------------------------------------------------------------------------


class IsolatedVerifier:
    """STATIC_ONLY inspector. Not a protected GPU verifier. Cannot promote.

    Holds its own verdict store. The proposer is never given a reference.
    """

    def __init__(self) -> None:
        self._verdicts: dict[str, dict[str, Any]] = {}

    def store_id(self) -> int:
        return id(self._verdicts)

    def inspect(self, hypothesis: Mapping[str, Any]) -> dict[str, Any]:
        """Structural check of a PROPOSAL. Does not mark it verified."""
        hid = str(hypothesis.get("id") or "")
        kind = str(hypothesis.get("kind") or "")
        problems: list[str] = []
        if not hid:
            problems.append("missing id")
        if kind not in KINDS:
            problems.append(f"unknown kind {kind!r}")
        if not list(hypothesis.get("evidence_parents") or []):
            problems.append("missing evidence_parents")
        try:
            if kind in KIND_OWNERS:
                validate_delegated_body(kind, hypothesis.get("delegated_body") or {})
        except (DelegationError, BoundViolation) as exc:
            problems.append(str(exc))
        if hypothesis.get("verified") is True:
            problems.append("proposal arrived already marked verified; inspector does not accept self-certification")
        verdict = {
            "hypothesis_id": hid,
            "verdict": "STRUCTURALLY_SOUND" if not problems else "STRUCTURALLY_UNSOUND",
            "problems": problems,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "promotion": "IMPOSSIBLE_FROM_THIS_LANE",
            "settles_physical_claim": False,
            "note": (
                "This inspector is not the protected verifier. Disk state is "
                "authority. A STRUCTURALLY_SOUND proposal is still unverified."
            ),
        }
        self._verdicts[hid] = dict(verdict)
        return dict(verdict)

    def record_verdict(self, hypothesis_id: str, verdict: str) -> dict[str, Any]:
        """Only this object may write a verdict, and never a promotion class."""
        banned = {
            "VERIFIED",
            "PROMOTED",
            "PROTECTED_ABSOLUTE",
            "DIAGNOSTIC_RELATIVE",
            "TRUE",
            "PASSED",
            "ACCEPTED",
        }
        if str(verdict).upper() in banned:
            raise VerifierSeparationError(
                f"sidecar IsolatedVerifier cannot record {verdict!r}; "
                "promotion and protected settlement are out of scope"
            )
        row = {
            "hypothesis_id": str(hypothesis_id),
            "verdict": str(verdict),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "promotion": "IMPOSSIBLE_FROM_THIS_LANE",
        }
        self._verdicts[str(hypothesis_id)] = row
        return dict(row)

    def weaken(self, *_args: Any, **_kwargs: Any) -> None:
        raise VerifierSeparationError("IsolatedVerifier cannot weaken itself")


class Proposer:
    """Generates and ranks PROPOSALs inside a bound. Cannot verify them."""

    def __init__(self, bound: OptimizerBound | None = None) -> None:
        object.__setattr__(self, "_bound", bound or OptimizerBound())
        object.__setattr__(self, "_emitted", ())
        object.__setattr__(self, "_authority", frozenset(self._bound.allowed_authority))
        object.__setattr__(self, "_frozen", True)

    def store_id(self) -> int:
        return id(self._emitted)

    def bound(self) -> OptimizerBound:
        return self._bound

    def authority(self) -> frozenset[str]:
        return self._authority

    def emitted(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(h) for h in self._emitted)

    def generate(self, *, seed: int = 0) -> tuple[dict[str, Any], ...]:
        """Self-generate hypotheses within the bound. `seed` is recorded, not shuffled."""
        del seed  # catalog is ordered; shuffling would break disk-state determinism
        catalog = candidate_catalog(self._bound)
        ranked = rank_hypotheses(catalog)
        admitted = clip_to_bound(ranked, self._bound)
        frozen = tuple(copy.deepcopy(h) for h in admitted)
        object.__setattr__(self, "_emitted", frozen)
        return self.emitted()

    def rank(self, rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
        payload = list(rows) if rows is not None else list(self._emitted)
        return rank_hypotheses(payload)

    def mark_verified(self, *_args: Any, **_kwargs: Any) -> None:
        raise VerifierSeparationError(
            "proposer cannot mark its own proposal verified"
        )

    def weaken_verifier(self, *_args: Any, **_kwargs: Any) -> None:
        raise VerifierSeparationError("proposer cannot weaken a verifier")

    def widen_authority(self, *_args: Any, **_kwargs: Any) -> None:
        raise VerifierSeparationError("proposer cannot widen its own authority")

    def __setattr__(self, name: str, value: Any) -> None:
        frozen = self.__dict__.get("_frozen", False)
        if frozen and name in {
            "_authority",
            "_bound",
            "_emitted",
            "verifier",
            "_verifier",
            "verdicts",
            "_verdicts",
            "verified",
        }:
            raise VerifierSeparationError(
                f"proposer cannot assign {name!r}; authority and verification are frozen"
            )
        object.__setattr__(self, name, value)


class ResidentOptimizer:
    """Facade holding a proposer and a verifier that do not share state.

    `promote()` does not exist. Emitting a PROPOSAL is the only write this
    object can do. A separate protected verifier (not IsolatedVerifier)
    would have to decide any later promotion, and that decision is out of
    this lane's authority.
    """

    def __init__(
        self,
        bound: OptimizerBound | None = None,
        *,
        verifier: IsolatedVerifier | None = None,
    ) -> None:
        self._bound = bound or OptimizerBound()
        self.proposer = Proposer(self._bound)
        self.verifier = verifier if verifier is not None else IsolatedVerifier()
        if self.proposer is self.verifier:
            raise VerifierSeparationError("proposer and verifier must be distinct objects")
        if self.proposer.store_id() == self.verifier.store_id():
            raise VerifierSeparationError("proposer and verifier share a mutable store")

    def bound(self) -> OptimizerBound:
        return self._bound

    def generate(self, *, seed: int = 0) -> tuple[dict[str, Any], ...]:
        return self.proposer.generate(seed=seed)

    def rank(self, rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
        return self.proposer.rank(rows)

    def economy(self, hypotheses: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        rows = list(hypotheses) if hypotheses is not None else list(self.proposer.emitted())
        units = [emit_workunit(h) for h in rows]
        species = _species_from_receipt()
        return {
            "schema": "hawking.future.resident_optimizer.economy.v1",
            "budget": default_budget(),
            "stop_conditions": list(default_stop_conditions()),
            "work_units": units,
            "count": len(units),
            "species_consumed": sorted({u["species"] for u in units}),
            "species_catalog_present": sorted(species),
            "workunits_receipt": WORKUNITS_RECEIPT,
            "workunits_receipt_present": _future_receipt(WORKUNITS_RECEIPT) is not None,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def recovery(self) -> dict[str, Any]:
        return recovery_contract()

    def shared_mutable_state(self) -> bool:
        """True only if the two objects share a store identity — which they must not."""
        return self.proposer.store_id() == self.verifier.store_id()

    def mark_verified(self, *args: Any, **kwargs: Any) -> None:
        # Facade must not launder the forbidden act.
        self.proposer.mark_verified(*args, **kwargs)

    def weaken_verifier(self, *args: Any, **kwargs: Any) -> None:
        self.proposer.weaken_verifier(*args, **kwargs)

    def widen_authority(self, *args: Any, **kwargs: Any) -> None:
        self.proposer.widen_authority(*args, **kwargs)


def _prove_verifier_separation() -> list[dict[str, Any]]:
    """Watch the three refusals actually fire. A guard nobody has seen fail is not a guard."""
    opt = ResidentOptimizer()
    trials = (
        ("mark_verified", lambda: opt.proposer.mark_verified("RO-CP-001")),
        ("weaken_verifier", lambda: opt.proposer.weaken_verifier("self")),
        ("widen_authority", lambda: opt.proposer.widen_authority("self_promotion")),
    )
    results: list[dict[str, Any]] = []
    for name, thunk in trials:
        try:
            thunk()
        except VerifierSeparationError as exc:
            results.append({"trial": name, "refused": True, "error": str(exc)})
            continue
        raise VerifierSeparationError(f"authority guard did not fire for {name}")
    if hasattr(ResidentOptimizer, "promote") or hasattr(Proposer, "promote"):
        raise VerifierSeparationError("promote() must not exist on the optimizer or proposer")
    if opt.shared_mutable_state():
        raise VerifierSeparationError("proposer and verifier share mutable state")
    return results


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "hcli/agentos/resident.py",
            "present": False,
            "what": "does not exist in HEAD; resident_gate.py is the live boundary",
        },
        {
            "path": "hcli/agentos/resident_gate.py",
            "present": True,
            "what": (
                "LIVE_RESIDENT_SEQUENTIAL_PROOF; measures process/lifecycle only; "
                "does not score model quality and does not promote generated prose"
            ),
        },
        {
            "path": "hcli/agentos/autonomy_gate.py",
            "present": True,
            "what": (
                "A1-A5 autonomy qualification; fixed verifier the model cannot nominate; "
                "A3 resident kill, A4 process kill, A5 post-mutation crash / replay refused"
            ),
        },
        {
            "path": "hcli/agentos/recovery.py",
            "present": True,
            "what": "fixture recovery gate; provider must not self-certify (nonsense_case)",
        },
        {
            "path": "hcli/verifier_pipeline.py",
            "present": True,
            "what": (
                "plan/execute/verify/synth; execute must not render a verdict; "
                "verify settles mechanically; Check-5 override"
            ),
        },
        {
            "path": "hcli/mutation.py",
            "present": True,
            "what": "bounded reversible mutations, snapshots, rollback_mutation, content_hash",
        },
        {
            "path": "hcli/workunit.py",
            "present": True,
            "what": "WorkUnit constructor, DEFAULT_RETRY_BUDGET=3, MAX_REPAIR_DEPTH=3, MAX_REPAIRS_PER_ROOT=6",
        },
        {
            "path": "lab/hcli/self_evolution.py",
            "present": (REPO / "lab/hcli/self_evolution.py").is_file(),
            "what": (
                "recovered via git show; tribunal separation: the proposer is never "
                "the admitter; missing evidence yields PENDING not fabricated ACCEPT"
            ),
        },
        {
            "path": "tools/future/workunit_species.py",
            "present": True,
            "consumed_as": WORKUNITS_RECEIPT,
            "what": "WorkUnit species + starting queue; read the receipt, do not import the module",
        },
        {
            "path": "tools/future/odyssey2_law_store.py",
            "present": True,
            "consumed_as": O2_RECEIPT,
            "what": "transfer law schema + scope lattice; promote() without evidence raises ScopeViolation",
        },
        {
            "path": "tools/future/hardware_doctor.py",
            "present": True,
            "consumed_as": HD_RECEIPT,
            "what": "hardware-profile schema + rank by expected information per unit cost",
        },
        {
            "path": "tools/future/lpc_dataset.py",
            "present": True,
            "consumed_as": LPC_RECEIPT,
            "what": "compiler-pass experiment row schema; DIAGNOSTIC_RELATIVE never completes",
        },
        {
            "path": "tools/future/resident_install.py",
            "present": True,
            "what": "generic NX install contract; tournament winner binds later; this lane does not install",
        },
        {
            "path": "tools/future/candidate_planner.py",
            "present": True,
            "what": "staged qualification plan; not a compiler-pass schema owner, not rivalled",
        },
    ]


def build() -> Path:
    opt = ResidentOptimizer()
    hypotheses = list(opt.generate(seed=0))
    economy = opt.economy(hypotheses)
    refusals = _prove_verifier_separation()
    owners_present = {
        kind: {
            "owner_module": spec["owner_module"],
            "owner_schema": spec["owner_schema"],
            "owner_receipt": spec["owner_receipt"],
            "owner_receipt_present": _future_receipt(spec["owner_receipt"]) is not None,
            "workunit_species": spec["workunit_species"],
            "verifier": spec["verifier"],
            "required_fields": list(spec["required_fields"]),
            "reimplemented": False,
        }
        for kind, spec in KIND_OWNERS.items()
    }
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "status": SIDECAR_STATUS,
        "promoted": False,
        "built": True,
        "purpose": (
            "Bounded hypothesis proposer a future resident will wake into. "
            "Verifier held strictly apart. Promotion is impossible from this lane."
        ),
        "head": git("rev-parse", "HEAD"),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": (
                "part of Accelerator / Physical Compiler / Fusion; not a civilization "
                "and this module does not build an FPGA backend"
            ),
            "disk_state_is_authority": True,
            "diagnostic_relative_never_promotes": True,
            "protected_absolute_not_emitted": True,
        },
        "bound": opt.bound().to_dict(),
        "proposal_kinds": owners_present,
        "ranking_rule": (
            "expected information per unit cost; integer key -(info*60//cost); "
            "same rule as hardware_doctor.rank_queue; never a hardware measurement"
        ),
        "hypotheses": hypotheses,
        "counts": {
            "hypotheses": len(hypotheses),
            "by_kind": {
                kind: sum(1 for h in hypotheses if h["kind"] == kind) for kind in KINDS
            },
        },
        "economy": economy,
        "recovery": opt.recovery(),
        "verifier_separation": {
            "proposer_type": "tools.future.resident_optimizer.Proposer",
            "verifier_type": "tools.future.resident_optimizer.IsolatedVerifier",
            "shared_mutable_state": opt.shared_mutable_state(),
            "promote_exists_on_optimizer": hasattr(ResidentOptimizer, "promote"),
            "promote_exists_on_proposer": hasattr(Proposer, "promote"),
            "refusals_proven": refusals,
            "principle": (
                "Models propose; protected deterministic evidence decides. "
                "The IsolatedVerifier in this module is STATIC_ONLY structure "
                "checking. A protected GPU verifier is Codex-owned and absent here."
            ),
        },
        "promotion": {
            "promote_method": None,
            "possible_from_this_lane": False,
            "emits": "PROPOSAL",
            "decider": "separate protected verifier (not this sidecar)",
            "waits_for": "tournament winner; tools/future/resident_install.py binds the NX",
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": [
            "bounded self-generated hypotheses, each carrying evidence parents",
            "ranking by expected information per unit cost (Hardware Doctor integer rule, mixed kinds)",
            "compiler-pass / transfer / hardware-profile delegated to owner schemas, not reimplemented",
            "WorkUnit economy with HCLI repair budgets and stop conditions that cannot grant promotion",
            "recovery contract pointing at resident_gate / recovery.py / autonomy A3-A5, not forked",
            "proposer and IsolatedVerifier are distinct objects with no shared mutable state",
            "mark_verified / weaken_verifier / widen_authority each RAISE on the proposer",
            "promote() does not exist; receipt status is BUILT_NOT_PROMOTED",
        ],
        "negative_findings": [
            "hcli/agentos/resident.py does not exist in HEAD",
            "lab/hcli/self_evolution.py is not materialized in this sparse worktree; recovered via git show",
            "tools/accelerator/ and tools/headless/ are not materialized here",
            "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "IsolatedVerifier cannot settle a physical claim and cannot record VERIFIED/PROMOTED",
            "tournament winner is not chosen; resident_install remains unbound",
            "workunit_species / odyssey2_law_store / hardware_doctor / lpc_dataset were consumed as receipts, not imported",
            "no GPU / FPGA board / power meter in this lane; hardware fields stay null/UNKNOWN",
        ],
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    refusals = _prove_verifier_separation()
    if len(refusals) != 3 or not all(r["refused"] for r in refusals):
        raise AssertionError(f"expected three watched refusals, got {refusals}")
    opt = ResidentOptimizer()
    hyps = opt.generate(seed=0)
    if not hyps:
        raise AssertionError("proposer emitted no hypotheses")
    if any(h.get("verified") for h in hyps):
        raise AssertionError("a proposal arrived marked verified")
    kinds = {h["kind"] for h in hyps}
    if not kinds.issubset(set(KINDS)):
        raise AssertionError(f"unexpected kinds {kinds}")
    if hasattr(opt, "promote") and callable(getattr(opt, "promote", None)):
        raise AssertionError("promote() must not exist")
    tight = ResidentOptimizer(OptimizerBound(max_hypotheses=2, max_total_cost_units=24))
    clipped = tight.generate(seed=0)
    if len(clipped) > 2:
        raise AssertionError("bound.max_hypotheses was not enforced")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
