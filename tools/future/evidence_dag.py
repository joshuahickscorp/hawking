"""EVIDENCE_DAG — V0–V9 verification hierarchy, reuse, precise invalidation.

Proof is expensive and must be earned, reused, and invalidated precisely.
This module is the sidecar guard that kills the full-suite reflex: a mutation
does not automatically demand V8, and a promotion-adjacent candidate cannot
be admitted below the level its factors require.

Foundation already landed: tools/future/repro_science.py (identity, provenance,
claim ledger). This module extends it; it does not fork the ledger or the
hasher. Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
V8 and V9 are UNAVAILABLE on this host and requesting one RAISES rather than
silently downgrading.

    python3 tools/future/evidence_dag.py --selftest
    python3 tools/future/evidence_dag.py --build
    python3 tools/future/evidence_dag.py --required-level '{"mutation_scope":"organ","uncertainty":0.1,"risk":0.1,"upside":0.1,"promotion_proximity":0.0}'
    python3 tools/future/evidence_dag.py --admit '{"mutation_scope":"organ","uncertainty":0.1,"risk":0.1,"upside":0.1,"promotion_proximity":0.0,"achieved_level":"V2"}'
    python3 tools/future/evidence_dag.py --reuse-or-rerun evidence_dag.cached_invariant
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, RECEIPTS, git, sha256_file

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import repro_science as rs
from tools.future import workunit_species as wus

RECEIPT = "EVIDENCE_DAG.json"
SCHEMA = "hawking.future.evidence_dag.v1"
VERSION = 1
RECORDED_BY = "tools/future/evidence_dag.py"

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

COMPILER_PIN = "sidecar-static-verifier"
PROXIMITY_V8 = 0.75
PROXIMITY_V9 = 0.95
UNCERTAINTY_MID = 0.25
UNCERTAINTY_HIGH = 0.60
RISK_MID = 0.25
RISK_HIGH = 0.60
UPSIDE_HIGH = 0.80

# Cached-invariant reuse. A name is not a cache key. Inputs are.
REUSE = "REUSE"
RERUN = "RERUN"
CACHED_INVARIANT_CLAIM = "evidence_dag.cached_invariant"
CACHED_INVARIANT_FAMILY = "evidence_dag.hierarchy"
CLAIM_RECEIPT_SCHEMA = "hawking.future.claim_receipt.v1"

# Rank is "how much the receipt actually proved". An ask may reuse only
# when the receipt is at least as strong. STATIC_ONLY cannot satisfy a
# protected ask; a protected receipt may satisfy a static ask.
EVIDENCE_CLASS_RANK: dict[str, int] = {
    "STATIC_ONLY": 0,
    "DIAGNOSTIC_RELATIVE": 1,
    "PROTECTED_ABSOLUTE": 2,
}

# Named claims this module itself can answer. Other claims pass a receipt path.
CLAIM_CATALOG: dict[str, dict[str, str]] = {
    CACHED_INVARIANT_CLAIM: {
        "receipt": RECEIPT,
        "family": CACHED_INVARIANT_FAMILY,
    },
    "evidence_dag.v0_v9_catalog": {
        "receipt": RECEIPT,
        "family": CACHED_INVARIANT_FAMILY,
    },
}

# Qualification-funnel rungs recovered from
# receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json.
# This sidecar produces none of the protected rungs.
FUNNEL_TO_LEVEL = (
    ("static_validation", "V0-V2"),
    ("native_parity", "V2-V3"),
    ("diagnostic_relative_ab", "not a V-level this module emits; DIAGNOSTIC_RELATIVE never promotes"),
    ("protected_absolute_complete_wall", "V8"),
    ("promotion", "V9"),
)

# composition_ladder.py rungs (git show; sparse checkout may hide the file).
# Unreached is not a death. A screen verdict is not a model verdict.
LADDER_TO_LEVEL = (
    ("local_functional_probe", "V1"),
    ("held_out_activation", "V3"),
    ("adjacent_layers", "V4"),
    ("short_chain", "V4"),
    ("complete_organ", "V2"),
    ("complete_token", "V6"),
    ("coherent_generation", "V7"),
    ("capability", "V7"),
)


# ---------------------------------------------------------------------------
# Fail-closed errors. A missing GPU is not a weaker passing level.
# ---------------------------------------------------------------------------


class UnavailableLevelError(rs.FailClosed):
    """V8/V9 requested without protected GPU authority. Never a silent V7."""

    def __init__(self, level: str, reason: str | None = None) -> None:
        self.level = str(level)
        super().__init__(
            "unavailable_level",
            reason
            or (
                f"{level} is UNAVAILABLE: protected GPU authority is required "
                "and this sidecar does not have it. Refusing rather than downgrading."
            ),
        )


class BelowRequiredLevelError(rs.FailClosed):
    """Admission attempted below required_level. The bar is not lowered."""

    def __init__(self, achieved: str, required: str) -> None:
        self.achieved = str(achieved)
        self.required = str(required)
        super().__init__(
            "below_required_level",
            f"achieved {achieved} is below required {required}; "
            "refusing admission rather than lowering the bar",
        )


# ---------------------------------------------------------------------------
# V0–V9 catalog. Each row states what it proves and what it does not.
# ---------------------------------------------------------------------------


def _level(
    ordinal: int,
    name: str,
    proves: str,
    does_not_prove: str,
    *,
    requires_gpu: bool,
) -> dict[str, Any]:
    available = not requires_gpu
    return {
        "id": f"V{ordinal}",
        "ordinal": ordinal,
        "name": name,
        "proves": proves,
        "does_not_prove": does_not_prove,
        "requires_gpu": requires_gpu,
        "availability": "AVAILABLE" if available else "UNAVAILABLE",
        "unavailable_reason": (
            None
            if available
            else (
                "protected GPU authority this sidecar does not have; "
                "Metal is Codex's lane and is not probed here"
            )
        ),
        "parents": [f"V{i}" for i in range(ordinal)],
        "emits": "STATIC_ONLY",
        "requires_protected_lease": requires_gpu,
    }


_LEVELS: tuple[dict[str, Any], ...] = (
    _level(
        0,
        "schema/identity",
        "The artifact is parseable, schema-tagged, and has a content identity.",
        "Numerical correctness, organ behaviour, chains, complete tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        1,
        "tiny numerical",
        "A tiny deterministic numerical fixture holds (host CPU, fixed arrays).",
        "Organ behaviour, held-out generalization, chains, complete tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        2,
        "organ",
        "One named organ satisfies a static contract on a fixture graph.",
        "Held-out organs, multi-organ chains, complete tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        3,
        "held-out organ",
        "An organ withheld from the V2 subject still satisfies the static contract.",
        "Multi-organ chains, complete tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        4,
        "short-chain",
        "A short (2-hop) organ path is well-formed on the fixture graph.",
        "Deep chains, complete tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        5,
        "deep-chain",
        "A longer organ path through the fixture graph is well-formed.",
        "Complete accepted tokens, capability, performance, or promotion.",
        requires_gpu=False,
    ),
    _level(
        6,
        "complete token",
        "A static complete-token recipe (ordered organ path covering the fixture token) is well-formed.",
        "Capability, protected performance, or promotion. Does not measure complete-token wall time.",
        requires_gpu=False,
    ),
    _level(
        7,
        "capability subset",
        "A named capability-subset contract is specified and statically checkable.",
        "Protected capability or performance, tournament admission, or promotion.",
        requires_gpu=False,
    ),
    _level(
        8,
        "protected capability/performance",
        "Would prove capability and performance under a real protected GPU lease.",
        "This host cannot prove it. Requesting V8 raises; it is not a V7 that passed.",
        requires_gpu=True,
    ),
    _level(
        9,
        "promotion/tournament",
        "Would prove tournament admission under protected authority.",
        "This host cannot prove it. Requesting V9 raises; it is not a quieter V8 or V7.",
        requires_gpu=True,
    ),
)

LEVEL_BY_ID: dict[str, dict[str, Any]] = {row["id"]: row for row in _LEVELS}
LEVEL_IDS: tuple[str, ...] = tuple(row["id"] for row in _LEVELS)

# mutation_scope → base ordinal. Aliases included so a resident can speak
# either V-ids or the composition-ladder / funnel vocabulary.
MUTATION_SCOPES: dict[str, int] = {
    "schema": 0,
    "identity": 0,
    "v0": 0,
    "tiny_numerical": 1,
    "tiny": 1,
    "numerical": 1,
    "v1": 1,
    "organ": 2,
    "v2": 2,
    "held_out_organ": 3,
    "heldout_organ": 3,
    "v3": 3,
    "short_chain": 4,
    "v4": 4,
    "deep_chain": 5,
    "v5": 5,
    "complete_token": 6,
    "v6": 6,
    "capability_subset": 7,
    "capability": 7,
    "v7": 7,
    "protected_capability": 8,
    "protected_performance": 8,
    "v8": 8,
    "promotion": 9,
    "tournament": 9,
    "v9": 9,
}


# Static organ fixture. Not a physical organ census and not a measurement.
# Linear parent chain so V4/V5/V6 path checks are well-formed by construction.
ORGAN_GRAPH: dict[str, dict[str, Any]] = {
    "affine": {"role": "gemv", "parents": []},
    "norm": {"role": "norm", "parents": []},
    "attn": {"role": "attention", "parents": ["norm"]},
    "gate": {"role": "elementwise", "parents": ["attn"]},
    "ffn": {"role": "ffn", "parents": ["gate"]},
    "out": {"role": "proj", "parents": ["ffn"]},
}
HELD_OUT_ORGAN = "held_out_router"
HELD_OUT_GRAPH: dict[str, dict[str, Any]] = {
    HELD_OUT_ORGAN: {"role": "router", "parents": ["norm"], "held_out": True},
}
SHORT_PATH: tuple[str, ...] = ("norm", "attn")
DEEP_PATH: tuple[str, ...] = ("norm", "attn", "gate", "ffn", "out")
TOKEN_PATH: tuple[str, ...] = DEEP_PATH
CAPABILITY_SUBSET: tuple[str, ...] = (
    "capability.fact-capital",
    "capability.json-answer",
    "capability.no-think-leak",
)


def levels() -> list[dict[str, Any]]:
    """Copy of the V0–V9 catalog. Count is derived from the tuple."""
    return [dict(row) for row in _LEVELS]


def level_ordinal(level: str) -> int:
    spec = LEVEL_BY_ID.get(str(level))
    if spec is None:
        raise rs.FailClosed("unknown_level", f"{level!r} is not a V0–V9 id")
    return int(spec["ordinal"])


def level_id(ordinal: int) -> str:
    if ordinal < 0 or ordinal >= len(_LEVELS):
        raise rs.FailClosed("unknown_level", f"ordinal {ordinal} is not a V0–V9 index")
    return f"V{ordinal}"


# ---------------------------------------------------------------------------
# Identity. Extends repro_science.experiment_identity; level is bound in.
# ---------------------------------------------------------------------------


def payload_hash(obj: Any) -> str:
    return rs.content_hash(obj)


def proof_identity(
    *,
    inputs: Mapping[str, str],
    code_sha256: str,
    machine_genome: Mapping[str, Any],
    level: str,
    compiler: str = COMPILER_PIN,
) -> str:
    """Content hash over inputs + code + machine genome + level.

    Byte-identical inputs with the same code and genome reuse. A different
    V-level is a different proof even on the same bytes.
    """
    if str(level) not in LEVEL_BY_ID:
        raise rs.FailClosed("unknown_level", f"{level!r} is not a V0–V9 id")
    eid = rs.experiment_identity(
        inputs={k: inputs[k] for k in sorted(inputs)},
        code_sha256=code_sha256,
        compiler=compiler,
        machine_genome=dict(machine_genome),
    )
    return rs.content_hash({"experiment_identity": eid, "level": str(level)})


# ---------------------------------------------------------------------------
# Adaptive depth. All five factors move the result. V8 is a floor, not a default.
# ---------------------------------------------------------------------------


def _clamp01(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise rs.FailClosed("invalid_factor", f"{name} must be a real in [0, 1], got {value!r}")
    x = float(value)
    if x != x or x < 0.0 or x > 1.0:
        raise rs.FailClosed("invalid_factor", f"{name}={value!r} is not in [0, 1]")
    return x


def _raise_boost(value: float, mid: float, high: float) -> int:
    if value < mid:
        return 0
    if value < high:
        return 1
    return 2


def required_level(
    mutation_scope: str,
    uncertainty: float,
    risk: float,
    upside: float,
    promotion_proximity: float,
) -> str:
    """Minimum V-level a candidate must reach before admission.

    A weak, cheap, reversible candidate does not get V8. A promotion-adjacent
    candidate floors at V8 (or V9), even though those levels are UNAVAILABLE
    here — unavailability is a refusal, not a quieter passing grade.
    """
    scope_key = str(mutation_scope).strip().lower().replace("-", "_").replace(" ", "_")
    if scope_key not in MUTATION_SCOPES:
        raise rs.FailClosed(
            "unknown_mutation_scope",
            f"{mutation_scope!r} is not a known mutation scope; "
            f"known={sorted(MUTATION_SCOPES)}",
        )
    base = MUTATION_SCOPES[scope_key]
    u = _clamp01("uncertainty", uncertainty)
    rk = _clamp01("risk", risk)
    up = _clamp01("upside", upside)
    prox = _clamp01("promotion_proximity", promotion_proximity)

    ordinal = base + _raise_boost(u, UNCERTAINTY_MID, UNCERTAINTY_HIGH)
    ordinal += _raise_boost(rk, RISK_MID, RISK_HIGH)
    ordinal += 1 if up >= UPSIDE_HIGH else 0
    if ordinal > len(_LEVELS) - 1:
        ordinal = len(_LEVELS) - 1

    max_available = max(s["ordinal"] for s in _LEVELS if s["availability"] == "AVAILABLE")
    if prox >= PROXIMITY_V9 or base >= 9:
        ordinal = 9
    elif prox >= PROXIMITY_V8 or base >= 8:
        ordinal = max(ordinal, 8)
    elif ordinal > max_available:
        # Stacking cheap boosts must not mint a protected level. Promotion
        # proximity or an explicit protected/promotion scope is required.
        ordinal = max_available

    return level_id(ordinal)


def request_level(level: str) -> dict[str, Any]:
    """Return the catalog row, or RAISE if that level is UNAVAILABLE.

    Never returns a weaker level. Silent downgrade is the defect this guards.
    """
    spec = LEVEL_BY_ID.get(str(level))
    if spec is None:
        raise rs.FailClosed("unknown_level", f"{level!r} is not a V0–V9 id")
    if spec["availability"] != "AVAILABLE":
        raise UnavailableLevelError(str(level), spec.get("unavailable_reason"))
    return dict(spec)


def admit_candidate(
    *,
    mutation_scope: str,
    uncertainty: float,
    risk: float,
    upside: float,
    promotion_proximity: float,
    achieved_level: str,
) -> str:
    """Admit only if achieved_level >= required_level AND the level is available.

    Both directions are enforced:
      * cheap candidate required_level is not V8
      * promotion-adjacent required_level is V8/V9 and admitting at V7 raises
      * achieving V8 on this host still raises (unavailable), not a V7 consolation
    """
    required = required_level(
        mutation_scope, uncertainty, risk, upside, promotion_proximity
    )
    achieved = str(achieved_level)
    if level_ordinal(achieved) < level_ordinal(required):
        raise BelowRequiredLevelError(achieved, required)
    # Refuse a claim at an unavailable level even when it meets the bar.
    request_level(achieved)
    return "ADMITTED"


# ---------------------------------------------------------------------------
# Static executors for V0–V7. V8/V9 never reach here.
# ---------------------------------------------------------------------------


def _path_well_formed(path: Sequence[str], graph: Mapping[str, Mapping[str, Any]]) -> bool:
    if len(path) < 2:
        return False
    for hop in path:
        if hop not in graph:
            return False
    for i in range(1, len(path)):
        parents = list(graph[path[i]].get("parents") or [])
        if path[i - 1] not in parents:
            return False
    return True


def execute_level(level: str, inputs: Mapping[str, str]) -> str:
    """Run the static check for an AVAILABLE level. Returns a result content hash.

    V8/V9 are not executed: request_level has already raised.
    """
    spec = request_level(level)
    ordinal = int(spec["ordinal"])
    if ordinal == 0:
        if "schema" not in inputs:
            raise rs.FailClosed("v0_schema", "V0 requires a schema identity in inputs")
        return payload_hash({"level": "V0", "schema": inputs["schema"]})
    if ordinal == 1:
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        got = a @ a
        expected = np.array([[7.0, 10.0], [15.0, 22.0]], dtype=np.float64)
        if not np.allclose(got, expected):
            raise rs.FailClosed("tiny_numerical", "fixture matmul diverged")
        return payload_hash({"level": "V1", "matmul": got.tolist()})
    if ordinal == 2:
        organ = inputs.get("organ", "affine")
        if organ not in ORGAN_GRAPH:
            raise rs.FailClosed("organ", f"organ {organ!r} is not in the fixture graph")
        return payload_hash({"level": "V2", "organ": organ, "role": ORGAN_GRAPH[organ]["role"]})
    if ordinal == 3:
        organ = inputs.get("held_out_organ", HELD_OUT_ORGAN)
        if organ not in HELD_OUT_GRAPH:
            raise rs.FailClosed("held_out_organ", f"{organ!r} is not a declared held-out organ")
        if organ in ORGAN_GRAPH:
            raise rs.FailClosed("held_out_organ", f"{organ!r} was not held out of the V2 subject")
        return payload_hash({"level": "V3", "held_out_organ": organ})
    if ordinal == 4:
        if not _path_well_formed(SHORT_PATH, ORGAN_GRAPH):
            raise rs.FailClosed("short_chain", "short-chain fixture path is not well-formed")
        return payload_hash({"level": "V4", "path": list(SHORT_PATH)})
    if ordinal == 5:
        if not _path_well_formed(DEEP_PATH, ORGAN_GRAPH):
            raise rs.FailClosed("deep_chain", "deep-chain fixture path is not well-formed")
        if len(DEEP_PATH) <= len(SHORT_PATH):
            raise rs.FailClosed("deep_chain", "deep path is not deeper than the short path")
        return payload_hash({"level": "V5", "path": list(DEEP_PATH)})
    if ordinal == 6:
        if not _path_well_formed(TOKEN_PATH, ORGAN_GRAPH):
            raise rs.FailClosed("complete_token", "complete-token recipe is not well-formed")
        missing = [name for name in DEEP_PATH if name not in TOKEN_PATH]
        if missing:
            raise rs.FailClosed(
                "complete_token",
                "token recipe missing deep-path organs: " + ",".join(missing),
            )
        return payload_hash({"level": "V6", "token_path": list(TOKEN_PATH)})
    if ordinal == 7:
        if len(CAPABILITY_SUBSET) < 1:
            raise rs.FailClosed("capability_subset", "capability subset contract is empty")
        names = list(CAPABILITY_SUBSET)
        if names != sorted(names):
            names = sorted(names)
        return payload_hash({"level": "V7", "capability_subset": names})
    raise UnavailableLevelError(level)


# ---------------------------------------------------------------------------
# Evidence DAG. Edges are parent → child (child depends on parent).
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    level: str
    inputs: dict[str, str]
    code_sha256: str
    machine_genome: dict[str, Any]
    status: str = "UNPROVEN"
    proof_identity: str | None = None
    result_sha256: str | None = None


@dataclass
class ProofRecord:
    identity: str
    node_id: str
    level: str
    status: str
    result_sha256: str
    reuse_count: int = 0


@dataclass
class EvidenceDAG:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    proofs: dict[str, ProofRecord] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=rs.new_ledger)
    work_count: int = 0
    executed_node_ids: list[str] = field(default_factory=list)

    def add_node(
        self,
        node_id: str,
        level: str,
        inputs: Mapping[str, str],
        *,
        code_sha256: str | None = None,
        machine_genome: Mapping[str, Any] | None = None,
    ) -> Node:
        if node_id in self.nodes:
            raise rs.FailClosed("duplicate_node", f"duplicate node {node_id!r}")
        if str(level) not in LEVEL_BY_ID:
            raise rs.FailClosed("unknown_level", f"{level!r} is not a V0–V9 id")
        node = Node(
            id=str(node_id),
            level=str(level),
            inputs={k: inputs[k] for k in sorted(inputs)},
            code_sha256=code_sha256 or rs.fixture_code_sha256(),
            machine_genome=dict(machine_genome or rs.fixture_machine_genome()),
        )
        self.nodes[node.id] = node
        if node.id not in self.ledger["nodes"]:
            rs.ledger_add(self.ledger, node.id, "evidence")
        return node

    def add_edge(self, parent: str, child: str) -> None:
        if parent not in self.nodes or child not in self.nodes:
            raise rs.FailClosed(
                "missing_node", f"edge {parent!r}->{child!r} names a missing node"
            )
        if parent == child:
            raise rs.FailClosed("cycle", f"self-edge on {parent!r}")
        if (parent, child) in self.edges:
            return
        self.edges.append((parent, child))
        if self._creates_cycle():
            self.edges.pop()
            raise rs.FailClosed("cycle", f"edge {parent!r}->{child!r} would cycle")

    def _creates_cycle(self) -> bool:
        children: dict[str, list[str]] = {}
        for p, c in self.edges:
            children.setdefault(p, []).append(c)
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {nid: WHITE for nid in self.nodes}

        def dfs(nid: str) -> bool:
            colour[nid] = GREY
            for dst in children.get(nid, ()):
                if colour.get(dst, WHITE) == GREY:
                    return True
                if colour.get(dst, WHITE) == WHITE and dfs(dst):
                    return True
            colour[nid] = BLACK
            return False

        for nid in sorted(self.nodes):
            if colour[nid] == WHITE and dfs(nid):
                return True
        return False

    def parents(self, node_id: str) -> list[str]:
        return sorted(p for p, c in self.edges if c == node_id)

    def children(self, node_id: str) -> list[str]:
        return sorted(c for p, c in self.edges if p == node_id)

    def descendants(self, node_id: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(self.children(cur))
        return out

    def ancestors(self, node_id: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(self.parents(cur))
        return out

    def identity_for(self, node_id: str) -> str:
        node = self.nodes[node_id]
        inputs = dict(node.inputs)
        for pid in self.parents(node_id):
            parent = self.nodes[pid]
            inputs[f"parent:{pid}"] = parent.proof_identity or f"status:{parent.status}"
        return proof_identity(
            inputs=inputs,
            code_sha256=node.code_sha256,
            machine_genome=node.machine_genome,
            level=node.level,
        )

    def prove(self, node_id: str) -> ProofRecord:
        if node_id not in self.nodes:
            raise rs.FailClosed("missing_node", f"cannot prove missing node {node_id!r}")
        node = self.nodes[node_id]
        for pid in self.parents(node_id):
            pst = self.nodes[pid].status
            if pst != "VALID":
                raise rs.FailClosed(
                    "stale_parent",
                    f"parent {pid} is {pst}; refusing to prove {node_id} on a non-VALID parent",
                )
        request_level(node.level)
        ident = self.identity_for(node_id)
        cached = self.proofs.get(ident)
        if cached is not None and cached.status == "VALID":
            cached.reuse_count += 1
            node.status = "VALID"
            node.proof_identity = ident
            node.result_sha256 = cached.result_sha256
            return cached
        result = execute_level(node.level, node.inputs)
        self.work_count += 1
        self.executed_node_ids.append(node_id)
        rec = ProofRecord(
            identity=ident,
            node_id=node_id,
            level=node.level,
            status="VALID",
            result_sha256=result,
            reuse_count=0,
        )
        self.proofs[ident] = rec
        node.status = "VALID"
        node.proof_identity = ident
        node.result_sha256 = result
        return rec

    def attach_claim(self, claim_id: str, evidence_node_id: str) -> None:
        if evidence_node_id not in self.nodes:
            raise rs.FailClosed(
                "missing_node", f"claim {claim_id!r} names missing evidence {evidence_node_id!r}"
            )
        if evidence_node_id not in self.ledger["nodes"]:
            rs.ledger_add(self.ledger, evidence_node_id, "evidence")
        if claim_id not in self.ledger["nodes"]:
            rs.ledger_add(self.ledger, claim_id, "claim")
        rs.ledger_link(self.ledger, claim_id, evidence_node_id)

    def invalidate(self, node_id: str) -> list[str]:
        """Invalidate node_id and its descendants. Nothing else.

        Ancestors and sibling branches keep VALID proofs. Derived claims on
        the affected set are transitively DOWNGRADED via the repro_science ledger.
        """
        if node_id not in self.nodes:
            raise rs.FailClosed("missing_node", f"cannot invalidate missing node {node_id!r}")
        affected = [node_id] + sorted(self.descendants(node_id))
        for nid in affected:
            node = self.nodes[nid]
            old = node.proof_identity
            if old and old in self.proofs:
                self.proofs[old].status = "INVALID" if nid == node_id else "STALE"
            node.status = "INVALID" if nid == node_id else "STALE"
            node.proof_identity = None
            node.result_sha256 = None
            if nid in self.ledger["nodes"]:
                rs.ledger_invalidate(self.ledger, nid)
        return affected

    def mutate(self, node_id: str, new_inputs: Mapping[str, str]) -> list[str]:
        if node_id not in self.nodes:
            raise rs.FailClosed("missing_node", f"cannot mutate missing node {node_id!r}")
        node = self.nodes[node_id]
        node.inputs = {k: new_inputs[k] for k in sorted(new_inputs)}
        return self.invalidate(node_id)

    def statuses(self) -> dict[str, str]:
        return {nid: self.nodes[nid].status for nid in sorted(self.nodes)}

    def claim_statuses(self) -> dict[str, str]:
        return {k: self.ledger["nodes"][k]["status"] for k in sorted(self.ledger["nodes"])}


def make_diamond(*, payload: str = "diamond-v1") -> EvidenceDAG:
    """Constructed diamond: A → B, A → C, B → D, C → D.

    Mutating B must invalidate B and D and leave A and C VALID.
    """
    dag = EvidenceDAG()
    code = rs.fixture_code_sha256()
    genome = rs.fixture_machine_genome()
    schema = payload_hash({"schema": SCHEMA, "payload": payload})
    dag.add_node(
        "A",
        "V0",
        {"schema": schema, "payload": payload_hash(payload)},
        code_sha256=code,
        machine_genome=genome,
    )
    dag.add_node(
        "B",
        "V1",
        {"schema": schema, "branch": payload_hash("left"), "payload": payload_hash(payload)},
        code_sha256=code,
        machine_genome=genome,
    )
    dag.add_node(
        "C",
        "V1",
        {"schema": schema, "branch": payload_hash("right"), "payload": payload_hash(payload)},
        code_sha256=code,
        machine_genome=genome,
    )
    dag.add_node(
        "D",
        "V2",
        {"schema": schema, "organ": "affine", "payload": payload_hash(payload)},
        code_sha256=code,
        machine_genome=genome,
    )
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    dag.attach_claim("CLAIM_B", "B")
    dag.attach_claim("CLAIM_C", "C")
    dag.attach_claim("CLAIM_D", "D")
    dag.attach_claim("CLAIM_JOIN", "D")
    rs.ledger_link(dag.ledger, "CLAIM_JOIN", "CLAIM_B")
    return dag


def prove_diamond(dag: EvidenceDAG) -> None:
    for nid in ("A", "B", "C", "D"):
        dag.prove(nid)


# ---------------------------------------------------------------------------
# Cached invariant reuse. Keyed on live input bytes, never on a claim name.
# ---------------------------------------------------------------------------
#
# EvidenceDAG.prove() already reuses in-memory ProofRecords whose identity
# covers hashed inputs + code + genome + level. That cache dies with the
# process and never re-hashes a file. A 3h trial that "reuses" by claim
# name will happily replay a result computed from inputs that have since
# moved. This path is the disk check: sealed receipt, every named input
# still hashes, no scar landed against the family since the seal, evidence
# class covers the ask. Fail any one of those and the answer is RERUN,
# naming the condition. REUSE without reused_from is a silent cache and
# is refused.


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _canon_family(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def _looks_like_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _pathish(key: str) -> bool:
    """A recorded input we can re-hash. Logical names ('schema') are not files."""
    s = str(key)
    if "/" in s or "\\" in s:
        return True
    name = s.replace("\\", "/").rsplit("/", 1)[-1]
    return "." in name and not name.startswith(".")


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _receipt_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def evidence_class_sufficient(have: str, need: str) -> bool:
    """True iff the receipt proved at least as much as the ask requires."""
    if have not in EVIDENCE_CLASS_RANK or need not in EVIDENCE_CLASS_RANK:
        return False
    return EVIDENCE_CLASS_RANK[have] >= EVIDENCE_CLASS_RANK[need]


def extract_recorded_inputs(
    doc: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str | None]:
    """Pull path+sha256 rows out of a receipt. Empty/logical → no_recorded_inputs.

    `recorded_inputs` wins over `input_hashes` over `inputs`. An explicit empty
    list is "nothing to verify", not a cue to fall through to another key.
    """
    raw: Any = None
    for key in ("recorded_inputs", "input_hashes", "inputs"):
        if key in doc and doc[key] is not None:
            raw = doc[key]
            break
    if raw is None or raw == [] or raw == {}:
        return [], "no_recorded_inputs"
    rows: list[dict[str, str]] = []
    if isinstance(raw, Mapping):
        pathish_keys = [k for k in raw if _pathish(str(k))]
        if not pathish_keys:
            return [], "no_recorded_inputs"
        for path in sorted(pathish_keys, key=lambda x: str(x)):
            sha = raw[path]
            if not _looks_like_sha256(sha):
                return [], "no_recorded_inputs"
            rows.append({"path": str(path), "sha256": str(sha).lower()})
        return rows, None
    if isinstance(raw, list):
        if not raw:
            return [], "no_recorded_inputs"
        for item in raw:
            if not isinstance(item, Mapping):
                return [], "no_recorded_inputs"
            path = item.get("path") or item.get("rel") or item.get("file")
            sha = item.get("sha256") or item.get("hash") or item.get("digest")
            if not path or not _looks_like_sha256(sha):
                return [], "no_recorded_inputs"
            rows.append({"path": str(path), "sha256": str(sha).lower()})
        return rows, None
    return [], "no_recorded_inputs"


def receipt_written_at(doc: Mapping[str, Any]) -> datetime | None:
    bench = doc.get("bench") if isinstance(doc.get("bench"), Mapping) else {}
    for value in (
        doc.get("written_at"),
        doc.get("recorded_at"),
        bench.get("recorded_at") if isinstance(bench, Mapping) else None,
    ):
        parsed = _parse_ts(value)
        if parsed is not None:
            return parsed
    return None


def receipt_evidence_class(doc: Mapping[str, Any]) -> str | None:
    for value in (
        doc.get("evidence_class"),
        (doc.get("bench") or {}).get("measurement_state")
        if isinstance(doc.get("bench"), Mapping)
        else None,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_input_path(rel: str, *, root: Path) -> Path | None:
    """Live file or nothing. A sparse-checkout hole is missing, not 'use git'."""
    p = Path(rel)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(root / rel)
    if root.resolve() != REPO.resolve():
        candidates.append(REPO / rel)
    seen: set[str] = set()
    for cand in candidates:
        try:
            key = str(cand.resolve()) if cand.exists() else str(cand)
        except OSError:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def write_claim_receipt(
    *,
    claim: str,
    family: str,
    input_paths: Sequence[str | Path],
    dest: str | Path,
    evidence_class: str = "STATIC_ONLY",
    recorded_at: str | None = None,
    root: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a claim receipt that records live file hashes. Missing input raises.

    Writes wherever `dest` says so tests can land in tmp. Does not go through
    write_receipt: that helper always lands under receipts/future/.
    """
    dest_path = Path(dest)
    root_p = Path(root) if root is not None else dest_path.parent
    if not input_paths:
        raise rs.FailClosed(
            "no_recorded_inputs",
            f"refusing to seal claim {claim!r} with no inputs; nothing would be verifiable",
        )
    if evidence_class not in EVIDENCE_CLASS_RANK:
        raise rs.FailClosed(
            "unknown_evidence_class",
            f"cannot seal claim {claim!r} with unknown evidence_class {evidence_class!r}",
        )
    rows: list[dict[str, str]] = []
    for raw in input_paths:
        p = Path(raw)
        if not p.is_absolute():
            cand = root_p / p
            p = cand if cand.is_file() else p
        if not p.is_file():
            raise rs.FailClosed(
                "input_missing",
                f"cannot seal claim {claim!r}: input {raw} is not a file",
            )
        rows.append({"path": _relpath(p, root_p), "sha256": sha256_file(p)})
    written = recorded_at or _utc_now_stamp()
    doc: dict[str, Any] = {
        "schema": CLAIM_RECEIPT_SCHEMA,
        "version": 1,
        "claim": str(claim),
        "claim_family": str(family),
        "evidence_class": str(evidence_class),
        "gpu_authority": False,
        "recorded_inputs": rows,
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "STATIC_ONLY",
            "recorded_at": written,
            "recorded_by": RECORDED_BY,
            "gpu_authority": False,
        },
        "claim_boundary": "Static sidecar artifact. No hardware measurement.",
    }
    if extra:
        for key, value in extra.items():
            if key == "seal_sha256":
                continue
            doc[key] = value
    # seal_doc copies; writing the original would emit an unsealed receipt.
    sealed = rs.seal_doc(doc)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n")
    return sealed


def module_recorded_inputs() -> list[dict[str, Any]]:
    """File hashes this receipt is allowed to be reused against.

    Named because they actually moved the proof, not because they were nearby.
    A missing file is recorded as missing: dropping it would let reuse skip
    a dependency that is not here to verify.
    """
    rels = (
        "tools/future/evidence_dag.py",
        "tools/future/_common.py",
        "tools/future/repro_science.py",
        "tools/future/freshness.py",
        "tools/future/evidence_snapshot.py",
    )
    rows: list[dict[str, Any]] = []
    for rel in rels:
        path = REPO / rel
        if path.is_file():
            rows.append({"path": rel, "sha256": sha256_file(path), "present": True})
        else:
            rows.append(
                {
                    "path": rel,
                    "sha256": None,
                    "present": False,
                    "reason": "not on disk in this worktree; cannot be a reuse key",
                }
            )
    return rows


def load_default_scars() -> tuple[list[dict[str, Any]] | None, str]:
    """Scars this partition can actually consult.

    negative_index walks a corpus this sparse worktree does not materialise.
    autonomy_scars is in the write partition and is the orchestrator ledger
    reuse has to respect. A matching-family untimestamped scar still RERUNs.
    """
    try:
        from tools.future import autonomy_scars as aus

        rows: list[dict[str, Any]] = []
        for scar in aus.scars():
            if isinstance(scar, Mapping):
                rows.append(dict(scar))
        return rows, "tools.future.autonomy_scars.scars"
    except Exception as exc:
        return None, f"autonomy_scars_unavailable:{type(exc).__name__}"


def _scar_families(scar: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("family", "claim_family", "hypothesis_family"):
        value = scar.get(key)
        if isinstance(value, str) and value.strip():
            found.append(_canon_family(value))
    extra = scar.get("families")
    if isinstance(extra, list):
        for value in extra:
            if isinstance(value, str) and value.strip():
                found.append(_canon_family(value))
    # Preserve order, drop dupes.
    out: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _scar_id(scar: Mapping[str, Any]) -> str:
    for key in ("id", "scar_id", "original_id"):
        value = scar.get(key)
        if value not in (None, ""):
            return str(value)
    return "unnamed_scar"


def _scar_landed_at(scar: Mapping[str, Any]) -> datetime | None:
    for key in ("landed_at", "recorded_at", "at", "written_at"):
        parsed = _parse_ts(scar.get(key))
        if parsed is not None:
            return parsed
    return None


def _invalidating_scar(
    scar: Mapping[str, Any],
    *,
    family: str,
    written: datetime | None,
) -> dict[str, Any] | None:
    """Return a hit if this scar forces RERUN of `family`, else None.

    Matching family + no timestamp: cannot prove it did not land after the
    receipt, so RERUN. Matching family + landed after written_at: RERUN.
    Matching family + landed at or before written_at: the receipt already
    had to live with it. Different family: ignore.
    """
    want = _canon_family(family)
    if not want:
        return None
    families = _scar_families(scar)
    if want not in families:
        return None
    sid = _scar_id(scar)
    landed = _scar_landed_at(scar)
    if landed is None:
        return {
            "id": sid,
            "family": want,
            "failed_condition": "scar_untimestamped",
            "reason": (
                f"scar {sid} matches family {want} but has no landed_at; "
                "cannot prove it did not land after the receipt"
            ),
        }
    if written is None:
        return {
            "id": sid,
            "family": want,
            "failed_condition": "receipt_untimestamped",
            "reason": (
                f"scar {sid} matches family {want} but the receipt has no "
                "written_at; cannot prove the scar landed first"
            ),
        }
    if landed > written:
        return {
            "id": sid,
            "family": want,
            "failed_condition": "scar_after_receipt",
            "reason": (
                f"scar {sid} landed at {landed.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"after receipt written_at "
                f"{written.strftime('%Y-%m-%dT%H:%M:%SZ')} and invalidates "
                f"family {want}"
            ),
        }
    return None


def _verdict(
    decision: str,
    *,
    claim: str,
    reason: str,
    failed_condition: str | None = None,
    reused_from: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if decision == REUSE:
        if not reused_from or not reused_from.get("receipt") or not reused_from.get("digest"):
            raise rs.FailClosed(
                "reuse_unreported",
                "REUSE without reused_from (receipt path + digest) is a silent cache; refused",
            )
    body: dict[str, Any] = {
        "decision": decision,
        "reason": reason,
        "failed_condition": failed_condition,
        "claim": claim,
        "reused_from": reused_from if decision == REUSE else None,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    body.update(extra)
    return body


def _load_receipt_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, "missing_receipt"
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return None, "corrupt_receipt"
    if not isinstance(doc, dict):
        return None, "corrupt_receipt"
    return doc, None


def _resolve_claim(
    claim: str | Mapping[str, Any],
    *,
    receipts_dir: Path,
    root: Path,
) -> dict[str, Any]:
    """Normalise a name or envelope into {id, receipt, family, asking, ...}."""
    envelope: dict[str, Any]
    if isinstance(claim, Mapping):
        envelope = dict(claim)
    elif isinstance(claim, str):
        text = claim.strip()
        envelope = {"id": text}
        if text in CLAIM_CATALOG:
            envelope.update(CLAIM_CATALOG[text])
        elif text.endswith(".json"):
            envelope["receipt"] = text
    else:
        envelope = {"id": ""}

    cid = str(envelope.get("id") or envelope.get("claim") or "").strip()
    receipt_raw = envelope.get("receipt") or envelope.get("receipt_path")
    family = envelope.get("family") or envelope.get("claim_family")
    asking = envelope.get("asking_evidence_class") or envelope.get("required_evidence_class")
    if asking is None:
        asking = "STATIC_ONLY"

    receipt_path: Path | None = None
    if isinstance(receipt_raw, Path):
        receipt_path = receipt_raw
    elif isinstance(receipt_raw, str) and receipt_raw.strip():
        p = Path(receipt_raw.strip())
        if p.is_absolute():
            receipt_path = p
        elif (receipts_dir / p.name).is_file() and not p.parent.parts:
            receipt_path = receipts_dir / p.name
        elif (root / p).is_file():
            receipt_path = root / p
        elif (receipts_dir / p).is_file():
            receipt_path = receipts_dir / p
        elif (REPO / p).is_file():
            receipt_path = REPO / p
        else:
            receipt_path = receipts_dir / p if not p.parent.parts else root / p

    if receipt_path is None and cid in CLAIM_CATALOG:
        receipt_path = receipts_dir / CLAIM_CATALOG[cid]["receipt"]
        family = family or CLAIM_CATALOG[cid].get("family")
        envelope.setdefault("receipt", CLAIM_CATALOG[cid]["receipt"])

    envelope["id"] = cid
    envelope["family"] = family
    envelope["asking_evidence_class"] = str(asking)
    envelope["_receipt_path"] = receipt_path
    return envelope


def reuse_or_rerun(
    claim: str | Mapping[str, Any],
    *,
    asking_evidence_class: str | None = None,
    root: Path | None = None,
    receipts_dir: Path | None = None,
    scars: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """REUSE a sealed receipt or RERUN, with the condition that failed.

    REUSE requires all of:
      * the claim names a sealed receipt whose seal still verifies
      * every input the receipt names still hashes to the recorded value
      * no scar landed since that invalidates the claim's family
      * the receipt's evidence class is sufficient for the asking context

    A receipt that names no inputs can never be reused: nothing to verify.
    A named input that is gone is RERUN, never REUSE. The cache key is the
    input bytes, not the claim name. Silent reuse is refused: a REUSE
    result always carries reused_from {receipt, digest}.
    """
    root_p = Path(root) if root is not None else REPO
    rec_dir = Path(receipts_dir) if receipts_dir is not None else RECEIPTS
    env = _resolve_claim(claim, receipts_dir=rec_dir, root=root_p)
    cid = env["id"] or "<unnamed>"
    asking = str(asking_evidence_class or env["asking_evidence_class"] or "STATIC_ONLY")

    receipt_path: Path | None = env.get("_receipt_path")
    if receipt_path is None:
        return _verdict(
            RERUN,
            claim=cid,
            reason="claim names no receipt path and is not in CLAIM_CATALOG",
            failed_condition="missing_receipt",
            asking_evidence_class=asking,
        )
    if not receipt_path.is_file():
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt {_receipt_relpath(receipt_path)} is not on disk",
            failed_condition="missing_receipt",
            receipt_path=_receipt_relpath(receipt_path),
            asking_evidence_class=asking,
        )

    doc, load_fault = _load_receipt_file(receipt_path)
    if doc is None:
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt {_receipt_relpath(receipt_path)} could not be loaded ({load_fault})",
            failed_condition=load_fault or "corrupt_receipt",
            receipt_path=_receipt_relpath(receipt_path),
            asking_evidence_class=asking,
        )

    if not cid or cid == "<unnamed>":
        cid = str(doc.get("claim") or env.get("id") or receipt_path.stem)

    family = env.get("family") or doc.get("claim_family") or doc.get("family")
    family = str(family).strip() if family not in (None, "") else ""

    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or not seal:
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt {_receipt_relpath(receipt_path)} has no seal_sha256",
            failed_condition="unsealed_receipt",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )
    if not rs.seal_is_valid(doc):
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt {_receipt_relpath(receipt_path)} seal does not match the body",
            failed_condition="corrupt_receipt",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )

    inputs, input_fault = extract_recorded_inputs(doc)
    if input_fault == "no_recorded_inputs" or not inputs:
        return _verdict(
            RERUN,
            claim=cid,
            reason=(
                f"receipt {_receipt_relpath(receipt_path)} records no inputs; "
                "nothing to verify; refuse reuse"
            ),
            failed_condition="no_recorded_inputs",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    verified: list[dict[str, str]] = []
    for row in inputs:
        rel = row["path"]
        recorded = row["sha256"]
        live = resolve_input_path(rel, root=root_p)
        if live is None:
            missing.append(rel)
            continue
        try:
            current = sha256_file(live)
        except OSError:
            missing.append(rel)
            continue
        if current.lower() != recorded.lower():
            mismatched.append(
                {
                    "path": rel,
                    "recorded_sha256": recorded,
                    "current_sha256": current,
                }
            )
            continue
        verified.append({"path": rel, "sha256": current})

    if missing:
        return _verdict(
            RERUN,
            claim=cid,
            reason=(
                "input "
                + ", ".join(missing)
                + " named by the receipt is not on disk"
            ),
            failed_condition="input_missing",
            named_input=missing[0],
            named_inputs=missing,
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )
    if mismatched:
        first = mismatched[0]
        return _verdict(
            RERUN,
            claim=cid,
            reason=(
                f"input {first['path']} sha256 changed: "
                f"recorded={first['recorded_sha256']} current={first['current_sha256']}"
            ),
            failed_condition="input_hash_mismatch",
            named_input=first["path"],
            named_inputs=[m["path"] for m in mismatched],
            mismatches=mismatched,
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )

    written = receipt_written_at(doc)

    scar_rows: list[Mapping[str, Any]]
    scar_source: str
    if scars is not None:
        scar_rows = [s for s in scars if isinstance(s, Mapping)]
        scar_source = "caller"
    else:
        loaded, scar_source = load_default_scars()
        if loaded is None:
            return _verdict(
                RERUN,
                claim=cid,
                reason=f"scar source unavailable ({scar_source}); cannot establish no invalidating scar",
                failed_condition="scar_source_absent",
                receipt_path=_receipt_relpath(receipt_path),
                family=family or None,
                asking_evidence_class=asking,
            )
        scar_rows = loaded

    if scar_rows and not family:
        return _verdict(
            RERUN,
            claim=cid,
            reason=(
                "receipt/claim records no family and scars were consulted; "
                "cannot prove none of them invalidate this claim"
            ),
            failed_condition="claim_family_unrecorded",
            receipt_path=_receipt_relpath(receipt_path),
            asking_evidence_class=asking,
            scar_source=scar_source,
        )

    if family:
        for scar in scar_rows:
            hit = _invalidating_scar(scar, family=family, written=written)
            if hit is not None:
                return _verdict(
                    RERUN,
                    claim=cid,
                    reason=hit["reason"],
                    failed_condition=hit["failed_condition"],
                    named_scar=hit["id"],
                    family=family,
                    receipt_path=_receipt_relpath(receipt_path),
                    asking_evidence_class=asking,
                    scar_source=scar_source,
                )

    have_cls = receipt_evidence_class(doc)
    if have_cls is None:
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt {_receipt_relpath(receipt_path)} records no evidence_class",
            failed_condition="evidence_class_unrecorded",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
        )
    if asking not in EVIDENCE_CLASS_RANK:
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"asking evidence class {asking!r} is not in {sorted(EVIDENCE_CLASS_RANK)}",
            failed_condition="evidence_class_unknown",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
            receipt_evidence_class=have_cls,
        )
    if have_cls not in EVIDENCE_CLASS_RANK:
        return _verdict(
            RERUN,
            claim=cid,
            reason=f"receipt evidence class {have_cls!r} is not in {sorted(EVIDENCE_CLASS_RANK)}",
            failed_condition="evidence_class_unknown",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
            receipt_evidence_class=have_cls,
        )
    if not evidence_class_sufficient(have_cls, asking):
        return _verdict(
            RERUN,
            claim=cid,
            reason=(
                f"receipt evidence class {have_cls} is not sufficient for "
                f"asking context {asking}"
            ),
            failed_condition="evidence_class_insufficient",
            receipt_path=_receipt_relpath(receipt_path),
            family=family or None,
            asking_evidence_class=asking,
            receipt_evidence_class=have_cls,
        )

    digest = str(doc["seal_sha256"])
    reused_from = {
        "receipt": _receipt_relpath(receipt_path),
        "digest": digest,
        "inputs_verified": verified,
    }
    return _verdict(
        REUSE,
        claim=cid,
        reason=(
            "sealed receipt; every named input still hashes; "
            "no invalidating scar since seal; evidence class sufficient"
        ),
        reused_from=reused_from,
        receipt_path=_receipt_relpath(receipt_path),
        family=family or None,
        asking_evidence_class=asking,
        receipt_evidence_class=have_cls,
        n_inputs_verified=len(verified),
        scar_source=scar_source if scars is not None or scar_rows else "caller_empty",
    )


def execute_reuse_workunit(
    claim: str | Mapping[str, Any] = CACHED_INVARIANT_CLAIM,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run reuse_or_rerun and shape a WorkUnit result. REUSE always reports it."""
    verdict = reuse_or_rerun(claim, **kwargs)
    result: dict[str, Any] = {
        "id": "future.evidence-dag.reuse-or-rerun",
        "claim": verdict.get("claim"),
        "decision": verdict["decision"],
        "reason": verdict["reason"],
        "failed_condition": verdict.get("failed_condition"),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "resource_class": "STATIC_ANALYSIS",
        "reused_from": verdict.get("reused_from"),
    }
    if verdict["decision"] == REUSE and not result["reused_from"]:
        raise rs.FailClosed(
            "reuse_unreported",
            "WorkUnit REUSE result missing reused_from; silent cache refused",
        )
    return result


def cached_invariant_reuse_proof() -> dict[str, Any]:
    """Four negative controls plus the earned REUSE path, on temp files.

    A validator nobody has watched reject is a validator that will drift.
    """
    with tempfile.TemporaryDirectory(prefix="evidence-dag-reuse-") as td:
        root = Path(td)
        payload = b"cached-invariant-v1\n"
        inp = root / "fixture.bin"
        inp.write_bytes(payload)
        dest = root / "CLAIM.json"
        write_claim_receipt(
            claim="fixture.cached_invariant",
            family="fixture.family",
            input_paths=[inp],
            dest=dest,
            root=root,
            recorded_at="2026-08-01T00:00:00Z",
        )
        envelope: dict[str, Any] = {
            "id": "fixture.cached_invariant",
            "receipt": str(dest),
            "family": "fixture.family",
            "asking_evidence_class": "STATIC_ONLY",
        }

        reuse = reuse_or_rerun(envelope, root=root, receipts_dir=root, scars=[])
        if reuse["decision"] != REUSE:
            raise rs.FailClosed(
                "cached_invariant",
                f"identical inputs did not REUSE: {reuse}",
            )
        reused_from = reuse.get("reused_from") or {}
        if reused_from.get("receipt") is None or reused_from.get("digest") is None:
            raise rs.FailClosed(
                "cached_invariant",
                f"REUSE did not report reused_from: {reuse}",
            )
        sealed = json.loads(dest.read_text())
        if reused_from.get("digest") != sealed.get("seal_sha256"):
            raise rs.FailClosed(
                "cached_invariant",
                "reused_from.digest is not the receipt seal",
            )
        wu = execute_reuse_workunit(envelope, root=root, receipts_dir=root, scars=[])
        if wu["decision"] != REUSE or not wu.get("reused_from"):
            raise rs.FailClosed(
                "cached_invariant",
                f"WorkUnit result hid reuse: {wu}",
            )

        # One input byte flips REUSE -> RERUN and names that input.
        flipped = bytearray(payload)
        flipped[0] ^= 0x01
        inp.write_bytes(bytes(flipped))
        after_flip = reuse_or_rerun(envelope, root=root, receipts_dir=root, scars=[])
        if after_flip["decision"] != RERUN:
            raise rs.FailClosed(
                "cached_invariant",
                f"byte-flipped input still REUSE: {after_flip}",
            )
        if after_flip.get("failed_condition") != "input_hash_mismatch":
            raise rs.FailClosed(
                "cached_invariant",
                f"byte-flip named {after_flip.get('failed_condition')}, not input_hash_mismatch",
            )
        if after_flip.get("named_input") != "fixture.bin":
            raise rs.FailClosed(
                "cached_invariant",
                f"byte-flip did not name the input: {after_flip}",
            )
        if after_flip.get("reused_from") is not None:
            raise rs.FailClosed(
                "cached_invariant",
                "RERUN carried reused_from; that is silent success",
            )

        # Restore, then delete: missing named input is RERUN, never REUSE.
        inp.write_bytes(payload)
        restored = reuse_or_rerun(envelope, root=root, receipts_dir=root, scars=[])
        if restored["decision"] != REUSE:
            raise rs.FailClosed(
                "cached_invariant",
                f"restored bytes did not REUSE: {restored}",
            )
        inp.unlink()
        after_del = reuse_or_rerun(envelope, root=root, receipts_dir=root, scars=[])
        if after_del["decision"] != RERUN or after_del.get("failed_condition") != "input_missing":
            raise rs.FailClosed(
                "cached_invariant",
                f"deleted input was not input_missing RERUN: {after_del}",
            )
        if after_del.get("named_input") != "fixture.bin":
            raise rs.FailClosed(
                "cached_invariant",
                f"deleted input did not name fixture.bin: {after_del}",
            )

        inp.write_bytes(payload)
        scar_after = {
            "id": "SCAR.fixture.after",
            "family": "fixture.family",
            "landed_at": "2026-08-15T00:00:00Z",
        }
        after_scar = reuse_or_rerun(
            envelope, root=root, receipts_dir=root, scars=[scar_after]
        )
        if after_scar["decision"] != RERUN or after_scar.get("failed_condition") != "scar_after_receipt":
            raise rs.FailClosed(
                "cached_invariant",
                f"scar after receipt did not RERUN: {after_scar}",
            )
        if after_scar.get("named_scar") != "SCAR.fixture.after":
            raise rs.FailClosed(
                "cached_invariant",
                f"scar after receipt did not name the scar: {after_scar}",
            )

        scar_before = {
            "id": "SCAR.fixture.before",
            "family": "fixture.family",
            "landed_at": "2026-07-01T00:00:00Z",
        }
        before_scar = reuse_or_rerun(
            envelope, root=root, receipts_dir=root, scars=[scar_before]
        )
        if before_scar["decision"] != REUSE:
            raise rs.FailClosed(
                "cached_invariant",
                f"scar landed before receipt forced RERUN: {before_scar}",
            )

        other_family = {
            "id": "SCAR.other",
            "family": "other.family",
            "landed_at": "2026-08-20T00:00:00Z",
        }
        other = reuse_or_rerun(
            envelope, root=root, receipts_dir=root, scars=[other_family]
        )
        if other["decision"] != REUSE:
            raise rs.FailClosed(
                "cached_invariant",
                f"unrelated-family scar forced RERUN: {other}",
            )

        empty_dest = root / "EMPTY.json"
        empty_doc = {
            "schema": CLAIM_RECEIPT_SCHEMA,
            "version": 1,
            "claim": "fixture.empty",
            "claim_family": "fixture.family",
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
            "recorded_inputs": [],
            "bench": {
                "state": "UNKNOWN",
                "measurement_state": "STATIC_ONLY",
                "recorded_at": "2026-08-01T00:00:00Z",
                "recorded_by": RECORDED_BY,
                "gpu_authority": False,
            },
        }
        empty_doc = rs.seal_doc(empty_doc)
        empty_dest.write_text(json.dumps(empty_doc, indent=1, sort_keys=True) + "\n")
        empty_v = reuse_or_rerun(
            {
                "id": "fixture.empty",
                "receipt": str(empty_dest),
                "family": "fixture.family",
                "asking_evidence_class": "STATIC_ONLY",
            },
            root=root,
            receipts_dir=root,
            scars=[],
        )
        if empty_v["decision"] != RERUN or empty_v.get("failed_condition") != "no_recorded_inputs":
            raise rs.FailClosed(
                "cached_invariant",
                f"empty recorded_inputs was not no_recorded_inputs RERUN: {empty_v}",
            )

        protected = reuse_or_rerun(
            envelope,
            root=root,
            receipts_dir=root,
            scars=[],
            asking_evidence_class="PROTECTED_ABSOLUTE",
        )
        if (
            protected["decision"] != RERUN
            or protected.get("failed_condition") != "evidence_class_insufficient"
        ):
            raise rs.FailClosed(
                "cached_invariant",
                f"STATIC_ONLY receipt satisfied a protected ask: {protected}",
            )

    return {
        "holds": True,
        "reuse_on_identical_inputs": True,
        "reused_from_reported": True,
        "reused_from_digest_is_seal": True,
        "workunit_result_carries_reused_from": True,
        "byte_flip_named_the_input": True,
        "deleted_input_is_rerun": True,
        "scar_after_receipt_is_rerun": True,
        "scar_before_receipt_still_reuses": True,
        "unrelated_family_scar_does_not_rerun": True,
        "no_recorded_inputs_never_reuses": True,
        "static_receipt_insufficient_for_protected_ask": True,
        "cache_key": "recorded input sha256, not claim name",
    }


# ---------------------------------------------------------------------------
# WorkUnits the resident can schedule. V8 is SLEEPING, never synthetic.
# ---------------------------------------------------------------------------


def emit_work_units() -> list[dict[str, Any]]:
    selftest = wus.emit_hcli_workunit(
        id="future.evidence-dag.selftest",
        role="science",
        description=(
            "Run the evidence DAG selftest: V0–V9 catalog, reuse, diamond "
            "invalidation, adaptive depth, V8/V9 refusal. Seals EVIDENCE_DAG.json."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.evidence_dag.selftest",
        provider="tools.future.evidence_dag",
        effect_class="READ_ONLY",
        status="pending",
        extras={
            "command": ["python3", "tools/future/evidence_dag.py", "--selftest"],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "claim_boundary": (
                "Static sidecar artifact. No hardware measurement. "
                "Cannot promote, take a GPU lease, or lower a verification bar."
            ),
        },
    )
    wus.validate_emitted_unit(selftest)

    adapt = wus.emit_hcli_workunit(
        id="future.evidence-dag.adapt-next-mutation",
        role="science",
        description=(
            "Choose required_level for the next mutation from scope, uncertainty, "
            "risk, upside and promotion proximity. Do not run the full suite by default."
        ),
        dependencies=["future.evidence-dag.selftest"],
        resource_class="STATIC_ANALYSIS",
        verifier="future.evidence_dag.required_level",
        provider="tools.future.evidence_dag",
        effect_class="READ_ONLY",
        status="pending",
        extras={
            "command": [
                "python3",
                "tools/future/evidence_dag.py",
                "--required-level",
                "-",
            ],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "claim_boundary": (
                "Static sidecar artifact. Adaptive depth is a plan, not a promotion."
            ),
        },
    )
    wus.validate_emitted_unit(adapt)

    sleeping = wus.emit_hcli_workunit(
        id="future.evidence-dag.v8-protected-capability",
        role="science",
        description=(
            "SLEEPING: V8 protected capability/performance. Wakes when a protected "
            "GPU lease exists and the machine is QUIESCENT. Never a synthetic result."
        ),
        dependencies=["future.evidence-dag.selftest"],
        resource_class="GPU_EXCLUSIVE",
        verifier="future.evidence_dag.request_level_v8",
        provider="tools.future.evidence_dag",
        preferred_backend="metal",
        status="blocked",
        classification="BLOCKED",
        extras={
            "sleeping": True,
            "wakes_when": "protected GPU authority qualifies AND machine is QUIESCENT",
            "blocked_reason": (
                "V8 is UNAVAILABLE: this sidecar has no GPU authority. "
                "The unit sleeps until hardware qualifies. It is not a V7 result."
            ),
            "requested_level": "V8",
            "species": "independent_reproduction",
            "requires_quiescence": True,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "claim_boundary": (
                "Proposal only. Sidecar cannot acquire a GPU lease or write a "
                "protected measurement. A sleeping unit is not a synthetic V8."
            ),
        },
    )
    wus.validate_emitted_unit(sleeping)

    reuse_unit = wus.emit_hcli_workunit(
        id="future.evidence-dag.reuse-or-rerun",
        role="science",
        description=(
            "Reuse a sealed claim if every named input still hashes, no scar "
            "has landed against its family since the seal, and the receipt's "
            "evidence class covers the ask. Otherwise RERUN, naming the "
            "condition. A REUSE result MUST carry reused_from."
        ),
        dependencies=["future.evidence-dag.selftest"],
        resource_class="STATIC_ANALYSIS",
        verifier="future.evidence_dag.reuse_or_rerun",
        provider="tools.future.evidence_dag",
        effect_class="READ_ONLY",
        status="pending",
        extras={
            "command": [
                "python3",
                "tools/future/evidence_dag.py",
                "--reuse-or-rerun",
                CACHED_INVARIANT_CLAIM,
            ],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "claim": CACHED_INVARIANT_CLAIM,
            "result_must_carry": "reused_from",
            "claim_boundary": (
                "Static sidecar artifact. No hardware measurement. "
                "Reuse is earned by live input hashes, not by claim name."
            ),
        },
    )
    wus.validate_emitted_unit(reuse_unit)
    return [selftest, adapt, reuse_unit, sleeping]


# ---------------------------------------------------------------------------
# Proofs that selftest / build seal into the receipt.
# ---------------------------------------------------------------------------


def levels_catalog_proof() -> dict[str, Any]:
    rows = levels()
    ids = [r["id"] for r in rows]
    holds = ids == [f"V{i}" for i in range(len(rows))]
    if not holds:
        raise rs.FailClosed("levels", f"level ids are not V0..Vn: {ids}")
    for row in rows:
        if not row["proves"] or not row["does_not_prove"]:
            raise rs.FailClosed("levels", f"{row['id']} missing proves/does_not_prove")
        if row["requires_gpu"] and row["availability"] == "AVAILABLE":
            raise rs.FailClosed("levels", f"{row['id']} requires GPU but is AVAILABLE")
        if (not row["requires_gpu"]) and row["availability"] != "AVAILABLE":
            raise rs.FailClosed("levels", f"{row['id']} is CPU-static but UNAVAILABLE")
    v8 = LEVEL_BY_ID["V8"]
    v9 = LEVEL_BY_ID["V9"]
    if v8["availability"] != "UNAVAILABLE" or v9["availability"] != "UNAVAILABLE":
        raise rs.FailClosed("levels", "V8/V9 must be UNAVAILABLE on this sidecar")
    return {
        "n_levels": len(rows),
        "ids": ids,
        "v8_availability": v8["availability"],
        "v9_availability": v9["availability"],
        "holds": True,
    }


def identity_sensitivity_proof() -> dict[str, Any]:
    genome = rs.fixture_machine_genome()
    code = rs.fixture_code_sha256()
    inputs = {"schema": payload_hash({"s": 1}), "payload": payload_hash("p")}
    a = proof_identity(inputs=inputs, code_sha256=code, machine_genome=genome, level="V1")
    b = proof_identity(inputs=inputs, code_sha256=code, machine_genome=genome, level="V1")
    other_in = dict(inputs)
    other_in["payload"] = payload_hash("q")
    c = proof_identity(inputs=other_in, code_sha256=code, machine_genome=genome, level="V1")
    d = proof_identity(inputs=inputs, code_sha256="c" * 64, machine_genome=genome, level="V1")
    other_g = dict(genome)
    other_g["arch"] = "x86_64"
    e = proof_identity(inputs=inputs, code_sha256=code, machine_genome=other_g, level="V1")
    f = proof_identity(inputs=inputs, code_sha256=code, machine_genome=genome, level="V2")
    proof = {
        "reproduces": a == b,
        "input_sensitive": a != c,
        "code_sensitive": a != d,
        "machine_genome_sensitive": a != e,
        "level_sensitive": a != f,
        "identity": a,
    }
    if not all(
        proof[k]
        for k in (
            "reproduces",
            "input_sensitive",
            "code_sensitive",
            "machine_genome_sensitive",
            "level_sensitive",
        )
    ):
        raise rs.FailClosed("proof_identity", f"identity is not immutable/sensitive: {proof}")
    return proof


def reuse_skips_work_proof() -> dict[str, Any]:
    dag = make_diamond()
    prove_diamond(dag)
    after_first = dag.work_count
    first_executed = list(dag.executed_node_ids)
    c1 = dag.prove("C")
    after_reuse = dag.work_count
    c2 = dag.prove("C")
    after_reuse2 = dag.work_count
    skipped = after_reuse == after_first == after_reuse2
    reused = c1.reuse_count >= 1 and c2.reuse_count >= 2
    if not skipped or not reused:
        raise rs.FailClosed(
            "reuse",
            f"reuse did not skip work: work={after_first}->{after_reuse}->{after_reuse2} "
            f"reuse_count={c2.reuse_count}",
        )
    # Changing inputs must NOT reuse.
    dag.mutate("C", {**dag.nodes["C"].inputs, "payload": payload_hash("mutated-c")})
    # C is now INVALID; parents of C (A) still VALID, so C can be re-proven.
    dag.prove("C")
    after_change = dag.work_count
    if after_change != after_first + 1:
        raise rs.FailClosed(
            "reuse",
            f"changed inputs still skipped work: {after_first} then {after_change}",
        )
    return {
        "work_first_pass": after_first,
        "executed_first_pass": first_executed,
        "work_after_identical_reuse": after_reuse2,
        "reuse_count_on_C": c2.reuse_count,
        "skipped_work_on_identical_inputs": True,
        "work_after_input_change": after_change,
        "changed_inputs_did_not_reuse": True,
    }


def diamond_invalidation_proof() -> dict[str, Any]:
    dag = make_diamond()
    prove_diamond(dag)
    before = dag.statuses()
    claims_before = dag.claim_statuses()
    work_before = dag.work_count
    c_ident = dag.nodes["C"].proof_identity
    c_reuse_before = dag.proofs[c_ident].reuse_count if c_ident else 0

    affected = dag.mutate(
        "B",
        {**dag.nodes["B"].inputs, "payload": payload_hash("mutated-left-branch")},
    )
    after = dag.statuses()
    claims_after = dag.claim_statuses()

    sibling_survived = after["C"] == "VALID" and after["A"] == "VALID"
    mutated_and_descendants = after["B"] == "INVALID" and after["D"] == "STALE"
    precise = set(affected) == {"B", "D"}
    if not (sibling_survived and mutated_and_descendants and precise):
        raise rs.FailClosed(
            "diamond_invalidation",
            f"precise invalidation failed: affected={affected} after={after}",
        )

    # Sibling proof actually still skips work.
    dag.prove("C")
    sibling_reused = dag.work_count == work_before
    if not sibling_reused:
        raise rs.FailClosed(
            "diamond_invalidation",
            f"sibling C was re-executed after mutating B; work {work_before}->{dag.work_count}",
        )

    # D must refuse until B is re-proven (stale parent), not silently pass.
    d_refused = False
    try:
        dag.prove("D")
    except rs.FailClosed as exc:
        d_refused = exc.fault == "stale_parent"
    if not d_refused:
        raise rs.FailClosed("diamond_invalidation", "D proved while parent B was INVALID")

    claim_join_down = claims_after.get("CLAIM_JOIN") == "DOWNGRADED"
    claim_b_down = claims_after.get("CLAIM_B") == "DOWNGRADED"
    claim_c_ok = claims_after.get("CLAIM_C") == "VALID"
    if not (claim_join_down and claim_b_down and claim_c_ok):
        raise rs.FailClosed(
            "claim_downgrade",
            f"claim downgrade on diamond failed: {claims_after}",
        )

    c_ident_after = dag.nodes["C"].proof_identity
    reuse_after = dag.proofs[c_ident_after].reuse_count if c_ident_after else 0
    return {
        "before": before,
        "after": after,
        "affected": affected,
        "sibling_C_survived": sibling_survived,
        "sibling_C_reused_without_work": sibling_reused,
        "sibling_C_reuse_count_before": c_reuse_before,
        "sibling_C_reuse_count_after": reuse_after,
        "D_refused_on_stale_parent": d_refused,
        "claims_before": claims_before,
        "claims_after": claims_after,
        "claim_C_stayed_VALID": claim_c_ok,
        "precise": precise,
        "holds": True,
    }


def adaptive_depth_proof() -> dict[str, Any]:
    cheap = required_level(
        mutation_scope="tiny_numerical",
        uncertainty=0.05,
        risk=0.05,
        upside=0.10,
        promotion_proximity=0.0,
    )
    if level_ordinal(cheap) >= 8:
        raise rs.FailClosed(
            "adaptive_depth",
            f"cheap reversible candidate required {cheap}; full-suite reflex is live",
        )
    organ_low = required_level("organ", 0.10, 0.10, 0.10, 0.0)
    organ_unc = required_level("organ", 0.90, 0.10, 0.10, 0.0)
    organ_risk = required_level("organ", 0.10, 0.90, 0.10, 0.0)
    organ_up = required_level("organ", 0.10, 0.10, 0.90, 0.0)
    deep = required_level("deep_chain", 0.10, 0.10, 0.10, 0.0)
    promo = required_level("organ", 0.05, 0.05, 0.10, 0.90)
    tourney = required_level("organ", 0.05, 0.05, 0.10, 0.97)
    explicit_promo = required_level("promotion", 0.0, 0.0, 0.0, 0.0)

    if not (level_ordinal(organ_unc) > level_ordinal(organ_low)):
        raise rs.FailClosed("adaptive_depth", "uncertainty did not raise required_level")
    if not (level_ordinal(organ_risk) > level_ordinal(organ_low)):
        raise rs.FailClosed("adaptive_depth", "risk did not raise required_level")
    if not (level_ordinal(organ_up) > level_ordinal(organ_low)):
        raise rs.FailClosed("adaptive_depth", "upside did not raise required_level")
    if not (level_ordinal(deep) > level_ordinal(required_level("schema", 0.1, 0.1, 0.1, 0.0))):
        raise rs.FailClosed("adaptive_depth", "mutation_scope did not raise required_level")
    if promo != "V8":
        raise rs.FailClosed(
            "adaptive_depth",
            f"promotion-adjacent required {promo}, not V8 — the bar was lowered",
        )
    if tourney != "V9" or explicit_promo != "V9":
        raise rs.FailClosed(
            "adaptive_depth",
            f"tournament-adjacent required tourney={tourney} explicit={explicit_promo}",
        )
    # Unavailability must not rewrite the bar.
    if LEVEL_BY_ID[promo]["availability"] != "UNAVAILABLE":
        raise rs.FailClosed("adaptive_depth", "V8 catalog row is not UNAVAILABLE")

    cheap_admitted = admit_candidate(
        mutation_scope="tiny_numerical",
        uncertainty=0.05,
        risk=0.05,
        upside=0.10,
        promotion_proximity=0.0,
        achieved_level=cheap,
    )
    below_refused = False
    below_required = None
    try:
        admit_candidate(
            mutation_scope="organ",
            uncertainty=0.05,
            risk=0.05,
            upside=0.10,
            promotion_proximity=0.90,
            achieved_level="V7",
        )
    except BelowRequiredLevelError as exc:
        below_refused = exc.required == "V8" and exc.achieved == "V7"
        below_required = exc.required
    if not below_refused:
        raise rs.FailClosed(
            "adaptive_depth",
            "promotion-adjacent candidate was admitted below required_level",
        )

    v8_claim_refused = False
    try:
        admit_candidate(
            mutation_scope="organ",
            uncertainty=0.05,
            risk=0.05,
            upside=0.10,
            promotion_proximity=0.90,
            achieved_level="V8",
        )
    except UnavailableLevelError as exc:
        v8_claim_refused = exc.level == "V8"
    if not v8_claim_refused:
        raise rs.FailClosed(
            "adaptive_depth",
            "achieved V8 on this host was admitted; that is a silent mint of protected proof",
        )

    return {
        "cheap_required": cheap,
        "cheap_not_v8": level_ordinal(cheap) < 8,
        "uncertainty_raises": True,
        "risk_raises": True,
        "upside_raises": True,
        "scope_raises": True,
        "promotion_adjacent_required": promo,
        "tournament_adjacent_required": tourney,
        "explicit_promotion_scope_required": explicit_promo,
        "bar_not_lowered_because_unavailable": promo == "V8",
        "cheap_admitted": cheap_admitted,
        "promotion_adjacent_v7_refused": below_refused,
        "promotion_adjacent_required_on_refusal": below_required,
        "promotion_adjacent_v8_claim_refused": v8_claim_refused,
        "holds": True,
    }


def unavailable_level_proof() -> dict[str, Any]:
    """Requesting V8/V9 RAISES. The return value is never a weaker level."""
    v8_raised = False
    v8_returned = None
    try:
        v8_returned = request_level("V8")
    except UnavailableLevelError as exc:
        v8_raised = exc.level == "V8" and exc.fault == "unavailable_level"
    v9_raised = False
    try:
        request_level("V9")
    except UnavailableLevelError as exc:
        v9_raised = exc.level == "V9"
    v7 = request_level("V7")
    if v8_returned is not None:
        raise rs.FailClosed(
            "unavailable_level",
            f"request_level('V8') returned {v8_returned!r} instead of raising",
        )
    if not v8_raised or not v9_raised:
        raise rs.FailClosed(
            "unavailable_level",
            f"V8/V9 request did not raise: v8_raised={v8_raised} v9_raised={v9_raised}",
        )
    if v7["id"] != "V7" or v7["availability"] != "AVAILABLE":
        raise rs.FailClosed("unavailable_level", "V7 should be AVAILABLE")
    # Proving a V8 node must raise before work is counted.
    dag = EvidenceDAG()
    dag.add_node("P8", "V8", {"schema": payload_hash("v8")})
    work_before = dag.work_count
    prove_raised = False
    try:
        dag.prove("P8")
    except UnavailableLevelError as exc:
        prove_raised = exc.level == "V8"
    if not prove_raised or dag.work_count != work_before or dag.nodes["P8"].status == "VALID":
        raise rs.FailClosed(
            "unavailable_level",
            "V8 prove() did not raise or counted work / stored a VALID proof",
        )
    return {
        "request_V8_raises": v8_raised,
        "request_V9_raises": v9_raised,
        "request_V8_returned": v8_returned,
        "silent_downgrade_to_V7": False,
        "V7_available": True,
        "prove_V8_raises_before_work": prove_raised,
        "work_count_unchanged": dag.work_count == work_before,
        "no_synthetic_V8_proof": dag.nodes["P8"].status != "VALID",
        "holds": True,
    }


def static_ladder_proof() -> dict[str, Any]:
    """Prove every AVAILABLE level once so execute_level is not a dead catalog."""
    dag = EvidenceDAG()
    code = rs.fixture_code_sha256()
    genome = rs.fixture_machine_genome()
    schema = payload_hash({"ladder": "v0-v7"})
    available = [s["id"] for s in _LEVELS if s["availability"] == "AVAILABLE"]
    prev = None
    for lid in available:
        spec = LEVEL_BY_ID[lid]
        inputs: dict[str, str] = {"schema": schema, "payload": payload_hash(lid)}
        if spec["ordinal"] == 2:
            inputs["organ"] = "affine"
        if spec["ordinal"] == 3:
            inputs["held_out_organ"] = HELD_OUT_ORGAN
        dag.add_node(lid, lid, inputs, code_sha256=code, machine_genome=genome)
        if prev is not None:
            dag.add_edge(prev, lid)
        rec = dag.prove(lid)
        if rec.status != "VALID":
            raise rs.FailClosed("static_ladder", f"{lid} did not prove VALID")
        prev = lid
    if dag.work_count != len(available):
        raise rs.FailClosed(
            "static_ladder",
            f"expected {len(available)} executions, got {dag.work_count}",
        )
    # Re-proving the top available level must skip work.
    top = available[-1]
    dag.prove(top)
    if dag.work_count != len(available):
        raise rs.FailClosed("static_ladder", "re-proving top available level did not reuse")
    return {
        "proved": available,
        "n_proved": len(available),
        "work_count": dag.work_count,
        "reused_top": True,
        "holds": True,
    }


def funnel_mapping() -> list[dict[str, Any]]:
    """Map the live qualification funnel keys onto V-levels. Counts from disk."""
    mapped = dict(FUNNEL_TO_LEVEL)
    path = REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
    if not path.is_file():
        return [{"funnel": k, "level": v, "n_ids": None, "source": "catalog"} for k, v in FUNNEL_TO_LEVEL]
    doc = load_json(path)
    funnel = doc.get("funnel") or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(funnel):
        if key == "promotion_rule":
            continue
        value = funnel[key]
        n_ids = len(value) if isinstance(value, list) else None
        rows.append(
            {
                "funnel": key,
                "level": mapped.get(key, "unmapped"),
                "n_ids": n_ids,
                "source": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            }
        )
    return rows


def claim_downgrade_on_dag_proof() -> dict[str, Any]:
    """Invalidating a parent transitively downgrades derived claims; siblings live.

    Uses the repro_science ledger (not a fork) wired onto DAG evidence nodes.
    """
    diamond = diamond_invalidation_proof()
    # Also run the foundation proof so a later reader sees we still honour it.
    foundation = rs.transitive_downgrade_proof()
    if not foundation["transitivity_holds"]:
        raise rs.FailClosed("claim_downgrade", "repro_science ledger transitivity failed")
    return {
        "diamond": {
            "CLAIM_B": diamond["claims_after"]["CLAIM_B"],
            "CLAIM_C": diamond["claims_after"]["CLAIM_C"],
            "CLAIM_D": diamond["claims_after"]["CLAIM_D"],
            "CLAIM_JOIN": diamond["claims_after"]["CLAIM_JOIN"],
        },
        "sibling_claim_survived": diamond["claim_C_stayed_VALID"],
        "foundation_transitivity_holds": foundation["transitivity_holds"],
        "holds": True,
    }


def run_all_proofs() -> dict[str, Any]:
    catalog = levels_catalog_proof()
    identity = identity_sensitivity_proof()
    ladder = static_ladder_proof()
    reuse = reuse_skips_work_proof()
    diamond = diamond_invalidation_proof()
    adaptive = adaptive_depth_proof()
    unavailable = unavailable_level_proof()
    claims = claim_downgrade_on_dag_proof()
    cached = cached_invariant_reuse_proof()
    units = emit_work_units()
    return {
        "levels": catalog,
        "identity": identity,
        "static_ladder": ladder,
        "reuse": reuse,
        "cached_invariant": cached,
        "diamond": diamond,
        "adaptive_depth": adaptive,
        "unavailable_levels": unavailable,
        "claim_downgrade": claims,
        "work_units": units,
    }


# ---------------------------------------------------------------------------
# Recovery notes, sealed into the receipt so a later reader does not re-derive.
# ---------------------------------------------------------------------------


def _probe(rel: str) -> dict[str, Any]:
    p = REPO / rel
    in_git = bool(git("ls-tree", "--name-only", "HEAD", rel).strip())
    return {"path": rel, "on_disk": p.is_file(), "in_git": bool(in_git)}


RECOVERED_IMPLEMENTATION = [
    {
        "path": "tools/future/repro_science.py",
        "what": (
            "Immutable experiment identity, provenance graph, claim ledger with "
            "transitive DOWNGRADED, FailClosed, content_hash. THIS IS THE FOUNDATION."
        ),
        "use": (
            "Imported. proof_identity wraps experiment_identity; invalidate() calls "
            "ledger_invalidate; FailClosed is subclassed, not forked."
        ),
    },
    {
        "path": "tools/future/_common.py",
        "what": "write_receipt seals STATIC_ONLY/UNKNOWN and raises HardwareClaimError",
        "use": "used as-is; this module does not reimplement or weaken HARDWARE_FIELDS",
    },
    {
        "path": "tools/headless/composition_ladder.py",
        "what": (
            "8-rung qualification ladder; a candidate that fails a rung is killed "
            "there; unreached ≠ failed; a screen verdict is not a model verdict"
        ),
        "use": (
            "Law kept as V0–V9 mapping (LADDER_TO_LEVEL). Not imported (Codex tools/ "
            "surface; sparse checkout). highest-claimable is admit_candidate here."
        ),
        "probe": "git show HEAD:tools/headless/composition_ladder.py",
    },
    {
        "path": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        "what": (
            "funnel: static_validation → native_parity → diagnostic_relative_ab → "
            "protected_absolute_complete_wall → promotion. measurement_contract "
            "nulls until a native protected complete-token receipt."
        ),
        "use": "FUNNEL_TO_LEVEL. V8/V9 map onto the last two rungs and stay UNAVAILABLE.",
    },
    {
        "path": "tools/future/lpc_dataset.py",
        "what": "Missing keys REJECTED; nulls carry a reason and are never imputed to 0",
        "use": "stale_parent refuse: a missing/INVALID parent is not treated as VALID",
    },
    {
        "path": "tools/future/integration_attack.py",
        "what": "Adversarial completion attack and severity model; a guard never watched to fail is not a guard",
        "use": "diamond sibling, below-required admission, and V8-raise are the negative controls",
    },
    {
        "path": "lab/verification_authority.py",
        "what": "Models propose; protected controller decides; forbidden self-promotion",
        "use": "this module never emits PROTECTED_ABSOLUTE or DIAGNOSTIC_RELATIVE",
        "probe": "git show HEAD:lab/verification_authority.py",
    },
    {
        "path": "hcli/dag_store.py",
        "what": "Durable DAG; disk is authority; interrupted ≠ verifier failure",
        "use": "in-memory EvidenceDAG is the static analog; swap named as integration point",
        "probe": "git show HEAD:hcli/dag_store.py",
    },
    {
        "path": "tools/future/workunit_species.py",
        "what": "HCLI WorkUnit field set, emit_hcli_workunit, validate_emitted_unit",
        "use": "selftest + adapt-next-mutation pending units; V8 unit blocked/SLEEPING",
    },
    {
        "path": "tools/future/qualification_pipeline.py",
        "what": "execute() raises without an existing HCLI lease AND quiescence AND --execute",
        "use": "same authority boundary: V8 is a sleeping unit, never seized",
    },
    {
        "path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
        "what": "49-system sidecar inventory; autonomous reproducible science already executable",
        "use": "repro_science was adequate for identity/downgrade; this module closes the V-hierarchy gap",
    },
    {
        "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
        "what": "F008 provenance gap (closed by repro_science). No frontier file is rewritten here.",
        "use": "WorkUnit refill is how a result changes the next verification depth",
    },
    {
        "path": "tools/future/_common.py",
        "what": "sha256_file + seal; HARDWARE_FIELDS raise on a numeric claim",
        "use": "reuse_or_rerun re-hashes every named input with sha256_file; does not reimplement the hasher",
    },
    {
        "path": "tools/future/freshness.py",
        "what": "byte sha vs semantic fingerprint for derived artifacts; FRESH only when sha matches",
        "use": (
            "The byte-match half is the same guarantee. Semantic fingerprint is "
            "deliberately NOT used here: one input byte must RERUN even if meaning is identical. "
            "A stale baseline surviving a cosmetic rewrite is the defect this lane kills."
        ),
    },
    {
        "path": "tools/future/evidence_snapshot.py",
        "what": "pinned copy + verify() that every captured file still hashes to the manifest",
        "use": "same live-hash-against-recorded-digest pattern; applied to named claims rather than the snapshot set",
    },
    {
        "path": "tools/future/repro_science.py",
        "what": "seal_is_valid, FailClosed, experiment_identity",
        "use": "reuse_or_rerun refuses an unsealed or corrupt receipt via seal_is_valid; does not fork the sealer",
    },
    {
        "path": "tools/future/autonomy_scars.py",
        "what": "orchestrator-defect scars with families; default scar source when the caller passes none",
        "use": "load_default_scars(). Matching family + landed_at after receipt_written_at forces RERUN",
    },
]

GAPS_CLOSED = [
    "V0–V9 catalog: each level declares what it proves and what it does not",
    "proof identity is a content hash over inputs + code + machine genome + level; reuse skips work on byte-identical inputs and does not skip when inputs change",
    "precise invalidation on a constructed diamond: mutating B invalidates B and D; sibling C's proof survives and is reused without work",
    "adaptive required_level(mutation_scope, uncertainty, risk, upside, promotion_proximity): cheap reversible stays below V8; promotion-adjacent floors at V8 even though V8 is UNAVAILABLE",
    "admission is two-sided: below-required raises; a V8 claim on this host also raises (no V7 consolation prize)",
    "request_level(V8) and request_level(V9) RAISE UnavailableLevelError rather than returning a weaker level; prove() does not count work or store a VALID V8 proof",
    "claim downgrade on the DAG uses repro_science.ledger_invalidate: parent invalidation transitively DOWNGRADES derived claims; sibling claims stay VALID",
    "V8 work is a SLEEPING/blocked HCLI WorkUnit, never a synthetic protected result",
    "reuse_or_rerun(claim) -> REUSE|RERUN: sealed receipt + live input sha256 match + no family scar since seal + evidence class sufficient; otherwise RERUN naming the failed condition",
    "a receipt that names no inputs can never be reused (nothing to verify); a named input that is gone is RERUN, never REUSE",
    "REUSE is reported: WorkUnit result carries reused_from {receipt, digest}; silent cache is FailClosed reuse_unreported",
    "one input byte, a deleted input, a post-seal family scar, and an empty recorded_inputs list each flip REUSE -> RERUN",
]

NEGATIVE_FINDINGS = [
    "this sidecar has no protected GPU lease; V8 and V9 are UNAVAILABLE by authority, not by a Metal probe (Metal was not queried)",
    "xcrun / Metal compiler / qualification HEAVY classification were not re-measured; Codex's blockers are accepted as given and become a sleeping WorkUnit",
    "tools/headless/composition_ladder.py and lab/verification_authority.py may be hidden by the sparse checkout; recovered via git show, not live import",
    "this module does not rewrite receipts/future/CLAUDE_GLOBAL_FRONTIER.json (owner: tools/future/global_frontier.py, prohibited)",
    "this-wave siblings (resident_api, workgraph, wakeup, protected_window, frontiers, super_resident) were not imported; local interfaces are named as integration points",
    "no DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE number was produced; bench.state stays UNKNOWN",
    "Flash NX remains SCAFFOLD_ONLY / teacher capture 0/256 — out of this lane; noted, not papered over",
    "freshness semantic fingerprints are not a reuse key; a byte change is RERUN even when meaning is identical",
    "default scar source is autonomy_scars in this partition; the negative_index corpus is not consulted (sparse checkout, hypothesis-keyed, not this DAG's families)",
    "a receipt that records only logical input names (schema/payload) has nothing to re-hash and cannot be reused by this checker",
    "STATIC_ONLY reuse never satisfies a DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE ask; this sidecar mints neither",
]


def resident_callable_doc(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "hcli_can_invoke": True,
        "entry_point": "python3 tools/future/evidence_dag.py --selftest",
        "discover": [
            "python3 tools/future/evidence_dag.py --selftest",
            "python3 tools/future/evidence_dag.py --build",
            "python3 tools/future/evidence_dag.py --required-level <candidate.json>",
            "python3 tools/future/evidence_dag.py --admit <candidate.json>",
            "python3 tools/future/evidence_dag.py --reuse-or-rerun evidence_dag.cached_invariant",
        ],
        "invoke": (
            "tools.future.evidence_dag:build|selftest|required_level|"
            "admit_candidate|request_level|EvidenceDAG.prove|"
            "reuse_or_rerun|execute_reuse_workunit"
        ),
        "schedule": [u["id"] for u in units],
        "work_units_emitted": [u["id"] for u in units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": {
            "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "writes_frontier_file": False,
            "how": (
                "A sealed proof changes which verification depth is still required. "
                "The next WorkUnit is refilled at required_level rather than the full "
                "suite. The frontier JSON is owned by global_frontier.py and is not "
                "mutated here."
            ),
            "related_entries": ["F008"],
        },
        "verify": "python3 -m pytest tools/future/test_evidence_dag.py -q",
        "fail_closed": [
            "request_level('V8'|'V9') raises UnavailableLevelError rather than returning a weaker level",
            "admit_candidate below required_level raises BelowRequiredLevelError rather than clamping to what is available",
            "an achieved V8 claim on this host raises UnavailableLevelError (no synthetic protected proof)",
            "prove() with a non-VALID parent raises FailClosed stale_parent",
            "write_receipt raises HardwareClaimError on a numeric hardware field",
            "unknown mutation_scope or out-of-range factor raises FailClosed",
            "a V8 WorkUnit is blocked/SLEEPING and is never a VALID proof record",
            "reuse_or_rerun returns RERUN (never REUSE) when the receipt is missing, unsealed, corrupt, names no inputs, names a missing input, names a hash-mismatched input, has a post-seal family scar, or is weaker than the asking evidence class",
            "REUSE without reused_from (receipt path + digest) raises FailClosed reuse_unreported",
        ],
        "result_changes_a_frontier": (
            "required_level output is the next verification WorkUnit's depth; "
            "a reused VALID proof is not re-run; an invalidated descendant is; "
            "reuse_or_rerun is the 3h-trial WorkUnit that demonstrates cached-invariant reuse"
        ),
        "workunit": (
            "one STATIC_ANALYSIS unit; reuse_or_rerun a named claim; "
            "REUSE result carries reused_from"
        ),
        "frontier": "FT.VERIFICATION.repro",
        "persists": f"receipts/future/{RECEIPT} via write_receipt",
        "next_work_refills": "future.evidence-dag.adapt-next-mutation depends on selftest",
    }


def build() -> Path:
    proofs = run_all_proofs()
    units = proofs["work_units"]
    recorded = module_recorded_inputs()
    verifiable_inputs = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in recorded
        if row.get("present") and _looks_like_sha256(row.get("sha256"))
    ]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "V0–V9 evidence DAG: reuse byte-identical proofs, invalidate only "
            "affected descendants, choose verification depth from mutation "
            "scope / uncertainty / risk / upside / promotion proximity, refuse "
            "V8/V9 on this host rather than silently downgrading, and reuse a "
            "sealed claim only when every named input still hashes."
        ),
        "claim": CACHED_INVARIANT_CLAIM,
        "claim_family": CACHED_INVARIANT_FAMILY,
        "recorded_inputs": verifiable_inputs,
        "recorded_inputs_probe": recorded,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "fpga_note": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. "
            "It is not its own civilization and this module does not build an FPGA backend."
        ),
        "measurement_states": {
            "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine; guides; never promotes; this module does not produce it",
            "PROTECTED_ABSOLUTE": "measurement under a real protected GPU lease; decides; this module does not produce it",
            "STATIC_ONLY": "everything this module emits",
        },
        "levels": levels(),
        "n_levels": len(_LEVELS),
        "funnel_mapping": funnel_mapping(),
        "ladder_mapping": [{"rung": a, "level": b} for a, b in LADDER_TO_LEVEL],
        "static_ladder": proofs["static_ladder"],
        "host": {
            "gpu_authority": False,
            "protected_gpu": False,
            "v8_availability": LEVEL_BY_ID["V8"]["availability"],
            "v9_availability": LEVEL_BY_ID["V9"]["availability"],
            "note": (
                "Unavailability is an authority fact of this sidecar, not a device "
                "query. MetalContext / xcrun were not probed."
            ),
        },
        "adaptive_depth": {
            "function": "required_level(mutation_scope, uncertainty, risk, upside, promotion_proximity)",
            "proximity_v8": PROXIMITY_V8,
            "proximity_v9": PROXIMITY_V9,
            "proof": proofs["adaptive_depth"],
        },
        "reuse": proofs["reuse"],
        "cached_invariant_reuse": proofs["cached_invariant"],
        "diamond_invalidation": proofs["diamond"],
        "unavailable_levels": proofs["unavailable_levels"],
        "claim_downgrade": proofs["claim_downgrade"],
        "proof_identity": proofs["identity"],
        "work_units": units,
        "n_work_units": len(units),
        "recovered_implementation": RECOVERED_IMPLEMENTATION,
        "recovery_probes": [
            _probe("tools/future/repro_science.py"),
            _probe("tools/headless/composition_ladder.py"),
            _probe("lab/verification_authority.py"),
            _probe("hcli/dag_store.py"),
            _probe("receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"),
            _probe("receipts/future/FUTURE_SUBSTRATE_HANDOFF.json"),
            _probe("receipts/future/CLAUDE_GLOBAL_FRONTIER.json"),
            _probe("receipts/future/REPRO_SCIENCE.json"),
        ],
        "gaps_closed": GAPS_CLOSED,
        "negative_findings": NEGATIVE_FINDINGS,
        "resident_callable": resident_callable_doc(units),
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. "
            "V8/V9 are UNAVAILABLE and must be requested as sleeping work, never synthesized."
        ),
        "integration_points": {
            "repro_science": "identity, FailClosed, claim ledger — imported",
            "workunit_species": "emit_hcli_workunit — imported",
            "resident_api.py": "this-wave; not imported. Swap: expose required_level/admit_candidate/request_level",
            "workgraph.py": "this-wave; not imported. Swap: persist EvidenceDAG through DagStore",
            "protected_window.py": "this-wave; not imported. Only that module may flip V8/V9 to AVAILABLE",
            "wakeup.py": "this-wave; not imported. Wakes the SLEEPING V8 WorkUnit when hardware qualifies",
            "frontiers.py": "this-wave; not imported. Consumes required_level to refill verification depth",
            "hcli/dag_store.py": "disk authority for live DAGs; this module's graph is in-memory + receipt",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def _load_candidate(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text == "-":
        text = _sys.stdin.read()
    elif not text.startswith("{") and not text.startswith("["):
        text = Path(text).read_text(encoding="utf-8")
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise rs.FailClosed("invalid_candidate", "candidate must be a JSON object")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--required-level", metavar="CANDIDATE")
    ap.add_argument("--admit", metavar="CANDIDATE")
    ap.add_argument("--reuse-or-rerun", metavar="CLAIM")
    args = ap.parse_args()
    if args.reuse_or_rerun is not None:
        raw = args.reuse_or_rerun.strip()
        try:
            loaded = json.loads(raw) if raw.startswith("{") or raw.startswith("[") else raw
        except json.JSONDecodeError:
            loaded = raw
        if isinstance(loaded, list):
            print(
                json.dumps(
                    {
                        "decision": RERUN,
                        "failed_condition": "unknown_claim",
                        "reason": "claim must be a name or a JSON object, not a list",
                        "reused_from": None,
                        "evidence_class": "STATIC_ONLY",
                        "gpu_authority": False,
                    },
                    indent=1,
                    sort_keys=True,
                )
            )
            return 1
        try:
            verdict = execute_reuse_workunit(loaded)
        except rs.FailClosed as exc:
            print(
                json.dumps(
                    {
                        "decision": RERUN,
                        "failed_condition": exc.fault,
                        "reason": exc.reason,
                        "reused_from": None,
                        "evidence_class": "STATIC_ONLY",
                        "gpu_authority": False,
                    },
                    indent=1,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(verdict, indent=1, sort_keys=True))
        return 0 if verdict["decision"] == REUSE else 1
    if args.required_level is not None:
        try:
            cand = _load_candidate(args.required_level)
            req = required_level(
                cand["mutation_scope"],
                cand["uncertainty"],
                cand["risk"],
                cand["upside"],
                cand["promotion_proximity"],
            )
        except rs.FailClosed as exc:
            print(
                json.dumps(
                    {"verdict": "REFUSED", "fault": exc.fault, "reason": exc.reason},
                    indent=1,
                    sort_keys=True,
                )
            )
            return 1
        print(
            json.dumps(
                {"required_level": req, "availability": LEVEL_BY_ID[req]["availability"]},
                indent=1,
                sort_keys=True,
            )
        )
        return 0
    if args.admit is not None:
        try:
            cand = _load_candidate(args.admit)
            verdict = admit_candidate(
                mutation_scope=cand["mutation_scope"],
                uncertainty=cand["uncertainty"],
                risk=cand["risk"],
                upside=cand["upside"],
                promotion_proximity=cand["promotion_proximity"],
                achieved_level=cand["achieved_level"],
            )
        except rs.FailClosed as exc:
            payload: dict[str, Any] = {
                "verdict": "REFUSED",
                "fault": exc.fault,
                "reason": exc.reason,
            }
            for key in ("level", "achieved", "required"):
                if hasattr(exc, key):
                    payload[key] = getattr(exc, key)
            print(json.dumps(payload, indent=1, sort_keys=True))
            return 1
        print(json.dumps({"verdict": verdict}, indent=1, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
