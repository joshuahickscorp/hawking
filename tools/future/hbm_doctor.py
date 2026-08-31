"""HBM Doctor — what should live in 8 GB of HBM.

An FPGA card's HBM is a BANDWIDTH SHARD, not overflow RAM. This module
answers the residency question by ranking candidates on marginal
complete-token critical-path reduction per resident byte. A missing
input makes the score UNKNOWN; it is not defaulted to zero and it is
not estimated. FPGA latency is a CLASS (LOW/MEDIUM/HIGH), never a
number — there is no board.

FPGA sits in Accelerator / Physical Compiler / Fusion. This is not an
FPGA backend and not a new civilization.

    python3 tools/future/hbm_doctor.py --build
    python3 tools/future/hbm_doctor.py --budget-bytes 8589934592
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "hawking.future.hbm_doctor.v1"
RECEIPT = "HBM_DOCTOR.json"
DEFAULT_BUDGET_BYTES = 8 * 1024 ** 3  # 8 GiB; configurable. Not read off a board.

FPGA_LATENCY_CLASSES = frozenset({"LOW", "MEDIUM", "HIGH"})
TRANSPORT_CLASSES = frozenset({"HOST_DRAM", "PCIE", "DISK", "ON_CHIP", "HBM"})
STATE_LIFETIMES = frozenset(
    {"STATIC_WEIGHTS", "SEQUENCE_STATE", "CONDITIONAL", "TOKEN_EPHEMERAL"}
)

# Contract-named paths. Absence is a negative finding, not a reason to invent.
CONTRACT_CENSUS = "receipts/headless/FLASH_ORGAN_CENSUS.json"
CONTRACT_QWEN27_BUDGET = "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json"
CONTRACT_LAYER30_CP = "receipts/headless/FLASH_LAYER30_CRITICAL_PATH.json"
CONTRACT_LAYER10_CP = "receipts/headless/FLASH_LAYER10_CRITICAL_PATH.json"

# Closest disk-backed substitutes (git HEAD; may be sparse-absent on disk).
FLASH_SCIENCE = "receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
FLASH_EBPW = "receipts/headless/FLASH_EBPW_BUDGET.json"
FLASH_TOKEN_NS = "receipts/headless/FLASH_TOKEN_NS_BUDGET.json"
FLASH_FPGA_MAP = "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json"
NOETIC_CENSUS = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
BYTES_ATLAS_RECEIPT = "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"
BYTES_ATLAS_MODULE = "tools/accelerator/bytes_atlas.py"
FRONTIER_RECEIPT = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"

# FPGA map names do not equal science organ ids. Join is explicit, not fuzzy.
FPGA_PRIORITY_JOIN: dict[str, str] = {
    "routed_experts": "expert_bank",
    "router": "router_topk_and_gather",
    "shared_expert": "routed_plus_shared_expert",
    "recurrent_state": "deltanet_persistent_state",
    "ngram_engine": "ngram_lookup_or_generator",
    "sparse_attention": "sparse_attention",
    "mtp": "mtp_draft_verify_rollback",
    "deltanet": "deltanet_persistent_state",
}

SCORE_MEANING = (
    "cited_weighted_latency / bytes, where cited_weighted_latency = "
    "mac_latency_ns * access_probability * reuse_count * dependency_criticality. "
    "mac_latency_ns is copied from a real receipt or is None; it is never estimated. "
    "projected_fpga_latency is a CLASS and is never converted into nanoseconds. "
    "The product is a ranking key, not an FPGA measurement."
)


class FpgaLatencyMustBeClass(ValueError):
    """projected_fpga_latency arrived as a number. There is no board."""


def _git_has(rel: str) -> bool:
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def load_repo_json(rel: str) -> dict[str, Any] | None:
    """Load JSON from the working tree, else from git HEAD (sparse checkout)."""
    p = REPO / rel
    if p.is_file():
        return load_json(p)
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nested(node: Any, *path: str) -> Any:
    cur = node
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class ResidentCandidate:
    """One HBM-residency candidate. Unknown inputs stay None, never 0-filled."""

    id: str
    bytes: int | None
    access_probability: float | None
    reuse_count: float | None
    transport_cost_class: str | None
    mac_latency_ns: float | None
    projected_fpga_latency: str | None
    state_lifetime: str | None
    dependency_criticality: float | None
    representation_format: str | None
    corpus: str = "unspecified"
    bytes_kind: str | None = None
    access_probability_source: str | None = None
    mac_latency_ns_source: str | None = None
    source_active_bytes_per_token: int | None = None
    fpga_map_priority: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        lat = self.projected_fpga_latency
        if _is_number(lat):
            raise FpgaLatencyMustBeClass(
                f"{self.id}: projected_fpga_latency={lat!r} is a number; "
                "FPGA latency is a CLASS (LOW/MEDIUM/HIGH), never a number"
            )
        if lat is not None and lat not in FPGA_LATENCY_CLASSES and lat != "UNKNOWN":
            raise FpgaLatencyMustBeClass(
                f"{self.id}: projected_fpga_latency={lat!r} is not a known class"
            )

    def missing_inputs(self) -> list[str]:
        missing: list[str] = []
        if not isinstance(self.bytes, int) or self.bytes <= 0:
            missing.append("bytes")
        if self.access_probability is None:
            missing.append("access_probability")
        elif not (0.0 <= float(self.access_probability) <= 1.0):
            missing.append("access_probability")
        if self.reuse_count is None:
            missing.append("reuse_count")
        if self.transport_cost_class not in TRANSPORT_CLASSES:
            missing.append("transport_cost_class")
        if self.mac_latency_ns is None:
            missing.append("mac_latency_ns")
        if self.projected_fpga_latency not in FPGA_LATENCY_CLASSES:
            missing.append("projected_fpga_latency")
        if self.state_lifetime not in STATE_LIFETIMES:
            missing.append("state_lifetime")
        if self.dependency_criticality is None:
            missing.append("dependency_criticality")
        if not self.representation_format or self.representation_format == "UNKNOWN":
            missing.append("representation_format")
        return missing

    def is_decidable(self) -> bool:
        return not self.missing_inputs()

    def cited_weighted_latency(self) -> float | None:
        if not self.is_decidable():
            return None
        assert self.mac_latency_ns is not None
        assert self.access_probability is not None
        assert self.reuse_count is not None
        assert self.dependency_criticality is not None
        return (
            float(self.mac_latency_ns)
            * float(self.access_probability)
            * float(self.reuse_count)
            * float(self.dependency_criticality)
        )

    def score_per_byte(self) -> float | str:
        missing = self.missing_inputs()
        if missing:
            return "UNKNOWN"
        value = self.cited_weighted_latency()
        assert value is not None and self.bytes
        return value / float(self.bytes)

    def record(self) -> dict[str, Any]:
        body = asdict(self)
        body["notes"] = list(self.notes)
        body["missing_inputs"] = self.missing_inputs()
        body["decidable"] = self.is_decidable()
        score = self.score_per_byte()
        body["score"] = score
        body["cited_weighted_latency"] = self.cited_weighted_latency()
        return body


@dataclass(frozen=True)
class SolveResult:
    budget_bytes: int
    selected: tuple[ResidentCandidate, ...]
    rejected: tuple[tuple[ResidentCandidate, str], ...]
    undecidable: tuple[ResidentCandidate, ...]
    selected_value: float
    selected_bytes: int

    def selected_ids(self) -> list[str]:
        return [c.id for c in self.selected]


def make_candidate(id: str, **kwargs: Any) -> ResidentCandidate:
    """Test/helper constructor. Defaults are DECIDABLE; override to punch holes."""
    base: dict[str, Any] = {
        "id": id,
        "bytes": 1,
        "access_probability": 1.0,
        "reuse_count": 1.0,
        "transport_cost_class": "PCIE",
        "mac_latency_ns": 1.0,
        "projected_fpga_latency": "MEDIUM",
        "state_lifetime": "STATIC_WEIGHTS",
        "dependency_criticality": 1.0,
        "representation_format": "test-format",
        "corpus": "test",
    }
    base.update(kwargs)
    return ResidentCandidate(**base)


def score_candidate(candidate: ResidentCandidate) -> dict[str, Any]:
    """Public scoring entry. UNKNOWN if any required input is missing."""
    missing = candidate.missing_inputs()
    if missing:
        return {
            "id": candidate.id,
            "score": "UNKNOWN",
            "missing_inputs": missing,
            "bucket": "undecidable",
            "cited_weighted_latency": None,
        }
    return {
        "id": candidate.id,
        "score": candidate.score_per_byte(),
        "missing_inputs": [],
        "bucket": "decidable",
        "cited_weighted_latency": candidate.cited_weighted_latency(),
    }


def _exact_knapsack(
    items: list[ResidentCandidate], budget: int
) -> tuple[tuple[ResidentCandidate, ...], float, int]:
    """0/1 knapsack maximising cited_weighted_latency under a byte budget.

    n is the organ count (Flash has 13). Enumeration is exact and deterministic:
    ties break toward less occupancy, then lexicographically smaller id-set.
    """
    ordered = sorted(items, key=lambda c: c.id)
    n = len(ordered)
    if n == 0:
        return (), 0.0, 0
    if n > 22:
        raise ValueError(
            f"exact knapsack refuses n={n} > 22; split the corpus rather than "
            "discretising the 8 GiB axis"
        )
    values = [c.cited_weighted_latency() or 0.0 for c in ordered]
    weights = [int(c.bytes or 0) for c in ordered]
    best_v = -1.0
    best_w = budget + 1
    best_mask = 0
    for mask in range(1 << n):
        w = 0
        v = 0.0
        feasible = True
        for i in range(n):
            if (mask >> i) & 1:
                w += weights[i]
                if w > budget:
                    feasible = False
                    break
                v += values[i]
        if not feasible:
            continue
        ids = tuple(ordered[i].id for i in range(n) if (mask >> i) & 1)
        better = False
        if v > best_v + 1e-15:
            better = True
        elif abs(v - best_v) <= 1e-15:
            if w < best_w:
                better = True
            elif w == best_w:
                current_ids = tuple(
                    ordered[i].id for i in range(n) if (best_mask >> i) & 1
                )
                if ids < current_ids:
                    better = True
        if better:
            best_v, best_w, best_mask = v, w, mask
    chosen = tuple(ordered[i] for i in range(n) if (best_mask >> i) & 1)
    return chosen, (best_v if best_v >= 0 else 0.0), (best_w if best_w <= budget else 0)


def size_ranked_select(
    items: list[ResidentCandidate], budget: int
) -> tuple[ResidentCandidate, ...]:
    """Naive fill-HBM-with-the-largest-items policy. The anti-pattern."""
    taken: list[ResidentCandidate] = []
    remaining = budget
    ranked = sorted(
        [c for c in items if isinstance(c.bytes, int) and c.bytes > 0],
        key=lambda c: (-int(c.bytes), c.id),
    )
    for cand in ranked:
        assert cand.bytes is not None
        if cand.bytes <= remaining:
            taken.append(cand)
            remaining -= cand.bytes
    return tuple(taken)


def solve(candidates: list[ResidentCandidate], budget_bytes: int) -> SolveResult:
    """Bounded knapsack over DECIDABLE items only. Undecidable never enter."""
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be >= 0")
    undecidable = tuple(
        sorted(
            [c for c in candidates if not c.is_decidable()],
            key=lambda c: c.id,
        )
    )
    decidable = [c for c in candidates if c.is_decidable()]
    selected, value, used = _exact_knapsack(decidable, budget_bytes)
    selected_ids = {c.id for c in selected}
    rejected: list[tuple[ResidentCandidate, str]] = []
    for cand in sorted(decidable, key=lambda c: c.id):
        if cand.id in selected_ids:
            continue
        assert cand.bytes is not None
        if cand.bytes > budget_bytes:
            reason = "bytes_exceed_budget"
        elif (cand.cited_weighted_latency() or 0.0) == 0.0:
            reason = "zero_cited_weighted_latency"
        else:
            reason = "not_in_optimal_set"
        rejected.append((cand, reason))
    return SolveResult(
        budget_bytes=budget_bytes,
        selected=selected,
        rejected=tuple(rejected),
        undecidable=undecidable,
        selected_value=value,
        selected_bytes=used,
    )


def refuse_size_ranking(
    candidates: list[ResidentCandidate], budget_bytes: int
) -> dict[str, Any]:
    """Compare size-greedy to the criticality-per-byte knapsack.

    Returns fired=True only when size-ranking is strictly worse on the
    DECIDABLE subset. A guard that cannot fire is not a guard.
    """
    decidable = [c for c in candidates if c.is_decidable()]
    if not decidable:
        return {
            "fired": False,
            "vacuous": True,
            "reason": "no_decidable_items",
            "size_ranked_ids": [],
            "objective_ids": [],
            "size_ranked_value": 0.0,
            "objective_value": 0.0,
        }
    objective = solve(decidable, budget_bytes)
    size_sel = size_ranked_select(decidable, budget_bytes)
    size_val = sum(c.cited_weighted_latency() or 0.0 for c in size_sel)
    fired = size_val + 1e-12 < objective.selected_value
    return {
        "fired": fired,
        "vacuous": False,
        "reason": (
            "size_ranking_captures_less_cited_weighted_latency"
            if fired
            else "size_ranking_not_worse_on_this_set"
        ),
        "size_ranked_ids": [c.id for c in size_sel],
        "objective_ids": objective.selected_ids(),
        "size_ranked_value": size_val,
        "objective_value": objective.selected_value,
        "size_ranked_bytes": sum(int(c.bytes or 0) for c in size_sel),
        "objective_bytes": objective.selected_bytes,
        "why_size_is_the_wrong_objective": (
            "HBM is a bandwidth shard. Ranking by stored bytes maximises "
            "occupancy, not complete-token critical-path reduction per "
            "resident byte. A large cold table crowds out a small hot path."
        ),
    }


def _present(rel: str) -> dict[str, Any]:
    on_disk = (REPO / rel).is_file()
    in_git = _git_has(rel)
    return {
        "path": rel,
        "on_disk": on_disk,
        "in_git": in_git,
        "present": on_disk or in_git,
    }


def recover_inputs() -> dict[str, Any]:
    """Inspect the contract paths and the closest existing substitutes."""
    consulted = {
        "flash_organ_census": _present(CONTRACT_CENSUS),
        "qwen27_token_ns_budget": _present(CONTRACT_QWEN27_BUDGET),
        "flash_layer30_critical_path": _present(CONTRACT_LAYER30_CP),
        "flash_layer10_critical_path": _present(CONTRACT_LAYER10_CP),
        "flash_science": _present(FLASH_SCIENCE),
        "flash_ebpw_budget": _present(FLASH_EBPW),
        "flash_token_ns_budget": _present(FLASH_TOKEN_NS),
        "flash_fpga_organ_map": _present(FLASH_FPGA_MAP),
        "noetic_organ_census": _present(NOETIC_CENSUS),
        "bytes_atlas_receipt": _present(BYTES_ATLAS_RECEIPT),
        "bytes_atlas_module": _present(BYTES_ATLAS_MODULE),
        "global_frontier": _present(FRONTIER_RECEIPT),
    }
    docs = {
        "flash_science": load_repo_json(FLASH_SCIENCE) if consulted["flash_science"]["present"] else None,
        "flash_ebpw": load_repo_json(FLASH_EBPW) if consulted["flash_ebpw_budget"]["present"] else None,
        "flash_token_ns": load_repo_json(FLASH_TOKEN_NS) if consulted["flash_token_ns_budget"]["present"] else None,
        "flash_fpga_map": load_repo_json(FLASH_FPGA_MAP) if consulted["flash_fpga_organ_map"]["present"] else None,
        "noetic_census": load_repo_json(NOETIC_CENSUS) if consulted["noetic_organ_census"]["present"] else None,
        "bytes_atlas": load_repo_json(BYTES_ATLAS_RECEIPT) if consulted["bytes_atlas_receipt"]["present"] else None,
        "frontier": load_repo_json(FRONTIER_RECEIPT) if consulted["global_frontier"]["present"] else None,
    }
    return {"consulted": consulted, "docs": docs}


def consume_bytes_atlas(doc: dict[str, Any] | None) -> dict[str, Any]:
    """Structural byte ranking only. No TPS, no GB/s, no token_ns copied."""
    if not doc:
        return {
            "consumed": False,
            "reason": "ACCELERATOR_TOKEN_BYTES_ATLAS.json not present",
        }
    pareto = doc.get("pareto_by_bytes") or []
    disagreement = doc.get("THE_COUNT_COLUMN_AND_THE_BYTES_COLUMN_DISAGREE") or {}
    artifact = doc.get("artifact") or {}
    headline = doc.get("headline") or {}
    rows = []
    for row in pareto:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "kernel": row.get("kernel"),
                "dispatches": row.get("dispatches"),
                "weight_bytes": row.get("weight_bytes"),
                "bytes_per_dispatch": row.get("bytes_per_dispatch"),
            }
        )
    return {
        "consumed": True,
        "path": BYTES_ATLAS_RECEIPT,
        "module_path": BYTES_ATLAS_MODULE,
        "module_role": (
            "measures per-dispatch weight bytes against the sealed catalog; "
            "does not solve HBM residency (frontier F005 duplication_check)"
        ),
        "catalog_total_bytes": artifact.get("catalog_total_bytes"),
        "active_weight_bytes_per_token": headline.get("active_weight_bytes_per_token"),
        "matvec_share_of_count": disagreement.get("matvec_share_of_count"),
        "matvec_share_of_bytes": disagreement.get("matvec_share_of_bytes"),
        "reading": disagreement.get("reading"),
        "pareto_by_bytes": rows,
        "claim_boundary": (
            "Byte counts and dimensionless shares only. Atlas TPS / GB/s "
            "figures are not copied: this sidecar has no GPU authority."
        ),
    }


def _fpga_priority(fpga_map: dict[str, Any] | None, organ_id: str) -> str | None:
    if not fpga_map:
        return None
    organs = fpga_map.get("organs") or []
    wanted = FPGA_PRIORITY_JOIN.get(organ_id)
    for row in organs:
        if not isinstance(row, dict):
            continue
        if row.get("organ") == wanted or row.get("organ") == organ_id:
            return row.get("priority")
    return None


def _state_lifetime_from_science(organ: dict[str, Any]) -> str | None:
    """Derive a lifetime class from stated science fields. None if unstated."""
    oid = organ.get("id")
    state = organ.get("state_bytes") or {}
    if oid == "recurrent_state":
        return "SEQUENCE_STATE"
    if oid in {"mtp", "vision_backbone"}:
        return "CONDITIONAL"
    status = state.get("status") if isinstance(state, dict) else None
    if status == "NO_PERSISTENT_STATE_IN_PRE_RUNTIME_MODEL":
        return "STATIC_WEIGHTS"
    if isinstance(state, dict) and _as_int(state.get("resident_bytes")):
        return "SEQUENCE_STATE"
    return None


def _access_probability_from_science(
    organ: dict[str, Any], science: dict[str, Any]
) -> tuple[float | None, str | None]:
    """Cite a fraction the receipt actually states. Do not use active/stored as p."""
    oid = organ.get("id")
    bounds = science.get("active_compute_bounds") or {}
    frac = _nested(bounds, "expert_activation_fraction", "routed")
    if oid == "routed_experts" and _as_float(frac) is not None:
        return float(frac), (
            "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.active_compute_bounds."
            "expert_activation_fraction.routed"
        )
    # Every other organ has no probability field. active/stored is intensity,
    # regularity strings are not probabilities, and neither is substituted.
    return None, None


def _resident_bytes(organ: dict[str, Any]) -> tuple[int | None, str | None]:
    oid = organ.get("id")
    if oid == "recurrent_state":
        resident = _as_int(_nested(organ, "state_bytes", "resident_bytes"))
        if resident:
            return resident, "organ_graph.recurrent_state.state_bytes.resident_bytes"
        return None, None
    stored = _as_int(_nested(organ, "stored_bytes", "value"))
    if stored is None:
        stored = _as_int(_nested(organ, "bytes", "value"))
    if stored is None:
        return None, None
    if stored == 0:
        return None, None
    return stored, "organ_graph.stored_bytes.value (pinned safetensors header payload)"


def flash_candidates_from_docs(
    science: dict[str, Any] | None,
    ebpw: dict[str, Any] | None,
    token_ns: dict[str, Any] | None,
    fpga_map: dict[str, Any] | None,
) -> list[ResidentCandidate]:
    """Build Flash candidates from the real organ inventory. Do not invent fields."""
    # token_ns is consulted: every actual_* field is null, so mac_latency stays None.
    if not science:
        return []
    eb_by_organ: dict[str, dict[str, Any]] = {}
    if ebpw:
        for row in ebpw.get("organs") or []:
            if isinstance(row, dict) and row.get("organ"):
                eb_by_organ[str(row["organ"])] = row
    out: list[ResidentCandidate] = []
    for organ in science.get("organ_graph") or []:
        if not isinstance(organ, dict) or not organ.get("id"):
            continue
        oid = str(organ["id"])
        nbytes, bytes_kind = _resident_bytes(organ)
        p, p_src = _access_probability_from_science(organ, science)
        eb_row = eb_by_organ.get(oid) or {}
        fmt = eb_row.get("chosen_representation")
        if not isinstance(fmt, str) or not fmt:
            fmt = None
        active = _as_int(_nested(organ, "active_bytes_per_token", "value"))
        notes: list[str] = []
        if oid == "routed_experts":
            notes.append(
                "whole-organ grain is 241 GB; FPGA P0 mapping is selected expert "
                "subsets. Census has no per-expert stored-byte vector, so shards "
                "are not invented."
            )
        if oid in {"embeddings", "ngram_engine"}:
            notes.append(
                "hot-row / lookup organ: access_probability of the whole blob is "
                "unstated; active/stored is intensity, not a probability, and is "
                "not substituted."
            )
        if oid == "mtp":
            notes.append(
                "active_bytes_per_token is CONDITIONAL on the draft/verify path; "
                "access_probability left UNKNOWN."
            )
        token_ns_row = None
        if token_ns:
            for row in token_ns.get("organs") or []:
                if isinstance(row, dict) and row.get("organ") == oid:
                    token_ns_row = row
                    break
        host_actual = (
            None if token_ns_row is None else token_ns_row.get("actual_gpu_ns_per_token")
        )
        # Host GPU ns is not FPGA MAC latency. Copying it would invent a MAC.
        mac_src = (
            "FLASH_TOKEN_NS_BUDGET has a host actual_gpu_ns_per_token; that is not "
            "FPGA MAC latency and is not copied into mac_latency_ns"
            if host_actual is not None
            else (
                "FLASH_TOKEN_NS_BUDGET.actual_gpu_ns_per_token is null; "
                "no FPGA MAC latency receipt exists"
            )
        )
        out.append(
            ResidentCandidate(
                id=f"flash.{oid}",
                bytes=nbytes,
                bytes_kind=bytes_kind,
                access_probability=p,
                access_probability_source=p_src,
                reuse_count=None,
                transport_cost_class=None,
                mac_latency_ns=None,
                mac_latency_ns_source=mac_src,
                projected_fpga_latency=None,
                state_lifetime=_state_lifetime_from_science(organ),
                dependency_criticality=None,
                representation_format=fmt,
                corpus="flash-next",
                source_active_bytes_per_token=active,
                fpga_map_priority=_fpga_priority(fpga_map, oid),
                notes=tuple(notes),
            )
        )
    return sorted(out, key=lambda c: c.id)


def noetic_size_counterexample(census: dict[str, Any] | None) -> dict[str, Any]:
    """Size order vs function-lost-per-byte order from the measured Qwen3.8 census."""
    if not census:
        return {"present": False, "path": NOETIC_CENSUS}
    ranking = (census.get("ranking") or {}).get("by_stored_byte") or []
    size_order = sorted(
        ranking,
        key=lambda r: (-int(r.get("bytes") or 0), r.get("organ") or ""),
    )
    precious = list(ranking)  # already by function_lost_per_stored_byte
    rows = []
    for row in ranking:
        rows.append(
            {
                "organ": row.get("organ"),
                "bytes": row.get("bytes"),
                "active_bytes_per_token": row.get("active_bytes_per_token"),
                "function_lost": row.get("function_lost"),
                "function_lost_per_stored_byte": row.get(
                    "function_lost_per_stored_byte"
                ),
                "rank_by_function_per_stored_byte": row.get(
                    "rank_by_function_per_stored_byte"
                ),
            }
        )
    mlp = next((r for r in rows if r["organ"] == "mlp"), None)
    embed = next((r for r in rows if r["organ"] == "embedding"), None)
    return {
        "present": True,
        "path": NOETIC_CENSUS,
        "model": "Qwen3.8-27B hybrid (NOT Flash). Cited as the measured organ census.",
        "ranking_definition": (census.get("ranking") or {}).get("definition"),
        "by_stored_byte": rows,
        "size_order": [r.get("organ") for r in size_order],
        "preciousness_order": [r.get("organ") for r in precious],
        "orders_disagree": [r.get("organ") for r in size_order]
        != [r.get("organ") for r in precious],
        "counterexample": {
            "largest_organ": mlp,
            "highest_preciousness_per_byte": embed,
            "reading": (
                "mlp is the largest organ and the lowest function-lost per stored "
                "byte; embedding is among the smallest and the highest. Filling "
                f"HBM by size would prefer mlp ({(mlp or {}).get('bytes')} bytes, "
                f"which already exceeds the 8 GiB budget of {DEFAULT_BUDGET_BYTES}). "
                "Size ranking is the wrong objective even before FPGA enters."
            ),
        },
    }


def flash_size_counterexample(
    candidates: list[ResidentCandidate], budget_bytes: int
) -> dict[str, Any]:
    """Size-greedy on the real Flash inventory, with intensity as the contrast."""
    with_bytes = [c for c in candidates if isinstance(c.bytes, int) and c.bytes > 0]
    greedy = size_ranked_select(with_bytes, budget_bytes)
    greedy_ids = [c.id for c in greedy]
    intensities = []
    for c in with_bytes:
        stored = c.bytes
        active = c.source_active_bytes_per_token
        intensity = None
        if stored and active is not None and stored > 0:
            intensity = active / stored
        intensities.append(
            {
                "id": c.id,
                "bytes": stored,
                "source_active_bytes_per_token": active,
                "active_over_stored": intensity,
                "fpga_map_priority": c.fpga_map_priority,
                "fits_whole_in_budget": bool(stored is not None and stored <= budget_bytes),
            }
        )
    intensities.sort(key=lambda r: r["id"])
    embed = next((c for c in with_bytes if c.id.endswith(".embeddings")), None)
    deltanet = next((c for c in with_bytes if c.id.endswith(".deltanet")), None)
    ngram = next((c for c in with_bytes if c.id.endswith(".ngram_engine")), None)
    experts = next((c for c in with_bytes if c.id.endswith(".routed_experts")), None)
    return {
        "budget_bytes": budget_bytes,
        "size_greedy_ids": greedy_ids,
        "size_greedy_bytes": sum(int(c.bytes or 0) for c in greedy),
        "too_large_to_reside_as_a_whole": [
            c.id for c in with_bytes if c.bytes and c.bytes > budget_bytes
        ],
        "intensities": intensities,
        "counterexamples": [
            {
                "id": (embed.id if embed else None),
                "bytes": embed.bytes if embed else None,
                "active_bytes_per_token": embed.source_active_bytes_per_token if embed else None,
                "point": (
                    "embeddings occupy ~1.18 GB of source payload for a 5120-byte "
                    "gathered row per token. Size packing after taking other large "
                    "fitters still treats this table as a serious HBM resident; "
                    "a bandwidth shard does not."
                ),
            },
            {
                "id": (deltanet.id if deltanet else None),
                "bytes": deltanet.bytes if deltanet else None,
                "active_bytes_per_token": deltanet.source_active_bytes_per_token if deltanet else None,
                "fpga_map_priority": deltanet.fpga_map_priority if deltanet else None,
                "point": (
                    "deltanet streams essentially its whole stored payload every "
                    "token and is FPGA-map P0 via persistent state. It loses a "
                    "size race to larger cold or conditional organs that happen "
                    "to fit."
                ),
            },
            {
                "id": (ngram.id if ngram else None),
                "bytes": ngram.bytes if ngram else None,
                "active_bytes_per_token": ngram.source_active_bytes_per_token if ngram else None,
                "point": (
                    "ngram_engine is ~95 GB stored and ~66 MB active per token. "
                    "Size ranking never considers it (does not fit as a whole) "
                    "and cannot consider a hot-row subset because the census "
                    "has no hot-row byte vector."
                ),
            },
            {
                "id": (experts.id if experts else None),
                "bytes": experts.bytes if experts else None,
                "active_bytes_per_token": experts.source_active_bytes_per_token if experts else None,
                "fpga_map_priority": experts.fpga_map_priority if experts else None,
                "point": (
                    "routed_experts is ~225 GB stored, 10/512 experts active per "
                    "token, FPGA-map P0 as 'HBM-resident selected expert subsets'. "
                    "The whole bank cannot live in 8 GiB. The census grain cannot "
                    "name the subset. Size ranking therefore either skips the "
                    "organ or would stuff the board with the wrong blob."
                ),
            },
        ],
    }


def _solve_json(result: SolveResult) -> dict[str, Any]:
    return {
        "budget_bytes": result.budget_bytes,
        "selected": [c.record() for c in result.selected],
        "rejected": [
            {"candidate": c.record(), "reason": reason} for c, reason in result.rejected
        ],
        "undecidable": [c.record() for c in result.undecidable],
        "selected_ids": result.selected_ids(),
        "selected_cited_weighted_latency": result.selected_value,
        "selected_bytes": result.selected_bytes,
        "counts": {
            "selected": len(result.selected),
            "rejected": len(result.rejected),
            "undecidable": len(result.undecidable),
            "decidable": len(result.selected) + len(result.rejected),
        },
    }


def build(budget_bytes: int = DEFAULT_BUDGET_BYTES) -> Path:
    recovered = recover_inputs()
    docs = recovered["docs"]
    flash = flash_candidates_from_docs(
        docs["flash_science"],
        docs["flash_ebpw"],
        docs["flash_token_ns"],
        docs["flash_fpga_map"],
    )
    result = solve(flash, budget_bytes)
    atlas = consume_bytes_atlas(docs["bytes_atlas"])
    noetic = noetic_size_counterexample(docs["noetic_census"])
    flash_anti = flash_size_counterexample(flash, budget_bytes)
    numeric_guard = refuse_size_ranking(flash, budget_bytes)

    consulted = recovered["consulted"]
    negative: list[str] = []
    if not consulted["flash_organ_census"]["present"]:
        negative.append(
            f"{CONTRACT_CENSUS} is absent from git HEAD. Closest Flash organ "
            f"inventory is {FLASH_SCIENCE} organ_graph + {FLASH_EBPW} organs."
        )
    if not consulted["qwen27_token_ns_budget"]["present"]:
        negative.append(
            f"{CONTRACT_QWEN27_BUDGET} is absent. Closest is {FLASH_TOKEN_NS} "
            "(every actual_* field is null) and the Qwen3.8 NOETIC census "
            "token_ns_ledger (wrong model, host GPU, not FPGA MAC)."
        )
    if not consulted["flash_layer30_critical_path"]["present"]:
        negative.append(f"{CONTRACT_LAYER30_CP} is absent from git HEAD.")
    if not consulted["flash_layer10_critical_path"]["present"]:
        negative.append(f"{CONTRACT_LAYER10_CP} is absent from git HEAD.")
    if docs["flash_token_ns"]:
        actuals = [
            row.get("actual_gpu_ns_per_token")
            for row in (docs["flash_token_ns"].get("organs") or [])
            if isinstance(row, dict)
        ]
        if actuals and all(v is None for v in actuals):
            negative.append(
                "FLASH_TOKEN_NS_BUDGET.organs.*.actual_gpu_ns_per_token is null "
                "for every organ (WAITING_FOR_NATIVE_EXECUTION). mac_latency_ns "
                "therefore cannot be filled from a receipt."
            )
    if docs["flash_fpga_map"]:
        cap = _nested(docs["flash_fpga_map"], "hbm_genome", "capacity_bytes")
        if cap is None:
            negative.append(
                "FLASH_NEXT_FPGA_ORGAN_MAP.hbm_genome.capacity_bytes is null "
                "(TARGET_UNSELECTED). 8 GiB is the workunit default, not a "
                "measured board capacity."
            )
    negative.append(
        "No receipt states reuse_count, transport_cost_class, "
        "projected_fpga_latency as LOW/MEDIUM/HIGH, or dependency_criticality "
        "for Flash organs. Those absences are not filled with defaults."
    )
    negative.append(
        "Census grain is whole-organ. FPGA P0 mapping is selected expert "
        "subsets and n-gram lookup rows. Per-expert / hot-row byte vectors "
        "are not in the census and are not invented."
    )
    if not consulted["bytes_atlas_module"]["on_disk"]:
        negative.append(
            f"{BYTES_ATLAS_MODULE} is in git but not on this sparse worktree; "
            "the sealed receipt was consumed instead of importing the module "
            "(the module also requires ~/noetic/NOETIC_PARENT_A)."
        )

    known_bytes = sum(int(c.bytes) for c in flash if isinstance(c.bytes, int) and c.bytes > 0)
    undecidable_bytes = sum(
        int(c.bytes) for c in result.undecidable if isinstance(c.bytes, int) and c.bytes > 0
    )
    headline = {
        "budget_bytes": budget_bytes,
        "candidate_count": len(flash),
        "decidable_count": len(result.selected) + len(result.rejected),
        "undecidable_count": len(result.undecidable),
        "undecidable_is_the_decision": len(result.undecidable) == len(flash) and len(flash) > 0,
        "known_resident_bytes": known_bytes,
        "undecidable_bytes": undecidable_bytes,
        "selected_count": len(result.selected),
        "selected_bytes": result.selected_bytes,
        "reading": (
            "The size of the undecidable bucket is the headline: every Flash "
            "organ is missing at least one input the objective requires "
            "(mac_latency_ns, a critical-path receipt, an FPGA latency CLASS, "
            "reuse_count, transport_cost_class). Reporting UNKNOWN is the "
            "correct residency answer until those receipts exist. A filled "
            "8 GiB set would be an invention."
        ),
    }

    frontier_lane = None
    if docs["frontier"]:
        for entry in docs["frontier"].get("entries") or []:
            if isinstance(entry, dict) and entry.get("id") == "F005":
                frontier_lane = {
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "classification": entry.get("classification"),
                    "integration_target": entry.get("integration_target"),
                    "duplication_check": entry.get("duplication_check"),
                    "prerequisite": entry.get("prerequisite"),
                }
                break

    recovered_implementation = [
        {
            "path": CONTRACT_CENSUS,
            "status": "ABSENT",
            "role": "contract-named Flash organ census",
        },
        {
            "path": FLASH_SCIENCE,
            "status": "PRESENT" if consulted["flash_science"]["present"] else "ABSENT",
            "role": "real Flash organ inventory: stored_bytes, active_bytes_per_token, state_bytes, expert fraction",
        },
        {
            "path": FLASH_EBPW,
            "status": "PRESENT" if consulted["flash_ebpw_budget"]["present"] else "ABSENT",
            "role": "source_bytes per organ + chosen_representation plan strings; actual_representation_bytes is null",
        },
        {
            "path": FLASH_TOKEN_NS,
            "status": "PRESENT" if consulted["flash_token_ns_budget"]["present"] else "ABSENT",
            "role": "per-organ token budget with every actual_* null",
        },
        {
            "path": FLASH_FPGA_MAP,
            "status": "PRESENT" if consulted["flash_fpga_organ_map"]["present"] else "ABSENT",
            "role": "P0/P1 HBM-residency hypotheses; hbm_genome.capacity_bytes is null",
        },
        {
            "path": NOETIC_CENSUS,
            "status": "PRESENT" if consulted["noetic_organ_census"]["present"] else "ABSENT",
            "role": "measured Qwen3.8 organ bytes + function-lost-per-byte ranking (size-ranking counterexample)",
        },
        {
            "path": BYTES_ATLAS_MODULE,
            "status": "PRESENT_IN_GIT" if consulted["bytes_atlas_module"]["in_git"] else "ABSENT",
            "role": "byte accounting; consumed via its sealed receipt, not forked",
        },
        {
            "path": BYTES_ATLAS_RECEIPT,
            "status": "PRESENT" if consulted["bytes_atlas_receipt"]["present"] else "ABSENT",
            "role": "sealed per-dispatch weight bytes; count column ≠ bytes column",
        },
        {
            "path": FRONTIER_RECEIPT,
            "status": "PRESENT" if consulted["global_frontier"]["present"] else "ABSENT",
            "role": "F005 No HBM Doctor; duplication_check: bytes_atlas measures bytes, does not solve residency",
        },
        {
            "path": "hcli/agentos/fpga_preboard.py",
            "status": "PRESENT_IN_GIT",
            "role": "HBMGenome.capacity_bytes Optional, default None; TARGET_UNSELECTED",
        },
        {
            "path": "tools/headless/noetic_organ_census.py",
            "status": "PRESENT_IN_GIT",
            "role": "producer of NOETIC_ORGAN_CENSUS; Qwen3.8, not Flash",
        },
        {
            "path": "tools/headless/doctor_diagnosis.py",
            "status": "PRESENT_IN_GIT",
            "role": "representation Doctor; not a hardware-axis / HBM residency solver",
        },
    ]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Decide what should live in 8 GiB of HBM by maximising cited "
            "complete-token critical-path reduction per resident byte. HBM is a "
            "bandwidth shard. FPGA is Accelerator / Physical Compiler / Fusion."
        ),
        "vocabulary": {
            "eras": ["I", "II", "III", "IV", "V"],
            "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?", "III WHERE IS HAWKING WRONG?"],
            "fpga_home": "Accelerator / Physical Compiler / Fusion",
            "not_an_fpga_backend": True,
            "evidence_classes": ["STATIC_ONLY", "DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
            "this_receipt": "STATIC_ONLY",
        },
        "objective": {
            "name": "marginal_complete_token_critical_path_reduction_per_resident_byte",
            "score_rule": SCORE_MEANING,
            "unknown_policy": (
                "Any missing required input => score UNKNOWN, bucket undecidable. "
                "Unknown is not zero. Unknown is not an estimate."
            ),
            "knapsack": (
                "exact 0/1 over decidable items, maximising cited_weighted_latency "
                "subject to sum(bytes) <= budget_bytes"
            ),
            "budget_bytes_default": DEFAULT_BUDGET_BYTES,
            "budget_bytes": budget_bytes,
            "budget_source": (
                "workunit default 8 GiB (8 * 1024**3). FPGA hbm_genome.capacity_bytes is null."
            ),
        },
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "frontier_lane": frontier_lane,
        "recovered_implementation": recovered_implementation,
        "gaps_closed": [
            "ResidentCandidate record with the nine contract fields, None-preserving.",
            "Score is UNKNOWN when any required input is missing; never 0-filled.",
            "Exact 0/1 knapsack over decidable items against a configurable 8 GiB budget.",
            "Undecidable items are a headline output and cannot enter the selected set.",
            "Anti-pattern guard: size-ranking vs criticality-per-byte, with Flash and NOETIC census counterexamples.",
            "Flash candidates joined from science organ_graph + EBPW + FPGA map without inventing latency.",
            "bytes_atlas consumed as byte ranking, not forked and not re-emitted as a TPS claim.",
        ],
        "negative_findings": negative,
        "inputs_consulted": consulted,
        "bytes_atlas_consumed": atlas,
        "candidates": [c.record() for c in flash],
        "headline": headline,
        "solution": _solve_json(result),
        "anti_pattern": {
            "refused_policy": "fill_HBM_with_the_largest_items",
            "why": (
                "Occupancy is not the objective. The board's only real advantage "
                "is HBM bandwidth on the complete-token critical path. Size "
                "ranking selects cold tables that fit and skips hot sparse "
                "structures that do not fit as wholes."
            ),
            "numeric_guard_on_flash_decidable_subset": numeric_guard,
            "flash_census": flash_anti,
            "noetic_census": noetic,
        },
        "field_absences_in_flash_census": {
            "bytes": "present as stored_bytes / resident_bytes on the science organ_graph",
            "access_probability": (
                "stated only for routed_experts (expert_activation_fraction.routed "
                "= 10/512). UNKNOWN for every other organ; active/stored is not "
                "substituted as a probability"
            ),
            "reuse_count": "ABSENT",
            "transport_cost_class": "ABSENT (no board, no PCIe measurement)",
            "mac_latency_ns": "ABSENT (FLASH_TOKEN_NS_BUDGET actuals are null; no FPGA MAC receipt)",
            "projected_fpga_latency": "ABSENT as a class; FPGA map has P0/P1 priority, which is not a latency",
            "state_lifetime": "DERIVED from science state_bytes.status where that field exists",
            "dependency_criticality": "ABSENT (layer critical-path receipts are missing)",
            "representation_format": "plan string from FLASH_EBPW; actual_representation_bytes is null",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/hbm_doctor.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--budget-bytes",
        type=int,
        default=DEFAULT_BUDGET_BYTES,
        help="HBM residency budget in bytes (default 8 GiB)",
    )
    a = ap.parse_args()
    out = build(budget_bytes=a.budget_bytes)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
