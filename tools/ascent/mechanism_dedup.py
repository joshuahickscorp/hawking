"""Mechanism-level launch gate and dry-queue synthesis.

Before a lane launches, retrieve related negative science, previous attempts,
and currently-running mechanisms, then refuse SEMANTIC duplicates. A bottleneck
may be retried; a mechanism may not, unless a precondition, implementation,
representation or the evidence changed, or the previous test was invalid.

When no admissible pending target exists and a measured bottleneck remains,
synthesize a NEW mechanism. Nine failed mechanisms against weight_addressing
means invent mechanism ten — not "bottleneck complete", and not re-run
mechanism one.

This module is the authority. The unattended loop should call ``admit`` before
launch and ``decide_next`` when the pending queue is empty or full of
inadmissible retries. It does not touch promotion, lineage, evictors, codecs
or kernels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .mechanism_identity import (
        SCHEMA as IDENTITY_SCHEMA,
        fingerprint,
        is_bottleneck_name,
        same_mechanism,
    )
except ImportError:  # python3 tools/ascent/mechanism_dedup.py
    from mechanism_identity import (  # type: ignore
        SCHEMA as IDENTITY_SCHEMA,
        fingerprint,
        is_bottleneck_name,
        same_mechanism,
    )

SCHEMA = "hawking.ascent.mechanism_dedup.v1"
DATE = "2026-08-16"

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCIENCE = REPO / "receipts" / "ascent-2026-08-16" / "NEGATIVE_SCIENCE_REGISTER.json"
DEFAULT_LEDGER = REPO / "receipts" / "ascent-2026-08-16" / "TOKEN_NS_QWEN38.json"
DEFAULT_RECEIPT = REPO / "receipts" / "ascent-2026-08-16" / "MECHANISM_DEDUP_EVIDENCE.json"

ACTIVE_MODELS = frozenset({"qwen38"})
SEALED_MODELS = frozenset({"q80", "dsv4f"})
TERMINAL_SCIENCE = frozenset(
    {"REFUTED", "DEAUTHORISED", "UNREACHABLE", "CATEGORY_ERROR"}
)
# A component below this is noise, not a remaining bottleneck worth a lane.
REMAINING_NS_FLOOR = 100_000
ACTIVE_STATUSES = frozenset({"pending", "running"})
# Roof axes that together form a precondition. Implementation and representation
# are reported separately when they are the thing that changed.
PRECONDITION_AXES = (
    "genome",
    "launch_geometry",
    "command_topology",
    "synchronization",
    "residency",
    "addressing",
    "host_gpu_partition",
    "kv_strategy",
    "artifact",
)

INVALID_MARKERS = (
    "gaussian",
    "synthetic activation",
    "synthetic activations",
    "degraded model",
    "undersampled",
    "underdetermined",
    "cpu-wait proxy",
    "cpu wait proxy",
    "projection as measurement",
    "cosine as capability",
    "organ cosine",
)

GENESIS_VEHICLE = {
    "model": "qwen38",
    "representation": "uniform-q4-group64",
    "implementation": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "genome": "1 CB / 964 dispatches; qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "bytes_per_token": 13_618_000_000.0,
    "launch_geometry": "tpr64_tg128",
    "command_topology": "1cb_964_dispatches",
    "artifact": "uniform-q4-v1",
}


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"
    SYNTHESIZE = "SYNTHESIZE"
    LAUNCH = "LAUNCH"
    HOLD = "HOLD"
    ERROR = "ERROR"


# ---------------------------------------------------------------- records


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ScienceEntry:
    id: str
    mechanism: str
    science_class: str
    models: tuple[str, ...] = ()
    retry_when: str = ""
    why_it_failed: str = ""
    settled_by: Any = None

    def terminal(self) -> bool:
        return self.science_class.upper() in TERMINAL_SCIENCE

    def applies_to(self, model: str | None) -> bool:
        if not model or not self.models:
            return True
        return model in self.models or model.lower() in {m.lower() for m in self.models}


@dataclass
class Attempt:
    """One prior or in-flight mechanism test."""

    id: str
    mechanism: str
    bottleneck: str = ""
    model: str = "qwen38"
    status: str = "failed"
    test_valid: bool = True
    invalid_reason: str | None = None
    representation: str | None = None
    implementation: str | None = None
    evidence_digest: str | None = None
    bytes_per_token: float | None = None
    genome: str | None = None
    launch_geometry: str | None = None
    command_topology: str | None = None
    synchronization: str | None = None
    residency: str | None = None
    addressing: str | None = None
    host_gpu_partition: str | None = None
    kv_strategy: str | None = None
    artifact: str | None = None
    outcome: str | None = None
    science_id: str | None = None

    def is_running(self) -> bool:
        return (self.status or "").lower() in ACTIVE_STATUSES

    def is_hollow(self) -> bool:
        return not (self.mechanism or "").strip()


@dataclass
class Proposal:
    """What a lane wants to do, before launch."""

    mechanism: str
    bottleneck: str
    model: str = "qwen38"
    id: str | None = None
    representation: str | None = None
    implementation: str | None = None
    evidence_digest: str | None = None
    bytes_per_token: float | None = None
    genome: str | None = None
    launch_geometry: str | None = None
    command_topology: str | None = None
    synchronization: str | None = None
    residency: str | None = None
    addressing: str | None = None
    host_gpu_partition: str | None = None
    kv_strategy: str | None = None
    artifact: str | None = None
    reopen_claim: str | None = None
    test_valid: bool = True

    def is_hollow(self) -> bool:
        return not (self.mechanism or "").strip()


@dataclass
class ScienceHit:
    entry: ScienceEntry
    score: float
    reason: str


@dataclass
class Retrieved:
    science: list[ScienceHit] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    running: list[Attempt] = field(default_factory=list)


@dataclass
class Synthesized:
    mechanism_id: str
    mechanism: str
    bottleneck: str
    axis: str
    assumption_attacked: str
    experiment: str
    model: str = "qwen38"
    target: dict[str, Any] = field(default_factory=dict)

    def as_proposal(self) -> Proposal:
        return Proposal(
            mechanism=self.mechanism,
            bottleneck=self.bottleneck,
            model=self.model,
            id=self.target.get("id"),
            representation=self.target.get("representation"),
            implementation=self.target.get("implementation"),
            artifact=self.target.get("artifact"),
        )


@dataclass
class Decision:
    verdict: Verdict
    reason: str
    gate: str
    matched_prior: Attempt | None = None
    matched_science: ScienceEntry | None = None
    match_score: float | None = None
    changed: list[str] = field(default_factory=list)
    retrieved: Retrieved | None = None
    synthesized: Synthesized | None = None
    proposal: Proposal | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def red(self) -> bool:
        return self.verdict in {Verdict.REFUSE, Verdict.ERROR}

    def to_dict(self) -> dict[str, Any]:
        def _att(a: Attempt | None) -> dict[str, Any] | None:
            return asdict(a) if a is not None else None

        def _sci(s: ScienceEntry | None) -> dict[str, Any] | None:
            return asdict(s) if s is not None else None

        retrieved = None
        if self.retrieved is not None:
            retrieved = {
                "science": [
                    {
                        "id": h.entry.id,
                        "class": h.entry.science_class,
                        "mechanism": h.entry.mechanism,
                        "score": h.score,
                        "reason": h.reason,
                    }
                    for h in self.retrieved.science
                ],
                "attempts": [a.id for a in self.retrieved.attempts],
                "running": [a.id for a in self.retrieved.running],
            }
        syn = None
        if self.synthesized is not None:
            syn = {
                "mechanism_id": self.synthesized.mechanism_id,
                "mechanism": self.synthesized.mechanism,
                "bottleneck": self.synthesized.bottleneck,
                "axis": self.synthesized.axis,
                "assumption_attacked": self.synthesized.assumption_attacked,
                "experiment": self.synthesized.experiment,
                "target_id": self.synthesized.target.get("id"),
            }
        return {
            "schema": SCHEMA,
            "verdict": self.verdict.value,
            "red": self.red,
            "gate": self.gate,
            "reason": self.reason,
            "changed": list(self.changed),
            "match_score": self.match_score,
            "matched_prior": _att(self.matched_prior),
            "matched_science": _sci(self.matched_science),
            "retrieved": retrieved,
            "synthesized": syn,
            "details": self.details,
        }


# ---------------------------------------------------------------- campaign catalog
# The nine MEASURED AND REFUTED mechanisms on weight_addressing / this campaign.
# Do not rediscover. Synthesis starts at ten.

def _v(**overrides: Any) -> dict[str, Any]:
    row = dict(GENESIS_VEHICLE)
    row.update(overrides)
    return row


def _attempt(i: int, mechanism: str, **kw: Any) -> Attempt:
    veh = _v()
    data = dict(
        id=f"wa-exhausted-{i:02d}",
        mechanism=mechanism,
        bottleneck="weight_addressing",
        model="qwen38",
        status="retained_negative",
        test_valid=True,
        representation=veh["representation"],
        implementation=veh["implementation"],
        genome=veh["genome"],
        bytes_per_token=veh["bytes_per_token"],
        launch_geometry=veh["launch_geometry"],
        command_topology=veh["command_topology"],
        artifact=veh["artifact"],
    )
    data.update(kw)
    return Attempt(**data)


CAMPAIGN_EXHAUSTED: tuple[Attempt, ...] = (
    _attempt(
        1,
        "fusing tiny kernels into the following GEMV",
        outcome="REGRESSION +10,679,189 ns",
    ),
    _attempt(
        2,
        "cross-token cache reuse",
        outcome="hot/cold gap only 2.5% on a 47 MiB tensor",
    ),
    _attempt(
        3,
        "N sessions sharing one resident weight body to amortize DRAM",
        outcome="4 GEMVs still cost 4.09x serial / 3.97x concurrent",
    ),
    _attempt(
        4,
        "separate processes share artifact pages",
        outcome="8.77 GB each against an 8.5 GB artifact",
    ),
    _attempt(
        5,
        "every codec tried on out_proj: absmax-per-64, median and p90, outlier islands, structured sparsity, per-head assignment",
        outcome="intra-group max/median ~4.05; Q4 holds; out_proj is the floor",
    ),
    _attempt(
        6,
        "lm_head drop-LSB",
        outcome="0 greedy flips; still loads every byte; sensitivity not a byte win",
    ),
    _attempt(
        7,
        "codec reconstruction cost on real activations",
        outcome="FREE at tpr64; penalty was launch geometry",
    ),
    _attempt(
        8,
        "evaluate or fit compression on gaussian / synthetic proxy activations",
        outcome="ARTIFACT; real captured activations only",
        test_valid=False,
        invalid_reason="gaussian/synthetic activations",
        status="invalid",
    ),
    _attempt(
        9,
        "fits taken from a degraded model",
        outcome="ranks the WRONG trajectory",
        test_valid=False,
        invalid_reason="degraded model",
        status="invalid",
    ),
)


@dataclass(frozen=True)
class CatalogItem:
    mechanism_id: str
    mechanism: str
    bottleneck: str
    axis: str
    assumption_attacked: str
    experiment: str


# Next mechanisms, in the order they should be invented. Item 10 is the first
# unused attack on the assumption that generates the 13.618 GB unique-once
# quantity: one codec applied uniformly to every layer. Per-layer / per-head
# assignment is UNEXPLORED on the coherent 4.2527 vehicle.
SYNTHESIS_CATALOG: tuple[CatalogItem, ...] = (
    CatalogItem(
        mechanism_id="per_layer_per_head_assignment",
        mechanism=(
            "Assign codecs per layer and per head on the coherent 4.2527 vehicle, "
            "instead of one uniform codec. Uniform Q4 is the genome-chosen quantity; "
            "per-layer assignment is UNEXPLORED."
        ),
        bottleneck="weight_addressing",
        axis="representation",
        assumption_attacked=(
            "one codec applied uniformly to every layer, which is how 4.2527 BPW "
            "and 13.618 GB/token are generated"
        ),
        experiment=(
            "On the coherent 4.2527 vehicle, assign a different codec to a named "
            "subset of layers/heads (not another uniform out_proj trial). Measure "
            "unique-once bytes and complete-token wall with >=3 alternating paired "
            "reps; capability is greedy ids unchanged on >=3 prompts."
        ),
    ),
    CatalogItem(
        mechanism_id="activation_gated_weight_skip",
        mechanism=(
            "Do not read every weight every token: gate weight traffic on this "
            "token's activations so unread rows/heads are not addressed."
        ),
        bottleneck="weight_addressing",
        axis="bytes",
        assumption_attacked="unique-bytes-once means every packed weight must be read every token",
        experiment=(
            "Measure, on real captured activations from the coherent vehicle, "
            "what fraction of addressed bytes are numerically unused this token. "
            "If that fraction is ~0, the skip is a negative. Do not use Gaussian proxies."
        ),
    ),
    CatalogItem(
        mechanism_id="sub1_bpw_attention_embed_norms",
        mechanism=(
            "Cut attention + embed + norms below 1 BPW on the coherent vehicle. "
            "Those organs are 74% of the artifact; out_proj Q3 is a different, "
            "already-refuted trial."
        ),
        bottleneck="weight_addressing",
        axis="representation",
        assumption_attacked="attention+embed+norms must stay at ~4.25 BPW",
        experiment=(
            "Pack only those organs below 1 BPW, leave out_proj at the Q4 floor, "
            "measure unique-once bytes and the complete token. Capability gate "
            "is greedy ids, not cosine."
        ),
    ),
    CatalogItem(
        mechanism_id="addressing_layout_not_codec",
        mechanism=(
            "Change addressing layout (blocked / organ-contiguous / Morton) "
            "without changing the codec. The bucket is named addressing."
        ),
        bottleneck="weight_addressing",
        axis="addressing",
        assumption_attacked="the current address generation / layout is free",
        experiment=(
            "Same representation and bytes, different layout. GPU timing is a "
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait."
        ),
    ),
    CatalogItem(
        mechanism_id="host_gpu_partition_of_addressing",
        mechanism=(
            "Move address generation / bind work across the host-GPU cut, "
            "without fusing kernels into the following GEMV."
        ),
        bottleneck="weight_addressing",
        axis="host_gpu_partition",
        assumption_attacked="the current host/GPU partition of addressing is given",
        experiment=(
            "Attribute host bind vs GPU address to the closed ledger. A host win "
            "that does not move the complete token is a negative."
        ),
    ),
    CatalogItem(
        mechanism_id="fuse_deltanet_activation_tails",
        mechanism=(
            "Fuse DeltaNet activation tails (gated_rmsnorm, ba_to_decay, rearrange) "
            "into the vi kernel or one encoder. Not GEMV fusion."
        ),
        bottleneck="deltanet",
        axis="fusion",
        assumption_attacked="each DeltaNet tail is its own launch",
        experiment=(
            "Paired complete-token measurement. Isolated tail win that does not "
            "move the token is a negative. Do not replay GEMV microkernel fusion."
        ),
    ),
    CatalogItem(
        mechanism_id="collapse_rmsnorm_launches",
        mechanism=(
            "Collapse 129 RMSNorm launches (input + post + final) by folding them "
            "into the following work or sharing one persistent encoder. Not GEMV fusion."
        ),
        bottleneck="normalization",
        axis="command_topology",
        assumption_attacked="one encoder-per-tiny-kernel for every RMSNorm",
        experiment=(
            "Paired complete-token measurement on the coherent vehicle. Report "
            "the token, not the isolated norm."
        ),
    ),
)


# ---------------------------------------------------------------- constructors


def attempt_from_mapping(raw: Mapping[str, Any]) -> Attempt:
    allowed = {f.name for f in fields(Attempt)}
    data = {k: raw[k] for k in allowed if k in raw}
    if "id" not in data:
        data["id"] = str(raw.get("lane_id") or raw.get("target_id") or "")
    if "mechanism" not in data:
        data["mechanism"] = str(
            raw.get("hypothesis") or raw.get("title") or raw.get("from_bottleneck") or ""
        )
    if "bottleneck" not in data:
        data["bottleneck"] = str(raw.get("from_bottleneck") or raw.get("target_stage") or "")
    if "science_class" in raw and "outcome" not in data:
        data["outcome"] = str(raw.get("science_class"))
    return Attempt(
        id=str(data.get("id") or ""),
        mechanism=str(data.get("mechanism") or ""),
        bottleneck=str(data.get("bottleneck") or ""),
        model=str(data.get("model") or "qwen38"),
        status=str(data.get("status") or "failed"),
        test_valid=bool(data.get("test_valid", True)),
        invalid_reason=_opt_str(data.get("invalid_reason")),
        representation=_opt_str(data.get("representation")),
        implementation=_opt_str(data.get("implementation")),
        evidence_digest=_opt_str(data.get("evidence_digest")),
        bytes_per_token=_opt_float(data.get("bytes_per_token")),
        genome=_opt_str(data.get("genome")),
        launch_geometry=_opt_str(data.get("launch_geometry")),
        command_topology=_opt_str(data.get("command_topology")),
        synchronization=_opt_str(data.get("synchronization")),
        residency=_opt_str(data.get("residency")),
        addressing=_opt_str(data.get("addressing")),
        host_gpu_partition=_opt_str(data.get("host_gpu_partition")),
        kv_strategy=_opt_str(data.get("kv_strategy")),
        artifact=_opt_str(data.get("artifact")),
        outcome=_opt_str(data.get("outcome")),
        science_id=_opt_str(data.get("science_id")),
    )


def proposal_from_mapping(raw: Mapping[str, Any]) -> Proposal:
    mechanism = raw.get("mechanism")
    if mechanism is None:
        mechanism = raw.get("proposed_mechanism") or raw.get("hypothesis") or ""
    bottleneck = raw.get("bottleneck") or raw.get("from_bottleneck") or ""
    return Proposal(
        mechanism=str(mechanism or ""),
        bottleneck=str(bottleneck or ""),
        model=str(raw.get("model") or "qwen38"),
        id=_opt_str(raw.get("id")),
        representation=_opt_str(raw.get("representation")),
        implementation=_opt_str(raw.get("implementation")),
        evidence_digest=_opt_str(raw.get("evidence_digest")),
        bytes_per_token=_opt_float(raw.get("bytes_per_token")),
        genome=_opt_str(raw.get("genome")),
        launch_geometry=_opt_str(raw.get("launch_geometry")),
        command_topology=_opt_str(raw.get("command_topology")),
        synchronization=_opt_str(raw.get("synchronization")),
        residency=_opt_str(raw.get("residency")),
        addressing=_opt_str(raw.get("addressing")),
        host_gpu_partition=_opt_str(raw.get("host_gpu_partition")),
        kv_strategy=_opt_str(raw.get("kv_strategy")),
        artifact=_opt_str(raw.get("artifact")),
        reopen_claim=_opt_str(raw.get("reopen_claim")),
        test_valid=bool(raw.get("test_valid", True)),
    )


def science_from_mapping(raw: Mapping[str, Any]) -> ScienceEntry:
    sid = raw.get("id")
    mechanism = raw.get("mechanism")
    klass = raw.get("class") if "class" in raw else raw.get("science_class")
    if sid is None or mechanism is None or klass is None:
        raise ValueError("science entry missing id, mechanism or class")
    models = raw.get("models") or ()
    if isinstance(models, str):
        models = (models,)
    return ScienceEntry(
        id=str(sid),
        mechanism=str(mechanism),
        science_class=str(klass).upper(),
        models=tuple(str(m) for m in models),
        retry_when=str(raw.get("retry_when") or ""),
        why_it_failed=str(raw.get("why_it_failed") or ""),
        settled_by=raw.get("settled_by"),
    )


def coerce_attempt(item: Attempt | Mapping[str, Any]) -> Attempt:
    return item if isinstance(item, Attempt) else attempt_from_mapping(item)


def coerce_proposal(item: Proposal | Mapping[str, Any]) -> Proposal:
    return item if isinstance(item, Proposal) else proposal_from_mapping(item)


def coerce_science(item: ScienceEntry | Mapping[str, Any]) -> ScienceEntry:
    return item if isinstance(item, ScienceEntry) else science_from_mapping(item)


# ---------------------------------------------------------------- loaders


def load_science_register(path: Path | str | None = None) -> list[ScienceEntry]:
    p = Path(path) if path is not None else DEFAULT_SCIENCE
    if not p.is_file():
        raise FileNotFoundError(f"negative science register missing: {p}")
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError(f"science register is not a {{entries: [...]}} object: {p}")
    entries = raw["entries"]
    if not isinstance(entries, list):
        raise ValueError("science register entries is not a list")
    out: list[ScienceEntry] = []
    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            raise ValueError(f"science entry {i} is not an object")
        out.append(science_from_mapping(row))
    return out


def load_measured_bottlenecks(source: Any = None) -> list[dict[str, Any]]:
    """Accept a ledger path, a parsed ledger, or a list of {component, ns}."""
    if source is None:
        source = DEFAULT_LEDGER
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"token ledger missing: {p}")
        source = json.loads(p.read_text())
    if isinstance(source, list):
        rows = []
        for row in source:
            if not isinstance(row, dict):
                raise ValueError("bottleneck list entries must be objects")
            name = row.get("component") or row.get("name") or row.get("bottleneck")
            ns = row.get("ns_per_token", row.get("ns", row.get("ns_per_tok")))
            if name is None or ns is None:
                raise ValueError("bottleneck row missing component or ns")
            rows.append(
                {
                    "component": str(name),
                    "ns_per_token": float(ns),
                    "pct": row.get("pct") or row.get("pct_of_token_wall"),
                    "triage": row.get("triage"),
                }
            )
        return rows
    if not isinstance(source, dict):
        raise ValueError("ledger must be a dict, list, or path")
    if "ranked_by_ns" in source and isinstance(source["ranked_by_ns"], list):
        rows = []
        for row in source["ranked_by_ns"]:
            rows.append(
                {
                    "component": str(row["component"]),
                    "ns_per_token": float(row.get("ns") or row.get("ns_per_token")),
                    "pct": row.get("pct"),
                    "triage": row.get("triage"),
                }
            )
        return rows
    comps = source.get("components")
    if not isinstance(comps, list):
        raise ValueError("ledger has no components or ranked_by_ns")
    rows = []
    for row in comps:
        if not isinstance(row, dict):
            raise ValueError("component row is not an object")
        name = row.get("component")
        ns = row.get("ns_per_token", row.get("ns"))
        if name is None or ns is None:
            raise ValueError("component missing name or ns")
        rows.append(
            {
                "component": str(name),
                "ns_per_token": float(ns),
                "pct": row.get("pct_of_token_wall") or row.get("pct"),
                "triage": row.get("triage"),
            }
        )
    return rows


def remaining_bottlenecks(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ns = float(row["ns_per_token"])
        triage = str(row.get("triage") or "").lower()
        if ns < REMAINING_NS_FLOOR:
            continue
        if triage == "noise":
            continue
        if row.get("solved"):
            continue
        out.append(row)
    out.sort(key=lambda r: r["ns_per_token"], reverse=True)
    return out


# ---------------------------------------------------------------- retrieve


def _science_hits(
    proposal: Proposal, entries: Iterable[ScienceEntry]
) -> list[ScienceHit]:
    hits: list[ScienceHit] = []
    for entry in entries:
        if not entry.applies_to(proposal.model):
            continue
        match = same_mechanism(proposal.mechanism, entry.mechanism)
        if match.same:
            hits.append(ScienceHit(entry, match.score, match.reason))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def retrieve(
    proposal: Proposal | Mapping[str, Any],
    *,
    science: Iterable[ScienceEntry | Mapping[str, Any]] | None = None,
    attempts: Iterable[Attempt | Mapping[str, Any]] | None = None,
    running: Iterable[Attempt | Mapping[str, Any]] | None = None,
) -> Retrieved:
    """Related negative science, previous attempts, currently-running mechanisms."""
    prop = coerce_proposal(proposal)
    sci_rows = [coerce_science(s) for s in (science or ())]
    att_rows = [coerce_attempt(a) for a in (attempts or ())]
    run_rows = [coerce_attempt(a) for a in (running or ())]
    # Running records also count as attempts for relatedness.
    related_attempts = []
    for att in att_rows:
        if att.is_hollow():
            continue
        same_bn = _same_bottleneck(prop.bottleneck, att.bottleneck)
        same_mech = same_mechanism(prop.mechanism, att.mechanism).same
        if same_bn or same_mech:
            related_attempts.append(att)
    related_running = []
    for att in run_rows:
        if att.is_hollow():
            continue
        if att.is_running() and (
            _same_bottleneck(prop.bottleneck, att.bottleneck)
            or same_mechanism(prop.mechanism, att.mechanism).same
        ):
            related_running.append(att)
    return Retrieved(
        science=_science_hits(prop, sci_rows),
        attempts=related_attempts,
        running=related_running,
    )


def _same_bottleneck(a: str, b: str) -> bool:
    if not a or not b:
        return False
    na = fingerprint(a)
    nb = fingerprint(b)
    if na.tokens and na.tokens == nb.tokens:
        return True
    # "weight_addressing 21293103 ns" vs "weight_addressing"
    if {"weight", "address"} <= na.tokens and {"weight", "address"} <= nb.tokens:
        return True
    return same_mechanism(a, b).same and is_bottleneck_name(a) and is_bottleneck_name(b)


# ---------------------------------------------------------------- deltas / invalid


def _close_float(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale < 1e-3


def _eq_opt(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return str(a).strip() == str(b).strip()


def changed_axes(proposal: Proposal, prior: Attempt) -> list[str]:
    """Named, explicitly supplied deltas that legally reopen a mechanism.

    Omission is not change. Treating ``None`` as different from the prior value
    lets a duplicate retry erase its metadata and talk its way past the gate.
    """
    changed: list[str] = []
    if proposal.representation is not None and not _eq_opt(
        proposal.representation, prior.representation
    ):
        changed.append("representation")
    if proposal.implementation is not None and not _eq_opt(
        proposal.implementation, prior.implementation
    ):
        changed.append("implementation")
    if proposal.evidence_digest is not None and not _eq_opt(
        proposal.evidence_digest, prior.evidence_digest
    ):
        changed.append("evidence")
    if proposal.bytes_per_token is not None and not _close_float(
        proposal.bytes_per_token, prior.bytes_per_token
    ):
        if "representation" not in changed:
            changed.append("precondition")
        # bytes moved with representation still counts as representation-primary;
        # a bytes-only change without a representation change is a precondition.
        elif "precondition" not in changed:
            pass
    pre_fields = (
        "genome",
        "launch_geometry",
        "command_topology",
        "synchronization",
        "residency",
        "addressing",
        "host_gpu_partition",
        "kv_strategy",
        "artifact",
    )
    for name in pre_fields:
        proposed = getattr(proposal, name)
        if proposed is not None and not _eq_opt(proposed, getattr(prior, name)):
            if "precondition" not in changed:
                changed.append("precondition")
            break
    return changed


def prior_test_invalid(prior: Attempt) -> str | None:
    if prior.test_valid is False:
        return prior.invalid_reason or "test_valid=false"
    blob = " ".join(
        str(x or "") for x in (prior.invalid_reason, prior.outcome, prior.status)
    ).lower()
    for marker in INVALID_MARKERS:
        if marker in blob:
            return marker
    return None


def science_reopen_allowed(entry: ScienceEntry, proposal: Proposal) -> tuple[bool, str]:
    text = (entry.retry_when or "").strip().lower()
    if not text:
        return False, f"{entry.id} has no retry_when; terminal science stays closed"
    if text.startswith("never"):
        # "never … Reopen only if X" still requires a structured claim we can check.
        # A free-text reopen_claim is not enough: that is how a retry talks its way
        # past a settled negative.
        return False, f"{entry.id} retry_when starts with never"
    if not proposal.reopen_claim:
        return False, f"{entry.id} retry_when is conditional and no reopen_claim was supplied"
    return True, f"{entry.id} retry_when does not start with never"


# ---------------------------------------------------------------- admit


def _error(gate: str, reason: str, **kw: Any) -> Decision:
    return Decision(Verdict.ERROR, reason, gate, **kw)


def _refuse(gate: str, reason: str, **kw: Any) -> Decision:
    return Decision(Verdict.REFUSE, reason, gate, **kw)


def _allow(gate: str, reason: str, **kw: Any) -> Decision:
    return Decision(Verdict.ALLOW, reason, gate, **kw)


def admit(
    proposal: Proposal | Mapping[str, Any],
    *,
    science: Iterable[ScienceEntry | Mapping[str, Any]] | None = None,
    attempts: Iterable[Attempt | Mapping[str, Any]] | None = None,
    running: Iterable[Attempt | Mapping[str, Any]] | None = None,
    include_campaign_exhausted: bool = False,
) -> Decision:
    """Refuse a semantic duplicate; allow a new mechanism or a listed delta.

    Fail closed on hollow or malformed input. Never treat a bottleneck name as
    a mechanism.
    """
    try:
        prop = coerce_proposal(proposal)
    except Exception as exc:  # noqa: BLE001 — fail closed
        return _error("named_mechanism_required", f"proposal unreadable: {exc}")

    if prop.is_hollow():
        return _error(
            "named_mechanism_required",
            "proposal has no mechanism; a bottleneck is not a mechanism",
            proposal=prop,
        )
    if not (prop.bottleneck or "").strip():
        return _error(
            "named_mechanism_required",
            "proposal has no bottleneck; refuse to launch an untargeted mechanism",
            proposal=prop,
        )
    if is_bottleneck_name(prop.mechanism):
        return _refuse(
            "named_mechanism_required",
            "mechanism text is only a bottleneck name; name the action, not the cost",
            proposal=prop,
            details={"mechanism": prop.mechanism},
        )
    model = (prop.model or "").lower()
    if model in SEALED_MODELS:
        return _refuse(
            "sealed_model",
            f"{prop.model} is sealed and its weights are deleted; exclude at launch",
            proposal=prop,
        )
    if model and model not in ACTIVE_MODELS:
        return _refuse(
            "sealed_model",
            f"{prop.model} is not an active model",
            proposal=prop,
        )

    att_list = [coerce_attempt(a) for a in (attempts or ())]
    if include_campaign_exhausted:
        att_list = list(CAMPAIGN_EXHAUSTED) + att_list
    run_list = [coerce_attempt(a) for a in (running or ())]
    # A running record that claims to be in flight but cannot name its mechanism
    # is corrupt: fail closed rather than launch beside an unknown in-flight test.
    for att in run_list:
        if att.is_running() and att.is_hollow():
            return _error(
                "inflight_duplicate",
                f"running record {att.id!r} has no mechanism; cannot tell if this is a duplicate",
                proposal=prop,
            )

    try:
        sci_list = [coerce_science(s) for s in (science or ())]
    except Exception as exc:  # noqa: BLE001
        return _error("negative_science", f"science register unreadable: {exc}", proposal=prop)

    for entry in sci_list:
        klass = (entry.science_class or "").upper()
        if klass not in TERMINAL_SCIENCE | {"INSUFFICIENT", "OPEN", "HYPOTHESIS"}:
            return _error(
                "negative_science",
                f"science {entry.id} has unknown class {entry.science_class!r}",
                proposal=prop,
                matched_science=entry,
            )

    retrieved = retrieve(prop, science=sci_list, attempts=att_list, running=run_list)

    # 1. In-flight semantic duplicate.
    for att in run_list:
        if not att.is_running() or att.is_hollow():
            continue
        match = same_mechanism(prop.mechanism, att.mechanism)
        if match.same:
            return _refuse(
                "inflight_duplicate",
                f"already in flight as {att.id} ({match.reason})",
                proposal=prop,
                matched_prior=att,
                match_score=match.score,
                retrieved=retrieved,
            )

    # 2. Terminal negative science, unless retry_when actually reopens it.
    for hit in retrieved.science:
        entry = hit.entry
        if not entry.terminal():
            continue
        ok, why = science_reopen_allowed(entry, prop)
        if not ok:
            return _refuse(
                "negative_science",
                f"matches {entry.id} ({entry.science_class}): {entry.mechanism} — {why}",
                proposal=prop,
                matched_science=entry,
                match_score=hit.score,
                retrieved=retrieved,
            )

    # 3. Previous attempts of the same mechanism.
    for att in att_list:
        if att.is_hollow():
            continue
        match = same_mechanism(prop.mechanism, att.mechanism)
        if not match.same:
            continue
        invalid = prior_test_invalid(att)
        if invalid:
            return _allow(
                "invalid_prior",
                f"same mechanism as {att.id} but previous test was invalid ({invalid})",
                proposal=prop,
                matched_prior=att,
                match_score=match.score,
                changed=["previous_test_invalid"],
                retrieved=retrieved,
            )
        delta = changed_axes(prop, att)
        if delta:
            return _allow(
                "retry_delta",
                f"same mechanism as {att.id} but {', '.join(delta)} changed",
                proposal=prop,
                matched_prior=att,
                match_score=match.score,
                changed=delta,
                retrieved=retrieved,
                details=_delta_detail(prop, att, delta),
            )
        return _refuse(
            "semantic_duplicate",
            f"semantic duplicate of {att.id} ({match.reason}); no precondition, "
            "implementation, representation or evidence changed, and the previous "
            "test was valid",
            proposal=prop,
            matched_prior=att,
            match_score=match.score,
            retrieved=retrieved,
        )

    return _allow(
        "new_mechanism",
        "no semantic match in science, prior attempts, or running mechanisms",
        proposal=prop,
        retrieved=retrieved,
    )


def _delta_detail(proposal: Proposal, prior: Attempt, changed: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "representation" in changed:
        out["representation"] = {"before": prior.representation, "after": proposal.representation}
    if "implementation" in changed:
        out["implementation"] = {"before": prior.implementation, "after": proposal.implementation}
    if "evidence" in changed:
        out["evidence"] = {"before": prior.evidence_digest, "after": proposal.evidence_digest}
    if "precondition" in changed:
        before = {k: getattr(prior, k) for k in PRECONDITION_AXES}
        after = {k: getattr(proposal, k) for k in PRECONDITION_AXES}
        out["precondition"] = {
            "before": before,
            "after": after,
            "which": [k for k in PRECONDITION_AXES if not _eq_opt(before[k], after[k])],
        }
    return out


# ---------------------------------------------------------------- synthesize


def _slug(mechanism_id: str, bottleneck: str, model: str) -> str:
    bn = "".join(ch if ch.isalnum() else "-" for ch in bottleneck.lower())
    bn = "-".join(p for p in bn.split("-") if p)[:24] or "bn"
    mid = mechanism_id.replace("_", "-")
    return f"auto-{model}-{bn}-{mid}"


def _exhausted_for(
    bottleneck: str, attempts: Iterable[Attempt]
) -> list[Attempt]:
    out = []
    seen: set[str] = set()
    for att in attempts:
        if att.is_hollow():
            continue
        key = att.id or att.mechanism
        if key in seen:
            continue
        if _same_bottleneck(bottleneck, att.bottleneck) or not att.bottleneck:
            seen.add(key)
            out.append(att)
    return out


def item_used(item: CatalogItem, exhausted: Iterable[Attempt]) -> Attempt | None:
    for att in exhausted:
        if same_mechanism(item.mechanism, att.mechanism).same:
            return att
        fp = fingerprint(att.mechanism)
        if fp.canonical_id == item.mechanism_id:
            return att
    return None


def synthesize(
    *,
    measured: Iterable[dict[str, Any]] | None,
    attempts: Iterable[Attempt | Mapping[str, Any]] | None = None,
    science: Iterable[ScienceEntry | Mapping[str, Any]] | None = None,
    running: Iterable[Attempt | Mapping[str, Any]] | None = None,
    include_campaign_exhausted: bool = True,
    model: str = "qwen38",
    catalog: Iterable[CatalogItem] | None = None,
) -> Decision:
    """Invent the next unused mechanism for the dominant remaining bottleneck."""
    if (model or "").lower() in SEALED_MODELS:
        return _error(
            "synthesis_when_dry",
            f"{model} is sealed; will not synthesize a mechanism for a deleted vehicle",
        )
    try:
        rows = load_measured_bottlenecks(measured)
    except Exception as exc:  # noqa: BLE001
        return _error("synthesis_needs_bottleneck", f"measured ledger unreadable: {exc}")

    remaining = remaining_bottlenecks(rows)
    if not remaining:
        return Decision(
            Verdict.HOLD,
            "no measured bottleneck remains at or above the floor; nothing to synthesize",
            "synthesis_needs_bottleneck",
            details={"floor_ns": REMAINING_NS_FLOOR, "n_rows": len(rows)},
        )

    att_list = [coerce_attempt(a) for a in (attempts or ())]
    if include_campaign_exhausted:
        att_list = list(CAMPAIGN_EXHAUSTED) + att_list
    run_list = [coerce_attempt(a) for a in (running or ())]
    try:
        sci_list = [coerce_science(s) for s in (science or ())]
    except Exception as exc:  # noqa: BLE001
        return _error("negative_science", f"science register unreadable: {exc}")

    items = tuple(catalog) if catalog is not None else SYNTHESIS_CATALOG
    if not items:
        return _error("synthesis_not_replay", "synthesis catalog is empty; refusing to invent from nothing")

    # Walk remaining bottlenecks (dominant first); skip items that are
    # semantically already exhausted, in flight, or terminal science.
    tried = []
    for bn_row in remaining:
        bn = bn_row["component"]
        exhausted = _exhausted_for(bn, att_list + run_list)
        for item in items:
            if item.bottleneck != bn and item.bottleneck != bn_row.get("bottleneck"):
                # Allow a catalog item whose bottleneck name matches after normalize.
                if not _same_bottleneck(item.bottleneck, bn):
                    continue
            used = item_used(item, exhausted)
            if used is not None:
                tried.append({"item": item.mechanism_id, "used_by": used.id})
                continue
            # Science veto.
            sci_hit = None
            for entry in sci_list:
                if not entry.terminal() or not entry.applies_to(model):
                    continue
                if same_mechanism(item.mechanism, entry.mechanism).same:
                    sci_hit = entry
                    break
            if sci_hit is not None:
                tried.append({"item": item.mechanism_id, "science": sci_hit.id})
                continue
            target = {
                "id": _slug(item.mechanism_id, bn, model),
                "model": model,
                "mechanism": item.mechanism,
                "mechanism_id": item.mechanism_id,
                "hypothesis": item.mechanism[:200],
                "from_bottleneck": f"{bn} {int(round(bn_row['ns_per_token']))} ns",
                "bottleneck": bn,
                "axis": item.axis,
                "assumption_attacked": item.assumption_attacked,
                "experiment": item.experiment,
                "status": "pending",
                "auto_generated": True,
                "synthesized": True,
                "representation": GENESIS_VEHICLE["representation"],
                "implementation": GENESIS_VEHICLE["implementation"],
                "artifact": GENESIS_VEHICLE["artifact"],
                "recoverable_ns_per_token": float(bn_row["ns_per_token"]),
            }
            syn = Synthesized(
                mechanism_id=item.mechanism_id,
                mechanism=item.mechanism,
                bottleneck=bn,
                axis=item.axis,
                assumption_attacked=item.assumption_attacked,
                experiment=item.experiment,
                model=model,
                target=target,
            )
            # Final self-check: the new mechanism must not match exhausted.
            for att in exhausted:
                if same_mechanism(syn.mechanism, att.mechanism).same:
                    return _error(
                        "synthesis_not_replay",
                        f"catalog item {item.mechanism_id} collapsed onto exhausted {att.id}",
                        synthesized=syn,
                    )
            return Decision(
                Verdict.SYNTHESIZE,
                f"queue dry; invented {item.mechanism_id} against remaining {bn} "
                f"({int(round(bn_row['ns_per_token']))} ns)",
                "synthesis_when_dry",
                synthesized=syn,
                details={
                    "remaining": remaining,
                    "skipped": tried,
                    "n_exhausted": len(exhausted),
                },
            )

    return _error(
        "synthesis_not_replay",
        "every catalog item for every remaining bottleneck is exhausted or "
        "science-blocked; will not replay mechanism one and will not declare "
        "the bottleneck complete",
        details={"remaining": remaining, "skipped": tried},
    )


def decide_next(
    *,
    pending: Iterable[Proposal | Mapping[str, Any]] | None = None,
    measured: Iterable[dict[str, Any]] | Mapping[str, Any] | str | Path | None = None,
    attempts: Iterable[Attempt | Mapping[str, Any]] | None = None,
    science: Iterable[ScienceEntry | Mapping[str, Any]] | None = None,
    running: Iterable[Attempt | Mapping[str, Any]] | None = None,
    include_campaign_exhausted: bool = True,
    model: str = "qwen38",
) -> Decision:
    """Launch an admissible pending target, or synthesize if the queue is dry."""
    pending_list = [coerce_proposal(p) for p in (pending or ())]
    refusals: list[dict[str, Any]] = []
    for prop in pending_list:
        decision = admit(
            prop,
            science=science,
            attempts=attempts,
            running=running,
            include_campaign_exhausted=include_campaign_exhausted,
        )
        if decision.verdict == Verdict.ALLOW:
            decision.verdict = Verdict.LAUNCH
            decision.reason = f"launch admissible {prop.id or prop.mechanism!r}: {decision.reason}"
            decision.details = {**decision.details, "refusals": refusals}
            return decision
        refusals.append(
            {
                "id": prop.id,
                "mechanism": prop.mechanism,
                "verdict": decision.verdict.value,
                "gate": decision.gate,
                "reason": decision.reason,
            }
        )

    # No admissible pending work: synthesize if a bottleneck remains.
    syn = synthesize(
        measured=measured,
        attempts=attempts,
        science=science,
        running=running,
        include_campaign_exhausted=include_campaign_exhausted,
        model=model,
    )
    syn.details = {**(syn.details or {}), "pending_refusals": refusals}
    return syn


# ---------------------------------------------------------------- evidence / selfcheck


def _vehicle_proposal(mechanism: str, **kw: Any) -> Proposal:
    base = dict(
        mechanism=mechanism,
        bottleneck="weight_addressing",
        model="qwen38",
        representation=GENESIS_VEHICLE["representation"],
        implementation=GENESIS_VEHICLE["implementation"],
        genome=GENESIS_VEHICLE["genome"],
        bytes_per_token=GENESIS_VEHICLE["bytes_per_token"],
        launch_geometry=GENESIS_VEHICLE["launch_geometry"],
        command_topology=GENESIS_VEHICLE["command_topology"],
        artifact=GENESIS_VEHICLE["artifact"],
    )
    base.update(kw)
    return Proposal(**base)


def demonstration_semantic_duplicate() -> Decision:
    prior = CAMPAIGN_EXHAUSTED[0]
    return admit(
        _vehicle_proposal("fuse small Metal kernels into the next GEMV"),
        attempts=[prior],
        science=[],
        running=[],
        include_campaign_exhausted=False,
    )


def demonstration_legitimate_retry() -> Decision:
    prior = CAMPAIGN_EXHAUSTED[0]
    return admit(
        _vehicle_proposal(
            "fusing tiny kernels into the following GEMV",
            launch_geometry="tpr32_tg256",
            genome="1 CB / 964 dispatches; geo_tpr32_tg256",
        ),
        attempts=[prior],
        science=[],
        running=[],
        include_campaign_exhausted=False,
    )


def demonstration_synthesis() -> Decision:
    ledger = [
        {"component": "weight_addressing", "ns_per_token": 21_293_103, "pct": 60.44, "triage": "EXISTENTIAL"},
        {"component": "deltanet", "ns_per_token": 3_732_795, "pct": 10.60, "triage": "research"},
        {"component": "command_submission", "ns_per_token": 12_084, "pct": 0.03, "triage": "noise"},
    ]
    return decide_next(
        pending=[],
        measured=ledger,
        attempts=list(CAMPAIGN_EXHAUSTED),
        science=[],
        running=[],
        include_campaign_exhausted=True,
    )


def corrupt_gate_cases() -> list[dict[str, Any]]:
    """Each gate, with a corrupted input that must go red (REFUSE or ERROR)."""
    prior = CAMPAIGN_EXHAUSTED[0]
    ledger = [
        {"component": "weight_addressing", "ns_per_token": 21_293_103, "triage": "EXISTENTIAL"},
    ]
    cases: list[dict[str, Any]] = []

    def run(name: str, corrupt: str, fn) -> None:
        decision = fn()
        cases.append(
            {
                "gate": name,
                "corrupt": corrupt,
                "verdict": decision.verdict.value,
                "red": decision.red,
                "reason": decision.reason,
            }
        )

    run(
        "named_mechanism_required",
        "empty mechanism text",
        lambda: admit(
            {"mechanism": "", "bottleneck": "weight_addressing", "model": "qwen38"},
            attempts=[prior],
        ),
    )
    run(
        "named_mechanism_required",
        "mechanism is only the bottleneck name",
        lambda: admit(
            {"mechanism": "weight_addressing", "bottleneck": "weight_addressing", "model": "qwen38"},
            attempts=[prior],
        ),
    )
    run(
        "semantic_duplicate",
        "paraphrase of exhausted fuse-into-GEMV with no delta",
        lambda: demonstration_semantic_duplicate(),
    )
    run(
        "retry_delta",
        "claimed change but launch_geometry and the rest are identical",
        lambda: admit(
            _vehicle_proposal("fusing tiny kernels into the following GEMV"),
            attempts=[prior],
            include_campaign_exhausted=False,
        ),
    )
    run(
        "invalid_prior",
        "prior marked valid and no delta, so invalid-prior path must not fire (duplicate red)",
        lambda: admit(
            _vehicle_proposal("fusing tiny kernels into the following GEMV", test_valid=True),
            attempts=[prior],
            include_campaign_exhausted=False,
        ),
    )
    run(
        "inflight_duplicate",
        "running record with empty mechanism",
        lambda: admit(
            _vehicle_proposal("per-layer and per-head assignment"),
            attempts=[],
            running=[Attempt(id="live-1", mechanism="", bottleneck="weight_addressing", status="running")],
        ),
    )
    run(
        "inflight_duplicate",
        "same mechanism already running",
        lambda: admit(
            _vehicle_proposal("per-layer and per-head assignment"),
            running=[
                Attempt(
                    id="live-assign",
                    mechanism="assign codecs per layer and per head",
                    bottleneck="weight_addressing",
                    status="running",
                    **{k: GENESIS_VEHICLE[k] for k in ("representation", "implementation", "artifact") if k in GENESIS_VEHICLE},
                )
            ],
        ),
    )
    run(
        "negative_science",
        "science entry missing class",
        lambda: admit(
            _vehicle_proposal("gaussian activations"),
            science=[{"id": "NS-009", "mechanism": "gaussian activations"}],
        ),
    )
    run(
        "negative_science",
        "unknown science class",
        lambda: admit(
            _vehicle_proposal("gaussian activations"),
            science=[
                {
                    "id": "NS-X",
                    "mechanism": "gaussian activations",
                    "class": "HAPPY",
                    "models": ["qwen38"],
                }
            ],
        ),
    )
    run(
        "negative_science",
        "terminal science with retry_when never",
        lambda: admit(
            _vehicle_proposal("evaluate or fit compression on gaussian / synthetic proxy activations"),
            science=[
                {
                    "id": "NS-009",
                    "mechanism": "Evaluate or fit compression on Gaussian / synthetic proxy activations",
                    "class": "REFUTED",
                    "models": ["qwen38"],
                    "retry_when": "never. Real captured activations only.",
                }
            ],
        ),
    )
    run(
        "sealed_model",
        "model=q80 (weights deleted)",
        lambda: admit(
            _vehicle_proposal("per-layer and per-head assignment", model="q80"),
        ),
    )
    run(
        "synthesis_needs_bottleneck",
        "ledger components all zero / missing ns",
        lambda: synthesize(measured=[{"component": "weight_addressing"}], include_campaign_exhausted=True),
    )
    run(
        "synthesis_needs_bottleneck",
        "ledger dict with no components and no ranked_by_ns",
        lambda: synthesize(measured={"schema": "broken"}, include_campaign_exhausted=True),
    )
    run(
        "synthesis_not_replay",
        "catalog contains only an already-exhausted item",
        lambda: synthesize(
            measured=ledger,
            attempts=list(CAMPAIGN_EXHAUSTED),
            include_campaign_exhausted=True,
            catalog=(
                CatalogItem(
                    mechanism_id="fuse_tiny_kernels_into_gemv",
                    mechanism="fusing tiny kernels into the following GEMV",
                    bottleneck="weight_addressing",
                    axis="fusion",
                    assumption_attacked="tiny kernels are free",
                    experiment="do not",
                ),
            ),
        ),
    )
    return cases


def emit_evidence(path: Path | str | None = None) -> dict[str, Any]:
    """Run the required demonstrations and write the receipt."""
    dup = demonstration_semantic_duplicate()
    retry = demonstration_legitimate_retry()
    syn = demonstration_synthesis()
    reds = corrupt_gate_cases()
    receipt = {
        "schema": "hawking.ascent.mechanism_dedup.evidence.v1",
        "date": DATE,
        "identity_schema": IDENTITY_SCHEMA,
        "law": (
            "Dedup BOTTLENECKS is not dedup MECHANISMS. weight_addressing may be "
            "attacked again; the same MECHANISM may not, unless a precondition, "
            "implementation, representation or the evidence changed, or the "
            "previous test was invalid."
        ),
        "demonstrations": {
            "semantic_duplicate_refused": {
                **dup.to_dict(),
                "proposal_mechanism": "fuse small Metal kernels into the next GEMV",
                "matched_prior_id": dup.matched_prior.id if dup.matched_prior else None,
                "matched_prior_mechanism": dup.matched_prior.mechanism if dup.matched_prior else None,
            },
            "legitimate_retry_allowed": {
                **retry.to_dict(),
                "which_precondition_changed": (retry.details or {}).get("precondition", {}).get("which"),
                "before": (retry.details or {}).get("precondition", {}).get("before"),
                "after": (retry.details or {}).get("precondition", {}).get("after"),
            },
            "synthesis_from_dry_queue": {
                **syn.to_dict(),
                "pending": [],
                "new_mechanism": syn.synthesized.mechanism if syn.synthesized else None,
                "new_mechanism_id": syn.synthesized.mechanism_id if syn.synthesized else None,
                "not_replay_of": [a.id for a in CAMPAIGN_EXHAUSTED],
            },
        },
        "gates_corrupted": reds,
        "all_corrupt_gates_red": all(c["red"] for c in reds),
        "n_campaign_exhausted": len(CAMPAIGN_EXHAUSTED),
    }
    dest = Path(path) if path is not None else DEFAULT_RECEIPT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["_path"] = str(dest)
    return receipt


def selfcheck() -> None:
    dup = demonstration_semantic_duplicate()
    assert dup.verdict == Verdict.REFUSE and dup.matched_prior is not None, dup
    retry = demonstration_legitimate_retry()
    assert retry.verdict == Verdict.ALLOW and "precondition" in retry.changed, retry
    syn = demonstration_synthesis()
    assert syn.verdict == Verdict.SYNTHESIZE and syn.synthesized is not None, syn
    assert syn.synthesized.mechanism_id != "fuse_tiny_kernels_into_gemv", syn
    reds = corrupt_gate_cases()
    assert reds and all(c["red"] for c in reds), reds
    print("selfcheck ok")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["selfcheck", "emit-receipts"])
    args = ap.parse_args()
    if args.command == "selfcheck":
        selfcheck()
    else:
        rec = emit_evidence()
        print(json.dumps({"wrote": rec["_path"], "all_corrupt_gates_red": rec["all_corrupt_gates_red"]}))
