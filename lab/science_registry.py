#!/usr/bin/env python3.12
"""Track V operator classification — counted Python authority (no data-file catalog)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

Handler = Callable[..., dict[str, Any]]
CONDENSE_ROOT = Path(__file__).resolve().parents[1] / "tools" / "condense"


class OperatorClass(str, Enum):
    OPERATOR = "operator"
    SPEC = "spec"
    DATASET_ADAPTER = "dataset_adapter"
    EVAL_RULE = "eval_rule"
    FIXTURE = "fixture"
    NUMERICAL_AUTHORITY = "numerical_authority"
    UNCLASSIFIED = "UNCLASSIFIED"


OP, DA, EV, FX, NA, UC = (
    OperatorClass.OPERATOR,
    OperatorClass.DATASET_ADAPTER,
    OperatorClass.EVAL_RULE,
    OperatorClass.FIXTURE,
    OperatorClass.NUMERICAL_AUTHORITY,
    OperatorClass.UNCLASSIFIED,
)


@dataclass(frozen=True)
class OperatorRecord:
    module: str
    class_: OperatorClass
    loc: int
    why: str
    path: str = ""
    handler_keys: tuple[str, ...] = ()
    path_sealed: bool = False
    science: bool = False

    @property
    def class_name(self) -> str:
        return self.class_.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "class": self.class_.value,
            "loc": self.loc,
            "why": self.why,
            "path": self.path or f"tools/condense/{self.module}.py",
            "handler_keys": list(self.handler_keys),
            "path_sealed": self.path_sealed,
            "science": self.science,
        }


def _loc(name: str) -> int:
    path = CONDENSE_ROOT / f"{name}.py"
    if not path.is_file():
        return 0
    return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def _m(
    module: str,
    class_: OperatorClass,
    key: str,
    why: str,
    *,
    sealed: bool = False,
    sci: bool = False,
) -> OperatorRecord:
    """One irreducible-module record. Omit sealed/sci when false (defaults)."""
    return OperatorRecord(
        module=module,
        class_=class_,
        loc=_loc(module),
        why=why,
        path=f"tools/condense/{module}.py",
        handler_keys=(key,) if key else (),
        path_sealed=sealed,
        science=sci,
    )


# Canonical Track V classification: every tools/condense top-level module, once.
IRREDUCIBLE_MODULES: tuple[OperatorRecord, ...] = (
    _m("glm52_common", NA, "artifact.resolve",
       "Single path authority resolve_artifact + canonical seal helpers; path-sealed into contracts and corpus "
       "receipts.", sealed=True),
    _m("glm52_state", UC, "state.controller",
       "Partially decomposed residual controller. Lease half absorbed into engine.lease.SingletonLease "
       "(TOCTOU-hardened flock, parent-chain O_NOFOLLOW open, post-lock revalidation). Remaining body is HashChainLog "
       "+ WindowLedger + TrustedArtifactStore + Controller + GLM52 evidence validators — dual-log claim-bound "
       "transitions under exclusive lease with official shard/coverage contracts. Thin SingletonLease subclass maps "
       "StateError; dies when Controller is absorbed.", sealed=True),
    _m("glm52_parity", EV, "measure.parity",
       "Adapter/twin/reference parity instrument; sealed parity surface.", sealed=True, sci=True),
    _m("glm52_contract", EV, "precheck.contract",
       "Immutable source contract and header-derived ledgers; live reader.", sealed=True, sci=True),
    _m("glm52_source_fetch", DA, "fetch.source",
       "BF16 source streamer with sealed schedule and manifest verification.", sci=True),
    _m("glm52_teacher_capture", OP, "measure.teacher_capture",
       "Teacher-evidence capture on the BF16 stream before eviction.", sci=True),
    _m("glm52_xet_autotune", EV, "plan.xet",
       "Offline planner and fail-closed authority gate for Xet autotuning (adjacent to pack science; not in the "
       "pack/parity/gravity floor).", sealed=True),
    _m("glm52_activation_aware_pack", OP, "pack.activation_v1",
       "Activation-aware packing program v1 (real-activation pilot order).", sci=True),
    _m("glm52_activation_aware_pack_v2", OP, "pack.activation_v2",
       "Activation-aware pack v2 — feasibility + fake-data codec (VARIANT of v1).", sci=True),
    _m("glm52_pack", OP, "pack.stream", "Serialize tensors into physically-exact sub-bit compact shards.", sci=True),
    _m("glm52_adapter", DA, "adapt.checkpoint",
       "Fail-closed checkpoint adapter and bounded safetensors reader.", sealed=True, sci=True),
    _m("glm52_assemble", OP, "assemble.model",
       "Assemble packed shards into one verified local model, or say why not.", sci=True),
    _m("glm52_capture_program", DA, "capture.program",
       "Natural teacher-capture program: real text, disjoint splits, domains.", sci=True),
    _m("glm52_corpus", DA, "corpus.verify",
       "Offline quality-corpus integrity contract; no model/network code.", sealed=True),
    _m("glm52_evidence_auth", EV, "auth.evidence", "Keychain-backed producer authentication for campaign evidence."),
    _m("glm52_functional_gauntlet", EV, "eval.gauntlet",
       "FS0–FS6 functional-escape survival suite across layers/documents.", sci=True),
    _m("glm52_grounding", EV, "eval.grounding",
       "Fail-closed read-only grounding observations (no write/network).", sci=True),
    _m("glm52_grounding_auth", EV, "auth.grounding", "Independent Keychain credential for filesystem observations."),
    _m("glm52_moe_student", NA, "student.moe",
       "Dense random-feature MoE student with ridge fit — function not weights.", sci=True),
    _m("glm52_reference", NA, "oracle.reference",
       "Inspectable NumPy reference forward for main + physical MTP (oracle).", sealed=True, sci=True),
    _m("glm52_shard_probe", OP, "probe.shard",
       "One-pass weight evidence capture for a resident BF16 source shard.", sci=True),
    _m("glm52_synthetic", FX, "fixture.synthetic",
       "Deterministic architecture-preserving safetensors fixture builder.", sealed=True, sci=True),
    _m("glm52_telegram", OP, "notify.telegram", "Secure Telegram credentials and delivery for campaign alerts."),
    _m("glm52_terminal_proofs", EV, "eval.terminal_proofs",
       "Pure semantic proofs for offline-ready stop conditions.", sealed=True, sci=True),
    _m("glm52_xet_live", OP, "exec.xet_live",
       "Authority-gated body-file-free live Xet trials (VARIANT of autotune).", sealed=True),
    _m("doctor_v5_gptoss_mxfp4", OP, "doctor.mxfp4", "Bounded-memory GPT-OSS MXFP4 inventory and staging primitives."),
    _m("gptoss_block", NA, "forward.gptoss_block",
       "Bounded single-block GPT-OSS-120B forward producing real MoE input."),
    _m("gptoss_moe_runtime", OP, "forward.gptoss_moe", "Per-expert STR2 loader + CPU-reference MoE runtime."),
    _m("gptoss_real_forward", OP, "forward.gptoss_real",
       "Full-model GPT-OSS-120B bounded-streaming parity-correct forward."),
    _m("gptoss_subbit_packer", OP, "pack.gptoss_subbit", "Sub-1-bit deployable packer on Metal/MPS.", sci=True),
    _m("gravity_bench_lab", EV, "eval.matched_bench",
       "Matched-benchmark harness — every speed claim must pass through here.", sci=True),
    _m("gravity_flop_ledger", NA, "ledger.flop",
       "FLOP-and-byte ledger separating compression from arithmetic savings.", sci=True),
    _m("gravity_forge", OP, "pack.gravity_forge", "Capability-preserving sub-bit representation foundry.", sci=True),
    _m("artifact_client", OP, "format.gravity",
       "Native .gravity container format (header/shard read/write/verify).", sci=True),
    _m("gravity_functional_codec", OP, "codec.functional_moe",
       "glm52.functional.moe.v1 codec storing a function, not weights.", sci=True),
    _m("gravity_kernel_select", NA, "select.kernel",
       "Kernel selection matrix: which grammar executes which geometry.", sci=True),
    _m("gravity_metal", OP, "kernel.metal",
       "Hand-written Metal kernel: decode inside accumulation, never in memory.", sci=True),
    _m("gravity_metal_lab_b", OP, "kernel.metal_lab_b",
       "Track B shared-table lookup-linear measured on unshared reality (VARIANT).", sci=True),
    _m("gravity_moe_layer", OP, "forward.moe_layer",
       "Complete GLM-5.2 MoE layer as one Metal command buffer, parity-gated.", sci=True),
    _m("gravity_real_fixtures", FX, "fixture.real_packed",
       "Real packed tensors as fixtures from live campaign without disturbance.", sci=True),
    _m("hawking_null_metric", NA, "metric.null",
       "Null-corrected promotion metric (constant-null vs raw cosine).", sci=True),
    _m("bounded_cache", OP, "cache.pressure", "Pressure-aware LRU for decoded experts / large reusable tensors."),
    _m("eco_common", NA, "eco.common", "Shared seal/hash/atomic helpers for Ecosystem Frontier scaffold."),
)


class OperatorRegistry:
    def __init__(
        self,
        records: tuple[OperatorRecord, ...] | None = None,
        *,
        handlers: Mapping[str, Handler] | None = None,
    ) -> None:
        self.records = records or IRREDUCIBLE_MODULES
        self._by_module = {r.module: r for r in self.records}
        self._by_handler: dict[str, OperatorRecord] = {}
        for rec in self.records:
            for key in rec.handler_keys:
                self._by_handler[key] = rec
        self.handlers = dict(handlers or {})

    def get(self, module: str) -> OperatorRecord | None:
        return self._by_module.get(module)

    def for_handler(self, key: str) -> OperatorRecord | None:
        return self._by_handler.get(key)

    def register_handler(self, key: str, handler: Handler) -> None:
        self.handlers[key] = handler

    def resolve_handler(self, key: str) -> Handler | None:
        return self.handlers.get(key)

    def classification(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def unclassifiable(self) -> list[dict[str, Any]]:
        return [
            {"module": r.module, "loc": r.loc, "what_it_is": r.why}
            for r in self.records
            if r.class_ is OperatorClass.UNCLASSIFIED
        ]

    def science_floor_loc(self) -> int:
        return sum(r.loc for r in self.records if r.science)

    def path_loc(self) -> int:
        return sum(r.loc for r in self.records if not r.science)

    def summary(self) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        loc_by_class: dict[str, int] = {}
        for r in self.records:
            by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
            loc_by_class[r.class_name] = loc_by_class.get(r.class_name, 0) + r.loc
        return {
            "schema": "hawking.condense.operator_registry.v1",
            "module_count": len(self.records),
            "total_loc": sum(r.loc for r in self.records),
            "science_floor_loc": self.science_floor_loc(),
            "path_residue_loc": self.path_loc(),
            "by_class_count": by_class,
            "by_class_loc": loc_by_class,
            "unclassifiable": self.unclassifiable(),
            "path_sealed": [r.module for r in self.records if r.path_sealed],
        }


DEFAULT_REGISTRY = OperatorRegistry()


def load_default_registry(
    handlers: Mapping[str, Handler] | None = None,
) -> OperatorRegistry:
    return OperatorRegistry(handlers=handlers)


def classify_all() -> list[dict[str, Any]]:
    return DEFAULT_REGISTRY.classification()
