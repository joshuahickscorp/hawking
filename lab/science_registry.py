#!/usr/bin/env python3.12
"""Track V operator registry — lab.operators authority (C-SCI-R1)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from lab.semantic_taxonomy import (
    CONDENSE_OPERATION,
    legacy_alias_for_schema,
    semantic_tags,
)

Handler = Callable[..., dict[str, Any]]
OPS_ROOT = Path(__file__).resolve().parent / "operators"
GRAVITY_OPERATOR_REGISTRY_SCHEMA = "hawking.gravity.operator_registry.v1"
CONDENSE_OPERATOR_REGISTRY_SCHEMA = "hawking.condense.operator_registry.v1"

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
            "path": self.path or f"lab/operators/{self.module}.py",
            "handler_keys": list(self.handler_keys),
            "path_sealed": self.path_sealed,
            "science": self.science,
        }

def _loc(name: str) -> int:
    path = OPS_ROOT / f"{name}.py"
    if path.is_file():
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    alt = Path(__file__).resolve().parents[1] / "tools" / "condense" / f"{name}.py"
    if alt.is_file():
        return len(alt.read_text(encoding="utf-8", errors="ignore").splitlines())
    return 0

def _m(
    module: str,
    class_: OperatorClass,
    key: str,
    why: str,
    *,
    sealed: bool = False,
    sci: bool = False,
    path: str | None = None,
) -> OperatorRecord:
    return OperatorRecord(
        module=module,
        class_=class_,
        loc=_loc(module),
        why=why,
        path=path or f"lab/operators/{module}.py",
        handler_keys=(key,) if key else (),
        path_sealed=sealed,
        science=sci,
    )

IRREDUCIBLE_MODULES: tuple[OperatorRecord, ...] = (
    _m("glm52_common", NA, "artifact.resolve", "Path authority + seal helpers.", sealed=True),
    _m("glm52_state", UC, "state.controller", "Campaign controller residual under lab lease.", sealed=True),
    _m("glm52_parity", EV, "measure.parity", "Adapter/twin/reference parity.", sealed=True, sci=True),
    _m("glm52_contract", EV, "precheck.contract", "Immutable source contract ledgers.", sealed=True, sci=True),
    _m("glm52_source_fetch", DA, "fetch.source", "BF16 source streamer body.", sci=True),
    _m("glm52_teacher_capture", OP, "measure.teacher_capture", "Teacher-evidence capture.", sci=True),
    _m("glm52_xet_autotune", EV, "plan.xet", "Xet autotune planner.", sealed=True),
    _m("deepseek_v4_xet_autotune", EV, "measure.deepseek_v4_public_xet", "Fresh-process public Xet throughput matrix and frozen profile.", sealed=True, path="tools/condense/deepseek_v4_xet_autotune.py"),
    _m("deepseek_v4_xet_sustained", EV, "measure.deepseek_v4_public_xet_sustained", "Long-form independent HTTP/2 versus direct-presigned public-path confirmation.", sealed=True, path="tools/condense/deepseek_v4_xet_sustained.py"),
    _m("frankenstein_pipeline", OP, "plan.frankenstein_pipeline", "Fail-closed Kimi-to-GLM-to-DeepSeek staged pipeline preflight.", sealed=True, path="tools/condense/frankenstein_pipeline.py"),
    _m("kimi_k3_source_admission", EV, "precheck.kimi_k3_source", "Official metadata-only Kimi K3 immutable source admission.", sealed=True, path="tools/condense/kimi_k3_source_admission.py"),
    _m("deepseek_v4_gravity", OP, "pack.deepseek_v4_stream", "DeepSeek-V4 Gravity stream and restart-safe content-addressed source windows.", sci=True, path="tools/condense/deepseek_v4_gravity.py"),
    _m("deepseek_v4_native_codec", NA, "codec.deepseek_v4_native", "Bounded DeepSeek-V4 FP4/FP8 byte-format codec.", sealed=True, sci=True, path="tools/condense/deepseek_v4_native_codec.py"),
    _m("deepseek_v4_native_fixture", FX, "fixture.deepseek_v4_native", "Bounded in-memory native codec fixture.", sealed=True, sci=True, path="tools/condense/deepseek_v4_native_fixture.py"),
    _m("deepseek_v4_stream_executor", OP, "plan.deepseek_v4_header", "Pinned header-only DeepSeek-V4 source admission.", sealed=True, path="tools/condense/deepseek_v4_stream_executor.py"),
    _m("deepseek_v4_xet_slice", OP, "exec.deepseek_v4_xet_slice", "Bounded zero-cache DeepSeek-V4 Xet range stream.", sealed=True, path="tools/condense/deepseek_v4_xet_slice.py"),
    _m("glm52_activation_aware_pack", OP, "pack.activation_v1", "Activation-aware pack v1.", sci=True),
    _m("glm52_activation_aware_pack_v2", OP, "pack.activation_v2", "Activation-aware pack v2.", sci=True),
    _m("glm52_pack", OP, "pack.stream", "Sub-bit compact shard serializer.", sci=True),
    _m("glm52_adapter", DA, "adapt.checkpoint", "Checkpoint adapter.", sealed=True, sci=True),
    _m("glm52_assemble", OP, "assemble.model", "Assemble packed shards.", sci=True),
    _m("glm52_capture_program", DA, "capture.program", "Teacher-capture program.", sci=True),
    _m("glm52_corpus", DA, "corpus.verify", "Corpus integrity contract.", sealed=True),
    _m("glm52_evidence_auth", EV, "auth.evidence", "Evidence HMAC auth."),
    _m("glm52_functional_gauntlet", EV, "eval.gauntlet", "FS0–FS6 gauntlet.", sci=True),
    _m("glm52_grounding", EV, "eval.grounding", "Grounding observations.", sci=True),
    _m("glm52_grounding_auth", EV, "auth.grounding", "Grounding credentials."),
    _m("glm52_moe_student", NA, "student.moe", "MoE student ridge fit.", sci=True),
    _m("glm52_reference", NA, "oracle.reference", "NumPy reference forward.", sealed=True, sci=True),
    _m("glm52_shard_probe", OP, "probe.shard", "Shard weight probe.", sci=True),
    _m("glm52_synthetic", FX, "fixture.synthetic", "Synthetic fixture builder.", sealed=True, sci=True),
    _m("glm52_telegram", OP, "notify.telegram", "Telegram campaign alerts."),
    _m("glm52_terminal_proofs", EV, "eval.terminal_proofs", "Terminal stop proofs.", sealed=True, sci=True),
    _m("glm52_xet_live", OP, "exec.xet_live", "Live Xet trials.", sealed=True),
    _m("glm52_restream_contract", NA, "", "Fail-closed <=90-GB range restream contract."),
    _m("glm52_range_stream_executor", OP, "", "Owner-gated official Xet framed-range executor."),
    _m("glm52_framed_window_operator", FX, "fixture.framed_window", "Fixture-only framed window lifecycle dry run."),
    _m("gptoss_live_probe", OP, "probe.gptoss", "Bounded GPT-OSS source expert-wave probe.", sci=True),
    _m("gptoss_subbit_packer", OP, "pack.gptoss_subbit", "GPT-OSS sub-bit packer.", sci=True),
    _m("gravity_bench_lab", EV, "eval.matched_bench", "Matched benchmark harness.", sci=True),
    _m("gravity_flop_ledger", NA, "ledger.flop", "FLOP/byte ledger.", sci=True),
    _m("gravity_forge", OP, "pack.gravity_forge", "Sub-bit representation foundry.", sci=True),
    _m("artifact_client", OP, "format.gravity",
       "Native .gravity container (S5 client).", sci=True,
       path="tools/condense/artifact_client.py"),
    _m("gravity_functional_codec", OP, "codec.functional_moe", "Functional MoE codec.", sci=True),
    _m("gravity_kernel_select", NA, "select.kernel", "Kernel selection matrix.", sci=True),
    _m("gravity_metal", OP, "kernel.metal", "Metal decode-in-accumulate kernel.", sci=True),
    _m("gravity_metal_lab_b", OP, "kernel.metal_lab_b", "Track B shared-table kernel.", sci=True),
    _m("gravity_moe_layer", OP, "forward.moe_layer", "MoE layer Metal executor.", sci=True),
    _m("gravity_real_fixtures", FX, "fixture.real_packed", "Real packed fixtures.", sci=True),
    _m("hawking_null_metric", NA, "metric.null", "Null-corrected metric.", sci=True),
    _m("bounded_cache", OP, "cache.pressure", "Pressure-aware LRU cache."),
    _m("condense_controller", NA, "", "Offline bounded rotation control-plan."),
    _m("eco_common", NA, "eco.common", "Shared seal/hash helpers absorbed into operators."),
    _m("subbit_closure", OP, "foundry.subbit_closure", "Sub-bit closure program.", sci=True),
    _m("gravity_potency", OP, "foundry.potency", "Potency registry/atlas.", sci=True),
    _m("one_bit_ceiling", NA, "foundry.ceiling", "One-bit ceiling ledger.", sci=True),
    _m("storage_modes", NA, "foundry.storage", "Storage mode contracts.", sci=True),
    _m("gravity_range_scheduler", NA, "", "Offline 90-GB range/tensor admission planning."),
    _m("acquisition", OP, "foundry.acquisition", "Acquisition proposals.", sci=True),
    _m("quality_contract", EV, "foundry.quality", "Quality contract gates.", sci=True),
    # Family facades: two lines each, `from lab.operators.<real> import *`.
    # They carry no logic, but they are named in lab/operators/__all__, so
    # `from lab.operators import pack` is a declared surface and they cannot
    # simply be deleted. Classified so the Track V contract covers every
    # module in the package rather than every module that happens to be real.
    _m("acquire", UC, "", "Family facade over acquisition."),
    _m("auth", UC, "", "Family facade over the evidence/grounding auth modules."),
    _m("evaluate", UC, "", "Family facade over the eval rules."),
    _m("forge", UC, "", "Family facade over gravity_forge."),
    _m("gravity_exec", UC, "", "Family facade over the gravity execution modules."),
    _m("gravity_math", UC, "", "Family facade over the gravity numeric modules."),
    _m("notify", UC, "", "Family facade over glm52_telegram."),
    _m("pack", UC, "", "Family facade over glm52_pack."),
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
            "schema": GRAVITY_OPERATOR_REGISTRY_SCHEMA,
            "semantic_tags": semantic_tags(
                operation=CONDENSE_OPERATION,
                artifact_kind="operator_registry",
            ),
            "legacy_schema_aliases": [CONDENSE_OPERATOR_REGISTRY_SCHEMA],
            # Keep the compact legacy list for older readers, and provide the
            # complete forward-only mapping for callers that need to know its
            # deprecation and Gravity successor.
            "legacy_schema_compatibility": [
                legacy_alias_for_schema(CONDENSE_OPERATOR_REGISTRY_SCHEMA)
            ],
            "module_count": len(self.records),
            "total_loc": sum(r.loc for r in self.records),
            "science_floor_loc": self.science_floor_loc(),
            "path_residue_loc": self.path_loc(),
            "by_class_count": by_class,
            "by_class_loc": loc_by_class,
            "unclassifiable": self.unclassifiable(),
            "path_sealed": [r.module for r in self.records if r.path_sealed],
            "process_authority": "python3.12 -m lab",
        }

DEFAULT_REGISTRY = OperatorRegistry()

def load_default_registry(handlers: Mapping[str, Handler] | None = None) -> OperatorRegistry:
    return OperatorRegistry(handlers=handlers)

def classify_all() -> list[dict[str, Any]]:
    return DEFAULT_REGISTRY.classification()

def build_operator_handlers() -> dict[str, Handler]:
    """Bind real operator callables for lab op dispatch; missing keys fail closed."""
    import importlib

    def bind(module: str, attr: str) -> Handler:
        def handler(runtime: Any = None, params: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
            mod = importlib.import_module(
                "tools.condense.artifact_client" if module == "artifact_client" else f"lab.operators.{module}"
            )
            fn = getattr(mod, attr)
            params = dict(params or {})
            params.update(kwargs)
            try:
                result = fn(**params) if params else fn()
            except TypeError:
                result = fn()
            if isinstance(result, dict):
                return result
            if isinstance(result, int):
                return {"ok": result == 0, "exit_code": result}
            return {"ok": True, "result": result}
        return handler

    handlers: dict[str, Handler] = {
        "pack.stream": bind("glm52_pack", "pack_indices"),
        "pack.activation_v1": bind("glm52_activation_aware_pack", "selftest"),
        "pack.activation_v2": bind("glm52_activation_aware_pack_v2", "assert_no_gaussian_promotion_path"),
        "fetch.source": bind("glm52_source_fetch", "selftest"),
        "ledger.flop": bind("gravity_flop_ledger", "official_geometry"),
        "notify.telegram": bind("glm52_telegram", "credential_status"),
        "auth.evidence": bind("glm52_evidence_auth", "credential_status"),
        "foundry.potency": bind("gravity_potency", "selftest"),
        "pack.gravity_forge": bind("gravity_forge", "selftest"),
        "eval.terminal_proofs": bind("glm52_terminal_proofs", "derive_all_ready_stop_proofs"),
        "format.gravity": bind("artifact_client", "selftest"),
    }
    return handlers
