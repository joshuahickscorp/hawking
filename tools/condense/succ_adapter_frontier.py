#!/usr/bin/env python3.12
"""Architecture adapter CONTRACTS for the three giant frontier parents (685B / 1T / 1.6T).

Master goal sections 5 and 19: before any giant-parent conversion runs, each architecture
family needs a typed, fail-closed adapter contract that DECLARES its capability conjunction
honestly and REFUSES to execute. This module is the successor analogue of the gpt-oss
0.1-contract adapter: it names the exact requirement set, it binds the real bound geometry
from ``succ_frontier.PARENTS`` (never invented from memory), and it never masquerades as a
completed codec experiment.

Three families are contracted here:
  - deepseek_v32  (685B, bf16 + fp8-scheme, MLA, MTP=1)   -> CORE vs CORE+MTP claim split
  - kimi_k25      (1T,   int4 compressed-tensors, MLA, multimodal) -> TEXT_CORE vs FULL_MULTIMODAL
  - deepseek_v4   (1.6T, fp4 experts + fp8 trunk, 6 selected experts) -> CORE vs CORE+MTP

Every adapter reports ``implementation_state='contract_not_executable'``. Nothing is built and
reviewed, no synthetic twin is green, so ``capabilities(model_type).ready_for_execution`` is
False for all three and ``run(...)`` hard-refuses with exit code 78. The claim taxonomy is
load-bearing: a text-core artifact enumerates its omitted vision bytes and never calls itself
the full model; a CORE artifact that omits the MTP head never claims full-model equivalence.

This module is additive and non-interfering: it launches nothing, downloads nothing, and
writes only under ``reports/condense/event_horizon_successor/adapter/``. The 595-1371 GB giant
sources are disk-walled; source acquisition is deferred to the remote bounded-stream Press.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError,
    seal_field,
    sealed,
    hash_value,
    now_iso,
    atomic_write_json,
    read_json_safe,
    repo_root,
)
from succ_frontier import PARENTS, GiantParent  # noqa: E402

# ── schema registry ────────────────────────────────────────────────────────────────────
SCHEMA_ADAPTER_FRONTIER = "hawking.successor.adapter_frontier.v1"
SCHEMA_ADAPTER_SPEC = "hawking.successor.adapter_spec.v1"
SCHEMA_ADAPTER_CAPABILITIES = "hawking.successor.adapter_capabilities.v1"
SCHEMA_ADAPTER_RUN_REFUSAL = "hawking.successor.adapter_run_refusal.v1"

# No twin is green and nothing is reviewed, so every adapter is a non-executable contract.
IMPLEMENTATION_STATE = "contract_not_executable"
REFUSAL_REASON = "contract_not_reviewed_and_twin_not_green"
REFUSAL_EXIT = 78

# The real requirement conjunction a giant adapter must satisfy before it can execute. Every
# one is FALSE/pending today; ready_for_execution is their logical AND (also False).
REQUIREMENT_KEYS = (
    "source_conversion",       # streamed source -> deterministic staged units
    "reassembly_provenance",   # every archive tensor bound to source shard SHA + byte ranges
    "runtime_specs",           # Apple-Silicon MoE/MLA runtime kernels for this family exist
    "tokenizer_template",      # tokenizer + chat template bound and validated
    "evaluator",               # standalone-capability + codec-fidelity evaluator wired
    "native_load_parity",      # native .tq loader parity vs source reference
    "exact_resume",            # exact resume from a press checkpoint
    "streamed_lifecycle",      # remote bounded-stream Press lifecycle proven
    "quality_path",            # quality-recovery (Doctor) path executable, not just declared
    "twin_green",              # synthetic geometry twin is green
    "reviewed",                # contract reviewed for live giant-parent execution
)

# Fixed policy strings shared by every giant adapter contract.
SOURCE_ACQUISITION_POLICY = "remote_bounded_stream_press"
STREAMED_CONVERSION_POLICY = "four_deterministic_passes"
NATIVE_LOAD_PARITY_STATE = "missing"
EXACT_RESUME_MECHANISM = "press_checkpoint"
QUALITY_PATH_STATE = "deferred"


class AdapterFrontierError(EcoError):
    """Fail-closed error for the giant-parent adapter contract layer."""


def adapter_state_root() -> str:
    """Successor-only artifact namespace. Never under the campaign (doctor_v5_ultra)."""
    return str(repo_root() / "reports" / "condense" / "event_horizon_successor" / "adapter")


# ── per-family source-precision handling (declared, not executed) ────────────────────────
def _source_precision_handling(p: GiantParent) -> dict[str, Any]:
    """Honest declaration of how each family's stored precision would be decoded. The three
    families differ materially, so the contract binds the family-specific scheme rather than
    a single generic path."""
    if p.model_type == "deepseek_v32":
        return {
            "family": "deepseek_v32",
            "stored": "bf16_weights_with_fp8_scheme",
            "decode_units": ["bf16_2d_weight", "fp8_block_scale"],
            "expert_payload": "bf16",
            "trunk_payload": "bf16",
            "vision_boundary": None,
            "note": "bf16 body with an fp8 quantization scheme on select blocks; ~2.0 B/param.",
        }
    if p.model_type == "kimi_k25":
        return {
            "family": "kimi_k25",
            "stored": "int4_compressed_tensors",
            "decode_units": ["int4_packed_weight", "group_scale", "group_zero_point"],
            "expert_payload": "int4",
            "trunk_payload": "int4",
            "vision_boundary": "text_tower_vs_vision_tower_split_required",
            "note": "native INT4 compressed-tensors source with a vision boundary; ~0.60 B/param.",
        }
    if p.model_type == "deepseek_v4":
        return {
            "family": "deepseek_v4",
            "stored": "fp4_experts_with_fp8_trunk",
            "decode_units": ["fp4_expert_weight", "fp8_trunk_weight", "block_scale"],
            "expert_payload": "fp4",
            "trunk_payload": "fp8",
            "vision_boundary": None,
            "note": "mixed native FP4 experts over an FP8 trunk; ~0.54 B/param.",
        }
    raise AdapterFrontierError(f"unknown model_type for precision handling: {p.model_type}")


def _attention_declaration(p: GiantParent) -> dict[str, Any]:
    """MLA is bound via kv_lora_rank. deepseek_v4 metadata carries no kv_lora_rank, so the
    contract records that honestly rather than asserting MLA."""
    if p.kv_lora_rank:
        return {
            "mechanism": "MLA",
            "kv_lora_rank": p.kv_lora_rank,
            "q_lora_rank": p.q_lora_rank,
            "state": "declared_not_wired",
        }
    return {
        "mechanism": "attention_kv_lora_rank_absent_in_metadata",
        "kv_lora_rank": None,
        "q_lora_rank": p.q_lora_rank,
        "state": "declared_not_wired",
        "note": "kv_lora_rank missing in bound metadata; MLA cannot be claimed for this family yet.",
    }


def _mtp_declaration(p: GiantParent) -> dict[str, Any]:
    """MTP handling. When mtp_layers > 0 the CORE artifact OMITS the MTP head and therefore
    must not claim full-model equivalence; the full-model claim requires the CORE+MTP artifact."""
    if p.mtp_layers > 0:
        return {
            "mtp_layers": p.mtp_layers,
            "artifact_components": ["CORE", "CORE+MTP"],
            "core_is_full_model": False,
            "full_model_component": "CORE+MTP",
            "rule": "never claim full-model equivalence while omitting the MTP head",
        }
    return {
        "mtp_layers": 0,
        "artifact_components": ["CORE"],
        "core_is_full_model": True,
        "full_model_component": "CORE",
        "rule": "no MTP head in source; CORE carries the full text body",
    }


def _vision_declaration(p: GiantParent) -> dict[str, Any]:
    """Vision handling. A multimodal parent forces the TEXT_CORE vs FULL_MULTIMODAL claim
    split: the text-core artifact enumerates its omitted vision bytes and never calls itself
    the full model."""
    if p.multimodal:
        return {
            "multimodal": True,
            "claim_split": ["TEXT_CORE", "FULL_MULTIMODAL"],
            "text_core_is_full_model": False,
            "full_model_component": "FULL_MULTIMODAL",
            "omitted_by_text_core": ["vision_tower_bytes", "vision_projector_bytes"],
            "rule": "text-core lists omitted vision bytes and never calls itself the full model",
        }
    return {
        "multimodal": False,
        "claim_split": None,
        "text_core_is_full_model": None,
        "full_model_component": None,
        "omitted_by_text_core": [],
        "rule": "text-only family; no vision boundary",
    }


def claim_components(p: GiantParent) -> list[dict[str, Any]]:
    """The artifact claim taxonomy for a parent, with honest byte components. Byte totals per
    component await tensor-shape summation over the pinned source, so component_bytes is null
    and byte_accounting stays 'pending_tensor_shape_summation'; the source total is bound now."""
    resident = "trunk+router+shared_experts+routed_experts+MLA_latents"
    components: list[dict[str, Any]] = []
    if p.multimodal:
        components.append({
            "component_id": "K2.6_TEXT_CORE",
            "label": "text_core",
            "full_model_equivalence": False,
            "includes": resident,
            "omitted_components": ["vision_tower", "vision_projector"],
            "omitted_bytes": ["vision_tower_bytes", "vision_projector_bytes"],
            "component_bytes": None,
            "byte_accounting": "pending_tensor_shape_summation",
            "source_total_bytes": p.source_bytes,
            "claim_note": "text-core only; omits vision bytes; NOT the full multimodal model.",
        })
        components.append({
            "component_id": "K2.6_FULL_MULTIMODAL",
            "label": "full_multimodal",
            "full_model_equivalence": True,
            "includes": resident + "+vision_tower+vision_projector",
            "omitted_components": [],
            "omitted_bytes": [],
            "component_bytes": None,
            "byte_accounting": "pending_tensor_shape_summation",
            "source_total_bytes": p.source_bytes,
            "claim_note": "full multimodal model including the vision tower and projector.",
        })
        return components
    # text-only families: CORE (omits MTP when present) and, if MTP exists, CORE+MTP.
    core_full = p.mtp_layers == 0
    components.append({
        "component_id": "CORE",
        "label": "core",
        "full_model_equivalence": core_full,
        "includes": resident,
        "omitted_components": [] if core_full else ["mtp_head"],
        "omitted_bytes": [] if core_full else ["mtp_head_bytes"],
        "component_bytes": None,
        "byte_accounting": "pending_tensor_shape_summation",
        "source_total_bytes": p.source_bytes,
        "claim_note": ("core body; carries the full text model" if core_full
                       else "core body; OMITS the MTP head; NOT full-model equivalent."),
    })
    if p.mtp_layers > 0:
        components.append({
            "component_id": "CORE+MTP",
            "label": "core_plus_mtp",
            "full_model_equivalence": True,
            "includes": resident + "+mtp_head",
            "omitted_components": [],
            "omitted_bytes": [],
            "component_bytes": None,
            "byte_accounting": "pending_tensor_shape_summation",
            "source_total_bytes": p.source_bytes,
            "claim_note": "core body plus the MTP head; full-model equivalent.",
        })
    return components


# ── the typed adapter spec (declared capability conjunction; not executable) ─────────────
@dataclasses.dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    model_type: str
    row_id: str
    architecture: str
    implementation_state: str
    source_precision: str
    source_precision_handling: dict[str, Any]
    expert_config: dict[str, Any]
    attention: dict[str, Any]
    mtp: dict[str, Any]
    vision: dict[str, Any]
    doctor_families: tuple[str, ...]
    source_acquisition_policy: str
    streamed_conversion_policy: str
    evaluation_contract: str
    native_load_parity: str
    exact_resume: str
    quality_path: str

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["doctor_families"] = list(self.doctor_families)
        d["schema"] = SCHEMA_ADAPTER_SPEC
        return d


def _adapter_id(p: GiantParent) -> str:
    return f"hawking-successor-adapter-{p.model_type}-contract"


def _build_adapter_spec(p: GiantParent) -> AdapterSpec:
    return AdapterSpec(
        adapter_id=_adapter_id(p),
        model_type=p.model_type,
        row_id=p.row_id,
        architecture=p.architecture,
        implementation_state=IMPLEMENTATION_STATE,
        source_precision=p.source_precision,
        source_precision_handling=_source_precision_handling(p),
        expert_config={
            "n_routed_experts": p.n_routed_experts,
            "experts_per_tok": p.experts_per_tok,
            "n_shared_experts": p.n_shared_experts,
            "moe_intermediate_size": p.moe_intermediate_size,
            "num_layers": p.num_layers,
        },
        attention=_attention_declaration(p),
        mtp=_mtp_declaration(p),
        vision=_vision_declaration(p),
        doctor_families=p.doctor_families,
        source_acquisition_policy=SOURCE_ACQUISITION_POLICY,
        streamed_conversion_policy=STREAMED_CONVERSION_POLICY,
        evaluation_contract="codec_fidelity+standalone_capability(text_core if multimodal)",
        native_load_parity=NATIVE_LOAD_PARITY_STATE,
        exact_resume=EXACT_RESUME_MECHANISM,
        quality_path=QUALITY_PATH_STATE,
    )


# ADAPTERS registry: model_type -> typed adapter spec. Built once from the bound parents.
_PARENTS_BY_TYPE: dict[str, GiantParent] = {p.model_type: p for p in PARENTS}
ADAPTERS: dict[str, AdapterSpec] = {
    p.model_type: _build_adapter_spec(p) for p in PARENTS
}


def _parent_for(model_type: str) -> GiantParent:
    parent = _PARENTS_BY_TYPE.get(model_type)
    if parent is None:
        raise AdapterFrontierError(
            f"unknown model_type {model_type!r}; known: {sorted(_PARENTS_BY_TYPE)}")
    return parent


# ── capability conjunction ───────────────────────────────────────────────────────────────
def _blockers(p: GiantParent) -> list[dict[str, str]]:
    blockers = [
        {"requirement": "twin_green",
         "detail": f"synthetic geometry twin for {p.model_type} is not green"},
        {"requirement": "reviewed",
         "detail": f"adapter contract for {p.model_type} not reviewed for live execution"},
        {"requirement": "source_conversion",
         "detail": f"streamed source->staged conversion for {p.architecture} not implemented"},
        {"requirement": "reassembly_provenance",
         "detail": "no manifest binds archive tensors to source shard SHA + byte ranges"},
        {"requirement": "runtime_specs",
         "detail": f"no Apple-Silicon MoE/MLA runtime for {p.architecture}"},
        {"requirement": "native_load_parity",
         "detail": "native .tq loader parity vs source reference is missing"},
        {"requirement": "exact_resume",
         "detail": "exact resume from a press checkpoint not proven"},
        {"requirement": "streamed_lifecycle",
         "detail": f"remote bounded-stream Press lifecycle not proven; source {round(p.source_bytes/1e9)} GB disk-walled"},
        {"requirement": "evaluator",
         "detail": "standalone-capability + codec-fidelity evaluator not wired"},
        {"requirement": "tokenizer_template",
         "detail": "tokenizer + chat template not bound and validated"},
        {"requirement": "quality_path",
         "detail": "Doctor quality-recovery path declared but not executable"},
    ]
    if p.multimodal:
        blockers.append({
            "requirement": "claim_split",
            "detail": "bind K2.6_TEXT_CORE vs K2.6_FULL_MULTIMODAL; vision tower + projector billed separately"})
    if p.mtp_layers > 0:
        blockers.append({
            "requirement": "mtp_component",
            "detail": "CORE omits the MTP head; CORE+MTP required for a full-model claim"})
    return blockers


def capabilities(model_type: str) -> dict[str, Any]:
    """The REAL requirement conjunction for a family, every requirement FALSE/pending today.
    ready_for_execution is the logical AND, so it is False, with named blockers."""
    p = _parent_for(model_type)
    spec = ADAPTERS[model_type]
    requirements = {key: False for key in REQUIREMENT_KEYS}
    ready = all(requirements.values())  # AND over all-False -> False
    out = {
        "schema": SCHEMA_ADAPTER_CAPABILITIES,
        "adapter_id": spec.adapter_id,
        "model_type": model_type,
        "row_id": p.row_id,
        "architecture": p.architecture,
        "implementation_state": IMPLEMENTATION_STATE,
        "requirements": requirements,
        "requirement_states": {key: "pending" for key in REQUIREMENT_KEYS},
        "ready_for_execution": ready,
        "reviewed_for_live_execution": False,
        "twin_green": False,
        "expert_config": dict(spec.expert_config),
        "attention": dict(spec.attention),
        "mtp": dict(spec.mtp),
        "vision": dict(spec.vision),
        "claim_components": [c["component_id"] for c in claim_components(p)],
        "doctor_families": list(p.doctor_families),
        "source_deletion_permitted": False,
        "quality_claims_permitted": False,
        "full_model_claim_permitted": False,
        "blockers": _blockers(p),
        "generated_at": now_iso(),
    }
    return seal_field(out, "capabilities_sha256")


def run(model_type: str, request: Any) -> dict[str, Any]:
    """Hard refusal. A giant adapter contract never masquerades as executable: it returns a
    sealed refusal with exit 78 and the named blockers until the twin is green and the
    contract is reviewed."""
    p = _parent_for(model_type)
    spec = ADAPTERS[model_type]
    out = {
        "schema": SCHEMA_ADAPTER_RUN_REFUSAL,
        "status": "refused",
        "reason": REFUSAL_REASON,
        "exit": REFUSAL_EXIT,
        "adapter_id": spec.adapter_id,
        "model_type": model_type,
        "row_id": p.row_id,
        "implementation_state": IMPLEMENTATION_STATE,
        "request_repr": repr(request)[:512],
        "ready_for_execution": False,
        "source_deletion_permitted": False,
        "quality_claims_permitted": False,
        "blockers": _blockers(p),
        "refused_at": now_iso(),
    }
    return seal_field(out, "refusal_sha256")


# ── sealed typed spec binding geometry + claim components ─────────────────────────────────
def build_spec(parent: GiantParent) -> dict[str, Any]:
    """A sealed typed spec binding the EXACT bound geometry, the declared adapter contract,
    and the claim components (CORE / CORE+MTP / TEXT_CORE / FULL_MULTIMODAL) with their byte
    components. Not executable: implementation_state == contract_not_executable."""
    if not isinstance(parent, GiantParent):
        raise AdapterFrontierError("build_spec requires a GiantParent")
    spec = ADAPTERS[parent.model_type]
    out = {
        "schema": SCHEMA_ADAPTER_SPEC,
        "adapter_id": spec.adapter_id,
        "model_type": parent.model_type,
        "row_id": parent.row_id,
        "hf_id": parent.hf_id,
        "exact_revision": parent.revision,
        "architecture": parent.architecture,
        "implementation_state": IMPLEMENTATION_STATE,
        "bound_geometry": {
            "num_layers": parent.num_layers,
            "n_routed_experts": parent.n_routed_experts,
            "experts_per_tok": parent.experts_per_tok,
            "n_shared_experts": parent.n_shared_experts,
            "moe_intermediate_size": parent.moe_intermediate_size,
            "kv_lora_rank": parent.kv_lora_rank,
            "q_lora_rank": parent.q_lora_rank,
            "mtp_layers": parent.mtp_layers,
            "hidden_size": parent.hidden_size,
            "vocab_size": parent.vocab_size,
            "max_context": parent.max_context,
            "multimodal": parent.multimodal,
            "n_source_shards": parent.n_source_shards,
            "n_tensors": parent.n_tensors,
            "source_bytes": parent.source_bytes,
            "source_precision": parent.source_precision,
        },
        "adapter_contract": spec.as_dict(),
        "source_precision_handling": _source_precision_handling(parent),
        "attention": _attention_declaration(parent),
        "mtp": _mtp_declaration(parent),
        "vision": _vision_declaration(parent),
        "claim_components": claim_components(parent),
        "doctor_families": list(parent.doctor_families),
        "source_acquisition_policy": SOURCE_ACQUISITION_POLICY,
        "streamed_conversion_policy": STREAMED_CONVERSION_POLICY,
        "evaluation_contract": spec.evaluation_contract,
        "native_load_parity": NATIVE_LOAD_PARITY_STATE,
        "exact_resume": EXACT_RESUME_MECHANISM,
        "quality_path": QUALITY_PATH_STATE,
        "executable": False,
        "full_model_claim_permitted": False,
        "generated_at": now_iso(),
    }
    return seal_field(out, "spec_sha256")


def frontier_adapter_manifest() -> dict[str, Any]:
    """A sealed manifest over all three giant adapter contracts."""
    specs = [build_spec(p) for p in PARENTS]
    caps = {p.model_type: capabilities(p.model_type) for p in PARENTS}
    out = {
        "schema": SCHEMA_ADAPTER_FRONTIER,
        "generated_at": now_iso(),
        "model_types": [p.model_type for p in PARENTS],
        "implementation_state": IMPLEMENTATION_STATE,
        "any_ready_for_execution": any(caps[mt]["ready_for_execution"] for mt in caps),
        "specs": specs,
        "capabilities": {mt: caps[mt] for mt in caps},
    }
    return seal_field(out, "manifest_sha256")


@dataclasses.dataclass(frozen=True)
class AdapterFrontierConfig:
    state_root: str
    implementation_state: str = IMPLEMENTATION_STATE
    refusal_exit: int = REFUSAL_EXIT

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def default_config() -> AdapterFrontierConfig:
    return AdapterFrontierConfig(state_root=adapter_state_root())


def write_manifest_snapshot(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Seal + atomically write the adapter-frontier manifest (successor namespace only)."""
    man = frontier_adapter_manifest()
    atomic_write_json(path, man)
    return man


# ── offline selftest ─────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    expected_types = ["deepseek_v32", "kimi_k25", "deepseek_v4"]
    if sorted(ADAPTERS) != sorted(expected_types):
        raise AdapterFrontierError(f"unexpected adapter registry: {sorted(ADAPTERS)}")

    for mt in expected_types:
        p = _parent_for(mt)

        # 1. capabilities: every requirement pending, AND is False, blockers present.
        caps = capabilities(mt)
        if not sealed(caps, "capabilities_sha256"):
            raise AdapterFrontierError(f"{mt} capabilities not sealed")
        if set(caps["requirements"]) != set(REQUIREMENT_KEYS):
            raise AdapterFrontierError(f"{mt} requirement keys drifted")
        if any(caps["requirements"].values()):
            raise AdapterFrontierError(f"{mt} has a requirement asserted true")
        if caps["ready_for_execution"] is not False:
            raise AdapterFrontierError(f"{mt} ready_for_execution must be False")
        if not caps["blockers"]:
            raise AdapterFrontierError(f"{mt} carries no blockers")
        if caps["full_model_claim_permitted"] is not False:
            raise AdapterFrontierError(f"{mt} must not permit a full-model claim")

        # 2. run: hard refusal with exit 78, sealed, never executable.
        refusal = run(mt, {"probe": "should_not_execute"})
        if refusal["status"] != "refused":
            raise AdapterFrontierError(f"{mt} run did not refuse")
        if refusal["exit"] != REFUSAL_EXIT:
            raise AdapterFrontierError(f"{mt} refusal exit != 78")
        if refusal["reason"] != REFUSAL_REASON:
            raise AdapterFrontierError(f"{mt} refusal reason drifted")
        if not refusal["blockers"]:
            raise AdapterFrontierError(f"{mt} refusal carries no blockers")
        if not sealed(refusal, "refusal_sha256"):
            raise AdapterFrontierError(f"{mt} refusal not sealed")

        # 3. build_spec: sealed, binds the real expert counts, not executable.
        spec = build_spec(p)
        if not sealed(spec, "spec_sha256"):
            raise AdapterFrontierError(f"{mt} spec not sealed")
        if spec["executable"] is not False:
            raise AdapterFrontierError(f"{mt} spec claims executable")
        if spec["implementation_state"] != IMPLEMENTATION_STATE:
            raise AdapterFrontierError(f"{mt} spec implementation_state drifted")
        geo = spec["bound_geometry"]
        if geo["n_routed_experts"] != p.n_routed_experts:
            raise AdapterFrontierError(f"{mt} routed-expert count not bound")
        if geo["experts_per_tok"] != p.experts_per_tok:
            raise AdapterFrontierError(f"{mt} experts_per_tok not bound")
        if geo["n_shared_experts"] != p.n_shared_experts:
            raise AdapterFrontierError(f"{mt} shared-expert count not bound")
        if spec["adapter_contract"]["expert_config"]["n_routed_experts"] != p.n_routed_experts:
            raise AdapterFrontierError(f"{mt} adapter expert_config not bound")

        # every claim component either omits bytes or is explicitly full-model, never both loose.
        for comp in spec["claim_components"]:
            if comp["full_model_equivalence"] and comp["omitted_components"]:
                raise AdapterFrontierError(f"{mt} full-model component omits components")

    # 4. Kimi: multimodal claim split present; text-core omits vision and is NOT full model.
    kimi = build_spec(_parent_for("kimi_k25"))
    kimi_ids = [c["component_id"] for c in kimi["claim_components"]]
    if kimi_ids != ["K2.6_TEXT_CORE", "K2.6_FULL_MULTIMODAL"]:
        raise AdapterFrontierError(f"kimi claim split wrong: {kimi_ids}")
    text_core = next(c for c in kimi["claim_components"] if c["component_id"] == "K2.6_TEXT_CORE")
    if text_core["full_model_equivalence"] is not False:
        raise AdapterFrontierError("kimi text-core claims full-model equivalence")
    if not text_core["omitted_bytes"]:
        raise AdapterFrontierError("kimi text-core must enumerate omitted vision bytes")
    full_mm = next(c for c in kimi["claim_components"] if c["component_id"] == "K2.6_FULL_MULTIMODAL")
    if full_mm["full_model_equivalence"] is not True:
        raise AdapterFrontierError("kimi full-multimodal must be full-model equivalent")

    # 5. deepseek families: CORE vs CORE+MTP; deepseek_v4 has experts_per_tok == 6.
    v32 = build_spec(_parent_for("deepseek_v32"))
    v32_ids = [c["component_id"] for c in v32["claim_components"]]
    if v32_ids != ["CORE", "CORE+MTP"]:
        raise AdapterFrontierError(f"deepseek_v32 claim split wrong: {v32_ids}")
    v32_core = next(c for c in v32["claim_components"] if c["component_id"] == "CORE")
    if v32_core["full_model_equivalence"] is not False:
        raise AdapterFrontierError("deepseek_v32 CORE must not claim full-model equivalence")

    v4 = build_spec(_parent_for("deepseek_v4"))
    if v4["bound_geometry"]["experts_per_tok"] != 6:
        raise AdapterFrontierError("deepseek_v4 experts_per_tok must be 6")
    v4_ids = [c["component_id"] for c in v4["claim_components"]]
    if v4_ids != ["CORE", "CORE+MTP"]:
        raise AdapterFrontierError(f"deepseek_v4 claim split wrong: {v4_ids}")

    # 6. manifest seals; no adapter is ever ready; snapshot round-trips offline.
    man = frontier_adapter_manifest()
    if not sealed(man, "manifest_sha256"):
        raise AdapterFrontierError("manifest not sealed")
    if man["any_ready_for_execution"] is not False:
        raise AdapterFrontierError("a giant adapter reported ready")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "adapter_frontier_manifest.json"
        written = write_manifest_snapshot(out_path)
        reread = read_json_safe(out_path)
        if reread.get("manifest_sha256") != written["manifest_sha256"]:
            raise AdapterFrontierError("snapshot round-trip seal mismatch")

    return {
        "ok": True,
        "adapters": len(ADAPTERS),
        "model_types": expected_types,
        "any_ready_for_execution": man["any_ready_for_execution"],
        "refusal_exit": REFUSAL_EXIT,
        "kimi_claim_split": kimi_ids,
        "deepseek_v4_experts_per_tok": v4["bound_geometry"]["experts_per_tok"],
        "manifest_sha256": man["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Giant-parent architecture adapter CONTRACTS (fail-closed, non-executable).")
    parser.add_argument("--selftest", action="store_true", help="run the offline selftest")
    parser.add_argument("--capabilities", metavar="MODEL_TYPE", default=None,
                        help="print the capability conjunction for a model_type")
    parser.add_argument("--spec", metavar="MODEL_TYPE", default=None,
                        help="print the sealed typed spec for a model_type")
    parser.add_argument("--run", metavar="MODEL_TYPE", default=None,
                        help="attempt to run a model_type (always refuses with exit 78)")
    parser.add_argument("--out", default=None, help="write the sealed adapter-frontier manifest")
    args = parser.parse_args(argv)

    try:
        if args.selftest:
            print(json.dumps(selftest(), indent=2, sort_keys=True))
            return 0
        if args.capabilities:
            print(json.dumps(capabilities(args.capabilities), indent=2, sort_keys=True))
            return 0
        if args.spec:
            print(json.dumps(build_spec(_parent_for(args.spec)), indent=2, sort_keys=True))
            return 0
        if args.run:
            refusal = run(args.run, {"cli": True})
            print(json.dumps(refusal, indent=2, sort_keys=True), file=sys.stderr)
            return int(refusal["exit"])
        man = frontier_adapter_manifest()
        if args.out:
            write_manifest_snapshot(args.out)
        print(json.dumps({
            "schema": man["schema"],
            "manifest_sha256": man["manifest_sha256"],
            "model_types": man["model_types"],
            "any_ready_for_execution": man["any_ready_for_execution"],
            "implementation_state": man["implementation_state"],
        }, indent=2, sort_keys=True))
        return 0
    except AdapterFrontierError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
