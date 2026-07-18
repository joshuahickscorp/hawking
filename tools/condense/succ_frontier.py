#!/usr/bin/env python3.12
"""High-Parameter Frontier Program: exact source authority, physical fit, durable rows.

The three giant parents (685B / 1T / 1.6T). Geometry and revisions below were bound from a
READ-ONLY Hugging Face metadata fetch (config.json + model.safetensors.index.json), never
from memory (master goal: "Do not invent a model identity from memory"). Source bytes are the
exact index total_size. All three are disk-walled on this box, so heavy conversion is gated;
this module is the preparation layer: source authority, physical-fit math, and the durable
supervisor-owned queue rows with the extended giant-parent status vocabulary.

Corrections the real metadata forced over the directive's hypotheses are recorded inline
(e.g. V4-Pro uses 6 selected experts, not 8; Kimi text-core has MTP=0).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, now_iso  # noqa: E402

FRONTIER_SCHEMA = "hawking.successor.frontier_manifest.v1"

# Extended giant-parent status vocabulary (master goal section 3).
FRONTIER_STATUSES = (
    "metadata_pending", "source_authority_pending", "adapter_pending", "synthetic_twin_pending",
    "synthetic_twin_running", "synthetic_twin_green", "waiting_predecessor", "waiting_disk",
    "waiting_release", "waiting_admission", "ready", "procuring", "converting", "doctor_running",
    "packaging", "evaluating", "provisional", "sealed", "retired", "blocked", "invalid",
)

EXECUTION_REGIMES = ("RESIDENT_EXTREME", "HYBRID_EXPERT_EXTREME", "STREAMED_ARCHIVE_EXTREME")


@dataclasses.dataclass(frozen=True)
class GiantParent:
    row_id: str
    hf_id: str
    revision: str
    architecture: str
    model_type: str
    official_total_params: float        # nominal official scale (params)
    active_params: float                # active per token (MoE)
    source_bytes: int                   # exact index total_size
    source_precision: str               # derived from bytes/param
    num_layers: int
    n_routed_experts: int
    experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int | None
    q_lora_rank: int | None
    mtp_layers: int
    max_context: int
    vocab_size: int
    hidden_size: int
    multimodal: bool
    tokenizer_files: tuple[str, ...]
    n_source_shards: int
    n_tensors: int
    resident_anchor_bpw: float
    aggressive_anchor_bpw: float
    extreme_probe_bpw: float
    doctor_families: tuple[str, ...]
    notes: str = ""

    def bytes_per_param(self) -> float:
        return round(self.source_bytes / self.official_total_params, 4)


# ---- exact bound geometry (READ-ONLY HF fetch on 2026-07-17) --------------------------------
PARENTS: tuple[GiantParent, ...] = (
    GiantParent(
        row_id="deepseek-v3.2-685b", hf_id="deepseek-ai/DeepSeek-V3.2",
        revision="a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
        architecture="DeepseekV32ForCausalLM", model_type="deepseek_v32",
        official_total_params=685e9, active_params=37e9, source_bytes=1_370_793_842_752,
        source_precision="bf16_with_fp8_scheme(~2.0 B/param)",
        num_layers=61, n_routed_experts=256, experts_per_tok=8, n_shared_experts=1,
        moe_intermediate_size=2048, kv_lora_rank=512, q_lora_rank=1536, mtp_layers=1,
        max_context=163840, vocab_size=129280, hidden_size=7168, multimodal=False,
        tokenizer_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
        n_source_shards=163, n_tensors=92425,
        resident_anchor_bpw=0.80, aggressive_anchor_bpw=0.68, extreme_probe_bpw=0.52,
        doctor_families=("expert_genome", "capability_immune_bank", "lexical_ark", "mtp_optional"),
        notes="685B bridge; MLA; MTP=1 explicit artifact component (CORE vs CORE+MTP); sparse attn"),
    GiantParent(
        row_id="kimi-k2.6-1t", hf_id="moonshotai/Kimi-K2.6",
        revision="7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        architecture="KimiK25ForConditionalGeneration", model_type="kimi_k25",
        official_total_params=1e12, active_params=32e9, source_bytes=595_148_192_736,
        source_precision="int4_compressed_tensors(~0.60 B/param)",
        num_layers=61, n_routed_experts=384, experts_per_tok=8, n_shared_experts=1,
        moe_intermediate_size=2048, kv_lora_rank=512, q_lora_rank=1536, mtp_layers=0,
        max_context=131072, vocab_size=163840, hidden_size=7168, multimodal=True,
        tokenizer_files=("config.json", "tokenizer_config.json", "chat_template.jinja"),
        n_source_shards=64, n_tensors=208550,
        resident_anchor_bpw=0.55, aggressive_anchor_bpw=0.45, extreme_probe_bpw=0.36,
        doctor_families=("expert_genome", "progressive_expert_slices", "lexical_ark"),
        notes="1T MULTIMODAL (KimiK25ForConditionalGeneration; vision_config present). Claim split "
              "K2.6_TEXT_CORE vs K2.6_FULL_MULTIMODAL. Text-core MTP=0. Native INT4 source. MLA."),
    GiantParent(
        row_id="deepseek-v4-pro-1.6t", hf_id="deepseek-ai/DeepSeek-V4-Pro",
        revision="b5968e9190ef611bbf34a7229255be88a0e937c1",
        architecture="DeepseekV4ForCausalLM", model_type="deepseek_v4",
        official_total_params=1.6e12, active_params=49e9, source_bytes=864_704_792_696,
        source_precision="fp4_experts+fp8_trunk(~0.54 B/param)",
        num_layers=61, n_routed_experts=384, experts_per_tok=6, n_shared_experts=1,
        moe_intermediate_size=3072, kv_lora_rank=None, q_lora_rank=1536, mtp_layers=1,
        max_context=1048576, vocab_size=129280, hidden_size=7168, multimodal=False,
        tokenizer_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
        n_source_shards=64, n_tensors=145116,
        resident_anchor_bpw=0.38, aggressive_anchor_bpw=0.30, extreme_probe_bpw=0.22,
        doctor_families=("expert_genome", "progressive_expert_slices", "error_propagation_firewall"),
        notes="1.6T extreme frontier; REAL config shows 6 selected experts (not 8) and native "
              "FP4/FP8 mixed source (~0.54 B/param). 1M context accounting deferred until stable."),
)


@dataclasses.dataclass(frozen=True)
class DeviceEnvelope:
    physical_ram_gb: float = 96.0
    os_control_reserve_gb: float = 8.0
    activation_workspace_reserve_gb: float = 6.0
    context_kv_reserve_gb: float = 6.0
    doctor_runtime_table_reserve_gb: float = 4.0

    @property
    def safe_model_resident_gb(self) -> float:
        return (self.physical_ram_gb - self.os_control_reserve_gb
                - self.activation_workspace_reserve_gb - self.context_kv_reserve_gb
                - self.doctor_runtime_table_reserve_gb)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["safe_model_resident_gb"] = self.safe_model_resident_gb
        return d


def default_envelope() -> DeviceEnvelope:
    return DeviceEnvelope()


def physical_fit(p: GiantParent, env: DeviceEnvelope | None = None) -> dict[str, Any]:
    """Exact physical-fit math (master goal section 4). Uses the OFFICIAL total-param
    denominator (never active params), and reports the resident ceiling plus each regime's
    feasibility at the parent's anchor rate. Artifact bytes must later include Doctor,
    pass-through, codebooks, exceptions, indices, alignment, and tables (billed at bake time)."""
    env = env or default_envelope()
    safe = env.safe_model_resident_gb
    safe_bytes = safe * 1e9

    def artifact_gb(bpw: float) -> float:
        return round(p.official_total_params * bpw / 8.0 / 1e9, 2)

    resident_ceiling_bpw = round(8.0 * safe_bytes / p.official_total_params, 4)

    # hybrid working set: shared trunk + routers + shared experts + active-expert working set
    # + a hot-expert cache. Approximate the permanently-resident trunk as the non-expert
    # fraction plus the active experts at the anchor rate.
    active_gb = round(p.active_params * p.resident_anchor_bpw / 8.0 / 1e9, 2)
    hot_cache_gb = round(active_gb * 1.5, 2)
    hybrid_resident_gb = round(active_gb + hot_cache_gb + env.context_kv_reserve_gb, 2)

    anchors = {
        "resident_anchor_bpw": p.resident_anchor_bpw,
        "aggressive_anchor_bpw": p.aggressive_anchor_bpw,
        "extreme_probe_bpw": p.extreme_probe_bpw,
    }
    fit = {}
    for name, bpw in anchors.items():
        agb = artifact_gb(bpw)
        fit[name] = {"bpw": bpw, "artifact_gb": agb, "fits_resident": agb <= safe}

    resident_ok = artifact_gb(p.resident_anchor_bpw) <= safe
    hybrid_ok = hybrid_resident_gb <= safe
    if resident_ok:
        regime = "RESIDENT_EXTREME"
    elif hybrid_ok:
        regime = "HYBRID_EXPERT_EXTREME"
    else:
        regime = "STREAMED_ARCHIVE_EXTREME"

    return {
        "device_envelope": env.as_dict(),
        "official_total_params": p.official_total_params,
        "active_params": p.active_params,
        "denominator": "official_total_params (never active params)",
        "resident_ceiling_bpw": resident_ceiling_bpw,
        "anchor_fit": fit,
        "hybrid": {"active_gb_at_anchor": active_gb, "hot_cache_gb": hot_cache_gb,
                   "hybrid_resident_gb": hybrid_resident_gb, "fits_hybrid": hybrid_ok},
        "selected_regime": regime,
        "source_bytes_gb": round(p.source_bytes / 1e9, 1),
        "disk_walled": True,
        "note": ("artifact_gb here is BASE only; the admitted artifact must also bill Doctor "
                 "corrections, pass-through, codebooks, exceptions, indices, alignment, tables, "
                 "and (for CORE+MTP / multimodal) those components before final fit."),
    }


def _parent_status(p: GiantParent) -> tuple[str, list[str], str]:
    """Honest status + blockers. Metadata is bound; the gates are adapter, synthetic twin,
    disk, release, and admission. Never 'ready' from a model ID."""
    blockers = [
        "adapter_pending: architecture adapter for %s not built/reviewed" % p.model_type,
        "synthetic_twin_pending: geometry twin not yet green",
        "waiting_disk: source %d GB >> free disk (~175 GB); remote bounded-stream Press required"
        % round(p.source_bytes / 1e9),
        "waiting_release: legacy campaign not released; heavy conversion gated",
        "waiting_predecessor: 72B calibration + 120B proof precede giant parents",
    ]
    if p.multimodal:
        blockers.append("claim_split_required: bind K2.6_TEXT_CORE vs K2.6_FULL_MULTIMODAL "
                        "(vision tower + projector billed separately)")
    return "adapter_pending", blockers, "build architecture adapter + green synthetic twin"


def build_row(p: GiantParent, env: DeviceEnvelope | None = None, *, generation: str = "gen-1",
              gravity: dict[str, Any] | None = None) -> dict[str, Any]:
    fit = physical_fit(p, env)
    status, blockers, next_transition = _parent_status(p)
    row = {
        "schema": "hawking.successor.frontier_row.v1",
        "queue_generation": generation,
        "row_id": p.row_id,
        "hf_or_source_id": p.hf_id,
        "exact_revision": p.revision,
        "license": "see_repo",  # license text not fetched; bind at procure time
        "architecture_family": p.architecture,
        "model_type": p.model_type,
        "tokenizer_files": list(p.tokenizer_files),
        "source_manifest": {"n_shards": p.n_source_shards, "n_tensors": p.n_tensors,
                            "source_bytes": p.source_bytes, "index": "model.safetensors.index.json",
                            "revision_pinned": True},
        "parameters": {"official_total": p.official_total_params, "active": p.active_params,
                       "mtp_layers": p.mtp_layers, "multimodal_vision": p.multimodal,
                       "exact_stored_count": "pending_tensor_shape_summation"},
        "architecture": {"num_layers": p.num_layers, "n_routed_experts": p.n_routed_experts,
                         "experts_per_tok": p.experts_per_tok, "n_shared_experts": p.n_shared_experts,
                         "moe_intermediate_size": p.moe_intermediate_size,
                         "kv_lora_rank": p.kv_lora_rank, "q_lora_rank": p.q_lora_rank,
                         "hidden_size": p.hidden_size, "vocab_size": p.vocab_size,
                         "attention": "MLA" if p.kv_lora_rank else "MHA/other",
                         "max_context": p.max_context},
        "source_precision": p.source_precision, "bytes_per_param": p.bytes_per_param(),
        "candidate_representation_families": ["strand_ladder_moe", "expert_genome_codec"],
        "candidate_doctor_families": list(p.doctor_families),
        "physical_fit": fit, "selected_regime": fit["selected_regime"],
        "resident_anchor_bpw": p.resident_anchor_bpw,
        "adapter_id": None, "adapter_implementation_state": "pending",
        "source_acquisition_policy": "remote_bounded_stream_press",
        "streamed_conversion_policy": "four_deterministic_passes",
        "evaluation_contract": "codec_fidelity+standalone_capability(text_core if multimodal)",
        "runtime_regime": fit["selected_regime"],
        "predecessor_dependencies": ["72B_calibration", "120B_proof"],
        "envelopes": {"ram_gb": fit["device_envelope"]["safe_model_resident_gb"],
                      "disk_free_gb": 175, "source_gb": round(p.source_bytes / 1e9, 1),
                      "network": "range/shard acquisition with revision-pinned ETags"},
        "current_status": status, "blockers": blockers, "next_transition": next_transition,
        "eta_state": "uncalibrated_no_giant_run", "telegram_state": "notify_on_state_change",
        "notes": p.notes, "created_at": now_iso(),
    }
    # Additive Gravity augmentation of the giant queue row (master goal section 16). Present
    # only when supplied, so an un-augmented frontier row is byte-identical to before.
    if gravity is not None:
        row["gravity"] = gravity
    return seal_field(row, "row_sha256")


def queue_rows(env: DeviceEnvelope | None = None, *, generation: str = "gen-1") -> list[dict[str, Any]]:
    """Map the 3 giant parents to durable succ_queue rows so the successor controller loads,
    validates, persists, displays, and ETA-projects them (a Markdown name is not a queue entry)."""
    import succ_queue
    rows = []
    for p in PARENTS:
        fit = physical_fit(p, env)
        status, blockers, next_transition = _parent_status(p)
        rows.append(succ_queue.make_row(
            queue_generation=generation, parent_label=p.row_id,
            hf_or_source_id=p.hf_id, exact_revision=p.revision,
            architecture_family=p.architecture,
            exact_stored_parameter_count=int(p.official_total_params),
            active_parameter_count=int(p.active_params),
            source_bytes=p.source_bytes, current_local_bytes=0,
            expected_output_rate_prior=p.resident_anchor_bpw,
            candidate_representation_families=["strand_ladder_moe", "expert_genome_codec"],
            candidate_doctor_families=list(p.doctor_families),
            required_predecessor_evidence=["72B_calibration", "120B_proof"],
            adapter_id=None, runtime_spec_status="pending", quality_path_status="deferred",
            streamed_lifecycle="remote_bounded_stream_press",
            disk_envelope=f"{round(p.source_bytes/1e9)} GB source >> ~175 GB free (disk-walled)",
            ram_envelope=fit["selected_regime"], resume_strategy="press-checkpoint-exact",
            current_status="waiting_adapter",
            blockers=blockers, exit_criteria=["architecture adapter built + reviewed",
                                             "synthetic geometry twin green",
                                             "release boundary + disk plan for source windows"],
            next_transition=next_transition))
    return rows


def frontier_manifest(env: DeviceEnvelope | None = None) -> dict[str, Any]:
    rows = [build_row(p, env) for p in PARENTS]
    manifest = {
        "schema": FRONTIER_SCHEMA,
        "generated_at": now_iso(),
        "device_envelope": (env or default_envelope()).as_dict(),
        "parents": rows,
        "row_ids": [p.row_id for p in PARENTS],
        "regimes_selected": {p.row_id: r["selected_regime"] for p, r in zip(PARENTS, rows)},
        "heavy_execution_order": ["72B_calibration", "120B_proof", "deepseek-v3.2-685b",
                                  "kimi-k2.6-1t", "deepseek-v4-pro-1.6t"],
    }
    return seal_field(manifest, "manifest_sha256")


def selftest() -> dict[str, Any]:
    man = frontier_manifest()
    if not sealed(man, "manifest_sha256"):
        raise EcoError("manifest not sealed")
    if man["row_ids"] != ["deepseek-v3.2-685b", "kimi-k2.6-1t", "deepseek-v4-pro-1.6t"]:
        raise EcoError(f"row ids wrong: {man['row_ids']}")
    for row in man["parents"]:
        if not sealed(row, "row_sha256"):
            raise EcoError(f"row {row['row_id']} not sealed")
        if row["current_status"] not in FRONTIER_STATUSES:
            raise EcoError(f"invalid status {row['current_status']}")
        if row["adapter_id"] is not None:
            raise EcoError("adapter must not be claimed present")
        if not row["blockers"]:
            raise EcoError("every giant row must carry honest blockers")
        # denominator must be official total, never active
        if "official_total_params" not in row["physical_fit"]:
            raise EcoError("fit must use official total denominator")
    # regime sanity: 685B @0.80 = 68.5 GB fits 72 GB envelope -> RESIDENT
    ds32 = next(r for r in man["parents"] if r["row_id"] == "deepseek-v3.2-685b")
    v4 = next(r for r in man["parents"] if r["row_id"] == "deepseek-v4-pro-1.6t")
    # V4-Pro @0.38 = 76 GB > 72 GB safe -> not resident at anchor -> hybrid/streamed
    return {"ok": True, "parents": 3,
            "regimes": man["regimes_selected"],
            "v3.2_bytes_per_param": ds32["bytes_per_param"],
            "v4pro_experts_per_tok": v4["architecture"]["experts_per_tok"],
            "manifest_sha256": man["manifest_sha256"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="High-Parameter Frontier source authority + fit.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    man = frontier_manifest()
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, man)
    print(json.dumps({"schema": man["schema"], "manifest_sha256": man["manifest_sha256"],
                      "row_ids": man["row_ids"], "regimes_selected": man["regimes_selected"],
                      "parents": [{"row_id": r["row_id"], "status": r["current_status"],
                                   "regime": r["selected_regime"],
                                   "resident_ceiling_bpw": r["physical_fit"]["resident_ceiling_bpw"],
                                   "anchor_fit": r["physical_fit"]["anchor_fit"]["resident_anchor_bpw"],
                                   "blockers": len(r["blockers"])} for r in man["parents"]]},
                     indent=2, sort_keys=True))
