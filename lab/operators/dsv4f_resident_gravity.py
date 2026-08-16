#!/usr/bin/env python3
"""DSV4F resident-gravity feasibility: census, complete-physical, working set.

Read-only. Re-walks the sealed manifest via ``dsv4f_tensor_schedule`` (no chunk
bodies, no forward, no runtime edits). Writes a compact receipt, never a giant
JSON index.

Claim boundary: this module bills bytes. It does not fit a codec, pack an
artifact, or run generation. A complete-physical number at 1.5 BPW is a
target footprint, not evidence that 1.5 BPW is coherent.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.operators.dsv4f_tensor_schedule import (
    BASE_LAYER_COUNT,
    COMPRESS_RATIOS,
    EXPECTED_CHUNK_SHA256,
    EXPECTED_MANIFEST_SEAL_PREFIX,
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_TENSOR_BYTES,
    HASH_LAYER_COUNT,
    HEAD_DIM,
    HIDDEN,
    MOE_INTER,
    ORGAN_CLASSES,
    PINNED_REPOSITORY,
    PINNED_REVISION,
    ROUTED_EXPERTS,
    TOP_K,
    VOCAB,
    classify_all,
    classify_tensor,
    extract_manifest_identity,
    iter_manifest_tensors,
    resolve_artifact_root,
)

SCHEMA = "hawking.dsv4f.resident_gravity.feasibility.v1"
LANE = "dsv-resident-gravity"
HOST_RAM_BYTES = 96 * 1024**3
ARTIFACT_INDEX_BYTES = 26_661_704  # receipts/dsv4f_artifact_index.json
PACKAGING_ALIGN_PAD = 4 * 1024**2
Q80_MIXED_EXPERT_BPW = 1.22957
Q80_W1_BPW = 1.1269  # binary_group, gate_proj analog
Q80_W3_BPW = 1.2918  # binary + rice_q1 @ 2% outliers, up_proj analog
Q80_W2_BPW = 1.27  # hgravs01_r160_b3 on post-SwiGLU X, down_proj analog
Q80_COMPLETE_8BIT_NONEXPERT = 1.43051
SLIDING_WINDOW = 128
INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
KV_ELEM_BF16 = 2
EXISTING_INDEX_BYTES = ARTIFACT_INDEX_BYTES

# Official inference/model.py Attention / Indexer / Compressor (pinned source).
# hc_pre reduces hc_mult copies to 1 before attention, so KV is not x4.
RATIO4_LAYERS = tuple(i for i, r in enumerate(COMPRESS_RATIOS[:BASE_LAYER_COUNT]) if r == 4)
RATIO128_LAYERS = tuple(i for i, r in enumerate(COMPRESS_RATIOS[:BASE_LAYER_COUNT]) if r == 128)
SLIDING_LAYERS = tuple(i for i, r in enumerate(COMPRESS_RATIOS[:BASE_LAYER_COUNT]) if r == 0)

# Warm streamed body from receipts/DSV4F_TOKEN_NS_LEDGER.json (other-lane dirty).
STREAMED_BODY_MS = (3275, 3323, 3425)


def gib(n_bytes: float) -> float:
    return float(n_bytes) / float(1024**3)


def complete_physical_bytes(logical_params: int, bpw: float) -> int:
    """Complete-physical payload for a rate billed against logical params."""

    if logical_params < 0:
        raise ValueError("logical_params must be >= 0")
    if bpw < 0:
        raise ValueError("bpw must be >= 0")
    return int(math.ceil(logical_params * bpw / 8.0))


def kv_slots_per_layer(seq_len: int, ratio: int, window: int = SLIDING_WINDOW) -> int:
    """Official ``kv_cache_size = window + (max_seq_len // ratio if ratio else 0)``."""

    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    if ratio < 0:
        raise ValueError("ratio must be >= 0")
    if ratio == 0:
        return min(seq_len, window) if seq_len else window
    return window + (seq_len // ratio)


def indexer_slots(seq_len: int, ratio: int = 4) -> int:
    """Official Indexer cache: ``max_seq_len // compress_ratio``."""

    if ratio <= 0:
        return 0
    return seq_len // ratio


def working_set_bytes(seq_len: int, *, elem_bytes: int = KV_ELEM_BF16) -> dict[str, int]:
    """MLA + indexer + compressor-state bytes for one decode batch.

    Attention KV is one 512-wide latent per slot (num_key_value_heads=1).
    Indexer KV is 128-wide, ratio-4 layers only. Compressor state does not
    grow with sequence length.
    """

    attn_slots = 0
    for layer in range(BASE_LAYER_COUNT):
        attn_slots += kv_slots_per_layer(seq_len, COMPRESS_RATIOS[layer])
    attn_kv = attn_slots * HEAD_DIM * elem_bytes
    index_kv = len(RATIO4_LAYERS) * indexer_slots(seq_len, 4) * INDEX_HEAD_DIM * elem_bytes

    # Compressor decode state is F32. coff = 1 + (ratio == 4).
    # Attention compressor: ratio-4 coff=2, (8 x 1024) x2; ratio-128 coff=1, (128 x 512) x2.
    # Indexer compressor: ratio-4 coff=2, head_dim=128, (8 x 256) x2.
    f32 = 4
    attn_r4_state = len(RATIO4_LAYERS) * (2 * 4) * (2 * HEAD_DIM) * f32 * 2
    attn_r128_state = len(RATIO128_LAYERS) * 128 * HEAD_DIM * f32 * 2
    idx_state = len(RATIO4_LAYERS) * (2 * 4) * (2 * INDEX_HEAD_DIM) * f32 * 2
    compressor_state = attn_r4_state + attn_r128_state + idx_state

    # Decode activations: hc residual [4, 4096] bf16 + a few 4096/2048 scratch rows.
    hc_residual = 4 * HIDDEN * 2
    scratch_rows = 16 * HIDDEN * 4  # f32 working rows, generous
    activations = hc_residual + scratch_rows

    return {
        "seq_len": seq_len,
        "elem_bytes": elem_bytes,
        "attention_kv_bytes": attn_kv,
        "indexer_kv_bytes": index_kv,
        "compressor_state_bytes": compressor_state,
        "decode_activation_bytes": activations,
        "total_kv_plus_index_bytes": attn_kv + index_kv,
        "total_working_bytes": attn_kv + index_kv + compressor_state + activations,
    }


def host_reserve_bytes() -> dict[str, int]:
    """Exclusive-use CLEAN envelope. Wired was measured ~5.4 GiB on this host."""

    os_wired = 8 * 1024**3
    metal_runtime = 2 * 1024**3
    decode_scratch = 1 * 1024**3
    reserved = os_wired + metal_runtime + decode_scratch
    return {
        "host_ram_bytes": HOST_RAM_BYTES,
        "os_wired_budget_bytes": os_wired,
        "metal_runtime_budget_bytes": metal_runtime,
        "decode_scratch_budget_bytes": decode_scratch,
        "reserved_bytes": reserved,
        "available_for_model_and_kv_bytes": HOST_RAM_BYTES - reserved,
    }


def max_routed_bpw(*, f_routed: float, protect_bpw: float, target: float) -> float:
    """Largest routed complete-physical BPW that still holds ``target``."""

    if not 0.0 < f_routed < 1.0:
        raise ValueError("f_routed must be in (0, 1)")
    return (target - (1.0 - f_routed) * protect_bpw) / f_routed


def complete_bpw(f_routed: float, routed_bpw: float, protect_bpw: float) -> float:
    return f_routed * routed_bpw + (1.0 - f_routed) * protect_bpw


def packaging_bytes() -> int:
    return EXISTING_INDEX_BYTES + PACKAGING_ALIGN_PAD


def organ_rows(classified: dict[str, Any]) -> dict[str, dict[str, int]]:
    organs = classified["organs"]
    return {
        name: {
            "tensor_count": organs[name].tensor_count,
            "byte_mass": organs[name].byte_mass,
            "logical_params": organs[name].logical_params,
            "stored_elements": organs[name].stored_elements,
            "weight_bytes": organs[name].weight_bytes,
            "scale_bytes": organs[name].scale_bytes,
        }
        for name in ORGAN_CLASSES
    }


def scope_logical(manifest: Path) -> dict[str, int]:
    out = {"base": 0, "mtp": 0, "global": 0}
    for row in iter_manifest_tensors(manifest):
        cls = classify_tensor(row)
        if cls.layer is None:
            scope = "global"
        elif cls.layer >= BASE_LAYER_COUNT:
            scope = "mtp"
        else:
            scope = "base"
        out[scope] += cls.logical_params
    return out


def q80_rate_hypothesis_bytes(routed_logical: int) -> dict[str, Any]:
    """Transfer Q80 per-component *rates* as an envelope. Not a DSV4F fit."""

    third = routed_logical // 3
    if third * 3 != routed_logical:
        raise ValueError("routed logical mass is not 3-way equal; mix bill would lie")
    w1 = complete_physical_bytes(third, Q80_W1_BPW)
    w3 = complete_physical_bytes(third, Q80_W3_BPW)
    w2 = complete_physical_bytes(third, Q80_W2_BPW)
    total = w1 + w3 + w2
    return {
        "status": "HYPOTHESIS_UNFITTED",
        "claim_boundary": (
            "Q80 organ rates transferred as an arithmetic envelope only. "
            "DSV4F experts are native FP4, not BF16. No activation-weighted "
            "score, no packed artifact, no generation."
        ),
        "w1_gate_analog_bpw": Q80_W1_BPW,
        "w3_up_analog_bpw": Q80_W3_BPW,
        "w2_down_analog_bpw": Q80_W2_BPW,
        "mixed_expert_bpw": Q80_MIXED_EXPERT_BPW,
        "per_proj_logical": third,
        "w1_bytes": w1,
        "w3_bytes": w3,
        "w2_bytes": w2,
        "routed_complete_bytes": total,
        "routed_complete_bpw_from_bytes": (8.0 * total) / routed_logical,
    }


def capture_honesty() -> dict[str, Any]:
    """Existing X is underdetermined. Do not score a fit against it."""

    w1_dim = HIDDEN
    w2_dim = MOE_INTER
    return {
        "existing": {
            "fullseq_L0_tokens": 255,
            "fullseq_L1_tokens": 1020,
            "fullseq_total_tokens": 1275,
            "fullseq_sequences": 160,
            "late_hidden_export": "32 x 4096 per layer (one vector per L0 sequence)",
            "activation_x_batch_parity_sample_rows": 1,
            "post_swiglu_x_captured": False,
            "per_expert_first_n_real_source": False,
        },
        "writer_defaults_are_underdetermined": {
            "default_max_hidden_tokens_per_expert": 64,
            "default_row_threshold": 16,
            "w1_w3_input_dim": w1_dim,
            "w2_input_dim": w2_dim,
            "rows_64_vs_w1_dim": "UNDERDETERMINED",
            "rows_16_vs_w1_dim": "UNDERDETERMINED",
        },
        "determined_fit_floor": {
            "rule": "rows < input_dim is underdetermined for a full-rank score; rows < rank is underdetermined for a rank-r score",
            "w1_w3_dim": w1_dim,
            "w2_dim": w2_dim,
            "recommended_rows_for_rank_256": 512,
            "do_not_cap_rank_to_n_fit_rows": True,
        },
        "layer_tiled_capture_bytes_rank256": {
            "rows": 512,
            "experts": ROUTED_EXPERTS,
            "w1_x_bytes_one_layer": 512 * ROUTED_EXPERTS * w1_dim * 4,
            "w2_x_bytes_one_layer": 512 * ROUTED_EXPERTS * w2_dim * 4,
            "note": "Keep one layer, fit, discard X. Never emit a giant JSON index.",
        },
        "tokens_to_fill_512_rows_if_uniform": {
            "formula": "experts * rows / top_k",
            "tokens_per_layer": int(math.ceil(ROUTED_EXPERTS * 512 / TOP_K)),
            "reuse_across_layers": (
                "The same sequences feed every layer; 21,846 tokens is the "
                "uniform-routing fill, not 43 independent corpora."
            ),
            "hash_layers_0_2": (
                "Layers 0-2 route by token id (tid2eid). Expert coverage is "
                "whatever the corpus token-id histogram hits. Report holes; "
                "do not fabricate rows."
            ),
        },
        "teacher": (
            "Official mixed-precision source (FP4 experts, FP8 control, BF16 "
            "embed/head). There is no local full-BF16 DSV4F. Fit against this "
            "teacher, not a degraded pack, not Gaussian X."
        ),
    }


def build_feasibility(classified: dict[str, Any], identity: dict[str, Any], scopes: dict[str, int]) -> dict[str, Any]:
    organs = organ_rows(classified)
    total_bytes = classified["byte_sum"]
    total_logical = sum(row["logical_params"] for row in organs.values())
    routed = organs["routed_expert"]
    shared = organs["shared_expert"]
    routed_logical = routed["logical_params"]
    shared_logical = shared["logical_params"]
    protect_organs = [name for name in ORGAN_CLASSES if name != "routed_expert"]
    protect_logical = sum(organs[name]["logical_params"] for name in protect_organs)
    protect_bytes = sum(organs[name]["byte_mass"] for name in protect_organs)
    f_routed = routed_logical / total_logical
    f_protect = 1.0 - f_routed
    f_routed_plus_shared = (routed_logical + shared_logical) / total_logical
    source_routed_bpw = (8.0 * routed["byte_mass"]) / routed_logical
    source_protect_bpw = (8.0 * protect_bytes) / protect_logical
    source_complete_bpw = (8.0 * total_bytes) / total_logical
    pack = packaging_bytes()

    targets = (1.5, 1.4, 1.3, 1.0)
    naive_targets = {}
    for bpw in targets:
        payload = complete_physical_bytes(total_logical, bpw)
        naive_targets[f"{bpw:.1f}"] = {
            "complete_bpw": bpw,
            "payload_bytes": payload,
            "payload_gib": gib(payload),
            "with_packaging_bytes": payload + pack,
            "with_packaging_gib": gib(payload + pack),
        }

    q80_hyp = q80_rate_hypothesis_bytes(routed_logical)
    mixed_protect_source = {
        "routed_bpw": Q80_MIXED_EXPERT_BPW,
        "protect_policy": "source_precision",
        "protect_bpw": source_protect_bpw,
        "complete_bpw": complete_bpw(f_routed, Q80_MIXED_EXPERT_BPW, source_protect_bpw),
    }
    mixed_protect_8bit = {
        "routed_bpw": Q80_MIXED_EXPERT_BPW,
        "protect_policy": "uniform_8bit",
        "protect_bpw": 8.0,
        "complete_bpw": complete_bpw(f_routed, Q80_MIXED_EXPERT_BPW, 8.0),
    }
    for row in (mixed_protect_source, mixed_protect_8bit):
        routed_bytes = complete_physical_bytes(routed_logical, row["routed_bpw"])
        prot_bytes = complete_physical_bytes(protect_logical, row["protect_bpw"])
        total = routed_bytes + prot_bytes + pack
        row["routed_bytes"] = routed_bytes
        row["protect_bytes"] = prot_bytes
        row["packaging_bytes"] = pack
        row["complete_physical_bytes"] = total
        row["complete_physical_gib"] = gib(total)
        row["complete_physical_bpw_from_bytes"] = (8.0 * total) / total_logical
        row["clears_1_5"] = row["complete_physical_bpw_from_bytes"] <= 1.5 + 1e-12

    envelope = []
    for routed_bpw in (1.0, 1.1269, 1.22957, 1.3, 1.3292, 1.4, 1.4609):
        for protect_name, protect_bpw in (
            ("source_precision", source_protect_bpw),
            ("uniform_8bit", 8.0),
        ):
            cbpw = complete_bpw(f_routed, routed_bpw, protect_bpw)
            envelope.append(
                {
                    "routed_bpw": routed_bpw,
                    "protect_policy": protect_name,
                    "protect_bpw": protect_bpw,
                    "complete_bpw": cbpw,
                    "clears_1_5": cbpw <= 1.5 + 1e-12,
                }
            )

    contexts = (4096, 32768, 131072, 1_048_576)
    working = {str(s): working_set_bytes(s) for s in contexts}
    reserve = host_reserve_bytes()

    def residency_row(model_bytes: int, seq_len: int) -> dict[str, Any]:
        ws = working[str(seq_len)]
        used = model_bytes + ws["total_working_bytes"]
        avail = reserve["available_for_model_and_kv_bytes"]
        return {
            "seq_len": seq_len,
            "model_bytes": model_bytes,
            "working_bytes": ws["total_working_bytes"],
            "model_plus_working_bytes": used,
            "model_plus_working_gib": gib(used),
            "available_gib": gib(avail),
            "margin_bytes": avail - used,
            "margin_gib": gib(avail - used),
            "fits_clean_exclusive": used <= avail,
            "fits_raw_96gib": used <= HOST_RAM_BYTES,
        }

    model_1_5 = naive_targets["1.5"]["with_packaging_bytes"]
    model_source = total_bytes + pack
    model_mixed_source = mixed_protect_source["complete_physical_bytes"]
    residency = {
        "at_1_5_uniform": {str(s): residency_row(model_1_5, s) for s in contexts},
        "at_q80_rate_protect_source": {
            str(s): residency_row(model_mixed_source, s) for s in contexts
        },
        "at_source_precision": {str(s): residency_row(model_source, s) for s in contexts},
    }

    active_one_layer_moe = TOP_K * 3 * MOE_INTER * HIDDEN + 3 * MOE_INTER * HIDDEN
    active_mla = (
        1024 * HIDDEN
        + 64 * HEAD_DIM * 1024
        + HEAD_DIM * HIDDEN
        + (64 * HEAD_DIM // 8) * (8 * 1024)
        + HIDDEN * (8 * 1024)
    )
    active_params = (
        BASE_LAYER_COUNT * (active_one_layer_moe + active_mla)
        + organs["embeddings"]["logical_params"]  # lookup, not matmul-all
        + organs["lm_head"]["logical_params"]
    )

    source_fits = any(
        residency["at_source_precision"][str(s)]["fits_clean_exclusive"] for s in contexts
    )
    target_fits = all(
        residency["at_1_5_uniform"][str(s)]["fits_clean_exclusive"] for s in contexts
    )

    verdict = {
        "arithmetic": "FEASIBLE" if target_fits and not source_fits else "INFEASIBLE",
        "quality": "UNPROVEN",
        "generation": "NOT_RUN",
        "plain": (
            "A <=1.5 complete-physical DSV4F *would* sit in the 96 GiB envelope "
            "with working room, because 97.43% of logical mass is routed experts "
            "and MLA KV is a 512-wide latent. Source precision (148.7 GiB) does "
            "not. Whether any codec actually reaches 1.5 complete *and* coherent "
            "generation is unproven: experts are already FP4, existing X is "
            "underdetermined, and organ cosine is not the gate."
        ),
        "do_not_declare_success": True,
        "source_precision_resident": False,
        "target_1_5_resident_if_rate_achieved": bool(target_fits),
    }

    return {
        "schema": SCHEMA,
        "lane": LANE,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ARITHMETIC_FEASIBLE_QUALITY_UNPROVEN",
        "claim_boundary": {
            "artifact_packed": False,
            "activation_weighted_fit": False,
            "coherence_generation_tested": False,
            "runtime_modified": False,
            "note": (
                "Census and residency arithmetic only. Q80 codec rates are an "
                "unfitted envelope. Model-level gate is coherent generation."
            ),
        },
        "identity": {
            "repository": PINNED_REPOSITORY,
            "revision": PINNED_REVISION,
            "manifest_seal_sha256": identity.get("seal_sha256"),
            "seal_matches_prefix": str(identity.get("seal_sha256", "")).startswith(
                EXPECTED_MANIFEST_SEAL_PREFIX
            ),
            "content_addressed_chunk_sha256": identity.get("content_addressed_chunk_sha256"),
            "chunk_sha_matches": identity.get("content_addressed_chunk_sha256")
            == EXPECTED_CHUNK_SHA256,
            "artifact_root": identity.get("artifact_root")
            or identity.get("path"),
            "schema": identity.get("schema"),
            "chunks_opened": 0,
        },
        "coverage": {
            "tensor_count": classified["count"],
            "expected_tensor_count": EXPECTED_TENSOR_COUNT,
            "byte_mass": total_bytes,
            "expected_byte_mass": EXPECTED_TOTAL_TENSOR_BYTES,
            "byte_residual": total_bytes - EXPECTED_TOTAL_TENSOR_BYTES,
            "logical_params": total_logical,
            "scope_logical": scopes,
            "shape_mismatch": classified["shape_mismatch"],
            "undetermined": classified["undetermined"],
            "covers_all_tensors": (
                classified["count"] == EXPECTED_TENSOR_COUNT
                and total_bytes == EXPECTED_TOTAL_TENSOR_BYTES
                and classified["shape_mismatch"] == 0
                and not classified["undetermined"]
            ),
        },
        "geometry": {
            "base_layers": BASE_LAYER_COUNT,
            "mtp_layers": 1,
            "hidden": HIDDEN,
            "moe_intermediate": MOE_INTER,
            "routed_experts": ROUTED_EXPERTS,
            "shared_experts": 1,
            "top_k": TOP_K,
            "vocab": VOCAB,
            "head_dim": HEAD_DIM,
            "n_heads": 64,
            "num_key_value_heads": 1,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "rope_head_dim": ROPE_HEAD_DIM,
            "sliding_window": SLIDING_WINDOW,
            "index_n_heads": 64,
            "index_head_dim": INDEX_HEAD_DIM,
            "index_topk": 512,
            "hc_mult": 4,
            "hash_layers": HASH_LAYER_COUNT,
            "max_position_embeddings": 1_048_576,
            "source_dtypes": "fp4 e2m1fn_x2 experts + ue8m0/32; fp8 e4m3 control; bf16 embed/head; f32 mHC",
            "ratio4_layers": list(RATIO4_LAYERS),
            "ratio128_layers": list(RATIO128_LAYERS),
            "sliding_layers": list(SLIDING_LAYERS),
        },
        "organs": organs,
        "mass_split": {
            "q80_analogous": {
                "crush": "routed_expert",
                "protect": protect_organs,
                "f_routed": f_routed,
                "f_protect": f_protect,
                "routed_logical": routed_logical,
                "protect_logical": protect_logical,
                "routed_bytes": routed["byte_mass"],
                "protect_bytes": protect_bytes,
                "source_routed_bpw": source_routed_bpw,
                "source_protect_bpw": source_protect_bpw,
                "why": (
                    "Q80 crushed routed experts and left shared + attention + "
                    "embed + lm_head at 8-bit. Shared fires every token."
                ),
            },
            "schedule_routed_plus_shared": {
                "f_expert": f_routed_plus_shared,
                "note": (
                    "DSV4F_TENSOR_SCHEDULE lumped shared with routed. This lane "
                    "does not: shared is protected."
                ),
            },
            "official_card": {
                "total_params_claimed": 284_000_000_000,
                "active_params_claimed": 13_000_000_000,
                "measured_routed_logical": routed_logical,
                "measured_all_logical": total_logical,
                "measured_base_logical": scopes["base"],
                "measured_mtp_logical": scopes["mtp"],
                "measured_global_logical": scopes["global"],
                "active_params_geometry_lower_bound": active_params,
                "active_note": (
                    "Lower bound is 43*(6+1)*3*2048*4096 + 43*MLA + embed + head. "
                    "Matches the 13B card to ~0.1B; embed is a lookup."
                ),
            },
        },
        "source_complete": {
            "bytes": total_bytes,
            "gib": gib(total_bytes),
            "bpw_vs_logical": source_complete_bpw,
            "routed_already_fp4_bpw": source_routed_bpw,
            "second_packing": True,
            "second_packing_note": (
                "Q80 1.23 BPW was BF16→low-bit. DSV4F routed source is already "
                "4.25 complete BPW (4-bit codes + UE8M0/32). 1.23 is a second "
                "collapse of a 16-level discrete matrix."
            ),
        },
        "complete_physical_targets": naive_targets,
        "packaging": {
            "compact_index_bytes": EXISTING_INDEX_BYTES,
            "align_pad_bytes": PACKAGING_ALIGN_PAD,
            "total_bytes": pack,
            "refused_144mb_json_manifest": True,
        },
        "q80_rate_hypothesis": q80_hyp,
        "mixed_policy": {
            "protect_at_source": mixed_protect_source,
            "protect_at_8bit": mixed_protect_8bit,
            "max_routed_bpw_protect_source_for_1_5": max_routed_bpw(
                f_routed=f_routed, protect_bpw=source_protect_bpw, target=1.5
            ),
            "max_routed_bpw_protect_8bit_for_1_5": max_routed_bpw(
                f_routed=f_routed, protect_bpw=8.0, target=1.5
            ),
            "envelope": envelope,
        },
        "working_set": working,
        "host_reserve": reserve,
        "residency": residency,
        "capture": capture_honesty(),
        "codec_proposal": {
            "status": "HYPOTHESIS_NOT_A_PRESCRIPTION",
            "routed_w1": "binary_group family (Q80 gate analog) — UNFITTED",
            "routed_w3": "binary + rice_q1 residual (Q80 up analog) — UNFITTED",
            "routed_w2": (
                "activation-weighted low-rank family against post-SwiGLU X, "
                "not the layer hidden (Q80 down analog) — UNFITTED"
            ),
            "shared_expert": "protect at source FP8 (always-on)",
            "mla_indexer_embed_head_router_norms_mhc": "protect at source",
            "forbidden": [
                "shared cross-expert basis (Q80 REFUTED, cos 0.004)",
                "single codec family across w1/w3/w2 (Q80 INSUFFICIENT; down inverts)",
                "fit w2 against hidden instead of post-SwiGLU X",
                "Q30 static <=1.5 approach (FAILED)",
                "Gaussian or degraded-pack activations",
                "rank = min(budget, n_fit_rows) caps on underdetermined X",
            ],
        },
        "streamed_baseline": {
            "label": "DIRTY_ENGINEERING",
            "source": "receipts/DSV4F_TOKEN_NS_LEDGER.json baseline_warm.reps_body_ms",
            "reps_body_ms": list(STREAMED_BODY_MS),
            "ns_per_token": [int(ms * 1_000_000) for ms in STREAMED_BODY_MS],
            "median_ns_per_token": int(sorted(STREAMED_BODY_MS)[1] * 1_000_000),
            "this_lane_did_not_retime": True,
        },
        "verdict": verdict,
        "next_bottleneck": {
            "name": "determined_teacher_x_capture_then_fit",
            "why": (
                "Bytes fit if 1.5 complete is achieved. Existing X is underdetermined. "
                "A 21846-token uniform fill is the next paid experiment."
            ),
            "estimated_streamed_forward_ns": int(21846 * 3_323_000_000),
            "estimated_from": "21846 tokens * 3323 ms ledger median, DIRTY_ENGINEERING",
            "post_resident_runtime_wall": {
                "name": "metal.gpu_act_quant",
                "ns": 2_082_931_000,
                "label": "DIRTY_ENGINEERING",
                "source": "receipts/DSV4F_TOKEN_NS_LEDGER.json metal.gpu",
            },
        },
        "negative_science_honored": [
            "Q80 cross-expert shared-basis REFUTED",
            "Q80 simply-bandwidth-bound REFUTED (not used as a premise)",
            "DSV4F route-ID readback serializer REFUTED",
            "shader compile as primary wall REFUTED / deprioritized",
            "single-family Q80 representation INSUFFICIENT",
            "Q30 static <=1.5 FAILED — not copied",
            "no giant JSON index",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    organs = report["organs"]
    split = report["mass_split"]["q80_analogous"]
    src = report["source_complete"]
    targets = report["complete_physical_targets"]
    mixed_s = report["mixed_policy"]["protect_at_source"]
    mixed_8 = report["mixed_policy"]["protect_at_8bit"]
    reserve = report["host_reserve"]
    v = report["verdict"]
    cap = report["capture"]
    lines = [
        "# DSV4F resident gravity — feasibility verdict",
        "",
        f"Lane `{report['lane']}`. Status **{report['status']}**.",
        "",
        v["plain"],
        "",
        "## Claim boundary",
        "",
        "- No packed artifact.",
        "- No activation-weighted DSV4F fit.",
        "- No coherent-generation test.",
        "- Runtime not modified.",
        "- Q80 codec rates are an unfitted envelope.",
        "- Model-level gate is coherent generation. Organ estimates are not that gate.",
        "",
        "## Geometry (live manifest re-walk)",
        "",
        f"- Artifact seal prefix match: `{report['identity']['seal_matches_prefix']}`",
        f"- Tensors: {report['coverage']['tensor_count']} / {report['coverage']['expected_tensor_count']}",
        f"- Byte residual: {report['coverage']['byte_residual']}",
        f"- Logical unpacked params: {report['coverage']['logical_params']}",
        f"- Scope logical: base={report['coverage']['scope_logical']['base']} "
        f"mtp={report['coverage']['scope_logical']['mtp']} "
        f"global={report['coverage']['scope_logical']['global']}",
        f"- Source bytes: {src['bytes']} ({src['gib']:.3f} GiB) at {src['bpw_vs_logical']:.6f} complete BPW",
        f"- Official card 284B/13B vs measured logical {report['coverage']['logical_params']} "
        f"(routed {report['organs']['routed_expert']['logical_params']}, "
        f"MTP {report['coverage']['scope_logical']['mtp']}, "
        f"active geometry lower bound {report['mass_split']['official_card']['active_params_geometry_lower_bound']}). "
        "The 284B card is the routed+shared neighborhood; this lane bills every unpacked logical param.",
        "",
        "| organ | tensors | bytes | logical params | % params | source BPW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_logical = report["coverage"]["logical_params"]
    for name in ORGAN_CLASSES:
        row = organs[name]
        bpw = (8.0 * row["byte_mass"] / row["logical_params"]) if row["logical_params"] else float("nan")
        pct = 100.0 * row["logical_params"] / total_logical
        bpw_s = f"{bpw:.4f}" if row["logical_params"] else "—"
        lines.append(
            f"| {name} | {row['tensor_count']} | {row['byte_mass']} | "
            f"{row['logical_params']} | {pct:.4f} | {bpw_s} |"
        )
    lines.extend(
        [
            "",
            "## What has to be non-expert",
            "",
            "Q80 crushed **routed** experts and left shared + attention + embeddings "
            "+ lm_head at 8-bit. The same structure holds, more extremely:",
            "",
            f"- f_routed = **{split['f_routed']:.9f}** ({split['routed_logical']} params, 97.431%)",
            f"- f_protect = **{split['f_protect']:.9f}** ({split['protect_logical']} params)",
            f"- source routed BPW (already FP4+UE8M0) = **{split['source_routed_bpw']:.6f}**",
            f"- source protect BPW = **{split['source_protect_bpw']:.6f}**",
            "",
            "Shared expert is **protected** (always-on, 0.381% of params). The earlier "
            "tensor-schedule envelope that lumped shared with routed is the wrong split "
            "for a Q80-analogous policy.",
            "",
            "## Complete-physical footprints",
            "",
            "Complete physical = codes + scales + codebooks + rank factors + indices + "
            f"padding + compact catalog ({report['packaging']['total_bytes']} bytes). "
            "Uniform rate against all logical params:",
            "",
            "| target BPW | payload GiB | +packaging GiB |",
            "|---:|---:|---:|",
        ]
    )
    for key in ("1.5", "1.4", "1.3", "1.0"):
        row = targets[key]
        lines.append(
            f"| {row['complete_bpw']:.1f} | {row['payload_gib']:.3f} | {row['with_packaging_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Mixed policy (Q80 rates as envelope only)",
            "",
            f"- Routed at {Q80_MIXED_EXPERT_BPW} (w1={Q80_W1_BPW}, w3={Q80_W3_BPW}, w2={Q80_W2_BPW}), protect at source: "
            f"**{mixed_s['complete_physical_bpw_from_bytes']:.6f} BPW, {mixed_s['complete_physical_gib']:.3f} GiB**, "
            f"clears 1.5 = {mixed_s['clears_1_5']}",
            f"- Same routed rate, protect at 8-bit: "
            f"**{mixed_8['complete_physical_bpw_from_bytes']:.6f} BPW, {mixed_8['complete_physical_gib']:.3f} GiB**, "
            f"clears 1.5 = {mixed_8['clears_1_5']}",
            f"- Max routed BPW with protect-at-source to hold 1.5: "
            f"**{report['mixed_policy']['max_routed_bpw_protect_source_for_1_5']:.6f}**",
            f"- Max routed BPW with protect-at-8-bit to hold 1.5: "
            f"**{report['mixed_policy']['max_routed_bpw_protect_8bit_for_1_5']:.6f}**",
            "",
            "These are not DSV4F scores. Q80's own 1.43051 was a screen "
            "(organ cosine), not a packed coherent artifact.",
            "",
            "## Residency arithmetic (96 GiB M3 Ultra)",
            "",
            f"- Host RAM: {gib(reserve['host_ram_bytes']):.0f} GiB",
            f"- CLEAN exclusive reserve: {gib(reserve['reserved_bytes']):.1f} GiB "
            f"(8 OS + 2 Metal + 1 scratch)",
            f"- Available for model + KV: {gib(reserve['available_for_model_and_kv_bytes']):.1f} GiB",
            "",
            "MLA KV is one 512-wide BF16 latent per slot (`num_key_value_heads=1`). "
            "`hc_pre` reduces 4 streams to 1 before attention, so KV is not ×4. "
            "Ratio-4 layers add a 128-wide indexer cache.",
            "",
        ]
    )
    for label, block in (
        ("uniform 1.5 complete + packaging", report["residency"]["at_1_5_uniform"]),
        ("Q80-rate routed + protect-at-source (unfitted)", report["residency"]["at_q80_rate_protect_source"]),
        ("source precision (what is on disk today)", report["residency"]["at_source_precision"]),
    ):
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| ctx | model+working GiB | margin GiB | fits CLEAN |")
        lines.append("|---:|---:|---:|---|")
        for key in ("4096", "32768", "131072", "1048576"):
            row = block[key]
            lines.append(
                f"| {row['seq_len']} | {row['model_plus_working_gib']:.3f} | "
                f"{row['margin_gib']:.3f} | {row['fits_clean_exclusive']} |"
            )
        lines.append("")
    ws1m = report["working_set"]["1048576"]
    lines.extend(
        [
            f"1M-context working set (BF16 KV): attn {gib(ws1m['attention_kv_bytes']):.3f} GiB + "
            f"indexer {gib(ws1m['indexer_kv_bytes']):.3f} GiB + "
            f"compressor state {gib(ws1m['compressor_state_bytes']):.3f} GiB.",
            "",
            "**Source precision does not fit.** **1.5 complete, if achieved, does — "
            "at 1M context with tens of GiB of CLEAN margin.**",
            "",
            "## Why this is not a yes",
            "",
            "1. Experts are already FP4. The transferable Q80 rates were measured on BF16.",
            "2. Existing capture is underdetermined: 1275 fullseq tokens, late_hidden 32×4096, "
            "AX batch sample 1 row/expert. w2 post-SwiGLU X is not captured.",
            f"3. Writer defaults (first-N {cap['writer_defaults_are_underdetermined']['default_max_hidden_tokens_per_expert']}, "
            f"threshold {cap['writer_defaults_are_underdetermined']['default_row_threshold']}) are "
            f"far below dim {HIDDEN}/{MOE_INTER}.",
            "4. Q30 static ≤1.5 coherence failed. This lane does not copy that approach.",
            "5. Q80 mixed 1.43051 itself was SCREEN_PASSED_NOT_YET_PACKED_OR_GENERATED.",
            "",
            "## Capture plan (if a later lane fits)",
            "",
            "- Teacher = official mixed source forward. Not Gaussian. Not a degraded pack.",
            "- Per-(layer, expert) first-N in the existing compact f32le mmap shape. No giant JSON.",
            "- w1/w3 X = post-ffn_norm hidden. w2 X = **post-SwiGLU**, dim 2048.",
            "- Determined floor: ≥512 rows for a rank-256 score; never `rank = min(budget, n_rows)`.",
            f"- Uniform-routing fill: {cap['tokens_to_fill_512_rows_if_uniform']['tokens_per_layer']} tokens, reused across 43 layers.",
            "- Layer-tile: capture layer L, fit, discard X. Peak X ≈ one layer's w1+w2 buffers.",
            "- Hash layers 0–2: report expert holes; do not invent rows.",
            "- GPU lock for the source forward. This tool does not run that forward.",
            "",
            "## Verdict",
            "",
            f"- Arithmetic: **{v['arithmetic']}** (source resident {v['source_precision_resident']}; "
            f"1.5-if-achieved resident {v['target_1_5_resident_if_rate_achieved']})",
            f"- Quality: **{v['quality']}**",
            f"- Generation: **{v['generation']}**",
            "",
            "A well-evidenced negative would be a *determined* activation-weighted fit "
            "that cannot clear a stated numeric gate. That experiment has not been run. "
            "Declaring ≤1.5 resident DSV4F a success from this census would be manufactured optimism.",
            "",
            "## Next bottleneck",
            "",
            "Bytes are not the wall *if* 1.5 complete is achieved. The wall is a determined "
            "teacher-X fit (existing capture is underdetermined). A 21,846-token fill at the "
            "streamed warm body (3,275 / 3,323 / 3,425 ms, DIRTY_ENGINEERING) is about "
            "20 hours of 43-layer source forward. After a hypothetical resident pack, the "
            "already-measured runtime wall is `metal.gpu` 2.083 s/token (act_quant), not I/O.",
            "",
        ]
    )
    return "\n".join(lines)


def default_out_dir(repo: Path) -> Path:
    return repo / "receipts" / "ascent-2026-08-16"


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dsv-resident-gravity.json"
    md_path = out_dir / "dsv-resident-gravity.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def analyze(artifact_root: Path | None = None) -> dict[str, Any]:
    root = resolve_artifact_root(artifact_root)
    manifest = root / "manifest.json"
    identity = extract_manifest_identity(manifest)
    identity["artifact_root"] = str(root)
    classified = classify_all(iter_manifest_tensors(manifest))
    scopes = scope_logical(manifest)
    return build_feasibility(classified, identity, scopes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(args.artifact)
    if args.write:
        repo = Path(__file__).resolve().parents[2]
        out_dir = args.out_dir or default_out_dir(repo)
        json_path, md_path = write_reports(report, out_dir)
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    cov = report["coverage"]
    v = report["verdict"]
    print(
        f"census {cov['tensor_count']}/{cov['expected_tensor_count']} "
        f"residual={cov['byte_residual']} covers={cov['covers_all_tensors']} "
        f"arithmetic={v['arithmetic']} quality={v['quality']}"
    )
    return 0 if cov["covers_all_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
