#!/usr/bin/env python3
"""EXECUTABLE ECONOMICS — score a representation before anyone fits it.

A compression ratio without an execution story is not a candidate. This
module is the cost model that says, in advance, whether a proposed
representation can mathematically matter on the measured token.

A unique byte is not a unit of time. The organ-average GB/s billed every
removed byte at 344.1 regardless of stream; AUX_U8_LUT then removed 0.535 GB
of broadcast aux, was billed +1.553 ms, and was slower on the wall clock.
This model prices a candidate at the calibrated marginal rate of the
stream class it declares. A candidate that does not declare a stream
class is REFUSED — defaulting to the organ average is how the error
stayed invisible.

Input: bytes removed, bytes added (generator, embeddings, residuals,
metadata, state), extra FLOPs per output element, dispatch delta, the
consuming primitive, the STREAM CLASS, and which bandwidth regime that
primitive plausibly runs in.

Output: predicted ms/token delta, predicted TPS, and MATERIAL or
IMMATERIAL against the S020 §20 bar — substantial work is deserved if
the candidate plausibly removes >= 1% of complete token time, or
creates a reusable representation family, or provides a high-information
falsifier.

    python3 tools/future/executable_economics.py --calibrate-from receipts/future/_ECONOMICS_CALIBRATION_raw.json
    python3 tools/future/executable_economics.py --record
    python3 -m pytest tools/future/test_executable_economics.py -q

Scoring is STATIC_ONLY arithmetic over cited organ times and a measured
per-stream calibration (ECONOMICS_CALIBRATION.json). The calibration is
SELF_MEASURED_DIRTY. Scoring does not touch crates/.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future import causal_budget_71 as cb
from tools.future._common import RECEIPTS, REPO, load_json, write_receipt
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "EXECUTABLE_ECONOMICS.json"
CALIBRATION_RECEIPT = "ECONOMICS_CALIBRATION.json"
CALIBRATION_RAW = "_ECONOMICS_CALIBRATION_raw.json"
SCHEMA = "hawking.future.executable_economics.v2"
CALIBRATION_SCHEMA = "hawking.future.economics_calibration.v1"
VERSION = 2
CALIBRATION_VERSION = 1
RECORDED_BY = "tools/future/executable_economics.py"
EVIDENCE_CLASS = "STATIC_ONLY"
CALIBRATION_EVIDENCE_CLASS = "SELF_MEASURED_DIRTY"

AUX_REL = "receipts/future/MLP_AUXILIARY_INFORMATION.json"
CODE_REL = "receipts/future/MLP_CODE_INFORMATION.json"
DN_REL = "receipts/future/DELTANET_REPRESENTATION.json"
SWEEP_REL = "receipts/future/DISPATCH_SIZE_SWEEP.json"
BA_REL = "receipts/future/BA_DELTA_AB.json"

# ---------------------------------------------------------------------------
# Cited measured constants. Arithmetic over these is STATIC_ONLY. None of
# them are re-measured here.
# ---------------------------------------------------------------------------

# THE BASELINE IS READ, NOT REMEMBERED. 28.722 was the pre-widen_f4 anchor and
# was hard-coded here through two rebases: G131 promoted three levers to sealed
# defaults and measured 21.9464 ms GPU in a protected window, so every
# predicted_token_ms in this gate was 6.8 ms stale and every predicted TPS with
# it. The unit matters too - this file prices BYTES at GPU stream rates, so the
# GPU figure is the right basis and the old wall/gpu split is not needed.
def _cited_token_ms() -> tuple[float, float, str]:
    """(gpu_ms, host_ms, basis). Falls back loudly rather than silently."""
    p = REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"
    if p.is_file():
        m = json.loads(p.read_text())["measured"]
        return (float(m["gpu_ms_per_token"]), float(m["host_gap_ms"]),
                "GPU_SEALED_DEFAULT")
    return 27.733, cb.HOST_GAP_MS, "CITED_PRE_PROMOTION_FALLBACK"


CITED_GPU_MS, CITED_HOST_MS, CITED_BASIS = _cited_token_ms()
CITED_TOKEN_MS = CITED_GPU_MS + CITED_HOST_MS
TPS_71_TOKEN_MS = 1000.0 / 71.0  # 14.085
TPS_71_GPU_MS = TPS_71_TOKEN_MS - CITED_HOST_MS  # 13.096
GPU_REDUCTION_FOR_71 = 1.0 - TPS_71_GPU_MS / CITED_GPU_MS  # 52.8%

CLEAN_GEMV_GB_S = cb.CLEAN_GEMV_GB_S  # 703.5
LM_HEAD_GB_S = cb.DEMONSTRATED_GB_S  # 497.4

# Affine-Q2 size curve (DISPATCH_SIZE_SWEEP). Saturates ~377 at any
# per-dispatch size past the 20 MB knee. Does not reach the LM head.
AFFINE_Q2_GB_S_AT_5MB = 223.4
AFFINE_Q2_GB_S_AT_20MB = 325.9
AFFINE_Q2_GB_S_AT_338MB = 374.4
AFFINE_Q2_GB_S_AT_700MB = 376.7
AFFINE_Q2_SATURATED_GB_S = 376.7

MARGINAL_US_MLP_GQA_NORM = 6.25
MARGINAL_US_DELTANET_BA = 2.884

S020_SECTION_20_BAR_FRAC = 0.01
S020_SECTION_20_BAR_MS = CITED_TOKEN_MS * S020_SECTION_20_BAR_FRAC

# Geometry of sealed-3.14. Same numbers as representation_decode_fusion /
# mlp_auxiliary_information.
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
DN_LAYERS = 48
GQA_LAYERS = 16
VOCAB = 248320
QKVZ_ROWS = 16384
BA_ROWS = 96
GQA_Q_ROWS = 12288
GQA_KV_ROWS = 1024
F16_BYTES = 2

MLP_PARAMS = LAYERS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE)
MLP_ACTIVE_BYTES = 5_347_795_776
MLP_CODE_BYTES = 4_278_190_080
MLP_AUX_BYTES = 1_069_605_696
DN_ACTIVE_BYTES = 2_961_659_904
GQA_ACTIVE_BYTES = 891_292_160
LM_HEAD_ACTIVE_BYTES = 675_430_440
ACTIVE_BYTES = cb.ACTIVE_BYTES
MLP_MS = 15.541
DN_MS = 8.227
GQA_MS = 2.607
LM_HEAD_MS = 1.358
MLP_GB_S = 344.1
DN_GB_S = 360.0
GQA_GB_S = 341.9

ORGAN_BYTES: dict[str, int] = {
    "mlp": MLP_ACTIVE_BYTES,
    "deltanet": DN_ACTIVE_BYTES,
    "gqa": GQA_ACTIVE_BYTES,
    "lm_head": LM_HEAD_ACTIVE_BYTES,
    "token": ACTIVE_BYTES,
}
ORGAN_MS: dict[str, float] = {
    "mlp": MLP_MS,
    "deltanet": DN_MS,
    "gqa": GQA_MS,
    "lm_head": LM_HEAD_MS,
    "token": CITED_GPU_MS,
}
ORGAN_GB_S: dict[str, float] = {
    "mlp": MLP_GB_S,
    "deltanet": DN_GB_S,
    "gqa": GQA_GB_S,
    "lm_head": LM_HEAD_GB_S,
    "token": ACTIVE_BYTES / 1e9 / (CITED_GPU_MS / 1000.0),
}
ORGAN_OUTPUT_ELEMENTS: dict[str, int] = {
    "mlp": LAYERS * (INTERMEDIATE + INTERMEDIATE + HIDDEN),
    "deltanet": DN_LAYERS * (QKVZ_ROWS + HIDDEN + BA_ROWS),
    "gqa": GQA_LAYERS * (GQA_Q_ROWS + GQA_KV_ROWS + GQA_KV_ROWS + HIDDEN),
    "lm_head": VOCAB,
}
ORGAN_OUTPUT_ELEMENTS["token"] = sum(
    ORGAN_OUTPUT_ELEMENTS[k] for k in ("mlp", "deltanet", "gqa", "lm_head")
)
ORGAN_PARAMS: dict[str, int] = {
    "mlp": MLP_PARAMS,
    "lm_head": VOCAB * HIDDEN,
}

# Effective GEMV-dressed FLOP rate of the MLP organ. Extra FLOPs are
# scored at this rate as an ASSUMPTION: same cost-per-FLOP as the
# incumbent MAC, which is itself mostly wait-on-memory. Hidden-under-
# bandwidth is 0; decode-ALU-like is FLOP_EXPOSED_MULT times this.
EFFECTIVE_FLOP_S = (2.0 * MLP_PARAMS) / (MLP_MS / 1000.0)
FLOP_EXPOSED_MULT = 4.0

BYTES_ADDED_FIELDS: tuple[str, ...] = (
    "generator",
    "embeddings",
    "residuals",
    "metadata",
    "state",
)

LIVE_STATUSES = frozenset({"OPEN", "UNMEASURED", "EXISTING_LEVER"})
DEAD_STATUSES = frozenset(
    {
        "MEASURED_NEGATIVE",
        "ALREADY_FALSIFIED",
        "REJECTED_DENSE_REMAT",
        "AT_THE_FLOOR",
        "PER_DISPATCH_SIZE_REFUTED",
        "GRANULARITY_REFUTED",
        "BOUNDED_TOO_SMALL",
        "ALREADY_FUSED",
        "CLOSED",
        "REFUTED",
    }
)

# Stream classes a candidate must declare. Defaulting to the organ average
# is how AUX_U8_LUT billed +1.553 ms for bytes that were never binding.
STREAM_CLASS_WEIGHT_CODES = "weight_codes"
STREAM_CLASS_BROADCAST_AUX = "broadcast_aux"
STREAM_CLASS_ACTIVATION = "activation"
STREAM_CLASS_NAMES = frozenset(
    {
        STREAM_CLASS_WEIGHT_CODES,
        STREAM_CLASS_BROADCAST_AUX,
        STREAM_CLASS_ACTIVATION,
    }
)

# Capability screen (AUX_CAPABILITY_SCREEN) already refuted these on
# held-out fit. Re-price their byte claims so the record is consistent.
GRANULARITY_REFUTED_IDS = frozenset({"group_size_256", "group_size_1024"})

# Every recorded candidate must appear here. Missing → refuse, not default.
STREAM_CLASS_BY_ID: dict[str, str] = {
    # MLP auxiliary (broadcast per-group scale/bias).
    "quantize_aux_u8": STREAM_CLASS_BROADCAST_AUX,
    "shared_scale_basis": STREAM_CLASS_BROADCAST_AUX,
    "per_tensor_curve_plus_residual": STREAM_CLASS_BROADCAST_AUX,
    "predict_scale_from_code_stats": STREAM_CLASS_BROADCAST_AUX,
    "low_rank_scale_matrix": STREAM_CLASS_BROADCAST_AUX,
    "parametric_scale_program": STREAM_CLASS_BROADCAST_AUX,
    "larger_group_size": STREAM_CLASS_BROADCAST_AUX,
    "group_size_256": STREAM_CLASS_BROADCAST_AUX,
    "group_size_512": STREAM_CLASS_BROADCAST_AUX,
    "group_size_1024": STREAM_CLASS_BROADCAST_AUX,
    "tie_bias_to_minus_half_codes": STREAM_CLASS_BROADCAST_AUX,
    "drop_bias": STREAM_CLASS_BROADCAST_AUX,
    "collapse_to_global_scale": STREAM_CLASS_BROADCAST_AUX,
    "cross_layer_scale_delta": STREAM_CLASS_BROADCAST_AUX,
    "pack_headers": STREAM_CLASS_BROADCAST_AUX,
    # MLP codes (per-thread unique, the binding stream).
    "lower_bit_native": STREAM_CLASS_WEIGHT_CODES,
    "heterogeneous_bit_allocation": STREAM_CLASS_WEIGHT_CODES,
    "generated_tensors": STREAM_CLASS_WEIGHT_CODES,
    "generated_programs": STREAM_CLASS_WEIGHT_CODES,
    "shared_code_bases": STREAM_CLASS_WEIGHT_CODES,
    "factorized_programs": STREAM_CLASS_WEIGHT_CODES,
    "dictionary_of_code_blocks": STREAM_CLASS_WEIGHT_CODES,
    "product_codebooks": STREAM_CLASS_WEIGHT_CODES,
    "lowrank_plus_sparse_residual": STREAM_CLASS_WEIGHT_CODES,
    "block_generators": STREAM_CLASS_WEIGHT_CODES,
    "cross_layer_code_prediction": STREAM_CLASS_WEIGHT_CODES,
    "capability_sensitive_literal_islands": STREAM_CLASS_WEIGHT_CODES,
    "function_replacement": STREAM_CLASS_WEIGHT_CODES,
    "entropy_coded_code_stream": STREAM_CLASS_WEIGHT_CODES,
    # Activation-side / tiled transforms of x.
    "shared_input_transforms": STREAM_CLASS_ACTIVATION,
    "latent_routed_accumulation": STREAM_CLASS_ACTIVATION,
    # DeltaNet weight codes.
    "heterogeneous_qkvz_bits": STREAM_CLASS_WEIGHT_CODES,
    "lower_bit_uniform_qkvz": STREAM_CLASS_WEIGHT_CODES,
    "lower_bit_out_proj": STREAM_CLASS_WEIGHT_CODES,
    "gravity_family_on_dn_weights": STREAM_CLASS_WEIGHT_CODES,
    "larger_q4_group": STREAM_CLASS_WEIGHT_CODES,
    "generated_coefficients": STREAM_CLASS_WEIGHT_CODES,
    "factorized_qkvz": STREAM_CLASS_WEIGHT_CODES,
    "conv1d_lower_bit": STREAM_CLASS_WEIGHT_CODES,
    # DeltaNet state / activation.
    "shared_transforms_across_layers": STREAM_CLASS_ACTIVATION,
    "lower_bit_recurrent_state": STREAM_CLASS_ACTIVATION,
    "structured_transition_state": STREAM_CLASS_ACTIVATION,
    "recurrent_state_replacement": STREAM_CLASS_ACTIVATION,
    "share_or_merge_state_across_depth": STREAM_CLASS_ACTIVATION,
    "direct_state_machine": STREAM_CLASS_ACTIVATION,
    "fused_update_consume": STREAM_CLASS_ACTIVATION,
    # Dispatch-size levers act on the code-bearing GEMV launch.
    "surviving_dispatch_size_amortize_sub20mb": STREAM_CLASS_WEIGHT_CODES,
    "dispatch_size_concat_to_lm_head_mb": STREAM_CLASS_WEIGHT_CODES,
}

_CALIBRATION: dict[str, Any] | None = None

# Packing / codec / operator families that would transfer, not one-off hacks.
REUSABLE_FAMILY_IDS = frozenset(
    {
        "quantize_aux_u8",
        "larger_group_size",
        "group_size_256",
        "group_size_512",
        "group_size_1024",
        "entropy_coded_code_stream",
        "lower_bit_native",
        "heterogeneous_bit_allocation",
        "heterogeneous_qkvz_bits",
        "lower_bit_uniform_qkvz",
        "lower_bit_out_proj",
        "larger_q4_group",
        "function_replacement",
        "shared_input_transforms",
        "generated_programs",
        "generated_coefficients",
        "factorized_programs",
        "factorized_qkvz",
        "structured_transition_state",
        "recurrent_state_replacement",
        "lower_bit_recurrent_state",
        "direct_state_machine",
        "surviving_dispatch_size_amortize_sub20mb",
    }
)

# Cheap experiments that would teach a whole school, not just one row.
HIGH_INFORMATION_FALSIFIER_IDS = frozenset(
    {
        "surviving_dispatch_size_amortize_sub20mb",
        "dispatch_size_concat_to_lm_head_mb",
        "entropy_coded_code_stream",
        "function_replacement",
        "shared_input_transforms",
        "direct_state_machine",
        "heterogeneous_qkvz_bits",
    }
)

# Group-size curve points that causal_budget_71 ranked separately.
# larger_group_size in the aux receipt is G=128 (534,773,760).
GROUP_SIZE_EXTRA: tuple[tuple[int, int], ...] = (
    (256, 802_160_640),
    (512, 935_854_080),
    (1024, 1_002_700_800),
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Predicted ms/token and predicted TPS are "
    "arithmetic over cited organ times, cited byte shares, cited "
    "dispatch-class costs, a declared stream class, and the measured "
    "per-stream marginal rate in ECONOMICS_CALIBRATION.json. They are not "
    "a protected measurement and not a promise that a fit will hold "
    "capability. Unique bytes of broadcast_aux are not billed at the organ "
    "average: that was the AUX_U8_LUT overcredit. A new representation is "
    "not assumed to run at the affine-Q2 saturation (~377 GB/s), the "
    "LM-head demonstrated 497.4 GB/s, or the clean GEMV roof 703.5 GB/s. "
    "evidence_class is STATIC_ONLY. gpu_authority is false. The calibration "
    "receipt is SELF_MEASURED_DIRTY with loadavg recorded."
)

_UNSET: Any = object()


class IncompleteEconomics(ValueError):
    """bytes_removed without bytes_added is not a candidate."""


class EconomicsRefuse(ValueError):
    """The cost model refused rather than guessing."""


# ---------------------------------------------------------------------------
# Bandwidth regimes. Each is a named ASSUMPTION with a range. Affine-Q2
# saturates near 377 GB/s at any per-dispatch size; the LM head reaches
# 497.4 and nothing else has been shown to; the clean GEMV roof is 703.5.
# ---------------------------------------------------------------------------

BANDWIDTH_REGIMES: dict[str, dict[str, Any]] = {
    "production_mlp": {
        "gb_s_lo": MLP_GB_S,
        "gb_s_hi": AFFINE_Q2_SATURATED_GB_S,
        "gb_s_nominal": MLP_GB_S,
        "measured": True,
        "assumption": (
            "nominal is the production MLP organ rate. The high end is the "
            "affine-Q2 size-curve saturation, which isolated launches reach "
            "and production has not. Not 497.4, not 703.5."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json + DISPATCH_SIZE_SWEEP.json",
    },
    "production_deltanet": {
        "gb_s_lo": DN_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": DN_GB_S,
        "measured": True,
        "assumption": (
            "nominal is the production DeltaNet organ rate (Q4). The high "
            "end is the LM head's demonstrated 497.4, also Q4, which DeltaNet "
            "has not been shown to reach. Including 497.4 in the range is an "
            "ASSUMPTION, not a prediction."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "production_gqa": {
        "gb_s_lo": GQA_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": GQA_GB_S,
        "measured": True,
        "assumption": (
            "nominal is production GQA. High end is the LM-head demonstrated "
            "rate; GQA has not been shown to reach it."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "production_lm_head": {
        "gb_s_lo": LM_HEAD_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": LM_HEAD_GB_S,
        "measured": True,
        "assumption": "the one organ that has been shown to reach 497.4 GB/s",
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "affine_q2_family": {
        "gb_s_lo": MLP_GB_S,
        "gb_s_hi": AFFINE_Q2_SATURATED_GB_S,
        "gb_s_nominal": MLP_GB_S,
        "measured": True,
        "assumption": (
            "production-shaped affine-Q2. The size curve saturates at "
            f"{AFFINE_Q2_SATURATED_GB_S} GB/s and does not rise toward "
            f"{LM_HEAD_GB_S}. A 5 MB launch is {AFFINE_Q2_GB_S_AT_5MB} GB/s; "
            "that is a different regime (affine_q2_unamortized)."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "affine_q2_unamortized": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_5MB,
        "gb_s_hi": AFFINE_Q2_GB_S_AT_20MB,
        "gb_s_nominal": AFFINE_Q2_GB_S_AT_5MB,
        "measured": True,
        "assumption": (
            "affine-Q2 launches around 5 MB. The surviving dispatch-size "
            "lever is this amortization to >= 20 MB, not the 350->497 gap."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "affine_q2_saturated": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_338MB,
        "gb_s_hi": AFFINE_Q2_GB_S_AT_700MB,
        "gb_s_nominal": AFFINE_Q2_SATURATED_GB_S,
        "measured": True,
        "assumption": (
            "the affine-Q2 family ceiling. Concatenating to the LM head's "
            "337.7 MB working set still sits here, not at 497.4."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "organ_cluster": {
        "gb_s_lo": GQA_GB_S,
        "gb_s_hi": DN_GB_S,
        "gb_s_nominal": 350.0,
        "measured": True,
        "assumption": "MLP, DeltaNet and GQA sit inside 5% of each other",
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "lm_head_demonstrated": {
        "gb_s_lo": LM_HEAD_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": LM_HEAD_GB_S,
        "measured": True,
        "assumption": (
            "ASSUMPTION if applied to any organ other than the LM head. "
            "No other organ has been shown to reach 497.4 GB/s. Affine-Q2 "
            f"saturates at {AFFINE_Q2_SATURATED_GB_S}."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "clean_gemv_roof": {
        "gb_s_lo": CLEAN_GEMV_GB_S,
        "gb_s_hi": CLEAN_GEMV_GB_S,
        "gb_s_nominal": CLEAN_GEMV_GB_S,
        "measured": True,
        "assumption": (
            "ASSUMPTION. Clean single-GEMV roof, not a packed-representation "
            "rate. Beating the LM head as well. Not a prediction."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "unknown_new_representation": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_5MB,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": 350.0,
        "measured": False,
        "assumption": (
            "ASSUMPTION with a range, not a point estimate. A new "
            "representation is bounded below by the slowest measured "
            f"affine-Q2 launch ({AFFINE_Q2_GB_S_AT_5MB} GB/s at 5 MB) and "
            f"above by the fastest demonstrated packed GEMV ({LM_HEAD_GB_S} "
            "GB/s, the LM head). The clean GEMV roof is excluded: nothing "
            "packed has been shown to reach it."
        ),
        "source": "DISPATCH_SIZE_SWEEP.json + ORGAN_BANDWIDTH.json",
    },
}

DISPATCH_CLASSES: dict[str, dict[str, Any]] = {
    "mlp_gqa_norm_fusion": {
        "us": MARGINAL_US_MLP_GQA_NORM,
        "source": "receipts/future/RESIDENT_TOKEN_BUDGET.json derived.measured_marginal_dispatch_us",
    },
    "deltanet_ba": {
        "us": MARGINAL_US_DELTANET_BA,
        "source": "receipts/future/BA_DELTA_AB.json derived.us_per_removed_dispatch",
    },
}


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _index_rungs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ident = str(row.get("id") or "")
        if ident:
            out[ident] = row
    return out


def _paired_noise_ns(rung: Mapping[str, Any]) -> float:
    mad = float(rung.get("paired_dt_ns_mad") or 0.0)
    return max(2.0 * mad, 2000.0)


def _fit_stream_class(
    *,
    name: str,
    rungs: Mapping[str, Mapping[str, Any]],
    keep_ids: dict[int, str],
    catalog_billing: str,
    organ_gb_s: float,
) -> dict[str, Any]:
    """Marginal unique-byte cost of one stream from paired keep-fraction rungs.

    Primary point is the 50% drop (operating-point derivative). A |dt|
    inside 2*MAD of the paired deltas (floor 2 us) is treated as 0: the
    stream is not on the critical path and must not be billed at the
    organ average.
    """
    samples = []
    for keep, ident in sorted(keep_ids.items(), reverse=True):
        row = rungs.get(ident)
        if not isinstance(row, Mapping):
            raise EconomicsRefuse(f"calibration missing rung {ident}")
        dt = int(row.get("dt_ns_vs_baseline") or 0)
        dropped = int(row.get("unique_bytes_dropped") or 0)
        mad = int(row.get("paired_dt_ns_mad") or 0)
        samples.append(
            {
                "id": ident,
                "keep_pct": keep,
                "drop_pct": 100 - keep,
                "unique_bytes_dropped": dropped,
                "paired_dt_ns": dt,
                "paired_dt_ns_mad": mad,
                "gpu_ns_median": int(row.get("gpu_ns_median") or 0),
            }
        )
    primary = rungs.get(keep_ids[50])
    if not isinstance(primary, Mapping):
        raise EconomicsRefuse(f"calibration missing 50% rung for {name}")
    dt = int(primary.get("dt_ns_vs_baseline") or 0)
    dropped = int(primary.get("unique_bytes_dropped") or 0)
    noise = _paired_noise_ns(primary)
    within_noise = abs(dt) <= noise
    faster = dt < -noise
    on_critical = bool(faster)

    ms_per_gb = 0.0
    billing_gb_s: float | None = None
    if on_critical and dropped > 0 and dt < 0:
        gb = dropped / 1e9
        ms_per_gb = (-dt / 1e6) / gb
        billing_gb_s = dropped / (-dt / 1e9) / 1e9

    # Activation unique-x rate (~2 GB/s) does not transfer to 100 MB+
    # state tensors. The probe's finding for this class is "not free";
    # catalog-scale activation/state candidates bill at the organ rate.
    catalog_gb_s: float | None
    catalog_ms_per_gb: float
    if catalog_billing == "organ_average":
        catalog_gb_s = float(organ_gb_s)
        catalog_ms_per_gb = 1000.0 / catalog_gb_s if catalog_gb_s else 0.0
        catalog_on_critical = True
    elif catalog_billing == "measured_or_zero":
        catalog_gb_s = billing_gb_s
        catalog_ms_per_gb = ms_per_gb
        catalog_on_critical = on_critical
    else:
        raise EconomicsRefuse(f"unknown catalog_billing {catalog_billing}")

    lo_gb = hi_gb = catalog_gb_s
    if on_critical and billing_gb_s is not None and catalog_billing == "measured_or_zero":
        rates = []
        for s in samples:
            sdt = int(s["paired_dt_ns"])
            sdrop = int(s["unique_bytes_dropped"])
            if sdt < -_paired_noise_ns({"paired_dt_ns_mad": s["paired_dt_ns_mad"]}) and sdrop > 0:
                rates.append(sdrop / (-sdt / 1e9) / 1e9)
        if rates:
            lo_gb = min(rates)
            hi_gb = max(rates)

    return {
        "name": name,
        "on_critical_path": catalog_on_critical if catalog_billing == "organ_average" else on_critical,
        "probe_on_critical_path": on_critical,
        "within_noise_at_50pct": within_noise,
        "primary_rung": keep_ids[50],
        "unique_bytes_dropped_at_50pct": dropped,
        "paired_dt_ns_at_50pct": dt,
        "paired_noise_floor_ns": noise,
        "ms_per_gb_saved_measured": _r(ms_per_gb, 6),
        "billing_gb_s_measured": None if billing_gb_s is None else _r(billing_gb_s, 2),
        "catalog_billing": catalog_billing,
        "billing_gb_s": None if catalog_gb_s is None else _r(catalog_gb_s, 2),
        "ms_per_gb_saved": _r(catalog_ms_per_gb, 6),
        "billing_gb_s_lo": None if lo_gb is None else _r(lo_gb, 2),
        "billing_gb_s_hi": None if hi_gb is None else _r(hi_gb, 2),
        "samples": samples,
    }


def calibrate_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a paired stream_criticality_probe JSON into a sealed calibration."""
    rungs = _index_rungs(raw.get("rungs") or [])
    for need in (
        "codes_keep_50",
        "codes_keep_0",
        "aux_keep_50",
        "aux_keep_0",
        "x_keep_50",
        "x_keep_0",
        "zero_load",
        "baseline",
    ):
        if need not in rungs:
            raise EconomicsRefuse(f"calibration raw is missing rung {need}")

    codes = _fit_stream_class(
        name=STREAM_CLASS_WEIGHT_CODES,
        rungs=rungs,
        keep_ids={75: "codes_keep_75", 50: "codes_keep_50", 25: "codes_keep_25", 0: "codes_keep_0"},
        catalog_billing="measured_or_zero",
        organ_gb_s=MLP_GB_S,
    )
    aux = _fit_stream_class(
        name=STREAM_CLASS_BROADCAST_AUX,
        rungs=rungs,
        keep_ids={75: "aux_keep_75", 50: "aux_keep_50", 25: "aux_keep_25", 0: "aux_keep_0"},
        catalog_billing="measured_or_zero",
        organ_gb_s=MLP_GB_S,
    )
    x = _fit_stream_class(
        name=STREAM_CLASS_ACTIVATION,
        rungs=rungs,
        keep_ids={75: "x_keep_75", 50: "x_keep_50", 25: "x_keep_25", 0: "x_keep_0"},
        catalog_billing="organ_average",
        organ_gb_s=MLP_GB_S,
    )

    load = raw.get("concurrent_load") or {}
    loadavg = load.get("loadavg")
    census = raw.get("census") or {}
    codes_ok = bool(codes["on_critical_path"])
    aux_free = not bool(aux["on_critical_path"])
    finding = (
        f"Paired A/B on Apple M3 Ultra layer {raw.get('layer')}, loadavg {loadavg}. "
        f"weight_codes 50% drop {codes['paired_dt_ns_at_50pct']} ns for "
        f"{codes['unique_bytes_dropped_at_50pct']} unique bytes "
        f"({'ON' if codes_ok else 'NOT ON'} the critical path, "
        f"{codes['ms_per_gb_saved']} ms/GB, billing "
        f"{codes['billing_gb_s']} GB/s unique). "
        f"broadcast_aux 50% drop {aux['paired_dt_ns_at_50pct']} ns for "
        f"{aux['unique_bytes_dropped_at_50pct']} unique bytes "
        f"(within_noise={aux['within_noise_at_50pct']}; "
        f"{'NOT billed' if aux_free else 'billed'} at the organ average). "
        f"activation 50% drop {x['paired_dt_ns_at_50pct']} ns for "
        f"{x['unique_bytes_dropped_at_50pct']} unique x bytes "
        f"(probe_on_critical_path={x['probe_on_critical_path']}; catalog "
        f"state/activation bills at the organ average {MLP_GB_S} GB/s because "
        f"the unique-x rate does not transfer to 100 MB+ tensors). "
        "A candidate that does not declare a stream class is refused."
    )
    return {
        "schema": CALIBRATION_SCHEMA,
        "version": CALIBRATION_VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": CALIBRATION_EVIDENCE_CLASS,
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/stream_criticality_probe.rs; "
            "paired MTLCommandBuffer GPUStartTime/GPUEndTime; one MLP layer "
            "of sealed-3.14, arithmetic stripped, trip count held"
        ),
        "claim_boundary": (
            "SELF_MEASURED_DIRTY. Absolute GPU ns is measured-under-load. "
            "The calibration is the paired dt (treat - baseline) of dropping "
            "a keep-fraction of ONE stream, unique catalog bytes of that "
            "stream times (1-keep) in the numerator. Order of the pair "
            "alternates. A |dt| inside 2*MAD (floor 2 us) is 0. Does not "
            "change the production decode path. Token-ms numbers in the "
            "economics receipt are arithmetic over this calibration and "
            "cited organ times."
        ),
        "finding": finding,
        "metal_device": raw.get("metal_device"),
        "layer": raw.get("layer"),
        "warmup": raw.get("warmup"),
        "reps": raw.get("reps"),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing"),
        "geometry": raw.get("geometry"),
        "organ": raw.get("organ", "mlp"),
        "loadavg": loadavg,
        "concurrent_load": load,
        "concurrent_load_start": raw.get("concurrent_load_start"),
        "census": census,
        "baseline_gpu_ns_median": raw.get("baseline_gpu_ns_median"),
        "baseline_gpu_ns_mad": raw.get("baseline_gpu_ns_mad"),
        "organ_average_gb_s_mlp": MLP_GB_S,
        "organ_average_ms_per_gb_mlp": _r(1000.0 / MLP_GB_S, 6),
        "stream_classes": {
            STREAM_CLASS_WEIGHT_CODES: codes,
            STREAM_CLASS_BROADCAST_AUX: aux,
            STREAM_CLASS_ACTIVATION: x,
        },
        "primary_marginal_is_50pct_drop": True,
        "rungs": list(raw.get("rungs") or []),
        "projections": raw.get("projections") or [],
        "pre_registered_interpretation": raw.get("pre_registered_interpretation"),
        # THE SCAR, EMITTED BY THE PRODUCER so a rebuild cannot delete it (S025
        # §19: a generated artifact is not fixed by hand-editing its receipt).
        # negative_index reads the `scars` ARRAY, so it goes there and nowhere else.
        "scars": [
            {
                "family": "BYTE_COUNT_TIMES_ORGAN_AVERAGE",
                "status": "MEASURED_NEGATIVE",
                "level": "MODEL_SPECIFIC",
                "parent": "qwen3.8-27b sealed-3.14",
                "organ": "mlp",
                "object": "any projection of token time from a byte count",
                "mechanism": (
                    "BYTE_COUNT x ORGAN_AVERAGE_RATE IS NOT A VALID COST MODEL "
                    "WHEN BYTE CLASSES HAVE DIFFERENT PHYSICAL BEHAVIOUR. Dropping "
                    "fractions of each stream and timing it: codes_keep_50 is "
                    "faster than 2*MAD, aux_keep_50 is NOT. weight_codes bill at "
                    f"{codes.get('ms_per_gb_saved')} ms/GB (an effective "
                    f"{codes.get('billing_gb_s')} GB/s, not the 344.1 organ "
                    f"average); broadcast_aux bills at {aux.get('ms_per_gb_saved')}. "
                    "Billing an auxiliary byte at the organ average credited "
                    "quantize_aux_u8 with 1.99 TPS and group_size_256 with 3.08 on "
                    "the 71 ladder. Both are 0.00."
                ),
                "not": (
                    "a claim that auxiliary bytes are free to STORE, or that a "
                    "different packing could not make them cost time; it is a "
                    "measurement of THIS packing on THIS body"
                ),
                "requires": "every candidate declares stream_class or score() refuses",
            },
            {
                "family": "BYTES_TRADED_FOR_DECODE_ARITHMETIC",
                "status": "MEASURED_NEGATIVE",
                "level": "MODEL_SPECIFIC",
                "parent": "qwen3.8-27b sealed-3.14",
                "organ": "mlp",
                "object": "any representation that stores fewer bytes at the cost "
                          "of more decode FMA per weight byte",
                "mechanism": (
                    "A REPRESENTATION CHANGE THAT TRADES BYTES FOR DECODE "
                    "ARITHMETIC TRADES A FREE RESOURCE FOR THE BINDING ONE. "
                    "broadcast_aux bytes measure 0.000 ms/GB while stripping "
                    "arithmetic at identical bytes buys 1.51x (MLP_ALU_ROOFLINE "
                    "ARM A), so the trade is negative before it starts. Both "
                    "attempts went the wrong way and were measured doing it: the "
                    "incumbent decodes at 1.3333 FMA/weight-byte, aux_u8_lut at "
                    "2.0, aux_u8_native at 2.5. That is aux_u8's measured "
                    "slowdown explained without a second theory - it removed "
                    "bytes that cost nothing and added arithmetic that costs."
                ),
                "not": (
                    "a claim that no cheaper decode exists, or that storing aux "
                    "compactly is wrong in itself. It refuses the TRADE, not the "
                    "storage: a decode cheaper than 1.3333 would be welcome and "
                    "nobody has built one."
                ),
                "requires": "a candidate must state its decode-FMA per weight byte",
            },
        ],
        "loads_survived": {
            "codes_50pct_faster_than_noise": codes_ok,
            "aux_50pct_within_noise": aux_free,
            "zero_load_dt_ns": (rungs["zero_load"].get("dt_ns_vs_baseline") if "zero_load" in rungs else None),
            "proof": (
                "paired codes_keep_50 is faster than 2*MAD and aux_keep_50 is not; "
                "the organ average is the code-bound rate, not a byte-class rate"
            ),
        },
    }


def record_calibration(raw: Mapping[str, Any]) -> Path:
    doc = calibrate_from_raw(raw)
    out = RECEIPTS / CALIBRATION_RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    global _CALIBRATION
    _CALIBRATION = doc
    return out


def load_calibration(*, path: Path | None = None, force: bool = False) -> dict[str, Any]:
    global _CALIBRATION
    if _CALIBRATION is not None and not force:
        return _CALIBRATION
    p = path or (RECEIPTS / CALIBRATION_RECEIPT)
    if not p.is_file():
        raise EconomicsRefuse(
            f"stream-class calibration missing: {p}. Run the Metal probe and "
            f"`python3 tools/future/executable_economics.py --calibrate-from "
            f"receipts/future/{CALIBRATION_RAW}`"
        )
    doc = load_json(p)
    if not isinstance(doc, dict):
        raise EconomicsRefuse(f"{p} is not an object")
    classes = doc.get("stream_classes")
    if not isinstance(classes, dict) or not STREAM_CLASS_NAMES.issubset(classes):
        raise EconomicsRefuse(
            f"{p} does not define stream_classes {sorted(STREAM_CLASS_NAMES)}"
        )
    _CALIBRATION = doc
    return doc


def stream_spec(stream_class: str) -> dict[str, Any]:
    cal = load_calibration()
    spec = (cal.get("stream_classes") or {}).get(stream_class)
    if not isinstance(spec, dict):
        raise EconomicsRefuse(f"calibration has no stream class {stream_class!r}")
    return spec


def bytes_to_ms(n_bytes: float, gb_s: float) -> float:
    """Milliseconds to move n_bytes at gb_s GB/s."""
    if gb_s <= 0.0:
        raise EconomicsRefuse(f"bandwidth must be positive, got {gb_s}")
    return float(n_bytes) / float(gb_s) * 1e-6


def tps_from_ms(ms: float) -> float:
    if ms <= 0.0:
        raise EconomicsRefuse(f"token time must be positive, got {ms}")
    return 1000.0 / float(ms)


def normalize_bytes_added(bytes_added: Any) -> dict[str, int]:
    """Canonical five-field added-byte ledger. Total is the sum."""
    out = {k: 0 for k in BYTES_ADDED_FIELDS}
    if isinstance(bytes_added, bool):
        raise IncompleteEconomics(
            "bytes_added must be a number or a mapping of "
            + ", ".join(BYTES_ADDED_FIELDS)
        )
    if isinstance(bytes_added, (int, float)):
        if bytes_added < 0:
            raise EconomicsRefuse(f"bytes_added cannot be negative: {bytes_added}")
        out["generator"] = int(bytes_added)
        out["total"] = int(bytes_added)
        return out
    if not isinstance(bytes_added, Mapping):
        raise IncompleteEconomics(
            "bytes_added must be supplied as a number or a mapping of "
            f"{BYTES_ADDED_FIELDS}; a ratio without executable economics "
            "is not a candidate"
        )
    total_check = 0
    for key, raw in bytes_added.items():
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EconomicsRefuse(f"bytes_added[{key!r}] is not a number: {raw!r}")
        if raw < 0:
            raise EconomicsRefuse(f"bytes_added[{key!r}] cannot be negative: {raw}")
        n = int(raw)
        total_check += n
        if key in out:
            out[key] += n
        else:
            # Extra named buckets (e.g. remat_dense) still count.
            out[str(key)] = out.get(str(key), 0) + n
    out["total"] = sum(int(v) for k, v in out.items() if k != "total")
    if out["total"] != total_check:
        out["total"] = total_check
    return out


def infer_regime(
    *,
    organ: str,
    consuming_primitive: str | None,
    bandwidth_regime: str | None,
) -> str:
    if bandwidth_regime:
        if bandwidth_regime not in BANDWIDTH_REGIMES:
            raise EconomicsRefuse(
                f"unknown bandwidth_regime {bandwidth_regime!r}. "
                f"known: {sorted(BANDWIDTH_REGIMES)}"
            )
        return bandwidth_regime
    if consuming_primitive is None or consuming_primitive not in ATLAS_PRIMITIVES:
        return "unknown_new_representation"
    if consuming_primitive in {"TiledProjection", "DirectRoutedAccumulate", "MoveOrRecompute"}:
        return "unknown_new_representation"
    if consuming_primitive == "LocalStateMachine":
        return "production_deltanet" if organ in {"deltanet", "token"} else "unknown_new_representation"
    if consuming_primitive == "LayoutTransform":
        return {
            "mlp": "production_mlp",
            "deltanet": "production_deltanet",
            "gqa": "production_gqa",
            "lm_head": "production_lm_head",
        }.get(organ, "unknown_new_representation")
    # FusedDecodeCompute and the rest of the packed-GEMV family.
    return {
        "mlp": "affine_q2_family",
        "deltanet": "production_deltanet",
        "gqa": "production_gqa",
        "lm_head": "production_lm_head",
    }.get(organ, "unknown_new_representation")


def infer_dispatch_class(organ: str, dispatch_class: str | None) -> str | None:
    if dispatch_class:
        if dispatch_class not in DISPATCH_CLASSES:
            raise EconomicsRefuse(
                f"unknown dispatch_class {dispatch_class!r}. "
                f"known: {sorted(DISPATCH_CLASSES)}"
            )
        return dispatch_class
    if organ in {"mlp", "gqa"}:
        return "mlp_gqa_norm_fusion"
    if organ == "deltanet":
        return "deltanet_ba"
    return None


def _flop_ms(
    extra_flops_per_output_element: float,
    n_output_elements: int,
    *,
    exposed_mult: float = 1.0,
) -> float:
    if extra_flops_per_output_element == 0.0:
        return 0.0
    extra = float(extra_flops_per_output_element) * int(n_output_elements)
    return extra / EFFECTIVE_FLOP_S * 1000.0 * float(exposed_mult)


def score(
    *,
    bytes_removed: Any = _UNSET,
    bytes_added: Any = _UNSET,
    extra_flops_per_output_element: float = 0.0,
    dispatch_delta: float = 0.0,
    consuming_primitive: str | None = None,
    bandwidth_regime: str | None = None,
    organ: str = "mlp",
    n_output_elements: int | None = None,
    dispatch_class: str | None = None,
    stream_class: str | None = None,
    stream_gb_s: float | None = None,
    stream_on_critical_path: bool | None = None,
    retime_organ: bool = False,
    reusable_family: bool = False,
    high_information_falsifier: bool = False,
    candidate_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Score a proposed representation. Does not fit anything.

    Refuses if bytes_removed is supplied without bytes_added: a ratio
    without executable economics is not a candidate.

    Refuses if stream_class is not declared: defaulting unique bytes to
    the organ average is how the aux-u8 overcredit stayed invisible.
    """
    if bytes_removed is not _UNSET and bytes_added is _UNSET:
        raise IncompleteEconomics(
            "bytes_removed without bytes_added: a ratio without executable "
            "economics is not a candidate"
        )
    if bytes_removed is not _UNSET and bytes_added is None:
        raise IncompleteEconomics(
            "bytes_removed without bytes_added: a ratio without executable "
            "economics is not a candidate"
        )
    if stream_class is None:
        raise IncompleteEconomics(
            "stream_class must be declared (weight_codes, broadcast_aux, "
            "activation); defaulting to the organ average is how the "
            "byte-overcredit stays invisible"
        )
    if stream_class not in STREAM_CLASS_NAMES:
        raise EconomicsRefuse(
            f"unknown stream_class {stream_class!r}. known: {sorted(STREAM_CLASS_NAMES)}"
        )

    if bytes_removed is _UNSET or bytes_removed is None:
        removed = 0
    else:
        if isinstance(bytes_removed, bool) or not isinstance(bytes_removed, (int, float)):
            raise EconomicsRefuse(f"bytes_removed is not a number: {bytes_removed!r}")
        if bytes_removed < 0:
            raise EconomicsRefuse(f"bytes_removed cannot be negative: {bytes_removed}")
        removed = int(bytes_removed)

    if bytes_added is _UNSET or bytes_added is None:
        added = normalize_bytes_added(0)
        added_was_supplied = False
    else:
        added = normalize_bytes_added(bytes_added)
        added_was_supplied = True

    if organ not in ORGAN_BYTES:
        raise EconomicsRefuse(
            f"unknown organ {organ!r}. known: {sorted(ORGAN_BYTES)}"
        )
    organ_bytes = ORGAN_BYTES[organ]
    if removed > organ_bytes:
        raise EconomicsRefuse(
            f"bytes_removed {removed} exceeds {organ} active bytes {organ_bytes}"
        )

    added_total = int(added.get("total", 0))
    net_bytes = added_total - removed

    regime_name = infer_regime(
        organ=organ,
        consuming_primitive=consuming_primitive,
        bandwidth_regime=bandwidth_regime,
    )
    regime = BANDWIDTH_REGIMES[regime_name]
    gb_s_nom = float(regime["gb_s_nominal"])
    gb_s_lo = float(regime["gb_s_lo"])
    gb_s_hi = float(regime["gb_s_hi"])
    if gb_s_lo > gb_s_hi:
        gb_s_lo, gb_s_hi = gb_s_hi, gb_s_lo

    # Stream-class rate, not the organ average. The organ average is how
    # unique aux bytes were billed as if they were on the critical path.
    organ_gb_s = ORGAN_GB_S[organ]
    organ_ms = ORGAN_MS[organ]
    spec: dict[str, Any] | None = None
    if stream_gb_s is not None:
        if stream_gb_s <= 0.0:
            raise EconomicsRefuse(f"stream_gb_s must be positive, got {stream_gb_s}")
        on_critical = True if stream_on_critical_path is None else bool(stream_on_critical_path)
        stream_rate = float(stream_gb_s)
        stream_rate_lo = stream_rate
        stream_rate_hi = stream_rate
        stream_rate_source = "caller_override"
    else:
        spec = stream_spec(stream_class)
        on_critical = (
            bool(spec.get("on_critical_path"))
            if stream_on_critical_path is None
            else bool(stream_on_critical_path)
        )
        raw_rate = spec.get("billing_gb_s")
        stream_rate = float(raw_rate) if raw_rate else None
        lo_raw = spec.get("billing_gb_s_lo")
        hi_raw = spec.get("billing_gb_s_hi")
        stream_rate_lo = float(lo_raw) if lo_raw else stream_rate
        stream_rate_hi = float(hi_raw) if hi_raw else stream_rate
        stream_rate_source = "ECONOMICS_CALIBRATION.json"

    def _byte_delta_at(gb_s: float) -> float:
        if retime_organ:
            new_bytes = organ_bytes + net_bytes
            if new_bytes < 0:
                raise EconomicsRefuse("retime would produce negative organ bytes")
            return bytes_to_ms(new_bytes, gb_s) - organ_ms
        return bytes_to_ms(net_bytes, gb_s)

    if retime_organ:
        rate_for_terms = gb_s_nom
        byte_nom = _byte_delta_at(gb_s_nom)
        byte_at_lo = _byte_delta_at(gb_s_lo)
        byte_at_hi = _byte_delta_at(gb_s_hi)
        byte_removed_ms = -bytes_to_ms(removed, rate_for_terms)
        byte_added_ms = bytes_to_ms(added_total, rate_for_terms)
    elif not on_critical or stream_rate is None:
        rate_for_terms = 0.0
        byte_nom = 0.0
        byte_at_lo = 0.0
        byte_at_hi = 0.0
        byte_removed_ms = 0.0
        byte_added_ms = 0.0
    else:
        rate_for_terms = stream_rate
        byte_nom = _byte_delta_at(stream_rate)
        lo = stream_rate_lo if stream_rate_lo else stream_rate
        hi = stream_rate_hi if stream_rate_hi else stream_rate
        if lo > hi:
            lo, hi = hi, lo
        byte_at_lo = _byte_delta_at(lo)
        byte_at_hi = _byte_delta_at(hi)
        byte_removed_ms = -bytes_to_ms(removed, rate_for_terms)
        byte_added_ms = bytes_to_ms(added_total, rate_for_terms)
        # A stream-class rate must not invent more organ time than the organ
        # has. Slack covers the rounded organ_ms vs bytes/rate arithmetic.
        cap = organ_ms * 1.05
        if byte_nom < -cap:
            byte_nom = -organ_ms
        if byte_nom > cap:
            byte_nom = organ_ms

    if extra_flops_per_output_element < 0:
        raise EconomicsRefuse(
            f"extra_flops_per_output_element cannot be negative: "
            f"{extra_flops_per_output_element}"
        )
    if n_output_elements is None:
        n_output_elements = ORGAN_OUTPUT_ELEMENTS[organ]
    if extra_flops_per_output_element != 0.0 and n_output_elements <= 0:
        raise EconomicsRefuse(
            "extra FLOPs require n_output_elements > 0 (pass n_output_elements "
            "or a known organ)"
        )

    flop_nom = _flop_ms(extra_flops_per_output_element, n_output_elements, exposed_mult=1.0)
    flop_hidden = 0.0
    flop_exposed = _flop_ms(
        extra_flops_per_output_element, n_output_elements, exposed_mult=FLOP_EXPOSED_MULT
    )

    dclass = infer_dispatch_class(organ, dispatch_class)
    if dispatch_delta != 0.0 and dclass is None:
        # Class unknown: carry the measured class range as ASSUMPTION.
        us_nom = 0.5 * (MARGINAL_US_MLP_GQA_NORM + MARGINAL_US_DELTANET_BA)
        us_lo, us_hi = MARGINAL_US_DELTANET_BA, MARGINAL_US_MLP_GQA_NORM
        dispatch_assumption = (
            "ASSUMPTION: dispatch class not named; range is the two measured "
            f"classes ({MARGINAL_US_DELTANET_BA} us DeltaNet BA, "
            f"{MARGINAL_US_MLP_GQA_NORM} us MLP/GQA/norm fusion)"
        )
    elif dclass is None:
        us_nom = us_lo = us_hi = 0.0
        dispatch_assumption = "no dispatch delta"
    else:
        us_nom = float(DISPATCH_CLASSES[dclass]["us"])
        us_lo = us_hi = us_nom
        dispatch_assumption = (
            f"measured class {dclass} at {us_nom} us; class-dependent, "
            "not a single marginal"
        )
    dispatch_nom = float(dispatch_delta) * us_nom / 1000.0
    dispatch_at_lo = float(dispatch_delta) * us_lo / 1000.0
    dispatch_at_hi = float(dispatch_delta) * us_hi / 1000.0

    delta_nom = byte_nom + flop_nom + dispatch_nom

    # Range: bandwidth lo/hi × flop hidden/exposed × dispatch class lo/hi.
    # "Plausibly" for the S020 time bar uses the most negative (most
    # saving) combination inside the stated assumptions.
    candidates_delta = []
    for b in (byte_at_lo, byte_at_hi, byte_nom):
        for f in (flop_hidden, flop_nom, flop_exposed):
            for d in (dispatch_at_lo, dispatch_at_hi, dispatch_nom):
                candidates_delta.append(b + f + d)
    delta_lo = min(candidates_delta)
    delta_hi = max(candidates_delta)

    token_nom = CITED_TOKEN_MS + delta_nom
    token_fast = CITED_TOKEN_MS + delta_lo
    token_slow = CITED_TOKEN_MS + delta_hi
    # Host gap is a floor: this model does not delete host work.
    token_nom = max(token_nom, CITED_HOST_MS)
    token_fast = max(token_fast, CITED_HOST_MS)
    token_slow = max(token_slow, CITED_HOST_MS)

    predicted_tps = tps_from_ms(token_nom)
    tps_fast = tps_from_ms(token_fast)
    tps_slow = tps_from_ms(token_slow)

    plausible_ms_saved = max(0.0, -delta_lo)
    clears_time_bar = plausible_ms_saved >= S020_SECTION_20_BAR_MS

    live = True if status is None else status in LIVE_STATUSES
    reasons: list[str] = []
    if live and clears_time_bar:
        reasons.append("clears_s020_section_20_time_bar")
    if live and reusable_family:
        reasons.append("reusable_representation_family")
    if live and high_information_falsifier:
        reasons.append("high_information_falsifier")
    if not live:
        reasons.append(f"not_live:{status}")

    verdict = "MATERIAL" if (live and reasons and any(
        r in {
            "clears_s020_section_20_time_bar",
            "reusable_representation_family",
            "high_information_falsifier",
        }
        for r in reasons
    )) else "IMMATERIAL"

    bandwidth_is_assumption = (not regime["measured"]) or (
        regime_name
        in {
            "unknown_new_representation",
            "lm_head_demonstrated",
            "clean_gemv_roof",
            "affine_q2_saturated",
        }
        and organ != "lm_head"
    )
    if regime_name == "lm_head_demonstrated" and organ != "lm_head":
        bandwidth_is_assumption = True
    if regime_name == "clean_gemv_roof":
        bandwidth_is_assumption = True
    if regime_name == "unknown_new_representation":
        bandwidth_is_assumption = True
    # Even a measured organ rate is an ASSUMPTION when applied to a new
    # primitive: the new kernel may not sit on that curve.
    if consuming_primitive not in {
        "FusedDecodeCompute",
        "LayoutTransform",
        None,
    } or (
        consuming_primitive == "FusedDecodeCompute"
        and organ == "mlp"
        and regime_name not in {"production_mlp", "affine_q2_family"}
    ):
        bandwidth_is_assumption = True
    if consuming_primitive and consuming_primitive not in ATLAS_PRIMITIVES:
        bandwidth_is_assumption = True

    return {
        "ok": True,
        "id": candidate_id,
        "organ": organ,
        "consuming_primitive": consuming_primitive,
        "stream_class": stream_class,
        "status": status,
        "live": live,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "bytes_removed": removed,
        "bytes_added": added,
        "bytes_added_supplied": added_was_supplied,
        "net_bytes": net_bytes,
        "extra_flops_per_output_element": float(extra_flops_per_output_element),
        "n_output_elements": int(n_output_elements),
        "dispatch_delta": float(dispatch_delta),
        "retime_organ": retime_organ,
        "predicted_ms_delta": delta_nom,
        "predicted_ms_delta_range": [delta_lo, delta_hi],
        "predicted_ms_saved": -delta_nom,
        "predicted_token_ms": token_nom,
        "predicted_token_ms_range": [token_fast, token_slow],
        "predicted_tps": predicted_tps,
        "predicted_tps_range": [tps_slow, tps_fast],
        "terms": {
            "byte_ms_delta": byte_nom,
            "byte_removed_ms": byte_removed_ms,
            "byte_added_ms": byte_added_ms,
            "flop_ms_delta": flop_nom,
            "dispatch_ms_delta": dispatch_nom,
        },
        "assumptions": {
            "bandwidth_regime": regime_name,
            "bandwidth_gb_s_nominal": gb_s_nom,
            "bandwidth_gb_s_range": [gb_s_lo, gb_s_hi],
            "bandwidth_is_assumption": bandwidth_is_assumption,
            "bandwidth_note": regime["assumption"],
            "stream_class": stream_class,
            "stream_on_critical_path": on_critical,
            "stream_billing_gb_s": rate_for_terms,
            "stream_rate_source": stream_rate_source,
            "incremental_byte_rate_gb_s": (
                rate_for_terms if (retime_organ or on_critical) else 0.0
            ),
            "organ_average_gb_s": organ_gb_s,
            "flop_model": (
                "extra FLOPs scored as extra_flops * n_output_elements / "
                "EFFECTIVE_FLOP_S, the MLP GEMV-dressed rate "
                f"({EFFECTIVE_FLOP_S / 1e12:.3f} TFLOP/s). ASSUMPTION: same "
                "cost-per-FLOP as the incumbent MAC (mostly wait-on-memory). "
                f"Range [0, {FLOP_EXPOSED_MULT}x] covers hidden-under-bandwidth "
                "to decode-ALU-like."
            ),
            "flop_ms_range": [flop_hidden, flop_exposed],
            "dispatch_class": dclass,
            "dispatch_us_nominal": us_nom,
            "dispatch_us_range": [us_lo, us_hi],
            "dispatch_note": dispatch_assumption,
            "cited_token_ms": CITED_TOKEN_MS,
            "host_gap_floor_ms": CITED_HOST_MS,
        },
        "s020_section_20": {
            "bar_frac": S020_SECTION_20_BAR_FRAC,
            "bar_ms": S020_SECTION_20_BAR_MS,
            "plausible_ms_saved": plausible_ms_saved,
            "clears_time_bar": clears_time_bar,
            "reusable_family": reusable_family,
            "high_information_falsifier": high_information_falsifier,
        },
    }


def score_proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    """Score a recorded or ad-hoc proposal mapping."""
    kwargs: dict[str, Any] = {}
    if "bytes_removed" in row:
        kwargs["bytes_removed"] = row["bytes_removed"]
    if "bytes_added" in row:
        kwargs["bytes_added"] = row["bytes_added"]
    elif "bytes_removed" in row:
        kwargs["bytes_added"] = 0
    for key in (
        "extra_flops_per_output_element",
        "dispatch_delta",
        "consuming_primitive",
        "bandwidth_regime",
        "organ",
        "n_output_elements",
        "dispatch_class",
        "stream_class",
        "stream_gb_s",
        "stream_on_critical_path",
        "retime_organ",
        "reusable_family",
        "high_information_falsifier",
        "status",
    ):
        if key in row and row[key] is not _UNSET:
            kwargs[key] = row[key]
    kwargs["candidate_id"] = row.get("id")
    if "stream_class" not in kwargs or kwargs["stream_class"] is None:
        cid = row.get("id")
        if cid in STREAM_CLASS_BY_ID:
            kwargs["stream_class"] = STREAM_CLASS_BY_ID[cid]
        else:
            raise IncompleteEconomics(
                f"candidate {cid!r} does not declare a stream_class; "
                "defaulting to the organ average is refused"
            )
    return score(**kwargs)


# ---------------------------------------------------------------------------
# Place every candidate already on record onto the curve.
# ---------------------------------------------------------------------------


def _load_future(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise EconomicsRefuse(f"required receipt missing: {rel} ({path})")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise EconomicsRefuse(f"{rel} is not an object")
    return doc


def _proposal_from_receipt_row(
    row: Mapping[str, Any],
    *,
    source: str,
    organ: str,
    default_regime: str,
) -> dict[str, Any]:
    cid = str(row["id"])
    status = str(row.get("status") or "OPEN")
    if cid in GRANULARITY_REFUTED_IDS:
        status = "GRANULARITY_REFUTED"
    if cid not in STREAM_CLASS_BY_ID:
        raise EconomicsRefuse(
            f"recorded candidate {cid} has no stream_class; refusing to default"
        )
    stream_class = STREAM_CLASS_BY_ID[cid]
    prim = row.get("physical_primitive")
    if prim is not None:
        prim = str(prim)
    raw_elim = row.get("bytes_eliminated_if_true")
    if raw_elim is None:
        removed = 0
        incomplete = True
    else:
        removed = int(raw_elim)
        incomplete = False
        if removed < 0:
            # Smaller group size grows the auxiliary. Not an attack.
            removed = 0
            incomplete = True

    added: dict[str, int] | int = 0
    remat = row.get("dense_rematerialization")
    if status == "REJECTED_DENSE_REMAT":
        params = ORGAN_PARAMS.get(organ)
        if params:
            added = {"generator": int(params) * F16_BYTES}

    dispatch_delta = float(row.get("dispatch_delta") or 0.0)
    extra_flops = float(row.get("extra_flops_per_output_element") or 0.0)
    retime = bool(row.get("retime_organ") or False)
    regime = row.get("bandwidth_regime") or default_regime
    dclass = row.get("dispatch_class")

    return {
        "id": cid,
        "name": row.get("name"),
        "source": source,
        "status": status,
        "organ": organ,
        "consuming_primitive": prim,
        "bytes_removed": removed,
        "bytes_added": added,
        "byte_model_incomplete": incomplete,
        "stream_class": stream_class,
        "extra_flops_per_output_element": extra_flops,
        "dispatch_delta": dispatch_delta,
        "dispatch_class": dclass,
        "bandwidth_regime": regime,
        "retime_organ": retime,
        "reusable_family": cid in REUSABLE_FAMILY_IDS,
        "high_information_falsifier": cid in HIGH_INFORMATION_FALSIFIER_IDS,
        "capability": row.get("capability"),
        "dense_rematerialization": remat,
    }


def recorded_proposals() -> list[dict[str, Any]]:
    """Every candidate already on record, plus the surviving size lever
    and the group-size curve points causal_budget ranked separately.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(p: dict[str, Any]) -> None:
        cid = p["id"]
        if cid in seen:
            raise EconomicsRefuse(f"duplicate recorded candidate id {cid}")
        seen.add(cid)
        rows.append(p)

    aux = _load_future(AUX_REL)
    for cand in aux.get("candidates") or []:
        _add(
            _proposal_from_receipt_row(
                cand,
                source=AUX_REL,
                organ="mlp",
                default_regime="affine_q2_family",
            )
        )
    for g, eliminated in GROUP_SIZE_EXTRA:
        gid = f"group_size_{g}"
        gstatus = "GRANULARITY_REFUTED" if gid in GRANULARITY_REFUTED_IDS else "OPEN"
        _add(
            {
                "id": gid,
                "name": f"MLP affine-Q2 group size {g} (byte curve exact; capability {'REFUTED' if gstatus != 'OPEN' else 'UNMEASURED'})",
                "source": AUX_REL,
                "status": gstatus,
                "organ": "mlp",
                "consuming_primitive": "FusedDecodeCompute",
                "bytes_removed": eliminated,
                "bytes_added": 0,
                "byte_model_incomplete": False,
                "stream_class": STREAM_CLASS_BROADCAST_AUX,
                "extra_flops_per_output_element": 0.0,
                "dispatch_delta": 0.0,
                "dispatch_class": "mlp_gqa_norm_fusion",
                "bandwidth_regime": "affine_q2_family",
                "retime_organ": False,
                "reusable_family": True,
                "high_information_falsifier": False,
                "capability": "FAILED_HELDOUT" if gstatus != "OPEN" else "UNMEASURED",
                "dense_rematerialization": "DIRECT_CONSUME",
            }
        )

    code = _load_future(CODE_REL)
    for cand in code.get("candidates") or []:
        _add(
            _proposal_from_receipt_row(
                cand,
                source=CODE_REL,
                organ="mlp",
                default_regime="affine_q2_family",
            )
        )

    dn = _load_future(DN_REL)
    for cand in dn.get("candidates") or []:
        p = _proposal_from_receipt_row(
            cand,
            source=DN_REL,
            organ="deltanet",
            default_regime="production_deltanet",
        )
        if p["id"] == "fused_update_consume":
            p["dispatch_delta"] = -48.0
            p["dispatch_class"] = "deltanet_ba"
            p["name"] = cand.get("name") or p.get("name")
        if p["consuming_primitive"] is None:
            p["bandwidth_regime"] = "unknown_new_representation"
        _add(p)

    # The size sweep killed "concatenate to 338 MB and reach 497". What
    # survives is the 5 MB -> 20 MB amortization on THIS kernel family,
    # and the fact that isolated affine-Q2 saturates near 377.
    _add(
        {
            "id": "surviving_dispatch_size_amortize_sub20mb",
            "name": (
                "amortize affine-Q2 launches from ~5 MB (223 GB/s) to >= 20 MB "
                "(326 GB/s); the only size effect the sweep left standing"
            ),
            "source": SWEEP_REL,
            "status": "OPEN",
            "organ": "mlp",
            "consuming_primitive": "FusedDecodeCompute",
            "bytes_removed": 0,
            "bytes_added": 0,
            "byte_model_incomplete": False,
            "stream_class": STREAM_CLASS_WEIGHT_CODES,
            "extra_flops_per_output_element": 0.0,
            "dispatch_delta": 0.0,
            "dispatch_class": "mlp_gqa_norm_fusion",
            "bandwidth_regime": "affine_q2_unamortized",
            "retime_organ": False,
            "reusable_family": True,
            "high_information_falsifier": True,
            "capability": "MEASURED_ON_AFFINE_Q2",
            "dense_rematerialization": "DIRECT_CONSUME",
            "note": (
                "Production organs already average >= 8.8 MB/dispatch and sit "
                "in the 342-360 cluster, so this is not a save on today's "
                "graph. It is a constraint on new representations: do not "
                "shard into 5 MB launches of this kernel family. MATERIAL "
                "as a reusable family / high-information falsifier, not as "
                "a 1% token-time lever on the current token."
            ),
        }
    )
    _add(
        {
            "id": "dispatch_size_concat_to_lm_head_mb",
            "name": (
                "concatenate affine-Q2 working set to the LM head's 337.7 MB "
                "in hope of 497.4 GB/s"
            ),
            "source": SWEEP_REL,
            "status": "PER_DISPATCH_SIZE_REFUTED",
            "organ": "mlp",
            "consuming_primitive": "FusedDecodeCompute",
            "bytes_removed": 0,
            "bytes_added": 0,
            "byte_model_incomplete": False,
            "stream_class": STREAM_CLASS_WEIGHT_CODES,
            "extra_flops_per_output_element": 0.0,
            "dispatch_delta": 0.0,
            "dispatch_class": "mlp_gqa_norm_fusion",
            "bandwidth_regime": "affine_q2_saturated",
            "retime_organ": True,
            "reusable_family": False,
            "high_information_falsifier": True,
            "capability": "REFUTED_AS_497_PATH",
            "dense_rematerialization": "DIRECT_CONSUME",
            "note": (
                "The sweep reached 374 GB/s at 338 MB and 377 GB/s at 700 MB, "
                "not 497. Layer-N+1 dependency means production cannot take "
                "the isolated concat. Scored at the measured affine-Q2 "
                "ceiling so the leftover 344->377 is not re-proposed as 497."
            ),
        }
    )
    return rows


def _compact(score_row: dict[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    s20 = score_row["s020_section_20"]
    assumptions = score_row["assumptions"]
    return {
        "id": proposal["id"],
        "name": proposal.get("name"),
        "source": proposal.get("source"),
        "status": proposal.get("status"),
        "live": score_row["live"],
        "organ": score_row["organ"],
        "consuming_primitive": score_row["consuming_primitive"],
        "stream_class": score_row["stream_class"],
        "bytes_removed": score_row["bytes_removed"],
        "bytes_added_total": int(score_row["bytes_added"].get("total", 0)),
        "bytes_added": {
            k: int(score_row["bytes_added"].get(k, 0)) for k in BYTES_ADDED_FIELDS
        },
        "byte_model_incomplete": bool(proposal.get("byte_model_incomplete")),
        "net_bytes": score_row["net_bytes"],
        "extra_flops_per_output_element": score_row["extra_flops_per_output_element"],
        "dispatch_delta": score_row["dispatch_delta"],
        "verdict": score_row["verdict"],
        "verdict_reasons": list(score_row["verdict_reasons"]),
        "predicted_ms_delta": _r(score_row["predicted_ms_delta"], 4),
        "predicted_ms_delta_range": [
            _r(score_row["predicted_ms_delta_range"][0], 4),
            _r(score_row["predicted_ms_delta_range"][1], 4),
        ],
        "predicted_ms_saved": _r(score_row["predicted_ms_saved"], 4),
        "predicted_token_ms": _r(score_row["predicted_token_ms"], 4),
        "predicted_token_ms_range": [
            _r(score_row["predicted_token_ms_range"][0], 4),
            _r(score_row["predicted_token_ms_range"][1], 4),
        ],
        "predicted_tps": _r(score_row["predicted_tps"], 3),
        "predicted_tps_range": [
            _r(score_row["predicted_tps_range"][0], 3),
            _r(score_row["predicted_tps_range"][1], 3),
        ],
        "terms": {k: _r(v, 4) for k, v in score_row["terms"].items()},
        "assumptions": {
            "bandwidth_regime": assumptions["bandwidth_regime"],
            "bandwidth_gb_s_nominal": _r(assumptions["bandwidth_gb_s_nominal"], 2),
            "bandwidth_gb_s_range": [
                _r(assumptions["bandwidth_gb_s_range"][0], 2),
                _r(assumptions["bandwidth_gb_s_range"][1], 2),
            ],
            "bandwidth_is_assumption": assumptions["bandwidth_is_assumption"],
            "bandwidth_note": assumptions["bandwidth_note"],
            "flop_ms_range": [
                _r(assumptions["flop_ms_range"][0], 4),
                _r(assumptions["flop_ms_range"][1], 4),
            ],
            "dispatch_class": assumptions["dispatch_class"],
            "dispatch_us_nominal": _r(assumptions["dispatch_us_nominal"], 3),
            "dispatch_note": assumptions["dispatch_note"],
            "stream_class": assumptions.get("stream_class"),
            "stream_on_critical_path": assumptions.get("stream_on_critical_path"),
            "stream_billing_gb_s": _r(float(assumptions.get("stream_billing_gb_s") or 0.0), 2),
            "organ_average_gb_s": _r(float(assumptions.get("organ_average_gb_s") or 0.0), 2),
        },
        "s020_section_20": {
            "bar_ms": _r(s20["bar_ms"], 4),
            "plausible_ms_saved": _r(s20["plausible_ms_saved"], 4),
            "clears_time_bar": s20["clears_time_bar"],
            "reusable_family": s20["reusable_family"],
            "high_information_falsifier": s20["high_information_falsifier"],
        },
        "capability": proposal.get("capability"),
        "note": proposal.get("note"),
    }


def _legacy_organ_average(p: Mapping[str, Any]) -> dict[str, Any]:
    """The old model: unique bytes at the organ average, stream ignored."""
    organ = str(p.get("organ") or "mlp")
    return score_proposal(
        {
            **dict(p),
            "stream_class": p.get("stream_class") or STREAM_CLASS_WEIGHT_CODES,
            "stream_gb_s": ORGAN_GB_S[organ],
            "stream_on_critical_path": True,
        }
    )


def rank_recorded() -> list[dict[str, Any]]:
    proposals = recorded_proposals()
    scored: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for p in proposals:
        s = score_proposal(p)
        legacy = _legacy_organ_average(p)
        scored.append((p, s, legacy))
    # Economic curve: largest predicted save first. Stable by id.
    scored.sort(key=lambda pair: (-pair[1]["predicted_ms_saved"], pair[0]["id"]))
    out = []
    for rank, (p, s, legacy) in enumerate(scored, start=1):
        row = _compact(s, p)
        row["rank"] = rank
        row["legacy_organ_average_ms_saved"] = _r(legacy["predicted_ms_saved"], 4)
        row["delta_from_legacy_ms_saved"] = _r(
            s["predicted_ms_saved"] - legacy["predicted_ms_saved"], 4
        )
        out.append(row)
    return out


def retrodictions(by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Predicted vs measured for the three real A/Bs that already exist."""
    # AUX_U8_LUT: 0.535 GB broadcast aux, extra FLOPs 0, LUT tables 393216 B.
    # Token projection from the layer A/B: +0.504 ms (SLOWER).
    lut = score(
        bytes_removed=534_773_760,
        bytes_added={"metadata": 393_216},
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=STREAM_CLASS_BROADCAST_AUX,
        candidate_id="quantize_aux_u8_lut",
        reusable_family=True,
    )
    lut_legacy = score(
        bytes_removed=534_773_760,
        bytes_added={"metadata": 393_216},
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=STREAM_CLASS_BROADCAST_AUX,
        stream_gb_s=MLP_GB_S,
        stream_on_critical_path=True,
        candidate_id="quantize_aux_u8_lut_legacy",
    )
    lut_measured_ms_delta = 0.504064  # slower; AUX_U8_LUT token projection
    lut_predicted_is_win = lut["predicted_ms_delta"] < 0.0

    # DELTANET_WIDEN_AB: layout/dispatch, not a byte lever. dispatch_delta=-48.
    widen = score(
        bytes_removed=0,
        bytes_added=0,
        dispatch_delta=-48.0,
        organ="deltanet",
        consuming_primitive="LocalStateMachine",
        dispatch_class="deltanet_ba",
        stream_class=STREAM_CLASS_ACTIVATION,
        candidate_id="deltanet_widen_ab",
    )
    widen_measured_ms_delta = -1.0245

    # MLP_DECODE_CHEAPEN fold_addqx: bit-identical, 329.2 -> 370.9 GB/s, 0 bytes.
    cheapen = score(
        bytes_removed=0,
        bytes_added=0,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=STREAM_CLASS_WEIGHT_CODES,
        candidate_id="mlp_decode_cheapen_fold_addqx",
    )
    cheapen_measured_ms_delta = -1.745  # projection from MLP_DECODE_CHEAPEN

    def _row(
        ident: str,
        predicted: Mapping[str, Any],
        measured_ms_delta: float,
        *,
        how: str,
        note: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pred = float(predicted["predicted_ms_delta"])
        out = {
            "id": ident,
            "how_scored": how,
            "stream_class": predicted["stream_class"],
            "bytes_removed": predicted["bytes_removed"],
            "predicted_ms_delta": _r(pred, 4),
            "predicted_ms_saved": _r(predicted["predicted_ms_saved"], 4),
            "measured_ms_delta": _r(measured_ms_delta, 4),
            "residual_ms": _r(pred - measured_ms_delta, 4),
            "predicted_is_win": pred < 0.0,
            "measured_is_win": measured_ms_delta < 0.0,
            "note": note,
        }
        if extra:
            out.update(extra)
        return out

    return [
        _row(
            "DELTANET_WIDEN_AB",
            widen,
            widen_measured_ms_delta,
            how="dispatch_delta=-48 at deltanet_ba 2.884 us; bytes_removed=0",
            note=(
                "The win is a kernel/layout change (widen_f4) plus a 48-launch "
                "fold, not a unique-byte removal. Sign agrees (a win); the "
                "byte model does not claim the 1.0245 ms."
            ),
            extra={"source": "receipts/future/DELTANET_WIDEN_AB.json"},
        ),
        _row(
            "MLP_DECODE_CHEAPEN",
            cheapen,
            cheapen_measured_ms_delta,
            how="bytes_removed=0, extra_flops=0, stream_class=weight_codes",
            note=(
                "fold_addqx is a decode-arithmetic cheapening (329.2 -> 370.9 "
                "GB/s, bit-identical). The byte model predicting 0 is correct "
                "for THIS defect: no unique bytes moved."
            ),
            extra={
                "source": "receipts/future/MLP_DECODE_CHEAPEN.json",
                "measured_gb_s": {"production": 329.2, "fold_addqx": 370.9},
            },
        ),
        _row(
            "AUX_U8_LUT",
            lut,
            lut_measured_ms_delta,
            how=(
                "bytes_removed=534773760 broadcast_aux, bytes_added.metadata="
                "393216, extra_flops=0"
            ),
            note=(
                "The case that matters. Old organ-average billed "
                f"{lut_legacy['predicted_ms_saved']:.4f} ms saved. Stream-class "
                f"bills {lut['predicted_ms_saved']:.4f} ms. Measured is slower "
                "(+0.504 ms token projection). Predicted is not a win."
            ),
            extra={
                "source": "receipts/future/AUX_U8_LUT.json",
                "legacy_organ_average_ms_saved": _r(lut_legacy["predicted_ms_saved"], 4),
                "legacy_predicted_is_win": lut_legacy["predicted_ms_delta"] < 0.0,
                "predicted_is_win": lut_predicted_is_win,
                "must_predict_not_a_win": True,
            },
        ),
        {
            "id": "quantize_aux_u8_catalog",
            "how_scored": "recorded catalog candidate, stream_class=broadcast_aux",
            "stream_class": STREAM_CLASS_BROADCAST_AUX,
            "delta_from_legacy_ms_saved": by_id.get("quantize_aux_u8", {}).get(
                "delta_from_legacy_ms_saved"
            ),
            "predicted_ms_saved": by_id.get("quantize_aux_u8", {}).get("predicted_ms_saved"),
            "legacy_organ_average_ms_saved": by_id.get("quantize_aux_u8", {}).get(
                "legacy_organ_average_ms_saved"
            ),
            "note": "live catalog row, same stream as AUX_U8_LUT",
        },
        {
            "id": "entropy_coded_code_stream",
            "how_scored": "recorded catalog candidate, stream_class=weight_codes (mlp_entropy_floor)",
            "stream_class": STREAM_CLASS_WEIGHT_CODES,
            "delta_from_legacy_ms_saved": by_id.get("entropy_coded_code_stream", {}).get(
                "delta_from_legacy_ms_saved"
            ),
            "predicted_ms_saved": by_id.get("entropy_coded_code_stream", {}).get(
                "predicted_ms_saved"
            ),
            "legacy_organ_average_ms_saved": by_id.get("entropy_coded_code_stream", {}).get(
                "legacy_organ_average_ms_saved"
            ),
            "note": "0.278 GB of binding code bytes; should remain a real save",
        },
        {
            "id": "group_size_1024",
            "how_scored": "recorded catalog candidate, stream_class=broadcast_aux, status GRANULARITY_REFUTED",
            "stream_class": STREAM_CLASS_BROADCAST_AUX,
            "delta_from_legacy_ms_saved": by_id.get("group_size_1024", {}).get(
                "delta_from_legacy_ms_saved"
            ),
            "predicted_ms_saved": by_id.get("group_size_1024", {}).get("predicted_ms_saved"),
            "legacy_organ_average_ms_saved": by_id.get("group_size_1024", {}).get(
                "legacy_organ_average_ms_saved"
            ),
            "live": by_id.get("group_size_1024", {}).get("live"),
            "note": "capability already REFUTED; byte claim re-priced for a consistent record",
        },
        {
            "id": "group_size_256",
            "how_scored": "recorded catalog candidate, stream_class=broadcast_aux, status GRANULARITY_REFUTED",
            "stream_class": STREAM_CLASS_BROADCAST_AUX,
            "delta_from_legacy_ms_saved": by_id.get("group_size_256", {}).get(
                "delta_from_legacy_ms_saved"
            ),
            "predicted_ms_saved": by_id.get("group_size_256", {}).get("predicted_ms_saved"),
            "legacy_organ_average_ms_saved": by_id.get("group_size_256", {}).get(
                "legacy_organ_average_ms_saved"
            ),
            "live": by_id.get("group_size_256", {}).get("live"),
            "note": "capability already REFUTED; byte claim re-priced for a consistent record",
        },
    ]


def build() -> dict[str, Any]:
    ranked = rank_recorded()
    live_material = [r["id"] for r in ranked if r["live"] and r["verdict"] == "MATERIAL"]
    live_immaterial = [
        r["id"] for r in ranked if r["live"] and r["verdict"] == "IMMATERIAL"
    ]
    dead = [r["id"] for r in ranked if not r["live"]]
    by_id = {r["id"]: r for r in ranked}

    # Pin the two load-bearing guards into the receipt so a closer that
    # dropped them is visible without rerunning pytest.
    refused = False
    try:
        score(bytes_removed=1)
    except IncompleteEconomics:
        refused = True

    stream_refused = False
    try:
        score(bytes_removed=1_000_000, bytes_added=0, organ="mlp")
    except IncompleteEconomics as exc:
        stream_refused = "stream_class" in str(exc)

    both_terms = score(
        bytes_removed=MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=100.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=STREAM_CLASS_WEIGHT_CODES,
        stream_gb_s=MLP_GB_S,
        stream_on_critical_path=True,
    )
    bytes_only = score(
        bytes_removed=MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=STREAM_CLASS_WEIGHT_CODES,
        stream_gb_s=MLP_GB_S,
        stream_on_critical_path=True,
    )

    cal = load_calibration()
    retro = retrodictions(by_id)

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "s020_section_20_bar": {
            "frac_of_complete_token_time": S020_SECTION_20_BAR_FRAC,
            "bar_ms": _r(S020_SECTION_20_BAR_MS, 4),
            "cited_token_ms": CITED_TOKEN_MS,
            "or_reusable_representation_family": True,
            "or_high_information_falsifier": True,
            "verbatim": (
                "a candidate deserves substantial work if it plausibly "
                "removes >= 1% of complete token time, or creates a "
                "reusable representation family, or provides a "
                "high-information falsifier"
            ),
        },
        "measured_constants_cited": {
            "token_ms": CITED_TOKEN_MS,
            "host_ms": CITED_HOST_MS,
            "gpu_ms": CITED_GPU_MS,
            "tps_71_needs_token_ms": _r(TPS_71_TOKEN_MS, 3),
            "tps_71_needs_gpu_ms": _r(TPS_71_GPU_MS, 3),
            "gpu_reduction_for_71_at_current_executor": _r(GPU_REDUCTION_FOR_71, 4),
            "affine_q2_gb_s_at_5mb": AFFINE_Q2_GB_S_AT_5MB,
            "affine_q2_gb_s_at_20mb": AFFINE_Q2_GB_S_AT_20MB,
            "affine_q2_gb_s_at_338mb": AFFINE_Q2_GB_S_AT_338MB,
            "affine_q2_gb_s_at_700mb": AFFINE_Q2_GB_S_AT_700MB,
            "affine_q2_saturated_gb_s": AFFINE_Q2_SATURATED_GB_S,
            "lm_head_gb_s": LM_HEAD_GB_S,
            "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
            "mlp_code_independent_fraction": 0.93509,
            "marginal_dispatch_us_mlp_gqa_norm": MARGINAL_US_MLP_GQA_NORM,
            "marginal_dispatch_us_deltanet_ba": MARGINAL_US_DELTANET_BA,
            "organs": [
                {
                    "organ": o["organ"],
                    "bytes": ORGAN_BYTES[o["organ"]],
                    "ms": o["ms"],
                    "gb_s": o["gb_s"],
                    "dispatches": o["dispatches"],
                }
                for o in cb.ORGANS
            ],
            "mlp_output_elements": ORGAN_OUTPUT_ELEMENTS["mlp"],
            "effective_flop_s_mlp_dressed": EFFECTIVE_FLOP_S,
        },
        "model": {
            "formula": (
                "predicted_token_ms = cited_token_ms "
                "+ ms(net_bytes, stream_class_rate) "
                "+ extra_flops * n_output_elements / EFFECTIVE_FLOP_S * 1000 "
                "+ dispatch_delta * class_us / 1000; "
                "net_bytes = bytes_added.total - bytes_removed; "
                "stream_class_rate is the calibrated unique-byte marginal of "
                "the declared stream (0 if that stream is not on the critical "
                "path); retime_organ still uses the named regime; "
                "the regime is an ASSUMPTION with a range"
            ),
            "refuses": [
                "bytes_removed without bytes_added",
                "stream_class undeclared (no default to the organ average)",
                "unknown stream_class",
            ],
            "stream_classes": cal.get("stream_classes"),
            "calibration_receipt": f"receipts/future/{CALIBRATION_RECEIPT}",
            "primary_marginal_is_50pct_drop": True,
            "bandwidth_regimes": {
                name: {
                    "gb_s_lo": spec["gb_s_lo"],
                    "gb_s_hi": spec["gb_s_hi"],
                    "gb_s_nominal": spec["gb_s_nominal"],
                    "measured": spec["measured"],
                    "assumption": spec["assumption"],
                }
                for name, spec in BANDWIDTH_REGIMES.items()
            },
            "dispatch_classes": DISPATCH_CLASSES,
            "flop_exposed_multiplier": FLOP_EXPOSED_MULT,
            "bytes_added_fields": list(BYTES_ADDED_FIELDS),
        },
        "guards": {
            "bytes_removed_without_bytes_added_refused": refused,
            "stream_class_undeclared_refused": stream_refused,
            "mlp_full_removal_byte_ms": _r(bytes_only["terms"]["byte_ms_delta"], 4),
            "mlp_full_removal_flop_ms_at_100": _r(both_terms["terms"]["flop_ms_delta"], 4),
            "mlp_full_removal_scores_both_terms": (
                both_terms["terms"]["byte_ms_delta"] < 0.0
                and both_terms["terms"]["flop_ms_delta"] > 0.0
                and abs(
                    both_terms["terms"]["byte_ms_delta"]
                    - bytes_only["terms"]["byte_ms_delta"]
                )
                < 1e-9
                and both_terms["predicted_ms_delta"] > bytes_only["predicted_ms_delta"]
            ),
            "aux_u8_lut_predicted_not_a_win": (
                not next(r for r in retro if r["id"] == "AUX_U8_LUT")["predicted_is_win"]
            ),
        },
        "calibration": {
            "schema": cal.get("schema"),
            "metal_device": cal.get("metal_device"),
            "loadavg": cal.get("loadavg"),
            "layer": cal.get("layer"),
            "finding": cal.get("finding"),
            "stream_classes": {
                name: {
                    "on_critical_path": spec.get("on_critical_path"),
                    "probe_on_critical_path": spec.get("probe_on_critical_path"),
                    "billing_gb_s": spec.get("billing_gb_s"),
                    "ms_per_gb_saved": spec.get("ms_per_gb_saved"),
                    "primary_rung": spec.get("primary_rung"),
                    "paired_dt_ns_at_50pct": spec.get("paired_dt_ns_at_50pct"),
                    "catalog_billing": spec.get("catalog_billing"),
                }
                for name, spec in (cal.get("stream_classes") or {}).items()
            },
        },
        "retrodictions": retro,
        "n_candidates": len(ranked),
        "n_live_material": len(live_material),
        "n_live_immaterial": len(live_immaterial),
        "n_dead": len(dead),
        "live_material_ranked": live_material,
        "live_immaterial_ranked": live_immaterial,
        "candidates_ranked": ranked,
        "what_the_model_cannot_know": [
            "whether a new representation runs in the affine-Q2 family, the LM-head Q4 regime, or somewhere off both curves — that is an ASSUMPTION with a range",
            "capability: no generate gate lives here. A MATERIAL byte lever can still be generation-incoherent",
            "actual_read_bytes_per_token: the catalog sum is not what the GPU read (no Metal memory counter on this box)",
            "extra decode ALU of a codec that is not the incumbent: the FLOP term is GEMV-dressed, range [0, 4x]",
            "whether remaining bytes keep the organ's measured rate or retiming the whole organ is the right counterfactual — retime_organ is an explicit switch",
            "host work a new generator might add; the host gap is a floor, not a claim that host stays 0.989 ms",
            "DeltaNet recurrent-state unique-byte traffic: the activation class is calibrated on MLP x, and catalog-scale state bills at the organ average rather than at the unique-x 2 GB/s number",
        ],
        "sources": [
            AUX_REL,
            CODE_REL,
            DN_REL,
            SWEEP_REL,
            BA_REL,
            f"receipts/future/{CALIBRATION_RECEIPT}",
            "receipts/future/AUX_U8_LUT.json",
            "receipts/future/DELTANET_WIDEN_AB.json",
            "receipts/future/MLP_DECODE_CHEAPEN.json",
            "receipts/future/AUX_CAPABILITY_SCREEN.json",
        ],
        "top_live_material": [
            {
                "id": cid,
                "predicted_ms_saved": by_id[cid]["predicted_ms_saved"],
                "predicted_tps": by_id[cid]["predicted_tps"],
                "verdict_reasons": by_id[cid]["verdict_reasons"],
                "status": by_id[cid]["status"],
            }
            for cid in live_material[:12]
        ],
    }


def record() -> Path:
    return write_receipt(RECEIPT, build(), RECORDED_BY)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Input:")[0].strip())
    parser.add_argument("--record", action="store_true", help=f"write receipts/future/{RECEIPT}")
    parser.add_argument("--build", action="store_true", help="alias of --record")
    parser.add_argument(
        "--calibrate-from",
        dest="calibrate_from",
        default=None,
        help=f"raw stream_criticality_probe JSON → receipts/future/{CALIBRATION_RECEIPT}",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.calibrate_from:
        raw_path = Path(args.calibrate_from)
        if not raw_path.is_absolute():
            raw_path = REPO / raw_path
        raw = load_json(raw_path)
        path = record_calibration(raw)
        print(f"wrote {path}")
        cal = load_calibration(force=True)
        for name, spec in (cal.get("stream_classes") or {}).items():
            print(
                f"  {name:16s} critical={spec.get('on_critical_path')}  "
                f"billing_gb_s={spec.get('billing_gb_s')}  "
                f"ms/GB={spec.get('ms_per_gb_saved')}  "
                f"dt50={spec.get('paired_dt_ns_at_50pct')} ns"
            )
        if not (args.record or args.build):
            return 0
    doc = build()
    if args.record or args.build:
        path = record()
        print(f"wrote {path}")
    print(
        f"  {doc['n_candidates']} on the curve; "
        f"{doc['n_live_material']} live MATERIAL; "
        f"{doc['n_live_immaterial']} live IMMATERIAL; "
        f"{doc['n_dead']} dead"
    )
    for row in doc["top_live_material"][:8]:
        print(
            f"  {row['predicted_ms_saved']:+7.3f} ms  "
            f"{row['predicted_tps']:6.2f} pred-TPS  "
            f"{row['status']:16s}  {row['id']}"
        )
    print("  retrodictions:")
    for r in doc.get("retrodictions") or []:
        if "predicted_ms_delta" not in r:
            continue
        print(
            f"    {r['id']:22s} pred {r['predicted_ms_delta']:+7.4f} ms  "
            f"meas {r['measured_ms_delta']:+7.4f} ms  "
            f"win_pred={r['predicted_is_win']} win_meas={r['measured_is_win']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
