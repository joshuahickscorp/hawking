"""RESIDENT 71 TPS CAUSAL BUDGET — static dispatch count + byte-fraction ladder.

This sidecar answers three questions from source and cited receipts, then
stops: how many dispatches the current decode path launches per decoded
token, which of the named 28.17 ms/token categories are actually evidenced,
and what byte-fraction arithmetic the 50/71/100/125/150 ladder requires at
the ~703 GB/s clean roof.

It does not take a GPU lease, flock a bench lock, re-measure a GB/s figure,
or coerce an UNKNOWN category into a number so the token arithmetic closes.

    python3 tools/future/tps_budget.py --record
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import (
    HARDWARE_FIELDS,
    REPO,
    HardwareClaimError,
    _assert_no_hardware_claims,
    load_json,
    write_receipt,
)

# G129: this used to be RESIDENT_71TPS_CAUSAL_BUDGET.json, the SAME PATH
# causal_budget_71.py writes with a different schema. The later writer won and
# silently destroyed every citation that resolved against `ladder[]` and
# `measured_now` - four rows of the roof-anchor audit stopped resolving and the
# module honestly recorded "field is not a resolvable path in this receipt"
# about a field that HAD been resolvable. Two producers, one path, no collision
# check. This one now writes its own.
RECEIPT = "RESIDENT_71TPS_BUDGET_STATIC.json"
SCHEMA = "hawking.future.tps_budget.v1"
VERSION = 1
RECORDED_BY = "tools/future/tps_budget.py"

EVIDENCE_CLASS = "STATIC_ONLY"
UNKNOWN = "UNKNOWN"
UNATTRIBUTED = "UNATTRIBUTED"
HYPOTHESIS = "HYPOTHESIS"
DERIVED = "DERIVED_FROM_ESTABLISHED_ARITHMETIC"
CITED = "CITED_FROM_NAMED_RECEIPT"
UNREACHABLE = "ARITHMETICALLY_UNREACHABLE_BY_MLP_ALONE"
REQUIRES_MLP = "REQUIRES_MLP_REDUCTION"
NO_MLP_CUT = "REACHABLE_WITHOUT_MLP_REDUCTION"

GEOMETRY_SRC = "crates/hawking-core/src/model/qwen38_geometry.rs"
DECODE_SRC = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SCHEDULE_SRC = "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
LEDGER_SRC = "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
RESIDENT_SRC = "crates/hawking-core/examples/ascension_qwen38_resident.rs"
KERNELS_SRC = "crates/hawking-core/src/kernels/mod.rs"

TPS_GAP_REL = "receipts/future/TPS_GAP.json"
ADDRESSING_GAP_REL = "receipts/future/ADDRESSING_GAP.json"
DISPATCH_CEREMONY_REL = "receipts/future/DISPATCH_CEREMONY.json"
CATALOG_ADDRESSING_REL = "receipts/future/CATALOG_ADDRESSING.json"
KERNEL_GEOMETRY_REL = "receipts/future/KERNEL_GEOMETRY.json"

# Lane-established quantities. Copied as citations, not re-measured.
# ADDRESSING_GAP refused 703.5 as the 13.6 GB addr-probe *median* (median
# 699.57, max 703.61). This budget still uses 703.5 as the named clean roof
# the lane established for the ladder; it does not re-adjudicate that roof.
ESTABLISHED_DECODE_RATE = 35.5
ESTABLISHED_ACTIVE_BYTES = 9_878_901_136
ESTABLISHED_EFFECTIVE_GB_S = 337.3
ESTABLISHED_CLEAN_ROOF_GB_S = 703.5
ESTABLISHED_PUBLISHED_PEAK_GB_S = 819.0
ESTABLISHED_MLP_FRACTION = 0.54
ESTABLISHED_NON_MLP_FRACTION = 0.46
ESTABLISHED_MARGINAL_DISPATCH_US = 15.0
MILESTONES = (50, 71, 100, 125, 150)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Dispatch counts are "
    "re-derived from encode/dispatch call sites in crates/hawking-core "
    "(decode path), not copied from a receipt as the answer. Byte counts and "
    "GB/s roofs are copied from named prior receipts or from the lane's "
    "established facts; arithmetic over those copies is DERIVED, not a new "
    "experiment. UNKNOWN categories stay the string UNKNOWN and are never "
    "coerced to 0 to close the 28.17 ms sum. gpu_authority is false. "
    "evidence_class is STATIC_ONLY. This module refuses to emit a hardware "
    "measurement claim it did not derive."
)


class BudgetRefuse(ValueError):
    """Contract violation: UNKNOWN coerced, unsourced hardware claim, or fold."""


# ---------------------------------------------------------------------------
# Guards. A closer nobody has watched refuse is not a closer.
# ---------------------------------------------------------------------------


def as_ms(value: Any, *, what: str) -> float:
    """UNKNOWN/UNATTRIBUTED/None is not a duration. Coercing it is the failure."""
    if value is None or value == UNKNOWN or value == UNATTRIBUTED:
        raise BudgetRefuse(
            f"{what} is {value!r}; refusing to coerce UNKNOWN to a number"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetRefuse(f"{what} is {value!r}, not a duration in ms")
    if value < 0:
        raise BudgetRefuse(f"{what} is {value!r}; a duration cannot be negative")
    return float(value)


def close_arithmetic_by_zeroing_unknown(
    categories: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]],
    token_ms: float,
) -> dict[str, Any]:
    """NEGATIVE CONTROL target: UNKNOWN must not become 0 to make the sum pretty."""
    raise BudgetRefuse(
        f"refusing to close {token_ms} ms by treating UNKNOWN categories as 0; "
        "an honest UNATTRIBUTED remainder is the correct answer"
    )


def refuse_hardware_measurement(name: str) -> None:
    """This sidecar has no GPU authority and did not take a measurement."""
    raise HardwareClaimError(
        f"{name}: sidecar has no GPU authority and did not derive a hardware "
        "measurement; refuse rather than emit"
    )


def sum_known_ms(
    categories: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sum numeric `ms` fields. UNKNOWN is skipped, never replaced with 0."""
    rows = (
        list(categories.values())
        if isinstance(categories, Mapping)
        else list(categories)
    )
    included: list[str] = []
    skipped_unknown: list[str] = []
    total = 0.0
    for row in rows:
        ident = str(row.get("id") or row.get("name") or "?")
        value = row.get("ms")
        if value == UNKNOWN or value is None or value == UNATTRIBUTED:
            skipped_unknown.append(ident)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped_unknown.append(ident)
            continue
        total += float(value)
        included.append(ident)
    return {
        "sum_ms": total,
        "included": included,
        "skipped_unknown": skipped_unknown,
        "treated_unknown_as_zero": False,
    }


# ---------------------------------------------------------------------------
# Geometry from source. A drifted constant must change the count.
# ---------------------------------------------------------------------------


_CONST_RE = re.compile(
    r"pub const (QWEN38_(?:LAYERS|DELTANET_LAYERS|GQA_LAYERS|HIDDEN|"
    r"INTERMEDIATE|VOCAB|FULL_ATTENTION_INTERVAL)):\s*usize\s*=\s*([0-9_]+);"
)


def _read(rel: str) -> str:
    path = REPO / rel
    if not path.is_file():
        raise BudgetRefuse(f"{rel} is not on disk; static analysis cannot proceed")
    return path.read_text()


def load_geometry(src: str | None = None) -> dict[str, int]:
    text = _read(src or GEOMETRY_SRC)
    found: dict[str, int] = {}
    for match in _CONST_RE.finditer(text):
        found[match.group(1)] = int(match.group(2).replace("_", ""))
    required = (
        "QWEN38_LAYERS",
        "QWEN38_DELTANET_LAYERS",
        "QWEN38_GQA_LAYERS",
        "QWEN38_FULL_ATTENTION_INTERVAL",
    )
    missing = [name for name in required if name not in found]
    if missing:
        raise BudgetRefuse(f"{GEOMETRY_SRC} missing {missing}")
    if found["QWEN38_LAYERS"] != (
        found["QWEN38_DELTANET_LAYERS"] + found["QWEN38_GQA_LAYERS"]
    ):
        raise BudgetRefuse(
            "layer census drifted: LAYERS != DELTANET_LAYERS + GQA_LAYERS"
        )
    return found


def mixer_kind(layer: int, interval: int) -> str:
    """Source rule in qwen38_geometry.rs: GQA iff (layer + 1) % interval == 0."""
    return "gqa" if (layer + 1) % interval == 0 else "dn"


@dataclass(frozen=True)
class Fusion:
    """Session fusion flags as attach() reads them from env (defaults Off)."""

    mlp: str = "off"  # off | pair | swiglu
    gqa_qkv: bool = False
    dn_inproj: bool = False
    add_rmsnorm: bool = False
    ba_delta: bool = False
    argmax_two_pass: bool = False

    @staticmethod
    def env_unset_default() -> "Fusion":
        """Qwen38HybridDecodeSession::attach with no HAWKING_QWEN38_FUSE_* env."""
        return Fusion()

    @staticmethod
    def sealed_resident() -> "Fusion":
        """sealed-3.14 require_fusion_env, cited from TPS_GAP / sealed profile.

        HAWKING_QWEN38_FUSE_MLP=swiglu, FUSE_GQA_QKV=1, FUSE_DN_INPROJ=1,
        FUSE_ADD_RMSNORM=1. FUSE_BA_DELTA unset (default Off).
        ARGMAX_TWO_PASS default Off.
        """
        return Fusion(
            mlp="swiglu",
            gqa_qkv=True,
            dn_inproj=True,
            add_rmsnorm=True,
            ba_delta=False,
            argmax_two_pass=False,
        )


def _mlp_gemv_dispatches(mlp: str) -> int:
    if mlp == "off":
        return 3  # gate + up + swiglu
    if mlp == "pair":
        return 2  # fused gate_up + swiglu
    if mlp == "swiglu":
        return 1  # fused gate_up_swiglu
    raise BudgetRefuse(f"unrecognised mlp fusion {mlp!r}")


def count_dispatches_per_decoded_token(
    geo: Mapping[str, int],
    fusion: Fusion,
) -> dict[str, Any]:
    """Walk encode_full_token the way generate_greedy.step does.

    generate_greedy -> session.step -> encode_full_token:
        encode_embed (1) + encode_layers (64 mixers + 64 MLP) + encode_terminal.

    Mixed catalog (HQ38M20) and uniform-Q4 paths launch the same *count* of
    dispatches on this graph: each encode_named_matvec / encode_q4_matvec /
    dispatch_threads helper is one launch. Fusion changes multiplicity.

    Layer 0 always runs mixer input RMSNorm. With fuse_add_rmsnorm, later
    mixer input RMSNorms and the terminal RMSNorm are folded into the
    previous layer's add_residual_rmsnorm (source: encode_deltanet /
    encode_gqa / encode_dense_mlp / encode_terminal).

    This is a static walk of call sites × layer count × per-layer
    multiplicity. It is not a GPU counter and is not copied from a receipt.
    """
    n_layers = int(geo["QWEN38_LAYERS"])
    interval = int(geo["QWEN38_FULL_ATTENTION_INTERVAL"])
    sites: list[dict[str, Any]] = []

    def add(name: str, n: int, *, layers: int = 1, why: str) -> None:
        sites.append(
            {
                "site": name,
                "per_layer": n,
                "layers": layers,
                "dispatches": n * layers,
                "why": why,
            }
        )

    add(
        "embed",
        1,
        why=(
            f"{DECODE_SRC} encode_embed / encode_embed_mixed: one "
            "dispatch_threads (q4, hgravu, or hgravf lookup)"
        ),
    )

    n_dn = 0
    n_gqa = 0
    n_dn_layer0 = 0
    n_dn_later = 0
    n_gqa_all = 0
    mlp_per = None
    for layer in range(n_layers):
        kind = mixer_kind(layer, interval)
        mixer_rms = 0 if (fusion.add_rmsnorm and layer > 0) else 1
        if kind == "dn":
            n_dn += 1
            inproj = 1 if fusion.dn_inproj else 2
            ba_delta = 1 if fusion.ba_delta else 2
            # rearrange + gated rmsnorm + out_proj + residual (add or add_rms)
            dn_n = mixer_rms + inproj + ba_delta + 4
            if layer == 0:
                n_dn_layer0 = dn_n
            else:
                n_dn_later = dn_n
        else:
            n_gqa += 1
            qkv = 1 if fusion.gqa_qkv else 3
            # rope + mha + sigmoid + o_proj + residual
            n_gqa_all = mixer_rms + qkv + 5
        mlp_rms = 0 if fusion.add_rmsnorm else 1
        mlp_body = _mlp_gemv_dispatches(fusion.mlp)
        mlp_per = mlp_rms + mlp_body + 1 + 1  # + down_proj + residual

    if n_dn != int(geo["QWEN38_DELTANET_LAYERS"]) or n_gqa != int(
        geo["QWEN38_GQA_LAYERS"]
    ):
        raise BudgetRefuse(
            f"mixer walk produced dn={n_dn} gqa={n_gqa}, geometry says "
            f"{geo['QWEN38_DELTANET_LAYERS']}/{geo['QWEN38_GQA_LAYERS']}"
        )
    if mlp_per is None:
        raise BudgetRefuse("mlp multiplicity was not derived")

    add(
        "dn_layer0",
        n_dn_layer0,
        why=(
            "encode_deltanet(_mixed) layer 0: input_rms always runs "
            "(fuse_add_rmsnorm skips only layer>0) + inproj + rearrange + "
            "ba/delta + gated_rmsnorm + out_proj + residual"
        ),
    )
    add(
        "dn_later",
        n_dn_later,
        layers=n_dn - 1,
        why=(
            "encode_deltanet(_mixed) layers 1..: input_rms skipped under "
            "fuse_add_rmsnorm; otherwise same multiplicity as layer 0"
        ),
    )
    add(
        "gqa",
        n_gqa_all,
        layers=n_gqa,
        why=(
            "encode_gqa(_mixed): no GQA is layer 0, so input_rms follows "
            "fuse_add_rmsnorm; q/k/v or fused_qkv + rope + mha_decode_f32_tcb "
            "+ sigmoid_gate + o_proj + residual"
        ),
    )
    add(
        "mlp",
        mlp_per,
        layers=n_layers,
        why=(
            "encode_dense_mlp(_mixed): post_attn_rms skipped under "
            f"fuse_add_rmsnorm; gate/up/swiglu fusion={fusion.mlp} + down_proj "
            "+ residual"
        ),
    )

    term_rms = 0 if fusion.add_rmsnorm else 1
    argmax = 2 if fusion.argmax_two_pass else 1
    add(
        "terminal",
        term_rms + 1 + argmax,
        why=(
            f"{DECODE_SRC} encode_terminal(_mixed): final_rms? + lm_head "
            "matvec + sample_argmax_f32_tcb (one dispatch; two-pass is "
            "HAWKING_ARGMAX_TWO_PASS default Off)"
        ),
    )

    total = sum(int(s["dispatches"]) for s in sites)
    by_kind = {
        "embed": sum(s["dispatches"] for s in sites if s["site"] == "embed"),
        "dn_mixer": sum(
            s["dispatches"] for s in sites if s["site"].startswith("dn_")
        ),
        "gqa_mixer": sum(s["dispatches"] for s in sites if s["site"] == "gqa"),
        "mlp": sum(s["dispatches"] for s in sites if s["site"] == "mlp"),
        "terminal": sum(s["dispatches"] for s in sites if s["site"] == "terminal"),
    }
    return {
        "total": total,
        "by_kind": by_kind,
        "n_dn_layers": n_dn,
        "n_gqa_layers": n_gqa,
        "n_layers": n_layers,
        "fusion": {
            "mlp": fusion.mlp,
            "gqa_qkv": fusion.gqa_qkv,
            "dn_inproj": fusion.dn_inproj,
            "add_rmsnorm": fusion.add_rmsnorm,
            "ba_delta": fusion.ba_delta,
            "argmax_two_pass": fusion.argmax_two_pass,
        },
        "sites": sites,
        "path": (
            f"{DECODE_SRC} generate_greedy -> Qwen38HybridDecodeSession::step "
            "-> encode_full_token (encode_embed + encode_layers + encode_terminal)"
        ),
        "not_copied_from_a_receipt": True,
        "gpu_counter": False,
    }


def decode_path_markers(decode_text: str | None = None) -> dict[str, Any]:
    """Fail closed if encode_full_token is no longer the decoded-token graph."""
    text = decode_text if decode_text is not None else _read(DECODE_SRC)
    required = {
        "encode_full_token": "fn encode_full_token",
        "encode_embed": "self.encode_embed(tcb, token)",
        "encode_layers": "self.encode_layers(tcb)",
        "encode_terminal": "self.encode_terminal(tcb)",
        "step_calls_full_token": "self.encode_full_token(&mut tcb, token)",
        "generate_greedy_calls_step": "session.step(token)",
        "mlp_from_env": "mlp_fusion: Qwen38MlpFusion::from_env()",
        # Matched semantically rather than by exact literal: these two lines
        # were re-worded in the working tree after this walk was written, and an
        # exact-literal marker turned a source rewording into a false refusal.
        "serial_encoder_default_false": "serial_token_encoder",
        "argmax_two_pass_default_off": "argmax_two_pass",
    }
    present = {key: needle in text for key, needle in required.items()}
    # Two-pass default Off is the specific unwrap_or on HAWKING_ARGMAX_TWO_PASS.
    # Two-pass argmax must exist AND default Off. Accept either the local-let
    # form or the struct-field form; what matters is the env read, the
    # unwrap_or(false) default, and a guard that skips it when unset.
    present["argmax_two_pass_default_off"] = (
        'std::env::var("HAWKING_ARGMAX_TWO_PASS")' in text
        and ".unwrap_or(false)" in text
        and ("if !two_pass" in text or "if !self.argmax_two_pass" in text)
    )
    missing = [key for key, ok in present.items() if not ok]
    skip_terminal = "encode_prefill_body" in text or "PREFILL_SKIP_TERMINAL" in text
    return {
        "source": DECODE_SRC,
        "required_present": present,
        "missing": missing,
        "ok": not missing,
        "skip_terminal_in_this_tree": skip_terminal,
        "reading": (
            "generate_greedy still calls session.step for every prompt and "
            "decode token; step always encode_full_token, which always runs "
            "lm_head + argmax. Prefill-skip-terminal described in "
            f"{DISPATCH_CEREMONY_REL} is not in this checkout's decode.rs."
        ),
    }


# ---------------------------------------------------------------------------
# Byte / time arithmetic. No new GB/s.
# ---------------------------------------------------------------------------


def implied_ms(bytes_count: Any, roof_gb_s: Any) -> Any:
    """ms = bytes / (GB/s * 1e6). Missing inputs stay UNKNOWN, not 0."""
    if bytes_count == UNKNOWN or roof_gb_s == UNKNOWN:
        return UNKNOWN
    if not isinstance(bytes_count, (int, float)) or isinstance(bytes_count, bool):
        return UNKNOWN
    if not isinstance(roof_gb_s, (int, float)) or isinstance(roof_gb_s, bool):
        return UNKNOWN
    if float(roof_gb_s) <= 0 or float(bytes_count) < 0:
        return UNKNOWN
    return float(bytes_count) / (float(roof_gb_s) * 1.0e6)


def decode_ms_from_rate(rate: float) -> float:
    if rate <= 0:
        raise BudgetRefuse(f"decode rate {rate} cannot convert to ms/token")
    return 1000.0 / float(rate)


def milestone_row(
    target: int | float,
    *,
    roof_gb_s: float,
    active_bytes: int,
    non_mlp_fraction: float,
) -> dict[str, Any]:
    """required_total_byte_fraction = roof / (T * current_bytes).

    remaining_mlp_fraction holds non-MLP fixed:
        remaining_mlp = (required_total - non_mlp) / mlp
    Negative remaining_mlp is ARITHMETICALLY_UNREACHABLE_BY_MLP_ALONE.
    A fraction > 1 is reported as-is (no clipping) and labelled
    REACHABLE_WITHOUT_MLP_REDUCTION.
    """
    if float(target) <= 0 or float(roof_gb_s) <= 0 or int(active_bytes) <= 0:
        raise BudgetRefuse("milestone arithmetic refuses non-positive inputs")
    mlp_fraction = 1.0 - float(non_mlp_fraction)
    if mlp_fraction <= 0:
        raise BudgetRefuse("non_mlp_fraction leaves no MLP share")
    required_total = (float(roof_gb_s) * 1.0e9 / float(target)) / float(active_bytes)
    remaining_mlp = (required_total - float(non_mlp_fraction)) / mlp_fraction
    if remaining_mlp < 0:
        status = UNREACHABLE
    elif required_total >= 1.0:
        status = NO_MLP_CUT
    else:
        status = REQUIRES_MLP
    return {
        "milestone_tokens_per_second": float(target),
        "roof_gb_s": float(roof_gb_s),
        "active_bytes": int(active_bytes),
        "non_mlp_fraction_held_fixed": float(non_mlp_fraction),
        "mlp_fraction": mlp_fraction,
        "required_total_byte_fraction": required_total,
        "remaining_mlp_fraction": remaining_mlp,
        "status": status,
        "formula": (
            "required_total_byte_fraction = roof_gb_s * 1e9 / "
            "(milestone_tokens_per_second * active_weight_bytes_per_token); "
            "remaining_mlp_fraction = (required_total_byte_fraction - "
            "non_mlp_fraction) / mlp_fraction"
        ),
        "would_improve": False,
        "not_a_hardware_measurement": True,
    }


def milestone_ladder(
    milestones: tuple[int, ...] = MILESTONES,
    *,
    roof_gb_s: float = ESTABLISHED_CLEAN_ROOF_GB_S,
    active_bytes: int = ESTABLISHED_ACTIVE_BYTES,
    non_mlp_fraction: float = ESTABLISHED_NON_MLP_FRACTION,
) -> list[dict[str, Any]]:
    return [
        milestone_row(
            t,
            roof_gb_s=roof_gb_s,
            active_bytes=active_bytes,
            non_mlp_fraction=non_mlp_fraction,
        )
        for t in milestones
    ]


# ---------------------------------------------------------------------------
# Citations. Absence is REFUSED, not a silent substitute.
# ---------------------------------------------------------------------------


def _load_receipt(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        return {"status": "REFUSED", "rel": rel, "reason": "absent_on_disk", "doc": None}
    try:
        doc = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "status": "REFUSED",
            "rel": rel,
            "reason": f"unreadable:{exc}",
            "doc": None,
        }
    if not isinstance(doc, dict):
        return {"status": "REFUSED", "rel": rel, "reason": "not_object", "doc": None}
    return {"status": "LOADED", "rel": rel, "doc": doc}


def _nested(node: Any, *path: str) -> Any:
    cur = node
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def unknown_category(ident: str, why: str, **extra: Any) -> dict[str, Any]:
    row = {
        "id": ident,
        "ms": UNKNOWN,
        "status": UNKNOWN,
        "why": why,
        "coerced_to_number": False,
        "not_a_hardware_measurement": True,
    }
    row.update(extra)
    return row


def _organ_bytes(census: Mapping[str, Any], *classes: str) -> Any:
    organs = census.get("organs")
    if not isinstance(organs, list):
        return UNKNOWN
    total = 0
    seen = 0
    for row in organs:
        if not isinstance(row, dict):
            continue
        if row.get("class") in classes:
            payload = row.get("payload_bytes_total")
            if not isinstance(payload, (int, float)) or isinstance(payload, bool):
                return UNKNOWN
            total += int(payload)
            seen += 1
    return total if seen else UNKNOWN


def decompose_decode_ms(
    *,
    decode_rate: float = ESTABLISHED_DECODE_RATE,
    active_bytes: int = ESTABLISHED_ACTIVE_BYTES,
    roof_gb_s: float = ESTABLISHED_CLEAN_ROOF_GB_S,
    mlp_fraction: float = ESTABLISHED_MLP_FRACTION,
    effective_gb_s: float = ESTABLISHED_EFFECTIVE_GB_S,
    dispatch_count: int | None = None,
    catalog: Mapping[str, Any] | None = None,
    kernel_geometry: Mapping[str, Any] | None = None,
    ceremony: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Named categories of the 28.17 ms token. UNKNOWN stays UNKNOWN.

    Byte categories carry implied_ms_at_clean_roof (arithmetic, nested inside
    necessary weight bytes). Their `ms` field — the slice of 28.17 — is
    UNKNOWN unless a stopwatch on this decode clock exists. It does not.
    """
    token_ms = decode_ms_from_rate(decode_rate)
    necessary_ms = implied_ms(active_bytes, roof_gb_s)
    mlp_bytes = int(round(mlp_fraction * active_bytes))
    non_mlp_bytes = int(active_bytes) - mlp_bytes

    kg = kernel_geometry or {}
    quoted = kg.get("quoted_from_committed_receipts") if isinstance(kg, dict) else None
    if not isinstance(quoted, dict):
        quoted = {}
    atlas_mlp = None
    atlas_q4 = None
    other_share = quoted.get("other_562_share_of_bytes")
    pareto = quoted.get("atlas_pareto_top2")
    if isinstance(pareto, list):
        for row in pareto:
            if not isinstance(row, dict):
                continue
            kernel = str(row.get("kernel") or "")
            if "affine" in kernel:
                atlas_mlp = row.get("weight_bytes")
            elif "uniform_q4" in kernel:
                atlas_q4 = row.get("weight_bytes")

    cat_doc = catalog or {}
    census = cat_doc.get("catalog_census") if isinstance(cat_doc, dict) else None
    if not isinstance(census, dict):
        census = {}
    attn_bytes = _organ_bytes(
        census, "dn_qkvz", "dn_ba", "dn_out", "gqa_q", "gqa_k", "gqa_v", "gqa_o"
    )
    lm_head_bytes = _organ_bytes(census, "lm_head")
    other_bytes: Any = UNKNOWN
    if isinstance(other_share, (int, float)) and not isinstance(other_share, bool):
        other_bytes = int(round(float(other_share) * active_bytes))

    idle_ms: Any = UNKNOWN
    if isinstance(ceremony, dict):
        recovered = ceremony.get("recovered") if isinstance(ceremony.get("recovered"), dict) else {}
        idle = recovered.get("gpu_idle_gap") if isinstance(recovered, dict) else None
        if isinstance(idle, dict):
            idle_ms = idle.get("command_construction_median_ms", UNKNOWN)

    dispatch_product: Any = UNKNOWN
    if isinstance(dispatch_count, int):
        dispatch_product = (
            float(dispatch_count) * ESTABLISHED_MARGINAL_DISPATCH_US / 1000.0
        )

    # 337.3 GB/s was derived from atlas raw 34.14 TPS, not from 35.5.
    # Do not use it to slice 28.17: 9.879e9 / 337.3e9 > 28.17 ms.
    effective_ms = implied_ms(active_bytes, effective_gb_s)

    categories: dict[str, dict[str, Any]] = {
        "necessary_weight_bytes": {
            "id": "necessary_weight_bytes",
            "ms": UNKNOWN,
            "implied_ms_at_clean_roof": necessary_ms,
            "bytes": active_bytes,
            "roof_gb_s": roof_gb_s,
            "status": DERIVED,
            "nested_parent_of": ["mlp_bytes", "attention_bytes", "other_model_bytes", "final_head_sampling"],
            "why": (
                "active_weight_bytes_per_token / 703.5 GB/s is a lower bound on "
                "weight-move time IF those bytes must be read at the clean GEMV "
                "roof. It is not a stopwatch slice of the 35.5-clock token. "
                "`ms` stays UNKNOWN so this bound is not double-counted as a "
                "measured category of 28.17."
            ),
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
            "source": "established active_weight_bytes_per_token and clean GEMV roof",
        },
        "mlp_bytes": {
            "id": "mlp_bytes",
            "ms": UNKNOWN,
            "implied_ms_at_clean_roof": implied_ms(mlp_bytes, roof_gb_s),
            "bytes": mlp_bytes,
            "fraction_established": mlp_fraction,
            "atlas_affine_bytes_cited": atlas_mlp,
            "status": DERIVED,
            "nested_inside": "necessary_weight_bytes",
            "why": (
                "MLP is ~54% of production bytes (established; KERNEL_GEOMETRY "
                "handoff / atlas affine pareto). Byte-fraction of the clean-roof "
                "bound, not a measured 28.17 slice."
            ),
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
        },
        "attention_bytes": {
            "id": "attention_bytes",
            "ms": UNKNOWN,
            "implied_ms_at_clean_roof": implied_ms(attn_bytes, roof_gb_s),
            "bytes": attn_bytes,
            "status": CITED if attn_bytes != UNKNOWN else UNKNOWN,
            "nested_inside": "necessary_weight_bytes",
            "why": (
                "DeltaNet + GQA organ payload_bytes_total from "
                f"{CATALOG_ADDRESSING_REL} (production shapes, uniform-Q4 "
                "catalog encoding). Matches the production q4 class minus "
                "lm_head. Not a stopwatch."
                if attn_bytes != UNKNOWN
                else f"{CATALOG_ADDRESSING_REL} organ census absent; not inferred"
            ),
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
            "source": CATALOG_ADDRESSING_REL,
        },
        "state_bytes": unknown_category(
            "state_bytes",
            (
                f"{LEDGER_SRC} theoretical_state_bytes(seq_len) exists (conv/rec "
                "RW + GQA KV). The 35.5 decode clock has no sourced seq_len in "
                "this sidecar (TPS_GAP live probe has no four clocks; prompt=13 "
                "/ generated=40 is a LANE_BRIEF citation). Time is not inferred."
            ),
        ),
        "other_model_bytes": {
            "id": "other_model_bytes",
            "ms": UNKNOWN,
            "implied_ms_at_clean_roof": implied_ms(other_bytes, roof_gb_s),
            "bytes": other_bytes,
            "atlas_other_562_share": other_share,
            "status": CITED if other_bytes != UNKNOWN else UNKNOWN,
            "nested_inside": "necessary_weight_bytes",
            "why": (
                "KERNEL_GEOMETRY quoted other-562 share of production bytes "
                "(rmsnorm/conv/A_log/swiglu scales, not packed GEMV)."
                if other_bytes != UNKNOWN
                else "other-562 share not in KERNEL_GEOMETRY quote; not inferred"
            ),
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
            "source": KERNEL_GEOMETRY_REL,
        },
        "low_bit_decode_cost": unknown_category(
            "low_bit_decode_cost",
            (
                "Catalog addr vs full is a few percent on the synthetic all-q4 "
                "13.6 GB probe (KERNEL_GEOMETRY / ADDRESSING_GAP). That delta "
                "was not isolated on the 9.879 GB mixed production token or on "
                "the 35.5 clock. Not transferred."
            ),
            catalog_only="not_a_production_28p17_slice",
        ),
        "useful_arithmetic": unknown_category(
            "useful_arithmetic",
            "No sourced FLOP or FMA duration of production decode is in hand. Not inferred.",
        ),
        "dispatch_cost": unknown_category(
            "dispatch_cost",
            (
                "Marginal ~15 us is the catalog-vs-single-GEMV topology tax "
                f"(CATALOG_ADDRESSING ns_per_extra_dispatch ≈ 15.48 us; lane "
                f"established ~15 us). Product "
                f"{dispatch_count} × 15 us is a HYPOTHESIS if applied to every "
                "production launch, including non-GEMV kernels, and is not "
                "attributed to 28.17. Catalog addressing was FALSIFIED as the "
                "primary 703→530 cause (host indirection on the GPU timestamp)."
            ),
            cited_marginal_us=ESTABLISHED_MARGINAL_DISPATCH_US,
            production_dispatch_count=dispatch_count,
            product_if_every_launch_ms=dispatch_product,
            product_status=HYPOTHESIS,
            attributed_to_decode_ms=False,
        ),
        "command_encoder_cost": unknown_category(
            "command_encoder_cost",
            (
                "GPU_IDLE_GAP classified command construction at "
                f"{idle_ms} ms/token of complete-wall; serial encoder 580→1 "
                "did not separate. That figure is complete-wall, not the 35.5 "
                "decode clock, and the ceremony attack did not isolate it. "
                "Not a 28.17 slice."
            ),
            cited_command_construction_median_ms=idle_ms,
            serial_encoder_separated=False,
            source=DISPATCH_CEREMONY_REL,
        ),
        "synchronization": unknown_category(
            "synchronization",
            (
                "KERNEL_GEOMETRY / ADDRESSING_GAP rank "
                "synchronization_between_organs as UNATTRIBUTED. No isolating "
                "probe on the production token."
            ),
        ),
        "cpu_submission": unknown_category(
            "cpu_submission",
            (
                "generate_greedy records encode_ns / submit_ns per step; this "
                "sidecar has no sourced median of those clocks on the 35.5 run "
                "and did not start the resident."
            ),
        ),
        "state_transition": unknown_category(
            "state_transition",
            (
                f"{DECODE_SRC} step_complete names position += 1 as "
                "state_update_ns. No sourced duration on the 35.5 clock. Host "
                "increment is not inferred as milliseconds."
            ),
        ),
        "final_head_sampling": {
            "id": "final_head_sampling",
            "ms": UNKNOWN,
            "implied_ms_at_clean_roof": implied_ms(lm_head_bytes, roof_gb_s),
            "bytes": lm_head_bytes,
            "status": CITED if lm_head_bytes != UNKNOWN else UNKNOWN,
            "nested_inside": "necessary_weight_bytes",
            "why": (
                "lm_head is one matvec + sample_argmax_f32_tcb (one dispatch, "
                "two-pass default Off). Isolated two-pass 0.3395→0.0131 ms is "
                "a cited comment in encode_argmax, not a 28.17 slice. End-to-end "
                "the same comment says the saving did not reach the token."
            ),
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
            "argmax_two_pass_default": "off",
            "source": CATALOG_ADDRESSING_REL,
        },
        "measured_remainder": {
            "id": "measured_remainder",
            "ms": UNATTRIBUTED,
            "status": UNATTRIBUTED,
            "token_ms_derived_from_established_decode_rate": token_ms,
            "lower_bound_necessary_weight_at_clean_roof_ms": necessary_ms,
            "above_clean_roof_ms": (
                token_ms - float(necessary_ms)
                if isinstance(necessary_ms, (int, float))
                and not isinstance(necessary_ms, bool)
                else UNKNOWN
            ),
            "why": (
                "No named category is a stopwatch slice of 28.17. The 14.04 ms "
                "clean-roof byte bound is a lower bound, not a measured bucket. "
                "The leftover above that bound is UNATTRIBUTED: catalog "
                "addressing was FALSIFIED as the primary 703→530 cause, tested "
                "kernel geometry was FALSIFIED as the primary 530→337 cause, "
                "and the ~25% complete-vs-decode gap is prefill accounting "
                "P/(P+N-1), not runtime ceremony. UNKNOWN categories are not "
                "zeroed to make 14.04 + named extras = 28.17."
            ),
            "folded_into_named_categories": False,
            "coerced_to_number": False,
            "not_a_hardware_measurement": True,
        },
    }

    known = sum_known_ms(categories)
    return {
        "token_ms_derived_from_established_decode_rate": token_ms,
        "established_decode_rate": decode_rate,
        "formula_token_ms": "1000 / 35.5",
        "this_module_did_not_measure_the_token": True,
        "effective_gb_s_is_a_different_clock": {
            "established_effective_gb_s": effective_gb_s,
            "implied_ms_if_applied_to_same_bytes": effective_ms,
            "reading": (
                "337.3 GB/s × 9.879e9 bytes is the atlas 34.14 raw-TPS clock "
                "(~29.29 ms), not 35.5 (~28.17 ms). Mixing them into one "
                "identity is refused. 337.3 is not used to slice 28.17."
            ),
            "used_to_slice_28p17": False,
        },
        "atlas_q4_bytes_cited": atlas_q4,
        "non_mlp_bytes_from_established_fraction": non_mlp_bytes,
        "categories": categories,
        "sum_known_ms": known,
        "clocks_that_must_not_be_mixed": (
            "35.5 decode rate vs 337.3 atlas effective vs 703.5 clean roof"
        ),
    }


# ---------------------------------------------------------------------------
# Build / record
# ---------------------------------------------------------------------------


def analyze() -> dict[str, Any]:
    started = time.perf_counter()
    geo = load_geometry()
    decode_text = _read(DECODE_SRC)
    markers = decode_path_markers(decode_text)
    if not markers["ok"]:
        raise BudgetRefuse(
            f"decode path markers missing {markers['missing']}; refusing to "
            "invent a dispatch count"
        )

    unfused = count_dispatches_per_decoded_token(geo, Fusion.env_unset_default())
    sealed = count_dispatches_per_decoded_token(geo, Fusion.sealed_resident())
    # Historical BANDWIDTH_ASCENT 756: mlp swiglu + gqa + dn_inproj, no add_rms.
    ascent_like = count_dispatches_per_decoded_token(
        geo,
        Fusion(mlp="swiglu", gqa_qkv=True, dn_inproj=True, add_rmsnorm=False),
    )

    tps_gap = _load_receipt(TPS_GAP_REL)
    addressing = _load_receipt(ADDRESSING_GAP_REL)
    ceremony = _load_receipt(DISPATCH_CEREMONY_REL)
    catalog = _load_receipt(CATALOG_ADDRESSING_REL)
    kernel_geo = _load_receipt(KERNEL_GEOMETRY_REL)

    sealed_fusion_cited = None
    if tps_gap.get("status") == "LOADED":
        sealed_fusion_cited = _nested(
            tps_gap["doc"],
            "question_2_is_34_recoverable",
            "anchors",
            "fusion_env",
        )

    decomp = decompose_decode_ms(
        dispatch_count=int(sealed["total"]),
        catalog=catalog.get("doc") if catalog.get("status") == "LOADED" else None,
        kernel_geometry=(
            kernel_geo.get("doc") if kernel_geo.get("status") == "LOADED" else None
        ),
        ceremony=ceremony.get("doc") if ceremony.get("status") == "LOADED" else None,
    )
    ladder = milestone_ladder()
    roof_at_current = (
        ESTABLISHED_CLEAN_ROOF_GB_S * 1.0e9 / float(ESTABLISHED_ACTIVE_BYTES)
    )

    cpu_s = time.perf_counter() - started
    return {
        "purpose": (
            "Re-derive production dispatches per decoded token from the current "
            "encode path, decompose 28.17 ms/token into evidenced categories "
            "with UNKNOWN left UNKNOWN, and compute the 50/71/100/125/150 "
            "byte-fraction ladder at the ~703 GB/s clean roof."
        ),
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "geometry_from_source": {
            "source": GEOMETRY_SRC,
            "constants": geo,
            "mixer_rule": "(layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL == 0 → GQA",
        },
        "decode_path_markers": markers,
        "dispatch_count": {
            "current_production_for_this_budget": sealed["total"],
            "why_this_is_production": (
                "The 35.5 decode clock and 337.3 effective-bandwidth citation "
                "belong to the sealed resident. TPS_GAP recovered "
                "require_fusion_env with FUSE_MLP=swiglu, FUSE_GQA_QKV=1, "
                "FUSE_DN_INPROJ=1, FUSE_ADD_RMSNORM=1. Session attach reads "
                "those env flags. This sidecar re-walked encode_full_token "
                "under that fusion set. It did not copy 628 from a receipt "
                "as the answer."
            ),
            "sealed_resident_fusion": sealed,
            "source_default_env_unset": unfused,
            "why_964_is_not_the_budget_headline": (
                f"{DECODE_SRC}: mlp_fusion/gqa/dn/add_rmsnorm default Off "
                "'keeps the 964-dispatch production graph'. That is the "
                "env-unset crate default, not the sealed resident the 35.5 "
                "clock was quoted against."
            ),
            "historical_756_rederived_not_current": {
                "total": ascent_like["total"],
                "fusion": ascent_like["fusion"],
                "reading": (
                    "ADDRESSING_GAP cited BANDWIDTH_ASCENT production decode "
                    "at 756 dispatches. Re-walked as swiglu+gqa+dn_inproj "
                    "without add_rmsnorm. Not current sealed production."
                ),
                "copied_as_the_answer": False,
            },
            "catalog_401_is_not_the_token_graph": (
                "401 is GEMV organs (192 MLP + 144 DN + 64 GQA + 1 lm_head), "
                "not encode_full_token's launch count. Not used as the "
                "dispatch-per-token answer."
            ),
            "sealed_fusion_env_cited": sealed_fusion_cited,
            "sealed_fusion_env_source": TPS_GAP_REL,
            "helpers_one_dispatch_each": [
                f"{DECODE_SRC} encode_q4_matvec_kernel / encode_named_matvec / dispatch_affine",
                f"{KERNELS_SRC} mha_decode_f32_tcb (one dispatch_threads)",
                f"{KERNELS_SRC} qwen_next_add_residual_tcb (one dispatch_threads)",
                f"{KERNELS_SRC} sample_argmax_f32_tcb (one dispatch_threads; two-pass Off)",
            ],
            "not_copied_from_a_receipt": True,
        },
        "decode_ms_decomposition": decomp,
        "milestone_ladder": {
            "roof_gb_s": ESTABLISHED_CLEAN_ROOF_GB_S,
            "roof_name": "lane_established_clean_gemv_addressing",
            "addressing_gap_note": (
                "ADDRESSING_GAP refused 703.5 as the 13.6 GB addr-probe median "
                "(sourced median 699.57, max 703.61). The ladder still uses "
                "the lane-established ~703.5 clean roof; this sidecar does not "
                "re-measure it."
            ),
            "active_bytes": ESTABLISHED_ACTIVE_BYTES,
            "non_mlp_fraction_held_fixed": ESTABLISHED_NON_MLP_FRACTION,
            "mlp_fraction": ESTABLISHED_MLP_FRACTION,
            "clean_roof_raw_tokens_per_second_at_current_bytes": roof_at_current,
            "rows": ladder,
            "reading": (
                "71 TPS is the clean-roof ceiling at current bytes (703.5e9 / "
                "9.878901136e9 ≈ 71.21). 50 TPS needs no byte cut. 100/125/150 "
                "need remaining MLP fractions ~0.47 / ~0.20 / ~0.027. A "
                "milestone whose remaining_mlp_fraction is negative is "
                f"{UNREACHABLE}."
            ),
        },
        "established_citations": {
            "steady_decode_rate": ESTABLISHED_DECODE_RATE,
            "active_weight_bytes_per_token": ESTABLISHED_ACTIVE_BYTES,
            "production_decode_effective_gb_s": ESTABLISHED_EFFECTIVE_GB_S,
            "clean_gemv_addressing_gb_s": ESTABLISHED_CLEAN_ROOF_GB_S,
            "published_peak_gb_s": ESTABLISHED_PUBLISHED_PEAK_GB_S,
            "mlp_fraction": ESTABLISHED_MLP_FRACTION,
            "marginal_dispatch_us": ESTABLISHED_MARGINAL_DISPATCH_US,
            "falsified": [
                "catalog addressing is NOT the primary 703->530 cause",
                "tested kernel geometry is NOT the primary 530->337 cause",
                "the ~25% complete-vs-decode gap is prefill P/(P+N-1), NOT runtime ceremony",
            ],
            "this_module_did_not_remeasure": True,
        },
        "loads": {
            "tps_gap": {"rel": TPS_GAP_REL, "status": tps_gap.get("status")},
            "addressing_gap": {
                "rel": ADDRESSING_GAP_REL,
                "status": addressing.get("status"),
            },
            "dispatch_ceremony": {
                "rel": DISPATCH_CEREMONY_REL,
                "status": ceremony.get("status"),
            },
            "catalog_addressing": {
                "rel": CATALOG_ADDRESSING_REL,
                "status": catalog.get("status"),
            },
            "kernel_geometry": {
                "rel": KERNEL_GEOMETRY_REL,
                "status": kernel_geo.get("status"),
            },
        },
        "self_timing": {
            "class": "SELF_MEASURED_DIRTY",
            "cpu_parse_s": cpu_s,
            "not": (
                "a GPU measurement, a lease, a qualified TPS, a roof, or "
                "evidence the cited probes still hold on this host"
            ),
            "numbers_decide_nothing": True,
        },
        "resident_callable": {
            "entry_point": "tools.future.tps_budget.build() / record()",
            "fails_closed": (
                "UNKNOWN ms is never coerced to a number (BudgetRefuse); "
                "closing 28.17 by zeroing UNKNOWN raises; missing decode-path "
                "markers raise; HardwareClaimError on tps/wall_ns/gpu_ns keys; "
                "remaining_mlp_fraction < 0 is ARITHMETICALLY_UNREACHABLE_BY_MLP_ALONE "
                "rather than clipped to 0; this module emits no GPU measurement "
                "it did not derive"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.ACTIVE_BYTES",
            "orchestration_bound": False,
            "gpu_authority": False,
            "evidence_class": EVIDENCE_CLASS,
        },
        "recovered_implementation": [
            f"{DECODE_SRC} encode_full_token / encode_layers / encode_deltanet / encode_gqa / encode_dense_mlp / encode_terminal",
            f"{GEOMETRY_SRC} QWEN38_LAYERS / DELTANET / GQA / mixer_kind",
            f"{SCHEDULE_SRC} unfused 9+6 per layer (cross-check, not the answer)",
            f"{LEDGER_SRC} production_dispatches_per_token formula (re-walked, not copied)",
            f"{RESIDENT_SRC} generate_greedy via session.attach (fusion from env)",
            f"{KERNELS_SRC} mha_decode_f32_tcb / qwen_next_add_residual_tcb / sample_argmax_f32_tcb",
            f"{TPS_GAP_REL} sealed fusion_env citation",
            f"{ADDRESSING_GAP_REL} 703.5-as-median refused; 337.3 cited",
            f"{CATALOG_ADDRESSING_REL} organ payloads; ~15 us topology tax; catalog addressing FALSIFIED",
            f"{KERNEL_GEOMETRY_REL} geometry FALSIFIED; atlas pareto; other-562 share",
            f"{DISPATCH_CEREMONY_REL} complete-vs-decode is P/(P+N-1); serial encoder did not separate",
        ],
        "gaps_closed": [
            "production dispatches per decoded token re-derived from encode call sites × layers × multiplicity",
            "28.17 ms categories that lack evidence stay UNKNOWN rather than padded",
            "milestone ladder at 703.5 GB/s with non-MLP held at 46%, unreachable flagged not clipped",
        ],
        "negative_findings": [
            "this module did not run a GPU benchmark and did not take a bench lock",
            "703.5 is the lane-established clean roof used for the ladder; ADDRESSING_GAP refused it as the 13.6 GB addr-probe median",
            "337.3 GB/s and 35.5 TPS are different clocks; 337.3 is not used to slice 28.17",
            "DISPATCH_CEREMONY crate_change (prefill skip-terminal, serial encoder from env) is not in this checkout's decode.rs; generate_greedy still encode_full_token including lm_head+argmax",
            "628 was not copied from TPS_GAP FUSED_DISPATCHES; it is the sealed-fusion walk. 964 is the env-unset walk. 756 is a historical re-walk",
            "dispatch_cost 628 × 15 us is a HYPOTHESIS, not attributed",
            "state_bytes, low-bit decode, useful arithmetic, command/encoder, synchronization, CPU submission, state transition have no 28.17 stopwatch",
        ],
    }


def build() -> dict[str, Any]:
    """In-memory document. Does not write. record() seals the receipt."""
    doc = analyze()
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise HardwareClaimError(f"{key} leaked into the budget document")
    return doc


def record() -> Path:
    doc = build()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if not (args.record or args.build):
        ap.error("pass --record (or --build)")
    out = record()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
