#!/usr/bin/env python3.12
"""SUB-BIT CLOSURE: the legal one-bit rebudget and the method program that follows it.

GOVERNING LAW
    Hawking does not climb above one bit to discover where conventional quantization
    works. Hawking changes the representation, model, allocation and treatment until
    useful intelligence survives at one bit or below.

    complete_artifact_bits / original_weight_count <= 1/1

A1_1p0 realized complete 1.0075 BPW on Qwen3-235B. It is ILLEGAL. The response is a
rebudget, never a raised ceiling. This module solves for allocations that land at or
under 1/1 with every organ and every overhead byte itemized, and it schedules what to
try AT the ceiling.

WHAT THE MEASURED COLLAPSE DOES AND DOES NOT SAY
    A1_1p0 (complete 1.0075) and R2_subhalf (complete 0.4930) both collapsed 6/6 on a
    real Qwen3-235B forward. That is a negative result for the RAW-WEIGHT PQ/VQ family
    at that rate. It is not a proof about every method below one bit. Methods that
    CHANGE THE SOURCE (QAT, distillation, compressibility training, structured pruning,
    learned sharing) are not bound by the rate-distortion limit of the original weights,
    and that distinction is the whole reason the ceiling is defensible rather than
    wishful.

ACCOUNTING DISCIPLINE
    Bits are exact integers, BPW is an exact Fraction, and every bit lands in one of the
    ten named slots of one_bit_ceiling.CompleteByteLedger plus an explicit reserve.
    Nothing is excluded as "overhead". An expert-only BPW is never reported as the whole
    model rate. This ledger is deliberately STRICTER than the sealed A1 ledger: replaying
    A1's own organ rates through it lands ABOVE 1.0075, so no variant here is made legal
    by a slacker ruler.

    This module launches nothing, downloads nothing and reads no weight bytes. It reads
    config.json / model.safetensors.index.json metadata only, and falls back to the
    sealed geometry constants when that metadata is not on this host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# The ceiling is owned elsewhere. Import it if present; otherwise guard with an exact
# local equivalent so this module still refuses anything above 1/1.
try:  # pragma: no cover - exercised by whichever side of the guard the tree is on
    import one_bit_ceiling as OBC

    CEILING: Fraction = OBC.CEILING
    HAVE_OBC = True
except Exception:  # noqa: BLE001 - a missing sibling module must not disable the law
    OBC = None
    CEILING = Fraction(1, 1)
    HAVE_OBC = False

SCHEMA = "hawking.foundry.subbit_closure_program.v1"
REPORT_RELPATH = Path("reports/foundry/SUBBIT_CLOSURE_PROGRAM.json")

DEFAULT_PARENT_DIR = Path(
    "/Users/scammermike/hawking-qwen-recovery-20260720/models/qwen3-235b-a22b"
)


class ClosureError(AssertionError):
    """A candidate or a schedule violated the law."""


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARENT INVENTORY  (metadata only, cross-checked byte-exact)
# ══════════════════════════════════════════════════════════════════════════════
# Sealed geometry of Qwen/Qwen3-235B-A22B-Instruct-2507 @ ac9c66cc9b46af7306746a9250f23d47083d689e.
SEALED_CONFIG: dict[str, int | bool] = {
    "hidden_size": 4096,
    "vocab_size": 151936,
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "moe_intermediate_size": 1536,
    "tie_word_embeddings": False,
}
SEALED_TOTAL_SIZE_BYTES = 470_187_269_120   # index metadata.total_size, bf16
SEALED_PARAMS = 235_093_634_560             # total_size / 2
SEALED_TENSOR_COUNT = 36_945                # len(weight_map)


@dataclass(frozen=True)
class Organ:
    """One collapsed-across-layers/experts tensor class."""

    name: str
    tensors: int
    rows: int          # 0 for the vector organs (norms), which are never coded
    cols: int
    params: int
    sensitive: bool = False

    @property
    def codeable(self) -> bool:
        return self.rows > 0 and self.cols > 0


@dataclass(frozen=True)
class Inventory:
    organs: dict[str, Organ]
    params: int
    tensors: int
    source: str

    def organ(self, name: str) -> Organ:
        try:
            return self.organs[name]
        except KeyError as exc:
            raise ClosureError(f"unknown organ {name!r}") from exc


def build_inventory(config: dict[str, Any] | None = None,
                    index: dict[str, Any] | None = None) -> Inventory:
    """Derive the whole-model organ inventory from config geometry.

    Cross-checked two ways: the analytic parameter total must equal the index
    metadata.total_size / 2 to the byte, and the analytic tensor count must equal
    len(weight_map). Either mismatch is fatal; a wrong denominator would make every
    BPW in this module a lie.
    """
    cfg = dict(SEALED_CONFIG if config is None else config)
    h = int(cfg["hidden_size"])
    v = int(cfg["vocab_size"])
    L = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    hd = int(cfg.get("head_dim") or h // heads)
    E = int(cfg["num_experts"])
    mi = int(cfg["moe_intermediate_size"])
    if cfg.get("tie_word_embeddings"):
        raise ClosureError("tied embeddings change the inventory; this plan assumes untied")

    q_out, kv_out = heads * hd, kv_heads * hd
    cells = L * E
    # norms: q_norm + k_norm (head_dim each) + 2 layernorms (hidden each) per layer, + final
    norm_params = L * (2 * hd + 2 * h) + h
    norm_tensors = L * 4 + 1

    organs = [
        Organ("embed_tokens", 1, v, h, v * h),
        Organ("lm_head", 1, v, h, v * h),
        Organ("attn_q", L, q_out, h, L * q_out * h),
        Organ("attn_k", L, kv_out, h, L * kv_out * h),
        Organ("attn_v", L, kv_out, h, L * kv_out * h),
        Organ("attn_o", L, h, q_out, L * h * q_out),
        Organ("router", L, E, h, L * E * h),
        Organ("expert_gate", cells, mi, h, cells * mi * h, sensitive=True),
        Organ("expert_up", cells, mi, h, cells * mi * h, sensitive=True),
        Organ("expert_down", cells, h, mi, cells * h * mi),
        Organ("norms", norm_tensors, 0, 0, norm_params),
    ]
    table = {o.name: o for o in organs}
    params = sum(o.params for o in organs)
    tensors = sum(o.tensors for o in organs)

    src = "sealed_constants"
    if index is not None:
        total = int((index.get("metadata") or {}).get("total_size", 0))
        n_named = len(index.get("weight_map") or {})
        if total and total != params * 2:
            raise ClosureError(
                f"inventory cross-check failed: analytic {params * 2} bytes != index {total}")
        if n_named and n_named != tensors:
            raise ClosureError(
                f"tensor-count cross-check failed: analytic {tensors} != index {n_named}")
        src = "config_json + model.safetensors.index.json"
    elif config is None:
        if params != SEALED_PARAMS or tensors != SEALED_TENSOR_COUNT:
            raise ClosureError("sealed constants no longer self-consistent")
    return Inventory(organs=table, params=params, tensors=tensors, source=src)


def load_parent(parent_dir: Path = DEFAULT_PARENT_DIR) -> Inventory:
    """Read real metadata when present; fall back to the sealed geometry when not."""
    cfg_p, idx_p = Path(parent_dir) / "config.json", Path(parent_dir) / "model.safetensors.index.json"
    if not (cfg_p.exists() and idx_p.exists()):
        alt = Path(parent_dir) / "_meta"
        cfg_p, idx_p = alt / "config.json", alt / "model.safetensors.index.json"
    if cfg_p.exists() and idx_p.exists():
        with open(cfg_p) as fh:
            cfg = json.load(fh)
        with open(idx_p) as fh:
            idx = json.load(fh)
        return build_inventory(cfg, idx)
    return build_inventory()


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXACT BYTE MODEL
# ══════════════════════════════════════════════════════════════════════════════
FP16_BITS = 16
NATIVE_BITS = 16                     # bf16 pass-through
METADATA_BITS_PER_TENSOR = 64 * 8    # same fixed descriptor charge as gravity_forge.ByteLedger
ALIGN_BYTES = 64                     # worst-case padding is billed, never averaged
ALIGN_BITS_PER_TENSOR = (ALIGN_BYTES - 1) * 8
PACKAGING_BITS = 16 * 1024 ** 2 * 8  # container header, manifest, per-tensor checksums
RESERVE_BITS = 32 * 1024 ** 2 * 8    # explicit unallocated headroom, declared not hidden

CODEBOOK_SCOPES = ("per_tensor", "per_layer", "global")
LAYERS = 94


def ceil_log2(k: int) -> int:
    if k < 2 or (k & (k - 1)):
        raise ClosureError(f"codebook cardinality must be a power of two >= 2, got {k}")
    return k.bit_length() - 1


@dataclass(frozen=True)
class VQ:
    """A vector-quantized lane. rate = subspaces * log2(k) / dim, exact."""

    dim: int
    subspaces: int
    k: int
    codebook_scope: str = "per_tensor"
    row_scale: bool = True
    protected_row_frac: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.subspaces < 1 or self.subspaces > self.dim:
            raise ClosureError(f"subspaces must be in 1..dim, got {self.subspaces}/{self.dim}")
        ceil_log2(self.k)   # refuse a non-power-of-two cardinality at construction, not at billing
        if self.codebook_scope not in CODEBOOK_SCOPES:
            raise ClosureError(f"unknown codebook scope {self.codebook_scope!r}")
        if not (Fraction(0) <= self.protected_row_frac < Fraction(1)):
            raise ClosureError("protected_row_frac must be in [0, 1)")

    @property
    def rate(self) -> Fraction:
        return Fraction(self.subspaces * ceil_log2(self.k), self.dim)

    @property
    def label(self) -> str:
        return f"d{self.dim}s{self.subspaces}k{self.k}"

    def as_dict(self) -> dict[str, Any]:
        r = self.rate
        return {
            "kind": "vq", "dim": self.dim, "subspaces": self.subspaces, "k": self.k,
            "codebook_scope": self.codebook_scope, "row_scale": self.row_scale,
            "protected_row_frac": f"{self.protected_row_frac.numerator}/"
                                  f"{self.protected_row_frac.denominator}",
            "index_rate_exact": f"{r.numerator}/{r.denominator}",
            "index_rate_float": float(r), "label": self.label,
        }


@dataclass(frozen=True)
class Native:
    """Kept bf16. Billed under pass_through_tensors, never as free overhead."""

    @property
    def rate(self) -> Fraction:
        return Fraction(NATIVE_BITS, 1)

    @property
    def label(self) -> str:
        return "bf16_native"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "native", "bits_per_weight": NATIVE_BITS, "label": self.label}


@dataclass(frozen=True)
class Omit:
    """Structured omission. The cell ships no weights, only a presence bit.

    SOURCE-CHANGING: the artifact is no longer a coding of the original weights, so the
    rate-distortion limit of those weights does not bind it. The DENOMINATOR is still the
    original weight count, so omission cannot be used to flatter the rate by shrinking
    what it divides by.
    """

    @property
    def rate(self) -> Fraction:
        return Fraction(0)

    @property
    def label(self) -> str:
        return "omitted"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "omit", "bits_per_weight": 0, "label": self.label,
                "changes_source": True}


Spec = VQ | Native | Omit


@dataclass(frozen=True)
class Band:
    """A slice of one organ's tensors carrying one spec."""

    organ: str
    tensors: int
    spec: Spec
    note: str = ""


@dataclass(frozen=True)
class Variant:
    name: str
    pressure_taken_from: str
    rationale: str
    bands: tuple[Band, ...]
    changes_source: bool = False
    runtime_table_bits: int = 0
    doctor_bits: int = 0
    requires: tuple[str, ...] = ()


def _bill_band(organ: Organ, band: Band) -> dict[str, int]:
    """Exact integer bits for one band, itemized into the ten named slots."""
    n = band.tensors
    if n < 0 or n > organ.tensors:
        raise ClosureError(f"{organ.name}: band of {n} tensors out of {organ.tensors}")
    out = {c: 0 for c in ("indices", "codebooks", "scales", "protected_islands",
                          "pass_through_tensors")}
    if n == 0:
        return out
    per_tensor_params = organ.params // organ.tensors

    if isinstance(band.spec, Omit):
        return out
    if isinstance(band.spec, Native):
        out["pass_through_tensors"] = n * per_tensor_params * NATIVE_BITS
        return out

    vq = band.spec
    if not organ.codeable:
        raise ClosureError(f"{organ.name} is a vector organ; it can only be Native or Omit")
    if organ.cols % vq.dim:
        raise ClosureError(
            f"{organ.name}: cols {organ.cols} not divisible by VQ dim {vq.dim}")

    protected_rows = math.ceil(organ.rows * vq.protected_row_frac)
    coded_rows = organ.rows - protected_rows
    if coded_rows <= 0:
        raise ClosureError(f"{organ.name}: protected islands consumed every row")

    coded_params = n * coded_rows * organ.cols
    out["indices"] = (coded_params // vq.dim) * vq.subspaces * ceil_log2(vq.k)
    out["scales"] = n * coded_rows * FP16_BITS if vq.row_scale else 0
    if protected_rows:
        row_ix = max(1, (organ.rows - 1).bit_length())
        out["protected_islands"] = n * protected_rows * (organ.cols * NATIVE_BITS + row_ix)

    sets = {"per_tensor": n, "per_layer": min(n, LAYERS), "global": 1}[vq.codebook_scope]
    out["codebooks"] = sets * vq.k * vq.dim * FP16_BITS
    return out


def bill(variant: Variant, inv: Inventory) -> dict[str, Any]:
    """Complete, itemized bit accounting for one variant. Exact integers throughout."""
    slots = {c: 0 for c in ("indices", "codebooks", "scales", "metadata", "alignment",
                            "protected_islands", "doctor", "pass_through_tensors",
                            "packaging", "runtime_tables")}
    per_organ: dict[str, dict[str, Any]] = {}
    covered: dict[str, int] = {}

    for band in variant.bands:
        organ = inv.organ(band.organ)
        got = _bill_band(organ, band)
        for key, val in got.items():
            slots[key] += val
        covered[band.organ] = covered.get(band.organ, 0) + band.tensors
        row = per_organ.setdefault(band.organ, {"params": organ.params, "bands": [],
                                                "bits": 0})
        band_bits = sum(got.values())
        row["bits"] += band_bits
        row["bands"].append({
            "tensors": band.tensors, "spec": band.spec.as_dict(), "note": band.note,
            "bits": band_bits,
        })

    missing = {name: o.tensors - covered.get(name, 0) for name, o in inv.organs.items()}
    bad = {k: v for k, v in missing.items() if v != 0}
    if bad:
        raise ClosureError(
            f"{variant.name}: every tensor of every organ must be allocated exactly once; "
            f"unallocated/over-allocated counts {bad}")

    # Descriptors and padding are charged for EVERY tensor slot, including omitted cells:
    # the container still has to say the cell is not there.
    slots["metadata"] = inv.tensors * METADATA_BITS_PER_TENSOR
    slots["alignment"] = inv.tensors * ALIGN_BITS_PER_TENSOR
    slots["packaging"] = PACKAGING_BITS
    slots["runtime_tables"] = int(variant.runtime_table_bits)
    slots["doctor"] = int(variant.doctor_bits)

    total = sum(slots.values()) + RESERVE_BITS
    bpw = Fraction(total, inv.params)

    for name, row in per_organ.items():
        row["realized_organ_bpw_exact"] = _frac_str(Fraction(row["bits"], row["params"]))
        row["realized_organ_bpw_float"] = row["bits"] / row["params"]
        row["share_of_complete_bits"] = row["bits"] / total

    receipt: dict[str, Any] = {
        "variant": variant.name,
        "pressure_taken_from": variant.pressure_taken_from,
        "rationale": variant.rationale,
        "changes_source": variant.changes_source,
        "requires": list(variant.requires),
        "components_bits": dict(slots),
        "reserve_bits": RESERVE_BITS,
        "complete_bits": total,
        "original_weight_count": inv.params,
        "complete_bpw_exact": _frac_str(bpw),
        "complete_bpw_float": float(bpw),
        "complete_bytes": math.ceil(total / 8),
        "complete_gib": total / 8 / 1024 ** 3,
        "headroom_bits": inv.params - total,
        "legal": bpw <= CEILING,
        "per_organ": per_organ,
        "scope": "whole_model",
        "expert_only_bpw_is_not_the_model_rate": True,
    }
    if HAVE_OBC:
        receipt["ledger"] = OBC.CompleteByteLedger(
            **{k: v for k, v in slots.items()},
            **{OBC.RESERVE: RESERVE_BITS},
            note=f"{variant.name}: {variant.pressure_taken_from}",
        ).as_dict(inv.params)
    return receipt


def _frac_str(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def check_ceiling(variant: Variant, inv: Inventory) -> dict[str, Any]:
    """Enforce the ceiling. Raises on any candidate above 1/1."""
    receipt = bill(variant, inv)
    bpw = Fraction(receipt["complete_bits"], inv.params)
    if HAVE_OBC:
        ledger = OBC.CompleteByteLedger(
            **receipt["components_bits"], **{OBC.RESERVE: RESERVE_BITS},
            note=variant.name)
        OBC.assert_complete_bpw_le_one(ledger, inv.params)   # raises CeilingViolation
        return receipt
    if bpw > CEILING:
        over = bpw - CEILING
        raise ClosureError(
            f"one-bit ceiling violated by {variant.name}: complete {float(bpw):.9f} BPW "
            f"(exact {_frac_str(bpw)}); overage {float(over):.9f} BPW = "
            f"{receipt['complete_bits'] - inv.params} bits; rebudget, do not raise the ceiling")
    return receipt


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE REBUDGET
# ══════════════════════════════════════════════════════════════════════════════
# Rate vocabulary, all exact and all realizable as (dim, subspaces, k):
#   d16s1k4 0.125 | d32s6k4 0.375 | d32s7k4 0.4375 | d16s2k16 0.5 | d16s3k16 0.75
#   d16s4k16 1.0  | d16s3k64 1.125 | d16s4k32 1.25 | d16s4k64 1.5 | d16s4k256 2.0
#   d8s1k256 1.0 (attn) | d8s2k256 2.0 | d8s3k256 3.0 | d8s4k256 4.0
CELLS = LAYERS * SEALED_CONFIG["num_experts"]          # 12032 layer-expert cells
NEVER_ROUTED_CELLS = int(CELLS * 261 // 1000)          # 26.1 percent measured on the holdout
LIVE_CELLS = CELLS - NEVER_ROUTED_CELLS
QUARTILE = CELLS // 4


def _experts(gate: Spec, up: Spec, down: Spec, n: int = CELLS, note: str = "") -> list[Band]:
    return [Band("expert_gate", n, gate, note), Band("expert_up", n, up, note),
            Band("expert_down", n, down, note)]


def _nonexpert(q_o: Spec, k_v: Spec, embed: Spec, *, router: Spec = Native()) -> list[Band]:
    return [
        Band("attn_q", LAYERS, q_o), Band("attn_o", LAYERS, q_o),
        Band("attn_k", LAYERS, k_v), Band("attn_v", LAYERS, k_v),
        Band("embed_tokens", 1, embed), Band("lm_head", 1, embed),
        Band("router", LAYERS, router, "routers protected: 0.003 BPW buys routing stability"),
        Band("norms", LAYERS * 4 + 1, Native(), "norms are 0.00003 BPW; never worth coding"),
    ]


# ── V1: all pressure on the tolerant organ ────────────────────────────────────
V1 = Variant(
    name="C1_down_pressure",
    pressure_taken_from="expert_down (the tolerant organ) and nothing else",
    rationale=(
        "Preserves A1_1p0's sensitive-organ rate EXACTLY (gate/up 1.25) and pays the whole "
        "illegality out of expert_down, which the measured organ inversion says tolerates "
        "more. Attention, embeddings and the LM head stay generous. If this collapses while "
        "A1 collapsed identically, down's bits were never the binding constraint and the "
        "organ inversion is exhausted as an allocation lever."),
    bands=tuple(
        _experts(VQ(16, 4, 32), VQ(16, 4, 32), VQ(32, 1, 256),
                 note="gate/up 1.25 preserved from A1; down cut 0.5 -> 0.25")
        + _nonexpert(VQ(8, 2, 256), VQ(8, 4, 256), VQ(8, 3, 256))),
)

# ── V2: all pressure on the non-expert organs ─────────────────────────────────
V2 = Variant(
    name="C2_attention_embedding_pressure",
    pressure_taken_from="attention q/k/v/o, embeddings and the LM head",
    rationale=(
        "Keeps BOTH expert organs near A1 (gate/up 1.25, down 0.4375) and buys the legality "
        "by compressing the 3.4 percent of parameters that are not experts. Tests the "
        "opposite hypothesis to C1: that the non-expert organs were quietly holding the "
        "budget hostage and the expert allocation was never the problem."),
    bands=tuple(
        _experts(VQ(16, 4, 32), VQ(16, 4, 32), VQ(32, 7, 4),
                 note="gate/up 1.25 preserved from A1; down 0.5 -> 0.4375")
        + _nonexpert(VQ(8, 1, 256), VQ(8, 2, 256), VQ(16, 3, 256))),
)

# ── V3: narrowed inversion ────────────────────────────────────────────────────
V3 = Variant(
    name="C3_narrowed_inversion",
    pressure_taken_from="expert_gate and expert_up (the sensitive organ), spent on expert_down",
    rationale=(
        "Moves bits the other way: gate/up down to 1.0, down up to 0.75. The organ inversion "
        "is a measured PRIOR, not a law, and it was measured as a failure ordering, not as a "
        "marginal-utility curve. C3 is the control that tells C1 and C2 apart: if C3 beats "
        "them, the inversion is being applied past the point where it still pays."),
    bands=tuple(
        _experts(VQ(16, 4, 16), VQ(16, 4, 16), VQ(16, 3, 16),
                 note="inversion narrowed to 1.0 / 1.0 / 0.75")
        + _nonexpert(VQ(8, 2, 256), VQ(8, 4, 256), VQ(8, 3, 256))),
)

# ── V4: structured omission (SOURCE-CHANGING) ─────────────────────────────────
V4 = Variant(
    name="C4_structured_omission",
    pressure_taken_from="the 26.1 percent of layer-expert cells that never routed on the holdout",
    rationale=(
        "Ships nothing at all for cells that never route, and spends the freed bytes to give "
        "the SURVIVING experts gate/up 1.5, which is HIGHER than the illegal A1_1p0 rate, "
        "while the complete artifact stays under one bit. This is the operator's thesis made "
        "arithmetic: change the model, do not raise the ceiling. The denominator remains the "
        "ORIGINAL weight count, so omission cannot flatter the rate."),
    bands=tuple(
        _experts(VQ(16, 4, 64), VQ(16, 4, 64), VQ(16, 3, 16), n=LIVE_CELLS,
                 note="surviving cells at 1.5 / 1.5 / 0.75")
        + _experts(Omit(), Omit(), Omit(), n=NEVER_ROUTED_CELLS,
                   note="never routed on the holdout; presence bit only")
        + _nonexpert(VQ(8, 2, 256), VQ(8, 4, 256), VQ(8, 3, 256))),
    changes_source=True,
    runtime_table_bits=CELLS,   # one presence bit per layer-expert cell
    requires=("routing census over >= 1000 calibration tokens on a disjoint holdout",),
)

# ── V5: hot/cold stratified allocation ────────────────────────────────────────
_HOT = QUARTILE
_COLD = QUARTILE
_MID = CELLS - _HOT - _COLD
V5 = Variant(
    name="C5_hot_cold_stratified",
    pressure_taken_from="the coldest routing quartile, spent on the hottest quartile",
    rationale=(
        "Rate follows measured routing mass: the hot quartile gets gate/up 2.0 (double the "
        "illegal A1 rate), the middle half gets 1.0, the cold quartile gets 0.5. Deliberately "
        "the QUARTILE band and not the median split, because the 88-token calibration is only "
        "63.6 percent stable at the median. Requires a >= 1000 token census; 88 tokens is "
        "refused."),
    bands=tuple(
        _experts(VQ(16, 4, 256), VQ(16, 4, 256), VQ(16, 2, 16), n=_HOT,
                 note="hot quartile 2.0 / 2.0 / 0.5")
        + _experts(VQ(16, 4, 16), VQ(16, 4, 16), VQ(16, 2, 16), n=_MID,
                   note="middle half 1.0 / 1.0 / 0.5")
        + _experts(VQ(16, 2, 16), VQ(16, 2, 16), VQ(32, 6, 4), n=_COLD,
                   note="cold quartile 0.5 / 0.5 / 0.375")
        + _nonexpert(VQ(8, 2, 256), VQ(8, 4, 256), VQ(8, 3, 256))),
    runtime_table_bits=CELLS * 2,   # two stratum bits per cell
    requires=("routing census over >= 1000 calibration tokens; the 88-token split is refused",),
)

# ── V6: same-budget Doctor ────────────────────────────────────────────────────
V6 = Variant(
    name="C6_same_budget_doctor",
    pressure_taken_from="index bits, converted into shipped Doctor correction bytes",
    rationale=(
        "Spends a fixed 1.5 GiB slice of the SAME budget on Doctor correction bytes instead "
        "of more index bits: gate/up drop to 1.0 to pay for it. Isolates the one question the "
        "index-rate ladder cannot answer, namely whether a bit is worth more as resolution or "
        "as repair."),
    bands=tuple(
        _experts(VQ(16, 4, 16), VQ(16, 4, 16), VQ(16, 2, 16),
                 note="1.0 / 1.0 / 0.5, the slack handed to Doctor")
        + _nonexpert(VQ(8, 2, 256), VQ(8, 4, 256), VQ(8, 3, 256))),
    doctor_bits=1536 * 1024 ** 2 * 8,
)

VARIANTS: tuple[Variant, ...] = (V1, V2, V3, V4, V5, V6)

# The illegal historical candidate, replayed through THIS ledger. Kept as a rejection
# fixture: a rebudget that only looks legal under a slacker ruler is not a rebudget.
A1_REPLAY = Variant(
    name="A1_1p0_replay_ILLEGAL",
    pressure_taken_from="nowhere; this is the candidate that broke the ceiling",
    rationale=(
        "gate/up 1.25, down 0.5, non-expert organs at 1.0, routers and norms native. Sealed "
        "as complete 1.007471652 BPW on the campaign ledger; this stricter ledger, which "
        "bills worst-case alignment and every per-row scale, puts it higher still."),
    bands=tuple(
        _experts(VQ(16, 4, 32), VQ(16, 4, 32), VQ(16, 2, 16))
        + _nonexpert(VQ(8, 1, 256), VQ(8, 1, 256), VQ(8, 1, 256))),
)
A1_SEALED_COMPLETE_BPW = Fraction(1_007_471_652, 1_000_000_000)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIDELITY TIERS AND THE SUB-BIT SCHEDULING RULE
# ══════════════════════════════════════════════════════════════════════════════
FIDELITY_TIERS: tuple[str, ...] = (
    "physical",          # bytes, exact BPW, checksums. Says nothing about capability.
    "tensor",            # per-tensor reconstruction / codeword-occupancy statistics.
    "expert",            # one expert's output on real activations.
    "layer",             # one layer's output divergence on real activations.
    "short_end_to_end",  # real parent-vs-packed forward, few prompts, non-holdout.
    "capability",        # real forward, holdout, gate metrics, >= 1000 tokens, all domains.
)
_TIER_RANK = {t: i for i, t in enumerate(FIDELITY_TIERS)}
CHEAP_TIERS = FIDELITY_TIERS[:4]     # physical / tensor / expert / layer
EXPENSIVE_TIERS = FIDELITY_TIERS[4:]  # short_end_to_end / capability

# Probe rates that stay ACTIVE below the ceiling. Exact rationals only.
SUBBIT_PROBE_RATES: tuple[Fraction, ...] = (
    Fraction(17, 20), Fraction(3, 4), Fraction(2, 3), Fraction(1, 2),
    Fraction(2, 5), Fraction(1, 3), Fraction(1, 4),
)


def may_schedule(rate: Fraction | str, tier: str, *,
                 one_bit_method_selected: bool = False) -> dict[str, Any]:
    """The scheduling rule. Returns {allowed, reason}; never raises for a legal refusal.

    Three refusals:
      1. anything above 1/1 is refused outright, at every tier;
      2. below 1/1, only the cheap tiers run until a serious one-bit method has been
         SELECTED on capability evidence. Full-model capability compute is not spent
         proving that sub-bit collapses again;
      3. an unknown tier is refused rather than silently treated as cheap.
    """
    q = rate if isinstance(rate, Fraction) else _parse_rate(rate)
    if tier not in _TIER_RANK:
        return {"allowed": False, "rate": _frac_str(q), "tier": tier,
                "reason": f"unknown fidelity tier {tier!r}; expected one of {FIDELITY_TIERS}"}
    if q > CEILING:
        return {"allowed": False, "rate": _frac_str(q), "tier": tier,
                "reason": ("above the one-bit ceiling; upward bracketing is REJECTED and no "
                           "escape receipt or safety anchor may be scheduled")}
    if q == CEILING:
        return {"allowed": True, "rate": _frac_str(q), "tier": tier,
                "reason": "at the ceiling; every tier including capability is in scope"}
    if tier in EXPENSIVE_TIERS and not one_bit_method_selected:
        return {"allowed": False, "rate": _frac_str(q), "tier": tier,
                "reason": ("sub-bit probe: full-model capability compute is NOT spent below one "
                           "bit until a serious one-bit method is selected on capability "
                           "evidence. Physical, tensor, expert and layer tiers stay active.")}
    return {"allowed": True, "rate": _frac_str(q), "tier": tier,
            "reason": ("sub-bit probe at a cheap tier" if tier in CHEAP_TIERS
                       else "a one-bit method is selected; sub-bit may now spend real compute")}


def _parse_rate(text: Any) -> Fraction:
    """Exact rationals only. '0.85' is refused; pass '17/20'."""
    if isinstance(text, Fraction):
        return text
    if isinstance(text, int):
        return Fraction(text, 1)
    s = str(text).strip()
    if "/" in s:
        n, d = s.split("/", 1)
        return Fraction(int(n), int(d))
    if s.lstrip("-").isdigit():
        return Fraction(int(s), 1)
    raise ClosureError(f"rate must be an exact rational 'n/d', not {text!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. THE CLOSURE METHOD PROGRAM  (decisive first)
# ══════════════════════════════════════════════════════════════════════════════
# rank: cheapest decisive kill first. changes_source marks the methods that are NOT bound
# by the rate-distortion limit of the original weights.
CLOSURE_METHODS: tuple[dict[str, Any], ...] = (
    {
        "rank": 1, "id": "M01_row_norm_stratified_codebooks", "changes_source": False,
        "changes": ("Partition each gate/up tensor's rows into norm strata and fit a separate "
                    "codebook per stratum, plus one stratum id per row. Equivalently: the "
                    "codebook stops being fitted on a distribution spanning five orders of "
                    "magnitude."),
        "why_not_falsified": ("The atlas kills entropy coding of LLOYD-optimal indices because "
                              "they are near-uniform. The measurement here is the opposite "
                              "pathology: 94 percent of gate/up rows collapse onto ONE codeword, "
                              "so the realized index entropy is far BELOW uniform and the billed "
                              "rate is not being spent. This is the atlas's own named reopen "
                              "condition for biased/stratified codebooks."),
        "byte_cost": ("S extra codebooks amortized over the stratum, plus ceil(log2 S) bits per "
                      "row. At S=4, d16 k32, per-tensor scope: about 0.0002 whole-model BPW. "
                      "Fits inside every C-variant's reserve without moving an index rate."),
        "first_tier": "tensor",
        "falsification": ("Measure codeword occupancy per gate/up tensor before and after. If "
                          "the single-codeword share does not fall from 94 percent below 20 "
                          "percent, or if tensor reconstruction error does not improve at the "
                          "identical exact rate, the lever is dead and goes to the atlas."),
    },
    {
        "rank": 2, "id": "M02_activation_aware_fitting", "changes_source": False,
        "changes": ("Fit codebooks and assignments to minimize output error on real calibration "
                    "activations (a Hessian/Fisher-weighted objective), not weight-space MSE. "
                    "The reconstruction stops being a conditional mean of the weights."),
        "why_not_falsified": ("The atlas's post-hoc scalar gain is pinned at exactly 1.0 because "
                              "k-means reconstruction is a conditional mean whose residual is "
                              "orthogonal to the reconstruction. That algebra holds IN WEIGHT "
                              "SPACE and is precisely what an activation-weighted objective "
                              "breaks: the residual is no longer orthogonal under the activation "
                              "metric, so gain and assignment both regain leverage."),
        "byte_cost": ("Zero artifact bytes. The fitting is offline; the shipped format is "
                      "unchanged. Cost is calibration compute only."),
        "first_tier": "expert",
        "falsification": ("Same exact rate, same geometry, two fits. If activation-aware fitting "
                          "does not reduce per-expert OUTPUT divergence on held-out activations, "
                          "the objective is not the binding constraint."),
    },
    {
        "rank": 3, "id": "M03_high_dim_vq_chunked_assignment", "changes_source": False,
        "changes": ("Raise VQ dimension at a FIXED index rate using chunked k-means assignment, "
                    "so k=65536 at d=32 becomes computable where the in-core solver OOMs."),
        "why_not_falsified": ("Already measured monotone in the right direction at fixed rate: "
                              "d8k16 0.782, d16k256 0.757, d32k65536 0.668 rel_error. Nothing in "
                              "the atlas touches space-filling gain."),
        "byte_cost": ("Codebook cost explodes with k and must be amortized: k=65536 at d=32 is "
                      "33.5 Mbit per codebook set, which is 0.0134 whole-model BPW at per-layer "
                      "scope and CATASTROPHIC at per-tensor scope. The scope choice is billed, "
                      "not assumed."),
        "first_tier": "tensor",
        "falsification": ("Extrapolate the measured rel_error curve. If d64 k=2^20 does not "
                          "continue the trend at fixed rate, or if the codebook amortization "
                          "erases the gain in COMPLETE BPW, the axis is exhausted."),
    },
    {
        "rank": 4, "id": "M04_residual_additive_codebooks", "changes_source": False,
        "changes": ("Split the same total rate into two additive stages: a coarse codebook plus "
                    "a residual codebook fitted on what the first stage missed."),
        "why_not_falsified": ("The atlas kills delta coding ACROSS experts (pairwise cosine 1e-4, "
                              "no shared component). A residual stage is within a single tensor "
                              "against its own first-stage error, where the residual is by "
                              "construction non-trivial."),
        "byte_cost": ("Rate-neutral by construction: 4+4 bits instead of 8. Second codebook adds "
                      "one more k*d*16 term per scope unit, the same order as the first."),
        "first_tier": "tensor",
        "falsification": ("At identical exact total rate, if a 2-stage residual split does not "
                          "beat the 1-stage code on tensor error AND on expert output "
                          "divergence, additive structure buys nothing here."),
    },
    {
        "rank": 5, "id": "M05_structured_expert_omission", "changes_source": True,
        "changes": ("Ship no weights for layer-expert cells that never route, only a presence "
                    "bit, and spend the freed bytes on the surviving experts."),
        "why_not_falsified": ("Not a coding claim at all, so no coding result can falsify it. "
                              "26.1 percent of cells never routed on the holdout. Because it "
                              "changes the MODEL rather than the code, the rate-distortion limit "
                              "of the original weights does not bind it."),
        "byte_cost": ("One presence bit per cell (12032 bits total, 5e-8 BPW). Frees ~26 percent "
                      "of expert bytes. Realized in variant C4, which reaches gate/up 1.5 under "
                      "a complete rate below one bit."),
        "first_tier": "expert",
        "falsification": ("Re-census routing over >= 1000 held-out tokens. If the never-routed "
                          "set is not stable across disjoint splits, or if omitting it moves "
                          "layer output divergence at all, the free bytes were not free."),
        "guard": "88-token calibration is refused; >= 1000 tokens required",
    },
    {
        "rank": 6, "id": "M06_tensor_organ_specific_representation", "changes_source": False,
        "changes": ("Stop shipping one representation family everywhere. Different organs get "
                    "different families: heavy-tailed down projections get an outlier-split "
                    "code, near-Gaussian gate/up get a lattice, embeddings get a shared-vocab "
                    "code."),
        "why_not_falsified": ("The atlas kills UNIFORM sub-bit allocation and kills ternary as a "
                              "GLOBAL substitute for VQ. Neither result says the same family is "
                              "optimal for every organ; the measured organ inversion says the "
                              "opposite."),
        "byte_cost": ("Per-organ codebook and side tables. Budget 0.005 whole-model BPW; the "
                      "reserve in every C-variant covers it."),
        "first_tier": "layer",
        "falsification": ("Per-organ bake-off at matched exact rate. If the best family is the "
                          "same family for every organ, the axis collapses back to one code and "
                          "the lever is dead."),
    },
    {
        "rank": 7, "id": "M07_global_marginal_utility_allocation", "changes_source": False,
        "changes": ("Replace hand-set per-organ rates with a global allocator that measures "
                    "d(output divergence)/d(bit) per organ and equalizes the marginal utility "
                    "across the whole model under the 1/1 constraint."),
        "why_not_falsified": ("The organ inversion is a measured FAILURE ORDERING, not a "
                              "marginal-utility curve. C1, C2 and C3 exist precisely to produce "
                              "the three points this allocator needs; nothing in the atlas "
                              "measures a slope."),
        "byte_cost": ("Zero artifact bytes. The allocation table is already in metadata."),
        "first_tier": "layer",
        "falsification": ("If the equalized-marginal allocation does not beat the best "
                          "hand-set C-variant at identical complete BPW, hand allocation was "
                          "already at the optimum and the allocator is ceremony."),
        "requires": "C1, C2 and C3 layer-tier results",
    },
    {
        "rank": 8, "id": "M08_router_sensitive_protection", "changes_source": False,
        "changes": ("Quantize with the ROUTER's decision as the objective: preserve the top-k "
                    "argmax and the top-k margin, not the router's weights. Cells near the "
                    "decision boundary get protection."),
        "why_not_falsified": ("Untouched by the atlas. The dominant failure organ is gate, which "
                              "is what the router selects INTO, so a routing flip and an expert "
                              "error are confounded in every result so far and have never been "
                              "separated."),
        "byte_cost": ("Routers native is 0.00335 whole-model BPW and is already bought in every "
                      "C-variant. Margin protection adds a per-layer boundary table, ~1e-4 BPW."),
        "first_tier": "layer",
        "falsification": ("Measure top-8 routing agreement parent vs packed. If it is already "
                          ">= 0.99 with native routers, routing is not a failure channel and "
                          "this lever is dead on arrival."),
    },
    {
        "rank": 9, "id": "M09_stable_hot_cold_allocation", "changes_source": False,
        "changes": ("Rate follows measured routing mass, on the QUARTILE partition rather than "
                    "the median split. Realized as variant C5."),
        "why_not_falsified": ("The atlas killed the 88-TOKEN CALIBRATION, not the lever, and "
                              "names >= 1000 tokens as the reopen condition. The instability is "
                              "concentrated at the median, so the quartile band is the "
                              "defensible partition."),
        "byte_cost": ("Two stratum bits per layer-expert cell: 24064 bits, 1e-7 BPW."),
        "first_tier": "expert",
        "falsification": ("Bootstrap the quartile membership over disjoint >= 1000 token splits. "
                          "If quartile membership is under 90 percent stable, refuse the lever "
                          "again and record the token count that would reopen it."),
        "guard": "88-token calibration is refused; >= 1000 tokens required",
    },
    {
        "rank": 10, "id": "M10_sparse_functional_correction", "changes_source": False,
        "changes": ("Ship a small sparse or rank-1 correction placed where measured OUTPUT error "
                    "is worst, instead of raising the index rate everywhere."),
        "why_not_falsified": ("Distinct from post-hoc scalar gain, which is one global scalar "
                              "pinned at 1.0 by conditional-mean algebra. A sparse correction is "
                              "chosen by output error and is not a conditional mean of anything."),
        "byte_cost": ("Explicit: nnz * (value 16 bits + index bits). Budget 0.01 whole-model BPW "
                      "and take it out of the index rate, never out of the reserve."),
        "first_tier": "layer",
        "falsification": ("At identical complete BPW, if moving 0.01 BPW from indices into sparse "
                          "correction does not reduce layer output divergence, corrections lose "
                          "to resolution and the lever is dead."),
    },
    {
        "rank": 11, "id": "M11_same_budget_doctor", "changes_source": False,
        "changes": ("Spend a fixed slice of the SAME budget on shipped Doctor repair bytes "
                    "rather than on index bits. Realized as variant C6."),
        "why_not_falsified": ("Doctor has never been run inside a fixed complete-BPW envelope; "
                              "every prior Doctor result added bytes on top. The atlas has no "
                              "entry for budget-neutral repair."),
        "byte_cost": ("1.5 GiB shipped, exactly 0.0548 whole-model BPW, paid for by dropping "
                      "gate/up from 1.25 to 1.0. Billed in the doctor slot, never as overhead."),
        "first_tier": "short_end_to_end",
        "falsification": ("C6 against C3 at matched complete BPW. If C6 does not beat C3 on the "
                          "gate metrics, a bit is worth more as resolution than as repair and "
                          "same-budget Doctor is closed."),
    },
    {
        "rank": 12, "id": "M12_compressibility_training", "changes_source": True,
        "changes": ("Continue-train the parent with a penalty that makes its weights cheap to "
                    "code: pull rows toward a shared codebook, flatten the row-norm spread that "
                    "currently spans 1e-5..0.91."),
        "why_not_falsified": ("Every atlas entry is a statement about THESE weights. This method "
                              "produces different weights, so the rate-distortion limit measured "
                              "on the originals does not bind it. It also attacks the exact "
                              "pathology M01 only works around."),
        "byte_cost": ("Zero artifact bytes; the format is unchanged. Cost is training compute and "
                      "a re-derived parent reference."),
        "first_tier": "expert",
        "falsification": ("Train a small proxy, then measure whether the same exact rate yields "
                          "lower output divergence than the untrained parent at that rate. If "
                          "not, compressibility training does not transfer."),
        "note": "the parent BF16 model remains the quality reference; a trained parent is a NEW "
                "parent and needs its own reference forward",
    },
    {
        "rank": 13, "id": "M13_quantization_aware_training", "changes_source": True,
        "changes": ("Train through the one-bit codec with straight-through estimation so the "
                    "weights adapt to the code instead of the code chasing the weights."),
        "why_not_falsified": ("Not bound by the original weights' rate-distortion limit. Every "
                              "collapse measured so far is post-hoc coding of a fixed parent."),
        "byte_cost": ("Zero artifact bytes beyond the chosen code."),
        "first_tier": "expert",
        "falsification": ("QAT one layer at the C3 allocation. If layer output divergence does "
                          "not improve over post-hoc fitting at the identical exact rate, QAT "
                          "does not rescue this rate."),
    },
    {
        "rank": 14, "id": "M14_distillation_into_the_one_bit_student", "changes_source": True,
        "changes": ("Treat the one-bit artifact as a student and distil the BF16 parent's "
                    "logits and routing into it, rather than reconstructing its weights."),
        "why_not_falsified": ("Changes the source. Also directly targets the metric the contract "
                              "gates on (symmetric KL against the parent), which no weight-space "
                              "method optimizes."),
        "byte_cost": ("Zero artifact bytes."),
        "first_tier": "short_end_to_end",
        "falsification": ("Distil a short schedule at the C3 allocation. If mean symmetric KL "
                          "does not fall by an order of magnitude from the measured 7.6-10.9, "
                          "distillation does not close a gap this large."),
    },
    {
        "rank": 15, "id": "M15_learned_sharing_generated_weights", "changes_source": True,
        "changes": ("Replace stored expert weights with a learned generator: a shared basis plus "
                    "small per-expert coefficients, or a hypernetwork producing expert weights "
                    "from a compact code."),
        "why_not_falsified": ("The atlas killed shared bases and cluster-mean subtraction on the "
                              "RAW weights, where pairwise expert cosine is 1e-4. A LEARNED "
                              "sharing structure is trained to be shareable; the 1e-4 measurement "
                              "is about the parent's arrangement, not about what a trained "
                              "generator can represent. The atlas's own reopen condition is a "
                              "parent whose expert cosine is >= 0.10, and this method MAKES one."),
        "byte_cost": ("Generator parameters plus per-expert codes, both billed in full. The "
                      "generator is a runtime table and lands in the runtime_tables slot."),
        "first_tier": "expert",
        "falsification": ("Fit a generator for one layer's 128 experts at a complete rate below "
                          "one bit. If reconstructed expert outputs do not beat the C3 code at "
                          "matched complete BPW, learned sharing loses to direct coding."),
    },
)

DECISIVE_ORDER: tuple[str, ...] = tuple(m["id"] for m in CLOSURE_METHODS)


def source_changing_methods() -> tuple[str, ...]:
    return tuple(m["id"] for m in CLOSURE_METHODS if m["changes_source"])


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROGRAM ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════
def build_program(inv: Inventory | None = None) -> dict[str, Any]:
    inv = inv or load_parent()

    receipts, rejected = [], []
    for variant in VARIANTS:
        try:
            receipts.append(check_ceiling(variant, inv))
        except Exception as exc:  # noqa: BLE001 - a rejection is data, not a crash
            rejected.append({"variant": variant.name, "rejected_because": str(exc)})
    if not receipts:
        raise ClosureError("no legal variant survived the ceiling; the rebudget failed")

    a1 = bill(A1_REPLAY, inv)
    a1_bpw = Fraction(a1["complete_bits"], inv.params)

    schedule = []
    for q in SUBBIT_PROBE_RATES:
        row = {"rate_exact": _frac_str(q), "rate_float": float(q), "tiers": {}}
        for tier in FIDELITY_TIERS:
            row["tiers"][tier] = may_schedule(q, tier, one_bit_method_selected=False)["allowed"]
        schedule.append(row)

    program: dict[str, Any] = {
        "schema": SCHEMA,
        "governing_law": (
            "Hawking does not climb above one bit to discover where conventional quantization "
            "works. Hawking changes the representation, model, allocation and treatment until "
            "useful intelligence survives at one bit or below."),
        "ceiling": {
            "rule": "complete_artifact_bits / original_weight_count <= 1/1",
            "value_exact": _frac_str(CEILING),
            "enforced_by": "one_bit_ceiling.assert_complete_bpw_le_one" if HAVE_OBC
                           else "subbit_closure.check_ceiling (local guard; sibling module absent)",
            "complete_bits_include": [
                "indices", "codebooks", "scales", "metadata", "alignment", "protected_islands",
                "doctor", "pass_through_tensors", "packaging", "runtime_tables",
                "and an explicitly declared reserve",
            ],
            "forbidden": [
                "any candidate above 1 BPW",
                "1.2 / 1.5 / 2.0 / 3.0 safety or quality anchors",
                "automatic escape receipts",
                "upward bracketing",
                "reporting an expert-only or payload-only BPW as the whole-model BPW",
            ],
            "quality_reference": "the parent BF16 model; a compressed high-rate anchor is NOT "
                                 "required and must not be scheduled",
        },
        "parent": {
            "repo": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "revision": "ac9c66cc9b46af7306746a9250f23d47083d689e",
            "original_weight_count": inv.params,
            "tensor_count": inv.tensors,
            "source_bytes_bf16": inv.params * 2,
            "inventory_source": inv.source,
            "organs": {n: {"tensors": o.tensors, "rows": o.rows, "cols": o.cols,
                           "params": o.params, "share": o.params / inv.params,
                           "sensitive": o.sensitive}
                       for n, o in inv.organs.items()},
        },
        "measured_context": {
            "A1_1p0": {"sealed_complete_bpw": float(A1_SEALED_COMPLETE_BPW),
                       "verdict": "COLLAPSE 6/6, symKL 7.6-10.9, argmax agreement 0.0",
                       "status": "ILLEGAL under the ceiling; rebudgeted here"},
            "R2_subhalf": {"sealed_complete_bpw": 0.4930,
                           "verdict": "COLLAPSE 6/6, symKL 9.3-13.5, argmax agreement 0.0"},
            "parent_control": "healthy, ppl 1.61-39.33 over 6 prompts, 94 layers",
            "correct_interpretation": (
                "a negative result for the RAW-WEIGHT PQ/VQ representation family at ~1 bit. "
                "NOT evidence that every Hawking method below one bit is impossible. Methods "
                "that change the source are not bound by the rate-distortion limit of the "
                "original weights."),
            "negative_transfer_preserved": [
                "inter-expert similarity negligible (pairwise cosine 1e-4)",
                "entropy coding of trained indices buys 0.0-0.7 percent",
                "post-hoc scalar gain pinned at exactly 1.0 in weight space",
                "uniform allocation fails",
                "the current VQ geometry fails",
                "94 percent of gate/up rows collapse onto ONE codeword; row norms span 1e-5..0.91",
                "26.1 percent of layer-expert cells never routed on the holdout",
                "88 calibration tokens is not evidence; >= 1000 required",
            ],
            "use_of_these_findings": "to stop dead methods, never to raise the ceiling",
        },
        "rejected_candidates": rejected + [{
            "variant": a1["variant"],
            "complete_bpw_exact": a1["complete_bpw_exact"],
            "complete_bpw_float": a1["complete_bpw_float"],
            "sealed_campaign_bpw": float(A1_SEALED_COMPLETE_BPW),
            "legal": a1["legal"],
            "rejected_because": (
                f"complete {a1['complete_bpw_float']:.9f} BPW > 1/1; overage "
                f"{a1['complete_bits'] - inv.params} bits. This ledger is stricter than the "
                f"campaign ledger that sealed {float(A1_SEALED_COMPLETE_BPW):.9f}, so no variant "
                f"below is made legal by a slacker ruler."),
        }],
        "legal_variants": receipts,
        "closure_methods": list(CLOSURE_METHODS),
        "decisive_first_order": list(DECISIVE_ORDER),
        "source_changing_methods": list(source_changing_methods()),
        "source_changing_note": (
            "These methods CHANGE THE SOURCE, so the rate-distortion limit of the original "
            "weights does not bind them. That distinction is the whole reason the one-bit "
            "ceiling is defensible: when raw-weight coding runs out, the answer is to change "
            "the model, never to raise the rate."),
        "fidelity_tiers": list(FIDELITY_TIERS),
        "subbit_probe_policy": {
            "rates_active": [_frac_str(q) for q in SUBBIT_PROBE_RATES],
            "allowed_tiers_below_one_bit": list(CHEAP_TIERS),
            "refused_tiers_below_one_bit": list(EXPENSIVE_TIERS),
            "rule": ("Lower-rate probes stay ACTIVE at physical / tensor / expert / layer "
                     "fidelity. Full-model capability compute is NOT spent below one bit until "
                     "a serious one-bit method is selected on capability evidence."),
            "unlock_condition": ("a one-bit method passes mean symmetric KL <= 0.10 AND "
                                 "next-token argmax agreement >= 0.95 on >= 1000 holdout tokens"),
            "schedule": schedule,
        },
        "claim": ("BYTE PLAN AND PROGRAM ONLY. Every variant here is PHYSICAL evidence. No "
                  "capability claim is made or implied; only a real parent-vs-packed forward "
                  "on holdout tokens may select a frontier."),
    }
    program["sha256"] = hashlib.sha256(
        json.dumps(program, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return program


def repo_root() -> Path:
    return _HERE.parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebudget to <= 1/1 and emit the closure program.")
    ap.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    ap.add_argument("--output", type=Path, default=repo_root() / REPORT_RELPATH)
    ap.add_argument("--summary", action="store_true", help="print the variant table only")
    args = ap.parse_args(argv)

    program = build_program(load_parent(args.parent_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(program, indent=2, sort_keys=True, default=str) + "\n")

    if args.summary:
        print(f"{'variant':34s} {'complete BPW (exact)':>46s} {'float':>12s}  pressure")
        for r in program["legal_variants"]:
            print(f"{r['variant']:34s} {r['complete_bpw_exact']:>46s} "
                  f"{r['complete_bpw_float']:12.9f}  {r['pressure_taken_from']}")
        rej = program["rejected_candidates"][-1]
        print(f"{rej['variant']:34s} {rej['complete_bpw_exact']:>46s} "
              f"{rej['complete_bpw_float']:12.9f}  REJECTED")
    else:
        print(json.dumps(program, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
