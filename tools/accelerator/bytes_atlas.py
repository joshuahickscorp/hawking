"""The BYTES column of the token execution atlas (S031 s3).

TOKEN_EXECUTION_ATLAS_COUNTS.json discharged the COUNT column and said so:
"No dispatch here carries a duration, bytes moved, or FLOPs". Then
ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE.json measured that a removed dispatch is
priced by its BYTES -- 1.042 us when it moves none, 70.708 us when it reads
weights, a factor of 68. So the dispatch ladder 964 -> 200 is expressed in the
column that is not the price, and this builds the one that is.

STATIC, not instrumented. The sealed artifact's catalog carries one segment per
tensor with an exact stored byte count, so per-dispatch bytes are read off the
artifact rather than measured through a runtime. That needs no GPU and cannot be
perturbed by contention -- but it prices the WEIGHT traffic only, which is the
claim boundary and is stated rather than assumed: activations, the KV cache and
the DeltaNet recurrent state are NOT in the catalog and are NOT counted here.

The reconciliation is what makes it falsifiable. A per-dispatch attribution that
sums to the catalog total, over a catalog whose total equals the bytes on disk,
over an artifact whose 8*bytes/params equals its own sealed EBPW, cannot have
dropped or double-counted a tensor. `assert_reconciles` refuses when it does.
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

MAGIC = b"HQ38M20\0"
RECORD_SIZE = 128

# The sealed resident. S031 s0 binds the artifact, not the tag.
RESIDENT = Path(os.path.expanduser("~/noetic/NOETIC_PARENT_A"))
PARENT_PARAMS = 26_895_998_464
SEALED_COMPLETE_EBPW = 3.139300850311054

# This machine's MEASURED bandwidth roof, GB/s, from MACHINE_GENOME.json -- the
# 589.73 median at 1.89% IQR, NOT a datasheet figure and NOT the 595.9 that a
# kernel family's scoring reference was once promoted into.
MEASURED_ROOF_GB_S = 589.73

# The recorded frontier of the sealed body (S031 s0).
SEALED_RAW_TPS = 34.14
SEALED_CAPABILITY = 30 / 43


class Unreconciled(RuntimeError):
    """The per-dispatch attribution does not account for the artifact."""


def parse_catalog(path: Path) -> list[tuple[str, int]]:
    """(tensor_name, stored_bytes) per record, in catalog order.

    Asserts the record<->segment bijection rather than assuming it: the byte
    count is read off the SEGMENT, so if two records shared a segment the sum
    would silently double-count.
    """
    b = path.read_bytes()
    if b[:8] != MAGIC:
        raise Unreconciled(f"bad catalog magic {b[:8]!r} in {path}")
    _ver, n_rec, n_seg, _a, name_len, _c = struct.unpack("<IIIIII", b[8:32])
    off = 32
    segs: dict[int, int] = {}
    for _ in range(n_seg):
        sid, nlen, nbytes, _dg = struct.unpack("<HHQ32s", b[off:off + 44])
        off += 44
        segs[sid] = nbytes
        off += nlen
    tbl = b[off:off + n_rec * RECORD_SIZE]
    names = b[off + n_rec * RECORD_SIZE:]
    if len(names) != name_len:
        raise Unreconciled("name blob length disagrees with the header")
    out, seen = [], set()
    for i in range(n_rec):
        r = tbl[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
        noff, nlen = struct.unpack("<IH", r[0:6])
        sid = struct.unpack("<H", r[36:38])[0]
        if sid in seen:
            raise Unreconciled(
                f"segment {sid} is referenced by more than one record; the byte "
                "count is per SEGMENT so a shared segment would be counted twice")
        seen.add(sid)
        out.append((names[noff:noff + nlen].decode(), segs[sid]))
    return out


def roles(catalog: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Collapse the layer index so 64 layers of one tensor become one role."""
    out: dict[str, list[int]] = {}
    for name, nbytes in catalog:
        out.setdefault(re.sub(r"\.layers\.\d+\.", ".layers.N.", name), []).append(nbytes)
    return out


L = "language_model.model.layers.N."
# kernel -> (dispatches per token, roles read, whole_tensor_per_token)
# Counts are TOKEN_EXECUTION_ATLAS_COUNTS.json's measured histogram, not declared.
DISPATCH_MAP: tuple[tuple[str, int, tuple[str, ...], bool], ...] = (
    ("qwen_uniform_q4_group64_matvec_geo_tpr64_tg128", 209, (
        L + "linear_attn.in_proj_qkvz.weight", L + "linear_attn.in_proj_ba.weight",
        L + "linear_attn.out_proj.weight", L + "self_attn.q_proj.weight",
        L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight",
        L + "self_attn.o_proj.weight", "language_model.lm_head.weight"), True),
    ("qwen_affine_q2_group32_matvec_geo_tpr64_tg128", 192, (
        L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight",
        L + "mlp.down_proj.weight"), True),
    # The ONE dispatch that does not read its whole tensor: decode looks up a
    # single row of the embedding table. Counting the table here would inflate
    # per-token traffic by 675 MB and is the obvious way to get this wrong.
    ("qwen_uniform_q4_embedding_lookup", 1, ("language_model.model.embed_tokens.weight",), False),
    ("qwen80_residual_rmsnorm_tg", 129, (
        L + "input_layernorm.weight", L + "post_attention_layernorm.weight",
        "language_model.model.norm.weight"), True),
    ("qwen80_deltanet_gated_rmsnorm_tg", 48, (L + "linear_attn.norm.weight",), True),
    ("qwen38_qkvz_rearrange_conv_l2_f32", 48, (L + "linear_attn.conv1d.weight",), True),
    ("qwen80_ba_to_decay_beta_f32", 48, (
        L + "linear_attn.A_log", L + "linear_attn.dt_bias"), True),
    ("qwen38_gqa_qk_norm_rope_cache_tg", 16, (
        L + "self_attn.q_norm.weight", L + "self_attn.k_norm.weight"), True),
    # Weight-free by construction. They read activations, the KV cache and the
    # DeltaNet recurrent state, none of which the catalog describes, so their
    # WEIGHT bytes are exactly zero and their real traffic is NOT measured here.
    ("qwen38_attention_apply_sigmoid_gate", 16, (), True),
    ("qwen_next_add_residual", 128, (), True),
    ("gk_swiglu_f32", 64, (), True),
    ("qwen38_gated_delta_decode_vi_simd", 48, (), True),
    ("mha_decode_f32", 16, (), True),
    ("sample_argmax_f32", 1, (), True),
)

EMBED_HIDDEN = 5120
EMBED_VOCAB = 248320

# The largest traffic the catalog CANNOT describe. Named with a number rather
# than left as "activations too": the DeltaNet recurrent state is read and
# written once per token per layer and is the only weight-free term big enough
# to move the bandwidth arithmetic.
DELTANET_LAYERS = 48
DELTANET_STATE_BYTES = 48 * 128 * 128 * 4  # value heads x key dim x value dim, f32


def build(root: Path = RESIDENT) -> dict:
    catalog = parse_catalog(root / "catalog.hq38m20")
    by_role = roles(catalog)
    catalog_total = sum(n for _, n in catalog)

    disk = sum(os.path.getsize(root / "segments" / f)
               for f in os.listdir(root / "segments"))

    rows, claimed, unmapped_check = [], set(), 0
    for kernel, n, role_names, whole in DISPATCH_MAP:
        b = 0
        for rn in role_names:
            if rn not in by_role:
                raise Unreconciled(
                    f"{kernel} claims role {rn!r} which the catalog does not hold")
            if rn in claimed:
                raise Unreconciled(f"role {rn!r} is claimed by two kernels")
            claimed.add(rn)
            tot = sum(by_role[rn])
            if whole:
                b += tot
            else:
                # One row: nibbles + one f16 scale per group of 64.
                b += EMBED_HIDDEN // 2 + (EMBED_HIDDEN // 64) * 2
                unmapped_check += tot
        rows.append({"kernel": kernel, "dispatches": n, "weight_bytes": b,
                     "bytes_per_dispatch": b // n if n else 0,
                     "roles": list(role_names)})

    missing = sorted(set(by_role) - claimed)
    if missing:
        raise Unreconciled(
            "these catalog tensors are read by NO dispatch in the map, so the "
            f"attribution does not account for the artifact: {missing}")

    dispatches = sum(r["dispatches"] for r in rows)
    active = sum(r["weight_bytes"] for r in rows)
    # R1: everything is accounted for -- what a token reads, plus the part of the
    # embedding table a token does not read, is the whole artifact.
    accounted = active + unmapped_check - (EMBED_HIDDEN // 2 + (EMBED_HIDDEN // 64) * 2)
    if accounted != catalog_total:
        raise Unreconciled(
            f"attribution sums to {accounted:,} against a catalog total of "
            f"{catalog_total:,}; a tensor is missing or counted twice")

    rows.sort(key=lambda r: -r["weight_bytes"])
    matvec = [r for r in rows if r["weight_bytes"] > 0.001 * active]
    state = DELTANET_LAYERS * DELTANET_STATE_BYTES * 2  # read and write
    ceiling_tps = MEASURED_ROOF_GB_S * 1e9 / active
    needed_raw = 50.0 / SEALED_CAPABILITY
    return {
        "artifact": str(root),
        "dispatches_per_token": dispatches,
        "reconciliation": {
            "R1_attribution_equals_catalog": accounted == catalog_total,
            "R2_catalog_equals_disk": catalog_total == disk,
            "R2_delta_bytes": disk - catalog_total,
            "R3_ebpw_from_bytes": 8 * catalog_total / PARENT_PARAMS,
            "R3_sealed_complete_ebpw": SEALED_COMPLETE_EBPW,
        },
        "catalog_total_bytes": catalog_total,
        "active_weight_bytes_per_token": active,
        "active_ebpw_per_token": 8 * active / PARENT_PARAMS,
        "pareto_by_bytes": rows,
        "count_vs_bytes": {
            "weight_reading_dispatches": sum(r["dispatches"] for r in rows if r["weight_bytes"]),
            "weight_free_dispatches": sum(r["dispatches"] for r in rows if not r["weight_bytes"]),
            "top_by_bytes_dispatch_share": sum(r["dispatches"] for r in matvec) / dispatches,
            "top_by_bytes_byte_share": sum(r["weight_bytes"] for r in matvec) / active,
        },
        "bandwidth": {
            "sealed_raw_tps": SEALED_RAW_TPS,
            "effective_gb_s": SEALED_RAW_TPS * active / 1e9,
            "measured_roof_gb_s": MEASURED_ROOF_GB_S,
            "fraction_of_roof": SEALED_RAW_TPS * active / 1e9 / MEASURED_ROOF_GB_S,
            "raw_tps_ceiling_at_roof": ceiling_tps,
            "raw_tps_needed_for_50_accepted": needed_raw,
            "ceiling_reaches_the_target": ceiling_tps >= needed_raw,
        },
        # Every uncounted byte makes the ceiling LOWER, so the conclusion is on
        # the safe side of its own omission. This bounds the largest omission.
        "uncounted_sensitivity": {
            "deltanet_state_bytes_per_token": state,
            "as_fraction_of_weight_bytes": state / active,
            "ceiling_with_state_included": MEASURED_ROOF_GB_S * 1e9 / (active + state),
            "note": "KV cache grows with context and is negligible at the 11-token "
                    "prompt the count atlas used; it is not negligible at long context.",
        },
    }


def assert_reconciles(root: Path = RESIDENT) -> dict:
    r = build(root)
    rec = r["reconciliation"]
    if not rec["R1_attribution_equals_catalog"] or not rec["R2_catalog_equals_disk"]:
        raise Unreconciled("reconciliation failed")
    if abs(rec["R3_ebpw_from_bytes"] - rec["R3_sealed_complete_ebpw"]) > 1e-9:
        raise Unreconciled(
            f"bytes imply EBPW {rec['R3_ebpw_from_bytes']!r}, the seal says "
            f"{rec['R3_sealed_complete_ebpw']!r}")
    return r


if __name__ == "__main__":
    print(json.dumps(assert_reconciles(), indent=1))
