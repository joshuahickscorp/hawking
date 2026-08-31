"""NGRAM_SCHOOL — five-axis candidate generator for Flash's n-gram organ.

Flash's n-gram surface is a representation/execution school, not a storage
problem. This module measures each idea on five conceptual axes (executable
bytes, active lookup bytes, lookup operations, decode cost, capability
sensitivity) and REFUSES to rank on storage alone.

Analytical models only, over the sealed organ shape. No fit to the 360 GB
specimen, no timing, no hardware claim.

    python3 tools/future/ngram_school.py --build
    python3 tools/future/ngram_school.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import write_receipt, load_json, REPO
from tools.future._common import git

RECEIPT = "NGRAM_SCHOOL.json"
SCHEMA = "hawking.future.ngram_school.v1"
VERSION = 1

# ---------------------------------------------------------------------------
# Sealed organ shape. Cited, not re-derived from tensor bodies.
# ---------------------------------------------------------------------------
#
# FLASH_ORGAN_CENSUS.json / FLASH_META_REPRESENTATION_SUB1.json /
# FLASH_COMPLETE_V2.nr.json are NOT in this HEAD. The live shape is the
# ngram_engine entry of receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json
# (organ_graph) plus the matching row in receipts/headless/FLASH_EBPW_BUDGET.json.
#
# Table: 128 shards of BF16 [2500012, 160] = 102_400_491_520 bytes.
# PLE aux (9 tensors): 65_679_640 bytes, of which 280 are I64 metadata.
# Source payload: 102_466_171_160 bytes.
# Active/token (STRUCTURAL_ESTIMATE): aux 65_679_640 + 3 * 320 = 65_680_600.
# The table is ~28.5 % of the 359_999_963_128-byte Flash specimen; the
# per-token DRAM of this organ is the PLE projection path, not the table.

SHARD_ROWS = 2_500_012
SHARD_DIM = 160
SHARD_COUNT = 128
N_ROWS = SHARD_ROWS * SHARD_COUNT  # 320_001_536
ROW_VALUES = SHARD_DIM
DTYPE_BYTES = 2  # BF16
DTYPE_BITS = 16
TABLE_BYTES = N_ROWS * ROW_VALUES * DTYPE_BYTES  # 102_400_491_520
TABLE_VALUES = N_ROWS * ROW_VALUES

PLE_CONV1D_BYTES = 10_240 * 1 * 4 * DTYPE_BYTES  # 81_920
PLE_KEY_PROJ_BYTES = 10_240 * 2_560 * DTYPE_BYTES  # 52_428_800
PLE_VALUE_PROJ_BYTES = 2_560 * 2_560 * DTYPE_BYTES  # 13_107_200
PLE_NORM_BYTES = 3 * 10_240 * DTYPE_BYTES  # 61_440
PLE_BF16_BYTES = (
    PLE_CONV1D_BYTES + PLE_KEY_PROJ_BYTES + PLE_VALUE_PROJ_BYTES + PLE_NORM_BYTES
)  # 65_679_360
PLE_I64_BYTES = 24 + 128 + 128  # layer_multipliers + offsets + vocab_sizes
AUX_BYTES = PLE_BF16_BYTES + PLE_I64_BYTES  # 65_679_640
AUX_BF16_VALUES = PLE_BF16_BYTES // DTYPE_BYTES

SOURCE_BYTES = TABLE_BYTES + AUX_BYTES  # 102_466_171_160
LOOKUP_ROW_BYTES = ROW_VALUES * DTYPE_BYTES  # 320
NGRAM_SIZE = 3
LOOKUP_ROWS_NOMINAL = NGRAM_SIZE
SOURCE_ACTIVE_BYTES = AUX_BYTES + LOOKUP_ROWS_NOMINAL * LOOKUP_ROW_BYTES  # 65_680_600
SOURCE_FLOPS_PER_TOKEN = 65_617_920  # cited STRUCTURAL_ESTIMATE; not a measurement
SOURCE_TENSOR_COUNT = 137
SOURCE_ALLOCATION_FRACTION = 0.2846282823744834
SPECIMEN_PAYLOAD_BYTES = 359_999_963_128
SOURCE_PARAMETER_COUNT = 179_999_981_564
NGRAM_VOCAB_SIZE_BASE = 20_000_000
NGRAM_HEADS = 16  # ngram_heads_vocab_sizes shape; bodies not loaded
SPLIT_NGRAM_PARTS = 128
HIDDEN_SIZE = 2560
VOCAB_SIZE = 248_320
GROUP = 64
Q4_CODE_BITS = 4
Q3_CODE_BITS = 3
SCALE_BYTES_PER_GROUP = 2  # fp16 scale; Flash independent_q4_g64 = 4.25 bpw

# Family-budget proxy. The named FLASH_META_REPRESENTATION_SUB1.json receipt
# is absent from HEAD. FLASH_EBPW_BUDGET.json sets complete_system_ebpw_max
# to 1.0 with target_ceiling_bytes == source_parameter_count, i.e. 1.0
# effective *bytes* per parameter for the whole specimen.
EBPW_MAX_BYTES_PER_PARAM = 1.0
NGRAM_FAIR_SHARE_BYTES = SOURCE_BYTES // DTYPE_BYTES  # 51_233_085_580

AXES = (
    "executable_bytes",
    "active_lookup_bytes_per_token",
    "lookup_operations_per_token",
    "decode_cost_class",
    "capability_sensitivity_class",
)

# Lower is cheaper. Q3 dequant is strictly above Q4 so a smaller packing
# cannot dominate the Q4 control on the decode axis by construction.
DECODE_COST_ORDER: dict[str, int] = {
    "IDENTITY_GATHER": 0,
    "GATHER_DEQUANT_Q4": 1,
    "GATHER_DEQUANT_Q3": 2,
    "ISLAND_BYPASS_THEN_DEQUANT": 3,
    "MULTI_GATHER_ADD": 4,
    "RESIDUAL_COMBINE": 5,
    "HIERARCHICAL_WALK": 6,
    "HASH_THEN_GATHER": 7,
    "CONTEXT_SELECT_THEN_GATHER": 8,
    "GENERATE": 9,
}

# Lower is safer. No Flash n-gram null-organ capability measurement exists;
# these are analytical classes, not observed survival scores.
# Islands sit between Q4 and naked Q3: they exist to bandage Q3's capability
# risk, so they must not be ranked as more fragile than the Q3 control or
# the packed Q3 control would dominate them on every axis by construction.
SENSITIVITY_ORDER: dict[str, int] = {
    "CONTROL_Q4": 0,
    "ISLAND_REPAIRABLE": 1,
    "CONTROL_Q3": 2,
    "STRUCTURE_PRESERVING": 3,
    "CODEBOOK_RESIDUAL": 4,
    "CODEBOOK": 5,
    "COLLISION_PRONE": 6,
    "CONTEXT_MISMATCH": 7,
    "GENERATIVE_UNMEASURED": 8,
}

CONTROL_IDS = ("packed_q4_control", "packed_q3_control")


class StorageOnlyRankingError(ValueError):
    """rank() refuses a storage-only or otherwise incomplete-axis ordering."""


class ControlsMissingError(ValueError):
    """rank() requires packed Q4 and Q3 controls in every comparison."""


class IncompleteVectorError(ValueError):
    """A candidate is missing one of the five axes."""


# ---------------------------------------------------------------------------
# Packing arithmetic (group-64, fp16 scale). Matches Flash independent_q4_g64
# 4.25 bits/value and the C4 UNIFORM_Q4_GROUP=64 / 34-byte group.
# ---------------------------------------------------------------------------

def packed_group64_bytes(n_values: int, code_bits: int) -> int:
    if n_values < 0:
        raise ValueError("n_values must be >= 0")
    if n_values == 0:
        return 0
    groups = (n_values + GROUP - 1) // GROUP
    code_bytes = (GROUP * code_bits + 7) // 8
    return groups * (code_bytes + SCALE_BYTES_PER_GROUP)


def q4_bytes(n_values: int) -> int:
    return packed_group64_bytes(n_values, Q4_CODE_BITS)


def q3_bytes(n_values: int) -> int:
    return packed_group64_bytes(n_values, Q3_CODE_BITS)


def q4_row_bytes() -> int:
    return q4_bytes(ROW_VALUES)


def q3_row_bytes() -> int:
    return q3_bytes(ROW_VALUES)


def aux_q4_bytes() -> int:
    return q4_bytes(AUX_BF16_VALUES) + PLE_I64_BYTES


def aux_q3_bytes() -> int:
    return q3_bytes(AUX_BF16_VALUES) + PLE_I64_BYTES


def _isqrt_ceil(n: int) -> int:
    x = int(n**0.5)
    while x * x < n:
        x += 1
    while x > 0 and (x - 1) * (x - 1) >= n:
        x -= 1
    return x


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _require_axes(candidate: Mapping[str, Any]) -> None:
    missing = [a for a in AXES if a not in candidate]
    if missing:
        raise IncompleteVectorError(
            f"{candidate.get('id', '<unnamed>')} missing axes {missing}"
        )
    for a in AXES[:3]:
        v = candidate[a]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise IncompleteVectorError(
                f"{candidate.get('id', '<unnamed>')} axis {a} must be a "
                f"non-negative int, got {v!r}"
            )
    if candidate["decode_cost_class"] not in DECODE_COST_ORDER:
        raise IncompleteVectorError(
            f"{candidate.get('id', '<unnamed>')} unknown decode_cost_class "
            f"{candidate['decode_cost_class']!r}"
        )
    if candidate["capability_sensitivity_class"] not in SENSITIVITY_ORDER:
        raise IncompleteVectorError(
            f"{candidate.get('id', '<unnamed>')} unknown "
            f"capability_sensitivity_class "
            f"{candidate['capability_sensitivity_class']!r}"
        )


def axis_tuple(candidate: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    _require_axes(candidate)
    return (
        int(candidate["executable_bytes"]),
        int(candidate["active_lookup_bytes_per_token"]),
        int(candidate["lookup_operations_per_token"]),
        DECODE_COST_ORDER[str(candidate["decode_cost_class"])],
        SENSITIVITY_ORDER[str(candidate["capability_sensitivity_class"])],
    )


def _cand(
    *,
    ident: str,
    family: str,
    is_control: bool,
    executable_bytes: int,
    active_lookup_bytes_per_token: int,
    lookup_operations_per_token: int,
    decode_cost_class: str,
    capability_sensitivity_class: str,
    assumptions: Sequence[str],
    formula: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "id": ident,
        "family": family,
        "is_control": is_control,
        "executable_bytes": int(executable_bytes),
        "active_lookup_bytes_per_token": int(active_lookup_bytes_per_token),
        "lookup_operations_per_token": int(lookup_operations_per_token),
        "decode_cost_class": decode_cost_class,
        "capability_sensitivity_class": capability_sensitivity_class,
        "assumptions": list(assumptions),
        "formula": dict(formula),
    }
    _require_axes(row)
    return row


def packed_q4_control() -> dict[str, Any]:
    table = q4_bytes(TABLE_VALUES)
    aux = aux_q4_bytes()
    active = aux + LOOKUP_ROWS_NOMINAL * q4_row_bytes()
    return _cand(
        ident="packed_q4_control",
        family="packed_q4_control",
        is_control=True,
        executable_bytes=table + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=LOOKUP_ROWS_NOMINAL,
        decode_cost_class="GATHER_DEQUANT_Q4",
        capability_sensitivity_class="CONTROL_Q4",
        assumptions=(
            "group-64 4-bit codes + fp16 scale (34 bytes / 64 values = 4.25 bpw)",
            "same gather count as the sealed lookup_rows_nominal=3",
            "PLE aux packed the same way as the table (full-organ control)",
            "cited from Flash independent_q4_g64 effective 4.25 bits/value and "
            "C4 UNIFORM_Q4_GROUP=64; not a hardware measurement",
        ),
        formula={
            "table_bytes": table,
            "aux_bytes": aux,
            "row_bytes": q4_row_bytes(),
            "group": GROUP,
            "code_bits": Q4_CODE_BITS,
            "bytes_per_group": 34,
        },
    )


def packed_q3_control() -> dict[str, Any]:
    table = q3_bytes(TABLE_VALUES)
    aux = aux_q3_bytes()
    active = aux + LOOKUP_ROWS_NOMINAL * q3_row_bytes()
    return _cand(
        ident="packed_q3_control",
        family="packed_q3_control",
        is_control=True,
        executable_bytes=table + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=LOOKUP_ROWS_NOMINAL,
        decode_cost_class="GATHER_DEQUANT_Q3",
        capability_sensitivity_class="CONTROL_Q3",
        assumptions=(
            "group-64 3-bit codes + fp16 scale (26 bytes / 64 values = 3.25 bpw)",
            "same gather count as packed Q4; unpack is a strictly higher decode class",
            "capability class CONTROL_Q3 is analytical: no n-gram null-organ test exists",
        ),
        formula={
            "table_bytes": table,
            "aux_bytes": aux,
            "row_bytes": q3_row_bytes(),
            "group": GROUP,
            "code_bits": Q3_CODE_BITS,
            "bytes_per_group": 26,
        },
    )


def product_codebooks() -> dict[str, Any]:
    # Classic PQ: 16 subspaces of 10 dims, 8-bit codes, BF16 centroids.
    m_sub = 16
    d_sub = ROW_VALUES // m_sub  # 10
    n_codes = 256
    code_bytes = N_ROWS * m_sub  # 1 byte per subspace
    codebook_bytes = m_sub * n_codes * d_sub * DTYPE_BYTES
    aux = aux_q4_bytes()
    # Do not assume the codebook is cache-resident (that would be a hardware claim).
    per_row_active = m_sub * 1 + m_sub * d_sub * DTYPE_BYTES
    active = aux + LOOKUP_ROWS_NOMINAL * per_row_active
    lookups = LOOKUP_ROWS_NOMINAL * m_sub
    return _cand(
        ident="product_codebooks",
        family="product_codebooks",
        is_control=False,
        executable_bytes=code_bytes + codebook_bytes + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="MULTI_GATHER_ADD",
        capability_sensitivity_class="CODEBOOK",
        assumptions=(
            "M=16 subspaces x 10 dims, 256-entry BF16 codebooks, 8-bit codes",
            "PLE aux remains packed Q4 (this family does not reinvent the projection)",
            "centroid fetches counted as active DRAM; cache residency is UNKNOWN",
            "lookup_ops = ngram_size * M (one gather per subspace per n-gram row)",
        ),
        formula={
            "M": m_sub,
            "d_sub": d_sub,
            "n_codes": n_codes,
            "code_bytes": code_bytes,
            "codebook_bytes": codebook_bytes,
            "aux_bytes": aux,
            "per_row_active_bytes": per_row_active,
        },
    )


def residual_product_quantization() -> dict[str, Any]:
    # Two-stage PQ: 8-dim-subspace coarse + 8-dim-subspace residual, 8-bit each.
    m_stage = 8
    d_sub = ROW_VALUES // m_stage  # 20
    n_codes = 256
    stages = 2
    code_bytes = N_ROWS * m_stage * stages
    codebook_bytes = stages * m_stage * n_codes * d_sub * DTYPE_BYTES
    aux = aux_q4_bytes()
    per_row_active = stages * (m_stage * 1 + m_stage * d_sub * DTYPE_BYTES)
    active = aux + LOOKUP_ROWS_NOMINAL * per_row_active
    lookups = LOOKUP_ROWS_NOMINAL * m_stage * stages
    return _cand(
        ident="residual_product_quantization",
        family="residual_product_quantization",
        is_control=False,
        executable_bytes=code_bytes + codebook_bytes + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="RESIDUAL_COMBINE",
        capability_sensitivity_class="CODEBOOK_RESIDUAL",
        assumptions=(
            "coarse PQ M=8 plus residual PQ M=8, 8-bit codes, BF16 centroids",
            "residual is hypothesized to reconstruct better than one-shot PQ "
            "at equal code bytes; that is a class, not a measured error",
            "PLE aux remains packed Q4",
            "lookup_ops = ngram_size * 8 * 2",
        ),
        formula={
            "M_per_stage": m_stage,
            "stages": stages,
            "d_sub": d_sub,
            "code_bytes": code_bytes,
            "codebook_bytes": codebook_bytes,
            "aux_bytes": aux,
        },
    )


def hierarchical_codebooks() -> dict[str, Any]:
    n_coarse = 1024
    m_res = 8
    d_sub = ROW_VALUES // m_res  # 20
    n_codes = 256
    coarse_codebook = n_coarse * ROW_VALUES * DTYPE_BYTES
    residual_codebook = m_res * n_codes * d_sub * DTYPE_BYTES
    # uint16 coarse id + 8 residual code bytes per row
    code_bytes = N_ROWS * (2 + m_res)
    aux = aux_q4_bytes()
    # Coarse codebook is 1024*320 bytes: count one 160-d coarse vector plus residual
    # gathers per n-gram row, not the whole book.
    per_row_active = LOOKUP_ROW_BYTES + m_res * (1 + d_sub * DTYPE_BYTES) + 2
    active = aux + LOOKUP_ROWS_NOMINAL * per_row_active
    lookups = LOOKUP_ROWS_NOMINAL * (1 + m_res)
    return _cand(
        ident="hierarchical_codebooks",
        family="hierarchical_codebooks",
        is_control=False,
        executable_bytes=code_bytes + coarse_codebook + residual_codebook + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="HIERARCHICAL_WALK",
        capability_sensitivity_class="CODEBOOK",
        assumptions=(
            "1024 coarse 160-d centroids, then residual PQ M=8",
            "walk is one coarse gather plus M residual gathers per n-gram row",
            "PLE aux remains packed Q4",
        ),
        formula={
            "n_coarse": n_coarse,
            "M_residual": m_res,
            "code_bytes": code_bytes,
            "coarse_codebook_bytes": coarse_codebook,
            "residual_codebook_bytes": residual_codebook,
            "aux_bytes": aux,
        },
    )


def clustered_dictionaries() -> dict[str, Any]:
    n_clusters = 4096
    dict_size = 256
    codebook = n_clusters * dict_size * ROW_VALUES * DTYPE_BYTES
    # uint16 cluster id + uint8 local code
    assign_bytes = N_ROWS * 3
    aux = aux_q4_bytes()
    per_row_active = 3 + ROW_VALUES * DTYPE_BYTES  # ids + one 160-d centroid
    active = aux + LOOKUP_ROWS_NOMINAL * per_row_active
    lookups = LOOKUP_ROWS_NOMINAL * 2  # cluster then dictionary entry
    return _cand(
        ident="clustered_dictionaries",
        family="clustered_dictionaries",
        is_control=False,
        executable_bytes=codebook + assign_bytes + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="HIERARCHICAL_WALK",
        capability_sensitivity_class="CODEBOOK",
        assumptions=(
            "4096 clusters, 256-entry 160-d dictionary per cluster, BF16 centroids",
            "row stored as (cluster_id, local_code); two gathers per n-gram row",
            "PLE aux remains packed Q4",
        ),
        formula={
            "n_clusters": n_clusters,
            "dict_size": dict_size,
            "codebook_bytes": codebook,
            "assign_bytes": assign_bytes,
            "aux_bytes": aux,
        },
    )


def factorized_lookup() -> dict[str, Any]:
    # Product-of-tables over the declared 20M base vocab, once per 16 heads:
    # 20_000_000 ≈ 4473^2. Two factors of shape [4473, 160] BF16 per head.
    side = _isqrt_ceil(NGRAM_VOCAB_SIZE_BASE)
    per_head = 2 * side * ROW_VALUES * DTYPE_BYTES
    table = NGRAM_HEADS * per_head
    aux = aux_q4_bytes()
    # Two factor-row gathers per head per n-gram position, then add.
    per_position_active = NGRAM_HEADS * 2 * LOOKUP_ROW_BYTES
    active = aux + LOOKUP_ROWS_NOMINAL * per_position_active
    lookups = LOOKUP_ROWS_NOMINAL * NGRAM_HEADS * 2
    return _cand(
        ident="factorized_lookup",
        family="factorized_lookup",
        is_control=False,
        executable_bytes=table + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="MULTI_GATHER_ADD",
        capability_sensitivity_class="STRUCTURE_PRESERVING",
        assumptions=(
            "each of 16 heads is a product of two sqrt(20e6)-sided 160-d tables",
            "head count is the I64 ngram_heads_vocab_sizes length; per-head vocab "
            "bodies were not loaded, so 20e6 is the config base not a measured size",
            "composition is A[i] + B[j] (additive); multiplicative is untested",
            "lookup_ops = ngram_size * 16 heads * 2 factors",
            "PLE aux remains packed Q4",
        ),
        formula={
            "side": side,
            "heads": NGRAM_HEADS,
            "per_head_bytes": per_head,
            "table_bytes": table,
            "aux_bytes": aux,
            "side_squared": side * side,
        },
    )


def context_conditioned_lookup() -> dict[str, Any]:
    n_context = 64
    rows_per_context = 65_536
    table = n_context * rows_per_context * ROW_VALUES * DTYPE_BYTES
    selector = n_context * HIDDEN_SIZE * DTYPE_BYTES  # tiny linear selector hypothesis
    aux = aux_q4_bytes()
    per_row_active = LOOKUP_ROW_BYTES
    # one extra gather for the context id, then 3 row gathers from the selected table
    active = aux + selector + LOOKUP_ROWS_NOMINAL * per_row_active
    lookups = 1 + LOOKUP_ROWS_NOMINAL
    return _cand(
        ident="context_conditioned_lookup",
        family="context_conditioned_lookup",
        is_control=False,
        executable_bytes=table + selector + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="CONTEXT_SELECT_THEN_GATHER",
        capability_sensitivity_class="CONTEXT_MISMATCH",
        assumptions=(
            "64 context bins, each a 2^16 x 160 BF16 table; selector is a 64 x 2560 map",
            "the 3-gram already conditions on local tokens; this adds a longer-window bin",
            "wrong bin is a capability miss, not a recoverable residual",
            "selector counted fully active (tiny); PLE aux remains packed Q4",
        ),
        formula={
            "n_context": n_context,
            "rows_per_context": rows_per_context,
            "table_bytes": table,
            "selector_bytes": selector,
            "aux_bytes": aux,
        },
    )


def generated_lookup() -> dict[str, Any]:
    # Replace the 320M-row table with vocab embeds + a small MLP over 3-grams.
    token_table = VOCAB_SIZE * ROW_VALUES * DTYPE_BYTES
    # concat 3*160 -> 256 -> 160
    mlp = (3 * ROW_VALUES * 256 + 256 * ROW_VALUES) * DTYPE_BYTES
    aux = aux_q4_bytes()
    active = aux + LOOKUP_ROWS_NOMINAL * LOOKUP_ROW_BYTES + mlp
    lookups = LOOKUP_ROWS_NOMINAL  # three token-embed gathers, then generate
    return _cand(
        ident="generated_lookup",
        family="generated_lookup",
        is_control=False,
        executable_bytes=token_table + mlp + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="GENERATE",
        capability_sensitivity_class="GENERATIVE_UNMEASURED",
        assumptions=(
            "no 20M-gram table: generate the 160-d row from 3 token embeds + MLP",
            "MLP 480->256->160 BF16, fully active every token (it is small)",
            "lookup_ops stays at 3 (vocab gathers); decode class is GENERATE",
            "capability is UNMEASURED: no Flash n-gram null-organ survival exists",
            "PLE aux remains packed Q4",
        ),
        formula={
            "token_table_bytes": token_table,
            "mlp_bytes": mlp,
            "aux_bytes": aux,
        },
    )


def semantic_hashing() -> dict[str, Any]:
    n_hash = 8
    n_buckets = 65_536
    table = n_hash * n_buckets * ROW_VALUES * DTYPE_BYTES
    aux = aux_q4_bytes()
    per_position_active = n_hash * LOOKUP_ROW_BYTES
    active = aux + LOOKUP_ROWS_NOMINAL * per_position_active
    lookups = LOOKUP_ROWS_NOMINAL * n_hash
    return _cand(
        ident="semantic_hashing",
        family="semantic_hashing",
        is_control=False,
        executable_bytes=table + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="HASH_THEN_GATHER",
        capability_sensitivity_class="COLLISION_PRONE",
        assumptions=(
            "8 independent 2^16-bucket 160-d tables; row is the sum of 8 hash gathers",
            "320M source rows into 64k buckets is an extreme collision regime",
            "lookup_ops = ngram_size * 8",
            "PLE aux remains packed Q4",
        ),
        formula={
            "n_hash": n_hash,
            "n_buckets": n_buckets,
            "table_bytes": table,
            "aux_bytes": aux,
        },
    )


def literal_exception_islands() -> dict[str, Any]:
    # Q3 backbone on 99 % of rows, BF16 literals on 1 % heavy-hitters, plus a bitmap.
    island_frac_num, island_frac_den = 1, 100
    n_islands = (N_ROWS * island_frac_num) // island_frac_den
    n_q3 = N_ROWS - n_islands
    q3_table = q3_bytes(n_q3 * ROW_VALUES)
    island_table = n_islands * LOOKUP_ROW_BYTES
    bitmap = (N_ROWS + 7) // 8
    aux = aux_q4_bytes()
    # Worst-case active: miss the island (Q3 row). Bitmap test is 1 bit, ignored.
    active = aux + LOOKUP_ROWS_NOMINAL * q3_row_bytes()
    lookups = LOOKUP_ROWS_NOMINAL
    return _cand(
        ident="literal_exception_islands",
        family="literal_exception_islands",
        is_control=False,
        executable_bytes=q3_table + island_table + bitmap + aux,
        active_lookup_bytes_per_token=active,
        lookup_operations_per_token=lookups,
        decode_cost_class="ISLAND_BYPASS_THEN_DEQUANT",
        capability_sensitivity_class="ISLAND_REPAIRABLE",
        assumptions=(
            "1 % of rows kept as BF16 literals (Zipf heavy-hitters, not fitted)",
            "remaining 99 % packed Q3 group-64; island membership is a bitset",
            "active path uses the Q3 row (miss); a hit would be a BF16 gather",
            "islands are a capability bandage, not a proof that Q3 is safe",
            "PLE aux remains packed Q4",
        ),
        formula={
            "n_islands": n_islands,
            "n_q3": n_q3,
            "q3_table_bytes": q3_table,
            "island_table_bytes": island_table,
            "bitmap_bytes": bitmap,
            "aux_bytes": aux,
        },
    )


FAMILY_BUILDERS: tuple[tuple[str, Any], ...] = (
    ("packed_q4_control", packed_q4_control),
    ("packed_q3_control", packed_q3_control),
    ("product_codebooks", product_codebooks),
    ("residual_product_quantization", residual_product_quantization),
    ("hierarchical_codebooks", hierarchical_codebooks),
    ("clustered_dictionaries", clustered_dictionaries),
    ("factorized_lookup", factorized_lookup),
    ("context_conditioned_lookup", context_conditioned_lookup),
    ("generated_lookup", generated_lookup),
    ("semantic_hashing", semantic_hashing),
    ("literal_exception_islands", literal_exception_islands),
)


def candidates() -> list[dict[str, Any]]:
    """Every school family, controls first, then the rest in a fixed order."""
    return [builder() for _name, builder in FAMILY_BUILDERS]


def storage_trap(
    control: Mapping[str, Any],
    *,
    byte_factor_num: int = 1,
    byte_factor_den: int = 2,
    lookup_factor: int = 3,
) -> dict[str, Any]:
    """Synthetic: half the executable bytes, triple the lookup ops.

    Not a school family. Used to prove rank() cannot promote a storage-only win.
    """
    _require_axes(control)
    trap = dict(control)
    trap["id"] = "storage_trap_half_bytes_triple_lookups"
    trap["family"] = "synthetic_negative_control"
    trap["is_control"] = False
    trap["executable_bytes"] = int(control["executable_bytes"]) * byte_factor_num // byte_factor_den
    trap["lookup_operations_per_token"] = int(control["lookup_operations_per_token"]) * lookup_factor
    trap["assumptions"] = [
        "synthetic negative control: half executable_bytes, triple lookup_operations",
        "other axes copied from the packed Q4 control",
        "not a proposed representation",
    ]
    trap["formula"] = {
        "byte_factor": [byte_factor_num, byte_factor_den],
        "lookup_factor": lookup_factor,
        "copied_from": control["id"],
    }
    _require_axes(trap)
    return trap


# ---------------------------------------------------------------------------
# Dominance, Pareto, rank (storage-only refusal)
# ---------------------------------------------------------------------------

def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True iff A is <= B on every axis and < on at least one (all minimized)."""
    ta, tb = axis_tuple(a), axis_tuple(b)
    if ta == tb:
        return False
    return all(x <= y for x, y in zip(ta, tb)) and any(x < y for x, y in zip(ta, tb))


def _check_full_vector_request(by: str | None, axes: Sequence[str] | None) -> None:
    if by is not None:
        raise StorageOnlyRankingError(
            f"rank() refuses scalar ordering (by={by!r}); the n-gram school "
            "reports a five-axis Pareto front, never a storage-only winner"
        )
    if axes is None:
        return
    if tuple(axes) == AXES:
        return
    if set(axes) == set(AXES) and len(axes) == len(AXES):
        return
    raise StorageOnlyRankingError(
        f"rank() refuses incomplete-axis ordering; requires {AXES}, got {tuple(axes)}"
    )


def _require_controls(cands: Sequence[Mapping[str, Any]]) -> None:
    ids = {c["id"] for c in cands}
    missing = [i for i in CONTROL_IDS if i not in ids]
    if missing:
        raise ControlsMissingError(
            f"rank() requires packed Q4/Q3 controls in every comparison; missing {missing}"
        )


def pareto_front(cands: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(c["id"], dict(c)) for c in cands]
    front: list[dict[str, Any]] = []
    for ident, c in sorted(indexed, key=lambda kv: kv[0]):
        if any(other["id"] != ident and dominates(other, c) for other in cands):
            continue
        front.append(c)
    return front


def rank(
    cands: Sequence[Mapping[str, Any]],
    *,
    by: str | None = None,
    axes: Sequence[str] | None = None,
    require_controls: bool = True,
) -> dict[str, Any]:
    """Pareto-rank the five-axis vectors. Never a scalar winner.

    Raises StorageOnlyRankingError if asked to order by executable_bytes
    (or any other incomplete axis set). Raises ControlsMissingError if the
    packed Q4/Q3 controls are absent. Raises IncompleteVectorError if any
    candidate lacks the five-axis vector.
    """
    _check_full_vector_request(by, axes)
    if require_controls:
        _require_controls(cands)
    for c in cands:
        _require_axes(c)
    front = pareto_front(cands)
    front_ids = [c["id"] for c in front]
    dominated_ids = sorted(c["id"] for c in cands if c["id"] not in front_ids)
    controls_on_front = [i for i in CONTROL_IDS if i in front_ids]
    return {
        "scalar_winner": None,
        "ranking_rule": "pareto_five_axis",
        "storage_only": "REFUSED",
        "axes": list(AXES),
        "pareto_front_ids": front_ids,
        "dominated_ids": dominated_ids,
        "controls_present": list(CONTROL_IDS),
        "controls_on_front": controls_on_front,
        "n_candidates": len(cands),
        "n_front": len(front_ids),
        "note": (
            "A candidate that halves executable_bytes and triples "
            "lookup_operations_per_token is incomparable to packed Q4 and "
            "cannot be a scalar winner."
        ),
    }


def naive_storage_order(cands: Sequence[Mapping[str, Any]]) -> list[str]:
    """The ranking rank() refuses: sort by executable_bytes alone."""
    return [
        c["id"]
        for c in sorted(cands, key=lambda x: (int(x["executable_bytes"]), x["id"]))
    ]


# ---------------------------------------------------------------------------
# Negative-control proof (must actually fire)
# ---------------------------------------------------------------------------

def prove_storage_only_refusal() -> dict[str, Any]:
    q4 = packed_q4_control()
    q3 = packed_q3_control()
    trap = storage_trap(q4)
    raised = False
    raised_axes = False
    try:
        rank([q4, q3, trap], by="executable_bytes")
    except StorageOnlyRankingError:
        raised = True
    try:
        rank([q4, q3, trap], axes=("executable_bytes",))
    except StorageOnlyRankingError:
        raised_axes = True
    naive = naive_storage_order([q4, trap])
    result = rank([q4, q3, trap])
    proof = {
        "storage_only_by_keyword_raises": raised,
        "storage_only_axes_raises": raised_axes,
        "trap_dominates_q4": dominates(trap, q4),
        "q4_dominates_trap": dominates(q4, trap),
        "trap_executable_bytes": trap["executable_bytes"],
        "q4_executable_bytes": q4["executable_bytes"],
        "trap_lookup_operations_per_token": trap["lookup_operations_per_token"],
        "q4_lookup_operations_per_token": q4["lookup_operations_per_token"],
        "naive_storage_winner": naive[0],
        "naive_storage_order": naive,
        "pareto_ids_with_trap": result["pareto_front_ids"],
        "scalar_winner": result["scalar_winner"],
    }
    if not raised:
        raise RuntimeError("negative control did not fire: by='executable_bytes' was accepted")
    if not raised_axes:
        raise RuntimeError("negative control did not fire: axes=('executable_bytes',) was accepted")
    if proof["trap_dominates_q4"]:
        raise RuntimeError("negative control failed: half-bytes/triple-lookups dominates Q4")
    if proof["naive_storage_winner"] != trap["id"]:
        raise RuntimeError("negative control failed: naive storage sort did not put the trap first")
    if result["scalar_winner"] is not None:
        raise RuntimeError("negative control failed: rank() produced a scalar winner")
    if q4["id"] not in result["pareto_front_ids"]:
        raise RuntimeError("negative control failed: packed Q4 dropped off the Pareto front")
    return proof


# ---------------------------------------------------------------------------
# Recovery / citation
# ---------------------------------------------------------------------------

NAMED_ABSENT = (
    "tools/flash_ngram_lookup_oracle.py",
    "tools/flash_doctor_ngram_screen.py",
    "receipts/headless/FLASH_ORGAN_CENSUS.json",
    "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
    "receipts/headless/FLASH_COMPLETE_V2.nr.json",
)


def _present(rel: str) -> bool:
    return (REPO / rel).exists()


def recovered_implementation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in NAMED_ABSENT:
        rows.append(
            {
                "path": rel,
                "present_in_HEAD_worktree": _present(rel),
                "role": "named by the work unit; not in this commit",
                "adequate_as_ngram_school": False,
            }
        )
    found = (
        (
            "hcli/agentos/flash_science.py",
            "organ graph, ngram_engine tensor layouts, PLE-aux vs shard split, "
            "lookup_rows_nominal=ngram_size=3, gravity GENERATE/FACTORIZE/SHARE stages",
            False,
        ),
        (
            "hcli/flash_next.py",
            "20M n-gram vocabulary, 3-gram, 128-way split identity; "
            "ngram_representation is a required complete-system byte field",
            False,
        ),
        (
            "hcli/agentos/flash_executable.py",
            "chosen_representation.ngram = lookup/compositional, not generic dense quant; "
            "native kernel lookup_or_compositional_generator NOT_IMPLEMENTED",
            False,
        ),
        (
            "hcli/agentos/fpga_preboard.py",
            "ngram_lookup_or_generator P1 mapping: HBM lookup/compositional generator "
            "rather than matrix GEMV; compact_ngram_lookup scenario uses source payload bytes",
            False,
        ),
        (
            "receipts/headless/FLASH_EBPW_BUDGET.json",
            "ngram_engine source_bytes=102466171160, active/tok=65680600, "
            "allocation 0.2846, representation CANDIDATE_NOT_BUILT; "
            "ngram_representation accounting WAITING_FOR_REPRESENTATION_AND_LOADER",
            False,
        ),
        (
            "receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json",
            "architecture ngram_size=3, ngram_vocab_size_base=20e6, split_ngram_parts=128; "
            "organ_graph.ngram_engine tensor_layout 128 x [2500012,160] BF16 + 9 PLE tensors; "
            "lookup_row_bytes=320, auxiliary_projection_bytes=65679640",
            False,
        ),
        (
            "receipts/headless/FLASH_TOKEN_NS_BUDGET.json",
            "ngram_engine source_active/flops cited; all actual_* timing fields null",
            False,
        ),
        (
            "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
            "organ ngram_lookup_or_generator priority P1",
            False,
        ),
        (
            "receipts/headless/REPRESENTATION_LIBRARY.json",
            "seven families (q2_affine, q4_control, binary, ternary, shared_basis, "
            "binary_sparse_residual, low_rank_sparse); law 'fewer stored bits != fewer nanoseconds'; "
            "no n-gram family",
            False,
        ),
        (
            "tools/spec/ngram_analysis.py",
            "spec-decode draft-acceptance oracle over token sequences; different school "
            "(user-ngram draft, not Flash PLE table representation)",
            False,
        ),
        (
            "tools/spec/ngram_baseline.sh",
            "Track 6.1 spec-decode n-gram oracle driver",
            False,
        ),
        (
            "crates/hawking-speculate/src/user_ngram.rs",
            "per-user 2-gram/1-gram draft index for spec-decode; lossless by construction; "
            "KB–MB CPU automaton, not the 102 GB Flash ngram_engine",
            False,
        ),
        (
            "tools/headless/c4codebook_design.py",
            "C4 additive codebook / fused ADC design for Qwen3.8 GEMV organs; "
            "group-64 4.25 bpw packing arithmetic reused as the Q4 control formula",
            False,
        ),
        (
            "tools/llama_residual_pq_pack.py",
            "Llama residual-PQ packer for FFN/attn weights; different organ, cited as PQ prior art",
            False,
        ),
        (
            "tools/gravity_exception_selection.py",
            "functional vs magnitude vs residual exception ranking; cited as island prior art",
            False,
        ),
        (
            "receipts/headless/NOETIC_ORGAN_CENSUS.json",
            "Qwen3.8-27B hybrid organs (embed/gqa/deltanet/mlp/output); no ngram_engine",
            False,
        ),
        (
            "tools/future/_common.py",
            "write_receipt / bench_block / HardwareClaimError; used, not reimplemented",
            True,
        ),
    )
    for path, role, adequate in found:
        rows.append(
            {
                "path": path,
                "present_in_HEAD_worktree": _present(path),
                "role": role,
                "adequate_as_ngram_school": adequate,
            }
        )
    return rows


def _try_cite_ebpw() -> dict[str, Any]:
    rel = "receipts/headless/FLASH_EBPW_BUDGET.json"
    path = REPO / rel
    if not path.exists():
        return {
            "path": rel,
            "materialized": False,
            "used": "hardcoded_cited_constants",
            "reason": "sparse checkout / file not on disk; constants taken from git show",
        }
    doc = load_json(path)
    organs = doc.get("organs") or []
    ngram = next((o for o in organs if o.get("organ") == "ngram_engine"), None)
    mismatches: list[str] = []
    if ngram is None:
        mismatches.append("ngram_engine organ missing")
    else:
        if ngram.get("source_bytes") != SOURCE_BYTES:
            mismatches.append(
                f"source_bytes {ngram.get('source_bytes')} != {SOURCE_BYTES}"
            )
        if ngram.get("source_active_bytes_per_token") != SOURCE_ACTIVE_BYTES:
            mismatches.append(
                "source_active_bytes_per_token "
                f"{ngram.get('source_active_bytes_per_token')} != {SOURCE_ACTIVE_BYTES}"
            )
    return {
        "path": rel,
        "materialized": True,
        "used": "hardcoded_cited_constants_cross_checked",
        "mismatches": mismatches,
    }


def organ_shape() -> dict[str, Any]:
    return {
        "organ": "ngram_engine",
        "citations": [
            "receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json organ_graph.ngram_engine",
            "receipts/headless/FLASH_EBPW_BUDGET.json organs[ngram_engine]",
            "hcli/flash_next.py architecture ngram_vocab_size/ngram_size/split_ngram_parts",
        ],
        "source_payload_bytes": SOURCE_BYTES,
        "table_bytes": TABLE_BYTES,
        "aux_bytes": AUX_BYTES,
        "aux_bf16_bytes": PLE_BF16_BYTES,
        "aux_i64_bytes": PLE_I64_BYTES,
        "source_active_bytes_per_token": SOURCE_ACTIVE_BYTES,
        "source_flops_per_token_structural": SOURCE_FLOPS_PER_TOKEN,
        "source_flops_are_not_a_measurement": True,
        "source_tensor_count": SOURCE_TENSOR_COUNT,
        "source_allocation_fraction": SOURCE_ALLOCATION_FRACTION,
        "specimen_payload_bytes": SPECIMEN_PAYLOAD_BYTES,
        "shard_shape": [SHARD_ROWS, SHARD_DIM],
        "shard_count": SHARD_COUNT,
        "n_rows": N_ROWS,
        "row_values": ROW_VALUES,
        "lookup_row_bytes": LOOKUP_ROW_BYTES,
        "lookup_rows_nominal": LOOKUP_ROWS_NOMINAL,
        "ngram_size": NGRAM_SIZE,
        "ngram_vocab_size_base": NGRAM_VOCAB_SIZE_BASE,
        "ngram_heads_from_tensor_shape": NGRAM_HEADS,
        "split_ngram_parts": SPLIT_NGRAM_PARTS,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "dtype": "BF16",
        "active_is_ple_aux_dominated": True,
        "table_bytes_per_token_source": LOOKUP_ROWS_NOMINAL * LOOKUP_ROW_BYTES,
        "aux_bytes_per_token_source": AUX_BYTES,
        "bottleneck_cited": "lookup latency, dictionary locality, and representation expansion",
        "execution_regularity_cited": "irregular bounded lookup plus small projection path",
        "cross_check": _try_cite_ebpw(),
    }


def family_budget() -> dict[str, Any]:
    return {
        "named_receipt": "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
        "named_receipt_present": _present("receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"),
        "proxy_receipt": "receipts/headless/FLASH_EBPW_BUDGET.json",
        "complete_system_ebpw_max": EBPW_MAX_BYTES_PER_PARAM,
        "ebpw_unit": (
            "effective bytes per parameter: target_ceiling_bytes equals "
            "source_parameter_count in FLASH_EBPW_BUDGET.json"
        ),
        "source_parameter_count": SOURCE_PARAMETER_COUNT,
        "target_ceiling_bytes_complete_system": SOURCE_PARAMETER_COUNT,
        "ngram_source_allocation_fraction": SOURCE_ALLOCATION_FRACTION,
        "ngram_fair_share_bytes": NGRAM_FAIR_SHARE_BYTES,
        "ngram_representation_actual_bytes": None,
        "ngram_representation_status": "WAITING_FOR_REPRESENTATION_AND_LOADER",
        "note": (
            "family_budget is a proxy reconstructed from FLASH_EBPW_BUDGET "
            "because FLASH_META_REPRESENTATION_SUB1.json is not in HEAD"
        ),
    }


def negative_findings() -> list[dict[str, str]]:
    return [
        {
            "wanted": "tools/flash_ngram_lookup_oracle.py",
            "found": "absent from HEAD (git show fatal)",
        },
        {
            "wanted": "tools/flash_doctor_ngram_screen.py",
            "found": "absent from HEAD (git show fatal)",
        },
        {
            "wanted": "receipts/headless/FLASH_ORGAN_CENSUS.json",
            "found": "absent; used HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json organ_graph.ngram_engine",
        },
        {
            "wanted": "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json family_budget",
            "found": "absent; proxied FLASH_EBPW_BUDGET.json complete_system_ebpw_max=1.0",
        },
        {
            "wanted": "receipts/headless/FLASH_COMPLETE_V2.nr.json representation",
            "found": "absent; FLASH_NEXT_NOETIC_EXECUTABLE.json is SCAFFOLD_ONLY with ngram kernel NOT_IMPLEMENTED",
        },
        {
            "wanted": "ngram_heads_vocab_sizes tensor body",
            "found": "I64[16] header only; per-head vocab sizes unknown without loading weights",
        },
        {
            "wanted": "Flash n-gram null-organ capability survival",
            "found": "NOETIC_ORGAN_CENSUS is the 27B hybrid, not Flash ngram_engine; sensitivity classes are analytical",
        },
        {
            "wanted": "existing five-axis n-gram candidate school",
            "found": "none; representation_library has no n-gram family; gravity plans say FACTORIZE/GENERATE only",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "candidate generator covering packed Q4/Q3 controls plus nine n-gram-school families",
        "five-axis analytical measure over the sealed ngram_engine shape, assumptions recorded",
        "rank() refuses storage-only ordering and reports a Pareto front with no scalar winner",
        "controls are first-class: rank() raises if packed Q4 or Q3 is missing",
        "negative control that actually fires: half-bytes/triple-lookups does not dominate Q4",
        "active-byte observation: PLE aux dominates per-token DRAM, so table-storage ranking is a trap",
    ]


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def build() -> Path:
    school = candidates()
    proof = prove_storage_only_refusal()
    ranked = rank(school)
    q4 = next(c for c in school if c["id"] == "packed_q4_control")
    q3 = next(c for c in school if c["id"] == "packed_q3_control")
    beats_q4_on_bytes = [
        c["id"]
        for c in school
        if (not c["is_control"]) and c["executable_bytes"] < q4["executable_bytes"]
    ]
    still_not_dominating = [
        c["id"]
        for c in school
        if (not c["is_control"]) and not dominates(c, q4)
    ]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Five-axis n-gram representation/execution school. Storage-only "
            "ranking is refused. Analytical models over the sealed Flash "
            "ngram_engine organ; STATIC_ONLY, no hardware measurement."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "organ_shape": organ_shape(),
        "family_budget": family_budget(),
        "axes": list(AXES),
        "axis_direction": {a: "minimize" for a in AXES},
        "decode_cost_classes": [
            {"name": k, "order": v} for k, v in sorted(DECODE_COST_ORDER.items(), key=lambda kv: kv[1])
        ],
        "capability_sensitivity_classes": [
            {"name": k, "order": v} for k, v in sorted(SENSITIVITY_ORDER.items(), key=lambda kv: kv[1])
        ],
        "packing": {
            "group": GROUP,
            "q4_code_bits": Q4_CODE_BITS,
            "q3_code_bits": Q3_CODE_BITS,
            "scale_bytes_per_group": SCALE_BYTES_PER_GROUP,
            "q4_bytes_per_group": 34,
            "q3_bytes_per_group": 26,
            "q4_bpw": "4.25",
            "q3_bpw": "3.25",
        },
        "controls": list(CONTROL_IDS),
        "q4_executable_bytes": q4["executable_bytes"],
        "q3_executable_bytes": q3["executable_bytes"],
        "candidates": school,
        "rank": ranked,
        "families_smaller_than_q4": beats_q4_on_bytes,
        "families_that_do_not_dominate_q4": still_not_dominating,
        "storage_only_policy": "REFUSED",
        "negative_control": proof,
        "analytical_only": True,
        "no_specimen_fit": True,
        "no_timing": True,
        "evidence_class": "STATIC_ONLY",
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "integration": {
            "module": "tools.future.ngram_school",
            "candidates": "candidates() -> list[dict]",
            "rank": "rank(cands, *, by=None, axes=None, require_controls=True) -> dict",
            "dominates": "dominates(a, b) -> bool",
            "storage_trap": "storage_trap(packed_q4_control()) -> dict",
            "build": "build() -> Path",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/ngram_school.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = selftest() if a.selftest else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
