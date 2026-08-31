"""REPRESENTATION DECODE FUSION — why 258 of 628 is a family, not an unpack pass.

Of 628 sealed dispatches per decoded token, the motif census attributes 258
to representation_decode (41%). Those launches are packed GEMVs and one
embedding row lookup. Production Metal already turns codes into a contribution
to y in-register and never writes a dense W. Dispatch count is independently
refuted as the ~350 GB/s cause. This sidecar is therefore not a "fuse to
shrink the integer" plan.

It asks a physical question: where is a representation still materialized
before it is consumed, and can decode+consume become one operation?

    unpack + matvec
    scale + matvec
    codebook lookup + accumulate
    sparse residual decode + consume

A candidate that decodes to a full f16 tensor and then runs the ordinary GEMV
is REJECTED_DENSE_REMAT. Rank is bytes of intermediate W traffic eliminated
versus a split decode, not dispatches removed.

    python3 tools/future/representation_decode_fusion.py --record
    python3 -m pytest tools/future/test_representation_decode_fusion.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, RECEIPTS, git, load_json, write_receipt
from tools.future.dispatch_motifs import (
    ESTABLISHED_SEALED,
    ESTABLISHED_UNFUSED,
    cluster_launches,
    family_counts,
    walk_launches,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES
from tools.future.tps_budget import DECODE_SRC, Fusion, load_geometry


RECEIPT = "REPRESENTATION_DECODE_FUSION.json"
SCHEMA = "hawking.future.representation_decode_fusion.v1"
VERSION = 1
RECORDED_BY = "tools/future/representation_decode_fusion.py"
EVIDENCE_CLASS = "STATIC_ONLY"

DISPATCH_MOTIFS_REL = "receipts/future/DISPATCH_MOTIFS.json"
ORGAN_BANDWIDTH_REL = "receipts/future/ORGAN_BANDWIDTH.json"
MLP_AUX_REL = "receipts/future/MLP_AUXILIARY_INFORMATION.json"
KERNEL_GEOMETRY_REL = "receipts/future/KERNEL_GEOMETRY.json"
MLP_CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"

AFFINE2_SHADER = "crates/hawking-core/shaders/affine2_group32_matvec.metal"
Q80_SHADER = "crates/hawking-core/shaders/q80_mixed_decode.metal"
Q4_SHADER = "crates/hawking-core/shaders/qwen_uniform_q4.metal"
PQ_SHADER = "crates/hawking-core/shaders/gravity_pq.metal"

ESTABLISHED_REPRESENTATION_DECODE_SEALED = 258
ESTABLISHED_REPRESENTATION_DECODE_UNFUSED = 402
ESTABLISHED_SHARE = ESTABLISHED_REPRESENTATION_DECODE_SEALED / ESTABLISHED_SEALED

# Sealed-3.14 geometry. Same constants as qwen38_geometry / KERNEL_GEOMETRY.
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
LAYERS = 64
DN_LAYERS = 48
GQA_LAYERS = 16
QKVZ_ROWS = 16384
BA_ROWS = 96
O_PROJ_COLS = 6144
Q_PROJ_ROWS = 12288
KV_PROJ_ROWS = 1024
HGRAVS_RANK = 160
HGRAVS_X_CAP = 512  # q80_hgravs01_two_stage_matvec kXCap
F16_BYTES = 2
F32_BYTES = 4
GROUP = 64

# MLP affine-Q2, recorded by mlp_auxiliary_information / mlp_byte_census.
MLP_PARAMS = LAYERS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE)
MLP_ACTIVE = 5_347_795_776
MLP_CODE_BYTES = 4_278_190_080
MLP_AUX_BYTES = 1_069_605_696
MLP_SCALE_BYTES = 534_773_760
MLP_BIAS_BYTES = 534_773_760
LM_HEAD_ACTIVE = 675_430_440
ATTENTION_ACTIVE = 3_852_952_064  # DeltaNet + GQA catalog bytes
AFFINE_TENSORS = 192  # 64 layers × gate/up/down
Q4_TENSORS_QUOTED = 209

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FUSED = "ALREADY_FUSED"
OPEN = "OPEN"
NOT_THIS_ARTIFACT = "NOT_THIS_ARTIFACT"
LOAD_TIME_ONLY = "LOAD_TIME_ONLY"

UNPACK_PLUS_MATVEC = "unpack_plus_matvec"
SCALE_PLUS_MATVEC = "scale_plus_matvec"
CODEBOOK_LOOKUP_PLUS_ACCUMULATE = "codebook_lookup_plus_accumulate"
SPARSE_RESIDUAL_DECODE_PLUS_CONSUME = "sparse_residual_decode_plus_consume"

OPERATIONS: tuple[str, ...] = (
    UNPACK_PLUS_MATVEC,
    SCALE_PLUS_MATVEC,
    CODEBOOK_LOOKUP_PLUS_ACCUMULATE,
    SPARSE_RESIDUAL_DECODE_PLUS_CONSUME,
)

STATUSES = {
    ALREADY_FUSED,
    OPEN,
    REJECTED_DENSE_REMAT,
    NOT_THIS_ARTIFACT,
    LOAD_TIME_ONLY,
}

# Sealed family partition. Sum is the load-bearing 258.
SEALED_REPRESENTATION_DECODE_MOTIFS: tuple[tuple[str, int, str], ...] = (
    ("embed_lookup", 1, "packed row lookup, not a GEMV"),
    ("dn_inproj_pair_concat", 48, "fused qkvz+ba packed GEMV"),
    ("dn_out_proj", 48, "packed GEMV"),
    ("gqa_fused_qkv", 16, "fused QKV packed GEMV"),
    ("gqa_o_proj", 16, "packed GEMV"),
    ("mlp_fused_gate_up_swiglu", 64, "affine-Q2 gate+up+SwiGLU"),
    ("mlp_down_proj", 64, "affine-Q2 GEMV"),
    ("lm_head", 1, "Q4 GEMV"),
)

REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "affine_q2_unpack_plus_matvec",
    "q4_unpack_plus_matvec",
    "affine_q2_scale_plus_matvec",
    "q4_scale_plus_matvec",
    "codebook_lookup_plus_accumulate",
    "sparse_residual_decode_plus_consume",
    "hgravs_two_stage_mid",
    "rice_index_expansion_at_upload",
    "unpack_to_dense_f16_then_gemv",
)

METAL_BLOCKER_KEYS: tuple[str, ...] = (
    "argument_buffers",
    "dynamic_shapes",
    "icb",
    "resource_residency",
    "routing_dependence",
    "threadgroup_capacity",
)

HELPER_MARKERS: dict[str, tuple[str, str]] = {
    "affine2_never_writes_dense_w": (
        AFFINE2_SHADER,
        "never writes a dense W",
    ),
    "affine2_codes_stay_packed": (
        AFFINE2_SHADER,
        "Codes stay packed",
    ),
    "q80_packed_bytes_read_directly": (
        Q80_SHADER,
        "packed bytes are read directly",
    ),
    "q80_must_never_write_dense": (
        Q80_SHADER,
        "write a dense (rows × cols) weight reconstruction",
    ),
    "q80_two_stage_kernel": (
        Q80_SHADER,
        "kernel void q80_hgravs01_two_stage_matvec(",
    ),
    "q80_x_cap_512": (
        Q80_SHADER,
        "constexpr uint kXCap = 512u",
    ),
    "q80_rank_cap_160": (
        Q80_SHADER,
        "constexpr uint kRankCap = 160u",
    ),
    "gravity_pq_no_dense_row": (
        PQ_SHADER,
        "never materializes a dense weight",
    ),
    "gravity_pq_no_temporary_residual": (
        PQ_SHADER,
        "never expands a dense row or a temporary residual tensor",
    ),
    "decode_dense_w_counter": (
        DECODE_SRC,
        "pub dense_w_materialized:",
    ),
    "decode_dispatch_hgravs": (
        DECODE_SRC,
        "fn dispatch_hgravs(",
    ),
    "decode_hgravs_mid": (
        DECODE_SRC,
        "hgravs_mid",
    ),
    "decode_expand_rice": (
        DECODE_SRC,
        "expand_rice_indices",
    ),
    "decode_sparse_csr_split": (
        DECODE_SRC,
        "q80_sparse_q1_apply_csr",
    ),
    "decode_two_stage_unbound": (
        DECODE_SRC,
        "q80_hgravs01_two_stage_matvec",
    ),
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. The 258 is re-walked "
    "from encode/dispatch call sites (tools/future/dispatch_motifs.py) and "
    f"must equal {DISPATCH_MOTIFS_REL} families.sealed.representation_decode "
    "and the established 258, or this module refuses to emit. Intermediate "
    "byte counts are geometry × dtype of buffers named in "
    f"{DECODE_SRC} and the affine-Q2 / Q4 / HGRAVS shaders, not measured "
    "traffic. gpu_authority is false. evidence_class is STATIC_ONLY."
)


class DecodeFusionRefuse(ValueError):
    """Census or packing does not reconcile; the module will not emit."""


class UnreconciledRepresentationDecode(DecodeFusionRefuse):
    """The 258 attributed dispatches do not match the motif census."""

    def __init__(self, got: int, want: int = ESTABLISHED_REPRESENTATION_DECODE_SEALED, *, detail: str = "") -> None:
        self.got = int(got)
        self.want = int(want)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: representation_decode dispatches {got} != established "
            f"{want}{extra}"
        )


# Import-time pin so a drifted partition cannot silently ship.
if sum(n for _id, n, _why in SEALED_REPRESENTATION_DECODE_MOTIFS) != ESTABLISHED_REPRESENTATION_DECODE_SEALED:
    raise DecodeFusionRefuse(
        "SEALED_REPRESENTATION_DECODE_MOTIFS does not sum to 258"
    )
if MLP_PARAMS != 17_112_760_320:
    raise DecodeFusionRefuse(f"MLP_PARAMS drifted: {MLP_PARAMS}")


def _read_rel(rel: str) -> tuple[str | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace"), "disk"
        except OSError:
            pass
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git:HEAD"
    return None, "missing"


def _load_receipt(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if path.is_file():
        doc = load_json(path)
        if isinstance(doc, dict):
            return doc
    text, via = _read_rel(rel)
    if not text:
        raise DecodeFusionRefuse(f"REFUSED: missing {rel} (via={via})")
    import json

    try:
        doc = json.loads(text)
    except Exception as exc:
        raise DecodeFusionRefuse(f"REFUSED: unparseable {rel}: {exc}") from exc
    if not isinstance(doc, dict):
        raise DecodeFusionRefuse(f"REFUSED: {rel} is not an object")
    return doc


def f16_bytes(n_params: int) -> int:
    return int(n_params) * F16_BYTES


def f32_bytes(n: int) -> int:
    return int(n) * F32_BYTES


def mlp_params() -> int:
    return MLP_PARAMS


def lm_head_params() -> int:
    return VOCAB * HIDDEN


def attention_gemv_params() -> int:
    """Exact GEMV parameter counts from KERNEL_GEOMETRY body constants."""
    dn_qkvz = DN_LAYERS * QKVZ_ROWS * HIDDEN
    dn_ba = DN_LAYERS * BA_ROWS * HIDDEN
    dn_out = DN_LAYERS * HIDDEN * O_PROJ_COLS
    gqa_q = GQA_LAYERS * Q_PROJ_ROWS * HIDDEN
    gqa_k = GQA_LAYERS * KV_PROJ_ROWS * HIDDEN
    gqa_v = GQA_LAYERS * KV_PROJ_ROWS * HIDDEN
    gqa_o = GQA_LAYERS * HIDDEN * O_PROJ_COLS
    return dn_qkvz + dn_ba + dn_out + gqa_q + gqa_k + gqa_v + gqa_o


def dense_f16_w_bytes() -> dict[str, int]:
    """Bytes a split decode would write if it rematerialized f16 W.

    Embed is a row lookup (one hidden vector), not a full-table remat on
    the token. The table is omitted from per-token intermediate traffic.
    """
    mlp = f16_bytes(mlp_params())
    attn = f16_bytes(attention_gemv_params())
    head = f16_bytes(lm_head_params())
    return {
        "mlp": mlp,
        "attention_gemv": attn,
        "lm_head": head,
        "token_gemv": mlp + attn + head,
    }


def activation_intermediates() -> dict[str, Any]:
    """GEMV *outputs* later kernels read. Not decoded weights.

    Recorded so act / qkvz / logits are not laundered as representation
    decode. Fusing those is a producer-consumer region question
    (DISPATCH_MOTIFS), not unpack+matvec.
    """
    act = f32_bytes(INTERMEDIATE)
    hidden = f32_bytes(HIDDEN)
    qkvz = f32_bytes(QKVZ_ROWS)
    ba = f32_bytes(BA_ROWS)
    logits = f32_bytes(VOCAB)
    hgravs_mid = f32_bytes(HGRAVS_RANK)
    xsum64 = f32_bytes(INTERMEDIATE // GROUP)
    return {
        "are_decoded_weights": False,
        "reading": (
            "These are consume *results* of the 258 packed GEMVs, re-read by "
            "the next kernel. They are not a materialized W. Ranking W-decode "
            "fusion by these bytes would confuse two different physical objects."
        ),
        "rows": [
            {
                "id": "mlp_act",
                "element": "f32[intermediate]",
                "bytes_per_launch": act,
                "sealed_launches": LAYERS,
                "write_plus_reread_per_token": act * LAYERS * 2,
                "producer": "mlp_fused_gate_up_swiglu",
                "consumer": "mlp_down_proj",
            },
            {
                "id": "dn_qkvz",
                "element": "f32[qkvz_rows]",
                "bytes_per_launch": qkvz,
                "sealed_launches": DN_LAYERS,
                "write_plus_reread_per_token": qkvz * DN_LAYERS * 2,
                "producer": "dn_inproj_pair_concat",
                "consumer": "dn_rearrange_conv",
            },
            {
                "id": "dn_ba",
                "element": "f32[ba_rows]",
                "bytes_per_launch": ba,
                "sealed_launches": DN_LAYERS,
                "write_plus_reread_per_token": ba * DN_LAYERS * 2,
                "producer": "dn_inproj_pair_concat",
                "consumer": "dn_ba_to_decay",
            },
            {
                "id": "lm_head_logits",
                "element": "f32[vocab]",
                "bytes_per_launch": logits,
                "sealed_launches": 1,
                "write_plus_reread_per_token": logits * 2,
                "producer": "lm_head",
                "consumer": "argmax",
            },
            {
                "id": "embed_hidden_row",
                "element": "f32[hidden]",
                "bytes_per_launch": hidden,
                "sealed_launches": 1,
                "write_plus_reread_per_token": hidden * 2,
                "producer": "embed_lookup",
                "consumer": "mixer_input_rmsnorm",
            },
            {
                "id": "hgravs_mid",
                "element": "f32[rank160]",
                "bytes_per_launch": hgravs_mid,
                "sealed_launches": 0,
                "write_plus_reread_per_token": 0,
                "producer": "dispatch_hgravs first factor",
                "consumer": "dispatch_hgravs second factor",
                "note": "workspace exists; sealed-3.14 GEMV census is affine+q4, so this buffer is not on the 258.",
            },
            {
                "id": "xsum64_biasprep",
                "element": "f32[intermediate/64]",
                "bytes_per_launch": xsum64,
                "sealed_launches": 0,
                "write_plus_reread_per_token": 0,
                "producer": "RMSNorm BiasPrep opt-in",
                "consumer": "affine2 BiasPrep gate_up_swiglu",
                "note": "HAWKING_AFFINE2_GEO=biasprep only; not the sealed tpr64/splitk graph.",
            },
        ],
    }


def _empty_blockers(**kwargs: str) -> dict[str, str]:
    out = {k: "n/a" for k in METAL_BLOCKER_KEYS}
    out.update(kwargs)
    missing = [k for k in METAL_BLOCKER_KEYS if k not in out]
    if missing:
        raise DecodeFusionRefuse(f"metal_blockers missing {missing}")
    return {k: out[k] for k in METAL_BLOCKER_KEYS}


def _candidate(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("evidence_class", EVIDENCE_CLASS)
    kwargs.setdefault("gpu_authority", False)
    kwargs.setdefault("not_a_dispatch_count_plan", True)
    required = (
        "id",
        "name",
        "operation",
        "status",
        "mechanism",
        "what_is_materialized",
        "intermediate_bytes_written_and_reread",
        "bytes_eliminated_vs_split_decode",
        "bytes_eliminated_if_true",
        "physical_primitive",
        "dense_rematerialization",
        "dense_rematerialization_reason",
        "metal_blockers",
        "cheapest_falsifier",
        "sealed_launches",
        "on_sealed_258",
        "evidence_class",
        "gpu_authority",
    )
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise DecodeFusionRefuse(f"candidate {kwargs.get('id')!r} missing {missing}")
    if kwargs["operation"] not in OPERATIONS:
        raise DecodeFusionRefuse(f"unknown operation {kwargs['operation']!r}")
    if kwargs["status"] not in STATUSES:
        raise DecodeFusionRefuse(f"unknown status {kwargs['status']!r}")
    if kwargs["dense_rematerialization"] not in {
        DIRECT_CONSUME,
        REJECTED_DENSE_REMAT,
        DEPENDS_ON_LOWERING,
    }:
        raise DecodeFusionRefuse(
            f"{kwargs['id']}: bad dense_rematerialization "
            f"{kwargs['dense_rematerialization']!r}"
        )
    if kwargs["status"] != REJECTED_DENSE_REMAT:
        if kwargs["physical_primitive"] not in ATLAS_PRIMITIVES:
            raise DecodeFusionRefuse(
                f"{kwargs['id']}: primitive {kwargs['physical_primitive']!r} "
                "is not an atlas primitive"
            )
    blockers = kwargs["metal_blockers"]
    if set(blockers) != set(METAL_BLOCKER_KEYS):
        raise DecodeFusionRefuse(
            f"{kwargs['id']}: metal_blockers {sorted(blockers)} != "
            f"{list(METAL_BLOCKER_KEYS)}"
        )
    kwargs.setdefault(
        "blocked_today",
        kwargs["status"] not in {ALREADY_FUSED, NOT_THIS_ARTIFACT, LOAD_TIME_ONLY},
    )
    kwargs["evidence_class"] = EVIDENCE_CLASS
    kwargs["gpu_authority"] = False
    return kwargs


def helper_markers(sources: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Source contracts that make fused in-register decode the production path."""
    present: dict[str, bool] = {}
    missing: list[str] = []
    via: dict[str, str] = {}
    for key, (rel, needle) in HELPER_MARKERS.items():
        if sources is not None and rel in sources:
            text, origin = sources[rel], "injected"
        else:
            text, origin = _read_rel(rel)
        via[rel] = origin
        ok = bool(text) and needle in (text or "")
        # two_stage kernel name is required in the SHADER and required ABSENT
        # from the decode bind path (production still splits HGRAVS).
        if key == "decode_two_stage_unbound":
            ok = bool(text) and needle not in (text or "")
        present[key] = ok
        if not ok:
            missing.append(key)
    return {
        "ok": not missing,
        "missing": missing,
        "required_present": present,
        "via": via,
        "reading": (
            "affine2 / q80 shaders contract in-register dequant and no dense W. "
            "dispatch_hgravs still writes hgravs_mid; the fused two-stage "
            "kernel exists in q80_mixed_decode.metal and is not bound. "
            "expand_rice_indices is a load-time CSR expansion."
        ),
    }


def sealed_decode_motif_counts() -> dict[str, int]:
    return {mid: n for mid, n, _why in SEALED_REPRESENTATION_DECODE_MOTIFS}


def walk_representation_decode(geo: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Independent encode-path walk. Must match DISPATCH_MOTIFS and 258."""
    geo = geo if geo is not None else load_geometry()
    sealed = walk_launches(geo, Fusion.sealed_resident())
    unfused = walk_launches(geo, Fusion.env_unset_default())
    sealed_counts = cluster_launches(sealed)
    unfused_counts = cluster_launches(unfused)
    sealed_fam = family_counts(sealed_counts)
    unfused_fam = family_counts(unfused_counts)
    return {
        "sealed_total": len(sealed),
        "unfused_total": len(unfused),
        "sealed_families": sealed_fam,
        "unfused_families": unfused_fam,
        "sealed_motif_counts": {
            mid: int(sealed_counts[mid]) for mid, _n, _why in SEALED_REPRESENTATION_DECODE_MOTIFS
        },
        "unfused_representation_decode": int(unfused_fam.get("representation_decode", -1)),
    }


def motifs_representation_decode(doc: Mapping[str, Any] | None = None) -> int:
    """Extract families.sealed.representation_decode from DISPATCH_MOTIFS.json."""
    if doc is None:
        doc = _load_receipt(DISPATCH_MOTIFS_REL)
    families = doc.get("families")
    if not isinstance(families, Mapping):
        raise DecodeFusionRefuse("REFUSED: DISPATCH_MOTIFS.json missing families")
    sealed = families.get("sealed")
    if not isinstance(sealed, Mapping):
        raise DecodeFusionRefuse("REFUSED: DISPATCH_MOTIFS.json missing families.sealed")
    raw = sealed.get("representation_decode")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DecodeFusionRefuse(
            "REFUSED: DISPATCH_MOTIFS families.sealed.representation_decode "
            "is not an int"
        )
    return int(raw)


def reconcile_representation_decode(
    got: int,
    *,
    want: int = ESTABLISHED_REPRESENTATION_DECODE_SEALED,
    motifs_got: int | None = None,
    walk_got: int | None = None,
    motif_counts: Mapping[str, int] | None = None,
    sealed_total: int | None = None,
    detail: str = "",
) -> dict[str, Any]:
    """Refuse rather than emit an unreconciled 258.

    Three oracles, one integer: the established constant, DISPATCH_MOTIFS.json,
    and the encode-path walk. A partition of motif ids must also sum to 258.
    """
    got = int(got)
    want = int(want)
    if got != want:
        raise UnreconciledRepresentationDecode(got, want, detail=detail or "primary got")
    if motifs_got is not None and int(motifs_got) != want:
        raise UnreconciledRepresentationDecode(
            int(motifs_got), want, detail=detail or f"{DISPATCH_MOTIFS_REL}"
        )
    if walk_got is not None and int(walk_got) != want:
        raise UnreconciledRepresentationDecode(
            int(walk_got), want, detail=detail or "encode-path walk"
        )
    if motif_counts is not None:
        pinned = sealed_decode_motif_counts()
        extra = sorted(set(motif_counts) - set(pinned))
        missing = sorted(set(pinned) - set(motif_counts))
        if extra or missing:
            raise DecodeFusionRefuse(
                f"REFUSED: representation_decode motif set drifted extra={extra} "
                f"missing={missing}"
            )
        part = sum(int(motif_counts[mid]) for mid in pinned)
        if part != want:
            raise UnreconciledRepresentationDecode(
                part, want, detail=detail or "motif partition sum"
            )
        for mid, n in pinned.items():
            got_n = int(motif_counts[mid])
            if got_n != n:
                raise DecodeFusionRefuse(
                    f"REFUSED: motif {mid} count {got_n} != pinned {n}"
                )
    if sealed_total is not None and int(sealed_total) != ESTABLISHED_SEALED:
        raise DecodeFusionRefuse(
            f"REFUSED: sealed token dispatches {sealed_total} != {ESTABLISHED_SEALED}"
        )
    return {
        "ok": True,
        "representation_decode": want,
        "sealed_total": ESTABLISHED_SEALED,
        "share": ESTABLISHED_SHARE,
        "unfused_representation_decode": ESTABLISHED_REPRESENTATION_DECODE_UNFUSED,
        "treated_unknown_as_zero": False,
    }


def packing_from_receipts() -> dict[str, Any]:
    """Sealed-3.14 packing: MLP is affine-Q2; the other GEMVs are Q4."""
    kg = _load_receipt(KERNEL_GEOMETRY_REL)
    census = kg.get("census") if isinstance(kg.get("census"), Mapping) else {}
    affine = int(census.get("affine", -1))
    q4 = int(census.get("q4", -1))
    if affine != AFFINE_TENSORS:
        raise DecodeFusionRefuse(
            f"REFUSED: KERNEL_GEOMETRY affine tensors {affine} != {AFFINE_TENSORS}"
        )
    if q4 != Q4_TENSORS_QUOTED:
        raise DecodeFusionRefuse(
            f"REFUSED: KERNEL_GEOMETRY q4 tensors {q4} != {Q4_TENSORS_QUOTED}"
        )
    aux = _load_receipt(MLP_AUX_REL)
    acc = aux.get("accounting") if isinstance(aux.get("accounting"), Mapping) else {}
    if int(acc.get("n_parameters", -1)) != MLP_PARAMS:
        raise DecodeFusionRefuse("REFUSED: MLP_AUX n_parameters drifted from geometry")
    if int(acc.get("code_bytes", -1)) != MLP_CODE_BYTES:
        raise DecodeFusionRefuse("REFUSED: MLP_AUX code_bytes drifted")
    if int(acc.get("auxiliary_bytes", -1)) != MLP_AUX_BYTES:
        raise DecodeFusionRefuse("REFUSED: MLP_AUX auxiliary_bytes drifted")
    inc = acc.get("incumbent_packing") if isinstance(acc.get("incumbent_packing"), Mapping) else {}
    return {
        "affine_tensors": affine,
        "q4_tensors": q4,
        "mlp_family": inc.get("family", "affine_q2_group64_ls"),
        "mlp_reconstruction": inc.get("reconstruction"),
        "mlp_params": MLP_PARAMS,
        "mlp_code_bytes": MLP_CODE_BYTES,
        "mlp_aux_bytes": MLP_AUX_BYTES,
        "hgravs_gemv_tensors_in_kernel_geometry_census": 0,
        "residual_gemv_tensors_in_kernel_geometry_census": 0,
        "binary_gemv_tensors_in_kernel_geometry_census": 0,
        "reading": (
            "KERNEL_GEOMETRY census is 192 affine + 209 q4. That is the "
            "sealed-3.14 GEMV mix: MLP affine-Q2 group-64, everything else "
            "a packed Q4 GEMV or the embedding row lookup. HGRAVS / residual "
            "/ binary / PQ are codec lanes in decode.rs and shaders, not "
            "members of the 258 on this artifact."
        ),
    }


def dispatch_count_is_not_the_cost() -> dict[str, Any]:
    """Cite ORGAN_BANDWIDTH; do not refit the negative per-dispatch slope."""
    doc = _load_receipt(ORGAN_BANDWIDTH_REL)
    findings = doc.get("findings") if isinstance(doc.get("findings"), list) else []
    ids = [f.get("id") for f in findings if isinstance(f, Mapping)]
    if "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE" not in ids:
        raise DecodeFusionRefuse(
            "REFUSED: ORGAN_BANDWIDTH no longer records the dispatch-count "
            "refutation this module is not allowed to reopen"
        )
    return {
        "cited_from": ORGAN_BANDWIDTH_REL,
        "finding_id": "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE",
        "what": next(
            f.get("what")
            for f in findings
            if isinstance(f, Mapping)
            and f.get("id") == "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE"
        ),
        "this_module_does_not_target_258_to_a_smaller_integer": True,
    }


def candidates() -> list[dict[str, Any]]:
    remat = dense_f16_w_bytes()
    mlp_w = remat["mlp"]
    q4_w = remat["attention_gemv"] + remat["lm_head"]
    token_w = remat["token_gemv"]
    hgravs_mid = f32_bytes(HGRAVS_RANK)

    already = _empty_blockers(
        argument_buffers="already bound per GEMV with set_buffer + set_bytes",
        dynamic_shapes="none on these GEMVs; grids are rows × tpr64",
        icb="not required; decode is already inside the kernel",
        resource_residency="packed codes, scales, biases already UMA-resident",
        routing_dependence="none; mixer kind is static, MLP is dense",
        threadgroup_capacity="tpr64 TG 128 already holds the reduction tree",
    )
    rejected_blockers = _empty_blockers(
        argument_buffers="a dense-W GEMV would still need rows/cols set_bytes",
        dynamic_shapes="a (rows×cols) f16 W is a dynamic working set of tens of GB",
        icb="irrelevant; the rejection is the write of W, not the launch",
        resource_residency=(
            f"UMA would have to hold {token_w} extra f16 bytes of W on top of "
            "the packed catalog; that is the physical problem, not a fix"
        ),
        routing_dependence="none",
        threadgroup_capacity="n/a; the illegal step is the dense write",
    )
    hgravs_blockers = _empty_blockers(
        argument_buffers=(
            "the fused two-stage kernel takes 12 set_bytes slots (rows/cols/"
            "group/bits/bound for both factors). An ICB replay would need "
            "those promoted. Bind, not ICB, is the first gap: dispatch_hgravs "
            "never names q80_hgravs01_two_stage_matvec"
        ),
        dynamic_shapes="none; rank is 160, K is a compile-time organ width",
        icb="wrong textbook; this is one kernel versus two, not graph replay",
        resource_residency="both factors already resident; mid is 640 B",
        routing_dependence="none",
        threadgroup_capacity=(
            f"q80_hgravs01_two_stage_matvec stages x in threadgroup with "
            f"kXCap={HGRAVS_X_CAP}. This model's K is in {{{HIDDEN}, "
            f"{INTERMEDIATE}, {O_PROJ_COLS}}}. All exceed 512, so the fused "
            "kernel returns without writing. That is the Metal blocker on "
            "this body even if a future catalog grew HGRAVS GEMVs."
        ),
    )

    rows = [
        _candidate(
            id="affine_q2_unpack_plus_matvec",
            name="affine-Q2 unpack already lives inside the GEMV",
            operation=UNPACK_PLUS_MATVEC,
            status=ALREADY_FUSED,
            on_sealed_258=True,
            sealed_launches=LAYERS + LAYERS,  # gate_up_swiglu + down
            organs=("mlp.gate", "mlp.up", "mlp.down"),
            mechanism=(
                "HGRAVF01 codes stay packed (4 q per byte). "
                "qwen_affine_q2_group32_matvec_geo_tpr64_tg128 / gate_up_swiglu "
                "siblings reconstruct w = float(q)*scale + bias in registers "
                "and FMA with x in the same loop. No (rows×cols) buffer."
            ),
            what_is_materialized="nothing of W. The GEMV writes y, the consume result.",
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=mlp_w,
            bytes_eliminated_if_true=0,
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "Production already consumes packed affine-Q2. Writing f16 W "
                f"then binding a generic GEMV would add {mlp_w} bytes of "
                "intermediate traffic and is the thing this candidate has "
                "already refused."
            ),
            metal_blockers=already,
            blocked_today=False,
            cheapest_falsifier=(
                "STATIC: affine2_group32_matvec.metal header still says "
                "'never writes a dense W', and MixedCatalogCensus."
                "dense_w_materialized stays 0 on HQ38M20 load. A new kernel "
                "that stores (rows×cols) f16 W, or a census that increments "
                "dense_w_materialized, kills this finding without a GPU lease."
            ),
        ),
        _candidate(
            id="q4_unpack_plus_matvec",
            name="Q4 unpack already lives inside the GEMV / row lookup",
            operation=UNPACK_PLUS_MATVEC,
            status=ALREADY_FUSED,
            on_sealed_258=True,
            sealed_launches=DN_LAYERS + DN_LAYERS + GQA_LAYERS + GQA_LAYERS + 1 + 1,
            organs=(
                "attention.linear_qkvz",
                "attention.linear_ba",
                "attention.linear_out",
                "attention.q",
                "attention.k",
                "attention.v",
                "attention.o",
                "lm_head",
                "embedding",
            ),
            mechanism=(
                "Uniform Q4 group-64: q = nibble-8, w = float(q)*scale, in "
                "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 and the fused "
                "pair-concat / QKV siblings. Embed is a packed row lookup, "
                "not a GEMV. Codes stay packed."
            ),
            what_is_materialized="nothing of W. Embed writes one hidden row (the lookup result).",
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=q4_w,
            bytes_eliminated_if_true=0,
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                f"A split Q4→f16 W then GEMV would write {q4_w} bytes of "
                "attention+lm_head W that production never writes."
            ),
            metal_blockers=already,
            blocked_today=False,
            cheapest_falsifier=(
                "STATIC: qwen_uniform_q4.metal still decodes a nibble in "
                "qwen_uniform_q4_value / unpack8 and accumulates against x. "
                "QWEN38_Q4_DECODE_PROBE is a diagnostic isolate, not a 628 "
                "graph launch. Kill if a production bind starts writing a "
                "decoded Q4 row-major W."
            ),
        ),
        _candidate(
            id="affine_q2_scale_plus_matvec",
            name="affine scale/bias already apply inside the same FMA",
            operation=SCALE_PLUS_MATVEC,
            status=ALREADY_FUSED,
            on_sealed_258=True,
            sealed_launches=LAYERS + LAYERS,
            organs=("mlp.gate", "mlp.up", "mlp.down"),
            mechanism=(
                "geo_tpr64 affine2 binds on-disk f16 scale/bias as half* and "
                "widens with float(scales[rgb]) in-register. AccFuse rewrites "
                "sum((q*s+b)*x) = s*sum(q x)+b*sum(x) as an opt-in ALU "
                "association (HAWKING_AFFINE2_GEO=accfuse); it does not "
                "eliminate the scale/bias *load*. tgsb stages aux in "
                "threadgroup to avoid re-fetching per 8-wide tile. Neither "
                "is a separate decode dispatch. The 1.07 GB aux is stored "
                "representation (MLP_AUXILIARY_INFORMATION), not an "
                "intermediate of a decode pass."
            ),
            what_is_materialized=(
                "no scale buffer is written at token time. Aux is catalog "
                f"storage: {MLP_SCALE_BYTES} scale + {MLP_BIAS_BYTES} bias."
            ),
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            bytes_eliminated_if_true_note=(
                "Generating or dropping aux is MLP_AUXILIARY_INFORMATION, "
                "not a decode+consume fuse. A lowering that writes an f16 W "
                "from q,s,b then GEMVs is unpack_to_dense_f16_then_gemv."
            ),
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "Scale applies in-register. Materializing S or W is not the "
                "production path."
            ),
            metal_blockers=already,
            blocked_today=False,
            cheapest_falsifier=(
                "STATIC: production geo_tpr64 kernel still takes "
                "device const half* scales/biases and does float(s) in the "
                "FMA. Kill if bind starts widening aux to a resident f32 "
                "buffer that the token then re-reads, or if a new kernel "
                "writes q*s+b into a W tensor."
            ),
        ),
        _candidate(
            id="q4_scale_plus_matvec",
            name="Q4 group scale already applies inside the same FMA",
            operation=SCALE_PLUS_MATVEC,
            status=ALREADY_FUSED,
            on_sealed_258=True,
            sealed_launches=DN_LAYERS + DN_LAYERS + GQA_LAYERS + GQA_LAYERS + 1 + 1,
            mechanism=(
                "Each Q4 group owns one f16 scale. The GEMV loads it as half "
                "and does float(q)*scale * x[i] in-register. There is no "
                "separate scale-apply kernel on the 628 graph."
            ),
            what_is_materialized="nothing. Scale is catalog storage, consumed in-register.",
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason="Q4 scale is already a direct consumer.",
            metal_blockers=already,
            blocked_today=False,
            cheapest_falsifier=(
                "STATIC: qwen_uniform_q4_value still returns float(q)*float(scales[group]) "
                "and the geo_tpr64 matvec calls it (or unpack8) against x. "
                "Kill if a production pass writes dequantized Q4 into a buffer "
                "the GEMV then reads."
            ),
        ),
        _candidate(
            id="codebook_lookup_plus_accumulate",
            name="codebook lookup + accumulate is fused in gravity_pq, absent here",
            operation=CODEBOOK_LOOKUP_PLUS_ACCUMULATE,
            status=NOT_THIS_ARTIFACT,
            on_sealed_258=False,
            sealed_launches=0,
            mechanism=(
                "gravity_pq_matvec / gravity_residual_pq_matvec index a half "
                "codebook and accumulate against x without expanding a dense "
                "row. That is FusedDecodeCompute for PQ. Sealed-3.14 GEMVs "
                "are affine-Q2 and Q4, not PQ. Expanding a codebook to W "
                "then running the ordinary GEMV is unpack_to_dense_f16_then_gemv."
            ),
            what_is_materialized="no codebook intermediate on the 258.",
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "The PQ kernels already consume entries in-register. A "
                "codebook→dense-W lowering is REJECTED_DENSE_REMAT and is "
                "filed under unpack_to_dense_f16_then_gemv, not here."
            ),
            metal_blockers=_empty_blockers(
                argument_buffers="gravity_pq_matvec already binds codebooks + codes + x",
                dynamic_shapes="sub/card are kernel params; not on this graph",
                icb="n/a; 0 launches",
                resource_residency="n/a on sealed-3.14",
                routing_dependence="none",
                threadgroup_capacity="n/a",
            ),
            blocked_today=True,
            cheapest_falsifier=(
                "STATIC: KERNEL_GEOMETRY census affine=192 q4=209, no PQ "
                "organ. Kill this NOT_THIS_ARTIFACT if a future catalog binds "
                "gravity_pq_matvec on the decode token graph; then the fused "
                "kernel is already the consumer and the remaining question is "
                "only a dense-W lowering, which stays REJECTED."
            ),
        ),
        _candidate(
            id="sparse_residual_decode_plus_consume",
            name="HGRAVR02 CSR residual is fused when recon_fuse=1, absent here",
            operation=SPARSE_RESIDUAL_DECODE_PLUS_CONSUME,
            status=NOT_THIS_ARTIFACT,
            on_sealed_258=False,
            sealed_launches=0,
            mechanism=(
                "dispatch_residual with HAWKING_QWEN38_RECON_FUSE default ON "
                "launches q80_binary_group_csr_matvec_* : binary GEMV and CSR "
                "outlier apply in one kernel. recon_fuse=0 is the split: "
                "binary GEMV writes y, then q80_sparse_q1_apply_csr re-reads y. "
                "Sealed-3.14 GEMV census has no residual tensors, so neither "
                "form is in the 258."
            ),
            what_is_materialized=(
                "on the split path only: the binary GEMV output y, then the "
                "CSR kernel reads y and adds. On the fused path: nothing extra. "
                "On this artifact: nothing."
            ),
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "CSR indices + residual signs are the representation. The "
                "fused kernel consumes them. Expanding residual to a dense W "
                "is REJECTED_DENSE_REMAT."
            ),
            metal_blockers=_empty_blockers(
                argument_buffers="fused CSR kernel already binds signs, scales, indices, row_ptr, residual_signs",
                dynamic_shapes="none; cols must be a 256-tile multiple when cols>2048 (qwen38_assert_k_complete_cols)",
                icb="n/a on this artifact",
                resource_residency="n/a on sealed-3.14 GEMVs",
                routing_dependence="none",
                threadgroup_capacity="tg256 / simd_bytes already sized",
            ),
            blocked_today=True,
            cheapest_falsifier=(
                "STATIC: KERNEL_GEOMETRY census has no residual GEMVs. "
                "Source still contains the split q80_sparse_q1_apply_csr "
                "under recon_fuse=0. Kill NOT_THIS_ARTIFACT if a catalog row "
                "binds MixedGpuWeight::Residual on the 628 graph; then check "
                "recon_fuse is still ON so the split y round-trip does not "
                "return."
            ),
        ),
        _candidate(
            id="hgravs_two_stage_mid",
            name="HGRAVS y=L@(R@x) still writes a rank-160 mid buffer",
            operation=UNPACK_PLUS_MATVEC,
            status=OPEN,
            on_sealed_258=False,
            sealed_launches=0,
            mechanism=(
                "dispatch_hgravs runs two factor GEMVs. First writes "
                "workspace.hgravs_mid (f32[160]), second reads it. "
                "q80_hgravs01_two_stage_matvec already fuses that into one "
                "launch with mid in threadgroup (640 B, not dense W) but is "
                "not bound. This is the only remaining decode-then-consume "
                "*buffer* on the mixed codec path."
            ),
            what_is_materialized="f32[160] hgravs_mid, 640 bytes, written then re-read.",
            intermediate_bytes_written_and_reread=0,  # 0 launches on the 258
            bytes_per_hgravs_gemv_write_plus_reread=hgravs_mid * 2,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            bytes_eliminated_if_true_note=(
                f"Per HGRAVS GEMV the fused kernel would eliminate {hgravs_mid * 2} "
                "bytes of mid traffic. Sealed-3.14 has 0 HGRAVS GEMVs, so the "
                "token number is 0. Ranked so a future mixed catalog cannot "
                "pretend the split is undiscovered."
            ),
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "Two-stage fused keeps mid in threadgroup. Reconstructing "
                "L@R as dense W is REJECTED_DENSE_REMAT."
            ),
            metal_blockers=hgravs_blockers,
            blocked_today=True,
            cheapest_falsifier=(
                "STATIC, already visible: (1) KERNEL_GEOMETRY has 0 HGRAVS "
                "GEMVs on sealed-3.14, so the 258 does not contain this "
                "split. (2) kXCap=512 < hidden 5120, so binding the existing "
                "fused kernel on this body would no-op. A real fuse needs a "
                "K-complete x stage (or no x_tg) AND a dispatch_hgravs bind. "
                "Do not GPU-lease a kernel that returns on K>512."
            ),
        ),
        _candidate(
            id="rice_index_expansion_at_upload",
            name="rice bitstream is expanded to CSR indices at catalog load",
            operation=SPARSE_RESIDUAL_DECODE_PLUS_CONSUME,
            status=LOAD_TIME_ONLY,
            on_sealed_258=False,
            sealed_launches=0,
            mechanism=(
                "upload_mixed Residual calls expand_rice_indices and "
                "rice_q1_row_ptr, then binds the expanded u32 indices + row_ptr. "
                "Every later token re-reads CSR, not rice. That is a load-time "
                "LayoutTransform of the residual representation, not a per-token "
                "decode launch. An in-kernel rice consumer would skip the "
                "expanded index buffer; none exists, and this artifact has no "
                "residual GEMVs."
            ),
            what_is_materialized=(
                "CSR indices + row_ptr, once per residual tensor at upload. "
                "Not on the 258."
            ),
            intermediate_bytes_written_and_reread=0,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=0,
            physical_primitive="LayoutTransform",
            dense_rematerialization=DIRECT_CONSUME,
            dense_rematerialization_reason=(
                "Expanding rice→CSR is not expanding to W. Expanding residual "
                "to dense W is REJECTED_DENSE_REMAT."
            ),
            metal_blockers=_empty_blockers(
                argument_buffers="CSR consumer already bound; a rice consumer does not exist",
                dynamic_shapes="rice length is per-tensor, known at upload",
                icb="n/a; this is load-time",
                resource_residency="expanded CSR is resident for the process",
                routing_dependence="none",
                threadgroup_capacity="a bitstream walker would be divergent; that is why upload expands",
            ),
            blocked_today=True,
            cheapest_falsifier=(
                "STATIC: sealed-3.14 has no residual tensors. Reopen only if "
                "a residual catalog row appears AND a kernel consumes rice "
                "bits in-register. Measuring CSR bytes without a residual "
                "tensor is inventing a denominator."
            ),
        ),
        _candidate(
            id="unpack_to_dense_f16_then_gemv",
            name="decode packed codes to f16 W, then run the ordinary GEMV",
            operation=UNPACK_PLUS_MATVEC,
            status=REJECTED_DENSE_REMAT,
            on_sealed_258=False,
            sealed_launches=0,
            mechanism=(
                "The textbook split this sidecar is forbidding: materialize "
                "W_f16[rows, cols] from affine-Q2 / Q4 / PQ / HGRAVS, then "
                "bind a generic f16 GEMV. That is the dense rematerialization "
                "the atlas primitive FusedDecodeCompute exists to remove. "
                "Production dense_w_materialized stays 0 so this cannot hide."
            ),
            what_is_materialized=f"f16 W of every GEMV organ, {token_w} bytes per token if whole tensors are decoded.",
            intermediate_bytes_written_and_reread=token_w * 2,
            bytes_eliminated_vs_split_decode=0,
            bytes_eliminated_if_true=token_w,
            bytes_eliminated_if_true_note=(
                "These bytes are what the split would add, not what a legal "
                "fuse still has to remove. Production already does not write "
                "them. Ranked so a 'just dequant then GEMV' proposal cannot "
                "out-rank real fuses by pretending the write is free."
            ),
            physical_primitive="FusedDecodeCompute",
            dense_rematerialization=REJECTED_DENSE_REMAT,
            dense_rematerialization_reason=(
                "Writes a dense weight tensor and then runs an ordinary GEMV. "
                "That is the production path this module is forbidden to "
                "recommend. Verification MAY reconstruct W; production may not."
            ),
            metal_blockers=rejected_blockers,
            blocked_today=True,
            cheapest_falsifier=(
                "STATIC: this candidate is rejected by contract, not by a "
                "measurement. A generate gate cannot undeny it. The cheap "
                "check that production has not silently become this candidate "
                "is dense_w_materialized == 0 on mixed catalog load."
            ),
        ),
    ]
    ids = [r["id"] for r in rows]
    if ids != list(REQUIRED_CANDIDATE_IDS):
        raise DecodeFusionRefuse(
            f"candidate catalog {ids} != required {list(REQUIRED_CANDIDATE_IDS)}"
        )
    return rows


def rank_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank by intermediate W bytes, never by launches removed.

    Legal ranking uses bytes_eliminated_vs_split_decode (what production
    already does not write, or what an OPEN fuse would still remove).
    REJECTED_DENSE_REMAT is recorded, then excluded from top_legal.
    Remaining OPEN bytes_eliminated_if_true is a second list so a 0-byte
    leftover cannot impersonate the 34 GB already fused.
    """
    def _bytes(row: Mapping[str, Any], key: str) -> int:
        raw = row.get(key)
        if raw is None:
            return 0
        return int(raw)

    legal = [r for r in rows if r["status"] != REJECTED_DENSE_REMAT]
    rejected = [r for r in rows if r["status"] == REJECTED_DENSE_REMAT]
    legal_sorted = sorted(
        legal,
        key=lambda r: (
            _bytes(r, "bytes_eliminated_vs_split_decode"),
            _bytes(r, "bytes_eliminated_if_true"),
            -int(r.get("sealed_launches") or 0),  # fewer launches lose when bytes tie
        ),
        reverse=True,
    )
    remaining = sorted(
        [r for r in legal if r["status"] == OPEN],
        key=lambda r: _bytes(r, "bytes_eliminated_if_true"),
        reverse=True,
    )
    if not legal_sorted:
        raise DecodeFusionRefuse("REFUSED: no legal candidates to rank")
    top = legal_sorted[0]
    # Guard: ranking must not be "whoever has more launches".
    by_launches = sorted(
        legal, key=lambda r: int(r.get("sealed_launches") or 0), reverse=True
    )
    return {
        "rank_by": "bytes_eliminated_vs_split_decode",
        "not_by": "dispatches_removed",
        "order": [r["id"] for r in legal_sorted],
        "top_legal": top["id"],
        "top_bytes_eliminated_vs_split_decode": _bytes(top, "bytes_eliminated_vs_split_decode"),
        "cheapest_falsifier_for_top": top["cheapest_falsifier"],
        "top_status": top["status"],
        "remaining_open": [r["id"] for r in remaining],
        "rejected_dense_remat": [r["id"] for r in rejected],
        "launch_order_differs_from_byte_order": [r["id"] for r in by_launches]
        != [r["id"] for r in legal_sorted],
        "reading": (
            "The largest intermediate W traffic on this graph is the dense f16 "
            "W that production already does not write. affine-Q2 unpack+matvec "
            "is therefore the top legal candidate: it is already one operation. "
            "The OPEN leftover (HGRAVS mid) is 0 bytes on sealed-3.14 and is "
            "blocked on Metal by kXCap=512. REJECTED_DENSE_REMAT is the same "
            "34+ GB in the opposite direction and is not a plan."
        ),
    }


def sites() -> list[dict[str, Any]]:
    """Every named place decode could be materialized, including zeros."""
    remat = dense_f16_w_bytes()
    return [
        {
            "id": "affine_q2_inregister",
            "file": AFFINE2_SHADER,
            "also": Q80_SHADER,
            "operation": UNPACK_PLUS_MATVEC,
            "materialized": None,
            "intermediate_bytes": 0,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "already the production MLP GEMV",
        },
        {
            "id": "q4_inregister",
            "file": Q4_SHADER,
            "operation": UNPACK_PLUS_MATVEC,
            "materialized": None,
            "intermediate_bytes": 0,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "already the production attention / lm_head GEMV",
        },
        {
            "id": "affine_scale_inregister",
            "file": AFFINE2_SHADER,
            "operation": SCALE_PLUS_MATVEC,
            "materialized": None,
            "intermediate_bytes": 0,
            "stored_aux_bytes": MLP_AUX_BYTES,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "half scale/bias widened in-register; aux is storage",
        },
        {
            "id": "gravity_pq_inregister",
            "file": PQ_SHADER,
            "operation": CODEBOOK_LOOKUP_PLUS_ACCUMULATE,
            "materialized": None,
            "intermediate_bytes": 0,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "fused in shaders; 0 launches on sealed-3.14",
        },
        {
            "id": "residual_csr_fused",
            "file": DECODE_SRC,
            "operation": SPARSE_RESIDUAL_DECODE_PLUS_CONSUME,
            "materialized": None,
            "intermediate_bytes": 0,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "recon_fuse=1 fused kernel; 0 residual GEMVs here",
        },
        {
            "id": "hgravs_mid_buffer",
            "file": DECODE_SRC,
            "operation": UNPACK_PLUS_MATVEC,
            "materialized": "workspace.hgravs_mid f32[160]",
            "intermediate_bytes": f32_bytes(HGRAVS_RANK) * 2,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": (
                "two launches; fused kernel exists but kXCap=512 blocks this K; "
                "0 HGRAVS GEMVs on sealed-3.14"
            ),
        },
        {
            "id": "rice_csr_at_upload",
            "file": DECODE_SRC,
            "operation": SPARSE_RESIDUAL_DECODE_PLUS_CONSUME,
            "materialized": "expand_rice_indices → u32 CSR",
            "intermediate_bytes": 0,
            "consumer_primitive": "LayoutTransform",
            "metal_today": "load-time; no residual tensors on this artifact",
        },
        {
            "id": "dense_f16_w_counterfactual",
            "file": DECODE_SRC,
            "operation": UNPACK_PLUS_MATVEC,
            "materialized": "f16[rows, cols] W",
            "intermediate_bytes": remat["token_gemv"] * 2,
            "consumer_primitive": "FusedDecodeCompute",
            "metal_today": "REJECTED_DENSE_REMAT; dense_w_materialized stays 0",
        },
    ]


def analyze() -> dict[str, Any]:
    markers = helper_markers()
    if not markers["ok"]:
        raise DecodeFusionRefuse(
            f"REFUSED: decode/shader markers missing {markers['missing']}"
        )
    # The two-stage kernel must exist in the shader (fused consumer ready)
    # and must not be bound from decode.rs. helper_markers encodes the second
    # as decode_two_stage_unbound. The first is q80_two_stage_kernel.
    if not markers["required_present"].get("q80_two_stage_kernel"):
        raise DecodeFusionRefuse("REFUSED: fused HGRAVS two-stage kernel missing from shaders")

    motifs_doc = _load_receipt(DISPATCH_MOTIFS_REL)
    motifs_n = motifs_representation_decode(motifs_doc)
    walked = walk_representation_decode()
    rec = reconcile_representation_decode(
        ESTABLISHED_REPRESENTATION_DECODE_SEALED,
        motifs_got=motifs_n,
        walk_got=int(walked["sealed_families"]["representation_decode"]),
        motif_counts=walked["sealed_motif_counts"],
        sealed_total=int(walked["sealed_total"]),
        detail="analyze()",
    )
    if int(walked["unfused_families"]["representation_decode"]) != ESTABLISHED_REPRESENTATION_DECODE_UNFUSED:
        raise UnreconciledRepresentationDecode(
            int(walked["unfused_families"]["representation_decode"]),
            ESTABLISHED_REPRESENTATION_DECODE_UNFUSED,
            detail="unfused walk",
        )
    packing = packing_from_receipts()
    cands = candidates()
    ranking = rank_candidates(cands)
    remat = dense_f16_w_bytes()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "purpose": (
            "Explain why representation_decode is 258 of 628 without treating "
            "that integer as a fusion target, and judge decode+consume as one "
            "physical operation by intermediate bytes."
        ),
        "established": {
            "sealed_dispatches_per_decoded_token": ESTABLISHED_SEALED,
            "unfused_dispatches_per_decoded_token": ESTABLISHED_UNFUSED,
            "representation_decode_sealed": ESTABLISHED_REPRESENTATION_DECODE_SEALED,
            "representation_decode_unfused": ESTABLISHED_REPRESENTATION_DECODE_UNFUSED,
            "share_of_sealed": ESTABLISHED_SHARE,
            "dispatch_count_is_not_the_350GBs_cause": dispatch_count_is_not_the_cost(),
            "not_a_fuse_to_reduce_the_integer": True,
        },
        "reconciliation": rec,
        "partition": {
            "motifs": [
                {
                    "id": mid,
                    "sealed_count": n,
                    "why": why,
                }
                for mid, n, why in SEALED_REPRESENTATION_DECODE_MOTIFS
            ],
            "sum": ESTABLISHED_REPRESENTATION_DECODE_SEALED,
        },
        "walk": {
            "sealed_total": walked["sealed_total"],
            "unfused_total": walked["unfused_total"],
            "sealed_families": walked["sealed_families"],
            "unfused_representation_decode": walked["unfused_representation_decode"],
            "sealed_motif_counts": walked["sealed_motif_counts"],
        },
        "packing": packing,
        "answer": {
            "why_is_it_separate": (
                "It is not, for W, on the sealed Metal path. The 258 "
                "representation_decode launches *are* the consume: packed "
                "GEMVs (and one embedding row lookup) that dequant in-register. "
                "The family is large because there are 258 packed weight "
                "consumers, not because a decode pass writes numbers that a "
                "later GEMV rereads. ORGAN_BANDWIDTH already folded "
                "low_bit_decode into the organ kernels; this sidecar names "
                "the sites."
            ),
            "what_would_look_like_separate_and_is_forbidden": (
                "Decode packed codes to a full f16 W, then run the ordinary "
                f"GEMV. That would write {remat['token_gemv']} bytes of W "
                "and reread them. REJECTED_DENSE_REMAT."
            ),
            "what_is_still_a_split_on_a_codec_path": (
                "HGRAVS two-stage writes f32[160] mid (640 B) and the fused "
                "kernel is unbound and K-capped at 512. Zero such GEMVs on "
                "sealed-3.14. Rice→CSR expansion is load-time and also has "
                "zero residual tensors here."
            ),
        },
        "dense_f16_w_counterfactual_bytes": remat,
        "activation_intermediates_not_weight_decode": activation_intermediates(),
        "sites": sites(),
        "candidates": cands,
        "ranking": ranking,
        "helper_markers": markers,
        "nomenclature": {
            "already_fused": ALREADY_FUSED,
            "open": OPEN,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "not_this_artifact": NOT_THIS_ARTIFACT,
            "load_time_only": LOAD_TIME_ONLY,
            "direct_consume": DIRECT_CONSUME,
            "depends_on_lowering": DEPENDS_ON_LOWERING,
        },
        "gaps_closed": [
            "258 re-walked and reconciled against DISPATCH_MOTIFS.json or refused",
            "four decode+consume operations judged; dense remat marked REJECTED_DENSE_REMAT",
            "rank is intermediate W bytes, not launches",
            "HGRAVS mid named with the kXCap=512 Metal blocker",
            "activation intermediates recorded as not W",
        ],
        "what_this_does_not_prove": [
            "a GB/s change from keeping fused decode (no GPU)",
            "capability of a K-complete HGRAVS two-stage kernel",
            "that 1.07 GB of affine aux can shrink (that is MLP_AUXILIARY_INFORMATION)",
        ],
    }


def snapshot() -> dict[str, Any]:
    return analyze()


def build() -> Path:
    doc = analyze()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", action="store_true", help="write the receipt")
    parser.add_argument("--build", action="store_true", help="alias of --record")
    parser.add_argument("--selftest", action="store_true", help="alias of --record")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.record or args.build or args.selftest:
        path = build()
        print(path)
        return 0
    doc = analyze()
    rec = doc["reconciliation"]
    rank = doc["ranking"]
    print(
        f"representation_decode {rec['representation_decode']}/{ESTABLISHED_SEALED} "
        f"top={rank['top_legal']} "
        f"bytes_vs_split={rank['top_bytes_eliminated_vs_split_decode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
