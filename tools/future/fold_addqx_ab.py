#!/usr/bin/env python3
"""Complete-token A/B: post-widen_f4 incumbent vs fold_addqx.

MLP_DECODE_CHEAPEN measured fold_addqx at 370.9 GB/s, bit-identical, on
ONE LAYER (1.127x, projected −1.745 ms). That sat at DIRTY_DIAGNOSTIC.
This sidecar is the 628/580-graph A/B the same way DELTANET_WIDEN_AB
promoted widen_f4: incumbent against the candidate on the real decode
path, MTLCommandBuffer GPUStartTime/GPUEndTime, token-id bytes, fallbacks
0. The incumbent arm IS the new post-widen_f4 baseline.

    python3 tools/future/fold_addqx_ab.py --measure --record
    python3 tools/future/fold_addqx_ab.py --from RAW.json --record
    python3 -m pytest tools/future/test_fold_addqx_ab.py -q

PARITY IS TOKEN-IDENTICAL AND THE ARITHMETIC IS EXACT. A byte comparison
at the token (and layer-0 named-matvec buffers) is required; citing the
probe is refused. SELF_MEASURED_DIRTY qualifies nothing. No TPS is
labelled QUALIFIED. If the complete-token saving does not reproduce the
1.745 ms projection, that is the result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "FOLD_ADDQX_AB.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_FOLD_ADDQX_AB_raw.json"
SCHEMA = "hawking.future.fold_addqx_ab.v1"
VERSION = 1
RECORDED_BY = "tools/future/fold_addqx_ab.py"

INCUMBENT_SWIGLU = "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"
INCUMBENT_DOWN = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"
FOLD_SWIGLU = "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_fold_addqx"
FOLD_DOWN = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_fold_addqx"
DN_F4 = "qwen38_gated_delta_decode_vi_simd_ba_f4"
SEALED_DISPATCHES = 580

CITED_PROBE_GB_S = 370.9
CITED_RATIO = 1.1265
CITED_PROJECTION_MS = 1.745
CITED_PRODUCTION_GB_S = 329.6
CITED_ARM_A_GB_S = 497.4
MATERIALITY_MS = 1.0
WIDEN_F4_SAVING_MS = 1.0245
WIDEN_F4_TOKEN_MS = 26.382083

SEALED_ENV = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
    "HAWKING_QWEN38_DN_STATE": "widen_f4",
}

DEFAULT_ARTIFACT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")
DEFAULT_TOKENIZER = DEFAULT_ARTIFACT / "tokenizer.json"

CLAIM_BOUNDARY = (
    "One-process complete-token A/B of the post-widen_f4 incumbent "
    "(GateUpSwiglu, FUSE_GQA_QKV, FUSE_DN_INPROJ, FUSE_ADD_RMSNORM, "
    "dn_state=widen_f4, affine2 geo=tpr64) against fold_addqx on the real "
    "encode_mlp path. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime. "
    "Complete-token ms is the median of per-generated-token GPU timestamps. "
    "Absolute ms are measured-under-load; the A/B ratio is back-to-back in "
    "the same process. The incumbent arm IS the new post-widen_f4 baseline "
    "(PATH_TO_71 28.722 ms is stale; DELTANET_WIDEN_AB's 26.382 ms is the "
    "previous measurement of this arm). Parity is token-id equality with "
    "fallbacks 0 AND a byte comparison proving bit-identity; argmax "
    "agreement is not parity; citing MLP_DECODE_CHEAPEN is not the identity "
    "proof. If the cited 1.745 ms projection does not appear in the complete "
    "token, the receipt names where it went. evidence_class "
    "SELF_MEASURED_DIRTY. No TPS is labelled QUALIFIED. A protected window "
    "is still required."
)


class FoldAbRefuse(ValueError):
    """Raised rather than emit an A/B receipt that cannot be defended."""


class ArgmaxIsNotParity(FoldAbRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity. Token-id equality "
            f"across runs is required{extra}."
        )


class NoByteComparison(FoldAbRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: fold_addqx claims bit-identity; a byte comparison is "
            f"required, citing the probe is not a substitute{extra}."
        )


class MissingArm(FoldAbRefuse):
    pass


class ProductionDidNotLaunch(FoldAbRefuse):
    pass


class EmptyGpuSample(FoldAbRefuse):
    pass


class QualifiedTpsRefused(FoldAbRefuse):
    pass


def ns_to_ms(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return int(ns) / 1e6


def _as_int_list(values: Sequence[Any] | None) -> list[int] | None:
    if values is None:
        return None
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise FoldAbRefuse(f"token id is not an integer: {v!r}")
        out.append(int(v))
    return out


def first_divergence(
    incumbent: Sequence[int], candidate: Sequence[int]
) -> dict[str, Any] | None:
    n = min(len(incumbent), len(candidate))
    for i in range(n):
        if int(incumbent[i]) != int(candidate[i]):
            return {
                "index": i,
                "incumbent": int(incumbent[i]),
                "candidate": int(candidate[i]),
            }
    if len(incumbent) != len(candidate):
        return {
            "index": n,
            "reason": "length",
            "incumbent_len": len(incumbent),
            "candidate_len": len(candidate),
        }
    return None


def byte_compare_u32(incumbent: Sequence[int], candidate: Sequence[int]) -> dict[str, Any]:
    """Little-endian u32 byte comparison of token-id lists."""
    a = b"".join(int(x).to_bytes(4, "little", signed=False) for x in incumbent)
    b = b"".join(int(x).to_bytes(4, "little", signed=False) for x in candidate)
    n = min(len(a), len(b))
    mismatch = 0
    first = None
    for i in range(n):
        if a[i] != b[i]:
            mismatch += 1
            if first is None:
                first = i
    if len(a) != len(b) and first is None:
        first = n
        mismatch += abs(len(a) - len(b))
    return {
        "compared_against": (
            "complete-token new_token_ids as little-endian u32 bytes; not the probe"
        ),
        "n_bytes_compared": n,
        "incumbent_len": len(a),
        "candidate_len": len(b),
        "n_mismatch_bytes": mismatch,
        "first_mismatch_index": first,
        "bit_identical": mismatch == 0 and len(a) == len(b) and n > 0,
    }


def require_byte_identity(compare: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if not isinstance(compare, Mapping):
        raise NoByteComparison(label)
    n = int(compare.get("n_bytes_compared") or 0)
    if n <= 0:
        raise NoByteComparison(f"{label}: n_bytes_compared is 0")
    mismatch = int(compare.get("n_mismatch_bytes") or 0)
    identical = bool(compare.get("bit_identical")) and mismatch == 0
    return {
        "label": label,
        "n_bytes_compared": n,
        "n_mismatch_bytes": mismatch,
        "first_mismatch_index": compare.get("first_mismatch_index"),
        "bit_identical": identical,
        "compared_against": compare.get("compared_against"),
    }


def report_token_parity(
    *,
    incumbent_token_ids: Sequence[int] | None = None,
    candidate_token_ids: Sequence[int] | None = None,
    fallbacks: int = 0,
    argmax_agreement: float | None = None,
    token_byte_compare: Mapping[str, Any] | None = None,
    layer0_byte_compare: Mapping[str, Any] | None = None,
    runs_compared: int | None = None,
) -> dict[str, Any]:
    if incumbent_token_ids is None or candidate_token_ids is None:
        raise ArgmaxIsNotParity(
            "token-id lists are required; argmax agreement on a sample "
            "is not a substitute"
            + (
                f" (argmax_agreement={argmax_agreement})"
                if argmax_agreement is not None
                else ""
            )
        )
    inc = _as_int_list(incumbent_token_ids)
    cand = _as_int_list(candidate_token_ids)
    assert inc is not None and cand is not None
    if token_byte_compare is None:
        token_byte_compare = byte_compare_u32(inc, cand)
    token_bytes = require_byte_identity(token_byte_compare, label="token_ids")
    layer0 = None
    layer0_ok = True
    if layer0_byte_compare is not None:
        layer0 = require_byte_identity(
            {
                "n_bytes_compared": int(
                    nested_bytes_compared(layer0_byte_compare) or 0
                ),
                "n_mismatch_bytes": int(
                    nested_mismatch(layer0_byte_compare) or 0
                ),
                "bit_identical": bool(layer0_byte_compare.get("bit_identical")),
                "first_mismatch_index": None,
                "compared_against": layer0_byte_compare.get("compared_against"),
            },
            label="layer0_named_matvec",
        )
        layer0_ok = bool(layer0["bit_identical"])
    identical = inc == cand
    div = None if identical else first_divergence(inc, cand)
    fb = int(fallbacks)
    arithmetic_exact = bool(token_bytes["bit_identical"] and layer0_ok)
    parity = bool(identical and fb == 0 and arithmetic_exact)
    return {
        "token_ids_identical": identical,
        "tokens_compared": min(len(inc), len(cand)),
        "incumbent_len": len(inc),
        "candidate_len": len(cand),
        "first_divergence": div,
        "fallbacks": fb,
        "parity": parity,
        "parity_basis": "token_id_equality_and_byte_identity",
        "argmax_is_not_parity": True,
        "argmax_agreement": argmax_agreement,
        "argmax_agreement_ignored": argmax_agreement is not None,
        "token_id_byte_compare": token_bytes,
        "layer0_byte_compare": layer0,
        "arithmetic_exact": arithmetic_exact,
        "runs_compared": runs_compared,
        "cited_probe_is_not_the_identity_proof": True,
    }


def nested_bytes_compared(doc: Mapping[str, Any]) -> int:
    n = 0
    if isinstance(doc.get("n_bytes_compared"), int):
        n += int(doc["n_bytes_compared"])
    for key in ("gate", "up", "down"):
        row = doc.get(key)
        if isinstance(row, Mapping) and isinstance(row.get("n_bytes_compared"), int):
            n += int(row["n_bytes_compared"])
    return n


def nested_mismatch(doc: Mapping[str, Any]) -> int:
    n = 0
    if isinstance(doc.get("n_mismatch_bytes"), int):
        n += int(doc["n_mismatch_bytes"])
    for key in ("gate", "up", "down"):
        row = doc.get(key)
        if isinstance(row, Mapping) and isinstance(row.get("n_mismatch_bytes"), int):
            n += int(row["n_mismatch_bytes"])
    return n


def kernel_count(histogram: Sequence[Mapping[str, Any]] | None, name: str) -> int:
    if not histogram:
        return 0
    total = 0
    for row in histogram:
        if row.get("kernel") == name:
            total += int(row.get("count") or 0)
    return total


def _median_u64(values: Sequence[int]) -> int | None:
    if not values:
        return None
    s = sorted(int(v) for v in values)
    return s[len(s) // 2]


def _require_gpu_ns(arm: Mapping[str, Any], label: str) -> int:
    ns = arm.get("gpu_ns_median")
    if ns is None:
        raise EmptyGpuSample(f"{label} is missing gpu_ns_median")
    ns_i = int(ns)
    if ns_i <= 0:
        raise EmptyGpuSample(f"{label} gpu_ns_median must be positive, got {ns_i}")
    return ns_i


def locate_saving(
    *,
    isolated_incumbent_ms: float,
    isolated_fold_ms: float,
    token_incumbent_ms: float,
    token_fold_ms: float,
    cited_projection_ms: float = CITED_PROJECTION_MS,
    materiality_ms: float = MATERIALITY_MS,
    isolated_matvec_incumbent_ms: float | None = None,
    isolated_matvec_fold_ms: float | None = None,
) -> dict[str, Any]:
    organ_ms = isolated_incumbent_ms - isolated_fold_ms
    token_saved_ms = token_incumbent_ms - token_fold_ms
    matvec_ms = None
    if isolated_matvec_incumbent_ms is not None and isolated_matvec_fold_ms is not None:
        matvec_ms = isolated_matvec_incumbent_ms - isolated_matvec_fold_ms
    reached = (
        organ_ms > 0
        and token_saved_ms > 0
        and token_saved_ms >= 0.5 * organ_ms
    )
    reproduced_projection = (
        token_saved_ms > 0 and abs(token_saved_ms - cited_projection_ms) <= 0.35
    )
    displaced_ms = organ_ms - token_saved_ms
    extra_ms = token_saved_ms - organ_ms
    if token_saved_ms <= 0 and organ_ms <= 0:
        where = (
            f"Neither the isolated MLP organ ({round(organ_ms, 4)} ms) nor the "
            f"complete token ({round(token_saved_ms, 4)} ms) reproduced the "
            f"cited {cited_projection_ms} ms one-layer projection. The probe "
            "did not survive integration; fold_addqx is not a token-level "
            "saving on this graph."
        )
    elif token_saved_ms <= 0:
        where = (
            f"Isolated MLP full saved {round(organ_ms, 4)} ms but the complete "
            f"token did not (incumbent {round(token_incumbent_ms, 4)} ms, "
            f"fold_addqx {round(token_fold_ms, 4)} ms, delta "
            f"{round(token_saved_ms, 4)} ms). The 1.745 ms projection did not "
            "survive integration; cost moved into the rest of the 580-graph."
        )
    elif not reached:
        where = (
            "The complete token kept only "
            f"{round(token_saved_ms, 4)} ms of the isolated MLP cut "
            f"{round(organ_ms, 4)} ms (displaced {round(displaced_ms, 4)} ms). "
            f"Cited projection {cited_projection_ms} ms."
        )
    elif not reproduced_projection:
        where = (
            "The organ-level decode saving reached the complete token "
            f"({round(token_saved_ms, 4)} ms vs isolated "
            f"{round(organ_ms, 4)} ms) but did not reproduce the cited "
            f"{cited_projection_ms} ms one-layer projection. "
        )
        if extra_ms > 0.1:
            where += (
                f"The extra {round(extra_ms, 4)} ms versus the isolated organ "
                "is integration, not the probe."
            )
        elif token_saved_ms > cited_projection_ms:
            where += (
                f"The complete token saved {round(token_saved_ms, 4)} ms, "
                f"{round(token_saved_ms - cited_projection_ms, 4)} ms more than "
                f"the {cited_projection_ms} ms 3-GEMV unfused one-layer "
                "projection. The fused GateUpSwiglu 580-graph is not the probe. "
                "That is the result this A/B exists to catch."
            )
        else:
            where += (
                "The missing "
                f"{round(cited_projection_ms - token_saved_ms, 4)} ms is the "
                "gap between a 3-GEMV unfused one-layer probe (192 launches "
                "scaled) and the fused GateUpSwiglu 580-graph. That is the "
                "result this A/B exists to catch."
            )
    else:
        where = (
            "The organ-level decode saving reached the complete token and "
            f"reproduced the cited {cited_projection_ms} ms projection "
            f"(complete token {round(token_saved_ms, 4)} ms, isolated "
            f"{round(organ_ms, 4)} ms)."
        )
    return {
        "cited_projection_ms": cited_projection_ms,
        "isolated_mlp_full_incumbent_ms": round(isolated_incumbent_ms, 4),
        "isolated_mlp_full_fold_addqx_ms": round(isolated_fold_ms, 4),
        "isolated_mlp_full_saving_ms": round(organ_ms, 4),
        "isolated_mlp_matvecs_saving_ms": None if matvec_ms is None else round(matvec_ms, 4),
        "complete_token_incumbent_ms": round(token_incumbent_ms, 4),
        "complete_token_fold_addqx_ms": round(token_fold_ms, 4),
        "complete_token_saving_ms": round(token_saved_ms, 4),
        "displaced_ms": round(displaced_ms, 4),
        "reached_the_token": reached,
        "reproduced_1p745_projection": reproduced_projection,
        "materiality_bar_ms": materiality_ms,
        "clears_materiality": bool(token_saved_ms >= materiality_ms),
        "where": where,
    }


def _run_ids(runs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    out: list[list[int]] = []
    for run in runs:
        ids = run.get("new_token_ids")
        if not isinstance(ids, list) or not ids:
            raise MissingArm("a decode run is missing new_token_ids")
        out.append(_as_int_list(ids) or [])
    return out


def _all_identical(series: Sequence[Sequence[int]]) -> bool:
    if not series:
        return False
    first = list(series[0])
    return all(list(s) == first for s in series)


def _fallbacks_sum(runs: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for run in runs:
        total += int(run.get("fallbacks") or 0)
    return total


def _require_kernel(runs: Sequence[Mapping[str, Any]], kernel: str, arm: str) -> int:
    counts = []
    for i, run in enumerate(runs):
        n = kernel_count(run.get("kernel_histogram") or [], kernel)
        if n <= 0:
            raise ProductionDidNotLaunch(
                f"{arm} run {i} did not dispatch {kernel}. Production did not "
                "launch the candidate; this is not a complete-token A/B of it."
            )
        counts.append(n)
    return min(counts)


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    iso = raw.get("isolated_mlp_full")
    decode = raw.get("decode")
    if not isinstance(iso, Mapping):
        raise MissingArm("raw is missing isolated_mlp_full")
    if not isinstance(decode, Mapping):
        raise MissingArm("raw is missing decode")
    inc_iso = iso.get("incumbent")
    fold_iso = iso.get("fold_addqx")
    if not isinstance(inc_iso, Mapping) or not isinstance(fold_iso, Mapping):
        raise MissingArm("isolated_mlp_full must carry incumbent and fold_addqx")
    inc_iso_ns = _require_gpu_ns(inc_iso, "isolated mlp_full incumbent")
    fold_iso_ns = _require_gpu_ns(fold_iso, "isolated mlp_full fold_addqx")

    matvec = raw.get("isolated_mlp_matvecs") if isinstance(raw.get("isolated_mlp_matvecs"), Mapping) else {}
    matvec_inc_ms = None
    matvec_fold_ms = None
    if isinstance(matvec, Mapping) and isinstance(matvec.get("incumbent"), Mapping) and isinstance(matvec.get("fold_addqx"), Mapping):
        try:
            matvec_inc_ms = ns_to_ms(_require_gpu_ns(matvec["incumbent"], "matvec incumbent"))
            matvec_fold_ms = ns_to_ms(_require_gpu_ns(matvec["fold_addqx"], "matvec fold_addqx"))
        except EmptyGpuSample:
            matvec_inc_ms = None
            matvec_fold_ms = None

    inc_runs = decode.get("incumbent")
    fold_runs = decode.get("fold_addqx")
    if not isinstance(inc_runs, list) or not inc_runs:
        raise MissingArm("decode.incumbent is empty")
    if not isinstance(fold_runs, list) or not fold_runs:
        raise MissingArm("decode.fold_addqx is empty")
    if len(inc_runs) != len(fold_runs):
        raise MissingArm(
            f"decode arms have different rep counts: incumbent {len(inc_runs)} "
            f"vs fold_addqx {len(fold_runs)}"
        )

    _require_kernel(inc_runs, INCUMBENT_SWIGLU, "incumbent")
    _require_kernel(fold_runs, FOLD_SWIGLU, "fold_addqx")
    # Candidate must not keep launching the production swiglu unpack.
    for i, run in enumerate(fold_runs):
        if kernel_count(run.get("kernel_histogram") or [], INCUMBENT_SWIGLU) > 0:
            raise ProductionDidNotLaunch(
                f"fold_addqx run {i} still dispatched {INCUMBENT_SWIGLU}; "
                "the candidate arm is not on fold_addqx"
            )

    inc_ids = _run_ids(inc_runs)
    fold_ids = _run_ids(fold_runs)
    fallbacks = _fallbacks_sum(inc_runs) + _fallbacks_sum(fold_runs)

    token_compare = None
    compares = decode.get("token_id_byte_compare")
    if isinstance(compares, list) and compares and isinstance(compares[0], Mapping):
        token_compare = compares[0]
    layer0 = raw.get("layer0_byte_compare")
    if not isinstance(layer0, Mapping):
        raise NoByteComparison(
            "raw is missing layer0_byte_compare; citing the probe is not identity"
        )

    if not _all_identical(inc_ids) or not _all_identical(fold_ids):
        parity = report_token_parity(
            incumbent_token_ids=inc_ids[0],
            candidate_token_ids=fold_ids[0],
            fallbacks=fallbacks,
            token_byte_compare=token_compare,
            layer0_byte_compare=layer0,
            runs_compared=len(inc_runs) + len(fold_runs),
        )
        parity["token_ids_identical"] = False
        parity["parity"] = False
        parity["within_arm_identical"] = {
            "incumbent": _all_identical(inc_ids),
            "fold_addqx": _all_identical(fold_ids),
        }
    else:
        parity = report_token_parity(
            incumbent_token_ids=inc_ids[0],
            candidate_token_ids=fold_ids[0],
            fallbacks=fallbacks,
            token_byte_compare=token_compare,
            layer0_byte_compare=layer0,
            runs_compared=len(inc_runs) + len(fold_runs),
        )
        parity["within_arm_identical"] = {"incumbent": True, "fold_addqx": True}

    inc_token_ns = decode.get("incumbent_complete_token_gpu_ns_median")
    fold_token_ns = decode.get("fold_addqx_complete_token_gpu_ns_median")
    if inc_token_ns is None:
        inc_token_ns = _median_u64(
            [int(r["complete_token_gpu_ns_median"]) for r in inc_runs if r.get("complete_token_gpu_ns_median")]
        )
    if fold_token_ns is None:
        fold_token_ns = _median_u64(
            [int(r["complete_token_gpu_ns_median"]) for r in fold_runs if r.get("complete_token_gpu_ns_median")]
        )
    if not inc_token_ns or not fold_token_ns:
        raise EmptyGpuSample("complete-token gpu_ns_median missing on an arm")

    inc_disp = inc_runs[0].get("complete_token_dispatches_last") or inc_runs[0].get(
        "theoretical_dispatches"
    )
    fold_disp = fold_runs[0].get("complete_token_dispatches_last") or fold_runs[0].get(
        "theoretical_dispatches"
    )

    organ = locate_saving(
        isolated_incumbent_ms=ns_to_ms(inc_iso_ns) or 0.0,
        isolated_fold_ms=ns_to_ms(fold_iso_ns) or 0.0,
        token_incumbent_ms=ns_to_ms(int(inc_token_ns)) or 0.0,
        token_fold_ms=ns_to_ms(int(fold_token_ns)) or 0.0,
        isolated_matvec_incumbent_ms=matvec_inc_ms,
        isolated_matvec_fold_ms=matvec_fold_ms,
    )

    faster_not_exact = (
        organ["complete_token_saving_ms"] > 0 and not parity.get("arithmetic_exact")
    )
    if faster_not_exact:
        organ = dict(organ)
        organ["where"] = (
            organ["where"]
            + " FASTER-BUT-NOT-EXACT: this saving is a different class and is "
            "not blended into the bit-identical verdict."
        )
        organ["faster_not_exact"] = True
        organ["class"] = "approx_candidate"
    else:
        organ["faster_not_exact"] = False
        organ["class"] = "exact_candidate" if parity.get("arithmetic_exact") else "unresolved"

    return {
        "raw": dict(raw),
        "isolated": {
            "mlp_full_incumbent_gpu_ns": inc_iso_ns,
            "mlp_full_fold_addqx_gpu_ns": fold_iso_ns,
            "mlp_full_incumbent_ms": ns_to_ms(inc_iso_ns),
            "mlp_full_fold_addqx_ms": ns_to_ms(fold_iso_ns),
            "mlp_full_incumbent_dispatches": inc_iso.get("dispatches"),
            "mlp_full_fold_addqx_dispatches": fold_iso.get("dispatches"),
            "matvecs": matvec,
        },
        "complete_token": {
            "incumbent_gpu_ns_median": int(inc_token_ns),
            "fold_addqx_gpu_ns_median": int(fold_token_ns),
            "incumbent_ms": ns_to_ms(int(inc_token_ns)),
            "fold_addqx_ms": ns_to_ms(int(fold_token_ns)),
            "incumbent_reps": decode.get("incumbent_complete_token_gpu_ns_median_reps"),
            "fold_addqx_reps": decode.get("fold_addqx_complete_token_gpu_ns_median_reps"),
            "incumbent_dispatches_last": inc_disp,
            "fold_addqx_dispatches_last": fold_disp,
            "incumbent_is_post_widen_f4_baseline": True,
            "stale_path_to_71_token_ms": 28.722,
            "cited_widen_f4_token_ms": WIDEN_F4_TOKEN_MS,
        },
        "parity": parity,
        "saving": organ,
        "launched": {
            "incumbent_swiglu": INCUMBENT_SWIGLU,
            "fold_addqx_swiglu": FOLD_SWIGLU,
            "incumbent_histogram_count": kernel_count(
                inc_runs[0].get("kernel_histogram") or [], INCUMBENT_SWIGLU
            ),
            "fold_addqx_histogram_count": kernel_count(
                fold_runs[0].get("kernel_histogram") or [], FOLD_SWIGLU
            ),
            "dn_state": DN_F4,
        },
        "layer0_byte_compare": layer0,
    }


def _finding(measured: Mapping[str, Any]) -> str:
    s = measured["saving"]
    p = measured["parity"]
    if p.get("parity"):
        parity_bit = "token-id identical, fallbacks 0, arithmetic byte-identical"
    elif not p.get("token_ids_identical"):
        parity_bit = "NOT token-identical"
    elif not p.get("arithmetic_exact"):
        parity_bit = "token ids match but arithmetic is not byte-identical"
    else:
        parity_bit = f"token ids match but fallbacks={p.get('fallbacks')}"
    cls = s.get("class")
    return (
        f"Complete-token A/B, post-widen_f4 incumbent vs fold_addqx, {parity_bit} "
        f"({cls}). Incumbent (new baseline) {s['complete_token_incumbent_ms']} ms vs "
        f"fold_addqx {s['complete_token_fold_addqx_ms']} ms "
        f"(saved {s['complete_token_saving_ms']} ms). Isolated MLP full "
        f"{s['isolated_mlp_full_incumbent_ms']} vs {s['isolated_mlp_full_fold_addqx_ms']} "
        f"(saved {s['isolated_mlp_full_saving_ms']} ms; cited projection "
        f"{s['cited_projection_ms']} ms). {s['where']} Materiality bar "
        f"{s['materiality_bar_ms']} ms: "
        f"{'clears' if s['clears_materiality'] else 'does not clear'}. "
        "SELF_MEASURED_DIRTY; no TPS QUALIFIED."
    )


def build(measured: Mapping[str, Any]) -> dict[str, Any]:
    raw = measured["raw"]
    s = measured["saving"]
    p = measured["parity"]
    if p.get("argmax_is_not_parity") is not True:
        raise ArgmaxIsNotParity("build refused a receipt that treats argmax as parity")
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/fold_addqx_token_ab.rs; "
            "production encode_mlp on sealed-3.14 post-widen_f4 580-graph; "
            "incumbent tpr64 vs HAWKING_AFFINE2_GEO=fold_addqx back-to-back "
            "in one process"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing"),
        "absolute_ms_are_measured_under_load": True,
        "concurrent_load_start": raw.get("concurrent_load_start"),
        "concurrent_load": raw.get("concurrent_load"),
        "session_open_s": raw.get("session_open_s"),
        "reps": raw.get("reps"),
        "warmup": raw.get("warmup"),
        "max_new_tokens": raw.get("max_new_tokens"),
        "dense_w_materialized": raw.get("dense_w_materialized", 0),
        "production_fusions": raw.get("production_fusions"),
        "sealed_env": SEALED_ENV,
        "lever": "HAWKING_AFFINE2_GEO=fold_addqx / apply_affine2_geo(FoldAddqx)",
        "lever_semantics": (
            "launches fold_addqx siblings of production geo_tpr64 on the real "
            "encode_mlp / fused gate_up_swiglu path. Default production stays "
            "tpr64 unpack8. Reversible: unset the lever."
        ),
        "production_kernel_swap": {
            "required": True,
            "why": (
                "fold_addqx lived only in decode_cheapen_mlp.metal as a probe. "
                "A complete-token A/B cannot launch a kernel production cannot "
                "bind. Minimal reversible sibling of geo_tpr64, same occupancy "
                "and binds, default off — same shape as widen_f4."
            ),
            "shaders": "crates/hawking-core/shaders/q80_mixed_decode.metal",
            "selector": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs Affine2Geo::FoldAddqx",
            "default_unchanged": True,
        },
        "cited_diagnostic": {
            "source": "receipts/future/MLP_DECODE_CHEAPEN.json",
            "fold_addqx_gb_s": CITED_PROBE_GB_S,
            "production_gb_s": CITED_PRODUCTION_GB_S,
            "arm_a_gb_s": CITED_ARM_A_GB_S,
            "ratio": CITED_RATIO,
            "projection_token_ms": CITED_PROJECTION_MS,
            "note": (
                "one-layer probe, 3 geo_tpr64 dispatches. Projection is "
                "arithmetic over MLP 15.541 ms of a 28.722 ms token, not a "
                "resident measurement. DIRTY_DIAGNOSTIC until this A/B."
            ),
        },
        "incumbent_is_post_widen_f4_baseline": True,
        "widen_f4_cited_saving_ms": WIDEN_F4_SAVING_MS,
        "widen_f4_cited_token_ms": WIDEN_F4_TOKEN_MS,
        "isolated_mlp": measured["isolated"],
        "complete_token": measured["complete_token"],
        "parity": p,
        "saving": s,
        "launched": measured["launched"],
        "layer0_byte_compare": measured.get("layer0_byte_compare"),
        "expected_dispatches": {
            "incumbent": SEALED_DISPATCHES,
            "fold_addqx": SEALED_DISPATCHES,
        },
        "tps_qualification": {
            "any_tps_labelled_qualified": False,
            "protected_window_required": True,
            "reason": (
                "SELF_MEASURED_DIRTY on a contaminated host qualifies nothing."
            ),
        },
        "finding": _finding(measured),
        "findings": [
            {
                "id": "PRODUCTION_LAUNCHED_FOLD_ADDQX",
                "what": (
                    f"candidate histogram counted {measured['launched']['fold_addqx_histogram_count']} "
                    f"{FOLD_SWIGLU} dispatches; incumbent counted "
                    f"{measured['launched']['incumbent_histogram_count']} "
                    f"{INCUMBENT_SWIGLU}"
                ),
                "why_it_matters": (
                    "the 1.127x diagnostic was a probe beside production; "
                    "this A/B is the production encode path"
                ),
            },
            {
                "id": "TOKEN_IDENTITY",
                "what": (
                    "token-id identical, fallbacks 0, arithmetic byte-identical"
                    if p.get("parity")
                    else "NOT bit-identical at the token; see parity"
                ),
                "parity": p.get("parity"),
                "token_ids_identical": p.get("token_ids_identical"),
                "fallbacks": p.get("fallbacks"),
                "arithmetic_exact": p.get("arithmetic_exact"),
                "argmax_is_not_parity": True,
                "cited_probe_is_not_the_identity_proof": True,
            },
            {
                "id": "INCUMBENT_IS_POST_WIDEN_F4_BASELINE",
                "what": (
                    f"incumbent complete-token {s['complete_token_incumbent_ms']} ms "
                    "is the new post-widen_f4 baseline. PATH_TO_71 28.722 ms is "
                    f"stale. DELTANET_WIDEN_AB cited {WIDEN_F4_TOKEN_MS} ms "
                    f"(saved {WIDEN_F4_SAVING_MS} ms) and nothing had re-profiled since."
                ),
            },
            {
                "id": "ORGAN_SAVING_VS_COMPLETE_TOKEN",
                "what": s["where"],
                "reached_the_token": s["reached_the_token"],
                "reproduced_1p745_projection": s["reproduced_1p745_projection"],
                "clears_materiality": s["clears_materiality"],
                "complete_token_saving_ms": s["complete_token_saving_ms"],
                "isolated_mlp_full_saving_ms": s["isolated_mlp_full_saving_ms"],
                "cited_projection_ms": s["cited_projection_ms"],
                "faster_not_exact": s.get("faster_not_exact"),
                "class": s.get("class"),
            },
        ],
        "bit_identical_is_non_negotiable": True,
        "gpu_lane_lock_waited_for": raw.get("gpu_lane_lock_waited_for"),
    }
    if doc["tps_qualification"]["any_tps_labelled_qualified"]:
        raise QualifiedTpsRefused("QUALIFIED TPS leaked")
    return doc


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise FoldAbRefuse("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("fold_addqx_token_ab",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "workspace" / "ops" / "build" / "rust",
            REPO / "target",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    tokenizer: Path,
    *,
    reps: int = 7,
    warmup: int = 1,
    max_new_tokens: int = 32,
    max_seq_len: int = 128,
    out: Path | None = None,
    binary: Path | None = None,
    use_lock: bool = True,
    lock_waited: list[str] | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "fold_addqx_token_ab binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core "
            "--example fold_addqx_token_ab`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(tokenizer),
        "--reps",
        str(reps),
        "--warmup",
        str(warmup),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--out",
        str(out),
    ]
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    from tools.future._common import gpu_lane_lock_path
    lock_path = gpu_lane_lock_path()
    if lock_path.exists() and lock_waited is not None:
        owner = ""
        try:
            owner = (lock_path / "owner").read_text().strip()
        except OSError:
            owner = "unknown"
        lock_waited.append(owner)
    cmd = ["bash", str(lock), "g1address", *inner] if use_lock and lock.is_file() else inner
    env = os.environ.copy()
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    waited: list[str] = []
    if args.measure:
        raw = run_example(
            args.artifact_root,
            args.tokenizer,
            reps=args.reps,
            warmup=args.warmup,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            out=RAW_DEFAULT,
            use_lock=not args.no_lock,
            lock_waited=waited,
        )
        if waited:
            raw["gpu_lane_lock_waited_for"] = waited
    elif args.raw_path:
        raw = load_raw(Path(args.raw_path))
    elif RAW_DEFAULT.is_file():
        raw = load_raw(RAW_DEFAULT)

    if raw is None:
        print(
            "no measurement: pass --from RAW.json, --measure, or write "
            f"{RAW_DEFAULT}",
            file=sys.stderr,
        )
        return 2

    measured = measurement_from_raw(raw)
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path}")
        print(measured["saving"]["where"])
        print(build(measured)["finding"])
    else:
        s = measured["saving"]
        p = measured["parity"]
        print(
            f"complete token incumbent {s['complete_token_incumbent_ms']} ms  "
            f"fold_addqx {s['complete_token_fold_addqx_ms']} ms  "
            f"saved {s['complete_token_saving_ms']} ms"
        )
        print(
            f"isolated MLP full {s['isolated_mlp_full_saving_ms']} ms  "
            f"parity={p['parity']} fallbacks={p['fallbacks']} "
            f"arithmetic_exact={p['arithmetic_exact']}"
        )
        print(s["where"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
