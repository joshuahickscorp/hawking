#!/usr/bin/env python3
"""Pack L00 Qwen3.8 organs for the in-register reconstruction discriminator.

Uses REAL BF16 weights and REAL captured activations. Writes raw bodies the
Metal harness consumes (no JSON envelopes). Does not use the GPU.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    GROUP_BINARY,
    GROUP_UNIFORM,
    _additive_residual_codec,
    _binary_codec,
    _binary_parts,
    _hadamard_lattice_codec,
    _parse_container,
    _ternary_codec,
    _uniform_codec,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402
from lab.operators.residual_compact_codec import (  # noqa: E402
    MAGIC_RESIDUAL_COMPACT,
    encode_residual_compact,
    select_outlier_indices,
)

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen38-27b/bf16"
CAPTURE_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
DEFAULT_OUT = REPO_ROOT / "scratch/qwen38_recon"

HIDDEN = 5120
N_TOKENS = 256
FIT_N = 192
HOLD_N = 64


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_f32(path: Path, arr: np.ndarray) -> None:
    write_bytes(path, np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def write_u32(path: Path, arr: np.ndarray) -> None:
    write_bytes(path, np.ascontiguousarray(arr, dtype="<u4").tobytes())


def load_hidden(layer: int) -> np.ndarray:
    path = CAPTURE_DIR / "hidden" / f"L{layer:02d}.f32"
    x = np.fromfile(path, dtype=np.float32)
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} has {x.size} floats")
    return np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN), dtype=np.float32)


def split_uniform(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVU01")
    scale_bytes = int(header["scale_bytes"])
    return body[:scale_bytes], body[scale_bytes:], header


def split_binary(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVB01")
    scale_bytes = int(header["scale_bytes"])
    return body[:scale_bytes], body[scale_bytes:], header


def split_ternary(payload: bytes) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVT01")
    thr = int(header["threshold_bytes"])
    sc = int(header["scale_bytes"])
    return body[:thr], body[thr : thr + sc], body[thr + sc :], header


def split_hadamard(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVH01")
    scale_bytes = int(header["scale_bytes"])
    return body[:scale_bytes], body[scale_bytes:], header


def split_additive(payload: bytes) -> dict[str, Any]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVA01")
    bsb = int(header["base_scale_bytes"])
    rsb = int(header["residual_scale_bytes"])
    bcb = int(header["base_code_bytes"])
    cursor = 0
    out = {"header": header}
    out["base_scales"] = body[cursor : cursor + bsb]
    cursor += bsb
    out["residual_scales"] = body[cursor : cursor + rsb]
    cursor += rsb
    out["base_codes"] = body[cursor : cursor + bcb]
    cursor += bcb
    out["residual_codes"] = body[cursor:]
    return out


def pack_production_q4(W: np.ndarray) -> tuple[bytes, bytes, np.ndarray]:
    """Nibble layout matching qwen_uniform_q4_group64_matvec_geo_tpr64_tg128."""
    rows, cols = W.shape
    assert cols % 64 == 0
    groups_per_row = cols // 64
    groups = rows * groups_per_row
    w = W.reshape(groups, 64)
    bound = 8.0
    scales = (np.max(np.abs(w), axis=1) / bound).astype("<f2")
    den = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    q = np.rint(w / den[:, None]).clip(-8, 7).astype(np.int16)
    nib = (q + 8).astype(np.uint8)
    codes = np.empty((groups, 32), dtype=np.uint8)
    codes[:, :] = nib[:, 0::2] | (nib[:, 1::2] << 4)
    rebuilt = (q.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(rows, cols)
    return scales.tobytes(), codes.tobytes(), rebuilt


def orthonormal_hadamard_vec(x: np.ndarray, group: int = 128) -> np.ndarray:
    work = np.array(x, dtype=np.float32, copy=True, order="C")
    n = work.size
    assert n % group == 0
    g = work.reshape(-1, group)
    width = group
    stride = 1
    while stride < width:
        view = g.reshape(g.shape[0], width // (2 * stride), 2, stride)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2
    g /= np.float32(np.sqrt(width))
    return g.reshape(-1)


def pack_organ(
    dest: Path,
    name: str,
    W: np.ndarray,
    X_token: np.ndarray,
    *,
    do_twostage: bool,
    X_fit: np.ndarray | None = None,
) -> dict[str, Any]:
    rows, cols = int(W.shape[0]), int(W.shape[1])
    elements = int(W.size)
    dest.mkdir(parents=True, exist_ok=True)
    write_f32(dest / "x.f32", X_token)
    y_ref = W @ X_token
    write_f32(dest / "y_ref_dense.f32", y_ref)

    rec: dict[str, Any] = {
        "name": name,
        "rows": rows,
        "cols": cols,
        "elements": elements,
        "x": str(dest / "x.f32"),
        "codecs": {},
    }

    def add(codec: str, meta: dict[str, Any], y_hat: np.ndarray, payload_bytes: int) -> None:
        write_f32(dest / f"{codec}.y_hat.f32", y_hat)
        meta["payload_bytes"] = int(payload_bytes)
        meta["storage_bpw"] = 8.0 * payload_bytes / elements
        num = float(y_ref @ y_hat)
        den = float(np.linalg.norm(y_ref) * np.linalg.norm(y_hat))
        meta["token_output_cosine"] = 1.0 if den <= 1e-12 else num / den
        rec["codecs"][codec] = meta
        print(
            f"  {name:4s} {codec:22s} bpw={meta['storage_bpw']:.4f} "
            f"cos={meta['token_output_cosine']:.4f} bytes={payload_bytes}",
            flush=True,
        )

    # production nibble q4
    t0 = time.perf_counter()
    sc, codes, hat = pack_production_q4(W)
    write_bytes(dest / "prod_q4.scales.f16", sc)
    write_bytes(dest / "prod_q4.codes.bin", codes)
    add(
        "prod_q4_nibble_g64",
        {
            "kind": "uniform_nibble",
            "bits": 4,
            "group_size": 64,
            "scales": str(dest / "prod_q4.scales.f16"),
            "codes": str(dest / "prod_q4.codes.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        hat @ X_token,
        len(sc) + len(codes),
    )

    t0 = time.perf_counter()
    u4 = _uniform_codec(W, bits=4, group_size=GROUP_UNIFORM)
    sc, codes, hdr = split_uniform(u4.payload)
    write_bytes(dest / "u4.scales.f16", sc)
    write_bytes(dest / "u4.codes.bin", codes)
    add(
        "uniform_q4_g64",
        {
            "kind": "uniform_bits",
            "bits": 4,
            "bound": 7,
            "group_size": 64,
            "scales": str(dest / "u4.scales.f16"),
            "codes": str(dest / "u4.codes.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        u4.reconstruction @ X_token,
        len(u4.payload),
    )

    t0 = time.perf_counter()
    u3 = _uniform_codec(W, bits=3, group_size=GROUP_UNIFORM)
    sc, codes, hdr = split_uniform(u3.payload)
    write_bytes(dest / "u3.scales.f16", sc)
    write_bytes(dest / "u3.codes.bin", codes)
    add(
        "uniform_q3_g64",
        {
            "kind": "uniform_bits",
            "bits": 3,
            "bound": 3,
            "group_size": 64,
            "scales": str(dest / "u3.scales.f16"),
            "codes": str(dest / "u3.codes.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        u3.reconstruction @ X_token,
        len(u3.payload),
    )

    t0 = time.perf_counter()
    u2 = _uniform_codec(W, bits=2, group_size=GROUP_UNIFORM)
    sc, codes, hdr = split_uniform(u2.payload)
    write_bytes(dest / "u2.scales.f16", sc)
    write_bytes(dest / "u2.codes.bin", codes)
    add(
        "uniform_q2_g64",
        {
            "kind": "uniform_bits",
            "bits": 2,
            "bound": 1,
            "group_size": 64,
            "scales": str(dest / "u2.scales.f16"),
            "codes": str(dest / "u2.codes.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        u2.reconstruction @ X_token,
        len(u2.payload),
    )

    t0 = time.perf_counter()
    bn = _binary_codec(W, group_size=GROUP_BINARY)
    sc, signs, hdr = split_binary(bn.payload)
    write_bytes(dest / "bin.scales.f16", sc)
    write_bytes(dest / "bin.signs.bin", signs)
    add(
        "binary_g128",
        {
            "kind": "binary",
            "group_size": 128,
            "scales": str(dest / "bin.scales.f16"),
            "signs": str(dest / "bin.signs.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        bn.reconstruction @ X_token,
        len(bn.payload),
    )

    t0 = time.perf_counter()
    tr = _ternary_codec(W, threshold_multiplier=0.7, group_size=GROUP_BINARY)
    _thr, sc, codes, hdr = split_ternary(tr.payload)
    write_bytes(dest / "ter.scales.f16", sc)
    write_bytes(dest / "ter.codes.bin", codes)
    add(
        "ternary_t0.7_g128",
        {
            "kind": "ternary",
            "group_size": 128,
            "bits": 2,
            "scales": str(dest / "ter.scales.f16"),
            "codes": str(dest / "ter.codes.bin"),
            "encode_s": time.perf_counter() - t0,
            "note": "threshold is encode-only; decode is 2-bit {0,+s,-s}",
        },
        tr.reconstruction @ X_token,
        len(tr.payload),
    )

    t0 = time.perf_counter()
    hd = _hadamard_lattice_codec(W, bits=2, group_size=GROUP_BINARY)
    sc, codes, hdr = split_hadamard(hd.payload)
    write_bytes(dest / "had.scales.f16", sc)
    write_bytes(dest / "had.codes.bin", codes)
    x_h = orthonormal_hadamard_vec(X_token, 128)
    write_f32(dest / "x_hadamard.f32", x_h)
    add(
        "hadamard_q2_g128",
        {
            "kind": "hadamard_q2",
            "bits": 2,
            "bound": 1,
            "group_size": 128,
            "scales": str(dest / "had.scales.f16"),
            "codes": str(dest / "had.codes.bin"),
            "x_transformed": str(dest / "x_hadamard.f32"),
            "encode_s": time.perf_counter() - t0,
            "note": "self-inverse WH: y = Q @ (H x). WH(x) is per-token and timed.",
        },
        hd.reconstruction @ X_token,
        len(hd.payload),
    )

    t0 = time.perf_counter()
    ad = _additive_residual_codec(W, group_size=GROUP_UNIFORM)
    parts = split_additive(ad.payload)
    write_bytes(dest / "add.base_scales.f16", parts["base_scales"])
    write_bytes(dest / "add.res_scales.f16", parts["residual_scales"])
    write_bytes(dest / "add.base_codes.bin", parts["base_codes"])
    write_bytes(dest / "add.res_codes.bin", parts["residual_codes"])
    add(
        "additive_q2q2_g64",
        {
            "kind": "additive_q2q2",
            "group_size": 64,
            "base_scales": str(dest / "add.base_scales.f16"),
            "residual_scales": str(dest / "add.res_scales.f16"),
            "base_codes": str(dest / "add.base_codes.bin"),
            "residual_codes": str(dest / "add.res_codes.bin"),
            "encode_s": time.perf_counter() - t0,
        },
        ad.reconstruction @ X_token,
        len(ad.payload),
    )

    # rice = binary + 2% compact residual. Expand to CSR for Q80-won consume.
    t0 = time.perf_counter()
    rice = encode_residual_compact(
        W,
        outlier_ratio=0.02,
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    header, body = _parse_container(rice.payload, expected_magic=MAGIC_RESIDUAL_COMPACT)
    scale_bytes = int(header["scale_bytes"])
    sign_bytes = int(header["sign_bytes"])
    rice_scales = body[:scale_bytes]
    rice_signs = body[scale_bytes : scale_bytes + sign_bytes]
    write_bytes(dest / "rice.bin_scales.f16", rice_scales)
    write_bytes(dest / "rice.bin_signs.bin", rice_signs)

    _, _sc, _sg, base = _binary_parts(W, group_size=GROUP_BINARY)
    residual = W.reshape(-1) - np.ascontiguousarray(base, dtype=np.float32).reshape(-1)
    indices, count = select_outlier_indices(residual, 0.02)
    cols_u = cols
    rows_of = (indices.astype(np.int64) // cols_u).astype(np.int64)
    cols_of = (indices.astype(np.int64) % cols_u).astype(np.uint32)
    counts = np.bincount(rows_of, minlength=rows).astype(np.int64)
    row_ptr = np.zeros(rows + 1, dtype=np.uint32)
    np.cumsum(counts, out=row_ptr[1:])
    # CSR col indices in row-major outlier order (already sorted globally => per-row sorted)
    write_u32(dest / "rice.csr_cols.u32", cols_of)
    write_u32(dest / "rice.csr_row_ptr.u32", row_ptr)
    # 1-bit residual signs at outlier positions, same order as indices
    rsigns = (residual[indices.astype(np.int64)] >= 0.0).astype(np.uint8)
    rsign_bytes = np.packbits(rsigns, bitorder="little").tobytes()
    write_bytes(dest / "rice.csr_signs.bin", rsign_bytes)
    # rms scale is 2 bytes after the rice index blob
    index_bytes = int(header["index_bytes"])
    rscale_off = scale_bytes + sign_bytes + index_bytes
    rscale_n = int(header["residual_scale_bytes"])
    rscale = body[rscale_off : rscale_off + rscale_n]
    write_bytes(dest / "rice.csr_scale.f16", rscale)
    write_bytes(dest / "rice.stream.bin", rice.payload)
    csr_traffic = (
        len(rice_scales)
        + len(rice_signs)
        + cols_of.nbytes
        + row_ptr.nbytes
        + len(rsign_bytes)
        + len(rscale)
    )
    add(
        "rice_q1_rms_2pct",
        {
            "kind": "binary_csr",
            "group_size": 128,
            "scales": str(dest / "rice.bin_scales.f16"),
            "signs": str(dest / "rice.bin_signs.bin"),
            "csr_cols": str(dest / "rice.csr_cols.u32"),
            "csr_row_ptr": str(dest / "rice.csr_row_ptr.u32"),
            "csr_signs": str(dest / "rice.csr_signs.bin"),
            "csr_scale": str(dest / "rice.csr_scale.f16"),
            "rice_payload": str(dest / "rice.stream.bin"),
            "outlier_count": int(count),
            "rice_k": int(header.get("rice_k", -1)),
            "csr_traffic_bytes": int(csr_traffic),
            "storage_payload_bytes": len(rice.payload),
            "encode_s": time.perf_counter() - t0,
            "note": "storage is rice bitstream; Q80-won consume is bind-expanded CSR",
        },
        rice.reconstruction @ X_token,
        len(rice.payload),
    )
    rec["codecs"]["rice_q1_rms_2pct"]["csr_traffic_bpw"] = 8.0 * csr_traffic / elements

    if do_twostage:
        if X_fit is None:
            raise RuntimeError("two-stage needs real post-SwiGLU X_fit")
        t0 = time.perf_counter()
        rank = 160
        # Activation-subspace factors on REAL X (QR of 160 captured tokens).
        # Cost geometry matches HGRAVS01 r160. Quality is reported but the
        # sibling pack owns the production SVD fit.
        A = np.ascontiguousarray(X_fit[:rank].T, dtype=np.float64)  # cols x rank
        q, _r = np.linalg.qr(A, mode="reduced")
        q = np.ascontiguousarray(q, dtype=np.float32)
        left = np.ascontiguousarray(W @ q, dtype=np.float32)  # rows x rank
        right = np.ascontiguousarray(q.T, dtype=np.float32)  # rank x cols
        lf = _uniform_codec(left, bits=3, group_size=GROUP_UNIFORM)
        rf = _uniform_codec(right, bits=3, group_size=GROUP_UNIFORM)
        lsc, lco, _ = split_uniform(lf.payload)
        rsc, rco, _ = split_uniform(rf.payload)
        write_bytes(dest / "ts.left_scales.f16", lsc)
        write_bytes(dest / "ts.left_codes.bin", lco)
        write_bytes(dest / "ts.right_scales.f16", rsc)
        write_bytes(dest / "ts.right_codes.bin", rco)
        hat = lf.reconstruction @ rf.reconstruction
        payload_bytes = len(lf.payload) + len(rf.payload)
        add(
            "hgravs01_r160_q3",
            {
                "kind": "two_stage",
                "rank": rank,
                "left_rows": rows,
                "left_cols": rank,
                "right_rows": rank,
                "right_cols": cols,
                "bits": 3,
                "bound": 3,
                "group_size": 64,
                "left_scales": str(dest / "ts.left_scales.f16"),
                "left_codes": str(dest / "ts.left_codes.bin"),
                "right_scales": str(dest / "ts.right_scales.f16"),
                "right_codes": str(dest / "ts.right_codes.bin"),
                "fit": "qr_activation_subspace_160_real_X",
                "encode_s": time.perf_counter() - t0,
                "note": "L@(R@x). Factors from real post-SwiGLU X QR, not the sibling SVD.",
            },
            hat @ X_token,
            payload_bytes,
        )

    # f32 control lives as a view of W; harness can optionally skip upload.
    rec["f32_bytes"] = elements * 4
    write_f32(dest / "W.f32", W)
    rec["W"] = str(dest / "W.f32")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    capture_meta = json.loads((CAPTURE_DIR / "capture-result.json").read_text())
    if capture_meta.get("status") != "CAPTURED_REAL_BF16_POST_NORM_HIDDEN":
        raise SystemExit(f"capture is not real: {capture_meta.get('status')}")
    if capture_meta.get("source", {}).get("not_synthetic") is not True:
        raise SystemExit("capture claims synthetic — refuse")

    weight_map = load_weight_map(MODEL_DIR)
    layer = 0
    prefix = f"language_model.model.layers.{layer}."
    X = load_hidden(layer)
    # Holdout token: first token after the 192-token fit split.
    x_h = np.ascontiguousarray(X[FIT_N], dtype=np.float32)

    print("loading L00 gate/up/down …", flush=True)
    gate = np.ascontiguousarray(
        load_tensor(MODEL_DIR, weight_map, prefix + "mlp.gate_proj.weight"),
        dtype=np.float32,
    )
    up = np.ascontiguousarray(
        load_tensor(MODEL_DIR, weight_map, prefix + "mlp.up_proj.weight"),
        dtype=np.float32,
    )
    print(f"  gate {gate.shape} up {up.shape}", flush=True)
    x_swiglu_all = silu(X @ gate.T) * (X @ up.T)
    x_swiglu = np.ascontiguousarray(x_swiglu_all[FIT_N], dtype=np.float32)
    x_swiglu_fit = np.ascontiguousarray(x_swiglu_all[:FIT_N], dtype=np.float32)
    del x_swiglu_all, up

    t0 = time.perf_counter()
    gate_rec = pack_organ(out / "gate", "gate", gate, x_h, do_twostage=False)
    del gate
    print(f"gate packed in {time.perf_counter()-t0:.1f}s", flush=True)

    down = np.ascontiguousarray(
        load_tensor(MODEL_DIR, weight_map, prefix + "mlp.down_proj.weight"),
        dtype=np.float32,
    )
    print(f"  down {down.shape}", flush=True)
    t0 = time.perf_counter()
    down_rec = pack_organ(
        out / "down",
        "down",
        down,
        x_swiglu,
        do_twostage=True,
        X_fit=x_swiglu_fit,
    )
    del down
    print(f"down packed in {time.perf_counter()-t0:.1f}s", flush=True)

    man = {
        "schema": "hawking.special_unit.qwen38_recon_pack.v1",
        "date": "2026-08-16",
        "activation": {
            "path": str(CAPTURE_DIR),
            "status": capture_meta.get("status"),
            "sha256_self": capture_meta.get("sha256_self"),
            "not_synthetic": True,
            "token_index": FIT_N,
            "note": "first holdout token of the 256-token real capture",
        },
        "organs": {"gate": gate_rec, "down": down_rec},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
