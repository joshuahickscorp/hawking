#!/usr/bin/env python3
"""CPU pack probe for Q80 lm_head + embed. No GPU. No remote.

Measures whether the terminal tensors can be compressed, and whether a cheap
draft + exact top-k rescore preserves greedy token identity.

Hidden source is the bound 25258-token capture's L47 *router-input* rows
(post-mixer residual entering the last MoE). Final-norm + lm_head were not
executed during that capture; every quality number is labeled as such.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

VOCAB = 151_936
HIDDEN = 2048
TOKENIZER_VOCAB = 151_669
RESERVED_TAIL = VOCAB - TOKENIZER_VOCAB
GROUP_UNIFORM = 64
GROUP_BINARY = 128
RMS_EPS = 1.0e-6

MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
)
CAPTURE_HIDDEN = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-diagnostics/source-bf16-capture-n192-scale64/hidden"
)
CANDIDATES = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates"
)

# Measured Q4-vehicle terminal CB (receipts/QWEN80_TOKEN_NS_LEDGER.json):
# 16 tokens, gpu_ns median 1_253_249 for 165_306_368 bytes.
Q4_LM_HEAD_GPU_NS_MEDIAN = 1_253_249
Q4_LM_HEAD_BYTES = 165_306_368
Q4_LM_HEAD_GB_S = Q4_LM_HEAD_BYTES / (Q4_LM_HEAD_GPU_NS_MEDIAN * 1e-9)

# Mixed reconstruction wall (receipts/ascent-2026-08-16/Q80_MIXED_RECONSTRUCTION_WALL.json)
MIXED_GB_S = 2.57
Q4_VEHICLE_GB_S = 15.2


def physical_uniform_bytes(n_elem: int, bits: int, group_size: int = GROUP_UNIFORM) -> int:
    groups = math.ceil(n_elem / group_size)
    code = math.ceil(groups * group_size * bits / 8)
    scales = groups * 2
    return scales + code


def physical_binary_bytes(n_elem: int, group_size: int = GROUP_BINARY) -> int:
    groups = math.ceil(n_elem / group_size)
    return groups * 2 + math.ceil(n_elem / 8)


def uniform_hat(w: np.ndarray, bits: int, group_size: int = GROUP_UNIFORM) -> np.ndarray:
    """Symmetric group absmax RTN. Matches HGRAVU01 / _uniform_codec."""
    flat = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    pad = groups * group_size - flat.size
    padded = np.pad(flat, (0, pad), constant_values=0.0).reshape(groups, group_size)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(padded), axis=1) / max(bound, 1)).astype(np.float32)
    denom = np.where(scales > 0.0, scales, 1.0)
    signed = np.rint(padded / denom[:, None]).clip(-bound, bound)
    hat = (signed * scales[:, None]).reshape(-1)[: flat.size]
    return np.ascontiguousarray(hat.reshape(w.shape), dtype=np.float32)


def binary_hat(w: np.ndarray, group_size: int = GROUP_BINARY) -> np.ndarray:
    flat = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    pad = groups * group_size - flat.size
    padded = np.pad(flat, (0, pad), constant_values=0.0).reshape(groups, group_size)
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float32)
    signs = np.where(padded >= 0.0, 1.0, -1.0).astype(np.float32)
    hat = (signs * scales[:, None]).reshape(-1)[: flat.size]
    return np.ascontiguousarray(hat.reshape(w.shape), dtype=np.float32)


def read_safetensors_header(shard: Path) -> dict[str, Any]:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def load_bf16_tensor(model_dir: Path, name: str) -> np.ndarray:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = model_dir / idx["weight_map"][name]
    header = read_safetensors_header(shard)
    info = header[name]
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + lo)
        raw = fh.read(hi - lo)
    dtype = info.get("dtype", "BF16")
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.ascontiguousarray(u32.view(np.float32).reshape(shape))
    if dtype in ("F16", "FLOAT16"):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(var + eps)) * weight


def load_l47_hiddens(n: int, seed: int) -> tuple[np.ndarray, list[str]]:
    files = sorted(CAPTURE_HIDDEN.glob("L47/*/*.f32le"))
    if not files:
        raise FileNotFoundError(f"no L47 hidden files under {CAPTURE_HIDDEN}")
    rng = np.random.default_rng(seed)
    # Stratify by probe directory so one prompt family cannot dominate.
    by_probe: dict[str, list[Path]] = {}
    for f in files:
        by_probe.setdefault(f.parent.name, []).append(f)
    probes = sorted(by_probe)
    picked: list[Path] = []
    # Round-robin one from each probe, then fill.
    for probe in probes:
        cand = by_probe[probe]
        picked.append(cand[int(rng.integers(0, len(cand)))])
        if len(picked) >= n:
            break
    if len(picked) < n:
        rest = [f for f in files if f not in picked]
        extra = rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False)
        picked.extend(Path(p) for p in extra)
    picked = picked[:n]
    rows = np.empty((len(picked), HIDDEN), dtype=np.float32)
    labels = []
    for i, p in enumerate(picked):
        raw = np.fromfile(p, dtype="<f4")
        if raw.size != HIDDEN:
            raise RuntimeError(f"{p} has {raw.size} floats, expected {HIDDEN}")
        rows[i] = raw
        labels.append(f"{p.parent.name}/{p.name}")
    return rows, labels


def cosine(a: np.ndarray, b: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if axis is None:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a.ravel(), b.ravel()) / (na * nb))
    na = np.linalg.norm(a, axis=axis)
    nb = np.linalg.norm(b, axis=axis)
    den = np.where((na > 0) & (nb > 0), na * nb, 1.0)
    return np.sum(a * b, axis=axis) / den


def mask_reserved(logits: np.ndarray) -> np.ndarray:
    out = np.array(logits, copy=True)
    out[..., TOKENIZER_VOCAB:] = -np.inf
    return out


def greedy(logits: np.ndarray) -> np.ndarray:
    return np.argmax(mask_reserved(logits), axis=-1)


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    masked = mask_reserved(logits)
    # argpartition then sort the k
    idx = np.argpartition(masked, -k, axis=-1)[..., -k:]
    part = np.take_along_axis(masked, idx, axis=-1)
    order = np.argsort(-part, axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def score_logits(ref: np.ndarray, hat: np.ndarray) -> dict[str, Any]:
    n = ref.shape[0]
    ref_g = greedy(ref)
    hat_g = greedy(hat)
    match = ref_g == hat_g
    ref_top2 = topk_indices(ref, 2)
    hat_at_ref = hat[np.arange(n), ref_g]
    ref_at_ref = ref[np.arange(n), ref_g]
    ref_second = ref[np.arange(n), ref_top2[:, 1]]
    margin = ref_at_ref - ref_second
    err_at_winner = np.abs(hat_at_ref - ref_at_ref)
    logit_cos = cosine(ref, hat, axis=-1)
    # Finite entries only for RMSE (reserved tail is -inf after mask; use raw).
    raw_err = hat - ref
    rmse = float(np.sqrt(np.mean(raw_err.astype(np.float64) ** 2)))
    return {
        "n": int(n),
        "greedy_match": int(match.sum()),
        "greedy_match_rate": float(match.mean()),
        "greedy_mismatch": int((~match).sum()),
        "mean_logit_cosine": float(np.mean(logit_cos)),
        "min_logit_cosine": float(np.min(logit_cos)),
        "p10_logit_cosine": float(np.percentile(logit_cos, 10)),
        "logit_rmse": rmse,
        "mean_abs_err_at_ref_winner": float(np.mean(err_at_winner)),
        "mean_ref_top1_top2_margin": float(np.mean(margin)),
        "min_ref_top1_top2_margin": float(np.min(margin)),
        "frac_err_ge_margin": float(np.mean(err_at_winner >= margin)),
        "mismatched_ids": [
            {"i": int(i), "ref": int(ref_g[i]), "hat": int(hat_g[i]), "margin": float(margin[i])}
            for i in np.where(~match)[0][:16]
        ],
    }


def coverage_table(ref: np.ndarray, draft: np.ndarray, ks: list[int]) -> dict[str, Any]:
    ref_g = greedy(ref)
    n = ref.shape[0]
    out = {}
    for k in ks:
        idx = topk_indices(draft, k)
        hit = np.any(idx == ref_g[:, None], axis=1)
        out[str(k)] = {
            "k": k,
            "hit": int(hit.sum()),
            "hit_rate": float(hit.mean()),
            "miss": int((~hit).sum()),
        }
    return out


def randomized_svd(w: np.ndarray, rank: int, n_oversample: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = rank + n_oversample
    omega = rng.standard_normal((w.shape[1], k), dtype=np.float32)
    y = w @ omega
    # one power iteration for a stabler basis
    y = w @ (w.T @ y)
    q, _ = np.linalg.qr(y.astype(np.float64), mode="reduced")
    q = q.astype(np.float32)
    b = q.T @ w
    u_hat, s, vt = np.linalg.svd(b.astype(np.float64), full_matrices=False)
    u = (q.astype(np.float64) @ u_hat)[:, :rank]
    return u.astype(np.float32), s[:rank].astype(np.float32), vt[:rank].astype(np.float32)


def svd_logits(u: np.ndarray, s: np.ndarray, vt: np.ndarray, h: np.ndarray, rank: int) -> np.ndarray:
    # logits = (U[:,:r] * S[:r]) @ (V[:r] @ h)
    mid = vt[:rank] @ h.T  # r x n
    return (u[:, :rank] * s[:rank]) @ mid  # V x n  -> we want n x V
    # wait: (V,r) @ (r,n) = (V,n). Caller wants (n,V).


def svd_logits_nV(u: np.ndarray, s: np.ndarray, vt: np.ndarray, h: np.ndarray, rank: int) -> np.ndarray:
    mid = vt[:rank] @ h.T
    return ((u[:, :rank] * s[:rank]) @ mid).T


def estimate_recon_ns(bytes_: int, class_name: str) -> dict[str, Any]:
    """Cost-class reconstruction ns. Not a GPU claim.

    uniform_q4_q8: scale from the measured Q4 lm_head terminal CB (132 GB/s).
    mixed_scatter: scale from the mixed-vehicle wall (2.57 GB/s).
    two_matvec / prefix / gather: arithmetic from the same Q4 sequential rate
    (these are sequential GEMV / prefix reads, not rice/scatter).
    """
    if class_name in {"uniform_dequant_gemv", "two_matvec", "prefix_then_gather", "row_gather"}:
        gb_s = Q4_LM_HEAD_GB_S
    elif class_name == "mixed_scatter_or_rice":
        gb_s = MIXED_GB_S
    else:
        gb_s = Q4_VEHICLE_GB_S
    ns = bytes_ / gb_s * 1e9
    return {
        "class": class_name,
        "assumed_gb_s": gb_s,
        "bytes": int(bytes_),
        "recon_ns": ns,
        "label": "DIRTY_ENGINEERING_SCALED_FROM_MEASURED_RATES",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hidden", type=int, default=384)
    ap.add_argument("--seed", type=int, default=80)
    ap.add_argument("--svd-rank", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "hawking.ascent.q80_terminal_head_pack_probe.v1",
        "date": "2026-08-16",
        "lane": "q80-terminal-head",
        "gpu_used": False,
        "hidden_caveat": (
            "L47 router-input rows from the 25258-token source-bf16 capture. "
            "final_norm_lm_head_sampler_not_executed_during_capture=true. "
            "These are late-layer residuals BEFORE the last MoE add, then "
            "RMSNorm'd with model.norm.weight. Not the exact pre-lm_head hidden."
        ),
        "reserved_tail_rows_masked": RESERVED_TAIL,
        "tokenizer_vocab": TOKENIZER_VOCAB,
        "vocab": VOCAB,
        "hidden": HIDDEN,
    }

    print("[probe] loading L47 hiddens", flush=True)
    raw_h, labels = load_l47_hiddens(args.n_hidden, args.seed)
    print(f"[probe] loaded {raw_h.shape[0]} L47 rows from {len({l.split('/')[0] for l in labels})} probes", flush=True)

    print("[probe] loading model.norm.weight + lm_head.weight", flush=True)
    t_load = time.perf_counter()
    norm_w = load_bf16_tensor(MODEL_DIR, "model.norm.weight")
    w = load_bf16_tensor(MODEL_DIR, "lm_head.weight")
    print(
        f"[probe] lm_head {w.shape} {w.dtype} {w.nbytes} bytes  load_s={time.perf_counter()-t_load:.1f}",
        flush=True,
    )
    if w.shape != (VOCAB, HIDDEN):
        raise RuntimeError(f"lm_head shape {w.shape}")
    h = rmsnorm(raw_h, norm_w)
    report["n_hidden"] = int(h.shape[0])
    report["hidden_rms_mean"] = float(np.sqrt(np.mean(h * h)))
    report["hidden_probes"] = sorted({l.split("/")[0] for l in labels})
    report["load_lm_head_s"] = time.perf_counter() - t_load

    print("[probe] reference logits W @ h", flush=True)
    t_ref = time.perf_counter()
    ref = h @ w.T
    report["ref_logits_s"] = time.perf_counter() - t_ref
    report["ref_greedy_preview"] = greedy(ref)[:16].tolist()
    print(f"[probe] ref logits {ref.shape} in {report['ref_logits_s']:.2f}s", flush=True)

    # Weight-space cosine helper
    def wcos(hat: np.ndarray) -> float:
        return cosine(w, hat)

    rows: list[dict[str, Any]] = []

    def add_full_eval(name: str, hat: np.ndarray, bytes_today: int, bytes_proj: int, recon_class: str) -> dict[str, Any]:
        t1 = time.perf_counter()
        logits = h @ hat.T
        dt = time.perf_counter() - t1
        sc = score_logits(ref, logits)
        rec = {
            "tensor": "lm_head.weight",
            "candidate": name,
            "kind": "full_eval_lossy",
            "bytes_today_q4_traffic": Q4_LM_HEAD_BYTES,
            "bytes_today_mixed1p5": 320_889_115,
            "projected_bytes": int(bytes_proj),
            "bytes_saved_vs_q4": int(Q4_LM_HEAD_BYTES - bytes_proj),
            "bytes_saved_vs_mixed_q8": int(320_889_115 - bytes_proj),
            "reconstruction_cost": estimate_recon_ns(bytes_proj, recon_class),
            "cpu_gemv_s": dt,
            "weight_cosine_vs_bf16": wcos(hat),
            "quality_vs_bf16": sc,
            "alters_emitted_tokens_vs_bf16": sc["greedy_mismatch"] > 0,
            "hidden_caveat": report["hidden_caveat"],
        }
        rec["bytes_saved_per_recon_ns_vs_q4"] = (
            rec["bytes_saved_vs_q4"] / rec["reconstruction_cost"]["recon_ns"]
            if rec["reconstruction_cost"]["recon_ns"] > 0
            else None
        )
        rec["bytes_saved_per_recon_ns_vs_q8"] = (
            rec["bytes_saved_vs_mixed_q8"] / rec["reconstruction_cost"]["recon_ns"]
            if rec["reconstruction_cost"]["recon_ns"] > 0
            else None
        )
        rows.append(rec)
        print(
            f"[probe] {name}: match={sc['greedy_match_rate']:.4f} "
            f"wcos={rec['weight_cosine_vs_bf16']:.6f} "
            f"logit_cos={sc['mean_logit_cosine']:.6f} "
            f"bytes={bytes_proj} mismatch={sc['greedy_mismatch']}",
            flush=True,
        )
        return rec

    print("[probe] uniform Q8/Q6/Q4/Q3/Q2", flush=True)
    for bits in (8, 6, 4, 3, 2):
        tq = time.perf_counter()
        hat = uniform_hat(w, bits)
        print(f"[probe]   q{bits} encode {time.perf_counter()-tq:.2f}s", flush=True)
        add_full_eval(
            f"HGRAVU01_q{bits}_g64",
            hat,
            Q4_LM_HEAD_BYTES,
            physical_uniform_bytes(w.size, bits),
            "uniform_dequant_gemv",
        )
        if bits == 8:
            q8_logits = h @ hat.T
            q8_hat = hat
        elif bits == 4:
            q4_logits = h @ hat.T
            q4_hat = hat
        del hat

    print("[probe] binary g128", flush=True)
    hat = binary_hat(w)
    add_full_eval(
        "HGRAVB01_binary_g128",
        hat,
        Q4_LM_HEAD_BYTES,
        physical_binary_bytes(w.size),
        "mixed_scatter_or_rice",  # same cheap sign*scale as mixed gate; billed at mixed wall? 
        # Actually binary is cheap (cheaper than Q4). Reclass below.
    )
    # Binary is cheap sequential. Override class to uniform-like sequential rate
    # but keep a note that a *rice residual on top* would fall to mixed_scatter.
    rows[-1]["reconstruction_cost"] = estimate_recon_ns(
        physical_binary_bytes(w.size), "uniform_dequant_gemv"
    )
    rows[-1]["reconstruction_cost"]["note"] = (
        "plain binary (sign*mean-abs) is cheap sequential; rice/outlier residual on "
        "top would drop to mixed_scatter_or_rice at 2.57 GB/s"
    )
    del hat

    # Q4 vs Q8 identity (the packed mixed-1p5 -> ne4 question)
    q4_vs_q8 = score_logits(q8_logits, q4_logits)
    report["q4_vs_q8_artifact"] = {
        **q4_vs_q8,
        "note": (
            "mixed-1p5 stores Q8; mixed-1p5-ne4 stores Q4. If this mismatches, "
            "dropping non-expert 8->4 on lm_head alone changes greedy tokens "
            "relative to the mixed-1p5 artifact oracle."
        ),
        "alters_emitted_tokens_vs_mixed1p5_q8": q4_vs_q8["greedy_mismatch"] > 0,
    }
    print(f"[probe] Q4 vs Q8 greedy match={q4_vs_q8['greedy_match_rate']:.4f}", flush=True)

    print(f"[probe] randomized SVD rank={args.svd_rank}", flush=True)
    t_svd = time.perf_counter()
    u, s, vt = randomized_svd(w, rank=args.svd_rank, n_oversample=32, seed=args.seed)
    report["svd_s"] = time.perf_counter() - t_svd
    report["svd_singular_head"] = s[:16].tolist()
    report["svd_energy"] = {
        str(r): float(np.sum(s[:r] ** 2) / np.sum(s ** 2)) for r in (16, 32, 64, 128, 160, 256) if r <= args.svd_rank
    }
    print(f"[probe] SVD done in {report['svd_s']:.1f}s  energy={report['svd_energy']}", flush=True)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    two_pass_rows: list[dict[str, Any]] = []

    def add_two_pass(name: str, draft: np.ndarray, draft_bytes: int, recon_class: str) -> None:
        cov = coverage_table(ref, draft, ks)
        # exact rescore of draft top-k against SOURCE W (capability-preserving if hit)
        identity_k = None
        for k in ks:
            if cov[str(k)]["miss"] == 0:
                identity_k = k
                break
        # also vs Q8 artifact
        cov_q8 = coverage_table(q8_logits, draft, ks)
        identity_k_q8 = None
        for k in ks:
            if cov_q8[str(k)]["miss"] == 0:
                identity_k_q8 = k
                break
        # cheap prefix/SVD as FULL eval (no rescore) — capability change if mismatch
        full = score_logits(ref, draft)
        k_rescore = identity_k if identity_k is not None else 1024
        exact_bytes = k_rescore * physical_uniform_bytes(HIDDEN, 4)
        # Q4 row is 2048 * 4.25 / 8 = 1088 theoretical; use that.
        exact_bytes = k_rescore * (HIDDEN * 4 + 16) // 8  # 4-bit + f16 scale/64 = 1088
        traffic = draft_bytes + exact_bytes
        rec = {
            "tensor": "lm_head.weight",
            "candidate": name,
            "kind": "cheap_draft_plus_exact_rescore",
            "draft_bytes": int(draft_bytes),
            "exact_rescore_bytes_at_identity_k": int(exact_bytes) if identity_k else None,
            "projected_traffic_bytes": int(traffic) if identity_k else None,
            "identity_k_vs_bf16": identity_k,
            "identity_k_vs_q8": identity_k_q8,
            "coverage_vs_bf16": cov,
            "coverage_vs_q8": cov_q8,
            "full_eval_without_rescore_vs_bf16": full,
            "alters_emitted_tokens_if_used_as_full_eval": full["greedy_mismatch"] > 0,
            "alters_emitted_tokens_if_rescored_at_identity_k": False if identity_k is not None else "UNKNOWN_NO_K_WITH_100PCT",
            "reconstruction_cost_draft": estimate_recon_ns(draft_bytes, recon_class),
            "reconstruction_cost_rescore_q4_rows": (
                estimate_recon_ns(exact_bytes, "row_gather") if identity_k is not None else None
            ),
        }
        if identity_k is not None:
            rec["bytes_saved_vs_q4"] = Q4_LM_HEAD_BYTES - traffic
            rec["bytes_saved_vs_q8"] = 320_889_115 - traffic
            total_ns = (
                rec["reconstruction_cost_draft"]["recon_ns"]
                + rec["reconstruction_cost_rescore_q4_rows"]["recon_ns"]
            )
            rec["bytes_saved_per_recon_ns_vs_q4"] = rec["bytes_saved_vs_q4"] / total_ns if total_ns else None
        two_pass_rows.append(rec)
        print(
            f"[probe] two-pass {name}: full_match={full['greedy_match_rate']:.4f} "
            f"identity_k={identity_k} k32={cov['32']['hit_rate']:.4f} "
            f"k128={cov['128']['hit_rate']:.4f} k1024={cov['1024']['hit_rate']:.4f}",
            flush=True,
        )

    for r in (16, 32, 64, 128, 160, 256):
        if r > args.svd_rank:
            continue
        draft = svd_logits_nV(u, s, vt, h, r)
        # f16 factors: U[V,r] + V[r,H]  (absorb S into U)
        draft_bytes = r * (VOCAB + HIDDEN) * 2
        add_two_pass(f"svd_r{r}_f16_factors", draft, draft_bytes, "two_matvec")
        # also score SVD as a REPLACEMENT (lossy full eval)
        # materializing UV is expensive; we already have draft logits
        sc = score_logits(ref, draft)
        rec = {
            "tensor": "lm_head.weight",
            "candidate": f"replace_lm_head_with_svd_r{r}_f16",
            "kind": "full_eval_lossy",
            "projected_bytes": draft_bytes,
            "bytes_saved_vs_q4": Q4_LM_HEAD_BYTES - draft_bytes,
            "bytes_saved_vs_mixed_q8": 320_889_115 - draft_bytes,
            "reconstruction_cost": estimate_recon_ns(draft_bytes, "two_matvec"),
            "quality_vs_bf16": sc,
            "svd_energy_captured": report["svd_energy"].get(str(r)),
            "alters_emitted_tokens_vs_bf16": sc["greedy_mismatch"] > 0,
        }
        rec["bytes_saved_per_recon_ns_vs_q4"] = (
            rec["bytes_saved_vs_q4"] / rec["reconstruction_cost"]["recon_ns"]
            if rec["reconstruction_cost"]["recon_ns"]
            else None
        )
        rows.append(rec)

    print("[probe] prefix drafts (row-major first-d dims, pack-friendly)", flush=True)
    for d in (64, 128, 256, 512):
        draft = h[:, :d] @ w[:, :d].T
        # Q4 of the prefix only, if we store full Q4 and read a prefix
        draft_bytes_if_stored_prefix_q4 = physical_uniform_bytes(VOCAB * d, 4)
        add_two_pass(f"prefix_d{d}_q4_read_of_stored_rows", draft, draft_bytes_if_stored_prefix_q4, "prefix_then_gather")

    print("[probe] Q2/Q3 as draft against bf16 (same W, cheaper read)", flush=True)
    # Recompute cheap hats only for draft; we already scored them as full eval.
    for bits in (3, 2):
        hat = uniform_hat(w, bits)
        draft = h @ hat.T
        add_two_pass(
            f"HGRAVU01_q{bits}_as_draft_then_exact_q4_rows",
            draft,
            physical_uniform_bytes(w.size, bits),
            "uniform_dequant_gemv",
        )
        del hat

    # CPU timing of two-pass vs full, one hidden, DIRTY
    print("[probe] CPU timing class (one hidden, DIRTY_ENGINEERING)", flush=True)
    x = np.ascontiguousarray(h[0])
    timings = {}
    t = time.perf_counter()
    _ = w @ x
    timings["full_bf16_gemv_s"] = time.perf_counter() - t
    t = time.perf_counter()
    _ = q4_hat @ x
    timings["full_q4_materialized_gemv_s"] = time.perf_counter() - t
    t = time.perf_counter()
    _ = q8_hat @ x
    timings["full_q8_materialized_gemv_s"] = time.perf_counter() - t
    for r in (64, 128, 256):
        if r > args.svd_rank:
            continue
        t = time.perf_counter()
        mid = vt[:r] @ x
        _ = (u[:, :r] * s[:r]) @ mid
        timings[f"svd_r{r}_two_matvec_s"] = time.perf_counter() - t
    t = time.perf_counter()
    _ = w[:64] @ x
    timings["exact_64_rows_s"] = time.perf_counter() - t
    report["cpu_timings_one_hidden"] = timings
    print(json.dumps(timings, indent=2), flush=True)

    # ---------------- embed ----------------
    print("[probe] loading embed_tokens", flush=True)
    del w, q4_hat, q8_hat, u, s, vt, ref, q8_logits, q4_logits
    embed = load_bf16_tensor(MODEL_DIR, "model.embed_tokens.weight")
    print(f"[probe] embed {embed.shape} {embed.nbytes} bytes", flush=True)
    rng = np.random.default_rng(args.seed + 1)
    # Include a few special / common ids plus a random sample.
    special = np.array([0, 1, 2, 151643, 151644, 151645, 151646, 872, 198, 7985, 264, 729], dtype=np.int64)
    special = special[(special >= 0) & (special < TOKENIZER_VOCAB)]
    extra = rng.integers(0, TOKENIZER_VOCAB, size=512, dtype=np.int64)
    ids = np.unique(np.concatenate([special, extra]))
    src_rows = embed[ids]
    embed_rows: list[dict[str, Any]] = []
    for bits in (8, 6, 4, 3, 2):
        hat = uniform_hat(embed, bits)
        hr = hat[ids]
        row_cos = cosine(src_rows, hr, axis=-1)
        rec = {
            "tensor": "model.embed_tokens.weight",
            "candidate": f"HGRAVU01_q{bits}_g64",
            "kind": "row_gather_lossy",
            "table_bytes_today_q4": 165_306_651,
            "table_bytes_today_mixed1p5": 320_889_115,
            "projected_table_bytes": physical_uniform_bytes(embed.size, bits),
            "per_token_traffic_bytes": physical_uniform_bytes(HIDDEN, bits) if False else (HIDDEN * bits + 16) // 8,
            "per_token_traffic_note": (
                "Exactly one row is gathered per token. Traffic is one row, not the table. "
                f"Q4 row = {HIDDEN * 4.25 / 8:.0f} B; Q8 row = {HIDDEN * 8.25 / 8:.0f} B."
            ),
            "mean_row_cosine": float(np.mean(row_cos)),
            "min_row_cosine": float(np.min(row_cos)),
            "p10_row_cosine": float(np.percentile(row_cos, 10)),
            "row_rmse": float(np.sqrt(np.mean((hr - src_rows).astype(np.float64) ** 2))),
            "n_rows_scored": int(ids.size),
            "reconstruction_cost": estimate_recon_ns((HIDDEN * bits + 16) // 8, "row_gather"),
            "alters_emitted_tokens": (
                "INDIRECT. A row error shifts every subsequent residual. Not a logit-local "
                "error. Token identity after 48 layers is UNMEASURED here; treat any "
                "sub-Q4 embed change as a capability risk until a generate gate."
            ),
        }
        embed_rows.append(rec)
        print(
            f"[probe] embed q{bits}: mean_row_cos={rec['mean_row_cosine']:.6f} "
            f"min={rec['min_row_cosine']:.6f} table={rec['projected_table_bytes']}",
            flush=True,
        )
        del hat
    hat = binary_hat(embed)
    hr = hat[ids]
    row_cos = cosine(src_rows, hr, axis=-1)
    embed_rows.append(
        {
            "tensor": "model.embed_tokens.weight",
            "candidate": "HGRAVB01_binary_g128",
            "kind": "row_gather_lossy",
            "projected_table_bytes": physical_binary_bytes(embed.size),
            "per_token_traffic_bytes": HIDDEN // 8 + 2,
            "mean_row_cosine": float(np.mean(row_cos)),
            "min_row_cosine": float(np.min(row_cos)),
            "p10_row_cosine": float(np.percentile(row_cos, 10)),
            "row_rmse": float(np.sqrt(np.mean((hr - src_rows).astype(np.float64) ** 2))),
            "n_rows_scored": int(ids.size),
            "reconstruction_cost": estimate_recon_ns(HIDDEN // 8 + 2, "row_gather"),
            "alters_emitted_tokens": "INDIRECT; binary embed is a capability risk. GLM R0 sub-bit embed was catastrophic.",
        }
    )
    print(
        f"[probe] embed binary: mean_row_cos={embed_rows[-1]['mean_row_cosine']:.6f} "
        f"min={embed_rows[-1]['min_row_cosine']:.6f}",
        flush=True,
    )
    del hat, embed

    # Catalog facts
    report["on_disk_catalog"] = {
        "mixed-1p5-v1": {
            "lm_head.weight": {"nbytes": 320889115, "codec_bpw": 8.250007629394531, "codec": "HGRAVU01_q8", "flags": "SENSITIVE_UNTOUCHED"},
            "model.embed_tokens.weight": {"nbytes": 320889115, "codec_bpw": 8.250007629394531, "codec": "HGRAVU01_q8", "flags": "SENSITIVE_UNTOUCHED"},
        },
        "mixed-1p5-ne4-v1": {
            "lm_head.weight": {"nbytes": 165306651, "codec_bpw": 4.250007152557373, "codec": "HGRAVU01_q4"},
            "model.embed_tokens.weight": {"nbytes": 165306651, "codec_bpw": 4.250007152557373, "codec": "HGRAVU01_q4"},
        },
    }
    report["q4_vehicle_terminal"] = {
        "source": "receipts/QWEN80_TOKEN_NS_LEDGER.json",
        "n_tokens": 16,
        "gpu_ns_median": Q4_LM_HEAD_GPU_NS_MEDIAN,
        "bytes": Q4_LM_HEAD_BYTES,
        "implied_gb_s": Q4_LM_HEAD_GB_S,
        "embed_gpu_ns_median": 4874,
        "embed_bytes": 1088,
        "note": (
            "lm_head is 14.7% of per-token BYTES on the Q4 vehicle and ~1.25 ms of GPU "
            f"({Q4_LM_HEAD_GB_S:.1f} GB/s sequential). It is not 14.7% of token TIME. "
            "Embed is 1088 B / ~5 us GPU (launch-bound)."
        ),
    }
    report["qwen38_geometry"] = {
        "lm_head_shape": [248320, 5120],
        "embed_shape": [248320, 5120],
        "q4_table_bytes_measured": 675430440,
        "tie_word_embeddings": False,
        "embed_is_not_traffic": True,
        "lm_head_is_traffic": True,
        "note": "Do not confuse the 675,430,440 B table with per-token embed traffic (one 5120-wide row).",
    }

    report["lm_head_full_eval_candidates"] = rows
    report["lm_head_two_pass_candidates"] = two_pass_rows
    report["embed_candidates"] = embed_rows
    report["wall_s"] = time.perf_counter() - t0
    report["claim_boundary"] = {
        "gpu_benchmark_not_run": True,
        "generate_not_run": True,
        "hidden_is_l47_router_input_not_post_final_moe": True,
        "graded_against_source_bf16_lm_head": True,
        "n_hidden": int(h.shape[0]) if False else report["n_hidden"],
        "reserved_tail_masked": True,
        "qwen38_quality_not_measured": True,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[probe] wrote {args.out} wall_s={report['wall_s']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
