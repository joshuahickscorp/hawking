#!/usr/bin/env python3
"""Qwen3.8 generator + residual measurement. CPU only. Peak RSS kept low.

Reads BF16 safetensors from the on-disk Qwen3.8-27B artifact. Does not write
model artifacts. Does not touch GPU / Metal / the live Genesis process.
"""
from __future__ import annotations

import gc
import json
import math
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("G1_TORCH_THREADS", "8")))
torch.set_grad_enabled(False)

BF16_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
OUT_JSON = Path("/tmp/g1_gen_residual.json")
OUT_LOG = Path("/tmp/g1_gen_residual.log")

R_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256)
R_RESIDUAL = (8, 16, 32, 64, 128, 256)
TARGETS = (0.05, 0.10, 0.15)
RSVD_K = 256
RSVD_P = 16
RSVD_Q = 2
EXACT_SVD_MAX = 1536


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss_max={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    with OUT_LOG.open("a") as f:
        f.write(line + "\n")


def classify(name: str) -> str:
    if "vision_tower" in name or "visual" in name:
        return "vision"
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".mlp.gate_proj.weight" in name:
        return "mlp.gate_proj"
    if ".mlp.up_proj.weight" in name:
        return "mlp.up_proj"
    if ".mlp.down_proj.weight" in name:
        return "mlp.down_proj"
    if ".self_attn.q_proj.weight" in name:
        return "full.q_proj"
    if ".self_attn.k_proj.weight" in name:
        return "full.k_proj"
    if ".self_attn.v_proj.weight" in name:
        return "full.v_proj"
    if ".self_attn.o_proj.weight" in name:
        return "full.o_proj"
    if ".linear_attn.in_proj_qkv.weight" in name:
        return "lin.in_proj_qkv"
    if ".linear_attn.in_proj_z.weight" in name:
        return "lin.in_proj_z"
    if ".linear_attn.in_proj_a.weight" in name:
        return "lin.in_proj_a"
    if ".linear_attn.in_proj_b.weight" in name:
        return "lin.in_proj_b"
    if ".linear_attn.out_proj.weight" in name:
        return "lin.out_proj"
    return "other"


class ShardIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        idx = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self._headers: dict[str, dict] = {}
        self._header_nbytes: dict[str, int] = {}

    def header(self, shard: str) -> dict:
        if shard in self._headers:
            return self._headers[shard]
        path = self.root / shard
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        self._headers[shard] = h
        self._header_nbytes[shard] = n
        return h

    def meta(self, name: str) -> tuple[str, dict, int]:
        shard = self.weight_map[name]
        h = self.header(shard)
        return shard, h[name], self._header_nbytes[shard]

    def load_f32(self, name: str) -> torch.Tensor:
        shard, meta, hbytes = self.meta(name)
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        n_elem = int(np.prod(shape))
        if dtype != "BF16":
            raise RuntimeError(f"{name} dtype {dtype}, expected BF16")
        if (end - start) != n_elem * 2:
            raise RuntimeError(f"{name} size mismatch {end-start} vs {shape}")
        path = self.root / shard
        out = np.empty(n_elem, dtype=np.float32)
        chunk = 1 << 22  # 4M elements, 8 MiB bf16
        with open(path, "rb") as f:
            f.seek(8 + hbytes + start)
            done = 0
            while done < n_elem:
                take = min(chunk, n_elem - done)
                raw = f.read(take * 2)
                u16 = np.frombuffer(raw, dtype=np.uint16, count=take)
                u32 = u16.astype(np.uint32) << 16
                out[done : done + take] = u32.view(np.float32)
                done += take
        return torch.from_numpy(out.reshape(shape))


def rsvd(W: torch.Tensor, k: int, p: int = RSVD_P, q: int = RSVD_Q, seed: int = 0):
    m, n = W.shape
    k_req = min(k, min(m, n))
    if min(m, n) <= EXACT_SVD_MAX:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U, S, Vh, "exact"
    l = min(k_req + p, min(m, n))
    g = torch.Generator().manual_seed(seed)
    if m >= n:
        Omega = torch.randn(n, l, generator=g, dtype=W.dtype)
        Y = W @ Omega
        for _ in range(q):
            Y = W @ (W.t() @ Y)
        Qb, _ = torch.linalg.qr(Y, mode="reduced")
        B = Qb.t() @ W
        Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Qb @ Uhat
    else:
        Omega = torch.randn(l, m, generator=g, dtype=W.dtype)
        Y = Omega @ W
        for _ in range(q):
            Y = (Y @ W.t()) @ W
        Qb, _ = torch.linalg.qr(Y.t(), mode="reduced")
        B = W @ Qb
        U, S, Vh_s = torch.linalg.svd(B, full_matrices=False)
        Vh = Vh_s @ Qb.t()
    return U[:, :k_req], S[:k_req], Vh[:k_req], "rsvd"


def frob2(t: torch.Tensor) -> float:
    return float(t.float().square().sum().item())


def stats(x: torch.Tensor) -> dict:
    """One-pass chunked moments. No extra full-matrix copies."""
    xf = x.reshape(-1)
    n = int(xf.numel())
    chunk = 1 << 20
    s1 = 0.0
    s2 = 0.0
    s4 = 0.0
    sabs = 0.0
    max_abs = 0.0
    # first pass: mean, m2, max, mean_abs
    for i in range(0, n, chunk):
        sl = xf[i : i + chunk].float()
        s1 += float(sl.sum().item())
        s2 += float(sl.square().sum().item())
        sabs += float(sl.abs().sum().item())
        max_abs = max(max_abs, float(sl.abs().max().item()))
    mean = s1 / n
    m2 = s2 / n - mean * mean
    if m2 < 0:
        m2 = 0.0
    std = math.sqrt(m2)
    rms = math.sqrt(s2 / n)
    mean_abs = sabs / n
    # second pass: m4 and tail + subsample for percentiles
    sample_target = min(n, 4_000_000)
    step = max(1, n // sample_target)
    sample_vals = []
    gt3 = 0
    thresh = 3.0 * rms
    for i in range(0, n, chunk):
        sl = xf[i : i + chunk].float()
        xc = sl - mean
        s4 += float((xc * xc * xc * xc).sum().item())
        if thresh > 0:
            gt3 += int((sl.abs() > thresh).sum().item())
        sample_vals.append(sl[::step].abs().cpu())
    m4 = s4 / n
    kurt = (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0
    sample = torch.cat(sample_vals)
    if sample.numel() > 4_000_000:
        sample = sample[:4_000_000]
    qs = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=sample.dtype))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "rms": rms,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "peak_over_rms": (max_abs / rms) if rms > 0 else None,
        "excess_kurtosis": kurt,
        "frac_gt_3rms": gt3 / n,
        "abs_p50": float(qs[0]),
        "abs_p90": float(qs[1]),
        "abs_p99": float(qs[2]),
        "abs_p999": float(qs[3]),
    }


def _group_view(X: torch.Tensor, group: int):
    m, n = X.shape
    pad = (group - (n % group)) % group
    if pad:
        Xp = torch.nn.functional.pad(X, (0, pad))
    else:
        Xp = X
    G = Xp.reshape(m, -1, group)
    n_groups = m * ((n + group - 1) // group)
    return G, pad, n_groups, n


def quant_uniform(X: torch.Tensor, bits: int, group: int):
    G, pad, n_groups, n = _group_view(X, group)
    qmax = (1 << (bits - 1)) - 1
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = (G / scale).round().clamp(-qmax, qmax)
    recon = (q * scale).reshape(G.shape[0], -1)
    if pad:
        recon = recon[:, :n]
    m = X.shape[0]
    payload = (bits * m * n + 7) // 8 + n_groups * 2  # f16 scale
    return recon, payload


def quant_binary(X: torch.Tensor, group: int = 128):
    G, pad, n_groups, n = _group_view(X, group)
    scale = G.abs().mean(dim=-1, keepdim=True)
    recon = G.sign() * scale
    if pad:
        recon = recon.reshape(G.shape[0], -1)[:, :n]
    else:
        recon = recon.reshape(G.shape[0], -1)
    m = X.shape[0]
    payload = (m * n + 7) // 8 + n_groups * 2
    return recon, payload


CODECS = (
    ("binary_g128", lambda X: quant_binary(X, 128)),
    ("uniform_q2_g64", lambda X: quant_uniform(X, 2, 64)),
    ("uniform_q3_g64", lambda X: quant_uniform(X, 3, 64)),
    ("uniform_q4_g64", lambda X: quant_uniform(X, 4, 64)),
)


def codec_error(W: torch.Tensor, recon: torch.Tensor, w_f2: float) -> dict:
    diff = W - recon
    e2 = frob2(diff)
    rel = math.sqrt(e2 / w_f2) if w_f2 > 0 else 0.0
    # cosine
    num = float((W * recon).sum().item())
    den = math.sqrt(w_f2 * frob2(recon))
    cos = (num / den) if den > 0 else 0.0
    return {"rel_l2": rel, "cosine": cos, "err_frob2": e2}


def factor_bytes(m: int, n: int, r: int) -> int:
    # A (m x r) f16 + B (r x n) f16. S folded into A.
    return r * (m + n) * 2


def select_targets() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    mlp_layers = (0, 3, 15, 31, 47, 63)
    for L in mlp_layers:
        for role in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            items.append(
                (
                    f"language_model.model.layers.{L}.{role}.weight",
                    role,
                )
            )
    for L in (3, 15, 31, 63):
        for role in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ):
            items.append(
                (
                    f"language_model.model.layers.{L}.{role}.weight",
                    "full." + role.split(".")[1],
                )
            )
    for L in (0, 16, 32, 48):
        for role in (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ):
            items.append(
                (
                    f"language_model.model.layers.{L}.{role}.weight",
                    "lin." + role.split(".")[1],
                )
            )
    items.append(("language_model.model.embed_tokens.weight", "embed"))
    items.append(("language_model.lm_head.weight", "lm_head"))
    return items


def analyze_one(idx: ShardIndex, name: str, cls: str) -> dict:
    t0 = time.perf_counter()
    log(f"LOAD {cls} {name}")
    W = idx.load_f32(name)
    m, n = int(W.shape[0]), int(W.shape[1])
    w_f2 = frob2(W)
    w_f = math.sqrt(w_f2)
    w_stats = stats(W)
    log(f"  shape=({m},{n}) ||W||_F={w_f:.6g} load_s={time.perf_counter()-t0:.2f}")

    k_use = RSVD_K
    if cls in ("embed", "lm_head"):
        k_use = 128
    t1 = time.perf_counter()
    U, S, Vh, method = rsvd(W, k_use)
    svd_s = time.perf_counter() - t1
    S_np = S.detach().cpu().numpy().astype(np.float64)
    captured_all = float(np.square(S_np).sum())
    energy = {}
    for r in R_SWEEP:
        if r > S_np.size:
            continue
        energy[str(r)] = float(np.square(S_np[:r]).sum() / w_f2) if w_f2 else 0.0
    # spectrum samples
    spec_idx = [0, 1, 3, 7, 15, 31, 63, 127, 255]
    spectrum = {str(i): float(S_np[i]) for i in spec_idx if i < S_np.size}
    decay = {}
    if S_np.size > 1 and S_np[-1] != 0:
        pass
    for a, b in ((0, 7), (0, 31), (0, 63), (0, 127), (0, 255)):
        if b < S_np.size and S_np[b] > 0:
            decay[f"s{a+1}_over_s{b+1}"] = float(S_np[a] / S_np[b])

    # original codec errors (incumbent)
    orig_codecs = {}
    for cname, fn in CODECS:
        recon, payload = fn(W)
        rec = codec_error(W, recon, w_f2)
        rec["payload_bytes"] = int(payload)
        rec["bpw"] = 8.0 * payload / (m * n)
        orig_codecs[cname] = rec
        del recon
    q4_rel = orig_codecs["uniform_q4_g64"]["rel_l2"]

    # residual ranks
    residual_ranks = []
    r_list = [r for r in R_RESIDUAL if r <= S_np.size]
    if cls in ("embed", "lm_head"):
        r_list = [r for r in (8, 32, 64, 128) if r <= S_np.size]

    for r in r_list:
        A32 = U[:, :r] * S[:r]
        B32 = Vh[:r]
        G32 = A32 @ B32
        g32_rel = math.sqrt(frob2(W - G32) / w_f2)
        A16 = A32.half().float()
        B16 = B32.half().float()
        # form f16 generator in-place over G32 to avoid a third full matrix
        G32.copy_(A16 @ B16)
        G16 = G32
        g16_rel = math.sqrt(frob2(W - G16) / w_f2)
        f16_extra = math.sqrt(max(g16_rel * g16_rel - g32_rel * g32_rel, 0.0))
        large = (m * n) > 200_000_000
        if large:
            W.sub_(G16)
            R = W
            inplace = True
        else:
            R = W - G16
            inplace = False
        r_stats = stats(R)
        r_f2 = frob2(R)
        # residual spectrum (cheap, k=32 or exact if small)
        try:
            _, Sr, _, rmethod = rsvd(R, min(32, min(m, n)))
            Sr_np = Sr.detach().cpu().numpy().astype(np.float64)
            r_energy32 = float(np.square(Sr_np[: min(32, Sr_np.size)]).sum() / r_f2) if r_f2 else 0.0
            r_s1s8 = float(Sr_np[0] / Sr_np[7]) if Sr_np.size > 7 and Sr_np[7] > 0 else None
        except Exception as e:
            r_energy32 = None
            r_s1s8 = None
            rmethod = f"fail:{e}"
        res_codecs = {}
        for cname, fn in CODECS:
            rq, payload = fn(R)
            # ||W - (G + Q(R))|| = ||R - Q(R)||
            rec = codec_error(R, rq, w_f2)
            fb = factor_bytes(m, n, r)
            rec["residual_payload_bytes"] = int(payload)
            rec["factor_bytes_f16"] = int(fb)
            rec["total_bytes"] = int(payload + fb)
            rec["residual_bpw"] = 8.0 * payload / (m * n)
            rec["factor_bpw"] = 8.0 * fb / (m * n)
            rec["total_bpw"] = 8.0 * (payload + fb) / (m * n)
            rec["beats_orig_rel"] = rec["rel_l2"] < orig_codecs[cname]["rel_l2"]
            rec["rel_improvement"] = (
                orig_codecs[cname]["rel_l2"] / rec["rel_l2"]
                if rec["rel_l2"] > 0
                else None
            )
            res_codecs[cname] = rec
            del rq
        if inplace:
            W.add_(G16)  # restore original weights for the next rank
        # bits to hit targets: among residual codecs, pick cheapest total_bpw
        bits_to_target = {}
        for tgt in TARGETS + (q4_rel,):
            key = f"rel_l2<={tgt:.6f}"
            hits = []
            for cname, rec in res_codecs.items():
                if rec["rel_l2"] <= tgt:
                    hits.append((rec["total_bpw"], cname, rec["rel_l2"]))
            if hits:
                hits.sort()
                bits_to_target[key] = {
                    "total_bpw": hits[0][0],
                    "codec": hits[0][1],
                    "rel_l2": hits[0][2],
                    "target": tgt,
                }
            else:
                bits_to_target[key] = {
                    "total_bpw": None,
                    "codec": None,
                    "rel_l2": None,
                    "target": tgt,
                    "note": "no tested residual codec hits target",
                }
        residual_ranks.append(
            {
                "r": r,
                "explained_frob": energy.get(str(r)),
                "lr_only_rel_l2_f32": g32_rel,
                "lr_only_rel_l2_f16": g16_rel,
                "f16_factor_extra_rel_l2": f16_extra,
                "residual_stats": r_stats,
                "residual_top32_energy_frac": r_energy32,
                "residual_s1_over_s8": r_s1s8,
                "residual_spectrum_method": rmethod,
                "orig_peak_over_rms": w_stats["peak_over_rms"],
                "resid_peak_over_rms": r_stats["peak_over_rms"],
                "orig_excess_kurtosis": w_stats["excess_kurtosis"],
                "resid_excess_kurtosis": r_stats["excess_kurtosis"],
                "codecs": res_codecs,
                "bits_to_target": bits_to_target,
            }
        )
        del A32, B32, G32, A16, B16, G16, R
        gc.collect()

    rec = {
        "name": name,
        "class": cls,
        "shape": [m, n],
        "n_elem": m * n,
        "dtype_on_disk": "BF16",
        "frob": w_f,
        "weight_stats": w_stats,
        "svd_method": method,
        "svd_k": int(S_np.size),
        "svd_s": svd_s,
        "approx_captured_energy_of_computed_S": captured_all / w_f2 if w_f2 else 0.0,
        "energy_frac": energy,
        "spectrum": spectrum,
        "decay": decay,
        "orig_codecs": orig_codecs,
        "q4_rel_l2": q4_rel,
        "residual": residual_ranks,
        "wall_s": time.perf_counter() - t0,
        "rss_max_gb_after": rss_gb(),
    }
    del W, U, S, Vh
    gc.collect()
    log(f"  DONE {cls} energy@64={energy.get('64')} q4_rel={q4_rel:.4f} wall={rec['wall_s']:.1f}s")
    return rec


def gaussian_control(m: int = 17408, n: int = 5120, seed: int = 1) -> dict:
    log(f"GAUSSIAN CONTROL {m}x{n}")
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(m, n, generator=g, dtype=torch.float32)
    W *= 0.02
    w_f2 = frob2(W)
    U, S, Vh, method = rsvd(W, 256)
    S_np = S.detach().cpu().numpy().astype(np.float64)
    energy = {
        str(r): float(np.square(S_np[:r]).sum() / w_f2)
        for r in R_SWEEP
        if r <= S_np.size
    }
    out = {
        "kind": "iid_gaussian_control",
        "shape": [m, n],
        "svd_method": method,
        "energy_frac": energy,
        "s1_over_s64": float(S_np[0] / S_np[63]) if S_np.size > 63 else None,
        "note": "Marchenko-Pastur-like flat spectrum reference at same MLP shape",
    }
    del W, U, S, Vh
    gc.collect()
    return out


def main() -> None:
    if OUT_LOG.exists():
        OUT_LOG.unlink()
    log(f"start threads={torch.get_num_threads()} bf16={BF16_DIR}")
    idx = ShardIndex(BF16_DIR)
    targets = select_targets()
    log(f"n_targets={len(targets)}")
    results = {
        "schema": "hawking.g1.generator_residual.v1",
        "measured_at_unix": time.time(),
        "artifact": str(BF16_DIR),
        "artifact_identity": "Qwen3.8-27B BF16 safetensors (PocketAiHub / qwen3_5)",
        "method": {
            "svd": "exact if min(m,n)<=1536 else randomized SVD k=256 p=16 q=2 (k=128 on embed/lm_head)",
            "factors": "A=U[:,:r]*S[:r] and B=Vh[:r] stored as f16; S folded",
            "codecs": [c[0] for c in CODECS],
            "error": "weight-space relative Frobenius ||W-hat||_F/||W||_F and cosine",
            "no_gpu": True,
            "no_generation": True,
            "torch_threads": torch.get_num_threads(),
        },
        "tensors": [],
        "gaussian_control": None,
        "errors": [],
    }
    # gaussian first (cheap kill reference)
    try:
        results["gaussian_control"] = gaussian_control()
        OUT_JSON.write_text(json.dumps(results, indent=2))
    except Exception as e:
        results["errors"].append({"where": "gaussian", "error": str(e)})
        log(f"GAUSSIAN FAIL {e}")

    for name, cls in targets:
        if name not in idx.weight_map:
            results["errors"].append({"name": name, "error": "missing_from_index"})
            log(f"MISSING {name}")
            continue
        try:
            rec = analyze_one(idx, name, cls)
            results["tensors"].append(rec)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            results["errors"].append({"name": name, "error": str(e), "tb": tb})
            log(f"FAIL {name}: {e}")
        OUT_JSON.write_text(json.dumps(results, indent=2))
        if rss_gb() > 18.0:
            log("ABORT rss_max>18G")
            results["errors"].append({"where": "rss_guard", "rss_max_gb": rss_gb()})
            break

    # class summary
    by = {}
    for t in results["tensors"]:
        by.setdefault(t["class"], []).append(t)
    summary = {}
    for cls, rows in by.items():
        e64 = [r["energy_frac"].get("64") for r in rows if r["energy_frac"].get("64") is not None]
        e128 = [r["energy_frac"].get("128") for r in rows if r["energy_frac"].get("128") is not None]
        e256 = [r["energy_frac"].get("256") for r in rows if r["energy_frac"].get("256") is not None]
        q4 = [r["q4_rel_l2"] for r in rows]
        # at r=64, binary residual vs original binary
        bin_imp = []
        q4_imp = []
        for r in rows:
            for rr in r["residual"]:
                if rr["r"] == 64:
                    bc = rr["codecs"]["binary_g128"]
                    qc = rr["codecs"]["uniform_q4_g64"]
                    bin_imp.append(bc.get("rel_improvement"))
                    q4_imp.append(qc.get("rel_improvement"))
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return float(sum(xs) / len(xs)) if xs else None

        summary[cls] = {
            "n": len(rows),
            "energy64_mean": mean(e64),
            "energy64_min": min(e64) if e64 else None,
            "energy64_max": max(e64) if e64 else None,
            "energy128_mean": mean(e128),
            "energy256_mean": mean(e256),
            "q4_rel_l2_mean": mean(q4),
            "r64_binary_rel_improvement_mean": mean(bin_imp),
            "r64_q4_rel_improvement_mean": mean(q4_imp),
            "flat_vs_gaussian64": (
                None
                if not e64 or not results.get("gaussian_control")
                else mean(e64)
                / results["gaussian_control"]["energy_frac"]["64"]
            ),
        }
    results["summary_by_class"] = summary
    results["rss_max_gb"] = rss_gb()
    OUT_JSON.write_text(json.dumps(results, indent=2))
    log(f"WROTE {OUT_JSON} tensors={len(results['tensors'])} errors={len(results['errors'])}")
    print(json.dumps({"summary_by_class": summary, "n": len(results["tensors"])}, indent=2))


if __name__ == "__main__":
    main()
