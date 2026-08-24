#!/usr/bin/env python3
"""Does GLM's 0.167 BPW / 0.755 cosine activation-aware fit transfer to DENSE organs?

The 2026-07-27 GLM result is real: A1 activation-aware rank-16 f16 factors on
real teacher capsules scored 0.755 cosine (e0 0.75515; 12/12 mean 0.75425)
at 0.16667 BPW, beating a constant-mean null of 0.651, while raw-weight
rank-16 at the same rate scored 0.189 and failed the null. That was a MoE
expert. This lane applies the SAME method to dense Qwen3.8 organs.

Not a kernel. Not a 27B server. Real captured activations only.

    python3 tools/headless/dense_subbit_transfer.py
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
OUT_PATH = ROOT / "receipts" / "headless" / "DENSE_SUBBIT_TRANSFER.json"

HIDDEN = 5120
INTERMEDIATE = 17408
F16_BPW = 16.0
HEADER_BYTES = 64  # GLM52AAP per-tensor header; billed separately from factor BPW
GLM_TARGET_BPW = 0.16667
GLM_TARGET_COSINE = 0.755
GLM_RANK = 16
SEED = 10582654  # glm52_activation_aware_pack.SEED
SCALE_TRAP = 0.01
CHUNK = 512
MAX_RANK = 1024
# Rank ladder. 41 is the dense-organ rank whose factor-storage BPW is closest
# to GLM's 0.16667 (16*k*(H+I)/(H*I)). 16 is the GLM rank (lower BPW here
# because the dense matrix is larger).
RANKS = (8, 16, 32, 41, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
LAYERS = (0, 31)
ORGANS = ("up_proj", "down_proj")  # two types; down inverts the family ranking
ORGAN_SEED = {"up_proj": 1, "down_proj": 2, "gate_proj": 3}
GAIN_REJECT_SCALE = 0.05  # 0.01*W must land below this
GAIN_HEALTHY_FLOOR = 0.50

PARENT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
    ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    ROOT / "workspace/campaign/phaseB/capture_diverse2",
    Path(
        "/Users/scammermike/Downloads/hawking-copy/workspace/campaign/"
        "records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_attn_norm"
    ),
]
GLM_SWEEP_GIT = "workspace/campaign/evidence/models/glm52/GLM52_REAL_ACTIVATION_SWEEP.json"
GLM_REPL_GIT = "workspace/campaign/evidence/models/glm52/GLM52_ACTIVATION_AWARE_REPLICATION.json"
GLM_PILOT_GIT = "workspace/campaign/evidence/models/glm52/GLM52_BASIS_PILOT_RECEIPT.json"
Q80_ORTHO = ROOT / "receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json"


def _ensure_torch():
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
    sys.exit("torch required (tried sys python and ~/.grok-vision/bin/python)")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_json(rel: str):
    """Load a JSON blob from git even when the sparse checkout omitted it."""
    disk = ROOT / rel
    if disk.is_file():
        return json.loads(disk.read_text()), f"disk:{disk}"
    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
        return json.loads(raw), f"git:HEAD:{rel}"
    except Exception as e:
        return None, f"missing:{rel} ({e})"


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass
    return str(x)


def find_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if (p / "model.safetensors.index.json").is_file():
            return p
    raise FileNotFoundError("qualified parent bf16 not found")


def find_capture() -> Path:
    for p in CAPTURE_CANDIDATES:
        if (p / "L00.f16").is_file() or (p / "L0.f16").is_file():
            return p
    raise FileNotFoundError("real post_attn_norm capture not found")


def tensor_name(layer: int, organ: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.{organ}.weight"


def load_tensor(parent: Path, name: str):
    import numpy as np

    index = json.loads((parent / "model.safetensors.index.json").read_text())
    shard = parent / index["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name} dtype {meta['dtype']}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return np.array(f32.reshape(meta["shape"]), dtype=np.float32, copy=True)


def capture_path(cap: Path, layer: int) -> Path:
    for name in (f"L{layer:02d}.f16", f"L{layer}.f16"):
        p = cap / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no capture for layer {layer} in {cap}")


def load_X(cap: Path, layer: int):
    import numpy as np

    p = capture_path(cap, layer)
    raw = np.fromfile(p, dtype=np.float16)
    if raw.size % HIDDEN != 0:
        raise ValueError(f"{p} size {raw.size} not divisible by hidden {HIDDEN}")
    X = raw.reshape(-1, HIDDEN).astype(np.float32)
    if X.shape[0] < 256:
        raise ValueError(f"{p} only {X.shape[0]} rows; refusing a toy capture")
    return X


def silu(x):
    import numpy as np

    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def gemm(a, b):
    """Threaded GEMM. Torch CPU; never a Gaussian proxy for X."""
    import numpy as np
    import torch

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    return (ta @ tb).numpy()


def x_wt(X, W, chunk: int = CHUNK):
    """Y = X @ W.T, chunked on the token axis."""
    import numpy as np

    n = X.shape[0]
    out_dim = W.shape[0]
    if n <= chunk:
        return gemm(X, W.T)
    y = np.empty((n, out_dim), dtype=np.float32)
    for i in range(0, n, chunk):
        y[i : i + chunk] = gemm(X[i : i + chunk], W.T)
    return y


def swiglu_intermediate(X, Wg, Wu, chunk: int = CHUNK):
    """Real post-SwiGLU X for down_proj. Not Gaussian. Not a unit-sphere probe."""
    import numpy as np

    n = X.shape[0]
    parts = []
    for i in range(0, n, chunk):
        xb = X[i : i + chunk]
        parts.append(silu(gemm(xb, Wg.T)) * gemm(xb, Wu.T))
    return np.concatenate(parts, axis=0)


def row_cosine(A, B) -> float:
    import numpy as np

    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    ok = den > 1e-20
    if not np.any(ok):
        return float("nan")
    return float((num[ok] / den[ok]).mean())


def rel_fro(A, B) -> float:
    import numpy as np

    na = np.linalg.norm(A)
    if na == 0:
        return float("nan")
    return float(np.linalg.norm(A - B) / na)


def gain_score(A, B) -> float:
    """min(r, 1/r) on per-row mean and per-unit min. Rejects 0.01*W."""
    import numpy as np

    def ratio(axis):
        na = np.linalg.norm(A, axis=axis)
        nb = np.linalg.norm(B, axis=axis)
        r = nb / (na + 1e-30)
        return np.minimum(r, 1.0 / (r + 1e-30))

    return float(min(np.mean(ratio(1)), ratio(0).min()))


def constant_mean_null(Y) -> float:
    import numpy as np

    mu = Y.mean(axis=0, keepdims=True)
    return row_cosine(Y, np.broadcast_to(mu, Y.shape))


def score_pair(Y, Yh) -> dict:
    cos = row_cosine(Y, Yh)
    null = constant_mean_null(Y)
    return {
        "rel_fro": rel_fro(Y, Yh),
        "cosine": cos,
        "gain": gain_score(Y, Yh),
        "null": null,
        "beats_null": bool(cos > null),
        "surplus_over_null": cos - null,
    }


def factor_bpw(rows: int, cols: int, rank: int) -> dict:
    """GLM-sweep billing: f16 factors, no header. Active = two f16 GEMMs.

    GLM k16 up_proj [2048, 6144]: 16*16*(2048+6144)/(2048*6144) = 0.16667.
    Reconstructing W_hat to dense f16 for a naive kernel is 16 active BPW.
    """
    n_w = rows * cols
    factor_elems = rank * (rows + cols)
    storage_bits = int(F16_BPW * factor_elems)
    header_bits = HEADER_BYTES * 8
    complete_bits = storage_bits + header_bits
    return {
        "rank": int(rank),
        "n_weights": int(n_w),
        "factor_elems": int(factor_elems),
        "storage_bits": storage_bits,
        "storage_bpw": storage_bits / n_w,
        "complete_storage_bits": complete_bits,
        "complete_storage_bpw": complete_bits / n_w,
        "active_two_gemm_f16_bits": storage_bits,
        "active_two_gemm_f16_bpw": storage_bits / n_w,
        "active_two_gemm_f32_bpw": 2.0 * storage_bits / n_w,
        "active_reconstructed_f16_bpw": F16_BPW,
        "header_bytes": HEADER_BYTES,
        "note": (
            "storage_bpw matches GLM52_REAL_ACTIVATION_SWEEP (f16 L,B; no header). "
            "active_two_gemm_f16_bpw equals storage for factored f16 matmul. "
            "active_two_gemm_f32_bpw is the decoded-f32 analogue of the Q80 3.9x gap. "
            "active_reconstructed_f16_bpw=16 if W_hat is materialised dense."
        ),
    }


def rank_for_bpw(rows: int, cols: int, bpw: float) -> int:
    return max(1, int(round(bpw * rows * cols / (F16_BPW * (rows + cols)))))


def exact_input_pcs(X, max_rank: int):
    """Right singular vectors of X via the smaller Gram. Uncentered."""
    import numpy as np

    n, p = X.shape
    if p > n:
        # wide: Gram on rows would be n x n; still may be large. Caller branches.
        raise ValueError("exact_input_pcs expects tall-or-square (n >= p)")
    G = gemm(X.T, X)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    V = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
    s = np.sqrt(evals).astype(np.float32)
    r = min(int(max_rank), V.shape[1])
    return V[:, :r], s[:r], float(evals.sum())


def randomized_input_pcs(X, max_rank: int, niter: int = 2, seed: int = SEED):
    """Randomized range-finder for wide X (post-SwiGLU). Algorithmic randomness
    only; the matrix being factored is real captured (or real SwiGLU) X."""
    import numpy as np
    import torch

    n, p = X.shape
    r = int(min(max_rank + 16, n, p))
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((p, r)).astype(np.float32)
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    Qt = torch.from_numpy(Q)
    Y = Xt @ Qt
    Qout, _ = torch.linalg.qr(Y, mode="reduced")
    for _ in range(niter):
        Qout, _ = torch.linalg.qr(Xt.T @ Qout, mode="reduced")
        Qout, _ = torch.linalg.qr(Xt @ Qout, mode="reduced")
    B = Qout.T @ Xt
    _u, s, Vh = torch.linalg.svd(B, full_matrices=False)
    k = min(int(max_rank), int(Vh.shape[0]))
    V = Vh[:k].T.contiguous().numpy().astype(np.float32)
    s = s[:k].numpy().astype(np.float32)
    fro2 = float((X.astype(np.float64) ** 2).sum())
    return V, s, fro2


def input_pcs(X, max_rank: int, *, centered: bool, seed: int):
    import numpy as np

    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = (X - mu) if centered else X
    n, p = Xc.shape
    if p <= 8192 and n >= p:
        V, s, fro2 = exact_input_pcs(Xc, max_rank)
        method = "exact_gram_eigh"
    else:
        V, s, fro2 = randomized_input_pcs(Xc, max_rank, niter=2, seed=seed)
        method = "randomized_svd_niter2"
    energy = (s.astype(np.float64) ** 2)
    total = float(fro2) + 1e-30
    cum = np.cumsum(energy) / total
    return {
        "V": V,
        "s": s,
        "fro2": total,
        "cum_energy": cum.astype(np.float64),
        "centered": bool(centered),
        "method": method,
        "mean": mu.reshape(-1),
        "n": int(n),
        "p": int(p),
    }


def spectral_stats(s, fro2, ks=(16, 41, 64, 256, 512, 1024)) -> dict:
    import numpy as np

    energy = s.astype(np.float64) ** 2
    total = float(fro2) + 1e-30
    p = energy / (energy.sum() + 1e-30)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    entropy_bits = entropy / np.log(2.0)
    # Participation ratio on the *observed* spectrum. Remaining energy (fro2
    # minus captured) is treated as one extra bin, which raises effective rank
    # toward the truth when the SVD is truncated.
    captured = float(energy.sum())
    remaining = max(total - captured, 0.0)
    e_full = np.concatenate([energy, np.array([remaining])]) if remaining > 0 else energy
    erank = float((e_full.sum() ** 2) / ((e_full ** 2).sum() + 1e-30))
    at = {}
    cum = np.cumsum(energy) / total
    for k in ks:
        if k <= len(cum):
            at[str(k)] = float(cum[k - 1])
        else:
            at[str(k)] = None
    return {
        "n_s": int(len(s)),
        "spectral_entropy_nats": entropy,
        "spectral_entropy_bits": float(entropy_bits),
        "effective_rank_participation": erank,
        "captured_energy_frac": captured / total,
        "remaining_energy_frac": remaining / total,
        "cum_energy_at_rank": at,
        "s0": float(s[0]) if len(s) else None,
        "s_ratio_1_16": float(s[0] / s[15]) if len(s) >= 16 else None,
    }


def aa_reconstruct_Y(X, W, V, rank: int):
    """Activation-aware: W_hat = W @ V_k @ V_k.T; Yh = (X @ V_k) @ (W @ V_k).T."""
    B = V[:, :rank]
    L = gemm(W, B)  # [out, k]
    Z = gemm(X, B)  # [n, k]
    return gemm(Z, L.T)


def weight_pcs(W, max_rank: int):
    """Truncated SVD of W via the smaller Gram. A1_raw_weights."""
    import numpy as np

    out, inn = W.shape
    if inn <= out:
        G = gemm(W.T, W)  # in x in
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        V = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
        s = np.sqrt(evals).astype(np.float32)
        r = min(int(max_rank), V.shape[1])
        return {"side": "right", "V": V[:, :r], "s": s[:r], "fro2": float(evals.sum())}
    G = gemm(W, W.T)  # out x out
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    U = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
    s = np.sqrt(evals).astype(np.float32)
    r = min(int(max_rank), U.shape[1])
    return {"side": "left", "U": U[:, :r], "s": s[:r], "fro2": float(evals.sum())}


def raw_weight_Y(X, W, pcs, rank: int):
    """Yh = X @ W_hat.T with W_hat the rank-k truncated SVD of W."""
    if pcs["side"] == "right":
        B = pcs["V"][:, :rank]
        L = gemm(W, B)
        Z = gemm(X, B)
        return gemm(Z, L.T)
    U = pcs["U"][:, :rank]
    # W_hat = U @ U.T @ W; Yh = X @ W.T @ U @ U.T
    Yfull_U = gemm(x_wt(X, W), U)
    return gemm(Yfull_U, U.T)


def x_recon_cosine(X, V, rank: int) -> float:
    B = V[:, :rank]
    Z = gemm(X, B)
    Xh = gemm(Z, B.T)
    return row_cosine(X, Xh)


def health_of(score: dict, scale: dict, target_cos: float) -> str:
    if scale["gain"] >= GAIN_REJECT_SCALE:
        return "INSTRUMENT_FAIL_scale_trap_not_rejected"
    if score["gain"] < GAIN_HEALTHY_FLOOR:
        return "UNHEALTHY_gain"
    if not score["beats_null"]:
        return "UNHEALTHY_fails_null"
    if score["cosine"] + 1e-12 < target_cos:
        return "UNHEALTHY_below_glm_cosine"
    return "HEALTHY"


def crossover(curve: list, target: float, key="cosine"):
    """Smallest storage_bpw at which `key` reaches `target`, linear in BPW."""
    if not curve:
        return {"reaches": False, "reason": "empty"}
    ordered = sorted(curve, key=lambda p: p["storage_bpw"])
    if ordered[0][key] >= target:
        return {
            "reaches": True,
            "storage_bpw": ordered[0]["storage_bpw"],
            "rank": ordered[0]["rank"],
            "kind": "at_or_below_first_point",
            "value": ordered[0][key],
        }
    for a, b in zip(ordered, ordered[1:]):
        if a[key] < target <= b[key]:
            span = b[key] - a[key]
            t = 0.0 if span == 0 else (target - a[key]) / span
            bpw = a["storage_bpw"] + t * (b["storage_bpw"] - a["storage_bpw"])
            rk = a["rank"] + t * (b["rank"] - a["rank"])
            return {
                "reaches": True,
                "storage_bpw": float(bpw),
                "rank_interpolated": float(rk),
                "kind": "interpolated",
                "lo": {"rank": a["rank"], "bpw": a["storage_bpw"], "value": a[key]},
                "hi": {"rank": b["rank"], "bpw": b["storage_bpw"], "value": b[key]},
            }
    last = ordered[-1]
    return {
        "reaches": False,
        "reason": "does_not_reach_on_ladder",
        "last_rank": last["rank"],
        "last_storage_bpw": last["storage_bpw"],
        "last_value": last[key],
        "target": target,
    }


def split_from_manifest(cap: Path, n_tokens: int):
    import numpy as np

    man_path = cap / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        if man.get("manifest"):
            fit, hold = [], []
            for m in man["manifest"]:
                sl = np.arange(m["row_start"], m["row_start"] + m["n_tokens"])
                (hold if m.get("split") == "hold" else fit).append(sl)
            return (
                np.concatenate(fit),
                np.concatenate(hold),
                man,
                "prompt_hold: last 3 prompts/family",
            )
    n_hold = max(256, n_tokens // 5)
    return (
        np.arange(0, n_tokens - n_hold),
        np.arange(n_tokens - n_hold, n_tokens),
        None,
        "last 20% rows (no prompt manifest)",
    )


def characterize_glm():
    sweep, sweep_src = git_json(GLM_SWEEP_GIT)
    repl, repl_src = git_json(GLM_REPL_GIT)
    pilot, pilot_src = git_json(GLM_PILOT_GIT)
    ortho = None
    if Q80_ORTHO.is_file():
        ortho = json.loads(Q80_ORTHO.read_text())

    glm = {
        "sweep_source": sweep_src,
        "replication_source": repl_src,
        "pilot_source": pilot_src,
        "method": {
            "name": "A1_activation_aware",
            "operator": (
                "Uncentered (v2) / centered (v1 sweep) SVD of REAL captured "
                "input activations X_fit; project W onto the top-k input PCs: "
                "W_hat = (W @ V_k) @ V_k.T. Scored as mean per-row cosine of "
                "Y=X_hold@W.T vs Yh=X_hold@W_hat.T against a constant-mean null."
            ),
            "objective": "function-space ||f(X)-f_hat(X)|| via output cosine; weight reconstruction is INADMISSIBLE",
            "capture": (
                "layer_10/pre_router_hidden from teacher capsule L10_L10.npz; "
                "4096 rows, 820 held out (20%). REAL teacher activations, not Gaussian."
            ),
            "acceptance": (
                "beats constant-mean null on held-out real activations. "
                "e0 up_proj rank-16 f16 = 0.75515 cosine vs null 0.65101 at 0.16667 BPW. "
                "Replicated 12/12 experts at k16; raw-weight rank-64 = 0/12."
            ),
            "billing": (
                "f16 factors, no header: BPW = 16*k*(m+n)/(m*n). "
                "GLM expert up_proj is [2048, 6144]; k=16 → 0.16667."
            ),
            "v1_vs_v2": (
                "GLM52_REAL_ACTIVATION_SWEEP.json and the 12/12 replication used "
                "the v1 packer, whose build_basis centers X. GLM52_BASIS_PILOT later "
                "showed uncentered/explicit-mean lift median +0.082 vs centered and "
                "tied with each other. This lane runs BOTH: centered is the receipt "
                "twin; uncentered is the method we would actually ship."
            ),
            "down_proj_law": (
                "Output-side hidden-basis (or Gaussian probe of the input width) is "
                "forbidden. Real SwiGLU X is the right input; GLM pilot rank-256 "
                "down on matched input-side scored 0.989-0.999 vs 0.36 on the "
                "production output-side negative control."
            ),
        },
        "headline": None,
        "replication": None,
        "pilot_lift_uncentered_over_centered": None,
        "expert_orthogonality_cited": None,
    }
    if sweep:
        pts = { (p["arm"], p.get("rank")): p for p in sweep.get("points", []) }
        aa16 = pts.get(("A1_activation_aware", 16))
        raw16 = pts.get(("A1_raw_weights", 16))
        glm["headline"] = {
            "schema": sweep.get("schema"),
            "at": sweep.get("at"),
            "expert": sweep.get("expert"),
            "activations": sweep.get("activations"),
            "activation_rows": sweep.get("activation_rows"),
            "held_out_rows": sweep.get("held_out_rows"),
            "null": sweep.get("constant_mean_cosine_null_on_real_activations"),
            "aa_k16": aa16,
            "raw_k16": raw16,
        }
    if repl:
        rows = repl.get("rows") or []
        aa = [r["aa_k16_cos"] for r in rows]
        nu = [r["null"] for r in rows]
        glm["replication"] = {
            "schema": repl.get("schema"),
            "at": repl.get("at"),
            "activations": repl.get("activations"),
            "experts_tested": repl.get("experts_tested"),
            "activation_aware_k16_beat_null": repl.get("activation_aware_k16_beat_null"),
            "activation_aware_k64_beat_null": repl.get("activation_aware_k64_beat_null"),
            "raw_weight_rank64_beat_null": repl.get("raw_weight_rank64_beat_null"),
            "aa_k16_cosine_mean": sum(aa) / len(aa) if aa else None,
            "aa_k16_cosine_min": min(aa) if aa else None,
            "aa_k16_cosine_max": max(aa) if aa else None,
            "null_mean": sum(nu) / len(nu) if nu else None,
            "experts": [r["expert"] for r in rows],
            "rows": rows,
        }
    if pilot and isinstance(pilot.get("verdict"), dict):
        v = pilot["verdict"]
        glm["pilot_lift_uncentered_over_centered"] = v.get(
            "mean_median_lift_uncentered_over_centered"
        )
        glm["pilot_story"] = v.get("distinguishing_story")
    if ortho and "components" in ortho:
        glm["expert_orthogonality_cited"] = {
            "receipt": str(Q80_ORTHO.relative_to(ROOT)),
            "layer": ortho.get("layer"),
            "n_experts": ortho.get("n_experts"),
            "gate_proj_pairwise_cosine_mean": (ortho["components"]
                                               .get("gate_proj", {})
                                               .get("pairwise_cosine_mean")),
            "up_proj_pairwise_cosine_mean": (ortho["components"]
                                             .get("up_proj", {})
                                             .get("pairwise_cosine_mean")),
            "note": (
                "NS-010 wording 'experts mutually orthogonal at cosine 0.004' "
                "is this Q80 L10 number, not GLM and not DSV4F. Cited as the "
                "reason to *doubt* transfer, not as a dense measurement."
            ),
        }
    return glm


def point_at_rank(curve, rank: int):
    for p in curve:
        if p["rank"] == rank:
            return p
    return None


def nearest_bpw(curve, bpw: float):
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p["storage_bpw"] - bpw))


def run_organ(layer, organ, W, X_fit, X_hold, *, seed, max_rank, ranks):
    import numpy as np

    t0 = time.time()
    out_f, in_f = int(W.shape[0]), int(W.shape[1])
    assert X_fit.shape[1] == in_f and X_hold.shape[1] == in_f, (
        f"{organ} X width {X_fit.shape[1]} != W.in {in_f}"
    )
    Y_hold = x_wt(X_hold, W)
    null_hold = constant_mean_null(Y_hold)
    act_null_hold = constant_mean_null(X_hold)
    act_null_fit = constant_mean_null(X_fit)

    scale = score_pair(Y_hold, SCALE_TRAP * Y_hold)
    scale["artifact"] = f"{SCALE_TRAP}*Y = X @ ({SCALE_TRAP}*W).T"
    scale["cosine_must_be_one"] = abs(scale["cosine"] - 1.0) < 1e-5
    scale["gain_rejects"] = bool(scale["gain"] < GAIN_REJECT_SCALE)
    scale["instrument_ok"] = bool(scale["cosine_must_be_one"] and scale["gain_rejects"])

    k_match = rank_for_bpw(out_f, in_f, GLM_TARGET_BPW)
    use_ranks = sorted(set(list(ranks) + [k_match, GLM_RANK]))
    use_ranks = [k for k in use_ranks if k <= max_rank and k <= min(X_fit.shape)]
    need = max(use_ranks)

    bases = {}
    for centered, name in ((False, "uncentered"), (True, "centered")):
        bases[name] = input_pcs(
            X_fit, need, centered=centered, seed=seed ^ (1 if centered else 0)
        )
    wpcs = weight_pcs(W, need)

    def pack_curve(arm, fn, V_for_xrecon=None):
        rows = []
        for k in use_ranks:
            Yh = fn(k)
            sc = score_pair(Y_hold, Yh)
            bp = factor_bpw(out_f, in_f, k)
            rec = {
                "arm": arm,
                "rank": int(k),
                "storage_bpw": bp["storage_bpw"],
                "complete_storage_bpw": bp["complete_storage_bpw"],
                "active_two_gemm_f16_bpw": bp["active_two_gemm_f16_bpw"],
                "active_two_gemm_f32_bpw": bp["active_two_gemm_f32_bpw"],
                "active_reconstructed_f16_bpw": bp["active_reconstructed_f16_bpw"],
                **sc,
                "health": health_of(sc, scale, GLM_TARGET_COSINE),
                "matches_glm_rank": k == GLM_RANK,
                "matches_glm_bpw": k == k_match,
            }
            if V_for_xrecon is not None:
                rec["x_recon_cosine_hold"] = x_recon_cosine(X_hold, V_for_xrecon, k)
            rows.append(rec)
            del Yh
        return rows

    curve_u = pack_curve(
        "A1_activation_aware_uncentered",
        lambda k: aa_reconstruct_Y(X_hold, W, bases["uncentered"]["V"], k),
        bases["uncentered"]["V"],
    )
    curve_c = pack_curve(
        "A1_activation_aware_centered",
        lambda k: aa_reconstruct_Y(X_hold, W, bases["centered"]["V"], k),
        bases["centered"]["V"],
    )
    def _raw_Y(k):
        if wpcs["side"] == "right":
            return raw_weight_Y(X_hold, W, wpcs, k)
        U = wpcs["U"][:, :k]
        return gemm(gemm(Y_hold, U), U.T)

    curve_w = pack_curve("A1_raw_weights", _raw_Y)

    # f16 factor round-trip at the BPW-matched rank (receipt billed f16, scored f32)
    k = k_match
    B = bases["uncentered"]["V"][:, :k]
    L = gemm(W, B)
    L16 = L.astype(np.float16).astype(np.float32)
    B16 = B.astype(np.float16).astype(np.float32)
    Yh16 = gemm(gemm(X_hold, B16), L16.T)
    f16_rt = score_pair(Y_hold, Yh16)
    f16_rt["health"] = health_of(f16_rt, scale, GLM_TARGET_COSINE)

    struct = {
        "activation_uncentered": spectral_stats(bases["uncentered"]["s"], bases["uncentered"]["fro2"]),
        "activation_centered": spectral_stats(bases["centered"]["s"], bases["centered"]["fro2"]),
        "weight": spectral_stats(wpcs["s"], wpcs["fro2"]),
        "basis_method_uncentered": bases["uncentered"]["method"],
        "basis_method_centered": bases["centered"]["method"],
        "weight_svd_side": wpcs["side"],
    }
    # drop bulky mean vector from serialised basis
    for b in bases.values():
        b.pop("mean", None)
        b.pop("V", None)

    out = {
        "layer": int(layer),
        "organ": organ,
        "tensor": tensor_name(layer, organ),
        "W_shape": [out_f, in_f],
        "site": "post_swiglu" if organ == "down_proj" else "post_attn_norm",
        "n_fit": int(X_fit.shape[0]),
        "n_hold": int(X_hold.shape[0]),
        "null_output_hold": null_hold,
        "null_activation_hold": act_null_hold,
        "null_activation_fit": act_null_fit,
        "rank_at_glm_bpw": int(k_match),
        "scale_trap_001W": scale,
        "curves": {
            "A1_activation_aware_uncentered": curve_u,
            "A1_activation_aware_centered": curve_c,
            "A1_raw_weights": curve_w,
        },
        "f16_roundtrip_at_glm_bpw": {"rank": int(k), **f16_rt, **factor_bpw(out_f, in_f, k)},
        "crossover_cosine_0.755": {
            "uncentered": crossover(curve_u, GLM_TARGET_COSINE),
            "centered": crossover(curve_c, GLM_TARGET_COSINE),
            "raw_weights": crossover(curve_w, GLM_TARGET_COSINE),
        },
        "crossover_beats_glm_surplus": {
            "glm_surplus": 0.75515 - 0.65101,
            "uncentered": crossover(curve_u, 0.75515 - 0.65101, key="surplus_over_null"),
        },
        "point_glm_bpw_uncentered": nearest_bpw(curve_u, GLM_TARGET_BPW),
        "point_glm_rank_uncentered": point_at_rank(curve_u, GLM_RANK),
        "point_glm_bpw_centered": nearest_bpw(curve_c, GLM_TARGET_BPW),
        "point_glm_rank_centered": point_at_rank(curve_c, GLM_RANK),
        "point_glm_bpw_raw": nearest_bpw(curve_w, GLM_TARGET_BPW),
        "structure": struct,
        "wall_s": time.time() - t0,
    }
    del Y_hold, Yh16, L, L16, B, B16
    return out


def decide(organs_out: list, glm: dict, scale_ok: bool) -> dict:
    pts = []
    for o in organs_out:
        p = o.get("point_glm_bpw_uncentered")
        if p:
            pts.append({**p, "layer": o["layer"], "organ": o["organ"], "null": o["null_output_hold"]})
    worst = min(pts, key=lambda p: p["cosine"]) if pts else None
    crosses = []
    for o in organs_out:
        c = o["crossover_cosine_0.755"]["uncentered"]
        crosses.append({"layer": o["layer"], "organ": o["organ"], **c})
    any_miss = [c for c in crosses if not c.get("reaches")]
    max_cross_bpw = None
    if not any_miss and crosses:
        max_cross_bpw = max(c["storage_bpw"] for c in crosses)

    nogo = []
    go = []
    if not scale_ok:
        nogo.append("instrument failed to reject 0.01*W")
    if worst is None:
        nogo.append("no 0.167-BPW point measured")
    else:
        if worst["cosine"] + 1e-12 < GLM_TARGET_COSINE:
            nogo.append(
                f"worst dense cosine at ~0.167 BPW is {worst['cosine']:.4f} "
                f"(L{worst['layer']} {worst['organ']}) < GLM 0.755"
            )
        else:
            go.append(
                f"worst dense cosine at ~0.167 BPW is {worst['cosine']:.4f} >= 0.755"
            )
        if not worst["beats_null"]:
            nogo.append(
                f"L{worst['layer']} {worst['organ']} cosine {worst['cosine']:.4f} "
                f"FAILS its own null {worst['null']:.4f}"
            )
        if worst.get("health") != "HEALTHY":
            nogo.append(f"health at 0.167 is {worst.get('health')}")
    if any_miss:
        nogo.append(
            "cosine 0.755 is not reached on the rank ladder for "
            + ", ".join(f"L{c['layer']} {c['organ']} (last {c.get('last_storage_bpw')})"
                        for c in any_miss)
        )
    elif max_cross_bpw is not None and max_cross_bpw >= 1.0:
        nogo.append(
            f"all organs reach 0.755 but the last one to do so is at "
            f"{max_cross_bpw:.3f} storage BPW — not fractional-bit"
        )

    decision = "NO-GO" if nogo else "GO"
    deciding = None
    meaning = None
    if worst is not None:
        deciding = worst["cosine"]
        meaning = (
            f"worst held-out uncentered-AA cosine at the dense rank whose "
            f"f16-factor storage BPW matches GLM 0.16667 "
            f"(L{worst['layer']} {worst['organ']}, rank {worst['rank']}, "
            f"storage_bpw {worst['storage_bpw']:.5f}, null {worst['null']:.4f}, "
            f"surplus {worst['surplus_over_null']:+.4f}, health {worst['health']}). "
            f"GLM e0 was 0.75515 vs null 0.65101 at rank 16."
        )
    return {
        "decision": decision,
        "deciding_number": deciding,
        "deciding_number_meaning": meaning,
        "go_reasons": go,
        "nogo_reasons": nogo,
        "worst_at_glm_bpw": worst,
        "crossover_0.755_by_organ": crosses,
        "max_crossover_storage_bpw": max_cross_bpw,
        "fractional_bit_means": "storage_bpw < 1.0 on f16 factors, health paired",
        "local_not_composed": (
            "This is an organ-local screen. A local win is not a composed win "
            "(students have diverged by layers 4-8 in all 40 layers despite "
            "acceptable local fits). No hop, no residual product, no generation."
        ),
    }


def watched_fail(organs_out, glm, verdict, scale_global) -> list:
    out = []
    out.append(
        {
            "what": "Gaussian / synthetic-X evaluation",
            "result": "REFUSED",
            "why": (
                "Every prior sub-bit negative here was a Gaussian-proxy artifact. "
                "X is capture_diverse2 post_attn_norm (real BF16 parent forward). "
                "down_proj X is silu(X@Wg.T)*(X@Wu.T) from the same parent. "
                "The GLM packer's output-side Gaussian probe for down_proj is not used."
            ),
        }
    )
    out.append(
        {
            "what": "cosine as a GO metric on 0.01*W",
            "result": (
                f"cosine={scale_global['cosine']:.6f} gain={scale_global['gain']:.6f} "
                f"{'REJECTED by gain' if scale_global.get('gain_rejects') else 'NOT REJECTED'}"
            ),
            "why": (
                "Cosine is scale-invariant. 0.01*W scored 1.000000 on every cosine axis "
                "for an entire campaign. Gain must reject it; this run exhibits that."
            ),
        }
    )
    if glm.get("headline") and glm["headline"].get("raw_k16"):
        raw = glm["headline"]["raw_k16"]
        aa = glm["headline"]["aa_k16"]
        out.append(
            {
                "what": "raw-weight low-rank at the GLM 0.167 BPW point",
                "result": (
                    f"GLM e0 raw k16 cosine={raw['cosine']} beats_null={raw['beats_null']}; "
                    f"AA k16 cosine={aa['cosine']} beats_null={aa['beats_null']}"
                ),
                "why": (
                    "Fitted-to-weights is the A2/A3 framing the 2026-07-27 result "
                    "refuted on experts. Dense transfer must not quietly revert to it."
                ),
            }
        )
    for o in organs_out:
        raw_pt = o.get("point_glm_bpw_raw")
        aa_pt = o.get("point_glm_bpw_uncentered")
        if raw_pt and aa_pt:
            out.append(
                {
                    "what": f"L{o['layer']} {o['organ']} raw-weight vs AA at 0.167 BPW",
                    "result": (
                        f"raw cosine={raw_pt['cosine']:.4f} beats_null={raw_pt['beats_null']}; "
                        f"AA cosine={aa_pt['cosine']:.4f} beats_null={aa_pt['beats_null']} "
                        f"health={aa_pt['health']}"
                    ),
                    "why": "Same budget, two objectives. Weight-SVD is the control the GLM receipt used.",
                }
            )
        c_pt = o.get("point_glm_bpw_centered")
        if c_pt and aa_pt:
            out.append(
                {
                    "what": f"L{o['layer']} {o['organ']} centered (receipt twin) vs uncentered",
                    "result": (
                        f"centered cosine={c_pt['cosine']:.4f}; "
                        f"uncentered cosine={aa_pt['cosine']:.4f}; "
                        f"Δ={aa_pt['cosine']-c_pt['cosine']:+.4f}"
                    ),
                    "why": (
                        "v1 sweep centered X; v2 pilot found centering is a defect "
                        "(median lift +0.082). Transfer reports both."
                    ),
                }
            )
        if aa_pt and aa_pt["cosine"] < o["null_output_hold"]:
            out.append(
                {
                    "what": f"L{o['layer']} {o['organ']} GLM-number 0.755 against THIS organ's null",
                    "result": (
                        f"AA cosine={aa_pt['cosine']:.4f} vs null={o['null_output_hold']:.4f} "
                        f"(activation null={o['null_activation_hold']:.4f})"
                    ),
                    "why": (
                        "Raw activation cosine null on this family sits near 0.898. "
                        "A number that beat GLM's 0.651 null can fail a denser organ's null. "
                        "Every fidelity number is reported against its own null."
                    ),
                }
            )
        cr = o["crossover_cosine_0.755"]["uncentered"]
        if not cr.get("reaches"):
            out.append(
                {
                    "what": f"L{o['layer']} {o['organ']} hunt for the 0.755 cosine",
                    "result": (
                        f"does not reach 0.755 by rank {cr.get('last_rank')} "
                        f"(storage_bpw={cr.get('last_storage_bpw')}, "
                        f"cosine={cr.get('last_value')})"
                    ),
                    "why": "The transfer question is not only 'how good at 0.167' but 'when, if ever'.",
                }
            )
        st = o["structure"]["activation_uncentered"]
        out.append(
            {
                "what": f"L{o['layer']} {o['organ']} activation effective rank vs GLM k=16",
                "result": (
                    f"eff_rank={st['effective_rank_participation']:.1f} "
                    f"entropy_bits={st['spectral_entropy_bits']:.2f} "
                    f"energy@16={st['cum_energy_at_rank'].get('16')} "
                    f"energy@41={st['cum_energy_at_rank'].get('41')} "
                    f"energy@256={st['cum_energy_at_rank'].get('256')}"
                ),
                "why": (
                    "Hypothesis, not conclusion: a dense organ carries the whole "
                    "distribution and is higher-entropy than a routed expert. "
                    "This is the number that shows it on this capture."
                ),
            }
        )
        if o["layer"] == 0 and o["organ"] == "up_proj":
            # composition trap: L0 looking better than L31
            pass
    l0 = [o for o in organs_out if o["layer"] == 0]
    l31 = [o for o in organs_out if o["layer"] == 31]
    if l0 and l31:
        c0 = min(o["point_glm_bpw_uncentered"]["cosine"] for o in l0)
        c31 = min(o["point_glm_bpw_uncentered"]["cosine"] for o in l31)
        out.append(
            {
                "what": "depth split (local win is not a composed win)",
                "result": f"worst L0 cosine@0.167={c0:.4f}; worst L31={c31:.4f}",
                "why": (
                    "A student once diverged by layers 4-8 in all 40 layers despite "
                    "acceptable local fits. Early-layer AA looking less bad does not "
                    "authorise a stack."
                ),
            }
        )
    down = [o for o in organs_out if o["organ"] == "down_proj"]
    up = [o for o in organs_out if o["organ"] == "up_proj"]
    if down and up:
        out.append(
            {
                "what": "organ-type split (down_proj inverts the family ranking)",
                "result": (
                    "up  "
                    + ", ".join(
                        f"L{o['layer']}@{o['point_glm_bpw_uncentered']['cosine']:.4f}"
                        for o in up
                    )
                    + "  down "
                    + ", ".join(
                        f"L{o['layer']}@{o['point_glm_bpw_uncentered']['cosine']:.4f}"
                        for o in down
                    )
                ),
                "why": (
                    "down_proj is known to invert the usual ranking on this family "
                    "(low-rank beats binary; post-SwiGLU X is the right input). "
                    "Two organ types were required because they may differ sharply."
                ),
            }
        )
    if verdict["decision"] == "NO-GO":
        out.append(
            {
                "what": "fractional-bit GO for dense organs",
                "result": "NO-GO",
                "why": "; ".join(verdict["nogo_reasons"]),
            }
        )
    return out


def print_report(doc: dict) -> None:
    print("DENSE SUB-BIT TRANSFER")
    print("=" * 72)
    print(f"git_head: {doc['git_head']}")
    print(f"python:   {doc['python']}")
    print(f"parent:   {doc['parent']}")
    print(f"capture:  {doc['capture']['path']}")
    print(f"torch:    {doc['torch']}")
    print()

    print("## GLM RESULT (characterized from receipt, not re-run)")
    glm = doc["glm"]
    h = glm.get("headline") or {}
    r = glm.get("replication") or {}
    print(f"  sweep:  {glm['sweep_source']}")
    print(f"  repl:   {glm['replication_source']}")
    if h:
        print(f"  expert: {h.get('expert')}")
        print(f"  X:      {h.get('activations')}  rows={h.get('activation_rows')} hold={h.get('held_out_rows')}")
        print(f"  null:   {h.get('null')}")
        aa, raw = h.get("aa_k16") or {}, h.get("raw_k16") or {}
        print(
            f"  AA  k16 bpw={aa.get('bpw')} cosine={aa.get('cosine')} "
            f"beats_null={aa.get('beats_null')}"
        )
        print(
            f"  RAW k16 bpw={raw.get('bpw')} cosine={raw.get('cosine')} "
            f"beats_null={raw.get('beats_null')}"
        )
    if r:
        print(
            f"  12/12:  AA k16 beat null {r.get('activation_aware_k16_beat_null')} "
            f"mean cosine={r.get('aa_k16_cosine_mean')} "
            f"min={r.get('aa_k16_cosine_min')}  "
            f"raw r64 beat null {r.get('raw_weight_rank64_beat_null')}"
        )
    print(f"  method: {glm['method']['name']}")
    print(f"          {glm['method']['operator'][:200]}")
    print()

    print("## METHOD ON DENSE QWEN3.8")
    print(f"  organs:  {doc['organs']}  layers: {doc['layers']}")
    print(f"  ranks:   {doc['ranks']}")
    print(f"  k@0.167: {doc['rank_at_glm_bpw_dense_matrix']}  "
          f"(16*k*(H+I)/(H*I) for H={HIDDEN} I={INTERMEDIATE})")
    print(f"  X site:  post_attn_norm (gate/up); real SwiGLU intermediate (down)")
    print(f"  split:   {doc['capture']['split_rule']}  "
          f"fit={doc['capture']['n_fit']} hold={doc['capture']['n_hold']}")
    print(f"  no Gaussian scoring path; llama-server not opened")
    print()

    print("## SCALE TRAP (must REJECT 0.01*W)")
    st = doc["scale_trap_global"]
    print(
        f"  0.01*W  cosine={st['cosine']:.6f}  gain={st['gain']:.6f}  "
        f"rel_fro={st['rel_fro']:.6f}  "
        f"{'REJECTED' if st.get('gain_rejects') else 'NOT REJECTED'}"
    )
    print()

    print("## DENSE FIDELITY CURVE")
    hdr = (
        f"  {'L':>2} {'organ':<9} {'arm':<12} {'k':>4} {'stor':>7} {'act16':>7} "
        f"{'cos':>7} {'null':>7} {'surp':>7} {'gain':>6} {'rel':>6} {'health'}"
    )
    print(hdr)
    for o in doc["organs_out"]:
        for arm in (
            "A1_activation_aware_uncentered",
            "A1_activation_aware_centered",
            "A1_raw_weights",
        ):
            short = {
                "A1_activation_aware_uncentered": "AA_uncent",
                "A1_activation_aware_centered": "AA_center",
                "A1_raw_weights": "raw_Wsvd",
            }[arm]
            for p in o["curves"][arm]:
                mark = ""
                if p.get("matches_glm_bpw"):
                    mark = " <--0.167"
                elif p.get("matches_glm_rank"):
                    mark = " <--k16"
                print(
                    f"  {o['layer']:>2} {o['organ']:<9} {short:<12} {p['rank']:>4} "
                    f"{p['storage_bpw']:>7.4f} {p['active_two_gemm_f16_bpw']:>7.4f} "
                    f"{p['cosine']:>7.4f} {p['null']:>7.4f} {p['surplus_over_null']:>+7.4f} "
                    f"{p['gain']:>6.3f} {p['rel_fro']:>6.3f} {p['health']}{mark}"
                )
        print(
            f"       activation_null_hold={o['null_activation_hold']:.4f}  "
            f"output_null_hold={o['null_output_hold']:.4f}  "
            f"f16_roundtrip_cos={o['f16_roundtrip_at_glm_bpw']['cosine']:.4f}"
        )
    print()

    print("## CROSSOVER TO GLM 0.755")
    for o in doc["organs_out"]:
        c = o["crossover_cosine_0.755"]["uncentered"]
        if c.get("reaches"):
            print(
                f"  L{o['layer']} {o['organ']}: REACHES at storage_bpw={c['storage_bpw']:.4f} "
                f"({c['kind']})"
            )
        else:
            print(
                f"  L{o['layer']} {o['organ']}: DOES NOT REACH by rank {c.get('last_rank')} "
                f"storage_bpw={c.get('last_storage_bpw')} cosine={c.get('last_value')}"
            )
    print()

    print("## STRUCTURE (why dense may differ)")
    for o in doc["organs_out"]:
        s = o["structure"]["activation_uncentered"]
        w = o["structure"]["weight"]
        print(
            f"  L{o['layer']} {o['organ']} site={o['site']}  "
            f"X_eff_rank={s['effective_rank_participation']:.1f} "
            f"X_entropy_bits={s['spectral_entropy_bits']:.2f} "
            f"X_E@16={s['cum_energy_at_rank'].get('16')} "
            f"X_E@41={s['cum_energy_at_rank'].get('41')} "
            f"X_E@256={s['cum_energy_at_rank'].get('256')}"
        )
        print(
            f"           W_eff_rank={w['effective_rank_participation']:.1f} "
            f"W_entropy_bits={w['spectral_entropy_bits']:.2f} "
            f"W_E@16={w['cum_energy_at_rank'].get('16')} "
            f"W_E@41={w['cum_energy_at_rank'].get('41')}"
        )
    if glm.get("expert_orthogonality_cited"):
        e = glm["expert_orthogonality_cited"]
        print(
            f"  cited Q80 expert pairwise cosine: gate={e['gate_proj_pairwise_cosine_mean']} "
            f"up={e['up_proj_pairwise_cosine_mean']} ({e['receipt']})"
        )
    print()

    v = doc["verdict"]
    print("## VERDICT")
    print(f"  {v['decision']}")
    print(f"  deciding_number: {v['deciding_number']}")
    print(f"  meaning: {v['deciding_number_meaning']}")
    for rsn in v["nogo_reasons"]:
        print(f"  NO-GO: {rsn}")
    for rsn in v["go_reasons"]:
        print(f"  GO:    {rsn}")
    print(f"  {v['local_not_composed']}")
    print()

    print("## WHAT I WATCHED FAIL")
    for i, w in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {w['what']}: {w['result']}")
        print(f"      {w['why']}")
    print()
    print(f"wrote: {doc['written_to']}")
    print(f"wall_s: {doc['wall_s']:.1f}")


def main() -> int:
    _ensure_torch()
    import numpy as np
    import torch

    torch.set_num_threads(min(16, os.cpu_count() or 8))
    t_all = time.time()
    print("DENSE SUB-BIT TRANSFER")
    print("=" * 72)
    head = git_head()
    print(f"git_head: {head}")
    print(f"python:   {sys.executable}")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print(f"torch:    {torch.__version__} mps={mps} threads={torch.get_num_threads()}")

    parent = find_parent()
    cap = find_capture()
    print(f"parent:   {parent}")
    print(f"capture:  {cap}")
    print("teacher:  qualified parent BF16; no llama-server; no second 27B")
    print()

    glm = characterize_glm()
    print("## GLM RESULT (from receipt)")
    if glm.get("headline"):
        h = glm["headline"]
        aa, raw = h.get("aa_k16") or {}, h.get("raw_k16") or {}
        print(f"  {h.get('expert')}  {h.get('activations')}")
        print(
            f"  AA k16 cosine={aa.get('cosine')} bpw={aa.get('bpw')} "
            f"null={h.get('null')} beats_null={aa.get('beats_null')}"
        )
        print(
            f"  RAW k16 cosine={raw.get('cosine')} beats_null={raw.get('beats_null')}"
        )
    if glm.get("replication"):
        r = glm["replication"]
        print(
            f"  replication {r.get('activation_aware_k16_beat_null')} "
            f"mean={r.get('aa_k16_cosine_mean')}"
        )
    print()

    X0 = load_X(cap, LAYERS[0])
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx, man, split_rule = split_from_manifest(cap, n_tokens)
    print(
        f"CAPTURE  tokens={n_tokens} fit={len(fit_idx)} hold={len(hold_idx)} "
        f"split={split_rule}"
    )
    k_dense = rank_for_bpw(INTERMEDIATE, HIDDEN, GLM_TARGET_BPW)
    print(f"dense rank @ {GLM_TARGET_BPW} BPW = {k_dense} "
          f"(up/gate/down all {HIDDEN}x{INTERMEDIATE} or transpose)")
    print()

    organs_out = []
    scale_global = None
    for layer in LAYERS:
        print(f"-- layer {layer} --")
        X = load_X(cap, layer)
        if X.shape[0] != n_tokens:
            raise ValueError(f"L{layer} rows {X.shape[0]} != {n_tokens}")
        print(f"  loading W gate/up/down from parent...")
        Wg = load_tensor(parent, tensor_name(layer, "gate_proj"))
        Wu = load_tensor(parent, tensor_name(layer, "up_proj"))
        Wd = load_tensor(parent, tensor_name(layer, "down_proj"))
        print(f"  Wg {Wg.shape} Wu {Wu.shape} Wd {Wd.shape}")
        print(f"  computing real post-SwiGLU X (silu(gate)*up)...")
        t_s = time.time()
        S = swiglu_intermediate(X, Wg, Wu)
        print(f"  post_swiglu {S.shape} in {time.time()-t_s:.1f}s")
        X_fit, X_hold = X[fit_idx], X[hold_idx]
        S_fit, S_hold = S[fit_idx], S[hold_idx]
        del X, S

        jobs = [
            ("up_proj", Wu, X_fit, X_hold),
            ("down_proj", Wd, S_fit, S_hold),
        ]
        for organ, W, Xin_f, Xin_h in jobs:
            print(f"  {organ} ...")
            rec = run_organ(
                layer,
                organ,
                W,
                Xin_f,
                Xin_h,
                seed=SEED ^ (layer * 1009) ^ ORGAN_SEED[organ],
                max_rank=MAX_RANK,
                ranks=RANKS,
            )
            organs_out.append(rec)
            if scale_global is None:
                scale_global = rec["scale_trap_001W"]
            p = rec["point_glm_bpw_uncentered"]
            print(
                f"    AA_uncent k={p['rank']} stor={p['storage_bpw']:.4f} "
                f"act16={p['active_two_gemm_f16_bpw']:.4f} "
                f"cos={p['cosine']:.4f} null={p['null']:.4f} "
                f"surp={p['surplus_over_null']:+.4f} gain={p['gain']:.3f} "
                f"{p['health']}  ({rec['wall_s']:.1f}s)"
            )
            cr = rec["crossover_cosine_0.755"]["uncentered"]
            if cr.get("reaches"):
                print(f"    crossover 0.755 @ storage_bpw={cr['storage_bpw']:.4f}")
            else:
                print(
                    f"    crossover 0.755 NOT reached "
                    f"(last k={cr.get('last_rank')} cos={cr.get('last_value')})"
                )
        del Wg, Wu, Wd, X_fit, X_hold, S_fit, S_hold

    scale_ok = bool(scale_global and scale_global.get("instrument_ok"))
    verdict = decide(organs_out, glm, scale_ok)
    fails = watched_fail(organs_out, glm, verdict, scale_global)

    results = {
        "schema": "hawking.headless.dense_subbit_transfer.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "python": sys.executable,
        "torch": f"{torch.__version__} mps={mps}",
        "parent": str(parent),
        "artifact_under_study": str(Path.home() / "models/qwen38-gravity-uniform-q4-v1"),
        "question": (
            "Fractional-bit activation-aware fitting reached 0.755 cosine at "
            "0.167 BPW on 12/12 GLM experts. Does that transfer to dense Qwen3.8 organs?"
        ),
        "glm": glm,
        "capture": {
            "path": str(cap),
            "site_gate_up": "post_attn_norm",
            "site_down": "real silu(X@Wg.T)*(X@Wu.T) from qualified-parent BF16",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "split_rule": split_rule,
            "manifest_families": (man or {}).get("families"),
            "not_gaussian": True,
            "not_llama_server": True,
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward. "
                "Not Gaussian. Not Q5_K llama-server."
            ),
        },
        "organs": list(ORGANS),
        "layers": list(LAYERS),
        "ranks": list(RANKS),
        "rank_at_glm_bpw_dense_matrix": k_dense,
        "accounting": {
            "factor_storage_bpw_formula": "16*k*(m+n)/(m*n)  # GLM sweep, f16, no header",
            "complete_storage_adds_header_bytes": HEADER_BYTES,
            "active_two_gemm_f16_equals_storage": True,
            "active_two_gemm_f32_is_2x": True,
            "active_reconstructed_f16_bpw": F16_BPW,
            "q80_storage_vs_active_cited": {"storage": 0.6462, "active": 2.518, "factor": 3.9},
            "note": "Report both storage and active, or neither.",
        },
        "scale_trap_global": scale_global,
        "organs_out": organs_out,
        "verdict": verdict,
        "what_i_watched_fail": fails,
        "wall_s": None,
        "written_to": str(OUT_PATH),
    }
    results["wall_s"] = time.time() - t_all
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j(results), indent=2) + "\n")
    tmp.replace(OUT_PATH)
    print()
    print_report(results)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
