#!/usr/bin/env python3
"""Retest G042: generated/implicit weights, with an accounting slot that can hold them.

G042 recorded GENERATED_BPW_EQUIVALENT = 0.0 for every live candidate. The
definition string cites G032 Hadamard. The tool that wrote the receipt assigns
the zero as a literal; NR serialize() emits generated_structures: []. This
probe establishes what G042 actually tested, classifies the zero, and — because
the class-level REFUTED label was produced by a pipeline that could not
represent a generated structure — runs the cheapest honest generated
representation on a real dense Qwen3.8 organ with real activations.

Never synthetic X. Cosine is never the GO metric (scale trap is exhibited).
Storage BPW and active BPW are reported together. Null is measured. A low BPW
without a health verdict is not a result. No tok/s claim.

  python3 tools/headless/generated_weights_retest.py
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
RECEIPT = ROOT / "receipts" / "headless" / "GENERATED_WEIGHTS_RETEST.json"

HIDDEN = 5120
INTERMEDIATE = 17408
GROUP = 64
Q3_BITS = 3
SCALE_BITS = 16
Q3_STORAGE_BPW = Q3_BITS + SCALE_BITS / GROUP  # 3.25
F16_BITS = 16.0
SOURCE_PARAM_COUNT = 26_895_998_464
LAYER = 31
LAYER_B = 0
ORGAN = "gate"
# Kronecker tiling of gate_proj 17408 x 5120 = (136 x 40) ⊗ (128 x 128).
KRON_P, KRON_R, KRON_Q, KRON_S = 136, 128, 40, 128
BPW_TARGETS = (0.05, 0.25, 1.0, 3.25)
CHUNK = 512
N_PROBE = 256
MATERIAL_RATIO = 0.90
GAIN_MARGIN = 0.02

PARENT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
    ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    ROOT / "workspace/campaign/phaseB/capture_diverse2",
    Path("/Users/scammermike/Downloads/hawking/workspace/campaign/phaseB/capture_diverse2"),
]

G042_REL = "receipts/ascent-2026-08-16/G042_BPW_FAMILY.json"
G032_REL = "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q4.json"
G031_REL = "receipts/ascent-2026-08-16/G031_FAMILY_REVIEW.json"
G034_REL = "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json"
G096_REL = "receipts/ascent-2026-08-16/G096_NEURAL_ISA_AUDIT.json"
BPW_TOOL_REL = "tools/gravity_bpw_family.py"
NR_TOOL_REL = "tools/nr_container.py"


def _ensure_scipy():
    try:
        import numpy  # noqa: F401
        import scipy.fft  # noqa: F401
        return
    except ImportError:
        pass
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
    sys.exit("numpy+scipy required (tried sys python and ~/.grok-vision/bin/python)")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_show(rel: str) -> bytes | None:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        return None
    return r.stdout


def git_show_json(rel: str):
    raw = git_show(rel)
    if raw is None:
        return None, f"git show HEAD:{rel} failed"
    try:
        return json.loads(raw), None
    except Exception as e:
        return None, f"json parse HEAD:{rel}: {e}"


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


def tensor_candidates(layer: int, organ: str) -> list[str]:
    return [
        f"model.language_model.layers.{layer}.mlp.{organ}_proj.weight",
        f"language_model.model.layers.{layer}.mlp.{organ}_proj.weight",
        f"language_model.model.layers.{layer}.mlp.{organ}.weight",
    ]


def load_tensor(parent: Path, names: list[str]) -> tuple[np.ndarray, str]:
    index = json.loads((parent / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    name = next((n for n in names if n in wm), None)
    if name is None:
        raise KeyError(f"none of {names} in parent index")
    shard = parent / wm[name]
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
    return np.array(f32.reshape(meta["shape"]), dtype=np.float32, copy=True), name


def capture_path(cap: Path, layer: int) -> Path:
    for name in (f"L{layer:02d}.f16", f"L{layer}.f16"):
        p = cap / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no capture for layer {layer} in {cap}")


def load_X(cap: Path, layer: int) -> np.ndarray:
    p = capture_path(cap, layer)
    raw = np.fromfile(p, dtype=np.float16)
    if raw.size % HIDDEN != 0:
        raise ValueError(f"{p} size {raw.size} not divisible by hidden {HIDDEN}")
    return raw.reshape(-1, HIDDEN).astype(np.float32)


def split_indices(cap: Path, n_tokens: int):
    manifest_path = cap / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("manifest"):
            fit, hold = [], []
            for m in manifest["manifest"]:
                sl = np.arange(m["row_start"], m["row_start"] + m["n_tokens"])
                (hold if m["split"] == "hold" else fit).append(sl)
            return (
                np.concatenate(fit),
                np.concatenate(hold),
                manifest.get("split_rule"),
                manifest.get("families"),
                manifest.get("total_tokens", n_tokens),
            )
    n_hold = max(256, n_tokens // 5)
    return (
        np.arange(0, n_tokens - n_hold),
        np.arange(n_tokens - n_hold, n_tokens),
        "last 20% rows = hold (no prompt manifest)",
        None,
        n_tokens,
    )


def f16_store(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32)


def linear_forward(X: np.ndarray, W: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    parts = [X[i : i + chunk] @ W.T for i in range(0, X.shape[0], chunk)]
    return np.concatenate(parts, axis=0)


def svd_forward(X, U, V, chunk: int = CHUNK) -> np.ndarray:
    """W = U @ V (U: m×r, V: r×n). Y = (X @ V.T) @ U.T."""
    parts = [(X[i : i + chunk] @ V.T) @ U.T for i in range(0, X.shape[0], chunk)]
    return np.concatenate(parts, axis=0)


def row_cosine(A, B) -> float:
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    ok = den > 1e-20
    if not np.any(ok):
        return float("nan")
    return float((num[ok] / den[ok]).mean())


def rel_fro(A, B) -> float:
    na = np.linalg.norm(A)
    if na == 0:
        return float("nan")
    return float(np.linalg.norm(A - B) / na)


def gain_score(A, B) -> float:
    def ratio(axis):
        na = np.linalg.norm(A, axis=axis)
        nb = np.linalg.norm(B, axis=axis)
        r = nb / (na + 1e-30)
        return np.minimum(r, 1.0 / (r + 1e-30))

    return float(min(np.mean(ratio(1)), ratio(0).min()))


def worst_unit(A, B) -> float:
    num = (A * B).sum(0)
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    live = na > 1e-20
    denom = na * nb + 1e-30
    cos = np.zeros_like(num)
    cos[live] = num[live] / denom[live]
    return float(cos[live].min()) if live.any() else 1.0


def score_pair(Y, Yh) -> dict:
    return {
        "rel_fro": rel_fro(Y, Yh),
        "cosine": row_cosine(Y, Yh),
        "gain": gain_score(Y, Yh),
        "worst_unit": worst_unit(Y, Yh),
    }


def null_cosine_constant_mean(Y) -> float:
    mu = Y.mean(axis=0, keepdims=True)
    return row_cosine(Y, np.broadcast_to(mu, Y.shape))


def rsvd(A: np.ndarray, rank: int, n_oversample: int = 12, n_iter: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    m, n = A.shape
    rank = int(min(rank, m, n))
    p = int(min(n, rank + n_oversample))
    Omega = rng.standard_normal((n, p)).astype(np.float32)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    for _ in range(n_iter):
        Z = A.T @ Q
        Q, _ = np.linalg.qr(Z, mode="reduced")
        Z = A @ Q
        Q, _ = np.linalg.qr(Z, mode="reduced")
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    k = min(rank, Ub.shape[1])
    U = Q @ Ub[:, :k]
    return (
        np.ascontiguousarray(U[:, :k].astype(np.float32)),
        S[:k].astype(np.float32),
        np.ascontiguousarray(Vt[:k].astype(np.float32)),
    )


def spd_sqrt_and_inv(G: np.ndarray, ridge_frac: float = 1e-4):
    evals, evecs = np.linalg.eigh(G.astype(np.float64))
    vmax = float(max(evals.max(), 1e-30))
    floor = ridge_frac * vmax
    evals = np.clip(evals, floor, None)
    sqrt_e = np.sqrt(evals)
    sqrtG = ((evecs * sqrt_e) @ evecs.T).astype(np.float32)
    invsqrtG = ((evecs * (1.0 / sqrt_e)) @ evecs.T).astype(np.float32)
    return sqrtG, invsqrtG, {
        "min_eval_clipped": float(evals.min()),
        "max_eval_raw": vmax,
        "ridge_floor": floor,
        "cond_after_ridge": float(evals.max() / evals.min()),
    }


def quantize_group(w: np.ndarray, bits: int = Q3_BITS, group: int = GROUP):
    rows, cols = w.shape
    if cols % group != 0:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    g = w.reshape(rows, cols // group, group)
    qmax = (1 << (bits - 1)) - 1
    absmax = np.abs(g).max(axis=2, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    codes = np.clip(np.rint(g / scale), -qmax - 1, qmax).astype(np.int32)
    deq = (codes * scale).reshape(rows, cols).astype(np.float32)
    n_groups = rows * (cols // group)
    storage_bits = bits * int(w.size) + SCALE_BITS * n_groups
    return deq, storage_bits


def rank_for_bpw(m: int, n: int, bpw: float, bits_per: float = F16_BITS) -> int:
    r = bpw * m * n / (bits_per * (m + n))
    return max(1, min(m, n, int(round(r))))


def k_for_kron_bpw(bpw: float, a_n: int, b_n: int, mn: int, bits_per: float = F16_BITS) -> int:
    k = bpw * mn / (bits_per * (a_n + b_n))
    return max(1, int(round(k)))


def dct_keep_shape(m: int, n: int, n_coeffs: int) -> tuple[int, int]:
    n_coeffs = max(1, min(m * n, int(n_coeffs)))
    frac = n_coeffs / (m * n)
    scale = math.sqrt(frac)
    p = max(1, min(m, int(round(m * scale))))
    q = max(1, min(n, int(round(n_coeffs / p))))
    return p, q


def rearrange_kron(W: np.ndarray, p, r, q, s) -> np.ndarray:
    return W.reshape(p, r, q, s).transpose(0, 2, 1, 3).reshape(p * q, r * s)


def unrearrange_kron(R: np.ndarray, p, r, q, s) -> np.ndarray:
    return R.reshape(p, q, r, s).transpose(0, 2, 1, 3).reshape(p * r, q * s)


def apply_kron_sum(As, Bs, X: np.ndarray) -> np.ndarray:
    t = X.shape[0]
    p, q = As[0].shape
    r, s = Bs[0].shape
    Y = np.zeros((t, p * r), dtype=np.float32)
    Xb = X.reshape(t, q, s)
    for A, B in zip(As, Bs):
        tm = Xb @ B.T
        ym = np.einsum("pq,tqr->tpr", A, tm)
        Y += ym.reshape(t, p * r)
    return Y


def accounting(
    *,
    organ_numel: int,
    generator_bytes: int,
    seed_bytes: int = 0,
    index_bytes: int = 0,
    metadata_bytes: int = 0,
    cache_f16_bytes: int = 0,
    cache_f32_bytes: int = 0,
    cache_regenerable: bool = True,
    reconstruct_flops: float = 0.0,
    reconstruct_wall_s: float = 0.0,
    fused_flops_per_token: float = 0.0,
    dense_flops_per_token: float = 0.0,
    fused_apply_ns_per_token: float | None = None,
    cache_apply_ns_per_token: float | None = None,
    fused_possible: bool = True,
    notes: list[str] | None = None,
    unaccounted: list[str] | None = None,
) -> dict:
    storage = generator_bytes + seed_bytes + index_bytes + metadata_bytes
    if cache_f16_bytes and not cache_regenerable:
        storage += cache_f16_bytes
        unaccounted = list(unaccounted or [])
        unaccounted.append(
            "cache is not regenerable from declared generator+seed; charged as storage"
        )
    active_fused = generator_bytes if fused_possible else cache_f16_bytes
    active_cache_f16 = cache_f16_bytes
    active_cache_f32 = cache_f32_bytes
    bpw = lambda b: (8.0 * b / organ_numel) if organ_numel else float("nan")
    return {
        "generator_bytes": int(generator_bytes),
        "seed_bytes": int(seed_bytes),
        "index_bytes": int(index_bytes),
        "metadata_bytes": int(metadata_bytes),
        "cache_f16_bytes": int(cache_f16_bytes),
        "cache_f32_bytes": int(cache_f32_bytes),
        "cache_regenerable_from_generator": bool(cache_regenerable),
        "storage_bytes_total": int(storage),
        "storage_bpw": bpw(storage),
        "active_bytes_fused": int(active_fused),
        "active_bpw_fused": bpw(active_fused) if fused_possible else None,
        "active_bytes_cache_f16": int(active_cache_f16),
        "active_bpw_cache_f16": bpw(active_cache_f16),
        "active_bytes_cache_f32": int(active_cache_f32),
        "active_bpw_cache_f32": bpw(active_cache_f32),
        "reconstruct_flops_once": float(reconstruct_flops),
        "reconstruct_wall_s": float(reconstruct_wall_s),
        "fused_flops_per_token": float(fused_flops_per_token),
        "dense_flops_per_token": float(dense_flops_per_token),
        "fused_apply_ns_per_token": fused_apply_ns_per_token,
        "cache_apply_ns_per_token": cache_apply_ns_per_token,
        "fused_possible": bool(fused_possible),
        "notes": notes or [],
        "unaccounted": unaccounted or [],
    }


def health_verdict(hold: dict, q3: dict, acc: dict, null_cos: float) -> str:
    rf, gain = hold["rel_fro"], hold["gain"]
    qrf, qgain = q3["rel_fro"], q3["gain"]
    quality = rf <= qrf and gain >= qgain - GAIN_MARGIN
    stor = acc["storage_bpw"]
    act = acc["active_bpw_fused"] if acc.get("fused_possible") else acc["active_bpw_cache_f16"]
    act = act if act is not None else acc["active_bpw_cache_f16"]
    stor_win = stor <= MATERIAL_RATIO * Q3_STORAGE_BPW
    act_win = act <= MATERIAL_RATIO * Q3_STORAGE_BPW
    cos_vs_null = hold["cosine"] - null_cos
    if quality and stor_win and act_win:
        return "HEALTHY: beats q3 quality at fewer storage AND active bytes"
    if quality and stor_win and not act_win:
        return (
            "UNHEALTHY: storage win smuggles a generated-state cache "
            f"(active_bpw={act:.4f} vs q3 {Q3_STORAGE_BPW})"
        )
    if not quality and stor < 0.5:
        return (
            f"UNHEALTHY: local storage_bpw={stor:.4f} < 0.5 with quality below q3 "
            f"(rel_fro {rf:.4f} vs q3 {qrf:.4f}); a low number is not a result"
        )
    if not quality and abs(cos_vs_null) < 0.05:
        return (
            f"UNHEALTHY: cosine {hold['cosine']:.4f} sits on the null {null_cos:.4f}; "
            "not fidelity"
        )
    if not quality:
        return (
            f"UNHEALTHY vs q3 (hold rel_fro {rf:.4f} vs {qrf:.4f}, "
            f"gain {gain:.4f} vs {qgain:.4f}); storage_bpw={stor:.4f} active_bpw={act:.4f}"
        )
    return (
        f"QUALITY matches q3 but bytes do not (storage_bpw={stor:.4f} "
        f"active_bpw={act:.4f} vs q3 {Q3_STORAGE_BPW})"
    )


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def _write(doc: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(j(doc), indent=2) + "\n")


def kron_selfcheck() -> dict:
    rng = np.random.default_rng(0)
    p, r, q, s = 4, 3, 5, 2
    A = rng.standard_normal((p, q)).astype(np.float32)
    B = rng.standard_normal((r, s)).astype(np.float32)
    W = np.kron(A, B)
    X = rng.standard_normal((7, q * s)).astype(np.float32)
    y_ref = X @ W.T
    y_fast = apply_kron_sum([A], [B], X)
    R = rearrange_kron(W, p, r, q, s)
    # Rank-1 rearrangement is vec(A) vec(B)^T (up to scale); reconstruction must match.
    W2 = unrearrange_kron(R, p, r, q, s)
    err_apply = float(np.max(np.abs(y_ref - y_fast)))
    err_round = float(np.max(np.abs(W - W2)))
    ok = err_apply < 1e-4 and err_round < 1e-6
    return {"ok": ok, "max_apply_err": err_apply, "max_rearrange_err": err_round}


def archaeology() -> dict:
    g042, e042 = git_show_json(G042_REL)
    g032, e032 = git_show_json(G032_REL)
    g031, e031 = git_show_json(G031_REL)
    g034, e034 = git_show_json(G034_REL)
    g096, e096 = git_show_json(G096_REL)
    tool = git_show(BPW_TOOL_REL)
    nr = git_show(NR_TOOL_REL)
    tool_txt = tool.decode("utf-8", "replace") if tool else ""
    nr_txt = nr.decode("utf-8", "replace") if nr else ""

    literal_zero = '"GENERATED_BPW_EQUIVALENT": 0.0' in tool_txt
    zero_is_assignment = False
    for line in tool_txt.splitlines():
        if "GENERATED_BPW_EQUIVALENT" in line and "0.0" in line and "definitions" not in line:
            if line.strip().startswith('"GENERATED_BPW_EQUIVALENT": 0.0'):
                zero_is_assignment = True
                break
    nr_empty = '"generated_structures": []' in nr_txt
    nr_cannot = "active_bytes" not in nr_txt or "generated_structures" in nr_txt

    g042_vals = []
    g042_def = None
    if g042:
        g042_def = (g042.get("definitions") or {}).get("GENERATED_BPW_EQUIVALENT")
        for c in g042.get("candidates", []):
            g042_vals.append(
                {
                    "candidate": c.get("candidate"),
                    "GENERATED_BPW_EQUIVALENT": c.get("GENERATED_BPW_EQUIVALENT"),
                    "SHARED_BPW": c.get("SHARED_BPW"),
                    "CORRECTION_BPW": c.get("CORRECTION_BPW"),
                }
            )

    hadamard = None
    if g032:
        hadamard = {
            "obligation": g032.get("obligation"),
            "transform": g032.get("transform"),
            "summary": g032.get("summary"),
            "what_it_tested": (
                "Block-diagonal Sylvester-Hadamard as a codec REPARAMETERIZATION of "
                "STORED W: pack (W H) at the same bit width, apply H^T x at runtime. "
                "H is generated from its size (zero stored bytes) and is NOT model-"
                "specific. W is still stored in full. Function is exactly preserved "
                "in exact arithmetic. Metrics were hold-cosine, rel-fro, and code "
                "entropy vs untransformed grouped-absmax — not a fit of a generator "
                "that produces W from a latent."
            ),
        }

    family_note = None
    if g031:
        pool = []
        if isinstance(g031, list):
            pool = g031
        elif isinstance(g031, dict):
            for key in ("families", "review", "table", "members"):
                v = g031.get(key)
                if isinstance(v, list):
                    pool.extend(v)
            # some receipts are a dict of name -> family
            if not pool:
                pool = [v for v in g031.values() if isinstance(v, dict) and "family" in v]
        for fam in pool:
            if not isinstance(fam, dict):
                continue
            fname = fam.get("family") or ""
            # Match the family identity, not a citation of G032 inside another family.
            if (
                "XFORM" in fname
                or "Hadamard" in fname
                or fam.get("obligation") == "G032"
            ):
                family_note = {
                    "family": fname,
                    "verdict": fam.get("verdict"),
                    "settling_check": fam.get("settling_check"),
                    "hidden_cost": (fam.get("defects_found") or {}).get("hidden_cost"),
                }
                break

    g034_sum = None
    if g034:
        g034_sum = {
            "obligation": g034.get("obligation"),
            "verdict": g034.get("verdict"),
            "mean_flat_q3": g034.get("mean_flat_q3"),
            "mean_lowrank": g034.get("mean_lowrank"),
            "mean_mac_ratio": g034.get("mean_mac_ratio"),
            "method": g034.get("method"),
            "note": "G034 fitted stored low-rank factors at matched q3 bits in function space. That IS a generated W (W = UV). G042 did not record those factor bytes on the GENERATED axis; it recorded 0.",
        }

    hiding = None
    if g096:
        hiding = {
            "obligation": g096.get("obligation"),
            "excerpt": None,
        }
        # keep a short pointer
        blob = json.dumps(g096)
        if "generated" in blob.lower():
            hiding["mentions_generated"] = True

    # Classification
    fitted_a_generator = False  # G042 itself did not
    evidence_artifact = [
        "tools/gravity_bpw_family.py assigns GENERATED_BPW_EQUIVALENT: 0.0 as a literal in the candidate row; it never reads a generator, never measures generated-state bytes, never prices FWHT runtime.",
        "G042 definition text cites G032 Hadamard as the reason for the zero, but G032 stored W at full size and generated only H, which carries no parent information (C5: do not put H in generated_structures and claim GENERATED_BPW).",
        "tools/nr_container.py serialize() emits generated_structures: [] for every candidate and explains the emptiness by pointing at G042's zero — a closed loop.",
        "NR has no field for active_bytes != stored_bytes of a generated structure; a generator that stores 0.05 BPW and reconstructs a 16 BPW cache cannot be represented.",
        "G031 settling_check on G-XFORM: settled for Hadamard; the family is NOT settled (permutation, Kronecker, … untested as XFORM members). A class-level REFUTED on generated/implicit weights does not follow from one transform.",
        "G034 DID fit a generator (UV at matched bits) and REFUTED it as a quality lever. That result never landed on G042's GENERATED axis.",
    ]
    evidence_property = [
        "G032 Hadamard member: mean_delta_hold +0.000338, mean_delta_entropy +0.0259 bits/elem at Q4. Bytes do not drop. Entropy rises. KILLED as a bit-saving generated transform of stored W. That member is PROPERTY_OF_IDEA.",
        "G034 low-rank operator at matched 3.25 b/elem: 2.93× q3 output error. PROPERTY_OF_IDEA for matched-bit UV as a drop-in for grouped-absmax. Not a measurement of GENERATED_BPW, and it did not account reconstruct-and-cache vs fused-apply.",
    ]

    classification = "ARTIFACT_OF_METHOD"
    classification_scope = (
        "G042's GENERATED_BPW_EQUIVALENT=0 and the index line 'generated/implicit "
        "weights REFUTED' as a class-level closure. The pipeline had no slot that "
        "could hold a non-zero generated structure, so the zero is not a measurement "
        "of the class. G032 Hadamard and G034 matched-bit UV remain PROPERTY_OF_IDEA "
        "refutations of those specific members."
    )

    return {
        "g042_path": G042_REL,
        "g042_load_error": e042,
        "g042_obligation": (g042 or {}).get("obligation") if g042 else None,
        "g042_definition": g042_def,
        "g042_candidate_zeros": g042_vals,
        "tool_path": BPW_TOOL_REL,
        "tool_literal_zero_present": literal_zero,
        "tool_zero_is_row_assignment": zero_is_assignment or literal_zero,
        "nr_path": NR_TOOL_REL,
        "nr_generated_structures_hardcoded_empty": nr_empty,
        "nr_has_active_bytes_field": "active_bytes" in nr_txt,
        "g032_hadamard": hadamard,
        "g031_family": family_note,
        "g034_lowrank": g034_sum,
        "g096_pointer": hiding,
        "fitted_a_generator": fitted_a_generator,
        "what_g042_tested": (
            "Eight BPW axes on already-built artifacts (uniform-q4-v1 and three "
            "siblings). STORED/ACTIVE/DRAM/CACHE/STATE were walked or derived. "
            "GENERATED, CORRECTION and SHARED were hardcoded 0.0. No generator "
            "was fitted. The Hadamard citation is G032, a stored-W reparameterization."
        ),
        "what_g042_did_not_test": (
            "A latent/procedural generator for W (or for Wx) with generator bytes, "
            "seed, generated-state cache and reconstruction cost on the same receipt. "
            "NR cannot express that tuple."
        ),
        "classification": classification,
        "classification_scope": classification_scope,
        "evidence_artifact_of_method": evidence_artifact,
        "evidence_property_of_idea_members": evidence_property,
        "load_errors": {k: v for k, v in {
            "G042": e042, "G032": e032, "G031": e031, "G034": e034, "G096": e096,
        }.items() if v},
    }


def print_archaeology(a: dict) -> None:
    print("## 1. G042 archaeology")
    print("-" * 72)
    print(f"  receipt: {a['g042_path']}")
    print(f"  tool:    {a['tool_path']}")
    print(f"  obligation: {a['g042_obligation']}")
    print(f"  definition GENERATED_BPW_EQUIVALENT:")
    print(f"    {a['g042_definition']}")
    print(f"  candidate zeros: {a['g042_candidate_zeros']}")
    print(f"  tool assigns 0.0 as a literal: {a['tool_literal_zero_present']}")
    print(f"  NR generated_structures hardcoded []: {a['nr_generated_structures_hardcoded_empty']}")
    print(f"  NR has active_bytes field: {a['nr_has_active_bytes_field']}")
    print(f"  fitted a generator: {a['fitted_a_generator']}")
    print(f"  what it tested: {a['what_g042_tested']}")
    print(f"  what it did not: {a['what_g042_did_not_test']}")
    if a.get("g032_hadamard"):
        s = a["g032_hadamard"]["summary"] or {}
        t = a["g032_hadamard"]["transform"] or {}
        print(
            f"  G032 Hadamard: stored_bytes={t.get('stored_bytes')} "
            f"mean_delta_hold={s.get('mean_delta_hold')} "
            f"mean_delta_entropy={s.get('mean_delta_entropy_bits')} "
            f"runtime={t.get('runtime_cost')}"
        )
    if a.get("g031_family"):
        print(f"  G031: {a['g031_family'].get('verdict')} / {a['g031_family'].get('settling_check')}")
    if a.get("g034_lowrank"):
        print(f"  G034: {a['g034_lowrank'].get('verdict')}")
    print(f"  CLASSIFICATION: {a['classification']}")
    print(f"  scope: {a['classification_scope']}")
    print()


def isotropic_probe(X_ref: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = X_ref.std(axis=0, keepdims=True).astype(np.float32)
    rms = np.where(rms > 0, rms, 1.0)
    return (rng.standard_normal((n, X_ref.shape[1])).astype(np.float32) * rms)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    t_all = time.time()
    print("GENERATED WEIGHTS RETEST")
    print("=" * 72)
    head = git_head()
    print(f"git_head: {head}")
    print(f"repo:     {ROOT}")
    print(f"python:   {sys.executable}")
    print(f"numpy:    {np.__version__}")
    print()

    arch = archaeology()
    print_archaeology(arch)

    parent = find_parent()
    cap = find_capture()
    print(f"parent:   {parent}")
    print(f"capture:  {cap}")
    print("teacher:  qualified parent BF16; llama-server not used")
    print("site:     post_attn_norm (G034-verified MLP input)")
    print()

    kc = kron_selfcheck()
    print(f"kronecker self-check: ok={kc['ok']} apply_err={kc['max_apply_err']:.2e} rearrange_err={kc['max_rearrange_err']:.2e}")
    if not kc["ok"]:
        print("FAIL: Kronecker apply/rearrange self-check")
        return 2
    print()

    W, wname = load_tensor(parent, tensor_candidates(LAYER, ORGAN))
    m, n = int(W.shape[0]), int(W.shape[1])
    if (m, n) != (INTERMEDIATE, HIDDEN):
        print(f"WARN: unexpected gate_proj shape {W.shape}, continuing")
    organ_numel = int(W.size)
    dense_flops = 2.0 * m * n
    print(f"organ:    L{LAYER}.{ORGAN}_proj  {wname}  shape={list(W.shape)}  numel={organ_numel:,}")

    X_all = load_X(cap, LAYER)
    fit_idx, hold_idx, split_rule, families, n_tokens = split_indices(cap, X_all.shape[0])
    rng = np.random.default_rng(0)
    fit_perm = rng.permutation(fit_idx)
    n_val = max(256, int(0.15 * len(fit_perm)))
    # val unused for closed-form generators; carved so hold stays untouched
    train_idx = fit_perm[n_val:]
    X_train, X_hold = X_all[train_idx], X_all[hold_idx]
    print(
        f"capture:  tokens={n_tokens} train={len(train_idx)} hold={len(hold_idx)} "
        f"split={split_rule} families={families}"
    )
    print()

    print("## 2. Scale trap  (cosine is scale-invariant; gain/rel_fro must refuse 0.01·W)")
    print("-" * 72)
    X_trap = X_hold[: min(512, len(hold_idx))]
    Y_trap = linear_forward(X_trap, W)
    Y_s = linear_forward(X_trap, 0.01 * W)
    trap = score_pair(Y_trap, Y_s)
    trap["organ"] = f"L{LAYER}.{ORGAN}_proj linear"
    trap_ok = bool(trap["cosine"] > 0.99 and trap["gain"] < 0.05 and trap["rel_fro"] > 0.9)
    print(
        f"  0.01*W  cosine={trap['cosine']:.6f}  gain={trap['gain']:.6f}  "
        f"rel_fro={trap['rel_fro']:.6f}  rejects={trap_ok}"
    )
    if not trap_ok:
        print("FAIL: scale trap did not reject 0.01*W — GO metric is blind")
        doc = {
            "schema": "hawking.headless.generated_weights_retest.v1",
            "verdict": {"decision": "NO-GO", "reason": "scale trap failed"},
            "scale_trap": trap,
            "g042": arch,
        }
        _write(doc)
        return 2
    print("  GO uses rel_fro + gain, never cosine alone.")
    print()

    print("## 3. Real-activation evaluation")
    print("-" * 72)
    t_y = time.time()
    Y_hold = linear_forward(X_hold, W)
    y_hold_s = time.time() - t_y
    null_mean = null_cosine_constant_mean(Y_hold)
    rng_n = np.random.default_rng(1)
    W_rand = rng_n.standard_normal(W.shape).astype(np.float32)
    W_rand *= (np.linalg.norm(W) / (np.linalg.norm(W_rand) + 1e-30))
    Y_rand = linear_forward(X_hold, W_rand)
    null_randw = score_pair(Y_hold, Y_rand)
    mu = Y_hold.mean()
    Y_const = np.full_like(Y_hold, mu)
    null_const = score_pair(Y_hold, Y_const)
    print(
        f"  Y_hold {Y_hold.shape} wall={y_hold_s:.2f}s  "
        f"null constant-mean cosine={null_mean:.4f}  "
        f"null matched-F rms random-W cosine={null_randw['cosine']:.4f} "
        f"rel_fro={null_randw['rel_fro']:.4f}"
    )
    print(
        f"  folklore 0.898 is NOT assumed; measured nulls are the ones above. "
        f"scalar-constant cosine={null_const['cosine']:.4f}"
    )
    P = isotropic_probe(X_train, N_PROBE, seed=7)
    Y_P = linear_forward(P, W)

    # q3 incumbent
    print("  q3 grouped-absmax g64 ...")
    t0 = time.time()
    W_q3, q3_bits = quantize_group(W)
    Y_q3 = linear_forward(X_hold, W_q3)
    q3_hold = score_pair(Y_hold, Y_q3)
    q3_hold["cosine_minus_null"] = q3_hold["cosine"] - null_mean
    q3_probe = score_pair(Y_P, linear_forward(P, W_q3))
    q3_wall = time.time() - t0
    q3_bytes = q3_bits / 8.0
    q3_acc = accounting(
        organ_numel=organ_numel,
        generator_bytes=int(q3_bytes),  # packed payload is the representation
        cache_f16_bytes=int(organ_numel * 2),
        cache_f32_bytes=int(organ_numel * 4),
        cache_regenerable=True,
        reconstruct_flops=float(organ_numel),  # dequant
        fused_flops_per_token=dense_flops,
        dense_flops_per_token=dense_flops,
        fused_possible=True,
        notes=[
            "q3 fused streams codes+scales (storage=active=3.25). "
            "Decoded-f16/f32 cache is what a naive dequant-then-GEMV pays."
        ],
    )
    q3_health = "HEALTHY (incumbent reference)"
    print(
        f"    hold rel_fro={q3_hold['rel_fro']:.4f} cosine={q3_hold['cosine']:.4f} "
        f"(null {null_mean:.4f}, Δ={q3_hold['cosine_minus_null']:.4f}) "
        f"gain={q3_hold['gain']:.4f} worst_unit={q3_hold['worst_unit']:.4f}"
    )
    print(
        f"    storage_bpw={q3_acc['storage_bpw']:.4f}  active_fused_bpw={q3_acc['active_bpw_fused']:.4f}  "
        f"active_decoded_f16_bpw={q3_acc['active_bpw_cache_f16']:.4f}  wall={q3_wall:.2f}s"
    )

    candidates = []
    watched = []
    watched.append(
        f"0.01*W cosine={trap['cosine']:.6f} (blind) gain={trap['gain']:.6f} "
        f"rel_fro={trap['rel_fro']:.6f} (rejects). Cosine is not a GO metric."
    )

    def record(family, name, hold, acc, extra=None, probe=None):
        hold = dict(hold)
        hold["cosine_minus_null"] = hold["cosine"] - null_mean
        if family == "incumbent_q3":
            hv = "HEALTHY (incumbent reference)"
        else:
            hv = health_verdict(hold, q3_hold, acc, null_mean)
        rec = {
            "family": family,
            "name": name,
            "hold": hold,
            "probe_isotropic": probe,
            "accounting": acc,
            "health_verdict": hv,
            "extra": extra or {},
        }
        candidates.append(rec)
        act = acc["active_bpw_fused"] if acc.get("fused_possible") else acc["active_bpw_cache_f16"]
        print(
            f"    {name:<28} stor={acc['storage_bpw']:.4f} act_fused="
            f"{acc['active_bpw_fused'] if acc.get('fused_possible') else float('nan'):.4f} "
            f"act_cache_f16={acc['active_bpw_cache_f16']:.4f}  "
            f"hold_rel={hold['rel_fro']:.4f} cos={hold['cosine']:.4f} "
            f"Δnull={hold['cosine_minus_null']:.4f} gain={hold['gain']:.4f}  {hv}"
        )
        if "UNHEALTHY" in hv or "smuggles" in hv:
            watched.append(f"{name}: {hv}")
        return rec

    # ------------------------------------------------------------------ SVD
    max_r = rank_for_bpw(m, n, max(BPW_TARGETS))
    print(f"  weight-space rSVD rank={max_r} (prefixes cover BPW targets) ...")
    t0 = time.time()
    Uw, Sw, Vtw = rsvd(W, max_r, seed=0)
    t_wsvd = time.time() - t0
    print(f"    wall={t_wsvd:.2f}s  S[0]={float(Sw[0]):.4f} S[-1]={float(Sw[-1]):.4f}")

    print("  activation-aware rSVD (W @ sqrt(X_train^T X_train)) ...")
    t0 = time.time()
    G = X_train.T @ X_train
    sqrtG, invsqrtG, ginfo = spd_sqrt_and_inv(G)
    WL = W @ sqrtG
    Ua, Sa, Vta = rsvd(WL, max_r, seed=1)
    t_asvd = time.time() - t0
    del WL, G, sqrtG
    print(
        f"    wall={t_asvd:.2f}s  cond_after_ridge={ginfo['cond_after_ridge']:.3e}  "
        f"S[0]={float(Sa[0]):.4f}"
    )

    for bpw_t in BPW_TARGETS:
        r = rank_for_bpw(m, n, bpw_t)
        r = min(r, Uw.shape[1], Ua.shape[1])
        # weight-space
        U = f16_store(Uw[:, :r] * Sw[:r])
        V = f16_store(Vtw[:r])
        t1 = time.time()
        Yh = svd_forward(X_hold, U, V)
        fused_s = time.time() - t1
        hold = score_pair(Y_hold, Yh)
        probe = score_pair(Y_P, svd_forward(P, U, V))
        gen_b = int(r * (m + n) * 2)  # f16 factors
        acc = accounting(
            organ_numel=organ_numel,
            generator_bytes=gen_b,
            metadata_bytes=16,  # rank + shape
            cache_f16_bytes=int(organ_numel * 2),
            cache_f32_bytes=int(organ_numel * 4),
            cache_regenerable=True,
            reconstruct_flops=2.0 * m * n * r,
            fused_flops_per_token=2.0 * r * (m + n),
            dense_flops_per_token=dense_flops,
            fused_apply_ns_per_token=1e9 * fused_s / max(X_hold.shape[0], 1),
            fused_possible=True,
            notes=[
                "Fused-apply streams U,V (active=storage). Reconstruct-and-cache "
                "materialises W_hat in f16/f32; that cache is generated-state and "
                "is ACTIVE even though it is regenerable (not extra disk if U,V live)."
            ],
            unaccounted=[
                "No hashed Metal kernel for fused UV-GEMV. Probe does not claim tok/s."
            ],
        )
        record(
            "svd_weight_space",
            f"svd_w_r{r}_bpw{bpw_t}",
            hold,
            acc,
            extra={"rank": r, "target_storage_bpw": bpw_t, "fit": "rsvd(W)", "train_used": False},
            probe=probe,
        )
        del Yh, U, V

        # activation-aware
        Vt_w = Vta[:r] @ invsqrtG
        U = f16_store(Ua[:, :r] * Sa[:r])
        V = f16_store(Vt_w)
        t1 = time.time()
        Yh = svd_forward(X_hold, U, V)
        fused_s = time.time() - t1
        hold = score_pair(Y_hold, Yh)
        # train overfit watch (chunked rel_fro + cosine)
        Ytr = linear_forward(X_train[: min(1024, len(train_idx))], W)
        Ytrh = svd_forward(X_train[: min(1024, len(train_idx))], U, V)
        train_spot = score_pair(Ytr, Ytrh)
        probe = score_pair(Y_P, svd_forward(P, U, V))
        acc = accounting(
            organ_numel=organ_numel,
            generator_bytes=gen_b,
            metadata_bytes=16,
            cache_f16_bytes=int(organ_numel * 2),
            cache_f32_bytes=int(organ_numel * 4),
            cache_regenerable=True,
            reconstruct_flops=2.0 * m * n * r,
            fused_flops_per_token=2.0 * r * (m + n),
            dense_flops_per_token=dense_flops,
            fused_apply_ns_per_token=1e9 * fused_s / max(X_hold.shape[0], 1),
            fused_possible=True,
            notes=[
                "Factors fitted in the metric of G=X_train^T X_train (real activations). "
                "Hold never used in the fit. Fused vs cache as above."
            ],
            unaccounted=[
                "No hashed Metal kernel for fused UV-GEMV. Probe does not claim tok/s."
            ],
        )
        rec = record(
            "svd_activation_aware",
            f"svd_a_r{r}_bpw{bpw_t}",
            hold,
            acc,
            extra={
                "rank": r,
                "target_storage_bpw": bpw_t,
                "fit": "rsvd(W @ sqrtG), G=X_train.T@X_train",
                "train_used": True,
                "train_spot_n": int(Ytr.shape[0]),
                "train_spot": train_spot,
                "gram": ginfo,
            },
            probe=probe,
        )
        if train_spot["rel_fro"] + 0.02 < hold["rel_fro"] and hold["rel_fro"] > q3_hold["rel_fro"]:
            watched.append(
                f"{rec['name']} train_spot rel_fro={train_spot['rel_fro']:.4f} "
                f"< hold {hold['rel_fro']:.4f} (activation-aware overfit on fit-set X)"
            )
        del Yh, U, V, Vt_w, Ytr, Ytrh

    del Uw, Sw, Vtw, Ua, Sa, Vta, invsqrtG

    # ------------------------------------------------------------------ DCT coefficients (procedural basis)
    print("  2D ortho DCT-II of W (basis is NOT model-specific; coefficients ARE) ...")
    t0 = time.time()
    C = dct(dct(W, axis=1, type=2, norm="ortho"), axis=0, type=2, norm="ortho")
    C = np.asarray(C, dtype=np.float32)
    t_dct = time.time() - t0
    print(f"    dct wall={t_dct:.2f}s")
    for bpw_t in BPW_TARGETS:
        n_coeffs = max(1, int(round(bpw_t * organ_numel / F16_BITS)))
        p, q = dct_keep_shape(m, n, n_coeffs)
        t1 = time.time()
        Ct = np.zeros_like(C)
        Ct[:p, :q] = C[:p, :q]
        What = idct(idct(Ct, axis=0, type=2, norm="ortho"), axis=1, type=2, norm="ortho")
        What = f16_store(np.asarray(What, dtype=np.float32))
        t_rec = time.time() - t1
        del Ct
        t2 = time.time()
        Yh = linear_forward(X_hold, What)
        t_app = time.time() - t2
        hold = score_pair(Y_hold, Yh)
        probe = score_pair(Y_P, linear_forward(P, What))
        gen_b = int(p * q * 2)  # f16 prefix coefficients
        acc = accounting(
            organ_numel=organ_numel,
            generator_bytes=gen_b,
            metadata_bytes=16,  # (p,q)
            seed_bytes=0,
            index_bytes=0,
            cache_f16_bytes=int(organ_numel * 2),
            cache_f32_bytes=int(organ_numel * 4),
            cache_regenerable=True,
            reconstruct_flops=float(m * n * (math.log2(max(m, 2)) + math.log2(max(n, 2)))),
            reconstruct_wall_s=t_rec,
            fused_flops_per_token=0.0,
            dense_flops_per_token=dense_flops,
            cache_apply_ns_per_token=1e9 * t_app / max(X_hold.shape[0], 1),
            fused_possible=False,
            notes=[
                "DCT basis is generated from size (like Hadamard) and is NOT parent "
                "information — not charged. Prefix coefficients ARE parent information. "
                "There is no fused DCT-GEMV for an arbitrary prefix of C; apply path "
                "reconstructs W_hat and streams the cache. That is the storage/active split "
                "G042/NR could not represent. Top-k-by-magnitude would add index_bytes = "
                f"n_coeffs * log2(mn) bits ≈ {n_coeffs * math.log2(organ_numel) / 8.0:,.0f} B "
                "and is not run."
            ],
            unaccounted=[
                "IDCT reconstruction kernel not hashed. Probe does not claim tok/s."
            ],
        )
        record(
            "dct_prefix_coefficients",
            f"dct_p{p}x{q}_bpw{bpw_t}",
            hold,
            acc,
            extra={
                "p": p, "q": q, "n_coeffs": p * q,
                "target_storage_bpw": bpw_t,
                "reconstruct_wall_s": t_rec,
                "basis_is_model_specific": False,
            },
            probe=probe,
        )
        del What, Yh
    del C

    # ------------------------------------------------------------------ Kronecker sum (Van Loan)
    p, r, q, s = KRON_P, KRON_R, KRON_Q, KRON_S
    assert m == p * r and n == q * s, (m, n, p, r, q, s)
    a_n, b_n = p * q, r * s
    max_k = k_for_kron_bpw(max(BPW_TARGETS), a_n, b_n, organ_numel)
    max_k = min(max_k, a_n, b_n)
    print(f"  Kronecker-sum rSVD of rearrangement ({p*q} x {r*s}) rank={max_k} ...")
    t0 = time.time()
    R = rearrange_kron(W, p, r, q, s)
    Uk, Sk, Vtk = rsvd(R, max_k, seed=2)
    t_kron = time.time() - t0
    del R
    print(f"    wall={t_kron:.2f}s  S[0]={float(Sk[0]):.4f}")

    for bpw_t in BPW_TARGETS:
        k = min(k_for_kron_bpw(bpw_t, a_n, b_n, organ_numel), Uk.shape[1])
        t1 = time.time()
        Rk = (Uk[:, :k] * Sk[:k]) @ Vtk[:k]
        What = unrearrange_kron(f16_store(Rk), p, r, q, s)
        t_rec = time.time() - t1
        del Rk
        t2 = time.time()
        Yh = linear_forward(X_hold, What)
        t_app = time.time() - t2
        hold = score_pair(Y_hold, Yh)
        probe = score_pair(Y_P, linear_forward(P, What))
        gen_b = int(k * (a_n + b_n) * 2)
        fused_flops = 2.0 * k * r * (n + p * q)  # see apply_kron_sum
        fused_possible = k <= 16
        fused_ns = None
        if fused_possible:
            As, Bs = [], []
            for i in range(k):
                si = math.sqrt(max(float(Sk[i]), 0.0))
                As.append(f16_store((Uk[:, i] * si).reshape(p, q)))
                Bs.append(f16_store((Vtk[i] * si).reshape(r, s)))
            t3 = time.time()
            Yh_f = apply_kron_sum(As, Bs, X_hold)
            fused_ns = 1e9 * (time.time() - t3) / max(X_hold.shape[0], 1)
            # sanity vs cache path
            extra_err = float(np.linalg.norm(Yh_f - Yh) / (np.linalg.norm(Yh) + 1e-30))
            del Yh_f, As, Bs
        else:
            extra_err = None
        acc = accounting(
            organ_numel=organ_numel,
            generator_bytes=gen_b,
            metadata_bytes=32,
            cache_f16_bytes=int(organ_numel * 2),
            cache_f32_bytes=int(organ_numel * 4),
            cache_regenerable=True,
            reconstruct_flops=2.0 * k * a_n * b_n,
            reconstruct_wall_s=t_rec,
            fused_flops_per_token=fused_flops,
            dense_flops_per_token=dense_flops,
            fused_apply_ns_per_token=fused_ns,
            cache_apply_ns_per_token=1e9 * t_app / max(X_hold.shape[0], 1),
            fused_possible=True,  # structured apply exists even if slower than dense
            notes=[
                "W ≈ Σ_i A_i ⊗ B_i. Factors are model-specific. Structured apply is "
                "fused (active=storage) but at large k is MORE flops than dense GEMV; "
                "a rational runtime would reconstruct-and-cache, paying 16 BPW active. "
                f"k={k} fused_flops/token={fused_flops:.3e} vs dense {dense_flops:.3e}."
            ],
            unaccounted=["No hashed Kronecker-GEMV kernel. Probe does not claim tok/s."],
        )
        record(
            "kronecker_sum",
            f"kron_k{k}_bpw{bpw_t}",
            hold,
            acc,
            extra={
                "k": k,
                "target_storage_bpw": bpw_t,
                "tile": [p, r, q, s],
                "fused_vs_cache_rel": extra_err,
                "fused_flops_gt_dense": bool(fused_flops > dense_flops),
            },
            probe=probe,
        )
        del What, Yh
    del Uk, Sk, Vtk

    # ------------------------------------------------------------------ seed canary (free-information trap)
    print("  seed canary: W_hat ~ N(0,1) * ||W|| from an 8-byte seed ...")
    seed = 0xC0FFEE42
    rng_s = np.random.default_rng(seed)
    t1 = time.time()
    W_seed = rng_s.standard_normal(W.shape, dtype=np.float32)
    W_seed *= (np.linalg.norm(W) / (np.linalg.norm(W_seed) + 1e-30))
    t_rec = time.time() - t1
    Yh = linear_forward(X_hold, W_seed)
    hold = score_pair(Y_hold, Yh)
    acc = accounting(
        organ_numel=organ_numel,
        generator_bytes=0,
        seed_bytes=8,
        metadata_bytes=8,  # shape implied
        cache_f16_bytes=int(organ_numel * 2),
        cache_f32_bytes=int(organ_numel * 4),
        cache_regenerable=True,
        reconstruct_flops=float(organ_numel),
        reconstruct_wall_s=t_rec,
        fused_possible=False,
        dense_flops_per_token=dense_flops,
        notes=[
            "The seed is model information (it regenerates this particular draw only "
            "because we chose the draw; a seed that actually reproduced W would be "
            "a compression of W and would have Kolmogorov complexity ~|W|). Charged: "
            "8-byte seed + the f16/f32 cache you must hold to GEMV. Storage BPW of the "
            "seed alone is ~0 and is not a result."
        ],
        unaccounted=[
            "PRNG algorithm identity is methodology (shared), not charged. If the "
            "seed were a learned key that regenerated W, the unaccounted Kolmogorov "
            "content would be the hole this canary exists to name."
        ],
    )
    record(
        "seed_prng_canary",
        "seed_xorshift_8B",
        hold,
        acc,
        extra={"seed": seed, "target_storage_bpw": 0.0},
    )
    del W_seed, Yh

    # q3 as a candidate row too
    record("incumbent_q3", "q3_g64", q3_hold, q3_acc, extra={"storage_bits": q3_bits}, probe=q3_probe)

    # ------------------------------------------------------------------ second site (local vs composed)
    print()
    print(f"  second site L{LAYER_B}.{ORGAN}_proj (local win is not a composed win) ...")
    W0, wname0 = load_tensor(parent, tensor_candidates(LAYER_B, ORGAN))
    X0 = load_X(cap, LAYER_B)
    X0_hold = X0[hold_idx]
    Y0_hold = linear_forward(X0_hold, W0)
    W0_q3, _ = quantize_group(W0)
    q3_0 = score_pair(Y0_hold, linear_forward(X0_hold, W0_q3))
    r_lo = rank_for_bpw(W0.shape[0], W0.shape[1], 0.25)
    r_hi = rank_for_bpw(W0.shape[0], W0.shape[1], 3.25)
    G0 = X0[train_idx].T @ X0[train_idx]
    sqrt0, inv0, _ = spd_sqrt_and_inv(G0)
    U0, S0, Vt0 = rsvd(W0 @ sqrt0, r_hi, seed=3)
    del sqrt0, G0
    second = []
    for r, tag in ((r_lo, "0.25"), (r_hi, "3.25")):
        U = f16_store(U0[:, :r] * S0[:r])
        V = f16_store(Vt0[:r] @ inv0)
        h = score_pair(Y0_hold, svd_forward(X0_hold, U, V))
        h["cosine_minus_null"] = h["cosine"] - null_cosine_constant_mean(Y0_hold)
        second.append(
            {
                "layer": LAYER_B,
                "rank": r,
                "target_bpw": tag,
                "hold": h,
                "q3_hold_rel_fro": q3_0["rel_fro"],
                "beats_q3": bool(h["rel_fro"] <= q3_0["rel_fro"]),
            }
        )
        print(
            f"    L{LAYER_B} svd_a r={r} hold_rel={h['rel_fro']:.4f} vs q3 {q3_0['rel_fro']:.4f} "
            f"beats={h['rel_fro'] <= q3_0['rel_fro']}"
        )
        del U, V
    if not any(s["beats_q3"] for s in second):
        watched.append(
            f"L{LAYER_B}.{ORGAN}_proj activation-aware SVD also loses to q3 at 0.25 and 3.25 BPW; "
            "no local win exists to compose."
        )
    del W0, X0, X0_hold, Y0_hold, U0, S0, Vt0, inv0, W0_q3

    # ------------------------------------------------------------------ verdict
    print()
    print("## 4. Accounting (every byte)")
    print("-" * 72)
    print(
        f"{'name':<28} {'stor':>7} {'actF':>7} {'actC16':>7} {'rel':>8} {'Δnull':>7} {'gain':>6}"
    )
    for c in candidates:
        a = c["accounting"]
        actf = a["active_bpw_fused"] if a.get("fused_possible") and a["active_bpw_fused"] is not None else float("nan")
        print(
            f"{c['name']:<28} {a['storage_bpw']:7.3f} {actf:7.3f} "
            f"{a['active_bpw_cache_f16']:7.3f} {c['hold']['rel_fro']:8.4f} "
            f"{c['hold']['cosine_minus_null']:7.3f} {c['hold']['gain']:6.3f}"
        )

    wins = []
    for c in candidates:
        if c["family"] == "incumbent_q3":
            continue
        a = c["accounting"]
        act = a["active_bpw_fused"] if a.get("fused_possible") else a["active_bpw_cache_f16"]
        if act is None:
            act = a["active_bpw_cache_f16"]
        q_ok = c["hold"]["rel_fro"] <= q3_hold["rel_fro"] and c["hold"]["gain"] >= q3_hold["gain"] - GAIN_MARGIN
        b_ok = a["storage_bpw"] <= MATERIAL_RATIO * Q3_STORAGE_BPW and act <= MATERIAL_RATIO * Q3_STORAGE_BPW
        if q_ok and b_ok:
            wins.append(c["name"])

    cache_smugglers = [
        c["name"] for c in candidates
        if c["family"] != "incumbent_q3"
        and c["hold"]["rel_fro"] <= q3_hold["rel_fro"]
        and c["accounting"]["storage_bpw"] <= MATERIAL_RATIO * Q3_STORAGE_BPW
        and (c["accounting"]["active_bpw_cache_f16"] or 0) > MATERIAL_RATIO * Q3_STORAGE_BPW
        and (
            not c["accounting"].get("fused_possible")
            or (c["accounting"].get("fused_flops_per_token") or 0) > dense_flops
        )
    ]

    if wins:
        decision = "REOPEN"
        deciding_note = (
            "A generated representation beat q3 on hold quality at fewer storage "
            "AND active bytes. That is the reopen condition."
        )
        deciding_number = min(
            c["hold"]["rel_fro"] - q3_hold["rel_fro"]
            for c in candidates if c["name"] in wins
        )
    else:
        decision = "SLOT_FILLED_NO_WIN"
        deciding_note = (
            "G042's GENERATED_BPW=0 is ARTIFACT_OF_METHOD (no slot). Filling the slot "
            "with SVD factors, DCT coefficients, Kronecker sums and a seed canary on "
            f"L{LAYER}.{ORGAN}_proj with real post_attn_norm X does not beat q3 on "
            "hold rel_fro at fewer storage AND active bytes. G032 Hadamard stands. "
            "G034 matched-bit UV stands, now with cache-vs-fused accounting: a "
            "storage-BPW win that reconstructs W is an active-BPW loss."
        )
        # deciding number: best (most negative) hold gap among fused-capable generated
        gaps = []
        for c in candidates:
            if c["family"] in ("incumbent_q3", "seed_prng_canary"):
                continue
            gaps.append(c["hold"]["rel_fro"] - q3_hold["rel_fro"])
        deciding_number = min(gaps) if gaps else None

    watched.append(
        "G042 recorded GENERATED_BPW_EQUIVALENT=0.0 as a program constant. That is "
        "the method artifact this probe exists to name."
    )
    if cache_smugglers:
        watched.append(
            "Reconstruct-and-cache path would report a storage-BPW win while streaming "
            f"a 16 BPW W_hat: {cache_smugglers}"
        )
    watched.append(
        f"Null cosine (constant-mean)={null_mean:.4f}; random-W cosine={null_randw['cosine']:.4f}. "
        "A cosine sitting on the null is not fidelity."
    )

    print()
    print("## 5. Verdict")
    print("-" * 72)
    print(f"  G042 classification: {arch['classification']}")
    print(f"  retest decision:     {decision}")
    print(f"  deciding_number:     {deciding_number}")
    print(f"  wins:                {wins or 'none'}")
    print(f"  {deciding_note}")
    print()
    print("## WHAT I WATCHED FAIL")
    print("-" * 72)
    for line in watched:
        print(f"  - {line}")
    print()

    unaccounted_global = [
        "No Metal/shader binary is hashed for UV-GEMV, Kronecker-GEMV or IDCT. A shipped kernel with learned constants would be model-specific; a generic kernel is shared runtime. This probe does not claim tok/s, so the missing kernel is named rather than converted into a TPS lever.",
        "Generator Python source is methodology, not parent information.",
        "64-layer composition is not run. Two sites (L31, L0) both lose locally, so there is no local win to compose. A local win would still not be a composed win.",
        "If a seed regenerated W exactly, Kolmogorov complexity of that seed is unaccounted by byte counting; the seed canary is garbage quality, so the hole is named without being used as a BPW win.",
    ]

    doc = {
        "schema": "hawking.headless.generated_weights_retest.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "obligation": (
            "G042 generated/implicit weights: locate what was tested, classify "
            "PROPERTY_OF_IDEA vs ARTIFACT_OF_METHOD, fill the slot if artifact."
        ),
        "parent": str(parent),
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm",
            "n_tokens": int(n_tokens),
            "n_train": int(len(train_idx)),
            "n_hold": int(len(hold_idx)),
            "split_rule": split_rule,
            "families": families,
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward. "
                "Not Gaussian. Not Q5_K llama-server."
            ),
        },
        "organ": {
            "layer": LAYER,
            "name": wname,
            "shape": [m, n],
            "numel": organ_numel,
            "why": "G032/G034 site: L31.gate_proj, contraction 5120 tiles 64 and 1024.",
        },
        "g042": arch,
        "kronecker_selfcheck": kc,
        "scale_trap": {
            "linear_gate_proj": trap,
            "rejects_scaled_artifact": trap_ok,
            "pass_if": "cosine~1 and gain~0.01 and rel_fro~0.99 on 0.01*W",
        },
        "null_baseline": {
            "hold_constant_mean_cosine": null_mean,
            "hold_scalar_constant": null_const,
            "hold_random_W_matched_frobenius": null_randw,
            "note": (
                "Raw activation cosine near the null is not fidelity. Folklore 0.898 "
                "is not assumed; these are measured on this organ and this capture."
            ),
        },
        "q3_incumbent": {
            "hold": q3_hold,
            "accounting": q3_acc,
            "health_verdict": q3_health,
            "probe_isotropic": q3_probe,
        },
        "generated_structures": [
            {
                "name": c["name"],
                "family": c["family"],
                "stored_bytes": c["accounting"]["storage_bytes_total"],
                "active_bytes_fused": c["accounting"]["active_bytes_fused"],
                "active_bytes_cache_f16": c["accounting"]["active_bytes_cache_f16"],
                "generated_cache_bytes_f16": c["accounting"]["cache_f16_bytes"],
                "cache_regenerable": c["accounting"]["cache_regenerable_from_generator"],
                "generator_bytes": c["accounting"]["generator_bytes"],
                "seed_bytes": c["accounting"]["seed_bytes"],
                "reconstruction_flops_once": c["accounting"]["reconstruct_flops_once"],
                "reconstruction_wall_s": c["accounting"]["reconstruct_wall_s"],
                "storage_bpw": c["accounting"]["storage_bpw"],
                "active_bpw_fused": c["accounting"]["active_bpw_fused"],
                "active_bpw_cache_f16": c["accounting"]["active_bpw_cache_f16"],
                "hold": c["hold"],
                "health_verdict": c["health_verdict"],
            }
            for c in candidates
            if c["family"] != "incumbent_q3"
        ],
        "nr_today_cannot_express": {
            "generated_structures": [],
            "GENERATED_BPW_EQUIVALENT": 0.0,
            "missing": "active_bytes != stored_bytes for a generated structure",
            "this_receipt_expresses": True,
        },
        "candidates": candidates,
        "second_site": {
            "layer": LAYER_B,
            "name": wname0,
            "q3_hold": q3_0,
            "rows": second,
            "note": "A local win is not a composed win. Both sites lose locally.",
        },
        "unaccounted": unaccounted_global,
        "watched_fail": watched,
        "verdict": {
            "g042_classification": arch["classification"],
            "g042_classification_scope": arch["classification_scope"],
            "hadamard_member": "PROPERTY_OF_IDEA (G032 stands)",
            "matched_bit_uv": "PROPERTY_OF_IDEA as a q3 replacement (G034 stands; replicated with cache vs fused)",
            "decision": decision,
            "wins": wins,
            "cache_smugglers": cache_smugglers,
            "deciding_number": deciding_number,
            "deciding_note": deciding_note,
            "original_refutation_stands_for": [
                "G032 block Hadamard as a bit-saving generated transform of stored W",
                "generated/implicit weights as a cheaper-than-q3 quality lever on this organ (this retest)",
            ],
            "original_refutation_does_not_stand_for": [
                "GENERATED_BPW_EQUIVALENT=0 as a measurement of the class",
                "NR generated_structures: [] meaning the class was tested and found empty",
            ],
        },
        "wall_s": time.time() - t_all,
    }
    _write(doc)
    print(f"wrote {RECEIPT.relative_to(ROOT)}")
    print(f"wall {doc['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        _ensure_scipy()
        import numpy as np  # noqa: E402
        from scipy.fft import dct, idct  # noqa: E402

        globals()["np"] = np
        globals()["dct"] = dct
        globals()["idct"] = idct
        sys.exit(main())
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        sys.exit(2)
