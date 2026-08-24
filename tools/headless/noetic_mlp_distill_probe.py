#!/usr/bin/env python3
"""NNS-015 probe: distill the MLP *function*, not a student model.

PHASE_B_HYBRID left one surviving avenue: a distilled/generated operator trained
to match the MLP function at q3 quality with fewer active bytes, Doctor holding.
That experiment had not been run. This is the probe, not a campaign.

Function-space fitting, not student-model distillation:
  objective  ||f(X) - f_hat(X)||  on real activations, not ||W - W_hat||
  executable need not resemble the teacher's parameterisation
  a local win is not a global one — one residual hop is measured, no more

The candidate is a *thinner SwiGLU* (same functional form, width I' < I) trained
on MLP outputs. The matched-bits weight-space twin is channel-prune of the
teacher. The incumbent is grouped-absmax q3 (g64, 3.25 b/elem with f16 scales).

Linear output-PCA of down_proj at rank-803 is run as a control: that is NNS-014
and is expected to win the fit set and lose held-out. It is not the headline.

Real activations only. Cosine is never the GO metric (scale trap is exhibited).
Storage BPW and active BPW are reported together. Null baseline is measured,
not assumed. No Metal TPS claim — quality probe; pairing discipline does not
apply. llama-server on 52484 is not the teacher (NS-015: Q5_K is degraded).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"

HIDDEN = 5120
INTERMEDIATE = 17408
GROUP = 64
Q3_BITS = 3
SCALE_BITS = 16  # f16 scale per group; G033/G034/G105 family
Q3_STORAGE_BPW = Q3_BITS + SCALE_BITS / GROUP  # 3.25
F16_BPW = 16.0
MATERIAL_BYTE_RATIO = 0.90  # "materially fewer" vs fused-q3 active bytes
QUALITY_REL_TOL = 0.00  # distill hold rel-fro must be <= q3 (not merely close)
HOP_AMPLIFICATION_CAP = 1.5
AXIS_MARGIN = {"observed": 0.02, "probed": 0.02, "worst_unit": 0.10, "gain": 0.02}

# Widths: 3536 matches q3 storage/active under f16; 2560 is ~72% (the fewer-byte
# candidate); 1024 is the aggressive undershoot. Headline GO can only come from
# a width strictly below the matched point.
WIDTHS = (1024, 2560, 3536)
DISTILL_LAYERS = (0, 31)
HOP_SRC, HOP_DST = 0, 1
TRAIN_STEPS = 400
BATCH = 512
LR = 2e-4
EVAL_EVERY = 25
GRAD_CLIP = 1.0
N_PROBE = 256
N_PROBE_SETS = 3
CHUNK = 1024

PARENT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
    ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    ROOT / "workspace/campaign/phaseB/capture_diverse2",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_attn_norm"),
]


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
    return f"model.language_model.layers.{layer}.mlp.{organ}_proj.weight"


def load_tensor(parent: Path, name: str) -> np.ndarray:
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


def load_mlp(parent: Path, layer: int) -> dict[str, np.ndarray]:
    return {
        "gate": load_tensor(parent, tensor_name(layer, "gate")),
        "up": load_tensor(parent, tensor_name(layer, "up")),
        "down": load_tensor(parent, tensor_name(layer, "down")),
    }


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


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def mlp_forward_np(X, Wg, Wu, Wd, chunk: int = CHUNK):
    """Teacher/student SwiGLU MLP. Returns (Y [n,H], swiglu [n,I])."""
    n = X.shape[0]
    y_parts, s_parts = [], []
    for i in range(0, n, chunk):
        xb = X[i : i + chunk]
        pre = xb @ Wg.T
        up = xb @ Wu.T
        s = silu(pre) * up
        y_parts.append(s @ Wd.T)
        s_parts.append(s)
    return np.concatenate(y_parts, axis=0), np.concatenate(s_parts, axis=0)


def quantize_group(w: np.ndarray, bits: int = Q3_BITS, group: int = GROUP):
    rows, cols = w.shape
    if cols % group != 0:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    g = w.reshape(rows, cols // group, group)
    qmax = (1 << (bits - 1)) - 1
    absmax = np.abs(g).max(axis=2, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    codes = np.clip(np.rint(g / scale), -qmax - 1, qmax).astype(np.int32)
    deq = (codes * scale).reshape(rows, cols)
    n_groups = rows * (cols // group)
    storage_bits = bits * w.size + SCALE_BITS * n_groups
    return deq.astype(np.float32), storage_bits


def q3_mlp(weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    out = {}
    bits = 0
    for k, w in weights.items():
        deq, b = quantize_group(w)
        out[k] = deq
        bits += b
    return out, bits


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
    """min(r, 1/r) on per-row mean and per-unit min. Rejects 0.01*W."""

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


def null_cosine_constant_mean(Y) -> float:
    mu = Y.mean(axis=0, keepdims=True)
    return row_cosine(Y, np.broadcast_to(mu, Y.shape))


def score_pair(Y, Yh) -> dict:
    return {
        "rel_fro": rel_fro(Y, Yh),
        "cosine": row_cosine(Y, Yh),
        "gain": gain_score(Y, Yh),
        "worst_unit": worst_unit(Y, Yh),
    }


def isotropic_probe(X_ref: np.ndarray, n: int = N_PROBE, seed: int = 0) -> np.ndarray:
    """Unused probe axis: Gaussian with matched per-channel RMS. Not unit-sphere
    — SwiGLU is nonlinear, wrong scale is a different operating point."""
    rng = np.random.default_rng(seed)
    rms = X_ref.std(axis=0, keepdims=True).astype(np.float32)
    rms = np.where(rms > 0, rms, 1.0)
    return (rng.standard_normal((n, X_ref.shape[1])).astype(np.float32) * rms)


def doctor_mlp(X, y_fn, yh_fn, ref=None, seed=0) -> dict:
    Y = y_fn(X)
    Yh = yh_fn(X)
    obs = score_pair(Y, Yh)
    probed = []
    ss = np.random.SeedSequence(seed)
    for child in ss.spawn(N_PROBE_SETS):
        P = isotropic_probe(X, N_PROBE, seed=int(child.generate_state(1)[0]))
        probed.append(score_pair(y_fn(P), yh_fn(P)))
    axes = {
        "observed": obs["cosine"],
        "probed": min(p["cosine"] for p in probed),
        "worst_unit": min(obs["worst_unit"], min(p["worst_unit"] for p in probed)),
        "gain": min(obs["gain"], min(p["gain"] for p in probed)),
        "observed_rel_fro": obs["rel_fro"],
        "probed_rel_fro": max(p["rel_fro"] for p in probed),
        "probed_gain": min(p["gain"] for p in probed),
    }
    if ref is None:
        # Absolute floors are the synthetic self-check only. Probe uses relative.
        axes.update({"healthy": None, "mode": "unreferenced", "worst_axis": None})
        return axes
    deficits = {k: axes[k] - (ref[k] - AXIS_MARGIN[k]) for k in AXIS_MARGIN}
    worst = min(deficits, key=deficits.get)
    axes.update(
        {
            "mode": "relative_to_q3",
            "deficit": {k: float(v) for k, v in deficits.items()},
            "worst_axis": worst,
            "healthy": bool(deficits[worst] >= 0.0),
        }
    )
    return axes


def channel_importance(W: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.linalg.norm(W["gate"], axis=1)
        * np.linalg.norm(W["up"], axis=1)
        * np.linalg.norm(W["down"], axis=0)
    )


def prune_mlp(W: dict[str, np.ndarray], width: int) -> dict[str, np.ndarray]:
    imp = channel_importance(W)
    idx = np.argpartition(imp, -width)[-width:]
    idx.sort()
    return {
        "gate": np.ascontiguousarray(W["gate"][idx]),
        "up": np.ascontiguousarray(W["up"][idx]),
        "down": np.ascontiguousarray(W["down"][:, idx]),
        "idx": idx,
    }


def numel_mlp(width: int = INTERMEDIATE) -> int:
    return 3 * HIDDEN * width


def bpw_pack(storage_bits: int, active_bits: int, orig_numel: int) -> dict:
    return {
        "storage_bits": int(storage_bits),
        "active_bits": int(active_bits),
        "storage_bpw": storage_bits / orig_numel,
        "active_bpw": active_bits / orig_numel,
        "orig_numel": int(orig_numel),
        "storage_bytes": storage_bits / 8.0,
        "active_bytes": active_bits / 8.0,
    }


def f16_cast(W: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        k: (v.astype(np.float16).astype(np.float32) if k != "idx" else v)
        for k, v in W.items()
    }


def output_pca_lowrank(X: np.ndarray, W: np.ndarray, rank: int) -> np.ndarray:
    """NNS-014 control: project W onto top-`rank` PCs of the *fit-set* outputs."""
    Y = X @ W.T
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Vt rows are output principal axes. n x out, out=5120, n~4k.
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    P = Vt[:rank]
    return (P.T @ (P @ W)).astype(np.float32)


def weight_svd_lowrank(W: np.ndarray, rank: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return ((U[:, :rank] * S[:rank]) @ Vt[:rank]).astype(np.float32)


def distill_thin_swiglu(
    X_train,
    Y_train,
    X_val,
    Y_val,
    init: dict[str, np.ndarray],
    steps: int,
    batch: int,
    lr: float,
):
    """Train a thinner SwiGLU to match teacher MLP outputs. Returns f16-cast
    weights, train curve, best-val checkpoint (val is carved from the FIT split;
    official HOLD is never used here)."""
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 8))
    device = torch.device("cpu")
    Wg = torch.nn.Parameter(torch.from_numpy(init["gate"].copy()).to(device))
    Wu = torch.nn.Parameter(torch.from_numpy(init["up"].copy()).to(device))
    Wd = torch.nn.Parameter(torch.from_numpy(init["down"].copy()).to(device))
    opt = torch.optim.Adam([Wg, Wu, Wd], lr=lr)
    Xt = torch.from_numpy(np.ascontiguousarray(X_train))
    Yt = torch.from_numpy(np.ascontiguousarray(Y_train))
    Xv = torch.from_numpy(np.ascontiguousarray(X_val))
    Yv = torch.from_numpy(np.ascontiguousarray(Y_val))
    n = Xt.shape[0]
    rng = np.random.default_rng(0)
    curve = []
    best = {"val_rel_fro": 1e9, "step": -1, "state": None}

    def forward(xb, wg, wu, wd):
        return torch.nn.functional.silu(xb @ wg.T) * (xb @ wu.T) @ wd.T

    @torch.no_grad()
    def eval_rel(x, y):
        yh = forward(x, Wg, Wu, Wd)
        return float(torch.linalg.vector_norm(yh - y) / torch.linalg.vector_norm(y))

    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        xb = Xt[idx]
        yb = Yt[idx]
        opt.zero_grad(set_to_none=True)
        yh = forward(xb, Wg, Wu, Wd)
        # Relative Frobenius on the batch — same family as the GO metric, and
        # less willing to ignore magnitude than raw MSE.
        loss = torch.linalg.vector_norm(yh - yb) / (torch.linalg.vector_norm(yb) + 1e-12)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Wg, Wu, Wd], GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps - 1:
            with torch.no_grad():
                tr = float(loss.detach().cpu())
            vr = eval_rel(Xv, Yv)
            curve.append({"step": step, "train_rel_fro": tr, "val_rel_fro": vr})
            if vr < best["val_rel_fro"]:
                best = {
                    "val_rel_fro": vr,
                    "step": step,
                    "state": {
                        "gate": Wg.detach().cpu().numpy().copy(),
                        "up": Wu.detach().cpu().numpy().copy(),
                        "down": Wd.detach().cpu().numpy().copy(),
                    },
                }
            print(
                f"      step {step:3d}/{steps}  train_rel_fro={tr:.4f}  val_rel_fro={vr:.4f}"
                f"{'  *best' if step == best['step'] else ''}",
                flush=True,
            )
    if best["state"] is None:
        best["state"] = {
            "gate": Wg.detach().cpu().numpy(),
            "up": Wu.detach().cpu().numpy(),
            "down": Wd.detach().cpu().numpy(),
        }
    out = f16_cast(best["state"])
    plateau = False
    if len(curve) >= 3:
        tail = [c["val_rel_fro"] for c in curve[-3:]]
        plateau = (max(tail) - min(tail)) < 0.005
    still_dropping = False
    if len(curve) >= 2:
        still_dropping = curve[-1]["val_rel_fro"] < curve[-2]["val_rel_fro"] - 0.002
    return {
        "weights": out,
        "curve": curve,
        "best_step": best["step"],
        "best_val_rel_fro": best["val_rel_fro"],
        "wall_s": time.time() - t0,
        "n_params": int(numel_mlp(init["gate"].shape[0])),
        "plateau": plateau,
        "still_dropping_at_end": still_dropping,
    }


def make_fn(W):
    def fn(X):
        y, _ = mlp_forward_np(X, W["gate"], W["up"], W["down"])
        return y

    return fn


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def main() -> int:
    _ensure_torch()
    t_all = time.time()
    print("NOETIC MLP DISTILL PROBE")
    print("=" * 72)
    head = git_head()
    print(f"git_head: {head}")
    print(f"repo:     {ROOT}")
    print(f"python:   {sys.executable}")
    try:
        import torch

        print(f"torch:    {torch.__version__} mps={torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"torch:    import failed after ensure ({e})")

    parent = find_parent()
    cap = find_capture()
    print(f"parent:   {parent}")
    print(f"capture:  {cap}")
    print("teacher:  qualified parent BF16 tensors; llama-server:52484 NOT used (Q5_K, NS-015)")
    print("metal:    no TPS claim; pairing/page-cache discipline N/A for this quality probe")
    print()

    manifest_path = cap / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        n_rows = load_X(cap, DISTILL_LAYERS[0]).shape[0]
        manifest = {
            "total_tokens": n_rows,
            "hidden": HIDDEN,
            "input": "post_attn_norm",
            "split_rule": "last 20% rows = hold (no prompt manifest)",
            "manifest": None,
        }

    X0 = load_X(cap, DISTILL_LAYERS[0])
    n_tokens = X0.shape[0]
    if manifest.get("manifest"):
        fit_idx, hold_idx = [], []
        for m in manifest["manifest"]:
            sl = np.arange(m["row_start"], m["row_start"] + m["n_tokens"])
            (hold_idx if m["split"] == "hold" else fit_idx).append(sl)
        fit_idx = np.concatenate(fit_idx)
        hold_idx = np.concatenate(hold_idx)
    else:
        n_hold = max(256, n_tokens // 5)
        fit_idx = np.arange(0, n_tokens - n_hold)
        hold_idx = np.arange(n_tokens - n_hold, n_tokens)

    rng = np.random.default_rng(0)
    fit_perm = rng.permutation(fit_idx)
    n_val = max(256, int(0.15 * len(fit_perm)))
    val_idx = fit_perm[:n_val]
    train_idx = fit_perm[n_val:]
    print(
        f"CAPTURE  site=post_attn_norm  tokens={n_tokens}  "
        f"train={len(train_idx)} val={len(val_idx)} hold={len(hold_idx)}"
    )
    print(f"         source={cap}")
    print(f"         split={manifest.get('split_rule')}")
    print(f"         families={manifest.get('families')}")
    print()

    orig_numel = numel_mlp(INTERMEDIATE)
    q3_storage_bits_full = int(Q3_STORAGE_BPW * orig_numel)
    # fused q3 reads packed payload; decoded-f16 q3 reads 16 b/elem
    q3_fused_active_bits = q3_storage_bits_full
    q3_decoded_active_bits = int(F16_BPW * orig_numel)

    results = {
        "schema": "hawking.headless.noetic_mlp_distill_probe.v1",
        "obligation": "NNS-015",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "what_this_is": (
            "function-space fit of a thinner SwiGLU to the MLP map on real X; "
            "not a student-model distillation arc"
        ),
        "parent": str(parent),
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm",
            "post_swiglu": "computed as silu(X@Wg.T)*(X@Wu.T) from qualified-parent BF16",
            "n_tokens": int(n_tokens),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_hold": int(len(hold_idx)),
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward, "
                "11269 tokens, 6 families, last 3 prompts/family held out. "
                "Not Gaussian. Not Q5_K llama-server."
            ),
            "manifest_families": manifest.get("families"),
            "split_rule": manifest.get("split_rule"),
        },
        "accounting": {
            "q3_storage_bpw": Q3_STORAGE_BPW,
            "q3_fused_active_bpw": Q3_STORAGE_BPW,
            "q3_decoded_f16_active_bpw": F16_BPW,
            "scale_bits": SCALE_BITS,
            "group": GROUP,
            "note": (
                "PHASE_B_HYBRID quoted 3.5 b/elem (f32 scales). Native family "
                "(G033/G034/G105) is 3 + 16/64 = 3.25 with f16 scales. This probe "
                "matches bits against 3.25. Storage and active are both reported."
            ),
            "orig_numel_per_mlp": orig_numel,
            "q3_storage_bytes": q3_storage_bits_full / 8.0,
            "q3_fused_active_bytes": q3_fused_active_bits / 8.0,
            "q3_decoded_active_bytes": q3_decoded_active_bits / 8.0,
            "student_f16_active_equals_storage": True,
            "materially_fewer_ratio": MATERIAL_BYTE_RATIO,
        },
        "widths": list(WIDTHS),
        "layers": list(DISTILL_LAYERS),
        "train": {
            "steps": TRAIN_STEPS,
            "batch": BATCH,
            "lr": LR,
            "grad_clip": GRAD_CLIP,
            "init": "teacher channel-prune by ||Wg_i||*||Wu_i||*||Wd_i||",
            "hold_never_used_in_fit": True,
            "val_carved_from_fit": True,
            "unused_probe_axis": "isotropic Gaussian, per-channel RMS matched to train X; Doctor observed scored on HOLD",
        },
        "layers_out": {},
        "scale_trap": {},
        "null_baseline": {},
        "one_hop": {},
        "nns014_control": {},
        "watched_fail": [],
        "verdict": {},
    }

    # ------------------------------------------------------------------ scale trap
    print("SCALE TRAP  (cosine is scale-invariant; gain must refuse 0.01·W)")
    W0 = load_mlp(parent, DISTILL_LAYERS[0])
    X_trap = X0[train_idx[:512]]
    Y_trap, S_trap = mlp_forward_np(X_trap, W0["gate"], W0["up"], W0["down"])
    # Linear organ: the historical blindness. 0.01*Wd leaves direction intact.
    Yd = S_trap @ W0["down"].T
    Yd_s = S_trap @ (0.01 * W0["down"]).T
    trap_lin = score_pair(Yd, Yd_s)
    trap_lin["organ"] = "down_proj"
    # Full MLP: SwiGLU is nonlinear so cosine is not exactly 1, but still high
    # enough to fool a cosine-only screen while magnitude dies.
    Y_s, _ = mlp_forward_np(
        X_trap, 0.01 * W0["gate"], 0.01 * W0["up"], 0.01 * W0["down"]
    )
    trap_mlp = score_pair(Y_trap, Y_s)
    trap_mlp["organ"] = "mlp_swiglu"
    trap = {
        "linear_down_proj": trap_lin,
        "mlp_swiglu": trap_mlp,
        "pass_if": (
            "linear: cosine~1 and gain~0.01; MLP: cosine still high, gain~0, rel_fro~1. "
            "GO uses rel_fro+gain, never cosine alone."
        ),
        "rejects_scaled_artifact": bool(
            trap_lin["cosine"] > 0.99
            and trap_lin["gain"] < 0.05
            and trap_mlp["gain"] < 0.05
            and trap_mlp["rel_fro"] > 0.9
        ),
    }
    print(
        f"  linear down_proj 0.01*W  cosine={trap_lin['cosine']:.6f}  "
        f"gain={trap_lin['gain']:.6f}  rel_fro={trap_lin['rel_fro']:.6f}"
    )
    print(
        f"  MLP SwiGLU    0.01*W  cosine={trap_mlp['cosine']:.6f}  "
        f"gain={trap_mlp['gain']:.6f}  rel_fro={trap_mlp['rel_fro']:.6f}"
    )
    print(f"  rejects={trap['rejects_scaled_artifact']}")
    if not trap["rejects_scaled_artifact"]:
        print("FAIL: scale trap did not reject 0.01*W — GO metric is blind")
        results["scale_trap"] = trap
        results["verdict"] = {"decision": "NO-GO", "reason": "scale trap failed"}
        _write(results)
        return 2
    results["scale_trap"] = trap
    results["watched_fail"].append(
        f"0.01*W linear down_proj cosine={trap_lin['cosine']:.6f} (blind) "
        f"gain={trap_lin['gain']:.6f} (rejects); MLP cosine={trap_mlp['cosine']:.6f} "
        f"gain={trap_mlp['gain']:.6f} rel_fro={trap_mlp['rel_fro']:.6f}. "
        f"Cosine is not a GO metric."
    )
    print()

    # ------------------------------------------------------------------ per layer
    hop_cache = {}  # layer -> {X, Y_teacher, err_by_name}

    for layer in DISTILL_LAYERS:
        print(f"LAYER {layer}")
        print("-" * 72)
        W = load_mlp(parent, layer) if layer != DISTILL_LAYERS[0] else W0
        X = load_X(cap, layer) if layer != DISTILL_LAYERS[0] else X0
        print(f"  loaded Wg {W['gate'].shape} Wu {W['up'].shape} Wd {W['down'].shape}")

        Y_all, S_all = mlp_forward_np(X, W["gate"], W["up"], W["down"])
        print(
            f"  post-SwiGLU captured via teacher: shape={S_all.shape}  "
            f"rms={float(S_all.std()):.5f}  frac_|s|<1%max={float((np.abs(S_all) < 0.01 * np.abs(S_all).max()).mean()):.3f}"
        )

        Y_train, Y_val, Y_hold = Y_all[train_idx], Y_all[val_idx], Y_all[hold_idx]
        X_train, X_val, X_hold = X[train_idx], X[val_idx], X[hold_idx]
        S_train, S_hold = S_all[train_idx], S_all[hold_idx]

        null_hold = null_cosine_constant_mean(Y_hold)
        print(f"  null (constant-mean) hold cosine={null_hold:.4f}  (GLM folklore 0.898; measured, not assumed)")
        results["null_baseline"][str(layer)] = {
            "hold_constant_mean_cosine": null_hold,
            "note": "every cosine below is paired with this null; raw cosine near the null is not fidelity",
        }

        def y_teacher(Z):
            y, _ = mlp_forward_np(Z, W["gate"], W["up"], W["down"])
            return y

        # q3 incumbent
        print("  q3 grouped-absmax g64 ...")
        Wq, qbits = q3_mlp(W)
        Yq_hold, _ = mlp_forward_np(X_hold, Wq["gate"], Wq["up"], Wq["down"])
        Yq_train, _ = mlp_forward_np(X_train, Wq["gate"], Wq["up"], Wq["down"])
        q3_hold = score_pair(Y_hold, Yq_hold)
        q3_train = score_pair(Y_train, Yq_train)
        q3_hold["cosine_minus_null"] = q3_hold["cosine"] - null_hold
        q3_fn = make_fn(Wq)
        # Doctor observed is scored on HOLD (unused in fitting). Isotropic probe
        # is a second axis never used in fitting. Train X is only for the fit.
        q3_doc = doctor_mlp(X_hold, y_teacher, q3_fn, ref=None, seed=layer + 1)
        q3_ref = {k: q3_doc[k] for k in AXIS_MARGIN}
        q3_doc = doctor_mlp(X_hold, y_teacher, q3_fn, ref=q3_ref, seed=layer + 1)
        q3_bpw = bpw_pack(qbits, qbits, orig_numel)  # fused: active = storage
        q3_bpw_decoded = bpw_pack(qbits, int(F16_BPW * orig_numel), orig_numel)
        print(
            f"    hold rel_fro={q3_hold['rel_fro']:.4f} cosine={q3_hold['cosine']:.4f} "
            f"(null {null_hold:.4f}, Δ={q3_hold['cosine_minus_null']:.4f}) "
            f"gain={q3_hold['gain']:.4f}"
        )
        print(
            f"    storage_bpw={q3_bpw['storage_bpw']:.4f}  "
            f"active_fused_bpw={q3_bpw['active_bpw']:.4f}  "
            f"active_decoded_f16_bpw={q3_bpw_decoded['active_bpw']:.4f}"
        )
        print(f"    doctor healthy-vs-self={q3_doc['healthy']} (reference)")

        layer_out = {
            "q3": {
                "hold": q3_hold,
                "train": q3_train,
                "doctor": q3_doc,
                "bpw_fused": q3_bpw,
                "bpw_decoded_f16": q3_bpw_decoded,
                "health_verdict": "HEALTHY (incumbent reference)",
            },
            "widths": {},
        }
        hop_cache[layer] = {
            "X": X,
            "Y": Y_all,
            "err": {"q3": Yq_hold - Y_hold},  # hold-aligned; hop uses full later
            "W": W,
        }

        # NNS-014 control on down_proj (linear), layer 31 or both
        r_matched = int(round((Q3_STORAGE_BPW / F16_BPW) * INTERMEDIATE * HIDDEN / (INTERMEDIATE + HIDDEN)))
        # rank-803 is the historical matched-byte point for down_proj alone
        r_hist = 803
        print(f"  NNS-014 control  down_proj output-PCA rank={r_hist} (historical matched-byte)")
        Wd_func = output_pca_lowrank(S_train, W["down"], r_hist)
        Yd_t_hold = S_hold @ W["down"].T
        Yd_f_hold = S_hold @ Wd_func.T
        Yd_f_train = S_train @ Wd_func.T
        Yd_t_train = S_train @ W["down"].T
        nns_hold = score_pair(Yd_t_hold, Yd_f_hold)
        nns_train = score_pair(Yd_t_train, Yd_f_train)
        Wd_wsvd = weight_svd_lowrank(W["down"], r_hist)
        nns_w_hold = score_pair(Yd_t_hold, S_hold @ Wd_wsvd.T)
        Wd_q3, q3_down_bits = quantize_group(W["down"])
        nns_q3_hold = score_pair(Yd_t_hold, S_hold @ Wd_q3.T)
        down_numel = int(W["down"].size)
        func_bits = int(F16_BPW * r_hist * (HIDDEN + INTERMEDIATE))
        print(
            f"    functional PCA  train rel_fro={nns_train['rel_fro']:.4f}  "
            f"hold rel_fro={nns_hold['rel_fro']:.4f}  "
            f"q3 hold={nns_q3_hold['rel_fro']:.4f}  "
            f"weight-SVD hold={nns_w_hold['rel_fro']:.4f}"
        )
        overfit = nns_train["rel_fro"] < nns_q3_hold["rel_fro"] and nns_hold["rel_fro"] > nns_q3_hold["rel_fro"]
        results["nns014_control"][str(layer)] = {
            "rank": r_hist,
            "matched_byte_rank_formula": r_matched,
            "functional_pca_train": nns_train,
            "functional_pca_hold": nns_hold,
            "weight_svd_hold": nns_w_hold,
            "q3_hold": nns_q3_hold,
            "bpw_functional_factors": bpw_pack(func_bits, func_bits, down_numel),
            "bpw_q3": bpw_pack(q3_down_bits, q3_down_bits, down_numel),
            "overfit_as_nns014": overfit,
            "health_verdict": "UNHEALTHY as a generalizing codec" if overfit else "see numbers",
        }
        if overfit:
            results["watched_fail"].append(
                f"L{layer} down_proj output-PCA r={r_hist} train {nns_train['rel_fro']:.4f} "
                f"< q3 {nns_q3_hold['rel_fro']:.4f} but hold {nns_hold['rel_fro']:.4f} > q3 "
                f"(NNS-014 replicated on this capture)"
            )

        for width in WIDTHS:
            print(f"  width I'={width}  (f16 storage_bpw={F16_BPW * width / INTERMEDIATE:.4f} vs q3 {Q3_STORAGE_BPW})")
            pruned = prune_mlp(W, width)
            pruned_f16 = f16_cast(pruned)
            student_bits = int(F16_BPW * numel_mlp(width))
            student_bpw = bpw_pack(student_bits, student_bits, orig_numel)
            underdet = numel_mlp(width) > len(train_idx) * HIDDEN
            print(
                f"    n_params={numel_mlp(width):,}  n_train_scalars={len(train_idx)*HIDDEN:,}  "
                f"underdetermined={underdet}  byte_ratio_vs_q3_fused={student_bpw['active_bytes']/q3_bpw['active_bytes']:.3f}"
            )

            # weight-space: prune, f16
            Yp_hold, _ = mlp_forward_np(
                X_hold, pruned_f16["gate"], pruned_f16["up"], pruned_f16["down"]
            )
            Yp_train, _ = mlp_forward_np(
                X_train, pruned_f16["gate"], pruned_f16["up"], pruned_f16["down"]
            )
            prune_hold = score_pair(Y_hold, Yp_hold)
            prune_train = score_pair(Y_train, Yp_train)
            prune_hold["cosine_minus_null"] = prune_hold["cosine"] - null_hold
            prune_fn = make_fn(pruned_f16)
            prune_doc = doctor_mlp(X_hold, y_teacher, prune_fn, ref=q3_ref, seed=layer + 17)
            print(
                f"    weight-space prune  train rel_fro={prune_train['rel_fro']:.4f}  "
                f"hold rel_fro={prune_hold['rel_fro']:.4f}  gain={prune_hold['gain']:.4f}  "
                f"doctor_healthy={prune_doc['healthy']}  worst={prune_doc.get('worst_axis')}"
            )

            # function-space: distill
            print(f"    function-space distill ({TRAIN_STEPS} Adam steps, init=prune) ...")
            dist = distill_thin_swiglu(
                X_train, Y_train, X_val, Y_val, pruned, TRAIN_STEPS, BATCH, LR
            )
            Wd_s = dist["weights"]
            Ys_hold, _ = mlp_forward_np(X_hold, Wd_s["gate"], Wd_s["up"], Wd_s["down"])
            Ys_train, _ = mlp_forward_np(X_train, Wd_s["gate"], Wd_s["up"], Wd_s["down"])
            dist_hold = score_pair(Y_hold, Ys_hold)
            dist_train = score_pair(Y_train, Ys_train)
            dist_hold["cosine_minus_null"] = dist_hold["cosine"] - null_hold
            dist_fn = make_fn(Wd_s)
            dist_doc = doctor_mlp(X_hold, y_teacher, dist_fn, ref=q3_ref, seed=layer + 29)
            beats_q3_hold = dist_hold["rel_fro"] <= q3_hold["rel_fro"] + QUALITY_REL_TOL
            fewer = student_bpw["active_bytes"] <= MATERIAL_BYTE_RATIO * q3_bpw["active_bytes"]
            health = "HEALTHY" if dist_doc["healthy"] and beats_q3_hold else "UNHEALTHY"
            print(
                f"    distill  train rel_fro={dist_train['rel_fro']:.4f}  "
                f"hold rel_fro={dist_hold['rel_fro']:.4f}  gain={dist_hold['gain']:.4f}  "
                f"doctor_healthy={dist_doc['healthy']}  worst={dist_doc.get('worst_axis')}"
            )
            print(
                f"    vs q3 hold Δrel_fro={dist_hold['rel_fro']-q3_hold['rel_fro']:+.4f}  "
                f"vs prune hold Δrel_fro={dist_hold['rel_fro']-prune_hold['rel_fro']:+.4f}  "
                f"beats_q3={beats_q3_hold}  fewer_bytes={fewer}  {health}"
            )
            print(
                f"    storage_bpw={student_bpw['storage_bpw']:.4f}  "
                f"active_bpw={student_bpw['active_bpw']:.4f}  "
                f"(q3 fused active {q3_bpw['active_bpw']:.4f} / decoded {q3_bpw_decoded['active_bpw']:.4f})"
            )
            print(
                f"    plateau={dist['plateau']} still_dropping={dist['still_dropping_at_end']} "
                f"best_step={dist['best_step']} train_wall_s={dist['wall_s']:.1f}"
            )

            layer_out["widths"][str(width)] = {
                "width": width,
                "underdetermined": underdet,
                "bpw": student_bpw,
                "byte_ratio_vs_q3_fused_active": student_bpw["active_bytes"] / q3_bpw["active_bytes"],
                "weight_space_prune": {
                    "hold": prune_hold,
                    "train": prune_train,
                    "doctor": prune_doc,
                    "health_verdict": "HEALTHY" if prune_doc["healthy"] else "UNHEALTHY",
                },
                "function_space_distill": {
                    "hold": dist_hold,
                    "train": dist_train,
                    "doctor": dist_doc,
                    "best_step": dist["best_step"],
                    "best_val_rel_fro": dist["best_val_rel_fro"],
                    "plateau": dist["plateau"],
                    "still_dropping_at_end": dist["still_dropping_at_end"],
                    "n_params": dist["n_params"],
                    "curve": dist["curve"],
                    "train_wall_s": dist["wall_s"],
                    "beats_q3_hold": beats_q3_hold,
                    "materially_fewer_active_bytes": fewer,
                    "health_verdict": health,
                },
            }
            hop_cache[layer]["err"][f"prune_{width}"] = Yp_hold - Y_hold
            hop_cache[layer]["err"][f"distill_{width}"] = Ys_hold - Y_hold
            hop_cache[layer][f"Yh_hold_distill_{width}"] = Ys_hold
            hop_cache[layer][f"W_distill_{width}"] = Wd_s

            if dist_train["rel_fro"] < q3_train["rel_fro"] and not beats_q3_hold:
                results["watched_fail"].append(
                    f"L{layer} I'={width} distill WINS train ({dist_train['rel_fro']:.4f} < q3 "
                    f"{q3_train['rel_fro']:.4f}) and LOSES hold ({dist_hold['rel_fro']:.4f} vs "
                    f"{q3_hold['rel_fro']:.4f}) — Goodhart/overfit, same shape as NNS-014"
                )
            if not dist_doc["healthy"]:
                results["watched_fail"].append(
                    f"L{layer} I'={width} distill Doctor UNHEALTHY vs q3 on {dist_doc.get('worst_axis')} "
                    f"(observed={dist_doc['observed']:.4f} probed={dist_doc['probed']:.4f} "
                    f"gain={dist_doc['gain']:.4f} worst_unit={dist_doc['worst_unit']:.4f})"
                )

        results["layers_out"][str(layer)] = layer_out
        # free big arrays we no longer need except hop
        if layer not in (HOP_SRC, HOP_DST):
            hop_cache[layer].pop("X", None)
            hop_cache[layer].pop("Y", None)
        print()

    # ------------------------------------------------------------------ one hop
    print("ONE-HOP  residual-injection L0 → L1 (not a full layer replay; mixer skipped)")
    print("-" * 72)
    X1 = load_X(cap, HOP_DST)
    W1 = load_mlp(parent, HOP_DST)
    # hold-aligned rows
    X1_hold = X1[hold_idx]
    Y1_hold, _ = mlp_forward_np(X1_hold, W1["gate"], W1["up"], W1["down"])
    # teacher residual error at L0 on hold tokens
    Y0_hold = hop_cache[HOP_SRC]["Y"][hold_idx]
    hop_rows = {}
    # q3 hop
    W0q, _ = q3_mlp(hop_cache[HOP_SRC]["W"])
    Y0q_hold, _ = mlp_forward_np(
        hop_cache[HOP_SRC]["X"][hold_idx], W0q["gate"], W0q["up"], W0q["down"]
    )
    e_q3 = Y0q_hold - Y0_hold
    Y1_q3hop, _ = mlp_forward_np(X1_hold + e_q3, W1["gate"], W1["up"], W1["down"])
    hop_q3 = score_pair(Y1_hold, Y1_q3hop)
    hop_q3["injected_rel"] = rel_fro(Y0_hold, Y0q_hold)
    hop_q3["amplification"] = hop_q3["rel_fro"] / max(hop_q3["injected_rel"], 1e-12)
    print(
        f"  q3      inject_rel={hop_q3['injected_rel']:.4f}  L1_rel_fro={hop_q3['rel_fro']:.4f}  "
        f"amp={hop_q3['amplification']:.3f}"
    )
    hop_rows["q3"] = hop_q3

    for width in WIDTHS:
        key = f"distill_{width}"
        e = hop_cache[HOP_SRC]["err"][key]
        Y1h, _ = mlp_forward_np(X1_hold + e, W1["gate"], W1["up"], W1["down"])
        h = score_pair(Y1_hold, Y1h)
        h["injected_rel"] = rel_fro(Y0_hold, Y0_hold + e)
        h["amplification"] = h["rel_fro"] / max(h["injected_rel"], 1e-12)
        h["amp_vs_q3"] = h["amplification"] / max(hop_q3["amplification"], 1e-12)
        hop_rows[key] = h
        print(
            f"  dist I'={width} inject_rel={h['injected_rel']:.4f}  L1_rel_fro={h['rel_fro']:.4f}  "
            f"amp={h['amplification']:.3f}  amp_vs_q3={h['amp_vs_q3']:.3f}"
        )
        e_p = hop_cache[HOP_SRC]["err"][f"prune_{width}"]
        Y1p, _ = mlp_forward_np(X1_hold + e_p, W1["gate"], W1["up"], W1["down"])
        hp = score_pair(Y1_hold, Y1p)
        hp["injected_rel"] = rel_fro(Y0_hold, Y0_hold + e_p)
        hp["amplification"] = hp["rel_fro"] / max(hp["injected_rel"], 1e-12)
        hop_rows[f"prune_{width}"] = hp
        print(
            f"  prune I'={width} inject_rel={hp['injected_rel']:.4f}  L1_rel_fro={hp['rel_fro']:.4f}  "
            f"amp={hp['amplification']:.3f}"
        )

    results["one_hop"] = {
        "kind": "residual_injection",
        "claim_boundary": "ONE hop only. Mixer/attn of L1 is not replayed; e_L0 is added to captured L1 MLP input. First-order residual model, not a 64-layer cascade.",
        "src_layer": HOP_SRC,
        "dst_layer": HOP_DST,
        "n_hold": int(len(hold_idx)),
        "rows": hop_rows,
    }
    results["watched_fail"].append(
        f"one-hop residual injection L{HOP_SRC}→L{HOP_DST}: q3 amp={hop_q3['amplification']:.3f}; "
        + ", ".join(
            f"distill I'={w} amp={hop_rows[f'distill_{w}']['amplification']:.3f} "
            f"(vs q3 {hop_rows[f'distill_{w}']['amp_vs_q3']:.3f}x)"
            for w in WIDTHS
        )
    )

    # ------------------------------------------------------------------ verdict
    print()
    print("VERDICT")
    print("=" * 72)
    go_reasons = []
    nogo_reasons = []
    deciding = []
    for layer in DISTILL_LAYERS:
        q3h = results["layers_out"][str(layer)]["q3"]["hold"]["rel_fro"]
        for width in WIDTHS:
            rec = results["layers_out"][str(layer)]["widths"][str(width)]
            d = rec["function_space_distill"]
            p = rec["weight_space_prune"]
            ratio = rec["byte_ratio_vs_q3_fused_active"]
            gap = d["hold"]["rel_fro"] - q3h
            hop_ok = True
            if layer == HOP_SRC:
                amp_vs = hop_rows[f"distill_{width}"]["amp_vs_q3"]
                hop_ok = amp_vs <= HOP_AMPLIFICATION_CAP
            row = {
                "layer": layer,
                "width": width,
                "hold_rel_fro": d["hold"]["rel_fro"],
                "q3_hold_rel_fro": q3h,
                "quality_gap": gap,
                "prune_hold_rel_fro": p["hold"]["rel_fro"],
                "distill_beats_prune": d["hold"]["rel_fro"] < p["hold"]["rel_fro"],
                "byte_ratio_vs_q3_fused_active": ratio,
                "storage_bpw": rec["bpw"]["storage_bpw"],
                "active_bpw": rec["bpw"]["active_bpw"],
                "q3_storage_bpw": Q3_STORAGE_BPW,
                "q3_active_fused_bpw": Q3_STORAGE_BPW,
                "doctor_healthy": d["doctor"]["healthy"],
                "health_verdict": d["health_verdict"],
                "hop_ok": hop_ok,
                "plateau": d["plateau"],
                "still_dropping_at_end": d["still_dropping_at_end"],
                "underdetermined": rec["underdetermined"],
            }
            pass_quality = gap <= QUALITY_REL_TOL
            pass_bytes = ratio <= MATERIAL_BYTE_RATIO
            pass_doc = bool(d["doctor"]["healthy"])
            row["pass_quality"] = pass_quality
            row["pass_bytes"] = pass_bytes
            row["pass_doctor"] = pass_doc
            row["pass_all"] = bool(pass_quality and pass_bytes and pass_doc and hop_ok)
            deciding.append(row)
            tag = f"L{layer} I'={width}"
            if row["pass_all"]:
                go_reasons.append(
                    f"{tag} hold rel_fro {d['hold']['rel_fro']:.4f} <= q3 {q3h:.4f}, "
                    f"active bytes {ratio:.3f} of q3 fused, Doctor healthy, hop ok"
                )
            else:
                bits = []
                if not pass_quality:
                    bits.append(f"hold rel_fro {d['hold']['rel_fro']:.4f} > q3 {q3h:.4f} (gap {gap:+.4f})")
                if not pass_bytes:
                    bits.append(f"active bytes {ratio:.3f} of q3 fused (need ≤ {MATERIAL_BYTE_RATIO})")
                if not pass_doc:
                    bits.append(f"Doctor UNHEALTHY on {d['doctor'].get('worst_axis')}")
                if not hop_ok:
                    bits.append("one-hop amplification vs q3 exceeds cap")
                nogo_reasons.append(f"{tag}: " + "; ".join(bits))

    # Headline candidate: I'=2560 (materially fewer). Matched 3536 cannot GO on bytes.
    headline_width = 2560
    headline_pass = all(
        r["pass_all"] for r in deciding if r["width"] == headline_width
    )
    any_pass = any(r["pass_all"] for r in deciding)
    # Campaign-worthiness: a near-matched width that beats q3 held-out AND doctor
    # would still not satisfy reopen (not fewer bytes) but would justify more work.
    near_matched_quality = [
        r for r in deciding if r["width"] == 3536 and r["pass_quality"] and r["pass_doctor"]
    ]

    if any_pass and headline_pass:
        decision = "GO"
        deciding_number = min(
            r["quality_gap"] for r in deciding if r["width"] == headline_width
        )
        deciding_note = (
            f"headline I'={headline_width} held-out rel-fro gap vs q3 (max-over-layers of "
            f"distill-q3; more negative is better). Must be <= 0 with byte_ratio<=0.90."
        )
    else:
        decision = "NO-GO"
        # The number that decides: worst headline hold gap vs q3.
        hl = [r for r in deciding if r["width"] == headline_width]
        if hl:
            deciding_number = max(r["quality_gap"] for r in hl)
            deciding_note = (
                f"worst L0/L31 hold rel-fro gap of I'={headline_width} vs q3 "
                f"(positive = worse than q3). Reopen needs this <= 0 AND byte_ratio "
                f"<= {MATERIAL_BYTE_RATIO} AND Doctor healthy."
            )
        else:
            deciding_number = None
            deciding_note = "headline width missing"

    # Do not GO a campaign on undertrained hope if the gap is large.
    still_dropping = any(
        r["still_dropping_at_end"] for r in deciding if r["width"] == headline_width
    )
    campaign_note = None
    if decision == "NO-GO":
        if near_matched_quality and not headline_pass:
            campaign_note = (
                "Near-matched I'=3536 hit q3 quality somewhere but is not fewer active "
                "bytes. That is not the reopen condition. No campaign: q3 remains Pareto."
            )
        elif still_dropping and deciding_number is not None and deciding_number < 0.05:
            campaign_note = (
                "Gap is small and val was still dropping. A longer training arc could "
                "be argued. This probe still NO-GOs: it did not produce the reopen win."
            )
        else:
            campaign_note = (
                "Trustworthy negative for a cheap/local distill of the MLP function. "
                "The last surviving avenue named by PHASE_B_HYBRID does not deserve a "
                "campaign on this family at this capture size: held-out function-space "
                "fitting did not beat q3 at fewer active bytes, Doctor holding."
            )

    results["verdict"] = {
        "decision": decision,
        "deciding_number": deciding_number,
        "deciding_number_meaning": deciding_note,
        "headline_width": headline_width,
        "materially_fewer_ratio": MATERIAL_BYTE_RATIO,
        "go_reasons": go_reasons,
        "nogo_reasons": nogo_reasons,
        "per_candidate": deciding,
        "campaign_note": campaign_note,
        "reopen_condition": (
            "A distilled/generated operator, trained to match the MLP function, "
            "achieves q3 quality at materially fewer active bytes held-out across "
            "layers, with Doctor holding."
        ),
        "scale_trap_rejects_001W": True,
        "null_baseline_measured": True,
        "storage_and_active_both_reported": True,
        "one_hop_claimed": "one residual injection only",
        "llama_server_52484": "not used as teacher (Q5_K; NS-015)",
    }

    print(f"decision: {decision}")
    print(f"deciding number: {deciding_number}  ({deciding_note})")
    for r in nogo_reasons:
        print(f"  NO-GO  {r}")
    for r in go_reasons:
        print(f"  GO     {r}")
    if campaign_note:
        print(f"campaign: {campaign_note}")
    print()
    print("## WHAT I WATCHED FAIL")
    for line in results["watched_fail"]:
        print(f"  - {line}")
    # Always include the structural failures the standing discipline names.
    extras = [
        "Synthetic activations were not used; prior sub-bit negatives on this family were Gaussian-proxy artifacts.",
        f"Raw cosine without null is meaningless; measured null on hold MLP outputs is per-layer in null_baseline (not assumed 0.898).",
        "Low storage BPW without a health verdict is not a result; every candidate has HEALTHY/UNHEALTHY vs q3 Doctor.",
        "q3 storage BPW 3.25 is not q3 decoded-f16 active BPW 16.0; both are in accounting.",
        "Student-model compounding (closed at layers 4-8) is not this experiment; one residual hop is the only downstream claim.",
    ]
    for line in extras:
        print(f"  - {line}")
        results["watched_fail"].append(line)

    results["wall_s"] = time.time() - t_all
    path = _write(results)
    print()
    print(f"wrote: {path}")
    print(f"wall_s: {results['wall_s']:.1f}")
    return 0


def _write(results: dict) -> Path:
    out_dir = ROOT / "receipts" / "headless"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "NOETIC_MLP_DISTILL_PROBE.json"
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(j(results), indent=1)
    tmp.write_text(payload)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
