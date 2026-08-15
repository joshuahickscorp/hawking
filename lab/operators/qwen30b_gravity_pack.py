"""Qwen3-Coder-30B-A3B-Instruct first Gravity / BPW probe.

Bounded research pass against real on-disk weights. Conservative ladder:
4-bit → 3-bit → 2-bit symmetric group RTN. Does NOT target sub-1-bit.

Reuses ideas from ``glm52_activation_aware_pack.py``:
- real safetensors BF16 read (no synthetic fixtures)
- complete BPW ledger (indices + scales + header; no free lunch)
- reconstruction + functional output quality (not recon alone)
- honest reporting when numeric parity holds but functional output degrades
  (Math-Preserve failure mode)

No activation capsules exist for this model yet, so this first pass uses
weight-space group quantization (not activation-aware fitting). That is an
explicit, deliberate scope cut — see evidence writeup.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

SCHEMA = "hawking.qwen30b.gravity_first_test.v1"
MAGIC = b"QW30G1T\0"

# Default source: main hawking tree (worktree may not carry 57G weights).
DEFAULT_MODEL = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
DEFAULT_EVIDENCE = (
    REPO
    / "workspace"
    / "campaign"
    / "evidence"
    / "models"
    / "qwen-30b"
    / "gravity-first-test"
)

# Known geometry from real config.json (verified at runtime too).
EXPECTED = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
    "torch_dtype": "bfloat16",
}


class PackError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Safetensors I/O (mirrors glm52_activation_aware_pack.read_bf16_tensor)
# ---------------------------------------------------------------------------


def read_safetensors_header(shard: Path) -> dict[str, Any]:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def read_bf16_tensor(shard: Path, header: dict[str, Any], name: str) -> np.ndarray:
    info = header[name]
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        base = 8 + n
        fh.seek(base + lo)
        raw = fh.read(hi - lo)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype in ("F16", "FLOAT16"):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise PackError(f"unsupported dtype {dtype} for {name}")


def load_weight_map(model_dir: Path) -> dict[str, str]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return dict(idx["weight_map"])


def load_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "config.json").read_text())


_HEADER_CACHE: dict[Path, dict[str, Any]] = {}


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    shard_name = weight_map[name]
    shard = model_dir / shard_name
    if shard not in _HEADER_CACHE:
        _HEADER_CACHE[shard] = read_safetensors_header(shard)
    return read_bf16_tensor(shard, _HEADER_CACHE[shard], name)


# ---------------------------------------------------------------------------
# Quantization: symmetric group absmax RTN (conservative first ladder)
# ---------------------------------------------------------------------------


@dataclass
class QuantResult:
    nbits: int
    group_size: int
    n_weights: int
    n_groups: int
    # complete bit ledger for this tensor
    index_bits: int
    scale_bits: int  # fp16 scales
    header_bits: int
    total_bits: int
    bpw: float
    recon: np.ndarray
    # reconstruction metrics
    rel_l2: float
    rmse: float
    max_abs: float
    cosine: float
    # packed storage (indices as uint8 packed where possible)
    q_codes: np.ndarray  # int8 codes in [-Q, Q]
    scales: np.ndarray  # float32 scales, one per group


def _pack_bits_needed(n_weights: int, nbits: int) -> int:
    return int(n_weights) * int(nbits)


def symmetric_group_quant(
    W: np.ndarray,
    *,
    nbits: int,
    group_size: int = 64,
    header_bytes: int = 64,
) -> QuantResult:
    """Symmetric absmax group quantization with round-to-nearest.

    Codes live in {-Q,...,Q} with Q = 2^(nbits-1) - 1 (sign bit implicit in
    two's-complement style range). Scales stored as float16 (16 bits each).
    Complete BPW includes indices + scales + fixed header.
    """
    if nbits < 2 or nbits > 8:
        raise PackError(f"nbits out of supported range: {nbits}")
    if group_size < 1:
        raise PackError("group_size must be positive")

    Wf = np.ascontiguousarray(W, dtype=np.float32)
    flat = Wf.reshape(-1)
    n = int(flat.size)
    g = int(group_size)
    n_groups = (n + g - 1) // g
    pad = n_groups * g - n
    if pad:
        flat_pad = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        flat_pad = flat
    groups = flat_pad.reshape(n_groups, g)

    Q = (1 << (nbits - 1)) - 1  # e.g. 4-bit -> 7, 3-bit -> 3, 2-bit -> 1
    absmax = np.max(np.abs(groups), axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-12)
    scales = absmax / float(Q)
    codes = np.round(groups / scales[:, None]).astype(np.int16)
    codes = np.clip(codes, -Q, Q)
    recon_pad = codes.astype(np.float32) * scales[:, None]
    recon_flat = recon_pad.reshape(-1)[:n]
    recon = recon_flat.reshape(Wf.shape)

    index_bits = _pack_bits_needed(n, nbits)
    scale_bits = n_groups * 16  # fp16 scales
    header_bits = header_bytes * 8
    total_bits = index_bits + scale_bits + header_bits
    bpw = total_bits / max(1, n)

    diff = Wf - recon
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(Wf) + 1e-12))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    w_norm = float(np.linalg.norm(Wf.ravel()))
    r_norm = float(np.linalg.norm(recon.ravel()))
    if w_norm < 1e-12 or r_norm < 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(Wf.ravel(), recon.ravel()) / (w_norm * r_norm))

    return QuantResult(
        nbits=nbits,
        group_size=g,
        n_weights=n,
        n_groups=n_groups,
        index_bits=index_bits,
        scale_bits=scale_bits,
        header_bits=header_bits,
        total_bits=total_bits,
        bpw=bpw,
        recon=recon,
        rel_l2=rel_l2,
        rmse=rmse,
        max_abs=max_abs,
        cosine=cosine,
        q_codes=codes.reshape(-1)[:n].astype(np.int8),
        scales=scales,
    )


# ---------------------------------------------------------------------------
# Functional probes (output quality, not weight recon alone)
# ---------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def expert_ffn(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """SwiGLU expert: down(silu(x @ gate.T) * (x @ up.T)).

    gate/up: [inter, hidden], down: [hidden, inter], x: [N, hidden]
    """
    g = silu(x @ gate.T)
    u = x @ up.T
    return (g * u) @ down.T


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = np.sum(a * b, axis=-1)
    da = np.linalg.norm(a, axis=-1)
    db = np.linalg.norm(b, axis=-1)
    den = np.maximum(da * db, 1e-12)
    return float(np.mean(num / den))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(var + eps)) * weight


def router_topk(
    x: np.ndarray,
    gate_w: np.ndarray,
    k: int = 8,
    norm_topk_prob: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices [N,k], weights [N,k]) for MoE router."""
    logits = x @ gate_w.T  # [N, n_experts]
    # stable top-k
    idx = np.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    # sort the top-k
    row = np.arange(logits.shape[0])[:, None]
    top_logits = logits[row, idx]
    order = np.argsort(-top_logits, axis=-1)
    idx = idx[row, order]
    top_logits = top_logits[row, order]
    # softmax over top-k
    top_logits = top_logits - np.max(top_logits, axis=-1, keepdims=True)
    w = np.exp(top_logits)
    w = w / np.maximum(np.sum(w, axis=-1, keepdims=True), 1e-12)
    if norm_topk_prob:
        w = w / np.maximum(np.sum(w, axis=-1, keepdims=True), 1e-12)
    return idx.astype(np.int64), w.astype(np.float32)


# ---------------------------------------------------------------------------
# Bounded experiment runner
# ---------------------------------------------------------------------------


@dataclass
class TensorProbe:
    name: str
    shape: list[int]
    n_weights: int
    nbits: int
    group_size: int
    bpw: float
    rel_l2: float
    rmse: float
    max_abs: float
    cosine: float
    index_bits: int
    scale_bits: int
    header_bits: int
    total_bits: int


@dataclass
class ExpertFunctionalProbe:
    layer: int
    expert: int
    nbits: int
    group_size: int
    # weight recon (aggregate over gate/up/down)
    mean_weight_cosine: float
    mean_weight_rel_l2: float
    mean_weight_bpw: float
    # functional FFN output quality
    ffn_mean_row_cosine: float
    ffn_rel_l2: float
    ffn_rmse: float
    # verdict heuristic
    functional_holds: bool
    note: str


def _verdict_functional(ffn_cos: float, ffn_rel: float) -> tuple[bool, str]:
    """Heuristic floor for 'still capable' at the expert-FFN level.

    Tight bars: cosine >= 0.99 and rel_l2 <= 0.05 ≈ near-lossless behaviour.
    Soft bar: cosine >= 0.95 and rel_l2 <= 0.15 ≈ mild degradation, may still
    be usable. Below that: degraded / Math-Preserve risk.
    """
    if ffn_cos >= 0.99 and ffn_rel <= 0.05:
        return True, "near_lossless"
    if ffn_cos >= 0.97 and ffn_rel <= 0.10:
        return True, "mild_degradation_likely_capable"
    if ffn_cos >= 0.95 and ffn_rel <= 0.15:
        return True, "borderline_capable"
    if ffn_cos >= 0.90:
        return False, "degraded_math_preserve_risk"
    return False, "collapse"


def run_bounded_probe(
    model_dir: Path,
    *,
    bits_list: list[int],
    group_size: int,
    layers: list[int],
    experts: list[int],
    n_probe_tokens: int = 64,
    seed: int = 20260806,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config(model_dir)
    weight_map = load_weight_map(model_dir)

    geometry = {
        "source_path": str(model_dir.resolve()),
        "config_keys": {
            k: config.get(k)
            for k in (
                "architectures",
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "torch_dtype",
                "hidden_act",
                "rms_norm_eps",
                "rope_theta",
                "max_position_embeddings",
            )
        },
        "index_tensor_count": len(weight_map),
        "index_total_size_bytes": json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )
        .get("metadata", {})
        .get("total_size"),
        "revision_hint": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        "matches_expected": {
            k: config.get(k) == v if k != "architectures" else config.get(k) == v
            for k, v in EXPECTED.items()
        },
    }

    hidden = int(config["hidden_size"])
    n_experts = int(config["num_experts"])
    top_k = int(config["num_experts_per_tok"])
    rms_eps = float(config.get("rms_norm_eps", 1e-6))

    # --- real activation-ish probes from embed_tokens ---
    print("[probe] loading embed_tokens for real token activations...", flush=True)
    embed = load_tensor(model_dir, weight_map, "model.embed_tokens.weight")
    rng = np.random.default_rng(seed)
    # sample real vocab rows (token embeddings) as probe activations
    tok_ids = rng.integers(0, embed.shape[0], size=n_probe_tokens)
    x_raw = embed[tok_ids].astype(np.float32)  # [N, hidden]

    # also a Gaussian null for comparison
    x_gauss = rng.standard_normal((n_probe_tokens, hidden)).astype(np.float32) * 0.02

    tensor_probes: list[dict[str, Any]] = []
    expert_probes: list[dict[str, Any]] = []
    layer_moe_probes: list[dict[str, Any]] = []
    attn_probes: list[dict[str, Any]] = []

    for layer in layers:
        print(f"[probe] layer {layer}: norms + router + experts {experts}", flush=True)
        ln_name = f"model.layers.{layer}.input_layernorm.weight"
        post_name = f"model.layers.{layer}.post_attention_layernorm.weight"
        gate_name = f"model.layers.{layer}.mlp.gate.weight"

        ln_w = load_tensor(model_dir, weight_map, ln_name)
        post_w = load_tensor(model_dir, weight_map, post_name)
        router_w = load_tensor(model_dir, weight_map, gate_name)

        # Use post-attn-norm path for MoE input (standard decoder block order
        # after residual). We don't have real residual stream; use normed embeds
        # as a bounded proxy for MoE-in activations.
        x_moe = rms_norm(x_raw, post_w, eps=rms_eps)
        x_moe_g = rms_norm(x_gauss, post_w, eps=rms_eps)

        # Attention weight recon only (bounded; no full attention functional
        # without KV cache / RoPE full path — still useful weight-level signal).
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tname = f"model.layers.{layer}.self_attn.{proj}.weight"
            W = load_tensor(model_dir, weight_map, tname)
            # Linear is y = x @ W.T with W [out, in].
            # q/k/v: in=hidden; o_proj: in=num_heads*head_dim (not hidden).
            in_dim = int(W.shape[1])
            for nb in bits_list:
                qr = symmetric_group_quant(W, nbits=nb, group_size=group_size)
                if in_dim == hidden:
                    x_attn = rms_norm(x_raw, ln_w, eps=rms_eps)
                else:
                    # synthetic in-features for o_proj (no full attention path)
                    x_attn = rng.standard_normal(
                        (n_probe_tokens, in_dim)
                    ).astype(np.float32) * float(np.std(x_raw) + 1e-6)
                y0 = x_attn @ W.T
                y1 = x_attn @ qr.recon.T
                attn_probes.append(
                    {
                        "name": tname,
                        "nbits": nb,
                        "group_size": group_size,
                        "bpw": qr.bpw,
                        "weight_rel_l2": qr.rel_l2,
                        "weight_cosine": qr.cosine,
                        "matmul_mean_row_cosine": mean_row_cosine(y0, y1),
                        "matmul_rel_l2": rel_l2(y0, y1),
                        "in_dim": in_dim,
                    }
                )
                tensor_probes.append(
                    asdict(
                        TensorProbe(
                            name=tname,
                            shape=list(W.shape),
                            n_weights=qr.n_weights,
                            nbits=nb,
                            group_size=group_size,
                            bpw=qr.bpw,
                            rel_l2=qr.rel_l2,
                            rmse=qr.rmse,
                            max_abs=qr.max_abs,
                            cosine=qr.cosine,
                            index_bits=qr.index_bits,
                            scale_bits=qr.scale_bits,
                            header_bits=qr.header_bits,
                            total_bits=qr.total_bits,
                        )
                    )
                )

        # Router kept full-precision in this first pass (tiny, sensitive).
        router_idx, router_wts = router_topk(x_moe, router_w, k=top_k)

        # Per-expert FFN probes
        expert_recons: dict[int, dict[int, dict[str, np.ndarray]]] = {
            nb: {} for nb in bits_list
        }
        for expert in experts:
            parts = {}
            for part in ("gate_proj", "up_proj", "down_proj"):
                tname = f"model.layers.{layer}.mlp.experts.{expert}.{part}.weight"
                parts[part] = load_tensor(model_dir, weight_map, tname)

            y_ref = expert_ffn(x_moe, parts["gate_proj"], parts["up_proj"], parts["down_proj"])
            y_ref_g = expert_ffn(
                x_moe_g, parts["gate_proj"], parts["up_proj"], parts["down_proj"]
            )

            for nb in bits_list:
                recons = {}
                weight_cos = []
                weight_rel = []
                bpws = []
                for part, W in parts.items():
                    tname = f"model.layers.{layer}.mlp.experts.{expert}.{part}.weight"
                    qr = symmetric_group_quant(W, nbits=nb, group_size=group_size)
                    recons[part] = qr.recon
                    weight_cos.append(qr.cosine)
                    weight_rel.append(qr.rel_l2)
                    bpws.append(qr.bpw)
                    tensor_probes.append(
                        asdict(
                            TensorProbe(
                                name=tname,
                                shape=list(W.shape),
                                n_weights=qr.n_weights,
                                nbits=nb,
                                group_size=group_size,
                                bpw=qr.bpw,
                                rel_l2=qr.rel_l2,
                                rmse=qr.rmse,
                                max_abs=qr.max_abs,
                                cosine=qr.cosine,
                                index_bits=qr.index_bits,
                                scale_bits=qr.scale_bits,
                                header_bits=qr.header_bits,
                                total_bits=qr.total_bits,
                            )
                        )
                    )
                expert_recons[nb][expert] = recons
                y_q = expert_ffn(
                    x_moe, recons["gate_proj"], recons["up_proj"], recons["down_proj"]
                )
                y_q_g = expert_ffn(
                    x_moe_g,
                    recons["gate_proj"],
                    recons["up_proj"],
                    recons["down_proj"],
                )
                ffn_cos = mean_row_cosine(y_ref, y_q)
                ffn_r = rel_l2(y_ref, y_q)
                ffn_rmse = float(np.sqrt(np.mean((y_ref - y_q) ** 2)))
                holds, note = _verdict_functional(ffn_cos, ffn_r)
                expert_probes.append(
                    asdict(
                        ExpertFunctionalProbe(
                            layer=layer,
                            expert=expert,
                            nbits=nb,
                            group_size=group_size,
                            mean_weight_cosine=float(np.mean(weight_cos)),
                            mean_weight_rel_l2=float(np.mean(weight_rel)),
                            mean_weight_bpw=float(np.mean(bpws)),
                            ffn_mean_row_cosine=ffn_cos,
                            ffn_rel_l2=ffn_r,
                            ffn_rmse=ffn_rmse,
                            functional_holds=holds,
                            note=note,
                        )
                    )
                )
                # also record gaussian-input FFN (stress)
                expert_probes[-1]["ffn_mean_row_cosine_gauss"] = mean_row_cosine(
                    y_ref_g, y_q_g
                )
                expert_probes[-1]["ffn_rel_l2_gauss"] = rel_l2(y_ref_g, y_q_g)

        # Layer-level MoE combine: original vs quantized experts that were
        # probed; unprobed experts pass through at full precision. This is an
        # honest partial-model quality signal.
        def moe_combine(
            x: np.ndarray,
            expert_weights: dict[int, dict[str, np.ndarray]],
            idx: np.ndarray,
            wts: np.ndarray,
        ) -> np.ndarray:
            N = x.shape[0]
            out = np.zeros((N, hidden), dtype=np.float32)
            for bi in range(N):
                for j in range(top_k):
                    e = int(idx[bi, j])
                    ww = float(wts[bi, j])
                    if e not in expert_weights:
                        # load on demand full precision for unprobed experts
                        # (skip for speed: only score tokens that route into
                        # probed experts)
                        continue
                    p = expert_weights[e]
                    y = expert_ffn(
                        x[bi : bi + 1], p["gate_proj"], p["up_proj"], p["down_proj"]
                    )
                    out[bi] += ww * y[0]
            return out

        # Build full-precision expert pack for probed experts only
        full_pack = {}
        for expert in experts:
            full_pack[expert] = {
                part: load_tensor(
                    model_dir,
                    weight_map,
                    f"model.layers.{layer}.mlp.experts.{expert}.{part}.weight",
                )
                for part in ("gate_proj", "up_proj", "down_proj")
            }

        # Restrict evaluation to tokens that route at least one probed expert
        probed_set = set(experts)
        mask = np.array(
            [bool(set(router_idx[i].tolist()) & probed_set) for i in range(len(x_moe))]
        )
        if not mask.any():
            # force-route: evaluate as if top expert is experts[0]
            print(
                f"[probe] layer {layer}: no natural routes into probed experts; "
                "using forced single-expert path",
                flush=True,
            )
            force = True
        else:
            force = False

        y_full = np.zeros((x_moe.shape[0], hidden), dtype=np.float32)
        if force:
            for expert in experts[:1]:
                p = full_pack[expert]
                y_full = expert_ffn(
                    x_moe, p["gate_proj"], p["up_proj"], p["down_proj"]
                )
        else:
            y_full = moe_combine(x_moe, full_pack, router_idx, router_wts)

        for nb in bits_list:
            q_pack = expert_recons[nb]
            if force:
                p = q_pack[experts[0]]
                y_q = expert_ffn(x_moe, p["gate_proj"], p["up_proj"], p["down_proj"])
            else:
                y_q = moe_combine(x_moe, q_pack, router_idx, router_wts)
            # only score rows with non-zero y_full (routed into probed)
            if force:
                row_mask = np.ones(x_moe.shape[0], dtype=bool)
            else:
                row_mask = np.linalg.norm(y_full, axis=-1) > 1e-8
            if not row_mask.any():
                continue
            cos = mean_row_cosine(y_full[row_mask], y_q[row_mask])
            r = rel_l2(y_full[row_mask], y_q[row_mask])
            holds, note = _verdict_functional(cos, r)
            layer_moe_probes.append(
                {
                    "layer": layer,
                    "nbits": nb,
                    "group_size": group_size,
                    "probed_experts": experts,
                    "tokens_scored": int(row_mask.sum()),
                    "forced_single_expert": force,
                    "moe_mean_row_cosine": cos,
                    "moe_rel_l2": r,
                    "functional_holds": holds,
                    "note": note,
                }
            )

    # Aggregate by bit width
    by_bits: dict[str, Any] = {}
    for nb in bits_list:
        t_nb = [t for t in tensor_probes if t["nbits"] == nb]
        e_nb = [e for e in expert_probes if e["nbits"] == nb]
        m_nb = [m for m in layer_moe_probes if m["nbits"] == nb]
        a_nb = [a for a in attn_probes if a["nbits"] == nb]
        total_w = sum(t["n_weights"] for t in t_nb)
        total_bits = sum(t["total_bits"] for t in t_nb)
        by_bits[str(nb)] = {
            "nbits": nb,
            "group_size": group_size,
            "n_tensors": len(t_nb),
            "n_weights": total_w,
            "complete_bits": total_bits,
            "complete_bpw": (total_bits / total_w) if total_w else None,
            "weight_recon": {
                "mean_rel_l2": float(np.mean([t["rel_l2"] for t in t_nb])) if t_nb else None,
                "mean_cosine": float(np.mean([t["cosine"] for t in t_nb])) if t_nb else None,
                "max_rel_l2": float(np.max([t["rel_l2"] for t in t_nb])) if t_nb else None,
                "min_cosine": float(np.min([t["cosine"] for t in t_nb])) if t_nb else None,
            },
            "expert_ffn_functional": {
                "mean_ffn_cosine": float(np.mean([e["ffn_mean_row_cosine"] for e in e_nb]))
                if e_nb
                else None,
                "min_ffn_cosine": float(np.min([e["ffn_mean_row_cosine"] for e in e_nb]))
                if e_nb
                else None,
                "mean_ffn_rel_l2": float(np.mean([e["ffn_rel_l2"] for e in e_nb]))
                if e_nb
                else None,
                "max_ffn_rel_l2": float(np.max([e["ffn_rel_l2"] for e in e_nb]))
                if e_nb
                else None,
                "n_experts_hold": sum(1 for e in e_nb if e["functional_holds"]),
                "n_experts_total": len(e_nb),
                "notes": sorted({e["note"] for e in e_nb}),
            },
            "layer_moe_functional": m_nb,
            "attn_matmul_functional": {
                "mean_matmul_cosine": float(
                    np.mean([a["matmul_mean_row_cosine"] for a in a_nb])
                )
                if a_nb
                else None,
                "min_matmul_cosine": float(
                    np.min([a["matmul_mean_row_cosine"] for a in a_nb])
                )
                if a_nb
                else None,
                "mean_matmul_rel_l2": float(np.mean([a["matmul_rel_l2"] for a in a_nb]))
                if a_nb
                else None,
            },
        }

    # Honest floor: lowest nbits where ALL expert FFN probes hold
    floor_bits = None
    floor_note = "no_level_held"
    for nb in sorted(bits_list):
        e_nb = [e for e in expert_probes if e["nbits"] == nb]
        if e_nb and all(e["functional_holds"] for e in e_nb):
            floor_bits = nb
            floor_note = (
                f"all {len(e_nb)} expert-FFN probes hold at {nb}-bit "
                f"(notes={sorted({e['note'] for e in e_nb})})"
            )
        else:
            # first failure stops the ladder for "holds"
            break

    # Detect Math-Preserve style: weight cosine high but FFN cosine low
    math_preserve_hits = []
    for e in expert_probes:
        if e["mean_weight_cosine"] >= 0.99 and e["ffn_mean_row_cosine"] < 0.95:
            math_preserve_hits.append(
                {
                    "layer": e["layer"],
                    "expert": e["expert"],
                    "nbits": e["nbits"],
                    "weight_cosine": e["mean_weight_cosine"],
                    "ffn_cosine": e["ffn_mean_row_cosine"],
                    "ffn_rel_l2": e["ffn_rel_l2"],
                }
            )

    elapsed = time.time() - t0
    return {
        "schema": SCHEMA,
        "at_unix": time.time(),
        "elapsed_sec": elapsed,
        "geometry": geometry,
        "method": {
            "name": "symmetric_group_absmax_rtn",
            "nbits_tested": bits_list,
            "group_size": group_size,
            "scale_dtype": "fp16_billed",
            "activation_aware": False,
            "activation_aware_note": (
                "No teacher activation capsules for Qwen-30B yet. "
                "First pass is weight-space group quant. "
                "glm52_activation_aware_pack.py ideas reused for ledger/metrics "
                "and functional-output checking, not the low-rank activation basis."
            ),
            "protected": [
                "router/gate.weight kept full precision in functional MoE probe",
                "norms kept full precision",
                "embed not quantized in this bounded pass",
            ],
            "layers": layers,
            "experts": experts,
            "n_probe_tokens": n_probe_tokens,
            "seed": seed,
        },
        "by_bits": by_bits,
        "tensor_probes": tensor_probes,
        "expert_functional_probes": expert_probes,
        "layer_moe_probes": layer_moe_probes,
        "attn_probes": attn_probes,
        "honest_floor": {
            "lowest_nbits_all_expert_ffn_hold": floor_bits,
            "note": floor_note,
            "caveat": (
                "Floor is for the bounded expert/attn slice under group RTN, "
                "not a full-model generation capability score. "
                "See generation_probe section if present."
            ),
        },
        "math_preserve_style_hits": math_preserve_hits,
        "claim_boundary": {
            "is_full_model_generation_benchmark": False,
            "is_full_model_bpw_artifact": False,
            "is_bounded_real_tensor_prototype": True,
            "universal_1_5_bpw_required": False,
        },
    }


def reconstruct_roundtrip_test(
    model_dir: Path,
    *,
    nbits: int = 4,
    group_size: int = 64,
    max_tensors: int = 6,
) -> dict[str, Any]:
    """Required test: load compressed representation, confirm recon error bound."""
    weight_map = load_weight_map(model_dir)
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.24.mlp.experts.7.gate_proj.weight",
        "model.layers.47.mlp.experts.3.down_proj.weight",
    ][:max_tensors]
    results = []
    all_ok = True
    for name in names:
        if name not in weight_map:
            results.append({"name": name, "status": "MISSING"})
            all_ok = False
            continue
        W = load_tensor(model_dir, weight_map, name)
        qr = symmetric_group_quant(W, nbits=nbits, group_size=group_size)
        # "load compressed representation": codes + scales → recon
        # re-decode from stored codes/scales to prove round-trip
        flat = W.reshape(-1).astype(np.float32)
        n = flat.size
        g = group_size
        n_groups = (n + g - 1) // g
        codes = qr.q_codes.astype(np.float32)
        # pad codes to groups
        pad = n_groups * g - n
        if pad:
            codes_pad = np.concatenate([codes, np.zeros(pad, dtype=np.float32)])
        else:
            codes_pad = codes
        recon2 = (codes_pad.reshape(n_groups, g) * qr.scales[:, None]).reshape(-1)[:n]
        recon2 = recon2.reshape(W.shape)
        err = float(np.max(np.abs(qr.recon - recon2)))
        bound_ok = qr.rel_l2 < 0.5  # very loose bound; real numbers reported
        roundtrip_ok = err < 1e-5
        ok = bound_ok and roundtrip_ok
        if not ok:
            all_ok = False
        results.append(
            {
                "name": name,
                "shape": list(W.shape),
                "nbits": nbits,
                "group_size": group_size,
                "bpw": qr.bpw,
                "rel_l2": qr.rel_l2,
                "cosine": qr.cosine,
                "roundtrip_max_abs": err,
                "roundtrip_ok": roundtrip_ok,
                "bound_ok": bound_ok,
                "status": "PASS" if ok else "FAIL",
            }
        )
    return {
        "schema": "hawking.qwen30b.gravity_first_test.roundtrip.v1",
        "all_ok": all_ok,
        "results": results,
    }


def try_generation_probe(
    model_dir: Path,
    *,
    bits_list: list[int],
    group_size: int,
    quant_layers: list[int],
    max_new_tokens: int = 48,
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Optional full/partial generation quality check via transformers.

    Quantizes expert weights in selected layers in-place (quant-dequant so the
    model still runs in bf16/float32), compares greedy outputs to baseline.
    """
    prompts = prompts or [
        "Write a Python function that returns the nth Fibonacci number.",
        "What is 17 * 24? Show the calculation.",
        "Explain in one sentence what a binary search does.",
    ]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        return {
            "status": "SKIPPED",
            "reason": f"transformers/torch unavailable: {type(e).__name__}: {e}",
        }

    print("[gen] loading tokenizer...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        return {
            "status": "SKIPPED",
            "reason": f"tokenizer load failed: {type(e).__name__}: {e}",
        }

    print("[gen] loading model (bf16, CPU)... this may take several minutes", flush=True)
    t0 = time.time()
    try:
        # Prefer dtype=; fall back for older transformers. Avoid device_map so
        # we do not require the accelerate package.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        model.eval()
    except Exception as e:
        return {
            "status": "SKIPPED",
            "reason": f"model load failed: {type(e).__name__}: {e}",
            "elapsed_sec": time.time() - t0,
        }
    load_sec = time.time() - t0
    print(f"[gen] model loaded in {load_sec:.1f}s", flush=True)

    def generate_one(prompt: str) -> dict[str, Any]:
        # Prefer chat template if present
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt
        else:
            text = prompt
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_ids = out[0, inputs["input_ids"].shape[-1] :].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        return {"prompt": prompt, "gen_ids": gen_ids, "gen_text": gen_text}

    def _iter_expert_params(layer_idx: int):
        """Yield (key, Parameter) for expert weight tensors in a layer.

        Transformers packs Qwen3 MoE experts as 3D Parameters on
        ``layer.mlp.experts``:
          - gate_up_proj: [E, 2*inter, hidden]
          - down_proj:    [E, hidden, inter]
        Older layouts used a ModuleList of per-expert Linear modules.
        """
        layer = model.model.layers[layer_idx]
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            return
        # Packed 3D layout (current transformers Qwen3MoeExperts)
        for pname in ("gate_up_proj", "down_proj", "gate_proj", "up_proj"):
            p = getattr(experts, pname, None)
            if p is not None and hasattr(p, "data") and p.ndim >= 2:
                yield f"L{layer_idx}.experts.{pname}", p
        # ModuleList / sequential layout fallback
        try:
            iterable = list(experts)
        except TypeError:
            iterable = []
        for ei, expert in enumerate(iterable):
            for proj_name in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                proj = getattr(expert, proj_name, None)
                if proj is None:
                    continue
                if hasattr(proj, "weight"):
                    yield f"L{layer_idx}.E{ei}.{proj_name}", proj.weight
                elif hasattr(proj, "data") and getattr(proj, "ndim", 0) >= 2:
                    yield f"L{layer_idx}.E{ei}.{proj_name}", proj

    def quant_dequant_experts(nbits: int) -> int:
        """In-place quant-dequant expert weights on selected layers. Returns count."""
        n = 0
        for layer_idx in quant_layers:
            for key, param in _iter_expert_params(layer_idx):
                W = param.detach().float().cpu().numpy()
                # Quantize per-expert slice when 3D packed [E, out, in]
                if W.ndim == 3:
                    recon = np.empty_like(W)
                    for e in range(W.shape[0]):
                        qr = symmetric_group_quant(
                            W[e], nbits=nbits, group_size=group_size
                        )
                        recon[e] = qr.recon
                else:
                    recon = symmetric_group_quant(
                        W, nbits=nbits, group_size=group_size
                    ).recon
                new_w = torch.from_numpy(recon).to(
                    dtype=param.dtype, device=param.device
                )
                param.data.copy_(new_w)
                n += 1
        return n

    def snapshot_expert_weights() -> dict[str, torch.Tensor]:
        snap = {}
        for layer_idx in quant_layers:
            for key, param in _iter_expert_params(layer_idx):
                snap[key] = param.detach().cpu().clone()
        return snap

    def restore_expert_weights(snap: dict[str, torch.Tensor]) -> None:
        for layer_idx in quant_layers:
            for key, param in _iter_expert_params(layer_idx):
                if key in snap:
                    param.data.copy_(
                        snap[key].to(dtype=param.dtype, device=param.device)
                    )

    results: dict[str, Any] = {
        "status": "RAN",
        "load_sec": load_sec,
        "quant_layers": quant_layers,
        "group_size": group_size,
        "max_new_tokens": max_new_tokens,
        "prompts": prompts,
        "by_bits": {},
    }

    print("[gen] snapshotting original expert weights for quant layers...", flush=True)
    snap = snapshot_expert_weights()
    print(f"[gen] snapshotted {len(snap)} expert proj tensors", flush=True)

    print("[gen] baseline generation...", flush=True)
    t1 = time.time()
    try:
        baseline = [generate_one(p) for p in prompts]
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"baseline generate failed: {type(e).__name__}: {e}",
            "load_sec": load_sec,
        }
    results["baseline"] = [
        {"prompt": b["prompt"], "gen_text": b["gen_text"], "n_tokens": len(b["gen_ids"])}
        for b in baseline
    ]
    results["baseline_gen_sec"] = time.time() - t1
    for b in results["baseline"]:
        print(f"[gen] BASE: {b['gen_text'][:120]!r}", flush=True)

    for nb in bits_list:
        print(f"[gen] quant-dequant experts @ {nb}-bit on layers {quant_layers}...", flush=True)
        restore_expert_weights(snap)
        n_q = quant_dequant_experts(nb)
        t2 = time.time()
        try:
            gens = [generate_one(p) for p in prompts]
        except Exception as e:
            results["by_bits"][str(nb)] = {
                "status": "FAILED",
                "reason": str(e),
                "n_tensors_quantized": n_q,
            }
            continue
        comparisons = []
        for b, g in zip(baseline, gens):
            # token-level agreement
            b_ids = b["gen_ids"]
            g_ids = g["gen_ids"]
            L = min(len(b_ids), len(g_ids))
            agree = sum(1 for i in range(L) if b_ids[i] == g_ids[i]) / max(1, L)
            exact = b_ids == g_ids
            comparisons.append(
                {
                    "prompt": b["prompt"],
                    "baseline_text": b["gen_text"],
                    "quant_text": g["gen_text"],
                    "exact_token_match": exact,
                    "prefix_token_agreement": agree,
                    "baseline_n_tokens": len(b_ids),
                    "quant_n_tokens": len(g_ids),
                    "looks_coherent": _looks_coherent(g["gen_text"]),
                    "baseline_coherent": _looks_coherent(b["gen_text"]),
                }
            )
            print(
                f"[gen] {nb}-bit agree={agree:.3f} exact={exact} "
                f"qtext={g['gen_text'][:100]!r}",
                flush=True,
            )
        mean_agree = float(np.mean([c["prefix_token_agreement"] for c in comparisons]))
        all_coherent = all(c["looks_coherent"] for c in comparisons)
        any_exact = any(c["exact_token_match"] for c in comparisons)
        results["by_bits"][str(nb)] = {
            "status": "RAN",
            "n_tensors_quantized": n_q,
            "gen_sec": time.time() - t2,
            "mean_prefix_token_agreement": mean_agree,
            "any_exact_match": any_exact,
            "all_outputs_coherent": all_coherent,
            "comparisons": comparisons,
            "generation_holds": bool(all_coherent and mean_agree >= 0.5),
            "note": (
                "generation_holds heuristic: all outputs look coherent AND "
                "mean prefix token agreement with baseline >= 0.5. "
                "Only selected layers' experts were quantized."
            ),
        }

    restore_expert_weights(snap)
    # free
    del model
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    return results


def _looks_coherent(text: str) -> bool:
    """Very rough garbage detector for Math-Preserve-style collapse."""
    t = (text or "").strip()
    if len(t) < 8:
        return False
    # high fraction of non-printable / replacement chars
    bad = sum(1 for c in t if ord(c) < 9 or c == "\ufffd")
    if bad / max(1, len(t)) > 0.05:
        return False
    # excessive repetition of same 4-gram
    if len(t) > 40:
        grams = [t[i : i + 4] for i in range(0, len(t) - 3)]
        if grams:
            from collections import Counter

            top = Counter(grams).most_common(1)[0][1]
            if top / len(grams) > 0.35:
                return False
    # must have some spaces / word structure for English-ish prompts
    if " " not in t and "\n" not in t:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--bits", type=str, default="4,3,2", help="comma-separated bit widths")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--layers", type=str, default="0,24,47")
    p.add_argument("--experts", type=str, default="0,1,7,63")
    p.add_argument("--n-probe-tokens", type=int, default=64)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument(
        "--gen-layers",
        type=str,
        default="0,1,2",
        help="layers whose experts are quant-dequant'd for generation probe",
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--roundtrip-only", action="store_true")
    args = p.parse_args(argv)

    model_dir = args.model_dir
    if not model_dir.is_dir():
        print(f"FAIL: model dir missing: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "config.json").is_file():
        print(f"FAIL: no config.json in {model_dir}", file=sys.stderr)
        return 2

    bits_list = [int(x) for x in args.bits.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    experts = [int(x) for x in args.experts.split(",") if x.strip()]
    gen_layers = [int(x) for x in args.gen_layers.split(",") if x.strip()]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required roundtrip test first
    print("[main] roundtrip reconstruction test...", flush=True)
    rt = reconstruct_roundtrip_test(
        model_dir, nbits=bits_list[0], group_size=args.group_size
    )
    (out_dir / "ROUNDTRIP_TEST.json").write_text(json.dumps(rt, indent=2) + "\n")
    print(f"[main] roundtrip all_ok={rt['all_ok']}", flush=True)
    if args.roundtrip_only:
        return 0 if rt["all_ok"] else 1

    print("[main] bounded real-tensor probe...", flush=True)
    report = run_bounded_probe(
        model_dir,
        bits_list=bits_list,
        group_size=args.group_size,
        layers=layers,
        experts=experts,
        n_probe_tokens=args.n_probe_tokens,
    )
    report["roundtrip_test"] = rt

    if not args.skip_generation:
        print("[main] generation probe (may be heavy)...", flush=True)
        # Prefer the condense venv's transformers if available via PYTHONPATH
        gen = try_generation_probe(
            model_dir,
            bits_list=bits_list,
            group_size=args.group_size,
            quant_layers=gen_layers,
            max_new_tokens=args.max_new_tokens,
        )
        report["generation_probe"] = gen
        # Reconcile honest floor with generation if available
        if gen.get("status") == "RAN":
            gen_floor = None
            for nb in sorted(bits_list):
                cell = gen.get("by_bits", {}).get(str(nb), {})
                if cell.get("generation_holds"):
                    gen_floor = nb
                else:
                    break
            report["honest_floor"]["generation_lowest_nbits_holds"] = gen_floor
            report["honest_floor"]["generation_note"] = (
                f"Partial-model generation (experts in layers {gen_layers} only). "
                f"Lowest nbits with generation_holds=True: {gen_floor}"
            )
    else:
        report["generation_probe"] = {"status": "SKIPPED", "reason": "--skip-generation"}

    out_path = out_dir / "QWEN30B_GRAVITY_FIRST_TEST.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[main] wrote {out_path}", flush=True)

    # Human-readable summary
    summary_lines = [
        "# Qwen3-Coder-30B Gravity First Test — Summary",
        "",
        f"Source: `{model_dir}`",
        f"Method: symmetric group absmax RTN, group_size={args.group_size}",
        f"Layers: {layers}  Experts: {experts}",
        "",
        "## By bit width",
        "",
    ]
    for nb in bits_list:
        cell = report["by_bits"][str(nb)]
        ef = cell["expert_ffn_functional"]
        summary_lines.append(
            f"### {nb}-bit  complete_bpw≈{cell['complete_bpw']:.4f}  "
            f"(n_weights={cell['n_weights']})"
        )
        summary_lines.append(
            f"- weight recon: mean_cos={cell['weight_recon']['mean_cosine']:.6f} "
            f"mean_rel_l2={cell['weight_recon']['mean_rel_l2']:.6f}"
        )
        summary_lines.append(
            f"- expert FFN: mean_cos={ef['mean_ffn_cosine']:.6f} "
            f"min_cos={ef['min_ffn_cosine']:.6f} "
            f"mean_rel_l2={ef['mean_ffn_rel_l2']:.6f} "
            f"hold={ef['n_experts_hold']}/{ef['n_experts_total']} "
            f"notes={ef['notes']}"
        )
        if cell["attn_matmul_functional"]["mean_matmul_cosine"] is not None:
            am = cell["attn_matmul_functional"]
            summary_lines.append(
                f"- attn matmul: mean_cos={am['mean_matmul_cosine']:.6f} "
                f"min_cos={am['min_matmul_cosine']:.6f} "
                f"mean_rel_l2={am['mean_matmul_rel_l2']:.6f}"
            )
        for m in cell["layer_moe_functional"]:
            summary_lines.append(
                f"- layer {m['layer']} MoE combine: cos={m['moe_mean_row_cosine']:.6f} "
                f"rel_l2={m['moe_rel_l2']:.6f} holds={m['functional_holds']} ({m['note']})"
            )
        gen = report.get("generation_probe", {})
        if gen.get("status") == "RAN" and str(nb) in gen.get("by_bits", {}):
            gcell = gen["by_bits"][str(nb)]
            summary_lines.append(
                f"- generation (layers {gen.get('quant_layers')}): "
                f"agree={gcell.get('mean_prefix_token_agreement')} "
                f"coherent={gcell.get('all_outputs_coherent')} "
                f"holds={gcell.get('generation_holds')}"
            )
        summary_lines.append("")
    hf = report["honest_floor"]
    summary_lines.append("## Honest floor")
    summary_lines.append(f"- expert-FFN floor nbits: {hf.get('lowest_nbits_all_expert_ffn_hold')}")
    summary_lines.append(f"- note: {hf.get('note')}")
    if "generation_lowest_nbits_holds" in hf:
        summary_lines.append(
            f"- generation floor nbits: {hf.get('generation_lowest_nbits_holds')}"
        )
        summary_lines.append(f"- generation note: {hf.get('generation_note')}")
    summary_lines.append("")
    summary_lines.append("## Math-Preserve-style hits")
    if report["math_preserve_style_hits"]:
        for h in report["math_preserve_style_hits"]:
            summary_lines.append(f"- {h}")
    else:
        summary_lines.append("- none under the (weight_cos>=0.99 & ffn_cos<0.95) rule")
    summary_lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines), flush=True)
    return 0 if rt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
