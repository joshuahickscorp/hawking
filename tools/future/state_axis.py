"""G016: per-token persistent decode-state, from each specimen's own config.

Header/config only. No tensor payload, no GPU. A missing field is a named
refusal (the exact key and the path read), never an "unknown" measurement.

    python3.12 tools/future/state_axis.py --selftest
    python3.12 tools/future/state_axis.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import RECEIPTS, REPO, write_receipt  # noqa: E402
from lake_scheme_census import DTYPE_BITS, FLOAT_DTYPES, read_header  # noqa: E402

RECORDED_BY = "tools/future/state_axis.py"
RECEIPT_NAME = "G016_STATE_AXIS.json"
SCHEMA = "hawking.future.state_axis.v1"
CATALOG_DEFAULT = REPO / "receipts" / "future" / "modellake-index" / "catalog.json"
LAKE_DEFAULT = "/Volumes/corpdrive/hawking-modellake/specimens"

STATE_CLASSES = (
    "KV_FULL",
    "KV_GQA",
    "KV_MQA",
    "KV_MLA",
    "RECURRENT",
    "HYBRID",
    "NONE",
)

# Config strings that name a growing attention cache vs a fixed mixer state.
# Discovered from the lake's own layer_types / linear_attn_config, not a slug map.
_ATTN_MIXERS = frozenset({
    "attention", "full_attention", "sliding_attention", "sliding_window",
    "local_attention", "full", "attn", "mha", "gqa", "mqa", "mla",
    "deepseek_sparse_attention", "sparse_attention",
})
_RECURRENT_MIXERS = frozenset({
    "mamba", "mamba2", "mamba3", "conv", "linear_attention", "kda",
    "delta_net", "deltanet", "gated_deltanet", "rwkv", "hyena", "ssm",
    "linear", "short_conv",
})

# Nested decoder bodies. First match that actually carries layer/head fields wins.
_NEST_KEYS = (
    "text_config", "llm_cfg", "language_config", "thinker_config", "decoder",
)

_DTYPE_BYTES = {
    "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
    "float16": 2, "fp16": 2, "half": 2, "torch.float16": 2,
    "float32": 4, "fp32": 4, "float": 4, "torch.float32": 4,
    "float64": 8, "fp64": 8, "double": 8, "torch.float64": 8,
    "float8": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
    "f8_e4m3": 1, "f8_e5m2": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
}

_SAFETENSORS_DTYPE_BYTES = {k: DTYPE_BITS[k] // 8 for k in DTYPE_BITS}

# Encoder / non-AR architecture tokens. A substring on the architecture list,
# not a slug table: new MaskedLM bodies classify themselves.
_NON_AR_ARCH_TOKENS = (
    "maskedlm",
    "fordepthestimation",
    "videomodel",
    "forimageclassification",
    "forobjectdetection",
    "forvideoclassification",
)
_NON_AR_MODEL_TYPES = frozenset({
    "modernbert", "depth_anything", "vjepa2", "sam2_video", "t2v",
})


class StateRefused(Exception):
    """The config does not carry a field the formula needs. That is a finding."""

    def __init__(self, missing_key: str, path: str, mechanism: str):
        if not missing_key or missing_key.lower() == "unknown":
            raise ValueError("a refusal must name the missing key; 'unknown' is not a refusal")
        self.missing_key = missing_key
        self.path = path
        self.mechanism = mechanism
        super().__init__(mechanism)


def _refuse_volumes_write(path: str) -> None:
    abs_path = os.path.abspath(path)
    if abs_path == "/Volumes" or abs_path.startswith("/Volumes/"):
        raise RuntimeError(f"refusing to write under /Volumes: {abs_path}")


# ---------------------------------------------------------------------------
# field readers — authority is the config, never a family guess
# ---------------------------------------------------------------------------

def _looks_like_decoder(cfg: dict) -> bool:
    return any(
        cfg.get(k) is not None
        for k in (
            "num_hidden_layers", "num_decoder_layers", "decoder_layers",
            "n_layer", "num_attention_heads", "num_heads", "decoder_attention_heads",
            "layer_types", "linear_attn_config", "kv_lora_rank", "ssm_cfg",
            "mamba_d_state", "num_key_value_heads",
        )
    )


def resolve_body(cfg: dict) -> tuple[dict, str | None]:
    """Pick the decoder/text body. Vision/audio configs are not decode state."""
    for key in _NEST_KEYS:
        child = cfg.get(key)
        if isinstance(child, dict) and _looks_like_decoder(child):
            return child, key
    return cfg, None


def _cfg_path(path: str, nest: str | None) -> str:
    return f"{path}#{nest}" if nest else path


def require(cfg: dict, key: str, path: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise StateRefused(
            key, path,
            f"config at {path} is missing {key!r}",
        )
    return cfg[key]


def pick_first(cfg: dict, keys: tuple[str, ...], path: str, *, required: bool = True) -> tuple[Any, str | None]:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k], k
    if required:
        tried = ", ".join(repr(k) for k in keys)
        raise StateRefused(
            keys[0], path,
            f"config at {path} is missing {keys[0]!r} (also tried {tried})",
        )
    return None, None


def _as_int(value: Any, key: str, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateRefused(key, path, f"config at {path} field {key!r} is {value!r}, not an int")
    return int(value)


def dtype_bytes_of(name: Any, key: str, path: str) -> int:
    if name is None:
        raise StateRefused(key, path, f"config at {path} is missing {key!r}")
    s = str(name).strip().lower().replace("torch.", "")
    if s in _DTYPE_BYTES:
        return _DTYPE_BYTES[s]
    raise StateRefused(
        key, path,
        f"config at {path} field {key!r}={name!r} is not a known dtype",
    )


def _header_dtype_bytes(specimen_dir: str) -> tuple[int, str] | None:
    """Fallback only: first floating safetensors dtype. Headers, not payloads."""
    if not specimen_dir or not os.path.isdir(specimen_dir):
        return None
    import glob
    shards = sorted(glob.glob(os.path.join(specimen_dir, "**", "*.safetensors"), recursive=True))
    for shard in shards:
        try:
            hdr = read_header(shard)
        except (OSError, ValueError):
            continue
        for entry in hdr.values():
            dt = entry.get("dtype")
            if dt in FLOAT_DTYPES and dt in _SAFETENSORS_DTYPE_BYTES:
                return _SAFETENSORS_DTYPE_BYTES[dt], f"safetensors_header:{dt}:{os.path.basename(shard)}"
    return None


def resolve_dtype(body: dict, parent: dict | None, path: str, specimen_dir: str | None) -> tuple[int, str, str]:
    for src in (body, parent or {}):
        for key in ("dtype", "torch_dtype"):
            if src.get(key):
                return dtype_bytes_of(src[key], key, path), str(src[key]), key
    if specimen_dir:
        hit = _header_dtype_bytes(specimen_dir)
        if hit:
            n, origin = hit
            return n, origin, "safetensors_header.dtype"
    raise StateRefused(
        "dtype", path,
        f"config at {path} is missing 'dtype'/'torch_dtype' and no floating safetensors header was available",
    )


def resolve_n_layers(body: dict, path: str) -> tuple[int, str]:
    v, k = pick_first(
        body,
        ("num_decoder_layers", "decoder_layers", "num_hidden_layers", "n_layer", "n_layers"),
        path,
    )
    return _as_int(v, k, path), k  # type: ignore[arg-type]


def resolve_n_heads(body: dict, path: str) -> tuple[int, str]:
    v, k = pick_first(
        body,
        ("decoder_attention_heads", "num_attention_heads", "num_heads", "n_heads", "n_head"),
        path,
    )
    return _as_int(v, k, path), k  # type: ignore[arg-type]


def resolve_head_dim(body: dict, n_heads: int, path: str, fields: dict) -> tuple[int, str]:
    for key in ("head_dim", "d_kv", "hidden_size_per_head"):
        v = body.get(key)
        if v is not None and int(v) > 0:
            fields[key] = int(v)
            return int(v), key
    hidden, hk = pick_first(
        body,
        ("hidden_size", "d_model", "n_embd", "dim"),
        path,
        required=False,
    )
    if hidden is None:
        raise StateRefused(
            "head_dim", path,
            f"config at {path} is missing 'head_dim' and also 'hidden_size'/'d_model'",
        )
    hidden_i = _as_int(hidden, hk, path)  # type: ignore[arg-type]
    fields[hk] = hidden_i
    if n_heads <= 0 or hidden_i % n_heads:
        raise StateRefused(
            "head_dim", path,
            f"config at {path} cannot form head_dim: {hk}={hidden_i} is not divisible by n_heads={n_heads}",
        )
    return hidden_i // n_heads, f"{hk}/{ 'num_attention_heads' }"


def resolve_max_pos(body: dict, parent: dict | None, path: str) -> tuple[int, str]:
    for src in (body, parent or {}):
        v, k = pick_first(
            src,
            (
                "max_position_embeddings", "model_max_length", "max_target_positions",
                "n_positions", "context_length",
            ),
            path,
            required=False,
        )
        if v is not None:
            return _as_int(v, k, path), k  # type: ignore[arg-type]
    raise StateRefused(
        "max_position_embeddings", path,
        f"config at {path} is missing 'max_position_embeddings' (also tried model_max_length, max_target_positions, n_positions, context_length)",
    )


def kv_class(n_heads: int, n_kv: int) -> str:
    if n_kv <= 0:
        raise ValueError("n_kv must be positive")
    if n_kv == 1 and n_heads > 1:
        return "KV_MQA"
    if n_kv < n_heads:
        return "KV_GQA"
    return "KV_FULL"


# ---------------------------------------------------------------------------
# formulas — unique ANCHOR comments are the mutation-check targets
# ---------------------------------------------------------------------------

def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """n_layers * n_kv_heads * head_dim * 2 * dtype_bytes. The 2 is K and V."""
    return int(n_layers) * int(n_kv_heads) * int(head_dim) * 2 * int(dtype_bytes)  # ANCHOR:kv_per_token


def mla_bytes_per_token(n_layers: int, kv_lora_rank: int, qk_rope_head_dim: int, dtype_bytes: int) -> int:
    """Compressed latent + decoupled RoPE key, not n_heads * head_dim."""
    per_layer = int(kv_lora_rank) + int(qk_rope_head_dim)  # ANCHOR:mla_latent
    return int(n_layers) * per_layer * int(dtype_bytes)


def mamba1_fixed_bytes(
    n_layers: int, hidden: int, expand: int, d_state: int, d_conv: int, dtype_bytes: int,
) -> int:
    d_inner = int(hidden) * int(expand)
    ssm = d_inner * int(d_state)
    conv = d_inner * int(d_conv)
    return int(n_layers) * (ssm + conv) * int(dtype_bytes)


def mamba2_fixed_bytes(
    n_layers: int, n_heads: int, d_head: int, d_state: int, d_conv: int,
    n_groups: int, d_ssm: int, dtype_bytes: int,
) -> int:
    ssm = int(n_heads) * int(d_head) * int(d_state)
    conv_dim = int(d_ssm) + 2 * int(n_groups) * int(d_state)
    conv = conv_dim * int(d_conv)
    return int(n_layers) * (ssm + conv) * int(dtype_bytes)


def deltanet_fixed_bytes(n_layers: int, n_heads: int, head_dim: int, kernel: int, dtype_bytes: int) -> int:
    matrix = int(n_heads) * int(head_dim) * int(head_dim)
    conv = 3 * int(n_heads) * int(head_dim) * max(int(kernel) - 1, 0)
    return int(n_layers) * (matrix + conv) * int(dtype_bytes)


def qwen_linear_fixed_bytes(
    n_layers: int, n_k: int, d_k: int, n_v: int, d_v: int, kernel: int, dtype_bytes: int,
) -> int:
    # GQA-style gated-delta state is n_v * d_k * d_v (keys broadcast across value heads).
    matrix = int(n_v) * int(d_k) * int(d_v)
    conv = (int(n_k) * int(d_k) + 2 * int(n_v) * int(d_v)) * max(int(kernel) - 1, 0)
    return int(n_layers) * (matrix + conv) * int(dtype_bytes)


def rwkv7_fixed_bytes(
    n_layers: int, n_heads: int, head_dim: int, hidden: int,
    state_dtype_bytes: int, act_dtype_bytes: int,
) -> int:
    wkv = int(n_layers) * int(n_heads) * int(head_dim) * int(head_dim) * int(state_dtype_bytes)
    shift = int(n_layers) * int(hidden) * int(act_dtype_bytes)
    return wkv + shift


def lfm2_conv_fixed_bytes(n_layers: int, conv_dim: int, conv_l_cache: int, dtype_bytes: int) -> int:
    return int(n_layers) * int(conv_dim) * int(conv_l_cache) * int(dtype_bytes)


def window_extent(max_pos: int, window: int | None) -> int:
    if window is None:
        return int(max_pos)
    return min(int(max_pos), int(window))  # ANCHOR:window_bound


def mixer_kind(name: str) -> str:
    n = str(name).lower().replace("-", "_")
    if n in _RECURRENT_MIXERS:
        return "recurrent"
    if n in _ATTN_MIXERS:
        return "attn"
    return "unknown"


def _one_index_to_zero(indices: list[int], n_layers: int) -> list[int]:
    if not indices:
        return []
    if max(indices) >= n_layers:
        return [i - 1 for i in indices if 1 <= i <= n_layers]
    return [i for i in indices if 0 <= i < n_layers]


# ---------------------------------------------------------------------------
# non-AR discovery
# ---------------------------------------------------------------------------

def non_ar_reason(cfg: dict, body: dict) -> str | None:
    archs = cfg.get("architectures") or body.get("architectures") or []
    if not isinstance(archs, list):
        archs = [archs]
    joined = " ".join(str(a) for a in archs).lower()
    if any(tok in joined.replace("-", "") for tok in _NON_AR_ARCH_TOKENS):
        return f"architecture {archs} is not an autoregressive decoder"
    mt = str(cfg.get("model_type") or body.get("model_type") or "").lower()
    if mt in _NON_AR_MODEL_TYPES:
        return f"model_type {mt!r} has no persistent per-token decode state"
    if mt == "dream" or (cfg.get("mask_token_id") is not None and any("Dream" in str(a) for a in archs)):
        return "masked-diffusion body: no AR per-token decode state"
    if cfg.get("type") == "pi0":
        return "pi0 policy config carries no decoder attention fields"
    class_name = str(cfg.get("_class_name") or "")
    if "Video" in class_name or mt in {"hunyuanvideo"}:
        return f"video synthesis body ({class_name or mt}): no AR per-token decode state"
    name = cfg.get("Name") or cfg.get("name")
    if isinstance(name, list) and any("Video" in str(x) or "Hyena" in str(x) for x in name):
        # Hyena without dims is refused later; Video without dims is NONE.
        if any("Video" in str(x) for x in name):
            return f"Name {name} is a video synthesis body"
    if isinstance(name, str) and "Video" in name:
        return f"Name {name!r} is a video synthesis body"
    return None


# ---------------------------------------------------------------------------
# layer inventory
# ---------------------------------------------------------------------------

def _attn_window(body: dict, max_pos: int) -> int | None:
    """Body-wide sliding window. sliding_window_size with local_layer_ids is per-layer, not body-wide."""
    use = body.get("use_sliding_window")
    sw = body.get("sliding_window")
    if sw is None and not body.get("local_layer_ids"):
        sw = body.get("sliding_window_size")
    if use is False:
        return None
    if isinstance(sw, int) and sw > 0 and (max_pos <= 0 or sw < max_pos):
        return sw
    return None


def _jamba_attn_layers(n_layers: int, offset: int, period: int) -> set[int]:
    if period <= 0:
        return set()
    return {i for i in range(n_layers) if (i - offset) % period == 0}


def layer_plan(body: dict, n_layers: int, path: str) -> list[dict[str, Any]]:
    """One entry per layer: attn, recurrent, or both. Discovered from config fields."""
    types = body.get("layer_types")
    if isinstance(types, list) and types:
        if len(types) != n_layers:
            raise StateRefused(
                "layer_types", path,
                f"config at {path} layer_types has {len(types)} entries, num_hidden_layers={n_layers}",
            )
        plan = []
        for i, t in enumerate(types):
            kind = mixer_kind(str(t))
            if kind == "unknown":
                raise StateRefused(
                    f"layer_types[{i}]", path,
                    f"config at {path} layer_types[{i}]={t!r} is not a known mixer",
                )
            sliding = str(t).lower() in {"sliding_attention", "sliding_window", "local_attention"}
            plan.append({"kind": kind, "name": str(t), "sliding": sliding})
        return plan

    lin = body.get("linear_attn_config")
    if isinstance(lin, dict) and (lin.get("full_attn_layers") is not None or lin.get("kda_layers") is not None):
        full_raw = [int(x) for x in (lin.get("full_attn_layers") or [])]
        kda_raw = [int(x) for x in (lin.get("kda_layers") or [])]
        # One list being 1-indexed (an index == n_layers) means both lists are.
        one_based = bool(full_raw + kda_raw) and max(full_raw + kda_raw) >= n_layers
        if one_based:
            full = [i - 1 for i in full_raw if 1 <= i <= n_layers]
            kda = [i - 1 for i in kda_raw if 1 <= i <= n_layers]
        else:
            full = [i for i in full_raw if 0 <= i < n_layers]
            kda = [i for i in kda_raw if 0 <= i < n_layers]
        full_s, kda_s = set(full), set(kda)
        plan = []
        for i in range(n_layers):
            if i in full_s and i in kda_s:
                raise StateRefused(
                    "linear_attn_config", path,
                    f"config at {path} layer {i} is in both full_attn_layers and kda_layers",
                )
            if i in full_s:
                plan.append({"kind": "attn", "name": "full_attention", "sliding": False})
            elif i in kda_s:
                plan.append({"kind": "recurrent", "name": "kda", "sliding": False})
            else:
                raise StateRefused(
                    "linear_attn_config", path,
                    f"config at {path} layer {i} is in neither full_attn_layers nor kda_layers",
                )
        return plan

    if body.get("attn_layer_period") is not None and body.get("mamba_d_state") is not None:
        period = _as_int(body["attn_layer_period"], "attn_layer_period", path)
        offset = _as_int(body.get("attn_layer_offset", 0), "attn_layer_offset", path)
        attn = _jamba_attn_layers(n_layers, offset, period)
        plan = []
        for i in range(n_layers):
            if i in attn:
                plan.append({"kind": "attn", "name": "attention", "sliding": False})
            else:
                plan.append({"kind": "recurrent", "name": "mamba", "sliding": False})
        return plan

    # Parallel hybrid: every layer has both mixers (Falcon-H1: attn_layer_indices is
    # null and mamba_* + attention heads coexist with no layer_types).
    if (
        body.get("mamba_d_state") is not None
        and body.get("num_attention_heads") is not None
        and body.get("layer_types") is None
        and body.get("attn_layer_period") is None
    ):
        return [{"kind": "both", "name": "attn+mamba", "sliding": False} for _ in range(n_layers)]

    local_ids = body.get("local_layer_ids")
    if isinstance(local_ids, list) and local_ids:
        local = set(int(i) for i in local_ids)
        plan = []
        for i in range(n_layers):
            plan.append({"kind": "attn", "name": "attention", "sliding": i in local})
        return plan

    if body.get("ssm_cfg") is not None and body.get("num_attention_heads") is None:
        return [{"kind": "recurrent", "name": "mamba3", "sliding": False} for _ in range(n_layers)]

    if str(body.get("model_type") or "").lower() == "rwkv7" or body.get("wkv_state_dtype") is not None:
        return [{"kind": "recurrent", "name": "rwkv7", "sliding": False} for _ in range(n_layers)]

    return [{"kind": "attn", "name": "attention", "sliding": False} for _ in range(n_layers)]


# ---------------------------------------------------------------------------
# per-kind sizers
# ---------------------------------------------------------------------------

def _attn_per_layer_bytes(body: dict, path: str, dtype_bytes: int, fields: dict) -> tuple[int, str, dict]:
    """Bytes of persistent KV added by ONE attention layer per generated token."""
    kv_lora = body.get("kv_lora_rank")
    if isinstance(kv_lora, int) and kv_lora > 0:
        fields["kv_lora_rank"] = kv_lora
        rope = body.get("qk_rope_head_dim")
        if rope is None:
            raise StateRefused(
                "qk_rope_head_dim", path,
                f"config at {path} has kv_lora_rank={kv_lora} but is missing 'qk_rope_head_dim'",
            )
        rope_i = int(rope)
        fields["qk_rope_head_dim"] = rope_i
        # MLA stores one compressed latent + RoPE key, not K and V of n_heads*head_dim.
        return mla_bytes_per_token(1, kv_lora, rope_i, dtype_bytes), "KV_MLA", {
            "formula": "kv_lora_rank + qk_rope_head_dim",
        }

    n_heads, hk = resolve_n_heads(body, path)
    fields[hk] = n_heads
    n_kv_raw, n_kv_key = pick_first(
        body, ("num_key_value_heads", "n_kv_heads", "num_kv_heads"), path, required=False,
    )
    if n_kv_raw is None:
        n_kv = n_heads
        n_kv_key = hk
        fields["num_key_value_heads_absent"] = True
    else:
        n_kv = _as_int(n_kv_raw, n_kv_key, path)  # type: ignore[arg-type]
        fields[n_kv_key] = n_kv  # type: ignore[index]
    n_kv = int(body.get("num_key_value_heads", n_kv))  # ANCHOR:gqa_kv_heads
    fields["n_kv_heads_used"] = n_kv
    head_dim, hdk = resolve_head_dim(body, n_heads, path, fields)
    klass = kv_class(n_heads, n_kv)
    return kv_bytes_per_token(1, n_kv, head_dim, dtype_bytes), klass, {
        "formula": "n_kv_heads * head_dim * 2 * dtype_bytes",
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "head_dim_source": hdk,
    }


def _recurrent_per_layer_bytes(body: dict, name: str, path: str, dtype_bytes: int, fields: dict) -> tuple[int, str]:
    """Fixed persistent state of ONE recurrent/linear/mamba/conv layer."""
    rec_dtype = dtype_bytes
    for key in ("wkv_state_dtype", "mamba_ssm_dtype"):
        if body.get(key):
            rec_dtype = dtype_bytes_of(body[key], key, path)
            fields[key] = body[key]
            break

    if name in {"kda", "linear_attention", "deltanet", "delta_net", "gated_deltanet"} or body.get("linear_attn_config"):
        if body.get("linear_num_value_heads") and body.get("linear_key_head_dim"):
            n_k = int(body["linear_num_key_heads"])
            d_k = int(body["linear_key_head_dim"])
            n_v = int(body["linear_num_value_heads"])
            d_v = int(body["linear_value_head_dim"])
            k = int(body.get("linear_conv_kernel_dim") or 0)
            fields.update({
                "linear_num_key_heads": n_k, "linear_key_head_dim": d_k,
                "linear_num_value_heads": n_v, "linear_value_head_dim": d_v,
                "linear_conv_kernel_dim": k,
            })
            return qwen_linear_fixed_bytes(1, n_k, d_k, n_v, d_v, k, rec_dtype), (
                "n_v_heads * d_k * d_v (+ short conv)"
            )
        lin = body.get("linear_attn_config") or {}
        if lin.get("num_heads") and lin.get("head_dim"):
            nh, hd = int(lin["num_heads"]), int(lin["head_dim"])
            k = int(lin.get("short_conv_kernel_size") or 0)
            fields["linear_attn_config.num_heads"] = nh
            fields["linear_attn_config.head_dim"] = hd
            fields["linear_attn_config.short_conv_kernel_size"] = k
            return deltanet_fixed_bytes(1, nh, hd, k, rec_dtype), "n_heads * head_dim * head_dim (+ short conv)"
        raise StateRefused(
            "linear_attn_config.num_heads", path,
            f"config at {path} linear-attention layer {name!r} is missing linear_attn_config.num_heads/head_dim "
            f"and linear_num_value_heads/linear_key_head_dim",
        )

    if name in {"conv", "short_conv"} or (body.get("conv_L_cache") is not None and name == "conv"):
        conv_dim = body.get("conv_dim") or body.get("hidden_size")
        cache_l = body.get("conv_L_cache")
        if conv_dim is None or cache_l is None:
            raise StateRefused(
                "conv_L_cache" if cache_l is None else "conv_dim", path,
                f"config at {path} conv layer is missing conv_dim/hidden_size or conv_L_cache",
            )
        fields["conv_dim"] = int(conv_dim)
        fields["conv_L_cache"] = int(cache_l)
        return lfm2_conv_fixed_bytes(1, int(conv_dim), int(cache_l), rec_dtype), "conv_dim * conv_L_cache"

    if name == "rwkv7" or body.get("wkv_state_dtype") is not None:
        n_heads = int(require(body, "num_heads", path))
        head_dim = int(require(body, "head_dim", path))
        hidden = int(require(body, "hidden_size", path))
        fields.update({"num_heads": n_heads, "head_dim": head_dim, "hidden_size": hidden})
        state_dt = rec_dtype
        return rwkv7_fixed_bytes(1, n_heads, head_dim, hidden, state_dt, dtype_bytes), (
            "n_heads * head_dim * head_dim (wkv) + hidden_size (shift)"
        )

    ssm = body.get("ssm_cfg")
    if isinstance(ssm, dict) and name in {"mamba3", "mamba", "mamba2"}:
        d_model = int(require(body, "d_model", path))
        d_state = int(require(ssm, "d_state", path + "#ssm_cfg"))
        expand = int(require(ssm, "expand", path + "#ssm_cfg"))
        headdim = int(require(ssm, "headdim", path + "#ssm_cfg"))
        ngroups = int(ssm.get("ngroups") or 1)
        d_inner = d_model * expand
        if headdim <= 0 or d_inner % headdim:
            raise StateRefused(
                "ssm_cfg.headdim", path,
                f"config at {path} d_model*expand={d_inner} is not divisible by headdim={headdim}",
            )
        n_heads = d_inner // headdim
        d_conv = int(ssm["d_conv"]) if ssm.get("d_conv") is not None else 0
        fields.update({
            "d_model": d_model, "ssm_cfg.d_state": d_state, "ssm_cfg.expand": expand,
            "ssm_cfg.headdim": headdim, "ssm_cfg.ngroups": ngroups,
            "ssm_cfg.d_conv": ssm.get("d_conv"),
        })
        d_ssm = d_inner
        return mamba2_fixed_bytes(1, n_heads, headdim, d_state, d_conv or 0, ngroups, d_ssm, rec_dtype), (
            "mamba2/3: n_heads * d_head * d_state (+ conv if d_conv declared)"
        )

    if body.get("mamba_n_heads") is not None and body.get("mamba_d_head") is not None:
        n_heads = int(body["mamba_n_heads"])
        d_head = int(body["mamba_d_head"])
        d_state = int(require(body, "mamba_d_state", path))
        d_conv = int(require(body, "mamba_d_conv", path))
        n_groups = int(body.get("mamba_n_groups") or 1)
        d_ssm = int(body["mamba_d_ssm"]) if body.get("mamba_d_ssm") is not None else n_heads * d_head
        fields.update({
            "mamba_n_heads": n_heads, "mamba_d_head": d_head, "mamba_d_state": d_state,
            "mamba_d_conv": d_conv, "mamba_n_groups": n_groups, "mamba_d_ssm": d_ssm,
        })
        return mamba2_fixed_bytes(1, n_heads, d_head, d_state, d_conv, n_groups, d_ssm, rec_dtype), (
            "mamba2: n_heads * d_head * d_state + (d_ssm + 2*n_groups*d_state)*d_conv"
        )

    if body.get("mamba_d_state") is not None:
        hidden = int(require(body, "hidden_size", path))
        expand = int(require(body, "mamba_expand", path))
        d_state = int(body["mamba_d_state"])
        d_conv = int(require(body, "mamba_d_conv", path))
        fields.update({
            "hidden_size": hidden, "mamba_expand": expand,
            "mamba_d_state": d_state, "mamba_d_conv": d_conv,
        })
        return mamba1_fixed_bytes(1, hidden, expand, d_state, d_conv, rec_dtype), (
            "mamba1: (hidden*expand) * (d_state + d_conv)"
        )

    raise StateRefused(
        "mamba_d_state", path,
        f"config at {path} recurrent layer {name!r} is missing the fields the mixer formula needs "
        f"(mamba_d_state / linear_attn_config / conv_L_cache / ssm_cfg / wkv_state_dtype)",
    )


# ---------------------------------------------------------------------------
# measure one config
# ---------------------------------------------------------------------------

def measure_config(
    cfg: dict,
    *,
    path: str,
    specimen_dir: str | None = None,
    nest: str | None = None,
    parent: dict | None = None,
) -> dict[str, Any]:
    """Size persistent decode state from a parsed config. Never loads tensors."""
    if nest is None:
        body, nest = resolve_body(cfg)
        parent = cfg if nest else parent
    else:
        body = cfg
    cpath = _cfg_path(path, nest)
    fields: dict[str, Any] = {}
    if nest:
        fields["nested_body"] = nest
    archs = cfg.get("architectures") or body.get("architectures")
    fields["architectures"] = archs
    fields["model_type"] = body.get("model_type") or cfg.get("model_type")

    reason = non_ar_reason(parent or cfg, body)
    if reason:
        return _measured_row(
            state_class="NONE",
            per_token=0,
            fixed=0,
            at_ctx=0,
            dtype_bytes=None,
            dtype_name=None,
            fields=fields,
            growth="NONE",
            window=None,
            max_pos=None,
            formula=reason,
            path=cpath,
            layer_mix={"none": 1},
        )

    try:
        dtype_bytes, dtype_name, dtype_key = resolve_dtype(body, parent, cpath, specimen_dir)
        fields[dtype_key] = dtype_name
        n_layers, lk = resolve_n_layers(body, cpath)
        fields[lk] = n_layers
    except StateRefused:
        # A generative-looking body missing a sizing field is a refusal, not NONE.
        raise

    plan = layer_plan(body, n_layers, cpath)
    n_attn_layers = sum(1 for p in plan if p["kind"] in {"attn", "both"})
    max_pos: int | None = None
    pk = None
    if n_attn_layers:
        max_pos, pk = resolve_max_pos(body, parent, cpath)
        fields[pk] = max_pos
    else:
        v, k = pick_first(
            body,
            (
                "max_position_embeddings", "model_max_length", "max_target_positions",
                "n_positions", "context_length",
            ),
            cpath,
            required=False,
        )
        if v is None and parent:
            v, k = pick_first(
                parent,
                (
                    "max_position_embeddings", "model_max_length", "max_target_positions",
                    "n_positions", "context_length",
                ),
                cpath,
                required=False,
            )
        if v is not None:
            max_pos = _as_int(v, k, cpath)  # type: ignore[arg-type]
            fields[k] = max_pos

    global_window = _attn_window(body, max_pos or 0)
    if global_window is not None:
        fields["sliding_window"] = global_window
        if "use_sliding_window" in body:
            fields["use_sliding_window"] = body["use_sliding_window"]
    local_window = body.get("sliding_window_size")
    if isinstance(local_window, int) and local_window > 0:
        fields["sliding_window_size"] = local_window
    n_attn = n_attn_layers
    n_rec = sum(1 for p in plan if p["kind"] in {"recurrent", "both"})
    layer_mix = {
        "attention_layers": n_attn,
        "recurrent_layers": n_rec,
        "n_layers": n_layers,
        "names": dict(collections.Counter(p["name"] for p in plan)),
    }

    attn_per_layer = 0
    attn_klass = None
    attn_meta: dict[str, Any] = {}
    rec_per_layer_by_name: dict[str, int] = {}
    rec_formula = None

    if n_attn:
        attn_per_layer, attn_klass, attn_meta = _attn_per_layer_bytes(body, cpath, dtype_bytes, fields)
        fields["attn_formula"] = attn_meta.get("formula")
        for k in ("n_heads", "n_kv_heads", "head_dim", "head_dim_source"):
            if k in attn_meta:
                fields[k] = attn_meta[k]

    if n_rec:
        # Size each recurrent name once (granite mamba vs LFM2 conv vs KDA).
        for p in plan:
            if p["kind"] not in {"recurrent", "both"}:
                continue
            if p["name"] in rec_per_layer_by_name:
                continue
            rec_per_layer_by_name[p["name"]], rec_formula = _recurrent_per_layer_bytes(
                body, p["name"], cpath, dtype_bytes, fields,
            )
        fields["recurrent_formula"] = rec_formula

    attention_per_token = 0
    recurrent_fixed = 0
    attn_at_ctx = 0
    any_unbounded = False
    any_bounded = False
    mixed_windows = any(p["sliding"] for p in plan) and any(
        (not p["sliding"] and p["kind"] in {"attn", "both"}) for p in plan
    )
    uniform_windowed = (
        global_window is not None
        and not mixed_windows
        and not any(p["sliding"] for p in plan)
    )

    for p in plan:
        if p["kind"] in {"attn", "both"}:
            if max_pos is None:
                raise StateRefused(
                    "max_position_embeddings", cpath,
                    f"config at {cpath} has attention layers but is missing max_position_embeddings",
                )
            layer_kv = attn_per_layer
            attention_per_token += layer_kv  # ANCHOR:hybrid_attn_separate
            if p["sliding"]:
                win = local_window if isinstance(local_window, int) and local_window > 0 else global_window
                if win is None:
                    raise StateRefused(
                        "sliding_window", cpath,
                        f"config at {cpath} has a sliding/local layer named {p['name']!r} but no sliding_window/sliding_window_size",
                    )
                attn_at_ctx += layer_kv * window_extent(max_pos, int(win))
                any_bounded = True
            elif uniform_windowed:
                attn_at_ctx += layer_kv * window_extent(max_pos, global_window)
                any_bounded = True
            else:
                attn_at_ctx += layer_kv * int(max_pos)
                any_unbounded = True
        if p["kind"] in {"recurrent", "both"}:
            layer_fixed = rec_per_layer_by_name[p["name"]]
            recurrent_fixed += layer_fixed  # ANCHOR:hybrid_recurrent_separate

    if n_attn == 0 and n_rec == 0:
        raise StateRefused(
            "num_hidden_layers", cpath,
            f"config at {cpath} produced an empty layer plan",
        )

    if n_attn == 0:
        state_class = "RECURRENT"
        per_token = 0  # ANCHOR:recurrent_zero_per_token
        growth = "NONE"
        at_ctx = recurrent_fixed
        window = None
        formula = fields.get("recurrent_formula")
    elif n_rec == 0:
        state_class = attn_klass or "KV_FULL"
        per_token = attention_per_token
        growth = "BOUNDED" if (any_bounded and not any_unbounded) else "LINEAR"
        at_ctx = attn_at_ctx
        window = global_window if growth == "BOUNDED" else None
        formula = fields.get("attn_formula")
    else:
        state_class = "HYBRID"
        per_token = attention_per_token
        growth = "BOUNDED" if (any_bounded and not any_unbounded) else ("LINEAR" if n_attn else "NONE")
        if not any_unbounded and not any_bounded:
            growth = "NONE" if attention_per_token == 0 else "LINEAR"
        at_ctx = attn_at_ctx + recurrent_fixed
        window = global_window if growth == "BOUNDED" else None
        formula = (
            f"attn[{fields.get('attn_formula')}]; "
            f"recurrent[{fields.get('recurrent_formula')}] — counted separately, not averaged"
        )

    if growth == "BOUNDED" and window is None and isinstance(local_window, int):
        window = local_window

    return _measured_row(
        state_class=state_class,
        per_token=int(per_token),
        fixed=int(recurrent_fixed),
        at_ctx=int(at_ctx),
        dtype_bytes=int(dtype_bytes),
        dtype_name=dtype_name,
        fields=fields,
        growth=growth,
        window=window,
        max_pos=int(max_pos) if max_pos is not None else None,
        formula=formula,
        path=cpath,
        layer_mix=layer_mix,
        attn_klass=attn_klass,
    )


def _measured_row(
    *,
    state_class: str,
    per_token: int,
    fixed: int,
    at_ctx: int,
    dtype_bytes: int | None,
    dtype_name: str | None,
    fields: dict,
    growth: str,
    window: int | None,
    max_pos: int | None,
    formula: str | None,
    path: str,
    layer_mix: dict,
    attn_klass: str | None = None,
) -> dict[str, Any]:
    if state_class not in STATE_CLASSES:
        raise RuntimeError(f"illegal state_class {state_class!r}")
    return {
        "status": "MEASURED",
        "state_class": state_class,
        "state_bytes_per_token": per_token,
        "state_bytes_fixed": fixed,
        "state_bytes_at_ctx": at_ctx,
        "dtype_bytes": dtype_bytes,
        "dtype_name": dtype_name,
        "fields_read": fields,
        "growth": growth,
        "window": window,
        "max_position_embeddings": max_pos,
        "formula": formula,
        "config_path": path,
        "layer_mix": layer_mix,
        "attn_class": attn_klass,
        "refusal": None,
    }


def refusal_row(exc: StateRefused, *, state_class: str = "NONE") -> dict[str, Any]:
    if state_class not in STATE_CLASSES:
        state_class = "NONE"
    return {
        "status": "REFUSED",
        "state_class": state_class,
        "state_bytes_per_token": None,
        "state_bytes_fixed": None,
        "state_bytes_at_ctx": None,
        "dtype_bytes": None,
        "dtype_name": None,
        "fields_read": {},
        "growth": None,
        "window": None,
        "max_position_embeddings": None,
        "formula": None,
        "config_path": exc.path,
        "layer_mix": None,
        "attn_class": None,
        "refusal": {
            "missing_key": exc.missing_key,
            "path": exc.path,
            "mechanism": exc.mechanism,
        },
    }


# ---------------------------------------------------------------------------
# on-disk config
# ---------------------------------------------------------------------------

def load_config_file(specimen_dir: str) -> tuple[dict, str]:
    """Read the specimen's own config. Nested configs are used only when the root is absent."""
    root = os.path.join(specimen_dir, "config.json")
    if os.path.isfile(root):
        try:
            return json.loads(Path(root).read_text()), root
        except json.JSONDecodeError as exc:
            raise StateRefused(
                "config.json", root,
                f"config at {root} is not valid JSON: {exc}",
            ) from exc
    nested: list[str] = []
    if os.path.isdir(specimen_dir):
        for dirpath, dirnames, filenames in os.walk(specimen_dir):
            dirnames[:] = [d for d in dirnames if d not in {".cache", ".git"}]
            if "config.json" in filenames:
                nested.append(os.path.join(dirpath, "config.json"))
            if len(nested) >= 8:
                break
    if not nested:
        raise StateRefused(
            "config.json", specimen_dir,
            f"config.json is missing under {specimen_dir}",
        )
    # Two DiT twins (Wan high/low noise) share a layout; read the first that parses.
    last_err: Exception | None = None
    for cand in nested:
        try:
            return json.loads(Path(cand).read_text()), cand
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
    raise StateRefused(
        "config.json", nested[0],
        f"nested config at {nested[0]} is not valid JSON: {last_err}",
    )


def _infer_class(cfg: dict) -> str:
    """Best-effort class when a required field is missing. Never invents a size."""
    body, _ = resolve_body(cfg)
    types = body.get("layer_types")
    if isinstance(types, list) and types:
        kinds = {mixer_kind(str(t)) for t in types}
        if "unknown" not in kinds and "recurrent" in kinds and "attn" in kinds:
            return "HYBRID"
        if kinds == {"recurrent"}:
            return "RECURRENT"
    if body.get("linear_attn_config") and (body.get("kv_lora_rank") or body.get("num_attention_heads")):
        return "HYBRID"
    if body.get("ssm_cfg") is not None or body.get("wkv_state_dtype") is not None:
        return "RECURRENT"
    if isinstance(body.get("kv_lora_rank"), int) and body["kv_lora_rank"] > 0:
        return "KV_MLA"
    n_heads = body.get("num_attention_heads") or body.get("num_heads")
    n_kv = body.get("num_key_value_heads")
    if n_heads and n_kv:
        return kv_class(int(n_heads), int(n_kv))
    if n_heads:
        return "KV_FULL"
    return "NONE"


def measure_specimen(specimen_dir: str) -> dict[str, Any]:
    try:
        cfg, cfg_path = load_config_file(specimen_dir)
    except StateRefused as exc:
        return refusal_row(exc, state_class="NONE")
    try:
        return measure_config(cfg, path=cfg_path, specimen_dir=specimen_dir)
    except StateRefused as exc:
        return refusal_row(exc, state_class=_infer_class(cfg))


def measure_catalog(catalog_path: str) -> dict[str, Any]:
    cat = json.loads(Path(catalog_path).read_text())
    specimens = cat["specimens"]
    rows = []
    for s in specimens:
        slug = s["slug"]
        path = s["path"]
        family = s.get("architecture_family")
        try:
            if not os.path.isdir(path):
                raise StateRefused(
                    "path", path,
                    f"specimen path {path} is not a directory",
                )
            row = measure_specimen(path)
        except StateRefused as exc:
            row = refusal_row(exc)
        row["slug"] = slug
        row["family"] = family
        row["path"] = path
        if row["state_class"] not in STATE_CLASSES:
            raise RuntimeError(f"{slug}: illegal state_class {row['state_class']!r}")
        if row["status"] == "REFUSED":
            ref = row.get("refusal") or {}
            if not ref.get("missing_key") or str(ref.get("missing_key")).lower() == "unknown":
                raise RuntimeError(f"{slug}: refusal without a named missing key")
            if not ref.get("path"):
                raise RuntimeError(f"{slug}: refusal without a path")
        rows.append(row)
    return {
        "n": len(rows),
        "n_catalog": len(specimens),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def gqa_negative_control(cfg: dict | None = None) -> dict[str, Any]:
    """A GQA body must not produce the full-MHA number. Ratio equals the grouping factor."""
    if cfg is None:
        cfg = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 40960,
        }
    row = measure_config(cfg, path="synthetic-gqa")
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    dt = 2
    gqa = kv_bytes_per_token(n_layers, n_kv, head_dim, dt)
    mha = kv_bytes_per_token(n_layers, n_heads, head_dim, dt)
    grouping = n_heads / n_kv
    measured = row["state_bytes_per_token"]
    if measured != gqa:
        raise AssertionError(f"GQA measured {measured}, expected {gqa}")
    if measured == mha:
        raise AssertionError("GQA control produced the full-MHA number")
    if mha / measured != grouping:
        raise AssertionError(f"MHA/GQA ratio {mha / measured} != grouping {grouping}")
    if row["state_class"] != "KV_GQA":
        raise AssertionError(f"expected KV_GQA, got {row['state_class']}")
    return {
        "name": "gqa_negative",
        "gqa_bytes_per_token": gqa,
        "mha_bytes_per_token": mha,
        "measured_bytes_per_token": measured,
        "grouping_factor": grouping,
        "measured_ratio_mha_over_gqa": mha / measured,
        "predicted_ratio": grouping,
        "passed": True,
    }


def mla_control(cfg: dict | None = None) -> dict[str, Any]:
    """An MLA body must not be sized by head_dim * n_heads."""
    if cfg is None:
        cfg = {
            "architectures": ["KimiVLForConditionalGeneration"],
            "model_type": "kimi_vl",
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "hidden_size": 2048,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072,
        }
    row = measure_config(cfg, path="synthetic-mla")
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    kv_lora = int(cfg["kv_lora_rank"])
    rope = int(cfg["qk_rope_head_dim"])
    dt = 2
    mla = mla_bytes_per_token(n_layers, kv_lora, rope, dt)
    # Wrong MHA using hidden/n_heads as head_dim (128) — the trap the task names.
    wrong = kv_bytes_per_token(n_layers, n_heads, 128, dt)
    measured = row["state_bytes_per_token"]
    if measured != mla:
        raise AssertionError(f"MLA measured {measured}, expected {mla}")
    if measured == wrong:
        raise AssertionError("MLA control was sized by head_dim * n_heads")
    if row["state_class"] != "KV_MLA":
        raise AssertionError(f"expected KV_MLA, got {row['state_class']}")
    return {
        "name": "mla",
        "mla_bytes_per_token": mla,
        "wrong_mha_bytes_per_token": wrong,
        "measured_bytes_per_token": measured,
        "passed": True,
    }


def twin_pair_control(catalog_path: str) -> dict[str, Any]:
    """Two Qwen3 bodies, same n_kv and head_dim, different n_layers. Ratio is n_layers ratio."""
    cat = json.loads(Path(catalog_path).read_text())
    by_slug = {s["slug"]: s for s in cat["specimens"]}
    a_slug = "Qwen--Qwen3-0.6B@c1899de289a0"
    b_slug = "Qwen--Qwen3-14B@40c069824f42"
    if a_slug not in by_slug or b_slug not in by_slug:
        raise RuntimeError(
            f"TWIN-PAIR NOT RUN — {a_slug} or {b_slug} absent from catalog. "
            "This control did not execute; do not read its silence as a pass."
        )
    a_path, b_path = by_slug[a_slug]["path"], by_slug[b_slug]["path"]
    if not (os.path.isdir(a_path) and os.path.isdir(b_path)):
        raise RuntimeError(
            f"TWIN-PAIR NOT RUN — specimen dirs missing under the lake "
            f"({a_path}, {b_path}). This control did not execute."
        )
    a = measure_specimen(a_path)
    b = measure_specimen(b_path)
    if a["status"] != "MEASURED" or b["status"] != "MEASURED":
        raise AssertionError(f"twin pair refused: {a.get('refusal')} / {b.get('refusal')}")
    fa, fb = a["fields_read"], b["fields_read"]
    pred = (fb["num_hidden_layers"] * fb["n_kv_heads"] * fb["head_dim"]) / (
        fa["num_hidden_layers"] * fa["n_kv_heads"] * fa["head_dim"]
    )
    meas = b["state_bytes_per_token"] / a["state_bytes_per_token"]
    if abs(pred - meas) > 1e-12:
        raise AssertionError(f"twin predicted ratio {pred} != measured {meas}")
    return {
        "name": "twin_pair",
        "family": "qwen3",
        "a": {"slug": a_slug, "state_bytes_per_token": a["state_bytes_per_token"],
              "n_layers": fa["num_hidden_layers"], "n_kv_heads": fa["n_kv_heads"],
              "head_dim": fa["head_dim"]},
        "b": {"slug": b_slug, "state_bytes_per_token": b["state_bytes_per_token"],
              "n_layers": fb["num_hidden_layers"], "n_kv_heads": fb["n_kv_heads"],
              "head_dim": fb["head_dim"]},
        "predicted_ratio": pred,
        "measured_ratio": meas,
        "passed": True,
    }


def lake_gqa_negative(catalog_path: str) -> dict[str, Any]:
    """Real Qwen3-0.6B is GQA; full-MHA would be 2x (16/8)."""
    cat = json.loads(Path(catalog_path).read_text())
    slug = "Qwen--Qwen3-0.6B@c1899de289a0"
    by_slug = {s["slug"]: s for s in cat["specimens"]}
    if slug not in by_slug:
        raise RuntimeError(f"LAKE GQA CONTROL NOT RUN — {slug} absent from catalog")
    path = by_slug[slug]["path"]
    if not os.path.isdir(path):
        raise RuntimeError(f"LAKE GQA CONTROL NOT RUN — {path} missing")
    cfg, cfg_path = load_config_file(path)
    body, nest = resolve_body(cfg)
    row = measure_config(cfg, path=cfg_path, specimen_dir=path)
    n_layers = int(body["num_hidden_layers"])
    n_heads = int(body["num_attention_heads"])
    n_kv = int(body["num_key_value_heads"])
    head_dim = int(body["head_dim"])
    dt = 2
    gqa = kv_bytes_per_token(n_layers, n_kv, head_dim, dt)
    mha = kv_bytes_per_token(n_layers, n_heads, head_dim, dt)
    grouping = n_heads / n_kv
    if row["state_bytes_per_token"] != gqa:
        raise AssertionError(f"lake GQA {row['state_bytes_per_token']} != {gqa}")
    if row["state_bytes_per_token"] == mha:
        raise AssertionError("lake GQA produced full-MHA")
    if mha / row["state_bytes_per_token"] != grouping:
        raise AssertionError("lake grouping factor mismatch")
    return {
        "name": "lake_gqa_negative",
        "slug": slug,
        "config_path": _cfg_path(cfg_path, nest),
        "gqa_bytes_per_token": gqa,
        "mha_bytes_per_token": mha,
        "grouping_factor": grouping,
        "measured_ratio_mha_over_gqa": mha / row["state_bytes_per_token"],
        "predicted_ratio": grouping,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# mutation checks — revert the line, confirm FAIL, restore, prove sha256 moved
# ---------------------------------------------------------------------------

_MUTATION_MARKER = "MUTATION_LIVE"


def leftover_mutation_live(source: str) -> bool:
    """True when a mutant line is executable in source, not just quoted in a check."""
    for line in source.splitlines():
        if _MUTATION_MARKER not in line:
            continue
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("_MUTATION_MARKER"):
            continue
        if stripped.startswith(("'", '"', "f'", 'f"', "'''", '"""')):
            continue
        return True
    return False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _mutation_check(anchor: str, old: str, new: str, probe: str) -> dict[str, Any]:
    """Replace a unique load-bearing line, prove the probe FAILS, restore.

    `probe` is a python snippet that must exit 0 on the healthy module and
    exit non-zero on the mutant. The subprocess imports THIS file by path so
    a stale .pyc cannot hide a no-op mutation.
    """
    src_path = Path(__file__).resolve()
    original = src_path.read_text()
    before = _sha256_text(original)
    if old not in original:
        raise RuntimeError(f"mutation did not apply: anchor {anchor!r} string not found in source")
    if original.count(old) != 1:
        raise RuntimeError(
            f"mutation did not apply: anchor {anchor!r} string is not unique "
            f"(count={original.count(old)})"
        )
    mutated = original.replace(old, new, 1)
    if _MUTATION_MARKER not in new:
        raise RuntimeError(f"mutant for {anchor} does not carry {_MUTATION_MARKER}")
    src_path.write_text(mutated)
    after = _sha256_text(src_path.read_text())
    applied = after != before
    if not applied:
        src_path.write_text(original)
        raise RuntimeError(f"mutation did not apply: sha256 unchanged for {anchor}")
    env = dict(os.environ)
    env["STATE_AXIS_INNER"] = "1"
    py = sys.executable
    mutant = None
    healthy = None
    restored = None
    try:
        mutant = subprocess.run([py, "-c", probe], capture_output=True, text=True, env=env)
        src_path.write_text(original)
        restored = _sha256_text(src_path.read_text())
        if restored != before:
            raise RuntimeError(f"restore failed for {anchor}: sha {restored} != {before}")
        if leftover_mutation_live(src_path.read_text()):
            raise RuntimeError(f"mutant marker left in source after {anchor}")
        healthy = subprocess.run([py, "-c", probe], capture_output=True, text=True, env=env)
    finally:
        if src_path.read_text() != original:
            src_path.write_text(original)
    if mutant is None or mutant.returncode == 0:
        raise RuntimeError(
            f"mutation {anchor} stayed green\n"
            f"stdout={getattr(mutant, 'stdout', '')}\nstderr={getattr(mutant, 'stderr', '')}"
        )
    if healthy is None or healthy.returncode != 0:
        raise RuntimeError(
            f"mutation {anchor}: healthy source failed after restore\n"
            f"stdout={getattr(healthy, 'stdout', '')}\nstderr={getattr(healthy, 'stderr', '')}"
        )
    return {
        "anchor": anchor,
        "sha256_before": before,
        "sha256_after": after,
        "sha256_restored": restored,
        "applied": True,
        "mutant_failed": True,
        "mutant_returncode": mutant.returncode,
        "healthy_returncode": healthy.returncode,
        "old": old.strip(),
        "new": new.strip(),
    }


def _probe_src(expr: str) -> str:
    here = Path(__file__).resolve()
    return (
        "import importlib.util\n"
        f"p = {str(here)!r}\n"
        "spec = importlib.util.spec_from_file_location('state_axis_mut', p)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        f"{expr}\n"
    )


def run_mutation_checks() -> list[dict[str, Any]]:
    if os.environ.get("STATE_AXIS_INNER"):
        return []
    checks = []

    gqa_probe = _probe_src(
        "cfg = dict(architectures=['Qwen3ForCausalLM'], model_type='qwen3',"
        " num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,"
        " head_dim=128, torch_dtype='bfloat16', max_position_embeddings=40960)\n"
        "row = m.measure_config(cfg, path='mut-gqa')\n"
        "assert row['state_bytes_per_token'] == 114688, row['state_bytes_per_token']\n"
        "assert row['state_class'] == 'KV_GQA'\n"
    )
    checks.append(_mutation_check(
        "gqa_kv_heads",
        '    n_kv = int(body.get("num_key_value_heads", n_kv))  # ANCHOR:' + 'gqa_kv_heads',
        '    n_kv = int(body.get("num_attention_heads", n_kv))  # MUTATION_LIVE ANCHOR:' + 'gqa_kv_heads',
        gqa_probe,
    ))

    mla_probe = _probe_src(
        "cfg = dict(architectures=['X'], model_type='kimi_vl', num_hidden_layers=27,"
        " num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512,"
        " qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128, hidden_size=2048,"
        " torch_dtype='bfloat16', max_position_embeddings=131072)\n"
        "row = m.measure_config(cfg, path='mut-mla')\n"
        "assert row['state_bytes_per_token'] == 31104, row['state_bytes_per_token']\n"
        "assert row['state_class'] == 'KV_MLA'\n"
    )
    checks.append(_mutation_check(
        "mla_latent",
        "    per_layer = int(kv_lora_rank) + int(qk_rope_head_dim)  # ANCHOR:" + "mla_latent",
        "    per_layer = int(kv_lora_rank) * int(qk_rope_head_dim)  # MUTATION_LIVE ANCHOR:" + "mla_latent",
        mla_probe,
    ))

    rec_probe = _probe_src(
        "cfg = dict(architectures=['Rwkv7ForCausalLM'], model_type='rwkv7',"
        " num_hidden_layers=2, num_heads=4, head_dim=8, hidden_size=32,"
        " dtype='bfloat16', wkv_state_dtype='float32', max_position_embeddings=64)\n"
        "row = m.measure_config(cfg, path='mut-rwkv')\n"
        "assert row['state_bytes_per_token'] == 0, row['state_bytes_per_token']\n"
        "assert row['state_class'] == 'RECURRENT'\n"
    )
    checks.append(_mutation_check(
        "recurrent_zero_per_token",
        "        per_token = 0  # ANCHOR:" + "recurrent_zero_per_token",
        "        per_token = 12345  # MUTATION_LIVE ANCHOR:" + "recurrent_zero_per_token",
        rec_probe,
    ))

    win_probe = _probe_src(
        "cfg = dict(architectures=['Starcoder2ForCausalLM'], model_type='starcoder2',"
        " num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,"
        " head_dim=8, torch_dtype='bfloat16', max_position_embeddings=100,"
        " sliding_window=10)\n"
        "row = m.measure_config(cfg, path='mut-win')\n"
        "assert row['growth'] == 'BOUNDED'\n"
        "assert row['state_bytes_at_ctx'] == row['state_bytes_per_token'] * 10, row\n"
    )
    checks.append(_mutation_check(
        "window_bound",
        "    return min(int(max_pos), int(window))  # ANCHOR:" + "window_bound",
        "    return int(max_pos)  # MUTATION_LIVE ANCHOR:" + "window_bound",
        win_probe,
    ))

    hyb_probe = _probe_src(
        "cfg = dict(architectures=['Lfm2ForCausalLM'], model_type='lfm2',"
        " num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,"
        " hidden_size=32, dtype='bfloat16', max_position_embeddings=16,"
        " conv_L_cache=3, conv_dim=32,"
        " layer_types=['conv', 'full_attention'])\n"
        "row = m.measure_config(cfg, path='mut-hyb')\n"
        "assert row['state_class'] == 'HYBRID'\n"
        "assert row['state_bytes_per_token'] == 1 * 2 * 8 * 2 * 2, row  # one attn layer, head_dim=32/4=8\n"
        "assert row['state_bytes_fixed'] > 0 and row['state_bytes_per_token'] > 0\n"
        # averaging would make per_token include half the conv state
        "assert row['state_bytes_per_token'] != (row['state_bytes_fixed'] + row['state_bytes_per_token']) // 2\n"
    )
    checks.append(_mutation_check(
        "hybrid_attn_separate",
        "            attention_per_token += layer_kv  # ANCHOR:" + "hybrid_attn_separate",
        "            attention_per_token += layer_kv + 1  # MUTATION_LIVE ANCHOR:" + "hybrid_attn_separate",
        hyb_probe,
    ))

    refuse_probe = _probe_src(
        "cfg = dict(architectures=['Rwkv7ForCausalLM'], model_type='rwkv7',"
        " num_hidden_layers=2, head_dim=8, hidden_size=32,"
        " dtype='bfloat16', wkv_state_dtype='float32', max_position_embeddings=64)\n"
        "try:\n"
        "    row = m.measure_config(cfg, path='mut-ref')\n"
        "    raise SystemExit('expected StateRefused, got ' + str(row.get('state_class')))\n"
        "except m.StateRefused as e:\n"
        "    assert e.missing_key == 'num_heads', e.missing_key\n"
    )
    checks.append(_mutation_check(
        "refuse_missing",
        "        raise State" + "Refused(\n"
        "            key, path,\n"
        "            f\"config at {path} is missing {key!r}\",\n"
        "        )",
        "        return 0  # MUTATION_" + "LIVE\n"
        "        raise State" + "Refused(\n"
        "            key, path,\n"
        "            f\"config at {path} is missing {key!r}\",\n"
        "        )",
        refuse_probe,
    ))

    if leftover_mutation_live(Path(__file__).read_text()):
        raise RuntimeError("mutant marker left in source after all checks")
    return checks


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

def _summarise(rows: list[dict]) -> dict[str, Any]:
    dist = dict(collections.Counter(r["state_class"] for r in rows))
    measured = [r for r in rows if r["status"] == "MEASURED"]
    refused = [r for r in rows if r["status"] == "REFUSED"]
    with_pt = [r for r in measured if isinstance(r.get("state_bytes_per_token"), int)]
    cheapest = min(with_pt, key=lambda r: (r["state_bytes_per_token"], r["slug"])) if with_pt else None
    expensive = max(with_pt, key=lambda r: (r["state_bytes_per_token"], r["slug"])) if with_pt else None
    with_ctx = [r for r in measured if isinstance(r.get("state_bytes_at_ctx"), int)]
    cheap_ctx = min(with_ctx, key=lambda r: (r["state_bytes_at_ctx"], r["slug"])) if with_ctx else None
    exp_ctx = max(with_ctx, key=lambda r: (r["state_bytes_at_ctx"], r["slug"])) if with_ctx else None
    # Cheapest *positive* per-token among growing caches — recurrent/NONE sit at 0.
    positive = [r for r in with_pt if r["state_bytes_per_token"] > 0]
    cheapest_positive = min(positive, key=lambda r: (r["state_bytes_per_token"], r["slug"])) if positive else None

    def slim(r: dict | None) -> dict | None:
        if r is None:
            return None
        return {
            "slug": r["slug"], "family": r.get("family"), "state_class": r["state_class"],
            "state_bytes_per_token": r["state_bytes_per_token"],
            "state_bytes_fixed": r["state_bytes_fixed"],
            "state_bytes_at_ctx": r["state_bytes_at_ctx"],
            "growth": r.get("growth"), "window": r.get("window"),
        }

    return {
        "state_class_distribution": dist,
        "n_measured": len(measured),
        "n_refused": len(refused),
        "cheapest_per_token": slim(cheapest),
        "cheapest_positive_per_token": slim(cheapest_positive),
        "most_expensive_per_token": slim(expensive),
        "cheapest_at_ctx": slim(cheap_ctx),
        "most_expensive_at_ctx": slim(exp_ctx),
        "refusals": [
            {
                "slug": r["slug"],
                "family": r.get("family"),
                "missing_key": r["refusal"]["missing_key"],
                "path": r["refusal"]["path"],
                "mechanism": r["refusal"]["mechanism"],
            }
            for r in refused
        ],
    }


def build_receipt(
    catalog_path: str,
    *,
    controls: dict[str, Any] | None = None,
    mutations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sweep = measure_catalog(catalog_path)
    if sweep["n"] != sweep["n_catalog"]:
        raise RuntimeError("catalog specimen count and row count diverged")
    summary = _summarise(sweep["rows"])
    doc = {
        "schema": SCHEMA,
        "recorded_by": RECORDED_BY,
        "what": (
            "Per-token persistent decode state from each ModelLake specimen's own "
            "config.json. KV bodies report bytes added per generated token; "
            "recurrent/DeltaNet/Mamba report fixed state and zero per-token growth; "
            "hybrids count each kind separately. A missing field is a named refusal."
        ),
        "claim_boundary": (
            "Static sidecar artifact. Config/header only. No tensor payload, no GPU, "
            "no hardware measurement."
        ),
        "catalog": str(catalog_path),
        "n": sweep["n"],
        "n_catalog": sweep["n_catalog"],
        "state_classes": list(STATE_CLASSES),
        "rows": sweep["rows"],
        "summary": summary,
        "controls": controls or {},
        "mutations": mutations or [],
    }
    return doc


def write_state_axis_receipt(
    catalog_path: str,
    *,
    controls: dict[str, Any] | None = None,
    mutations: list[dict[str, Any]] | None = None,
) -> Path:
    _refuse_volumes_write(str(RECEIPTS / RECEIPT_NAME))
    doc = build_receipt(catalog_path, controls=controls, mutations=mutations)
    return write_receipt(RECEIPT_NAME, doc, RECORDED_BY)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _assert_gqa_synthetic() -> None:
    gqa_negative_control()


def _assert_mla_synthetic() -> None:
    mla_control()


def _assert_refusal_names_key() -> None:
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "torch_dtype": "bfloat16",
    }
    try:
        measure_config(cfg, path="/tmp/missing-max.json")
    except StateRefused as exc:
        assert exc.missing_key == "max_position_embeddings", exc.missing_key
        assert "/tmp/missing-max.json" in exc.path
        assert "unknown" not in exc.mechanism.lower() or "max_position" in exc.mechanism
        return
    raise AssertionError("missing max_position_embeddings did not refuse")


def selftest(catalog_path: str) -> dict[str, Any]:
    _assert_gqa_synthetic()
    _assert_mla_synthetic()
    _assert_refusal_names_key()
    # Recurrent zero growth.
    rw = measure_config(
        {
            "architectures": ["Rwkv7ForCausalLM"],
            "model_type": "rwkv7",
            "num_hidden_layers": 2,
            "num_heads": 4,
            "head_dim": 8,
            "hidden_size": 32,
            "dtype": "bfloat16",
            "wkv_state_dtype": "float32",
            "max_position_embeddings": 64,
        },
        path="synthetic-rwkv",
    )
    assert rw["state_class"] == "RECURRENT" and rw["state_bytes_per_token"] == 0, rw
    assert rw["state_bytes_fixed"] > 0

    # Sliding window bounds at_ctx.
    sc = measure_config(
        {
            "architectures": ["Starcoder2ForCausalLM"],
            "model_type": "starcoder2",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 100,
            "sliding_window": 10,
        },
        path="synthetic-window",
    )
    assert sc["growth"] == "BOUNDED", sc
    assert sc["state_bytes_at_ctx"] == sc["state_bytes_per_token"] * 10, sc

    mutations = run_mutation_checks()
    controls = {
        "gqa_negative": gqa_negative_control(),
        "mla": mla_control(),
    }
    if Path(catalog_path).is_file():
        controls["twin_pair"] = twin_pair_control(catalog_path)
        controls["lake_gqa_negative"] = lake_gqa_negative(catalog_path)
        out = write_state_axis_receipt(catalog_path, controls=controls, mutations=mutations)
        controls["receipt"] = str(out)
    if leftover_mutation_live(Path(__file__).read_text()):
        raise RuntimeError("mutant marker left in source")
    print("selftest OK")
    print(json.dumps({
        "gqa": controls["gqa_negative"],
        "mla": controls["mla"],
        "twin_pair": controls.get("twin_pair"),
        "lake_gqa_negative": controls.get("lake_gqa_negative"),
        "mutations": [
            {k: m[k] for k in ("anchor", "sha256_before", "sha256_after", "applied", "mutant_failed")}
            for m in mutations
        ],
        "receipt": controls.get("receipt"),
    }, indent=2))
    return {"controls": controls, "mutations": mutations}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--catalog", default=str(CATALOG_DEFAULT))
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(args.catalog)
        return 0
    out = write_state_axis_receipt(args.catalog)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
