#!/usr/bin/env python3.12
"""DeepSeek-V4-Flash-DSpark organ adapter (deepseek_v4 / DeepseekV4ForCausalLM).

Reads official metadata ONLY (config.json + model.safetensors.index.json) and produces the organ
inventory the foundry needs: every tensor mapped to an organ class, the routed-expert grammar
confirmed against the real index, the weight/scale companion structure recorded, and a bounded
per-shard read plan. Nothing here downloads or loads the 155.4 GiB source.

Why this parent is different, and it matters for the ledger
  config.json declares  dtype "fp8",  scale_fmt "ue8m0",  expert_dtype "fp4".
  The routed experts - the exact organ this programme has been trying to push below one bit - are
  ALREADY four bits in the official checkpoint, and every quantised tensor ships a companion
  `.scale`. So:
    * "original_weight_count" must be bound from the resident tensor index, never from the 284B
      name-derived figure, and the scale tensors are part of the artifact's byte reality.
    * A 1.0 complete-BPW target against this parent is a ~4x reduction, not the ~16x that a bf16
      parent implies. That is a materially easier and better-posed sub-bit question, and it is the
      honest reason this parent outranks the bf16 giants on information gain per source byte.
    * Any BPW comparison against the sealed Qwen3-235B (bf16) is NOT like-for-like and must carry
      the source-format caveat.

Honesty boundary
  The HF safetensors index carries name->shard plus metadata.total_size only: no shapes, no
  dtypes. This adapter therefore reports EXACT tensor counts and shard mapping from the index, and
  leaves per-tensor bytes as pending-source until the shards are resident, at which point
  `verify_against_source` reads the real safetensors headers and reconciles byte-exactly against
  total_size. No analytic shape model is invented for a block-scaled fp4/fp8 layout that has not
  been read.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HF_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
IMMUTABLE_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
EXPECTED_MODEL_TYPE = "deepseek_v4"
DEFAULT_META = Path("models/deepseek-v4-flash-dspark/_meta")
MAX_HEADER_BYTES = 200 * 1024 * 1024

_ST_DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
                   "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
                   "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
                   "F4": 0.5, "U4": 0.5}


class DeepSeekV4AdapterError(Exception):
    """Raised on any config/index inconsistency or unrecognised tensor name."""


# ── organ classes ───────────────────────────────────────────────────────────────────────
ORGAN_EMBED = "embed"
ORGAN_HEAD = "lm_head"
ORGAN_FINAL_NORM = "norm"
ORGAN_HC_HEAD = "hyper_connection.head"
ORGAN_HC_BLOCK = "hyper_connection.block"
ORGAN_ATTN_MLA = "attn.mla"                    # wq_a/wq_b/wkv/wo_a/wo_b
ORGAN_ATTN_NORM = "attn.norm"                  # q_norm/kv_norm/attn_norm/ffn_norm
ORGAN_ATTN_SINK = "attn.sink"
ORGAN_COMPRESSOR = "attn.compressor"           # the DSpark long-context compressor
ORGAN_INDEXER = "attn.indexer"                 # DeepSeek sparse-attention indexer
ORGAN_ROUTER = "ffn.gate"
ORGAN_EXPERT = "ffn.experts"                   # THE routed-expert organ (fp4 native)
ORGAN_SHARED_EXPERT = "ffn.shared_experts"
ORGAN_MTP = "mtp"                              # multi-token-prediction stack
ORGAN_MARKOV = "mtp.markov_head"
ORGAN_CONFIDENCE = "mtp.confidence_head"

# Organs whose bytes the sub-bit campaign is allowed to spend against.
COMPRESSIBLE_ORGANS = {ORGAN_EXPERT, ORGAN_SHARED_EXPERT}

# The MLA projections that actually exist in this checkpoint's index.
_MLA_PROJECTIONS = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}

_TOP = {"embed.weight": ORGAN_EMBED, "head.weight": ORGAN_HEAD, "norm.weight": ORGAN_FINAL_NORM,
        "hc_head_base": ORGAN_HC_HEAD, "hc_head_fn": ORGAN_HC_HEAD,
        "hc_head_scale": ORGAN_HC_HEAD}

_RE_EXPERT = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.w(\d+)\.(weight|scale)$")
_RE_SHARED = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.shared_experts\.w(\d+)\.(weight|scale)$")
_RE_BLOCK = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")


def classify(name: str) -> dict[str, Any]:
    """Map a state-dict tensor name to organ class + coordinates. Raises on anything unknown."""
    if name in _TOP:
        return {"organ": _TOP[name], "stack": "top", "layer": None, "expert": None,
                "is_scale": name.endswith("_scale")}
    m = _RE_EXPERT.match(name)
    if m:
        return {"organ": ORGAN_EXPERT, "stack": m.group(1), "layer": int(m.group(2)),
                "expert": int(m.group(3)), "proj": f"w{m.group(4)}",
                "is_scale": m.group(5) == "scale"}
    m = _RE_SHARED.match(name)
    if m:
        return {"organ": ORGAN_SHARED_EXPERT, "stack": m.group(1), "layer": int(m.group(2)),
                "expert": None, "proj": f"w{m.group(3)}", "is_scale": m.group(4) == "scale"}
    m = _RE_BLOCK.match(name)
    if not m:
        raise DeepSeekV4AdapterError(f"unrecognized tensor name: {name!r}")
    stack, layer, rest = m.group(1), int(m.group(2)), m.group(3)
    is_scale = rest.endswith(".scale")
    organ = _block_organ(rest)
    return {"organ": organ, "stack": stack, "layer": layer, "expert": None,
            "is_scale": is_scale}


def _block_organ(rest: str) -> str:
    if rest.startswith("attn.indexer."):
        return ORGAN_INDEXER
    if rest.startswith("attn.compressor."):
        return ORGAN_COMPRESSOR
    if rest == "attn.attn_sink":
        return ORGAN_ATTN_SINK
    if rest.startswith("attn.") and rest.split(".")[1] in ("q_norm", "kv_norm"):
        return ORGAN_ATTN_NORM
    # Explicit allowlist, not a startswith("attn.") catch-all: a new attention sub-module must
    # raise so it gets an organ decision, rather than being silently billed as MLA.
    if rest.startswith("attn.") and rest.split(".")[1] in _MLA_PROJECTIONS:
        return ORGAN_ATTN_MLA
    if rest in ("attn_norm.weight", "ffn_norm.weight", "norm.weight", "main_norm.weight"):
        return ORGAN_ATTN_NORM
    if rest.startswith("hc_"):
        return ORGAN_HC_BLOCK
    if rest.startswith("ffn.gate"):
        return ORGAN_ROUTER
    if rest.startswith("markov_head."):
        return ORGAN_MARKOV
    if rest.startswith("confidence_head."):
        return ORGAN_CONFIDENCE
    if rest.startswith("main_proj"):
        return ORGAN_MTP
    # The MTP block's own projections: e_proj lifts the token embedding and h_proj lifts
    # the previous hidden state before they are combined. Both ship a .scale, so they are
    # quantized MTP infrastructure, not routed experts, and are billed at their real bytes.
    if rest.split(".")[0] in ("e_proj", "h_proj"):
        return ORGAN_MTP
    # enorm/hnorm are the MTP block's two input norms, the same organ class as every other
    # block norm.
    if rest in ("enorm.weight", "hnorm.weight"):
        return ORGAN_ATTN_NORM
    raise DeepSeekV4AdapterError(f"unrecognized block tensor suffix: {rest!r}")


# ── loading ─────────────────────────────────────────────────────────────────────────────
# The repo ships TWO configs with different spellings for the same geometry: the HF-style root
# config.json (model_type, architectures, quantization_config, num_hidden_layers, hidden_size) and
# inference/config.json, the reference implementation's own (n_layers, dim, dtype, scale_fmt,
# expert_dtype). Neither alone is sufficient: only the root names the architecture, only the
# inference one names the expert dtype. Merge them and fail loudly if they disagree.
_ALIASES = {"n_layers": "num_hidden_layers", "dim": "hidden_size",
            "n_activated_experts": "num_experts_per_tok", "n_heads": "num_attention_heads",
            "moe_inter_dim": "moe_intermediate_size", "n_hash_layers": "num_hash_layers"}
# NOT an alias pair: inference/config.json n_mtp_layers=3 counts stacked MTP BLOCKS, while root
# config.json num_nextn_predict_layers=1 counts tokens predicted per step. The tensor index
# arbitrates and agrees with the former: mtp.{N}.ffn.experts.* holds 2304 = 3 x 256 x 3 tensors,
# i.e. three MTP stacks. Both values are kept, neither is reconciled away.
_MTP_SEMANTICS = {"n_mtp_layers": "stacked MTP blocks (inference/config.json; index-confirmed)",
                  "num_nextn_predict_layers": "tokens predicted per step (root config.json)"}


def load_config(meta: Path = DEFAULT_META) -> dict[str, Any]:
    meta = Path(meta)
    root = json.loads((meta / "config.json").read_text())
    inf_path = meta / "inference_config.json"
    inf = json.loads(inf_path.read_text()) if inf_path.exists() else {}
    cfg: dict[str, Any] = {**root}
    for short, long in _ALIASES.items():
        a, b = inf.get(short), root.get(long)
        if a is not None and b is not None and int(a) != int(b):
            raise DeepSeekV4AdapterError(
                f"the two configs disagree on {short}/{long}: {a} vs {b}")
        cfg[short] = a if a is not None else b
    for k, v in inf.items():
        cfg.setdefault(k, v)
    # dtype/scale_fmt live only in the inference config; quantization_config only in the root.
    q = root.get("quantization_config") or {}
    cfg.setdefault("dtype", q.get("quant_method"))
    for k in ("n_layers", "n_routed_experts", "dim", "vocab_size"):
        if cfg.get(k) is None:
            raise DeepSeekV4AdapterError(f"config missing {k!r} in both config files")
    if root.get("model_type") and root["model_type"] != EXPECTED_MODEL_TYPE:
        raise DeepSeekV4AdapterError(
            f"model_type {root['model_type']!r} != {EXPECTED_MODEL_TYPE!r}")
    return cfg


def load_index(meta: Path = DEFAULT_META) -> dict[str, Any]:
    idx = json.loads((Path(meta) / "model.safetensors.index.json").read_text())
    if not isinstance(idx.get("weight_map"), dict):
        raise DeepSeekV4AdapterError("index missing weight_map")
    return idx


# ── inventory ───────────────────────────────────────────────────────────────────────────
def inventory(meta: Path = DEFAULT_META) -> dict[str, Any]:
    cfg, idx = load_config(meta), load_index(meta)
    wm = idx["weight_map"]
    total_size = int(idx.get("metadata", {}).get("total_size", 0))

    organs: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "weight_tensors": 0, "scale_tensors": 0, "layers": set(),
                 "shards": set()})
    shards: Counter = Counter()
    unknown: list[str] = []
    for name, shard in wm.items():
        try:
            c = classify(name)
        except DeepSeekV4AdapterError:
            unknown.append(name)
            continue
        o = organs[c["organ"]]
        o["tensors"] += 1
        o["scale_tensors" if c["is_scale"] else "weight_tensors"] += 1
        if c["layer"] is not None:
            o["layers"].add((c["stack"], c["layer"]))
        o["shards"].add(shard)
        shards[shard] += 1

    # Companion invariant: every quantised .weight in a scaled organ needs a matching .scale.
    weights = {n[: -len(".weight")] for n in wm if n.endswith(".weight")}
    scales = {n[: -len(".scale")] for n in wm if n.endswith(".scale")}
    unscaled_experts = sorted(
        s for s in weights - scales
        if ".ffn.experts." in s or ".ffn.shared_experts." in s)
    orphan_scales = sorted(scales - weights)

    n_layers, n_exp = int(cfg["n_layers"]), int(cfg["n_routed_experts"])
    expected_expert_weights = n_layers * n_exp * 3
    actual_expert_weights = organs[ORGAN_EXPERT]["weight_tensors"] - sum(
        1 for n in wm if n.startswith("mtp.") and ".ffn.experts." in n and n.endswith(".weight"))

    out = {
        "schema": "hawking.foundry.deepseek_v4_adapter.v1",
        "hf_id": HF_ID, "immutable_revision": IMMUTABLE_REVISION,
        "model_type": EXPECTED_MODEL_TYPE,
        "source_format": {"dtype": cfg.get("dtype"), "scale_fmt": cfg.get("scale_fmt"),
                          "expert_dtype": cfg.get("expert_dtype")},
        "geometry": {k: cfg.get(k) for k in
                     ("n_layers", "n_mtp_layers", "n_hash_layers", "dim", "vocab_size",
                      "n_routed_experts", "n_shared_experts", "n_activated_experts",
                      "moe_inter_dim", "n_heads", "head_dim", "q_lora_rank", "o_lora_rank",
                      "index_n_heads", "index_head_dim", "index_topk", "hc_mult",
                      "dspark_target_layer_ids", "dspark_markov_rank", "window_size")},
        "mtp_semantics": _MTP_SEMANTICS,
        "mtp_stacks_from_index": len({int(n.split(".")[1]) for n in wm if n.startswith("mtp.")}),
        "index_total_size_bytes": total_size,
        "tensor_count": len(wm),
        "shard_count": len(shards),
        "organs": {k: {"tensors": v["tensors"], "weight_tensors": v["weight_tensors"],
                       "scale_tensors": v["scale_tensors"],
                       "distinct_layers": len(v["layers"]), "shards_touched": len(v["shards"])}
                   for k, v in sorted(organs.items())},
        "compressible_organs": sorted(COMPRESSIBLE_ORGANS),
        "expert_grammar": r"(layers|mtp)\.{L}\.ffn\.experts\.{E}\.w(1|2|3)\.(weight|scale)",
        "router_grammar": r"layers\.{L}\.ffn\.gate\.(weight|bias)",
        "checks": {
            "unknown_tensor_names": unknown,
            "routed_expert_weights_expected": expected_expert_weights,
            "routed_expert_weights_found": actual_expert_weights,
            "routed_expert_grammar_ok": actual_expert_weights == expected_expert_weights,
            "experts_missing_scale": unscaled_experts,
            "orphan_scales": orphan_scales,
            "every_expert_weight_has_a_scale": not unscaled_experts,
        },
        "ledger_warnings": [
            "original_weight_count is NOT bound here: the index carries no shapes. Bind it from "
            "the resident safetensors headers at admission before any Fraction ledger is built.",
            f"the routed experts ship natively as {cfg.get('expert_dtype')!r} with "
            f"{cfg.get('scale_fmt')!r} scales; a 1.0 complete-BPW target is a ~4x reduction here, "
            f"not the ~16x a bf16 parent implies. BPW is NOT comparable to the sealed "
            f"Qwen3-235B bf16 numbers without that caveat.",
            "the .scale companions are real artifact bytes and must be billed, not treated as "
            "metadata.",
            "the MTP stack, markov_head and confidence_head must be explicitly INCLUDED or "
            "EXCLUDED from the ledger denominator, with the choice recorded.",
        ],
        "unbuilt_for_execution": [
            "no Rust route for deepseek_v4 (MLA + sparse indexer + DSpark compressor + "
            "hyper-connections); every claim is PYTHON REFERENCE FORWARD ONLY",
            "reference forward exists upstream at _meta/inference/model.py and is the honest "
            "starting point rather than a from-scratch reimplementation",
        ],
    }
    if unknown:
        raise DeepSeekV4AdapterError(
            f"{len(unknown)} tensor names did not classify, e.g. {unknown[:5]}")
    if not out["checks"]["routed_expert_grammar_ok"]:
        raise DeepSeekV4AdapterError(
            f"routed-expert grammar mismatch: expected {expected_expert_weights}, "
            f"found {actual_expert_weights}")
    return out


def shard_plan(meta: Path = DEFAULT_META, organ: str = ORGAN_EXPERT) -> dict[str, list[str]]:
    """Bounded-read plan: which shards hold an organ, and which tensors to take from each."""
    wm = load_index(meta)["weight_map"]
    plan: dict[str, list[str]] = defaultdict(list)
    for name, shard in wm.items():
        try:
            if classify(name)["organ"] == organ:
                plan[shard].append(name)
        except DeepSeekV4AdapterError:
            continue
    return {k: sorted(v) for k, v in sorted(plan.items())}


def routed_expert_pack_factor(name: str, shape: list[int], cfg: dict[str, Any]) -> int:
    """How many logical weights are packed into each stored element.

    MEASURED, not assumed. The routed experts of this checkpoint store two fp4 values per int8
    byte: `layers.0.ffn.experts.0.w1.weight` is I8 [2048, 2048] while hidden_size is 4096, and the
    SHARED expert's w1 is F8_E4M3 [2048, 4096] unpacked. Taking numel straight from the header
    would halve the routed-expert denominator and make every future BPW read twice as compressed
    as it truly is. Verified factor 2.0 on w1, w2 and w3 alike.
    """
    m = _RE_EXPERT.match(name)
    if not m or m.group(5) != "weight" or len(shape) != 2:
        return 1
    H = int(cfg["hidden_size"]) if "hidden_size" in cfg else int(cfg["dim"])
    I = int(cfg.get("moe_intermediate_size") or cfg["moe_inter_dim"])
    expected_in = I if m.group(4) == "2" else H     # w2 maps I->H; w1/w3 map H->I
    stored_in = shape[1]
    if stored_in == expected_in:
        return 1
    if stored_in * 2 == expected_in:
        return 2
    raise DeepSeekV4AdapterError(
        f"{name}: stored in-features {stored_in} is neither {expected_in} nor half of it; "
        f"the packing layout is not what this adapter was verified against")


def verify_against_source(model_dir: Path) -> dict[str, Any]:
    """Read the REAL safetensors headers once the shards are resident and reconcile byte-exactly."""
    model_dir = Path(model_dir)
    cfg = load_config(model_dir / "_meta") if (model_dir / "_meta" / "config.json").exists() \
        else json.loads((model_dir / "config.json").read_text())
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    total_size = int(idx.get("metadata", {}).get("total_size", 0))
    per_organ: Counter = Counter()
    per_organ_params: Counter = Counter()
    per_organ_stored: Counter = Counter()
    pack_factors: Counter = Counter()
    seen = 0
    for shard in sorted({*idx["weight_map"].values()}):
        p = model_dir / shard
        if not p.exists():
            return {"status": "PENDING_SOURCE", "missing_shard": shard}
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > MAX_HEADER_BYTES:
                raise DeepSeekV4AdapterError(f"{shard}: implausible header length {n}")
            header = json.loads(fh.read(n))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            begin, end = spec["data_offsets"]
            nbytes = end - begin
            organ = classify(name)["organ"]
            per_organ[organ] += nbytes
            numel = 1
            for d in spec["shape"]:
                numel *= d
            factor = routed_expert_pack_factor(name, spec["shape"], cfg)
            if factor != 1:
                pack_factors[f"{organ}:{spec['dtype']}"] = factor
            per_organ_stored[organ] += numel
            per_organ_params[organ] += numel * factor
            seen += nbytes
    owc_compressible = sum(per_organ_params[o] for o in COMPRESSIBLE_ORGANS)
    owc_all = sum(per_organ_params.values())
    return {
        "status": "GREEN" if seen == total_size else "RED",
        "bytes_seen": seen, "index_total_size": total_size,
        "byte_exact": seen == total_size,
        "bytes_by_organ": dict(per_organ.most_common()),
        "stored_elements_by_organ": dict(per_organ_stored.most_common()),
        "logical_weights_by_organ": dict(per_organ_params.most_common()),
        "pack_factors_detected": dict(pack_factors),
        "original_weight_count_compressible": owc_compressible,
        "original_weight_count_all": owc_all,
        "source_bits_per_weight_by_organ": {
            o: round(per_organ[o] * 8 / per_organ_params[o], 4)
            for o in per_organ if per_organ_params[o]},
        "denominator_warning":
            "original_weight_count uses LOGICAL weights, i.e. stored elements x the measured pack "
            "factor. The routed experts pack two fp4 values per int8 byte, so taking numel "
            "straight from the safetensors header would HALVE this denominator and make every "
            "candidate report roughly twice the compression it actually achieved.",
    }


# ── synthetic twin ──────────────────────────────────────────────────────────────────────
def synthetic_twin(n_layers: int = 2, n_experts: int = 4, n_mtp: int = 1) -> dict[str, Any]:
    """A tiny index with the SAME grammar, for exercising the adapter without the source."""
    wm: dict[str, str] = {}
    add = lambda n: wm.__setitem__(n, "model-00001-of-00001.safetensors")  # noqa: E731
    for n in ("embed.weight", "head.weight", "norm.weight", "hc_head_base", "hc_head_fn",
              "hc_head_scale"):
        add(n)
    for stack, count in (("layers", n_layers), ("mtp", n_mtp)):
        for L in range(count):
            for suf in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"):
                add(f"{stack}.{L}.{suf}.weight")
                add(f"{stack}.{L}.{suf}.scale")
            for suf in ("attn.q_norm.weight", "attn.kv_norm.weight", "attn_norm.weight",
                        "ffn_norm.weight"):
                add(f"{stack}.{L}.{suf}")
            add(f"{stack}.{L}.attn.attn_sink")
            for suf in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base",
                        "hc_ffn_fn", "hc_ffn_scale"):
                add(f"{stack}.{L}.{suf}")
            add(f"{stack}.{L}.ffn.gate.weight")
            add(f"{stack}.{L}.ffn.gate.bias")
            for suf in ("compressor.ape", "compressor.wkv.weight", "compressor.wgate.weight",
                        "compressor.norm.weight"):
                add(f"{stack}.{L}.attn.{suf}")
            for suf in ("indexer.wq_b.weight", "indexer.wq_b.scale", "indexer.weights_proj.weight",
                        "indexer.compressor.ape", "indexer.compressor.wkv.weight"):
                add(f"{stack}.{L}.attn.{suf}")
            for E in range(n_experts):
                for w in (1, 2, 3):
                    add(f"{stack}.{L}.ffn.experts.{E}.w{w}.weight")
                    add(f"{stack}.{L}.ffn.experts.{E}.w{w}.scale")
            for w in (1, 2, 3):
                add(f"{stack}.{L}.ffn.shared_experts.w{w}.weight")
                add(f"{stack}.{L}.ffn.shared_experts.w{w}.scale")
    return {"weight_map": wm, "metadata": {"total_size": 0}}


def self_check() -> None:
    twin = synthetic_twin()
    organs: Counter = Counter()
    for name in twin["weight_map"]:
        organs[classify(name)["organ"]] += 1
    # 2 real layers + 1 mtp layer, 4 experts, 3 projections, weight+scale each.
    assert organs[ORGAN_EXPERT] == 3 * 4 * 3 * 2, organs[ORGAN_EXPERT]
    assert organs[ORGAN_SHARED_EXPERT] == 3 * 3 * 2, organs[ORGAN_SHARED_EXPERT]
    assert organs[ORGAN_INDEXER] == 3 * 5, organs[ORGAN_INDEXER]
    assert organs[ORGAN_COMPRESSOR] == 3 * 4, organs[ORGAN_COMPRESSOR]
    assert organs[ORGAN_ROUTER] == 3 * 2, organs[ORGAN_ROUTER]
    assert organs[ORGAN_EMBED] == 1 and organs[ORGAN_HEAD] == 1

    # Compressor and indexer must NOT be swallowed by the generic MLA organ: the DSpark
    # compressor is the whole reason this parent is architecturally novel, and misfiling it
    # would silently spend expert bytes on attention.
    assert classify("layers.7.attn.compressor.wkv.weight")["organ"] == ORGAN_COMPRESSOR
    assert classify("layers.7.attn.indexer.compressor.wkv.weight")["organ"] == ORGAN_INDEXER
    assert classify("layers.7.attn.wkv.weight")["organ"] == ORGAN_ATTN_MLA

    # The fp4 pack factor must be MEASURED from the shape, and an unexpected layout must raise
    # rather than silently produce a wrong ledger denominator.
    c = {"hidden_size": 4096, "moe_intermediate_size": 2048}
    assert routed_expert_pack_factor("layers.0.ffn.experts.0.w1.weight", [2048, 2048], c) == 2
    assert routed_expert_pack_factor("layers.0.ffn.experts.0.w2.weight", [4096, 1024], c) == 2
    assert routed_expert_pack_factor("layers.0.ffn.experts.0.w3.weight", [2048, 2048], c) == 2
    assert routed_expert_pack_factor("layers.0.ffn.experts.0.w1.weight", [2048, 4096], c) == 1
    assert routed_expert_pack_factor("layers.0.ffn.shared_experts.w1.weight", [2048, 4096], c) == 1
    try:
        routed_expert_pack_factor("layers.0.ffn.experts.0.w1.weight", [2048, 999], c)
    except DeepSeekV4AdapterError:
        pass
    else:
        raise AssertionError("an unrecognised packing layout did not raise")

    # An unknown name must RAISE, never be silently bucketed.
    for bad in ("layers.3.attn.brand_new_thing", "totally.unknown", "layers.3.ffn.mystery"):
        try:
            classify(bad)
        except DeepSeekV4AdapterError:
            pass
        else:
            raise AssertionError(f"{bad!r} classified instead of raising")

    # Against the real index when it is present.
    if (DEFAULT_META / "model.safetensors.index.json").exists():
        inv = inventory()
        assert inv["checks"]["routed_expert_grammar_ok"], inv["checks"]
        assert inv["checks"]["every_expert_weight_has_a_scale"], \
            inv["checks"]["experts_missing_scale"][:5]
        assert not inv["checks"]["unknown_tensor_names"]
        assert inv["tensor_count"] == 72317, inv["tensor_count"]
        assert inv["index_total_size_bytes"] == 166878536440, inv["index_total_size_bytes"]
        print(f"self_check: OK (synthetic twin + real index: {inv['tensor_count']:,} tensors, "
              f"{inv['organs'][ORGAN_EXPERT]['weight_tensors']:,} routed-expert weights, "
              f"expert dtype {inv['source_format']['expert_dtype']})")
    else:
        print("self_check: OK (synthetic twin only; real index not resident)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["inventory", "shard-plan", "verify", "self-check"])
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--model-dir", default="models/deepseek-v4-flash-dspark")
    ap.add_argument("--organ", default=ORGAN_EXPERT)
    a = ap.parse_args()
    if a.cmd == "self-check":
        self_check()
        return 0
    if a.cmd == "inventory":
        print(json.dumps(inventory(Path(a.meta)), indent=2, sort_keys=True))
    elif a.cmd == "shard-plan":
        plan = shard_plan(Path(a.meta), a.organ)
        print(json.dumps({"organ": a.organ, "shards": len(plan),
                          "tensors": sum(len(v) for v in plan.values()),
                          "plan_head": {k: v[:3] for k, v in list(plan.items())[:3]}}, indent=2))
    else:
        print(json.dumps(verify_against_source(Path(a.model_dir)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
