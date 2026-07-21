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
    raise DeepSeekV4AdapterError(f"unrecognized block tensor suffix: {rest!r}")


# ── loading ─────────────────────────────────────────────────────────────────────────────
def load_config(meta: Path = DEFAULT_META) -> dict[str, Any]:
    cfg = json.loads((Path(meta) / "config.json").read_text())
    for k in ("n_layers", "n_routed_experts", "dim", "vocab_size"):
        if k not in cfg:
            raise DeepSeekV4AdapterError(f"config missing {k!r}")
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


def verify_against_source(model_dir: Path) -> dict[str, Any]:
    """Read the REAL safetensors headers once the shards are resident and reconcile byte-exactly."""
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    total_size = int(idx.get("metadata", {}).get("total_size", 0))
    per_organ: Counter = Counter()
    per_organ_params: Counter = Counter()
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
            per_organ_params[organ] += numel
            seen += nbytes
    return {
        "status": "GREEN" if seen == total_size else "RED",
        "bytes_seen": seen, "index_total_size": total_size,
        "byte_exact": seen == total_size,
        "bytes_by_organ": dict(per_organ.most_common()),
        "elements_by_organ": dict(per_organ_params.most_common()),
        "original_weight_count_compressible": sum(
            per_organ_params[o] for o in COMPRESSIBLE_ORGANS),
        "original_weight_count_all": sum(per_organ_params.values()),
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
