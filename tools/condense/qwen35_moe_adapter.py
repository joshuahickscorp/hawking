#!/usr/bin/env python3.12
"""Qwen3.5-397B-A17B organ adapter (qwen3_5_moe / Qwen3_5MoeForConditionalGeneration).

This parent is the amplification challenger, chosen because its residual stream is
structurally unlike GLM-5.2's: 45 of 60 text layers are mamba-style linear attention and
only 15 are full attention, on a period of 4. If EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE is a
fact about deep MoE transformers rather than about GLM specifically, a hybrid recurrent
stack is where that would show.

Metadata only. This reads the real safetensors index and classifies every tensor into an
organ, so the contraction-first pilot knows which shards a bounded strata probe must fetch.
It downloads and loads nothing; the 806.8 GB body is never resident under the storage
policy, so this is the whole of the Qwen preparation that can run alongside a DeepSeek
fetch.

The linear-attention organ is genuinely new: A_log, conv1d, dt_bias and the split input
projections have no counterpart in the GLM or DeepSeek adapters, which is exactly why the
readiness gate recorded no qwen3_5_moe adapter existed.

    inventory   organ -> {tensors, templates, shards} from the real index
    windows     bounded per-stratum shard plan for the amplification probe
    self-check
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = "Qwen/Qwen3.5-397B-A17B"
REVISION = "8472618112abcbd45acbcdc58436aff4233c23f7"

SUPPORT = Path(os.environ.get(
    "QWEN35_SUPPORT_ROOT",
    str(Path.home() / "Library/Application Support/Hawking/Qwen35_397B")))
DEFAULT_META = SUPPORT / "meta"

# Text-stack organs. linear_attn and self_attn are separate organs on purpose: the pilot
# fits a functional student against the MoE output, but the amplification probe runs
# through whichever attention the layer actually uses, and those are different operators.
ORGAN_EMBED = "embed"
ORGAN_NORM = "norm"
ORGAN_HEAD = "lm_head"
ORGAN_LINEAR_ATTN = "attn.linear"          # mamba-style recurrent mixer
ORGAN_FULL_ATTN = "attn.full"              # standard softmax attention
ORGAN_ATTN_NORM = "attn.norm"
ORGAN_ROUTER = "moe.gate"
ORGAN_EXPERT = "moe.experts"               # fused gate_up_proj + down_proj (stacked)
ORGAN_SHARED_EXPERT = "moe.shared_expert"
ORGAN_VISION = "vision"
ORGAN_MTP = "mtp"

COMPRESSIBLE_ORGANS = {ORGAN_EXPERT, ORGAN_SHARED_EXPERT}

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_MTP_LAYER = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")


class Qwen35AdapterError(RuntimeError):
    pass


def _text_block_organ(rest: str) -> str:
    if rest.startswith("linear_attn."):
        return ORGAN_LINEAR_ATTN
    if rest.startswith("self_attn."):
        head = rest.split(".")[1]
        return ORGAN_ATTN_NORM if head in ("q_norm", "k_norm") else ORGAN_FULL_ATTN
    if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return ORGAN_ATTN_NORM
    if rest.startswith("mlp.gate.") or rest == "mlp.gate.weight":
        return ORGAN_ROUTER
    if rest.startswith("mlp.experts."):
        return ORGAN_EXPERT
    if rest.startswith("mlp.shared_expert"):
        return ORGAN_SHARED_EXPERT
    raise Qwen35AdapterError(f"unrecognized text block suffix: {rest!r}")


def classify(name: str) -> dict[str, Any]:
    if name == "model.language_model.embed_tokens.weight":
        return {"organ": ORGAN_EMBED, "stack": "text", "layer": None}
    if name == "model.language_model.norm.weight":
        return {"organ": ORGAN_NORM, "stack": "text", "layer": None}
    if name == "lm_head.weight":
        return {"organ": ORGAN_HEAD, "stack": "top", "layer": None}
    if name.startswith("model.visual."):
        return {"organ": ORGAN_VISION, "stack": "vision", "layer": None}
    m = _LAYER.match(name)
    if m:
        return {"organ": _text_block_organ(m.group(2)), "stack": "text",
                "layer": int(m.group(1))}
    m = _MTP_LAYER.match(name)
    if m:
        rest = m.group(2)
        try:
            organ = _text_block_organ(rest)
        except Qwen35AdapterError:
            organ = ORGAN_MTP
        return {"organ": ORGAN_MTP if organ in (ORGAN_MTP,) else organ,
                "stack": "mtp", "layer": int(m.group(1))}
    if name.startswith("mtp."):
        return {"organ": ORGAN_MTP, "stack": "mtp", "layer": None}
    raise Qwen35AdapterError(f"unrecognized tensor name: {name!r}")


def load_index(meta: Path = DEFAULT_META) -> dict:
    return json.loads((Path(meta) / "model.safetensors.index.json").read_text())


def _layer_type_map(meta: Path) -> dict[int, str]:
    """From config text_config.layer_types if present, else empty."""
    config = Path(meta) / "config.json"
    if not config.exists():
        return {}
    cfg = json.loads(config.read_text())
    text = cfg.get("text_config", cfg)
    types = text.get("layer_types") or []
    return {i: t for i, t in enumerate(types)}


def inventory(meta: Path = DEFAULT_META) -> dict:
    idx = load_index(meta)
    wm = idx["weight_map"]
    organs: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "templates": set(), "shards": set(), "layers": set(),
                 "stacks": set()})
    unknown = []
    for name, shard in wm.items():
        try:
            c = classify(name)
        except Qwen35AdapterError:
            unknown.append(name)
            continue
        o = organs[c["organ"]]
        o["tensors"] += 1
        o["templates"].add(re.sub(r"\.\d+\.", ".N.", name))
        o["shards"].add(shard)
        o["stacks"].add(c["stack"])
        if c["layer"] is not None:
            o["layers"].add(c["layer"])
    if unknown:
        raise Qwen35AdapterError(f"{len(unknown)} tensors did not classify, "
                                 f"e.g. {unknown[:4]}")
    layer_types = _layer_type_map(meta)
    return {
        "schema": "hawking.qwen35_moe.inventory.v1",
        "repo": REPO, "revision": REVISION,
        "total_tensors": len(wm),
        "total_size_gb": round(int(idx.get("metadata", {}).get("total_size", 0)) / 1e9, 1),
        "layer_types_histogram": {
            t: sum(1 for v in layer_types.values() if v == t)
            for t in sorted(set(layer_types.values()))},
        "organs": {name: {"tensors": o["tensors"],
                          "templates": sorted(o["templates"]),
                          "shard_count": len(o["shards"]),
                          "stacks": sorted(o["stacks"]),
                          "layer_count": len(o["layers"]),
                          "compressible": name in COMPRESSIBLE_ORGANS}
                   for name, o in sorted(organs.items())},
        "linear_attention_is_new_organ": ORGAN_LINEAR_ATTN in organs,
    }


def windows(meta: Path = DEFAULT_META) -> dict:
    """Bounded strata for the amplification probe, mapped to the shards they need.

    The probe needs a stratum layer and its immediate successor, at three depths, plus the
    head organs for a logit lens. It deliberately spans both attention types: an early
    linear-attention layer, a middle boundary where a full-attention layer sits, and a late
    linear-attention layer, so the amplification measurement covers the hybrid stack rather
    than one operator.
    """
    idx = load_index(meta)
    wm = idx["weight_map"]
    layer_types = _layer_type_map(meta)
    num_layers = (max((classify(n)["layer"] for n in wm
                       if classify(n)["stack"] == "text"
                       and classify(n)["layer"] is not None), default=59) + 1)

    def shards_for_layer(layer: int) -> set[str]:
        prefix = f"model.language_model.layers.{layer}."
        return {shard for name, shard in wm.items() if name.startswith(prefix)}

    # full_attention_interval is 4, so full-attention layers are the ones divisible by it;
    # pick a middle stratum that lands on one to exercise that operator.
    full_layers = [i for i, t in layer_types.items() if t == "full_attention"]
    linear_layers = [i for i, t in layer_types.items() if t == "linear_attention"]
    early = linear_layers[1] if len(linear_layers) > 1 else 1
    middle = full_layers[len(full_layers) // 2] if full_layers else num_layers // 2
    late = linear_layers[-3] if len(linear_layers) > 2 else num_layers - 3

    strata = {}
    for label, layer in (("early_linear", early), ("middle_full", middle),
                         ("late_linear", late)):
        partner = layer + 1
        needed = shards_for_layer(layer) | shards_for_layer(partner)
        strata[label] = {
            "stratum_layer": layer,
            "layer_type": layer_types.get(layer, "unknown"),
            "successor_layer": partner,
            "successor_type": layer_types.get(partner, "unknown"),
            "shards": sorted(needed),
            "shard_count": len(needed),
        }
    head_shards = sorted({shard for name, shard in wm.items()
                          if name in ("model.language_model.embed_tokens.weight",
                                      "model.language_model.norm.weight",
                                      "lm_head.weight")})
    unique = sorted({s for w in strata.values() for s in w["shards"]} | set(head_shards))
    return {
        "schema": "hawking.qwen35_moe.window_plan.v1",
        "repo": REPO, "revision": REVISION,
        "policy": "PARTIAL_ONLY; the 806.8 GB body is 6.2x free disk and is never resident",
        "num_text_layers": num_layers,
        "strata": strata,
        "head_shards": head_shards,
        "unique_shards_to_fetch": unique,
        "unique_shard_count": len(unique),
        "of_total_shards": 94,
        "note": "actual partial fetch is deferred: the heavy lane is the DeepSeek complete "
                "download, and the amplification probe needs a qwen3_5_moe reference forward "
                "(linear-attention recurrence + full attention + fused-expert MoE) that does "
                "not exist yet. This plan is what that fetch will pull when its turn comes.",
    }


def self_check(meta: Path = DEFAULT_META) -> dict:
    inv = inventory(meta)
    win = windows(meta)
    assert inv["linear_attention_is_new_organ"], "linear attention organ missing"
    assert ORGAN_EXPERT in inv["organs"], "expert organ missing"
    # The window plan must be a strict subset of the model and must span both operators.
    assert win["unique_shard_count"] < win["of_total_shards"], "window plan is not bounded"
    types = {w["layer_type"] for w in win["strata"].values()}
    assert "linear_attention" in types and "full_attention" in types, types
    print(json.dumps({
        "self_check": "OK",
        "total_tensors": inv["total_tensors"],
        "organs": len(inv["organs"]),
        "linear_attention_tensors": inv["organs"][ORGAN_LINEAR_ATTN]["tensors"],
        "full_attention_tensors": inv["organs"][ORGAN_FULL_ATTN]["tensors"],
        "layer_types": inv["layer_types_histogram"],
        "window_shards": f"{win['unique_shard_count']}/{win['of_total_shards']}",
    }))
    return {"inventory": inv, "windows": win}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "self-check"
    meta = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_META
    if command == "inventory":
        print(json.dumps(inventory(meta), indent=2, default=list))
    elif command == "windows":
        print(json.dumps(windows(meta), indent=2))
    elif command == "self-check":
        self_check(meta)
    else:
        raise SystemExit(f"unknown command: {command}")
