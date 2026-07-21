#!/usr/bin/env python3.12
"""Resident-first ladder: does a parent FIT IN FULL on the stripped disk, measured not assumed.

Storage law (declared by the campaign that commissioned this tool):

    MIN_RESERVE       = max(32 GiB, 2 x largest official shard, 3% of the writable volume)
    WORKING_HEADROOM  = max(16 GiB, one largest shard, projected compact-checkpoint bytes)

    FULL_RESIDENT_COMFORTABLE : source + WORKING_HEADROOM + 80 GiB <= free
    FULL_RESIDENT_SQUEEZED    : source + WORKING_HEADROOM + MIN_RESERVE <= free
    DOES_NOT_FIT_FULLY        : otherwise

Every source figure is a LIVE HfApi files_metadata total at a pinned revision. No nominal sizes,
no name-derived parameter counts used as a fit input. Parameter counts are labelled by origin.

Ordering law: every parent that fits fully runs before any parent that must stream, regardless of
parameter count.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]
OUT = ROOT / "reports/condense/storage_stripdown"
DATA_VOLUME = "/System/Volumes/Data"
GIB = 1 << 30

# Candidate field. Every entry is resolved LIVE; the local fields here are scheduling metadata
# (why we care) and the parameter count used only for the compact-checkpoint projection.
CANDIDATES = [
    # repo, total params (origin), active params, ladder slot, why
    ("deepseek-ai/DeepSeek-V4-Flash-DSpark", 284e9, 13e9, "procurement",
     "native fp8 e4m3 frontier MoE, 1M context, DeepSeek family; maximal distance from the "
     "just-sealed Qwen3-235B and the only frontier-class parent whose source is ALREADY sub-8-bit"),
    ("MiniMaxAI/MiniMax-M2", 230e9, 10e9, "off-ladder",
     "native fp8 MoE, distinct family; second native-low-bit parent"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", 80e9, 3e9, "off-ladder",
     "qwen3_next Gated-DeltaNet hybrid: the exact architecture the F2 397B needs an adapter for, "
     "at 1/5 the source bytes. Prerequisite value, not diversity value (Qwen-adjacent)"),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", 48e9, 3e9, "off-ladder",
     "Kimi Delta Attention linear-attention hybrid; cheapest distinct-architecture probe"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", 49e9, None, "off-ladder",
     "dense NAS-derived control; no MoE routing term at all"),
    ("ibm-granite/granite-4.0-h-small", 32e9, 9e9, "off-ladder",
     "hybrid Mamba-2 + MoE; state-space control"),
    ("openai/gpt-oss-20b", 21e9, 3.6e9, "off-ladder",
     "MXFP4 native small control; same family as the sealed F0"),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", 235e9, 22e9, "off-ladder",
     "same 94x128 top-8 text core as the sealed F1 plus a vision tower: LOW information gain, "
     "carried only to show the resident group is not size-ordered"),
    ("openai/gpt-oss-120b", 117e9, 5.1e9, "F0",
     "already measured and sealed NEGATIVE; carried as the completed reference"),
    ("moonshotai/Kimi-K2.6", 1070e9, 32e9, "F4",
     "the campaign's expected first candidate; verified rather than assumed"),
    ("moonshotai/Kimi-K2.7-Code", 1070e9, 32e9, "off-ladder", "K2.6 coding sibling"),
    ("deepseek-ai/DeepSeek-V3.2", 671e9, 37e9, "F3", "DeepSeek capstone"),
    ("deepseek-ai/DeepSeek-V4-Pro-DSpark", 1600e9, 49e9, "OPT_1_6T", "1.6T mega frontier"),
    ("Qwen/Qwen3.5-397B-A17B", 396234469376, 17e9, "F2",
     "the F-ladder's nominal successor edge from F1"),
    ("Qwen/Qwen3-Coder-480B-A35B-Instruct", 480e9, 35e9, "OPT_480B", "coder capstone"),
    ("zai-org/GLM-5.2", 753e9, 39e9, "off-ladder", "GLM capstone"),
    ("moonshotai/Kimi-K2-Instruct", 1000e9, 32e9, "off-ladder", "K2 text control"),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 400e9, 17e9, "off-ladder", "Llama-4 MoE (gated)"),
]

PARAM_ORIGIN = {
    "Qwen/Qwen3.5-397B-A17B": "analytic from config geometry (QWEN35_397B_ADAPTER_PLAN)",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, sort_keys=True, default=str)
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def free_bytes() -> int:
    st = os.statvfs(DATA_VOLUME)
    return st.f_bavail * st.f_frsize


def volume_bytes() -> int:
    st = os.statvfs(DATA_VOLUME)
    return st.f_blocks * st.f_frsize


def resolve_live(repo: str) -> dict:
    from huggingface_hub import HfApi
    api = HfApi()
    mi = api.model_info(repo, files_metadata=True)
    sibs = [s for s in mi.siblings if s.size is not None]
    weights = [s for s in sibs
               if s.rfilename.endswith((".safetensors", ".bin", ".gguf", ".pt"))]
    return {
        "repo": repo,
        "immutable_revision": mi.sha,
        "license": (mi.cardData or {}).get("license") if mi.cardData else None,
        "gated": bool(mi.gated),
        "last_modified": str(mi.lastModified),
        "n_files": len(mi.siblings),
        "n_weight_shards": len(weights),
        "source_bytes": sum(s.size for s in sibs),
        "weight_bytes": sum(s.size for s in weights),
        "largest_shard_bytes": max((s.size for s in weights), default=0),
        "has_tokenizer": any(s.rfilename.startswith("tokenizer") for s in mi.siblings),
        "has_index": any("index.json" in s.rfilename for s in mi.siblings),
        "has_config": any(s.rfilename == "config.json" for s in mi.siblings),
    }


def resolve_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
    except Exception as exc:  # noqa: BLE001
        return {"config_error": f"{type(exc).__name__}: {exc}"[:200]}
    text = cfg.get("text_config", cfg)
    q = cfg.get("quantization_config") or {}
    return {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "source_format": (f"{q.get('quant_method')}/{q.get('fmt') or q.get('activation_scheme') or ''}"
                          .strip("/") if q else (cfg.get("torch_dtype") or cfg.get("dtype") or "unknown")),
        "natively_sub_8_bit": bool(q) and str(q.get("quant_method", "")).lower() in
                              {"fp8", "mxfp4", "compressed-tensors", "awq", "gptq", "int4"},
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_experts": text.get("n_routed_experts") or text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "multimodal": "vision_config" in cfg or "ForConditionalGeneration" in
                      str(cfg.get("architectures", "")),
    }


def fit(source_bytes: int, largest_shard: int, total_params: float, free: int,
        volume: int) -> dict:
    min_reserve = max(32 * GIB, 2 * largest_shard, int(0.03 * volume))
    # A complete artifact at the campaign's 1/1 BPW ceiling is one bit per original weight.
    compact_ckpt = int(total_params / 8) if total_params else 0
    working = max(16 * GIB, largest_shard, compact_ckpt)
    comfortable = source_bytes + working + 80 * GIB
    squeezed = source_bytes + working + min_reserve
    if comfortable <= free:
        klass = "FULL_RESIDENT_COMFORTABLE"
    elif squeezed <= free:
        klass = "FULL_RESIDENT_SQUEEZED"
    else:
        klass = "DOES_NOT_FIT_FULLY"
    return {
        "min_reserve_bytes": min_reserve,
        "working_headroom_bytes": working,
        "projected_compact_checkpoint_bytes": compact_ckpt,
        "required_comfortable_bytes": comfortable,
        "required_squeezed_bytes": squeezed,
        "free_bytes_assumed": free,
        "fit_class": klass,
        "margin_bytes": free - squeezed,
    }


def priority(row: dict) -> tuple:
    """Scientific priority within the full-resident group. Higher score sorts first."""
    c = row.get("config", {})
    slot = row["ladder_slot"]
    score = 0
    if c.get("natively_sub_8_bit"):
        score += 40   # a natively sub-8-bit parent asks the sub-bit question at its sharpest
    fam = row["repo"].split("/")[0].lower()
    if "qwen" not in fam:
        score += 25   # distance from the just-finished Qwen family
    if c.get("num_experts"):
        score += 10   # MoE keeps the organ/expert machinery in scope
    if (c.get("max_position_embeddings") or 0) >= 1 << 20:
        score += 5
    params = row.get("total_params") or 0
    if row["source_bytes"]:
        score += min(20, int(params / row["source_bytes"] * 10))  # value per source byte
    if slot == "F0":
        score = -100  # already sealed
    if row["repo"] == "Qwen/Qwen3-VL-235B-A22B-Instruct":
        score -= 30   # same text core as the sealed F1: near-zero information gain
    if row.get("gated"):
        score -= 50
    return (score, -row["source_bytes"])


def build(free_override: int | None, comment: str) -> dict:
    volume = volume_bytes()
    free = free_override if free_override is not None else free_bytes()
    rows = []
    for repo, total_p, active_p, slot, why in CANDIDATES:
        try:
            live = resolve_live(repo)
        except Exception as exc:  # noqa: BLE001
            rows.append({"repo": repo, "ladder_slot": slot, "why": why,
                         "resolution_error": f"{type(exc).__name__}: {exc}"[:200],
                         "fit_class": "PENDING_OFFICIAL_SOURCE"})
            continue
        cfg = resolve_config(repo) if live["has_config"] else {}
        row = {**live, "ladder_slot": slot, "why": why, "config": cfg,
               "total_params": total_p, "active_params": active_p,
               "total_params_origin": PARAM_ORIGIN.get(repo, "nominal from repo name / model card"),
               **fit(live["source_bytes"], live["largest_shard_bytes"], total_p, free, volume)}
        rows.append(row)

    resident = [r for r in rows if r.get("fit_class", "").startswith("FULL_RESIDENT")]
    streaming = [r for r in rows if r.get("fit_class") == "DOES_NOT_FIT_FULLY"]
    pending = [r for r in rows if r.get("fit_class") == "PENDING_OFFICIAL_SOURCE"]
    resident.sort(key=priority, reverse=True)
    streaming.sort(key=lambda r: r["source_bytes"])

    order = 0
    for r in resident:
        order += 1
        r["selected_order"] = order
    for r in streaming:
        order += 1
        r["selected_order"] = order
        r["reason"] = "streaming-only: moved behind every full-resident parent by the ordering law"

    selected = next((r for r in resident if r["ladder_slot"] != "F0"), None)
    out = {
        "schema": "hawking.resident_first_ladder.v1",
        "generated_at": now(),
        "comment": comment,
        "law": {
            "min_reserve": "max(32 GiB, 2 x largest shard, 3% of volume)",
            "working_headroom": "max(16 GiB, largest shard, projected compact checkpoint)",
            "compact_checkpoint_projection": "total_params / 8 bytes, i.e. one complete bit per "
                                             "original weight - the campaign's 1/1 BPW ceiling",
            "ordering": "every FULL_RESIDENT_* parent runs before any DOES_NOT_FIT_FULLY parent, "
                        "regardless of parameter count",
        },
        "volume_total_bytes": volume,
        "free_bytes_used_for_fit": free,
        "free_bytes_measured_now": free_bytes(),
        "counts": {"full_resident_comfortable":
                   sum(1 for r in rows if r.get("fit_class") == "FULL_RESIDENT_COMFORTABLE"),
                   "full_resident_squeezed":
                   sum(1 for r in rows if r.get("fit_class") == "FULL_RESIDENT_SQUEEZED"),
                   "does_not_fit_fully": len(streaming),
                   "pending_official_source": len(pending)},
        "selected_next_parent": selected["repo"] if selected else None,
        "selected_revision": selected["immutable_revision"] if selected else None,
        "resident_frontier_exhausted": selected is None,
        "rows": resident + streaming + pending,
        "honesty": [
            "Every source_bytes figure is a live HfApi files_metadata sum at the pinned sha, not "
            "a nominal or model-card size.",
            "total_params is used ONLY to project the compact-checkpoint headroom term. It is "
            "name-derived for most rows and must not be used as a ledger denominator; bind "
            "original_weight_count from the resident tensor index at admission.",
            "free_bytes_used_for_fit may be a PROJECTED post-cleanup figure. The fit class is "
            "re-computed against measured free space before any download starts.",
            "Fit class is a STORAGE verdict. It says nothing about whether the parent is "
            "compressible, and nothing about whether the Rust engine can execute the artifact.",
        ],
    }
    write(OUT / "FULL_RESIDENT_ELIGIBILITY.json", out)
    write(OUT / "HAWKING_FULL_RESIDENT_FIRST_LADDER.json", out)
    write(OUT / "HAWKING_FULL_RESIDENT_FIRST_LADDER.md", render(out))
    return out


def _g(n) -> str:
    return f"{(n or 0) / GIB:,.1f}"


def render(out: dict) -> str:
    L = ["# Hawking resident-first ladder", "",
         f"generated {out['generated_at']}", "",
         f"- volume: {_g(out['volume_total_bytes'])} GiB",
         f"- free used for fit: {_g(out['free_bytes_used_for_fit'])} GiB "
         f"({out['comment']})",
         f"- free measured now: {_g(out['free_bytes_measured_now'])} GiB",
         f"- selected next parent: **{out['selected_next_parent']}**", "",
         "Ordering law: every parent that fits fully runs before any parent that must stream, "
         "regardless of parameter count.", "",
         "| # | parent | slot | params | active | source GiB | largest shard | reserve | "
         "headroom | fit class | margin GiB |",
         "|--:|---|---|--:|--:|--:|--:|--:|--:|---|--:|"]
    for r in out["rows"]:
        if "source_bytes" not in r:
            L.append(f"| - | {r['repo']} | {r['ladder_slot']} | | | | | | | "
                     f"PENDING_OFFICIAL_SOURCE | |")
            continue
        L.append(
            f"| {r['selected_order']} | {r['repo']} | {r['ladder_slot']} | "
            f"{(r['total_params'] or 0)/1e9:.0f}B | "
            f"{(r['active_params'] or 0)/1e9:.0f}B | {_g(r['source_bytes'])} | "
            f"{_g(r['largest_shard_bytes'])} | {_g(r['min_reserve_bytes'])} | "
            f"{_g(r['working_headroom_bytes'])} | {r['fit_class']} | "
            f"{_g(r['margin_bytes'])} |")
    L += ["", "## Why the leader leads", ""]
    for r in out["rows"][:4]:
        if "source_bytes" in r:
            c = r.get("config", {})
            L.append(f"- **{r['repo']}** ({c.get('model_type')}, {c.get('source_format')}, "
                     f"{c.get('num_hidden_layers')} layers, {c.get('num_experts')} experts, "
                     f"top-{c.get('num_experts_per_tok')}): {r['why']}")
    L += ["", "## Honesty", ""] + [f"- {h}" for h in out["honesty"]] + [""]
    return "\n".join(L)


def self_check() -> None:
    v = 926 * GIB
    # A parent whose source alone exceeds the volume can never be resident.
    r = fit(800 * GIB, 9 * GIB, 1.07e12, 600 * GIB, v)
    assert r["fit_class"] == "DOES_NOT_FIT_FULLY", r
    # Zero operational room must never read as resident.
    r = fit(600 * GIB, 9 * GIB, 1.07e12, 600 * GIB, v)
    assert r["fit_class"] == "DOES_NOT_FIT_FULLY", "a zero-headroom fit was called resident"
    # A small parent with room to work is comfortable.
    r = fit(155 * GIB, 4 * GIB, 284e9, 560 * GIB, v)
    assert r["fit_class"] == "FULL_RESIDENT_COMFORTABLE", r
    # The squeezed band exists and sits between the two.
    r = fit(430 * GIB, 4 * GIB, 284e9, 500 * GIB, v)
    assert r["fit_class"] == "FULL_RESIDENT_SQUEEZED", r
    print("self_check: OK (no-fit, zero-headroom, comfortable and squeezed all classify correctly)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--free-gib", type=float,
                    help="projected post-cleanup free space to classify against")
    ap.add_argument("--comment", default="measured live")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
        return 0
    out = build(int(a.free_gib * GIB) if a.free_gib else None, a.comment)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
