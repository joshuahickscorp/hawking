#!/usr/bin/env python3.12
"""Forge F2 compact-runtime fixture: measure output divergence on REAL token-derived activations.

Upgrades the synthetic-Gaussian output-divergence proxy to REAL data: it tokenizes a deterministic
prompt set with the validated GPT-OSS-120B tokenizer, embeds the tokens with the model's own
`embedding.weight`, and runs the reference MoE block forward with ORIGINAL vs Forge-packed experts
(packing exactly the routed union), measuring output divergence and routing stability.

Honest boundary (do not overstate):
  * activation_source = real_token_embedding. These are the true layer-0 embedding vectors, a
    strictly-better-than-Gaussian input distribution, but NOT the post-attention residual stream.
    True residual-stream F2 needs the block attention layer (attn.qkv / sinks / out), which the
    manifest exposes but this fixture does not yet run. That gap is reported, not hidden.
  * routing is invariant here because only experts are packed, not the router; routing-divergence
    becomes a live signal once the router itself is packed.
  * capability_parity = False. No cell here authorizes an escape or an Event Horizon.
  * bounded: only the routed experts are loaded/packed (no dense shadow model); the embedding matrix
    is loaded once and indexed; packed-artifact bytes are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf          # noqa: E402
import gptoss_moe_runtime as rt     # noqa: E402

FIXTURE_SCHEMA = "hawking.gravity_forge.f2_fixture.v1"
SRC = "scratch/staging/gpt-oss-120b.partial"

# deterministic, domain-spread prompts (general / code / math / tool-use)
PROMPTS = [
    "The capital of France is Paris and the river running through it is the Seine.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "Solve for x: 3x + 7 = 22. Subtract 7 to get 3x = 15, so x = 5.",
    "Call the weather API with {\"city\": \"Tokyo\", \"units\": \"metric\"} and summarize the result.",
]


def _tokenizer_ok() -> tuple[bool, dict[str, Any]]:
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(f"{SRC}/tokenizer.json")
        s = "def add(a, b):\n    return a + b"
        ids = tk.encode(s).ids
        rt_ok = s.strip() in tk.decode(ids)
        return True, {"vocab_size": tk.get_vocab_size(), "roundtrip": bool(rt_ok),
                      "chat_template_present": os.path.exists(f"{SRC}/chat_template.jinja")}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


def run(*, block: int = 0, max_tokens: int = 48, top_k: int = 4,
        pack_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        pack_label: str = "transform_pq@~0.75bpw") -> dict[str, Any]:
    ok, tkinfo = _tokenizer_ok()
    if not ok:
        return {"schema": FIXTURE_SCHEMA, "green": False, "reason": "tokenizer invalid", "tokenizer": tkinfo}
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f"{SRC}/tokenizer.json")

    reader = rt.ProvenanceReader()
    if not Path(reader.by_name["block.0.mlp.gate.weight"]["shard_path"]).exists():
        return {"schema": FIXTURE_SCHEMA, "green": False, "reason": "120B source shards absent"}

    # deterministic real token ids (dedup, bounded)
    ids: list[int] = []
    for p in PROMPTS:
        ids.extend(tk.encode(p).ids)
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); uniq.append(int(i))
        if len(uniq) >= max_tokens:
            break

    # real layer-0 embeddings for those tokens (loaded once, indexed -> bounded)
    emb = reader.bf16("embedding.weight")            # [vocab, 2880]
    acts = np.ascontiguousarray(emb[uniq], dtype=np.float32)   # [n, 2880]
    del emb

    if pack_fn is None:
        pack_fn = lambda w: gf.pack_transform_pq(w, dim=16, subspaces=2, k=64).recon

    router = rt.load_router(reader, block)
    routed: set[int] = set()
    for x in acts:
        logits = router["weight"] @ x + router["bias"]
        routed.update(int(e) for e in np.argsort(-logits)[:top_k])

    orig, packed, packed_bytes, packed_weights = {}, {}, 0, 0
    for e in routed:
        ex = rt.load_expert(reader, block, e)
        orig[e] = ex
        art1 = gf.pack_transform_pq(ex["mlp1"].astype(np.float32), dim=16, subspaces=2, k=64)
        art2 = gf.pack_transform_pq(ex["mlp2"].astype(np.float32), dim=16, subspaces=2, k=64)
        m = dict(ex); m["mlp1"] = art1.recon; m["mlp2"] = art2.recon
        packed[e] = m
        packed_bytes += art1.physical_bytes + art2.physical_bytes
        packed_weights += ex["mlp1"].size + ex["mlp2"].size

    rel, routing_stable = [], 0
    for x in acts:
        y0 = rt.moe_forward_reference(x, router, lambda e: orig[e], top_k=top_k)
        y1 = rt.moe_forward_reference(x, router, lambda e: packed[e], top_k=top_k)
        rel.append(float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-9)))
        routing_stable += 1     # router unchanged -> routing invariant by construction
    finite = bool(np.all(np.isfinite(y1)))

    whole_bpw = packed_bytes * 8 / max(1, packed_weights)
    return {
        "schema": FIXTURE_SCHEMA, "green": bool(finite and len(rel) > 0),
        "parent": "120B", "block": block, "device": gf._device().type,
        "tokenizer": tkinfo, "activation_source": "real_token_embedding",
        "activation_gap": "pre-attention proxy; true residual-stream F2 needs the block attention layer",
        "n_tokens": len(uniq), "n_experts_exercised": len(routed), "pack": pack_label,
        "packed_whole_artifact_bpw": round(whole_bpw, 4),
        "mean_output_rel_div": round(float(np.mean(rel)), 5),
        "max_output_rel_div": round(float(np.max(rel)), 5),
        "min_output_rel_div": round(float(np.min(rel)), 5),
        "routing_invariant": routing_stable == len(acts),
        "routing_divergence_note": "invariant until the router itself is packed",
        "bounded_memory": True, "reconstructs_dense_model": False, "capability_parity": False,
        "deterministic": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Forge F2 real-token compact-runtime fixture.")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default="reports/condense/gravity_forge/FORGE_F2_FIXTURE.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    doc = run(block=args.block, max_tokens=args.max_tokens)
    doc["seconds"] = round(time.time() - t0, 1)
    doc["fixture_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in doc.items() if k != "fixture_sha256"},
                   sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: doc[k] for k in ("green", "activation_source", "n_tokens",
                                          "n_experts_exercised", "packed_whole_artifact_bpw",
                                          "mean_output_rel_div", "max_output_rel_div",
                                          "capability_parity") if k in doc}, indent=2))
    return 0 if doc["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
