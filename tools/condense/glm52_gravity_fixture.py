#!/usr/bin/env python3.12
"""Build a tiny GLM-5.2-shaped `.gravity` artifact and freeze the logits it encodes.

The Rust GLM adapter has to be graded on MLA, DSA, IndexShare, the noaux_tc router and
the routed/shared expert split before the 1.5 TB flagship traversal finishes. A tiny
model with the flagship's exact *semantics* -- same layer schedule shape, same interleaved
RoPE, same grouped router, same IndexShare reuse -- grades all of that in a second, and
the flagship then only has to prove scale, not correctness.

Every dimension here is small; none of the semantics are. The layer schedule deliberately
contains a dense layer, a full-indexer sparse layer, a shared-indexer sparse layer that
must reuse the previous full layer's top-k, and one more full layer, so a runtime that
ignores IndexShare cannot pass.

The oracle reads the artifact back through the container and the same codec, so it grades
what the artifact encodes rather than the float32 weights that were packed into it -- the
bf16 rounding is part of what is being reproduced, not an error term.

    python3.12 tools/condense/glm52_gravity_fixture.py OUT_DIR
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack as pack  # noqa: E402
import glm52_reference as ref  # noqa: E402
import gravity_format as gravity  # noqa: E402
import gravity_forge as forge  # noqa: E402
from glm52_gravity_source import GravityGlmSource  # noqa: E402

# Small enough to grade in a second, large enough that every code path is real.
CONFIG = {
    "model_type": "glm_moe_dsa",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "index_n_heads": 4,
    "index_head_dim": 16,
    "index_topk": 8,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "n_group": 2,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "moe_intermediate_size": 16,
    "intermediate_size": 128,
    "first_k_dense_replace": 1,
    "vocab_size": 4096,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000.0},
    # Layer 2 is `shared`: it must reuse layer 1's top-k. A runtime that
    # recomputes an index there gets different attention and fails.
    "indexer_types": ["full", "full", "shared", "full"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
}
TOKENS = [7, 1234, 9]
SEED = 20260724


def _weights(config: dict) -> dict[str, np.ndarray]:
    """Deterministic pseudo-random weights, scaled so activations stay in range."""
    rng = np.random.default_rng(SEED)
    h = config["hidden_size"]
    heads = config["num_attention_heads"]
    qk = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    v = config["v_head_dim"]
    qr, kr = config["q_lora_rank"], config["kv_lora_rank"]
    ih, idim = config["index_n_heads"], config["index_head_dim"]

    def w(*shape: int) -> np.ndarray:
        # 1/sqrt(fan_in) keeps a 4-layer stack from saturating, which would
        # make every logit comparison trivially pass.
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return (1.0 + 0.02 * rng.standard_normal(n)).astype(np.float32)

    t: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": w(config["vocab_size"], h),
        "model.norm.weight": norm(h),
        "lm_head.weight": w(config["vocab_size"], h),
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        t[f"{p}.input_layernorm.weight"] = norm(h)
        t[f"{p}.post_attention_layernorm.weight"] = norm(h)
        a = f"{p}.self_attn"
        t[f"{a}.q_a_proj.weight"] = w(qr, h)
        t[f"{a}.q_a_layernorm.weight"] = norm(qr)
        t[f"{a}.q_b_proj.weight"] = w(heads * qk, qr)
        t[f"{a}.kv_a_proj_with_mqa.weight"] = w(kr + config["qk_rope_head_dim"], h)
        t[f"{a}.kv_a_layernorm.weight"] = norm(kr)
        t[f"{a}.kv_b_proj.weight"] = w(heads * (config["qk_nope_head_dim"] + v), kr)
        t[f"{a}.o_proj.weight"] = w(h, heads * v)
        if config["indexer_types"][layer] == "full":
            i = f"{a}.indexer"
            t[f"{i}.wq_b.weight"] = w(ih * idim, qr)
            t[f"{i}.wk.weight"] = w(idim, h)
            t[f"{i}.k_norm.weight"] = norm(idim)
            t[f"{i}.k_norm.bias"] = (0.01 * rng.standard_normal(idim)).astype(np.float32)
            t[f"{i}.weights_proj.weight"] = w(ih, h)
        if config["mlp_layer_types"][layer] == "dense":
            m = f"{p}.mlp"
            t[f"{m}.gate_proj.weight"] = w(config["intermediate_size"], h)
            t[f"{m}.up_proj.weight"] = w(config["intermediate_size"], h)
            t[f"{m}.down_proj.weight"] = w(h, config["intermediate_size"])
        else:
            m = f"{p}.mlp"
            t[f"{m}.gate.weight"] = w(config["n_routed_experts"], h)
            t[f"{m}.gate.e_score_correction_bias"] = (
                0.1 * rng.standard_normal(config["n_routed_experts"])
            ).astype(np.float32)
            mi = config["moe_intermediate_size"]
            for e in range(config["n_routed_experts"]):
                s = f"{m}.experts.{e}"
                t[f"{s}.gate_proj.weight"] = w(mi, h)
                t[f"{s}.up_proj.weight"] = w(mi, h)
                t[f"{s}.down_proj.weight"] = w(h, mi)
            s = f"{m}.shared_experts"
            t[f"{s}.gate_proj.weight"] = w(mi, h)
            t[f"{s}.up_proj.weight"] = w(mi, h)
            t[f"{s}.down_proj.weight"] = w(h, mi)
    return t


def _bf16_bytes(a: np.ndarray) -> bytes:
    """Round-to-nearest-even f32 -> bf16, matching what a real GLM checkpoint carries."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes()


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "glm52-tiny-R0.gravity"
    rung = next(r for r in pack.LADDER if r["rung"] == "R0")
    tensors = _weights(CONFIG)

    payloads: list[tuple[dict, bytes]] = []
    pq_bits = native_bits = pq_weights = native_weights = 0
    for name in sorted(tensors):
        arr = tensors[name]
        shape = list(arr.shape)
        elements = int(arr.size)
        # Same admissibility rule as the flagship pipeline, so both bill alike:
        # a codebook costs the same on a 4x64 tensor as on a routed expert.
        if len(shape) == 2 and shape[1] % rung["dim"] == 0 and pack.rung_is_admissible(rung, elements):
            blob = pack.serialize(
                forge.pack_product_quant(arr, dim=rung["dim"], subspaces=1, k=rung["k"], seed=0)
            )
            descriptor = {"name": name, "codec": "gravity-pq", "shape": shape,
                          "elements": elements, "rung": rung["rung"]}
            pq_bits += len(blob) * 8
            pq_weights += elements
        else:
            blob = _bf16_bytes(arr)
            descriptor = {"name": name, "codec": "native.bf16", "shape": shape,
                          "elements": elements,
                          "terminal_state": "PROTECTED_SOURCE_NATIVE"}
            native_bits += len(blob) * 8
            native_weights += elements
        descriptor["bytes"] = len(blob)
        payloads.append((descriptor, blob))

    total = pq_weights + native_weights
    gravity.write_shard(
        artifact, payloads,
        model={"repo": "glm52-tiny", "revision": "fixture",
               "representation": "QUANTIZED_TRANSFORMER"},
        architecture=CONFIG,
        tokenizer={"source": "none", "dir": "synthetic"},
        compression={"codec": "gravity-pq", "production_rung": rung["rung"],
                     "representation": "QUANTIZED_TRANSFORMER",
                     "complete_bpw": (pq_bits + native_bits) / max(1, total),
                     "packed_bpw": pq_bits / max(1, pq_weights),
                     "rate_basis": "artifact bytes over ALL declared weights, "
                                   "native-carried tensors included in the denominator"},
        shard={"index": 0, "count": 1},
    )

    # Grade against what the ARTIFACT encodes: read every tensor back through the
    # container and the same codec, exactly as the runtime will. GravityGlmSource
    # is the same source class the flagship oracle run uses, so the fixture and
    # the 77 GB artifact are graded by one methodology, not two that could drift.
    source = GravityGlmSource(out_dir, single_shard=artifact.name)
    logits, _, trace = ref.main_forward(np.array([TOKENS], dtype=np.int64), source, CONFIG)
    flat = np.asarray(logits[0, -1], dtype=np.float32)
    order = np.argsort(-flat, kind="stable")

    (out_dir / "ref_logits.f32").write_bytes(np.ascontiguousarray(flat).tobytes())
    meta = {
        "tokens": TOKENS,
        "argmax": int(order[0]),
        "top5": [int(i) for i in order[:5]],
        "logits_head": [float(x) for x in flat[:8]],
        "final_topk_indices": np.asarray(trace["final_main_topk"]).reshape(-1).tolist(),
        "artifact": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "complete_bpw": (pq_bits + native_bits) / max(1, total),
        "tensors": len(payloads),
        "tensors_pq": sum(1 for d, _ in payloads if d["codec"] == "gravity-pq"),
    }
    (out_dir / "ref_glm.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1].joinpath(
        "crates/hawking-core/tests/fixtures/gravity_glm")
    print(json.dumps(build(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
