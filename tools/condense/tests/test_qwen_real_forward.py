#!/usr/bin/env python3.12
"""Tests for the real Qwen3-235B-A22B (qwen3_moe) full-model forward.

The real 438 GiB source is intentionally NOT staged (only models/qwen3-235b-a22b/_meta/ exists), so
logits_for() against the true shards is untested-pending-source. What IS validated here is a tiny
SYNTHETIC TWIN that exercises the entire machine end to end in milliseconds:

  * a 2-layer / 4-expert / top-2 / hidden-16 / head_dim-8 qwen3_moe config with random bf16 tensors
    written to REAL safetensors shards + index (so the exact byte-range streaming loader is exercised
    for real, not mocked);
  * the full residual chain: embed -> [q/k-norm attention + RoPE + GQA -> residual -> MoE SwiGLU ->
    residual] x2 -> final norm -> untied lm_head -> logits;
  * router (softmax-first top-k + norm_topk_prob), q/k-norm detection+application, and NLL/ppl;
  * an assertion that this module's STANDARD SwiGLU differs from gpt-oss's clamped interleaved gate.

Also a metadata-only check against the real _meta config confirming the wiring targets qwen3_moe.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import qwen_real_forward as Q   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
META = REPO / "models" / "qwen3-235b-a22b" / "_meta"


# --------------------------------------------------------------------------- #
# Synthetic twin: tiny qwen3_moe with random tensors on real safetensors shards
# --------------------------------------------------------------------------- #
TINY_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 8,
    "vocab_size": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000.0,
    "norm_topk_prob": True,
    "tie_word_embeddings": False,
}


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a real bf16 safetensors shard: 8-byte header len + JSON header + concatenated data."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = Q.f32_to_bf16_bits(arr).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        for b in blobs:
            fh.write(b)


def _build_tiny_checkpoint(tmp: Path, seed: int = 0) -> Q.QwenGeometry:
    """Materialize a tiny qwen3_moe across TWO shards (exercises multi-shard mapping) + write index."""
    g = Q.QwenGeometry(TINY_CONFIG)
    rng = np.random.default_rng(seed)

    def rnd(*shape: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        # norm weights near 1 with tiny jitter (keeps activations well-scaled)
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    shard1: dict[str, np.ndarray] = {}
    shard2: dict[str, np.ndarray] = {}

    # top-level: embed in shard1, final norm + lm_head in shard2
    shard1["model.embed_tokens.weight"] = rnd(g.vocab, g.hidden, scale=0.2)
    shard2["model.norm.weight"] = ones(g.hidden)
    shard2["lm_head.weight"] = rnd(g.vocab, g.hidden, scale=0.2)

    def emit_layer(L: int, into: dict[str, np.ndarray]) -> None:
        b = f"model.layers.{L}."
        into[b + "self_attn.q_proj.weight"] = rnd(g.q_out, g.hidden)
        into[b + "self_attn.k_proj.weight"] = rnd(g.kv_out, g.hidden)
        into[b + "self_attn.v_proj.weight"] = rnd(g.kv_out, g.hidden)
        into[b + "self_attn.o_proj.weight"] = rnd(g.hidden, g.q_out)
        into[b + "self_attn.q_norm.weight"] = ones(g.head_dim)
        into[b + "self_attn.k_norm.weight"] = ones(g.head_dim)
        into[b + "input_layernorm.weight"] = ones(g.hidden)
        into[b + "post_attention_layernorm.weight"] = ones(g.hidden)
        into[b + "mlp.gate.weight"] = rnd(g.n_experts, g.hidden)
        for e in range(g.n_experts):
            into[b + f"mlp.experts.{e}.gate_proj.weight"] = rnd(g.moe_inter, g.hidden)
            into[b + f"mlp.experts.{e}.up_proj.weight"] = rnd(g.moe_inter, g.hidden)
            into[b + f"mlp.experts.{e}.down_proj.weight"] = rnd(g.hidden, g.moe_inter)

    emit_layer(0, shard1)
    emit_layer(1, shard2)

    _write_safetensors(tmp / "model-00001-of-00002.safetensors", shard1)
    _write_safetensors(tmp / "model-00002-of-00002.safetensors", shard2)

    weight_map = {n: "model-00001-of-00002.safetensors" for n in shard1}
    weight_map.update({n: "model-00002-of-00002.safetensors" for n in shard2})
    with open(tmp / "model.safetensors.index.json", "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, fh)
    with open(tmp / "config.json", "w") as fh:
        json.dump(TINY_CONFIG, fh)
    return g


@pytest.fixture()
def tiny(tmp_path: Path):
    g = _build_tiny_checkpoint(tmp_path)
    reader = Q.SafetensorsIndexReader(tmp_path)
    fwd = Q.QwenRealForward(reader, g)
    yield fwd, g, reader
    reader.close()


# --------------------------------------------------------------------------- #
# Reader / streaming loader
# --------------------------------------------------------------------------- #
def test_reader_roundtrips_bf16_and_maps_shards(tiny):
    _fwd, g, reader = tiny
    assert reader.source_present()
    # a shard-1 tensor and a shard-2 tensor both resolve + read
    q = reader.bf16("model.layers.0.self_attn.q_proj.weight")
    assert q.shape == (g.q_out, g.hidden) and np.isfinite(q).all()
    lm = reader.bf16("lm_head.weight")
    assert lm.shape == (g.vocab, g.hidden)
    assert reader.shard_of("model.layers.0.mlp.gate.weight").endswith("00001-of-00002.safetensors")
    assert reader.shard_of("model.layers.1.mlp.gate.weight").endswith("00002-of-00002.safetensors")


def test_bf16_rows_matches_full_gather(tiny):
    _fwd, _g, reader = tiny
    full = reader.bf16("model.embed_tokens.weight")
    rows = [3, 0, 7, 3]
    gathered = reader.bf16_rows("model.embed_tokens.weight", rows)
    assert np.array_equal(gathered, full[rows])   # bounded gather == full-table indexing, exactly


def test_remote_reader_uses_exact_ranges_and_cache(tmp_path):
    """The remote path is exercised without a network: the injected transport sees only exact
    half-open ranges, while the second read is served by the content-addressed cache."""
    source = tmp_path / "source"
    source.mkdir()
    g = _build_tiny_checkpoint(source)
    blobs = {p.name: p.read_bytes() for p in source.glob("*.safetensors")}
    calls: list[tuple[str, int, int]] = []

    def fetch(shard: str, start: int, end: int) -> bytes:
        calls.append((shard, start, end))
        return blobs[shard][start:end]

    cache = tmp_path / "range-cache"
    reader = Q.RemoteSafetensorsIndexReader(
        source / "model.safetensors.index.json", cache_dir=cache, range_fetcher=fetch,
        repo_id="fixture/tiny", revision="fixture-revision",
    )
    q1 = reader.bf16("model.layers.0.self_attn.q_proj.weight")
    n_calls = len(calls)
    q2 = reader.bf16("model.layers.0.self_attn.q_proj.weight")
    assert np.array_equal(q1, q2)
    assert q1.shape == (g.q_out, g.hidden)
    assert len(calls) == n_calls                         # tensor + header were cache hits
    telem = reader.telemetry_json()
    assert telem["cache_hits"] >= 1 and telem["network_bytes"] > 0
    assert "model.layers.0.self_attn.q_proj.weight" in telem["tensors_read"]


def test_remote_forward_matches_local_forward(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    g = _build_tiny_checkpoint(source, seed=9)
    blobs = {p.name: p.read_bytes() for p in source.glob("*.safetensors")}
    remote = Q.RemoteSafetensorsIndexReader(
        source / "model.safetensors.index.json",
        range_fetcher=lambda shard, start, end: blobs[shard][start:end],
        repo_id="fixture/tiny", revision="fixture-revision",
    )
    local = Q.SafetensorsIndexReader(source)
    ids = [1, 5, 9, 2]
    got = Q.QwenRealForward(remote, g).logits_for(ids, positions="all")
    expected = Q.QwenRealForward(local, g).logits_for(ids, positions="all")
    local.close(); remote.close()
    assert np.array_equal(got, expected)


# --------------------------------------------------------------------------- #
# Full forward chain
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_finiteness(tiny):
    fwd, g, _reader = tiny
    ids = [1, 5, 9, 2, 17, 4]
    last = fwd.logits_for(ids, positions="last")
    allp = fwd.logits_for(ids, positions="all")
    assert last.shape == (1, g.vocab) and np.isfinite(last).all()
    assert allp.shape == (len(ids), g.vocab) and np.isfinite(allp).all()
    # the "last" logits equal the final row of the "all" logits (same computation, same selection)
    assert np.allclose(last[0], allp[-1], atol=1e-4)


def test_qk_norm_detected_and_partial_blocks(tiny):
    fwd, g, _reader = tiny
    assert fwd.has_qk_norm is True                      # q_norm/k_norm present in the index
    ids = [1, 5, 9, 2]
    hidden = fwd.logits_for(ids, positions="all", max_blocks=1)
    assert hidden.shape == (len(ids), g.hidden)         # partial forward returns raw hidden state
    assert np.isfinite(hidden).all()


def test_nll_and_perplexity(tiny):
    fwd, _g, _reader = tiny
    ids = [1, 5, 9, 2, 17, 4, 8]
    out = fwd.nll(ids)
    assert out["n_pred"] == len(ids) - 1
    assert np.isfinite(out["nll"]) and out["nll"] > 0.0
    assert out["perplexity"] > 1.0                      # ppl = exp(nll) > 1 for a nontrivial nll


def test_forward_is_deterministic(tiny):
    fwd, _g, _reader = tiny
    ids = [1, 5, 9, 2, 17]
    a = fwd.logits_for(ids, positions="last")
    b = fwd.logits_for(ids, positions="last")
    assert np.array_equal(a, b)                         # cache + streaming reload is deterministic


# --------------------------------------------------------------------------- #
# expert_hook substitution (the Gravity/Doctor original-vs-packed seam)
# --------------------------------------------------------------------------- #
def test_expert_hook_is_called_and_changes_output(tiny):
    fwd, _g, _reader = tiny
    ids = [1, 5, 9, 2, 17, 4]
    base = fwd.logits_for(ids, positions="all")

    seen: list[tuple[int, int]] = []

    def zero_hook(L, e, w):
        seen.append((L, e))
        return {k: np.zeros_like(v) for k, v in w.items()}

    # fresh forward with a distinct cache so the hook actually runs on load
    reader2 = Q.SafetensorsIndexReader(_reader_dir(fwd))
    from bounded_cache import PressureAwareCache
    fwd2 = Q.QwenRealForward(reader2, fwd.g, cache=PressureAwareCache("hooktest", verbose=False))
    hooked = fwd2.logits_for(ids, positions="all", expert_hook=zero_hook)
    reader2.close()

    assert seen, "expert_hook was never invoked"
    # zeroing experts kills the MoE contribution -> logits must move
    assert not np.allclose(base, hooked, atol=1e-3)


def _reader_dir(fwd) -> Path:
    return fwd.reader.source_dir


# --------------------------------------------------------------------------- #
# The load-bearing activation difference: standard SwiGLU vs gpt-oss clamped gate
# --------------------------------------------------------------------------- #
def test_standard_swiglu_differs_from_gptoss_clamped_gate():
    import gptoss_real_forward as GPT   # the gpt-oss interleaved/clamped activation
    rng = np.random.default_rng(1)
    inter = 64
    gate = (rng.standard_normal(inter) * 4.0).astype(np.float32)   # large values -> clamp matters
    up = (rng.standard_normal(inter) * 4.0).astype(np.float32)

    standard = Q.swiglu(gate, up)                                  # silu(gate) * up

    # gpt-oss consumes an INTERLEAVED [2*inter] gate_up: [::2]=gate, [1::2]=up, then
    # clamp(gate,max=7), clamp(up,+-7), glu=gate*sigmoid(1.702*gate), out=(up+1)*glu
    gate_up = np.empty(2 * inter, dtype=np.float32)
    gate_up[0::2] = gate
    gate_up[1::2] = up
    gptoss = GPT.apply_gate(gate_up)

    assert standard.shape == gptoss.shape == (inter,)
    assert not np.allclose(standard, gptoss)                       # genuinely different activations
    # and specifically: gpt-oss clamps, standard does not -> diverge most where |gate| is large
    assert np.max(np.abs(standard - gptoss)) > 1.0


def test_router_softmax_first_topk_and_renorm():
    logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0], dtype=np.float32)
    idx, w = Q.route_topk(logits, top_k=2, norm_topk_prob=True)
    assert list(idx) == [4, 0]                                     # top-2 by logit (== by softmax)
    assert abs(float(w.sum()) - 1.0) < 1e-6                        # renormalized to sum 1
    # without renorm, weights are the raw softmax mass of the selected experts (< 1)
    _idx2, w2 = Q.route_topk(logits, top_k=2, norm_topk_prob=False)
    assert float(w2.sum()) < 1.0


def test_rope_is_norm_preserving_and_identity_at_pos0():
    rng = np.random.default_rng(2)
    head_dim = 8
    x = rng.standard_normal((3, 2, head_dim)).astype(np.float32)   # [seq, heads, head_dim]
    pos = np.arange(3)
    y = Q.rope(x, pos, head_dim, theta=5_000_000.0)
    assert np.allclose(y[0], x[0], atol=1e-5)                      # position 0: cos=1, sin=0
    assert np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-4)


# --------------------------------------------------------------------------- #
# Real metadata sanity (no weights, no 438 GiB source touched)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (META / "config.json").exists(), reason="_meta not present")
def test_real_meta_geometry_targets_qwen3_moe():
    with open(META / "config.json") as fh:
        cfg = json.load(fh)
    g = Q.QwenGeometry(cfg)
    assert cfg["model_type"] == "qwen3_moe"
    assert (g.n_layers, g.hidden, g.head_dim) == (94, 4096, 128)
    assert (g.n_heads, g.n_kv) == (64, 4)
    assert (g.n_experts, g.top_k, g.moe_inter) == (128, 8, 1536)
    assert g.vocab == 151936 and g.tie is False
    assert g.q_out == 8192 and g.kv_out == 512          # decoupled head_dim (q_out != hidden)


@pytest.mark.skipif(not (META / "model.safetensors.index.json").exists(), reason="_meta not present")
def test_real_source_is_untested_pending_source():
    # The forward wires against the real index but the 438 GiB shards are not staged: source_present
    # must be False and no shard bytes are ever read here.
    fwd = Q.from_source()
    assert fwd.source_present() is False
    assert fwd.has_qk_norm is True                      # q_norm/k_norm ARE in the real index
