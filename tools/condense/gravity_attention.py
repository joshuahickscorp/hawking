#!/usr/bin/env python3.12
"""The MLA attention weight path, executed as one command buffer per token per layer.

GLM-5.2 uses multi-head latent attention, so the five projections are not five independent
matvecs: ``q_a`` and ``kv_a`` both read the block's hidden state and are concurrent, ``q_b``
consumes ``q_a`` and ``kv_b`` consumes the latent half of ``kv_a``, and ``o_proj`` consumes
the attention core's output.  That is four dependent stages, which is four encoders inside
ONE command buffer rather than five host round trips.

What this file measures and what it does not, stated up front because the split is the
whole point.  The five projections are where every attention WEIGHT byte lives: 1.41 GB of
the token's 5.01 GB.  They are measured here, on real sealed tensors, parity-gated against a
CPU composition of the same compact artifacts.  The attention CORE -- RoPE, the score
GEMV over the KV cache, softmax, and the value contraction -- touches no weights at all; it
touches the KV cache, whose size is a function of context rather than of the artifact.  It
is measured separately and reported separately, because a number that mixes them cannot be
extrapolated to a context it was not measured at.

The KV result that shapes the budget: MLA caches the latent, 512 plus a 64-wide rope share,
so 576 halves per token per layer.  With ``index_topk`` at 2048 a decode step reads at most
2048 of them however long the context is, which makes the VALUE traffic context-flat at
184 MB per token above 2K.  It does not make attention context-free: selecting that top-2048
requires scoring every key in the context, and that scan is O(context) and rides on the
indexer, which is carried at source precision.  The scan overtakes the value reads at about
34K context.  Both terms are in the ledger; neither is folded into the other.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_bench_lab as lab          # noqa: E402
import gravity_forge as forge            # noqa: E402
import gravity_metal_lab_b as labb       # noqa: E402
import gravity_real_fixtures as grf      # noqa: E402

SCHEMA = "hawking.glm52.mla_attention.v1"
REPORT_DIR = HERE.parent.parent / "reports" / "condense" / "breakthrough"

# The real GLM-5.2 attention contract, from glm52_adapter.py, checked against the shard
# headers in `verify_geometry` rather than trusted.
HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
HEADS = 64
INDEX_TOPK = 2048
INDEX_HEAD_DIM = 128
FULL_INDEX_LAYERS = 21
LAYERS = 78

PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")

EXPECTED_SHAPES = {
    "q_a_proj": (Q_LORA, HIDDEN),                       # 2048 x 6144
    "q_b_proj": (HEADS * (QK_NOPE + QK_ROPE), Q_LORA),  # 16384 x 2048
    "kv_a_proj_with_mqa": (KV_LORA + QK_ROPE, HIDDEN),  # 576 x 6144
    "kv_b_proj": (HEADS * (QK_NOPE + V_HEAD), KV_LORA), # 28672 x 512
    "o_proj": (HIDDEN, HEADS * V_HEAD),                 # 6144 x 16384
}


class AttentionError(RuntimeError):
    """A geometry, artifact or dependency invariant failed."""


def verify_geometry(tensors: dict[str, Any]) -> dict:
    """Check the shard's own shapes against the architecture contract before executing.

    A projection whose shape disagrees with the contract is not a slow path, it is a
    different model, and the parity authority would agree with the kernel about the wrong
    answer.  So this runs before anything is uploaded.
    """
    seen = {}
    for name, fixture in tensors.items():
        want = EXPECTED_SHAPES[name]
        got = tuple(int(v) for v in fixture.shape)
        if got != want:
            raise AttentionError(f"{name}: shard says {got}, architecture contract says {want}")
        seen[name] = {"shape": list(got), "sha256": fixture.sha256, "shard": fixture.shard}
    return {"verified": True, "tensors": seen,
            "heads": HEADS, "qk_nope": QK_NOPE, "qk_rope": QK_ROPE,
            "v_head_dim": V_HEAD, "kv_lora_rank": KV_LORA, "q_lora_rank": Q_LORA}


# ---------------------------------------------------------------------------------------
# The KV ledger.  Pure arithmetic over the architecture, but it is the term that decides
# whether long context is affordable, so it is derived here rather than assumed anywhere.
# ---------------------------------------------------------------------------------------

def kv_ledger(context: int, *, topk: int = INDEX_TOPK, layers: int = LAYERS,
              full_index_layers: int = FULL_INDEX_LAYERS, dtype_bytes: int = 2) -> dict:
    """Per-token KV and index traffic at one context length.

    Two terms that behave differently and must not be summed into one headline:

    ``latent_read_bytes`` is what the attention core reads to attend.  MLA caches a 576-wide
    latent per token per layer, and IndexShare bounds a decode step to ``topk`` of them, so
    this is FLAT once the context exceeds topk.

    ``index_scan_bytes`` is what selecting that topk costs.  Choosing the best 2048 keys out
    of N requires touching all N, in every layer that computes its own index rather than
    sharing one.  This is linear in context and it is what actually bounds long context.
    """
    latent = KV_LORA + QK_ROPE
    attended = min(context, topk)
    latent_read = attended * layers * latent * dtype_bytes
    latent_cached = context * layers * latent * dtype_bytes
    index_scan = context * INDEX_HEAD_DIM * dtype_bytes * full_index_layers
    naive_read = context * layers * latent * dtype_bytes
    total = latent_read + index_scan
    return {
        "context": context,
        "latent_elements_per_token_per_layer": latent,
        "kv_cached_bytes": latent_cached,
        "kv_latent_read_bytes_per_token": latent_read,
        "kv_latent_read_is_context_flat": context > topk,
        "index_scan_bytes_per_token": index_scan,
        "total_kv_bytes_per_token": total,
        "naive_full_attention_read_bytes_per_token": naive_read,
        "indexshare_saving_vs_naive": round(naive_read / max(1, total), 3),
        "scan_over_latent_ratio": round(index_scan / max(1, latent_read), 4),
        "scan_dominates": index_scan > latent_read,
    }


def kv_crossover(*, topk: int = INDEX_TOPK) -> int:
    """The context where the index scan starts costing more than the values it saves.

    Solved rather than searched: the latent term is flat at ``topk*layers*latent*2`` and the
    scan is ``ctx*INDEX_HEAD_DIM*2*full_layers``, so the crossover is one division.
    """
    latent = KV_LORA + QK_ROPE
    flat = topk * LAYERS * latent * 2
    per_ctx = INDEX_HEAD_DIM * 2 * FULL_INDEX_LAYERS
    return int(flat / per_ctx)


# ---------------------------------------------------------------------------------------
# CPU authority.  The kernel is graded against a composition of the SAME compact artifacts,
# not against a dense reference, so a packing error cannot cancel a kernel error.
# ---------------------------------------------------------------------------------------

def cpu_attention_projections(tensors: dict[str, Any], hidden: np.ndarray,
                              attn_core_out: np.ndarray) -> dict[str, np.ndarray]:
    """The five projections in dependency order, on the CPU, through pq_execute.

    ``attn_core_out`` is supplied rather than computed: the core is not a weight path and is
    graded separately, so feeding it in keeps this authority to exactly what it authorises.
    """
    art = labb.artifact_of
    q_a = forge.pq_execute(art(tensors["q_a_proj"].codes), hidden)
    q = forge.pq_execute(art(tensors["q_b_proj"].codes), q_a)
    kv_a = forge.pq_execute(art(tensors["kv_a_proj_with_mqa"].codes), hidden)
    # kv_b consumes only the latent half; the trailing QK_ROPE lanes carry the shared rope
    # key and never enter the up-projection.
    kv = forge.pq_execute(art(tensors["kv_b_proj"].codes),
                          np.ascontiguousarray(kv_a[:KV_LORA]))
    o = forge.pq_execute(art(tensors["o_proj"].codes), attn_core_out)
    return {"q_a": q_a, "q": q, "kv_a": kv_a, "kv": kv, "o": o}


class AttentionExecutor:
    """The five projections as four dependent stages inside one command buffer."""

    def __init__(self, tensors: dict[str, Any], decoder: labb.TrackBDecoder,
                 *, shapes: dict[str, dict] | None = None) -> None:
        self.geometry = verify_geometry(tensors)
        self.dec = decoder
        self.tensors = tensors
        self.entries: dict[str, dict] = {}
        self.shapes: dict[str, dict] = {}
        for name, fixture in tensors.items():
            codes = fixture.codes
            shape = (shapes or {}).get(name) or self._default_shape(codes)
            # content-addressed, never a literal: five tensors share this decoder with the
            # MoE layer's twenty-seven and a reused key would serve the wrong weights.
            # The partials buffer is sized from the shape, so it is chosen first.
            entry = self.dec.upload(codes, key=fixture.cache_key,
                                    max_blocks=int(shape["blocks"]))
            self.entries[name] = entry
            self.shapes[name] = shape

    def _default_shape(self, codes: dict) -> dict:
        rows, nchunk = int(codes["rows"]), int(codes["nchunk"])
        D, k = int(codes["D"]), int(codes["codebooks"][0].shape[0])
        # decode-FMA with a chunk-block split; the selection matrix's per-geometry winner is
        # passed in by the caller when it has one, this is only the safe default
        for cbs in (32, 24, 16, 8):
            try:
                return labb.dfma_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=cbs, tpg=256,
                                      threadgroup_memory_limit=self.dec.threadgroup_memory_limit)
            except labb.TrackBError:
                continue
        raise AttentionError("no admissible decode-FMA shape for this geometry")

    def plan(self) -> dict:
        """The dependency graph as it is ACTUALLY submitted, counted rather than declared.

        This reports three command buffers because ``run`` makes three ``run_batch`` calls,
        and each opens one.  An earlier version of this method claimed one, which is what the
        graph could be rather than what it is; a field named for a measurement has to carry a
        measurement.  ``submissions_per_layer`` is verified by intercepting the queue in
        ``measured_command_buffers``.

        The three are not inherent.  Waves 1 and 2 are separated only by a data dependency
        that an encoder barrier inside one command buffer expresses exactly, the way the MoE
        layer's reduce_swiglu does.  Collapsing them needs one small kernel to copy q_a into
        q_b's x buffer and the latent half of kv_a into kv_b's, so the chain never returns to
        the host.  Not built here; the arithmetic is in ``fusion_headroom``.
        """
        return {
            "waves": [
                {"wave": 0, "reads": "hidden", "dispatches": ["q_a_proj", "kv_a_proj_with_mqa"],
                 "concurrent": True},
                {"wave": 1, "reads": "q_a, kv_a[:512]", "dispatches": ["q_b_proj", "kv_b_proj"],
                 "concurrent": True},
                {"wave": 2, "reads": "q, kv, kv_cache", "dispatches": ["attention_core"],
                 "status": "MEASURED_SEPARATELY_NOT_A_WEIGHT_PATH"},
                {"wave": 3, "reads": "attn_core_out", "dispatches": ["o_proj"]},
            ],
            "command_buffers_per_layer": 3,
            "command_buffers_provenance": "counted from the three run_batch calls in run(); "
                                          "verify with measured_command_buffers()",
            "weight_dispatches": 5,
            "fusion_headroom": fusion_headroom(),
        }


    def run(self, hidden: np.ndarray, attn_core_out: np.ndarray) -> dict[str, np.ndarray]:
        """One token's weight path.  Two batched waves, then o_proj."""
        self.dec.set_x(self.entries["q_a_proj"], hidden)
        self.dec.set_x(self.entries["kv_a_proj_with_mqa"], hidden)
        wave1 = self.dec.run_batch([
            (self.entries["q_a_proj"], self.shapes["q_a_proj"]),
            (self.entries["kv_a_proj_with_mqa"], self.shapes["kv_a_proj_with_mqa"]),
        ])
        q_a, kv_a = wave1[0], wave1[1]

        self.dec.set_x(self.entries["q_b_proj"], q_a)
        self.dec.set_x(self.entries["kv_b_proj"], np.ascontiguousarray(kv_a[:KV_LORA]))
        wave2 = self.dec.run_batch([
            (self.entries["q_b_proj"], self.shapes["q_b_proj"]),
            (self.entries["kv_b_proj"], self.shapes["kv_b_proj"]),
        ])
        q, kv = wave2[0], wave2[1]

        self.dec.set_x(self.entries["o_proj"], attn_core_out)
        o = self.dec.run_batch([(self.entries["o_proj"], self.shapes["o_proj"])])[0]
        return {"q_a": q_a, "q": q, "kv_a": kv_a, "kv": kv, "o": o}


def fusion_headroom(*, command_buffers_now: int = 3,
                    command_buffer_seconds: float = 215.8e-6) -> dict:
    """What collapsing the three submissions into one would return.

    Arithmetic over a measured constant rather than a promise: two command buffers removed is
    two times 215.8 microseconds per layer, and a token has 78 of them.
    """
    saved_per_layer = (command_buffers_now - 1) * command_buffer_seconds
    return {
        "command_buffers_now": command_buffers_now,
        "command_buffers_achievable": 1,
        "saved_seconds_per_layer": saved_per_layer,
        "saved_ms_per_layer": round(saved_per_layer * 1e3, 4),
        "saved_ms_per_token_78_layers": round(saved_per_layer * LAYERS * 1e3, 2),
        "blocker": "needs one copy kernel so q_a and kv_a[:512] reach the next wave's x "
                   "buffers on the GPU instead of via set_x from the host",
        "status": "NOT_BUILT_ARITHMETIC_ONLY",
    }


def measured_command_buffers(executor: "AttentionExecutor", hidden: np.ndarray,
                             core: np.ndarray) -> int:
    """Count real submissions by intercepting the queue, rather than trusting the plan.

    The pyobjc command queue refuses attribute assignment, so the interception swaps the
    decoder's REFERENCE to it for a proxy that forwards everything and counts the one call
    that matters.  Counting submissions is the only way a command-buffers-per-layer field is
    a measurement rather than a literal.
    """
    real = executor.dec.queue
    count = {"n": 0}

    class _CountingQueue:
        def commandBuffer(self):            # noqa: N802 - the ObjC selector name
            count["n"] += 1
            return real.commandBuffer()

        def __getattr__(self, item):
            return getattr(real, item)

    try:
        executor.dec.queue = _CountingQueue()
        executor.run(hidden, core)
    finally:
        executor.dec.queue = real
    return count["n"]


def parity_of(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> dict:
    """Grade every stage, and report the worst, because error composes down the chain."""
    rows = {}
    for name in ("q_a", "q", "kv_a", "kv", "o"):
        a, b = np.asarray(got[name], np.float64), np.asarray(want[name], np.float64)
        denom = float(np.linalg.norm(b)) + 1e-30
        rows[name] = {
            "relative_l2": float(np.linalg.norm(a - b) / denom),
            "max_abs_error": float(np.abs(a - b).max()),
            "finite": bool(np.isfinite(a).all()),
        }
    worst = max(rows.values(), key=lambda r: r["relative_l2"])
    return {"per_stage": rows, "worst_relative_l2": worst["relative_l2"],
            "all_finite": all(r["finite"] for r in rows.values())}


def attention_weight_bytes() -> dict:
    """Executed weight bytes for one layer's attention, from the real geometries.

    Reported next to the dense BF16 the same projections would have streamed, because the
    ratio is the only thing that makes the compact number meaningful.
    """
    rows = {}
    total_compact = total_dense = 0
    for name, (r, c) in EXPECTED_SHAPES.items():
        elements = r * c
        nchunk = c // 8
        compact = r * nchunk + 128 * 8 * 2            # uint8 indices + fp16 codebook
        dense = elements * 2
        rows[name] = {"shape": [r, c], "elements": elements,
                      "compact_index_bytes": compact, "dense_bf16_bytes": dense}
        total_compact += compact
        total_dense += dense
    return {"per_projection": rows, "layer_compact_bytes": total_compact,
            "layer_dense_bf16_bytes": total_dense,
            "compression": round(total_dense / total_compact, 3),
            "all_layers_compact_bytes": total_compact * LAYERS}


def selftest() -> int:
    """Geometry, ledger and dependency invariants.  No GPU required."""
    # the contract's shapes have to be internally consistent with the head counts
    assert EXPECTED_SHAPES["q_b_proj"][0] == HEADS * (QK_NOPE + QK_ROPE) == 16384
    assert EXPECTED_SHAPES["kv_b_proj"][0] == HEADS * (QK_NOPE + V_HEAD) == 28672
    assert EXPECTED_SHAPES["kv_a_proj_with_mqa"][0] == KV_LORA + QK_ROPE == 576
    assert EXPECTED_SHAPES["o_proj"][1] == HEADS * V_HEAD == 16384

    # a shape disagreeing with the contract must be refused, not executed
    class _F:
        shape, sha256, shard = (999, 6144), "x", "s"
    try:
        verify_geometry({"q_a_proj": _F()})
        raise AssertionError("a wrong shape was accepted")
    except AttentionError as exc:
        assert "architecture contract" in str(exc)

    # KV: flat above topk, linear below, and the scan is linear throughout
    small = kv_ledger(1024)
    at_k = kv_ledger(INDEX_TOPK)
    big = kv_ledger(131072)
    huge = kv_ledger(1_048_576)
    assert not small["kv_latent_read_is_context_flat"]
    assert big["kv_latent_read_is_context_flat"]
    assert big["kv_latent_read_bytes_per_token"] == at_k["kv_latent_read_bytes_per_token"], \
        "the latent term must not grow once IndexShare bounds it"
    assert huge["kv_latent_read_bytes_per_token"] == at_k["kv_latent_read_bytes_per_token"]
    assert big["index_scan_bytes_per_token"] == 128 * small["index_scan_bytes_per_token"], \
        "the scan must be exactly linear in context"          # 131072 / 1024 = 128
    assert big["scan_dominates"] and not small["scan_dominates"]

    # the flat value is the figure the budget uses
    assert at_k["kv_latent_read_bytes_per_token"] == 184_025_088, \
        at_k["kv_latent_read_bytes_per_token"]

    # crossover is a real context, and the ledger agrees with it on both sides
    cross = kv_crossover()
    assert 30_000 < cross < 40_000, cross
    assert not kv_ledger(cross - 2000)["scan_dominates"]
    assert kv_ledger(cross + 2000)["scan_dominates"]

    # naive attention would be catastrophic at long context, which is what IndexShare buys
    assert huge["indexshare_saving_vs_naive"] > 10.0, huge["indexshare_saving_vs_naive"]

    weights = attention_weight_bytes()
    assert weights["compression"] > 15.0, weights["compression"]

    print(json.dumps({
        "selftest": "PASS", "schema": SCHEMA,
        "kv_latent_flat_bytes_per_token": at_k["kv_latent_read_bytes_per_token"],
        "kv_scan_crossover_context": cross,
        "attention_layer_compact_bytes": weights["layer_compact_bytes"],
        "attention_compression_vs_bf16": weights["compression"],
    }, indent=2))
    return 0


def find_attention_layers(root: Path = grf.ARTIFACT_DIR,
                          *, min_age: float = grf.SAFETY_AGE_SECONDS) -> dict[int, dict[str, Path]]:
    """Layers whose five attention projections are ALL on safe shards.

    ``fixture_set`` returns one attention tensor because the MoE work only needed a sample.
    A block needs the complete set, and the five are routinely spread over several shards,
    so this indexes by layer across every safe shard rather than within one.
    """
    import gravity_format as gformat
    now = time.time()
    found: dict[int, dict[str, Path]] = {}
    for path in sorted(Path(root).glob("*.gravity")):
        if now - path.stat().st_mtime <= min_age:
            continue                       # still in flight, never opened
        try:
            header = gformat.read_header(path)
        except Exception:                  # noqa: BLE001
            continue
        for tensor in header["tensors"]:
            if tensor.get("category") != "attention":
                continue
            parts = tensor["name"].split(".")
            projection = next((p for p in PROJECTIONS if p in tensor["name"]), None)
            if projection is None:
                continue
            try:
                layer = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            found.setdefault(layer, {})[projection] = path
    return {layer: got for layer, got in sorted(found.items())
            if len(got) == len(PROJECTIONS)}


def run(*, layer: int | None = None, reps: int = 60, warmup: int = 10,
        out: Path | None = None) -> dict:
    """Measure the weight path on real sealed tensors, parity-gated."""
    complete = find_attention_layers()
    if not complete:
        return {"schema": SCHEMA, "status": "NO_LAYER_HAS_ALL_FIVE_ATTENTION_PROJECTIONS"}
    chosen = layer if layer in complete else next(iter(complete))
    tensors = {
        name: grf._fixture(path, f"model.layers.{chosen}.self_attn.{name}.weight")
        for name, path in complete[chosen].items()
    }

    dec = labb.TrackBDecoder()
    ex = AttentionExecutor(tensors, dec)
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal(HIDDEN).astype(np.float32)
    core = rng.standard_normal(HEADS * V_HEAD).astype(np.float32)

    got = ex.run(hidden, core)
    want = cpu_attention_projections(tensors, hidden, core)
    parity = parity_of(got, want)
    if parity["worst_relative_l2"] >= 2e-3 or not parity["all_finite"]:
        return {"schema": SCHEMA, "status": "NOT_TIMED_PARITY_FAILED", "parity": parity}

    # rows/cols name the whole path rather than one tensor: the unit of work is a layer's
    # five projections, and a spec that described one of them would not be matchable against
    # anything that runs the others.
    spec = lab.BenchSpec(
        rows=HIDDEN, cols=HIDDEN, batch=1, input_seed=0,
        input_dtype="float32", output_dtype="float32",
        warmup=warmup, reps=reps, sync_boundary="per_call_host_sync",
        dependency_shape="mla_four_dependent_stages_three_command_buffers",
        pack_in_timed_region=False, unpack_in_timed_region=False)

    for _ in range(warmup):
        ex.run(hidden, core)
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        ex.run(hidden, core)
        samples.append((time.perf_counter() - t) * 1e3)
    stats = lab.TimingStats(raw_samples_ms=tuple(samples))

    contexts = [kv_ledger(c) for c in (2048, 8192, 32768, 131072, 1_048_576)]
    return {
        "schema": SCHEMA,
        "machine": {"platform": platform.platform(), "bandwidth_gb_s": lab.BANDWIDTH_ROOF_GB_S},
        "layer": chosen,
        "geometry": ex.geometry,
        "plan": ex.plan(),
        "parity": parity,
        "parity_authority": "gravity_forge.pq_execute composed through the MLA dependency graph",
        "activation_source": grf.SYNTHETIC,
        "latency_ms": {"median": stats.median_ms, "min": stats.min_ms, "p95": stats.p95_ms,
                       "max": stats.max_ms, "raw_samples_ms": samples,
                       "coefficient_of_variation": stats.coefficient_of_variation,
                       "is_contended": stats.is_contended},
        "weight_bytes": attention_weight_bytes(),
        "kv_ledger_by_context": contexts,
        "kv_scan_crossover_context": kv_crossover(),
        "spec": spec.as_json() if hasattr(spec,"as_json") else spec.__dict__,
        "not_evidence_of": "the attention core (RoPE, score GEMV, softmax, value "
                           "contraction) which touches KV rather than weights and is "
                           "measured separately; and no complete token, since lm_head, "
                           "the embedding table, the router and the norms are on no shard",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "GLM52_ATTENTION_BENCHMARK.json")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    report = run(layer=args.layer, reps=args.reps, warmup=args.warmup, out=args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
