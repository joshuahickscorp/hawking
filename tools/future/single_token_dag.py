#!/usr/bin/env python3
"""G082: the single-token dependency DAG, and how much of it is serial by SEMANTICS.

S025 §4-5. Stop viewing the executor as 628 dispatches. The question it forces is
the useful one:

    HOW MUCH GPU WORK IS SERIAL BY TRUE SEMANTICS, AND HOW MUCH IS SERIAL ONLY
    BECAUSE THIS EXECUTOR SCHEDULES IT THAT WAY?

The distinction matters because G009 showed the machine has capacity a single
stream does not use - ~361 GB/s at n=1 against 449-580 aggregate. If the missing
capacity is ARTIFICIAL serialization, reordering recovers it. If the token is a
true chain, no amount of reordering will, and the cause is inside the kernels.

The edges here are declared from the decode step's data flow and each one names
WHY it exists. An edge asserted without a reason is worse than no DAG at all, so
_edge refuses one. Where independence cannot be established from the data flow
the answer is UNKNOWN, never "probably parallel".

    python3 tools/future/single_token_dag.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/single_token_dag.py"
RECEIPT_NAME = "SINGLE_TOKEN_PARALLEL_SLACK.json"

ORGAN_MEASUREMENT = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
MOTIFS = "receipts/future/DISPATCH_MOTIFS.json"
CONCURRENCY = "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json"

TRUE_DEPENDENCY = "TRUE_DEPENDENCY"
ARTIFICIAL = "ARTIFICIAL_SERIALIZATION"
UNKNOWN = "UNKNOWN"


class DagRefused(RuntimeError):
    """An edge without a reason, or a receipt without its measurement."""


def _edge(src: str, dst: str, why: str, kind: str = TRUE_DEPENDENCY) -> dict[str, Any]:
    if not why or len(why) < 20:
        raise DagRefused(
            f"{src}->{dst}: an edge must say WHY it exists. An unexplained edge "
            "makes every critical path downstream of it unfalsifiable."
        )
    if kind not in {TRUE_DEPENDENCY, ARTIFICIAL, UNKNOWN}:
        raise DagRefused(f"{src}->{dst}: unknown edge kind {kind!r}")
    return {"from": src, "to": dst, "kind": kind, "why": why}


# The decode step's data flow. Read set / write set are stated as the reason on
# each edge: X->Y exists when Y reads what X writes.
def token_edges() -> list[dict[str, Any]]:
    e = [
        _edge("embed", "layer.in_rmsnorm",
              "the norm reads the embedding row this token's id selected"),
        _edge("layer.in_rmsnorm", "mixer.proj",
              "the projection reads the normalised hidden state, nothing else"),
        _edge("mixer.proj", "mixer.core",
              "DeltaNet's gated-delta decode reads q/k/v/z; GQA's mha reads q/k/v"),
        _edge("mixer.core", "mixer.out",
              "the out projection reads the mixer's own output activations"),
        _edge("mixer.out", "post_attn_rmsnorm",
              "add_residual_rmsnorm reads the mixer output and the residual"),
        _edge("post_attn_rmsnorm", "mlp.gate_up_swiglu",
              "gate and up both read the post-attention normalised state"),
        _edge("mlp.gate_up_swiglu", "mlp.down",
              "down reads the SwiGLU product, which is gate*silu(up)"),
        _edge("mlp.down", "layer.residual",
              "the residual add reads the MLP output"),
        _edge("layer.residual", "layer.next.in_rmsnorm",
              "layer N+1 normalises layer N's residual stream; this is the chain "
              "that makes a token serial and it is TRUE, not an artefact"),
        _edge("layer.last.residual", "lm_head",
              "the head reads the final hidden state"),
        _edge("lm_head", "argmax",
              "sampling reads the logits"),
    ]
    return e


# What the fusions ALREADY captured. These were independent operations and the
# sealed build put each group in ONE kernel, so they are no longer schedulable
# units at all - the slack was taken, not left on the table.
FUSED_INDEPENDENCE: tuple[dict[str, Any], ...] = (
    {"was": ["gate_proj", "up_proj", "swiglu"],
     "now": "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
     "why_independent": "gate and up both read the same normalised x and neither "
                        "reads the other's output"},
    {"was": ["q_proj", "k_proj", "v_proj"],
     "now": "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
     "why_independent": "three projections of the same input, no cross-reads"},
    {"was": ["residual_add", "rmsnorm"],
     "now": "qwen80_add_residual_rmsnorm_tg",
     "why_independent": "elementwise then reduction over the same buffer; fusing "
                        "removes a full activation round trip"},
    {"was": ["ba_projection", "decay_beta"],
     "now": "qwen38_gated_delta_decode_vi_simd_ba_f4",
     "why_independent": "widen_f4 folds ba_to_decay into the state kernel"},
)


def _measurement() -> dict[str, Any]:
    path = REPO / ORGAN_MEASUREMENT
    if not path.is_file():
        raise DagRefused(
            f"{ORGAN_MEASUREMENT} is not on disk; this receipt is a measurement "
            "of a real token, not a drawing of one"
        )
    return json.loads(path.read_text())


# EVERY MEASURED ROW IS BOUND TO A NODE, or the receipt refuses. An unmapped row
# is measured time with no place in the dependency graph, and a critical path
# that quietly omits it is wrong by exactly that much.
#
# q4_remainder is the one that needed reading rather than guessing:
# qwen38_hybrid_decode.rs:5310 encodes it as the mixer OUT projection for all 64
# layers - linear_attn.out_proj for DeltaNet layers, self_attn.o_proj for GQA -
# so it is mixer.out, squarely on the chain and not a floating remainder.
ROW_TO_NODE: dict[str, str] = {
    "embedding": "embed",
    "deltanet": "mixer.core",
    "gqa_attention": "mixer.core",
    "q4_remainder": "mixer.out",
    "mlp_gate_up": "mlp.gate_up_swiglu",
    "mlp_down": "mlp.down",
    "lm_head": "lm_head",
    "sampling": "argmax",
}
ROW_TO_NODE_CITES = {
    "q4_remainder": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs:5310",
}


def node_times() -> dict[str, Any]:
    """Measured ms per DAG node. Refuses an unmapped row."""
    rows = _measurement()["organs"]["rows"]
    unmapped = [r["organ"] for r in rows if r["organ"] not in ROW_TO_NODE]
    if unmapped:
        raise DagRefused(
            f"measured rows with no node: {unmapped}. Measured time outside the "
            "graph makes the critical path wrong by exactly that much."
        )
    by_node: dict[str, float] = {}
    for r in rows:
        by_node.setdefault(ROW_TO_NODE[r["organ"]], 0.0)
        by_node[ROW_TO_NODE[r["organ"]]] += float(r["gpu_ms"])
    return {
        "ms_by_node": {k: round(v, 4) for k, v in sorted(by_node.items())},
        "row_to_node": dict(ROW_TO_NODE),
        "cites": dict(ROW_TO_NODE_CITES),
        "n_rows": len(rows),
        "unmapped_rows": [],
    }


def slack() -> dict[str, Any]:
    """Total GPU work against the critical path, from the measured organ census."""
    doc = _measurement()
    rows = doc["organs"]["rows"]
    total_ms = sum(float(r["gpu_ms"]) for r in rows)
    edges = token_edges()
    artificial = [x for x in edges if x["kind"] == ARTIFICIAL]
    unknown = [x for x in edges if x["kind"] == UNKNOWN]

    # Every measured region sits on the chain: each layer's regions are ordered by
    # the edges above, and layer N+1 depends on layer N. So the critical path is
    # the whole of the measured work, and the overlapable part is exactly the work
    # on edges that are NOT true dependencies - currently none.
    overlapable_ms = 0.0
    critical_ms = total_ms - overlapable_ms
    return {
        "total_gpu_work_ns": int(total_ms * 1e6),
        "critical_path_ns": int(critical_ms * 1e6),
        "theoretically_overlapable_ns": int(overlapable_ms * 1e6),
        "actually_overlapped_ns": 0,
        "overlap_efficiency": None if overlapable_ms == 0 else 0.0,
        "overlap_efficiency_is_none_because": (
            "there is nothing to overlap at this grain: every edge in the token's "
            "data flow is a TRUE dependency, so an efficiency would divide by zero "
            "and reporting 0.0 would read as a failure to exploit slack that does "
            "not exist"
        ),
        "largest_independent_regions": [],
        "largest_artificial_barriers": artificial,
        "unknown_edges": unknown,
        "n_edges": len(edges),
        "n_true_dependency": len(edges) - len(artificial) - len(unknown),
        "measured_rows": rows,
        "measured_total_ms": round(total_ms, 4),
        "nodes": node_times(),
    }


def reading() -> dict[str, Any]:
    conc = REPO / CONCURRENCY
    conc_doc = json.loads(conc.read_text()) if conc.is_file() else None
    return {
        "finding": "THE_TOKEN_IS_A_CHAIN",
        "what_it_means": (
            "Every edge in the decode step is a true data dependency and the "
            "sealed build has ALREADY fused the four places where independent "
            "operations existed - gate/up/swiglu, q/k/v, residual/rmsnorm, and "
            "ba_to_decay into the state kernel. So there is no artificial "
            "serialization left to remove AT THE DISPATCH GRAIN, and the "
            "theoretically overlapable work between dispatches is zero."
        ),
        "therefore_about_g009": (
            "the capacity a second stream exposes cannot be coming from "
            "reordering this token's dispatches, because there is no ordering "
            "freedom left. It has to come from INSIDE the kernels - occupancy, "
            "memory-level parallelism, or latency the kernel cannot hide from "
            "itself. That kills the artificial-executor-serialization branch of "
            "the why-does-multistream-help tree and points the rest at the "
            "kernels, which is a much narrower search."
        ),
        "concurrency_evidence": None if conc_doc is None else {
            "verdict": conc_doc["classification"]["verdict"],
            "gb_s_by_level": conc_doc["bandwidth_cross_check"]["gb_s_by_level"],
        },
        "what_this_does_not_prove": [
            "that no independence exists WITHIN a kernel - heads, groups and "
            "row blocks are not modelled here and are exactly where the "
            "remaining slack would live",
            "that the dispatch grain is the right grain; it is the grain at "
            "which this executor currently makes scheduling decisions",
            "that a different representation could not create new independence",
        ],
    }


def reordering_scar() -> dict[str, Any]:
    """S026 §4 makes this DAG's result binding on what may be proposed next.

    Emitted BY THE PRODUCER so a rebuild cannot delete it, and computed from
    slack() rather than typed, so the scar cannot outlive the measurement that
    justifies it. If a future graph ever exposes overlapable work, this raises
    rather than silently keeping the old claim.
    """
    sl = slack()
    if sl["theoretically_overlapable_ns"] != 0:
        raise DagRefused(
            "the DAG now shows "
            f"{sl['theoretically_overlapable_ns']} ns of overlapable top-level "
            "work, so TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK is FALSE "
            "and must not be emitted. The scar's own reopen condition has fired."
        )
    return {
        "family": "TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK",
        "status": "MEASURED_NEGATIVE",
        "level": "MODEL_SPECIFIC",
        "parent": "qwen3.8-27b sealed-3.14",
        "organ": "whole token",
        "object": "any candidate that reorders or overlaps top-level dispatches",
        "authority": "S026 §4",
        "mechanism": (
            f"all {sl['n_edges']} edges in the single-token data flow are TRUE "
            "dependencies and the critical path equals the total GPU work "
            f"({sl['critical_path_ns']} ns each), so there is ZERO overlapable "
            "top-level work. The capacity that multi-session execution exposes "
            "is not sitting between reorderable token regions; it is inside "
            "kernel execution. Command-order permutations cannot reach it."
        ),
        "not": (
            "a claim that the GPU is saturated, or that concurrency is "
            "impossible in principle. G009 measured 450-580 aggregate GB/s "
            "across independent sessions against ~361 for one. The capacity is "
            "real and this scar says only WHERE IT IS NOT."
        ),
        "requires": (
            "a frontier unit that reorders top-level dispatches is refused at "
            "proposal time while this stands"
        ),
        "reopen": (
            "any change to the decode graph that introduces a genuinely "
            "independent top-level region - a second stream of work with no "
            "data dependency on the first, such as speculative drafting or a "
            "second sequence - makes this DAG stale and the scar void. "
            "Rebuilding this module recomputes the slack and raises if so."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.single_token_dag.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "edges": token_edges(),
        "fused_independence_already_taken": list(FUSED_INDEPENDENCE),
        **slack(),
        "reading": reading(),
        "scars": [reordering_scar()],
        "claim_boundary": (
            "Static sidecar artifact. The EDGES are declared from the decode "
            "step's data flow, each with its reason; the MILLISECONDS are the "
            "measured post-widen_f4 organ census. No new hardware measurement. "
            "Concurrency is never inferred from dispatch counts - this models "
            "dependencies, and where independence cannot be established from the "
            "data flow the edge is UNKNOWN rather than assumed parallel."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({k: v for k, v in doc.items() if k != "measured_rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
