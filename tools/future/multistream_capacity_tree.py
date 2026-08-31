#!/usr/bin/env python3
"""G084: why does multistream help? Nine classes, and one already dead.

S025 §10, §13. G009 measured ~361 GB/s at n=1 against 449-580 aggregate, so the
machine has capacity a single stream does not use. The steer is explicit about
the trap: do NOT call the effect "serial dependency" merely because concurrency
helped. Prove the class.

Each class carries a maximum plausible payoff, the CHEAPEST discriminator that
would settle it, and reopen/kill criteria written before any of them run. Do not
launch all nine - S022 picks, and the point of the tree is that it can.

One class is already dead on landed evidence, which is what a tree is for.

    python3 tools/future/multistream_capacity_tree.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/multistream_capacity_tree.py"
RECEIPT_NAME = "MULTISTREAM_CAPACITY_TREE.json"

DAG_REL = "receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json"
CONC_REL = "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json"

OPEN = "OPEN"
KILLED = "KILLED"
SHARPENED = "SHARPENED"
BLOCKED_ON_TOOLING = "BLOCKED_ON_TOOLING"


class TreeRefused(RuntimeError):
    """A class without a discriminator, or a kill without evidence."""


def _cls(
    *,
    id: str,
    claim: str,
    max_payoff_ms: float | None,
    discriminator: str,
    kills_if: str,
    reopens_if: str,
    status: str = OPEN,
    killed_by: str | None = None,
    sharpened_by: str | None = None,
    not_answered_by: str | None = None,
) -> dict[str, Any]:
    if len(discriminator) < 40:
        raise TreeRefused(f"{id}: a class without a cheap discriminator is a guess")
    if not kills_if or not reopens_if:
        raise TreeRefused(f"{id}: kill and reopen criteria are written BEFORE the run")
    if status == KILLED and not killed_by:
        raise TreeRefused(f"{id}: killed without naming the evidence that killed it")
    if status in {SHARPENED, BLOCKED_ON_TOOLING} and not sharpened_by:
        raise TreeRefused(
            f"{id}: SHARPENED without naming the evidence that sharpened it. A "
            "class that narrowed for no stated reason is a class nobody can check."
        )
    return {
        "id": id,
        "claim": claim,
        "max_payoff_ms": max_payoff_ms,
        "cheapest_discriminator": discriminator,
        "kills_if": kills_if,
        "reopens_if": reopens_if,
        "status": status,
        "killed_by": killed_by,
        "sharpened_by": sharpened_by,
        # A lookalike receipt that does NOT answer this class. Naming it is
        # cheaper than someone re-deriving that it does not apply, and stops a
        # class being quietly closed by the wrong evidence.
        "not_answered_by": not_answered_by,
    }


def classes() -> list[dict[str, Any]]:
    return [
        _cls(
            id="H_artificial_executor_serialization",
            claim="the executor orders independent work serially, so one stream "
                  "under-fills the machine that two streams fill",
            max_payoff_ms=None,
            discriminator="build the single-token dependency DAG and count edges "
                          "that are NOT true data dependencies; any artificial "
                          "barrier is schedulable slack",
            kills_if="every edge in the token is a true data dependency and the "
                     "independent operations are already fused",
            reopens_if="a representation change creates new independence, or a "
                       "region is found that the DAG did not model",
            status=KILLED,
            killed_by=(
                "receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json: 11 edges, 11 "
                "true dependencies, 0 artificial barriers, 0 unknown, "
                "theoretically_overlapable_ns = 0. gate/up/swiglu, q/k/v, "
                "residual/rmsnorm and ba_to_decay are ALREADY fused, so the "
                "reorderable slack was taken, not left."
            ),
        ),
        _cls(
            id="B_occupancy_too_few_active_threads",
            claim="the decode kernels launch too little parallel work to keep the "
                  "memory system busy; a second stream supplies the missing waves",
            max_payoff_ms=None,
            discriminator="sweep threadgroup size and grid on ONE production "
                          "matvec at fixed bytes and fixed arithmetic; if effective "
                          "GB/s rises with occupancy alone the class is live",
            kills_if="GB/s is flat across a real occupancy sweep at identical bytes",
            reopens_if="a kernel is changed in a way that alters its register or "
                       "threadgroup footprint",
            status=SHARPENED,
            sharpened_by=(
                "receipts/future/GEOMETRY_TABLE.json already RAN this "
                "discriminator: threadgroup size and rows-per-threadgroup swept "
                "at fixed bytes across 11 production shapes. flat is FALSE with "
                "EIGHT distinct launch winners - mlp.gate_proj wins at tg128_r2, "
                "up_proj at tg256_r4, down_proj at tg512_r2, in_proj_qkvz at "
                "tg64_r2. So the kill criterion is not met and this class "
                "survives: geometry is a TABLE, not a constant, and the winner "
                "depends on the hot dimension. "
                "The spread is shape-dependent and mostly small - gate_proj "
                "220.8 against 217.4 for its runner-up, up_proj 188.8 against "
                "188.4 - but down_proj is 214.7 against 191.6, about 12%. "
                "WHAT IS NOT ESTABLISHED, and must not be assumed: whether "
                "production already selects the per-shape winner. That is the "
                "next question on this class, not a payoff to claim."
            ),
        ),
        _cls(
            id="E_memory_level_parallelism",
            claim="each thread has too few independent loads in flight, so latency "
                  "is exposed rather than hidden; independent streams interleave",
            max_payoff_ms=None,
            discriminator="unroll the per-thread load stride on one matvec so each "
                          "thread issues N independent loads before consuming any; "
                          "bytes and arithmetic identical, output bit-identical",
            kills_if="deeper per-thread load pipelining does not move GB/s",
            reopens_if="the representation changes the access pattern",
            not_answered_by=(
                "MLP_ISSUE_RATE_LADDER looks like this discriminator and is NOT. "
                "Its ilp2/ilp4/ilp8 vary INDEPENDENT ACCUMULATOR CHAINS - its own "
                "note reads 'production 16 FMA split into 2 independent acc "
                "chains' - which is arithmetic structure, not the load pipeline. "
                "Nothing landed has varied how many independent LOADS a thread "
                "holds in flight, so this class is untouched. Reading the ILP "
                "ladder as an answer here would be the easiest available mistake."
            ),
        ),
        _cls(
            id="D_instruction_dependency_chain",
            claim="the decode inner loop is a serial FMA/decode chain, so ALU "
                  "latency rather than memory is the stall",
            max_payoff_ms=None,
            discriminator="the matched ARM A pair already built: bytes identical, "
                          "arithmetic stripped. If stripping arithmetic moves the "
                          "time, the chain is arithmetic-bound",
            kills_if="ARM A (arithmetic stripped, bytes identical) does not move "
                     "meaningfully faster than production",
            reopens_if="a representation with a longer decode chain is proposed",
            status=SHARPENED,
            killed_by=None,
            sharpened_by=(
                "receipts/future/MLP_ALU_ROOFLINE.json. ARM A strips the "
                "arithmetic at IDENTICAL BYTES and MLP goes 329.6 -> 497.4 GB/s, "
                "a 1.5089x jump; DeltaNet qkvz goes 600.9 -> 943.2, 1.5697x. ARM "
                "B halves K and returns time/byte 1.0427 - bytes behave like "
                "bytes. The pre-registered rule reads that conjunction as MIXED: "
                "both terms are real. "
                "So this class is NOT killed and NOT alone. What it now carries "
                "is a NUMBER: about 1.5x is available at constant bytes, which "
                "sits inside the 1.24-1.60x the concurrency ladder measured from "
                "a completely different direction. Two independent probes, one "
                "magnitude. "
                "SELF_MEASURED_DIRTY and taken under load 8.41, so the ABSOLUTE "
                "GB/s are not promotable - but ARM A is a back-to-back RATIO at "
                "identical bytes, and ratios hold under load. "
                "REFINED BY MLP_ISSUE_RATE_LADDER, which splits the production "
                "16-FMA inner loop into 2, 4 and 8 INDEPENDENT accumulator "
                "chains, every variant bit-identical to production: 328.5, "
                "325.5 and 327.4 GB/s, flat inside 1%. So making the arithmetic "
                "more PARALLEL buys nothing while REMOVING it buys 1.51x. The "
                "term is arithmetic THROUGHPUT, not dependency-chain LATENCY - "
                "the machine is not stalling on the chain, it is spending real "
                "issue slots. That narrows the remedy: restructuring cannot "
                "recover this, only having LESS arithmetic can, which points at "
                "the decode tax of 1.3333 decode-FMA per weight byte inside a "
                "2.6667 total."
            ),
        ),
        _cls(
            id="C_register_limited_occupancy",
            claim="register pressure caps resident waves per core below what the "
                  "memory system needs",
            max_payoff_ms=None,
            discriminator="read the compiled pipeline's register footprint and "
                          "max threads per threadgroup from the Metal reflection, "
                          "then compare against the occupancy sweep in class B",
            kills_if="the achieved threadgroup size is already at the device "
                     "maximum and occupancy is not the binding term",
            reopens_if="a toolchain that reports the register count becomes "
                       "available, or a kernel rewrite changes the footprint",
            status=BLOCKED_ON_TOOLING,
            sharpened_by=(
                "UNANSWERABLE ON THIS TOOLCHAIN, and that is a finding rather "
                "than a gap. MLP_ALU_ROOFLINE's occupancy block reports "
                "registers_per_thread NULL with the reason: 'Metal pipeline "
                "state does not report register count on this toolchain; xcrun "
                "metal is not on PATH'. The discriminator asked for the register "
                "footprint from the reflection and the reflection does not carry "
                "it. "
                "What IS known: threads_per_threadgroup 128 against a device "
                "maximum of 1024 - occupancy_of_max_threads 0.125 - with 8704 "
                "threadgroups over 60 cores, so 145 per core. Whether 128 is "
                "chosen BECAUSE of register pressure or for another reason "
                "cannot be read here, so the class is neither confirmed nor "
                "killed. Same shape as the memory-traffic counter: this device "
                "does not expose what the question needs."
            ),
        ),
        _cls(
            id="F_cache_behaviour",
            claim="one stream's working set thrashes a level that two interleaved "
                  "streams happen to use better",
            max_payoff_ms=None,
            discriminator="vary only the per-launch working-set size at constant "
                          "bytes and constant arithmetic and look for a knee",
            kills_if="no knee appears across a working-set sweep spanning the "
                     "plausible cache sizes",
            reopens_if="a layout change alters locality",
        ),
        _cls(
            id="G_command_scheduling_topology",
            claim="one command queue or one encoder serialises submission in a way "
                  "two processes escape",
            max_payoff_ms=None,
            discriminator="the same token issued across two command queues versus "
                          "one, identical kernels and identical order, token output "
                          "bit-identical",
            kills_if="multiple queues do not change GPU time for identical work",
            reopens_if="the driver or OS scheduler changes",
        ),
        _cls(
            id="I_kernel_shape_underfill",
            claim="the decode shapes (batch 1, one row block) leave the kernel's "
                  "own tiling underfilled regardless of occupancy",
            max_payoff_ms=None,
            discriminator="run the SAME kernel at batch>1 on synthetic input and "
                          "compare GB/s per byte moved against the batch-1 rate",
            kills_if="per-byte rate is flat from batch 1 upward",
            reopens_if="a batched or speculative decode path is adopted",
        ),
        _cls(
            id="A_latency_hiding_generic",
            claim="something is exposing latency that a second stream hides, "
                  "without yet saying which latency",
            max_payoff_ms=None,
            discriminator="this class exists to be SPLIT, not tested: it is the "
                          "parent of B, C, D, E and F and must be replaced by one "
                          "of them before any payoff is claimed",
            kills_if="one of its children is confirmed, which makes the parent "
                     "redundant rather than false",
            reopens_if="every child is killed and the effect survives",
        ),
    ]


def summary() -> dict[str, Any]:
    cs = classes()
    dag_path = REPO / DAG_REL
    conc_path = REPO / CONC_REL
    if not dag_path.is_file():
        raise TreeRefused(f"{DAG_REL} is missing; class H cannot be killed without it")
    dag = json.loads(dag_path.read_text())
    if dag["theoretically_overlapable_ns"] != 0:
        raise TreeRefused(
            "the DAG reports overlapable work, so class H is NOT dead and this "
            "tree must not say it is"
        )
    conc = json.loads(conc_path.read_text()) if conc_path.is_file() else None
    return {
        "n_classes": len(cs),
        "n_killed": sum(1 for c in cs if c["status"] == KILLED),
        "n_open": sum(1 for c in cs if c["status"] == OPEN),
        "n_sharpened": sum(1 for c in cs if c["status"] == SHARPENED),
        "n_blocked_on_tooling": sum(1 for c in cs if c["status"] == BLOCKED_ON_TOOLING),
        "blocked_on_tooling": [c["id"] for c in cs if c["status"] == BLOCKED_ON_TOOLING],
        "killed": [c["id"] for c in cs if c["status"] == KILLED],
        "sharpened": [c["id"] for c in cs if c["status"] == SHARPENED],
        "next_cheapest": [c["id"] for c in cs if c["status"] == OPEN][:3],
        "the_effect_being_explained": None if conc is None else {
            "verdict": conc["classification"]["verdict"],
            "gb_s_by_level": conc["bandwidth_cross_check"]["gb_s_by_level"],
        },
        "do_not_call_it_serial_dependency": (
            "concurrency helping is the OBSERVATION, not the cause. Class H - the "
            "one that would have made 'serial dependency' literally true at the "
            "dispatch grain - is the one class already dead. What remains is "
            "inside the kernels, and naming it before measuring would be the "
            "error this tree exists to prevent."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.multistream_capacity_tree.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "classes": classes(),
        "summary": summary(),
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. Every class carries "
            "a discriminator written BEFORE any of them run, and the one KILLED "
            "class names the landed receipt that killed it. Maximum payoffs are "
            "None because none has been bounded yet, and a null is honest where a "
            "number would be invented."
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
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
