"""The CUDA Capability Ledger. FRONT B (G044, steer S015).

The authoritative inventory of useful CUDA/NVIDIA compute capabilities, with the
twelve fields the steer names on every entry. The validator refuses an incomplete
entry, because a ledger with holes silently reads as a ledger with coverage.

Two disciplines are enforced in code rather than by intention:

  1. NEVER "CUDA PARITY" UNQUALIFIED. A semantic_gap of NONE may only be claimed
     with evidence attached; without it the entry must say UNKNOWN.
  2. A performance_gap may not be stated without a receipt path. This program has
     measured exactly four things, and everything else says so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.cuda_capability_ledger.v1"

CLASSES = ("COMPILER", "EXECUTION", "MEMORY", "MATH", "RUNTIME", "PROFILING",
           "DEBUGGING", "MULTI_DEVICE")

FIELDS = ("capability_id", "cuda_mechanism", "why_it_exists", "underlying_problem",
          "apple_equivalent", "hawking_equivalent", "semantic_gap", "performance_gap",
          "priority", "test_corpus", "current_winner", "remaining_limitation")

GAP = ("NONE", "PARTIAL", "LARGE", "UNKNOWN", "DELETED_BY_UNIFIED_MEMORY")
PRIORITY = ("P0", "P1", "P2", "P3")


class CCLError(ValueError):
    pass


def entry(**kw: Any) -> dict[str, Any]:
    missing = [f for f in FIELDS if f not in kw]
    if missing:
        raise CCLError(f"entry {kw.get('capability_id')!r} is missing {missing}; a "
                       f"ledger with holes reads as a ledger with coverage")
    if kw["capability_id"].split(".")[0] not in CLASSES:
        raise CCLError(f"{kw['capability_id']!r} must start with one of {CLASSES}")
    if kw["semantic_gap"] not in GAP:
        raise CCLError(f"semantic_gap {kw['semantic_gap']!r} not in {GAP}")
    if kw["priority"] not in PRIORITY:
        raise CCLError(f"priority {kw['priority']!r} not in {PRIORITY}")
    if kw["semantic_gap"] == "NONE" and not kw.get("evidence"):
        raise CCLError(f"{kw['capability_id']}: a semantic gap of NONE is a PARITY "
                       f"CLAIM and needs evidence; say UNKNOWN instead")
    pg = kw["performance_gap"]
    if isinstance(pg, dict) and pg.get("measured") and not pg.get("receipt"):
        raise CCLError(f"{kw['capability_id']}: a measured performance gap needs a "
                       f"receipt path")
    return dict(kw)


def unmeasured(reason: str) -> dict[str, Any]:
    return {"measured": False, "reason": reason}


def measured(value: str, receipt: str) -> dict[str, Any]:
    return {"measured": True, "value": value, "receipt": receipt}


def recount(led: dict[str, Any]) -> dict[str, Any]:
    """Recompute every derived count from the entries.

    Exists because an edit that changed one entry's performance_gap without
    recomputing left the ledger claiming 10 measured gaps when it had 11 -- the
    ledger disagreeing with itself, which is precisely the failure its own validator
    is for. Any writer must call this rather than maintaining counts by hand.
    """
    led["count"] = len(led["entries"])
    led["performance_gaps_measured"] = sum(
        1 for e in led["entries"]
        if isinstance(e.get("performance_gap"), dict)
        and e["performance_gap"].get("measured"))
    led["performance_gaps_unmeasured"] = led["count"] - led["performance_gaps_measured"]
    by: dict[str, int] = {}
    for e in led["entries"]:
        c = e["capability_id"].split(".")[0]
        by[c] = by.get(c, 0) + 1
    led["by_class"] = by
    led["classes_with_no_entry"] = [c for c in CLASSES if c not in by]
    return led


def build(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [e["capability_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise CCLError("duplicate capability_id")
    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["capability_id"].split(".")[0]] = \
            by_class.get(e["capability_id"].split(".")[0], 0) + 1
    n_measured = sum(1 for e in entries
                     if isinstance(e["performance_gap"], dict)
                     and e["performance_gap"].get("measured"))
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries,
        "count": len(entries),
        "by_class": by_class,
        "classes_with_no_entry": [c for c in CLASSES if c not in by_class],
        "performance_gaps_measured": n_measured,
        "performance_gaps_unmeasured": len(entries) - n_measured,
        "parity_claim": ("NOT CLAIMED. This is a capability inventory, not a parity "
                         "statement. No entry may report a semantic gap of NONE "
                         "without evidence, and the steer forbids the phrase 'CUDA "
                         "parity' unqualified."),
        "coverage_honesty": ("The CUDA surface is vastly larger than this. These are "
                             "the capabilities reachable or relevant from what has "
                             "actually been built here; absence from this ledger means "
                             "NOT YET STUDIED, never NOT NEEDED."),
    }


# =============================================================================
# CENSUS EXPANSION (G055): PROGRAMMING MODEL, MEMORY, EXECUTION/RUNTIME,
# COMPILER, MATH ECOSYSTEM, PROFILING -- solved-problem archaeology.
#
# For each entry the interesting field is WHAT PROBLEM the CUDA mechanism
# solves and whether Apple/Metal has that problem at all. Several CUDA
# mechanisms exist only because a discrete GPU has separate physical memory;
# on unified memory the honest answer is semantic_gap=DELETED_BY_UNIFIED_MEMORY,
# and that is a finding, not a hole.
#
# Absence from this census (and from the whole ledger) means NOT YET STUDIED,
# never NOT NEEDED -- see receipts/headless/CUDA_CAPABILITY_CENSUS.json for the
# explicit list of what this pass did not reach.
#
# MULTI_DEVICE.peer_access is deliberately NOT extended here: it is blocked on
# hardware (no NVIDIA hardware exists on this machine) and stays that way.
# NCCL collectives, which need a second device just as much, are filed under
# MATH instead of MULTI_DEVICE for exactly this reason.
# =============================================================================

NEW_ENTRIES: list[dict[str, Any]] = [

    # --- PROGRAMMING MODEL -------------------------------------------------

    entry(
        capability_id="EXECUTION.cooperative_groups",
        cuda_mechanism="cooperative_groups::thread_block/tile/grid_group: a C++ "
            "API that names an explicit group of cooperating threads (a subset "
            "of a warp, a whole block, or the whole grid via "
            "cudaLaunchCooperativeKernel) and gives it its own sync() and "
            "collective operations",
        why_it_exists="raw __syncthreads()/__shfl_sync() require every caller "
            "to agree by convention on which threads participate; cooperative "
            "groups make the participating set an explicit, type-checked "
            "object instead of an implicit convention",
        underlying_problem="expressing 'these specific threads act together' "
            "as a first-class value instead of an implicit contract enforced "
            "by discipline",
        apple_equivalent="simdgroups (fixed 32-wide) and threadgroups are the "
            "two built-in granularities MSL exposes; there is no "
            "user-composable arbitrary-tile group object between them",
        hawking_equivalent="none; AIR's barrier and atomic strategies each "
            "hard-code which scope they operate at rather than taking a group "
            "as a parameter",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no AIR program has ever needed a group "
            "finer than 'the whole threadgroup' or coarser than 'one "
            "simdgroup', so nothing has been built to time"),
        priority="P2",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="grid-wide cooperative groups "
            "(cudaLaunchCooperativeKernel, letting an entire grid "
            "synchronize) has no Metal analog investigated at all -- indirect "
            "command buffers and AirGraph's DAG submission are the closest "
            "adjacent mechanisms and neither has been evaluated for this "
            "specific purpose",
    ),

    entry(
        capability_id="EXECUTION.warp_collectives",
        cuda_mechanism="__shfl_sync/__shfl_down_sync/__shfl_xor_sync "
            "(lane-to-lane register exchange), __ballot_sync/__activemask "
            "(per-lane predicate to bitmask), __all_sync/__any_sync (vote), "
            "__match_any_sync/__match_all_sync (find lanes agreeing on a "
            "value) -- all operating across the 32 lanes of one warp without "
            "touching shared memory",
        why_it_exists="many reduction/broadcast/compaction patterns only need "
            "to move data among the 32 lanes already executing in lockstep; "
            "going through shared memory for that is a barrier and a memory "
            "round trip for something the hardware can do in one instruction",
        underlying_problem="intra-warp data exchange and cross-lane agreement "
            "without paying for shared memory and a barrier",
        apple_equivalent="simd_shuffle/simd_shuffle_down/simd_shuffle_xor "
            "cover the shuffle family directly; simd_ballot and "
            "simd_all/simd_any cover ballot and vote; an MSL equivalent to "
            "match_any/match_all (which lanes share MY value) was not found "
            "in the reference material available here",
        hawking_equivalent="AIR's ONLY warp-level primitive is "
            "simdgroup_float8x8 matrix multiply-accumulate "
            "(EXECUTION.warp_primitives); no shuffle, ballot, vote, or match "
            "instruction has been emitted by AIR at all",
        semantic_gap="LARGE",
        performance_gap=unmeasured("this is the exact gap "
            "EXECUTION.warp_primitives already named in its own "
            "remaining_limitation ('shuffle, ballot and vote are still "
            "untouched') and it remains untouched"),
        priority="P1",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="match_any/match_all have not been confirmed to "
            "exist in MSL at all from documentation available on this "
            "machine -- marked here as UNVERIFIED rather than assumed "
            "present or absent",
    ),

    entry(
        capability_id="MEMORY.constant_memory",
        cuda_mechanism="__constant__ memory: a small (64KB) cached read-only "
            "region broadcast to all threads in a warp reading the same "
            "address",
        why_it_exists="a warp reading one shared address from constant cache "
            "costs one broadcast read instead of contending for the general "
            "L1/L2 path",
        underlying_problem="many threads in lockstep reading the SAME small "
            "read-only value (a coefficient table, a kernel parameter) "
            "should not each pay a general memory load",
        apple_equivalent="the `constant` address space in MSL, backed by the "
            "same read-only broadcast path",
        hawking_equivalent="AIR emits ordinary `device`-space buffers for "
            "every kernel argument; nothing marks a buffer `constant` yet",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("AIR has never emitted a "
            "`constant`-address-space parameter, so there is nothing built "
            "to time against a device-space equivalent"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN -- never measured",
        remaining_limitation="Metal's constant address space is a "
            "straightforward capability that AIR simply has not reached for; "
            "unlike cp.async or managed memory this is not a hardware-model "
            "mismatch, it is an unbuilt door",
    ),

    entry(
        capability_id="EXECUTION.register_pressure_and_occupancy",
        cuda_mechanism="the CUDA occupancy model: registers-per-thread and "
            "shared-memory-per-block are FIXED hardware budgets per SM; a "
            "kernel using more registers fits FEWER concurrent warps, "
            "trading per-thread speed for the SM's ability to hide latency "
            "across warps. cudaOccupancyMaxPotentialBlockSize computes the "
            "tradeoff",
        why_it_exists="a compiler choosing to keep more values in registers "
            "speeds up ONE warp but can starve the SM of enough OTHER warps "
            "to hide memory latency; the two effects pull opposite "
            "directions and the optimum is workload-dependent",
        underlying_problem="a fixed per-core resource budget shared between "
            "per-thread speed and cross-thread latency hiding",
        apple_equivalent="Apple GPUs have an analogous occupancy limit "
            "(threads-per-threadgroup vs registers vs threadgroup memory, "
            "reported by [[max_total_threads_per_threadgroup]] and the "
            "pipeline state's maxTotalThreadsPerThreadgroup), but there is "
            "no evidence in this repo that anyone has measured Hawking's own "
            "kernels against it",
        hawking_equivalent="ACCELERATOR_CONCURRENCY_SWEEP.json measures a "
            "related but DIFFERENT axis: OS-process-level concurrency of a "
            "whole kernel (1/2/4/8 processes), not in-kernel "
            "register/occupancy tradeoff within one dispatch",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no AIR kernel has had its register usage "
            "or occupancy queried or varied; the closest existing data, the "
            "concurrency sweep, measures a different axis and its own "
            "reconciliation note says explicitly not to conflate the two "
            "regimes"),
        priority="P1",
        test_corpus="none for occupancy specifically; "
            "ACCELERATOR_CONCURRENCY_SWEEP.json for the adjacent "
            "process-concurrency question",
        current_winner="UNKNOWN",
        remaining_limitation="xcrun metal is ABSENT on this machine (already "
            "named in COMPILER.specialization_constants), which blocks "
            "inspecting generated register counts directly; occupancy could "
            "only be measured indirectly by varying threadgroup size and "
            "timing, which has not been done",
    ),

    entry(
        capability_id="EXECUTION.persistent_kernels",
        cuda_mechanism="a kernel launched ONCE that internally loops, "
            "pulling work items from a queue instead of returning after "
            "processing one item -- avoiding per-item launch overhead "
            "entirely, at the cost of managing the work queue and "
            "termination inside the kernel",
        why_it_exists="even with CUDA Graphs cutting per-launch CPU cost, "
            "each launch still has SOME fixed overhead; a persistent kernel "
            "amortizes that overhead over an entire workload instead of per "
            "item",
        underlying_problem="launch overhead when the number of small work "
            "items is very large relative to the work per item",
        apple_equivalent="no restriction preventing an MSL kernel from "
            "looping internally and reading a device-memory work queue; "
            "nothing Metal-specific blocks the pattern",
        hawking_equivalent="none; every AIR-generated kernel processes "
            "exactly the launch's index space once and returns. "
            "RUNTIME.graphs already found that batching k dispatches into "
            "ONE submission cuts marginal per-dispatch cost 15.5x -- the "
            "same problem persistent kernels solve, attacked from the "
            "submission side rather than by keeping one kernel resident",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no persistent-kernel AIR program has "
            "been written; RUNTIME.graphs' 15.5x marginal-cost reduction is "
            "evidence the same PROBLEM is worth solving here, not a "
            "measurement of THIS mechanism"),
        priority="P2",
        test_corpus="none",
        current_winner="RUNTIME.graphs' batched-submission approach, by "
            "default, since it is what has actually been built and "
            "measured",
        remaining_limitation="whether a persistent kernel would beat "
            "AirGraph's already-measured 15.5x marginal-cost reduction, or "
            "whether the two attack the same waste and the cheaper one "
            "(graph submission, no new kernel-internal queue/termination "
            "logic) already captures most of the win, is unmeasured "
            "speculation either way",
    ),

    entry(
        capability_id="EXECUTION.dynamic_shared_memory",
        cuda_mechanism="shared memory whose SIZE is set at LAUNCH time (the "
            "third kernel<<<>>> launch parameter) rather than compiled into "
            "the kernel, letting one kernel binary serve multiple tile sizes",
        why_it_exists="compiling a separate kernel per tile size multiplies "
            "binary count and compile time for what is otherwise identical "
            "code",
        underlying_problem="a resource whose size should vary per-launch "
            "without recompiling the kernel",
        apple_equivalent="MTLComputeCommandEncoder."
            "setThreadgroupMemoryLength(_:index:) sets threadgroup memory "
            "size per-dispatch, the same 'declared at compile time in the "
            "shader but SIZED at dispatch time' pattern",
        hawking_equivalent="tools/accelerator/gemm.py's tiled GEMM declares "
            "one fixed tile size in its MSL source; AIR has never emitted a "
            "threadgroup array with a launch-time-determined length",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no AIR program varies its threadgroup "
            "memory size at dispatch time to compare against a fixed-size "
            "kernel"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="the Metal-side mechanism is confirmed to exist "
            "(setThreadgroupMemoryLength is a documented API) but AIR has no "
            "code path that emits a variable-length threadgroup array or "
            "calls it, so this is an unbuilt door, not a hardware gap",
    ),

    entry(
        capability_id="MEMORY.async_copy_pipeline",
        cuda_mechanism="cp.async / cuda::memcpy_async: a warp issues a "
            "global->shared copy that proceeds in the background without "
            "occupying a thread for the transfer, so compute on a PREVIOUS "
            "tile can overlap the copy of the NEXT tile (software "
            "pipelining)",
        why_it_exists="shared memory is filled by explicit thread-issued "
            "loads; without an async path, filling the next tile blocks on "
            "every thread completing its load before compute can proceed",
        underlying_problem="hiding global memory latency behind compute "
            "within a single kernel body, across pipeline stages",
        apple_equivalent="none at the ISA level -- MSL has no instruction "
            "that starts an async global->threadgroup copy and returns "
            "control to the thread immediately; threadgroup memory is "
            "filled by ordinary (synchronous, from the thread's point of "
            "view) loads and stores",
        hawking_equivalent="none; tools/accelerator/gemm.py's tiled GEMM "
            "loads each tile synchronously with a barrier between load and "
            "compute phases",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no software-pipelined kernel exists "
            "here to time against the single-buffered tiled GEMM"),
        priority="P1",
        test_corpus="none",
        current_winner="NOT APPLICABLE -- no async-copy-pipelined kernel "
            "exists on either side to compare",
        remaining_limitation="whether Apple's GPU scheduler already hides "
            "this latency by running OTHER threadgroups while one stalls on "
            "a load (occupancy-driven latency hiding, the same mechanism "
            "CUDA itself relies on when cp.async is unavailable) is "
            "unmeasured; cp.async's real edge is letting a SINGLE "
            "threadgroup pipeline within itself, which matters most when "
            "occupancy is too low to hide latency across threadgroups",
    ),

    entry(
        capability_id="EXECUTION.thread_block_clusters",
        cuda_mechanism="thread block clusters (Hopper, compute capability "
            "9.0+): a group of thread blocks guaranteed to be co-scheduled "
            "on the same GPU Processing Cluster, able to access each "
            "other's shared memory directly via 'distributed shared memory' "
            "without going through global memory",
        why_it_exists="some algorithms (e.g. large-tile GEMM) want "
            "cooperation at a granularity BIGGER than one block's shared "
            "memory but smaller than the whole grid; clusters give that a "
            "hardware-guaranteed co-location",
        underlying_problem="cooperation and shared-memory-like sharing "
            "across MULTIPLE thread blocks without falling back to slow "
            "global memory",
        apple_equivalent="none located; Apple GPU documentation available "
            "on this machine describes threadgroups and simdgroups only, no "
            "intermediate co-scheduled cluster of threadgroups with a "
            "shared distributed memory space",
        hawking_equivalent="none",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no cluster-level mechanism exists on "
            "either side of the comparison to time"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="this is also the NEWEST CUDA mechanism in "
            "this census (Hopper-only, compute capability 9.0+), so its "
            "absence here may reflect that it is not yet load-bearing even "
            "in the CUDA ecosystem outside frontier GEMM/attention kernels, "
            "not only that Apple lacks it",
    ),

    entry(
        capability_id="EXECUTION.device_side_launch",
        cuda_mechanism="dynamic parallelism (cudaLaunchDevice / launching a "
            "kernel FROM inside another kernel, no host round trip): a "
            "running kernel can spawn child kernels based on data it just "
            "computed",
        why_it_exists="some algorithms (adaptive mesh refinement, graph "
            "traversal with data-dependent branching factor) don't know how "
            "much work to launch until partway through -- waiting for the "
            "host to see the result and issue a new launch adds a full "
            "round trip",
        underlying_problem="launching new GPU work based on a decision made "
            "ON the GPU, without returning control to the host first",
        apple_equivalent="none; Metal has no device-side dispatch of new "
            "compute work from within a running kernel -- indirect command "
            "buffers let the HOST encode commands whose ARGUMENTS a kernel "
            "fills in, which is close but the launch decision and encoding "
            "still originate host-side or from a pre-encoded buffer, not "
            "from arbitrary in-kernel control flow",
        hawking_equivalent="none; every AIR launch is decided and issued "
            "from Python/MLX before the kernel runs",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no data-dependent-launch-count workload "
            "has been attempted on either side"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="indirect command buffers (which DO exist on "
            "Metal and are already the answer this ledger gives for "
            "RUNTIME.graphs) are the nearest adjacent mechanism and have "
            "not been evaluated specifically for whether they can "
            "approximate data-dependent launch counts",
    ),

    # --- MEMORY --------------------------------------------------------------

    entry(
        capability_id="MEMORY.stream_ordered_allocation",
        cuda_mechanism="cudaMallocAsync/cudaFreeAsync: allocation and free "
            "are enqueued on a stream like any other operation, so a hot "
            "loop that allocates scratch space every iteration does not pay "
            "a synchronizing device-wide allocator call",
        why_it_exists="cudaMalloc is a synchronizing call across the whole "
            "device; a kernel launched every iteration that also allocates "
            "scratch on every iteration serializes on the allocator",
        underlying_problem="allocation in a tight, repeated launch loop "
            "should not be slower than the compute it scaffolds",
        apple_equivalent="none exposed directly by Metal to app code, but "
            "MLX has its own caching/pool allocator: `mx.array` creation "
            "does not call the OS allocator per-array in steady state",
        hawking_equivalent="none built directly; Hawking rides MLX's "
            "allocator for every mx.array and has never measured whether it "
            "is actually stream-ordered or just cached",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("MLX's allocator internals have not been "
            "inspected or benchmarked in isolation from the kernels that "
            "use it"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="this is a property of MLX's C++ runtime, not "
            "of anything Hawking built; claiming a gap size here would be "
            "claiming knowledge of MLX internals that has not been verified "
            "by reading MLX's source or timing its allocator directly",
    ),

    entry(
        capability_id="MEMORY.memory_pools",
        cuda_mechanism="cudaMemPool_t: an explicit pool object with a "
            "release threshold, trim API, and the ability to share one pool "
            "across streams or export it for IPC",
        why_it_exists="letting an async allocator return device memory to "
            "the OS/driver only when actually idle, instead of on every "
            "free, avoids alloc/free churn at the cost of holding memory a "
            "program is not using this instant",
        underlying_problem="trading peak memory headroom against allocator "
            "churn is a POLICY, and a single implicit pool cannot serve "
            "every workload's policy",
        apple_equivalent="MTLHeap gives explicit sub-allocation with "
            "app-controlled lifetime; MLX exposes `mx.set_memory_limit` / "
            "`mx.set_cache_limit` as a coarser, single global policy knob "
            "rather than a per-pool object",
        hawking_equivalent="none; Hawking has never set or measured MLX's "
            "cache limit against a workload",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no experiment has varied MLX's "
            "cache/memory limit and measured the effect"),
        priority="P3",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="a single global limit is a coarser instrument "
            "than a per-pool object with a release threshold; whether that "
            "coarseness ever matters on a 96GB unified-memory machine with "
            "no allocator contention across processes is untested",
    ),

    entry(
        capability_id="MEMORY.graph_memory_allocations",
        cuda_mechanism="cudaGraphAddMemAllocNode / MemFreeNode: allocation "
            "and free become NODES inside a captured graph, so the "
            "allocator's lifetime tracking is part of the graph's own "
            "dependency structure and memory can be reused across graph "
            "replays without re-issuing malloc",
        why_it_exists="once a command sequence is captured as a graph, an "
            "allocation the graph makes every replay should not require "
            "leaving the graph to call an allocator",
        underlying_problem="resource lifetime that repeats identically on "
            "every replay of a fixed command sequence",
        apple_equivalent="none; indirect/reusable command buffers do not "
            "model allocation as a node type",
        hawking_equivalent="none; AirGraph (RUNTIME.graphs) has no "
            "allocation node and no memory-node concept at all",
        semantic_gap="LARGE",
        performance_gap=unmeasured("AirGraph re-records its DAG per call "
            "already (see RUNTIME.graphs' remaining_limitation); adding an "
            "allocation-node concept has not been attempted"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="depends on RUNTIME.graphs first growing graph "
            "instantiation/update reuse; building an allocation-node "
            "concept on top of a graph that is re-recorded every call would "
            "not be measuring what CUDA's version measures",
    ),

    entry(
        capability_id="MEMORY.managed_unified_memory",
        cuda_mechanism="cudaMallocManaged: a single pointer valid on host "
            "and device, with the driver migrating pages on demand between "
            "them at first-touch",
        why_it_exists="a discrete GPU has separate physical memory; managed "
            "memory hides the two address spaces behind page-fault-driven "
            "migration so the programmer writes one pointer instead of a "
            "host/device pair",
        underlying_problem="presenting ONE address space when the hardware "
            "actually has TWO physical memories",
        apple_equivalent="none needed; CPU and GPU already share one "
            "physical memory and one address space, so there is no second "
            "copy to migrate",
        hawking_equivalent="every mx.array already lives in the one memory; "
            "MEMORY.host_device_copy's resident-operand path is the "
            "measured instance of this same fact",
        semantic_gap="DELETED_BY_UNIFIED_MEMORY",
        performance_gap=measured("the migration cost this API exists to "
            "hide is not paid here at all: keeping the WorkUnit gate and "
            "its operands device-resident (never round-tripping through "
            "the host) measured 635,422 -> 2,151,304 WorkUnits/hour, a "
            "3.39x end-to-end gain with every arm inside a 10% IQR gate",
            "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json"),
        priority="P1",
        test_corpus="the three-arm A/B/C WorkUnit residency test of "
            "ACCELERATOR_DEVICE_RESIDENT",
        current_winner="Hawking's resident-operand path; there is no "
            "migration to compare it against because there is no second "
            "memory",
        remaining_limitation="this entry reuses MEMORY.host_device_copy's "
            "measurement rather than a fresh one specific to page-fault-"
            "driven migration semantics (first-touch, oversubscription, "
            "prefetch hints) -- those semantics have no analog to test "
            "since there is only one memory, so 'deleted' is a statement "
            "about the API's REASON FOR EXISTING, not a claim that every "
            "corner of its behavior was independently verified absent",
        evidence="receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json",
    ),

    entry(
        capability_id="MEMORY.pinned_host_memory",
        cuda_mechanism="cudaHostAlloc/cudaMallocHost: page-locked host "
            "memory that the DMA engine can transfer directly, without CPU "
            "staging through a bounce buffer, and which enables true async "
            "cudaMemcpyAsync",
        why_it_exists="pageable host memory can be relocated or swapped by "
            "the OS mid-DMA, so the driver must stage it through an "
            "internal pinned buffer first, which is slower and cannot "
            "overlap with a kernel",
        underlying_problem="a DMA engine needs a physical address that will "
            "not move; ordinary OS-managed memory does not guarantee that",
        apple_equivalent="none needed; there is no host-to-device DMA to "
            "enable, because there is no host-to-device copy at all",
        hawking_equivalent="none; nothing pins memory because nothing needs "
            "a DMA-safe address",
        semantic_gap="DELETED_BY_UNIFIED_MEMORY",
        performance_gap=unmeasured("there is no copy here to speed up by "
            "pinning its source; benchmarking pinning would require "
            "inventing a copy that would not otherwise exist"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="macOS memory compression and paging under "
            "real memory pressure is a distinct, unstudied concern -- a "
            "large mx.array COULD in principle be paged out under pressure "
            "the way any process memory can, and whether that ever matters "
            "here (a 96GB box running one workload at a time) is "
            "unmeasured, not ruled out",
    ),

    entry(
        capability_id="MEMORY.mapped_memory",
        cuda_mechanism="cudaHostAllocMapped ('zero-copy'): pinned host "
            "memory given a matching DEVICE pointer, so a kernel can read "
            "it directly over PCIe without an explicit cudaMemcpy -- at "
            "PCIe bandwidth and latency per access, not device-memory "
            "bandwidth",
        why_it_exists="some access patterns touch host data once or rarely "
            "enough that a full copy costs more than reading it in place, "
            "slowly, from the device",
        underlying_problem="avoiding a copy for infrequently-accessed host "
            "data, at the cost of every access paying interconnect latency",
        apple_equivalent="this IS, structurally, what every Apple "
            "unified-memory allocation already is -- a single pointer both "
            "sides can read -- except unconditional and at full "
            "device-memory bandwidth rather than PCIe-limited",
        hawking_equivalent="every mx.array",
        semantic_gap="DELETED_BY_UNIFIED_MEMORY",
        performance_gap=unmeasured("comparing PCIe-limited zero-copy "
            "bandwidth against Apple's full-bandwidth unified access would "
            "require NVIDIA hardware to measure the CUDA side of the "
            "comparison, which does not exist on this machine"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE -- no comparison is measurable "
            "without the CUDA side",
        remaining_limitation="the claim that Apple's version is "
            "unconditionally better is a REASONING claim, not a measured "
            "one; it has not been verified against any real CUDA zero-copy "
            "benchmark, only argued from the two architectures' known "
            "bandwidth numbers",
    ),

    entry(
        capability_id="MEMORY.virtual_memory_management",
        cuda_mechanism="the low-level VMM API "
            "(cuMemAddressReserve/cuMemCreate/cuMemMap/cuMemSetAccess): "
            "reserve a virtual address range and map physical chunks into "
            "it lazily, letting an allocator grow a buffer without a "
            "realloc-and-copy and share physical pages across virtual "
            "mappings",
        why_it_exists="a device has its own page tables separate from the "
            "host's; growing a device allocation in place, or exporting one "
            "physical allocation under two different device mappings, "
            "needs an API that operates on the DEVICE's virtual memory",
        underlying_problem="two separable problems bundled in one API: (a) "
            "growable allocation without copy, and (b) a device having its "
            "own address space to manage at all",
        apple_equivalent="problem (b) does not exist -- a device pointer "
            "already IS a process virtual address here, managed by the one "
            "kernel VM system every process uses; problem (a) is a real, "
            "general concern (growable buffers without a realloc-copy) that "
            "mmap/vm_allocate could serve on macOS but nothing in Hawking "
            "uses for this purpose",
        hawking_equivalent="none; every mx.array is allocated at a fixed "
            "size for its lifetime",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no growable-allocation experiment has "
            "been built or timed"),
        priority="P3",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="conflating 'device has separate address "
            "space' (deleted here) with 'growable allocation is useful' "
            "(not deleted, just unbuilt) would overstate the gap in either "
            "direction; this entry keeps them separate on purpose",
    ),

    entry(
        capability_id="MEMORY.ipc_memory",
        cuda_mechanism="cudaIpcGetMemHandle/cudaIpcOpenMemHandle: export a "
            "device allocation as a handle another PROCESS can map into its "
            "own address space, for multi-process pipelines sharing one "
            "GPU's memory without a host round trip",
        why_it_exists="two processes cannot normally see each other's "
            "device allocations; IPC handles let a producer/consumer pair "
            "on separate processes share device memory directly",
        underlying_problem="cross-process sharing of GPU-resident data "
            "without serializing it through the host",
        apple_equivalent="macOS supports cross-process shared memory (POSIX "
            "shm, Mach ports) for host memory generally, but nothing "
            "Metal-specific for a GPU buffer handle shared cross-process "
            "has been checked",
        hawking_equivalent="none; Hawking is single-process",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no multi-process Hawking pipeline "
            "exists to need this"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE -- single-process today",
        remaining_limitation="whether MTLBuffer/MTLHeap objects can be "
            "shared cross-process on macOS at all, and at what cost, is "
            "unresearched; this entry states that honestly rather than "
            "guessing",
    ),

    entry(
        capability_id="MEMORY.prefetch_and_advise",
        cuda_mechanism="cudaMemPrefetchAsync/cudaMemAdvise: hints that tell "
            "the managed-memory migration engine to move pages to a given "
            "device BEFORE they are touched, or to bias its migration "
            "heuristic (read-mostly, preferred location, accessed-by)",
        why_it_exists="first-touch page-fault migration is reactive and "
            "pays a fault latency; prefetch hints make migration proactive "
            "for a known future access pattern",
        underlying_problem="hiding page-migration latency behind a known "
            "future access, and steering a heuristic that would otherwise "
            "guess",
        apple_equivalent="none needed; there is no migration engine to hint "
            "because there is nothing to migrate",
        hawking_equivalent="none",
        semantic_gap="DELETED_BY_UNIFIED_MEMORY",
        performance_gap=unmeasured("this API's entire value proposition is "
            "hiding a migration cost that does not exist here to hide"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="same caveat as MEMORY.managed_unified_memory: "
            "macOS-level paging/compression under memory pressure is a "
            "distinct mechanism this entry does not claim to have ruled "
            "out, only that CUDA's specific migration-hint problem does not "
            "exist here",
    ),

    entry(
        capability_id="MEMORY.copy_engines",
        cuda_mechanism="independent DMA copy engines (separate from the "
            "SM) that let a device run one or more memcpy operations "
            "CONCURRENTLY with kernel execution, when using multiple "
            "streams",
        why_it_exists="a discrete GPU's PCIe transfer and its compute units "
            "are physically separate hardware; without a dedicated copy "
            "engine, moving the next batch would compete with the SMs for "
            "the memory controller in a way that blocks compute",
        underlying_problem="overlapping a transfer with unrelated compute "
            "on a device where transfer and compute are different engines",
        apple_equivalent="none needed as a DISTINCT capability; there is no "
            "transfer to overlap with compute because there is no copy. "
            "Whatever concurrency exists between multiple GPU command "
            "encoders is a general scheduling question already covered by "
            "RUNTIME.streams_and_events, not a copy-engine question",
        hawking_equivalent="none; nothing here has a transfer phase to "
            "schedule",
        semantic_gap="DELETED_BY_UNIFIED_MEMORY",
        performance_gap=unmeasured("there is no transfer phase to overlap"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="this collapses into MEMORY.host_device_copy's "
            "finding; it is listed separately because the CUDA mechanism (a "
            "distinct hardware engine) is conceptually distinct from the "
            "API (cudaMemcpy) even though the Apple-side answer is the same "
            "for both",
    ),

    entry(
        capability_id="MEMORY.memory_ordering_and_scopes",
        cuda_mechanism="__threadfence()/__threadfence_block()/"
            "__threadfence_system() and, in newer CUDA, explicit "
            "acquire/release atomics: memory ordering guarantees scoped to "
            "a thread block, a device, or the whole system (including the "
            "host)",
        why_it_exists="a GPU has many independent cores without a single "
            "global clock; without an explicit fence, one thread's write is "
            "not guaranteed visible to another thread (in a different "
            "block, or on the host) in program order",
        underlying_problem="making a write by one execution context "
            "visible, in order, to another context that did not itself "
            "issue the write",
        apple_equivalent="Metal's memory_coherence / mem_flags on "
            "threadgroup_barrier cover the BLOCK scope; MSL exposes "
            "memory_order_relaxed/acquire/release for atomics but Apple GPU "
            "documentation of a DEVICE-wide or SYSTEM-wide (CPU-visible) "
            "fence has not been located",
        hawking_equivalent="AIR's barrier_msl instruction (from "
            "ACCELERATOR_BARRIER_SCOPES) lowers a SIMDGROUP-scope barrier "
            "to a real MSL instruction; DEVICE scope is explicitly refused "
            "rather than silently downgraded",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("ACCELERATOR_BARRIER_SCOPES.json is "
            "correctness/IR-structure work only -- its own claim_boundary "
            "states nothing was timed there ('no_performance_claim: "
            "nothing was timed; this is correctness and IR structure "
            "only')"),
        priority="P1",
        test_corpus="ACCELERATOR_BARRIER_SCOPES.json's SIMDGROUP-lowering "
            "and DEVICE-refusal cases",
        current_winner="UNKNOWN -- SIMDGROUP lowers, DEVICE is refused "
            "rather than emulated incorrectly",
        remaining_limitation="DEVICE-scope barriers 'remain unlowerable and "
            "always will be at the instruction level' per that receipt's "
            "own wording; the redirection to AirGraph (splitting into two "
            "command-buffer submissions) is a WORKAROUND via the runtime, "
            "not a closure of the gap at the instruction level, and SYSTEM "
            "scope (GPU-write visible to CPU without an explicit sync) is "
            "untouched entirely",
    ),

    # --- EXECUTION (streams/graphs; filed under RUNTIME, matching the -------
    # --- existing class the ledger already uses for streams_and_events and --
    # --- graphs) --------------------------------------------------------------

    entry(
        capability_id="RUNTIME.stream_priorities",
        cuda_mechanism="cudaStreamCreateWithPriority: streams can be "
            "assigned a priority so the scheduler prefers a high-priority "
            "stream's blocks when both are ready, useful for a "
            "latency-sensitive stream sharing a device with a "
            "throughput-oriented one",
        why_it_exists="a GPU is one shared resource; without priority, a "
            "big background job can delay a small latency-sensitive one "
            "that arrives after it",
        underlying_problem="scheduling fairness/preference between "
            "concurrent streams of unequal urgency on one device",
        apple_equivalent="MTLCommandQueue has no documented priority field "
            "found in the reference material here; command buffers on one "
            "queue execute in submission order by default",
        hawking_equivalent="none; RUNTIME.streams_and_events already found "
            "'no named streams and no events' for AirGraph, and priority is "
            "a further capability layered on TOP of streams that do not "
            "exist here yet",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no priority mechanism has been located "
            "on the Metal side to test, and no multi-priority workload has "
            "been built"),
        priority="P2",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="whether multiple MTLCommandQueues on the same "
            "device get any relative scheduling preference (as opposed to "
            "being fully independent, first-come-first-served) is "
            "unresearched, not confirmed absent",
    ),

    entry(
        capability_id="RUNTIME.graph_capture",
        cuda_mechanism="cudaStreamBeginCapture/cudaStreamEndCapture: record "
            "whatever operations a stream WOULD have executed, without "
            "executing them, producing a cudaGraph_t -- capture is "
            "transparent to code that was already written for a plain "
            "stream",
        why_it_exists="letting EXISTING stream-based code become a "
            "replayable graph without rewriting it as explicit "
            "node-construction calls",
        underlying_problem="getting the low-launch-overhead benefit of a "
            "graph without a second, parallel graph-construction API "
            "surface to maintain",
        apple_equivalent="none exactly matching 'transparently record "
            "ordinary calls'; indirect command buffers are built via their "
            "OWN construction API from the start, closer to CUDA's "
            "EXPLICIT construction path (RUNTIME.explicit_graph_"
            "construction) than to capture",
        hawking_equivalent="AirGraph IS an explicit construction API (a DAG "
            "built and submitted deliberately) -- there is no 'capture' "
            "mode where ordinary AIR dispatch calls get transparently "
            "recorded instead of executed",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("AirGraph's measured 15.5x "
            "marginal-cost reduction (RUNTIME.graphs) is about explicit "
            "construction, not capture; capture's specific value -- "
            "reusing ALREADY-WRITTEN stream code -- has no equivalent test "
            "because AirGraph was never written as ordinary sequential "
            "dispatch code first"),
        priority="P2",
        test_corpus="none specific to capture; RUNTIME.graphs' chain/fan "
            "corpus for the underlying graph-submission benefit",
        current_winner="AirGraph's explicit form, by default",
        remaining_limitation="capture's real advantage is ergonomic (no "
            "second API to learn) more than it is a distinct PERFORMANCE "
            "mechanism from explicit construction -- once captured, a "
            "cudaGraph_t executes the same way either construction path "
            "produces it, so this entry's semantic_gap is about API "
            "ergonomics, not about a different runtime behavior",
    ),

    entry(
        capability_id="RUNTIME.explicit_graph_construction",
        cuda_mechanism="cudaGraphAddKernelNode/AddMemcpyNode/"
            "AddDependencies etc: build the DAG node-by-node and "
            "edge-by-edge directly, without ever running the operations as "
            "ordinary stream calls first",
        why_it_exists="some graphs are built programmatically from data "
            "(e.g. a graph whose shape depends on a model architecture) "
            "where there is no 'ordinary sequential code' to capture in the "
            "first place",
        underlying_problem="constructing a reusable command DAG when its "
            "shape is known analytically rather than by having already "
            "been executed once",
        apple_equivalent="indirect command buffers, built by explicit "
            "encode calls ahead of execution",
        hawking_equivalent="AirGraph, built explicitly from AIR's "
            "dependency analysis of a multi-op chain (this is the SAME "
            "mechanism the ledger already measures under RUNTIME.graphs)",
        semantic_gap="PARTIAL",
        performance_gap=measured("batching k dispatches into one "
            "submission cuts the marginal cost of an added dispatch 15.5x "
            "(0.168 -> 0.011 ms); total 2.9x at k=4, 4.8x at k=8, 8.0x at "
            "k=16",
            "receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json"),
        priority="P0",
        test_corpus="chain and fan of k=1,2,4,8,16 elementwise dispatches "
            "at 4096 and 2^20 (same corpus as RUNTIME.graphs)",
        current_winner="Hawking AirGraph",
        remaining_limitation="this entry is a RENAME of RUNTIME.graphs' "
            "finding onto the more precise CUDA-side vocabulary (explicit "
            "construction, as opposed to stream capture); it does not add "
            "new evidence beyond what RUNTIME.graphs already measured, and "
            "is kept as a separate entry so the census distinguishes "
            "capture from construction the way CUDA's own API does",
        evidence="receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json",
    ),

    entry(
        capability_id="RUNTIME.graph_update_and_instantiation_reuse",
        cuda_mechanism="cudaGraphInstantiate produces an executable "
            "cudaGraphExec_t that can be REUSED across many replays; "
            "cudaGraphExecUpdate patches an already-instantiated graph's "
            "parameters (e.g. a changed pointer or launch dimension) "
            "without re-instantiating from scratch, which is itself cheaper "
            "than a fresh instantiate",
        why_it_exists="instantiation does real work (validating the DAG, "
            "allocating internal scheduling structures); paying that cost "
            "every replay of an otherwise-unchanged graph is waste, and "
            "even a graph that changes SLIGHTLY (one pointer) need not pay "
            "a full re-instantiation",
        underlying_problem="amortizing one-time graph-validation cost "
            "across many replays, and cheaply patching a graph that "
            "changes only a little between replays",
        apple_equivalent="reusable indirect command buffers are the "
            "structural analog -- encoded once, then re-executed, with "
            "individual argument buffers patchable without a full "
            "re-encode",
        hawking_equivalent="none; AirGraph's own remaining_limitation "
            "already states 'the DAG is re-recorded per call; only the "
            "compiled kernels are cached' -- this IS the identified gap, "
            "named explicitly here as its own census entry",
        semantic_gap="LARGE",
        performance_gap=unmeasured("RUNTIME.graphs measured the cost of a "
            "submission with NO reuse of the recording step at all; the "
            "marginal value of adding reuse on top of that has never been "
            "isolated"),
        priority="P1",
        test_corpus="none new; RUNTIME.graphs' corpus with per-call "
            "re-recording is the baseline this entry says has not been "
            "improved on",
        current_winner="NOT APPLICABLE -- nothing here reuses an "
            "instantiation yet",
        remaining_limitation="this is the single most concretely "
            "actionable item in this census: RUNTIME.graphs already names "
            "the exact fix ('no graph instantiation or update reuse') and "
            "this entry exists to make sure that named gap has its own "
            "tracked line rather than staying buried in another entry's "
            "remaining_limitation",
    ),

    entry(
        capability_id="RUNTIME.graph_memory_nodes",
        cuda_mechanism="cudaGraphAddMemAllocNode/MemFreeNode as node TYPES "
            "within the graph's dependency structure -- listed here as the "
            "EXECUTION-side counterpart to MEMORY.graph_memory_allocations, "
            "which covers the same mechanism from the allocator-lifetime "
            "angle",
        why_it_exists="same as MEMORY.graph_memory_allocations: an "
            "allocation that repeats identically every graph replay should "
            "be scheduled as part of the graph's own dependency ordering",
        underlying_problem="same as MEMORY.graph_memory_allocations",
        apple_equivalent="none; indirect command buffers have no "
            "allocation node type",
        hawking_equivalent="none; AirGraph has no node type for allocation "
            "at all",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no allocation-node concept exists on "
            "either side"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="genuinely the same underlying capability as "
            "MEMORY.graph_memory_allocations; kept as two entries because "
            "the census brief itself lists 'graph allocations' under "
            "MEMORY and 'graph memory nodes' under EXECUTION, and "
            "collapsing them would have hidden that the brief asked for "
            "both angles -- a future consolidation pass should probably "
            "merge them",
    ),

    entry(
        capability_id="RUNTIME.launch_attributes",
        cuda_mechanism="cudaLaunchAttribute / cudaLaunchKernelEx: a typed, "
            "extensible attribute list attached to a launch (cooperative "
            "launch, programmatic dependent launch letting one grid start "
            "before a prior one fully finishes, cluster dimensions, "
            "launch-time priority) replacing a growing pile of separate "
            "launch-configuration APIs",
        why_it_exists="as launch-time options multiplied (cooperative "
            "groups, clusters, priority, dependent launch) a single "
            "struct-based extensible attribute list scales better than one "
            "new function parameter per feature",
        underlying_problem="extensible per-launch configuration without an "
            "ever-growing function signature",
        apple_equivalent="MTLComputeCommandEncoder's dispatch calls take a "
            "fixed, non-extensible parameter set (grid size, threadgroup "
            "size, and a handful of encoder-level state); no attribute-list "
            "pattern for per-dispatch options was found",
        hawking_equivalent="none; AIR's dispatch path has exactly the "
            "parameters it needs today and no extensibility mechanism",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("this is an API design question, not a "
            "runtime behavior to benchmark on its own"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="programmatic dependent launch specifically "
            "(starting grid N+1 before grid N fully drains) has no "
            "identified Metal equivalent at all and is the one sub-feature "
            "of this entry that is a real capability question, not just an "
            "API-shape question -- it has not been investigated separately",
    ),

    entry(
        capability_id="RUNTIME.host_callbacks",
        cuda_mechanism="cudaLaunchHostFunc: enqueue a CPU-side function "
            "call INTO a stream, executed by a driver thread once all "
            "prior work in the stream completes, without the host having "
            "to poll or block on a synchronize",
        why_it_exists="some CPU-side bookkeeping (freeing a buffer, "
            "signaling another thread, updating a progress counter) should "
            "happen exactly when a stream reaches a point, without the "
            "host busy-waiting or explicitly synchronizing and stalling "
            "its own async submission of further work",
        underlying_problem="running host code in POSITION within an async "
            "device command stream, not before or after it",
        apple_equivalent="MTLCommandBuffer.addCompletedHandler: runs a "
            "host closure when a command buffer finishes -- the same shape "
            "of mechanism (host code invoked by the completion of GPU "
            "work, off the submitting thread)",
        hawking_equivalent="none observed; whether Hawking's Python/MLX "
            "layer uses completion handlers anywhere has not been checked",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("nothing in this codebase has been "
            "searched for or measured using a completion-handler pattern"),
        priority="P3",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="addCompletedHandler's existence in the Metal "
            "API is confident (it is a standard, widely documented API), "
            "but whether MLX exposes it or uses it internally for anything "
            "Hawking depends on has not been checked in MLX's source from "
            "this machine",
    ),

    # --- COMPILER --------------------------------------------------------------

    entry(
        capability_id="COMPILER.nvrtc_runtime_compilation",
        cuda_mechanism="NVRTC: a standalone library that compiles CUDA C++ "
            "SOURCE (not just PTX) to PTX at runtime, given only a source "
            "string -- the CUDA-C++-specific counterpart to the more "
            "general 'JIT compile at runtime' capability",
        why_it_exists="some programs generate CUDA C++ source text at "
            "runtime (a DSL, a specialized kernel body) and need a "
            "compiler callable as a library, not a subprocess invoking "
            "nvcc",
        underlying_problem="compiling a HIGH-LEVEL device-language source "
            "string, not just lowering an already-parallel IR, without "
            "shelling out to a separate compiler binary",
        apple_equivalent="MTLDevice.makeLibrary(source:options:) compiles "
            "MSL source text to a usable library, in-process, at runtime -- "
            "the direct structural analog",
        hawking_equivalent="already measured under "
            "COMPILER.jit_kernel_compilation: mx.fast.metal_kernel "
            "JIT-compiles the MSL string AIR emits, at 36.8ms per distinct "
            "kernel and 0.24ms cached",
        semantic_gap="PARTIAL",
        performance_gap=measured("36.8 ms per distinct kernel, 0.24 ms "
            "cached on repeat -- same measurement as "
            "COMPILER.jit_kernel_compilation, since it is the same compile "
            "step viewed from NVRTC's angle instead of AIR's",
            "receipts/headless/ACCELERATOR_JIT_COMPILE.json"),
        priority="P1",
        test_corpus="the six AIR programs of P3 (same as "
            "COMPILER.jit_kernel_compilation)",
        current_winner="MLX JIT, retained",
        remaining_limitation="this entry exists to answer the census "
            "brief's explicit request for an NVRTC line item; it does not "
            "add new measurement beyond COMPILER.jit_kernel_compilation and "
            "a future pass should consider whether keeping both is worth "
            "the duplication",
        evidence="receipts/headless/ACCELERATOR_JIT_COMPILE.json",
    ),

    entry(
        capability_id="COMPILER.fatbinaries",
        cuda_mechanism="a 'fat binary' embeds compiled code for MULTIPLE "
            "compute capabilities (sm_70, sm_80, sm_90, ...) plus a PTX "
            "fallback in one file; the driver picks the best match for the "
            "actual GPU at load time, or JITs the PTX fallback if no exact "
            "match is embedded",
        why_it_exists="CUDA binaries must run across many GPU generations "
            "with different instruction sets; a fat binary avoids shipping "
            "a separate binary per generation while still using native "
            "code where available",
        underlying_problem="one distributable artifact that runs "
            "efficiently across hardware generations with incompatible "
            "native instruction sets",
        apple_equivalent="not needed in the same shape -- Apple GPUs "
            "across the relevant product line share one instruction set "
            "family closely enough that Metal shader source (or a single "
            "metallib) targets them without a multi-architecture "
            "fat-binary format; Apple's real cross-hardware problem "
            "(Intel-era Macs, different GPU vendors) predates the "
            "unified-memory Apple Silicon line this ledger is scoped to",
        hawking_equivalent="none needed; AIR emits MSL text compiled fresh "
            "by MLX's JIT on the one machine it runs on",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no multi-generation Apple Silicon "
            "compatibility question has been tested; this machine is the "
            "only target"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="whether Apple Silicon GPU generations (M1 "
            "through M3+) actually share enough of an instruction set that "
            "this problem TRULY does not exist, versus Metal simply hiding "
            "a real recompilation behind its own JIT on every machine, is "
            "asserted from general knowledge and not verified against "
            "Apple's own architecture documentation on this machine",
    ),

    entry(
        capability_id="COMPILER.link_time_optimization",
        cuda_mechanism="device-side LTO (nvcc -dlto / -rdc=true with "
            "link-time code generation): separately-compiled device object "
            "files are optimized ACROSS their boundary at link time "
            "(inlining across translation units, dead code elimination) "
            "rather than only within each file",
        why_it_exists="splitting a large device program across files for "
            "build hygiene should not force every cross-file call to stay "
            "a real, un-inlined function call",
        underlying_problem="cross-translation-unit optimization for device "
            "code compiled separately",
        apple_equivalent="not directly applicable in the same shape -- a "
            "single AIR/MSL program is emitted as one source string per "
            "kernel, not compiled from multiple separately-compiled device "
            "object files, so there is no link boundary to optimize across "
            "yet",
        hawking_equivalent="none; C2M and AIR both currently produce one "
            "MSL string per program, not a multi-file device build",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no multi-file AIR program exists to "
            "need cross-file optimization"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE",
        remaining_limitation="this only becomes a real question if AIR "
            "ever grows a multi-translation-unit device build; today it "
            "does not, so calling this DELETED_BY_UNIFIED_MEMORY would "
            "overclaim -- the problem doesn't exist YET here because the "
            "codebase hasn't reached the scale where it would come up, not "
            "because unified memory or Apple's architecture removes it",
    ),

    entry(
        capability_id="COMPILER.launch_bounds",
        cuda_mechanism="__launch_bounds__(maxThreadsPerBlock, "
            "minBlocksPerSM) attached to a kernel definition, or the "
            "-maxrregcount compiler flag: tells the compiler the intended "
            "launch configuration so it can constrain register allocation "
            "to guarantee that occupancy is ACHIEVABLE, instead of the "
            "compiler greedily using registers and the kernel failing to "
            "launch at the intended block size",
        why_it_exists="a compiler optimizing one kernel in isolation has "
            "no way to know the programmer's INTENDED occupancy target; "
            "without a hint it may allocate registers that make a large "
            "block size fail to fit",
        underlying_problem="communicating an occupancy INTENT to a "
            "compiler that otherwise only sees one kernel body in "
            "isolation",
        apple_equivalent="[[max_total_threads_per_threadgroup(N)]] is a "
            "documented MSL attribute serving the same role -- a "
            "compile-time hint constraining the compiler's resource "
            "allocation to guarantee a launch shape works",
        hawking_equivalent="none; AIR has never emitted this attribute on "
            "any generated kernel",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no AIR kernel has been compiled with "
            "and without this attribute to compare"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="directly related to "
            "EXECUTION.register_pressure_and_occupancy -- this is the "
            "COMPILE-TIME hint, that entry is the RUNTIME consequence; "
            "neither has been measured, and xcrun metal's absence on this "
            "machine blocks inspecting whether AIR-generated code would "
            "even benefit",
    ),

    entry(
        capability_id="COMPILER.kernel_metadata",
        cuda_mechanism="cudaFuncGetAttributes / cuFuncGetAttribute queried "
            "on a compiled kernel at runtime: register count, shared "
            "memory usage, constant memory usage, max threads per block, "
            "PTX/binary version -- resource usage baked into the compiled "
            "artifact and introspectable without a separate profiling pass",
        why_it_exists="an occupancy calculator (or a program tuning its "
            "own launch shape at runtime) needs to know what a kernel "
            "ACTUALLY costs in registers and shared memory, not just what "
            "the source asked for",
        underlying_problem="runtime introspection of a compiled kernel's "
            "resource footprint",
        apple_equivalent="MTLComputePipelineState exposes "
            "maxTotalThreadsPerThreadgroup, threadExecutionWidth, and "
            "staticThreadgroupMemoryLength -- a smaller but structurally "
            "equivalent introspection surface",
        hawking_equivalent="none; nothing in AIR/bench.py queries a "
            "compiled MTLComputePipelineState for these fields",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("not a performance capability by "
            "itself -- introspection, not execution"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="MTLComputePipelineState's introspection "
            "surface is smaller than CUDA's (no reported register count, "
            "which Apple does not appear to expose to app code at all as "
            "far as this census could establish) -- that specific field "
            "may be a genuine, permanent gap rather than an unbuilt door, "
            "but this has not been verified against current Metal "
            "documentation on this machine",
    ),

    # --- MATH ECOSYSTEM ------------------------------------------------------

    entry(
        capability_id="MATH.cublas_library_surface",
        cuda_mechanism="cuBLAS beyond plain GEMM: batched and "
            "strided-batched GEMM (many small independent matrix "
            "multiplies in one call), mixed-precision and tensor-core code "
            "paths selected automatically per shape/dtype, and the full "
            "BLAS level 1/2/3 function set (not just level-3 matmul)",
        why_it_exists="a transformer's expert-routed or per-head matmuls "
            "are often many SMALL matrices rather than one big one; "
            "batched GEMM avoids one kernel launch per matrix, and BLAS "
            "1/2 cover vector and matrix-vector operations GEMM does not",
        underlying_problem="efficient execution of many small, independent "
            "matrix operations, and of the non-GEMM BLAS operations a "
            "model still needs",
        apple_equivalent="MLX's matmul supports batched/broadcast matrix "
            "multiplication; MPS also exposes batched GEMM. Level 1/2 "
            "BLAS-equivalent operations exist as individual MLX ops "
            "(mx.matmul for mat-vec, elementwise ops for level 1) rather "
            "than as one named 'BLAS' library",
        hawking_equivalent="the native quantized matvec kernel of "
            "MEMORY.quantized_weight_access, exercised at BATCH sizes on "
            "real Qwen3 expert tensors",
        semantic_gap="PARTIAL",
        performance_gap=measured("64 real Qwen3-30B-A3B expert tensors, "
            "shared activation (the decode case): B=1 1.02x (NO_CLAIM, "
            "reproduces the no-advantage-below-64MiB finding), B=4 1.06x "
            "(INDISTINGUISHABLE), B=16 1.45x, B=43 2.27x, B=64 2.70x "
            "(CANDIDATE_WINS_DESPITE_NOISE, 13.43% IQR). The native arm "
            "reaches only 161.9 GB/s against the dense arm's 426.4 GB/s "
            "(72% of this machine's measured 589.73 GB/s roof) -- the "
            "native kernel is ARITHMETIC-bound on the unpack, not "
            "bandwidth-bound, which is why the win is 2.70x and not the "
            "7.1x byte ratio",
            "receipts/headless/ACCELERATOR_EXPERT_BATCH.json"),
        priority="P0",
        test_corpus="64 real Qwen3-30B-A3B expert gate_proj tensors at "
            "B=1,4,16,43,64, native ws_rtn_q4_g64 vs dense f32",
        current_winner="Hawking native batched matvec above B~16; "
            "NO_CLAIM at B=1 (reproduces MEMORY.quantized_weight_access's "
            "own crossover)",
        remaining_limitation="the timed arms all share ONE activation "
            "vector across the batch (the realistic decode-time MoE "
            "shape); a genuinely batched kernel with INDEPENDENT "
            "activations per matrix (the more general cuBLAS "
            "batched-GEMM shape) is a different, untested case. The "
            "measured 10.58% relative error against the dense arm is the "
            "REPRESENTATION's quantization error, not a kernel bug -- "
            "verified separately against the representation's own oracle "
            "at 2.861e-06",
        evidence="receipts/headless/ACCELERATOR_EXPERT_BATCH.json",
    ),

    entry(
        capability_id="MATH.cublaslt_epilogue_fusion",
        cuda_mechanism="cuBLASLt: an explicit-algorithm-selection API "
            "layered on cuBLAS that lets a caller fuse a bias-add and "
            "activation function INTO the GEMM epilogue (the same output "
            "tile that was just computed in registers gets the "
            "bias/activation applied before it is written to memory, "
            "instead of a separate kernel reading it back)",
        why_it_exists="GEMM followed by bias-add and activation is "
            "bandwidth-bound on the INTERMEDIATE if done as separate "
            "kernels, for the same reason COMPILER.kernel_fusion exists "
            "generally",
        underlying_problem="avoiding a memory round trip for a fixed, "
            "common post-GEMM operation sequence specifically",
        apple_equivalent="MLX does not expose a fused GEMM+bias+activation "
            "epilogue as a single call; MPSMatrixMultiplication similarly "
            "does not fuse an arbitrary epilogue",
        hawking_equivalent="COMPILER.kernel_fusion's AIR chain-fusion "
            "(mul->relu->silu, 1.71x over naive) is the closest evidence "
            "-- general elementwise fusion, not specifically a GEMM "
            "epilogue -- and no GEMM+bias+activation fusion has been built "
            "or measured",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no GEMM-epilogue-fusion kernel exists "
            "on either side of the comparison; COMPILER.kernel_fusion's "
            "number is for a pure elementwise chain, not a matmul "
            "epilogue"),
        priority="P1",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="GEMM+activation is exactly the shape of a "
            "transformer's feedforward block, so this is a second "
            "high-relevance untested item alongside "
            "MATH.cublas_library_surface",
    ),

    entry(
        capability_id="MATH.cudnn_graph_engine_model",
        cuda_mechanism="cuDNN's graph API: describe a subgraph of "
            "operations (conv, pointwise, reduction) declaratively, and "
            "cuDNN's own heuristics engine picks and JIT-assembles a fused "
            "kernel ('engine') to execute it, rather than the caller "
            "choosing a specific pre-written kernel",
        why_it_exists="the best fused kernel for a given "
            "shape/dtype/hardware combination changes too often for a "
            "library to hand-write and name every combination; a "
            "heuristics engine that ASSEMBLES a kernel from the graph "
            "description scales better than an enumerated kernel catalog",
        underlying_problem="automatic kernel selection/assembly for a "
            "described computation graph, rather than manual selection by "
            "the caller",
        apple_equivalent="MLX's own lazy graph + its internal kernel "
            "dispatch is a heuristics-driven system in the same SPIRIT "
            "(the caller describes ops, MLX decides how to execute them), "
            "but this census has not compared MLX's actual selection "
            "LOGIC to cuDNN's documented heuristics engine -- that would "
            "require reading MLX internals not yet examined",
        hawking_equivalent="AIR's own fusion (COMPILER.kernel_fusion) "
            "plays a similar role for elementwise chains specifically, "
            "hand-written rather than heuristically assembled. Separately, "
            "receipts/headless/ACCELERATOR_CONVOLUTION.json already built "
            "and correctness-verified a direct convolution kernel (88 "
            "width-4 conv1d tensors censused in one real specimen) -- but "
            "its own claim_boundary states 'nothing claimed against a "
            "baseline', so it is evidence of a kernel existing, not of an "
            "engine-selection capability",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("comparing 'a heuristics engine selects "
            "the kernel' as a CAPABILITY, not a specific kernel's speed, "
            "is not something a single benchmark settles"),
        priority="P2",
        test_corpus="none for the engine-selection question; "
            "ACCELERATOR_CONVOLUTION.json for the adjacent "
            "kernel-existence question",
        current_winner="UNKNOWN",
        remaining_limitation="this entry is the most abstract in the "
            "census -- it is about a LIBRARY ARCHITECTURE PATTERN "
            "(declarative graph + automatic engine selection) rather than "
            "a specific instruction or API, and judging Apple's "
            "equivalent honestly would require reading MLX's dispatch "
            "internals, which has not been done",
    ),

    entry(
        capability_id="MATH.cusparse",
        cuda_mechanism="cuSPARSE: sparse matrix formats (CSR, COO, "
            "blocked-ELL) and sparse-dense/sparse-sparse matrix operations "
            "(SpMM, SpMV) exploiting structural zeros to skip work and "
            "memory traffic a dense kernel would waste",
        why_it_exists="a matrix with mostly-zero entries wastes both "
            "FLOPs and bandwidth if processed as if dense; structured "
            "sparse formats and kernels skip the zeros",
        underlying_problem="exploiting known-zero structure to avoid "
            "computing and moving values that are zero",
        apple_equivalent="no dedicated sparse-matrix-multiply library "
            "located in MLX's documented op set; MLX's op set is "
            "dense-array-first",
        hawking_equivalent="receipts/headless/ACCELERATOR_SPARSE.json "
            "measured a DIFFERENT sense of 'sparse': whether a sparse "
            "WEIGHT REPRESENTATION (for compression) beats a dense "
            "representation at the same bit budget. It does not, by "
            "2.5-3.9x per bit -- 'the outlier hurts by setting the group "
            "scale, not by being stored badly, so the error it causes is "
            "DENSE and a sparse fix cannot reach it'. That finding is "
            "about QUANTIZATION representation, not about a sparse "
            "COMPUTE kernel (SpMM/SpMV), and does not answer this entry's "
            "question",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no sparse-matrix COMPUTE kernel "
            "(SpMM/SpMV skipping known zeros at runtime) has been built or "
            "timed here; ACCELERATOR_SPARSE.json answers a representation "
            "question, not a compute-kernel question"),
        priority="P2",
        test_corpus="none for sparse compute; "
            "ACCELERATOR_SPARSE.json's 8 expert gate_proj tensors for the "
            "adjacent representation question",
        current_winner="UNKNOWN",
        remaining_limitation="this entry initially risked conflating two "
            "different meanings of 'sparse' (compressed weight "
            "representation vs. a compute kernel that skips zero "
            "operands); they are kept explicitly separate here because "
            "ACCELERATOR_SPARSE.json's negative result does not transfer "
            "to the cuSPARSE-shaped question at all",
    ),

    entry(
        capability_id="MATH.cufft",
        cuda_mechanism="cuFFT: FFT-based convolution and general Fourier "
            "transforms on the GPU, valuable because FFT convolution beats "
            "direct convolution once the kernel is LONG (O(n log n) vs "
            "O(n*k))",
        why_it_exists="some models (long convolutions, spectral methods) "
            "have kernels long enough that FFT's asymptotic advantage over "
            "direct convolution pays for the two transforms and pointwise "
            "product it costs",
        underlying_problem="convolution with a kernel long enough that "
            "direct multiply-add is the wrong algorithm",
        apple_equivalent="vDSP/Accelerate provides FFT on Apple Silicon; "
            "MLX itself does not expose an FFT-based convolution path as "
            "far as documented here",
        hawking_equivalent="none; AIR has a DIRECT convolution kernel only "
            "(receipts/headless/ACCELERATOR_CONVOLUTION.json), and FFT was "
            "a deliberate, investigated choice, not an oversight",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("ACCELERATOR_CONVOLUTION.json "
            "investigated and REFUSED FFT for the one convolution "
            "specimen present, with a stated algorithmic reason rather "
            "than a name-grep absence: 'FFT convolution beats direct "
            "convolution only for LONG kernels. The one convolution that "
            "exists on disk here has WIDTH 4: direct costs four "
            "multiply-adds per output while an FFT costs two transforms "
            "and a pointwise product. FFT WOULD BE THE WRONG ALGORITHM "
            "for the only convolution these specimens contain.' Nothing "
            "wider than width 4 has been tried, so this is not timed "
            "either way"),
        priority="P2",
        test_corpus="the four model specimens censused in "
            "ACCELERATOR_CONVOLUTION.json (Falcon-H1's 88 width-4 conv1d "
            "tensors; zero elsewhere)",
        current_winner="direct convolution, for the only specimen on disk",
        remaining_limitation="the refusal is honestly scoped to width-4 "
            "kernels: ACCELERATOR_CONVOLUTION.json itself names 'a "
            "long-convolution specimen' as what would change the answer. "
            "This entry should be read as 'FFT is not useful for the ONE "
            "convolution actually present', never as 'FFT is never useful "
            "here'",
    ),

    entry(
        capability_id="MATH.cusolver",
        cuda_mechanism="cuSOLVER: dense linear algebra decompositions -- "
            "LU, Cholesky, QR, SVD, eigenvalue solvers -- built on top of "
            "cuBLAS for problems that need a FACTORIZATION, not just a "
            "product",
        why_it_exists="a factorization is a different algorithmic shape "
            "than a matmul (sequential dependency between pivots/columns "
            "rather than embarrassingly parallel tiles) and needs its own "
            "library rather than being expressible as a GEMM call",
        underlying_problem="solving linear systems / decomposing matrices, "
            "which have real data dependencies between steps that a pure "
            "GEMM kernel does not",
        apple_equivalent="Accelerate/LAPACK on the CPU covers this; MLX "
            "has begun exposing mx.linalg (QR, SVD, Cholesky, inverse) for "
            "at least some of these -- whether each dispatches to CPU or "
            "GPU has not been independently verified from this machine",
        hawking_equivalent="none; nothing in tools/accelerator uses a "
            "decomposition",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("no decomposition kernel has been "
            "built or timed under tools/accelerator"),
        priority="P3",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="whether mx.linalg's GPU-named coverage is "
            "real Metal-side compute or a silent CPU fallback per-"
            "operation has not been checked; that distinction would "
            "materially change this entry's semantic_gap and is exactly "
            "the kind of claim this census refuses to guess at",
    ),

    entry(
        capability_id="MATH.curand",
        cuda_mechanism="cuRAND: device-side pseudorandom and quasirandom "
            "number generation, callable FROM a running kernel so sampling "
            "does not require returning to the host for random values",
        why_it_exists="a kernel that needs randomness per-thread (dropout, "
            "sampling, Monte Carlo) would otherwise have to either "
            "precompute random numbers on the host and transfer them, or "
            "synchronize back to the host mid-algorithm -- device-side "
            "generation avoids both",
        underlying_problem="supplying randomness to many parallel threads "
            "without a host round trip and without threads colliding on "
            "the same stream/seed",
        apple_equivalent="mx.random generates via MLX's own RNG; whether "
            "MLX's kernels generate randomness ON-DEVICE per-thread (the "
            "cuRAND shape) or generate host-side and transfer has not been "
            "checked",
        hawking_equivalent="receipts/headless/ACCELERATOR_TOPK_SAMPLING."
            "json built a CDF-walk sampler in AIR -- its own headline "
            "states 'a sampler cannot be graded the way every other kernel "
            "here has been' -- but that receipt is explicit that it "
            "carries NO performance claim, and its concern is CORRECTNESS "
            "of sampling (can a broken sampler that always returns argmax "
            "pass a naive check), not where the randomness is generated",
        semantic_gap="UNKNOWN",
        performance_gap=unmeasured("ACCELERATOR_TOPK_SAMPLING.json "
            "explicitly carries no performance claim; whether AIR's "
            "sampler generates its random draws device-side or is fed "
            "them from the host has not been examined"),
        priority="P2",
        test_corpus="ACCELERATOR_TOPK_SAMPLING.json's CDF-walk sampler, "
            "for the correctness question; nothing for the "
            "generation-location question",
        current_winner="UNKNOWN",
        remaining_limitation="this entry surfaces a real open question "
            "(where do AIR's random draws actually come from) that the "
            "existing TOPK_SAMPLING receipt does not answer, because that "
            "receipt's own concern was grading correctness, not RNG "
            "placement",
    ),

    entry(
        capability_id="MATH.cub_primitives",
        cuda_mechanism="CUB's primitives BEYOND reduce/scan (already "
            "covered by MATH.reduction_and_scan): block/device-wide SORT "
            "(radix and merge), SELECT/PARTITION (stream compaction by "
            "predicate), and HISTOGRAM",
        why_it_exists="sort, compaction, and histogram are each a "
            "distinct parallel algorithm pattern (data-dependent output "
            "size or position) that a reduce/scan primitive does not "
            "cover, and each is common enough to warrant a hand-tuned "
            "library implementation rather than a hand-rolled kernel per "
            "use",
        underlying_problem="parallel algorithms whose OUTPUT SIZE or "
            "POSITION depends on the data, not just its VALUE -- the thing "
            "reduce/scan do not have to solve because their output shape "
            "is fixed",
        apple_equivalent="MLX exposes mx.sort/mx.argsort; a "
            "partition/compaction primitive and a histogram primitive "
            "were not located in MLX's documented op set from this "
            "machine",
        hawking_equivalent="none; AIR has no sort, compaction, or "
            "histogram kernel -- MATH.reduction_and_scan's scan (which "
            "DOES beat mx.cumsum 8.09x at 2^24) is the only CUB-family "
            "primitive built here",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no sort, compaction, or histogram "
            "kernel exists in AIR"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN -- mx.sort exists for sort specifically "
            "and has never been benchmarked against anything here",
        remaining_limitation="this is a real, unbuilt gap distinct from "
            "MATH.reduction_and_scan, kept separate because reduce/scan's "
            "output shape is fixed and known ahead of time while "
            "sort/compaction/histogram's is not, which is a materially "
            "different parallel algorithm design problem",
    ),

    entry(
        capability_id="MATH.thrust",
        cuda_mechanism="Thrust: a C++ STL-like library of parallel "
            "algorithms (transform, sort, reduce, scan, unique, set "
            "operations) over device or host vectors, at a HIGHER level of "
            "abstraction than CUB -- Thrust composes CUB-level primitives "
            "(and cuBLAS/cuSOLVER where relevant) behind an "
            "iterator-based, container-agnostic interface",
        why_it_exists="writing a custom kernel for every combination of "
            "transform+reduce+filter is both slow to develop and easy to "
            "get subtly wrong; a composable high-level algorithm library "
            "trades some performance headroom for correctness and "
            "development speed",
        underlying_problem="the same problem every high-level "
            "parallel-algorithms library solves: making common patterns "
            "fast to WRITE, not just fast to RUN",
        apple_equivalent="MLX's own Python-level array API (broadcasting, "
            "vectorized ops, mx.sort, reductions) plays this role for the "
            "NumPy-shaped audience; there is no C++-iterator-based library "
            "with Thrust's specific shape",
        hawking_equivalent="AIR itself is the closer analog in SPIRIT -- a "
            "chain of ops gets lowered and (per COMPILER.kernel_fusion) "
            "FUSED into one kernel, which is actually a stronger guarantee "
            "than Thrust gives (Thrust's composed algorithms are "
            "typically separate kernel launches per stage, not "
            "automatically fused)",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no direct Thrust-vs-AIR comparison "
            "exists or is measurable without CUDA hardware"),
        priority="P3",
        test_corpus="COMPILER.kernel_fusion's mul->relu->silu corpus is "
            "the closest existing evidence for 'a composed chain, fused', "
            "though it was never compared to Thrust specifically",
        current_winner="UNKNOWN -- the comparison would need to control "
            "for Thrust's typical lack of auto-fusion versus AIR's, and "
            "that framing has not been verified against Thrust's actual "
            "current behavior",
        remaining_limitation="the claim that Thrust does not auto-fuse "
            "chained algorithms the way AIR does is stated from general "
            "knowledge of the library's design, NOT VERIFIED AGAINST "
            "NVIDIA DOCUMENTATION ON THIS MACHINE, per this census's own "
            "rule about unverified API claims",
    ),

    entry(
        capability_id="MATH.cutlass_concepts",
        cuda_mechanism="CUTLASS: a header-only TEMPLATE LIBRARY (not a "
            "compiled binary library like cuBLAS) exposing GEMM as "
            "composable building blocks across a tile hierarchy "
            "(thread-block tile -> warp tile -> instruction-level tile), "
            "letting a user hand-assemble a custom fused GEMM variant "
            "cuBLAS does not ship as a named function",
        why_it_exists="cuBLAS ships a FIXED catalog of GEMM variants; some "
            "fusions or epilogues a real workload needs are not in that "
            "catalog, and CUTLASS lets a user compose one from the same "
            "building blocks NVIDIA's own engineers use, without writing a "
            "GEMM from scratch",
        underlying_problem="customizing a near-peak-performance GEMM for "
            "a fusion pattern the vendor library does not ship as a named "
            "function, without abandoning peak performance to do it",
        apple_equivalent="no template-library equivalent exposing a tile "
            "hierarchy was located for Metal; MLX's mx.fast.metal_kernel "
            "lets a caller write raw MSL by hand, which is a much "
            "lower-level escape hatch than a composable tile-template "
            "library",
        hawking_equivalent="AIR's simdgroup matmul WITH 2x2 register "
            "blocking (MATH.gemm) is exactly this kind of hand-assembled "
            "tile hierarchy, built the hard way (by hand) rather than "
            "composed from a template library, because no such library "
            "exists on this platform",
        semantic_gap="PARTIAL",
        performance_gap=measured("parity with MLX at 1024 and 2048; MLX "
            "wins 1.24x at 4096 -- the same measurement as MATH.gemm, "
            "since AIR's register-blocked matmul IS this census's "
            "CUTLASS-shaped answer",
            "receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json"),
        priority="P1",
        test_corpus="MATH.gemm's 512/1024/2048/4096 corpus",
        current_winner="MLX at 4096 and above; AIR at parity below",
        remaining_limitation="the tile-hierarchy CONCEPT is what CUTLASS "
            "names explicitly and reusably as a library; AIR has one "
            "hand-built instance of the pattern (2x2 register blocking), "
            "not a reusable template library that could assemble a "
            "DIFFERENT fused variant on demand the way CUTLASS's actual "
            "users do",
        evidence="receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json",
    ),

    entry(
        capability_id="MATH.nccl_collectives",
        cuda_mechanism="NCCL: topology-aware collective communication "
            "algorithms (ring/tree all-reduce, all-gather, broadcast, "
            "reduce-scatter) across multiple GPUs, choosing an algorithm "
            "shape based on the actual interconnect topology (NVLink vs "
            "PCIe vs network)",
        why_it_exists="naive all-to-all communication among N GPUs does "
            "not use the interconnect topology efficiently; NCCL's "
            "algorithms are specifically shaped to the bandwidth/latency "
            "profile of how GPUs are actually wired together",
        underlying_problem="the ALGORITHM SHAPE of a collective "
            "communication pattern, as distinct from the physical "
            "transport it rides on",
        apple_equivalent="none; there is exactly one GPU in this machine, "
            "so there is no collective to shape an algorithm for. "
            "MULTI_DEVICE.peer_access already covers the PHYSICAL "
            "transport question (no second device exists) -- this entry "
            "is the algorithmic layer ABOVE that transport, and is "
            "equally inapplicable for the same underlying reason",
        hawking_equivalent="none; HUMF (MULTI_DEVICE.peer_access) models "
            "domains but has no collective-algorithm concept at all, mock "
            "or real",
        semantic_gap="LARGE",
        performance_gap=unmeasured("no second physical device exists; "
            "every multi-device number would be simulated and per this "
            "ledger's own hard rule may not be cited as evidence"),
        priority="P3",
        test_corpus="none",
        current_winner="NOT APPLICABLE -- no second physical device",
        remaining_limitation="identical hardware blocker to "
            "MULTI_DEVICE.peer_access, kept as a SEPARATE entry because "
            "NCCL is a distinct algorithmic/library layer from raw peer "
            "access even though both are blocked by the same missing "
            "hardware; MULTI_DEVICE.peer_access remains the SOLE "
            "MULTI_DEVICE-class entry, unchanged, per this campaign's own "
            "instruction not to add anything there while blocked on "
            "hardware -- this entry sits in MATH instead",
    ),

    # --- PROFILING -----------------------------------------------------------

    entry(
        capability_id="PROFILING.cupti",
        cuda_mechanism="CUPTI: the low-level instrumentation API that "
            "Nsight Systems, Nsight Compute, and most third-party CUDA "
            "profilers are BUILT ON -- activity records, callback API, "
            "hardware performance counters, correlation IDs tying a host "
            "API call to the device work it produced",
        why_it_exists="every profiler needs the same underlying data "
            "(when did a kernel run, what did it read/write, what stalled "
            "it); CUPTI exists so that data collection is written ONCE, in "
            "the driver, rather than reimplemented per profiling tool",
        underlying_problem="a single, reusable, low-overhead source of "
            "ground-truth execution data that multiple tools can build on",
        apple_equivalent="Instruments' Metal System Trace / GPU counters, "
            "and MTLCaptureManager's GPU frame capture, serve an "
            "analogous role -- a system-level, driver-backed data source "
            "other tools could consume, though whether third-party "
            "(non-Apple) tools CAN consume it the way third-party CUDA "
            "profilers consume CUPTI has not been checked",
        hawking_equivalent="PROFILING.kernel_timing's bench.py measures "
            "WALL CLOCK ONLY from the calling process, which is a "
            "different tier entirely -- it is not a driver-level "
            "instrumentation API, it is a stopwatch around a call",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("not something to benchmark -- it is "
            "an instrumentation capability, and this census has not "
            "exercised Instruments' GPU counters at all"),
        priority="P1",
        test_corpus="none",
        current_winner="UNKNOWN -- bench.py is what is actually used, by "
            "necessity, not because it was chosen over a CUPTI-equivalent "
            "that was tried and lost",
        remaining_limitation="this is the ROOT cause of "
            "PROFILING.kernel_timing's own remaining_limitation ('no DRAM "
            "byte counters, no occupancy, no stall reasons'): those "
            "specific numbers are exactly what a CUPTI-tier "
            "instrumentation source would supply, and Hawking has never "
            "used Instruments' GPU counter APIs to get them",
    ),

    entry(
        capability_id="PROFILING.nsight_systems",
        cuda_mechanism="Nsight Systems: a system-WIDE timeline showing CPU "
            "threads, CUDA streams, memory transfers, and kernel execution "
            "together, for diagnosing where time actually goes ACROSS the "
            "whole pipeline rather than inside one kernel",
        why_it_exists="a fast kernel in a pipeline with poor overlap or an "
            "idle GPU waiting on the CPU is still a slow PROGRAM; "
            "per-kernel profiling alone cannot see that, so a system-wide "
            "timeline view is a separate, necessary tool",
        underlying_problem="seeing GPU idle time and CPU/GPU overlap "
            "across an entire program, not just a single kernel's "
            "internal cost",
        apple_equivalent="Instruments' Metal System Trace template covers "
            "the same system-wide timeline shape",
        hawking_equivalent="none; nothing in this repo has captured a "
            "system-wide timeline. ACCELERATOR_CONCURRENCY_SWEEP's "
            "finding that per-process efficiency falls 1.0->0.42 from 1 "
            "to 8 processes is the closest INDIRECT evidence of "
            "contention, inferred from aggregate throughput rather than "
            "observed on a timeline",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("no Instruments trace has been "
            "captured for any Hawking workload during this program"),
        priority="P1",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="given that ACCELERATOR_DEVICE_RESIDENT.json "
            "found GPU-idle-adjacent effects (bytes not 'coming home' cost "
            "3.39x) purely through wall-clock A/B testing rather than a "
            "timeline tool, some of what Nsight Systems would show has "
            "been INFERRED here through careful experiment design rather "
            "than OBSERVED directly -- a real methodological difference "
            "worth naming, not a claim that the inference is wrong",
    ),

    entry(
        capability_id="PROFILING.nsight_compute",
        cuda_mechanism="Nsight Compute: per-kernel hardware counters -- "
            "achieved occupancy, memory throughput as a percentage of the "
            "roofline, warp stall reasons (broken down by cause: memory "
            "dependency, execution dependency, not-selected, etc.), "
            "instruction mix -- collected by replaying the kernel with "
            "counters armed",
        why_it_exists="knowing a kernel is SLOW is not the same as "
            "knowing WHY; without stall-reason and occupancy counters, "
            "diagnosing a slow kernel is limited to hypothesis-and-rewrite "
            "instead of measurement-and-target",
        underlying_problem="attributing a kernel's lost performance to a "
            "SPECIFIC hardware cause (bandwidth, occupancy, a specific "
            "stall reason) rather than inferring it from comparative "
            "wall-clock timing alone",
        apple_equivalent="Instruments' GPU counters / the Metal "
            "Performance HUD expose SOME comparable data (GPU "
            "utilization, memory bandwidth); a stall-reason breakdown as "
            "granular as Nsight Compute's was not confirmed present in "
            "the documentation available on this machine",
        hawking_equivalent="PROFILING.kernel_timing's own "
            "remaining_limitation states this exactly: 'no DRAM byte "
            "counters, no occupancy, no stall reasons, so mechanism is "
            "inferred rather than observed' -- every diagnosis in this "
            "whole census program (e.g. the occupancy story in "
            "receipts/headless/ACCELERATOR_ATTENTION_OCCUPANCY.json) has "
            "been built by controlled A/B wall-clock experiments "
            "substituting for what this tool would show directly",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("this is the instrument itself; "
            "PROFILING.kernel_timing already states it is not "
            "benchmarked, for the same reason"),
        priority="P0",
        test_corpus="every A/B diagnosis in this program is, "
            "collectively, the corpus of evidence that this tool's "
            "ABSENCE has a real cost -- each one is slower and less "
            "certain than a direct counter read would be",
        current_winner="controlled A/B wall-clock experiments, by "
            "necessity",
        remaining_limitation="this is arguably the single highest-leverage "
            "gap in the entire census: nearly every inferred-mechanism "
            "finding elsewhere in this ledger (e.g. MATH.attention's "
            "occupancy story) exists BECAUSE this tool's counters are "
            "unavailable, and would likely be resolved faster and more "
            "certainly with them",
    ),

    entry(
        capability_id="DEBUGGING.race_and_memory_diagnostics",
        cuda_mechanism="compute-sanitizer's four tools: memcheck "
            "(out-of-bounds and misaligned access), racecheck "
            "(shared-memory hazards between threads in a block), "
            "initcheck (reads of uninitialized device memory), synccheck "
            "(divergent/invalid barrier usage) -- each a dedicated, "
            "mechanical detector, distinct from comparing output to an "
            "oracle",
        why_it_exists="an oracle comparison (Hawking's actual practice, "
            "per DEBUGGING.numerical_diagnostics) catches a WRONG ANSWER "
            "but not a race that happens to produce the RIGHT answer on "
            "this run and a wrong one on the next -- mechanical hazard "
            "detection catches the hazard itself, independent of whether "
            "it manifested this time",
        underlying_problem="a race, an out-of-bounds access, or a read of "
            "uninitialized memory can be silently harmless on one run and "
            "corrupt data on the next; testing against an oracle cannot "
            "distinguish 'correct' from 'got lucky this run'",
        apple_equivalent="Metal API Validation (enabled via "
            "MTL_DEBUG_LAYER / the Xcode scheme's Metal Validation "
            "setting) and GPU frame capture together cover some of this "
            "-- resource-usage validation and shader debugging -- but a "
            "dedicated racecheck-equivalent (detecting a "
            "THREADGROUP-MEMORY data race specifically) was not confirmed "
            "present in the documentation available on this machine",
        hawking_equivalent="DEBUGGING.numerical_diagnostics' "
            "verify-before-time discipline: 'every candidate verified "
            "against an oracle BEFORE it is timed' -- this is the "
            "oracle-comparison half only; its own remaining_limitation "
            "already names the gap this entry tracks explicitly: 'no race "
            "detection, no OOB checking, no device-side assertion'",
        semantic_gap="LARGE",
        performance_gap=unmeasured("not a performance capability -- a "
            "diagnostic capability, same reasoning as "
            "DEBUGGING.numerical_diagnostics"),
        priority="P1",
        test_corpus="none; this is the specific gap "
            "DEBUGGING.numerical_diagnostics already named without giving "
            "it its own census line",
        current_winner="NOT APPLICABLE",
        remaining_limitation="AIR's atomic-reduction strategy "
            "(EXECUTION.atomics) explicitly notes summation order is "
            "NONDETERMINISTIC -- that is precisely the kind of property a "
            "racecheck-equivalent tool would characterize systematically "
            "instead of leaving as a stated caveat",
    ),

    entry(
        capability_id="PROFILING.kernel_trace_and_provenance",
        cuda_mechanism="CUPTI activity records + NVTX ranges: a causal "
            "trace tying a specific kernel launch back to the host call "
            "site (and, with NVTX, a user-named logical range like "
            "'forward pass layer 12') that produced it, with correlation "
            "IDs linking async host submission to eventual device "
            "execution",
        why_it_exists="on an async device, 'what is running right now' "
            "and 'what CPU code caused it' are not the same question, and "
            "answering the second one (attributing device time back to "
            "source-level provenance) needs an explicit correlation "
            "mechanism, not just a timestamp",
        underlying_problem="attributing device-side execution back to the "
            "host-side logical operation that caused it, across an "
            "asynchronous boundary",
        apple_equivalent="MTLCommandBuffer.label and "
            "MTLComputeCommandEncoder.pushDebugGroup/popDebugGroup provide "
            "the same NAMED-RANGE provenance capability, visible in "
            "Instruments and GPU frame capture",
        hawking_equivalent="none observed; nothing in bench.py or AIR "
            "labels a command buffer or pushes a debug group, so a "
            "captured Metal trace of a Hawking run would show anonymous "
            "dispatches rather than AIR's own op names",
        semantic_gap="PARTIAL",
        performance_gap=unmeasured("not a performance capability -- a "
            "debuggability/attribution capability"),
        priority="P2",
        test_corpus="none",
        current_winner="UNKNOWN",
        remaining_limitation="this is a small, concrete, unbuilt door "
            "(label command buffers and push debug groups with AIR op "
            "names) that would make every OTHER profiling gap in this "
            "family easier to close, since a captured trace would at "
            "least be readable instead of anonymous",
    ),

]


# The 20 seed entries, frozen here VERBATIM (this is literally the JSON that
# was on disk before the G055 census pass, re-embedded rather than read back
# from receipts/headless/CUDA_CAPABILITY_LEDGER.json) so that this module
# stays the stable source of truth even after that file has been regenerated
# once -- reading the seed back from an already-expanded output file would
# make a second run see 68 "seed" entries and collide with NEW_ENTRIES on
# duplicate capability_id. Keeping it frozen here means the census's own
# "keep every existing entry byte-identical" rule is enforced by
# construction, not by remembering not to re-run a generator twice.
_SEED_ENTRIES_JSON = r"""[
 {
  "capability_id": "MEMORY.host_device_copy",
  "cuda_mechanism": "cudaMalloc + cudaMemcpy H2D/D2H around every kernel",
  "why_it_exists": "a discrete GPU has its own physical memory the CPU cannot address",
  "underlying_problem": "data must physically cross PCIe to be computed on",
  "apple_equivalent": "none needed; CPU and GPU address the same physical memory",
  "hawking_equivalent": "cuda_runtime.execute_host UNIFIED mode: cudaMalloc becomes an alias and both cudaMemcpy directions become no-ops, guarded by may_delete_copies()",
  "semantic_gap": "DELETED_BY_UNIFIED_MEMORY",
  "performance_gap": {
   "measured": true,
   "value": "eliminating the copies is worth 7.4x on the copy itself (P6 microbenchmark) and 1.7x inside a translated host program",
   "receipt": "receipts/headless/ACCELERATOR_FRONT_G_P6.json + ACCELERATOR_C2M_T1_RUNTIME.json"
  },
  "priority": "P0",
  "test_corpus": "vadd and vmul host programs at 4096 and 2^22 in both modes, plus a hazard program that writes its host buffer between the copy and the launch",
  "current_winner": "Hawking resident-operand path",
  "remaining_limitation": "THE DELETION IS CONDITIONAL, and calling this gap deleted without the condition would be wrong. A CUDA host-to-device copy is a SNAPSHOT; an alias is a LIVE REFERENCE. A program that writes the host buffer after the copy and before the launch DIVERGES -- demonstrated at 999.36 in exactly one element with every other element bit-identical. may_delete_copies() refuses those programs by name. Also unhandled: device-to-device copies, async copies, pinned and managed memory, all refused rather than emulated.",
  "evidence": "receipts/headless/ACCELERATOR_FRONT_G_P6.json"
 },
 {
  "capability_id": "MEMORY.quantized_weight_access",
  "cuda_mechanism": "dequantize to dense then call cuBLAS, or a fused dequant-GEMM kernel",
  "why_it_exists": "tensor cores consume dense formats; compact formats must be expanded",
  "underlying_problem": "compact weights are not directly consumable by library GEMM",
  "apple_equivalent": "none provided; MLX quantized ops cover its own formats only",
  "hawking_equivalent": "representation-native matvec reading packed nibbles + f16 group scales",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "no advantage below ~64 MiB; 2.08x at 256 MiB where it becomes bandwidth-bound",
   "receipt": "receipts/headless/ACCELERATOR_FRONT_D_P7.json"
  },
  "priority": "P0",
  "test_corpus": "real Qwen3-30B-A3B expert tensor at ws_rtn_q4_g64",
  "current_winner": "Hawking native kernel above the crossover; nothing below it",
  "remaining_limitation": "matvec only; no GEMM, no batched form, and one expert sits BELOW the crossover so single-expert use gains nothing",
  "evidence": "receipts/headless/ACCELERATOR_FRONT_D_P7.json"
 },
 {
  "capability_id": "COMPILER.kernel_fusion",
  "cuda_mechanism": "hand-fused kernels, or a compiler like Triton/torch.compile",
  "why_it_exists": "each launch re-reads and re-writes memory an op chain need not touch",
  "underlying_problem": "unfused chains are bandwidth-bound on intermediates",
  "apple_equivalent": "MLX does not fuse an arbitrary elementwise chain automatically",
  "hawking_equivalent": "AIR lowers a multi-op SSA chain into one kernel",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "1.71x over the naive 3-kernel translation, 2.62x over MLX",
   "receipt": "receipts/headless/ACCELERATOR_FRONT_D_P5.json"
  },
  "priority": "P0",
  "test_corpus": "mul->relu->silu chain, 2^24 f32",
  "current_winner": "Hawking fused AIR program",
  "remaining_limitation": "elementwise chains over one shape and dtype only",
  "evidence": "receipts/headless/ACCELERATOR_FRONT_D_P5.json"
 },
 {
  "capability_id": "COMPILER.jit_kernel_compilation",
  "cuda_mechanism": "NVRTC / driver JIT of PTX at runtime",
  "why_it_exists": "shapes and constants are not known until run time",
  "underlying_problem": "AOT compilation cannot specialise on runtime values",
  "apple_equivalent": "Metal runtime library compilation; MLX exposes it via metal_kernel",
  "hawking_equivalent": "AIR emits MSL, MLX JIT-compiles it",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "36.8 ms per DISTINCT kernel on this machine, near-flat in source size (102 chars 36.8 ms, 5990 chars 43.6 ms) and CACHED on repeat: the same name+source recompiles in 0.24 ms. Compile is 23.2% of a WorkUnit's first-run excess; MLX's own first-use costs are 55.1%, dominated by the FIRST mx.array at 151 ms, which is device/allocator init and NOT compilation.",
   "receipt": "receipts/headless/ACCELERATOR_JIT_COMPILE.json"
  },
  "priority": "P1",
  "test_corpus": "the six AIR programs of P3",
  "current_winner": "MLX JIT (retained; nothing here replaces it)",
  "remaining_limitation": "measured on mx.fast.metal_kernel JIT only; xcrun metal AOT is ABSENT on this machine so an AOT metallib comparison cannot be made, and CUDA's own nvrtc latency is unmeasured because no NVIDIA hardware exists"
 },
 {
  "capability_id": "COMPILER.specialization_constants",
  "cuda_mechanism": "template parameters and __constant__ values baked at compile time",
  "why_it_exists": "a constant known at compile time removes a load and enables unrolling",
  "underlying_problem": "runtime values block optimisation",
  "apple_equivalent": "Metal function constants",
  "hawking_equivalent": "AirProgram.specialization emits const floats into the MSL",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "NO MEASURABLE BENEFIT on this backend. A literal scalar against a buffer read is INDISTINGUISHABLE (1.62% margin vs 7.95% noise); a compute-bound loop bound where unrolling should pay is INDISTINGUISHABLE with the literal 3.09% SLOWER, both arms clean at 5.25%/4.96%.",
   "receipt": "receipts/headless/ACCELERATOR_SPEC_CONSTANTS.json"
  },
  "priority": "P2",
  "test_corpus": "saxpy->relu with ALPHA as a specialization constant",
  "current_winner": "UNKNOWN \u2014 never measured against a non-specialised variant",
  "remaining_limitation": "two shapes on one machine; whether the literal loop is actually unrolled cannot be checked because xcrun metal is ABSENT so generated code cannot be inspected; and a constant that would enable BRANCH ELIMINATION or a different memory layout is untested"
 },
 {
  "capability_id": "EXECUTION.grid_block_thread",
  "cuda_mechanism": "blockIdx/blockDim/threadIdx and the launch configuration",
  "why_it_exists": "a portable way to index a data-parallel launch",
  "underlying_problem": "mapping work onto a hierarchy of parallel units",
  "apple_equivalent": "thread_position_in_grid and threadgroup sizing",
  "hawking_equivalent": "C2M recognises the canonical global index and maps it",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": false,
   "reason": "indexing is not independently measurable"
  },
  "priority": "P0",
  "test_corpus": "the four translated kernels of P4",
  "current_winner": "Hawking C2M for the canonical form",
  "remaining_limitation": "ONLY the 1-D canonical form; no 2-D or 3-D grids, no block-relative indexing, no grid-stride loops"
 },
 {
  "capability_id": "EXECUTION.shared_memory",
  "cuda_mechanism": "__shared__ arrays with block-scoped lifetime",
  "why_it_exists": "a software-managed scratchpad far faster than global memory",
  "underlying_problem": "reuse within a block should not go to DRAM",
  "apple_equivalent": "threadgroup address space",
  "hawking_equivalent": "tools/accelerator/gemm.py tiled GEMM",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "MLX 12733 GFLOP/s vs ours 2683 at 2048^2 = 4.7x in MLX's favour",
   "receipt": "receipts/headless/ACCELERATOR_GEMM.json"
  },
  "priority": "P0",
  "test_corpus": "512/1024/2048 square f32 GEMM vs MLX, plus 2 negative controls",
  "current_winner": "MLX, retained (\u00a7141) -- replacement is not justified by evidence",
  "remaining_limitation": "threadgroup tiles execute; no double buffering, no bank-conflict work"
 },
 {
  "capability_id": "EXECUTION.atomics",
  "cuda_mechanism": "atomicAdd/atomicCAS on global and shared memory",
  "why_it_exists": "concurrent accumulation without a separate reduction pass",
  "underlying_problem": "many threads updating one location",
  "apple_equivalent": "atomic_fetch_add_explicit in MSL",
  "hawking_equivalent": "atomic_fetch_add_explicit in AIR's atomic reduction strategy",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "single-pass atomic sum is 3.05x/1.80x over two-stage; INDISTINGUISHABLE from MLX",
   "receipt": "receipts/headless/ACCELERATOR_ATOMICS.json"
  },
  "priority": "P1",
  "test_corpus": "sum at 2^20 and 2^24, atomic vs two-stage vs MLX, plus a refusal for max",
  "current_winner": "MLX and AIR indistinguishable; MLX retained as incumbent",
  "remaining_limitation": "float ADD only. No atomic max (Metal lacks it here, so the strategy refuses), no CAS, no integer atomics, no shared-memory atomics. Summation order is NONDETERMINISTIC and the result is 3.4x less exact than the two-stage form."
 },
 {
  "capability_id": "EXECUTION.warp_primitives",
  "cuda_mechanism": "__shfl_sync, __ballot_sync, warp-level reduction",
  "why_it_exists": "intra-warp exchange without shared memory or barriers",
  "underlying_problem": "lane-to-lane communication inside a scheduling unit",
  "apple_equivalent": "simd_shuffle and friends over a 32-wide SIMD group",
  "hawking_equivalent": "simdgroup_float8x8 matrix ops in AIR's simdgroup matmul strategy",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "1.6x over a scalar tile; 91-98% of MLX at 512/1024 but 36.7% at 2048",
   "receipt": "receipts/headless/ACCELERATOR_SIMDGROUP_GEMM.json"
  },
  "priority": "P1",
  "test_corpus": "512/1024/2048 f32 GEMM, simdgroup vs tiled vs MLX",
  "current_winner": "MLX, retained",
  "remaining_limitation": "simdgroup MATRIX ops work; shuffle, ballot and vote are still untouched, and Apple's divergence semantics remain uncompared to CUDA's"
 },
 {
  "capability_id": "EXECUTION.synchronization",
  "cuda_mechanism": "__syncthreads and cooperative groups",
  "why_it_exists": "phase separation inside a block",
  "underlying_problem": "ordering between cooperating threads",
  "apple_equivalent": "threadgroup_barrier",
  "hawking_equivalent": "tools/accelerator/gemm.py tiled GEMM",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "MLX 12733 GFLOP/s vs ours 2683 at 2048^2 = 4.7x in MLX's favour",
   "receipt": "receipts/headless/ACCELERATOR_GEMM.json"
  },
  "priority": "P0",
  "test_corpus": "512/1024/2048 square f32 GEMM vs MLX, plus 2 negative controls",
  "current_winner": "MLX, retained (\u00a7141) -- replacement is not justified by evidence",
  "remaining_limitation": "threadgroup_barrier executes; SIMDGROUP and DEVICE scope untested, and AIR can represent a barrier but cannot lower one"
 },
 {
  "capability_id": "MATH.gemm",
  "cuda_mechanism": "cuBLAS GEMM on tensor cores",
  "why_it_exists": "matrix multiply is the dominant cost of a transformer",
  "underlying_problem": "peak arithmetic needs tiling, blocking and hardware MMA units",
  "apple_equivalent": "MLX matmul; Apple matrix coprocessor via MPS",
  "hawking_equivalent": "AIR simdgroup matmul with 2x2 register blocking",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "parity with MLX at 1024 and 2048; MLX wins 1.24x at 4096",
   "receipt": "receipts/headless/ACCELERATOR_REGISTER_BLOCKING.json"
  },
  "priority": "P0",
  "test_corpus": "512/1024/2048 square f32 GEMM vs MLX, plus 2 negative controls",
  "current_winner": "MLX at 4096 and above; parity below, so MLX is retained",
  "remaining_limitation": "parity to 2048, then MLX pulls ahead 1.24x at 4096; no threadgroup-level staging, no double buffering, no batching, no transpose, f32 only"
 },
 {
  "capability_id": "MATH.reduction_and_scan",
  "cuda_mechanism": "CUB block/device reduce and scan",
  "why_it_exists": "reductions are latency-critical and hard to write well",
  "underlying_problem": "tree reduction across a hierarchy with correct synchronization",
  "apple_equivalent": "MLX reductions; simd_sum",
  "hawking_equivalent": "AIR two-stage reduction (sum, max) and three-phase scan (sum, max) with two strategies each",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "scan beats mx.cumsum 8.09x at 2^24 and 1.84x at 2^20; reduction loses 1.75x to mx.sum at 2^24 and is not distinguishable at 2^20. The earlier 'atomics win 3.05x on launch count' is REFUTED -- the cost was host round trips, not launches.",
   "receipt": "receipts/headless/ACCELERATOR_SCAN.json + ACCELERATOR_SYNC_CORRECTION.json"
  },
  "priority": "P1",
  "test_corpus": "sum/max reduce and inclusive/exclusive/max scan at 7, 256, 1000, non-power-of-two 4099, 2^16, 2^20, 2^22, 2^24, graded against an f64 oracle and cross-checked against mx.cumsum",
  "current_winner": "SPLIT: Hawking for scan (mx.cumsum runs at 19 GB/s on a 589 GB/s machine here); MLX for reduction, retained",
  "remaining_limitation": "f32 only, 1-D only. Scan is sum and max; exclusive is sum-only (derived as inclusive minus x) and the simd_prefix strategy is sum-only because Metal has no prefix max. No segmented scan, no scan-by-key, no multi-dimensional or strided axis. Nothing here has run inside a model."
 },
 {
  "capability_id": "RUNTIME.streams_and_events",
  "cuda_mechanism": "cudaStream_t, cudaEvent_t, async launch and dependency ordering",
  "why_it_exists": "overlap of compute and transfer, and explicit dependency expression",
  "underlying_problem": "serialising independent work wastes the device",
  "apple_equivalent": "MTLCommandQueue, command buffers, MTLEvent",
  "hawking_equivalent": "AirGraph expresses independence as DAG width; there are no named streams and no events",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "independent branches overlap only weakly: 1.09x at k=8 and 1.16x at k=16 over a serial chain of the same work at 2^20, and NO_CLAIM at 4096. Concurrency is the SMALLER half of what a graph buys here",
   "receipt": "receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json"
  },
  "priority": "P1",
  "test_corpus": "fan (depth 1, width k) vs chain (depth k, width 1), same total work",
  "current_winner": "MLX's own scheduling, retained; we do not control it",
  "remaining_limitation": "No named streams, no events, no cross-stream dependencies, no host callbacks, no priority. Only DAG width is expressed, and the runtime decides what to do with it."
 },
 {
  "capability_id": "RUNTIME.graphs",
  "cuda_mechanism": "CUDA Graphs: capture once, replay with minimal launch overhead",
  "why_it_exists": "per-launch CPU cost dominates short kernels in a tight loop",
  "underlying_problem": "re-encoding an unchanged command stream every iteration",
  "apple_equivalent": "indirect command buffers; reusable command buffers",
  "hawking_equivalent": "AIR AirGraph: a recorded dependency DAG executed as one command-buffer submission with one host sync",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "batching k dispatches into one submission cuts the marginal cost of an added dispatch 15.5x (0.168 -> 0.011 ms); total 2.9x at k=4, 4.8x at k=8, 8.0x at k=16",
   "receipt": "receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json"
  },
  "priority": "P0",
  "test_corpus": "chain and fan of k=1,2,4,8,16 elementwise dispatches at 4096 and 2^20",
  "current_winner": "Hawking AirGraph (MLX's laziness supplies the batching; AirGraph makes it explicit and measurable)",
  "remaining_limitation": "No graph instantiation or update reuse, no conditional or dynamic nodes, no cross-device nodes. The DAG is re-recorded per call; only the compiled kernels are cached."
 },
 {
  "capability_id": "PROFILING.kernel_timing",
  "cuda_mechanism": "cudaEventElapsedTime, Nsight Compute counters",
  "why_it_exists": "optimisation without measurement is guessing",
  "underlying_problem": "attributing time to a specific kernel on an async device",
  "apple_equivalent": "Metal GPU counters; Instruments",
  "hawking_equivalent": "bench.py wall-clock with warmup, IQR gate and an implausibility rejection",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": false,
   "reason": "the instrument is not itself benchmarked"
  },
  "priority": "P0",
  "test_corpus": "every timed arm in P5, P6, P7",
  "current_winner": "Hawking bench.py for comparisons",
  "remaining_limitation": "WALL CLOCK ONLY \u2014 no DRAM byte counters, no occupancy, no stall reasons, so mechanism is inferred rather than observed"
 },
 {
  "capability_id": "DEBUGGING.numerical_diagnostics",
  "cuda_mechanism": "compute-sanitizer, device printf, cuda-gdb",
  "why_it_exists": "silent numerical corruption is the worst failure mode",
  "underlying_problem": "a wrong answer at speed looks like a right answer",
  "apple_equivalent": "Metal shader validation, GPU frame capture",
  "hawking_equivalent": "every candidate verified against an oracle BEFORE it is timed",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": false,
   "reason": "not a performance capability"
  },
  "priority": "P0",
  "test_corpus": "P3 six programs, P4 four kernels, forge 15 variants",
  "current_winner": "Hawking verify-before-time discipline",
  "remaining_limitation": "oracle comparison only; no race detection, no OOB checking, no device-side assertion"
 },
 {
  "capability_id": "MULTI_DEVICE.peer_access",
  "cuda_mechanism": "cudaDeviceEnablePeerAccess, NVLink, NCCL collectives",
  "why_it_exists": "multi-GPU training and inference need direct device-to-device paths",
  "underlying_problem": "moving tensors between devices without a host bounce",
  "apple_equivalent": "none on a single SoC; Thunderbolt/RDMA between machines",
  "hawking_equivalent": "HUMF models the domains but only a MOCK external one exists",
  "semantic_gap": "LARGE",
  "performance_gap": {
   "measured": false,
   "reason": "no real second device exists; every transport number so far is SIMULATED and may not be cited as evidence"
  },
  "priority": "P2",
  "test_corpus": "HUMF P8 across APPLE_UM and MOCK_EXTERNAL_VRAM",
  "current_winner": "NOT APPLICABLE \u2014 no second physical device",
  "remaining_limitation": "EGB-D0 has not been reached; no hardware is present"
 },
 {
  "capability_id": "MATH.softmax",
  "cuda_mechanism": "hand-fused softmax kernels, or cuDNN's",
  "why_it_exists": "softmax is on the critical path of every attention layer and is bandwidth-bound, so its passes over memory dominate",
  "underlying_problem": "three dependent reductions over the same row, naively three passes",
  "apple_equivalent": "mx.softmax",
  "hawking_equivalent": "AIR fused row softmax, one threadgroup per row, two barriers",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "1.46x/1.35x over the naive three-pass shape; INDISTINGUISHABLE from MLX",
   "receipt": "receipts/headless/ACCELERATOR_SOFTMAX.json"
  },
  "priority": "P0",
  "test_corpus": "64x128, 512x1024, 128x4099, 8x3, an overflow row, and a barrier-stripped negative control",
  "current_winner": "MLX and AIR are indistinguishable; MLX retained as incumbent",
  "remaining_limitation": "f32 only, last axis only, no masking, no causal variant. ATTENTION ITSELF REMAINS ABSENT -- this is a piece of it.",
  "evidence": "receipts/headless/ACCELERATOR_SOFTMAX.json"
 },
 {
  "capability_id": "MATH.attention",
  "cuda_mechanism": "FlashAttention and cuDNN fused attention kernels",
  "why_it_exists": "attention materialises an O(seq^2) score matrix that dominates memory traffic and capacity at long sequence",
  "underlying_problem": "a chain of matmul, softmax and matmul where the intermediates are far larger than the inputs",
  "apple_equivalent": "mx.fast.scaled_dot_product_attention",
  "hawking_equivalent": "AIR fused single-head attention, score row in threadgroup memory",
  "semantic_gap": "PARTIAL",
  "performance_gap": {
   "measured": true,
   "value": "loses to naive 1.37x at 1024; a simdgroup rebuild is SLOWER still (0.70x)",
   "receipt": "receipts/headless/ACCELERATOR_ATTENTION_SIMDGROUP_REFUTED.json"
  },
  "priority": "P0",
  "test_corpus": "512/1024/2048 with head_dim 64 and 128, a causal variant, a non-power-of-two 37x40, and a refusal case at seq 100000",
  "current_winner": "MLX, and at 2048 even the NAIVE three-call path beats ours",
  "remaining_limitation": "not flash-attention: no online softmax, score row must fit threadgroup memory. Diagnosis history: 'scalar dot products' REFUTED; 'occupancy' DIRECTIONALLY CONFIRMED -- 4 simdgroups on the same query block take the simdgroup variant from 0.70x losing to INDISTINGUISHABLE from the scalar kernel. But the NAIVE three-MLX-call path STILL WINS 1.08x with clean arms, so fused attention is not useful here at these shapes. Single head, f32, no batching, no KV cache.",
  "evidence": "receipts/headless/ACCELERATOR_ATTENTION.json"
 },
 {
  "capability_id": "COMPILER.frontend_parsing",
  "cuda_mechanism": "nvcc / clang CUDA frontend: a full C++ frontend for device and host code",
  "why_it_exists": "CUDA's corpus is written in C++, so a compiler must accept C++",
  "underlying_problem": "turning arbitrary source into an executable device program",
  "apple_equivalent": "the Metal compiler accepts MSL, a different C++ dialect; nothing accepts CUDA",
  "hawking_equivalent": "C2M's FOUR narrow doors: T0 expression matching over 4 elementwise forms, a PTX instruction subset, the T1 host runtime statement subset, and T2 algorithm-level IDIOM RECOGNITION over 2 kernels",
  "semantic_gap": "LARGE",
  "performance_gap": {
   "measured": false,
   "reason": "a frontend gap is about COVERAGE, not speed; measuring a translation rate would say nothing about how much of CUDA is reachable"
  },
  "priority": "P1",
  "test_corpus": "4 elementwise kernels, the pinned seed's 291 .ptx files, 4 host programs, 2 recognised idioms and 12 near-miss kernels",
  "current_winner": "the pinned seed Lulzx/cuda-metal, which has a REAL clang-based CUDA frontend and builds here; C2M does not compete with it on coverage and is not claimed to",
  "remaining_limitation": "THIS IS THE LARGEST REMAINING SOFTWARE GAP AND IT IS NOT NARROWING QUICKLY. None of the four doors parses C++. PTX coverage measured 1.0% of the seed's own corpus. Idiom recognition covers exactly 2 textbook kernels and REFUSES the same algorithm written with different variable names or a different guard style. Adding doors widens what is reachable without widening what is COMPILABLE, and saying so keeps a tier number from reading as CUDA coverage."
 }
]"""


def _load_seed_entries() -> list[dict[str, Any]]:
    return json.loads(_SEED_ENTRIES_JSON)


def census_ledger() -> dict[str, Any]:
    """The full ledger: the 20-entry seed plus this pass's 48-entry census."""
    return build(_load_seed_entries() + NEW_ENTRIES)


if __name__ == "__main__":
    led = census_ledger()
    out = REPO / "receipts/headless/CUDA_CAPABILITY_LEDGER.json"
    out.write_text(json.dumps(led, indent=1) + "\n")
    print(f"wrote {led['count']} entries "
          f"({led['performance_gaps_measured']} measured, "
          f"{led['performance_gaps_unmeasured']} unmeasured) to {out}")
