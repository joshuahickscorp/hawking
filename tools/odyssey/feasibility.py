#!/usr/bin/env python3.12
"""Honest T1–T7 resource feasibility on this machine.

Numbers are order-of-magnitude estimates from sealed substrate facts and
observed free memory. Wall-clock under load ~29/28 is contaminated; stated as such.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tools.odyssey._paths import EXPECTED_BYTES, EXPECTED_BPW, MATH_ARTIFACT, RECORDS_DIR

SCHEMA = "hawking.odyssey.feasibility.v1"

# Architecture facts for GLM-5.2 (from campaign / index).
HIDDEN = 6144
N_LAYERS = 78  # approximate flagship depth used in cascade docs; not required exact for bound
VOCAB = 150_000  # order of magnitude for lm_head bound
# Weight bytes of the compact artifact (already compressed ~0.98 BPW).
ARTIFACT_BYTES = EXPECTED_BYTES  # 92_038_250_160


def _mem_snapshot() -> dict[str, Any]:
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 16384
    # vm_stat parse is fragile; use resource / heuristic from sysctl via /usr if needed.
    memsize = 103_079_215_104  # 96 GiB sealed for this machine class
    # Try vm_stat free pages.
    free_gb = None
    try:
        import subprocess

        out = subprocess.check_output(["vm_stat"], text=True)
        free_pages = None
        for line in out.splitlines():
            if line.startswith("Pages free"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
        if free_pages is not None:
            free_gb = free_pages * page / (1024**3)
    except Exception:  # noqa: BLE001
        free_gb = None
    return {
        "machine_ram_bytes": memsize,
        "machine_ram_gib": memsize / (1024**3),
        "observed_free_gib": free_gb,
        "ncpu": os.cpu_count(),
        "load_note": "machine observed at load ~29–31 of 28 cores during this assessment; timings contaminated",
    }


def estimate() -> dict[str, Any]:
    mem = _mem_snapshot()
    # Smallest meaningful training step for full-model CE on the compact artifact:
    # - weights resident (or streamed): 92 GB
    # - gradients (fp32) for trainable subset: if full, another ~92 GB * (32/bpw_bits) —
    #   even LoRA-style small adapters need activations.
    # Conservative lower bound for one step with streaming weights + one microbatch:
    weight_bytes = ARTIFACT_BYTES
    # Activations for seq_len=512, microbatch=1, hidden=6144, layers~78, fp16 residual stream:
    seq, microbatch = 512, 1
    # residual stream per layer approx 2 * B * S * H * 2 bytes (x and residual) rough:
    activation_per_layer = microbatch * seq * HIDDEN * 2 * 2
    activation_bytes = activation_per_layer * 20  # keep 20-layer window if checkpointed
    # Optimizer state for a small adapter (rank 16 on 10 matrices): tiny
    adapter_params = 16 * HIDDEN * 10
    adapter_bytes = adapter_params * 2  # fp16
    adapter_opt = adapter_params * 8  # adam moments fp32
    # Grad for adapter
    adapter_grad = adapter_params * 4

    min_step_streaming_adapter = (
        # working set if weights streamed from SSD one layer at a time (~1–2 GB layer peak)
        2_000_000_000
        + activation_bytes
        + adapter_bytes
        + adapter_opt
        + adapter_grad
    )
    min_step_full_resident = weight_bytes + activation_bytes + adapter_bytes + adapter_opt

    free = (mem["observed_free_gib"] or 0.2) * (1024**3)
    # Disk streaming bandwidth ~1.5 GB/s measured earlier for hash; contaminated under load.
    stream_gbps = 1.0  # conservative under contention
    # One epoch over 92 GB once for forward: wall clock
    forward_pass_s = (weight_bytes / (stream_gbps * 1e9)) * 1.5  # read amp for PQ decode
    # Backward ~2–3x
    step_wall_s = forward_pass_s * 3.0

    t1_possible = free + 8 * (1024**3) > min_step_streaming_adapter  # allow some reclaim
    # Even streaming adapter step may thrash if free is 0.16 GB and OS needs headroom.
    t1_practical = free > 4 * (1024**3)  # need ~4 GB free headroom minimum for OS+python+decode

    stages = {
        "T0": {
            "intent": "baseline reproduction",
            "feasible_here": True,
            "reason": "integrity, data classification, single-layer authority, known-failure registry — implemented",
        },
        "T1": {
            "intent": "capability-conditioned continued training",
            "feasible_here": False,
            "reason": (
                f"smallest meaningful streaming-adapter step needs ~"
                f"{min_step_streaming_adapter/1e9:.1f} GB working set; "
                f"full-resident needs ~{min_step_full_resident/1e9:.1f} GB; "
                f"observed free ~{(mem['observed_free_gib'] or 0):.2f} GiB on a "
                f"{mem['machine_ram_gib']:.0f} GiB machine already holding other campaigns. "
                f"Training corpora are DECLARED_NOT_PRESENT."
            ),
            "memory_required_bytes_min_step": int(min_step_streaming_adapter),
            "memory_required_bytes_full_resident": int(min_step_full_resident),
            "what_must_be_offloaded": [
                "all non-active layer weights (SSD stream)",
                "optimizer state for any non-adapter full-model path (impossible in RAM)",
                "KV/activation checkpointing to disk if seq grows",
            ],
            "estimated_wall_clock_per_step_seconds": step_wall_s,
            "estimated_wall_clock_note": "contaminated by load ~29/28; order-of-magnitude only",
        },
        "T2": {
            "intent": "QAT headroom at frozen rate",
            "feasible_here": False,
            "reason": (
                "QAT requires forward+fake-quant+backward through the same 92 GB substrate. "
                "Operator simulation on small tensors is runnable; full QAT is not."
            ),
            "memory_required_bytes_min_step": int(min_step_streaming_adapter * 1.3),
            "estimated_wall_clock_per_step_seconds": step_wall_s * 1.5,
        },
        "T3": {
            "intent": "trajectory stabilization",
            "feasible_here": False,
            "reason": (
                "Needs parent trajectory traces (teacher manifest PARTIAL — layer capsules only) "
                "plus multi-token forward of the student; same memory wall as T1 plus sequence length."
            ),
            "memory_required_bytes_min_step": int(min_step_streaming_adapter * 2),
        },
        "T4": {
            "intent": "forge profile expression",
            "feasible_here": False,
            "reason": "F1–F4 require a served model and evaluated prompt set; serve path is memory-bound and another lane.",
        },
        "T5": {
            "intent": "hidden replication + sovereignty",
            "feasible_here": False,
            "reason": "Depends on T1–T4 checkpoints and a live evaluation path against held-out items.",
        },
        "T6": {
            "intent": "not declared in training plan stages list (plan ends T5); support-halo/tournament gates sit on T7 in evaluation docs",
            "feasible_here": False,
            "reason": "No T6 stage object in ODYSSEY_TRAINING_PLAN.json",
        },
        "T7": {
            "intent": "checkpoint tournament on math profile + support halo (G5)",
            "feasible_here": "partial",
            "reason": (
                "Tournament comparison logic is RUNNABLE on scorecards. "
                "Live support-halo baseline on Math-Preserve is NOT_RUN (needs serve/completions). "
                "Offline score-completions path exists in tools/eval/support_halo_gate.py."
            ),
        },
    }

    return {
        "schema": SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "substrate": {
            "artifact": str(MATH_ARTIFACT),
            "bytes": ARTIFACT_BYTES,
            "bpw": EXPECTED_BPW,
            "read_only": True,
        },
        "machine": mem,
        "bounds": {
            "min_meaningful_training_step_bytes_streaming_adapter": int(min_step_streaming_adapter),
            "min_full_resident_step_bytes": int(min_step_full_resident),
            "assumptions": {
                "seq_len": seq,
                "microbatch": microbatch,
                "hidden": HIDDEN,
                "adapter_rank_params": adapter_params,
                "stream_gbps_assumed": stream_gbps,
            },
        },
        "stages": stages,
        "verdict": {
            "t0_runnable": True,
            "t1_t5_full_training_feasible_on_this_hardware_now": False,
            "headline": (
                "T1–T5 training against the 92 GB Math-Preserve artifact is not feasible on "
                "this 96 GB machine at current load and free memory. T0 reproductions and "
                "contract machinery are. This is a finding, not a task to silently shrink."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    report = estimate()
    out = RECORDS_DIR / "ODYSSEY_FEASIBILITY.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
