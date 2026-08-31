#!/usr/bin/env python3
"""G075: the resident, reprofiled on the body that actually runs.

The ladder in PATH_TO_71 and RESIDENT_71TPS_CAUSAL_BUDGET was arithmetic over a
28.722 ms token. deltanet_widen_f4 landed a measured token-identical win, so
every rung above it was computed against a body that no longer exists.

Reprofiling it turned out to need three things nobody had listed:

  1. A REBUILD. widen_f4's selection path landed at 71e186909 (2026-08-31 04:46)
     and the newest release binary was 2026-08-30 23:35. It could not select the
     mode whatever kernels it contained.

  2. THE RIGHT PROFILE. The serving binary was release-fast, whose own Cargo.toml
     comment reads "Correctness-testing profile ONLY... NEVER benchmark with
     this: TPS numbers must come from release (lto=fat, codegen-units=1)". A TPS
     read off it was never eligible to be the protected absolute.

  3. A PROTECTED WINDOW. ModelLake was streaming five downloads. This
     obligation's own text says absolutes move with load while ratios do not, so
     the five hf processes were SIGSTOPped for each measurement and resumed after.

The proof widen_f4 is live is not a strings(1) grep of the binary - I tried that
and it was wrong, because "widen_f4" is only the as_str() display form and
dead-code elimination removes it under fat LTO. The proof is that
qwen38_gated_delta_decode_vi_simd_ba_f4 appears in dispatched_kernels_rep0 of the
run itself.

    python3 tools/future/resident_reprofile.py --build
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/resident_reprofile.py"
RECEIPT_NAME = "RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

GREEDY_BASELINE = REPO / "receipts" / "future" / "_G075_greedy_baseline.json"
GREEDY_WIDEN = REPO / "receipts" / "future" / "_G075_greedy_widen_f4.json"
ORGAN_WIDEN = REPO / "receipts" / "future" / "_G075_organ_widen_f4.json"

F4_KERNEL = "qwen38_gated_delta_decode_vi_simd_ba_f4"
STALE_LADDER_MS = 28.722


class ReprofileRefused(RuntimeError):
    """A raw input is missing, or the arms are not comparable."""


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        try:
            shown = path.relative_to(REPO)
        except ValueError:
            shown = path
        raise ReprofileRefused(
            f"{shown} is not on disk; run the protected window first. "
            "This receipt is never assembled from remembered numbers."
        )
    return json.loads(path.read_text())


def _control(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["control_uninstrumented_generate_greedy"]


def arms() -> dict[str, Any]:
    """The paired A/B. Token-identical or the comparison is void."""
    a, b = _control(_load(GREEDY_BASELINE)), _control(_load(GREEDY_WIDEN))
    if a["new_token_ids"] != b["new_token_ids"]:
        raise ReprofileRefused(
            "the two arms produced different tokens; a speed delta between "
            "different outputs is not a speedup"
        )
    if a["decode_steps"] != b["decode_steps"]:
        raise ReprofileRefused("arms differ in decode_steps; not a paired comparison")
    base_ms = a["decode_wall_ns_per_token"] / 1e6
    widen_ms = b["decode_wall_ns_per_token"] / 1e6
    return {
        "token_identical": True,
        "decode_steps": int(a["decode_steps"]),
        "baseline_ms_per_token": round(base_ms, 4),
        "widen_f4_ms_per_token": round(widen_ms, 4),
        "baseline_tps": round(1000.0 / base_ms, 3),
        "widen_f4_tps": round(1000.0 / widen_ms, 3),
        "delta_ms": round(base_ms - widen_ms, 4),
        "delta_tps": round(1000.0 / widen_ms - 1000.0 / base_ms, 3),
    }


def organs() -> dict[str, Any]:
    doc = _load(ORGAN_WIDEN)
    iso = doc["isolated_organs"]
    rows = [
        {
            "organ": r["organ"],
            "gpu_ms": round(r["gpu_ns_median"] / 1e6, 4),
            "dispatches": r.get("dispatches"),
        }
        for r in iso["organs"]
    ]
    kernels = doc["decode"]["add_rmsnorm_628"]["dispatched_kernels_rep0"]
    if F4_KERNEL not in kernels:
        raise ReprofileRefused(
            f"{F4_KERNEL} is not in the dispatched kernels; this run did not "
            "execute widen_f4 whatever the environment said. A strings(1) grep of "
            "the binary is NOT evidence: that literal is the as_str() display form "
            "and fat LTO removes it."
        )
    reps = doc["decode"]["add_rmsnorm_628"]["decode_wall_ns_reps"]
    n_tok = int(doc["max_new_tokens"])
    return {
        "rows": rows,
        "organ_ms_sum": round(sum(r["gpu_ms"] for r in rows), 4),
        "decode_gpu_ms_per_token": round(statistics.median(reps) / n_tok / 1e6, 4),
        "n_reps": len(reps),
        "n_tokens": n_tok,
        "dispatches": doc["counting"]["baseline_628"],
        "widen_f4_kernel_dispatched": True,
        "noop_floor_control": iso["noop_empty"],
        "git_head": doc["git_head"],
    }


def build() -> dict[str, Any]:
    ab = arms()
    org = organs()
    wall = ab["widen_f4_ms_per_token"]
    gpu = org["decode_gpu_ms_per_token"]
    return {
        "schema": "hawking.future.resident_reprofile.v1",
        "version": 1,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": True,
        "timing_label": "DIRTY_ENGINEERING",
        "timing_label_reason": (
            "GPU lane lock held and ModelLake SIGSTOPped for each window, but the "
            "crate source is dirty (DIRTY_SOURCE_SEAL) and other system lanes were "
            "live. Not offered as BASE_TRUE_TPS."
        ),
        # THE NUMBER THE LADDER NEEDS
        "decode_wall_ms_per_token": wall,
        "decode_wall_tps": ab["widen_f4_tps"],
        "decode_gpu_ms_per_token": gpu,
        "host_gap_ms_per_token": round(wall - gpu, 4),
        "supersedes": {
            "stale_ladder_ms": STALE_LADDER_MS,
            "ms_removed": round(STALE_LADDER_MS - wall, 4),
            "why": (
                "PATH_TO_71 and RESIDENT_71TPS_CAUSAL_BUDGET were arithmetic over "
                f"{STALE_LADDER_MS} ms. Every rung must be recomputed from this."
            ),
        },
        "ab": ab,
        "organs": org,
        "protected_window": {
            "modellake_sigstopped": True,
            "gpu_lane_lock_held": True,
            "why": (
                "absolutes move with load while ratios do not; this obligation "
                "asks for an absolute, so the five hf download processes were "
                "stopped for each measurement and resumed after"
            ),
        },
        "binary": {
            "profile": "release",
            "why_not_release_fast": (
                "release-fast is a correctness profile whose own Cargo.toml comment "
                "forbids benchmarking with it: TPS must come from release "
                "(lto=fat, codegen-units=1)"
            ),
        },
    }


def record() -> Path:
    doc = build()
    return write_measured_receipt(
        REPO / "receipts" / "future" / RECEIPT_NAME,
        doc,
        RECORDED_BY,
        provenance=measurement_provenance(lock_held=True, lane="g075-profile"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        print(record())
        return 0
    print(json.dumps(build(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
