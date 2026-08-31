#!/usr/bin/env python3
"""G083: the matched intra-token concurrency probe, and it finds nothing.

S025 §7 asks for one production region with two or more semantically independent
operations, run SERIAL against CONCURRENT at exact parity. §8 adds that separate
command buffers, queues and concurrent encoders are EXPERIMENTS rather than
dogma - Metal is not assumed to find the overlap and is not assumed to fail.

The knob already existed. HAWKING_QWEN38_CONCURRENT wraps independent matvec
pairs in a Metal concurrent compute group (begin_concurrent_group /
end_concurrent_group) instead of encoding them serially. So the probe is a pure
environment A/B on the production path: same bytes, same arithmetic, same
representation, same token.

    serial      27.1824 ms   36.789 TPS   gpu median 26.2452 ms
    concurrent  27.1358 ms   36.852 TPS   gpu median 26.2458 ms
    delta       0.0465 ms, 0.17%          gpu medians differ by 0.6 MICROseconds

Token-identical over 48 decode steps. The verdict is NO_MEASURABLE_EFFECT, not
"concurrent is 0.0465 ms faster": the same harness returned 27.290 ms for an
equivalent configuration earlier in the day, so 0.0465 sits inside the
run-to-run spread and the GPU medians are the same number.

That is exactly what SINGLE_TOKEN_PARALLEL_SLACK predicted. The independent
operations are ALREADY FUSED - gate/up/swiglu in one kernel, q/k/v in one - so a
concurrent group has nothing left to wrap. Two instruments, two directions, one
answer.

    python3 tools/future/intra_token_concurrency_ab.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/intra_token_concurrency_ab.py"
RECEIPT_NAME = "INTRA_TOKEN_CONCURRENCY_AB.json"

SERIAL = REPO / "receipts/future/_G083_serial.json"
CONCURRENT = REPO / "receipts/future/_G083_concurrent.json"

# The spread the same harness shows between equivalent configurations on
# different runs. A delta inside this is not a result.
RUN_TO_RUN_MS = 0.15


class ProbeRefused(RuntimeError):
    """A raw arm is missing, or the arms are not comparable."""


def _arm(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ProbeRefused(f"{path.name} is not on disk; run the A/B first")
    return json.loads(path.read_text())["control_uninstrumented_generate_greedy"]


def compare() -> dict[str, Any]:
    a, b = _arm(SERIAL), _arm(CONCURRENT)
    if a["new_token_ids"] != b["new_token_ids"]:
        raise ProbeRefused(
            "the arms produced different tokens; a timing difference between "
            "different outputs is not a speedup"
        )
    if a["decode_steps"] != b["decode_steps"]:
        raise ProbeRefused("arms differ in decode_steps; not a paired comparison")
    serial_ms = a["decode_wall_ns_per_token"] / 1e6
    conc_ms = b["decode_wall_ns_per_token"] / 1e6
    delta = serial_ms - conc_ms
    gpu_serial = a["median_gpu_ns_including_prefill_steps"] / 1e6
    gpu_conc = b["median_gpu_ns_including_prefill_steps"] / 1e6
    measurable = abs(delta) > RUN_TO_RUN_MS
    return {
        "token_identical": True,
        "decode_steps": int(a["decode_steps"]),
        "serial_ms_per_token": round(serial_ms, 4),
        "concurrent_ms_per_token": round(conc_ms, 4),
        "delta_ms": round(delta, 4),
        "delta_pct": round(100.0 * delta / serial_ms, 3),
        "gpu_median_serial_ms": round(gpu_serial, 4),
        "gpu_median_concurrent_ms": round(gpu_conc, 4),
        "gpu_median_delta_us": round((gpu_serial - gpu_conc) * 1000.0, 2),
        "run_to_run_bar_ms": RUN_TO_RUN_MS,
        "verdict": "CONCURRENT_HELPS" if measurable and delta > 0 else "NO_MEASURABLE_EFFECT",
        "why": (
            f"delta {delta:.4f} ms is inside the {RUN_TO_RUN_MS} ms run-to-run "
            "spread this harness shows between equivalent configurations, and the "
            "GPU medians differ by microseconds. Reporting it as a speedup would "
            "be reading noise."
            if not measurable else
            f"delta {delta:.4f} ms exceeds the {RUN_TO_RUN_MS} ms run-to-run bar"
        ),
    }


def reading() -> dict[str, Any]:
    return {
        "predicted_by": "receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json",
        "prediction": (
            "the token is a chain and its independent operations are ALREADY "
            "FUSED - gate/up/swiglu in one kernel, q/k/v in one - so a concurrent "
            "encoder group has nothing left to wrap"
        ),
        "outcome": "confirmed, from an independent direction",
        "what_it_kills": (
            "the last dispatch-grain hope. If wrapping independent pairs in a "
            "Metal concurrent group changes nothing, the missing capacity G009 "
            "measured is not reachable by scheduling this token differently."
        ),
        "what_it_does_not_kill": [
            "occupancy: the kernels may still launch too little parallel work",
            "memory-level parallelism: threads may hold too few loads in flight",
            "the instruction dependency chain inside the decode loop",
            "separate command QUEUES, which this probe did not vary - it varied "
            "the encoder topology within one queue",
        ],
        "evidence_class_note": (
            "this is a paired RATIO under a contaminated window (load 8.8), which "
            "S025 §42 explicitly permits: absolutes move with load, ratios do not. "
            "It is not offered as a protected absolute and the ms figures are not "
            "promotable."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.intra_token_concurrency_ab.v1",
        "version": 1,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": True,
        "knob": "HAWKING_QWEN38_CONCURRENT",
        "mechanism": (
            "begin_concurrent_group / end_concurrent_group around independent "
            "matvec pairs, instead of a serial encoder"
        ),
        "ab": compare(),
        "reading": reading(),
        "claim_boundary": (
            "One paired A/B on the sealed production path, same bytes, same "
            "arithmetic, same representation, token-identical over 48 decode "
            "steps. Contaminated window, so the RATIO is the result and the "
            "absolute ms are not promotable. It varies encoder topology within "
            "one command queue and says nothing about multiple queues."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=False, lane="g083-concurrency-ab"),
        ))
        return 0
    print(json.dumps(doc["ab"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
