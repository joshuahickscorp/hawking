#!/usr/bin/env python3.12
"""Matched-benchmark harness for the GLM-5.2 inference campaign (mandate section 3).

THE MATCHED BENCHMARK LAW: every speed claim in this campaign must be produced by this
module.  The previous campaign published a 35.9x speedup that rested on a dense fp16
baseline of 9.012 ms; the real dense fp16 MPS figure is 0.3674 ms, so the baseline was
24x wrong and the headline number was fiction.  The matched single-call truth is that the
custom kernel is currently SLOWER than dense (0.727x at down, 0.329x at gate/up).

This module makes that failure structurally impossible rather than merely discouraged:

  * a comparison is described by a BenchSpec, and two BenchSpecs that are not
    field-identical are not comparable.  speedup() raises on unmatched specs; it does
    not warn, and there is no override flag.
  * timings keep every RAW SAMPLE and report min/median/p95/max/count/stddev.  A bare
    mean is never reported, because on a machine with a live campaign on it the tail is
    contention, not hardware -- is_contended surfaces exactly that.
  * component accounting is explicit and sparse.  A backend that cannot measure
    host_encode reports UNMEASURED, never 0.0, because a zero would silently flatter it.
  * baselines are a closed set of names, so later phases register implementations
    against the same seven names instead of inventing a favourable new one.
  * the refuted claims of the previous campaign are seeded by name and by value, so the
    9.012 ms baseline, the 35.9x ratio and the mislabelled 1.4e-6 Metal parity figure
    cannot be revived by any caller.

Roofline fractions are billed against the roofs measured on THIS machine (736 GB/s,
17703 GFLOP/s), not against vendor figures; the 819 GB/s vendor number is not this
machine's roof.
"""
from __future__ import annotations

import hashlib
import math
import platform
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from glm52_common import Glm52Error, atomic_json, canonical, utc_now


SCHEMA = "hawking.glm52.matched_benchmark.v1"

# Measured on this machine (Mac15,14 / M3 Ultra / 60 GPU cores).  These are the roofs
# every fraction in this module is billed against.  The 819 GB/s vendor figure is NOT
# this machine's roof and must not be substituted.
MACHINE_FACTS: dict[str, Any] = {
    "model_identifier": "Mac15,14",
    "chip": "Apple M3 Ultra",
    "gpu_cores": 60,
    "cpu_cores": 28,
    "unified_memory_gib": 96,
    "metal_version": 4,
    "max_threadgroup_memory_bytes": 32768,
    "thread_execution_width": 32,
    "bandwidth_roof_gb_s": 736.0,          # sustained read, best median (759.8 max observed)
    "compute_roof_gflop_s": 17703.0,       # fp32 FMA
    "command_buffer_fixed_cost_us": 215.8,
    "marginal_dispatch_us": 0.71,
    "ridge_flop_per_byte": 24.05,
}

BANDWIDTH_ROOF_GB_S: float = MACHINE_FACTS["bandwidth_roof_gb_s"]
COMPUTE_ROOF_GFLOP_S: float = MACHINE_FACTS["compute_roof_gflop_s"]

# Closed set.  Later phases register implementations against these names; a new name is
# a mandate change, not a benchmark detail.
BASELINES: tuple[str, ...] = (
    "dense_fp16_mps",
    "torch_mps_compact",
    "cpu_authority",
    "custom_v2",
    "track_a",
    "track_b",
    "selected_hybrid",
)

# Component fields the diagnostic separated.  Missing components are None and serialise
# as UNMEASURED; they are never coerced to zero.
COMPONENTS: tuple[str, ...] = (
    "gpu_execution",
    "host_encode",
    "command_buffer",
    "end_to_end",
    "cold_start",
    "warm_steady_state",
)

# Roofline is billed against the first component present in this order, so a backend
# that measured pure GPU time is not credited with host encode it did not pay for.
PRIMARY_TIMING_ORDER: tuple[str, ...] = ("gpu_execution", "warm_steady_state", "end_to_end")

UNMEASURED = "UNMEASURED"

# Coefficient of variation above which the samples are treated as contended rather than
# as a hardware reading.  The diagnostic used 15%.  The denominator is the MEDIAN, not
# the mean: this harness never reports a mean, and the median is the robust centre.
CONTENTION_CV_THRESHOLD: float = 0.15


class MatchedBenchmarkError(Glm52Error):
    """Raised when the matched benchmark law is violated."""


@dataclass(frozen=True)
class RefutedClaim:
    """A number from the previous campaign that was disproved and may not be republished."""

    name: str
    kind: str            # "milliseconds" | "ratio" | "parity"
    value: float
    reason: str


REFUTED_CLAIMS: tuple[RefutedClaim, ...] = (
    RefutedClaim(
        "dense_fp16_9.012ms", "milliseconds", 9.012,
        "dense fp16 MPS was re-measured at 0.3674 ms; 9.012 ms is 24x wrong and is the "
        "baseline the retracted 35.9x speedup was divided by",
    ),
    RefutedClaim(
        "speedup_35.9x", "ratio", 35.9,
        "computed against the refuted 9.012 ms dense baseline; the matched single-call "
        "result is 0.727x at down and 0.329x at gate/up (custom is SLOWER)",
    ),
    RefutedClaim(
        "metal_parity_1.4e-6", "parity", 1.4e-6,
        "mislabelled: 1.4e-6 belongs to the torch fp32 path. Custom Metal parity is "
        "2.1e-4, caused by gravity_metal.py:202 casting the codebook to fp16",
    ),
)

_REFUTED_REL_TOL = 1e-3


@dataclass(frozen=True)
class BenchSpec:
    """The full description of a matched comparison.

    Every field participates in matching.  Two specs are comparable only when they are
    field-identical, so a comparison cannot drift in geometry, dtype, timed region or
    synchronisation policy between the baseline and the candidate.
    """

    rows: int
    cols: int
    batch: int
    input_seed: int
    input_dtype: str
    output_dtype: str
    warmup: int
    reps: int
    sync_boundary: str          # e.g. "per_call_host_sync", "per_batch_gpu_fence", "none"
    dependency_shape: str       # e.g. "independent_calls", "serial_dependent_chain"
    pack_in_timed_region: bool
    unpack_in_timed_region: bool

    def __post_init__(self) -> None:
        if min(self.rows, self.cols, self.batch) < 1:
            raise MatchedBenchmarkError("rows, cols and batch must all be >= 1")
        if self.reps < 2:
            raise MatchedBenchmarkError("reps must be >= 2; stddev of one sample is meaningless")
        if self.warmup < 0:
            raise MatchedBenchmarkError("warmup must be >= 0")

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "batch": self.batch,
            "input_seed": self.input_seed,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "warmup": self.warmup,
            "reps": self.reps,
            "sync_boundary": self.sync_boundary,
            "dependency_shape": self.dependency_shape,
            "pack_in_timed_region": self.pack_in_timed_region,
            "unpack_in_timed_region": self.unpack_in_timed_region,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BenchSpec":
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.to_json())).hexdigest()

    def make_input(self) -> np.ndarray:
        """The fixed-seed input vector.  Both sides of a matched comparison get this."""
        rng = np.random.default_rng(self.input_seed)
        x = rng.standard_normal((self.cols, self.batch))
        return np.ascontiguousarray(x, dtype=np.dtype(self.input_dtype))


def mismatched_fields(a: BenchSpec, b: BenchSpec) -> tuple[str, ...]:
    left, right = a.to_json(), b.to_json()
    return tuple(sorted(k for k in left if left[k] != right[k]))


def matched(a: BenchSpec, b: BenchSpec) -> bool:
    """True only when the two specs are field-identical."""
    return not mismatched_fields(a, b)


def require_matched(a: BenchSpec, b: BenchSpec, *, left: str = "baseline",
                    right: str = "candidate") -> None:
    diff = mismatched_fields(a, b)
    if diff:
        raise MatchedBenchmarkError(
            f"unmatched BenchSpecs ({left} vs {right}): fields differ {diff}. "
            "The matched benchmark law forbids a speedup across differing specs."
        )


@dataclass(frozen=True)
class TimingStats:
    """Raw samples plus the order statistics.  No mean is exposed, by design."""

    raw_samples_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.raw_samples_ms) < 2:
            raise MatchedBenchmarkError("need >= 2 raw samples to report a distribution")
        if any(s < 0 or not math.isfinite(s) for s in self.raw_samples_ms):
            raise MatchedBenchmarkError("raw samples must be finite and non-negative")

    @property
    def count(self) -> int:
        return len(self.raw_samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.raw_samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.raw_samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.raw_samples_ms)

    @property
    def p95_ms(self) -> float:
        """Nearest-rank p95: the smallest sample at or above the 95th percentile rank."""
        ordered = sorted(self.raw_samples_ms)
        rank = math.ceil(0.95 * len(ordered))
        return ordered[min(len(ordered), max(1, rank)) - 1]

    @property
    def stddev_ms(self) -> float:
        return statistics.stdev(self.raw_samples_ms)

    @property
    def coefficient_of_variation(self) -> float:
        """stddev / median.  Median denominator, deliberately: no mean is reported here."""
        median = self.median_ms
        return float("inf") if median == 0 else self.stddev_ms / median

    @property
    def is_contended(self) -> bool:
        """True when spread exceeds the threshold, i.e. p95/max read as machine load."""
        return self.coefficient_of_variation > CONTENTION_CV_THRESHOLD

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_samples_ms": list(self.raw_samples_ms),
            "count": self.count,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "stddev_ms": self.stddev_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
            "is_contended": self.is_contended,
            "contention_cv_threshold": CONTENTION_CV_THRESHOLD,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TimingStats":
        return cls(tuple(value["raw_samples_ms"]))


@dataclass(frozen=True)
class ComponentTimings:
    """The six components the diagnostic separated.  None means UNMEASURED, never zero."""

    gpu_execution: TimingStats | None = None
    host_encode: TimingStats | None = None
    command_buffer: TimingStats | None = None
    end_to_end: TimingStats | None = None
    cold_start: TimingStats | None = None
    warm_steady_state: TimingStats | None = None

    def __post_init__(self) -> None:
        if all(getattr(self, name) is None for name in COMPONENTS):
            raise MatchedBenchmarkError("at least one component must be measured")

    def primary(self) -> tuple[str, TimingStats]:
        for name in PRIMARY_TIMING_ORDER:
            stats = getattr(self, name)
            if stats is not None:
                return name, stats
        raise MatchedBenchmarkError(
            f"no roofline-billable component measured; need one of {PRIMARY_TIMING_ORDER}"
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in COMPONENTS:
            stats = getattr(self, name)
            out[name] = UNMEASURED if stats is None else stats.to_json()
        return out

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ComponentTimings":
        return cls(**{
            name: None if value[name] == UNMEASURED else TimingStats.from_json(value[name])
            for name in COMPONENTS
        })


@dataclass(frozen=True)
class BenchResult:
    """One implementation measured against one spec.

    `reproduced` is the guard the previous campaign lacked: a result carried forward from
    a prior report without being re-measured here is not reproduced, and no speedup may
    be computed against it.
    """

    baseline: str
    spec: BenchSpec
    timings: ComponentTimings
    reproduced: bool = True
    bytes_moved: int | None = None
    flops: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.baseline not in BASELINES:
            raise MatchedBenchmarkError(
                f"unknown baseline {self.baseline!r}; register against one of {BASELINES}"
            )

    def roofline(self) -> dict[str, Any]:
        source, stats = self.timings.primary()
        seconds = stats.median_ms / 1e3
        out: dict[str, Any] = {
            "timing_source": source,
            "seconds_used": seconds,
            "bytes_moved": self.bytes_moved if self.bytes_moved is not None else UNMEASURED,
            "flops": self.flops if self.flops is not None else UNMEASURED,
        }
        if self.bytes_moved is None or seconds <= 0:
            out["achieved_gb_s"] = UNMEASURED
            out["fraction_of_bandwidth_roof"] = UNMEASURED
        else:
            gb_s = self.bytes_moved / seconds / 1e9
            out["achieved_gb_s"] = gb_s
            out["fraction_of_bandwidth_roof"] = gb_s / BANDWIDTH_ROOF_GB_S
        if self.flops is None or seconds <= 0:
            out["achieved_gflop_s"] = UNMEASURED
            out["fraction_of_compute_roof"] = UNMEASURED
        else:
            gflop_s = self.flops / seconds / 1e9
            out["achieved_gflop_s"] = gflop_s
            out["fraction_of_compute_roof"] = gflop_s / COMPUTE_ROOF_GFLOP_S
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "reproduced": self.reproduced,
            "spec": self.spec.to_json(),
            "spec_fingerprint": self.spec.fingerprint,
            "timings": self.timings.to_json(),
            "roofline": self.roofline(),
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BenchResult":
        roof = value["roofline"]
        return cls(
            baseline=value["baseline"],
            spec=BenchSpec.from_json(value["spec"]),
            timings=ComponentTimings.from_json(value["timings"]),
            reproduced=value["reproduced"],
            bytes_moved=None if roof["bytes_moved"] == UNMEASURED else roof["bytes_moved"],
            flops=None if roof["flops"] == UNMEASURED else roof["flops"],
            notes=value["notes"],
        )


def assert_not_refuted(*, name: str | None = None, kind: str | None = None,
                       value: float | None = None) -> None:
    """Reject a refuted claim, by name and by numeric value.

    The three seeded claims are the retracted campaign's headline numbers.  They are
    rejected here so that no later phase can reintroduce them, whether by quoting the
    name or by arriving at the number again through an unmatched comparison.
    """
    for claim in REFUTED_CLAIMS:
        if name is not None and name == claim.name:
            raise MatchedBenchmarkError(f"REFUTED claim {claim.name}: {claim.reason}")
        if value is None or kind is None or kind != claim.kind:
            continue
        if math.isclose(value, claim.value, rel_tol=_REFUTED_REL_TOL):
            raise MatchedBenchmarkError(
                f"REFUTED claim {claim.name}: value {value!r} matches the disproved "
                f"{claim.value} ({claim.kind}). {claim.reason}"
            )


def measure(fn: Callable[[], Any], spec: BenchSpec) -> TimingStats:
    """Run fn for spec.warmup untimed reps then spec.reps timed reps, keeping every sample.

    CPU-only wall clock.  A GPU backend supplies its own component stats instead; this
    primitive exists so the sample-keeping and statistics are identical either way.
    """
    for _ in range(spec.warmup):
        fn()
    samples: list[float] = []
    for _ in range(spec.reps):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return TimingStats(tuple(samples))


def speedup(baseline: BenchResult, candidate: BenchResult) -> dict[str, Any]:
    """The only sanctioned way to state a speedup.

    Refuses unmatched specs, unreproduced baselines, and any refuted number on either
    side or in the resulting ratio.  Raises; never warns.
    """
    require_matched(baseline.spec, candidate.spec,
                    left=f"baseline:{baseline.baseline}", right=f"candidate:{candidate.baseline}")
    if not baseline.reproduced:
        raise MatchedBenchmarkError(
            f"baseline {baseline.baseline!r} is flagged unreproduced; no speedup may be "
            "emitted against a baseline this harness did not measure"
        )
    if not candidate.reproduced:
        raise MatchedBenchmarkError(
            f"candidate {candidate.baseline!r} is flagged unreproduced"
        )
    base_source, base_stats = baseline.timings.primary()
    cand_source, cand_stats = candidate.timings.primary()
    if base_source != cand_source:
        raise MatchedBenchmarkError(
            f"component mismatch: baseline billed {base_source}, candidate billed "
            f"{cand_source}. A speedup across different timed regions is unmatched."
        )
    assert_not_refuted(kind="milliseconds", value=base_stats.median_ms)
    assert_not_refuted(kind="milliseconds", value=cand_stats.median_ms)
    if cand_stats.median_ms <= 0:
        raise MatchedBenchmarkError("candidate median is zero; cannot form a ratio")
    ratio = base_stats.median_ms / cand_stats.median_ms
    assert_not_refuted(kind="ratio", value=ratio)
    return {
        "baseline": baseline.baseline,
        "candidate": candidate.baseline,
        "specs_matched": True,
        "spec_fingerprint": baseline.spec.fingerprint,
        "timing_source": base_source,
        "statistic": "median_ms",
        "baseline_median_ms": base_stats.median_ms,
        "candidate_median_ms": cand_stats.median_ms,
        "speedup": ratio,
        "slower_than_baseline": ratio < 1.0,
        "either_side_contended": base_stats.is_contended or cand_stats.is_contended,
    }


def build_report(results: Sequence[BenchResult], speedups: Sequence[dict[str, Any]], *,
                 label: str, notes: str = "") -> dict[str, Any]:
    for entry in speedups:
        if not entry.get("specs_matched"):
            raise MatchedBenchmarkError("a speedup entry does not assert specs_matched")
    return {
        "schema": SCHEMA,
        "label": label,
        "generated_at": utc_now(),
        "notes": notes,
        "machine": dict(MACHINE_FACTS, observed_platform=platform.platform(),
                        observed_machine=platform.machine()),
        "baselines": list(BASELINES),
        "components": list(COMPONENTS),
        "contention_cv_threshold": CONTENTION_CV_THRESHOLD,
        "refuted_claims": [
            {"name": c.name, "kind": c.kind, "value": c.value, "reason": c.reason}
            for c in REFUTED_CLAIMS
        ],
        "results": [r.to_json() for r in results],
        "matched": list(speedups),
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    if report.get("schema") != SCHEMA:
        raise MatchedBenchmarkError(f"refusing to write a report with schema {report.get('schema')!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return path


def selftest(*, rows: int = 256, cols: int = 512) -> dict[str, Any]:
    """Exercise the harness end to end on CPU only: dense numpy matvec vs gravity_forge PQ.

    No GPU work.  This proves the harness measures, matches, refuses and serialises; it
    is not a claim about either implementation's production speed.
    """
    import gravity_forge as gf

    spec = BenchSpec(
        rows=rows, cols=cols, batch=1, input_seed=20260722,
        input_dtype="float32", output_dtype="float32",
        warmup=3, reps=15,
        sync_boundary="none_cpu_wall_clock",
        dependency_shape="independent_calls",
        pack_in_timed_region=False, unpack_in_timed_region=True,
    )
    rng = np.random.default_rng(spec.input_seed)
    w = (rng.standard_normal((rows, 8)).astype(np.float32)
         @ rng.standard_normal((8, cols)).astype(np.float32)) * 0.1
    x = spec.make_input()[:, 0]

    # R0 geometry, reduced size: D=8, k=128, subspaces=1, rotate=False.
    art = gf.pack_product_quant(w, dim=8, subspaces=1, k=128, seed=0, iters=4)
    codes = art.config["pq_codes"]
    nchunk = codes["nchunk"]

    dense = BenchResult(
        baseline="cpu_authority",
        spec=spec,
        timings=ComponentTimings(end_to_end=measure(lambda: w @ x, spec)),
        bytes_moved=rows * cols * 4 + cols * 4 + rows * 4,
        flops=2 * rows * cols,
        notes="numpy float32 dense matvec, CPU",
    )
    pq = BenchResult(
        baseline="custom_v2",
        spec=spec,
        timings=ComponentTimings(end_to_end=measure(lambda: gf.pq_execute(art, x), spec)),
        # indices are 1 byte/entry in memory here; the on-disk R0 bill is 7-bit.
        bytes_moved=rows * nchunk * codes["S"] + 128 * codes["sub"] * 4 + cols * 4 + rows * 4,
        flops=2 * rows * cols,
        notes="gravity_forge.pq_execute, CPU, D=8 k=128 subspaces=1 rotate=False (R0 geometry, reduced size)",
    )
    report = build_report(
        [dense, pq],
        [speedup(dense, pq)],
        label="gravity_bench_lab selftest (CPU only, no GPU work)",
        notes="Harness proof only. Both sides are CPU; neither figure is a production claim. "
              "packed_bpw and on-disk 7-bit index billing are not exercised here.",
    )
    round_trip = [BenchResult.from_json(r).to_json() for r in report["results"]]
    report["selftest"] = {
        "json_round_trip_stable": round_trip == report["results"],
        "unmatched_comparison_refused": _refuses_unmatched(spec, dense),
        "refuted_guard_live": _refuses_refuted(),
    }
    return report


def _refuses_unmatched(spec: BenchSpec, result: BenchResult) -> bool:
    other = replace(spec, batch=spec.batch + 1)
    try:
        speedup(result, replace(result, spec=other, baseline="track_a"))
    except MatchedBenchmarkError:
        return True
    return False


def _refuses_refuted() -> bool:
    ok = 0
    for claim in REFUTED_CLAIMS:
        try:
            assert_not_refuted(name=claim.name)
        except MatchedBenchmarkError:
            ok += 1
    return ok == len(REFUTED_CLAIMS)


def main() -> None:
    report = selftest()
    out = Path(__file__).resolve().parents[2] / "reports" / "condense" / "breakthrough" / \
        "GLM52_BENCH_HARNESS_SELFTEST.json"
    write_report(out, report)
    print(f"wrote {out}")
    for entry in report["matched"]:
        print(f"{entry['candidate']} vs {entry['baseline']}: {entry['speedup']:.4f}x "
              f"(contended={entry['either_side_contended']})")


if __name__ == "__main__":
    main()
