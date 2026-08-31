"""EXPERIMENT_TURNAROUND — development-loop latency, measured like a token budget.

Token latency is already a first-class scoreboard field. Experiment turnaround
is the same shape of question (which repeated phase dominates, with evidence)
and is currently all-null on the Accelerator scoreboard. This sidecar fills the
CPU-side phases it can honestly time and leaves every build/GPU phase UNKNOWN.

    python3 tools/future/turnaround.py --measure
    python3 tools/future/turnaround.py --build
    python3 tools/future/turnaround.py --selftest

Does not run `cargo build`, does not touch the GPU, and does not write
scoreboard nanosecond fields from CPU timings.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "EXPERIMENT_TURNAROUND.json"
SCHEMA = "hawking.future.turnaround.v1"
DEFAULT_REPEATS = 5
MIN_REPEATS = 2

# Contract names, snake_cased. Same 11 slots the lane is required to report.
PHASES: tuple[str, ...] = (
    "source_discovery",
    "transform",
    "compile",
    "link",
    "shader_compile",
    "launch",
    "execution",
    "verify",
    "receipt",
    "ledger",
    "next_decision",
)

# These require cargo and/or a GPU. No CPU proxy is a measurement of them.
GPU_OR_BUILD_PHASES: frozenset[str] = frozenset(
    {"compile", "link", "shader_compile", "launch", "execution"}
)

# Exact keys of receipts/headless/ACCELERATOR_SCOREBOARD.json -> development_phases.
# Copied from tools/accelerator/scoreboard.py (Codex-owned, read-only).
SCOREBOARD_DEVELOPMENT_PHASES: tuple[str, ...] = (
    "transform_ns",
    "compile_ns",
    "load_ns",
    "benchmark_ns",
    "verification_ns",
    "receipt_ns",
    "total_experiment_turnaround_ns",
)

PHASE_TO_SCOREBOARD: dict[str, str] = {
    "transform": "transform_ns",
    "compile": "compile_ns",
    "launch": "load_ns",
    "execution": "benchmark_ns",
    "verify": "verification_ns",
    "receipt": "receipt_ns",
}

GPU_REASONS: dict[str, str] = {
    "compile": (
        "cargo_build_forbidden: would contend with the live campaign for CPU "
        "and the shared target-dir cache"
    ),
    "link": (
        "cargo_build_forbidden: link is part of cargo build; same contention"
    ),
    "shader_compile": (
        "requires Metal (newLibraryWithSource or xcrun metallib); sidecar has "
        "no GPU. Existing cache: crates/hawking-core/src/metal/mod.rs "
        "load_or_compile_shader_library"
    ),
    "launch": "requires the experiment binary and a GPU device",
    "execution": (
        "requires PROTECTED_ABSOLUTE GPU authority; this campaign is STATIC_ONLY"
    ),
}

CPU_MEASUREMENT_KIND: dict[str, str] = {
    "source_discovery": "git_ls_tree_full_inventory",
    "transform": "python_import_tools.future._common",
    "verify": "pytest_collect_only_tools.future",
    "receipt": "serialize_seal_fsync_tempfile",
    "ledger": "jsonl_append_fsync_tempfile",
    "next_decision": "dominant_cost_and_lever_selection",
}


class GpuDependentPhaseError(ValueError):
    """Raised when a caller tries to record a build/GPU phase as a CPU measurement."""


class HardwareLaunderingError(ValueError):
    """Raised when a caller tries to pour CPU timings into scoreboard ns fields."""


def record_cpu_side(phase: str, value_ms: float) -> None:
    """Guard used by the measure loop and by the negative-control test.

    GPU/build phases must raise. A guard nobody has watched fail is not a guard.
    """
    if phase in GPU_OR_BUILD_PHASES:
        raise GpuDependentPhaseError(
            f"{phase} requires a build or a GPU; refusing CPU-side proxy "
            f"{value_ms!r} ms"
        )
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    if not isinstance(value_ms, (int, float)) or isinstance(value_ms, bool):
        raise TypeError(f"cpu-side sample for {phase} must be numeric, got {value_ms!r}")


def record_scoreboard_ns(name: str, value: float) -> None:
    """Refuse to fill Accelerator scoreboard nanosecond fields from this sidecar.

    `experiment_turnaround_ns` / `*_ns` on the scoreboard are the full experiment
    loop, including compile and GPU. A CPU tempfile write is not that quantity.
    """
    raise HardwareLaunderingError(
        f"{name}={value!r}: sidecar must not fill scoreboard ns fields from "
        f"CPU-side timings (that is hardware-claim laundering)"
    )


def null_scoreboard_development_phases() -> dict[str, None]:
    return {name: None for name in SCOREBOARD_DEVELOPMENT_PHASES}


def _run(args: list[str], *, timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged,
        check=False,
    )


def scan_sources() -> dict[str, Any]:
    """Full-tree source inventory via git. Sparse checkout must not undercount."""
    raw = git("ls-tree", "-r", "--name-only", "HEAD")
    paths = [ln for ln in raw.splitlines() if ln]
    by_suffix: dict[str, int] = {}
    hawking_core_rs = 0
    hawking_core_metal = 0
    for rel in paths:
        suffix = Path(rel).suffix.lower() or "<none>"
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        if rel.startswith("crates/hawking-core/") and rel.endswith(".rs"):
            hawking_core_rs += 1
        if rel.startswith("crates/hawking-core/") and rel.endswith(".metal"):
            hawking_core_metal += 1
    ranked = dict(sorted(by_suffix.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "scan": "git ls-tree -r --name-only HEAD",
        "file_count": len(paths),
        "by_suffix": ranked,
        "rust_rs": by_suffix.get(".rs", 0),
        "metal": by_suffix.get(".metal", 0),
        "python": by_suffix.get(".py", 0),
        "toml": by_suffix.get(".toml", 0),
        "hawking_core_rs": hawking_core_rs,
        "hawking_core_metal": hawking_core_metal,
        "sparse_checkout": True,
        "note": (
            "git ls-tree sees the full committed tree; os.walk of this "
            "worktree would undercount because of sparse checkout"
        ),
    }


def _time_source_discovery() -> dict[str, Any]:
    return scan_sources()


def _time_transform() -> None:
    env = {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = _run(
        [sys.executable, "-c", "import tools.future._common"],
        timeout=30,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"python import failed: {proc.stderr[-400:]}")


def _time_verify() -> None:
    env = {
        "PYTHONPATH": str(REPO),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tools/future",
            "--collect-only",
            "-q",
        ],
        timeout=60,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pytest --collect-only failed: {proc.stderr[-400:]}")


def _time_receipt() -> None:
    doc = {"schema": "hawking.future.turnaround.probe.v1", "probe": True, "n": 0}
    body = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    doc["seal_sha256"] = hashlib.sha256(body).hexdigest()
    blob = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    fd, path = tempfile.mkstemp(prefix="hawking-turnaround-receipt-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        os.unlink(path)


def _time_ledger() -> None:
    record = {"obligation": "turnaround-probe", "state": "UNVERIFIED", "seq": 0}
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    fd, path = tempfile.mkstemp(prefix="hawking-turnaround-ledger-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "a") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        os.unlink(path)


def _summarize(samples: list[float]) -> dict[str, Any]:
    xs = sorted(samples)
    med = statistics.median(xs)
    min_ms = round(xs[0], 3)
    max_ms = round(xs[-1], 3)
    out: dict[str, Any] = {
        "n": len(xs),
        "median_ms": round(med, 3),
        "min_ms": min_ms,
        "max_ms": max_ms,
        "range_ms": round(max_ms - min_ms, 3),
        "samples_ms": [round(s, 3) for s in samples],
    }
    if len(xs) >= 4:
        q = statistics.quantiles(xs, n=4, method="inclusive")
        q1 = round(q[0], 3)
        q3 = round(q[2], 3)
        out["q1_ms"] = q1
        out["q3_ms"] = q3
        out["iqr_ms"] = round(q3 - q1, 3)
    return out


def read_cargo_evidence() -> dict[str, Any]:
    cargo_path = REPO / "Cargo.toml"
    cargo = tomllib.loads(cargo_path.read_text())
    profiles = cargo.get("profile", {})
    release = profiles.get("release", {})
    release_fast = profiles.get("release-fast", {})
    core_path = REPO / "crates" / "hawking-core" / "Cargo.toml"
    core = tomllib.loads(core_path.read_text()) if core_path.is_file() else {}
    cargo_config = REPO / ".cargo" / "config.toml"
    return {
        "workspace_members": list(cargo.get("workspace", {}).get("members", [])),
        "default_members": list(cargo.get("workspace", {}).get("default-members", [])),
        "profile_release": {
            "lto": release.get("lto"),
            "codegen_units": release.get("codegen-units"),
            "opt_level": release.get("opt-level"),
            "incremental": release.get("incremental"),  # None => cargo default (off for release)
        },
        "profile_release_fast": {
            "inherits": release_fast.get("inherits"),
            "lto": release_fast.get("lto"),
            "codegen_units": release_fast.get("codegen-units"),
            "incremental": release_fast.get("incremental"),
        },
        "hawking_core_crate_type": list(core.get("lib", {}).get("crate-type", [])),
        "cargo_config_in_this_worktree": cargo_config.is_file(),
        "shared_target_dir": "workspace/ops/build/rust",
        "shared_target_dir_source": (
            "parent-disk .cargo/config.toml (gitignored) plus context-pack "
            "build output path; not present in this sparse worktree"
        ),
    }


def _scoreboard_on_disk() -> dict[str, Any]:
    path = REPO / "receipts" / "headless" / "ACCELERATOR_SCOREBOARD.json"
    if not path.is_file():
        return {
            "present_in_this_worktree": False,
            "reason": "sparse checkout; file is also untracked in git HEAD",
        }
    body = load_json(path)
    rows = body.get("rows") or []
    sample = rows[0].get("development_phases") if rows else None
    return {
        "present_in_this_worktree": True,
        "schema": body.get("schema"),
        "development_phases_keys": list((sample or {}).keys()),
        "sample_development_phases": sample,
    }


def identify_dominant(
    measured: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
    cargo: dict[str, Any],
) -> dict[str, Any]:
    cpu = {
        name: rec
        for name, rec in measured.items()
        if rec.get("state") == "MEASURED_CPU_SIDE"
        and isinstance(rec.get("median_ms"), (int, float))
    }
    winner = max(cpu, key=lambda name: float(cpu[name]["median_ms"])) if cpu else None
    release = cargo.get("profile_release") or {}
    release_fast = cargo.get("profile_release_fast") or {}
    return {
        "among_measured_cpu_side": {
            "phase": winner,
            "median_ms": None if winner is None else cpu[winner]["median_ms"],
            "evidence": (
                "largest median_ms among MEASURED_CPU_SIDE phases; process "
                "timings, not a GPU measurement"
            ),
        },
        "full_experiment_loop": {
            "hypothesized_dominant_phase": "compile",
            "state": "UNKNOWN",
            "not_a_measurement": True,
            "why_not_measured": GPU_REASONS["compile"],
            "static_evidence": [
                (
                    f"inventory: {inventory.get('rust_rs')} .rs files, "
                    f"{inventory.get('hawking_core_rs')} in crates/hawking-core, "
                    f"{inventory.get('metal')} .metal files"
                ),
                (
                    "profile.release: "
                    f"lto={release.get('lto')!r} "
                    f"codegen-units={release.get('codegen_units')!r} "
                    f"incremental={release.get('incremental')!r} "
                    "(None means cargo's release default: incremental off)"
                ),
                (
                    "profile.release-fast already exists for iteration "
                    f"(incremental={release_fast.get('incremental')!r}, "
                    f"lto={release_fast.get('lto')!r}, "
                    f"codegen-units={release_fast.get('codegen_units')!r}) "
                    "and Cargo.toml forbids using it for TPS"
                ),
                (
                    "shared target-dir workspace/ops/build/rust is why this "
                    "lane must not cargo build: it contends with the live campaign"
                ),
                (
                    "scoreboard development_phases are all null — compile has "
                    "never been budgeted as an experiment cost"
                ),
                (
                    "default-members already drop hide-* from a bare cargo "
                    "build; that lever is pulled"
                ),
            ],
        },
        "lever": {
            "id": "target_isolation_plus_input_fingerprint",
            "title": (
                "Per-experiment CARGO_TARGET_DIR plus a content-addressed skip fingerprint"
            ),
            "what": (
                "Give each experiment lane its own CARGO_TARGET_DIR instead of the "
                "shared workspace/ops/build/rust, and skip cargo only when a "
                "fingerprint of (git tree of crates/**, Cargo.lock, rustc -vV, "
                "profile name, RUSTFLAGS) matches the artifact already in that dir."
            ),
            "why_this_and_not_the_others": (
                "release-fast already exists for non-protected iteration; the "
                "metallib cache already exists in crates/hawking-core/src/metal/mod.rs "
                "(keyed by device + shader sha256 + math mode); default-members "
                "already drop hide-* from a bare cargo build. The remaining "
                "repeated cost is cache contention on the shared target-dir plus "
                "fat-LTO rebuilds when that cache is invalidated."
            ),
            "does_not_weaken_reproducibility": (
                "target-dir is not a rustc codegen input; two directories compiling "
                "the same sources/profile/flags emit equivalent artifacts. The skip "
                "fingerprint includes every input that can change the binary, so a "
                "hit is a replay of an already-built artifact, not a skipped compile "
                "of dirty inputs. PROTECTED_ABSOLUTE measurements keep using "
                "[profile.release] (lto=fat); this lever does not substitute "
                "release-fast numbers for protected ones. Artifact identity remains "
                "the sealed binary hash already used by HCLI."
            ),
            "how_we_know": (
                "Cargo treats target-dir as an output location. The scoreboard "
                "already keys executables by sha256 (executable_id). Cargo.toml "
                "already splits release vs release-fast and comments that TPS "
                "must come from release."
            ),
        },
    }


def _time_next_decision(
    measured: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
    cargo: dict[str, Any],
) -> dict[str, Any]:
    return identify_dominant(measured, inventory, cargo)


CPU_TIMERS: dict[str, Callable[..., Any]] = {
    "source_discovery": _time_source_discovery,
    "transform": _time_transform,
    "verify": _time_verify,
    "receipt": _time_receipt,
    "ledger": _time_ledger,
}


def _unknown_record(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": "UNKNOWN",
        "reason": GPU_REASONS[name],
        "measurement_kind": None,
        "phase_wall_ms_cpu_side": None,
        "n": 0,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "range_ms": None,
        "samples_ms": None,
        "scoreboard_field": PHASE_TO_SCOREBOARD.get(name),
    }


def _measured_record(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "name": name,
        "state": "MEASURED_CPU_SIDE",
        "reason": None,
        "measurement_kind": CPU_MEASUREMENT_KIND[name],
        "phase_wall_ms_cpu_side": summary["median_ms"],
        "scoreboard_field": PHASE_TO_SCOREBOARD.get(name),
    }
    rec.update(summary)
    return rec


def measure(repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    if repeats < MIN_REPEATS:
        raise ValueError(
            f"repeats must be >= {MIN_REPEATS} (median with spread, never a single sample)"
        )
    if not CPU_TIMERS.keys().isdisjoint(GPU_OR_BUILD_PHASES):
        raise GpuDependentPhaseError(
            "CPU_TIMERS contains a GPU/build phase; that is a CPU proxy"
        )

    cargo = read_cargo_evidence()
    samples: dict[str, list[float]] = {name: [] for name in CPU_TIMERS}
    inventory: dict[str, Any] | None = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        inventory = _time_source_discovery()
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("source_discovery", dt)
        samples["source_discovery"].append(dt)

        t0 = time.perf_counter()
        _time_transform()
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("transform", dt)
        samples["transform"].append(dt)

        t0 = time.perf_counter()
        _time_verify()
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("verify", dt)
        samples["verify"].append(dt)

        t0 = time.perf_counter()
        _time_receipt()
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("receipt", dt)
        samples["receipt"].append(dt)

        t0 = time.perf_counter()
        _time_ledger()
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("ledger", dt)
        samples["ledger"].append(dt)

    if inventory is None:
        raise RuntimeError("source discovery produced no inventory")

    phase_records: dict[str, dict[str, Any]] = {}
    for name in PHASES:
        if name in GPU_OR_BUILD_PHASES:
            phase_records[name] = _unknown_record(name)
        elif name == "next_decision":
            continue
        else:
            phase_records[name] = _measured_record(name, _summarize(samples[name]))

    next_samples: list[float] = []
    decision: dict[str, Any] | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        decision = _time_next_decision(phase_records, inventory, cargo)
        dt = (time.perf_counter() - t0) * 1000.0
        record_cpu_side("next_decision", dt)
        next_samples.append(dt)
    assert decision is not None
    # Re-run selection after next_decision has its own summary so the receipt
    # describes the complete measured set. Selection is deterministic.
    phase_records["next_decision"] = _measured_record(
        "next_decision", _summarize(next_samples)
    )
    decision = identify_dominant(phase_records, inventory, cargo)

    cpu_side_medians = {
        name: rec["phase_wall_ms_cpu_side"]
        for name, rec in phase_records.items()
        if rec["state"] == "MEASURED_CPU_SIDE"
    }
    ordered = [phase_records[name] for name in PHASES]

    scoreboard_path = REPO / "receipts" / "headless" / "ACCELERATOR_SCOREBOARD.json"
    scoreboard_py = REPO / "tools" / "accelerator" / "scoreboard.py"

    return {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Measure experiment-loop turnaround the way a token budget is "
            "measured: named phases, median with spread, dominant cost with "
            "evidence, a lever that does not weaken reproducibility."
        ),
        "evidence_class": "STATIC_ONLY",
        "measurement_kind": "process_wall_cpu_side",
        "repeats": repeats,
        "experiment_turnaround_ns": None,
        "total_experiment_turnaround_ns": None,
        "development_phases": null_scoreboard_development_phases(),
        "phases": ordered,
        "phase_wall_ms_cpu_side": cpu_side_medians,
        "source_inventory": inventory,
        "cargo_evidence": cargo,
        "dominant_cost": decision,
        "scoreboard_schema": {
            "keys": list(SCOREBOARD_DEVELOPMENT_PHASES),
            "mapping_from_lane_phases": dict(PHASE_TO_SCOREBOARD),
            "values_in_this_receipt": "all null — CPU timings are not ns GPU/build quantities",
        },
        "recovered_implementation": [
            {
                "path": "receipts/headless/ACCELERATOR_SCOREBOARD.json",
                "present_here": scoreboard_path.is_file(),
                "what": (
                    "Already defines experiment_turnaround_ns, "
                    "total_experiment_turnaround_ns, and development_phases "
                    "(transform_ns, compile_ns, load_ns, benchmark_ns, "
                    "verification_ns, receipt_ns, total_experiment_turnaround_ns), "
                    "currently all null. Disk is authority: those are the "
                    "scoreboard names. The 11 lane names are a finer split."
                ),
            },
            {
                "path": "tools/accelerator/scoreboard.py",
                "present_here": scoreboard_py.is_file(),
                "what": (
                    "Codex-owned derived view. _explicit_development_phase reads "
                    "only named fields; never invents a sum from incomplete "
                    "phases. Read-only for this lane."
                ),
            },
            {
                "path": "crates/hawking-core/src/startup_timing.rs",
                "present_here": (REPO / "crates/hawking-core/src/startup_timing.rs").is_file(),
                "what": (
                    "Process-local startup phase timers (HAWKING_STARTUP_TIMING=1). "
                    "Times the running binary, including metallib load/compile. "
                    "Does not fill the scoreboard development_phases block and "
                    "requires launching the engine. Extended, not forked."
                ),
            },
            {
                "path": "crates/hawking-core/src/metal/mod.rs",
                "present_here": (REPO / "crates/hawking-core/src/metal/mod.rs").is_file(),
                "what": (
                    "Metallib disk cache keyed by (device name, shader source "
                    "sha256, math mode). Kernel specialization cache already "
                    "exists; this lane does not reimplement it. Shader compile "
                    "stays UNKNOWN here because exercising the cache needs a GPU."
                ),
            },
            {
                "path": "Cargo.toml [profile.release] / [profile.release-fast]",
                "present_here": True,
                "what": (
                    "release is fat LTO, codegen-units=1; release-fast is "
                    "incremental with codegen-units=16 and is already forbidden "
                    "for TPS. default-members already drop hide-* from a bare "
                    "cargo build. Those levers are pulled."
                ),
            },
            {
                "path": ".cargo/config.toml",
                "present_here": (REPO / ".cargo/config.toml").is_file(),
                "what": (
                    "gitignored. On the parent checkout it sets "
                    "build.target-dir = workspace/ops/build/rust. Shared cache "
                    "is the contention this lane is forbidden from joining."
                ),
            },
        ],
        "gaps_closed": [
            "CPU-side timings for source_discovery, transform (python import), "
            "verify (pytest collect-only), receipt, ledger, next_decision, "
            "each a median with spread over repeated samples",
            "GPU/build phases compile, link, shader_compile, launch, execution "
            "recorded as UNKNOWN with a refusal that actually raises",
            "Scoreboard development_phases schema mirrored with honest nulls "
            "(no CPU-ms laundered into *_ns)",
            "Dominant-cost split: measured CPU-side winner vs hypothesized "
            "full-loop compile (UNKNOWN), plus one reproducibility-preserving lever",
            "Sealed receipts/future/EXPERIMENT_TURNAROUND.json",
        ],
        "negative_findings": [
            "ACCELERATOR_SCOREBOARD.json is not in git HEAD and not in this "
            "sparse worktree; recovered from the parent checkout as untracked disk state",
            "tools/accelerator/scoreboard.py is not in this sparse worktree; "
            "recovered read-only from the parent checkout",
            "Did not run cargo build, cargo test, or any GPU path",
            "Could not fill experiment_turnaround_ns: a total without compile "
            "and execution is not the scoreboard quantity",
            "Full-repo pytest collection was not run (sparse checkout would "
            "miss paths); verify is tools/future --collect-only only",
            "No existing tools/future/turnaround.py — this module is new",
            "Did not touch hcli/ledger.py; ledger timing is a tempfile JSONL analog",
        ],
        "scoreboard_presence": _scoreboard_on_disk(),
    }


def build(*, repeats: int = DEFAULT_REPEATS) -> Path:
    doc = measure(repeats=repeats)
    return write_receipt(RECEIPT, doc, "tools/future/turnaround.py")


def selftest(*, repeats: int = DEFAULT_REPEATS) -> Path:
    return build(repeats=repeats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args()
    out = build(repeats=args.repeats)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
