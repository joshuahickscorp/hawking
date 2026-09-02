"""SYSTEMS / COMPILER laboratory — H.5 / compiler-runtime against this repo.

Runs the existing host/shader ABI preflight (`analyze`, `parse_metal`,
`load_repo_sources`) on crates/hawking-core in this checkout, then checks
KERNEL_GEOMETRY coverage identities as STATIC arithmetic.

This is STATIC_ONLY. Matching buffer indices or trips*stride==cols is not
a speed claim and not a hardware occupancy counter. GPU kernel A/B numbers
in other receipts are not re-claimed here.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import REPO, write_receipt
from tools.theia.bounty import BountyClass, make_internal_bounty
from tools.theia.intake import VerificationFailed, run_intake
from tools.theia.labs import LabKind
from tools.theia.value import DeclaredFactor, ValueInputs

RECORDED_BY = "tools/theia/systems_lab.py"
RECEIPT_NAME = "THEIA_SYSTEMS_COMPILER.json"
SCHEMA = "hawking.theia.systems_compiler.v1"
EVIDENCE_TIER = "STATIC"
KERNEL_GEOMETRY = REPO / "receipts" / "future" / "KERNEL_GEOMETRY.json"

UNIT_SOURCE = (
    "declared unit baseline so the H.1 denominator is defined; STATIC_ONLY; "
    "not a hardware measurement"
)


def apple_threadgroup_ceiling() -> int:
    from tools.future.static_kernel_verify import APPLE_MAX_THREADS_PER_THREADGROUP

    return int(APPLE_MAX_THREADS_PER_THREADGROUP)


def parse_repo_shaders(metal: Mapping[str, str]) -> dict[str, Any]:
    """Call parse_metal on this repo's own shader sources."""
    from tools.future.static_kernel_verify import parse_metal

    kernels = []
    structs = 0
    by_file: dict[str, int] = {}
    for path, src in sorted(metal.items()):
        ks, sts = parse_metal(src, path)
        by_file[path] = len(ks)
        structs += len(sts)
        for k in ks:
            kernels.append({"name": k.name, "path": k.path, "line": k.line})
    return {
        "n_shader_files": len(metal),
        "n_kernels": len(kernels),
        "n_structs": structs,
        "kernels_by_file": by_file,
        "kernel_names": sorted({k["name"] for k in kernels}),
        "symbol": "tools.future.static_kernel_verify.parse_metal",
    }


def geometry_identities(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """STATIC coverage identities from KERNEL_GEOMETRY.json. Not occupancy counters."""
    if doc is None:
        doc = json.loads(KERNEL_GEOMETRY.read_text())
    occ = doc["occupancy_class"]
    tpr = int(occ["threads_per_row"])
    rpt = int(occ["rows_per_tg"])
    tg = int(occ["threadgroup"])
    stride = int(occ["col_stride"])
    ceiling = apple_threadgroup_ceiling()
    occupancy_ok = tpr * rpt == tg
    under_ceiling = tg <= ceiling
    organs = []
    for organ in doc.get("organs") or []:
        cov = organ.get("coverage") or {}
        cols = int(organ["cols"])
        trips = int(cov["trips"])
        dropped = int(cov.get("dropped") or 0)
        match = trips * stride == cols and dropped == 0
        organs.append(
            {
                "role": organ.get("role"),
                "cols": cols,
                "trips": trips,
                "dropped": dropped,
                "trips_times_stride": trips * stride,
                "holds": match,
            }
        )
    n_hold = sum(1 for o in organs if o["holds"])
    return {
        "source": str(KERNEL_GEOMETRY),
        "occupancy_identity": {
            "threads_per_row": tpr,
            "rows_per_tg": rpt,
            "threadgroup": tg,
            "holds": occupancy_ok,
            "formula": "threadgroup == threads_per_row * rows_per_tg",
        },
        "under_apple_ceiling": under_ceiling,
        "apple_ceiling": ceiling,
        "ceiling_symbol": "tools.future.static_kernel_verify.APPLE_MAX_THREADS_PER_THREADGROUP",
        "col_stride": stride,
        "organs": organs,
        "n_organs": len(organs),
        "n_hold": n_hold,
        "evidence_tier": EVIDENCE_TIER,
        "not_a_hardware_occupancy_counter": True,
    }


def run_preflight() -> dict[str, Any]:
    """Call load_repo_sources and analyze on this checkout. STATIC_ONLY."""
    from tools.future.static_kernel_verify import (
        analyze,
        load_repo_sources,
        report_from_analyze,
    )

    metal, rust, membership = load_repo_sources(REPO)
    parsed = parse_repo_shaders(metal)
    raw = analyze(metal, rust, library_membership=membership)
    report = report_from_analyze(raw)
    errors = [
        {
            "severity": f.get("severity"),
            "check": f.get("check"),
            "message": f.get("message"),
            "kernel": f.get("kernel"),
            "shader": f.get("shader"),
            "host": f.get("host"),
        }
        for f in report.get("findings") or []
        if isinstance(f, dict) and f.get("severity") == "ERROR"
    ]
    return {
        "load_repo_sources": {
            "symbol": "tools.future.static_kernel_verify.load_repo_sources",
            "n_metal": len(metal),
            "n_rust": len(rust),
            "n_membership": len(membership),
        },
        "parse_metal": parsed,
        "analyze": {
            "symbol": "tools.future.static_kernel_verify.analyze",
            "counts": report.get("counts"),
            "coverage": {
                k: report.get("coverage", {}).get(k)
                for k in (
                    "metal_files",
                    "metal_kernels",
                    "host_dispatches",
                    "dispatches_resolved",
                    "binding_pairs_with_matching_index_sets",
                    "threadgroup_triples_resolved",
                    "structs_paired",
                )
            },
            "blocking_defect_count": report.get("blocking_defect_count"),
            "would_waste_a_protected_window": report.get(
                "would_waste_a_protected_window"
            ),
            "errors": errors,
        },
        "evidence_class": "STATIC_ONLY",
        "static_correctness_does_not_prove_speed": True,
        "does_not_substitute_for_protected_measurement": True,
    }


def verify_systems_receipt(path: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Re-parse shaders and re-check geometry. Does not re-run 17s analyze."""
    del path
    from tools.future.static_kernel_verify import load_repo_sources, parse_metal

    metal, _rust, _membership = load_repo_sources(REPO)
    live_names: set[str] = set()
    n_kernels = 0
    for pth, src in metal.items():
        ks, _sts = parse_metal(src, pth)
        n_kernels += len(ks)
        live_names.update(k.name for k in ks)
    claimed = (((doc.get("preflight") or {}).get("parse_metal") or {}).get("n_kernels"))
    if claimed != n_kernels:
        raise VerificationFailed(
            f"parse_metal kernel count {n_kernels} != receipt {claimed}"
        )
    claimed_names = set(
        ((doc.get("preflight") or {}).get("parse_metal") or {}).get("kernel_names")
        or []
    )
    if claimed_names != live_names:
        raise VerificationFailed("parse_metal kernel names disagree with the receipt")
    geo_live = geometry_identities()
    geo_doc = doc.get("geometry_identities") or {}
    if geo_live["n_hold"] != geo_doc.get("n_hold"):
        raise VerificationFailed("geometry_identities n_hold disagrees on re-check")
    if geo_live["occupancy_identity"]["holds"] is not True:
        raise VerificationFailed("occupancy identity does not hold")
    if geo_live["n_hold"] != geo_live["n_organs"]:
        raise VerificationFailed("not every KERNEL_GEOMETRY organ coverage identity holds")
    analyze_counts = (doc.get("preflight") or {}).get("analyze") or {}
    if int(analyze_counts.get("blocking_defect_count") or -1) < 0:
        raise VerificationFailed("systems receipt missing analyze.blocking_defect_count")
    return {
        "n_kernels": n_kernels,
        "n_geometry_hold": geo_live["n_hold"],
        "blocking_defect_count": analyze_counts.get("blocking_defect_count"),
        "independent_module": "tools.future.static_kernel_verify.parse_metal",
        "analyze_symbol_was": "tools.future.static_kernel_verify.analyze",
        "load_symbol": "tools.future.static_kernel_verify.load_repo_sources",
        "evidence_tier": EVIDENCE_TIER,
    }


def _value_inputs(preflight: Mapping[str, Any], geometry: Mapping[str, Any]) -> ValueInputs:
    n_errors = int((preflight.get("analyze") or {}).get("blocking_defect_count") or 0)
    n_geo = int(geometry.get("n_hold") or 0)
    n_kernels = int((preflight.get("parse_metal") or {}).get("n_kernels") or 0)
    if n_kernels < 1:
        raise VerificationFailed("systems lab parsed no kernels")
    if n_geo < 1:
        raise VerificationFailed("systems lab held no geometry identities")
    # Bound information_gain to the independent checks that ran, not kernel count
    # (766 would dominate H.1 and look like a capability claim).
    info = n_errors + n_geo
    if info < 1:
        info = 1

    def unit(name: str, source: str) -> DeclaredFactor:
        return DeclaredFactor(value=Fraction(1), name=name, source=source)

    return ValueInputs(
        verified_reward=unit(
            "verified_reward",
            "H.1: verified_reward may include an accepted patch / compiler finding; "
            "SYSTEMS/COMPILER static preflight",
        ),
        probability_of_success=unit(
            "probability_of_success",
            "local parse of shaders and receipts already on disk",
        ),
        information_gain=DeclaredFactor(
            value=Fraction(info),
            name="information_gain",
            source=(
                f"blocking_defect_count {n_errors} plus KERNEL_GEOMETRY organs "
                f"whose trips*col_stride==cols ({n_geo}); STATIC"
            ),
        ),
        transfer_value=unit(
            "transfer_value",
            "host/shader ABI preflight method plus occupancy identity reusable as a compiler law",
        ),
        strategic_relevance=unit(
            "strategic_relevance",
            "SYSTEMS / COMPILER laboratory, §19.12 / H.5",
        ),
        wall_time=unit("wall_time", UNIT_SOURCE),
        compute_cost=unit("compute_cost", UNIT_SOURCE),
        human_cost=unit("human_cost", UNIT_SOURCE),
        risk=unit(
            "risk",
            "local source parse; no network, no ACTIVE_TEST, no credentials, no dispatch",
        ),
        opportunity_cost=unit("opportunity_cost", UNIT_SOURCE),
    )


def assemble_receipt_doc(
    preflight: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": 1,
        "recorded_by": RECORDED_BY,
        "lab": LabKind.SYSTEMS_COMPILER.value,
        "recipe": "H.5",
        "evidence_class": "STATIC_ONLY",
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "Host/shader ABI preflight and KERNEL_GEOMETRY coverage identities "
            "against this checkout's crates/hawking-core. STATIC_ONLY. Static "
            "correctness does not prove speed. No command queue, no dispatch, "
            "no hardware measurement."
        ),
        "h5": {
            "baseline": "crates/hawking-core shaders + hosts in this sparse checkout",
            "profiler_or_cost": "static ABI / geometry, not a profiler",
            "correctness_gate": "parse_metal + analyze ERROR tally",
            "not_run": [
                "realistic GPU workload gate",
                "capability gate",
                "power/resource gate",
            ],
            "accept_or_reject": "ACCEPT as a scored static bounty; REJECT as a speed claim",
        },
        "executable_work": [
            "runtime_bugs",
            "compiler_bugs",
            "kernel_optimization",
            "performance_archaeology",
            "hardware_mapping",
        ],
        "preflight": preflight,
        "geometry_identities": geometry,
        "apple_threadgroup_ceiling": apple_threadgroup_ceiling(),
        "repo_paths": {
            "shaders": "crates/hawking-core/shaders",
            "hosts": "crates/hawking-core",
            "kernel_geometry_receipt": str(KERNEL_GEOMETRY),
        },
    }


def run_systems_bounty(*, write: bool = True):
    """Execute one real SYSTEMS/COMPILER bounty through H.2."""
    preflight = run_preflight()
    geometry = geometry_identities()
    if geometry["n_hold"] != geometry["n_organs"]:
        raise VerificationFailed("KERNEL_GEOMETRY coverage identity failed on an organ")
    if geometry["occupancy_identity"]["holds"] is not True:
        raise VerificationFailed("KERNEL_GEOMETRY occupancy identity failed")
    doc = assemble_receipt_doc(preflight, geometry)
    artifact = write_receipt(RECEIPT_NAME, doc, recorded_by=RECORDED_BY)
    if not write:
        del write
    bounty = make_internal_bounty(
        id=f"systems:{SCHEMA}:hawking-core-abi-preflight",
        source=str(artifact.resolve()),
        domain="systems",
        question_or_target=(
            "Run the host/shader ABI preflight against this repo's own "
            "crates/hawking-core Metal/Rust and check KERNEL_GEOMETRY coverage "
            "identities (trips*col_stride==cols; threadgroup==tpr*rows_per_tg)."
        ),
        nonmonetary_value="compiler/runtime bug",
        bounty_class=BountyClass.COMPILER_RUNTIME_BUG,
        lab=LabKind.SYSTEMS_COMPILER.value,
        extra_rules=("H.5 optimization recipe, static gates only", "no GPU dispatch"),
        evidence_required=(
            "receipt schema",
            "seal_sha256",
            "tools.future.static_kernel_verify.analyze",
            "tools.future.static_kernel_verify.parse_metal",
        ),
    )

    def inputs(_artifact: Path) -> ValueInputs:
        return _value_inputs(preflight, geometry)

    return run_intake(
        bounty,
        value_inputs_factory=inputs,
        expected_schema=SCHEMA,
    )
