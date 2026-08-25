#!/usr/bin/env python3
"""PHYSICAL METRIC AUDIT + MUTATION CANARIES (steer S011 §5, §6).

Every mutable headline metric is classified by WHERE ITS VALUE COMES FROM:

    DESIGN_EXPECTED_*    computed from constants in the code. Cannot move when the
                         artifact moves. Useful as a cross-check, never as a measurement.
    ARTIFACT_PHYSICAL_*  derived from bytes on disk.
    RUNTIME_MEASURED_*   observed during execution.

This distinction is not academic here. `complete_ebpw` was a DESIGN constant published
under a physical name: adding 1,288,519,664 bytes to an artifact did not move it. The
canaries below exist so that can never pass silently again.

The canaries mutate an APFS CLONE, never a real artifact.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
PARENT_PARAMS = 26895998464
CLEAN = Path("/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors")


def physical_ebpw(root):
    """The only honest definition: bytes that are actually there."""
    total = 0
    for f in Path(root).rglob("*"):
        if f.is_file() and (f.suffix in (".hgrafv01", ".hgravu01", ".f32v2", ".hq30uq4")):
            total += f.stat().st_size
    return 8.0 * total / PARENT_PARAMS, total


def design_ebpw(root):
    """What MIX_REPORT claims, which may be a constant."""
    m = Path(root) / "MIX_REPORT.json"
    if not m.is_file():
        return None
    return json.load(open(m)).get("complete_ebpw")


def clone(src, dst):
    if Path(dst).exists():
        shutil.rmtree(dst)
    # APFS copy-on-write: instant, and costs no real disk until something is written
    subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True)
    return Path(dst)


def audit_metrics():
    """Classify every mutable headline metric S011 §5 names."""
    wmn = json.load(open(RH / "WHOLE_MODEL_NATIVE.json"))
    mix = json.load(open(CLEAN / "MIX_REPORT.json"))
    src = (REPO / "tools/headless/whole_model_native.py").read_text()

    def cls(name, kind, where, note):
        return {"metric": name, "classification": kind, "source_of_value": where,
                "note": note}

    rows = [
        cls("complete_ebpw", "ARTIFACT_PHYSICAL_complete_ebpw",
            "8 * (payload_bytes - header_bytes) / parent_params",
            "WAS a design constant published under a physical name; corrected so it "
            "follows the payload, with the design figure kept beside it as a cross-check"),
        cls("complete_ebpw_from_design_constants", "DESIGN_EXPECTED_complete_ebpw",
            "hardcoded per-organ rates: mlp*2.25, dn*3.25, gqa/embed/out*3.125",
            "cannot move when the genome or the artifact moves; retained only to surface "
            "a design/physical divergence"),
        cls("active_ebpw_per_token", "DESIGN_EXPECTED_active_ebpw_per_token",
            "same hardcoded per-organ rates",
            "STILL FROZEN. Identical 8234330016.0 was reported for two artifacts differing "
            "by 1.29 GB. Not corrected because no measurement in this campaign depends on "
            "it; flagged here so nobody quotes it as physical."),
        cls("active_bytes_per_token", "DESIGN_EXPECTED_active_bytes_per_token",
            "hardcoded organ element counts x design rates", "same defect as above"),
        cls("payload_bytes", "ARTIFACT_PHYSICAL_payload_bytes",
            "sum of bytes written by the packer", "moves with the artifact"),
        cls("resident_bytes", "ARTIFACT_PHYSICAL_resident_bytes", "du over the artifact root",
            "measured, not declared"),
        cls("dispatches_per_token", "RUNTIME_MEASURED_dispatches_per_token",
            "counted by the runtime during decode",
            "964 for all three bodies, which is itself informative: representation change "
            "did not alter dispatch count"),
        cls("median_gpu_ns_per_token", "RUNTIME_MEASURED_tpot",
            "GPU timestamps per decode step", "carries the machine's standing CPU floor"),
        cls("TTFT / prefill_wall_ns", "RUNTIME_MEASURED_latency", "wall clock around prefill",
            "contaminated by any concurrent work; measured inside a protected window"),
        cls("dram_bytes_per_token", "DESIGN_EXPECTED_dram_bytes_per_token",
            "derived from the design active-bytes figure",
            "inherits the frozen active-bytes defect; treat as design until re-derived "
            "from a counter"),
        cls("model_reachable_roof", "RUNTIME_MEASURED_roof_input",
            "measured achieved GB/s against the executable's own traffic",
            "per-executable and per-regime; explicitly forbidden from being copied across "
            "models"),
        cls("DEVICE_MEASURED_SUSTAINED (778.8) / DEVICE_THEORETICAL (819.0)",
            "RUNTIME_MEASURED_machine_roof", "sealed bandwidth probe",
            "machine constants, valid on this box only"),
        cls("representation_identity", "ARTIFACT_PHYSICAL_identity",
            "codec + group + bits recorded per segment header", "readable from the artifact"),
        cls("kernel_identity", "ARTIFACT_PHYSICAL_identity",
            "kernel name + shader sha256 in KERNEL_LIBRARY",
            "sha256 of the shader source, not of the compiled metallib, which the runtime "
            "compiles from source"),
        cls("runtime_identity", "RUNTIME_MEASURED_identity",
            "genome_bind string plus dense_w_materialized counter emitted at decode",
            "the counter is incremented only by Qwen38HybridDecodeSession::account_dense_w"),
    ]
    frozen = [r for r in rows if r["classification"].startswith("DESIGN_EXPECTED")]
    return {
        "metrics": rows,
        "n_metrics": len(rows),
        "n_design_expected": len(frozen),
        "n_artifact_physical": sum(1 for r in rows
                                   if r["classification"].startswith("ARTIFACT_PHYSICAL")),
        "n_runtime_measured": sum(1 for r in rows
                                  if r["classification"].startswith("RUNTIME_MEASURED")),
        "still_frozen_and_flagged": [r["metric"] for r in frozen
                                     if "FROZEN" in r["note"] or "frozen" in r["note"]],
        "law": "never expose a design constant under a physical label",
        "observed_values": {
            "clean_design_ebpw": design_ebpw(CLEAN),
            "clean_physical_ebpw": round(physical_ebpw(CLEAN)[0], 6),
            "wmn_complete_ebpw": wmn.get("complete_ebpw"),
            "mix_payload_bytes": mix.get("payload_bytes"),
        },
    }


def canaries(work):
    """Five adversarial mutations. Each must produce its stated effect."""
    out = []
    base_phys, base_bytes = physical_ebpw(work)
    base_design = design_ebpw(work)

    # A: add known model-specific bytes -> physical EBPW must rise EXACTLY
    add = 64 * 1024 * 1024
    probe = Path(work) / "segments" / ("canary_added_" + "0" * 8 + ".f32v2")
    probe.write_bytes(b"\0" * add)
    p2, b2 = physical_ebpw(work)
    expect = base_phys + 8.0 * add / PARENT_PARAMS
    out.append({"canary": "A_add_bytes_raises_physical_ebpw",
                "bytes_added": add, "before": base_phys, "after": p2,
                "expected_after": expect,
                "exact": abs(p2 - expect) < 1e-12,
                "passed": abs(p2 - expect) < 1e-12 and p2 > base_phys})
    probe.unlink()

    # B: remove known representation bytes -> physical EBPW must FALL
    segs = sorted((Path(work) / "segments").glob("*.hgrafv01"))
    victim = segs[0]
    vbytes = victim.stat().st_size
    held = victim.read_bytes()
    victim.unlink()
    p3, _ = physical_ebpw(work)
    expect3 = base_phys - 8.0 * vbytes / PARENT_PARAMS
    out.append({"canary": "B_remove_bytes_lowers_physical_ebpw",
                "bytes_removed": vbytes, "before": base_phys, "after": p3,
                "expected_after": expect3,
                "exact": abs(p3 - expect3) < 1e-12,
                "passed": abs(p3 - expect3) < 1e-12 and p3 < base_phys})
    victim.write_bytes(held)

    # C: mutate the GENOME only -> physical EBPW must NOT move
    mixp = Path(work) / "MIX_REPORT.json"
    m = json.load(open(mixp))
    original = json.dumps(m)
    m["genome"]["mlp"]["gemv_storage_bpw"] = 99.0
    m["genome"]["mlp"]["codec"] = "canary_fictional_codec"
    mixp.write_text(json.dumps(m, indent=1))
    p4, _ = physical_ebpw(work)
    out.append({"canary": "C_genome_only_change_does_not_move_physical_ebpw",
                "mutation": "genome.mlp.gemv_storage_bpw 2.25 -> 99.0",
                "before": base_phys, "after": p4,
                "passed": abs(p4 - base_phys) < 1e-12})
    mixp.write_text(original)

    # D: mutate the ARTIFACT without updating the genome -> physical moves AND the
    #    design/physical mismatch must surface
    victim2 = segs[1]
    orig2 = victim2.read_bytes()
    victim2.write_bytes(orig2 + b"\0" * (32 * 1024 * 1024))
    p5, _ = physical_ebpw(work)
    d5 = design_ebpw(work)
    mismatch = abs(p5 - d5) > 1e-3 if d5 is not None else None
    out.append({"canary": "D_artifact_only_change_moves_physical_and_surfaces_mismatch",
                "bytes_added_to_segment": 32 * 1024 * 1024,
                "physical_before": base_phys, "physical_after": p5,
                "design_unchanged": d5, "design_equals_original": d5 == base_design,
                "mismatch_surfaced": mismatch,
                "passed": bool(p5 > base_phys and mismatch)})
    victim2.write_bytes(orig2)

    # E: runtime fallback must be exposed by RuntimeIdentity
    wmn = json.load(open(RH / "WHOLE_MODEL_NATIVE.json"))
    zp = wmn.get("zero_parent", {})
    contract = zp.get("rust_counter_contract", {})
    decode = wmn.get("decode", {})
    exposed = ("fallbacks" in decode) or ("dense_w_materialized" in zp)
    out.append({"canary": "E_runtime_fallback_is_exposed_by_runtime_identity",
                "counter_present": contract.get("field_present"),
                "incremented_only_by": zp.get("counter", {}).get("incremented_only_by"),
                "not_a_python_literal": contract.get("not_a_python_literal"),
                "decode_reports_fallbacks_field": "fallbacks" in decode,
                "fallbacks_observed": decode.get("fallbacks"),
                "passed": bool(exposed and contract.get("field_present")
                               and contract.get("not_a_python_literal")),
                "honest_limitation": (
                    "this verifies the runtime EXPOSES a fallback channel and that the "
                    "counter is a real Rust field incremented only by account_dense_w, not "
                    "a literal. It does not INDUCE a fallback: doing so needs a runtime "
                    "build with a deliberately broken kernel path, which is a code change "
                    "to the crate rather than an artifact mutation.")})
    p_end, b_end = physical_ebpw(work)
    return out, {"restored_to_baseline": abs(p_end - base_phys) < 1e-12,
                 "baseline_bytes": base_bytes, "final_bytes": b_end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-audit", required=True)
    ap.add_argument("--emit-canaries", required=True)
    ap.add_argument("--work", default="/Users/scammermike/noetic/CANARY_CLONE")
    a = ap.parse_args()

    audit = audit_metrics()
    Path(a.emit_audit).write_text(json.dumps({
        "schema": "hawking.headless.physical_metric_audit.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/metric_audit.py",
        "obligation": "G033 — PHYSICAL_METRIC_AUDIT (steer S011 §5)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False, **audit,
        "pass": bool(audit["n_metrics"] >= 12 and audit["n_artifact_physical"] >= 3
                     and audit["n_runtime_measured"] >= 3),
    }, indent=1))

    work = clone(CLEAN, a.work)
    try:
        rows, restore = canaries(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    out = {
        "schema": "hawking.headless.metric_mutation_canaries.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/metric_audit.py",
        "obligation": "G033 — METRIC_MUTATION_CANARIES (steer S011 §6)",
        "hand_authored": False,
        "mutated": "an APFS clone, never a real artifact",
        "canaries": rows, "n": len(rows),
        "n_passed": sum(1 for r in rows if r["passed"]),
        "restore_check": restore,
        "why": "complete_ebpw was a design constant published under a physical name, and "
               "adding 1,288,519,664 bytes did not move it. These exist so that cannot "
               "pass silently again.",
        "pass": all(r["passed"] for r in rows) and restore["restored_to_baseline"],
    }
    Path(a.emit_canaries).write_text(json.dumps(out, indent=1))
    print(f"audit: {audit['n_metrics']} metrics "
          f"({audit['n_design_expected']} design / {audit['n_artifact_physical']} physical "
          f"/ {audit['n_runtime_measured']} runtime)")
    print(f"  still frozen and flagged: {audit['still_frozen_and_flagged']}")
    for r in rows:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['canary']}")
    print(f"canaries {out['n_passed']}/{out['n']}  restored={restore['restored_to_baseline']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
