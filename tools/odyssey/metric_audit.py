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

G059 hardened three things:
  1. `active_ebpw_per_token` / `active_bytes_per_token` now have ARTIFACT_PHYSICAL
     derivations that follow bytes on disk, and the MIX_REPORT constants survive only
     under explicit DESIGN_EXPECTED_* names beside them.
  2. `dram_bytes_per_token` cannot be measured on this box (no userspace DRAM traffic
     counter), so it is RENAMED into DESIGN_EXPECTED_* and carries its blocking gate.
     It is not given a physical name it has not earned.
  3. `still_frozen_and_flagged` is now a STRUCTURAL rule -- a metric whose value comes
     from constants may not be published under a name outside the DESIGN_EXPECTED_*
     namespace -- instead of a substring search over the prose in its own note.

The canaries mutate an APFS CLONE, never a real artifact.
Run `--emit-negative-control` to watch A, C and D FAIL on injected defects.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(os.environ.get("HAWKING_REPO") or Path(__file__).resolve().parents[2])
RH = REPO / "receipts/headless"
PARENT_PARAMS = 26895998464
CLEAN = Path("/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors")
VARIANT_A = Path("/Users/scammermike/noetic/VARIANT_A_MLP_ONLY")
PAYLOAD_SUFFIXES = (".hgrafv01", ".hgravu01", ".f32v2", ".hq30uq4")


def physical_ebpw(root):
    """The only honest definition: bytes that are actually there."""
    total = 0
    for f in Path(root).rglob("*"):
        if f.is_file() and f.suffix in PAYLOAD_SUFFIXES:
            total += f.stat().st_size
    return 8.0 * total / PARENT_PARAMS, total


def design_ebpw(root):
    """What MIX_REPORT claims, which may be a constant."""
    m = Path(root) / "MIX_REPORT.json"
    if not m.is_file():
        return None
    return json.load(open(m)).get("complete_ebpw")


def active_physical(root):
    """ACTIVE bytes/token and active EBPW derived from BYTES ON DISK.

    Every organ is read whole once per decoded token except the embedding table, which
    is a single-row gather, so active bytes = payload bytes on disk minus the embedding
    table. This is a dense body: there is no routing to make it smaller.

    The role split itself comes from MIX_REPORT.payload_bytes_by_role, which is the one
    piece of this that is read rather than measured -- so the reconciliation of that
    split against the bytes actually on disk is reported, not assumed. The returned
    figure is anchored on the DISK total, so it moves when the artifact moves even if
    the packer's role table is stale.
    """
    mp = Path(root) / "MIX_REPORT.json"
    if not mp.is_file():
        return {"available": False, "why": "no MIX_REPORT.json under this root"}
    m = json.load(open(mp))
    by_role = m.get("payload_bytes_by_role")
    if not by_role:
        return {"available": False,
                "why": "MIX_REPORT has no payload_bytes_by_role; this body predates "
                       "per-role accounting and its ACTIVE vector cannot be derived "
                       "from bytes. It is UNMEASURED here, not estimated."}
    disk_total = physical_ebpw(root)[1]
    embed = by_role.get("embedding", 0)
    active = disk_total - embed
    return {
        "available": True,
        "active_bytes_per_token": active,
        "active_ebpw_per_token": 8.0 * active / PARENT_PARAMS,
        "payload_bytes_on_disk": disk_total,
        "embedding_bytes_excluded_as_gather": embed,
        "role_split_reconciles_with_disk": sum(by_role.values()) == disk_total,
        "design_active_bytes_per_token": m.get("active_bytes_per_token"),
        "design_active_ebpw_per_token": m.get("active_ebpw_per_token"),
    }


def frozen_design_evidence():
    """Two DIFFERENT artifacts, one identical design constant. Machine-read, not quoted."""
    rows = {}
    for tag, root in (("clean-2.60", CLEAN), ("variantA-2.98", VARIANT_A)):
        ap = active_physical(root)
        mp = Path(root) / "MIX_REPORT.json"
        m = json.load(open(mp)) if mp.is_file() else {}
        rows[tag] = {
            "payload_bytes": m.get("payload_bytes"),
            "DESIGN_EXPECTED_active_bytes_per_token": m.get("active_bytes_per_token"),
            "DESIGN_EXPECTED_active_ebpw_per_token": m.get("active_ebpw_per_token"),
            "ARTIFACT_PHYSICAL_active_bytes_per_token": ap.get("active_bytes_per_token"),
            "ARTIFACT_PHYSICAL_active_ebpw_per_token": (
                round(ap["active_ebpw_per_token"], 6) if ap.get("available") else None),
            "role_split_reconciles_with_disk": ap.get("role_split_reconciles_with_disk"),
        }
    tags = list(rows)
    a, b = rows[tags[0]], rows[tags[1]]
    design_same = (a["DESIGN_EXPECTED_active_bytes_per_token"]
                   == b["DESIGN_EXPECTED_active_bytes_per_token"])
    phys = [r["ARTIFACT_PHYSICAL_active_bytes_per_token"] for r in (a, b)]
    rel = (abs(phys[0] - phys[1]) / max(phys) if all(phys) else None)
    return {
        "bodies": rows,
        "payload_bytes_differ": a["payload_bytes"] != b["payload_bytes"],
        "design_constant_identical_across_both": design_same,
        "physical_figures_differ_by_rel": rel,
        "reading": (
            "the two bodies differ in payload bytes, the ARTIFACT_PHYSICAL active figure "
            "differs with them, and the MIX_REPORT design figure is bit-identical across "
            "both. That is the frozen-constant defect observed directly rather than "
            "recalled from a prior receipt."),
    }


def dram_gate():
    """Name the gate that blocks measuring DRAM traffic per token, by running into it."""
    p = subprocess.run(["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "200"],
                       capture_output=True, text=True)
    observed = (p.stdout + p.stderr).strip().splitlines()
    return {
        "metric": "dram_bytes_per_token",
        "why_unmeasurable_here": (
            "no userspace DRAM traffic counter is reachable on this box. powermetrics "
            "requires root, and Metal memory counters require MTLCounterSampleBuffer "
            "instrumentation inside crates/hawking-core, which is a runtime code change "
            "and outside this lane."),
        "probe_command": "powermetrics --samplers gpu_power -n 1 -i 200",
        "probe_exit_code": p.returncode,
        "probe_observed": observed[0] if observed else "",
        "what_is_published_instead": (
            "an UPPER BOUND computed from artifact bytes (one full read of every "
            "non-embedding organ per decoded token, assumed-zero inter-token reuse). "
            "The bound moves with the artifact; the reuse assumption is the part that "
            "is unmeasured, and it is a design assumption, so the figure is published "
            "under DESIGN_EXPECTED_dram_bytes_per_token."),
    }


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

    def cls(name, kind, where, note, blocking_gate=None, measured_twin=None):
        return {"metric": name, "classification": kind, "source_of_value": where,
                "note": note, "blocking_gate": blocking_gate,
                "measured_twin": measured_twin}

    rows = [
        cls("complete_ebpw", "ARTIFACT_PHYSICAL_complete_ebpw",
            "8 * (payload_bytes - header_bytes) / parent_params",
            "WAS a design constant published under a physical name; corrected so it "
            "follows the payload, with the design figure kept beside it as a cross-check"),
        cls("DESIGN_EXPECTED_complete_ebpw", "DESIGN_EXPECTED_complete_ebpw",
            "hardcoded per-organ rates: mlp*2.25, dn*3.25, gqa/embed/out*3.125",
            "cannot move when the genome or the artifact moves; retained only to surface "
            "a design/physical divergence. G059 gave it the namespace prefix it was "
            "already entitled to -- it used to be published as "
            "'complete_ebpw_from_design_constants', which is not a physical name but is "
            "not a declared one either.",
            measured_twin="complete_ebpw"),
        cls("active_ebpw_per_token", "ARTIFACT_PHYSICAL_active_ebpw_per_token",
            "8 * (payload_bytes_on_disk - embedding_bytes) / parent_params",
            "G059: was DESIGN_EXPECTED wearing a physical name -- identical "
            "8234330016.0 bytes was reported for two artifacts differing by 1.29 GB. "
            "Now anchored on the disk total, so canaries A and D move it and canary C "
            "cannot."),
        cls("active_bytes_per_token", "ARTIFACT_PHYSICAL_active_bytes_per_token",
            "payload_bytes_on_disk - payload_bytes_by_role['embedding']",
            "same derivation as above in bytes. The role split is read from MIX_REPORT; "
            "its reconciliation against the bytes on disk is reported per body rather "
            "than assumed."),
        cls("DESIGN_EXPECTED_active_ebpw_per_token", "DESIGN_EXPECTED_active_ebpw_per_token",
            "MIX_REPORT.active_ebpw_per_token: hardcoded organ element counts x design rates",
            "kept as the cross-check partner of the physical figure. Frozen by "
            "construction: see observed_values.frozen_design_evidence, where both bodies "
            "publish the same number while their physical figures differ.",
            measured_twin="active_ebpw_per_token"),
        cls("DESIGN_EXPECTED_active_bytes_per_token", "DESIGN_EXPECTED_active_bytes_per_token",
            "MIX_REPORT.active_bytes_per_token: hardcoded organ element counts x design rates",
            "same defect, same treatment", measured_twin="active_bytes_per_token"),
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
        cls("DESIGN_EXPECTED_dram_bytes_per_token", "DESIGN_EXPECTED_dram_bytes_per_token",
            "artifact active bytes under an ASSUMED-ZERO inter-token reuse model",
            "G059: this used to be published as 'dram_bytes_per_token' and (in "
            "perf_addendum) as ARTIFACT_PHYSICAL_dram_bytes_per_token. The byte count is "
            "physical; the claim that the memory system moves exactly those bytes per "
            "token is a design assumption nothing on this box can check. Renamed rather "
            "than fabricated.",
            blocking_gate="powermetrics requires root; Metal memory counters require "
                          "MTLCounterSampleBuffer instrumentation inside crates/, which "
                          "is a runtime code change and out of this lane's scope. See "
                          "observed_values.dram_gate for the probe that was actually run."),
        cls("model_reachable_roof", "RUNTIME_MEASURED_roof_input",
            "measured achieved GB/s against the executable's own traffic",
            "per-executable and per-regime; explicitly forbidden from being copied across "
            "models. HONEST LIMIT: the numerator is the DESIGN_EXPECTED dram bound above, "
            "so this is measured-time over modelled-bytes, not a measured bandwidth."),
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
    # STRUCTURAL rule, not a substring search over the metric's own prose: a value that
    # comes from constants may not be published under a name outside DESIGN_EXPECTED_*.
    # This is what "a design constant wearing a physical name" means, mechanically.
    violations = [r["metric"] for r in frozen
                  if not r["metric"].startswith("DESIGN_EXPECTED_")]
    undeclared = [r["metric"] for r in frozen
                  if not r["blocking_gate"] and not r["measured_twin"]]
    ap = active_physical(CLEAN)
    return {
        "metrics": rows,
        "n_metrics": len(rows),
        "n_design_expected": len(frozen),
        "n_artifact_physical": sum(1 for r in rows
                                   if r["classification"].startswith("ARTIFACT_PHYSICAL")),
        "n_runtime_measured": sum(1 for r in rows
                                  if r["classification"].startswith("RUNTIME_MEASURED")),
        "still_frozen_and_flagged": violations,
        "frozen_rule": "a metric classified DESIGN_EXPECTED_* must be PUBLISHED under a "
                       "DESIGN_EXPECTED_* name. Structural, so it cannot be cleared by "
                       "editing the note.",
        "design_metrics_without_gate_or_measured_twin": undeclared,
        "law": "never expose a design constant under a physical label",
        "observed_values": {
            "clean_design_ebpw": design_ebpw(CLEAN),
            "clean_physical_ebpw": round(physical_ebpw(CLEAN)[0], 6),
            "wmn_complete_ebpw": wmn.get("complete_ebpw"),
            "mix_payload_bytes": mix.get("payload_bytes"),
            "clean_ARTIFACT_PHYSICAL_active_bytes_per_token": ap.get("active_bytes_per_token"),
            "clean_ARTIFACT_PHYSICAL_active_ebpw_per_token": (
                round(ap["active_ebpw_per_token"], 6) if ap.get("available") else None),
            "clean_DESIGN_EXPECTED_active_bytes_per_token": ap.get("design_active_bytes_per_token"),
            "frozen_design_evidence": frozen_design_evidence(),
            "dram_gate": dram_gate(),
        },
    }


def canaries(work):
    """Five adversarial mutations. Each must produce its stated effect."""
    out = []
    base_phys, base_bytes = physical_ebpw(work)
    base_design = design_ebpw(work)
    base_active = active_physical(work)["active_bytes_per_token"]

    # A: add known model-specific bytes -> physical EBPW must rise EXACTLY
    add = 64 * 1024 * 1024
    probe = Path(work) / "segments" / ("canary_added_" + "0" * 8 + ".f32v2")
    probe.write_bytes(b"\0" * add)
    p2, b2 = physical_ebpw(work)
    a2 = active_physical(work)["active_bytes_per_token"]
    expect = base_phys + 8.0 * add / PARENT_PARAMS
    out.append({"canary": "A_add_bytes_raises_physical_ebpw",
                "bytes_added": add, "before": base_phys, "after": p2,
                "expected_after": expect,
                "exact": abs(p2 - expect) < 1e-12,
                "active_before": base_active, "active_after": a2,
                "active_expected_after": base_active + add,
                "active_exact": a2 == base_active + add,
                "passed": (abs(p2 - expect) < 1e-12 and p2 > base_phys
                           and a2 == base_active + add)})
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

    # C: mutate the GENOME only -> NO physical field may move. This is the load-bearing
    #    canary: it is what catches a genome literal reaching a physical accounting
    #    field. It now covers the ACTIVE figures too, which DO open MIX_REPORT.json.
    mixp = Path(work) / "MIX_REPORT.json"
    m = json.load(open(mixp))
    original = json.dumps(m)
    m["genome"]["mlp"]["gemv_storage_bpw"] = 99.0
    m["genome"]["mlp"]["codec"] = "canary_fictional_codec"
    mixp.write_text(json.dumps(m, indent=1))
    p4, _ = physical_ebpw(work)
    a4 = active_physical(work)["active_bytes_per_token"]
    out.append({"canary": "C_genome_only_change_does_not_move_physical_ebpw",
                "mutation": "genome.mlp.gemv_storage_bpw 2.25 -> 99.0, codec -> fictional",
                "before": base_phys, "after": p4,
                "active_before": base_active, "active_after": a4,
                "fields_covered": ["complete_ebpw", "active_bytes_per_token",
                                   "active_ebpw_per_token"],
                "passed": abs(p4 - base_phys) < 1e-12 and a4 == base_active})
    mixp.write_text(original)

    # D: mutate the ARTIFACT without updating the genome -> physical moves AND the
    #    design/physical mismatch must surface
    victim2 = segs[1]
    orig2 = victim2.read_bytes()
    victim2.write_bytes(orig2 + b"\0" * (32 * 1024 * 1024))
    p5, _ = physical_ebpw(work)
    a5 = active_physical(work)["active_bytes_per_token"]
    d5 = design_ebpw(work)
    mismatch = abs(p5 - d5) > 1e-3 if d5 is not None else None
    out.append({"canary": "D_artifact_only_change_moves_physical_and_surfaces_mismatch",
                "bytes_added_to_segment": 32 * 1024 * 1024,
                "physical_before": base_phys, "physical_after": p5,
                "active_before": base_active, "active_after": a5,
                "design_unchanged": d5, "design_equals_original": d5 == base_design,
                "mismatch_surfaced": mismatch,
                "passed": bool(p5 > base_phys and mismatch and a5 > base_active)})
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


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS. Each entry injects, into a COPY of this file, the exact defect
# one check exists to catch, and the run is only accepted if that check FLIPS TO FAIL.
# An anchor that no longer matches is a hard error: a control that silently patches
# nothing is the vacuous check this whole file exists to prevent.
# ---------------------------------------------------------------------------
NEG_CONTROLS = [
    {
        "name": "C_genome_literal_into_complete_ebpw",
        "target": {"kind": "canary", "canary": "C_genome_only_change"},
        "defect": "route a genome literal into the ARTIFACT_PHYSICAL complete_ebpw field",
        "old": "    return 8.0 * total / PARENT_PARAMS, total\n",
        "new": ("    _g = json.load(open(Path(root) / \"MIX_REPORT.json\"))"
                "[\"genome\"][\"mlp\"][\"gemv_storage_bpw\"]\n"
                "    return float(_g), total\n"),
    },
    {
        "name": "C_genome_literal_into_active_bytes",
        "target": {"kind": "canary", "canary": "C_genome_only_change"},
        "defect": ("scale the newly-derived ARTIFACT_PHYSICAL active bytes by a genome "
                   "literal. On an unmutated artifact the factor is 1.0, so this defect "
                   "is INVISIBLE to canaries A, B and D -- only C can see it"),
        "old": "    active = disk_total - embed\n",
        "new": ("    active = int(disk_total * m[\"genome\"][\"mlp\"]"
                "[\"gemv_storage_bpw\"] / 2.25) - embed\n"),
    },
    {
        "name": "A_accountant_stops_counting_a_codec",
        "target": {"kind": "canary", "canary": "A_add_bytes"},
        "defect": "drop .f32v2 from the payload suffix set, so bytes really on disk "
                  "stop reaching the physical total",
        "old": "        if f.is_file() and f.suffix in PAYLOAD_SUFFIXES:\n",
        "new": ("        if f.is_file() and f.suffix in tuple(\n"
                "                s for s in PAYLOAD_SUFFIXES if s != \".f32v2\"):\n"),
    },
    {
        "name": "D_design_crosscheck_silently_tracks_physical",
        "target": {"kind": "canary", "canary": "D_artifact_only_change"},
        "defect": "make the DESIGN cross-check recompute itself from bytes, so a "
                  "design/physical divergence can never surface",
        "old": "    return json.load(open(m)).get(\"complete_ebpw\")\n",
        "new": "    return physical_ebpw(root)[0]\n",
    },
    {
        "name": "NS_design_constant_republished_under_a_physical_name",
        "target": {"kind": "audit"},
        "defect": "publish the DESIGN_EXPECTED dram figure under the bare physical name "
                  "'dram_bytes_per_token' again",
        "old": "        cls(\"DESIGN_EXPECTED_dram_bytes_per_token\", "
               "\"DESIGN_EXPECTED_dram_bytes_per_token\",\n",
        "new": "        cls(\"dram_bytes_per_token\", "
               "\"DESIGN_EXPECTED_dram_bytes_per_token\",\n",
    },
]


def _run_variant(src_text, tag, scratch, work_root):
    """Run a (possibly patched) copy of this file and return its audit + canary output."""
    p = scratch / f"metric_audit_{tag}.py"
    p.write_text(src_text)
    a_out = scratch / f"audit_{tag}.json"
    c_out = scratch / f"canaries_{tag}.json"
    r = subprocess.run(
        [sys.executable, str(p), "--emit-audit", str(a_out), "--emit-canaries", str(c_out),
         "--work", str(work_root) + "_" + tag],
        capture_output=True, text=True,
        env=dict(os.environ, HAWKING_REPO=str(REPO)))
    return {
        "exit_code": r.returncode,
        "stdout": r.stdout.strip().splitlines(),
        "stderr_tail": r.stderr.strip().splitlines()[-5:],
        "audit": json.loads(a_out.read_text()) if a_out.is_file() else None,
        "canaries": json.loads(c_out.read_text()) if c_out.is_file() else None,
    }


def negative_controls(work_root):
    src_path = Path(__file__).resolve()
    src = src_path.read_text()
    scratch = Path(tempfile.mkdtemp(prefix="metric_audit_negctl_"))
    results = []
    try:
        for nc in NEG_CONTROLS:
            n = src.count(nc["old"])
            if n != 1:
                raise SystemExit(
                    f"negative control {nc['name']}: anchor matched {n} times, expected "
                    f"exactly 1. The control would patch nothing or patch too much, "
                    f"which is the vacuous-check failure mode this file exists to catch.")
            run = _run_variant(src.replace(nc["old"], nc["new"], 1), nc["name"],
                               scratch, work_root)
            row = {"name": nc["name"], "defect_injected": nc["defect"],
                   "patch": {"file": "tools/odyssey/metric_audit.py",
                             "old": nc["old"], "new": nc["new"]},
                   "target": nc["target"], "exit_code": run["exit_code"],
                   "observed_stdout": run["stdout"]}
            if nc["target"]["kind"] == "canary":
                pref = nc["target"]["canary"]
                cans = (run["canaries"] or {}).get("canaries", [])
                hit = next((c for c in cans if c["canary"].startswith(pref)), None)
                row["observed_canary_row"] = hit
                row["target_canary_passed"] = hit and hit["passed"]
                row["watched_failing"] = bool(hit and hit["passed"] is False
                                              and run["exit_code"] != 0)
                row["other_canaries_still_passing"] = [
                    c["canary"] for c in cans
                    if not c["canary"].startswith(pref) and c["passed"]]
                row["clone_restored_under_the_defect"] = (
                    (run["canaries"] or {}).get("restore_check", {})
                    .get("restored_to_baseline"))
            else:
                a = run["audit"] or {}
                row["observed_still_frozen_and_flagged"] = a.get("still_frozen_and_flagged")
                row["observed_audit_pass"] = a.get("pass")
                row["watched_failing"] = bool(a.get("still_frozen_and_flagged")
                                              and a.get("pass") is False
                                              and run["exit_code"] != 0)
            results.append(row)
        clean = _run_variant(src, "restored_unpatched", scratch, work_root)
        restored = {
            "exit_code": clean["exit_code"],
            "observed_stdout": clean["stdout"],
            "all_canaries_pass": (clean["canaries"] or {}).get("pass"),
            "still_frozen_and_flagged": (clean["audit"] or {}).get("still_frozen_and_flagged"),
            "audit_pass": (clean["audit"] or {}).get("pass"),
            "clone_restored_to_baseline": (
                (clean["canaries"] or {}).get("restore_check", {})
                .get("restored_to_baseline")),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        for nc in NEG_CONTROLS + [{"name": "restored_unpatched"}]:
            shutil.rmtree(str(work_root) + "_" + nc["name"], ignore_errors=True)
    return results, restored


def emit_namespace_receipt(path, audit, controls, restored):
    out = {
        "schema": "hawking.odyssey.ebpw_namespace_separation.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/metric_audit.py --emit-negative-control",
        "obligation": "G059 — EBPW namespace separation",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "claim": ("every mutable headline metric is published under the namespace its "
                  "VALUE comes from, and the checks that enforce that have each been "
                  "watched failing on an injected defect"),
        "still_frozen_and_flagged": audit["still_frozen_and_flagged"],
        "frozen_rule": audit["frozen_rule"],
        "metrics_moved_this_lane": {
            "active_ebpw_per_token": {
                "was": "DESIGN_EXPECTED_active_ebpw_per_token published as "
                       "'active_ebpw_per_token'",
                "now": "ARTIFACT_PHYSICAL_active_ebpw_per_token",
                "derivation": "8 * (payload_bytes_on_disk - embedding_bytes) / parent_params",
                "design_figure_retained_as": "DESIGN_EXPECTED_active_ebpw_per_token",
            },
            "active_bytes_per_token": {
                "was": "DESIGN_EXPECTED published as 'active_bytes_per_token'",
                "now": "ARTIFACT_PHYSICAL_active_bytes_per_token",
                "derivation": "payload_bytes_on_disk - payload_bytes_by_role['embedding']",
            },
            "dram_bytes_per_token": {
                "was": "ARTIFACT_PHYSICAL_dram_bytes_per_token (perf_addendum.py) and "
                       "'dram_bytes_per_token' (metric_audit.py)",
                "now": "DESIGN_EXPECTED_dram_bytes_per_token",
                "renamed_not_measured": True,
                "blocking_gate": audit["observed_values"]["dram_gate"],
            },
        },
        "frozen_design_evidence": audit["observed_values"]["frozen_design_evidence"],
        "negative_control": controls,
        "restored": restored,
        "pass": bool(all(c["watched_failing"] for c in controls)
                     and not audit["still_frozen_and_flagged"]
                     and restored["all_canaries_pass"] is True
                     and restored["audit_pass"] is True
                     and restored["clone_restored_to_baseline"] is True),
        "claim_boundary": [
            "This does NOT measure DRAM traffic. No userspace DRAM counter is reachable "
            "on this box (powermetrics requires root), so dram_bytes_per_token is a "
            "modelled upper bound that was RENAMED into DESIGN_EXPECTED_*, not measured.",
            "This does NOT prove the assumed-zero inter-token reuse model is correct. It "
            "proves only that the assumption is now labelled as an assumption.",
            "The ACTIVE figures are anchored on bytes on disk, but the organ ROLE split "
            "is read from MIX_REPORT.payload_bytes_by_role. If the packer mislabels a "
            "role, the split is wrong and only the reconciliation flag would show it.",
            "The 'one whole read per organ per token' rule behind active bytes is a "
            "dense-body assumption. It is not verified against runtime memory traffic.",
            "Canary E still does not INDUCE a runtime fallback; it verifies the channel "
            "exists. That limit is unchanged by this lane.",
            "RUNTIME_MEASURED_model_reachable_roof is measured TIME over MODELLED bytes. "
            "Its numerator is the design bound above, so it inherits that assumption. It "
            "is flagged in the audit rather than renamed, because renaming it would break "
            "tools/odyssey/pareto_archive.py, which is outside this lane's scope.",
            "Only PHYSICAL_METRIC_AUDIT / PHYSICAL_METRIC_CANARIES were regenerated. "
            "receipts/headless/QWEN_PERFORMANCE_ADDENDUM.json still carries the old "
            "ARTIFACT_PHYSICAL_dram_bytes_per_token key; the source that writes it is "
            "fixed, but that receipt is outside this lane's permitted scope and must be "
            "regenerated by re-running perf_addendum.py.",
            "Negative controls patch a COPY of metric_audit.py in a temp directory. They "
            "prove the CHECKS bite; they do not prove the checks are exhaustive.",
        ],
    }
    Path(path).write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-audit", required=True)
    ap.add_argument("--emit-canaries", required=True)
    ap.add_argument("--emit-negative-control",
                    help="also inject each defect into a copy of this file and require "
                         "the matching check to FAIL; writes the namespace receipt")
    ap.add_argument("--work", default="/Users/scammermike/noetic/CANARY_CLONE")
    a = ap.parse_args()

    audit = audit_metrics()
    audit_pass = bool(audit["n_metrics"] >= 12 and audit["n_artifact_physical"] >= 3
                      and audit["n_runtime_measured"] >= 3
                      and not audit["still_frozen_and_flagged"]
                      and not audit["design_metrics_without_gate_or_measured_twin"])
    Path(a.emit_audit).write_text(json.dumps({
        "schema": "hawking.headless.physical_metric_audit.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/metric_audit.py",
        "obligation": "G033 — PHYSICAL_METRIC_AUDIT (steer S011 §5); G059 namespaces",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False, **audit,
        "pass": audit_pass,
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
    print(f"audit pass={audit_pass}")

    nc_pass = True
    if a.emit_negative_control:
        controls, restored = negative_controls(a.work + "_NC")
        rec = emit_namespace_receipt(a.emit_negative_control, audit, controls, restored)
        for c in controls:
            print(f"  {'WATCHED-FAILING' if c['watched_failing'] else 'DID-NOT-BITE'}  "
                  f"{c['name']}")
        print(f"negative controls {sum(c['watched_failing'] for c in controls)}"
              f"/{len(controls)}  restored_clean_run_pass={restored['all_canaries_pass']}")
        print(f"-> {a.emit_negative_control}  pass={rec['pass']}")
        nc_pass = rec["pass"]

    return 0 if (out["pass"] and audit_pass and nc_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
