#!/usr/bin/env python3
"""G023 unblock step 1: register the MoE expert kernels that already exist.

The pipeline's KernelPlanner stage was recorded AUTOMATIC for model #2 while its library
held no kernel for any MoE organ. That was not because the kernels were missing -- 18 of
them are declared in crates/hawking-core/shaders/qwen30_device_expert_table.metal and
referenced by name from the routed runtime -- but because nobody catalogued them.

They are registered here with MEASURED fields where a fact exists on disk and ABSENT
fields, each carrying a reason, where one does not. The library's law is that a field may
be absent with a reason but never blank, and these kernels are genuinely unmeasured: no
model #2 artifact exists to run them against, so claiming a measurement or a parity result
would be the forged number this campaign keeps catching.
"""
import hashlib, json, re, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
KL = RH / "KERNEL_LIBRARY.json"
SHADER = REPO / "crates/hawking-core/shaders/qwen30_device_expert_table.metal"
RUNTIME = REPO / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"


def absent(reason):
    return {"kind": "ABSENT", "value": None, "absent_reason": reason}


def measured(v, note=None, source=None):
    d = {"kind": "MEASURED", "value": v}
    if note:
        d["note"] = note
    if source:
        d["source"] = source
    return d


def representation_of(name):
    if "binary" in name:
        return "binary_group"
    if "hgravs" in name:
        return "hgravs01_factored"
    if "uniform_q4" in name:
        return "uniform_q4_group"
    return "unknown"


def specialization_of(name):
    s = {"serial": "serial" in name, "simdgroup": "simdgroup" in name}
    m = re.search(r"rowblock(\d+)", name)
    s["rowblock"] = int(m.group(1)) if m else (1 if "rowblock" in name else None)
    s["fused_gate_up_swiglu"] = "paired_gate_up_swiglu" in name
    return s


def main():
    lib = json.load(open(KL))
    existing = {k["kernel_identity"] for k in lib["kernels"]}

    shader_src = SHADER.read_text()
    declared = sorted(set(re.findall(r"kernel void ([a-z0-9_]+)", shader_src)))
    referenced = sorted(set(re.findall(r'"(qwen30_expert_table_[a-z0-9_]+)"',
                                       RUNTIME.read_text())))
    both = [n for n in declared if n in referenced]
    sha = hashlib.sha256(SHADER.read_bytes()).hexdigest()
    machine = lib["kernels"][0]["machine_identity"]      # same box, cited not re-derived

    added = []
    for name in both:
        if name in existing:
            continue
        fused = "paired_gate_up_swiglu" in name
        entry = {
            "kernel_identity": name,
            "organ_identity": "moe_expert_gate_up" if fused else "moe_expert",
            "representation_identity": representation_of(name),
            "machine_identity": machine,
            "why_qualified": "declared in the expert-table shader AND referenced by name "
                             "from the routed runtime; both checked, not assumed",
            "shader": str(SHADER.relative_to(REPO)),
            "shader_sha256": measured(sha, note="sha256 of the shader SOURCE; the runtime "
                                                "compiles from source at admission",
                                      source=str(SHADER.relative_to(REPO))),
            "compiled_identity": absent(
                "the routed runtime compiles this shader from source at admission and "
                "does not emit a metallib, so there is no compiled artifact to hash"),
            "specialization": measured(specialization_of(name),
                                       note="read from the kernel name and confirmed "
                                            "against the shader declaration"),
            "memory_layout": measured(
                {"dispatch": "expert-table indexed", "operand": "per-expert weight rows "
                 "selected by device-produced route ids"},
                note="the runtime reads eight device-produced route ids per token and "
                     "never evaluates the router on the host",
                source=str(RUNTIME.relative_to(REPO))),
            "competence": measured(
                {"executes": representation_of(name),
                 "organ": "moe_expert_gate_up" if fused else "moe_expert"},
                note="competence is a claim about WHICH representation this kernel can "
                     "execute, not about how fast it does so"),
            "measurements": absent(
                "never benchmarked in this campaign. No throughput, occupancy or "
                "bandwidth figure exists for these kernels here."),
            "supported_capability_regime": absent(
                "unknown: no model #2 noetic artifact exists, so these kernels have "
                "never executed a whole-model decode in this campaign"),
            "parity": absent(
                "no parity oracle was run for these kernels here. The qwen38 kernels "
                "were graded against a numpy oracle; the equivalent has not been done "
                "for the expert-table family."),
            "citations": measured(
                [str(RUNTIME.relative_to(REPO)), str(SHADER.relative_to(REPO)),
                 "receipts/headless/NOETIC_COMPILER_STAGE_AUDIT.json"]),
        }
        added.append(entry)

    lib["kernels"].extend(added)
    lib["n_kernels"] = len(lib["kernels"])
    lib["moe_registration"] = {
        "added": [k["kernel_identity"] for k in added],
        "n_added": len(added),
        "organs_now_covered": sorted({k["organ_identity"] for k in lib["kernels"]}),
        "why": "KernelPlanner had no kernel for any MoE organ, so the pipeline could not "
               "plan for model #2 even though the kernels existed. Registering them is "
               "the first item of the revised G023 gap.",
        "all_fields_present_or_absent_with_reason": True,
        "nothing_measured_was_invented": "measurements, supported_capability_regime and "
                                         "parity are ABSENT with reasons on every added "
                                         "kernel, because none has been run here",
        "declared_in_shader": len(declared),
        "referenced_by_runtime": len(referenced),
        "in_both": len(both),
    }
    KL.write_text(json.dumps(lib, indent=1))
    print(f"  declared in shader: {len(declared)}   referenced by runtime: "
          f"{len(referenced)}   in both: {len(both)}")
    print(f"  added to library: {len(added)}")
    print(f"  organs now covered: {lib['moe_registration']['organs_now_covered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
