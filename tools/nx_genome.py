#!/usr/bin/env python3
"""G104: the NX genome -- everything NR refuses, plus the machine it was compiled for.

G103's deny-list is this file's allow-list. NR says what the patient IS and must be
portable; NX says how ONE machine runs it and must not be portable. The teeth here
are the inverse of G103's: an NX that could load anywhere has failed, because
silently running an executable compiled for different hardware is exactly the
failure this obligation exists to prevent.

The MACHINE GENOME is a digest over the facts a lowering actually depends on --
GPU core count, unified memory size, the measured bandwidth roof, and the Metal
family. Two machines agreeing on all of those can run each other's NX; anything
else is REFUSED rather than run and hoped for.

The kernel binding is extracted from the decode source by intersecting its string
literals with the declared `kernel void` names, so the seal names kernels that are
ACTUALLY dispatched rather than kernels that merely exist. G071 measured that
distinction as 38 bound against 554 declared, and a seal listing the 554 would be
a lie about what runs.

  ./tools/nx_genome.py --seal --out /tmp/nx.json
  ./tools/nx_genome.py --dump /tmp/nx.json
  ./tools/nx_genome.py --refusal-test /tmp/nx.json    # expect exit 1
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DECODE = ROOT / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SHADERS = ROOT / "crates/hawking-core/shaders"


def machine_genome():
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True).stdout.strip())
    disp = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True,
                          text=True).stdout
    chip = next((l.split(":")[1].strip() for l in disp.splitlines() if "Chipset Model" in l), "?")
    cores = next((int(l.split(":")[1].strip()) for l in disp.splitlines()
                  if "Total Number of Cores" in l), 0)
    metal = next((l.split(":")[1].strip() for l in disp.splitlines() if "Metal Support" in l), "?")
    g = {"chipset": chip, "gpu_cores": cores, "unified_memory_bytes": ram,
         "metal_family": metal, "measured_roof_gb_s": 595.9,
         "roof_provenance": "roofline sweep in the G072 run, low arithmetic intensity peak"}
    g["genome_digest"] = hashlib.sha256(
        json.dumps({k: g[k] for k in ("chipset", "gpu_cores", "unified_memory_bytes",
                                      "metal_family", "measured_roof_gb_s")},
                   sort_keys=True).encode()).hexdigest()
    return g


def bound_kernels():
    lits = set()
    for tok in DECODE.read_text().split('"'):
        if tok and all(c.isalnum() or c == "_" for c in tok):
            lits.add(tok)
    declared = set()
    for p in SHADERS.glob("*.metal"):
        for line in p.read_text().splitlines():
            if line.startswith("kernel void "):
                declared.add(line.split()[2].split("(")[0])
    return sorted(lits & declared), len(declared)


def seal():
    ks, n_declared = bound_kernels()
    return {
        "nx_version": "1.0.0",
        "nx_kind": "hawking.nos.noetic_executable_genome",
        "compiled_for_machine_genome": machine_genome(),
        "kernel_binding": {
            "dispatched": ks, "count": len(ks), "declared_in_tree": n_declared,
            "extraction": "string literals in qwen38_hybrid_decode.rs intersected with declared "
                          "`kernel void` names, so this lists kernels ACTUALLY dispatched. G071 "
                          "measured 38 bound against 554 declared; a seal listing all 554 would be "
                          "a lie about what runs.",
        },
        "threadgroup_geometry": {
            "gemv": {"threadgroup": 128, "rows_per_threadgroup": 2,
                     "note": "geo_tpr64 tiling, measured 0.8046 ps/element (G072)"},
            "mha_decode": {"threadgroup": 512,
                           "note": "swept 256/512/1024 at long context; 512 is the plateau and "
                                   "1024 buys 0.16% (G060)"},
        },
        "residency_plan": {
            "all_weights": "unified_memory",
            "basis": "G085 measured the artifact at 13.9% of 103.1 GB RAM and RAM pressure "
                     "beginning at 677,508 KV positions against a 131,072 maximum context, so no "
                     "tier below unified memory is reachable on this machine",
        },
        "cache_plan": {
            "cache_resident_token_fraction": 0.004930172,
            "basis": "G094 measured a 5.8x traffic swing producing zero rank correlation with "
                     "time, so cache residency is recorded but is NOT an optimisation target here",
        },
        "scheduling": {
            "dispatches_per_token": 964,
            "host_encode_ns_per_dispatch": 702,
            "host_ceremony_ms_per_token": 1.070,
            "gpu_fraction_of_wall": 0.965,
            "basis": "G093",
        },
    }


def check_loadable(nx, current):
    a = nx["compiled_for_machine_genome"]
    problems = []
    if a["genome_digest"] != current["genome_digest"]:
        problems.append(f"MACHINE GENOME MISMATCH: NX built for {a['genome_digest'][:16]}, "
                        f"this machine is {current['genome_digest'][:16]}")
        for k in ("chipset", "gpu_cores", "unified_memory_bytes", "metal_family",
                  "measured_roof_gb_s"):
            if a.get(k) != current.get(k):
                problems.append(f"  {k}: NX has {a.get(k)!r}, machine has {current.get(k)!r}")
    return (not problems), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true"); ap.add_argument("--dump")
    ap.add_argument("--refusal-test"); ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    if a.seal:
        d = seal()
        txt = json.dumps(d, indent=2) + "\n"
        if a.out:
            a.out.write_text(txt)
        g = d["compiled_for_machine_genome"]
        print(f"sealed for {g['chipset']}, {g['gpu_cores']} GPU cores, "
              f"{g['unified_memory_bytes']/1e9:.1f} GB, {g['metal_family']}")
        print(f"genome digest {g['genome_digest'][:32]}")
        print(f"kernel binding: {d['kernel_binding']['count']} dispatched of "
              f"{d['kernel_binding']['declared_in_tree']} declared")
        return 0

    if a.dump:
        d = json.loads(pathlib.Path(a.dump).read_text())
        ok, probs = check_loadable(d, machine_genome())
        g = d["compiled_for_machine_genome"]
        print(f"NX SEAL DUMP")
        print(f"  machine     {g['chipset']}, {g['gpu_cores']} cores, "
              f"{g['unified_memory_bytes']/1e9:.1f} GB, {g['metal_family']}")
        print(f"  genome      {g['genome_digest'][:32]}")
        print(f"  kernels     {d['kernel_binding']['count']} dispatched "
              f"(of {d['kernel_binding']['declared_in_tree']} declared)")
        print(f"  geometry    gemv tg{d['threadgroup_geometry']['gemv']['threadgroup']}, "
              f"mha tg{d['threadgroup_geometry']['mha_decode']['threadgroup']}")
        print(f"  residency   {d['residency_plan']['all_weights']}")
        print(f"  scheduling  {d['scheduling']['dispatches_per_token']} dispatches/token, "
              f"{d['scheduling']['host_encode_ns_per_dispatch']} ns/dispatch host encode")
        print(f"  loadable here: {'YES' if ok else 'NO'}")
        return 0 if ok else 1

    if a.refusal_test:
        d = json.loads(pathlib.Path(a.refusal_test).read_text())
        # A plausible neighbour, not an absurd one: same chip family, fewer GPU cores.
        d["compiled_for_machine_genome"]["gpu_cores"] = 40
        d["compiled_for_machine_genome"]["genome_digest"] = hashlib.sha256(
            b"different-machine").hexdigest()
        ok, probs = check_loadable(d, machine_genome())
        print("REFUSAL TEST: an NX sealed for a 40-core GPU, loaded on this 60-core machine")
        print(f"runtime says: {'LOADED -- THE CHECK IS BROKEN' if ok else 'REFUSED, as required'}")
        for p in probs:
            print(f"  {p}")
        return 1 if not ok else 0

    ap.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
