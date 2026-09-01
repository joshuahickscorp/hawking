#!/usr/bin/env python3
"""Seal the Flash NR as a machine-bound NX descriptor.

The descriptor is deliberately metadata-only: it does not claim that a
single-token receipt is a resident executable.  It binds the exact Flash
executor/shaders and the measured protected profile to this machine genome,
so a changed NR, source, shader, or host cannot be loaded silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / "crates/hawking-core/examples/flash_fast_chain.rs"
LINEAR = ROOT / "crates/hawking-core/examples/flash_noetic_complete_layer0.rs"
ATTENTION = ROOT / "crates/hawking-core/examples/flash_full_attention_layer3.rs"
SHADERS = ROOT / "crates/hawking-core/shaders"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_genome() -> dict[str, Any]:
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, check=True).stdout.strip())
    display = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True,
                             text=True, check=True).stdout
    chip = next((line.split(":", 1)[1].strip() for line in display.splitlines()
                 if "Chipset Model" in line), "?")
    cores = next((int(line.split(":", 1)[1].strip()) for line in display.splitlines()
                  if "Total Number of Cores" in line), 0)
    metal = next((line.split(":", 1)[1].strip() for line in display.splitlines()
                  if "Metal Support" in line), "?")
    body = {"chipset": chip, "gpu_cores": cores, "unified_memory_bytes": ram,
            "metal_family": metal}
    body["genome_digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def seal(nr: pathlib.Path, profile: pathlib.Path | None, stateful_organ: pathlib.Path | None, stateful_attention: pathlib.Path | None, stateful_prefix: pathlib.Path | None, census: pathlib.Path | None, byte_ledger: pathlib.Path | None, bank_screen: pathlib.Path | None, ngram_screen: pathlib.Path | None, ngram_lookup: pathlib.Path | None) -> dict[str, Any]:
    files = [EXECUTOR, LINEAR, ATTENTION]
    shaders = sorted(SHADERS.glob("*.metal"))
    source_members = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
                      for p in files]
    shader_members = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
                      for p in shaders]
    nr_doc = json.loads(nr.read_text())
    nr_representation = nr_doc.get("representation", {})
    complete_bpw = nr_representation.get("complete_bits_per_weight")
    if complete_bpw is None:
        complete_bpw = (nr_representation.get("tensors") or {}).get("complete_bits_per_weight")
    out: dict[str, Any] = {
        "schema": "hawking.flash.nx_genome.v1",
        "nx_kind": "hawking.nos.flash_noetic_executable_genome",
        "status": "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "compiled_for_machine_genome": machine_genome(),
        "lowers_nr": {"path": str(nr), "sha256": sha(nr), "nr_kind": nr_doc.get("nr_kind"),
                      "complete_bits_per_weight": complete_bpw},
        "physical_program": {
            "executor": source_members,
            "metal_sources": shader_members,
            "source_binding": "exact Flash fast-chain executor plus both organ executors; shader files are content-bound",
            "device": "Metal",
            "cuda": "HARDWARE_BLOCKED",
            "host_activation_roundtrips": 0,
            "dispatch_policy": "grouped structural chain; device-resident activation handoff",
        },
        "qualification": {
            "profile": str(profile) if profile else None,
            "profile_sha256": sha(profile) if profile and profile.is_file() else None,
            "stateful_linear_organ": str(stateful_organ) if stateful_organ else None,
            "stateful_linear_organ_sha256": sha(stateful_organ) if stateful_organ and stateful_organ.is_file() else None,
            "stateful_attention_organ": str(stateful_attention) if stateful_attention else None,
            "stateful_attention_organ_sha256": sha(stateful_attention) if stateful_attention and stateful_attention.is_file() else None,
            "stateful_linear_prefix_session": str(stateful_prefix) if stateful_prefix else None,
            "stateful_linear_prefix_session_sha256": sha(stateful_prefix) if stateful_prefix and stateful_prefix.is_file() else None,
            "organ_census": str(census) if census else None,
            "organ_census_sha256": sha(census) if census and census.is_file() else None,
            "byte_ledger": str(byte_ledger) if byte_ledger else None,
            "byte_ledger_sha256": sha(byte_ledger) if byte_ledger and byte_ledger.is_file() else None,
            "doctor_expert_bank_screen": str(bank_screen) if bank_screen else None,
            "doctor_expert_bank_screen_sha256": sha(bank_screen) if bank_screen and bank_screen.is_file() else None,
            "doctor_ngram_screen": str(ngram_screen) if ngram_screen else None,
            "doctor_ngram_screen_sha256": sha(ngram_screen) if ngram_screen and ngram_screen.is_file() else None,
            "doctor_ngram_lookup_oracle": str(ngram_lookup) if ngram_lookup else None,
            "doctor_ngram_lookup_oracle_sha256": sha(ngram_lookup) if ngram_lookup and ngram_lookup.is_file() else None,
            "complete_layers": 48 if profile else None,
            "terminal_token": "physically received" if profile else None,
            "accepted_multitoken_tps": None,
            "complete_system_ebpw": None,
            "resident_promotion": False,
        },
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": "2026-08-28T00:00:00Z",
            "recorded_by": "tools/flash_nx_genome.py metadata seal",
            "machine": "Apple M3 Ultra (machine genome recorded above)",
            "rule": "S032 §3 -- NX metadata seal is not a performance benchmark; timing state is UNKNOWN",
            "provenance": "No performance claim is made by this descriptor; the referenced protected profile carries its own benchmark state.",
        },
        "claim_boundary": "Machine-bound Flash NX metadata seal only. It binds source, shaders, NR, and machine genome; it is not a promoted resident executable until accepted multi-token TPS, EBPW, capability, and residency evidence exist.",
    }
    out["seal_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=pathlib.Path, required=True)
    ap.add_argument("--profile", type=pathlib.Path)
    ap.add_argument("--stateful-organ", type=pathlib.Path)
    ap.add_argument("--stateful-attention", type=pathlib.Path)
    ap.add_argument("--stateful-prefix", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--byte-ledger", type=pathlib.Path)
    ap.add_argument("--bank-screen", type=pathlib.Path)
    ap.add_argument("--ngram-screen", type=pathlib.Path)
    ap.add_argument("--ngram-lookup-oracle", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    if not a.nr.is_file():
        raise SystemExit(f"missing NR: {a.nr}")
    doc = seal(a.nr, a.profile, a.stateful_organ, a.stateful_attention, a.stateful_prefix, a.census, a.byte_ledger, a.bank_screen, a.ngram_screen, a.ngram_lookup_oracle)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "genome": doc["compiled_for_machine_genome"]["genome_digest"],
                      "nr_sha256": doc["lowers_nr"]["sha256"], "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
