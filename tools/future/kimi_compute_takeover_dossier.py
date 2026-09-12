#!/usr/bin/env python3.12
"""Build the first machine-wide compute dossier for KIMI physical gravity.

This composes Hawking's existing MachineGenome, ANE probe, GPU attribution,
and resource-guard owners.  It adds only small CPU workload probes so the
result compares domains without pretending that a microbenchmark is end-to-end
KIMI TPS.  CorpDrive access remains read-only through MachineGenome.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.accelerator import machine_genome  # noqa: E402
from tools.future import gpu_attribution  # noqa: E402
from tools.future.campaign_memory_guard import require_ok, sample  # noqa: E402

SCHEMA = "hawking.future.kimi_compute_takeover_dossier.v1"


def _cpu_probe(elements: int = 8 << 20, sort_elements: int = 1 << 20,
               reps: int = 5) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(20260910)
    a = rng.standard_normal(elements, dtype=np.float32)
    b = rng.standard_normal(elements, dtype=np.float32)
    c = np.empty_like(a)
    np.add(a, b, out=c)
    add_samples = []
    for _ in range(reps):
        started = time.perf_counter()
        np.add(a, b, out=c)
        add_samples.append((time.perf_counter() - started) * 1000.0)

    values = rng.integers(0, 2**31 - 1, size=sort_elements, dtype=np.int32)
    sort_samples = []
    for _ in range(reps):
        work = values.copy()
        started = time.perf_counter()
        work.sort(kind="quicksort")
        sort_samples.append((time.perf_counter() - started) * 1000.0)

    bytes_moved = elements * 3 * 4
    return {
        "evidence_tier": "HARDWARE_MEASURED",
        "vector_add_f32": {
            "elements": elements,
            "bytes_moved_per_rep": bytes_moved,
            "reps": reps,
            "median_ms": statistics.median(add_samples),
            "samples_ms": add_samples,
            "median_gb_s": bytes_moved / max(statistics.median(add_samples), 1e-9) / 1e6,
        },
        "sort_int32": {
            "elements": sort_elements,
            "reps": reps,
            "median_ms": statistics.median(sort_samples),
            "samples_ms": sort_samples,
        },
        "claim_boundary": "Small CPU primitive measurements only; not a KIMI TPS result and not a CPU roofline.",
    }


def _load_if_present(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build(output: Path) -> dict[str, Any]:
    guard_entry = require_ok("KIMI machine-wide compute takeover dossier", expected_gb=1.0).as_dict()
    cpu = _cpu_probe()
    cpu_end = sample(expected_gb=1.0).as_dict()

    genome = machine_genome.build(
        contended=True,
        contention_note="live HCLI daemon, ModelLake watcher, and local KIMI derivative server; bounded characterization only",
    )
    attribution = gpu_attribution.attribute("KIMI P0 / physical-gravity machine takeover characterization")

    tps_path = ROOT / "receipts/future/KIMI_CANONICAL_TPS_PARETO_RAW_VS_INT8_ROWRELATIVE_20260910.json"
    p0_path = ROOT / "receipts/future/KIMI_P0_DENSITY_FREEZE_20260910.json"
    tps = _load_if_present(tps_path)
    p0 = _load_if_present(p0_path)

    out = {
        "schema": SCHEMA,
        "status": "MACHINE_WIDE_COMPUTE_CHARACTERIZATION_COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "genome": genome,
        },
        "resource_guard": {"entry": guard_entry, "cpu_probe_end": cpu_end},
        "cpu_workload_probes": cpu,
        "gpu_attribution": attribution,
        "kimi_physical_anchor": {
            "canonical_tps_receipt": str(tps_path),
            "canonical_tps_status": (tps or {}).get("status", "MISSING"),
            "p0_freeze_receipt": str(p0_path),
            "p0_freeze_status": (p0 or {}).get("status", "MISSING"),
            "raw_steady_decode_tps": ((tps or {}).get("timing", {}).get("baseline", {})
                                       .get("contexts", [{}])[0].get("summary", {})
                                       .get("steady_decode_tps_median")),
        },
        "domains": {
            "cpu": "HARDWARE_MEASURED primitive probes plus MachineGenome identity",
            "gpu": "MachineGenome bounded triad probe; KIMI TPS remains separate canonical evidence",
            "ane": "MachineGenome supported-interface/presence characterization only; no ANE KIMI workload claim",
            "unified_memory": "MachineGenome capacity/pressure observations and resource guard",
            "storage": "MachineGenome bounded read-only CorpDrive and local mount characterization",
        },
        "summary": {
            "cpu_vector_add_median_gb_s": cpu["vector_add_f32"]["median_gb_s"],
            "gpu_triad_median_gb_s": (genome.get("measured_bandwidth") or {}).get("median_gb_s"),
            "gpu_triad_iqr_spread_pct": (genome.get("measured_bandwidth") or {}).get("iqr_spread_pct"),
            "ane_maturity": (genome.get("backend_maturity") or {}).get("ane_0"),
            "cpu_maturity": (genome.get("backend_maturity") or {}).get("cpu_0"),
            "storage_maturity": (genome.get("backend_maturity") or {}).get("storage"),
            "uma_maturity": (genome.get("backend_maturity") or {}).get("uma_0"),
            "gpu_attribution_hawking_processes": len(attribution.get("hawking_processes", [])),
            "gpu_attribution_foreign_model_servers": len(attribution.get("foreign_model_servers", [])),
        },
        "next_work": [
            "run matched CPU/GPU KIMI decode placement experiments under the canonical protocol",
            "attribute token latency into weights, routing, representation decode, attention/state, dispatch, and sampling",
            "extend supported Core ML/ANE organ probes only where a complete-token workload can be measured",
            "keep 100 and 500 TPS claims closed until end-to-end sustained decode receipts pass",
        ],
        "claim_boundary": "This dossier begins machine-wide characterization. It does not claim an SoC roof, ANE runtime throughput, KIMI CPU/GPU superiority, 100 TPS, 500 TPS, or promotion.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "receipts/future/KIMI_COMPUTE_TAKEOVER_DOSSIER_20260910.json")
    args = ap.parse_args()
    out = build(args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": out["status"],
        "cpu_vector_add": out["cpu_workload_probes"]["vector_add_f32"],
        "gpu_bandwidth": out["machine"]["genome"].get("measured_bandwidth"),
        "ane_maturity": out["machine"]["genome"]["backend_maturity"].get("ane_0"),
        "claim_boundary": out["claim_boundary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
