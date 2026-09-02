"""Generic Hardware Doctor — reasons OVER the genome, the ascension cycle, and HWIR.

The FPGA-axis sidecar (tools/future/hardware_doctor.py) ranks falsifiable
FPGA-organ experiments. This module is the host doctor those organs sit on:

  1. what devices exist right now
  2. what is measured vs modelled vs unknown, per device, per axis
  3. which workloads fit which device, given the measured genome
  4. which experiment would most reduce uncertainty (info-gain / cost)
  5. backend maturity per device

It must give a useful answer for a device that does NOT exist yet. Absent
U50DD/DGX/eGPU degrade to UNKNOWN axes + a named wake_condition; brochure
and third-party carrier figures stay STATIC / THIRD_PARTY_REPORTED. Nothing
here fabricates a HARDWARE_MEASURED number for a board that is not attached.

A module import is not a call site. diagnose() actually invokes:

  machine_genome.discover_identity
  machine_genome.build          (via device_ascension.characterize)
  machine_genome.axes_for_domain
  machine_genome.devices_exist
  device_ascension.characterize / economics / select
  hwir.u50_family_profile
  hwir.chestnut_current_firmware
  hwir.chestnut_hawking_optimized
  tools.roadmap.hardware.probe  (and probe_u50)

    python3 tools/accelerator/hardware_doctor.py
    python3 -m pytest tools/accelerator/test_hardware_doctor.py -o addopts="" -q
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_ACCEL = Path(__file__).resolve().parent
REPO = _ACCEL.parents[1]
for _p in (str(_ACCEL), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import device_ascension as da  # noqa: E402
import machine_genome as mg  # noqa: E402
from tools.future import hwir  # noqa: E402
from tools.roadmap import hardware as hw_wake  # noqa: E402

SCHEMA = "hawking.accelerator.hardware_doctor.v1"
RECEIPT = "HARDWARE_DOCTOR_GENERIC.json"
RECORDED_BY = "tools/accelerator/hardware_doctor.py"
RECEIPTS = REPO / "receipts" / "future"

# Cost ladder. Integer, not a wall-time claim.
COST_IDENTITY = 1
COST_BOUNDED = 2
COST_LAB = 3
COST_PRIVILEGED = 4
COST_PROTECTED_WINDOW = 5
COST_SUSTAINED = 6
COST_BOARD_ARRIVAL = 8

WAKE_IDS = ("U50_PRESENT", "DGX_PRESENT", "EGPU_PRESENT", "NEW_M_SERIES_PRESENT")


class HardwareDoctorError(ValueError):
    """Generic Hardware Doctor refused a fabricated or ill-typed claim."""


def _seal(doc: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    doc["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return doc


def rank_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by runnable-now, then expected information per unit cost.

    Information proxy = info_weight (UNKNOWN-on-present=4 … already-measured=0).
    Cost proxy = COST_* ladder. Integer key: runnable first, then
    -(info * 60 // cost), then cost, then id. Same arithmetic as the FPGA-axis
    doctor; never a hardware measurement.

    Mutation point: if this key drops `cost`, an expensive board probe with
    id 'A-…' outranks a cheap CPU probe with id 'Z-…' at equal info_weight.
    """
    decorated = []
    for rec in records:
        info = int(rec.get("info_weight") or 0)
        cost = max(int(rec.get("cost") or 1), 1)
        runnable = 0 if rec.get("runnable_now") else 1
        decorated.append((rec, info, cost, runnable))

    def _key(item: tuple[Any, ...]) -> tuple[Any, ...]:
        rec, info, cost, runnable = item
        return (runnable, -(info * 60 // cost), cost, rec.get("id") or "")

    ordered = [item[0] for item in sorted(decorated, key=_key)]
    ranked = []
    for i, rec in enumerate(ordered, start=1):
        row = dict(rec)
        info = int(rec.get("info_weight") or 0)
        cost = max(int(rec.get("cost") or 1), 1)
        row["rank"] = i
        row["information_per_cost"] = {
            "info_weight": info,
            "cost": cost,
            "runnable_now": bool(rec.get("runnable_now")),
            "rule": (
                "runnable first, then info_weight / cost via integer key "
                "-(info*60//cost); never a hardware measurement"
            ),
        }
        ranked.append(row)
    return ranked


def _info_weight(
    status: str,
    *,
    device_present: bool,
    runnable_now: bool,
) -> int:
    if status in {"MEASURED", "STATIC_IDENTITY"}:
        return 0
    if runnable_now and device_present and status in {"UNKNOWN", "ABSENT"}:
        return 4
    if runnable_now and status == "UNRELIABLE":
        return 3
    if runnable_now and device_present and status == "BLOCKED":
        return 3
    if runnable_now and not device_present:
        return 1
    if not runnable_now and status in {"UNKNOWN", "ABSENT", "BLOCKED"}:
        return 2
    return 1


def _axis_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["axis"]: r for r in rows if isinstance(r, dict) and r.get("axis")}


def _brochure(profile: Any) -> dict[str, Any]:
    """STATIC vendor literature, slim and labelled. Not a local census."""
    hwir.assert_variant_provenance(profile)
    doc = profile.to_dict()
    hwir.assert_no_hardware_measured(doc)
    if doc.get("evidence_tier") == "HARDWARE_MEASURED":
        raise HardwareDoctorError("u50 family profile must not claim HARDWARE_MEASURED")
    prov = doc.get("field_provenance") or {}
    resources: dict[str, Any] = {}
    unpinned: list[dict[str, Any]] = []
    for name in hwir.U50_VARIANT_REQUIRED_FIELDS:
        meta = dict(prov.get(name) or {})
        if meta.get("pinned"):
            resources[name] = {
                "value": meta.get("value"),
                "evidence_tier": meta.get("evidence_tier") or "STATIC",
                "document_class": meta.get("document_class"),
                "citation": meta.get("citation"),
                "hardware_measured": False,
                "vendor_literature_not_measurement": True,
            }
        else:
            unpinned.append({
                "field": name,
                "value": hwir.UNPINNED,
                "reason": meta.get("note") or meta.get("reason") or "UNPINNED",
                "evidence_tier": "STATIC",
                "hardware_measured": False,
            })
    return {
        "device_id": doc.get("device_id"),
        "sku": doc.get("sku"),
        "variant_id": doc.get("variant_id"),
        "origin": doc.get("origin"),
        "evidence_tier": doc.get("evidence_tier") or "STATIC",
        "declared_not_measured": True,
        "hardware_measured": False,
        "vendor_literature": doc.get("vendor_literature"),
        "resources": resources,
        "unpinned_fields": unpinned,
        "claim_boundary": (
            "STATIC vendor literature from hwir.u50_family_profile('u50dd'). "
            "Not a local board census. Never HARDWARE_MEASURED."
        ),
    }


def _carrier_view(env: Any, *, evidence_class: str) -> dict[str, Any]:
    doc = env.to_dict()
    return {
        "carrier_id": doc.get("carrier_id"),
        "origin": doc.get("origin"),
        "evidence_tier": doc.get("evidence_tier") or "STATIC",
        "evidence_class": evidence_class,
        "hardware_measured": False,
        "pcie_generation": doc.get("pcie_generation"),
        "pcie_lanes": doc.get("pcie_lanes"),
        "observed_payload_bytes_per_s": doc.get("observed_payload_bytes_per_s"),
        "airflow_class": doc.get("airflow_class"),
        "sustained_power_w": doc.get("sustained_power_w"),
        "note": doc.get("note"),
        "claim_boundary": (
            f"{evidence_class}. Must not be emitted as HARDWARE_MEASURED. "
            "Hardware Doctor overwrites chestnut_hawking_optimized on U50_PRESENT."
        ),
    }


def _probe_wakes() -> dict[str, dict[str, Any]]:
    """Call the wake-condition probes. Inventory, not a performance number."""
    out: dict[str, dict[str, Any]] = {}
    for wid in WAKE_IDS:
        rec = hw_wake.probe(wid)
        rec = dict(rec)
        rec["evidence_tier"] = rec.get("evidence_tier") or "STATIC"
        out[wid] = rec
    # Gate's own symbol, not just probe() dispatch.
    present, evidence = hw_wake.probe_u50()
    out["U50_PRESENT"]["probe_u50_present"] = bool(present)
    out["U50_PRESENT"]["probe_u50_evidence"] = evidence
    return out


def absent_u50dd(
    genome: Mapping[str, Any],
    *,
    profile: Any,
    chestnut: Any,
    chestnut_opt: Any,
    wake: Mapping[str, Any],
) -> dict[str, Any]:
    """Honest degradation for a board that is not here.

    Names exactly what is unknown and what U50_PRESENT would (and would not)
    resolve. Brochure numbers stay STATIC. Chestnut payload stays
    THIRD_PARTY_REPORTED.
    """
    domain = (genome.get("domains") or {}).get("u50dd_0") or {}
    axes = mg.axes_for_domain(domain, genome=genome)
    unknown = [
        r["axis"] for r in axes
        if r.get("status") in {"UNKNOWN", "ABSENT", "BLOCKED"} and r.get("axis") != "presence"
    ]
    brochure = _brochure(profile)
    # Speed grade is explicitly UNPINNED on the ES3 SKU (hwir note).
    if not any(u["field"] == "speed_grade" for u in brochure["unpinned_fields"]):
        brochure["unpinned_fields"].append({
            "field": "speed_grade",
            "value": hwir.UNPINNED,
            "reason": (
                "hwir: speed grade for the ES3 SKU is UNPINNED; not copied "
                "from production U50"
            ),
            "evidence_tier": "STATIC",
            "hardware_measured": False,
        })
    return {
        "name": "u50dd_0",
        "kind": "FPGA",
        "product": "Alveo U50DD",
        "present": False,
        "physical": False,
        "maturity": domain.get("maturity") or "DECLARED",
        "performance": "UNKNOWN",
        "evidence_tier": "STATIC",
        "hardware_measured": False,
        "wake_condition": "U50_PRESENT",
        "wake": {
            "id": wake.get("id") or "U50_PRESENT",
            "present": bool(wake.get("present")),
            "evidence": wake.get("evidence"),
            "probe_u50_present": wake.get("probe_u50_present"),
            "probe_u50_evidence": wake.get("probe_u50_evidence"),
            "evidence_tier": "STATIC",
            "description": wake.get("description") or hw_wake.WAKE_CONDITIONS.get("U50_PRESENT"),
        },
        "brochure": brochure,
        "carrier": _carrier_view(chestnut, evidence_class=hwir.CHESTNUT_THIRD_PARTY),
        "carrier_hawking_optimized": _carrier_view(
            chestnut_opt, evidence_class="UNPINNED_AWAITING_HAWKING_MEASUREMENT"
        ),
        "unknown_axes": unknown,
        "axes": axes,
        "what_wake_resolves": [
            "device identity/presence: PCIe/Thunderbolt inventory matching xilinx/alveo/u50/xcu50",
            "permission to flip u50dd_0.present after a local census (still not a rate)",
            "the U50_* capability-graph gates become runnable, not passed",
            "Hardware Doctor may overwrite chestnut_hawking_optimized with a Hawking measurement",
        ],
        "what_wake_does_not_resolve": [
            "measured HBM bandwidth (needs 15.13 M09 HBM round trip, still HARDWARE_MEASURED only after the ladder)",
            "measured H2C/C2H (M07/M08)",
            "local thermal on this carrier (U50DD is passively cooled; Chestnut supplies no airflow)",
            "Fmax, joules/token, capability-preserving TPS",
            "any DS965 LUT/DSP/HBM brochure field becoming HARDWARE_MEASURED by virtue of enumeration",
            "Chestnut usable payload: THIRD_PARTY_REPORTED until Hawking measures it",
        ],
        "wake_gates": list(hwir.U50_WAKE_GATES),
        "claim_boundary": (
            "U50DD is DECLARED-absent on this host. STATIC vendor literature "
            "and THIRD_PARTY_REPORTED carrier figures are models. No number "
            "in this object is HARDWARE_MEASURED."
        ),
    }


def workload_fit(
    genome: Mapping[str, Any],
    economics: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Which workloads fit which device. COST_MODEL over measured capacity."""
    domains = genome.get("domains") or {}

    def _present(kind: str, name: str | None = None) -> bool:
        for n, d in domains.items():
            if not isinstance(d, dict):
                continue
            if name and n != name:
                continue
            if d.get("kind") == kind and d.get("present"):
                return True
        return False

    catalog = [
        {
            "id": "interactive_decode",
            "requires_kind": "GPU",
            "also": ("UMA",),
            "profile": "INTERACTIVE",
            "fit": bool(_present("GPU") and _present("UMA")),
            "evidence_tier": "COST_MODEL",
            "reason": "G023 INTERACTIVE winner is a UMA-resident body on the host GPU",
        },
        {
            "id": "max_throughput_decode",
            "requires_kind": "GPU",
            "also": ("UMA",),
            "profile": "MAXX",
            "fit": bool(_present("GPU") and _present("UMA")),
            "evidence_tier": "COST_MODEL",
            "reason": "G023 MAXX winner is a UMA-resident body on the host GPU",
        },
        {
            "id": "ane_prefill",
            "requires_kind": "ANE",
            "fit": False,
            "unproven": True,
            "evidence_tier": "STATIC",
            "reason": (
                "ANE is present (ioreg H11ANEIn) but the lab receipt's add "
                "fixture preferred CPU; Flash/Qwen placement is not claimed"
            ),
        },
        {
            "id": "lake_stage",
            "requires_kind": "STORAGE",
            "fit": bool(_present("STORAGE")),
            "evidence_tier": "COST_MODEL",
            "reason": "staging cost is extrapolated from a sequential sample; not a measured stage of a body",
        },
        {
            "id": "u50dd_hbm_resident_shard",
            "requires_kind": "FPGA",
            "requires_name": "u50dd_0",
            "wake_condition": "U50_PRESENT",
            "fit": False,
            "evidence_tier": "STATIC",
            "reason": "u50dd_0 is DECLARED-absent; an FPGA-required resident is ineligible until U50_PRESENT",
        },
        {
            "id": "dgx_offload",
            "requires_kind": "EXTERNAL_ACCELERATOR",
            "requires_name": "nvidia_dgx_0",
            "wake_condition": "DGX_PRESENT",
            "fit": False,
            "evidence_tier": "STATIC",
            "reason": "no DGX on this host",
        },
        {
            "id": "egpu_offload",
            "requires_kind": "EXTERNAL_ACCELERATOR",
            "requires_name": "egpu_0",
            "wake_condition": "EGPU_PRESENT",
            "fit": False,
            "evidence_tier": "STATIC",
            "reason": "no eGPU enclosure; the Apple SoC GPU is not an eGPU",
        },
    ]
    bodies = []
    for b in economics.get("bodies") or []:
        bodies.append({
            "id": b.get("id"),
            "resident_bytes": b.get("resident_bytes"),
            "fits_uma": b.get("fits_uma"),
            "uma_headroom_bytes": b.get("uma_headroom_bytes"),
            "stage_from_corpdrive": b.get("stage_from_corpdrive"),
            "stage_from_ssd": b.get("stage_from_ssd"),
            "evidence_tier": "COST_MODEL",
        })
    return {
        "evidence_tier": "COST_MODEL",
        "uma_bytes": economics.get("uma_bytes"),
        "uma_present": economics.get("uma_present"),
        "ane_present": economics.get("ane_present"),
        "fpga_present": economics.get("fpga_present"),
        "external_accelerator_present": economics.get("external_accelerator_present"),
        "selected_resident": {
            "selected": selection.get("selected"),
            "installed": bool(selection.get("installed")),
            "profile": selection.get("profile") or economics.get("profile"),
            "evidence_tier": selection.get("evidence_tier") or "COST_MODEL",
            "note": "decision record, not an install",
        },
        "bodies": bodies,
        "workloads": catalog,
        "note": economics.get("note"),
    }


def candidate_probes(
    genome: Mapping[str, Any],
    axes_by_device: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Named probes bound to the live axis inventory. Info weight is not a guess
    about the outcome; it is how much the current status would move."""
    cpu = _axis_map(axes_by_device.get("cpu_0") or [])
    gpu = _axis_map(axes_by_device.get("gpu_uma_0") or [])
    uma = _axis_map(axes_by_device.get("uma_0") or [])
    ane = _axis_map(axes_by_device.get("ane_0") or [])
    u50 = _axis_map(axes_by_device.get("u50dd_0") or [])
    net = _axis_map(axes_by_device.get("network") or [])
    cpu_present = bool(((genome.get("domains") or {}).get("cpu_0") or {}).get("present"))
    gpu_present = bool(((genome.get("domains") or {}).get("gpu_uma_0") or {}).get("present"))
    ane_present = bool(((genome.get("domains") or {}).get("ane_0") or {}).get("present"))
    uma_present = bool(((genome.get("domains") or {}).get("uma_0") or {}).get("present"))

    def _probe(
        pid: str,
        *,
        axis: str,
        device: str,
        row: Mapping[str, Any] | None,
        cost: int,
        runnable_now: bool,
        device_present: bool,
        hypothesis: str,
        falsifier: str,
        resolves: str,
        prerequisite: str,
        ineligible_reason: str | None = None,
    ) -> dict[str, Any]:
        status = (row or {}).get("status") or "UNKNOWN"
        info = 0 if ineligible_reason else _info_weight(
            str(status), device_present=device_present, runnable_now=runnable_now
        )
        return {
            "id": pid,
            "axis": axis,
            "device": device,
            "status_now": status,
            "evidence_tier_now": (row or {}).get("evidence_tier") or "STATIC",
            "cost": cost,
            "runnable_now": bool(runnable_now) and not ineligible_reason,
            "device_present": bool(device_present),
            "info_weight": info,
            "hypothesis": hypothesis,
            "falsifier": falsifier,
            "resolves": resolves,
            "prerequisite": prerequisite,
            "ineligible_reason": ineligible_reason,
            "predicted_effect": {
                "direction": "reduce_uncertainty",
                "magnitude_class": "UNKNOWN",
            },
            "evidence_class_if_run": (
                "HARDWARE_MEASURED" if runnable_now and device_present else "STATIC"
            ),
        }

    wan_blocked = (net.get("bandwidth") or {}).get("status") == "BLOCKED"
    probes = [
        _probe(
            "HDG-CPU-STREAM",
            axis="bandwidth",
            device="cpu_0",
            row=cpu.get("bandwidth"),
            cost=COST_BOUNDED,
            runnable_now=True,
            device_present=cpu_present,
            hypothesis=(
                "A bounded CPU STREAM/triad on this SoC would convert cpu_0.bandwidth "
                "from UNKNOWN to HARDWARE_MEASURED (or UNRELIABLE if the machine is "
                "not held still). The GPU triad is not a CPU roof."
            ),
            falsifier="IQR > 10% under current contention, or no userspace triad can run",
            resolves="cpu_0.bandwidth",
            prerequisite="userspace CPU triad (numpy/mlx); no GPU lease required",
        ),
        _probe(
            "HDG-CPU-LATENCY",
            axis="latency",
            device="cpu_0",
            row=cpu.get("latency"),
            cost=COST_BOUNDED,
            runnable_now=True,
            device_present=cpu_present,
            hypothesis="A pointer-chase / cache-line sample would convert cpu_0.latency from UNKNOWN.",
            falsifier="sample window < 5ms, or contention makes the IQR miss the 10% gate",
            resolves="cpu_0.latency",
            prerequisite="userspace; no GPU lease required",
        ),
        _probe(
            "HDG-GPU-TRIAD",
            axis="bandwidth",
            device="gpu_uma_0",
            row=gpu.get("bandwidth"),
            cost=(
                COST_PROTECTED_WINDOW
                if (gpu.get("bandwidth") or {}).get("status") == "UNRELIABLE"
                else COST_BOUNDED
            ),
            runnable_now=True,
            device_present=gpu_present,
            hypothesis=(
                "A GPU triad converts gpu_uma_0.bandwidth from ABSENT/UNKNOWN to "
                "HARDWARE_MEASURED. If the live triad is UNRELIABLE, the same "
                "probe in a protected window is the one that can become a roof. "
                "If it is already MEASURED and reliable, info_weight is 0."
            ),
            falsifier="mlx unavailable, or IQR > 10% so the number is UNRELIABLE not a roof",
            resolves="gpu_uma_0.bandwidth",
            prerequisite=(
                "quiet GPU window"
                if (gpu.get("bandwidth") or {}).get("status") == "UNRELIABLE"
                else "mlx in this interpreter; one triad is not a roof"
            ),
        ),
        _probe(
            "HDG-GPU-DISPATCH-LATENCY",
            axis="latency",
            device="gpu_uma_0",
            row=gpu.get("latency"),
            cost=COST_BOUNDED,
            runnable_now=True,
            device_present=gpu_present,
            hypothesis="A small-dispatch sample would convert gpu_uma_0.latency from UNKNOWN.",
            falsifier="dispatch sample is not a roof; IQR misses the gate",
            resolves="gpu_uma_0.latency",
            prerequisite="mlx or Metal runtime; not a production ADP",
        ),
        _probe(
            "HDG-SUSTAINED-THERMAL",
            axis="thermal",
            device="gpu_uma_0",
            row=gpu.get("thermal"),
            cost=COST_SUSTAINED,
            runnable_now=True,
            device_present=gpu_present,
            hypothesis=(
                "A sustained thermal campaign is ABSENT and is required before a "
                "production ADP (G049). This is the probe that converts thermal "
                "from ABSENT to HARDWARE_MEASURED."
            ),
            falsifier="throttling appears, or the campaign cannot be isolated from lake fills",
            resolves="genome.thermal_envelope / gpu_uma_0.thermal",
            prerequisite="long exclusive GPU window; not a microbenchmark",
        ),
        _probe(
            "HDG-ANE-ORGAN-PLACEMENT",
            axis="placement",
            device="ane_0",
            row={
                "status": "UNKNOWN",
                "evidence_tier": "STATIC",
                "reason": "add-fixture / supported-device list is not Flash/Qwen placement",
            },
            cost=COST_LAB,
            runnable_now=True,
            device_present=ane_present,
            hypothesis=(
                "The lab receipt profiles an add fixture, not Flash/Qwen. A "
                "real-organ MLComputePlan placement probe would convert ANE "
                "residency from 'present but unproven' to PROFILED-for-that-organ."
            ),
            falsifier="preferred device remains CPU, or the organ is unsupported",
            resolves="ane_0.placement for a named organ (still not TOPS)",
            prerequisite="CoreML/MLComputePlan lab path; no TOPS invented",
        ),
        _probe(
            "HDG-ANE-ENERGY",
            axis="energy",
            device="ane_0",
            row=ane.get("energy"),
            cost=COST_PRIVILEGED,
            runnable_now=True,
            device_present=ane_present,
            hypothesis="powermetrics during an ANE-legal fixture would convert ane_0.energy from UNKNOWN.",
            falsifier="powermetrics unavailable without privileges, or ANE is not the preferred device",
            resolves="ane_0.energy",
            prerequisite="powermetrics; may require privileges. Still not TOPS.",
        ),
        _probe(
            "HDG-UMA-COPY-ELISION",
            axis="transport",
            device="uma_0",
            row={
                "status": "UNKNOWN",
                "evidence_tier": "STATIC",
                "reason": "UMA-COPY-ELISION is a STATIC law; the avoided-copy overlay is unmeasured",
            },
            cost=COST_BOUNDED,
            runnable_now=True,
            device_present=uma_present,
            hypothesis=(
                "UMA-COPY-ELISION is a STATIC topology law. A measurement of "
                "avoided HtoD/DtoH bytes would add a HARDWARE_MEASURED overlay "
                "without promoting the law to a speedup on discrete GPUs."
            ),
            falsifier="a discrete-memory copy is observed on this SoC, which would refute the law",
            resolves="uma_0.transport as a measured overlay on a STATIC law",
            prerequisite="a CUDA-shaped workload rewritten against UMA",
        ),
        _probe(
            "HDG-NETWORK-WAN",
            axis="bandwidth",
            device="network",
            row=net.get("bandwidth"),
            cost=COST_BOUNDED,
            runnable_now=False,
            device_present=True,
            hypothesis="A WAN sample would convert network.bandwidth from BLOCKED.",
            falsifier="the sample contends with live hf download workers",
            resolves="network.wan_throughput",
            prerequisite="lake fills idle",
            ineligible_reason=(
                "live hf download workers write to /Volumes/corpdrive; a WAN "
                "sample would be contended and could disturb them"
            ) if wan_blocked else None,
        ),
        _probe(
            "HDG-U50-WAKE-INVENTORY",
            axis="presence",
            device="u50dd_0",
            row=u50.get("presence"),
            cost=COST_IDENTITY,
            runnable_now=True,
            device_present=False,
            hypothesis="Re-running the U50_PRESENT inventory confirms the board has not arrived.",
            falsifier="PCIe/Thunderbolt identity matches xilinx/alveo/u50/xcu50",
            resolves="u50dd_0.presence (inventory only)",
            prerequisite="system_profiler/ioreg; already the wake probe",
        ),
        _probe(
            "HDG-U50DD-ARRIVAL-HBM",
            axis="bandwidth",
            device="u50dd_0",
            row=u50.get("bandwidth"),
            cost=COST_BOARD_ARRIVAL,
            runnable_now=False,
            device_present=False,
            hypothesis=(
                "An HBM round trip on an attached U50DD (15.13 M09) would convert "
                "u50dd_0.bandwidth from ABSENT/UNKNOWN to HARDWARE_MEASURED. "
                "U50_PRESENT alone does not produce that number."
            ),
            falsifier="M09 does not complete, or the carrier supplies no usable HBM path",
            resolves="u50dd_0.bandwidth (only after wake AND the DMA/HBM ladder)",
            prerequisite="wake_condition U50_PRESENT, then M00–M09",
        ),
        _probe(
            "HDG-U50DD-CHESTNUT-PAYLOAD",
            axis="transport",
            device="u50dd_0",
            row=u50.get("transport"),
            cost=COST_BOARD_ARRIVAL,
            runnable_now=False,
            device_present=False,
            hypothesis=(
                "On U50_PRESENT, a Hawking measurement overwrites "
                "chestnut_hawking_optimized. Until then the ~1.68 GB/s figure is "
                "THIRD_PARTY_REPORTED, not HARDWARE_MEASURED."
            ),
            falsifier="measured payload disagrees with the third-party completer ceiling",
            resolves="chestnut_hawking_optimized (Hardware Doctor overwrite on wake)",
            prerequisite="wake_condition U50_PRESENT plus a Hawking transport campaign",
        ),
        _probe(
            "HDG-DGX-WAKE-INVENTORY",
            axis="presence",
            device="nvidia_dgx_0",
            row=_axis_map(axes_by_device.get("nvidia_dgx_0") or []).get("presence"),
            cost=COST_IDENTITY,
            runnable_now=True,
            device_present=False,
            hypothesis="nvidia-smi inventory confirms no DGX on this host.",
            falsifier="nvidia-smi -L names a DGX product",
            resolves="nvidia_dgx_0.presence",
            prerequisite="nvidia-smi",
        ),
        _probe(
            "HDG-EGPU-WAKE-INVENTORY",
            axis="presence",
            device="egpu_0",
            row=_axis_map(axes_by_device.get("egpu_0") or []).get("presence"),
            cost=COST_IDENTITY,
            runnable_now=True,
            device_present=False,
            hypothesis="Displays/Thunderbolt inventory confirms no eGPU enclosure.",
            falsifier="an eGPU enclosure is named in SPDisplays/SPThunderbolt",
            resolves="egpu_0.presence",
            prerequisite="system_profiler",
        ),
    ]
    return probes


def per_axis_inventory(genome: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Call machine_genome.axes_for_domain for every domain. Real call site."""
    out: dict[str, list[dict[str, Any]]] = {}
    for name, d in (genome.get("domains") or {}).items():
        if isinstance(d, dict):
            out[name] = mg.axes_for_domain(d, genome=genome)
    return out


def diagnose(
    *,
    live: bool = True,
    genome: Mapping[str, Any] | None = None,
    contended: bool = True,
    contention_note: str | None = None,
    profile: str = "INTERACTIVE",
    wakes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Answer the five questions. live=True probes this host."""
    called: list[str] = []
    note = contention_note or (
        "live HCLI daemon and hf download workers; genome probes are "
        "identity plus bounded read-only storage samples"
    )

    identity = mg.discover_identity()
    called.append("machine_genome.discover_identity")

    if live or genome is None:
        char = da.characterize(
            {"identity": identity, "called": "machine_genome.discover_identity"},
            contended=contended,
            contention_note=note,
        )
        called.append("device_ascension.characterize")
        called.append("machine_genome.build")
        genome_d = char["genome"]
    else:
        genome_d = dict(genome)
        char = {
            "stage": "characterize",
            "genome": genome_d,
            "genome_digest": genome_d.get("genome_digest"),
            "evidence_tier": genome_d.get("evidence_tiers_used") or "STATIC",
        }

    econ_stage = da.economics(char, profile=profile)
    called.append("device_ascension.economics")
    called.append(econ_stage.get("called") or "device_profiles.economics_from_genome")
    sel_stage = da.select(econ_stage, profile=profile)
    called.append("device_ascension.select")
    called.append(sel_stage.get("called") or "device_profiles.select_resident")

    inventory = mg.devices_exist(genome_d)
    called.append("machine_genome.devices_exist")
    axes = per_axis_inventory(genome_d)
    called.append("machine_genome.axes_for_domain")

    profile_u50dd = hwir.u50_family_profile("u50dd")
    called.append("hwir.u50_family_profile")
    chestnut = hwir.chestnut_current_firmware()
    called.append("hwir.chestnut_current_firmware")
    chestnut_opt = hwir.chestnut_hawking_optimized()
    called.append("hwir.chestnut_hawking_optimized")

    if wakes is None:
        wakes = _probe_wakes()
        called.append("tools.roadmap.hardware.probe")
        called.append("tools.roadmap.hardware.probe_u50")
    else:
        wakes = {k: dict(v) for k, v in wakes.items()}

    u50dd = absent_u50dd(
        genome_d,
        profile=profile_u50dd,
        chestnut=chestnut,
        chestnut_opt=chestnut_opt,
        wake=wakes["U50_PRESENT"],
    )

    workloads = workload_fit(
        genome_d, econ_stage.get("economics") or {}, sel_stage.get("decision") or {}
    )
    probes = candidate_probes(genome_d, axes)
    reducing = [p for p in probes if int(p.get("info_weight") or 0) > 0]
    already = [p for p in probes if int(p.get("info_weight") or 0) == 0]
    ranked = rank_queue(reducing)

    # Overlay wake-probe presence onto the inventory (STATIC inventory, not a rate).
    for row in inventory["present"] + inventory["absent"]:
        wake_id = row.get("wake_condition")
        if wake_id and wake_id in wakes:
            row["wake_probe_present"] = bool(wakes[wake_id].get("present"))
            row["wake_probe_evidence"] = wakes[wake_id].get("evidence")
            row["wake_probe_evidence_tier"] = "STATIC"

    host = {
        "soc": genome_d.get("soc") or identity.get("soc"),
        "arch": genome_d.get("arch") or identity.get("arch"),
        "cpu_cores": genome_d.get("cpu_cores"),
        "gpu_cores": genome_d.get("gpu_cores"),
        "memory_bytes": genome_d.get("memory_bytes"),
        "genome_digest": genome_d.get("genome_digest"),
        "evidence_tier": "STATIC",
        "note": (
            "identity from machine_genome.discover_identity / build. "
            "This is THIS machine, not a fixture."
        ),
    }

    return {
        "schema": SCHEMA,
        "recorded_by": RECORDED_BY,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "called": called,
        "host": host,
        "questions": {
            "devices_exist": inventory,
            "measured_vs_modelled_vs_unknown": axes,
            "workload_fit": workloads,
            "experiments_ranked": ranked,
            "backend_maturity": genome_d.get("backend_maturity"),
        },
        "already_known_probes": already,
        "absent_u50dd": u50dd,
        "wakes": wakes,
        "ranking_rule": (
            "runnable first, then info_weight / cost via integer key "
            "-(info*60//cost). Board-gated probes stay in the list but cannot "
            "outrank a cheap UNKNOWN-on-present probe."
        ),
        "evidence_tiers_used": sorted({
            t for rows in axes.values() for r in rows
            for t in [r.get("evidence_tier")]
            if t in mg.EVIDENCE_TIERS
        }),
        "claim_boundary": (
            "Host domains may carry HARDWARE_MEASURED axes from the genome. "
            "Absent U50DD/DGX/eGPU never do. STATIC vendor literature and "
            "THIRD_PARTY_REPORTED carrier figures are not measurements. Tiers "
            "are never merged."
        ),
        "recovered_implementation": [
            {
                "path": "tools/accelerator/machine_genome.py",
                "what": "CPU/GPU/UMA/ANE/storage/network + declared FPGA/DGX/U50DD/eGPU; axes_for_domain",
                "present": True,
            },
            {
                "path": "tools/accelerator/device_ascension.py",
                "what": "discover -> characterize -> economics -> select -> promote -> invalidate",
                "present": True,
            },
            {
                "path": "tools/future/hwir.py",
                "what": "u50_family_profile('u50dd'), Chestnut carrier, U50_WAKE_GATES",
                "present": True,
            },
            {
                "path": "tools/roadmap/hardware.py",
                "what": "wake probes U50_PRESENT / DGX_PRESENT / EGPU_PRESENT (STATIC inventory)",
                "present": True,
            },
            {
                "path": "tools/odyssey/device_profiles.py",
                "what": "economics_from_genome / select_resident (called via device_ascension, not edited)",
                "present": True,
            },
            {
                "path": "tools/future/hardware_doctor.py",
                "what": "FPGA-axis experiment proposer; sibling, not replaced",
                "present": True,
            },
        ],
    }


def _assert_honest(doc: dict[str, Any]) -> None:
    """Refuse a diagnosis that stamps HARDWARE_MEASURED on an absent device."""
    u50 = doc.get("absent_u50dd") or {}
    if u50.get("present"):
        raise HardwareDoctorError("absent_u50dd.present must be False on this host")
    if u50.get("hardware_measured"):
        raise HardwareDoctorError("absent_u50dd must not claim hardware_measured")
    if (u50.get("brochure") or {}).get("evidence_tier") == "HARDWARE_MEASURED":
        raise HardwareDoctorError("U50DD brochure is STATIC vendor literature")
    if (u50.get("carrier") or {}).get("hardware_measured"):
        raise HardwareDoctorError("Chestnut carrier is not a Hawking measurement")
    axes = (doc.get("questions") or {}).get("measured_vs_modelled_vs_unknown") or {}
    for name in ("u50dd_0", "fpga_hbm_0", "nvidia_dgx_0", "egpu_0"):
        for row in axes.get(name) or []:
            if row.get("axis") == "presence":
                continue
            if row.get("evidence_tier") == "HARDWARE_MEASURED":
                raise HardwareDoctorError(
                    f"{name}.{row.get('axis')} is HARDWARE_MEASURED on an absent device"
                )


def build(*, live: bool = True, **kwargs: Any) -> Path:
    doc = diagnose(live=live, **kwargs)
    _assert_honest(doc)
    _seal(doc)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS / RECEIPT
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return out


def _summary(doc: dict[str, Any]) -> str:
    q = doc["questions"]
    inv = q["devices_exist"]
    present = ", ".join(f"{r['name']}({r['kind']})" for r in inv["present"])
    absent = ", ".join(f"{r['name']}({r['kind']})" for r in inv["absent"])
    ranked = q["experiments_ranked"]
    lines = [
        f"schema {doc['schema']}",
        f"host {doc['host'].get('soc')} digest={doc['host'].get('genome_digest')}",
        f"1. present: {present}",
        f"   absent:  {absent}",
        "2. per-axis (status/tier):",
    ]
    for name, rows in q["measured_vs_modelled_vs_unknown"].items():
        bits = [f"{r['axis']}={r['status']}/{r['evidence_tier']}" for r in rows]
        lines.append(f"   {name}: " + "; ".join(bits))
    lines.append("3. workload fit:")
    for w in q["workload_fit"]["workloads"]:
        lines.append(f"   {w['id']}: fit={w.get('fit')} ({w.get('reason')})")
    sel = q["workload_fit"]["selected_resident"]
    lines.append(
        f"   selected={sel.get('selected')} installed={sel.get('installed')} "
        f"tier={sel.get('evidence_tier')}"
    )
    lines.append("4. experiments (runnable first, then info/cost):")
    for r in ranked[:8]:
        ipc = r["information_per_cost"]
        lines.append(
            f"   #{r['rank']} {r['id']} info={ipc['info_weight']} cost={ipc['cost']} "
            f"runnable={ipc['runnable_now']} -> {r['resolves']}"
        )
    lines.append("5. backend maturity: " + json.dumps(q["backend_maturity"], sort_keys=True))
    u = doc["absent_u50dd"]
    lines.append(
        f"absent U50DD: present={u['present']} wake={u['wake_condition']} "
        f"wake_present={u['wake']['present']} unknown_axes={u['unknown_axes']}"
    )
    lines.append("called: " + ", ".join(doc["called"]))
    return "\n".join(lines)


def main() -> int:
    path = build(live=True)
    doc = json.loads(path.read_text())
    print(path)
    print(_summary(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
