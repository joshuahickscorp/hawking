"""Machine-readable wake-condition probes.

A hardware-requiring gate never PASSes without the device. These probes only
answer "is the device present on this host", labelled STATIC inventory, never
a performance measurement.
"""
from __future__ import annotations

import re
import subprocess
from typing import Any

WAKE_CONDITIONS: dict[str, str] = {
    "U50_PRESENT": (
        "AMD/Xilinx Alveo U50DD/XCU50-class FPGA enumerated on this host "
        "(PCIe/Thunderbolt identity contains xilinx/alveo/u50/xcu50)."
    ),
    "DGX_PRESENT": (
        "NVIDIA DGX Spark or DGX-class node visible: nvidia-smi succeeds and "
        "the product name contains DGX."
    ),
    "NEW_M_SERIES_PRESENT": (
        "Apple SoC newer than the current M3 Ultra textbook (M4 or later)."
    ),
    "HMF_PRESENT": (
        "HMF/HGVAS managed-memory device visible (CXL expander / HBM appliance "
        "enumerated; identity contains HMF, HGVAS, or CXL memory device)."
    ),
    "EGPU_PRESENT": (
        "External GPU enumerated on Thunderbolt/USB4 (eGPU / AMD / NVIDIA "
        "enclosure that is not the Apple SoC GPU)."
    ),
}


_CMD_CACHE: dict[tuple[str, ...], str] = {}
_PROBE_ALL: dict[str, dict[str, Any]] | None = None


def _run(argv: list[str], timeout: float = 8.0) -> str:
    key = tuple(argv)
    cached = _CMD_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        result = ""
    else:
        result = (cp.stdout or "") + "\n" + (cp.stderr or "")
    _CMD_CACHE[key] = result
    return result


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key]).strip()


def probe_u50() -> tuple[bool, str]:
    blob = "\n".join(
        [
            _run(["system_profiler", "SPPCIeDataType", "-detailLevel", "mini"]),
            _run(["system_profiler", "SPThunderboltDataType", "-detailLevel", "mini"]),
            _run(["ioreg", "-r", "-c", "IOPCIDevice", "-l"]),
        ]
    )
    hit = re.search(r"xilinx|alveo|u50dd|xcu50|\bu50\b", blob, re.I)
    if hit:
        return True, f"device identity matched {hit.group(0)!r} in PCIe/Thunderbolt inventory"
    return False, "no Xilinx/Alveo/U50/XCU50 identity in PCIe/Thunderbolt/ioreg inventory"


def probe_dgx() -> tuple[bool, str]:
    smi = _run(["nvidia-smi", "-L"])
    if not smi.strip() or "command not found" in smi.lower() or "not found" in smi.lower():
        return False, "nvidia-smi not present or produced no GPU list"
    if re.search(r"\bDGX\b", smi, re.I):
        return True, "nvidia-smi -L named a DGX product"
    return False, "nvidia-smi present but product list is not DGX"


def probe_new_m_series() -> tuple[bool, str]:
    brand = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
    model = _sysctl("hw.model")
    blob = f"{brand} {model}"
    # M3 Ultra (Mac15,14) is the current textbook, not a newer M-series.
    if re.search(r"\bM([4-9]|[1-9][0-9])\b", blob):
        return True, f"SoC identity {blob!r} is M4 or later"
    return False, f"SoC identity {blob!r} is not M4-or-later (current textbook is M3 Ultra)"


def probe_hmf() -> tuple[bool, str]:
    blob = "\n".join(
        [
            _run(["system_profiler", "SPMemoryDataType", "-detailLevel", "mini"]),
            _run(["system_profiler", "SPPCIeDataType", "-detailLevel", "mini"]),
        ]
    )
    hit = re.search(r"\bHMF\b|\bHGVAS\b|CXL", blob, re.I)
    if hit:
        return True, f"memory/PCIe inventory matched {hit.group(0)!r}"
    return False, "no HMF/HGVAS/CXL memory appliance in inventory"


def probe_egpu() -> tuple[bool, str]:
    blob = "\n".join(
        [
            _run(["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"]),
            _run(["system_profiler", "SPThunderboltDataType", "-detailLevel", "mini"]),
        ]
    )
    if re.search(r"\beGPU\b|Thunderbolt Display|external gpu", blob, re.I):
        return True, "displays/thunderbolt inventory named an eGPU"
    # Apple M-series GPU is not an eGPU.
    return False, "no eGPU enclosure in displays/thunderbolt inventory"


_PROBES = {
    "U50_PRESENT": probe_u50,
    "DGX_PRESENT": probe_dgx,
    "NEW_M_SERIES_PRESENT": probe_new_m_series,
    "HMF_PRESENT": probe_hmf,
    "EGPU_PRESENT": probe_egpu,
}


def probe(wake_id: str) -> dict[str, Any]:
    if wake_id not in _PROBES:
        raise KeyError(f"unknown wake condition {wake_id!r}")
    present, evidence = _PROBES[wake_id]()
    return {
        "id": wake_id,
        "present": bool(present),
        "description": WAKE_CONDITIONS[wake_id],
        "evidence": evidence,
        "evidence_tier": "STATIC",
    }


def probe_all() -> dict[str, dict[str, Any]]:
    global _PROBE_ALL
    if _PROBE_ALL is None:
        _PROBE_ALL = {name: probe(name) for name in WAKE_CONDITIONS}
    return _PROBE_ALL


def blocked_hardware_wakes(gates: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (gate_id, wake_condition) for every BLOCKED_HARDWARE gate.

    Raises if any such gate has an empty wake condition. The daemon uses the
    wake id (U50_PRESENT, DGX_PRESENT, ...) to know what to activate when a
    device arrives.
    """
    out: list[tuple[str, str]] = []
    for gate in gates.values():
        if not isinstance(gate, dict) or gate.get("status") != "BLOCKED_HARDWARE":
            continue
        gid = str(gate.get("id") or "")
        wake = gate.get("wake_condition")
        if not isinstance(wake, str) or not wake.strip():
            raise ValueError(f"{gid or '<unknown>'} BLOCKED_HARDWARE with empty wake_condition")
        wake = wake.strip()
        if wake not in WAKE_CONDITIONS:
            raise ValueError(
                f"{gid} wake_condition {wake!r} is not a known hardware id; "
                f"known={sorted(WAKE_CONDITIONS)}"
            )
        out.append((gid, wake))
    return out
