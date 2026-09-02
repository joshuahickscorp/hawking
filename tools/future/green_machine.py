"""GREEN_MACHINE — II-E / H-ROADMAP §24, with per-value evidence tiers.

Roadmap §24: Green Machine becomes real only where power is measured.
Utilization is not energy efficiency. Report J/token or J/accepted WorkUnit
only when instrumentation supports it.

II-E gene card root phenotype: measure useful work per energy without
Goodharting. The eight SUBGENES are the categories this sidecar emits.
Token-attributed joules (the existing metric contract) stay UNKNOWN: this
process has no GPU lease and does not wrap TOKEN_NS. What THIS Apple M3
Ultra can measure without root is labeled HARDWARE_MEASURED. What it cannot
is modeled and labeled COST_MODEL. Tiers are never merged on one value.

    python3 tools/future/green_machine.py --build
    python3 tools/future/green_machine.py --probe
    python3 tools/future/green_machine.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import ctypes
import ctypes.util
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.future._common import HARDWARE_FIELDS, HardwareClaimError, git

RECEIPT = "GREEN_MACHINE.json"
SCHEMA = "hawking.future.green_machine.v1"
UNKNOWN = "UNKNOWN"
CLAIM_CLASS = "STATIC_ONLY"

# ---------------------------------------------------------------------------
# Metric contract. Definitions are the product. Values stay UNKNOWN here.
# ---------------------------------------------------------------------------

METRIC_CONTRACT: tuple[dict[str, Any], ...] = (
    {
        "id": "joules_per_token",
        "unit": "J/token",
        "definition": (
            "Joules integrated over the same closed interval as TOKEN_NS body_ns "
            "(complete-token wall), divided by tokens emitted in that interval. "
            "The interval must include draft + verify + rollback when speculation "
            "is on; emitting tokens that later get rejected still cost energy. "
            "A datasheet TDP, an idle GPU-rail sample, or a FLOP-derived CMOS "
            "guess is not this metric."
        ),
        "numerator": "joules over the TOKEN_NS interval (CPU+GPU+DRAM rails, or a documented subset)",
        "denominator": "tokens emitted in that interval, including rejected speculative drafts",
        "requires": (
            "PROTECTED_ABSOLUTE GPU lease",
            "energy wrap around the same interval as TOKEN_NS",
            "root powermetrics OR a working IOReport Energy Model subscription",
        ),
        "hardware_field": "joules_per_token",
        "blocked_by_write_receipt": True,
    },
    {
        "id": "joules_per_accepted_token",
        "unit": "J/accepted-token",
        "definition": (
            "Joules over the same closed interval as TOKEN_NS, divided by "
            "accepted tokens only. Speculative decoding drafts tokens that the "
            "verifier rejects; those drafts still move weights and still draw "
            "the GPU rail. Accepted tokens are the unit of useful work "
            "(see crates/hawking-speculate/src/metrics_sep.rs: "
            "ACCELERATED_ACCEPTED_TPS = accepted_tokens / (draft+verify+rollback)). "
            "joules_per_accepted_token is therefore the real energy axis of the "
            "scoreboard whenever speculation is enabled. It is always >= "
            "joules_per_token when any draft is rejected, and equal only when "
            "every drafted token is accepted (or speculation is off)."
        ),
        "numerator": "same joule integral as joules_per_token",
        "denominator": "accepted_tokens (target-verified tokens that advanced committed state)",
        "requires": (
            "everything joules_per_token requires",
            "an accepted-token count from the same interval (AccelCostLedger)",
        ),
        "hardware_field": None,
        "blocked_by_write_receipt": False,
    },
    {
        "id": "work_units_per_kwh",
        "unit": "WorkUnits/kWh",
        "definition": (
            "HCLI WorkUnits completed per kilowatt-hour of measured energy over "
            "the same wall. This is not tokens/watt and not VERIFIED_WUS_PER_HOUR "
            "(a time axis already present and ABSENT on the noetic scoreboard). "
            "It is useful work per joule, scaled to kWh so a production day is "
            "readable. Requires a WorkUnit completion ledger AND a trustworthy "
            "joule integral over the same window."
        ),
        "numerator": "WorkUnits sealed complete in the window",
        "denominator": "kilowatt-hours = joules / 3.6e6 over the same window",
        "requires": (
            "everything joules_per_token requires",
            "WorkUnit completion count with the same closed wall",
        ),
        "hardware_field": None,
        "blocked_by_write_receipt": False,
    },
    {
        "id": "idle_joules_per_second",
        "unit": "J/s",
        "definition": (
            "Mean joules per second while no decode/prefill is in flight. "
            "IOReport GPU Energy increments with a display even at idle "
            "(energy.rs standing finding: ~1 W GPU rail). Without an idle "
            "baseline, an 'active' sample attributes display/other-lane energy "
            "to the token. Idle is a measurement under a protected lease with "
            "the GPU held still, not a TDP fraction."
        ),
        "numerator": "joules over an idle window with no token work",
        "denominator": "idle window seconds",
        "requires": (
            "PROTECTED_ABSOLUTE GPU lease",
            "a still machine (no other GPU lane)",
            "working energy source",
        ),
        "hardware_field": None,
        "blocked_by_write_receipt": False,
    },
    {
        "id": "active_joules_per_second",
        "unit": "J/s",
        "definition": (
            "Mean joules per second while a token interval is open. Distinct "
            "from idle; the token-attributable power is (active - idle) only "
            "when both were taken under the same protected lease, same thermal "
            "state, and same rail set. Mixing a dirty idle with a dirty active "
            "is DIAGNOSTIC_RELATIVE at best and is not this metric."
        ),
        "numerator": "joules over a token-in-flight window",
        "denominator": "active window seconds",
        "requires": (
            "PROTECTED_ABSOLUTE GPU lease",
            "token work actually in flight",
            "working energy source",
        ),
        "hardware_field": None,
        "blocked_by_write_receipt": False,
    },
    {
        "id": "thermal_state",
        "unit": "enum",
        "definition": (
            "Machine thermal pressure/throttling state during the energy wrap "
            "(cool / warming / throttling, or a measured die temperature). A "
            "throttled run is not comparable to a cool run; thermal_envelope on "
            "the machine genome is currently ABSENT. 'No thermal warning "
            "recorded' from pmset is the absence of a warning log, not a "
            "temperature, and is not this metric."
        ),
        "numerator": None,
        "denominator": None,
        "requires": (
            "a thermal sensor or throttle flag that this process can read",
            "the reading taken inside the same window as the joule integral",
        ),
        "hardware_field": None,
        "blocked_by_write_receipt": False,
    },
)

METRIC_IDS: tuple[str, ...] = tuple(m["id"] for m in METRIC_CONTRACT)

# II-E gene card SUBGENES (H-ROADMAP.md lines 6040-6048) and §24. Exact strings.
ROADMAP_SUBGENES: tuple[str, ...] = (
    "GPU/CPU/FPGA power receipts",
    "J/token",
    "J/accepted-token",
    "WU/kWh",
    "thermal stability",
    "idle-vs-active cost",
    "energy-aware scheduler",
    "power caps only when measured",
)

EVIDENCE_TIERS: tuple[str, ...] = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)

TIER_STATIC = "STATIC"
TIER_FUNCTIONAL_SIM = "FUNCTIONAL_SIM"
TIER_COST_MODEL = "COST_MODEL"
TIER_CYCLE_APPROX = "CYCLE_APPROX"
TIER_HARDWARE_MEASURED = "HARDWARE_MEASURED"

# Standing finding from crates/hawking-core/src/token_ns/energy.rs (2026-08-16).
# A citation, not a measurement this process took. Using it as watts is COST_MODEL.
CITED_IDLE_GPU_WATTS_PRIOR = 0.98
CITED_IDLE_GPU_WATTS_SOURCE = (
    "crates/hawking-core/src/token_ns/energy.rs IoreportFinding::documented "
    "(2026-08-16 this machine: GPU Energy nJ incremented ~0.98 W over 1 s idle)"
)
TOKEN_INTERVAL_RECEIPT = "receipts/future/TOKEN_NS_OBJECTIVE.json"

HONESTY_RULE = (
    "Token-attributed metrics (joules_per_token, joules_per_accepted_token, "
    "work_units_per_kwh, the contracted idle/active J/s, thermal_state) stay "
    "UNKNOWN: this sidecar has no GPU lease and does not wrap TOKEN_NS. "
    "Process-attributed ri_energy_nj idle-vs-active on this pid is a real "
    "sample and is labeled HARDWARE_MEASURED; it is not joules_per_token. "
    "A unit conversion of a sampled integral (nJ to J, J over the same "
    "window to mean watts) stays HARDWARE_MEASURED. Multiplying a wattage "
    "by a cited ms/token from another receipt is COST_MODEL. FPGA/U50 is "
    "absent and is never HARDWARE_MEASURED. write_receipt still refuses a "
    "numeric joules_per_token. This sidecar produces STATIC_ONLY for the "
    "token contract; it produces neither DIAGNOSTIC_RELATIVE nor "
    "PROTECTED_ABSOLUTE."
)

SCOREBOARD_SLOT = {
    "noetic_scoreboard": {
        "path": "receipts/headless/NOETIC_SCOREBOARD.json",
        "existing_columns": (
            "EBPW",
            "RESIDENT_GB",
            "ACTIVE_GB_PER_TOKEN",
            "DRAM_GB_PER_TOKEN",
            "FLOP_PER_TOKEN",
            "DISPATCHES_PER_TOKEN",
            "ROUTES_PER_TOKEN",
            "ROUTING_NS_PER_TOKEN",
            "COMPLETE_TOKEN_NS",
            "TPS",
            "AGGREGATE_TPS_C2",
            "AGGREGATE_TPS_C4",
            "VERIFIED_WUS_PER_HOUR",
            "CAPABILITY",
        ),
        "missing_energy_columns": (
            "JOULES_PER_TOKEN",
            "JOULES_PER_ACCEPTED_TOKEN",
            "WORK_UNITS_PER_KWH",
        ),
        "cell_today": "ABSENT/UNKNOWN",
        "honesty": (
            "An unmeasured cell is never rendered as 0. A plausible zero makes "
            "an unmeasured candidate look cheap, which is the specific way a "
            "scoreboard lies."
        ),
    },
    "accelerator_scoreboard": {
        "path": "receipts/headless/ACCELERATOR_SCOREBOARD.json",
        "note": (
            "Frontier F015 probed this path as present on the campaign disk. "
            "It is not in git HEAD of this worktree and is not materialized "
            "in the sparse checkout. An energy axis would slot next to the "
            "existing complete-token / TPS columns the same way: ABSENT "
            "until a PROTECTED_ABSOLUTE wrap exists."
        ),
    },
}


class UntrustworthyMeasurement(ValueError):
    """Raised when a caller asks this sidecar to treat energy as a number."""


def unknown_metrics() -> dict[str, dict[str, Any]]:
    """Every contracted metric, explicitly UNKNOWN. Missing is not zero."""
    return {
        m["id"]: {
            "value": UNKNOWN,
            "state": UNKNOWN,
            "unit": m["unit"],
            "claim_class": CLAIM_CLASS,
            "trustworthy": False,
        }
        for m in METRIC_CONTRACT
    }


def energy_number(value: Any, field: str) -> float:
    """There is no legal conversion. Always raises.

    UNKNOWN is not 0. A forged float is not authority. A guard nobody has
    watched fail is not a guard — tests call this with both.
    """
    raise UntrustworthyMeasurement(
        f"{field}={value!r}: sidecar energy numbers do not exist; "
        "UNKNOWN is not a default and a float is not authority"
    )


def estimate_from_tdp_watts(tdp_watts: Any, token_ns: Any = None) -> float:
    """Forbidden. TDP is a datasheet envelope, not a token joule."""
    raise UntrustworthyMeasurement(
        f"TDP-derived estimates are forbidden (tdp_watts={tdp_watts!r}, "
        f"token_ns={token_ns!r})"
    )


def estimate_from_flops(flops: Any, picojoules_per_flop: Any = None) -> float:
    """Forbidden. CMOS P = alpha C V^2 f is not a measurement on this machine."""
    raise UntrustworthyMeasurement(
        f"FLOP-derived joule estimates are forbidden (flops={flops!r}, "
        f"pJ_per_flop={picojoules_per_flop!r})"
    )


def measurement_is_trustworthy(
    *,
    gpu_authority: bool,
    protected_lease: bool,
    energy_wrap_around_token_ns: bool,
    root_powermetrics: bool,
    ioreport_live_samples: bool,
) -> bool:
    """Predicate a future protected lane would use. All five must be true.

    This sidecar cannot satisfy gpu_authority or protected_lease, so the
    predicate is False for every call we make. Tests prove partial flags
    do not sneak through.
    """
    return bool(
        gpu_authority
        and protected_lease
        and energy_wrap_around_token_ns
        and (root_powermetrics or ioreport_live_samples)
    )


# ---------------------------------------------------------------------------
# Energy-aware scheduler — inert while measurement is untrustworthy.
# ---------------------------------------------------------------------------

REASON_UNTRUSTWORTHY = "MEASUREMENT_UNTRUSTWORTHY"
REASON_NUMERIC_WITHOUT_AUTHORITY = "NUMERIC_CLAIM_WITHOUT_AUTHORITY"
ACTION_REFUSE = "REFUSE"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _lookup_metric(metrics: Mapping[str, Any] | None, mid: str) -> Any:
    """Missing and None are UNKNOWN. They are never 0."""
    if not metrics or mid not in metrics:
        return UNKNOWN
    entry = metrics[mid]
    if isinstance(entry, dict):
        if "value" not in entry:
            return UNKNOWN
        value = entry["value"]
        return UNKNOWN if value is None else value
    if entry is None:
        return UNKNOWN
    return entry


@dataclass(frozen=True)
class EnergyScheduleDecision:
    action: str
    reason_code: str
    detail: str
    work_id: str | None
    metrics_consulted: tuple[str, ...]
    numeric_energy_used: bool
    substituted_default: bool
    claim_class: str
    admit_implemented: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "work_id": self.work_id,
            "metrics_consulted": list(self.metrics_consulted),
            "numeric_energy_used": self.numeric_energy_used,
            "substituted_default": self.substituted_default,
            "claim_class": self.claim_class,
            "admit_implemented": self.admit_implemented,
        }


class EnergyAwareScheduler:
    """Refuse to schedule on energy grounds while measurement is untrustworthy.

    There is no Admit path in this sidecar. A future Codex module under a
    protected GPU lease would be the one to admit. Copying hawking-orch's
    on_battery+quiet heuristic would be scheduling on a guess; we do not.
    """

    REQUIRED = METRIC_IDS

    def schedule(
        self,
        work: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        *,
        gpu_authority: bool = False,
    ) -> EnergyScheduleDecision:
        work_id = None if not work else (work.get("id") or work.get("work_id"))
        consulted: list[str] = []
        for mid in self.REQUIRED:
            consulted.append(mid)
            raw = _lookup_metric(metrics, mid)
            if _is_number(raw):
                return EnergyScheduleDecision(
                    action=ACTION_REFUSE,
                    reason_code=REASON_NUMERIC_WITHOUT_AUTHORITY,
                    detail=(
                        f"{mid} arrived as a number; this sidecar has no hardware "
                        "authority and will not schedule on it"
                    ),
                    work_id=None if work_id is None else str(work_id),
                    metrics_consulted=tuple(consulted),
                    numeric_energy_used=False,
                    substituted_default=False,
                    claim_class=CLAIM_CLASS,
                    admit_implemented=False,
                )
            if raw is not UNKNOWN:
                return EnergyScheduleDecision(
                    action=ACTION_REFUSE,
                    reason_code=REASON_UNTRUSTWORTHY,
                    detail=(
                        f"{mid} is not UNKNOWN and not a trustworthy measurement; "
                        "scheduler is inert rather than guessing"
                    ),
                    work_id=None if work_id is None else str(work_id),
                    metrics_consulted=tuple(consulted),
                    numeric_energy_used=False,
                    substituted_default=False,
                    claim_class=CLAIM_CLASS,
                    admit_implemented=False,
                )
        why = (
            "all energy metrics are UNKNOWN; scheduler is inert rather than guessing"
        )
        if not gpu_authority:
            why = (
                "no GPU authority and all energy metrics are UNKNOWN; "
                "scheduler is inert rather than guessing"
            )
        return EnergyScheduleDecision(
            action=ACTION_REFUSE,
            reason_code=REASON_UNTRUSTWORTHY,
            detail=why,
            work_id=None if work_id is None else str(work_id),
            metrics_consulted=tuple(consulted),
            numeric_energy_used=False,
            substituted_default=False,
            claim_class=CLAIM_CLASS,
            admit_implemented=False,
        )


def admit_is_implemented() -> bool:
    """Energy-based Admit is not implemented in this sidecar."""
    return False


# ---------------------------------------------------------------------------
# Capability probes — no root, no password prompt, no GPU lease.
# ---------------------------------------------------------------------------

_PROBE_TIMEOUT_S = 8


def _run(argv: list[str], timeout: float = _PROBE_TIMEOUT_S) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "invoked": True,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-2000:],
            "error": None,
        }
    except FileNotFoundError:
        return {
            "argv": argv,
            "invoked": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "FileNotFoundError",
            "missing_dependency": argv[0],
        }
    except PermissionError as exc:
        return {
            "argv": argv,
            "invoked": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": "PermissionError",
            "missing_dependency": "permission",
        }
    except subprocess.TimeoutExpired:
        return {
            "argv": argv,
            "invoked": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "TimeoutExpired",
            "missing_dependency": None,
        }
    except OSError as exc:
        return {
            "argv": argv,
            "invoked": False,
            "returncode": getattr(exc, "errno", None),
            "stdout": "",
            "stderr": str(exc),
            "error": type(exc).__name__,
            "missing_dependency": "os",
        }


def _clip(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _probe_base(pid: str) -> dict[str, Any]:
    return {
        "id": pid,
        "invoked": False,
        "command_ok": False,
        "succeeded": False,
        "trustworthy_for_token_energy": False,
        "missing_dependency": None,
        "observation": None,
        "numeric_sample_recorded": False,
    }


def probe_powermetrics_without_root() -> dict[str, Any]:
    out = _probe_base("powermetrics_without_root")
    run = _run(["powermetrics", "-n", "1"])
    out["invoked"] = bool(run["invoked"])
    out["command_ok"] = run.get("returncode") == 0
    text = (run.get("stderr") or "") + (run.get("stdout") or "")
    needs_root = "must be invoked as the superuser" in text
    if run.get("error") == "FileNotFoundError":
        out["missing_dependency"] = "powermetrics"
        out["observation"] = "powermetrics binary not found"
        return out
    if needs_root or not out["command_ok"]:
        out["missing_dependency"] = "root"
        out["observation"] = _clip(text) or run.get("error")
        out["succeeded"] = False
        return out
    # A successful non-root powermetrics would still be DIRTY without a lease.
    out["succeeded"] = True
    out["observation"] = "powermetrics returned 0 without root; still not a token joule"
    out["trustworthy_for_token_energy"] = False
    return out


def probe_sudo_n_powermetrics() -> dict[str, Any]:
    """Non-interactive sudo only. Never prompts for a password."""
    out = _probe_base("sudo_n_powermetrics")
    run = _run(["sudo", "-n", "powermetrics", "-n", "1"])
    out["invoked"] = bool(run["invoked"])
    out["command_ok"] = run.get("returncode") == 0
    text = (run.get("stderr") or "") + (run.get("stdout") or "")
    if not run.get("invoked"):
        out["missing_dependency"] = run.get("missing_dependency") or "sudo"
        out["observation"] = _clip(text or str(run.get("error")))
        return out
    if out["command_ok"]:
        out["succeeded"] = True
        out["observation"] = "sudo -n powermetrics returned 0; still not a token wrap"
        out["trustworthy_for_token_energy"] = False
        return out
    out["missing_dependency"] = "root"
    out["observation"] = _clip(text) or run.get("error")
    return out


def probe_pmset_therm() -> dict[str, Any]:
    out = _probe_base("pmset_therm")
    run = _run(["pmset", "-g", "therm"])
    out["invoked"] = bool(run["invoked"])
    out["command_ok"] = run.get("returncode") == 0
    text = (run.get("stdout") or "") + (run.get("stderr") or "")
    if not out["command_ok"]:
        out["missing_dependency"] = "pmset" if not run.get("invoked") else None
        out["observation"] = _clip(text or str(run.get("error")))
        return out
    out["observation"] = _clip(text)
    # Command ran. It is not a thermal measurement.
    out["succeeded"] = False
    out["missing_dependency"] = None
    out["why_not_thermal_state"] = (
        "pmset -g therm reports whether a warning was logged; "
        "'No thermal warning level has been recorded' is not a temperature "
        "or headroom reading"
    )
    return out


def probe_pmset_batt() -> dict[str, Any]:
    out = _probe_base("pmset_batt")
    run = _run(["pmset", "-g", "batt"])
    out["invoked"] = bool(run["invoked"])
    out["command_ok"] = run.get("returncode") == 0
    text = (run.get("stdout") or "") + (run.get("stderr") or "")
    out["observation"] = _clip(text) or run.get("error")
    # Power *source* is not energy. Useful context, not a joule.
    out["succeeded"] = bool(out["command_ok"] and text.strip())
    out["trustworthy_for_token_energy"] = False
    out["why_not_token_energy"] = "power source (AC/battery) is not a joule integral"
    return out


def probe_sysctl_thermal() -> dict[str, Any]:
    out = _probe_base("sysctl_thermal_levels")
    keys = (
        "machdep.xcpm.cpu_thermal_level",
        "machdep.xcpm.gpu_thermal_level",
        "machdep.xcpm.io_thermal_level",
        "machdep.thermal",
    )
    present: dict[str, str] = {}
    absent: list[str] = []
    invoked = False
    for key in keys:
        run = _run(["sysctl", "-n", key])
        invoked = invoked or bool(run.get("invoked"))
        if run.get("returncode") == 0 and (run.get("stdout") or "").strip():
            present[key] = _clip(run["stdout"], 80)
        else:
            absent.append(key)
    out["invoked"] = invoked
    out["command_ok"] = invoked
    out["succeeded"] = bool(present)
    out["observation"] = {
        "present": present,
        "absent": absent,
        "note": (
            "Darwin 27 / Apple Silicon in this session: the historical "
            "machdep.xcpm.* thermal oids are unknown. Absent keys are not "
            "a thermal_state measurement."
        ),
    }
    if not present:
        out["missing_dependency"] = "sysctl_thermal_oids"
    return out


_K_CFSTRING_UTF8 = 0x08000100
_IOREPORT_CACHE: dict[str, Any] | None = None


def _cfstr(cf: Any, text: str) -> Any:
    return cf.CFStringCreateWithCString(None, text.encode(), _K_CFSTRING_UTF8)


def _cf_to_str(cf: Any, ref: Any) -> str | None:
    if not ref:
        return None
    if cf.CFGetTypeID(ref) != cf.CFStringGetTypeID():
        return None
    buf = ctypes.create_string_buffer(256)
    ok = cf.CFStringGetCString(ref, buf, 256, _K_CFSTRING_UTF8)
    if not ok:
        return None
    return buf.value.decode("utf-8", "replace") or None


def _load_ioreport() -> tuple[Any, Any]:
    cf_path = ctypes.util.find_library("CoreFoundation")
    if not cf_path:
        raise RuntimeError("CoreFoundation not found")
    cf = ctypes.cdll.LoadLibrary(cf_path)
    lib = ctypes.CDLL("libIOReport.dylib")
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    cf.CFStringGetCString.restype = ctypes.c_ubyte
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    cf.CFStringGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetTypeID.argtypes = []
    cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    cf.CFDictionaryGetTypeID.restype = ctypes.c_ulong
    cf.CFDictionaryGetTypeID.argtypes = []
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    lib.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
    lib.IOReportCopyChannelsInGroup.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    lib.IOReportCreateSubscription.restype = ctypes.c_void_p
    lib.IOReportCreateSubscription.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    lib.IOReportChannelGetChannelName.restype = ctypes.c_void_p
    lib.IOReportChannelGetChannelName.argtypes = [ctypes.c_void_p]
    lib.IOReportCreateSamples.restype = ctypes.c_void_p
    lib.IOReportCreateSamples.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
    lib.IOReportCreateSamplesDelta.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.IOReportSimpleGetIntegerValue.restype = ctypes.c_uint64
    lib.IOReportSimpleGetIntegerValue.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.IOReportChannelGetFormat.restype = ctypes.c_int
    lib.IOReportChannelGetFormat.argtypes = [ctypes.c_void_p]
    lib.IOReportArrayGetValueAtIndex.restype = ctypes.c_uint64
    lib.IOReportArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.IOReportChannelGetUnitLabel.restype = ctypes.c_void_p
    lib.IOReportChannelGetUnitLabel.argtypes = [ctypes.c_void_p]
    return cf, lib


def _ioreport_inprocess() -> dict[str, Any]:
    """ctypes IOReport probe. May crash the process; call via subprocess."""
    cf, lib = _load_ioreport()
    group = _cfstr(cf, "Energy Model")
    if not group:
        raise RuntimeError("CFString Energy Model failed")
    channels = lib.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
    cf.CFRelease(group)
    if not channels:
        raise RuntimeError("IOReportCopyChannelsInGroup(Energy Model) returned null")
    key = _cfstr(cf, "IOReportChannels")
    arr = cf.CFDictionaryGetValue(channels, key)
    cf.CFRelease(key)
    if not arr:
        raise RuntimeError("Energy Model dict missing IOReportChannels")
    dict_tid = cf.CFDictionaryGetTypeID()
    n = int(cf.CFArrayGetCount(arr))
    names: list[str] = []
    for i in range(n):
        item = cf.CFArrayGetValueAtIndex(arr, i)
        if not item or cf.CFGetTypeID(item) != dict_tid:
            continue
        name = _cf_to_str(cf, lib.IOReportChannelGetChannelName(item))
        if name:
            names.append(name)
    unique = sorted(set(names))
    subbed = ctypes.c_void_p()
    sub = lib.IOReportCreateSubscription(None, channels, ctypes.byref(subbed), 0, None)
    obtained = bool(sub) and bool(subbed)
    return {
        "channel_count": len(names),
        "unique_channel_names": len(unique),
        "gpu_energy_channel_present": "GPU Energy" in unique,
        "dram_channels_present": [n for n in ("DRAM0_0", "DRAM0_1") if n in unique],
        "cpu_energy_channels_present": sorted(
            n for n in unique if n.endswith("CPU Energy")
        ),
        "subscription_obtained": obtained,
        "sandbox_iokit_open_denied": _iokit_open_denied(),
        "libioreport_dlopen_without_root": True,
    }


_ENERGY_SENTINEL = 1 << 63
_ENERGY_CHANNEL_UNITS: dict[str, str] = {
    "GPU Energy": "nJ",
    "DIE_0_CPU Energy": "mJ",
    "DIE_1_CPU Energy": "mJ",
    "DRAM0_0": "mJ",
    "DRAM0_1": "mJ",
}


def _ioreport_open(cf: Any, lib: Any) -> dict[str, Any]:
    """One Energy Model subscription. May crash; call via subprocess."""
    group = _cfstr(cf, "Energy Model")
    if not group:
        return {"subscription_obtained": False, "error": "CFString Energy Model failed"}
    channels = lib.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
    cf.CFRelease(group)
    if not channels:
        return {
            "subscription_obtained": False,
            "error": "IOReportCopyChannelsInGroup(Energy Model) returned null",
        }
    subbed = ctypes.c_void_p()
    sub = lib.IOReportCreateSubscription(None, channels, ctypes.byref(subbed), 0, None)
    if not sub or not subbed:
        denied = _iokit_open_denied()
        why = "IOReportCreateSubscription returned null"
        if denied:
            why += (
                "; sandbox_check(iokit-open-user-client)=denied. "
                "CopyChannelsInGroup still works (catalog); live samples need "
                "an IOKit user client this seatbelt profile does not allow."
            )
        return {
            "subscription_obtained": False,
            "error": why,
            "sandbox_iokit_open_denied": denied,
        }
    return {
        "subscription_obtained": True,
        "error": None,
        "channels": channels,
        "sub": sub,
        "subbed": subbed,
    }


def _ioreport_read_channels(cf: Any, lib: Any, sub: Any, subbed: Any) -> dict[str, Any]:
    """Read wanted Energy Model channels from one CreateSamples snapshot."""
    samples = lib.IOReportCreateSamples(sub, subbed, None)
    if not samples:
        return {"error": "IOReportCreateSamples returned null", "channels": {}}
    key = _cfstr(cf, "IOReportChannels")
    arr = cf.CFDictionaryGetValue(samples, key)
    cf.CFRelease(key)
    if not arr:
        return {"error": "samples missing IOReportChannels", "channels": {}, "samples": samples}
    dict_tid = cf.CFDictionaryGetTypeID()
    n = int(cf.CFArrayGetCount(arr))
    wanted = set(_ENERGY_CHANNEL_UNITS)
    out: dict[str, dict[str, Any]] = {}
    for i in range(n):
        item = cf.CFArrayGetValueAtIndex(arr, i)
        if not item or cf.CFGetTypeID(item) != dict_tid:
            continue
        name = _cf_to_str(cf, lib.IOReportChannelGetChannelName(item))
        if not name:
            continue
        if name not in wanted and not name.endswith("CPU Energy"):
            continue
        ok = ctypes.c_int(0)
        raw = int(lib.IOReportSimpleGetIntegerValue(item, ctypes.byref(ok)))
        valid = int(ok.value) > 0 and raw != _ENERGY_SENTINEL
        fmt = None
        array_sum = None
        try:
            fmt = int(lib.IOReportChannelGetFormat(item))
        except Exception:  # noqa: BLE001
            fmt = None
        if fmt == 4:
            total = 0
            for idx in range(16):
                total += int(lib.IOReportArrayGetValueAtIndex(item, idx))
            array_sum = total
            if not valid and array_sum != 0:
                raw = array_sum
                valid = True
        unit_label = None
        try:
            unit_label = _cf_to_str(cf, lib.IOReportChannelGetUnitLabel(item))
        except Exception:  # noqa: BLE001
            unit_label = None
        out[name] = {
            "raw": raw if valid else None,
            "ok": int(ok.value),
            "unit": _ENERGY_CHANNEL_UNITS.get(name, "unknown"),
            "unit_label": unit_label,
            "format": fmt,
            "array_sum": array_sum,
            "valid": valid,
        }
    return {"error": None, "channels": out, "samples": samples}


def _cpu_burn(seconds: float) -> int:
    """Tight CPU loop. Does not touch the GPU; will not disturb a resident."""
    n = 0
    end = time.perf_counter() + max(0.0, seconds)
    while time.perf_counter() < end:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Process-attributed energy: proc_pid_rusage ri_energy_nj
# sys/resource.h RUSAGE_INFO_V6. Call site is measure_process_energy().
# ---------------------------------------------------------------------------

RUSAGE_INFO_V6 = 6
PROC_ALL_PIDS = 1
_SANDBOX_FILTER_NONE = 0
_ENERGY_NJ_QUANTIZATION_J = 1.0e-9
_TIMING_UNCERTAINTY_S = 0.001
_USEFUL_WORK_DENOMINATOR = "cpu_burn_iterations"
_USEFUL_WORK_DEFENSE = (
    "The only useful work this sidecar performed is the CPU-burn loop "
    "counted in the busy window. Tokens were not emitted and WorkUnits "
    "were not sealed, so J/token and WU/kWh stay UNKNOWN/COST_MODEL. "
    "Process CPU-seconds is a time axis (restating watts), not a work "
    "axis. Iterations are counted on the same closed wall as the joule "
    "integral. This is a calibration of integer-loop work on this M3 "
    "Ultra, not a token and not a WorkUnit."
)

# Contention name/path needles. argv matching uses pgrep -lf.
_CONTENTION_PGREP = (
    "hawkingd",
    "ascension_qwen38_resident",
    "hcli.agentos.resident",
)


class _RusageInfoV6(ctypes.Structure):
    """Layout from MacOSX.sdk usr/include/sys/resource.h rusage_info_v6."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
        ("ri_flags", ctypes.c_uint64),
        ("ri_user_ptime", ctypes.c_uint64),
        ("ri_system_ptime", ctypes.c_uint64),
        ("ri_pinstructions", ctypes.c_uint64),
        ("ri_pcycles", ctypes.c_uint64),
        ("ri_energy_nj", ctypes.c_uint64),
        ("ri_penergy_nj", ctypes.c_uint64),
        ("ri_secure_time_in_system", ctypes.c_uint64),
        ("ri_secure_ptime_in_system", ctypes.c_uint64),
        ("ri_neural_footprint", ctypes.c_uint64),
        ("ri_lifetime_max_neural_footprint", ctypes.c_uint64),
        ("ri_interval_max_neural_footprint", ctypes.c_uint64),
        ("ri_conclave_footprint", ctypes.c_uint64),
        ("ri_page_wait_time_mach", ctypes.c_uint64),
        ("ri_page_cache_hits", ctypes.c_uint64),
        ("ri_reserved", ctypes.c_uint64 * 6),
    ]


def _libsystem() -> Any:
    return ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def _iokit_open_denied() -> bool | None:
    """sandbox_check: 0 allowed, 1 denied, -1 unknown op / error."""
    try:
        libc = _libsystem()
        libc.sandbox_check.restype = ctypes.c_int
        rc = libc.sandbox_check(
            os.getpid(), b"iokit-open-user-client", _SANDBOX_FILTER_NONE
        )
        if rc == 1:
            return True
        if rc == 0:
            return False
        return None
    except Exception:  # noqa: BLE001
        return None


def proc_pid_rusage(pid: int) -> dict[str, Any] | None:
    """One RUSAGE_INFO_V6 snapshot. Returns None if the syscall fails.

    Call site of the kernel gate: this function invokes proc_pid_rusage.
    measure_process_energy calls this; tests call measure_process_energy
    and also this symbol directly.
    """
    libc = _libsystem()
    libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.proc_pid_rusage.restype = ctypes.c_int
    info = _RusageInfoV6()
    rc = int(libc.proc_pid_rusage(int(pid), RUSAGE_INFO_V6, ctypes.byref(info)))
    if rc != 0:
        return None
    return {
        "pid": int(pid),
        "rc": rc,
        "energy_nj": int(info.ri_energy_nj),
        "penergy_nj": int(info.ri_penergy_nj),
        "billed_energy": int(info.ri_billed_energy),
        "serviced_energy": int(info.ri_serviced_energy),
        "cycles": int(info.ri_cycles),
        "instructions": int(info.ri_instructions),
        "pcycles": int(info.ri_pcycles),
        "pinstructions": int(info.ri_pinstructions),
        "neural_footprint": int(info.ri_neural_footprint),
        "phys_footprint": int(info.ri_phys_footprint),
    }


def _pgrep_lf(pattern: str) -> list[dict[str, Any]]:
    run = _run(["/usr/bin/pgrep", "-lf", pattern], timeout=5)
    rows: list[dict[str, Any]] = []
    if run.get("returncode") not in (0, 1):
        return rows
    for line in (run.get("stdout") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, rest = line.partition(" ")
        if not pid_s.isdigit():
            continue
        rows.append({"pid": int(pid_s), "command": rest[:240], "pattern": pattern})
    return rows


def observe_contention() -> dict[str, Any]:
    """Read-only snapshot of other energy consumers. Never signals a pid."""
    denied = _iokit_open_denied()
    try:
        load = list(os.getloadavg())
    except (OSError, AttributeError):
        load = None
    peers: list[dict[str, Any]] = []
    seen: set[int] = set()
    hawkingd_present = False
    resident_present = False
    for pattern in _CONTENTION_PGREP:
        for row in _pgrep_lf(pattern):
            pid = row["pid"]
            if pid in seen or pid == os.getpid():
                continue
            seen.add(pid)
            if pattern == "hawkingd":
                hawkingd_present = True
            if pattern == "ascension_qwen38_resident":
                resident_present = True
            snap = proc_pid_rusage(pid)
            peers.append(
                {
                    "pid": pid,
                    "command": row["command"],
                    "matched": pattern,
                    "energy_nj": None if snap is None else snap["energy_nj"],
                    "cycles": None if snap is None else snap["cycles"],
                    "phys_footprint": None if snap is None else snap["phys_footprint"],
                    "rusage_ok": snap is not None,
                }
            )
    return {
        "self_pid": os.getpid(),
        "loadavg": load,
        "sandbox_iokit_open_user_client_denied": denied,
        "hawkingd_process_present": hawkingd_present,
        "ascension_resident_present": resident_present,
        "peers": peers[:16],
        "signaled": False,
        "gpu_touched": False,
        "why": (
            "proc_pid_rusage and pgrep -lf are read-only. No process was "
            "killed, restarted, or sent a signal. IOReport live samples "
            "need iokit-open-user-client; when that is denied the GPU/CPU "
            "package rails are not this measurement. ri_energy_nj is "
            "task-attributed, so peer energy is not in our numerator; "
            "peers still contend for the same package (DIRTY frequency/"
            "thermal)."
        ),
    }


def _mean(xs: Sequence[float]) -> float | None:
    return None if not xs else float(sum(xs) / len(xs))


def _stdev(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def _window_from_rusage(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    window_s: float,
    label: str,
    cpu_burn_iters: int,
) -> dict[str, Any]:
    delta_nj = int(b["energy_nj"]) - int(a["energy_nj"])
    joules = delta_nj / 1.0e9
    watts = None if window_s <= 0 else joules / window_s
    return {
        "label": label,
        "window_s": window_s,
        "energy_nj_t0": int(a["energy_nj"]),
        "energy_nj_t1": int(b["energy_nj"]),
        "delta_nj": delta_nj,
        "joules": joules,
        "watts": watts,
        "penergy_delta_nj": int(b["penergy_nj"]) - int(a["penergy_nj"]),
        "cycles_delta": int(b["cycles"]) - int(a["cycles"]),
        "instructions_delta": int(b["instructions"]) - int(a["instructions"]),
        "neural_footprint_t1": int(b["neural_footprint"]),
        "cpu_burn_iters": cpu_burn_iters,
        "evidence_tier": TIER_HARDWARE_MEASURED,
        "increments": delta_nj > 0,
    }


def measure_process_energy(
    *,
    idle_s: float = 0.25,
    busy_s: float = 0.25,
    repeats: int = 1,
    contention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Idle vs CPU-busy process energy on THIS pid via proc_pid_rusage.

    Production call site: build() -> roadmap_categories -> this function.
    Tests call this symbol directly. GPU is not touched. The live resident
    is observed with a read-only rusage snapshot, never signaled.

    Raw energy_nj bookends are HARDWARE_MEASURED. Watts and J/iteration
    are the same integral over the measured window / the counted
    iterations — not a TDP model. J/token is not this function.
    """
    pid = os.getpid()
    if contention is None:
        contention = observe_contention()
    method = {
        "source": "proc_pid_rusage",
        "flavor": "RUSAGE_INFO_V6",
        "field": "ri_energy_nj",
        "field_unit": "nJ",
        "header": (
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/"
            "usr/include/sys/resource.h rusage_info_v6.ri_energy_nj"
        ),
        "sampling": (
            "bookend snapshots at each window boundary; not a polling rate. "
            f"Requested idle_s={idle_s}, busy_s={busy_s}, repeats={repeats}."
        ),
        "workload": (
            "idle = time.sleep; busy = _cpu_burn tight integer increment. "
            "GPU untouched; no Metal, no TOKEN_NS wrap, no resident call."
        ),
        "clock": "time.perf_counter",
        "pid": pid,
    }
    s0 = proc_pid_rusage(pid)
    if s0 is None:
        return {
            "ok": False,
            "error": "proc_pid_rusage failed on self",
            "evidence_tier": TIER_STATIC,
            "gpu_touched": False,
            "method": method,
            "contention": dict(contention),
        }

    resident_pids = [
        p["pid"]
        for p in (contention.get("peers") or [])
        if p.get("matched") == "ascension_qwen38_resident"
    ]
    resident_t0 = proc_pid_rusage(resident_pids[0]) if resident_pids else None

    repeats_n = max(1, int(repeats))
    idle_s = max(0.05, float(idle_s))
    busy_s = max(0.05, float(busy_s))
    rows: list[dict[str, Any]] = []
    for _ in range(repeats_n):
        a = proc_pid_rusage(pid)
        t0 = time.perf_counter()
        time.sleep(idle_s)
        t1 = time.perf_counter()
        b = proc_pid_rusage(pid)
        if a is None or b is None:
            continue
        idle = _window_from_rusage(
            a, b, window_s=t1 - t0, label="idle", cpu_burn_iters=0
        )
        t_busy0 = time.perf_counter()
        iters = _cpu_burn(busy_s)
        t_busy1 = time.perf_counter()
        c = proc_pid_rusage(pid)
        if c is None:
            continue
        busy = _window_from_rusage(
            b, c, window_s=t_busy1 - t_busy0, label="cpu_busy_gpu_untouched",
            cpu_burn_iters=iters,
        )
        idle_watts = idle["watts"]
        work_j = None
        if idle_watts is not None and busy["joules"] is not None:
            work_j = float(busy["joules"]) - float(idle_watts) * float(busy["window_s"])
        j_per_iter = None
        if work_j is not None and iters > 0:
            j_per_iter = work_j / float(iters)
        rows.append(
            {
                "idle": idle,
                "busy": busy,
                "work_joules_idle_subtracted": work_j,
                "joules_per_iteration": j_per_iter,
            }
        )

    resident_t1 = proc_pid_rusage(resident_pids[0]) if resident_pids else None
    resident_during = None
    if resident_t0 and resident_t1:
        resident_during = {
            "pid": resident_pids[0],
            "energy_nj_t0": resident_t0["energy_nj"],
            "energy_nj_t1": resident_t1["energy_nj"],
            "delta_nj": int(resident_t1["energy_nj"]) - int(resident_t0["energy_nj"]),
            "cycles_delta": int(resident_t1["cycles"]) - int(resident_t0["cycles"]),
            "evidence_tier": TIER_HARDWARE_MEASURED,
            "why": (
                "proc_pid_rusage on ascension_qwen38_resident over the same "
                "wall as this sidecar's idle+busy windows. Read-only; the "
                "resident was not signaled."
            ),
        }

    idle_j = [r["idle"]["joules"] for r in rows]
    busy_j = [r["busy"]["joules"] for r in rows]
    idle_w = [r["idle"]["watts"] for r in rows if r["idle"]["watts"] is not None]
    busy_w = [r["busy"]["watts"] for r in rows if r["busy"]["watts"] is not None]
    work_j = [r["work_joules_idle_subtracted"] for r in rows if r["work_joules_idle_subtracted"] is not None]
    j_iter = [r["joules_per_iteration"] for r in rows if r["joules_per_iteration"] is not None]
    iters = [r["busy"]["cpu_burn_iters"] for r in rows]
    idle_win = [r["idle"]["window_s"] for r in rows]
    busy_win = [r["busy"]["window_s"] for r in rows]

    mean_idle_j, mean_busy_j = _mean(idle_j), _mean(busy_j)
    mean_idle_w, mean_busy_w = _mean(idle_w), _mean(busy_w)
    mean_work_j = _mean(work_j)
    mean_j_iter = _mean(j_iter)
    mean_iters = _mean(iters)
    mean_idle_win, mean_busy_win = _mean(idle_win), _mean(busy_win)
    stdev_work_j = _stdev(work_j)
    stdev_busy_j = _stdev(busy_j)

    watts_for_budget = mean_busy_w if mean_busy_w is not None else 0.0
    timing_j = abs(watts_for_budget) * _TIMING_UNCERTAINTY_S
    empirical_j = (2.0 * stdev_work_j) if stdev_work_j is not None else 0.0
    error_budget_j = max(_ENERGY_NJ_QUANTIZATION_J, timing_j, empirical_j)
    error_budget_j_iter = (
        None if not mean_iters or mean_iters <= 0 else error_budget_j / mean_iters
    )

    differential_ok = (
        mean_busy_j is not None
        and mean_idle_j is not None
        and mean_busy_j > mean_idle_j
        and any(r["busy"]["increments"] for r in rows)
    )
    iterations_per_kwh = None
    if mean_work_j is not None and mean_work_j > 0 and mean_iters is not None:
        kwh = mean_work_j / 3.6e6
        if kwh > 0:
            iterations_per_kwh = mean_iters / kwh

    return {
        "ok": bool(rows) and differential_ok,
        "error": None if rows else "no successful rusage windows",
        "evidence_tier": TIER_HARDWARE_MEASURED if rows else TIER_STATIC,
        "gpu_touched": False,
        "token_ns_wrap": False,
        "pid": pid,
        "method": method,
        "error_budget": {
            "quantization_j": _ENERGY_NJ_QUANTIZATION_J,
            "timing_uncertainty_s": _TIMING_UNCERTAINTY_S,
            "timing_term_j": timing_j,
            "repeat_stdev_work_j": stdev_work_j,
            "repeat_stdev_busy_j": stdev_busy_j,
            "combined_j": error_budget_j,
            "combined_j_per_iteration": error_budget_j_iter,
            "rule": (
                "max(1 nJ, |busy_watts| * 1 ms, 2 * sample_stdev of "
                "idle-subtracted work joules). Bookend sampling, not a Hz rate."
            ),
            "repeats": repeats_n,
        },
        "contention": dict(contention),
        "resident_during_measurement": resident_during,
        "repeats": rows,
        "idle": {
            "mean_joules": mean_idle_j,
            "mean_watts": mean_idle_w,
            "mean_window_s": mean_idle_win,
            "stdev_joules": _stdev(idle_j),
            "evidence_tier": TIER_HARDWARE_MEASURED,
            "raw_field": "ri_energy_nj",
        },
        "busy": {
            "mean_joules": mean_busy_j,
            "mean_watts": mean_busy_w,
            "mean_window_s": mean_busy_win,
            "mean_cpu_burn_iters": mean_iters,
            "stdev_joules": stdev_busy_j,
            "evidence_tier": TIER_HARDWARE_MEASURED,
            "raw_field": "ri_energy_nj",
        },
        "differential": {
            "id": "idle_vs_active_joules",
            "mean_work_joules": mean_work_j,
            "mean_busy_minus_mean_idle_joules": (
                None
                if mean_busy_j is None or mean_idle_j is None
                else mean_busy_j - mean_idle_j
            ),
            "busy_gt_idle": differential_ok,
            "evidence_tier": TIER_HARDWARE_MEASURED if differential_ok else TIER_STATIC,
            "why": (
                "Both windows are proc_pid_rusage ri_energy_nj on this pid. "
                "work_joules = busy_joules - idle_watts * busy_window_s. "
                "That is a subtraction of two samples, not of two guesses."
            ),
        },
        "useful_work": {
            "denominator": _USEFUL_WORK_DENOMINATOR,
            "denominator_definition": (
                "Iterations of a tight integer increment loop completed by "
                "this process during the busy window (_cpu_burn)."
            ),
            "defense": _USEFUL_WORK_DEFENSE,
            "mean_iterations": mean_iters,
            "joules_per_iteration": mean_j_iter,
            "iterations_per_kwh": iterations_per_kwh,
            "error_budget_j_per_iteration": error_budget_j_iter,
            "evidence_tier": TIER_HARDWARE_MEASURED if mean_j_iter is not None else TIER_STATIC,
            "not": (
                "J/token, J/accepted-token, WU/kWh — those denominators "
                "were not produced in this interval."
            ),
        },
        "billed_energy_unused": {
            "ri_billed_energy": s0["billed_energy"],
            "ri_serviced_energy": s0["serviced_energy"],
            "why": (
                "ri_billed_energy and ri_serviced_energy were 0 on this "
                "host in probe; they are not the measurement."
            ),
        },
    }


def _delta_to_watts(delta_raw: int | None, unit: str, window_s: float) -> float | None:
    if delta_raw is None or delta_raw <= 0 or window_s <= 0:
        return None
    if unit == "nJ":
        joules = delta_raw / 1.0e9
    elif unit == "mJ":
        joules = delta_raw / 1.0e3
    elif unit == "J":
        joules = float(delta_raw)
    else:
        return None
    return joules / window_s


def _ioreport_sample_inprocess(idle_s: float, busy_s: float) -> dict[str, Any]:
    """Idle then CPU-busy IOReport windows on ONE subscription. GPU untouched."""
    cf, lib = _load_ioreport()
    opened = _ioreport_open(cf, lib)
    if not opened.get("subscription_obtained"):
        return {
            "subscription_obtained": False,
            "error": opened.get("error"),
            "sandbox_iokit_open_denied": opened.get("sandbox_iokit_open_denied"),
            "idle": None,
            "cpu_busy": None,
        }
    sub, subbed = opened["sub"], opened["subbed"]
    t_wall0 = time.perf_counter()
    s0 = _ioreport_read_channels(cf, lib, sub, subbed)
    time.sleep(max(0.05, idle_s))
    t_wall1 = time.perf_counter()
    s1 = _ioreport_read_channels(cf, lib, sub, subbed)
    iters = _cpu_burn(max(0.05, busy_s))
    t_wall2 = time.perf_counter()
    s2 = _ioreport_read_channels(cf, lib, sub, subbed)

    delta_idle = None
    delta_busy = None
    try:
        if s0.get("samples") and s1.get("samples"):
            delta_idle = bool(lib.IOReportCreateSamplesDelta(s0["samples"], s1["samples"], None))
        if s1.get("samples") and s2.get("samples"):
            delta_busy = bool(lib.IOReportCreateSamplesDelta(s1["samples"], s2["samples"], None))
    except Exception as exc:  # noqa: BLE001
        delta_idle = f"delta_exc:{type(exc).__name__}"

    def _window(a: dict[str, Any], b: dict[str, Any], dt: float, label: str) -> dict[str, Any]:
        ch_a = a.get("channels") or {}
        ch_b = b.get("channels") or {}
        rows = []
        for name in sorted(set(ch_a) | set(ch_b)):
            ua = ch_a.get(name) or {}
            ub = ch_b.get(name) or {}
            raw0, raw1 = ua.get("raw"), ub.get("raw")
            unit = ub.get("unit") or ua.get("unit") or "unknown"
            delta = None if raw0 is None or raw1 is None else int(raw1) - int(raw0)
            watts = _delta_to_watts(delta, unit, dt)
            rows.append(
                {
                    "channel": name,
                    "unit": unit,
                    "raw_t0": raw0,
                    "raw_t1": raw1,
                    "delta": delta,
                    "window_s": dt,
                    "watts": watts,
                    "increments": bool(delta is not None and delta > 0),
                    "format": ub.get("format"),
                }
            )
        return {"label": label, "window_s": dt, "channels": rows}

    return {
        "subscription_obtained": True,
        "error": s0.get("error") or s1.get("error") or s2.get("error"),
        "cpu_burn_iters": iters,
        "idle": _window(s0, s1, t_wall1 - t_wall0, "idle"),
        "cpu_busy": _window(s1, s2, t_wall2 - t_wall1, "cpu_busy_gpu_untouched"),
        "create_samples_delta_idle": delta_idle,
        "create_samples_delta_busy": delta_busy,
        "note": (
            "GPU-token-active is not sampled: this sidecar has no GPU lease and "
            "must not disturb the live resident. cpu_busy is CPU-only. "
            "One subscription, three CreateSamples, optional CreateSamplesDelta."
        ),
    }


def _ioreport_via_subprocess() -> dict[str, Any]:
    """Isolate IOReport ctypes. A segfault becomes a failed probe, not a crash."""
    global _IOREPORT_CACHE
    if _IOREPORT_CACHE is not None:
        return _IOREPORT_CACHE
    if sys.platform != "darwin":
        _IOREPORT_CACHE = {"error": "macos-only", "missing_dependency": "macos"}
        return _IOREPORT_CACHE
    env = dict(_os.environ)
    env["HAWKING_GREEN_MACHINE_IOREPORT_WORKER"] = "1"
    run = subprocess.run(
        [sys.executable, __file__, "--ioreport-worker"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
        check=False,
        env=env,
    )
    if run.returncode != 0:
        _IOREPORT_CACHE = {
            "error": (
                f"ioreport worker exit {run.returncode}: "
                + _clip((run.stderr or "") + (run.stdout or ""), 400)
            ),
            "missing_dependency": "IOReportCreateSubscription_or_ctypes",
            "crashed": run.returncode < 0 or run.returncode == 139,
        }
        return _IOREPORT_CACHE
    try:
        _IOREPORT_CACHE = json.loads(run.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        _IOREPORT_CACHE = {
            "error": "ioreport worker produced non-JSON: " + _clip(run.stdout, 400),
            "missing_dependency": "libIOReport",
        }
    return _IOREPORT_CACHE


def probe_ioreport_catalog() -> dict[str, Any]:
    out = _probe_base("ioreport_energy_model_catalog")
    if sys.platform != "darwin":
        out["missing_dependency"] = "macos"
        out["observation"] = "IOReport Energy Model is macOS-only"
        return out
    raw = _ioreport_via_subprocess()
    out["invoked"] = True
    out["numeric_sample_recorded"] = False
    out["trustworthy_for_token_energy"] = False
    if raw.get("error") and "channel_count" not in raw:
        out["command_ok"] = False
        out["succeeded"] = False
        out["missing_dependency"] = raw.get("missing_dependency") or "libIOReport"
        out["observation"] = raw.get("error")
        return out
    out["command_ok"] = True
    out["succeeded"] = bool(raw.get("channel_count"))
    out["observation"] = {
        "libioreport_dlopen_without_root": bool(raw.get("libioreport_dlopen_without_root")),
        "channel_count": raw.get("channel_count"),
        "unique_channel_names": raw.get("unique_channel_names"),
        "gpu_energy_channel_present": bool(raw.get("gpu_energy_channel_present")),
        "dram_channels_present": raw.get("dram_channels_present") or [],
        "cpu_energy_channels_present": raw.get("cpu_energy_channels_present") or [],
    }
    out["why_not_token_energy"] = (
        "a catalog of channel names is not a joule integral over a token interval"
    )
    return out


def probe_ioreport_subscription() -> dict[str, Any]:
    out = _probe_base("ioreport_energy_model_subscription")
    if sys.platform != "darwin":
        out["missing_dependency"] = "macos"
        out["observation"] = "IOReport Energy Model is macOS-only"
        return out
    raw = _ioreport_via_subprocess()
    out["invoked"] = True
    out["numeric_sample_recorded"] = False
    out["trustworthy_for_token_energy"] = False
    if raw.get("error") and "subscription_obtained" not in raw:
        out["command_ok"] = False
        out["succeeded"] = False
        out["missing_dependency"] = raw.get("missing_dependency") or "libIOReport"
        out["observation"] = raw.get("error")
        return out
    obtained = bool(raw.get("subscription_obtained"))
    out["command_ok"] = True
    out["succeeded"] = obtained
    denied = raw.get("sandbox_iokit_open_denied")
    if obtained:
        out["observation"] = {
            "subscription_obtained": True,
            "sandbox_iokit_open_denied": denied,
            "note": (
                "Live IOReport samples are still not joules_per_token: this "
                "sidecar has no GPU lease and does not wrap TOKEN_NS. "
                "Process energy is measured via proc_pid_rusage, not here."
            ),
        }
        return out
    out["missing_dependency"] = "IOReportCreateSubscription"
    out["observation"] = {
        "subscription_obtained": False,
        "sandbox_iokit_open_denied": denied,
        "error": "IOReportCreateSubscription returned null",
        "note": (
            "energy.rs documents GPU Energy (nJ) incrementing without root "
            "on 2026-08-16. This seatbelt profile denies iokit-open-user-client "
            f"(denied={denied!r}), so this process cannot subscribe. "
            "The channel catalog still holds. Process ri_energy_nj is the "
            "measurement this sidecar actually took."
        ),
    }
    return out


def probe_ioreg_power_telemetry() -> dict[str, Any]:
    """AppleSmartBattery PowerTelemetryData exists on this desktop as ESTIMATES."""
    out = _probe_base("ioreg_power_telemetry")
    run = _run(["ioreg", "-r", "-n", "AppleSmartBattery", "-l"])
    out["invoked"] = bool(run["invoked"])
    out["command_ok"] = run.get("returncode") == 0
    text = run.get("stdout") or ""
    has_block = "PowerTelemetryData" in text
    named_estimate = "AccumulatedWallEnergyEstimate" in text
    out["succeeded"] = False
    out["observation"] = {
        "power_telemetry_block_present": has_block,
        "named_estimate": named_estimate,
        "why_untrustworthy": (
            "Keys are named Estimate; PowerTelemetryErrorCount is nonzero on "
            "this machine; AppleSmartBattery is a stub on this desktop "
            "(capacity/voltage/amperage 0). Whole-system and not wrapped "
            "around TOKEN_NS. Recording the milliwatt fields would be "
            "fantasy precision."
        ),
    }
    out["numeric_sample_recorded"] = False
    out["trustworthy_for_token_energy"] = False
    if not has_block:
        out["missing_dependency"] = "AppleSmartBattery.PowerTelemetryData"
    return out


PROBES: tuple[Any, ...] = (
    probe_powermetrics_without_root,
    probe_sudo_n_powermetrics,
    probe_pmset_therm,
    probe_pmset_batt,
    probe_sysctl_thermal,
    probe_ioreport_catalog,
    probe_ioreport_subscription,
    probe_ioreg_power_telemetry,
)


def run_probes() -> list[dict[str, Any]]:
    rows = [p() for p in PROBES]
    for row in rows:
        if row.get("trustworthy_for_token_energy"):
            # A probe must not declare token-energy trust without a lease.
            row["trustworthy_for_token_energy"] = False
            row["trustworthy_overridden"] = (
                "sidecar has no GPU lease; token-energy trust forced False"
            )
    return rows


class TierHonestyError(ValueError):
    """A COST_MODEL (or absent-hardware) value was labeled HARDWARE_MEASURED."""


def _cat_value(
    vid: str,
    value: Any,
    *,
    unit: str | None,
    evidence_tier: str,
    why: str,
    **extra: Any,
) -> dict[str, Any]:
    if evidence_tier not in EVIDENCE_TIERS:
        raise TierHonestyError(f"{vid}: unknown evidence_tier={evidence_tier!r}")
    row: dict[str, Any] = {
        "id": vid,
        "value": value,
        "unit": unit,
        "evidence_tier": evidence_tier,
        "why": why,
    }
    row.update(extra)
    return row


def measure_idle_vs_active_cpu(idle_s: float = 0.2, busy_s: float = 0.2) -> dict[str, Any]:
    """Process CPU-seconds idle vs a CPU burn. Not joules. HARDWARE_MEASURED."""
    p0, w0 = time.process_time(), time.perf_counter()
    time.sleep(max(0.05, idle_s))
    p1, w1 = time.process_time(), time.perf_counter()
    iters = _cpu_burn(max(0.05, busy_s))
    p2, w2 = time.process_time(), time.perf_counter()
    return {
        "idle_window_s": w1 - w0,
        "idle_process_cpu_s": p1 - p0,
        "active_window_s": w2 - w1,
        "active_process_cpu_s": p2 - p1,
        "cpu_burn_iters": iters,
        "gpu_touched": False,
        "evidence_tier": TIER_HARDWARE_MEASURED,
        "why": (
            "time.process_time() and time.perf_counter() on this process. "
            "CPU-seconds, not joules. GPU was not touched."
        ),
    }


def measure_host_identity() -> dict[str, Any]:
    """sysctl / pmset observations of THIS machine. Not a joule integral."""
    def _sysctl(key: str) -> str | None:
        run = _run(["sysctl", "-n", key])
        if run.get("returncode") == 0:
            text = (run.get("stdout") or "").strip()
            return text or None
        return None

    ncpu_s = _sysctl("hw.ncpu")
    brand = _sysctl("machdep.cpu.brand_string")
    mem_s = _sysctl("hw.memsize")
    batt = probe_pmset_batt()
    therm = probe_pmset_therm()
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = None
    ncpu = int(ncpu_s) if ncpu_s and ncpu_s.isdigit() else None
    mem_bytes = int(mem_s) if mem_s and mem_s.isdigit() else None
    power_source = None
    obs = batt.get("observation")
    if isinstance(obs, str) and "AC Power" in obs:
        power_source = "AC Power"
    elif isinstance(obs, str) and "Battery" in obs:
        power_source = "Battery"
    warning = None
    therm_obs = therm.get("observation")
    if isinstance(therm_obs, str):
        if "No thermal warning level has been recorded" in therm_obs:
            warning = False
        else:
            warning = "thermal warning" in therm_obs.lower()
    return {
        "chip": brand,
        "ncpu": ncpu,
        "mem_bytes": mem_bytes,
        "power_source": power_source,
        "thermal_warning_recorded": warning,
        "thermal_warning_observation": therm_obs if isinstance(therm_obs, str) else None,
        "loadavg": list(load) if load is not None else None,
        "evidence_tier": TIER_HARDWARE_MEASURED,
        "why": (
            "sysctl hw.ncpu/brand/memsize, pmset -g batt, pmset -g therm, "
            "os.getloadavg. Die temperature is not among these readings."
        ),
    }


def sample_energy_rails(idle_s: float = 0.25, busy_s: float = 0.25) -> dict[str, Any]:
    """IOReport Energy Model idle vs CPU-busy, isolated in a subprocess."""
    if sys.platform != "darwin":
        return {
            "subscription_obtained": False,
            "error": "macos-only",
            "idle": None,
            "cpu_busy": None,
        }
    env = dict(_os.environ)
    env["HAWKING_GREEN_MACHINE_IOREPORT_SAMPLE"] = "1"
    run = subprocess.run(
        [
            sys.executable,
            __file__,
            "--ioreport-sample",
            f"--idle-s={idle_s}",
            f"--busy-s={busy_s}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=25,
        check=False,
        env=env,
    )
    if run.returncode != 0:
        return {
            "subscription_obtained": False,
            "error": (
                f"ioreport sample worker exit {run.returncode}: "
                + _clip((run.stderr or "") + (run.stdout or ""), 400)
            ),
            "idle": None,
            "cpu_busy": None,
        }
    try:
        return json.loads(run.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "subscription_obtained": False,
            "error": "ioreport sample worker produced non-JSON: " + _clip(run.stdout, 400),
            "idle": None,
            "cpu_busy": None,
        }


def _channel_watts(window: dict[str, Any] | None, channel: str) -> dict[str, Any] | None:
    if not window:
        return None
    for row in window.get("channels") or []:
        if row.get("channel") == channel:
            return row
    return None


def cite_token_interval() -> dict[str, Any]:
    """Cited ms/token from TOKEN_NS_OBJECTIVE. Not a wrap this sidecar took."""
    rel = TOKEN_INTERVAL_RECEIPT
    path = REPO / rel
    if not path.is_file():
        return _cat_value(
            "cited_token_interval",
            None,
            unit="ms/token",
            evidence_tier=TIER_STATIC,
            why="TOKEN_NS_OBJECTIVE.json is not on disk in this worktree",
            cited_from=rel,
        )
    doc = load_json(path)
    current = doc.get("current") if isinstance(doc.get("current"), dict) else {}
    ms = current.get("ms_per_token")
    return _cat_value(
        "cited_token_interval",
        ms if isinstance(ms, (int, float)) else None,
        unit="ms/token",
        evidence_tier=TIER_STATIC,
        why=(
            "Cited from TOKEN_NS_OBJECTIVE.json current.ms_per_token. That "
            "receipt is not a TOKEN_NS wrap by this sidecar; multiplying it "
            "by a wattage to form J/token is COST_MODEL."
        ),
        cited_from=rel,
        cited_evidence_class=doc.get("evidence_class") or doc.get("claim_boundary"),
        token_ns_wrap=False,
    )


def decide_power_cap(
    *,
    proposed_cap_watts: float,
    power_value: Mapping[str, Any],
) -> dict[str, Any]:
    """II-E subgene: power caps only when measured.

    Call site: build() -> decide_power_cap. A COST_MODEL wattage is not a cap.
    """
    tier = power_value.get("evidence_tier")
    watts = power_value.get("value")
    if tier != TIER_HARDWARE_MEASURED or not isinstance(watts, (int, float)):
        return {
            "action": ACTION_REFUSE,
            "reason_code": "POWER_CAP_REQUIRES_MEASUREMENT",
            "detail": (
                "power caps only when measured; "
                f"power evidence_tier={tier!r} value={watts!r}"
            ),
            "evidence_tier": TIER_FUNCTIONAL_SIM,
            "applied_cap_watts": None,
            "proposed_cap_watts": proposed_cap_watts,
            "numeric_cap_applied": False,
        }
    return {
        "action": ACTION_REFUSE,
        "reason_code": "NO_GPU_LEASE_FOR_ENFORCEMENT",
        "detail": (
            "power is HARDWARE_MEASURED but this sidecar has no GPU lease "
            "and will not enforce a cap on the live machine"
        ),
        "evidence_tier": TIER_FUNCTIONAL_SIM,
        "applied_cap_watts": None,
        "proposed_cap_watts": proposed_cap_watts,
        "numeric_cap_applied": False,
        "measured_watts": watts,
    }


def collect_evidence_tiers(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        if "evidence_tier" in node and node["evidence_tier"] is not None:
            found.add(str(node["evidence_tier"]))
        for value in node.values():
            found |= collect_evidence_tiers(value)
    elif isinstance(node, list):
        for value in node:
            found |= collect_evidence_tiers(value)
    return found


def assert_tier_honesty(categories: Mapping[str, Any]) -> None:
    """Refuse COST_MODEL / absent-hardware labeled HARDWARE_MEASURED.

    A guard nobody has watched fail is not a guard. Tests mutate a COST_MODEL
    value to HARDWARE_MEASURED and require this to raise.
    """
    for cat_id, cat in categories.items():
        values = cat.get("values") if isinstance(cat, dict) else None
        if not isinstance(values, list):
            raise TierHonestyError(f"{cat_id}: missing values[]")
        for val in values:
            if not isinstance(val, dict):
                raise TierHonestyError(f"{cat_id}: value is not an object")
            vid = val.get("id")
            tier = val.get("evidence_tier")
            if tier not in EVIDENCE_TIERS:
                raise TierHonestyError(f"{vid}: evidence_tier={tier!r} is not a known tier")
            extra = val.get("also_evidence_tier") or val.get("merged_tiers")
            if extra:
                raise TierHonestyError(f"{vid}: tiers must not be merged ({extra!r})")
            name = f"{vid}".lower()
            if "fpga" in name and tier == TIER_HARDWARE_MEASURED:
                raise TierHonestyError(
                    f"{vid}: FPGA/U50 is absent on this machine; COST_MODEL only"
                )
            if val.get("hardware_present") is False and tier == TIER_HARDWARE_MEASURED:
                raise TierHonestyError(
                    f"{vid}: absent hardware cannot be HARDWARE_MEASURED"
                )
            if vid in {"J/token", "J/accepted-token", "WU/kWh"} and tier == TIER_HARDWARE_MEASURED:
                if not val.get("token_ns_wrap"):
                    raise TierHonestyError(
                        f"{vid} labeled HARDWARE_MEASURED without a TOKEN_NS wrap"
                    )


def _gpu_idle_watts_value(
    rails: Mapping[str, Any],
) -> dict[str, Any]:
    idle = _channel_watts(rails.get("idle") if isinstance(rails, dict) else None, "GPU Energy")
    if idle and idle.get("increments") and isinstance(idle.get("watts"), (int, float)):
        return _cat_value(
            "gpu_rail_watts_idle",
            float(idle["watts"]),
            unit="W",
            evidence_tier=TIER_HARDWARE_MEASURED,
            why=(
                "IOReport Energy Model GPU Energy (nJ) delta over an idle window "
                "in this process. GPU rail only, not DRAM, DIRTY if other lanes "
                "ran. Not joules_per_token."
            ),
            token_ns_wrap=False,
            channel="GPU Energy",
            window_s=idle.get("window_s"),
            delta=idle.get("delta"),
        )
    return _cat_value(
        "gpu_rail_watts_idle",
        CITED_IDLE_GPU_WATTS_PRIOR,
        unit="W",
        evidence_tier=TIER_COST_MODEL,
        why=(
            "IOReportCreateSubscription did not yield an incrementing GPU Energy "
            "sample in this process. Watts are the standing-finding prior from "
            "energy.rs, not a measurement this sidecar took."
        ),
        token_ns_wrap=False,
        cited_from=CITED_IDLE_GPU_WATTS_SOURCE,
        subscription_obtained=bool(rails.get("subscription_obtained")) if isinstance(rails, dict) else False,
        sample_error=rails.get("error") if isinstance(rails, dict) else None,
    )


def roadmap_categories(
    *,
    probes: Sequence[Mapping[str, Any]] | None = None,
    rails: Mapping[str, Any] | None = None,
    cpu_cost: Mapping[str, Any] | None = None,
    host: Mapping[str, Any] | None = None,
    scheduler_decision: Mapping[str, Any] | None = None,
    process_energy: Mapping[str, Any] | None = None,
    contention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """II-E SUBGENES as categories, each value carrying its own evidence tier."""
    probes = list(probes) if probes is not None else run_probes()
    rails = dict(rails) if rails is not None else sample_energy_rails()
    cpu_cost = dict(cpu_cost) if cpu_cost is not None else measure_idle_vs_active_cpu()
    host = dict(host) if host is not None else measure_host_identity()
    if contention is None:
        contention = observe_contention()
    else:
        contention = dict(contention)
    if process_energy is None:
        # Production call site of measure_process_energy (also invoked from build).
        process_energy = measure_process_energy(contention=contention)
    else:
        process_energy = dict(process_energy)
    if scheduler_decision is None:
        scheduler_decision = EnergyAwareScheduler().schedule(
            {"id": "green-machine-categories"}, unknown_metrics()
        ).as_dict()

    by_id = {p["id"]: p for p in probes if isinstance(p, dict) and "id" in p}
    catalog = by_id.get("ioreport_energy_model_catalog") or {}
    cat_obs = catalog.get("observation") if isinstance(catalog.get("observation"), dict) else {}
    gpu_ch = bool(cat_obs.get("gpu_energy_channel_present"))
    cpu_ch = list(cat_obs.get("cpu_energy_channels_present") or [])
    catalog_ok = bool(catalog.get("succeeded"))

    gpu_watts = _gpu_idle_watts_value(rails)
    cited = cite_token_interval()
    ms = cited.get("value")
    watts = gpu_watts.get("value")
    j_tok = None
    if isinstance(ms, (int, float)) and isinstance(watts, (int, float)):
        j_tok = float(watts) * (float(ms) / 1000.0)

    power_cap = decide_power_cap(proposed_cap_watts=50.0, power_value=gpu_watts)

    categories: dict[str, Any] = {
        "GPU/CPU/FPGA power receipts": {
            "id": "GPU/CPU/FPGA power receipts",
            "values": [
                _cat_value(
                    "gpu_energy_channel_present",
                    gpu_ch,
                    unit="bool",
                    evidence_tier=TIER_HARDWARE_MEASURED if catalog_ok else TIER_STATIC,
                    why=(
                        "IOReportCopyChannelsInGroup('Energy Model') on this Mac "
                        "returned the GPU Energy channel name."
                        if catalog_ok
                        else "Energy Model catalog did not succeed in this process."
                    ),
                ),
                _cat_value(
                    "cpu_energy_channels_present",
                    cpu_ch,
                    unit="channel-names",
                    evidence_tier=TIER_HARDWARE_MEASURED if catalog_ok else TIER_STATIC,
                    why=(
                        "IOReport Energy Model named DIE_*_CPU Energy channels "
                        "on this Mac."
                        if catalog_ok
                        else "Energy Model catalog did not succeed in this process."
                    ),
                ),
                gpu_watts,
                _cat_value(
                    "process_energy_watts_idle",
                    (process_energy.get("idle") or {}).get("mean_watts"),
                    unit="W",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if process_energy.get("ok")
                        else TIER_STATIC
                    ),
                    why=(
                        "Mean ri_energy_nj / window_s over idle sleep on this "
                        "pid. Process-attributed kernel energy, not the GPU "
                        "rail, not joules_per_token."
                    ),
                    token_ns_wrap=False,
                    source="proc_pid_rusage.ri_energy_nj",
                    window_s=(process_energy.get("idle") or {}).get("mean_window_s"),
                ),
                _cat_value(
                    "process_energy_watts_busy",
                    (process_energy.get("busy") or {}).get("mean_watts"),
                    unit="W",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if process_energy.get("ok")
                        else TIER_STATIC
                    ),
                    why=(
                        "Mean ri_energy_nj / window_s over a CPU-only burn on "
                        "this pid. GPU untouched. Not GPU-token-active."
                    ),
                    token_ns_wrap=False,
                    source="proc_pid_rusage.ri_energy_nj",
                    window_s=(process_energy.get("busy") or {}).get("mean_window_s"),
                    cpu_burn_iters=(process_energy.get("busy") or {}).get(
                        "mean_cpu_burn_iters"
                    ),
                ),
                _cat_value(
                    "fpga_power_watts",
                    None,
                    unit="W",
                    evidence_tier=TIER_COST_MODEL,
                    why=(
                        "FPGA/U50 is absent on this Apple M3 Ultra. Absent "
                        "hardware is a model, never a measurement."
                    ),
                    hardware_present=False,
                    token_ns_wrap=False,
                ),
            ],
        },
        "J/token": {
            "id": "J/token",
            "values": [
                _cat_value(
                    "J/token",
                    j_tok,
                    unit="J/token",
                    evidence_tier=TIER_COST_MODEL,
                    why=(
                        "gpu_rail_watts_idle × cited_ms_per_token / 1000. Not a "
                        "TOKEN_NS wrap, not joules_per_token. The contracted "
                        "metric stays UNKNOWN in metrics[]."
                    ),
                    token_ns_wrap=False,
                    model="gpu_rail_watts_idle * cited_ms_per_token / 1000",
                    cited_ms_per_token=ms,
                    cited_from=cited.get("cited_from"),
                    watts_input_tier=gpu_watts.get("evidence_tier"),
                ),
            ],
        },
        "J/accepted-token": {
            "id": "J/accepted-token",
            "values": [
                _cat_value(
                    "J/accepted-token",
                    j_tok,
                    unit="J/accepted-token",
                    evidence_tier=TIER_COST_MODEL,
                    why=(
                        "No accepted-token ledger is wrapped with energy here. "
                        "Equals the J/token COST_MODEL under the assumption "
                        "that speculation is off or every draft is accepted. "
                        "Not a measurement."
                    ),
                    token_ns_wrap=False,
                    assumption="speculation_off_or_accept_rate_1",
                ),
            ],
        },
        "WU/kWh": {
            "id": "WU/kWh",
            "values": [
                _cat_value(
                    "WU/kWh",
                    None,
                    unit="WorkUnits/kWh",
                    evidence_tier=TIER_COST_MODEL,
                    why=(
                        "No WorkUnit completion count shares a closed wall with "
                        "a joule integral. Formula WU / (J / 3.6e6) is defined; "
                        "the numerator is missing, so the value is not filled. "
                        "cpu_burn_iterations_per_kwh is a different denominator "
                        "and lives under idle-vs-active / useful_work; it is "
                        "not this metric."
                    ),
                    token_ns_wrap=False,
                    model="work_units_completed / (joules / 3.6e6)",
                ),
            ],
        },
        "thermal stability": {
            "id": "thermal stability",
            "values": [
                _cat_value(
                    "thermal_warning_recorded",
                    host.get("thermal_warning_recorded"),
                    unit="bool",
                    evidence_tier=TIER_HARDWARE_MEASURED,
                    why=(
                        "pmset -g therm ran on this Mac. This is the warning "
                        "log, not a die temperature. 'No thermal warning "
                        "recorded' is not thermal_state."
                    ),
                    observation=host.get("thermal_warning_observation"),
                ),
                _cat_value(
                    "die_temperature_c",
                    None,
                    unit="C",
                    evidence_tier=TIER_STATIC,
                    why=(
                        "sysctl machdep.xcpm.* / machdep.thermal oids are "
                        "absent; powermetrics needs root; no die thermometer "
                        "is readable in this process. No temperature is modeled."
                    ),
                ),
            ],
        },
        "idle-vs-active cost": {
            "id": "idle-vs-active cost",
            "values": [
                _cat_value(
                    "idle_process_cpu_s",
                    cpu_cost.get("idle_process_cpu_s"),
                    unit="s",
                    evidence_tier=TIER_HARDWARE_MEASURED,
                    why=cpu_cost.get("why") or "process CPU-seconds over an idle sleep",
                    window_s=cpu_cost.get("idle_window_s"),
                ),
                _cat_value(
                    "active_process_cpu_s",
                    cpu_cost.get("active_process_cpu_s"),
                    unit="s",
                    evidence_tier=TIER_HARDWARE_MEASURED,
                    why="process CPU-seconds over a CPU-only burn; GPU untouched",
                    window_s=cpu_cost.get("active_window_s"),
                    cpu_burn_iters=cpu_cost.get("cpu_burn_iters"),
                ),
                _cat_value(
                    "idle_process_energy_j",
                    (process_energy.get("idle") or {}).get("mean_joules"),
                    unit="J",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if process_energy.get("ok")
                        else TIER_STATIC
                    ),
                    why=(
                        "Raw sampled ri_energy_nj delta over idle sleep, "
                        "converted nJ→J. Bookend proc_pid_rusage on this pid."
                    ),
                    window_s=(process_energy.get("idle") or {}).get("mean_window_s"),
                    raw_field="ri_energy_nj",
                ),
                _cat_value(
                    "active_process_energy_j",
                    (process_energy.get("busy") or {}).get("mean_joules"),
                    unit="J",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if process_energy.get("ok")
                        else TIER_STATIC
                    ),
                    why=(
                        "Raw sampled ri_energy_nj delta over CPU-only burn, "
                        "converted nJ→J. GPU untouched."
                    ),
                    window_s=(process_energy.get("busy") or {}).get("mean_window_s"),
                    raw_field="ri_energy_nj",
                    cpu_burn_iters=(process_energy.get("busy") or {}).get(
                        "mean_cpu_burn_iters"
                    ),
                ),
                _cat_value(
                    "idle_vs_active_joules",
                    (process_energy.get("differential") or {}).get("mean_work_joules"),
                    unit="J",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if (process_energy.get("differential") or {}).get("busy_gt_idle")
                        else TIER_STATIC
                    ),
                    why=(
                        "busy_joules - idle_watts * busy_window_s from two "
                        "proc_pid_rusage windows on this pid. A real "
                        "differential of two samples, not two guesses. "
                        "Process-attributed; not GPU-token-active; not a "
                        "TOKEN_NS wrap."
                    ),
                    token_ns_wrap=False,
                    error_budget_j=(process_energy.get("error_budget") or {}).get(
                        "combined_j"
                    ),
                ),
                _cat_value(
                    "joules_per_cpu_burn_iteration",
                    (process_energy.get("useful_work") or {}).get("joules_per_iteration"),
                    unit="J/iteration",
                    evidence_tier=(
                        TIER_HARDWARE_MEASURED
                        if (process_energy.get("useful_work") or {}).get(
                            "joules_per_iteration"
                        )
                        is not None
                        else TIER_STATIC
                    ),
                    why=_USEFUL_WORK_DEFENSE,
                    denominator=_USEFUL_WORK_DENOMINATOR,
                    token_ns_wrap=False,
                    error_budget_j_per_iteration=(
                        process_energy.get("useful_work") or {}
                    ).get("error_budget_j_per_iteration"),
                    not_j_token=True,
                ),
            ],
        },
        "energy-aware scheduler": {
            "id": "energy-aware scheduler",
            "values": [
                _cat_value(
                    "scheduler_action",
                    scheduler_decision.get("action"),
                    unit="enum",
                    evidence_tier=TIER_FUNCTIONAL_SIM,
                    why=(
                        "EnergyAwareScheduler.schedule was invoked. It refuses "
                        "while token energy is UNKNOWN; there is no Admit path."
                    ),
                    reason_code=scheduler_decision.get("reason_code"),
                    admit_implemented=scheduler_decision.get("admit_implemented"),
                    numeric_energy_used=scheduler_decision.get("numeric_energy_used"),
                ),
            ],
        },
        "power caps only when measured": {
            "id": "power caps only when measured",
            "values": [
                _cat_value(
                    "power_cap_action",
                    power_cap.get("action"),
                    unit="enum",
                    evidence_tier=TIER_FUNCTIONAL_SIM,
                    why=power_cap.get("detail"),
                    reason_code=power_cap.get("reason_code"),
                    applied_cap_watts=power_cap.get("applied_cap_watts"),
                    numeric_cap_applied=power_cap.get("numeric_cap_applied"),
                    power_input_tier=gpu_watts.get("evidence_tier"),
                ),
            ],
        },
    }
    assert_tier_honesty(categories)
    return {
        "categories": categories,
        "gpu_watts": gpu_watts,
        "cited_token_interval": cited,
        "power_cap": power_cap,
        "host": host,
        "cpu_cost": cpu_cost,
        "rails": rails,
        "process_energy": process_energy,
        "contention": contention,
    }


def _git_exists(rel: str) -> bool:
    kind = git("cat-file", "-t", f"HEAD:{rel}")
    return kind.strip() in {"blob", "tree"}


def _disk_exists(rel: str) -> bool:
    return (REPO / rel).exists()


def recover_implementation() -> list[dict[str, Any]]:
    """What already existed, with paths. Disk state is authority."""
    specs = (
        {
            "path": "crates/hawking-core/src/token_ns/energy.rs",
            "what": (
                "Codex energy probe: powermetrics needs root; IOReport Energy "
                "Model GPU Energy (nJ) documented as readable without root on "
                "2026-08-16; idle sample is explicitly not joules_per_token; "
                "EnergySampler wraps the same interval as TOKEN_NS."
            ),
            "adequate_for_this_lane": False,
            "why_not_adequate": (
                "Rust runtime under crates/; sidecar must not mutate it. "
                "EnergySampler wraps IOReport GPU Energy (nJ) around TOKEN_NS; "
                "this sidecar has no GPU lease and iokit-open-user-client is "
                "denied in the seatbelt profile, so that path is not this "
                "measurement. proc_pid_rusage ri_energy_nj is the working "
                "non-root process energy gate; energy.rs does not call it."
            ),
        },
        {
            "path": "crates/hawking-core/src/token_ns/served_weight.rs",
            "what": "pJ_per_weight_served is None unless joules_per_token is supplied.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "geometry + optional caller joule; no metric contract for accepted-token or WU/kWh.",
        },
        {
            "path": "crates/hawking-core/src/token_ns/schema.rs",
            "what": "EmitMeta.joules_per_token: Option<f64>, default None.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "a field is not a measurement and not a scoreboard axis.",
        },
        {
            "path": "crates/hawking-orch/src/scheduler.rs",
            "what": (
                "Admission controller defers heavy roles when on_battery and "
                "PowerMode::Quiet (DeferReason::Energy). Thermal headroom is a "
                "caller-supplied proxy in [0,1]."
            ),
            "adequate_for_this_lane": False,
            "why_not_adequate": (
                "It schedules on a battery+quiet heuristic, which is a guess. "
                "This lane's scheduler must refuse while unmeasured, not copy "
                "that heuristic."
            ),
        },
        {
            "path": "crates/hawking-serve/src/lib.rs",
            "what": "EnergyMode {Off, Balanced, Efficient} sizes a gather window (0/3/8 ms).",
            "adequate_for_this_lane": False,
            "why_not_adequate": "batching heuristic for J/tok in the comment; no joule is measured.",
        },
        {
            "path": "crates/hawking-serve/tests/energy_gather_window.rs",
            "what": "Unit tests for EnergyMode.should_gather. Not a measurement.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "tests a gather window, not energy accounting.",
        },
        {
            "path": "crates/hawking-speculate/src/metrics_sep.rs",
            "what": "AccelCostLedger.accepted_tokens / draft_tokens / rejected_tokens; ACCELERATED_ACCEPTED_TPS.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "defines the accepted-token denominator; does not attach joules.",
        },
        {
            "path": "tools/accelerator/machine_genome.py",
            "what": "thermal_envelope and sustained_behaviour are ABSENT with reasons. No energy fields.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "honest ABSENT on thermal; no joule contract.",
        },
        {
            "path": "hcli/machine.py",
            "what": "MemGate / Metal working-set admission. Host snapshot is RAM/swap/pressure.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "memory admission, not energy.",
        },
        {
            "path": "tools/headless/noetic_scoreboard.py",
            "what": "S017 §44 columns. Every unmeasured cell is ABSENT, never 0. No joule column.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "the energy axis is exactly the missing column.",
        },
        {
            "path": "workspace/campaign/evidence/models/glm52/GLM52_FUNCTIONAL_FLOP_BYTE_JOULE.json",
            "what": "joule.status = UNAVAILABLE; 'no accepted on-device energy source is wired; no joules are inferred from FLOPs or bytes'.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "one model's honest UNAVAILABLE, not a civilization-wide contract.",
        },
        {
            "path": "receipts/headless/ACCELERATOR_MACHINE_GENOME.json",
            "what": "claim_boundary: thermal_envelope and sustained_behaviour both ABSENT.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "identity + bandwidth; no energy axis.",
        },
        {
            "path": "receipts/headless/ACCELERATOR_SCOREBOARD.json",
            "what": "Named by frontier F015 as the live accelerator scoreboard.",
            "adequate_for_this_lane": False,
            "why_not_adequate": "not in git HEAD of this worktree; not on disk in the sparse checkout.",
        },
        {
            "path": "tools/future/green_machine.py",
            "what": "this module (F006 integration target).",
            "adequate_for_this_lane": True,
            "why_not_adequate": None,
        },
    )
    rows = []
    for spec in specs:
        path = spec["path"]
        rows.append(
            {
                **spec,
                "in_git_head": _git_exists(path),
                "on_disk_this_worktree": _disk_exists(path),
            }
        )
    return rows


def _negative_findings(probes: list[dict[str, Any]], recovered: list[dict[str, Any]]) -> list[str]:
    findings = [
        "No tools/future/green_machine.py existed before this lane (frontier F006).",
        "No energy axis on the noetic scoreboard (JOULES_PER_TOKEN / JOULES_PER_ACCEPTED_TOKEN / WORK_UNITS_PER_KWH absent).",
        "receipts/headless/ACCELERATOR_SCOREBOARD.json is not in git HEAD and not on disk in this sparse worktree.",
        "Sidecar has no GPU lease and must not produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE.",
        "Cannot wrap TOKEN_NS: no complete-token interval is available to this process.",
        "Cannot run cargo test on hawking-core energy.rs (crates/ is Codex-owned; GPU touch forbidden).",
    ]
    for row in probes:
        if row["id"] == "powermetrics_without_root" and not row.get("succeeded"):
            findings.append(
                "powermetrics without root failed: "
                + str(row.get("observation") or row.get("missing_dependency"))
            )
        if row["id"] == "sudo_n_powermetrics" and not row.get("succeeded"):
            findings.append(
                "sudo -n powermetrics is not available (no password prompt was issued): "
                + str(row.get("observation") or row.get("missing_dependency"))
            )
        if row["id"] == "ioreport_energy_model_subscription" and not row.get("succeeded"):
            findings.append(
                "IOReportCreateSubscription returned null in this process; live GPU Energy nJ was not read."
            )
        if row["id"] == "sysctl_thermal_levels" and not row.get("succeeded"):
            findings.append(
                "sysctl thermal oids (machdep.xcpm.* / machdep.thermal) are absent on this Darwin."
            )
        if row["id"] == "pmset_therm":
            findings.append(
                "pmset -g therm ran but is not a thermal_state measurement: "
                + str(row.get("observation"))
            )
        if row["id"] == "ioreg_power_telemetry":
            findings.append(
                "ioreg AppleSmartBattery PowerTelemetryData is named Estimate and was not recorded as a joule."
            )
    missing_scoreboard = [
        r for r in recovered if r["path"].endswith("ACCELERATOR_SCOREBOARD.json") and not r["in_git_head"]
    ]
    if missing_scoreboard:
        findings.append(
            "Could not inspect ACCELERATOR_SCOREBOARD.json contents; energy-axis slot is described against the noetic scoreboard instead."
        )
    return findings


def _gaps_closed() -> list[str]:
    return [
        "Defined joules/token, joules/accepted-token, WorkUnits/kWh, idle vs active, and thermal_state as a sealed contract.",
        "Probed this Mac without root and without a GPU lease; recorded exactly which probes ran and which did not.",
        "Forced every metric value to UNKNOWN; no TDP/FLOP estimate path succeeds.",
        "Energy-aware scheduler refuses while untrustworthy and refuses numeric claims without authority; there is no Admit path.",
        "Named the scoreboard slot (JOULES_PER_TOKEN, JOULES_PER_ACCEPTED_TOKEN, WORK_UNITS_PER_KWH) as ABSENT/UNKNOWN cells.",
        "Cited recovered Codex energy.rs / orch scheduler / EnergyMode / accepted-token ledger so this is not a fork of them.",
        "Emitted II-E SUBGENES as roadmap_categories with per-value evidence tiers (HARDWARE_MEASURED vs COST_MODEL vs STATIC vs FUNCTIONAL_SIM).",
        "Measured a genuine idle-vs-active energy differential on this pid via proc_pid_rusage ri_energy_nj (kernel nJ), with stated method and error budget.",
        "Defined the useful-work denominator as cpu_burn_iterations of the same closed window; J/token and WU/kWh stay unfilled.",
        "Recorded live-resident / hcli / sandbox contention without signaling any process.",
        "Measured what this M3 Ultra can without iokit-open: process energy, IOReport catalog, pmset power source/thermal warning log, sysctl identity, process CPU-seconds.",
        "Modeled what it cannot: FPGA power, J/token, J/accepted-token, WU/kWh, die temperature, GPU-token-active joules, IOReport live rails under this seatbelt profile.",
        "Power-cap policy refuses unless the wattage is HARDWARE_MEASURED; still does not enforce a cap without a GPU lease.",
    ]


def _forbid_numeric_metric_values(metrics: Mapping[str, Any]) -> None:
    for mid, entry in metrics.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if _is_number(value):
            raise HardwareClaimError(
                f"{mid} = {value!r}: sidecar has no GPU authority, "
                "hardware fields must be null/UNKNOWN"
            )
        if mid in HARDWARE_FIELDS and _is_number(value):
            raise HardwareClaimError(f"{mid} numeric")


def build() -> Any:
    probes = run_probes()
    metrics = unknown_metrics()
    _forbid_numeric_metric_values(metrics)
    recovered = recover_implementation()
    scheduler = EnergyAwareScheduler()
    decision = scheduler.schedule({"id": "green-machine-self"}, metrics)
    if decision.action != ACTION_REFUSE:
        raise UntrustworthyMeasurement(
            "scheduler must refuse while this sidecar cannot measure energy"
        )
    if decision.numeric_energy_used or decision.substituted_default:
        raise UntrustworthyMeasurement("scheduler leaked a numeric energy use")

    any_token_energy = any(p.get("trustworthy_for_token_energy") for p in probes)
    trustworthy = measurement_is_trustworthy(
        gpu_authority=False,
        protected_lease=False,
        energy_wrap_around_token_ns=False,
        root_powermetrics=any(
            p["id"] == "sudo_n_powermetrics" and p.get("succeeded") for p in probes
        ),
        ioreport_live_samples=any(
            p["id"] == "ioreport_energy_model_subscription" and p.get("succeeded")
            for p in probes
        ),
    )

    contention = observe_contention()
    # Receipt measurement: 1 s windows × 3 repeats. Tests call
    # measure_process_energy() directly with shorter windows.
    process_energy = measure_process_energy(
        idle_s=1.0, busy_s=1.0, repeats=3, contention=contention
    )
    packed = roadmap_categories(
        probes=probes,
        scheduler_decision=decision.as_dict(),
        process_energy=process_energy,
        contention=contention,
    )
    categories = packed["categories"]
    assert_tier_honesty(categories)

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "II-E Green Machine / H-ROADMAP §24: emit the gene-card SUBGENES "
            "with per-value evidence tiers. Token-attributed joules stay "
            "UNKNOWN; measurable M3 Ultra observations are HARDWARE_MEASURED; "
            "the rest are COST_MODEL."
        ),
        "roadmap": {
            "section": "§24 GREEN MACHINE",
            "gene": "II-E_GREEN_MACHINE",
            "gene_card_lines": "5971-6050",
            "root_phenotype": "Measure useful work per energy without Goodharting.",
            "subgenes": list(ROADMAP_SUBGENES),
        },
        "honesty_rule": HONESTY_RULE,
        "claim_class": CLAIM_CLASS,
        "gpu_authority": False,
        "protected_lease": False,
        "produces_diagnostic_relative": False,
        "produces_protected_absolute": False,
        "measurement_is_trustworthy": trustworthy,
        "any_probe_declared_token_energy_trust": any_token_energy,
        "metric_contract": list(METRIC_CONTRACT),
        "metrics": metrics,
        "roadmap_categories": categories,
        "evidence_tiers_present": sorted(collect_evidence_tiers(categories)),
        "host": packed["host"],
        "cpu_cost": packed["cpu_cost"],
        "rail_samples": packed["rails"],
        "process_energy": packed["process_energy"],
        "contention": packed["contention"],
        "measurement": {
            "symbol": "measure_process_energy",
            "ok": bool(process_energy.get("ok")),
            "method": process_energy.get("method"),
            "error_budget": process_energy.get("error_budget"),
            "useful_work": process_energy.get("useful_work"),
            "resident_during_measurement": process_energy.get(
                "resident_during_measurement"
            ),
            "gpu_touched": False,
            "token_ns_wrap": False,
        },
        "cited_token_interval": packed["cited_token_interval"],
        "power_cap": packed["power_cap"],
        "probes": probes,
        "probes_succeeded": sorted(p["id"] for p in probes if p.get("succeeded")),
        "probes_failed": sorted(p["id"] for p in probes if not p.get("succeeded")),
        "scheduler": {
            "interface": (
                "EnergyAwareScheduler.schedule(work, metrics=None, *, gpu_authority=False) "
                "-> EnergyScheduleDecision"
            ),
            "inert": True,
            "admit_implemented": admit_is_implemented(),
            "self_decision": decision.as_dict(),
            "does_not_copy": (
                "crates/hawking-orch/src/scheduler.rs on_battery+quiet heuristic"
            ),
        },
        "scoreboard_slot": SCOREBOARD_SLOT,
        "recovered_implementation": recovered,
        "gaps_closed": _gaps_closed(),
        "negative_findings": _negative_findings(probes, recovered),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    return write_receipt(RECEIPT, doc, "tools/future/green_machine.py")


def selftest() -> int:
    """Refuse-path + receipt. Exit 0 only if the guard actually fires."""
    raised = False
    try:
        energy_number(UNKNOWN, "joules_per_token")
    except UntrustworthyMeasurement:
        raised = True
    if not raised:
        print("selftest: energy_number(UNKNOWN) did not raise", file=sys.stderr)
        return 1
    raised = False
    try:
        energy_number(0.0, "joules_per_token")
    except UntrustworthyMeasurement:
        raised = True
    if not raised:
        print("selftest: energy_number(0.0) did not raise", file=sys.stderr)
        return 1
    decision = EnergyAwareScheduler().schedule({"id": "selftest"})
    if decision.action != ACTION_REFUSE or decision.numeric_energy_used:
        print("selftest: scheduler did not refuse", file=sys.stderr)
        return 1
    out = build()
    print(out)
    return 0


def probe_main() -> int:
    probes = run_probes()
    metrics = unknown_metrics()
    decision = EnergyAwareScheduler().schedule({"id": "probe"}, metrics)
    summary = {
        "honesty_rule": HONESTY_RULE,
        "metrics": {k: v["value"] for k, v in metrics.items()},
        "probes": probes,
        "scheduler": decision.as_dict(),
        "measurement_is_trustworthy": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    out = build()
    print(out)
    return 0


def main() -> int:
    if "--ioreport-worker" in sys.argv:
        try:
            print(json.dumps(_ioreport_inprocess(), sort_keys=True))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
            return 1
    if "--ioreport-sample" in sys.argv:
        idle_s = 0.25
        busy_s = 0.25
        for arg in sys.argv:
            if arg.startswith("--idle-s="):
                idle_s = float(arg.split("=", 1)[1])
            if arg.startswith("--busy-s="):
                busy_s = float(arg.split("=", 1)[1])
        try:
            print(json.dumps(_ioreport_sample_inprocess(idle_s, busy_s), sort_keys=True))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
            return 1
    ap = argparse.ArgumentParser(
        description="Green Machine energy accounting (STATIC_ONLY / UNKNOWN)"
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.probe:
        return probe_main()
    print(build())
    return 0


if __name__ == "__main__":
    from _common import require_known_flags
    require_known_flags(
        [
            "--build",
            "--probe",
            "--selftest",
            "--ioreport-worker",
            "--ioreport-sample",
            "--idle-s",
            "--busy-s",
        ]
    )
    raise SystemExit(main())
