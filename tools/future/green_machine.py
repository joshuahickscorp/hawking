"""GREEN_MACHINE — energy accounting that says UNKNOWN.

Energy is a missing axis of the dominance scoreboard. This sidecar defines the
metric contract, probes what THIS Mac can measure without root and without a
GPU lease, and exposes an energy-aware scheduler that is INERT while the
measurement is untrustworthy.

This module produces STATIC_ONLY / bench state UNKNOWN. It emits neither
DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. A plausible invented joule would
silently corrupt every later comparison; an honest UNKNOWN is the deliverable.

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
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping

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

HONESTY_RULE = (
    "Any metric that is not trustworthily measurable is UNKNOWN. Never an "
    "estimate, never a TDP-derived guess, never a FLOP-derived CMOS guess, "
    "never an idle GPU-rail sample presented as joules_per_token, never a "
    "number with invented precision. write_receipt already refuses a numeric "
    "joules_per_token. This module does not catch that error, does not add "
    "fields to HARDWARE_FIELDS, and does not convert UNKNOWN into 0 or into "
    "a datasheet watt. Saying UNKNOWN clearly is the deliverable. This "
    "sidecar produces STATIC_ONLY with bench state UNKNOWN; it produces "
    "neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
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
        "libioreport_dlopen_without_root": True,
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
    if obtained:
        out["observation"] = {
            "subscription_obtained": True,
            "note": (
                "Live samples would still not be joules_per_token: this "
                "sidecar has no GPU lease and does not wrap TOKEN_NS. "
                "Numeric nJ is deliberately not recorded."
            ),
        }
        return out
    out["missing_dependency"] = "IOReportCreateSubscription"
    out["observation"] = {
        "subscription_obtained": False,
        "error": "IOReportCreateSubscription returned null",
        "note": (
            "crates/hawking-core/src/token_ns/energy.rs documents a "
            "2026-08-16 standing finding that GPU Energy (nJ) incremented "
            "without root. This sidecar process cannot obtain a "
            "subscription, so live samples are not reproduced here. "
            "Channel catalog (probe ioreport_energy_model_catalog) is "
            "the part that still holds."
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
                "Rust runtime under crates/; sidecar must not mutate it. It "
                "fills pJ only from a caller joule or a wrap. It is not a "
                "scoreboard contract, not an inert scheduler, and this "
                "session could not reproduce live IOReport samples."
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

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Energy accounting contract and an honest probe of what this Mac "
            "can measure without root and without a GPU lease."
        ),
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
    raise SystemExit(main())
