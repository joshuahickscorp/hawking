"""Device Ascension Pipeline — arrival of a new machine.

When a new machine arrives it must not restart the science from zero, and it
must not inherit another machine's laws as truth. This sidecar is the ordered
arrival PIPE around the existing Codex ADP campaign in
``tools/accelerator/device_ascension.py`` (eleven stages, fingerprint to ADP
seal). That module already exists and already refuses a production seal
without sustained evidence. This module does not fork it. This module is the
arrival sequence and the law-downgrade rule those eleven stages never had.

    python3 tools/future/device_ascension_pipeline.py --dry-run
    python3 tools/future/device_ascension_pipeline.py --build
    python3 -m pytest tools/future/test_device_ascension_pipeline.py -q

Everything emitted here is STATIC_ONLY, bench state UNKNOWN, gpu_authority
false. Stage 7 (protected measurement) is a declared stub: UNAVAILABLE. This
process has no protected GPU lease and must not invent one.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, REPO, git

import argparse
import hashlib
import json
import platform
import re
import subprocess
from typing import Any, Iterable, Mapping

RECEIPT = "DEVICE_ASCENSION_PIPELINE.json"
SCHEMA = "hawking.future.device_ascension.v1"
RECORDED_BY = "tools/future/device_ascension_pipeline.py"

# Ordered arrival stages. Not the eleven ADP stages in
# tools/accelerator/device_ascension.py (fingerprint..adp_seal).
STAGES = (
    "discover_hardware",
    "build_machine_genome",
    "derive_capabilities",
    "import_laws_as_hypotheses",
    "calibration_experiments",
    "recompile_physical_graph",
    "protected_measurement",
    "emit_machine_laws",
)

# Odyssey II law-record vocabulary. Field names only — do not import a sibling
# lane's module (tools/future/odyssey2_law_store.py is a different lane).
LAW_FIELDS = (
    "law_id",
    "statement",
    "scope",
    "evidence_strength",
    "evidence_class",
    "origin_machine_id",
    "status",
)
SCOPES = ("MACHINE_LOCAL", "MODEL_LOCAL", "SOC_FAMILY", "GENERIC_VERIFIED")
EVIDENCE_STRENGTHS = ("HYPOTHESIS", "DISCOVERED", "VERIFIED")
EVIDENCE_CLASSES = ("STATIC_ONLY", "DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE")

VERIFIED_STRENGTHS = frozenset({"VERIFIED"})
LIVE_MEASUREMENT_CLASSES = frozenset({"PROTECTED_ABSOLUTE", "DIAGNOSTIC_RELATIVE"})

# Codex ADP stages, recovered so this lane names them rather than reinventing.
CODEX_ADP_STAGES = (
    "fingerprint",
    "benchmark",
    "roof_measurement",
    "workload_census",
    "kernel_selection",
    "autotune",
    "memory_plan",
    "dispatch_plan",
    "concurrency_sweep",
    "sustained_qualification",
    "adp_seal",
)

# Cheapest-first. GPU kinds exist so a hypothesis has an honest experiment,
# not so this sidecar can run them.
EXPERIMENT_KINDS: tuple[dict[str, Any], ...] = (
    {
        "kind": "STATIC_SYSCTL_IDENTITY",
        "axis": "identity",
        "cost_rank": 0,
        "needs_gpu": False,
        "would_confirm": "dest sysctl/ioreg identity matches the hypothesis predicate",
        "would_refute": "dest identity disagrees with the hypothesis predicate",
    },
    {
        "kind": "TOOLCHAIN_PRESENCE",
        "axis": "toolchain",
        "cost_rank": 1,
        "needs_gpu": False,
        "would_confirm": "named toolchain binary is present on dest",
        "would_refute": "named toolchain binary is absent on dest",
    },
    {
        "kind": "PROTECTED_BANDWIDTH_TRIAD",
        "axis": "bandwidth",
        "cost_rank": 10,
        "needs_gpu": True,
        "would_confirm": "protected triad on dest reproduces the origin claim",
        "would_refute": "protected triad on dest disagrees with the origin claim",
    },
    {
        "kind": "PROTECTED_UNIFIED_MEMORY_AB",
        "axis": "unified_memory",
        "cost_rank": 11,
        "needs_gpu": True,
        "would_confirm": "protected A/B on dest shows copy-elimination is structural",
        "would_refute": "protected A/B on dest shows the copies are not optional",
    },
    {
        "kind": "PROTECTED_FUSION_AB",
        "axis": "fusion",
        "cost_rank": 12,
        "needs_gpu": True,
        "would_confirm": "protected fusion A/B on dest agrees in sign with the origin",
        "would_refute": "protected fusion A/B on dest disagrees in sign with the origin",
    },
    {
        "kind": "PROTECTED_REPRESENTATION_HOLD",
        "axis": "representation",
        "cost_rank": 13,
        "needs_gpu": True,
        "would_confirm": "protected hold-out on dest keeps the representation floor",
        "would_refute": "protected hold-out on dest breaks the representation floor",
    },
    {
        "kind": "PROTECTED_GENERIC_AB",
        "axis": "*",
        "cost_rank": 20,
        "needs_gpu": True,
        "would_confirm": "a protected measurement on dest agrees with the hypothesis",
        "would_refute": "a protected measurement on dest disagrees with the hypothesis",
    },
)


class VerifiedImportRefused(ValueError):
    """Raised when an import path is asked to keep VERIFIED/PROTECTED_ABSOLUTE."""


class ArrivalInvariantError(ValueError):
    """Raised when a pipeline result still carries a live foreign VERIFIED law."""


# --------------------------------------------------------------------------- identity


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def _sysctl(key: str) -> str | None:
    out = _run(["sysctl", "-n", key], timeout=5.0)
    if out is None:
        return None
    text = out.strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n


def _gpu_cores_ioreg() -> Any:
    """Chip identity, not a measurement. gpu-core-count is a fused-in property."""
    out = _run(["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator", "-w", "0"], timeout=8.0)
    if not out:
        return {
            "status": "UNKNOWN",
            "reason": "ioreg AGXAccelerator returned nothing; sidecar will not guess a core count",
        }
    m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    if not m:
        return {
            "status": "UNKNOWN",
            "reason": "ioreg AGXAccelerator had no gpu-core-count key",
        }
    return int(m.group(1))


def _ane_present_ioreg() -> dict[str, Any]:
    out = _run(["ioreg", "-r", "-d", "1", "-c", "H11ANEIn", "-w", "0"], timeout=8.0)
    if out and "H11ANEIn" in out:
        return {
            "present": True,
            "profile": "ABSENT",
            "profile_path": "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
            "reason": (
                "H11ANEIn is visible in ioreg; APPLE_ANE_DEVICE_PROFILE.json is not "
                "in HEAD, so ANE capability is present-unprofiled, not measured"
            ),
            "performance": "UNKNOWN",
        }
    return {
        "present": False,
        "profile": "ABSENT",
        "profile_path": "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
        "reason": "no H11ANEIn node from ioreg; ANE is UNKNOWN rather than claimed absent-as-hardware",
        "performance": "UNKNOWN",
    }


def _metal_compiler_present() -> dict[str, Any]:
    try:
        r = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return {
            "present": False,
            "status": "UNKNOWN",
            "reason": f"xcrun metal probe failed: {type(e).__name__}",
        }
    if r.returncode == 0 and r.stdout.strip():
        # Version string is identity of a toolchain, not a hardware rate.
        line = r.stdout.strip().splitlines()[0]
        return {"present": True, "status": "PRESENT", "version_line": line}
    return {
        "present": False,
        "status": "ABSENT",
        "reason": "xcrun metal is not installed; AOT metallib compilation is unavailable",
    }


def machine_id(identity: Mapping[str, Any]) -> str:
    """Canonical machine identity. No wall-clock. Sorted, fixed field order."""
    gpu = identity.get("gpu_cores")
    if isinstance(gpu, dict):
        gpu = gpu.get("status") or "UNKNOWN"
    parts = (
        str(identity.get("soc") or "UNKNOWN"),
        str(identity.get("arch") or "UNKNOWN"),
        f"cpu={identity.get('cpu_cores') if identity.get('cpu_cores') is not None else 'UNKNOWN'}",
        f"gpu={gpu if gpu is not None else 'UNKNOWN'}",
        f"mem={identity.get('memory_bytes') if identity.get('memory_bytes') is not None else 'UNKNOWN'}",
    )
    return "|".join(parts)


def machine_id_digest(mid: str) -> str:
    return hashlib.sha256(mid.encode()).hexdigest()


# --------------------------------------------------------------------------- stage 1 / 2 / 3


def discover_hardware() -> dict[str, Any]:
    """Static identity of THIS machine. No bandwidth, no tps, no joules."""
    mem = _sysctl("hw.memsize")
    cpu = _sysctl("hw.ncpu")
    perf = _sysctl("hw.perflevel0.physicalcpu")
    eff = _sysctl("hw.perflevel1.physicalcpu")
    gpu = _gpu_cores_ioreg()
    identity = {
        "soc": _sysctl("machdep.cpu.brand_string") or "UNKNOWN",
        "arch": platform.machine() or "UNKNOWN",
        "cpu_cores": _int_or_none(cpu),
        "perf_cores": _int_or_none(perf),
        "efficiency_cores": _int_or_none(eff),
        "gpu_cores": gpu,
        "memory_bytes": _int_or_none(mem),
        "os": f"{platform.system()} {platform.release()}".strip(),
        "os_product": _sysctl("kern.osproductversion") or "UNKNOWN",
        "discovery_class": "STATIC_ONLY",
        "discovery_means": ("sysctl", "ioreg AGXAccelerator", "platform"),
        "not_measured": (
            "bandwidth, tps, token_ns, gpu_ns, joules, sustained thermal envelope"
        ),
    }
    identity["machine_id"] = machine_id(identity)
    identity["machine_id_sha256"] = machine_id_digest(identity["machine_id"])
    return identity


def build_machine_genome(hardware: Mapping[str, Any]) -> dict[str, Any]:
    """STATIC genome. Not tools/accelerator/machine_genome.py (that measures)."""
    hw = dict(hardware)
    mid = hw.get("machine_id") or machine_id(hw)
    return {
        "schema": "hawking.future.machine_genome.static.v1",
        "producer": RECORDED_BY,
        "not_the_codex_producer": "tools/accelerator/machine_genome.py",
        "not_the_admission_bag": "hcli.machine.MachineGenome",
        "knowledge_level": "INSTANCE",
        "machine_id": mid,
        "machine_id_sha256": machine_id_digest(mid),
        "identity": {
            "soc": hw.get("soc"),
            "arch": hw.get("arch"),
            "cpu_cores": hw.get("cpu_cores"),
            "perf_cores": hw.get("perf_cores"),
            "efficiency_cores": hw.get("efficiency_cores"),
            "gpu_cores": hw.get("gpu_cores"),
            "memory_bytes": hw.get("memory_bytes"),
            "os": hw.get("os"),
            "os_product": hw.get("os_product"),
        },
        "measured_bandwidth": {
            "status": "UNKNOWN",
            "reason": (
                "sidecar has no GPU authority; a bandwidth number would be an "
                "invented measurement. Codex's sealed genome is "
                "receipts/headless/MACHINE_GENOME.json and is not re-asserted here"
            ),
        },
        "thermal_envelope": {
            "status": "ABSENT",
            "reason": "no sustained thermal campaign in this sidecar",
        },
        "sustained_behaviour": {
            "status": "ABSENT",
            "reason": "sidecar does not run a microbenchmark or a sustained campaign",
        },
        "measurement_state": "STATIC_ONLY",
    }


def derive_capabilities(
    hardware: Mapping[str, Any],
    genome: Mapping[str, Any],
    *,
    live_probes: bool = True,
) -> dict[str, Any]:
    """Capabilities from identity. Presence, not performance."""
    soc = str(hardware.get("soc") or "")
    arch = str(hardware.get("arch") or "")
    apple_silicon = arch in {"arm64", "aarch64"} and soc.startswith("Apple")
    if live_probes:
        metal = _metal_compiler_present()
        ane = _ane_present_ioreg()
    else:
        metal = {
            "present": False,
            "status": "UNKNOWN",
            "reason": "live toolchain probe skipped (injected hardware fixture)",
        }
        ane = {
            "present": False,
            "profile": "ABSENT",
            "profile_path": "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
            "reason": "live ANE probe skipped (injected hardware fixture)",
            "performance": "UNKNOWN",
        }
    unified = apple_silicon
    return {
        "machine_id": genome.get("machine_id") or machine_id(hardware),
        "apple_silicon": apple_silicon,
        "unified_memory": {
            "present": unified,
            "note": (
                "On unified memory the CUDA-era host<->device copies are structurally "
                "avoidable. That is a topology fact, not a speedup number, and it does "
                "not transfer to a discrete-GPU machine."
            ),
            "evidence": "STATIC_ONLY identity (arch+soc); not a protected A/B",
        },
        "metal": {
            "api": "Metal",
            "eligible": apple_silicon,
            "compiler": metal,
            "performance": "UNKNOWN",
        },
        "ane": ane,
        "cuda": {
            "present": False,
            "reason": "this arrival pipeline is the Apple host sidecar; CUDA is a future backend, not claimed here",
            "performance": "UNKNOWN",
        },
        "fpga": {
            "present": False,
            "role": "Accelerator / Physical Compiler / Fusion",
            "civilization": False,
            "backend_status": "FUTURE",
            "note": (
                "FPGA is part of Accelerator / Physical Compiler / Fusion. It is not "
                "its own civilization and this module does not build an FPGA backend."
            ),
            "performance": "UNKNOWN",
        },
        "protected_gpu_lease": {
            "available": False,
            "reason": "sidecar has no GPU authority; Codex owns the protected lane",
        },
        "measurement_state": "STATIC_ONLY",
    }


# --------------------------------------------------------------------------- stage 4 — THE DOWNGRADE RULE


def _copy_law(law: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(law, sort_keys=True, default=str))


def clamp_scope(scope: str, *, crosses_machine_boundary: bool) -> str:
    """Scope clamp across a machine boundary. GENERIC_VERIFIED cannot survive."""
    if not crosses_machine_boundary:
        return scope if scope in SCOPES else "MACHINE_LOCAL"
    if scope == "MODEL_LOCAL":
        return "MODEL_LOCAL"
    # MACHINE_LOCAL stays MACHINE_LOCAL (still about the origin box).
    # SOC_FAMILY and GENERIC_VERIFIED clamp to MACHINE_LOCAL: a family-wide
    # or generic-verified law is a hypothesis about THIS box, not a verified
    # generic.
    return "MACHINE_LOCAL"


def apply_downgrade_rule(
    law: Mapping[str, Any],
    dest_machine_id: str,
    *,
    preserve_verified: bool = False,
) -> dict[str, Any]:
    """Single chokepoint. Every import surface calls this.

    A MACHINE_LOCAL law from machine A is a HYPOTHESIS on machine B, never a
    verified law. preserve_verified is the watched refusal: it always raises.
    """
    if not dest_machine_id:
        raise VerifiedImportRefused("dest_machine_id is empty; refuse rather than guess")
    if preserve_verified:
        raise VerifiedImportRefused(
            "preserve_verified is refused: no import path may keep VERIFIED or "
            "PROTECTED_ABSOLUTE status across a machine boundary "
            f"(law_id={law.get('law_id')!r} dest={dest_machine_id!r})"
        )
    missing = [f for f in ("law_id", "scope", "evidence_strength") if f not in law]
    if missing:
        raise VerifiedImportRefused(
            f"law is missing required Odyssey II fields {missing}; "
            "a record that cannot be typed cannot be imported as verified"
        )

    origin = str(law.get("origin_machine_id") or "")
    # Fail closed: an untyped origin is treated as foreign.
    crosses = origin != dest_machine_id
    out = _copy_law(law)
    out["origin_machine_id"] = origin or "UNKNOWN_ORIGIN"
    out["origin_scope"] = law.get("scope")
    out["origin_evidence_strength"] = law.get("evidence_strength")
    out["origin_evidence_class"] = law.get("evidence_class")
    out["dest_machine_id"] = dest_machine_id
    out["imported"] = True
    out["crosses_machine_boundary"] = crosses
    out["downgrade_rule"] = "hawking.future.device_ascension.downgrade.v1"

    if crosses:
        out["scope"] = clamp_scope(str(law.get("scope")), crosses_machine_boundary=True)
        out["evidence_strength"] = "HYPOTHESIS"
        out["evidence_class"] = "STATIC_ONLY"
        out["status"] = "HYPOTHESIS"
        out["verified_on_dest"] = False
        out["downgrade_reason"] = (
            "origin_machine_id != dest_machine_id: a law imported from another "
            "machine arrives as a hypothesis with scope clamped; VERIFIED and "
            "PROTECTED_ABSOLUTE cannot survive the boundary"
        )
    else:
        # Same-machine re-ingest is still an import, not a measurement. Arrival
        # stage 4 is "import laws AS HYPOTHESES". Sidecar never re-seals VERIFIED.
        out["scope"] = clamp_scope(str(law.get("scope")), crosses_machine_boundary=False)
        if str(law.get("evidence_strength")) in VERIFIED_STRENGTHS:
            out["evidence_strength"] = "HYPOTHESIS"
            out["downgrade_reason"] = (
                "arrival import always hypothesises; VERIFIED is not re-sealed "
                "without a protected measurement on dest (stage 7 is UNAVAILABLE)"
            )
        else:
            out["evidence_strength"] = (
                str(law.get("evidence_strength"))
                if str(law.get("evidence_strength")) in EVIDENCE_STRENGTHS
                else "HYPOTHESIS"
            )
            out["downgrade_reason"] = "arrival import; evidence_class forced STATIC_ONLY"
        out["evidence_class"] = "STATIC_ONLY"
        out["status"] = "HYPOTHESIS" if out["evidence_strength"] == "HYPOTHESIS" else str(law.get("status") or "HYPOTHESIS")
        out["verified_on_dest"] = False

    _assert_imported_law_is_legal(out, dest_machine_id)
    return out


def _assert_imported_law_is_legal(law: Mapping[str, Any], dest_machine_id: str) -> None:
    if law.get("verified_on_dest") is True:
        raise ArrivalInvariantError(f"{law.get('law_id')}: verified_on_dest is True after import")
    if law.get("evidence_class") in LIVE_MEASUREMENT_CLASSES:
        raise ArrivalInvariantError(
            f"{law.get('law_id')}: live evidence_class {law.get('evidence_class')!r} "
            "survived import; sidecar may only carry STATIC_ONLY"
        )
    origin = str(law.get("origin_machine_id") or "")
    crosses = origin != dest_machine_id
    if crosses and law.get("evidence_strength") in VERIFIED_STRENGTHS:
        raise ArrivalInvariantError(
            f"{law.get('law_id')}: VERIFIED survived a machine boundary "
            f"origin={origin!r} dest={dest_machine_id!r}"
        )
    if crosses and law.get("scope") == "GENERIC_VERIFIED":
        raise ArrivalInvariantError(
            f"{law.get('law_id')}: GENERIC_VERIFIED survived a machine boundary"
        )


def import_law(
    law: Mapping[str, Any],
    dest_machine_id: str,
    *,
    preserve_verified: bool = False,
) -> dict[str, Any]:
    return apply_downgrade_rule(law, dest_machine_id, preserve_verified=preserve_verified)


def import_laws(
    laws: Iterable[Mapping[str, Any]],
    dest_machine_id: str,
    *,
    preserve_verified: bool = False,
) -> list[dict[str, Any]]:
    imported = [
        import_law(law, dest_machine_id, preserve_verified=preserve_verified)
        for law in laws
    ]
    imported.sort(key=lambda e: str(e.get("law_id") or ""))
    assert_no_foreign_verified(imported, dest_machine_id)
    return imported


def import_law_catalog(
    dest_machine_id: str,
    laws: Iterable[Mapping[str, Any]] | None = None,
    *,
    preserve_verified: bool = False,
) -> list[dict[str, Any]]:
    catalog = list(laws) if laws is not None else list(FOREIGN_SEED_LAWS)
    catalog.sort(key=lambda e: str(e.get("law_id") or ""))
    return import_laws(catalog, dest_machine_id, preserve_verified=preserve_verified)


def assert_no_foreign_verified(
    laws: Iterable[Mapping[str, Any]], dest_machine_id: str
) -> None:
    """Guard the output of every import path. Raises if the refusal was skipped."""
    bad = []
    for law in laws:
        origin = str(law.get("origin_machine_id") or "")
        crosses = origin != dest_machine_id
        strength = law.get("evidence_strength")
        eclass = law.get("evidence_class")
        if crosses and strength in VERIFIED_STRENGTHS:
            bad.append(f"{law.get('law_id')}: evidence_strength={strength}")
        if eclass in LIVE_MEASUREMENT_CLASSES:
            bad.append(f"{law.get('law_id')}: evidence_class={eclass}")
        if crosses and law.get("scope") == "GENERIC_VERIFIED":
            bad.append(f"{law.get('law_id')}: scope=GENERIC_VERIFIED")
        if law.get("verified_on_dest") is True:
            bad.append(f"{law.get('law_id')}: verified_on_dest=True")
    if bad:
        raise ArrivalInvariantError(
            "foreign VERIFIED/PROTECTED_ABSOLUTE survived import: " + "; ".join(bad)
        )


def import_surfaces() -> dict[str, Any]:
    """Every public import path. Tests iterate this so a new path cannot hide."""
    return {
        "apply_downgrade_rule": apply_downgrade_rule,
        "import_law": import_law,
        "import_laws": import_laws,
        "import_law_catalog": import_law_catalog,
    }


# --------------------------------------------------------------------------- seed catalog (foreign by construction)


def _seed(
    law_id: str,
    *,
    statement: str,
    scope: str,
    evidence_strength: str,
    evidence_class: str,
    origin_machine_id: str,
    calibration_axis: str,
    source: str,
) -> dict[str, Any]:
    return {
        "law_id": law_id,
        "statement": statement,
        "scope": scope,
        "evidence_strength": evidence_strength,
        "evidence_class": evidence_class,
        "origin_machine_id": origin_machine_id,
        "status": "ACTIVE",
        "calibration_axis": calibration_axis,
        "source": source,
    }


# Origin ids are fixtures, never this host's machine_id. AKB-MACHINE-BANDWIDTH
# is a Codex law of the M3 Ultra that *is* this host; it is cited in recovery
# notes and is NOT imported, because importing it onto this host would be
# same-machine re-ingest of a measurement this sidecar did not take.
FOREIGN_SEED_LAWS: tuple[dict[str, Any], ...] = (
    _seed(
        "LAW-FOREIGN-H100-BANDWIDTH",
        statement=(
            "Origin fixture: a discrete NVIDIA H100 reported a machine-local "
            "bandwidth law. The number is not repeated here. A MACHINE_LOCAL "
            "bandwidth law is a property of that box and must not be inherited."
        ),
        scope="MACHINE_LOCAL",
        evidence_strength="VERIFIED",
        evidence_class="PROTECTED_ABSOLUTE",
        origin_machine_id="foreign|NVIDIA-H100|cpu=UNKNOWN|gpu=H100|mem=UNKNOWN",
        calibration_axis="bandwidth",
        source="fixture:foreign-h100",
    ),
    _seed(
        "LAW-FOREIGN-M1MAX-UNIFIED-MEMORY",
        statement=(
            "Origin fixture: on an Apple M1 Max, eliminating per-call "
            "host<->device copies was a structural unified-memory fact. That "
            "MACHINE_LOCAL topology law is a hypothesis on any other box."
        ),
        scope="MACHINE_LOCAL",
        evidence_strength="VERIFIED",
        evidence_class="PROTECTED_ABSOLUTE",
        origin_machine_id="foreign|Apple-M1-Max|cpu=10|gpu=32|mem=68719476736",
        calibration_axis="unified_memory",
        source="fixture:foreign-m1-max",
    ),
    _seed(
        "LAW-FOREIGN-GENERIC-FUSION",
        statement=(
            "Origin fixture: a GENERIC_VERIFIED fusion-beats-materialising claim "
            "from another machine. Generic-verified cannot survive a machine "
            "boundary; it clamps to MACHINE_LOCAL hypothesis."
        ),
        scope="GENERIC_VERIFIED",
        evidence_strength="VERIFIED",
        evidence_class="PROTECTED_ABSOLUTE",
        origin_machine_id="foreign|Apple-M2-Ultra|cpu=24|gpu=76|mem=UNKNOWN",
        calibration_axis="fusion",
        source="fixture:foreign-m2-ultra",
    ),
    _seed(
        "LAW-FOREIGN-MODEL-MLP-FLOOR",
        statement=(
            "Origin fixture: a MODEL_LOCAL MLP information-floor law measured "
            "under DIAGNOSTIC_RELATIVE conditions on another machine. Model "
            "scope may survive; VERIFIED/DIAGNOSTIC_RELATIVE may not."
        ),
        scope="MODEL_LOCAL",
        evidence_strength="VERIFIED",
        evidence_class="DIAGNOSTIC_RELATIVE",
        origin_machine_id="foreign|lab-B|cpu=UNKNOWN|gpu=UNKNOWN|mem=UNKNOWN",
        calibration_axis="representation",
        source="fixture:foreign-lab-b",
    ),
    _seed(
        "LAW-FOREIGN-ALREADY-HYPOTHESIS",
        statement="Origin fixture: already a hypothesis on a foreign machine.",
        scope="MACHINE_LOCAL",
        evidence_strength="HYPOTHESIS",
        evidence_class="STATIC_ONLY",
        origin_machine_id="foreign|lab-C|cpu=UNKNOWN|gpu=UNKNOWN|mem=UNKNOWN",
        calibration_axis="identity",
        source="fixture:foreign-lab-c",
    ),
)


# --------------------------------------------------------------------------- stage 5 / 6 / 7 / 8


def _experiment_for(axis: str) -> dict[str, Any]:
    exact = [e for e in EXPERIMENT_KINDS if e["axis"] == axis]
    pool = exact if exact else [e for e in EXPERIMENT_KINDS if e["axis"] == "*"]
    pool = sorted(pool, key=lambda e: (int(e["cost_rank"]), str(e["kind"])))
    return dict(pool[0])


def generate_calibration_experiments(
    hypotheses: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One cheapest experiment per hypothesis. GPU kinds are declared UNAVAILABLE."""
    out: list[dict[str, Any]] = []
    for hyp in sorted(hypotheses, key=lambda e: str(e.get("law_id") or "")):
        axis = str(hyp.get("calibration_axis") or "*")
        spec = _experiment_for(axis)
        available = not bool(spec["needs_gpu"])
        out.append(
            {
                "hypothesis_law_id": hyp.get("law_id"),
                "calibration_axis": axis,
                "kind": spec["kind"],
                "cost_rank": spec["cost_rank"],
                "needs_gpu": spec["needs_gpu"],
                "status": "RUNNABLE" if available else "UNAVAILABLE",
                "would_confirm": spec["would_confirm"],
                "would_refute": spec["would_refute"],
                "reason": (
                    None
                    if available
                    else "sidecar has no protected GPU lease; this experiment is declared, not run"
                ),
            }
        )
    return out


def recompile_physical_graph(
    hardware: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    hypotheses: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """PLAN_ONLY graph. Does not import hcli.physical_graph (Codex/hcli surface)."""
    apple = bool(capabilities.get("apple_silicon"))
    selected = "metal" if apple else "cpu"
    hyp_ids = sorted(str(h.get("law_id") or "") for h in hypotheses)
    body = {
        "schema": "hawking.future.physical_graph.plan_only.v1",
        "qualification": "PLAN_ONLY",
        "compiler_stage": "arrival_recompile_static",
        "not_executed": True,
        "not_the_hcli_compiler": "hcli.physical_graph.compile_physical_graph",
        "device_placement": {
            "candidates": ["cpu", "gpu", "fpga", "ane"],
            "selected": selected,
            "fpga_note": (
                "FPGA is a candidate backend of Accelerator / Physical Compiler / "
                "Fusion, not a civilization; selected is never fpga here"
            ),
        },
        "residency": {
            "weights": "unified_memory_candidate" if capabilities.get("unified_memory", {}).get("present") else "unresolved",
            "state": "unresolved",
            "page_cache": "unresolved",
        },
        "precision": {
            "weight": "unresolved",
            "activation": "unresolved",
            "accumulator": "unresolved",
        },
        "hypotheses_in_scope": hyp_ids,
        "machine_id": capabilities.get("machine_id") or hardware.get("machine_id"),
        "measurement_state": "STATIC_ONLY",
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["fingerprint"] = hashlib.sha256(blob).hexdigest()
    return body


def protected_measurement_stub(*, dry_run: bool) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "dry_run": bool(dry_run),
        "measurement_state": "STATIC_ONLY",
        "gpu_authority": False,
        "bench_state": "UNKNOWN",
        "would_produce": "PROTECTED_ABSOLUTE",
        "reason": (
            "sidecar has no protected GPU lease; stage 7 is a declared stub. "
            "Codex owns tools/accelerator and the physical qualification queue. "
            "A number here would be an invented measurement."
        ),
        "refuses_to_emit": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
    }


def emit_machine_laws(
    hardware: Mapping[str, Any],
    genome: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    hypotheses: Iterable[Mapping[str, Any]],
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """This machine's own laws. Without stage 7, none are VERIFIED performance laws."""
    mid = str(genome.get("machine_id") or hardware.get("machine_id") or "")
    own = [
        {
            "law_id": "LAW-DEST-IDENTITY-SOC",
            "statement": f"THIS machine's SoC identity WAS {hardware.get('soc')!r} under STATIC sysctl discovery.",
            "scope": "MACHINE_LOCAL",
            "evidence_strength": "DISCOVERED",
            "evidence_class": "STATIC_ONLY",
            "origin_machine_id": mid,
            "status": "ACTIVE",
            "imported": False,
            "verified_on_dest": False,
            "note": "identity, not a performance law; DISCOVERED is not VERIFIED",
        },
        {
            "law_id": "LAW-DEST-UNIFIED-MEMORY-TOPOLOGY",
            "statement": (
                "THIS machine has unified memory"
                if capabilities.get("unified_memory", {}).get("present")
                else "THIS machine was not classified as unified-memory Apple Silicon"
            ),
            "scope": "MACHINE_LOCAL",
            "evidence_strength": "DISCOVERED",
            "evidence_class": "STATIC_ONLY",
            "origin_machine_id": mid,
            "status": "ACTIVE",
            "imported": False,
            "verified_on_dest": False,
            "note": "topology from arch+soc; copy-elimination MAGNITUDE is unmeasured",
        },
    ]
    own.sort(key=lambda e: e["law_id"])
    pending = [
        {
            "law_id": h.get("law_id"),
            "scope": h.get("scope"),
            "evidence_strength": h.get("evidence_strength"),
            "waiting_on": "protected_measurement",
            "origin_machine_id": h.get("origin_machine_id"),
        }
        for h in sorted(hypotheses, key=lambda e: str(e.get("law_id") or ""))
    ]
    verified = [
        e for e in own
        if e.get("evidence_strength") in VERIFIED_STRENGTHS
        or e.get("evidence_class") in LIVE_MEASUREMENT_CLASSES
    ]
    if verified:
        raise ArrivalInvariantError(
            "stage 8 emitted a VERIFIED/live-measurement law without stage 7: "
            + ", ".join(str(e.get("law_id")) for e in verified)
        )
    if measurement.get("status") != "UNAVAILABLE":
        # Sidecar has no other legal status. A future Codex hook must not
        # silently start promoting through this function.
        raise ArrivalInvariantError(
            f"stage 7 status is {measurement.get('status')!r}; sidecar may only record UNAVAILABLE"
        )
    return {
        "own_laws": own,
        "imported_hypotheses_pending_calibration": pending,
        "n_verified_performance_laws": 0,
        "n_discovered_identity_laws": len(own),
        "rule": (
            "this machine's own VERIFIED performance laws are produced only after "
            "protected measurement; stage 7 is UNAVAILABLE so none are produced"
        ),
    }


# --------------------------------------------------------------------------- pipeline


def _stage(name: str, index: int, status: str, payload: Any, reason: str | None = None) -> dict[str, Any]:
    rec = {
        "name": name,
        "index": index,
        "status": status,
        "measurement_state": "STATIC_ONLY",
        "payload": payload,
    }
    if reason is not None:
        rec["reason"] = reason
    return rec


def run_pipeline(
    *,
    dry_run: bool = True,
    hardware: Mapping[str, Any] | None = None,
    foreign_laws: Iterable[Mapping[str, Any]] | None = None,
    live_probes: bool | None = None,
) -> dict[str, Any]:
    """Execute the eight arrival stages. Stage 7 is always UNAVAILABLE here."""
    injected = hardware is not None
    hw = dict(hardware) if injected else discover_hardware()
    if "machine_id" not in hw:
        hw["machine_id"] = machine_id(hw)
        hw["machine_id_sha256"] = machine_id_digest(hw["machine_id"])
    dest = str(hw["machine_id"])
    probes = (not injected) if live_probes is None else bool(live_probes)

    genome = build_machine_genome(hw)
    caps = derive_capabilities(hw, genome, live_probes=probes)
    hypotheses = import_law_catalog(dest, laws=foreign_laws)
    calibrations = generate_calibration_experiments(hypotheses)
    graph = recompile_physical_graph(hw, caps, hypotheses)
    measurement = protected_measurement_stub(dry_run=dry_run)
    emitted = emit_machine_laws(hw, genome, caps, hypotheses, measurement)

    assert_no_foreign_verified(hypotheses, dest)
    assert_no_foreign_verified(emitted["own_laws"], dest)

    stages = [
        _stage("discover_hardware", 1, "COMPLETED", hw),
        _stage("build_machine_genome", 2, "COMPLETED", genome),
        _stage("derive_capabilities", 3, "COMPLETED", caps),
        _stage("import_laws_as_hypotheses", 4, "COMPLETED", {
            "n_imported": len(hypotheses),
            "laws": hypotheses,
            "downgrade_rule": "hawking.future.device_ascension.downgrade.v1",
        }),
        _stage("calibration_experiments", 5, "COMPLETED", {
            "n": len(calibrations),
            "experiments": calibrations,
            "n_runnable": sum(1 for e in calibrations if e["status"] == "RUNNABLE"),
            "n_unavailable": sum(1 for e in calibrations if e["status"] == "UNAVAILABLE"),
        }),
        _stage("recompile_physical_graph", 6, "COMPLETED", graph),
        _stage(
            "protected_measurement",
            7,
            "UNAVAILABLE",
            measurement,
            reason=str(measurement["reason"]),
        ),
        _stage("emit_machine_laws", 8, "COMPLETED", emitted),
    ]
    if [s["name"] for s in stages] != list(STAGES):
        raise ArrivalInvariantError("stage order drifted from STAGES")
    return {
        "dry_run": bool(dry_run),
        "dest_machine_id": dest,
        "dest_machine_id_sha256": machine_id_digest(dest),
        "stages": stages,
        "downgrade_rule": {
            "id": "hawking.future.device_ascension.downgrade.v1",
            "statement": (
                "Any law imported from another machine arrives with evidence_strength "
                "HYPOTHESIS and evidence_class STATIC_ONLY. Scope is clamped: "
                "GENERIC_VERIFIED and SOC_FAMILY become MACHINE_LOCAL; MODEL_LOCAL "
                "may keep model scope but never VERIFIED status. A MACHINE_LOCAL "
                "law from machine A is a HYPOTHESIS on machine B, never a verified law."
            ),
            "preserve_verified": "raises VerifiedImportRefused on every import surface",
            "import_surfaces": sorted(import_surfaces()),
        },
        "invariants_checked": [
            "no foreign VERIFIED survived import",
            "no PROTECTED_ABSOLUTE or DIAGNOSTIC_RELATIVE in sidecar-emitted laws",
            "stage 7 is UNAVAILABLE",
            "stage 8 emitted zero VERIFIED performance laws",
        ],
    }


# --------------------------------------------------------------------------- receipt


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/accelerator/device_ascension.py",
            "on_disk_in_this_worktree": (REPO / "tools/accelerator/device_ascension.py").is_file(),
            "in_git": True,
            "role": (
                "Codex ADP campaign: eleven stages fingerprint..adp_seal. Real "
                "functions; NOT_RUN rather than silent skip. Refuses production "
                "seal without sustained_qualification. Used by "
                "receipts/headless/ACCELERATOR_FRONT_G_P6.json. Not an arrival pipeline "
                "and has no law-downgrade rule. This sidecar extends AROUND it."
            ),
            "adequate_for_this_lane": False,
            "gap": "no import-laws-as-hypotheses, no machine-boundary downgrade, no arrival dry-run",
        },
        {
            "path": "tools/accelerator/test_device_ascension.py",
            "in_git": True,
            "role": "pins ADP seal refusals (unnamed profile, missing stages, INSTANCE vs EXACT_MACHINE vocabularies)",
        },
        {
            "path": "tools/accelerator/machine_genome.py",
            "in_git": True,
            "role": (
                "Codex producer of hawking.accelerator.machine_genome.v1 with a "
                "measured f32 triad. Sidecar must not re-measure and must not copy "
                "that sealed median as its own."
            ),
        },
        {
            "path": "receipts/headless/MACHINE_GENOME.json",
            "in_git": True,
            "role": "sealed INSTANCE genome for Apple M3 Ultra; Codex evidence, not sidecar evidence",
        },
        {
            "path": "receipts/headless/ACCELERATOR_MACHINE_GENOME.json",
            "in_git": True,
            "role": "ACCEL-DEVICE receipt wrapping the genome; AKB-MACHINE-BANDWIDTH cites it and says it must never be inherited by another machine",
        },
        {
            "path": "hcli/machine.py",
            "in_git": True,
            "role": "MachineGenome compatibility bag over a JSON file; not a producer; admission is resolve_runtime_limits; producer of admission numbers is tools/headless/machine_probe.py",
        },
        {
            "path": "hcli/genomes/runtime_genome.py",
            "in_git": True,
            "role": "RuntimeGenome from CONVENTIONAL_CONTROL_SET; per-backend science, not arrival",
        },
        {
            "path": "hcli/physical_graph.py",
            "in_git": True,
            "role": "PLAN_ONLY PhysicalGraph compiler. Sidecar recompile stage is a static plan that does not import hcli.",
        },
        {
            "path": "tools/headless/cross_model_laws.py",
            "in_git": True,
            "role": "promotion refusal on QWEN_SPECIFIC/FAMILY_TRANSFERRED/ARCHITECTURE_GENERAL/MACHINE_GENERAL. Different vocabulary from Odyssey II law_id/scope/evidence_strength. Not imported.",
        },
        {
            "path": "tools/accelerator/akb.py",
            "in_git": True,
            "role": "typed Accelerator laws with law_id + evidence_class Measured/Derived + 11 applicability axes. Closest existing law store; not Odyssey II scope/evidence_strength. Not imported (Codex surface, and sibling odyssey2_law_store is a different lane).",
        },
        {
            "path": "tools/odyssey/device_profiles.py",
            "in_git": True,
            "role": "INTERACTIVE/MAXX workload profiles, not hardware capability derivation",
        },
        {
            "path": "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
            "in_git": False,
            "role": "named in the lane brief; not present in HEAD",
        },
        {
            "path": "tools/future/odyssey2_law_store.py",
            "in_git": False,
            "role": "frontier F010; sibling lane. This module uses Odyssey II field names and does not import it.",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "eight-stage arrival pipeline (discover..emit) as real functions, not a narrative",
        "law-downgrade rule: foreign VERIFIED/PROTECTED_ABSOLUTE cannot survive any public import path",
        "scope clamp: GENERIC_VERIFIED/SOC_FAMILY -> MACHINE_LOCAL across a machine boundary",
        "calibration experiments generated one-cheapest-per imported hypothesis",
        "--dry-run executes the pipeline against THIS machine with stage 7 UNAVAILABLE",
        "stage 8 refuses to emit VERIFIED performance laws when stage 7 is UNAVAILABLE",
        "Odyssey II field names (law_id, scope, evidence_strength) without importing a sibling law store",
    ]


def negative_findings() -> list[str]:
    return [
        "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json is not in HEAD; ANE is present-unprofiled at best",
        "no Odyssey II scoped law store on disk (frontier F010); field names used locally",
        "tools/accelerator/* and hcli/* are not materialized in this sparse checkout; recovered via git show HEAD:<path>",
        "protected measurement cannot be taken here; stage 7 is UNAVAILABLE",
        "FPGA board is not present; FPGA backend is not built (and must not be)",
        "existing tools/accelerator/device_ascension.py is an ADP campaign, not an arrival pipeline — extended around, not forked",
        "cannot load Codex MACHINE_GENOME.json from disk in this worktree (sparse); cited from git, not copied as a measurement",
    ]


def build(pipeline: Mapping[str, Any] | None = None, *, dry_run: bool = True) -> Any:
    result = dict(pipeline) if pipeline is not None else run_pipeline(dry_run=dry_run)
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Arrival pipeline for a new machine: discover, genome, capabilities, "
            "import laws AS HYPOTHESES, calibrate, recompile, (stub) measure, "
            "emit this machine's own laws. Disk state is authority. Models propose; "
            "protected deterministic evidence decides. This sidecar produces neither "
            "DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
        "eras": [
            "I Genesis of the Laboratory",
            "II Compounding Civilization",
            "III Autonomous Science Civilization",
            "IV Synthetic Machine Civilization",
            "V Released Hawking Civilization",
        ],
        "odysseys": [
            "I WHAT IS TRUE?",
            "II WHAT DID HAWKING ALREADY LEARN?",
            "III WHERE IS HAWKING WRONG?",
        ],
        "around_not_instead_of": {
            "codex_adp_module": "tools/accelerator/device_ascension.py",
            "codex_adp_stages": list(CODEX_ADP_STAGES),
            "this_module_stages": list(STAGES),
            "relation": (
                "ADP campaign qualifies a profile on a machine that already has "
                "laws. Arrival pipeline is what runs when the machine is new: it "
                "feeds hypotheses into later ADP stages, and it never promotes "
                "those hypotheses itself."
            ),
        },
        "odyssey_ii_law_fields": list(LAW_FIELDS),
        "scopes": list(SCOPES),
        "evidence_strengths": list(EVIDENCE_STRENGTHS),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "pipeline": result,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("When a new machine", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="run the pipeline on THIS machine; stage 7 UNAVAILABLE")
    ap.add_argument("--build", action="store_true", help="emit the sealed receipt")
    a = ap.parse_args()
    result = run_pipeline(dry_run=True if a.dry_run or not a.build else True)
    out = build(pipeline=result, dry_run=True)
    if a.dry_run:
        summary = {
            "dest_machine_id": result["dest_machine_id"],
            "stages": [
                {"index": s["index"], "name": s["name"], "status": s["status"]}
                for s in result["stages"]
            ],
            "n_imported_hypotheses": result["stages"][3]["payload"]["n_imported"],
            "n_calibration_experiments": result["stages"][4]["payload"]["n"],
            "n_runnable_calibrations": result["stages"][4]["payload"]["n_runnable"],
            "stage_7": result["stages"][6]["status"],
            "n_verified_performance_laws": result["stages"][7]["payload"]["n_verified_performance_laws"],
            "receipt": str(out),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
