"""ANE_PROBE_LAB — parameterized capability for ANE placement experiments.

The resident designs the experiment. This module does not. It materializes a
small public ANE-expressible graph from caller-supplied (op, shape, dtype,
structure), runs that graph under named modes, times the predict call, and
records what actually ran where from a public API.

It extends tools/future/ane_preboard.py: the preboard is the prepared-but-unrun
contract (Flash/Qwen27 corpus, MLComputePlan parser, placement schema). This
lab reuses the parser, the device vocabulary, the toolchain probe, and the
Swift MLComputePlan inspector. It does not import the preboard corpus and
does not choose a graph. Encoding a recipe here would steal the experiment.

coremltools authors. If it is absent the authoring and run entry points
REFUSE. A simulated ANE number would be worse than none.

Evidence tags are closed. An inference cannot be recorded as an observation.

    python3 tools/future/ane_probe_lab.py --build
    python3 -m pytest tools/future/test_ane_probe_lab.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, REPO
from tools.future import ane_preboard as preboard

import argparse
import json
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


RECEIPT = "ANE_PROBE_LAB.json"
SCHEMA = "hawking.future.ane_probe_lab.v1"
VERSION = 1
RECORDED_BY = "tools/future/ane_probe_lab.py"

MODES: tuple[str, ...] = ("GPU_ONLY", "ANE_ONLY", "SERIAL", "CONCURRENT")

EVIDENCE_CLASSES: tuple[str, ...] = (
    "PUBLICLY_DOCUMENTED",
    "PUBLIC_API_OBSERVED",
    "PLACEMENT_INFERRED",
    "TIMING_INFERRED",
    "PHYSICAL_BEHAVIOR_INFERRED",
)
OBSERVATION_CLASSES = frozenset({"PUBLICLY_DOCUMENTED", "PUBLIC_API_OBSERVED"})
INFERENCE_CLASSES = frozenset(
    {
        "PLACEMENT_INFERRED",
        "TIMING_INFERRED",
        "PHYSICAL_BEHAVIOR_INFERRED",
    }
)

# Public MLComputeUnits / coremltools.ComputeUnit names. Apple has no
# GPU-without-CPU and no ANE-without-CPU case. GPU_ONLY / ANE_ONLY are this
# lab's mode names; the mapping is documentation, not a placement.
DOCUMENTED_COMPUTE_UNITS: dict[str, dict[str, Any]] = {
    "GPU_ONLY": {
        "ml_compute_units": "cpuAndGPU",
        "coremltools_compute_unit": "CPU_AND_GPU",
        "exclusive_without_cpu": False,
    },
    "ANE_ONLY": {
        "ml_compute_units": "cpuAndNeuralEngine",
        "coremltools_compute_unit": "CPU_AND_NE",
        "exclusive_without_cpu": False,
    },
    "SERIAL": {
        "ml_compute_units": "all",
        "coremltools_compute_unit": "ALL",
        "exclusive_without_cpu": False,
        "kind": "SCHEDULE",
    },
    "CONCURRENT": {
        "ml_compute_units": "all",
        "coremltools_compute_unit": "ALL",
        "exclusive_without_cpu": False,
        "kind": "SCHEDULE",
    },
}

MODE_KIND: dict[str, str] = {
    "GPU_ONLY": "COMPUTE_UNITS",
    "ANE_ONLY": "COMPUTE_UNITS",
    "SERIAL": "SCHEDULE",
    "CONCURRENT": "SCHEDULE",
}

DEVICE_NEEDED: dict[str, str | None] = {
    "GPU_ONLY": "GPU",
    "ANE_ONLY": "NEURAL_ENGINE",
    "SERIAL": None,
    "CONCURRENT": None,
}

# How to author, not what to test. The caller still picks the pair.
SUPPORTED_SPECS: frozenset[tuple[str, str]] = frozenset(
    {
        ("add", "elementwise"),
        ("mul", "elementwise"),
        ("silu", "unary"),
        ("silu", "elementwise"),
        ("sigmoid", "unary"),
        ("sigmoid", "elementwise"),
        ("softmax", "reduction"),
        ("matmul", "linear"),
    }
)

PUBLIC_DTYPES: frozenset[str] = frozenset({"fp16", "float16", "fp32", "float32"})

PLACEMENT_OBSERVATION_APIS = frozenset(
    {
        "MLComputePlan.computeDeviceUsage(for:)",
        "MLComputePlan.load(contentsOf:configuration:)",
    }
)

INFERENCE_SOURCE_KINDS = frozenset(
    {
        "TIMING_DELTA",
        "IOREG",
        "CHIP_NAME",
        "METAL_PRESENCE",
        "ASSUMPTION",
        "INFERENCE",
        "FASTER_THAN",
    }
)

SWIFT_DEVICE_ENUM = r'''
import Foundation
import CoreML

func kind(of device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "NEURAL_ENGINE"
    @unknown default: return "UNKNOWN"
    }
}

var out: [String: Any] = [:]
out["macos"] = ProcessInfo.processInfo.operatingSystemVersionString
var devices: [MLComputeDevice] = []
var api: String = "NONE"
if #available(macOS 13.0, *) {
    devices = MLModel.availableComputeDevices
    api = "MLModel.availableComputeDevices"
}
out["api"] = api
out["devices"] = devices.map { [
    "kind": kind(of: $0),
    "description": String(describing: $0)
] }
out["kinds"] = Array(Set(devices.map { kind(of: $0) })).sorted()
out["n_devices"] = devices.count
out["status"] = "OBSERVED"
let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
'''


class ProbeLabRefused(RuntimeError):
    """The lab will not guess a missing input, a missing toolchain, or a placement."""


class CoremltoolsUnavailableError(ProbeLabRefused):
    """Authoring and execution require coremltools. Absence is a refusal, not a simulation."""


class GraphSpecRefused(ProbeLabRefused):
    """op, shape, dtype, or structure is missing or not publicly expressible."""


class ModeRefused(ProbeLabRefused):
    """Mode is missing or not one of GPU_ONLY, ANE_ONLY, SERIAL, CONCURRENT."""


class ModeNotRunnableError(ProbeLabRefused):
    """The named mode is not runnable on this host right now."""


class EvidencePromotionError(ProbeLabRefused):
    """An inference cannot be recorded as an observation."""


_DEVICE_CACHE: dict[str, Any] | None = None


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def coremltools_status() -> dict[str, Any]:
    """Reuse the preboard's import probe. Do not install anything."""
    return dict(preboard._coremltools_status())


def require_coremltools() -> dict[str, Any]:
    status = coremltools_status()
    if not status.get("present"):
        err = status.get("error") or "coremltools is not importable"
        raise CoremltoolsUnavailableError(
            "coremltools is absent; refusing to author, run, or time a Core ML "
            f"graph rather than simulating an ANE result ({err})"
        )
    return status


def existing_harness() -> dict[str, Any]:
    """The preboard is the existing harness. This lab extends it."""
    rel = "tools/future/ane_preboard.py"
    state = preboard.path_state(rel)
    state["role"] = (
        "STATIC_ONLY prepared-but-unrun Core ML / ANE contract; parser, "
        "device vocabulary, and MLComputePlan inspector are reused here"
    )
    state["this_lab_does_not_import"] = [
        "graph_corpus",
        "test_matrix graphs",
        "Flash / Qwen27 experiment recipe",
    ]
    return state


def documented_compute_units(mode: str) -> dict[str, Any]:
    if mode is None or mode == "":
        raise ModeRefused("mode is missing; the lab does not default GPU_ONLY/ANE_ONLY/SERIAL/CONCURRENT")
    if mode not in DOCUMENTED_COMPUTE_UNITS:
        raise ModeRefused(
            f"mode {mode!r} is not one of {list(MODES)}; "
            "the lab will not invent a compute-unit mapping"
        )
    row = dict(DOCUMENTED_COMPUTE_UNITS[mode])
    row["mode"] = mode
    row["kind"] = MODE_KIND[mode]
    row["evidence_class"] = "PUBLICLY_DOCUMENTED"
    row["is_observation"] = True
    row["is_inference"] = False
    row["is_placement"] = False
    row["api"] = "MLModelConfiguration.computeUnits / coremltools.ComputeUnit"
    row["caveat"] = (
        "Apple's public MLComputeUnits has no GPU-without-CPU and no "
        "ANE-without-CPU case. Requesting GPU_ONLY sets cpuAndGPU; what "
        "actually ran is a later PUBLIC_API_OBSERVED placement, not this mapping."
    )
    return row


def validate_graph_spec(
    *,
    op: Any,
    shape: Any,
    dtype: Any,
    structure: Any,
) -> dict[str, Any]:
    """Refuse missing or inexpressible inputs. Does not compile. Does not default."""
    missing = [
        name
        for name, value in (
            ("op", op),
            ("shape", shape),
            ("dtype", dtype),
            ("structure", structure),
        )
        if value is None or value == "" or value == []
    ]
    if missing:
        raise GraphSpecRefused(
            f"{', '.join(missing)} is missing; the lab does not default "
            "op, shape, dtype, or structure"
        )
    if not isinstance(op, str):
        raise GraphSpecRefused(f"op must be a string, got {type(op).__name__}")
    if not isinstance(structure, str):
        raise GraphSpecRefused(f"structure must be a string, got {type(structure).__name__}")
    if not isinstance(dtype, str):
        raise GraphSpecRefused(f"dtype must be a string, got {type(dtype).__name__}")
    if not isinstance(shape, (list, tuple)):
        raise GraphSpecRefused("shape must be a list or tuple of positive ints")
    dims: list[int] = []
    for i, dim in enumerate(shape):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise GraphSpecRefused(
                f"shape[{i}]={dim!r} is not a positive int; the lab will not pad or replace it"
            )
        dims.append(int(dim))
    if dtype not in PUBLIC_DTYPES:
        raise GraphSpecRefused(
            f"dtype {dtype!r} is not a public MIL dtype this lab authors "
            f"({sorted(PUBLIC_DTYPES)}); refusing rather than substituting fp16"
        )
    pair = (op, structure)
    if pair not in SUPPORTED_SPECS:
        raise GraphSpecRefused(
            f"op={op!r} structure={structure!r} is not a public MIL pair this lab "
            "authors; refusing rather than substituting a mul as a compile vehicle"
        )
    if op == "matmul" and len(dims) < 2:
        raise GraphSpecRefused(
            "matmul needs rank >= 2 so the second operand can be a transpose of "
            "the input; the lab will not invent an inner dimension"
        )
    return {
        "op": op,
        "shape": dims,
        "dtype": "fp16" if dtype in {"fp16", "float16"} else "fp32",
        "dtype_in": dtype,
        "structure": structure,
        "public_api": "coremltools MIL Builder -> MLProgram",
        "compiled": False,
    }


def enumerate_compute_devices(*, force: bool = False) -> dict[str, Any]:
    """PUBLIC_API_OBSERVED list from MLModel.availableComputeDevices, or UNTESTED."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None and not force:
        return dict(_DEVICE_CACHE)

    if platform.system() != "Darwin":
        result = {
            "status": "UNTESTED",
            "api": "MLModel.availableComputeDevices",
            "evidence_class": "PUBLIC_API_OBSERVED",
            "kinds": [],
            "devices": [],
            "n_devices": 0,
            "error": f"host is {platform.system()}; Core ML device enumeration is a Darwin API",
            "did_not": ["infer devices from chip name", "use ioreg as a placement"],
        }
        _DEVICE_CACHE = result
        return dict(result)

    tmp = Path(tempfile.mkdtemp(prefix="hawking-ane-probe-lab-"))
    src = tmp / "enum.swift"
    binary = tmp / "enum"
    src.write_text(SWIFT_DEVICE_ENUM)
    compiled = _run(
        ["xcrun", "swiftc", "-O", "-framework", "CoreML", "-o", str(binary), str(src)]
    )
    if compiled.returncode != 0 or not binary.is_file():
        err = (compiled.stderr or compiled.stdout or "swiftc produced no binary").strip()
        result = {
            "status": "UNTESTED",
            "api": "MLModel.availableComputeDevices",
            "evidence_class": "PUBLIC_API_OBSERVED",
            "kinds": [],
            "devices": [],
            "n_devices": 0,
            "error": f"swiftc enumerator failed: {err[:400]}",
            "did_not": ["infer devices from chip name", "use ioreg as a placement"],
        }
        _DEVICE_CACHE = result
        return dict(result)
    ran = _run([str(binary)])
    if ran.returncode != 0:
        err = (ran.stderr or ran.stdout or "enumerator produced no output").strip()
        result = {
            "status": "UNTESTED",
            "api": "MLModel.availableComputeDevices",
            "evidence_class": "PUBLIC_API_OBSERVED",
            "kinds": [],
            "devices": [],
            "n_devices": 0,
            "error": f"enumerator run failed: {err[:400]}",
            "did_not": ["infer devices from chip name", "use ioreg as a placement"],
        }
        _DEVICE_CACHE = result
        return dict(result)
    try:
        document = json.loads(ran.stdout)
    except json.JSONDecodeError as exc:
        result = {
            "status": "UNTESTED",
            "api": "MLModel.availableComputeDevices",
            "evidence_class": "PUBLIC_API_OBSERVED",
            "kinds": [],
            "devices": [],
            "n_devices": 0,
            "error": f"enumerator stdout was not JSON: {exc}",
            "did_not": ["infer devices from chip name", "use ioreg as a placement"],
        }
        _DEVICE_CACHE = result
        return dict(result)
    kinds = [str(k) for k in (document.get("kinds") or [])]
    result = {
        "status": "OBSERVED",
        "api": document.get("api") or "MLModel.availableComputeDevices",
        "evidence_class": "PUBLIC_API_OBSERVED",
        "is_observation": True,
        "is_inference": False,
        "kinds": kinds,
        "devices": document.get("devices") or [],
        "n_devices": int(document.get("n_devices") or len(document.get("devices") or [])),
        "macos": document.get("macos"),
        "gpu_listed": "GPU" in kinds,
        "ane_listed": "NEURAL_ENGINE" in kinds,
        "cpu_listed": "CPU" in kinds,
        "error": None,
        "did_not": [
            "treat Metal GPU presence as a Core ML GPU compute device",
            "treat ioreg ANE classes as a placement",
        ],
    }
    _DEVICE_CACHE = result
    return dict(result)


def mode_runnability(
    *,
    devices: Mapping[str, Any] | None = None,
    coremltools: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Which of the four modes can actually run a caller graph on this host now.

    Measured gates, not assumptions:
      * coremltools importable (authoring)
      * MLModel.availableComputeDevices observed (target device listed)
    --build does not compile a graph. A mode is runnable when those gates are
    open, not because a chip name suggests they should be.
    """
    devices = dict(devices) if devices is not None else enumerate_compute_devices()
    coreml = dict(coremltools) if coremltools is not None else coremltools_status()
    authoring = bool(coreml.get("present"))
    observed = devices.get("status") == "OBSERVED"
    kinds = {str(k) for k in (devices.get("kinds") or [])}
    measured_by = [
        "importlib.util.find_spec('coremltools') / coremltools import",
        "MLModel.availableComputeDevices via xcrun swiftc -framework CoreML",
    ]
    rows: dict[str, Any] = {}
    for mode in MODES:
        needed = DEVICE_NEEDED[mode]
        reasons: list[str] = []
        if not authoring:
            reasons.append(
                "coremltools is not importable; cannot materialize a graph "
                f"({coreml.get('error') or 'absent'})"
            )
        if not observed:
            reasons.append(
                "compute devices UNTESTED "
                f"({devices.get('error') or devices.get('status')}); "
                "refusing to assume a device from the chip name"
            )
        elif needed is not None and needed not in kinds:
            reasons.append(
                f"{devices.get('api')} did not list {needed}; "
                f"listed={sorted(kinds)}"
            )
        elif needed is None and not kinds:
            reasons.append(
                f"{devices.get('api')} listed no compute devices; "
                "SERIAL/CONCURRENT still need a device to predict on"
            )
        runnable = authoring and observed and (
            (needed in kinds) if needed is not None else bool(kinds)
        )
        rows[mode] = {
            "mode": mode,
            "kind": MODE_KIND[mode],
            "runnable_now": bool(runnable),
            "why": reasons if not runnable else (
                "coremltools is importable and the public device list includes "
                + (needed or ("at least one of " + ",".join(sorted(kinds))))
            ),
            "needed_device": needed,
            "documented": documented_compute_units(mode),
            "measured_by": list(measured_by),
            "assumed": False,
        }
    return rows


def record_placement(
    *,
    preferred: Any,
    supported: Any,
    evidence_class: str,
    source: str,
    api: str | None = None,
    is_live: bool = False,
    is_fixture: bool = False,
    source_kind: str | None = None,
    operations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record a placement claim. Inferences stay inferences."""
    if evidence_class not in EVIDENCE_CLASSES:
        raise EvidencePromotionError(
            f"evidence_class {evidence_class!r} is not one of {list(EVIDENCE_CLASSES)}"
        )
    if evidence_class == "PUBLICLY_DOCUMENTED":
        raise EvidencePromotionError(
            "documentation is not a graph placement; use documented_compute_units "
            "for the MLComputeUnits mapping and PUBLIC_API_OBSERVED only for a "
            "live MLComputePlan"
        )
    kind = (source_kind or "").upper() or None
    if evidence_class == "PUBLIC_API_OBSERVED":
        if is_fixture:
            raise EvidencePromotionError(
                "a fixture plan cannot be recorded as PUBLIC_API_OBSERVED"
            )
        if not is_live:
            raise EvidencePromotionError(
                "inferred or non-live placement cannot be recorded as "
                "PUBLIC_API_OBSERVED"
            )
        if api not in PLACEMENT_OBSERVATION_APIS:
            raise EvidencePromotionError(
                f"api {api!r} is not a live placement API; PUBLIC_API_OBSERVED "
                f"placement requires one of {sorted(PLACEMENT_OBSERVATION_APIS)}"
            )
        if kind in INFERENCE_SOURCE_KINDS:
            raise EvidencePromotionError(
                f"source_kind {kind} is an inference and cannot be recorded as "
                "PUBLIC_API_OBSERVED"
            )
    if evidence_class in INFERENCE_CLASSES and evidence_class == "PUBLIC_API_OBSERVED":
        raise EvidencePromotionError("an inference class cannot also be an observation")

    is_inference = evidence_class in INFERENCE_CLASSES
    is_observation = evidence_class in OBSERVATION_CLASSES
    if is_inference and is_observation:
        raise EvidencePromotionError(
            f"{evidence_class} cannot be both an inference and an observation"
        )
    return {
        "preferred": preferred,
        "supported": list(supported) if isinstance(supported, (list, tuple)) else supported,
        "evidence_class": evidence_class,
        "is_observation": is_observation,
        "is_inference": is_inference,
        "source": source,
        "source_kind": kind,
        "api": api,
        "is_live": bool(is_live) and not is_fixture,
        "is_fixture": bool(is_fixture),
        "operations": list(operations) if operations is not None else None,
        "claim_boundary": (
            "PUBLIC_API_OBSERVED means a live MLComputePlan in this process. "
            "A timing delta, an ioreg class, or a chip name is an inference."
        ),
    }


def as_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Promote a record to 'used as observation'. Inferences refuse."""
    cls = record.get("evidence_class")
    if cls in INFERENCE_CLASSES:
        raise EvidencePromotionError(
            f"{cls} cannot be promoted to an observation"
        )
    if cls != "PUBLIC_API_OBSERVED":
        raise EvidencePromotionError(
            f"{cls!r} is not a placement observation"
        )
    if not record.get("is_observation") or record.get("is_inference"):
        raise EvidencePromotionError("record is not an observation")
    if record.get("is_fixture") or not record.get("is_live"):
        raise EvidencePromotionError("fixture or non-live record cannot be an observation")
    return dict(record)


def placement_evidence_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Live MLComputePlan -> PUBLIC_API_OBSERVED. Fixtures cannot enter."""
    parsed = preboard.parse_mlcompute_plan(plan)
    if parsed.get("is_fixture") or not parsed.get("is_live"):
        raise EvidencePromotionError(
            "a fixture or non-live plan cannot be recorded as PUBLIC_API_OBSERVED "
            f"(is_fixture={parsed.get('is_fixture')} is_live={parsed.get('is_live')})"
        )
    ops = parsed.get("operations") or []
    op0 = ops[0] if ops else {}
    return record_placement(
        preferred=op0.get("preferred"),
        supported=op0.get("supported") or [],
        evidence_class="PUBLIC_API_OBSERVED",
        source="MLComputePlan.computeDeviceUsage(for:)",
        source_kind="MLCOMPUTEPLAN",
        api="MLComputePlan.computeDeviceUsage(for:)",
        is_live=True,
        is_fixture=False,
        operations=ops,
    )


SWIFT_COMPILE_MODEL = r'''
import CoreML
import Foundation
guard CommandLine.arguments.count >= 2 else {
    fputs("usage: compile_mlmodel <model>\n", stderr)
    exit(2)
}
let src = URL(fileURLWithPath: CommandLine.arguments[1])
do {
    let compiled = try MLModel.compileModel(at: src)
    FileHandle.standardOutput.write(Data(compiled.path.utf8))
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fputs(String(describing: error), stderr)
    exit(1)
}
'''


def _ensure_mlmodelc(path: str) -> str:
    """MLComputePlan.load wants a compiled .mlmodelc. Public compileModel API."""
    src = Path(path)
    if src.suffix == ".mlmodelc" and src.is_dir():
        return str(src)
    tmp = Path(tempfile.mkdtemp(prefix="hawking-ane-probe-lab-compile-"))
    swift = tmp / "compile.swift"
    binary = tmp / "compile"
    swift.write_text(SWIFT_COMPILE_MODEL)
    compiled = _run(
        ["xcrun", "swiftc", "-O", "-framework", "CoreML", "-o", str(binary), str(swift)]
    )
    if compiled.returncode != 0 or not binary.is_file():
        err = (compiled.stderr or compiled.stdout or "swiftc failed").strip()
        raise ProbeLabRefused(f"MLModel.compileModel authoring helper failed: {err[:400]}")
    ran = _run([str(binary), str(src)])
    if ran.returncode != 0:
        err = (ran.stderr or ran.stdout or "compileModel failed").strip()
        raise ProbeLabRefused(f"MLModel.compileModel failed: {err[:400]}")
    out = (ran.stdout or "").strip()
    if not out:
        raise ProbeLabRefused("MLModel.compileModel printed no path")
    return out


def inspect_placement(compiled_model: str) -> dict[str, Any]:
    """Run the preboard's public Swift inspector against a compiled model."""
    if compiled_model is None or compiled_model == "":
        raise GraphSpecRefused("compiled_model is missing")
    if not Path(compiled_model).exists():
        raise GraphSpecRefused(f"compiled model {compiled_model} is not on disk")
    modelc = _ensure_mlmodelc(compiled_model)
    document = preboard._load_live_compute_plan_impl(modelc)
    return placement_evidence_from_plan(document)


def _mil_dtype(dtype: str) -> Any:
    from coremltools.converters.mil.mil import types as mil_types  # type: ignore

    if dtype in {"fp16", "float16"}:
        return mil_types.fp16
    if dtype in {"fp32", "float32"}:
        return mil_types.fp32
    raise GraphSpecRefused(f"dtype {dtype!r} has no MIL mapping")


def _author_program(spec: Mapping[str, Any]) -> Any:
    from coremltools.converters.mil import Builder as mb  # type: ignore

    shape = tuple(int(d) for d in spec["shape"])
    mil_dtype = _mil_dtype(spec["dtype"])
    op = spec["op"]
    structure = spec["structure"]

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=mil_dtype)])
    def prog(x):  # type: ignore[no-untyped-def]
        if op == "add" and structure == "elementwise":
            return mb.add(x=x, y=x)
        if op == "mul" and structure == "elementwise":
            return mb.mul(x=x, y=x)
        if op == "silu" and structure in {"unary", "elementwise"}:
            return mb.silu(x=x)
        if op == "sigmoid" and structure in {"unary", "elementwise"}:
            return mb.sigmoid(x=x)
        if op == "softmax" and structure == "reduction":
            return mb.softmax(x=x, axis=-1)
        if op == "matmul" and structure == "linear":
            rank = len(shape)
            perm = list(range(rank - 2)) + [rank - 1, rank - 2]
            xt = mb.transpose(x=x, perm=perm)
            return mb.matmul(x=x, y=xt)
        raise GraphSpecRefused(
            f"no public MIL authoring for op={op!r} structure={structure!r}"
        )

    return prog


def materialize_graph(
    *,
    op: Any,
    shape: Any,
    dtype: Any,
    structure: Any,
) -> dict[str, Any]:
    """Compile a caller-specified public MIL graph. REFUSES without coremltools."""
    spec = validate_graph_spec(op=op, shape=shape, dtype=dtype, structure=structure)
    require_coremltools()
    import coremltools as ct  # type: ignore

    prog = _author_program(spec)
    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    out_dir = Path(tempfile.mkdtemp(prefix="hawking-ane-probe-lab-graph-"))
    dest = out_dir / "graph.mlpackage"
    mlmodel.save(str(dest))
    spec["compiled"] = True
    spec["compiled_model"] = str(dest)
    spec["status"] = "COMPILED"
    spec["claim_boundary"] = (
        "Compilation is not placement and is not a timing. Placement comes "
        "from MLComputePlan; elapsed_s comes from time.perf_counter around predict."
    )
    return spec


def _load_mlmodel(compiled_model: str, compute_unit_name: str) -> Any:
    import coremltools as ct  # type: ignore

    units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    if compute_unit_name not in units:
        raise ModeRefused(f"compute unit {compute_unit_name!r} is not a public coremltools.ComputeUnit")
    return ct.models.MLModel(compiled_model, compute_units=units[compute_unit_name])


def _predict_once(mlmodel: Any) -> None:
    spec = mlmodel.get_spec()
    desc = spec.description
    inputs = {}
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - numpy is present on this host
        raise ProbeLabRefused(f"numpy is required to feed a predict: {exc}") from exc
    for inp in desc.input:
        ttype = inp.type.WhichOneof("Type")
        if ttype != "multiArrayType":
            raise GraphSpecRefused(f"input {inp.name!r} is {ttype}, not multiArrayType")
        arr_type = inp.type.multiArrayType
        shape = [int(s) for s in arr_type.shape]
        inputs[inp.name] = np.ones(shape, dtype=np.float16)
    mlmodel.predict(inputs)


def run_graph(
    *,
    op: Any,
    shape: Any,
    dtype: Any,
    structure: Any,
    mode: Any,
    repeats: Any,
) -> dict[str, Any]:
    """Materialize, run one named mode, time predict, record placement if the API talks."""
    if mode is None or mode == "":
        raise ModeRefused("mode is missing; the lab will not choose one")
    if mode not in MODES:
        raise ModeRefused(f"mode {mode!r} is not one of {list(MODES)}")
    if repeats is None:
        raise GraphSpecRefused("repeats is missing; the lab does not default a campaign length")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise GraphSpecRefused(f"repeats={repeats!r} must be a positive int")
    validate_graph_spec(op=op, shape=shape, dtype=dtype, structure=structure)
    require_coremltools()

    documented = documented_compute_units(mode)
    runnability = mode_runnability()
    row = runnability[mode]
    if not row["runnable_now"]:
        raise ModeNotRunnableError(
            f"{mode} is not runnable now: {row['why']}"
        )

    compiled = materialize_graph(op=op, shape=shape, dtype=dtype, structure=structure)
    mlmodel = _load_mlmodel(
        compiled["compiled_model"], documented["coremltools_compute_unit"]
    )

    predict_elapsed_s: list[float] = []
    concurrent_elapsed_s = None
    concurrent_errors: list[str] = []
    if mode == "CONCURRENT":
        barrier_err: list[str] = []

        def one() -> None:
            try:
                _predict_once(mlmodel)
            except Exception as exc:  # prediction failure is evidence, not a guess
                barrier_err.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=one) for _ in range(repeats)]
        t0 = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent_elapsed_s = time.perf_counter() - t0
        concurrent_errors = barrier_err
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            _predict_once(mlmodel)
            predict_elapsed_s.append(time.perf_counter() - t0)

    placement: dict[str, Any] | None = None
    placement_error = None
    try:
        placement = inspect_placement(compiled["compiled_model"])
    except Exception as exc:
        placement_error = f"{type(exc).__name__}: {exc}"
        placement = None

    return {
        "spec": compiled,
        "mode": mode,
        "documented_compute_units": documented,
        "requested_compute_units_are_not_placement": True,
        "repeats": repeats,
        "schedule": "CONCURRENT" if mode == "CONCURRENT" else "SERIAL",
        "predict_elapsed_s": predict_elapsed_s,
        "concurrent_elapsed_s": concurrent_elapsed_s,
        "concurrent_errors": concurrent_errors,
        "timing_evidence_class": "PUBLIC_API_OBSERVED",
        "timing_claim_boundary": (
            "elapsed_s is time.perf_counter around MLModel.predict in this "
            "process. It is not a placement. Attributing the duration to ANE "
            "or GPU is TIMING_INFERRED / PLACEMENT_INFERRED."
        ),
        "placement": placement,
        "placement_error": placement_error,
        "placement_untested": placement is None,
    }


def run_matched_modes(
    *,
    op: Any,
    shape: Any,
    dtype: Any,
    structure: Any,
    modes: Any,
    repeats: Any,
) -> dict[str, Any]:
    """Run the caller-supplied mode list. The lab will not pick the list."""
    if modes is None:
        raise ModeRefused(
            "modes is missing; the lab will not choose GPU_ONLY/ANE_ONLY/"
            "SERIAL/CONCURRENT for the caller"
        )
    if not isinstance(modes, (list, tuple)) or not modes:
        raise ModeRefused("modes must be a non-empty list supplied by the caller")
    results = []
    for mode in modes:
        results.append(
            run_graph(
                op=op,
                shape=shape,
                dtype=dtype,
                structure=structure,
                mode=mode,
                repeats=repeats,
            )
        )
    return {"modes": list(modes), "results": results}


def execution_entry_points() -> tuple[tuple[str, Any], ...]:
    """Every function that would author or run a graph. Tests iterate this."""
    kwargs = dict(op="add", shape=[1, 4], dtype="fp16", structure="elementwise")
    return (
        ("materialize_graph", lambda: materialize_graph(**kwargs)),
        ("run_graph", lambda: run_graph(**kwargs, mode="ANE_ONLY", repeats=1)),
        ("run_matched_modes", lambda: run_matched_modes(**kwargs, modes=["SERIAL"], repeats=1)),
        ("inspect_placement", lambda: inspect_placement("/nonexistent.mlmodelc")),
    )


def probe_lab() -> dict[str, Any]:
    """Capability and runnability snapshot. No graph is chosen or timed."""
    coreml = coremltools_status()
    devices = enumerate_compute_devices(force=True)
    runnability = mode_runnability(devices=devices, coremltools=coreml)
    runnable_now = {mode: bool(runnability[mode]["runnable_now"]) for mode in MODES}
    preboard_probe = preboard.probe_toolchain()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Parameterized ANE probe lab. The caller supplies (op, shape, dtype, "
            "structure) and the mode list. This receipt does not choose them."
        ),
        "extends": "tools/future/ane_preboard.py",
        "existing_harness": existing_harness(),
        "does_not_choose_the_experiment": True,
        "modes": list(MODES),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "observation_classes": sorted(OBSERVATION_CLASSES),
        "inference_classes": sorted(INFERENCE_CLASSES),
        "supported_specs": sorted(f"{op}/{structure}" for op, structure in SUPPORTED_SPECS),
        "public_dtypes": sorted(PUBLIC_DTYPES),
        "coremltools": coreml,
        "compute_devices": devices,
        "mode_runnability": runnability,
        "runnable_now": runnable_now,
        "runnable_now_plain": (
            "GPU_ONLY={GPU_ONLY} ANE_ONLY={ANE_ONLY} SERIAL={SERIAL} "
            "CONCURRENT={CONCURRENT}"
        ).format(**runnable_now),
        "documented_compute_units": {mode: documented_compute_units(mode) for mode in MODES},
        "capability": {
            "materialize_graph": "caller supplies op, shape, dtype, structure",
            "run_modes": list(MODES),
            "time": "time.perf_counter around MLModel.predict; not a placement",
            "placement_evidence": (
                "PUBLIC_API_OBSERVED only from a live MLComputePlan; "
                "inferences stay PLACEMENT_INFERRED / TIMING_INFERRED / "
                "PHYSICAL_BEHAVIOR_INFERRED"
            ),
        },
        "execution_entry_points": [name for name, _ in execution_entry_points()],
        "toolchain_preboard": preboard_probe,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "negative_findings": [
            (
                "coremltools is absent; authoring/run/time refuse"
                if not coreml.get("present")
                else f"coremltools {coreml.get('version')} is importable"
            ),
            (
                f"MLModel.availableComputeDevices listed {devices.get('kinds')}"
                if devices.get("status") == "OBSERVED"
                else f"compute devices UNTESTED: {devices.get('error')}"
            ),
            (
                "GPU is not a Core ML compute device on this host according to "
                "MLModel.availableComputeDevices; GPU_ONLY is not runnable"
                if devices.get("status") == "OBSERVED" and not devices.get("gpu_listed")
                else "GPU listing as reported by availableComputeDevices"
            ),
            "this --build compiled no caller graph and timed no predict",
            "Flash/Qwen27 corpus stays in the preboard; this lab has no campaign",
        ],
        "claim_class": "CAPABILITY_AND_RUNNABILITY",
        "claim_boundary": (
            "Capability and runnability only. Device list is from "
            "MLModel.availableComputeDevices in this process. No graph was "
            "chosen, compiled, or timed. No ANE latency. Requested compute "
            "units are not placement."
        ),
        "preboard_path": str(REPO / "tools" / "future" / "ane_preboard.py"),
    }


def build() -> Path:
    doc = probe_lab()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe and not args.build:
        print(json.dumps(probe_lab(), indent=1, sort_keys=True))
        return 0
    if not args.build and not args.probe:
        ap.print_help()
        return 2
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
