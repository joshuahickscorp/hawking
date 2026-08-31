"""METAL REACHABILITY — falsify a host-wide blocker, or confirm it.

The whole campaign carries one sentence:

    "this host has no Metal-capable GPU"

It appears as `status: BLOCKED_NO_METAL_GPU` on the Flash teacher capture, which
is gate 2 of the meta funnel and therefore the thing standing between Flash and
every downstream representation result. It was repeated into this sidecar's own
autonomy driver as a settled fact about the machine.

It is not a fact about the machine. This host is an Apple M3 Ultra with a 60-core
GPU reporting Metal 4, and `MTLCreateSystemDefaultDevice()` returns that device
from an ordinary command-line process here. So `Device::system_default()`
returning None inside one Rust binary is a property of THAT PROCESS, not of the
host, and the two have completely different remedies: a missing GPU means buy
hardware, while a process that cannot see a present GPU means change how the
process is launched.

What this module does NOT claim:

* It does not claim the Codex capture will now succeed. It did not run that
  binary, and the failure could still be anything downstream of device creation.
* It does not measure anything. Device enumeration is a capability query --
  no queue is used, no work is submitted, no lease is taken, no timing is kept.
* It does not identify the cause of the Codex-side failure. Naming the machine
  as innocent is not the same as naming the guilty party, and guessing between
  sandbox, launch context and build target would be inventing evidence.

    python3 tools/future/metal_reachability.py --probe
    python3 tools/future/metal_reachability.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt
from tools.future import status_causality as sc

RECEIPT = "METAL_REACHABILITY.json"
SCHEMA = "hawking.future.metal_reachability.v1"

FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    """Consumer-side emit. Sibling owns the routine; this checkout may predate it."""
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def record_verdict_causality(
    result: dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change the host-claim verdict.

    An unsupplied observation is UNTESTED, never a restatement of the verdict.
    """
    verdict_before = result.get("verdict")
    status = str(result.get("verdict") or "")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source="tools/future/metal_reachability.py::verdict",
    )
    for key in FIVE_RECORDED_FIELDS:
        result[key] = rec[key]
    result["causality_verdict"] = rec["verdict"]
    result["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        result["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        result["claim_kind"] = rec["claim_kind"]
    if result.get("verdict") != verdict_before:
        raise RuntimeError("status_causality.emit mutated the metal-reachability verdict")
    return rec

# The claim under test, and where it is written down.
CLAIM = "this host has no Metal-capable GPU"
CLAIM_SITES = (
    "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
    "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
    "crates/hawking-core/src/metal/mod.rs",
)

# Enumeration only. No command queue is created, so no GPU work can be submitted
# by this probe even accidentally.
PROBE_SWIFT = '''import Metal
var out: [String: Any] = [:]
if let d = MTLCreateSystemDefaultDevice() {
    out["system_default"] = d.name
    out["unified_memory"] = d.hasUnifiedMemory
    out["recommended_max_working_set_bytes"] = d.recommendedMaxWorkingSetSize
} else {
    out["system_default"] = NSNull()
}
let all = MTLCopyAllDevices()
out["n_devices"] = all.count
out["devices"] = all.map { $0.name }
let data = try! JSONSerialization.data(withJSONObject: out)
print(String(data: data, encoding: .utf8)!)
'''


# The same crate and the same call the runtime uses. Enumeration only again --
# no command queue, no shader library, nothing submitted. This is the probe that
# matters: it removes the language binding as a suspect, because it IS the
# binding that failed.
PROBE_RUST_MAIN = '''use metal::{CompileOptions, Device};
fn main() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => { println!("system_default=NONE"); println!("all_devices=0"); return; }
    };
    println!("system_default={}", device.name());
    println!("all_devices={}", Device::all().len());
    // Compiling a shader from SOURCE is the path the runtime actually takes when
    // no metallib cache is warm. It exercises the compiler service, not the GPU:
    // no command queue is created and nothing is dispatched.
    let src = "#include <metal_stdlib>\\nusing namespace metal;\\n\\
kernel void nop(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) \\
{ o[i] = 0.0f; }\\n";
    match device.new_library_with_source(src, &CompileOptions::new()) {
        Ok(lib) => {
            println!("runtime_source_compile=OK");
            match lib.get_function("nop", None) {
                Ok(_) => println!("function_lookup=OK"),
                Err(e) => println!("function_lookup=ERR {}", e),
            }
        }
        Err(e) => println!("runtime_source_compile=ERR {}", e),
    }
}
'''


def _runtime_metal_crate_version() -> str:
    """The metal crate version hawking-core actually resolves, from Cargo.lock."""
    lock = REPO / "Cargo.lock"
    if not lock.is_file():
        return ""
    seen = []
    lines = lock.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'name = "metal"' and i + 1 < len(lines):
            v = lines[i + 1].strip()
            if v.startswith("version = "):
                seen.append(v.split('"')[1])
    return max(seen) if seen else ""


class ProbeUnavailable(Exception):
    pass


def probe() -> dict[str, Any]:
    """Ask Metal whether it has a device. Enumeration, not measurement."""
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise ProbeUnavailable("swiftc is not on PATH; the claim cannot be tested here")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.swift"
        binary = Path(tmp) / "probe"
        src.write_text(PROBE_SWIFT)
        build = subprocess.run([swiftc, "-O", str(src), "-o", str(binary)],
                               capture_output=True, text=True, timeout=300)
        if build.returncode != 0:
            raise ProbeUnavailable(f"probe did not build: {build.stderr.strip()[:300]}")
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
        if run.returncode != 0:
            raise ProbeUnavailable(f"probe did not run: {run.stderr.strip()[:300]}")
        return json.loads(run.stdout.strip())


def probe_rust() -> dict[str, Any]:
    """Ask the SAME crate the runtime uses. Built out of tree, offline.

    Built in a temporary directory with its own CARGO_TARGET_DIR so nothing
    touches the repository's build state while a campaign is running.
    """
    cargo = shutil.which("cargo")
    if not cargo:
        raise ProbeUnavailable("cargo is not on PATH; the runtime binding cannot be tested")
    version = _runtime_metal_crate_version()
    if not version:
        raise ProbeUnavailable("Cargo.lock does not resolve a metal crate version")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "probe"
        (root / "src").mkdir(parents=True)
        (root / "Cargo.toml").write_text(
            '[package]\nname = "metalprobe"\nversion = "0.0.0"\nedition = "2021"\n'
            f'\n[dependencies]\nmetal = "{version}"\n'
        )
        (root / "src" / "main.rs").write_text(PROBE_RUST_MAIN)
        env = dict(_os.environ, CARGO_TARGET_DIR=str(Path(tmp) / "target"))
        run = subprocess.run([cargo, "run", "--release", "--offline", "-q"],
                             cwd=root, env=env, capture_output=True, text=True,
                             timeout=900)
        if run.returncode != 0:
            raise ProbeUnavailable(f"crate probe failed: {run.stderr.strip()[-300:]}")
        out: dict[str, Any] = {"metal_crate_version": version}
        for line in run.stdout.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "system_default":
                out["system_default"] = None if value.strip() == "NONE" else value.strip()
            elif key.strip() == "all_devices":
                out["n_devices"] = int(value.strip())
            elif key.strip() in ("runtime_source_compile", "function_lookup"):
                out[key.strip()] = value.strip()
        return out


def verdict(observed: dict[str, Any] | None, why_unavailable: str = "") -> dict[str, Any]:
    probe_performed = (
        "Swift Metal enumeration: MTLCreateSystemDefaultDevice() and "
        "MTLCopyAllDevices(); no command queue created, no work submitted"
    )
    if observed is None:
        row = {
            "claim": CLAIM,
            "verdict": "UNTESTED",
            "why": why_unavailable or "the probe could not run on this host",
        }
        record_verdict_causality(
            row,
            probe_performed=probe_performed,
            direct_observation=(
                f"probe_ran=False; why={why_unavailable or 'the probe could not run on this host'!r}"
            ),
            interpretation=(
                "the host-has-no-Metal-GPU claim was not tested because the "
                "enumeration probe did not run"
            ),
            probe_kind=sc.PROBE_PROCESS_ERROR,
            claim_kind=sc.CLAIM_PROCESS_FAILURE,
        )
        return row
    device = observed.get("system_default")
    n_devices = observed.get("n_devices")
    observation = (
        f"system_default={device!r}; n_devices={n_devices}; "
        f"devices={observed.get('devices')}"
    )
    if device:
        row = {
            "claim": CLAIM,
            "verdict": "FALSIFIED_AS_A_HOST_PROPERTY",
            "why": (
                f"MTLCreateSystemDefaultDevice() returned {device!r} from an ordinary "
                f"command-line process on this host, and MTLCopyAllDevices() reports "
                f"{observed.get('n_devices')} device(s). A Rust Device::system_default() "
                f"returning None is therefore a property of that process, not of the "
                f"machine."
            ),
            "what_this_does_not_establish": [
                "that the blocked capture will now succeed; that binary was not run here",
                "the cause of the process-side failure; sandbox, launch context and "
                "build target are all consistent with what was observed and nothing "
                "here distinguishes them",
            ],
            "different_remedies": {
                "if the host had no GPU": "acquire hardware; nothing else helps",
                "a process that cannot see a present GPU": (
                    "change how the process is launched, then re-run the capture"
                ),
            },
        }
        record_verdict_causality(
            row,
            probe_performed=probe_performed,
            direct_observation=observation,
            interpretation=(
                f"this process saw a Metal device ({device!r}); the campaign claim "
                f"{CLAIM!r} is falsified as a host property"
            ),
            probe_kind=sc.PROBE_ENUMERATION,
            claim_kind=sc.CLAIM_DEVICE_PRESENT,
        )
        return row
    row = {
        "claim": CLAIM,
        "verdict": "CONFIRMED",
        "why": "MTLCreateSystemDefaultDevice() returned nil from an ordinary process here",
    }
    record_verdict_causality(
        row,
        probe_performed=probe_performed,
        direct_observation=observation,
        interpretation=(
            "this ordinary process saw no Metal device; that is a process-side "
            "observation, not a completed host census"
        ),
        probe_kind=sc.PROBE_ENUMERATION,
        claim_kind=sc.CLAIM_PROCESS_FAILURE,
    )
    return row


def claim_sites() -> list[dict[str, Any]]:
    """Where the claim is written down, so a correction can find all of it."""
    rows = []
    for rel in CLAIM_SITES:
        path = REPO / rel
        rows.append({
            "path": rel,
            "present": path.is_file(),
            "carries_claim": bool(
                path.is_file()
                and "no Metal-capable GPU" in path.read_text(errors="replace")
            ),
        })
    return rows


def build() -> Path:
    observed: dict[str, Any] | None = None
    why = ""
    try:
        observed = probe()
    except (ProbeUnavailable, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        why = f"{type(exc).__name__}: {exc}"

    runtime_binding: dict[str, Any] | None = None
    runtime_why = ""
    try:
        runtime_binding = probe_rust()
    except (ProbeUnavailable, subprocess.TimeoutExpired, ValueError) as exc:
        runtime_why = f"{type(exc).__name__}: {exc}"

    v = verdict(observed, why)
    v["runtime_binding"] = verdict(runtime_binding, runtime_why)
    if runtime_binding and runtime_binding.get("runtime_source_compile") == "OK":
        v["shader_compilation"] = {
            "runtime_source_compile": "OK",
            "why_it_matters": (
                "load_or_compile_shader_library falls back to "
                "device.new_library_with_source when no metallib cache is warm, and "
                "the xcrun precompile path is optional and gated behind "
                "HAWKING_METALLIB_BUILD. So the absent offline compiler does not "
                "block execution here -- it forces source compilation on a cold "
                "start, which is a cost, not a wall."
            ),
            "still_not_a_measurement": (
                "the compiler service was exercised, not the GPU: no command queue "
                "was created and nothing was dispatched"
            ),
        }
    if runtime_binding and runtime_binding.get("system_default"):
        v["runtime_binding"]["why"] = (
            f"metal crate {runtime_binding['metal_crate_version']} -- the same crate and "
            f"the same Device::system_default() call the runtime uses -- returned "
            f"{runtime_binding['system_default']!r} here. The language binding is not "
            f"the suspect; it is the binding that failed in the blocked run."
        )
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Test the campaign-wide claim that this host has no Metal-capable GPU, "
            "because a host property and a process property have different remedies."
        ),
        "evidence_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "why_not_a_measurement": (
            "device enumeration only. No command queue is created, no work is "
            "submitted, no lease is taken, and no timing is recorded."
        ),
        "gpu_authority": False,
        "observed": observed,
        "observed_runtime_binding": runtime_binding,
        "verdict": v,
        "claim_sites": claim_sites(),
        "blocked_downstream_if_true": [
            "Flash teacher capture (meta funnel gate 2: real_teacher_fit)",
            "every representation result that funnel gates",
            "physical NX qualification",
            "protected absolute measurement",
        ],
        "recovered_implementation": [
            "crates/hawking-core/src/metal/mod.rs raises Error::Metal on "
            "Device::system_default() returning None",
            "crates/hawking-core/examples/flash_meta_teacher_trace.rs writes the "
            "boundary receipt on any prefix-initialization error",
        ],
        "gaps_closed": [
            "nothing had tested the blocker; it was carried as a settled host fact",
        ],
        "negative_findings": [
            "the boundary receipt hardcodes status BLOCKED_NO_METAL_GPU for ANY "
            "prefix-initialization error, so that status is not by itself evidence "
            "that the GPU was the problem -- here the error string did name Metal",
            "this module cannot say why the Rust process saw no device",
        ],
        "resident_callable": {
            "entry_point": "tools.future.metal_reachability.probe()",
            "workunit": "one CPU_ANALYSIS unit; cheap; no GPU authority",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": "an unavailable probe records UNTESTED, never CONFIRMED",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/metal_reachability.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.probe:
        try:
            print(json.dumps(probe(), indent=1, sort_keys=True))
        except ProbeUnavailable as exc:
            print(json.dumps({"untested": str(exc)}, indent=1))
            return 1
        return 0
    out = build()
    print(out)
    print(json.dumps(json.loads(out.read_text())["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
