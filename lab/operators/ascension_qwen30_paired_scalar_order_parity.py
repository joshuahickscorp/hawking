"""One-shot parity gate for Qwen30's scalar-order paired gate/up candidate.

This controller is intentionally separate from the detached scalar runtime
watcher.  It cannot alter the serving kernel, start HCLI, write a TPS receipt,
or reuse the earlier explicit-FMA fusion result.  It first requires the sealed
CPU direct-packed arithmetic discriminator, then builds an isolated candidate
binary and, under one shared Qwen GPU lease, compares two exact source-template
generations against the current scalar controls while retaining per-route
native-device activation parity.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical"
RUNTIME_ROOT = PHYSICAL_ROOT / "qwen30/complete-runtime"
PROFILER_ROOT = PHYSICAL_ROOT / "qwen30/complete-token-profiler"
COMPLETE_ROOT = PHYSICAL_ROOT / "qwen30/complete-gravity"
CANONICAL_RUNTIME = RUNTIME_ROOT / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
SCALAR_A = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_PROMPT_A_RESULT.json"
SCALAR_B = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_PROMPT_B_RESULT.json"
CPU_PARITY = PROFILER_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_CPU_PARITY_RECEIPT.json"
RESULT_A = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_CANDIDATE_PROMPT_A_RESULT.json"
RESULT_B = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_CANDIDATE_PROMPT_B_RESULT.json"
RECEIPT = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY_RECEIPT.json"
ACTIVE = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY_ACTIVE.json"
LAST_PROCESS = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY_LAST_PROCESS.json"
BUILD_STDOUT = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_BUILD.stdout.log"
BUILD_STDERR = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_BUILD.stderr.log"
CANDIDATE_TARGET = REPO_ROOT / "workspace/ops/build/rust-qwen30-paired-scalar-order"
CANDIDATE_EXECUTABLE = (
    CANDIDATE_TARGET / "debug/examples/ascension_qwen30_complete_native_runtime"
)
INITIAL_COMPILE_REFUSAL = RECEIPT
SUCCESSOR_ATTEMPT = "msl-pragma-successor"
ATTEMPT_ID = "initial"
PRODUCTION_SUCCESSOR_CPU_PARITY = PROFILER_ROOT / (
    "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_"
    "MSL_PRAGMA_SUCCESSOR_CPU_PARITY_RECEIPT.json"
)
PRODUCTION_SUCCESSOR_RESULT_A = RUNTIME_ROOT / (
    "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_"
    "MSL_PRAGMA_SUCCESSOR_CANDIDATE_PROMPT_A_RESULT.json"
)
PRODUCTION_SUCCESSOR_RESULT_B = RUNTIME_ROOT / (
    "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_"
    "MSL_PRAGMA_SUCCESSOR_CANDIDATE_PROMPT_B_RESULT.json"
)
PRODUCTION_SUCCESSOR_TEMPLATE_PARITY = RUNTIME_ROOT / (
    "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_"
    "MSL_PRAGMA_SUCCESSOR_TEMPLATE_PARITY_RECEIPT.json"
)
LEASE_ROOT = PHYSICAL_ROOT / "qwen-family/dual-gravity"
LEASE_LOCK = LEASE_ROOT / ".gpu-lease.lock"
LEASE_STATUS = LEASE_ROOT / "GPU_LEASE_STATUS.json"
SCHEMA = "hawking.ascension.qwen30_paired_scalar_order_gate_up_template_parity.v1"
EXPECTED_RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
EXPECTED_RUNTIME_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
EXPECTED_RESULT_SCHEMA = "hawking.ascension.qwen30_complete_native_runtime_result.v1"
EXPECTED_RESULT_STATUS = "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED"
EXPECTED_KERNEL = "paired_direct_packed_gate_up_swiglu_scalar_order_candidate_with_device_control_parity"


class GateError(RuntimeError):
    """The current candidate cannot safely run or promote."""


def _configure_attempt(attempt: str) -> None:
    """Select immutable paths for one explicitly named source successor.

    The initial candidate's f594… refusal is evidence, not a scratch file.
    A spelling repair must therefore create an independently bound set of CPU,
    build, prompt, process, and template-parity records rather than overwrite
    any first-attempt path.
    """

    global ACTIVE, ATTEMPT_ID, BUILD_STDERR, BUILD_STDOUT, CANDIDATE_EXECUTABLE
    global CANDIDATE_TARGET, CPU_PARITY, LAST_PROCESS, RECEIPT, RESULT_A, RESULT_B
    if attempt == "initial":
        ATTEMPT_ID = attempt
        return
    if attempt != SUCCESSOR_ATTEMPT:
        raise GateError(f"unsupported immutable candidate attempt {attempt!r}")
    ATTEMPT_ID = attempt
    suffix = "_MSL_PRAGMA_SUCCESSOR"
    CPU_PARITY = PROFILER_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_CPU_PARITY_RECEIPT.json"
    )
    RESULT_A = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_CANDIDATE_PROMPT_A_RESULT.json"
    )
    RESULT_B = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_CANDIDATE_PROMPT_B_RESULT.json"
    )
    RECEIPT = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_TEMPLATE_PARITY_RECEIPT.json"
    )
    ACTIVE = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_TEMPLATE_PARITY_ACTIVE.json"
    )
    LAST_PROCESS = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_TEMPLATE_PARITY_LAST_PROCESS.json"
    )
    BUILD_STDOUT = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_BUILD.stdout.log"
    )
    BUILD_STDERR = RUNTIME_ROOT / (
        "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER"
        f"{suffix}_BUILD.stderr.log"
    )
    CANDIDATE_TARGET = REPO_ROOT / "workspace/ops/build/rust-qwen30-paired-scalar-order-pragma-successor"
    CANDIDATE_EXECUTABLE = (
        CANDIDATE_TARGET / "debug/examples/ascension_qwen30_complete_native_runtime"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise GateError(f"refusing to overwrite immutable evidence {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise GateError(f"JSON root must be an object: {path}")
    return dict(value)


def _sealed(path: Path) -> dict[str, Any]:
    try:
        return verify(_read_object(path), label=str(path))
    except Exception as exc:  # lab receipt errors have several public subclasses
        raise GateError(f"sealed document is invalid: {path}: {exc}") from exc


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} must be an object")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty string")
    return value


def _current_binding() -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _sealed(CANONICAL_RUNTIME)
    if runtime.get("schema") != EXPECTED_RUNTIME_SCHEMA or runtime.get("status") != EXPECTED_RUNTIME_STATUS:
        raise GateError("canonical Q30 runtime is not an exact native full-token PASS")
    binding = _mapping(runtime.get("binding"), "canonical runtime binding")
    for key in (
        "complete_manifest_seal_sha256",
        "runtime_executable_sha256",
        "source_content_identity_sha256",
        "source_revalidation_seal_sha256",
    ):
        _string(binding.get(key), f"canonical runtime binding {key}")
    return runtime, binding


def _control(path: Path, evidence: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    path_value = _string(evidence.get("path"), "canonical control path")
    if Path(path_value).resolve() != path.resolve():
        raise GateError(f"canonical runtime evidence does not bind expected scalar control {path}")
    expected_sha = _string(evidence.get("sha256"), "canonical control SHA-256")
    if _sha256_file(path) != expected_sha:
        raise GateError(f"scalar control file hash drifted: {path}")
    document = _read_object(path)
    runtime_binding = _mapping(document.get("runtime_binding"), f"{path.name} runtime binding")
    execution = _mapping(document.get("execution"), f"{path.name} execution")
    template = _mapping(execution.get("prompt_template"), f"{path.name} prompt template")
    if (
        document.get("schema") != EXPECTED_RESULT_SCHEMA
        or document.get("status") != EXPECTED_RESULT_STATUS
        or runtime_binding.get("manifest_seal_sha256") != binding["complete_manifest_seal_sha256"]
        or runtime_binding.get("packed_matvec_kernel") != "scalar_one_thread_per_row_control"
        or runtime_binding.get("gate_up_swiglu_kernel")
        != "three_dispatch_direct_packed_gate_up_swiglu_control"
        or template.get("source_template_bound") is not True
        or template.get("applied_to_prompt") is not True
        or execution.get("all_48_layers_executed_for_each_forward") is not True
        or execution.get("final_norm_lm_head_device_argmax_executed") is not True
        or execution.get("autoregressive_feedback_executed") is not True
    ):
        raise GateError(f"scalar control is not a current exact source-template native execution: {path}")
    return document


def _cpu_gate(runtime: Mapping[str, Any]) -> dict[str, Any]:
    document = _sealed(CPU_PARITY)
    observations = _mapping(document.get("observations"), "CPU parity observations")
    runtime_binding = _mapping(_mapping(document.get("binding"), "CPU parity binding").get("runtime"), "CPU parity runtime binding")
    if (
        document.get("schema")
        != "hawking.ascension.qwen30_direct_packed_gate_up_precision_order_discriminator.v1"
        or document.get("status")
        != "EARNED_CPU_DIRECT_PACKED_GATE_UP_ORDER_PRECISION_DISCRIMINATOR"
        or document.get("outcome")
        != "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT"
        or observations.get("scalar_control_vs_paired_scalar_order_nonfused_difference_observed")
        is not False
        or runtime_binding.get("seal_sha256") != runtime.get("seal_sha256")
    ):
        raise GateError("CPU scalar-order paired parity gate is absent, stale, or not exact")
    return document


def _initial_compile_refusal(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Require the first compiler refusal as an immutable predecessor.

    A successor is allowed only for the documented unsupported syntax. It is
    not a generic rerun mechanism for an observed numerical/template failure.
    """

    document = _sealed(INITIAL_COMPILE_REFUSAL)
    binding = _mapping(document.get("binding"), "initial refusal binding")
    results = _mapping(document.get("candidate_results"), "initial refusal candidate results")
    facts = _mapping(
        document.get("all_layer_device_parity_and_exact_completion_parity"),
        "initial refusal device facts",
    )
    failures = document.get("failures")
    if not (
        document.get("schema") == SCHEMA
        and document.get("status")
        == "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
        and binding.get("scalar_runtime_receipt_path") == str(CANONICAL_RUNTIME)
        and binding.get("scalar_runtime_receipt_seal_sha256") == runtime.get("seal_sha256")
        and binding.get("scalar_runtime_executable_sha256")
        == _mapping(runtime.get("binding"), "canonical runtime binding").get(
            "runtime_executable_sha256"
        )
        and results.get("prompt_a_sha256") is None
        and results.get("prompt_b_sha256") is None
        and not facts
        and isinstance(failures, list)
        and any("candidate prompt A returned 2" == value for value in failures)
    ):
        raise GateError(
            "initial scalar-order result is not the sealed MSL compile refusal required "
            "for this successor"
        )
    return document


def production_no_parity_requalification_binding() -> dict[str, Any]:
    """Return the exact candidate authority needed to *prepare* a transition.

    This deliberately creates no runtime, binary, receipt, endpoint, or
    selection. It is a fail-closed handoff for a later explicitly authorized
    production requalification: the no-parity execution option may only be
    considered when the sealed scalar-order candidate completed both source
    templates with all-layer native device parity against the live scalar
    control, and its direct-packed CPU gate remains exact.
    """

    runtime, scalar_binding = _current_binding()
    receipt = _sealed(PRODUCTION_SUCCESSOR_TEMPLATE_PARITY)
    observed = _mapping(receipt.get("binding"), "production successor binding")
    predecessor = _mapping(
        observed.get("predecessor_compile_refusal"),
        "production successor predecessor refusal",
    )
    cpu_gate = _mapping(receipt.get("cpu_scalar_order_gate"), "production successor CPU gate")
    results = _mapping(receipt.get("candidate_results"), "production successor results")
    facts = _mapping(
        receipt.get("all_layer_device_parity_and_exact_completion_parity"),
        "production successor parity facts",
    )
    cpu_document = _sealed(PRODUCTION_SUCCESSOR_CPU_PARITY)
    cpu_observations = _mapping(cpu_document.get("observations"), "production CPU observations")
    source_hashes = _mapping(observed.get("candidate_source_sha256"), "candidate source hashes")
    current_shader_sha = _sha256_file(
        REPO_ROOT / "crates/hawking-core/shaders/qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal"
    )
    if not (
        receipt.get("schema") == SCHEMA
        and receipt.get("status")
        == "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
        and observed.get("attempt_id") == SUCCESSOR_ATTEMPT
        and observed.get("complete_manifest_seal_sha256")
        == scalar_binding["complete_manifest_seal_sha256"]
        and observed.get("scalar_runtime_receipt_path") == str(CANONICAL_RUNTIME)
        and observed.get("scalar_runtime_receipt_seal_sha256") == runtime.get("seal_sha256")
        and observed.get("scalar_runtime_executable_sha256")
        == scalar_binding["runtime_executable_sha256"]
        and predecessor.get("path") == str(INITIAL_COMPILE_REFUSAL)
        and predecessor.get("status")
        == "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
        and isinstance(predecessor.get("seal_sha256"), str)
        and cpu_gate.get("path") == str(PRODUCTION_SUCCESSOR_CPU_PARITY)
        and cpu_gate.get("seal_sha256") == cpu_document.get("seal_sha256")
        and cpu_gate.get("outcome")
        == "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT"
        and cpu_observations.get(
            "scalar_control_vs_paired_scalar_order_nonfused_difference_observed"
        )
        is False
        and source_hashes.get("paired_scalar_order_shader") == current_shader_sha
        and not receipt.get("failures")
    ):
        raise GateError(
            "no-parity production handoff lacks the current sealed scalar-order candidate/CPU binding"
        )
    for label, result_path in (("a", PRODUCTION_SUCCESSOR_RESULT_A), ("b", PRODUCTION_SUCCESSOR_RESULT_B)):
        expected_sha = _string(results.get(f"prompt_{label}_sha256"), f"candidate prompt {label} SHA")
        if _sha256_file(result_path) != expected_sha:
            raise GateError(f"candidate prompt {label} result drifted from its sealed parity receipt")
        device = _mapping(
            _mapping(facts.get(f"prompt_{label}_native_device_parity"), f"prompt {label} device facts").get(
                "device_parity"
            ),
            f"prompt {label} device parity",
        )
        completion = _mapping(
            facts.get(f"prompt_{label}_exact_token_parity"),
            f"prompt {label} exact completion parity",
        )
        exact_completion = all(
            _mapping(completion.get(field), f"prompt {label} {field}").get("candidate")
            == _mapping(completion.get(field), f"prompt {label} {field}").get("control")
            for field in (
                "prompt_token_ids",
                "completion_token_ids",
                "full_model_forward_count",
                "completion_feedback_full_forwards",
            )
        )
        if not (
            device.get("enabled") is True
            and device.get("valid") is True
            and device.get("all_selected_route_major_activations_compared_on_device") is True
            and device.get("full_model_forwards_without_device_parity") == 0
            and isinstance(device.get("full_model_forwards_compared"), int)
            and device["full_model_forwards_compared"] > 0
            and isinstance(device.get("layers_compared"), int)
            and device["layers_compared"] == device["full_model_forwards_compared"] * 48
            and isinstance(device.get("max_abs_error"), (int, float))
            and isinstance(device.get("tolerance_max_abs"), (int, float))
            and float(device["max_abs_error"]) <= float(device["tolerance_max_abs"])
            and exact_completion
        ):
            raise GateError(
                f"candidate prompt {label} does not prove all-layer device/template parity"
            )
    return {
        "production_gate_up_swiglu_kernel": "paired-scalar-order-production-no-parity",
        "production_kernel_receipt_id": "paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity",
        "candidate_template_parity_receipt_path": str(PRODUCTION_SUCCESSOR_TEMPLATE_PARITY),
        "candidate_template_parity_receipt_seal_sha256": receipt["seal_sha256"],
        "candidate_cpu_parity_receipt_path": str(PRODUCTION_SUCCESSOR_CPU_PARITY),
        "candidate_cpu_parity_receipt_seal_sha256": cpu_document["seal_sha256"],
        "candidate_shader_sha256": current_shader_sha,
        "scalar_control_runtime_receipt_path": str(CANONICAL_RUNTIME),
        "scalar_control_runtime_receipt_seal_sha256": runtime["seal_sha256"],
        "scalar_control_runtime_executable_sha256": scalar_binding["runtime_executable_sha256"],
        "predecessor_compile_refusal_path": str(INITIAL_COMPILE_REFUSAL),
        "predecessor_compile_refusal_seal_sha256": predecessor["seal_sha256"],
        "claim_boundary": {
            "preflight_handoff_only_not_a_runtime_transition": True,
            "does_not_select_or_serve_the_no_parity_kernel": True,
            "requires_fresh_preflight_full_token_template_profile_then_hcli": True,
        },
    }


def _candidate_sources() -> dict[str, str]:
    paths = {
        "paired_scalar_order_shader": REPO_ROOT
        / "crates/hawking-core/shaders/qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal",
        "qwen30_complete_runtime": REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs",
        "metal_registry": REPO_ROOT / "crates/hawking-core/src/metal/mod.rs",
        "native_runtime_entrypoint": REPO_ROOT
        / "crates/hawking-core/examples/ascension_qwen30_complete_native_runtime.rs",
        "paired_parity_controller": Path(__file__).resolve(),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _build() -> dict[str, Any]:
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_complete_native_runtime",
    ]
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(CANDIDATE_TARGET)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env=environment,
    )
    _atomic_text(BUILD_STDOUT, completed.stdout)
    _atomic_text(BUILD_STDERR, completed.stderr)
    built = completed.returncode == 0 and CANDIDATE_EXECUTABLE.is_file()
    result = {
        "command": command,
        "cargo_target_dir": str(CANDIDATE_TARGET),
        "returncode": completed.returncode,
        "stdout_path": str(BUILD_STDOUT),
        "stderr_path": str(BUILD_STDERR),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "candidate_executable_path": str(CANDIDATE_EXECUTABLE),
        "candidate_executable_sha256": _sha256_file(CANDIDATE_EXECUTABLE) if built else None,
        "built": built,
        "claim_boundary": "isolated candidate build only; it does not alter the scalar control runtime or HTTP adapter",
    }
    if not built:
        raise GateError("isolated scalar-order candidate build failed")
    return result


def _port_closed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        if sock.connect_ex(("127.0.0.1", 18430)) == 0:
            raise GateError("HCLI adapter port 18430 is open; candidate parity requires serving to stay closed")


@contextlib.contextmanager
def _quiet_lease(stage: str) -> Iterator[None]:
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    with LEASE_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError("shared Qwen GPU lease is busy; candidate was not started") from exc
        _atomic_json(
            LEASE_STATUS,
            {
                "schema": "hawking.ascension.gpu_lease.v1",
                "status": "ACTIVE_EXCLUSIVE_GPU_LEASE",
                "worker": "qwen30-native-runtime",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "stage": stage,
                "acquired_at": _utc_now(),
                "claim_boundary": "bounded current-binding parity lease only; not a clean performance benchmark",
            },
        )
        try:
            yield
        finally:
            _atomic_json(
                LEASE_STATUS,
                {
                    "schema": "hawking.ascension.gpu_lease.v1",
                    "status": "RELEASED",
                    "worker": "qwen30-native-runtime",
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "stage": stage,
                    "released_at": _utc_now(),
                    "claim_boundary": "bounded current-binding parity lease only; not a clean performance benchmark",
                },
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _candidate_command(binding: Mapping[str, Any], prompt: str) -> list[str]:
    manifest = COMPLETE_ROOT / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    return [
        str(CANDIDATE_EXECUTABLE),
        "--manifest",
        str(manifest),
        "--expected-manifest-seal-sha256",
        str(binding["complete_manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        "00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b",
        "--expected-source-revision",
        "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        "--mode",
        "generate-greedy",
        "--prompt",
        prompt,
        "--prompt-template",
        "source-user-chat",
        "--packed-matvec-kernel",
        "control",
        "--gate-up-swiglu-kernel",
        "paired-scalar-order-candidate-device-parity",
        "--max-new-tokens",
        "2",
        "--max-seq-len",
        "256",
    ]


def _parse_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    raise GateError("candidate stdout did not contain a JSON runtime result")


def _result_valid(
    document: Mapping[str, Any], binding: Mapping[str, Any], executable_sha256: str
) -> tuple[bool, dict[str, Any]]:
    runtime_binding = _mapping(document.get("runtime_binding"), "candidate runtime binding")
    execution = _mapping(document.get("execution"), "candidate execution")
    parity = _mapping(execution.get("gate_up_swiglu_device_control_parity"), "candidate device parity")
    prompt_template = _mapping(execution.get("prompt_template"), "candidate prompt template")
    full_forwards = execution.get("full_model_forward_count")
    expected_layers = full_forwards * 48 if isinstance(full_forwards, int) and full_forwards > 0 else None
    expected_routes = expected_layers * 8 if isinstance(expected_layers, int) else None
    expected_values = expected_routes * 768 if isinstance(expected_routes, int) else None
    max_error = parity.get("max_abs_error")
    tolerance = parity.get("tolerance_max_abs")
    valid = (
        document.get("schema") == EXPECTED_RESULT_SCHEMA
        and document.get("status") == EXPECTED_RESULT_STATUS
        and document.get("runtime_executable_sha256") == executable_sha256
        and runtime_binding.get("manifest_seal_sha256") == binding["complete_manifest_seal_sha256"]
        and runtime_binding.get("source_revision") == "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
        and runtime_binding.get("packed_matvec_kernel") == "scalar_one_thread_per_row_control"
        and runtime_binding.get("gate_up_swiglu_kernel") == EXPECTED_KERNEL
        and execution.get("all_48_layers_executed_for_each_forward") is True
        and execution.get("final_norm_lm_head_device_argmax_executed") is True
        and execution.get("autoregressive_feedback_executed") is True
        and prompt_template.get("source_template_bound") is True
        and prompt_template.get("applied_to_prompt") is True
        and parity.get("enabled") is True
        and parity.get("valid") is True
        and parity.get("all_selected_route_major_activations_compared_on_device") is True
        and parity.get("full_model_forwards_without_device_parity") == 0
        and parity.get("full_model_forwards_compared") == full_forwards
        and parity.get("layers_compared") == expected_layers
        and parity.get("routed_experts_compared") == expected_routes
        and parity.get("activation_values_compared") == expected_values
        and isinstance(max_error, (int, float))
        and not isinstance(max_error, bool)
        and isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and float(max_error) <= float(tolerance)
    )
    return valid, {
        "runtime_binding": runtime_binding,
        "full_model_forward_count": full_forwards,
        "expected_layers_compared": expected_layers,
        "expected_routed_experts_compared": expected_routes,
        "expected_activation_values_compared": expected_values,
        "device_parity": parity,
    }


def _template_parity(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    control_execution = _mapping(control.get("execution"), "scalar control execution")
    candidate_execution = _mapping(candidate.get("execution"), "candidate execution")
    fields = (
        "prompt_token_ids",
        "completion_token_ids",
        "full_model_forward_count",
        "completion_feedback_full_forwards",
    )
    detail = {field: {"control": control_execution.get(field), "candidate": candidate_execution.get(field)} for field in fields}
    exact = all(detail[field]["control"] == detail[field]["candidate"] for field in fields)
    return exact, detail


def _run_prompt(
    *, label: str, prompt: str, binding: Mapping[str, Any], executable_sha256: str, result_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = _candidate_command(binding, prompt)
    stdout_path = result_path.with_suffix(".stdout.log")
    stderr_path = result_path.with_suffix(".stderr.log")
    active = {
        "schema": "hawking.ascension.qwen30_paired_scalar_order_process.v1",
        "phase": "RUNNING",
        "label": label,
        "started_at": _utc_now(),
        "command": command,
        "candidate_executable_sha256": executable_sha256,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    _atomic_json(ACTIVE, active)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    _atomic_text(stdout_path, completed.stdout)
    _atomic_text(stderr_path, completed.stderr)
    terminal = {
        **active,
        "phase": "EXITED",
        "finished_at": _utc_now(),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    if completed.returncode != 0:
        terminal["outcome"] = "CANDIDATE_EXECUTION_FAILED"
        _atomic_json(LAST_PROCESS, terminal)
        _atomic_json(ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise GateError(f"candidate prompt {label} returned {completed.returncode}")
    document = _parse_stdout(completed.stdout)
    valid, facts = _result_valid(document, binding, executable_sha256)
    if not valid:
        terminal["outcome"] = "CANDIDATE_RESULT_FAILED_NATIVE_DEVICE_PARITY_CONTRACT"
        terminal["facts"] = facts
        _atomic_json(LAST_PROCESS, terminal)
        _atomic_json(ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise GateError(f"candidate prompt {label} failed its native-device parity contract")
    _atomic_json(result_path, document, overwrite=False)
    terminal["outcome"] = "EARNED_CANDIDATE_RESULT_WRITTEN"
    terminal["result_sha256"] = _sha256_file(result_path)
    _atomic_json(LAST_PROCESS, terminal)
    _atomic_json(ACTIVE, {**terminal, "phase": "TERMINAL"})
    return document, facts


def run() -> int:
    if RECEIPT.exists() or RESULT_A.exists() or RESULT_B.exists():
        raise GateError("immutable paired scalar-order output already exists; refusing a silent retry")
    _port_closed()
    runtime, binding = _current_binding()
    predecessor = _initial_compile_refusal(runtime) if ATTEMPT_ID == SUCCESSOR_ATTEMPT else None
    evidence = _mapping(runtime.get("evidence"), "canonical runtime evidence")
    control_a = _control(SCALAR_A, _mapping(evidence.get("source_user_prompt_a"), "prompt A evidence"), binding)
    control_b = _control(SCALAR_B, _mapping(evidence.get("source_user_prompt_b"), "prompt B evidence"), binding)
    cpu_gate = _cpu_gate(runtime)
    source_hashes = _candidate_sources()
    build = _build()
    executable_sha256 = _string(build.get("candidate_executable_sha256"), "candidate executable SHA-256")
    failures: list[str] = []
    candidate_a: dict[str, Any] | None = None
    candidate_b: dict[str, Any] | None = None
    facts: dict[str, Any] = {}
    try:
        with _quiet_lease("qwen30_paired_scalar_order_gate_up_source_template_parity"):
            candidate_a, facts_a = _run_prompt(
                label="A",
                prompt="Reply with the single word native.",
                binding=binding,
                executable_sha256=executable_sha256,
                result_path=RESULT_A,
            )
            facts["prompt_a_native_device_parity"] = facts_a
            matches_a, detail_a = _template_parity(control_a, candidate_a)
            facts["prompt_a_exact_token_parity"] = detail_a
            if not matches_a:
                failures.append("prompt A exact completion/template parity differs from scalar control")
            candidate_b, facts_b = _run_prompt(
                label="B",
                prompt="Write a one-line Python function named add.",
                binding=binding,
                executable_sha256=executable_sha256,
                result_path=RESULT_B,
            )
            facts["prompt_b_native_device_parity"] = facts_b
            matches_b, detail_b = _template_parity(control_b, candidate_b)
            facts["prompt_b_exact_token_parity"] = detail_b
            if not matches_b:
                failures.append("prompt B exact completion/template parity differs from scalar control")
    except (GateError, subprocess.TimeoutExpired) as exc:
        failures.append(str(exc))
    passed = not failures and candidate_a is not None and candidate_b is not None
    payload = {
        "schema": SCHEMA,
        "status": (
            "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
            if passed
            else "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
        ),
        "recorded_at": _utc_now(),
        "binding": {
            "attempt_id": ATTEMPT_ID,
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "complete_manifest_seal_sha256": binding["complete_manifest_seal_sha256"],
            "source_content_identity_sha256": binding["source_content_identity_sha256"],
            "source_revalidation_seal_sha256": binding["source_revalidation_seal_sha256"],
            "scalar_runtime_receipt_path": str(CANONICAL_RUNTIME),
            "scalar_runtime_receipt_seal_sha256": runtime["seal_sha256"],
            "scalar_runtime_executable_sha256": binding["runtime_executable_sha256"],
            "candidate_runtime_executable_path": str(CANDIDATE_EXECUTABLE),
            "candidate_runtime_executable_sha256": executable_sha256,
            "candidate_source_sha256": source_hashes,
            "predecessor_compile_refusal": (
                {
                    "path": str(INITIAL_COMPILE_REFUSAL),
                    "seal_sha256": predecessor["seal_sha256"],
                    "status": predecessor["status"],
                    "claim_boundary": "historical compiler refusal only; no device/template result is reused",
                }
                if predecessor is not None
                else None
            ),
        },
        "cpu_scalar_order_gate": {
            "path": str(CPU_PARITY),
            "sha256": _sha256_file(CPU_PARITY),
            "seal_sha256": cpu_gate["seal_sha256"],
            "outcome": cpu_gate["outcome"],
            "paired_scalar_order_cpu_bitwise_exact": True,
        },
        "build": build,
        "scalar_controls": {
            "prompt_a_path": str(SCALAR_A),
            "prompt_a_sha256": _sha256_file(SCALAR_A),
            "prompt_b_path": str(SCALAR_B),
            "prompt_b_sha256": _sha256_file(SCALAR_B),
        },
        "candidate_results": {
            "prompt_a_path": str(RESULT_A),
            "prompt_a_sha256": _sha256_file(RESULT_A) if RESULT_A.exists() else None,
            "prompt_b_path": str(RESULT_B),
            "prompt_b_sha256": _sha256_file(RESULT_B) if RESULT_B.exists() else None,
        },
        "all_layer_device_parity_and_exact_completion_parity": facts,
        "failures": failures,
        "gpu_lease_status_path": str(LEASE_STATUS),
        "claim_boundary": {
            "candidate_is_not_selected_for_scalar_runtime_or_http_adapter": True,
            "candidate_uses_only_admitted_direct_packed_weights_and_native_metal": True,
            "explicit_fma_candidate_and_its_prior_rejection_are_not_reused": True,
            "each_candidate_forward_requires_route_major_device_control_parity": True,
            "does_not_claim_hcli_tps_tg_coherence_capability_or_tournament": True,
            "fresh_complete_token_profile_and_clean_HCLI_TPS_gate_remain_required": True,
        },
    }
    _atomic_json(RECEIPT, seal(payload), overwrite=False)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", choices=("run",), default="run")
    parser.add_argument("--attempt", choices=("initial", SUCCESSOR_ATTEMPT), default="initial")
    arguments = parser.parse_args()
    try:
        _configure_attempt(arguments.attempt)
        return run()
    except GateError as exc:
        print(f"qwen30 paired scalar-order parity refused: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
