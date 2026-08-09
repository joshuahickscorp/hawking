"""Run one isolated, current-HCLI L0 route/hidden diagnostic capture.

The HQ30GR2 quality branch changes only L0/E0 gate/up after the L0 router.  A
historical HCLI record cannot establish which routes were active because its
selected-context text/token IDs were never persisted.  This operator consumes
the new, sealed compiler-only trace instead.  It never contacts the Qwen30
HTTP adapter or HCLI endpoint and it leaves the live watcher/server untouched.

The Rust helper intentionally executes only direct-packed embedding + L0
attention/post-attention norm/router for each exact source-template token.  It
persists device route IDs/weights and L0 router-input hidden buffers.  A typed
HQ30GR2 residual all-layer runtime does not exist, so candidate logits are
explicitly absent rather than approximated with a baseline logit or BF16
shadow path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.ascension_base_true_tps_gate import PROMPT_PROBES
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_TRACE_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_COMPILER_TRACE.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SELECTION_RECEIPT.json"
DEFAULT_CAUSAL_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CAUSAL_COVERAGE_CURRENT.json"
DEFAULT_CAPTURE_BINARY = (
    REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_current_hcli_layer0_route_capture"
)

INPUT_SCHEMA = "hawking.ascension.qwen30_current_hcli_layer0_route_capture_input.v1"
RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture.v1"
STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED"
TRACE_STATUS = "NEW_DIAGNOSTIC_NOT_HISTORICAL"
EXPECTED_PROBES = tuple(probe.identifier for probe in PROMPT_PROBES)


class CurrentRouteCaptureError(RuntimeError):
    """The source/compiler/candidate binding is incomplete or capture failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
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


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise CurrentRouteCaptureError(f"{label} is absent or has an invalid seal: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise CurrentRouteCaptureError(f"{label} is not an object")
    return dict(checked)


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurrentRouteCaptureError(f"{label} must be a non-empty string")
    return value


def _trace_input(
    *, trace_current_path: Path, selection_path: Path, causal_current_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    pointer = _sealed(trace_current_path, label="current HCLI compiler trace pointer")
    selected = pointer.get("compiler_trace_receipt")
    if pointer.get("status") != "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_HCLI_COMPILER_TRACE_SELECTED":
        raise CurrentRouteCaptureError("current compiler trace pointer is not selected")
    if not isinstance(selected, Mapping):
        raise CurrentRouteCaptureError("current compiler trace pointer lacks receipt binding")
    trace_receipt_path = Path(_nonempty_text(selected.get("path"), label="compiler trace receipt path"))
    trace = _sealed(trace_receipt_path, label="current HCLI compiler trace receipt")
    if trace.get("status") != "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_PRE_EXECUTION_HCLI_COMPILER_TRACE":
        raise CurrentRouteCaptureError("current compiler trace receipt does not have the expected diagnostic status")
    trace_binding = trace.get("binding")
    rows = trace.get("public_probe_compiler_traces")
    if not isinstance(trace_binding, Mapping) or not isinstance(rows, list):
        raise CurrentRouteCaptureError("current compiler trace receipt lacks binding/probe rows")
    run_root = Path(_nonempty_text(trace_binding.get("run_root"), label="compiler trace run root"))
    if not run_root.is_dir():
        raise CurrentRouteCaptureError("compiler trace run root is absent")
    probes: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CurrentRouteCaptureError("compiler trace probe row is not an object")
        probe_id = _nonempty_text(row.get("probe_id"), label="compiler trace probe id")
        annotated_relative = _nonempty_text(row.get("annotated_trace_path"), label=f"{probe_id} annotated trace path")
        annotated_path = run_root / annotated_relative
        try:
            annotated = json.loads(annotated_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurrentRouteCaptureError(f"{probe_id} annotated trace is unreadable: {exc}") from exc
        if _sha256_file(annotated_path) != row.get("annotated_trace_sha256"):
            raise CurrentRouteCaptureError(f"{probe_id} annotated trace hash differs from sealed compiler trace receipt")
        if not isinstance(annotated, Mapping):
            raise CurrentRouteCaptureError(f"{probe_id} annotated trace is not an object")
        compiler = annotated.get("compiler_trace")
        annotations = annotated.get("source_tokenizer_annotations")
        if not isinstance(compiler, Mapping) or not isinstance(annotations, Mapping):
            raise CurrentRouteCaptureError(f"{probe_id} annotated trace lacks compiler/tokenizer data")
        if compiler.get("status") != TRACE_STATUS or compiler.get("model_execution_started") is not False:
            raise CurrentRouteCaptureError(f"{probe_id} does not prove a pre-execution compiler trace")
        source_prompt = annotations.get("source_one_user_native_prompt")
        if not isinstance(source_prompt, Mapping):
            raise CurrentRouteCaptureError(f"{probe_id} lacks source template native prompt IDs")
        token_ids = source_prompt.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF for token in token_ids
        ):
            raise CurrentRouteCaptureError(f"{probe_id} source template token IDs are invalid")
        if source_prompt.get("add_special_tokens") is not True:
            raise CurrentRouteCaptureError(f"{probe_id} source template token IDs lack source-special-token binding")
        expected_hash = source_prompt.get("token_ids_u32le_sha256")
        actual_hash = hashlib.sha256(b"".join(int(token).to_bytes(4, "little") for token in token_ids)).hexdigest()
        if expected_hash != actual_hash:
            raise CurrentRouteCaptureError(f"{probe_id} source template token ID hash is invalid")
        probes.append(
            {
                "probe_id": probe_id,
                "compiler_trace": {
                    "annotated_trace_path": str(annotated_path.resolve()),
                    "annotated_trace_sha256": _sha256_file(annotated_path),
                    "raw_trace_path": str((annotated_path.parent / _nonempty_text(annotated.get("raw_compiler_trace", {}).get("path") if isinstance(annotated.get("raw_compiler_trace"), Mapping) else None, label=f"{probe_id} raw trace relative path")).resolve()),
                    "selected_context_span_count": len(annotations.get("selected_context_spans", [])),
                    "model_execution_started": False,
                },
                "source_one_user_native_prompt": {
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "token_ids_u32le_sha256": actual_hash,
                    "add_special_tokens": True,
                },
            }
        )
    if tuple(row["probe_id"] for row in probes) != EXPECTED_PROBES:
        raise CurrentRouteCaptureError("compiler trace rows are not exactly the three protected public probes")
    selection = _sealed(selection_path, label="HQ30GR2 selection receipt")
    selection_binding = selection.get("binding")
    if not isinstance(selection_binding, Mapping):
        raise CurrentRouteCaptureError("HQ30GR2 selection receipt lacks binding")
    baseline = selection_binding.get("baseline_control")
    source_audit = selection_binding.get("source_audit")
    source_revalidation = selection_binding.get("immutable_source_revalidation")
    if not isinstance(baseline, Mapping) or not isinstance(source_audit, Mapping) or not isinstance(source_revalidation, Mapping):
        raise CurrentRouteCaptureError("HQ30GR2 selection receipt lacks baseline/source binding")
    manifest = baseline.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CurrentRouteCaptureError("HQ30GR2 selection baseline lacks manifest")
    causal_pointer = _sealed(causal_current_path, label="HQ30GR2 causal coverage current pointer")
    causal_selected = causal_pointer.get("coverage_receipt")
    if not isinstance(causal_selected, Mapping):
        raise CurrentRouteCaptureError("HQ30GR2 causal coverage current pointer lacks receipt")
    causal_path = Path(_nonempty_text(causal_selected.get("path"), label="HQ30GR2 causal receipt path"))
    causal = _sealed(causal_path, label="HQ30GR2 causal coverage receipt")
    return (
        {
            "compiler_trace_current_pointer": {
                "path": str(trace_current_path.resolve()),
                "seal_sha256": pointer.get("seal_sha256"),
            },
            "compiler_trace_receipt": {
                "path": str(trace_receipt_path.resolve()),
                "seal_sha256": trace.get("seal_sha256"),
            },
            "candidate_selection": {
                "path": str(selection_path.resolve()),
                "seal_sha256": selection.get("seal_sha256"),
                "selected_organs": selection_binding.get("selected_organs"),
            },
            "candidate_causal_coverage": {
                "path": str(causal_path.resolve()),
                "seal_sha256": causal.get("seal_sha256"),
            },
            "baseline_direct_packed_control": {
                "manifest_path": _nonempty_text(manifest.get("path"), label="baseline manifest path"),
                "manifest_seal_sha256": _nonempty_text(manifest.get("seal_sha256"), label="baseline manifest seal"),
                "source_audit_seal_sha256": _nonempty_text(source_audit.get("seal_sha256"), label="source audit seal"),
                "source_revision": _nonempty_text(source_revalidation.get("source_revision"), label="source revision"),
            },
        },
        probes,
        causal,
    )


def prepare_input(
    *, root: Path, trace_current_path: Path, selection_path: Path, causal_current_path: Path
) -> tuple[Path, dict[str, Any]]:
    binding, probes, causal = _trace_input(
        trace_current_path=trace_current_path,
        selection_path=selection_path,
        causal_current_path=causal_current_path,
    )
    candidate_seal = binding["compiler_trace_receipt"]["seal_sha256"]
    document = seal(
        {
            "schema": INPUT_SCHEMA,
            "status": TRACE_STATUS,
            "recorded_at": _utc_now(),
            "binding": binding,
            "probes": probes,
            "claim_boundary": {
                "model_execution_started": False,
                "new_diagnostic_not_historical": True,
                "compiler_trace_selected_spans_and_source_template_token_ids_are_persisted": True,
                "baseline_l0_router_is_causally_pre_residual": True,
                "candidate_changes_only_l0_expert0_gate_up_after_l0_router_as_bound_by_causal_receipt": True,
                "does_not_claim_route_membership_until_the_isolated_metal_capture_completes": True,
            },
            "required_capture_boundary": {
                "metal_path": "direct-packed baseline embedding + L0 attention/postnorm/router only",
                "required_output": "per-token L0 top-8 routes/weights plus device-produced L0 router input hidden F32LE",
                "explicitly_not_executed": "L0 expert wave, later layers, final norm, lm_head, sampler, autoregressive feedback, generation, HCLI bench",
                "candidate_logit_divergence": "NOT_EXECUTED_UNTIL_TYPED_HQ30GR2_ALL_LAYER_DIAGNOSTIC_RUNTIME_EXISTS",
            },
            "causal_receipt_summary": {
                "status": causal.get("status"),
                "direct_middle_late_coverage": "ABSENT_BY_EXISTING_SEALED_CAUSAL_RECEIPT",
            },
        }
    )
    request_path = root / "current-hcli-route-capture/requests" / f"QWEN30_HQ30GR2_CURRENT_HCLI_L0_ROUTE_CAPTURE_INPUT_{candidate_seal}.json"
    if request_path.exists():
        existing = _sealed(request_path, label="existing current HCLI route capture input")
        comparable = dict(document)
        comparable["recorded_at"] = existing.get("recorded_at")
        comparable = seal({key: value for key, value in comparable.items() if key != "seal_sha256"})
        if existing != comparable:
            raise CurrentRouteCaptureError("refusing to overwrite a distinct current HCLI route capture input")
        return request_path, existing
    _atomic_json(request_path, document)
    return request_path, document


def _receipt_from_result(
    *,
    root: Path,
    input_path: Path,
    input_doc: Mapping[str, Any],
    capture_binary: Path,
    output_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentRouteCaptureError(f"capture result is unreadable: {exc}") from exc
    if not isinstance(result, Mapping):
        raise CurrentRouteCaptureError("capture result is not an object")
    if result.get("schema") != "hawking.ascension.qwen30_current_hcli_layer0_route_capture_result.v1":
        raise CurrentRouteCaptureError("capture result schema is unexpected")
    if result.get("status") != "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED":
        raise CurrentRouteCaptureError("capture result does not have its strict diagnostic status")
    if result.get("capture_protocol_revision") != "l0-route-hidden-capture-output-parent-v2":
        raise CurrentRouteCaptureError("capture result does not bind the successor route-capture protocol revision")
    input_binding = result.get("input")
    probe_rows = result.get("probes")
    runtime = result.get("runtime_binding")
    if not isinstance(input_binding, Mapping) or not isinstance(probe_rows, list) or not isinstance(runtime, Mapping):
        raise CurrentRouteCaptureError("capture result lacks input/probe/runtime details")
    if input_binding.get("sha256") != _sha256_file(input_path):
        raise CurrentRouteCaptureError("capture result does not bind exact input document")
    if runtime.get("runtime_executable_sha256") != _sha256_file(capture_binary):
        raise CurrentRouteCaptureError("capture result does not bind exact capture binary")
    expected = {row["probe_id"]: row for row in input_doc["probes"]}
    if tuple(row.get("probe_id") for row in probe_rows if isinstance(row, Mapping)) != EXPECTED_PROBES:
        raise CurrentRouteCaptureError("capture result probes differ from protected input order")
    summarized: list[dict[str, Any]] = []
    for probe in probe_rows:
        if not isinstance(probe, Mapping):
            raise CurrentRouteCaptureError("capture result probe is not an object")
        probe_id = _nonempty_text(probe.get("probe_id"), label="capture result probe id")
        steps = probe.get("steps")
        expected_probe = expected.get(probe_id)
        if not isinstance(steps, list) or not isinstance(expected_probe, Mapping):
            raise CurrentRouteCaptureError(f"{probe_id} capture result is malformed")
        expected_tokens = expected_probe["source_one_user_native_prompt"]["token_ids"]
        if len(steps) != len(expected_tokens):
            raise CurrentRouteCaptureError(f"{probe_id} capture did not process every sealed source-template token")
        active_e0_positions: list[int] = []
        hidden_files: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} is malformed")
            if step.get("position") != index or step.get("input_token_id") != expected_tokens[index]:
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} does not bind expected token/position")
            routes = step.get("selected_expert_ids")
            weights = step.get("normalized_route_weights")
            hidden = step.get("router_input_hidden_f32le")
            if not isinstance(routes, list) or len(routes) != 8 or not all(isinstance(item, int) for item in routes):
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} has invalid route IDs")
            if not isinstance(weights, list) or len(weights) != 8 or not all(isinstance(item, (int, float)) for item in weights):
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} has invalid route weights")
            if not isinstance(hidden, Mapping):
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} has no hidden capture")
            relative = _nonempty_text(hidden.get("relative_path"), label=f"{probe_id} hidden path")
            hidden_path = output_dir / relative
            if not hidden_path.is_file() or _sha256_file(hidden_path) != hidden.get("sha256"):
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} hidden payload is absent or hash-mismatched")
            if hidden.get("bytes") != 2048 * 4 or hidden.get("elements") != 2048:
                raise CurrentRouteCaptureError(f"{probe_id} capture step {index} hidden payload geometry is not Qwen30 hidden=2048")
            if 0 in routes:
                active_e0_positions.append(index)
            hidden_files.append({"relative_path": relative, "sha256": hidden.get("sha256")})
        summarized.append(
            {
                "probe_id": probe_id,
                "source_template_token_count": len(steps),
                "l0_expert0_selected_position_count": len(active_e0_positions),
                "l0_expert0_selected_positions": active_e0_positions,
                "route_membership_and_hidden_steps": len(hidden_files),
                "hidden_payloads": hidden_files,
            }
        )
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "binding": {
                "input_path": str(input_path.resolve()),
                "input_seal_sha256": input_doc.get("seal_sha256"),
                "capture_binary": {
                    "path": str(capture_binary.resolve()),
                    "sha256": _sha256_file(capture_binary),
                },
                "capture_output_root": str(output_dir.resolve()),
                "capture_result_path": str(result_path.resolve()),
                "capture_result_sha256": _sha256_file(result_path),
                "compiler_trace": input_doc["binding"]["compiler_trace_receipt"],
                "candidate_selection": input_doc["binding"]["candidate_selection"],
                "candidate_causal_coverage": input_doc["binding"]["candidate_causal_coverage"],
                "baseline_direct_packed_control": input_doc["binding"]["baseline_direct_packed_control"],
            },
            "probe_summary": summarized,
            "logit_divergence": {
                "status": "NOT_EXECUTED_FAIL_CLOSED",
                "reason": "typed HQ30GR2 residual all-layer candidate diagnostic runtime is absent; baseline logits cannot establish candidate-minus-control divergence",
            },
            "assessment": {
                "actual_current_hcli_l0_route_membership": "EARNED_FOR_EXACT_SEALED_CURRENT_COMPILER_INPUTS",
                "actual_current_hcli_l0_router_input_hidden": "EARNED_DEVICE_PRODUCED_F32LE_PER_TOKEN",
                "candidate_all_layer_or_logit_causal_reach": "NOT_EARNED",
            },
            "claim_boundary": {
                "new_diagnostic_not_historical": True,
                "baseline_l0_router_is_pre_candidate_residual_only": True,
                "l0_expert_wave_later_layers_lm_head_sampler_autoregressive_feedback_and_generation_not_executed": True,
                "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
                "server_runtime_watcher_and_adapter_untouched": True,
            },
        }
    )


def run_once(
    *,
    root: Path,
    trace_current_path: Path,
    selection_path: Path,
    causal_current_path: Path,
    capture_binary: Path,
    prepare_only: bool,
) -> dict[str, Any]:
    input_path, input_doc = prepare_input(
        root=root,
        trace_current_path=trace_current_path,
        selection_path=selection_path,
        causal_current_path=causal_current_path,
    )
    if prepare_only:
        return {
            "status": TRACE_STATUS,
            "input_path": str(input_path),
            "input_seal_sha256": input_doc.get("seal_sha256"),
            "prepared_only": True,
        }
    if not capture_binary.is_file() or not os.access(capture_binary, os.X_OK):
        raise CurrentRouteCaptureError(f"isolated route capture binary is not executable: {capture_binary}")
    baseline = input_doc["binding"]["baseline_direct_packed_control"]
    output_root = root / "current-hcli-route-capture"
    # The isolated Rust executable deliberately requires an existing parent
    # for a brand-new immutable run directory.  Create that parent here, not
    # inside the executable, so its create-new output invariant remains a
    # useful guard against accidental result replacement.
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    binary_sha = _sha256_file(capture_binary)
    output_dir = output_root / "runs" / f"{input_doc['seal_sha256']}_{binary_sha}"
    if output_dir.exists():
        raise CurrentRouteCaptureError(f"refusing to reuse capture output directory: {output_dir}")
    attempts_root = output_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{input_doc['seal_sha256']}_{binary_sha}"
    command = [
        str(capture_binary.resolve()),
        "--manifest",
        str(Path(baseline["manifest_path"]).resolve()),
        "--expected-manifest-seal-sha256",
        baseline["manifest_seal_sha256"],
        "--expected-source-audit-seal-sha256",
        baseline["source_audit_seal_sha256"],
        "--expected-source-revision",
        baseline["source_revision"],
        "--input-json",
        str(input_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--max-seq-len",
        "4096",
    ]
    running_status_path = root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_ROUTE_CAPTURE_STATUS.json"
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _atomic_json(
            running_status_path,
            {
                "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_status.v1",
                "phase": "NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_RUNNING",
                "recorded_at": _utc_now(),
                "capture_pid": process.pid,
                "input_path": str(input_path.resolve()),
                "input_seal_sha256": input_doc.get("seal_sha256"),
                "capture_binary_sha256": binary_sha,
                "output_dir": str(output_dir.resolve()),
                "command": command,
                "claim_boundary": "isolated Metal L0 capture only; no generation/HCLI/TPS",
            },
        )
        stdout, stderr = process.communicate(timeout=1800)
        returncode = process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            process.kill()
            process.communicate()
        raise CurrentRouteCaptureError(f"isolated L0 route capture could not run: {exc}") from exc
    stdout_path = attempts_root / f"{attempt_id}.stdout.txt"
    stderr_path = attempts_root / f"{attempt_id}.stderr.txt"
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    if returncode != 0:
        refusal = seal(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "BLOCKED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_REFUSED",
                "recorded_at": _utc_now(),
                "binding": {
                    "input_path": str(input_path.resolve()),
                    "input_seal_sha256": input_doc.get("seal_sha256"),
                    "capture_binary_path": str(capture_binary.resolve()),
                    "capture_binary_sha256": binary_sha,
                    "output_dir": str(output_dir.resolve()),
                    "command": command,
                    "returncode": returncode,
                    "stdout_path": str(stdout_path.resolve()),
                    "stdout_sha256": _sha256_file(stdout_path),
                    "stderr_path": str(stderr_path.resolve()),
                    "stderr_sha256": _sha256_file(stderr_path),
                },
                "claim_boundary": {
                    "new_diagnostic_not_historical": True,
                    "no_route_hidden_or_logit_claim_from_refused_capture": True,
                    "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
                },
            }
        )
        refusal_path = output_root / "negative-science" / f"QWEN30_HQ30GR2_CURRENT_HCLI_L0_ROUTE_CAPTURE_REFUSAL_{input_doc['seal_sha256']}_{binary_sha}.json"
        _atomic_json(refusal_path, refusal)
        _atomic_json(
            running_status_path,
            {
                "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_status.v1",
                "phase": "NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_REFUSED",
                "recorded_at": _utc_now(),
                "refusal_path": str(refusal_path.resolve()),
                "refusal_seal_sha256": refusal.get("seal_sha256"),
            },
        )
        raise CurrentRouteCaptureError(f"capture refused; sealed negative science at {refusal_path}")
    result_path = output_dir / "capture-result.json"
    receipt = _receipt_from_result(
        root=root,
        input_path=input_path,
        input_doc=input_doc,
        capture_binary=capture_binary,
        output_dir=output_dir,
        result_path=result_path,
    )
    receipt_path = output_root / "receipts" / f"QWEN30_HQ30GR2_CURRENT_HCLI_L0_ROUTE_CAPTURE_{input_doc['seal_sha256']}_{binary_sha}.json"
    _atomic_json(receipt_path, receipt)
    _atomic_json(
        root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_ROUTE_CAPTURE.json",
        seal(
            {
                "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_current.v1",
                "status": "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_SELECTED",
                "recorded_at": _utc_now(),
                "route_capture_receipt": {
                    "path": str(receipt_path.resolve()),
                    "seal_sha256": receipt.get("seal_sha256"),
                },
                "claim_boundary": receipt["claim_boundary"],
            }
        ),
    )
    _atomic_json(
        running_status_path,
        {
            "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_status.v1",
            "phase": "NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_COMPLETE",
            "recorded_at": _utc_now(),
            "receipt_path": str(receipt_path.resolve()),
            "receipt_seal_sha256": receipt.get("seal_sha256"),
        },
    )
    return {
        "status": receipt.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": receipt.get("seal_sha256"),
        "input_path": str(input_path),
        "input_seal_sha256": input_doc.get("seal_sha256"),
        "output_dir": str(output_dir),
        "probe_summary": receipt.get("probe_summary"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trace-current", type=Path, default=DEFAULT_TRACE_CURRENT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--causal-current", type=Path, default=DEFAULT_CAUSAL_CURRENT)
    parser.add_argument("--capture-binary", type=Path, default=DEFAULT_CAPTURE_BINARY)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            trace_current_path=args.trace_current.expanduser().resolve(),
            selection_path=args.selection.expanduser().resolve(),
            causal_current_path=args.causal_current.expanduser().resolve(),
            capture_binary=args.capture_binary.expanduser().resolve(),
            prepare_only=args.prepare_only,
        )
    except CurrentRouteCaptureError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_CURRENT_HCLI_L0_ROUTE_CAPTURE_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
