"""Capture a new, pre-execution HCLI compiler trace for the HQ30GR2 branch.

The prior coherence probes are immutable negative science: their persisted
``context.compiled`` events do not include the selected span bodies or native
token IDs.  This operator does not try to replay them from a mutable memory
database.  Instead it runs the real HCLI compiler in a fresh isolated
workspace with its explicit trace-only guard enabled.  The guard writes the
compiled context and then refuses *before* ``HttpModelProvider.generate`` can
be reached.  The operator subsequently annotates every compiler-selected span
and both prompt boundaries with the exact source-tokenizer IDs.

This is intentionally a diagnostic input-capture operation only.  It does not
load the HQ30GR2 candidate, start Qwen30's server or runtime watcher, use
Metal, produce logits, recover historical trajectories, or claim coherence,
HCLI, TPS, TG, capability, or tournament evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer

from lab.operators.ascension_base_true_tps_gate import PROMPT_PROBES
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CANDIDATE = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SELECTION_RECEIPT.json"
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SOURCE_BINDING_SNAPSHOT.json"
DEFAULT_ADMISSION_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_NATIVE_ADMISSION_CURRENT.json"
DEFAULT_HCLI = REPO_ROOT / "workspace/ops/build/rust/debug/hcli"
DEFAULT_SOURCE_MODEL = (
    REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
DEFAULT_TOKENIZER = DEFAULT_SOURCE_MODEL / "tokenizer.json"
DEFAULT_TEMPLATE = DEFAULT_SOURCE_MODEL / "chat_template.jinja"
DEFAULT_TOKENIZER_CONFIG = DEFAULT_SOURCE_MODEL / "tokenizer_config.json"
DEFAULT_RENDERER_SOURCE = REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"

SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace.v1"
STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_PRE_EXECUTION_HCLI_COMPILER_TRACE"
TRACE_SCHEMA = "hawking.ascension.hcli_compiler_pre_execution_trace.v1"
TRACE_MODE = "NEW_DIAGNOSTIC_NOT_HISTORICAL"
TRACE_ENDPOINT = "http://127.0.0.1:9"
SOURCE_TEMPLATE_RENDERER = "hawking-core::model::qwen30_complete_runtime::render_source_user_chat_template"
SOURCE_TEMPLATE_ANCHORS = (
    "{%- for message in loop_messages %}",
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
    "{%- if add_generation_prompt %}",
    "{{- '<|im_start|>assistant\\n' }}",
)


class CurrentHcliTraceError(RuntimeError):
    """The new trace is incomplete or does not meet its pre-execution boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise CurrentHcliTraceError(f"{label} is absent or has an invalid seal: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise CurrentHcliTraceError(f"{label} is not an object")
    return dict(checked)


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurrentHcliTraceError(f"{label} must be a non-empty string")
    return value


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CurrentHcliTraceError(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _candidate_binding(
    *,
    candidate_path: Path,
    selection_path: Path,
    source_snapshot_path: Path,
    admission_current_path: Path,
) -> dict[str, Any]:
    candidate = _sealed(candidate_path, label="HQ30GR2 candidate manifest")
    selection = _sealed(selection_path, label="HQ30GR2 selection receipt")
    snapshot = _sealed(source_snapshot_path, label="HQ30GR2 source snapshot")
    current = _sealed(admission_current_path, label="HQ30GR2 admission current pointer")
    candidate_seal = _nonempty_text(candidate.get("seal_sha256"), label="candidate seal")
    if candidate.get("schema") != "hawking.ascension.qwen30_quality_repack_candidate.v1":
        raise CurrentHcliTraceError("candidate does not use the HQ30GR2 candidate grammar")
    if candidate.get("status") != "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED":
        raise CurrentHcliTraceError("candidate is no longer the unqualified HQ30GR2 branch")
    selection_binding = selection.get("binding")
    if not isinstance(selection_binding, Mapping):
        raise CurrentHcliTraceError("selection receipt lacks binding")
    if selection_binding.get("branch_id") != "qwen30-gate-up-sparse-fp16-residual-v1":
        raise CurrentHcliTraceError("selection receipt does not bind the HQ30GR2 branch")
    if selection_binding.get("selected_organs") != [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]:
        raise CurrentHcliTraceError("selection receipt does not bind exactly the L0/E0 gate/up pair")
    snapshot_binding = snapshot.get("binding")
    if not isinstance(snapshot_binding, Mapping):
        raise CurrentHcliTraceError("source snapshot lacks binding")
    current_manifest = current.get("complete_manifest")
    current_admission = current.get("admission_receipt")
    if not isinstance(current_manifest, Mapping) or not isinstance(current_admission, Mapping):
        raise CurrentHcliTraceError("candidate admission pointer is malformed")
    if current_manifest.get("seal_sha256") != candidate_seal:
        raise CurrentHcliTraceError("candidate admission pointer does not bind candidate manifest")
    if current.get("status") != "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED":
        raise CurrentHcliTraceError("candidate admission pointer is not current")
    return {
        "candidate_manifest": {
            "path": str(candidate_path.resolve()),
            "seal_sha256": candidate_seal,
            "document_sha256": _sha256_file(candidate_path),
        },
        "selection_receipt": {
            "path": str(selection_path.resolve()),
            "seal_sha256": selection.get("seal_sha256"),
            "document_sha256": _sha256_file(selection_path),
        },
        "source_snapshot": {
            "path": str(source_snapshot_path.resolve()),
            "seal_sha256": snapshot.get("seal_sha256"),
            "document_sha256": _sha256_file(source_snapshot_path),
            "immutable_source_revalidation": snapshot_binding.get("immutable_source_revalidation"),
        },
        "candidate_native_admission": {
            "current_pointer_path": str(admission_current_path.resolve()),
            "current_pointer_seal_sha256": current.get("seal_sha256"),
            "admission_receipt": current_admission,
        },
    }


def _token_id_record(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
    try:
        token_ids = list(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)
    except Exception as exc:  # tokenizers exposes a Rust exception family
        raise CurrentHcliTraceError(f"source tokenizer failed: {exc}") from exc
    if any(not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF for token in token_ids):
        raise CurrentHcliTraceError("source tokenizer returned an invalid unsigned token ID")
    encoded = b"".join(struct.pack("<I", token) for token in token_ids)
    return {
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "token_ids_u32le_sha256": _sha256_bytes(encoded),
        "add_special_tokens": add_special_tokens,
    }


def _render_source_one_user_template(folded_prompt: str) -> str:
    return f"<|im_start|>user\n{folded_prompt}<|im_end|>\n<|im_start|>assistant\n"


def _validate_source_template(template_path: Path, tokenizer_config_path: Path) -> dict[str, Any]:
    template_bytes = template_path.read_bytes()
    template_text = template_bytes.decode("utf-8")
    for anchor in SOURCE_TEMPLATE_ANCHORS:
        if anchor not in template_text:
            raise CurrentHcliTraceError(f"source template is missing required one-user anchor {anchor!r}")
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentHcliTraceError(f"source tokenizer config is unreadable: {exc}") from exc
    configured = tokenizer_config.get("chat_template") if isinstance(tokenizer_config, Mapping) else None
    if not isinstance(configured, str) or configured.encode("utf-8") != template_bytes:
        raise CurrentHcliTraceError("source tokenizer_config chat_template does not byte-match chat_template.jinja")
    return {
        "source_template": _file_binding(template_path),
        "source_tokenizer_config": _file_binding(tokenizer_config_path),
        "renderer": SOURCE_TEMPLATE_RENDERER,
        "renderer_source": _file_binding(DEFAULT_RENDERER_SOURCE),
        "one_user_branch_anchors": list(SOURCE_TEMPLATE_ANCHORS),
    }


def _raw_trace(raw_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentHcliTraceError(f"raw compiler trace is unreadable: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CurrentHcliTraceError("raw compiler trace is not an object")
    raw = dict(value)
    if raw.get("schema") != TRACE_SCHEMA or raw.get("status") != TRACE_MODE:
        raise CurrentHcliTraceError("raw compiler trace does not use the trace-only schema/mode")
    if raw.get("capture_timing") != "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION":
        raise CurrentHcliTraceError("raw compiler trace has an unexpected capture timing")
    if raw.get("model_execution_started") is not False:
        raise CurrentHcliTraceError("raw compiler trace says model execution started")
    spans = raw.get("selected_context_spans")
    if not isinstance(spans, list):
        raise CurrentHcliTraceError("raw compiler trace lacks selected context spans")
    if not isinstance(raw.get("folded_native_prompt_utf8"), str):
        raise CurrentHcliTraceError("raw compiler trace lacks folded native prompt text")
    return raw


def annotate_compiler_trace(
    *,
    raw_trace: Mapping[str, Any],
    tokenizer: Tokenizer,
    source_template_binding: Mapping[str, Any],
    raw_trace_path: Path,
) -> dict[str, Any]:
    """Copy a raw pre-execution trace and add exact source-tokenizer IDs."""

    raw = dict(raw_trace)
    spans = raw.get("selected_context_spans")
    if not isinstance(spans, list):
        raise CurrentHcliTraceError("raw compiler trace selected spans are not a list")
    annotated_spans: list[dict[str, Any]] = []
    for ordinal, span in enumerate(spans):
        if not isinstance(span, Mapping):
            raise CurrentHcliTraceError(f"selected span {ordinal} is not an object")
        text = span.get("text")
        if not isinstance(text, str):
            raise CurrentHcliTraceError(f"selected span {ordinal} has no exact text")
        item = dict(span)
        item["text_utf8_sha256"] = _sha256_bytes(text.encode("utf-8"))
        item["text_utf8_bytes"] = len(text.encode("utf-8"))
        item["hcli_compiler_token_ids"] = _token_id_record(
            tokenizer, text, add_special_tokens=False
        )
        annotated_spans.append(item)
    folded_prompt = raw["folded_native_prompt_utf8"]
    assert isinstance(folded_prompt, str)  # validated in _raw_trace
    source_prompt = _render_source_one_user_template(folded_prompt)
    return {
        "schema": "hawking.ascension.qwen30_hcli_compiler_pre_execution_trace_annotated.v1",
        "status": TRACE_MODE,
        "recorded_at": _utc_now(),
        "raw_compiler_trace": {
            # The annotated trace moves with its immutable probe directory.
            # Keep this reference relative to that directory so an atomic
            # stage -> content-addressed move cannot leave a stale path inside
            # an otherwise valid evidence document.
            "path": raw_trace_path.name,
            "path_kind": "RELATIVE_TO_ANNOTATED_TRACE_PROBE_DIRECTORY",
            "sha256": _sha256_file(raw_trace_path),
            "schema": raw.get("schema"),
            "status": raw.get("status"),
            "capture_timing": raw.get("capture_timing"),
            "model_execution_started": raw.get("model_execution_started"),
        },
        "compiler_trace": raw,
        "source_tokenizer_annotations": {
            "hcli_compiler_encoding": {
                "description": "exact tokenizer.json encoding of each HCLI compiler span and folded prompt",
                "add_special_tokens": False,
            },
            "selected_context_spans": annotated_spans,
            "folded_native_prompt": {
                "utf8_sha256": _sha256_bytes(folded_prompt.encode("utf-8")),
                "utf8_bytes": len(folded_prompt.encode("utf-8")),
                **_token_id_record(tokenizer, folded_prompt, add_special_tokens=False),
            },
            "source_one_user_native_prompt": {
                "renderer": source_template_binding["renderer"],
                "utf8_sha256": _sha256_bytes(source_prompt.encode("utf-8")),
                "utf8_bytes": len(source_prompt.encode("utf-8")),
                **_token_id_record(tokenizer, source_prompt, add_special_tokens=True),
            },
        },
        "claim_boundary": {
            "new_diagnostic_not_historical": True,
            "actual_selected_span_text_and_token_ids_recorded_before_model_execution": True,
            "source_one_user_template_token_ids_are_a_diagnostic_render_not_a_model_forward": True,
            "does_not_load_or_execute_candidate_or_baseline_model": True,
            "does_not_claim_historical_trajectory_replay_route_membership_hidden_state_logits_topk_generation_hcli_tps_tg_capability_or_tournament": True,
        },
    }


def _invalidate_stale_current_pointer(*, root: Path, output_root: Path) -> None:
    """Fail closed if a prior current pointer has an unresolvable trace path.

    This only handles a short-lived staging-path bug in an earlier diagnostic
    implementation.  It preserves the old sealed receipt as history and
    replaces the mutable current pointer with a sealed invalidation until a
    fresh relocation-safe capture is earned.
    """

    pointer_path = root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_COMPILER_TRACE.json"
    if not pointer_path.exists():
        return
    pointer = _sealed(pointer_path, label="current HCLI compiler trace pointer")
    selected = pointer.get("compiler_trace_receipt")
    if not isinstance(selected, Mapping):
        return
    selected_path_value = selected.get("path")
    if not isinstance(selected_path_value, str):
        return
    receipt_path = Path(selected_path_value)
    if not receipt_path.is_file():
        return
    receipt = _sealed(receipt_path, label="current HCLI compiler trace receipt")
    binding = receipt.get("binding")
    traces = receipt.get("public_probe_compiler_traces")
    if not isinstance(binding, Mapping) or not isinstance(traces, list):
        return
    run_root_value = binding.get("run_root")
    if not isinstance(run_root_value, str):
        return
    run_root = Path(run_root_value)
    stale_paths: list[dict[str, str]] = []
    for row in traces:
        if not isinstance(row, Mapping):
            continue
        relative = row.get("annotated_trace_path")
        if not isinstance(relative, str):
            continue
        annotated_path = run_root / relative
        if not annotated_path.is_file():
            continue
        try:
            annotated = json.loads(annotated_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        raw_ref = annotated.get("raw_compiler_trace") if isinstance(annotated, Mapping) else None
        raw_path_value = raw_ref.get("path") if isinstance(raw_ref, Mapping) else None
        if isinstance(raw_path_value, str) and Path(raw_path_value).is_absolute() and not Path(raw_path_value).is_file():
            stale_paths.append(
                {
                    "probe_id": str(row.get("probe_id", "unknown")),
                    "annotated_trace_path": str(annotated_path.resolve()),
                    "stale_raw_trace_path": raw_path_value,
                }
            )
    if not stale_paths:
        return
    invalidation = seal(
        {
            "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace_supersession.v1",
            "status": "SUPERSEDED_PRE_EXECUTION_TRACE_PATH_BINDING_INVALID",
            "recorded_at": _utc_now(),
            "binding": {
                "candidate_root": str(root.resolve()),
                "historical_current_pointer_path": str(pointer_path.resolve()),
                "historical_current_pointer_seal_sha256": pointer.get("seal_sha256"),
                "historical_receipt_path": str(receipt_path.resolve()),
                "historical_receipt_seal_sha256": receipt.get("seal_sha256"),
                "historical_run_root": str(run_root.resolve()),
                "stale_annotated_raw_trace_references": stale_paths,
            },
            "reason": (
                "the annotated compiler trace recorded an absolute path inside a staging directory; "
                "the directory was atomically moved to its content-addressed run root, leaving that "
                "embedded raw-trace reference unresolved"
            ),
            "claim_boundary": {
                "historical_pre_execution_capture_is_preserved_but_not_current": True,
                "no_model_generation_or_runtime_evidence_is_revoked_or_reinterpreted": True,
                "fresh_relocation_safe_capture_required_before_route_or_hidden_state_diagnostic": True,
            },
        }
    )
    invalidation_path = output_root / "negative-science" / (
        "QWEN30_HQ30GR2_CURRENT_HCLI_COMPILER_TRACE_PATH_BINDING_INVALID_"
        f"{receipt.get('seal_sha256')}.json"
    )
    if invalidation_path.exists():
        existing = _sealed(invalidation_path, label="existing compiler trace path invalidation")
        if existing != invalidation:
            raise CurrentHcliTraceError("refusing to overwrite a different compiler trace path invalidation")
    else:
        _atomic_json(invalidation_path, invalidation)
    _atomic_json(
        pointer_path,
        seal(
            {
                "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace_current.v1",
                "status": "CURRENT_HCLI_COMPILER_TRACE_PATH_BINDING_INVALIDATED_AWAITING_FRESH_CAPTURE",
                "recorded_at": _utc_now(),
                "candidate_root": str(root.resolve()),
                "historical_compiler_trace_receipt": {
                    "path": str(receipt_path.resolve()),
                    "seal_sha256": receipt.get("seal_sha256"),
                },
                "supersession": {
                    "path": str(invalidation_path.resolve()),
                    "seal_sha256": invalidation.get("seal_sha256"),
                },
                "claim_boundary": invalidation["claim_boundary"],
            }
        ),
    )


def _event_log_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CurrentHcliTraceError(f"trace HCLI workspace has no event log: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentHcliTraceError(f"trace HCLI event log is unreadable: {exc}") from exc
    if not all(isinstance(row, Mapping) for row in rows):
        raise CurrentHcliTraceError("trace HCLI event log contains a non-object row")
    kinds = [str(row.get("kind", "")) for row in rows]
    forbidden = [
        kind
        for kind in kinds
        if any(marker in kind.lower() for marker in ("generation", "generate", "completion", "stream"))
        or kind == "context.compiled"
    ]
    if forbidden:
        raise CurrentHcliTraceError(
            f"trace-only HCLI event log contains post-compiler/model activity: {forbidden}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "event_count": len(rows),
        "event_kinds": kinds,
        "forbidden_post_compiler_or_generation_events": forbidden,
    }


def _run_trace_probe(
    *,
    stage_root: Path,
    probe_id: str,
    prompt: str,
    hcli_path: Path,
    tokenizer_path: Path,
    source_template_binding: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    probe_root = stage_root / "probes" / probe_id
    workspace = probe_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    raw_trace_path = probe_root / "compiler-pre-execution.raw.json"
    environment = os.environ.copy()
    environment.update(
        {
            "HIDE_TOKENIZER": str(tokenizer_path.resolve()),
            "HIDE_MAX_OUTPUT_TOKENS": "8",
            "HAWKING_HCLI_COMPILER_TRACE_MODE": TRACE_MODE,
            "HAWKING_HCLI_COMPILER_TRACE_PATH": str(raw_trace_path.resolve()),
        }
    )
    command = [
        str(hcli_path.resolve()),
        "run",
        "--prompt",
        prompt,
        "--session",
        f"qwen30-quality-current-trace-{probe_id}",
        "--model-url",
        TRACE_ENDPOINT,
        "--max-output-tokens",
        "8",
        "--workspace",
        str(workspace.resolve()),
        "--json",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CurrentHcliTraceError(f"{probe_id} HCLI compiler trace could not run: {exc}") from exc
    _atomic_text(probe_root / "hcli.stdout.txt", completed.stdout)
    _atomic_text(probe_root / "hcli.stderr.txt", completed.stderr)
    _atomic_json(
        probe_root / "hcli.command.json",
        {
            "schema": "hawking.ascension.qwen30_hcli_compiler_trace_command.v1",
            "status": TRACE_MODE,
            "command": command,
            "cwd": str(REPO_ROOT.resolve()),
            "endpoint": TRACE_ENDPOINT,
            "expected_terminal_error": "trace-only mode intentionally refuses generation",
            "returncode": completed.returncode,
            "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
            "claim_boundary": {
                "endpoint_is_intentionally_unbound_loopback_not_the_qwen30_server": True,
                "compiler_trace_guard_refuses_before_provider_or_model_execution": True,
            },
        },
    )
    if completed.returncode == 0:
        raise CurrentHcliTraceError(f"{probe_id} trace HCLI unexpectedly succeeded")
    if "trace-only mode intentionally refuses generation" not in completed.stderr:
        raise CurrentHcliTraceError(
            f"{probe_id} HCLI failed without the explicit pre-execution trace refusal"
        )
    raw = _raw_trace(raw_trace_path)
    annotated = annotate_compiler_trace(
        raw_trace=raw,
        tokenizer=tokenizer,
        source_template_binding=source_template_binding,
        raw_trace_path=raw_trace_path,
    )
    annotated_path = probe_root / "compiler-pre-execution.annotated.json"
    _atomic_json(annotated_path, annotated)
    event_log = workspace / ".hide/log/events.jsonl"
    event_summary = _event_log_summary(event_log)
    return {
        "probe_id": probe_id,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "command_path": str((probe_root / "hcli.command.json").resolve()),
        "command_sha256": _sha256_file(probe_root / "hcli.command.json"),
        "raw_trace_path": str(raw_trace_path.resolve()),
        "raw_trace_sha256": _sha256_file(raw_trace_path),
        "annotated_trace_path": str(annotated_path.resolve()),
        "annotated_trace_sha256": _sha256_file(annotated_path),
        "selected_context_span_count": len(raw["selected_context_spans"]),
        "event_log": event_summary,
        "hcli_returncode": completed.returncode,
        "model_execution_started": raw.get("model_execution_started"),
        "status": raw.get("status"),
    }


def _relative_probe_paths(stage_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return receipt-safe path references after a staging directory moves."""

    rewritten: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in (
            "command_path",
            "raw_trace_path",
            "annotated_trace_path",
        ):
            item[key] = str(Path(item[key]).resolve().relative_to(stage_root.resolve()))
        event_log = item.get("event_log")
        if not isinstance(event_log, Mapping):
            raise CurrentHcliTraceError("trace probe lacks event-log summary")
        event_item = dict(event_log)
        event_item["path"] = str(Path(_nonempty_text(event_item.get("path"), label="event log path")).resolve().relative_to(stage_root.resolve()))
        item["event_log"] = event_item
        rewritten.append(item)
    return rewritten


def run_once(
    *,
    root: Path,
    candidate_path: Path,
    selection_path: Path,
    source_snapshot_path: Path,
    admission_current_path: Path,
    hcli_path: Path,
    tokenizer_path: Path,
    template_path: Path,
    tokenizer_config_path: Path,
) -> dict[str, Any]:
    """Run all public probes through a source-bound, pre-execution compiler trace."""

    binding = _candidate_binding(
        candidate_path=candidate_path,
        selection_path=selection_path,
        source_snapshot_path=source_snapshot_path,
        admission_current_path=admission_current_path,
    )
    if not hcli_path.is_file() or not os.access(hcli_path, os.X_OK):
        raise CurrentHcliTraceError(f"trace HCLI binary is not executable: {hcli_path}")
    if not tokenizer_path.is_file():
        raise CurrentHcliTraceError(f"source tokenizer is absent: {tokenizer_path}")
    template_binding = _validate_source_template(template_path, tokenizer_config_path)
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise CurrentHcliTraceError(f"cannot load exact source tokenizer: {exc}") from exc
    output_root = root / "current-hcli-compiler-trace"
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    _invalidate_stale_current_pointer(root=root, output_root=output_root)
    stage_root = Path(
        tempfile.mkdtemp(prefix=".stage-current-hcli-compiler-trace-", dir=runs_root)
    )
    probe_rows: list[dict[str, Any]] = []
    try:
        for probe in PROMPT_PROBES:
            probe_rows.append(
                _run_trace_probe(
                    stage_root=stage_root,
                    probe_id=probe.identifier,
                    prompt=probe.prompt,
                    hcli_path=hcli_path,
                    tokenizer_path=tokenizer_path,
                    source_template_binding=template_binding,
                    tokenizer=tokenizer,
                )
            )
        candidate_seal = binding["candidate_manifest"]["seal_sha256"]
        bundle_identity = {
            "candidate_manifest_seal_sha256": candidate_seal,
            "selection_receipt_seal_sha256": binding["selection_receipt"]["seal_sha256"],
            "hcli_binary_sha256": _sha256_file(hcli_path),
            "source_tokenizer_sha256": _sha256_file(tokenizer_path),
            "source_template_sha256": _sha256_file(template_path),
            "probes": [
                {
                    "probe_id": row["probe_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "raw_trace_sha256": row["raw_trace_sha256"],
                    "annotated_trace_sha256": row["annotated_trace_sha256"],
                }
                for row in probe_rows
            ],
        }
        bundle_sha = _canonical_sha256(bundle_identity)
        final_run_root = runs_root / bundle_sha
        if final_run_root.exists():
            raise CurrentHcliTraceError(
                f"refusing to replace an existing content-addressed compiler trace run: {final_run_root}"
            )
        # Derive receipt-local references while the run still has its staging
        # pathname.  Once it moves, the absolute evidence paths captured by
        # subprocess are deliberately stale; resolving them after the move
        # would turn a successful trace into a non-bindable receipt.
        relative_rows = _relative_probe_paths(stage_root, probe_rows)
        os.replace(stage_root, final_run_root)
        stage_root = final_run_root
        receipt = seal(
            {
                "schema": SCHEMA,
                "status": STATUS,
                "recorded_at": _utc_now(),
                "binding": {
                    **binding,
                    "hcli_binary": _file_binding(hcli_path),
                    "source_tokenizer": _file_binding(tokenizer_path),
                    "source_template": template_binding,
                    "trace_guard": {
                        "mode": TRACE_MODE,
                        "raw_trace_schema": TRACE_SCHEMA,
                        "capture_timing": "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION",
                        "endpoint": TRACE_ENDPOINT,
                        "endpoint_is_intentionally_unbound_loopback_not_the_qwen30_server": True,
                    },
                    "run_root": str(final_run_root.resolve()),
                    "run_content_sha256": bundle_sha,
                },
                "public_probe_compiler_traces": relative_rows,
                "assessment": {
                    "actual_current_hcli_compiler_selected_spans": "PERSISTED_WITH_EXACT_SOURCE_TOKENIZER_IDS",
                    "actual_current_hcli_folded_prompt_token_ids": "PERSISTED",
                    "source_one_user_template_prompt_token_ids": "PERSISTED_DIAGNOSTIC_RENDER",
                    "historical_hcli_trajectory_recovery": "STILL_BLOCKED_AND_NOT_REWRITTEN",
                    "candidate_l0_expert0_actual_current_hcli_route_membership": "NOT_YET_EXECUTED_REQUIRES_SEPARATE_DIAGNOSTIC_MODEL_RUN",
                    "hidden_state_logit_topk_divergence": "NOT_YET_EXECUTED_REQUIRES_SEPARATE_DIAGNOSTIC_MODEL_RUN",
                },
                "claim_boundary": {
                    "new_diagnostic_not_historical": True,
                    "each_hcli_trace_refused_before_provider_or_model_execution": True,
                    "no_qwen30_runtime_server_watcher_or_gpu_was_started_or_used": True,
                    "no_candidate_or_baseline_weights_were_loaded": True,
                    "does_not_bypass_coherence_cooldown_or_claim_generation_hcli_tps_tg_capability_or_tournament": True,
                },
            }
        )
        receipts_root = output_root / "receipts"
        receipt_path = receipts_root / f"QWEN30_HQ30GR2_CURRENT_HCLI_COMPILER_TRACE_{candidate_seal}_{bundle_sha}.json"
        if receipt_path.exists():
            existing = _sealed(receipt_path, label="existing current HCLI compiler trace receipt")
            if existing != receipt:
                raise CurrentHcliTraceError("refusing to overwrite a distinct current compiler trace receipt")
            result = existing
            reused = True
        else:
            _atomic_json(receipt_path, receipt)
            result = receipt
            reused = False
        pointer = seal(
            {
                "schema": "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace_current.v1",
                "status": "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_HCLI_COMPILER_TRACE_SELECTED",
                "recorded_at": _utc_now(),
                "candidate_root": str(root.resolve()),
                "compiler_trace_receipt": {
                    "path": str(receipt_path.resolve()),
                    "seal_sha256": result.get("seal_sha256"),
                },
                "claim_boundary": result["claim_boundary"],
            }
        )
        _atomic_json(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_COMPILER_TRACE.json", pointer)
        return {
            "status": result.get("status"),
            "receipt_path": str(receipt_path),
            "receipt_seal_sha256": result.get("seal_sha256"),
            "current_path": str(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_COMPILER_TRACE.json"),
            "current_seal_sha256": pointer.get("seal_sha256"),
            "run_root": str(final_run_root),
            "run_content_sha256": bundle_sha,
            "reused": reused,
            "probe_count": len(relative_rows),
        }
    except BaseException:
        # A failed staging capture is intentionally left immutable under its
        # unique name for forensic inspection; it cannot overwrite a prior
        # complete run or current pointer.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--admission-current", type=Path, default=DEFAULT_ADMISSION_CURRENT)
    parser.add_argument("--hcli", type=Path, default=DEFAULT_HCLI)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            candidate_path=args.candidate.expanduser().resolve(),
            selection_path=args.selection.expanduser().resolve(),
            source_snapshot_path=args.source_snapshot.expanduser().resolve(),
            admission_current_path=args.admission_current.expanduser().resolve(),
            hcli_path=args.hcli.expanduser().resolve(),
            tokenizer_path=args.tokenizer.expanduser().resolve(),
            template_path=args.template.expanduser().resolve(),
            tokenizer_config_path=args.tokenizer_config.expanduser().resolve(),
        )
    except CurrentHcliTraceError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_CURRENT_HCLI_COMPILER_TRACE_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
