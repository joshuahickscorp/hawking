"""Fail-closed provenance ladder for the historical Qwen30 HCLI failures.

This is deliberately a CPU-only diagnostic.  It verifies the three recorded
HCLI prompts, their failed completions, and the source-bound admitted complete
Gravity artifact.  It then requires the *entire* compiled input prefix before
attempting any source-versus-packed embedding, operator, logit, or top-k
comparison.

The old HCLI record retained only summary information about selected memory
spans.  A visible user prompt is not a substitute for that compiled prefix.
Consequently a missing exact prefix is a sealed blocked result, not permission
to run a user-only forward and call it an explanation of the historical
trajectory.  This operator never starts a runtime, talks to an endpoint, uses
Metal, or loads model payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.ascension_base_true_tps_gate import PROMPT_PROBES
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
)
DEFAULT_NEGATIVE = (
    DEFAULT_PHYSICAL_ROOT
    / "tps-gate/negative-science"
    / "QWEN30_HCLI_COHERENCE_6959979797825d3cedf17b073e7c7a6071b23292c6b490c439daa41a0afda79e.json"
)
DEFAULT_ADMISSION = (
    DEFAULT_PHYSICAL_ROOT
    / "complete-gravity/QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
)
DEFAULT_EVENTS = DEFAULT_PHYSICAL_ROOT / "tps-gate/hcli-workspace/.hide/log/events.jsonl"
DEFAULT_OUTPUT_ROOT = DEFAULT_PHYSICAL_ROOT / "quality-diagnostics/hcli-trajectory-divergence"

SCHEMA = "hawking.ascension.qwen30_hcli_trajectory_divergence_ladder.v1"
STATUS = "BLOCKED_HISTORICAL_HCLI_TRAJECTORY_PREFIX_NOT_RECONSTRUCTABLE"


class DivergenceLadderError(RuntimeError):
    """The input evidence cannot safely bind this diagnostic."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(value, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise DivergenceLadderError(f"{label} is absent or has an invalid seal: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise DivergenceLadderError(f"{label} is not an object")
    return dict(checked)


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DivergenceLadderError(f"{label} must be a non-empty string")
    return value


def _exact_option(command: object, option: str, *, label: str) -> str:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise DivergenceLadderError(f"{label} command must be a string array")
    offsets = [index for index, item in enumerate(command) if item == option]
    if len(offsets) != 1 or offsets[0] + 1 >= len(command):
        raise DivergenceLadderError(f"{label} command must contain exactly one {option} value")
    return command[offsets[0] + 1]


def _probe_catalog() -> dict[str, tuple[str, str, str]]:
    catalog: dict[str, tuple[str, str, str]] = {}
    for probe in PROMPT_PROBES:
        if probe.identifier in catalog:
            raise DivergenceLadderError(f"duplicate source prompt probe {probe.identifier}")
        catalog[probe.identifier] = (probe.prompt, probe.prompt_sha256, probe.acceptance)
    return catalog


def _raw_hcli_evidence(
    *,
    observation: Mapping[str, Any],
    catalog: Mapping[str, tuple[str, str, str]],
) -> dict[str, Any]:
    probe_id = _nonempty_text(observation.get("probe_id"), label="negative probe id")
    expected = catalog.get(probe_id)
    if expected is None:
        raise DivergenceLadderError(f"negative evidence references unknown source probe {probe_id}")
    expected_prompt, expected_prompt_sha, expected_acceptance = expected
    raw_path = Path(_nonempty_text(observation.get("hcli_command_evidence_path"), label=f"{probe_id} raw HCLI path"))
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DivergenceLadderError(f"{probe_id} raw HCLI evidence is unreadable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DivergenceLadderError(f"{probe_id} raw HCLI evidence is not an object")
    prompt = _exact_option(raw.get("command"), "--prompt", label=probe_id)
    session_alias = _exact_option(raw.get("command"), "--session", label=probe_id)
    workspace = Path(_exact_option(raw.get("command"), "--workspace", label=probe_id))
    if prompt != expected_prompt:
        raise DivergenceLadderError(f"{probe_id} raw prompt differs from the protected prompt probe")
    observed_prompt_sha = _sha256_bytes(prompt.encode("utf-8"))
    if observed_prompt_sha != expected_prompt_sha or observed_prompt_sha != observation.get("prompt_sha256"):
        raise DivergenceLadderError(f"{probe_id} prompt SHA-256 does not bind raw/source/negative evidence")
    stdout = raw.get("stdout")
    if not isinstance(stdout, str):
        raise DivergenceLadderError(f"{probe_id} raw HCLI stdout is absent")
    try:
        hcli = json.loads(stdout)
        turn = hcli["result"]["turn"]
        completion = turn["completion"]
        generation_stats = turn["generation_stats"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DivergenceLadderError(f"{probe_id} raw HCLI stdout lacks its turn result: {exc}") from exc
    if not isinstance(completion, str) or not isinstance(generation_stats, Mapping):
        raise DivergenceLadderError(f"{probe_id} raw HCLI turn is malformed")
    completion_sha = _sha256_bytes(completion.encode("utf-8"))
    if completion_sha != observation.get("completion_sha256"):
        raise DivergenceLadderError(f"{probe_id} completion SHA-256 differs from negative evidence")
    observed_session = turn.get("session_id")
    if not isinstance(observed_session, str) or not observed_session:
        raise DivergenceLadderError(f"{probe_id} raw HCLI turn lacks a concrete session ID")
    alias_path = workspace / ".hide/kv/sessions" / f"{session_alias}.json"
    try:
        alias_record = json.loads(alias_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DivergenceLadderError(f"{probe_id} named HCLI session mapping is unreadable: {exc}") from exc
    if not isinstance(alias_record, Mapping) or alias_record.get("session_id") != observed_session:
        raise DivergenceLadderError(f"{probe_id} named HCLI session does not bind the recorded turn session")
    return {
        "probe_id": probe_id,
        "acceptance": expected_acceptance,
        "raw_hcli_evidence_path": str(raw_path.resolve()),
        "raw_hcli_evidence_sha256": _sha256_file(raw_path),
        "prompt_sha256": observed_prompt_sha,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "named_session": session_alias,
        "named_session_mapping_path": str(alias_path.resolve()),
        "named_session_mapping_sha256": _sha256_file(alias_path),
        "session_id": observed_session,
        "completion_sha256": completion_sha,
        "completion_utf8_bytes": len(completion.encode("utf-8")),
        "historical_input_tokens": generation_stats.get("input_tokens"),
        "historical_output_tokens": generation_stats.get("output_tokens"),
        "prompt_reconstruction": "EXACT_RAW_COMMAND_SOURCE_PROBE_AND_NEGATIVE_RECEIPT_MATCH",
    }


def _context_event_by_session(events_path: Path, session_id: str) -> dict[str, Any]:
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DivergenceLadderError(f"event log is unreadable: {exc}") from exc
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DivergenceLadderError(f"event log JSON is malformed at line {line_number}: {exc}") from exc
        if not isinstance(event, Mapping):
            raise DivergenceLadderError(f"event log row {line_number} is not an object")
        if event.get("session_id") == session_id and event.get("kind") == "context.compiled":
            matches.append(dict(event))
    if len(matches) != 1:
        raise DivergenceLadderError(
            f"historical session {session_id} has {len(matches)} context.compiled events; expected exactly one"
        )
    event = matches[0]
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise DivergenceLadderError(f"historical context.compiled event for {session_id} has no payload")
    # An exact replay needs the selected prefix token IDs or the exact selected
    # text and a tokenizer authority.  Counts, hit totals, and labels cannot
    # identify a prefix and are intentionally insufficient.
    persisted_exact_prefix = (
        isinstance(payload.get("exact_prefix_token_ids"), list)
        or isinstance(payload.get("exact_prefix_text"), str)
        or isinstance(payload.get("selected_memory_spans"), list)
    )
    meter = payload.get("meter")
    if not isinstance(meter, Mapping):
        meter = {}
    return {
        "event_id": event.get("id"),
        "event_seq": event.get("seq"),
        "event_chain_hash": event.get("chain_hash"),
        "event_log_path": str(events_path.resolve()),
        "event_log_sha256": _sha256_file(events_path),
        "retained_span_count": payload.get("retained"),
        "used_tokens": payload.get("used_tokens"),
        "used_tokens_estimated": meter.get("used_estimated"),
        "selected_prefix_text_or_token_ids_persisted": persisted_exact_prefix,
        "replayability": (
            "EXACT_PREFIX_RECOVERABLE" if persisted_exact_prefix else "BLOCKED_PREFIX_TEXT_AND_TOKEN_IDS_NOT_PERSISTED"
        ),
    }


def _source_bound_artifact(admission_path: Path, negative: Mapping[str, Any]) -> dict[str, Any]:
    admission = _sealed(admission_path, label="complete Gravity admission receipt")
    if admission.get("status") != "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED":
        raise DivergenceLadderError("complete Gravity admission receipt has an unexpected status")
    negative_binding = negative.get("binding")
    if not isinstance(negative_binding, Mapping):
        raise DivergenceLadderError("negative HCLI receipt lacks binding")
    if admission.get("seal_sha256") != negative_binding.get("complete_artifact_admission_seal_sha256"):
        raise DivergenceLadderError("historical negative does not bind the supplied complete Gravity admission")
    manifest = admission.get("complete_manifest")
    source = admission.get("current_source_revalidation")
    identity = admission.get("immutable_source_identity")
    if not isinstance(manifest, Mapping) or not isinstance(source, Mapping) or not isinstance(identity, Mapping):
        raise DivergenceLadderError("complete Gravity admission lacks manifest/source identity binding")
    if manifest.get("seal_sha256") != negative_binding.get("complete_manifest_seal_sha256"):
        raise DivergenceLadderError("historical negative complete manifest seal does not match admission")
    if source.get("seal_sha256") != negative_binding.get("source_revalidation_seal_sha256"):
        raise DivergenceLadderError("historical negative source revalidation seal does not match admission")
    if identity.get("content_identity_sha256") != negative_binding.get("source_content_identity_sha256"):
        raise DivergenceLadderError("historical negative source content identity does not match admission")
    return {
        "admission_receipt_path": str(admission_path.resolve()),
        "admission_receipt_seal_sha256": admission.get("seal_sha256"),
        "complete_manifest_path": manifest.get("path"),
        "complete_manifest_seal_sha256": manifest.get("seal_sha256"),
        "source_revalidation_path": source.get("path"),
        "source_revalidation_seal_sha256": source.get("seal_sha256"),
        "source_content_identity_sha256": identity.get("content_identity_sha256"),
        "source_repository": identity.get("repository"),
        "source_revision": identity.get("revision"),
        "artifact_scope": "ADMITTED_DIRECT_PACKED_ARTIFACT_BOUND_ONLY_NO_PAYLOAD_OR_RUNTIME_EXECUTION",
    }


def _blocked_ladder() -> list[dict[str, str]]:
    blocked = "NOT_EXECUTED_FAIL_CLOSED_EXACT_HISTORICAL_COMPILED_PREFIX_UNAVAILABLE"
    return [
        {
            "stage": "template_tokenizer",
            "status": blocked,
            "reason": "Exact template input cannot be formed without the selected retained span text/token IDs.",
        },
        {
            "stage": "embedding",
            "status": blocked,
            "reason": "Source-versus-direct-packed embedding comparison would use a guessed prefix.",
        },
        {
            "stage": "attention_router",
            "status": blocked,
            "reason": "No exact historical prefix means no source-bound activation trajectory.",
        },
        {
            "stage": "moe",
            "status": blocked,
            "reason": "No exact historical prefix means no source-bound routed-expert trajectory.",
        },
        {
            "stage": "final_norm_lm_head",
            "status": blocked,
            "reason": "No exact historical prefix means no comparable final logits.",
        },
        {
            "stage": "logit_top_k",
            "status": blocked,
            "reason": "No exact historical prefix means no top-k divergence can be attributed to the failed trajectory.",
        },
    ]


def run_once(
    *,
    negative_path: Path,
    admission_path: Path,
    events_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    negative = _sealed(negative_path, label="historical HCLI coherence negative")
    if negative.get("status") != "BLOCKED_HCLI_PROMPT_DEPENDENT_COHERENCE_NOT_EARNED":
        raise DivergenceLadderError("historical receipt is not the expected blocked HCLI coherence result")
    source_artifact = _source_bound_artifact(admission_path, negative)
    observations = negative.get("details", {}).get("probe_observations") if isinstance(negative.get("details"), Mapping) else None
    if not isinstance(observations, list) or len(observations) != 3:
        raise DivergenceLadderError("historical HCLI negative must have exactly three probe observations")
    catalog = _probe_catalog()
    trajectories: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise DivergenceLadderError("historical HCLI probe observation is not an object")
        raw = _raw_hcli_evidence(observation=observation, catalog=catalog)
        context = _context_event_by_session(events_path, _nonempty_text(raw.get("session_id"), label="raw HCLI session"))
        if context["replayability"] != "BLOCKED_PREFIX_TEXT_AND_TOKEN_IDS_NOT_PERSISTED":
            raise DivergenceLadderError(
                "this bounded fail-closed operator intentionally refuses to infer a replay path; "
                "exact-prefix replay needs a separate source-tokenizer authority"
            )
        trajectories.append(
            {
                "probe": raw,
                "compiled_context": context,
                "first_unisolated_historical_stage": "template_tokenizer",
                "first_unisolated_reason": "compiled context retained span count/estimated count but no selected span text or token IDs",
                "ladder": _blocked_ladder(),
            }
        )
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "binding": {
            "historical_hcli_negative_path": str(negative_path.resolve()),
            "historical_hcli_negative_seal_sha256": negative.get("seal_sha256"),
            "source_bound_admitted_artifact": source_artifact,
            "source_prompt_catalog_path": str((REPO_ROOT / "lab/operators/ascension_base_true_tps_gate.py").resolve()),
            "source_prompt_catalog_sha256": _sha256_file(REPO_ROOT / "lab/operators/ascension_base_true_tps_gate.py"),
        },
        "trajectories": trajectories,
        "assessment": {
            "exact_visible_user_prompt_bodies_reconstructed": True,
            "exact_historical_compiled_prefix_reconstructed": False,
            "first_unisolated_stage_for_all_three": "template_tokenizer",
            "reason": "Each context.compiled record retained two memory spans and an estimated nine-token count, but persisted neither selected text nor token IDs.",
            "embedding_attention_router_moe_final_norm_lm_head_logit_and_top_k_comparisons": "NOT_RUN_FAIL_CLOSED",
        },
        "claim_boundary": {
            "cpu_only": True,
            "no_model_payload_loaded": True,
            "no_source_forward_or_packed_forward_executed": True,
            "no_runtime_server_watcher_endpoint_or_gpu_touched": True,
            "does_not_duplicate_route_membership_or_cross_depth_gate_up_checks": True,
            "does_not_claim_a_causal_quality_defect_or_any_hcli_tps_generation_capability_or_tournament_result": True,
            "visible_user_only_or_guessed_context_is_not_treated_as_historical_hcli_trajectory": True,
        },
        "reopen_condition": {
            "requires_for_each_probe": [
                "immutable exact compiled prefix text or token IDs",
                "binding from that prefix to the historical HCLI request/session",
                "source tokenizer/template authority for text-only prefix reconstruction",
            ],
            "then": "run a separate source-versus-admitted-direct-packed CPU ladder from tokenizer through logits/top-k",
        },
    }
    sealed = seal(body)
    receipt_path = output_root / f"QWEN30_HCLI_TRAJECTORY_DIVERGENCE_LADDER_{negative['seal_sha256']}.json"
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing HCLI trajectory divergence receipt")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != STATUS:
            raise DivergenceLadderError("refusing to overwrite a different historical divergence receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "reused": reused,
        "first_unisolated_stage": "template_tokenizer",
        "exact_historical_compiled_prefix_reconstructed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            negative_path=args.negative.expanduser().resolve(),
            admission_path=args.admission.expanduser().resolve(),
            events_path=args.events.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
        )
    except DivergenceLadderError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HCLI_TRAJECTORY_DIVERGENCE_LADDER_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
