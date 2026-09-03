"""Fail-closed recovery audit for actual Qwen30 HCLI layer-0 routes.

The HQ30GR2 candidate changes only layer-0/expert-0 gate/up organs.  Whether
that organ can affect the three failed HCLI probes depends on the *actual*
native prompt token IDs that reached the model, including HCLI's compiled
context.  This diagnostic refuses to substitute visible user text, a current
memory DB, or a text re-encoding for those historical token IDs.  It records
the exact missing authority so a later source-bound HCLI capture can close the
gap without rewriting history.
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

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CANDIDATE = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SELECTION_RECEIPT.json"
DEFAULT_HCLI_NEGATIVE = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/tps-gate/negative-science"
    / "QWEN30_HCLI_COHERENCE_6959979797825d3cedf17b073e7c7a6071b23292c6b490c439daa41a0afda79e.json"
)

SCHEMA = "hawking.ascension.qwen30_quality_repack_hcli_route_recovery.v1"
STATUS = "BLOCKED_ACTUAL_HCLI_LAYER0_ROUTE_MEMBERSHIP_TOKEN_TRAJECTORY_UNAVAILABLE"
EXPECTED_PROBES = ("literal_hawking", "json_status", "python_add")


class RouteRecoveryError(RuntimeError):
    """Historical HCLI evidence cannot safely support a route-membership claim."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise RouteRecoveryError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise RouteRecoveryError(f"{label} is not an object")
    return dict(checked)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RouteRecoveryError(f"{label} must be a non-empty string")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, Mapping):
                events.append(dict(row))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteRecoveryError(f"cannot parse HCLI event log {path}: {exc}") from exc
    if not events:
        raise RouteRecoveryError("HCLI event log is empty")
    return events


def _command_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteRecoveryError(f"cannot parse HCLI command evidence {path}: {exc}") from exc
    if not isinstance(evidence, Mapping):
        raise RouteRecoveryError(f"HCLI command evidence is not an object: {path}")
    command = evidence.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RouteRecoveryError(f"HCLI command is malformed: {path}")
    stdout = evidence.get("stdout")
    if not isinstance(stdout, str):
        raise RouteRecoveryError(f"HCLI command has no stdout: {path}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RouteRecoveryError(f"HCLI command stdout is not JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RouteRecoveryError(f"HCLI stdout payload is not an object: {path}")
    return dict(evidence), dict(payload), list(command)


def _option(command: Sequence[str], name: str) -> str:
    try:
        index = list(command).index(name)
        value = command[index + 1]
    except (ValueError, IndexError) as exc:
        raise RouteRecoveryError(f"HCLI command lacks {name}") from exc
    if not value:
        raise RouteRecoveryError(f"HCLI command has empty {name}")
    return value


def inspect_probe(
    observation: Mapping[str, Any], *, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify exactly what historical evidence has—and does not have."""

    probe_id = _text(observation.get("probe_id"), "HCLI probe id")
    evidence_path = Path(_text(observation.get("hcli_command_evidence_path"), f"{probe_id} evidence path"))
    raw, payload, command = _command_payload(evidence_path)
    prompt = _option(command, "--prompt")
    session_name = _option(command, "--session")
    raw_result = payload.get("result")
    if not isinstance(raw_result, Mapping):
        raise RouteRecoveryError(f"{probe_id} HCLI result is missing")
    turn = raw_result.get("turn")
    if not isinstance(turn, Mapping):
        raise RouteRecoveryError(f"{probe_id} HCLI turn is missing")
    stats = turn.get("generation_stats")
    if not isinstance(stats, Mapping):
        raise RouteRecoveryError(f"{probe_id} HCLI generation stats are missing")
    prompt_sha = _sha256(prompt)
    if prompt_sha != observation.get("prompt_sha256"):
        raise RouteRecoveryError(f"{probe_id} command prompt no longer matches sealed negative receipt")
    completion = _text(turn.get("completion"), f"{probe_id} completion")
    completion_sha = _sha256(completion)
    if completion_sha != observation.get("completion_sha256"):
        raise RouteRecoveryError(f"{probe_id} command completion no longer matches sealed negative receipt")
    input_tokens = stats.get("input_tokens")
    output_tokens = stats.get("output_tokens")
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise RouteRecoveryError(f"{probe_id} has no positive historical input token count")
    if not isinstance(output_tokens, int) or output_tokens <= 0:
        raise RouteRecoveryError(f"{probe_id} has no positive historical output token count")
    intent_event_id = _text(turn.get("intent_event_id"), f"{probe_id} intent event id")
    session_id = _text(turn.get("session_id"), f"{probe_id} session id")
    matching = [event for event in events if event.get("id") == intent_event_id]
    if len(matching) != 1:
        raise RouteRecoveryError(f"{probe_id} intent event cannot be uniquely recovered")
    intent = matching[0]
    if intent.get("session_id") != session_id:
        raise RouteRecoveryError(f"{probe_id} intent session differs from HCLI result")
    intent_text = (
        intent.get("payload", {}).get("args", {}).get("text")
        if isinstance(intent.get("payload"), Mapping)
        else None
    )
    if intent_text != prompt:
        raise RouteRecoveryError(f"{probe_id} intent text differs from HCLI command prompt")
    sequence = intent.get("seq")
    if not isinstance(sequence, int):
        raise RouteRecoveryError(f"{probe_id} intent sequence is missing")
    compiled = next(
        (
            event
            for event in events
            if event.get("session_id") == session_id
            and event.get("seq") == sequence + 1
            and event.get("kind") == "context.compiled"
        ),
        None,
    )
    if not isinstance(compiled, Mapping) or not isinstance(compiled.get("payload"), Mapping):
        raise RouteRecoveryError(f"{probe_id} compiled-context event is absent")
    context = dict(compiled["payload"])
    missing_authority = [
        key
        for key in (
            "native_prompt_token_ids",
            "native_prompt_token_ids_sha256",
            "folded_prompt_token_ids",
            "compiled_prompt_token_ids",
            "compiled_prompt_sha256",
            "retained_span_text_or_token_ids",
        )
        if key not in context
    ]
    return {
        "probe_id": probe_id,
        "hcli_command_evidence_path": str(evidence_path.resolve()),
        "hcli_command_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "raw_hcli_session_name": session_name,
        "hcli_session_id": session_id,
        "intent_event_id": intent_event_id,
        "intent_event_seq": sequence,
        "prompt_sha256": prompt_sha,
        "completion_sha256": completion_sha,
        "historical_input_tokens": input_tokens,
        "historical_output_tokens": output_tokens,
        "historical_completed_decode_forwards": stats.get("completed_decode_forwards"),
        "historical_context_compiled_event_id": compiled.get("id"),
        "historical_context_compiled_chain_hash": compiled.get("chain_hash"),
        "historical_context_retained_count": context.get("retained"),
        "historical_context_used_tokens_estimated": context.get("used_tokens"),
        "actual_native_prompt_token_ids_recorded": False,
        "actual_generated_token_ids_recorded": False,
        "missing_authority": missing_authority,
        "result": "NO_ROUTE_MEMBERSHIP_COMPUTED",
        "reason": (
            "the historical record has only aggregate input/output counts and redacted compiled-context "
            "metadata; its full native prompt token IDs and generated token IDs were never sealed"
        ),
    }


def run_once(*, root: Path, candidate_path: Path, selection_path: Path, hcli_negative_path: Path) -> dict[str, Any]:
    candidate = _sealed(candidate_path, label="isolated HQ30GR2 candidate manifest")
    selection = _sealed(selection_path, label="HQ30GR2 selection receipt")
    negative = _sealed(hcli_negative_path, label="Qwen30 HCLI coherence negative")
    if candidate.get("schema") != "hawking.ascension.qwen30_quality_repack_candidate.v1":
        raise RouteRecoveryError("candidate does not use the isolated HQ30GR2 manifest grammar")
    candidate_seal = _text(candidate.get("seal_sha256"), "candidate manifest seal")
    binding = selection.get("binding")
    if not isinstance(binding, Mapping) or binding.get("selected_organs") != [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]:
        raise RouteRecoveryError("selection receipt is not the exact L0/E0 HQ30GR2 candidate")
    if negative.get("status") != "BLOCKED_HCLI_PROMPT_DEPENDENT_COHERENCE_NOT_EARNED":
        raise RouteRecoveryError("target HCLI evidence is not the sealed coherence negative")
    details = negative.get("details")
    observations = details.get("probe_observations") if isinstance(details, Mapping) else None
    if not isinstance(observations, list) or tuple(row.get("probe_id") for row in observations if isinstance(row, Mapping)) != EXPECTED_PROBES:
        raise RouteRecoveryError("sealed HCLI negative does not contain the three expected probes")
    first_path = Path(_text(observations[0].get("hcli_command_evidence_path"), "first HCLI evidence path"))
    # The tps gate gives all three invocations one workspace; derive only from
    # their own durable evidence rather than querying a current mutable server.
    # ``probe-XX.json`` lives at ``tps-gate/evidence/<fingerprint>/``.  The
    # companion HCLI workspace is directly under that same ``tps-gate`` root.
    events_path = first_path.parents[2] / "hcli-workspace/.hide/log/events.jsonl"
    events = _read_events(events_path)
    probes = [inspect_probe(dict(row), events=events) for row in observations if isinstance(row, Mapping)]
    if any(row["actual_native_prompt_token_ids_recorded"] for row in probes):
        raise RouteRecoveryError("this refusal lane expects the historical HCLI record to be token-ID incomplete")
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "binding": {
            "candidate_manifest_path": str(candidate_path.resolve()),
            "candidate_manifest_seal_sha256": candidate_seal,
            "selection_receipt_path": str(selection_path.resolve()),
            "selection_receipt_seal_sha256": selection.get("seal_sha256"),
            "hcli_coherence_negative_path": str(hcli_negative_path.resolve()),
            "hcli_coherence_negative_seal_sha256": negative.get("seal_sha256"),
            "hcli_event_log_path": str(events_path.resolve()),
            "hcli_event_log_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
        },
        "probe_recovery": probes,
        "assessment": {
            "candidate_l0_expert0_actual_hcli_route_membership": "UNMEASURED",
            "candidate_has_causal_reach_to_all_three_failed_hcli_trajectories": "NOT_EARNED",
            "why_no_source_user_or_current_memory_substitution_was_used": (
                "the source-user template omits HCLI's historical compiled context; current user-memory "
                "state is mutable and cannot stand in for the sealed historical native prompt"
            ),
            "required_next_evidence": (
                "a new source-bound HCLI capture must seal exact native prompt token IDs, each generated "
                "feedback token ID, tokenizer/template hashes, and the context compilation binding before "
                "a CPU layer-0 router oracle may label memberships as actual"
            ),
        },
        "claim_boundary": {
            "cpu_only_evidence_recovery": True,
            "does_not_query_or_execute_live_qwen30_server_runtime_or_gpu": True,
            "does_not_substitute_visible_user_text_or_current_memory_for_actual_hcli_tokens": True,
            "does_not_claim_route_membership_generation_hcli_tps_tg_capability_or_tournament": True,
            "candidate_runtime_and_baseline_control_untouched": True,
        },
    }
    sealed = seal(body)
    receipt_root = root / "route-membership" / "negative-science"
    receipt_path = receipt_root / (
        f"QWEN30_HQ30GR2_HCLI_L0_ROUTE_RECOVERY_{candidate_seal}_{negative['seal_sha256']}.json"
    )
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing HCLI route recovery receipt")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != STATUS:
            raise RouteRecoveryError("refusing to overwrite a historical HCLI route recovery receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    pointer = seal(
        {
            "schema": "hawking.ascension.qwen30_quality_repack_hcli_route_recovery_current.v1",
            "status": "CURRENT_QWEN30_HQ30GR2_HCLI_L0_ROUTE_RECOVERY_REFUSAL_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(root.resolve()),
            "route_recovery_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": result.get("seal_sha256"),
            },
            "claim_boundary": body["claim_boundary"],
        }
    )
    _atomic_json(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_HCLI_L0_ROUTE_RECOVERY_CURRENT.json", pointer)
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "current_path": str(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_HCLI_L0_ROUTE_RECOVERY_CURRENT.json"),
        "current_seal_sha256": pointer.get("seal_sha256"),
        "reused": reused,
        "probe_recovery": result.get("probe_recovery"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--hcli-negative", type=Path, default=DEFAULT_HCLI_NEGATIVE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            candidate_path=args.candidate.expanduser().resolve(),
            selection_path=args.selection.expanduser().resolve(),
            hcli_negative_path=args.hcli_negative.expanduser().resolve(),
        )
    except RouteRecoveryError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HQ30GR2_HCLI_L0_ROUTE_RECOVERY_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
