#!/usr/bin/env python3
"""Execute the first teacher-bound compact Flash route-union control.

The control is intentionally narrow.  It does not discover routes, extrapolate
cache behavior, or report TPS.  It replays a source-bound accepted sequence
using only unions observed in a complete dense teacher session, and treats a
route or terminal mismatch as the first physical failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Support package imports and direct repository-script invocation.
    from tools.odyssey.flash_repeated_accepted_decode import (
        PINNED_REVISION,
        PROMPT_IDS,
        REPO_ID,
        _json_object,
        load_external_reference_contract,
    )
except ModuleNotFoundError:  # pragma: no cover - direct CLI invocation.
    from flash_repeated_accepted_decode import (  # type: ignore[no-redef]
        PINNED_REVISION,
        PROMPT_IDS,
        REPO_ID,
        _json_object,
        load_external_reference_contract,
    )


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
SESSION_SCHEMA = "hawking.flash.stateful_complete_token_session.v1"
SESSION_STATUS = "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
TOKEN_MAJOR_STATUS = "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE"


@dataclass(frozen=True)
class AcceptedChain:
    """The one explicitly selected source-token chain a route control may replay."""

    token_ids: tuple[int, ...]
    prompt_length: int

    @property
    def continuations(self) -> tuple[int, ...]:
        return self.token_ids[self.prompt_length:]


def _load(path: Path) -> dict[str, Any]:
    # The route wrapper validates a pre-existing teacher and a native-written
    # candidate before it writes provenance. Reuse the reference gate's
    # duplicate-key rejection so a JSON parser cannot silently collapse an
    # ambiguous raw receipt before its seal and transcript are checked.
    return _json_object(path.read_bytes(), str(path))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_utf8_sorted_seal(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _verify_receipt_seal(doc: dict[str, Any], *, role: str) -> str:
    seal = doc.get("seal_sha256")
    body = dict(doc)
    body.pop("seal_sha256", None)
    if not isinstance(seal, str) or seal != _compact_utf8_sorted_seal(body):
        raise ValueError(f"{role} has an invalid compact UTF-8 receipt seal")
    return seal


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _accepted_chain(
    doc: dict[str, Any],
    *,
    role: str,
    source_input_identity: dict[str, Any] | None = None,
) -> AcceptedChain:
    """Validate a complete accepted chain without assuming its token values.

    The old adapter carried a historical continuation as a constant.  This
    derives both the prompt boundary and reference continuation solely from a
    chosen, sealed receipt, so an operator cannot silently reuse the old chain
    when a source oracle establishes a different one.
    """

    if doc.get("schema") != SESSION_SCHEMA or doc.get("status") != SESSION_STATUS:
        raise ValueError(f"{role} is not a passed repeated accepted Flash session")
    if doc.get("model") != REPO_ID or doc.get("pinned_revision") != PINNED_REVISION:
        raise ValueError(f"{role} source identity mismatch")
    _verify_receipt_seal(doc, role=role)
    if source_input_identity is not None:
        expected_root = source_input_identity.get("model_root")
        if (
            not isinstance(expected_root, str)
            or doc.get("root") != expected_root
            or doc.get("source_input_identity") != source_input_identity
        ):
            raise ValueError(
                f"{role} source input identity differs from the owner-authorized V2 ModelLake source"
            )
    token_ids = doc.get("token_ids")
    if not isinstance(token_ids, list) or not all(_is_token_id(token) for token in token_ids):
        raise ValueError(f"{role} has an invalid token sequence")
    execution = doc.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("process_boundary") != "one native process"
        or execution.get("source_reset_or_reprefill") is not False
    ):
        raise ValueError(f"{role} lacks one-process persistent-session proof")
    terminal = doc.get("terminal")
    checks = terminal.get("reference_checks") if isinstance(terminal, dict) else None
    if not isinstance(checks, list) or len(checks) < 2:
        raise ValueError(f"{role} needs at least two terminal reference checks")
    first = checks[0] if checks else None
    first_input = first.get("input_state_index") if isinstance(first, dict) else None
    if not isinstance(first_input, int) or first_input < 0:
        raise ValueError(f"{role} omits its first terminal input-state boundary")
    prompt_length = first_input + 1
    if prompt_length <= 0 or prompt_length >= len(token_ids):
        raise ValueError(f"{role} has an invalid prompt/reference boundary")
    chain = AcceptedChain(tuple(token_ids), prompt_length)
    if tuple(token_ids[:prompt_length]) != tuple(PROMPT_IDS):
        raise ValueError(f"{role} prompt token sequence does not match the canonical Flash gate")
    if len(chain.continuations) != len(checks):
        raise ValueError(f"{role} terminal checks do not cover exactly its continuation chain")
    if doc.get("accepted_generation_tokens") != len(chain.continuations):
        raise ValueError(f"{role} did not accept every declared continuation token")
    for generation_index, (expected, check) in enumerate(zip(chain.continuations, checks, strict=True)):
        if (
            not isinstance(check, dict)
            or check.get("generation_index") != generation_index
            or check.get("input_state_index") != prompt_length - 1 + generation_index
            or check.get("expected_token_id") != expected
            or check.get("predicted_token_id") != expected
            or check.get("accepted") is not True
        ):
            raise ValueError(f"{role} terminal reference chain is not exact")
    return chain


def _segment_layer(segment: dict[str, Any], *, role: str) -> int:
    """Read one unambiguous single-layer token-major segment."""

    direct = segment.get("layer")
    ranged = segment.get("layers")
    range_layer: object | None = None
    if ranged is not None:
        if not isinstance(ranged, list) or len(ranged) != 2 or ranged[0] != ranged[1]:
            raise ValueError(f"{role} has a non-single-layer token-major segment")
        range_layer = ranged[0]
    if direct is not None and range_layer is not None and direct != range_layer:
        raise ValueError(f"{role} has contradictory token-major layer identifiers")
    layer = direct if direct is not None else range_layer
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 or layer >= 48:
        raise ValueError(f"{role} has incomplete or invalid token-major layer coverage")
    return layer


def _token_major_rows(
    doc: dict[str, Any],
    chain: AcceptedChain,
    *,
    role: str,
    expert_bank_mode: str,
) -> dict[tuple[int, int], tuple[tuple[int, ...], str]]:
    """Return the complete exact per-layer/per-token route-state transcript."""

    segments = doc.get("segments")
    if not isinstance(segments, list) or len(segments) != 48:
        raise ValueError(f"{role} omits one record for each token-major layer")
    seen_layers: set[int] = set()
    transcript: dict[tuple[int, int], tuple[tuple[int, ...], str]] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError(f"{role} has a non-object layer record")
        layer = _segment_layer(segment, role=role)
        if layer in seen_layers:
            raise ValueError(f"{role} has incomplete or duplicate token-major layer coverage")
        seen_layers.add(layer)
        if segment.get("expert_bank_mode") != expert_bank_mode:
            raise ValueError(f"{role} has an unexpected compact/dense layer segment")
        rows = segment.get("steps")
        if not isinstance(rows, list) or len(rows) != len(chain.token_ids):
            raise ValueError(f"{role} token-major layer omits the selected chain")
        for step, (row, token) in enumerate(zip(rows, chain.token_ids, strict=True)):
            route_ids = row.get("route_ids") if isinstance(row, dict) else None
            state_hash = row.get("final_state_sha256") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or row.get("step") != step
                or row.get("token_id") != token
                or not isinstance(route_ids, list)
                or len(route_ids) != 10
                or not all(_is_token_id(route_id) and route_id < 512 for route_id in route_ids)
                or len(set(route_ids)) != 10
                or not isinstance(state_hash, str)
                or len(state_hash) != 64
                or any(char not in "0123456789abcdef" for char in state_hash.lower())
            ):
                raise ValueError(f"{role} lacks exact route/state-hash evidence for the selected chain")
            transcript[(layer, step)] = (tuple(route_ids), state_hash)
    if len(seen_layers) != 48:
        raise ValueError(f"{role} does not cover all 48 layers")
    return transcript


def _verify_token_major_chain(
    doc: dict[str, Any], chain: AcceptedChain, *, role: str
) -> dict[tuple[int, int], tuple[tuple[int, ...], str]]:
    """Reject layer-major/segmented controls as route teachers for a full body."""

    execution = doc.get("execution")
    if not isinstance(execution, dict) or execution.get("expert_bank_mode") != "dense":
        raise ValueError(f"{role} is not an exact dense teacher")
    token_major = execution.get("token_major_resident_banks") if isinstance(execution, dict) else None
    if not isinstance(token_major, dict) or token_major.get("status") != TOKEN_MAJOR_STATUS:
        raise ValueError(f"{role} is segmented or lacks all-48 token-major residency")
    if (
        token_major.get("resident_layers") != 48
        or token_major.get("expected_layers") != 48
        or token_major.get("cross_layer_activation_handoff") != "device buffers"
        or token_major.get("source_reset_or_reprefill") is not False
    ):
        raise ValueError(f"{role} lacks complete token-major state ownership")
    return _token_major_rows(doc, chain, role=role, expert_bank_mode="dense")


def _verify_compact_token_major_chain(
    doc: dict[str, Any], chain: AcceptedChain
) -> dict[tuple[int, int], tuple[tuple[int, ...], str]]:
    """Require the compact candidate to expose the same full transcript shape."""

    execution = doc.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("compact candidate omitted execution evidence")
    token_major = execution.get("token_major_resident_banks")
    if not isinstance(token_major, dict) or token_major.get("status") != TOKEN_MAJOR_STATUS:
        raise ValueError("compact candidate is segmented or lacks all-48 token-major residency")
    if (
        token_major.get("resident_layers") != 48
        or token_major.get("expected_layers") != 48
        or token_major.get("cross_layer_activation_handoff") != "device buffers"
        or token_major.get("source_reset_or_reprefill") is not False
    ):
        raise ValueError("compact candidate lacks complete token-major state ownership")
    return _token_major_rows(
        doc,
        chain,
        role="compact candidate",
        expert_bank_mode="route_union_compact_teacher_bound",
    )


def _verify_teacher(
    doc: dict[str, Any], *, source_input_identity: dict[str, Any] | None = None
) -> AcceptedChain:
    chain = _accepted_chain(
        doc, role="route teacher", source_input_identity=source_input_identity
    )
    _verify_token_major_chain(doc, chain, role="route teacher")
    return chain


def _verify_reference_matches_teacher(reference: dict[str, Any], chain: AcceptedChain) -> None:
    expected = tuple(reference["prompt_token_ids"] + reference["generated_token_ids"])
    if chain.token_ids != expected:
        raise ValueError(
            "route teacher chain differs from the independently sealed PLE-inclusive source reference"
        )


def _verify_candidate(
    doc: dict[str, Any],
    teacher: dict[str, Any],
    *,
    device_only_resident: bool = False,
    token_major_resident: bool = False,
    source_input_identity: dict[str, Any] | None = None,
) -> None:
    chain = _verify_teacher(teacher, source_input_identity=source_input_identity)
    teacher_rows = _verify_token_major_chain(teacher, chain, role="route teacher")
    candidate_chain = _accepted_chain(
        doc, role="compact candidate", source_input_identity=source_input_identity
    )
    if candidate_chain != chain:
        raise ValueError("compact candidate token sequence drifted from its selected route teacher")
    if doc.get("execution", {}).get("expert_bank_mode") != "route_union_compact_teacher_bound":
        raise ValueError("candidate did not execute the teacher-bound compact expert path")
    candidate_rows = _verify_compact_token_major_chain(doc, chain)
    if candidate_rows != teacher_rows:
        raise ValueError("compact candidate route/state transcript drifted from its dense teacher")
    teacher_bytes = int(teacher.get("execution", {}).get("source_payload_bytes_read", 0))
    candidate_bytes = int(doc.get("execution", {}).get("source_payload_bytes_read", 0))
    if teacher_bytes <= 0 or candidate_bytes <= 0 or candidate_bytes >= teacher_bytes:
        raise ValueError("compact candidate did not reduce source payload reads")
    if device_only_resident:
        resident = doc.get("execution", {}).get("linear_compact_bank_residency", {})
        expected_linear_status = (
            "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE"
            if token_major_resident
            else "PARTIAL_PROCESS_LIFETIME_DEVICE_ONLY_LINEAR_BANKS_RETAINED"
        )
        if resident.get("status") != expected_linear_status:
            raise ValueError("device-only resident control did not retain compact linear banks")
        if resident.get("immutable_weight_ownership") != "device_only_after_source_upload":
            raise ValueError("device-only resident control retained an ambiguous weight owner")
        if int(resident.get("retained_linear_layers", 0)) <= 0:
            raise ValueError("device-only resident control retained no linear layers")
        if int(resident.get("retained_linear_source_weight_bytes", -1)) != 0:
            raise ValueError("device-only resident control retained source-side linear weights")
        if int(resident.get("retained_linear_device_weight_bytes", 0)) <= 0:
            raise ValueError("device-only resident control omitted device weight accounting")
        if int(resident.get("observed_process_rss_bytes", 0)) <= 0:
            raise ValueError("device-only resident control omitted observed process RSS")
        attention_segments = [
            segment
            for segment in doc.get("segments", [])
            if isinstance(segment, dict) and segment.get("species") == "full_attention"
        ]
        if not attention_segments:
            raise ValueError("device-only resident control omitted full-attention segments")
        for segment in attention_segments:
            if segment.get("immutable_weight_ownership") != "device_only_after_source_upload":
                raise ValueError("full-attention segment retained an ambiguous weight owner")
            if segment.get("host_source_weights_retained_during_token_loop") is not False:
                raise ValueError("full-attention segment retained source weights during token execution")
        attention_residency = doc.get("execution", {}).get("full_attention_compact_bank_residency", {})
        expected_attention_status = (
            "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE"
            if token_major_resident
            else "PER_LAYER_DEVICE_ONLY_FULL_ATTENTION_BANKS_REUSED_ACROSS_SESSION_TOKENS"
        )
        if attention_residency.get("status") != expected_attention_status:
            raise ValueError("device-only resident control did not account for full-attention bank reuse")
        if int(attention_residency.get("resident_full_attention_layers", 0)) != len(attention_segments):
            raise ValueError("full-attention resident-layer count does not match the session")
        if int(attention_residency.get("expected_full_attention_layers", 0)) != len(attention_segments):
            raise ValueError("full-attention resident control omitted expected layer count")
        if int(attention_residency.get("device_weight_bytes", 0)) <= 0:
            raise ValueError("full-attention resident control omitted device weight accounting")
        if attention_residency.get("host_source_weights_retained_during_token_loop") is not False:
            raise ValueError("full-attention resident control retained source weights")
        if not token_major_resident and attention_residency.get("process_lifetime_all_banks_retained") is not False:
            raise ValueError("full-attention resident control overstates layer-major bank lifetime")
    if token_major_resident:
        if not device_only_resident:
            raise ValueError("token-major resident control requires device-only resident verification")
        token_major = doc.get("execution", {}).get("token_major_resident_banks", {})
        if token_major.get("status") != "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control did not retain all 48 banks")
        if int(token_major.get("resident_layers", 0)) != 48 or int(token_major.get("expected_layers", 0)) != 48:
            raise ValueError("token-major resident control omitted complete 48-layer accounting")
        if token_major.get("cross_layer_activation_handoff") != "device buffers":
            raise ValueError("token-major resident control did not use device activation handoffs")
        if token_major.get("source_reset_or_reprefill") is not False:
            raise ValueError("token-major resident control lacks persistent-session proof")
        linear = doc.get("execution", {}).get("linear_compact_bank_residency", {})
        if linear.get("status") != "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control omitted full linear-bank lifetime")
        attention = doc.get("execution", {}).get("full_attention_compact_bank_residency", {})
        if attention.get("status") != "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control omitted full-attention lifetime")
        if attention.get("process_lifetime_all_banks_retained") is not True:
            raise ValueError("token-major resident control did not retain full-attention banks")
        segments = doc.get("segments")
        if not isinstance(segments, list) or len(segments) != 48:
            raise ValueError("token-major resident control omitted per-layer execution evidence")
        seen_layers: set[int] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("token-major resident segment is not an object")
            raw_layer = segment.get("layer")
            if raw_layer is None:
                layers = segment.get("layers")
                raw_layer = layers[0] if isinstance(layers, list) and len(layers) == 2 and layers[0] == layers[1] else None
            if not isinstance(raw_layer, int) or raw_layer < 0 or raw_layer >= 48 or raw_layer in seen_layers:
                raise ValueError("token-major resident layer inventory is incomplete or duplicated")
            seen_layers.add(raw_layer)
            rows = segment.get("steps")
            if not isinstance(rows, list) or len(rows) != len(chain.token_ids):
                raise ValueError("token-major resident segment omitted repeated-token rows")
            for step, (row, token) in enumerate(zip(rows, chain.token_ids, strict=True)):
                if not isinstance(row, dict) or row.get("step") != step or row.get("token_id") != token:
                    raise ValueError("token-major resident token sequence drifted")
        if len(seen_layers) != 48:
            raise ValueError("token-major resident control did not cover all layers")


def _source_file_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256_file(resolved)}


def _native_session_artifact_dir(receipt: Path) -> Path:
    """Match the native executor's receipt-adjacent artifact namespace."""

    return receipt.with_name(f"{receipt.stem}.session_artifacts")


def _path_entry_exists(path: Path) -> bool:
    """Treat a dangling symlink as an occupied output target too."""

    return os.path.lexists(path)


def _reserve_fresh_output_namespace(teacher_path: Path, out: Path) -> tuple[Path, Path]:
    """Reserve a disjoint candidate namespace before source/model access.

    A route-union candidate must not merely avoid replacing the teacher JSON.
    Writing beneath the teacher's native session-artifact directory mixes two
    receipts' owned namespaces and makes later provenance ambiguous, even when
    every individual path is new.  Reject all nesting in either direction.
    """

    provenance_path = out.with_suffix(".route_union_provenance.json")
    session_artifacts = _native_session_artifact_dir(out)
    teacher_artifacts = _native_session_artifact_dir(teacher_path)
    candidate_targets = (out, provenance_path, session_artifacts)
    teacher_targets = (teacher_path, teacher_artifacts)
    for candidate in candidate_targets:
        for teacher_target in teacher_targets:
            if (
                candidate == teacher_target
                or teacher_target in candidate.parents
                or candidate in teacher_target.parents
            ):
                raise ValueError(
                    "route-union output namespace may not collide with or nest inside "
                    "the teacher receipt/session-artifact namespace"
                )
    existing = [str(target) for target in candidate_targets if _path_entry_exists(target)]
    if existing:
        raise ValueError(
            "route-union output, provenance, or session-artifact path already exists; "
            "choose fresh paths: "
            + ", ".join(existing)
        )
    return provenance_path, session_artifacts


def _reject_lexically_occupied_output_targets(out: Path) -> None:
    """Reject output symlinks before canonicalization can hide them.

    ``Path.resolve()`` follows a dangling output symlink to its missing target,
    so checking only the resolved spelling would accidentally treat the
    existing link as a fresh filename.  First reserve the lexical output name
    and its sibling targets; the later canonical reservation catches aliases
    through an existing parent symlink and namespace nesting.
    """

    targets = (
        out,
        out.with_suffix(".route_union_provenance.json"),
        _native_session_artifact_dir(out),
    )
    existing = [str(target) for target in targets if _path_entry_exists(target)]
    if existing:
        raise ValueError(
            "route-union output, provenance, or session-artifact path already exists; "
            "choose fresh paths: "
            + ", ".join(existing)
        )


def _native_binary_path() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    return target / "release/examples/flash_stateful_complete_token_session"


def _command_text(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _preexecution_build_binding(binary: Path) -> dict[str, Any]:
    """Fingerprint observed executable/source inputs before it runs.

    This is deliberately not called a compiler attestation: it has no signed
    build receipt showing that the executable was produced from this closure.
    The run remains non-promotable until a separate verified build attestation
    binds a cargo invocation, complete inputs, and this exact binary digest.
    """

    if not binary.is_file():
        raise FileNotFoundError(
            "prebuilt native executable is absent; build and inspect it separately before a route run: "
            f"{binary}"
        )
    git_status = _command_text(["git", "status", "--porcelain=v1"])
    source_paths = (
        Path(__file__),
        ROOT / "crates/hawking-core/examples/flash_stateful_complete_token_session.rs",
        ROOT / "crates/hawking-core/examples/flash_full_attention_layer3.rs",
        ROOT / "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
        ROOT / "crates/hawking-core/examples/flash_source_bf16_terminal.rs",
        ROOT / "crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs",
        ROOT / "Cargo.lock",
    )
    return {
        "captured_before_native_execution": True,
        "git_head": _command_text(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
        "worktree_dirty": bool(git_status),
        "rustc_version": _command_text(["rustc", "-Vv"]),
        "native_executable": _source_file_binding(binary),
        "source_closure": [_source_file_binding(path) for path in source_paths],
    }


def _same_executable_closure(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Ignore expected receipt creation while rejecting code/binary drift mid-run."""

    return all(
        before.get(key) == after.get(key)
        for key in ("git_head", "rustc_version", "native_executable", "source_closure")
    )


def _protected_native_lane() -> dict[str, Any]:
    """Observe known Flash/native and resident owners before a model launch.

    This is a conservative preflight, not a timing witness.  It prevents this
    adapter from starting alongside a visible protected Flash session or a
    live Hawking resident; future benchmark receipts still require their own
    before/after lane evidence.
    """

    completed = subprocess.run(
        ["ps", "-axo", "pid=,stat=,command="], check=True, capture_output=True, text=True
    )
    markers = (
        "flash_stateful_complete_token_session",
        "flash_repeated_accepted_decode.py",
        "flash_route_union_control.py",
        "flash_noetic_complete_layer0",
        "mlx_vlm.server",
        "hawkingd",
    )
    matches = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        raw_pid, state, command = parts
        if raw_pid == str(os.getpid()):
            continue
        if state.startswith("Z") or not command:
            continue
        if any(marker in command for marker in markers):
            matches.append({"state": state, "command": command})
    return {"clean": not matches, "matches": matches}


def _write_run_provenance(
    path: Path,
    *,
    command: list[str],
    teacher_path: Path,
    teacher: dict[str, Any],
    candidate_path: Path,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    build: dict[str, Any],
    lane_preflight: dict[str, Any],
) -> None:
    """Bind a future native candidate to observed executable/source state.

    The executor receipt owns numerical/session evidence.  This sidecar adds
    the wrapper's invocation, selected source contract, observed binary, and
    source-tree state without rewriting that native receipt.  It does not
    assert the binary was compiled from the listed files.
    """

    document: dict[str, Any] = {
        "schema": "hawking.flash.route_union_control_provenance.v1",
        "status": "BOUND_TEACHER_ROUTE_UNION_CANDIDATE__NOT_A_PROMOTION",
        "command": command,
        "reference_authority": reference,
        "teacher": {
            "path": str(teacher_path),
            "sha256": _sha256_file(teacher_path),
            "seal_sha256": teacher["seal_sha256"],
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "seal_sha256": candidate["seal_sha256"],
        },
        "build": build,
        "native_lane_preflight": lane_preflight,
        "claim_boundary": (
            "This binds one teacher-bound route-union candidate to a selected source-token contract and an "
            "observed executable/source closure; it is not a compiler build attestation. It does not establish "
            "source-independent execution, a "
            "native PLE body, complete EBPW, qualified TPS, capability, or promotion."
        ),
        "promotion_allowed": False,
    }
    document["seal_sha256"] = _compact_utf8_sorted_seal(document)
    try:
        # The native executor owns its receipt/artifact namespace.  This is
        # the wrapper-owned final artifact, so create it exclusively instead
        # of relying on the earlier existence check across a long native run.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise ValueError("route-union provenance path appeared during execution; preserving it") from exc


def _build_command(
    *,
    native_binary: Path,
    model_root: Path,
    teacher_path: Path,
    out: Path,
    chain: AcceptedChain,
    retain_linear_banks: bool,
    device_only_compact_banks: bool,
    token_major_resident_banks: bool,
) -> list[str]:
    command = [
        str(native_binary),
        "--root", str(model_root),
        "--token-ids", ",".join(str(token) for token in chain.token_ids),
        "--prompt-length", str(chain.prompt_length),
        "--route-teacher", str(teacher_path),
        "--out", str(out),
        # Rust rejects an unmarked raw launch before Metal.  This wrapper adds
        # the marker only after owner-authorized V2 source admission and its
        # own protected-lane preflight; it is an explicit routing marker, not
        # a substitute for the owner signature or a lease.
        "--wrapper-admitted-source-control",
        "--supervising-launcher-pid", str(os.getpid()),
    ]
    if retain_linear_banks:
        command.append("--retain-linear-banks")
    if device_only_compact_banks:
        command.append("--device-only-compact-banks")
    if token_major_resident_banks:
        command.append("--token-major-resident-banks")
    return command


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MODEL_ROOT)
    ap.add_argument(
        "--native-binary",
        type=Path,
        default=_native_binary_path(),
        help="prebuilt, inspected session executable; cargo run is intentionally not used here",
    )
    ap.add_argument(
        "--teacher", type=Path,
        help="explicit sealed all-48 token-major route teacher; no historical default is admitted",
    )
    ap.add_argument(
        "--reference-contract", type=Path,
        help="configured-trust PLE-inclusive source greedy-reference contract",
    )
    ap.add_argument(
        "--owner-authorization",
        type=Path,
        help="detached machine-admin Ed25519 authorization for --reference-contract",
    )
    ap.add_argument(
        "--retain-linear-banks",
        action="store_true",
        help="retain compact linear banks through the complete control process",
    )
    ap.add_argument(
        "--device-only-compact-banks",
        action="store_true",
        help="drop source-side compact linear weights after their Metal upload",
    )
    ap.add_argument(
        "--token-major-resident-banks",
        action="store_true",
        help="retain all 48 exact teacher-bound compact banks across the full token-major session",
    )
    ap.add_argument("--out", type=Path, help="new candidate receipt path; existing receipts are never overwritten")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.device_only_compact_banks and not args.retain_linear_banks:
        ap.error("--device-only-compact-banks requires --retain-linear-banks")
    if args.token_major_resident_banks and not (
        args.retain_linear_banks and args.device_only_compact_banks
    ):
        ap.error("--token-major-resident-banks requires --retain-linear-banks --device-only-compact-banks")
    missing = [
        flag
        for flag, value in (
            ("--teacher", args.teacher),
            ("--reference-contract", args.reference_contract),
            ("--owner-authorization", args.owner_authorization),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        if args.dry_run:
            print(json.dumps({
                "status": "DRY_RUN_WITHHELD_MISSING_AUTHORITATIVE_INPUTS",
                "required": missing,
                "claim_boundary": (
                    "No model, GPU, native executor, or receipt write occurred. A route-union run requires an "
                    "explicit complete token-major teacher, a configured-trust PLE-inclusive source contract, "
                    "and a fresh output path."
                ),
            }, indent=2))
            return 0
        raise ValueError("route-union control requires " + ", ".join(missing))
    assert (
        args.teacher is not None
        and args.reference_contract is not None
        and args.owner_authorization is not None
        and args.out is not None
    )
    if not args.dry_run and not (
        args.retain_linear_banks
        and args.device_only_compact_banks
        and args.token_major_resident_banks
    ):
        raise ValueError(
            "a native route-union control is restricted to the all-48 device-only token-major path; "
            "supply --retain-linear-banks --device-only-compact-banks --token-major-resident-banks"
        )
    model_root = args.root.resolve()
    native_binary = args.native_binary.resolve()
    teacher_path = args.teacher.resolve()
    raw_out = args.out.expanduser()
    if not raw_out.is_absolute():
        raw_out = Path.cwd() / raw_out
    _reject_lexically_occupied_output_targets(raw_out)
    out = raw_out.resolve()
    provenance_path, _session_artifacts = _reserve_fresh_output_namespace(
        teacher_path, out
    )
    if not model_root.is_dir():
        raise FileNotFoundError(f"Flash source root does not exist: {model_root}")
    reference = load_external_reference_contract(
        args.reference_contract,
        model_root,
        owner_authorization_path=args.owner_authorization,
    )
    source_input_identity = reference.get("source_input_identity")
    if not isinstance(source_input_identity, dict):
        raise ValueError("owner-authorized V2 source contract omitted native source_input_identity")
    teacher = _load(teacher_path)
    chain = _verify_teacher(teacher, source_input_identity=source_input_identity)
    _verify_reference_matches_teacher(reference, chain)
    command = _build_command(
        native_binary=native_binary,
        model_root=model_root,
        teacher_path=teacher_path,
        out=out,
        chain=chain,
        retain_linear_banks=args.retain_linear_banks,
        device_only_compact_banks=args.device_only_compact_banks,
        token_major_resident_banks=args.token_major_resident_banks,
    )
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_SOURCE_BOUND_TOKEN_MAJOR_ROUTE_CONTROL",
            "teacher": str(teacher_path),
            "token_ids": list(chain.token_ids),
            "prompt_length": chain.prompt_length,
            "reference_authority": reference,
            "native_binary": {
                "path": str(native_binary),
                "exists": native_binary.is_file(),
                "reopen_condition": "prebuild and inspect the exact binary before a non-dry route run",
            },
            "command": command,
            "claim_boundary": "no model, GPU, native executor, or receipt write occurred",
        }, indent=2))
        return 0
    build = _preexecution_build_binding(native_binary)
    lane_preflight = _protected_native_lane()
    if not lane_preflight["clean"]:
        raise ValueError("protected native lane is occupied; refusing route-union launch")
    subprocess.run(command, cwd=ROOT, check=True)
    if not _same_executable_closure(build, _preexecution_build_binding(native_binary)):
        raise ValueError(
            "native executable or its declared source closure changed during route-union execution; "
            "withholding candidate provenance"
        )
    candidate = _load(out)
    _verify_candidate(
        candidate,
        teacher,
        device_only_resident=args.device_only_compact_banks,
        token_major_resident=args.token_major_resident_banks,
        source_input_identity=source_input_identity,
    )
    _write_run_provenance(
        provenance_path,
        command=command,
        teacher_path=teacher_path,
        teacher=teacher,
        candidate_path=out,
        candidate=candidate,
        reference=reference,
        build=build,
        lane_preflight=lane_preflight,
    )
    print(json.dumps({
        "status": "PASSED_TEACHER_BOUND_ROUTE_UNION_CONTROL",
        "teacher": str(teacher_path),
        "candidate": str(out),
        "provenance": str(provenance_path),
        "teacher_source_bytes": teacher["execution"]["source_payload_bytes_read"],
        "candidate_source_bytes": candidate["execution"]["source_payload_bytes_read"],
        "device_only_linear_residency": args.device_only_compact_banks,
        "token_major_residency": args.token_major_resident_banks,
        "claim_boundary": "bounded teacher control only; when requested, all 48 compact banks remain device-only through the accepted token sequence with device activation handoffs. Per-layer diagnostics remain enabled, so this is not warmed full-model TPS, EBPW, capability, future-route coverage, or resident promotion.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "FAILED_PRECISE_DIAGNOSIS", "error": str(exc)}))
        raise SystemExit(2)
