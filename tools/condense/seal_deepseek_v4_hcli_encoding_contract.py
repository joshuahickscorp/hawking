#!/usr/bin/env python3
"""Seal the pinned DeepSeek-V4 structured HCLI encoding contract.

The full streamed Gravity artifact intentionally carries tokenizer metadata but
not the upstream ``encoding/`` reference folder.  This tool admits that folder
only when its local Hugging Face download metadata, file digests, the sealed
Gravity manifest, and the prior tokenizer admission all bind to the same pinned
revision.  It runs the exact upstream test driver and checks its golden prompt
strings through the exact sealed tokenizer.

It does not load model weights, start HCLI, or claim generation/capability/TPS.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.receipts import seal, verify  # noqa: E402

try:
    import tokenizers
    from tokenizers import Tokenizer
except ImportError as exc:  # pragma: no cover - environment admission error
    raise SystemExit(
        "The real `tokenizers` backend is required; do not substitute a custom tokenizer. "
        "Run with the Condense virtual environment."
    ) from exc


SCHEMA = "hawking.gravity.deepseek_v4.hcli_encoding_contract.v1"
TOKENIZER_ADMISSION_SCHEMA = "hawking.gravity.deepseek_v4.tokenizer_template_admission.v1"
EXPECTED_MANIFEST_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
EXPECTED_MANIFEST_STATUS = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY"
EXPECTED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
EXPECTED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

SOURCE_FILES = (
    "README.md",
    "encoding_dsv4.py",
    "test_encoding_dsv4.py",
    "tests/test_input_1.json",
    "tests/test_input_2.json",
    "tests/test_input_3.json",
    "tests/test_input_4.json",
    "tests/test_output_1.txt",
    "tests/test_output_2.txt",
    "tests/test_output_3.txt",
    "tests/test_output_4.txt",
)

CASE_THINKING_MODE = {1: "thinking", 2: "thinking", 3: "thinking", 4: "chat"}

GRAMMAR_MARKERS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜latest_reminder｜>",
    "<｜action｜>",
    "<｜query｜>",
    "<｜authority｜>",
    "<｜domain｜>",
    "<｜title｜>",
    "<｜read_url｜>",
)


class ContractError(RuntimeError):
    """A pinned source, tokenizer, or reference execution condition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"{label} root must be a JSON object")
    return raw


def _regular(path: Path, *, label: str) -> os.stat_result:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    return st


def _verify_sealed(path: Path, *, label: str) -> dict[str, Any]:
    _regular(path, label=label)
    value = _read_json(path, label=label)
    try:
        verify(value, label=label)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return value


def _manifest_binding(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    manifest = _verify_sealed(manifest_path, label="full streamed Gravity manifest")
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise ContractError(f"unexpected Gravity manifest schema: {manifest.get('schema')!r}")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise ContractError(f"unexpected Gravity manifest status: {manifest.get('status')!r}")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ContractError("Gravity manifest source must be an object")
    if source.get("repository") != EXPECTED_REPOSITORY or source.get("revision") != EXPECTED_REVISION:
        raise ContractError("Gravity manifest repository/revision drift")
    if source.get("source_parent_persisted") is not False:
        raise ContractError("Gravity manifest source_parent_persisted must remain false")
    metadata_assets = source.get("metadata_assets")
    if not isinstance(metadata_assets, dict):
        raise ContractError("Gravity manifest source.metadata_assets must be an object")
    declared_tokenizer = metadata_assets.get("tokenizer.json")
    if not isinstance(declared_tokenizer, dict):
        raise ContractError("Gravity manifest does not bind tokenizer.json")
    tokenizer_path = artifact_dir / "metadata" / "tokenizer.json"
    st = _regular(tokenizer_path, label="sealed tokenizer.json")
    actual_hash = _sha256_file(tokenizer_path)
    if actual_hash != declared_tokenizer.get("sha256") or st.st_size != declared_tokenizer.get("bytes"):
        raise ContractError("sealed tokenizer.json does not match the Gravity manifest")
    binding = {
        "path": str(manifest_path),
        "file_sha256": _sha256_file(manifest_path),
        "schema": manifest["schema"],
        "status": manifest["status"],
        "seal_sha256": manifest["seal_sha256"],
        "repository": source["repository"],
        "revision": source["revision"],
    }
    tokenizer_binding = {
        "path": str(tokenizer_path),
        "bytes": st.st_size,
        "sha256": actual_hash,
        "manifest_declared_sha256": declared_tokenizer["sha256"],
        "manifest_declared_bytes": declared_tokenizer["bytes"],
        "regular_non_symlink": True,
    }
    return manifest, {"manifest": binding, "tokenizer": tokenizer_binding}, tokenizer_path


def _tokenizer_admission_binding(path: Path, *, expected_tokenizer_sha256: str) -> dict[str, Any]:
    receipt = _verify_sealed(path, label="prior DSV4F tokenizer/template admission")
    if receipt.get("schema") != TOKENIZER_ADMISSION_SCHEMA:
        raise ContractError("prior tokenizer admission schema mismatch")
    if receipt.get("source", {}).get("repository") != EXPECTED_REPOSITORY or receipt.get("source", {}).get("revision") != EXPECTED_REVISION:
        raise ContractError("prior tokenizer admission source identity drift")
    observed = receipt.get("evidence_bindings", {}).get("metadata_assets", {}).get("tokenizer.json", {})
    if observed.get("sha256") != expected_tokenizer_sha256:
        raise ContractError("prior tokenizer admission does not bind the exact sealed tokenizer")
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "schema": receipt["schema"],
        "status": receipt.get("status"),
        "seal_sha256": receipt["seal_sha256"],
    }


def _source_file_binding(source_root: Path, relative: str) -> dict[str, Any]:
    encoding_dir = source_root / "encoding"
    path = encoding_dir / relative
    st = _regular(path, label=f"encoding source {relative}")
    metadata_path = source_root / ".cache" / "huggingface" / "download" / "encoding" / f"{relative}.metadata"
    _regular(metadata_path, label=f"Hugging Face download metadata for {relative}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0] != EXPECTED_REVISION:
        raise ContractError(f"download metadata revision drift for {relative}")
    blob_id = lines[1]
    if len(blob_id) != 40 or any(c not in "0123456789abcdef" for c in blob_id):
        raise ContractError(f"download metadata git blob id is invalid for {relative}")
    return {
        "relative_path": relative,
        "path": str(path.resolve()),
        "bytes": st.st_size,
        "sha256": _sha256_file(path),
        "huggingface_download_metadata_path": str(metadata_path.resolve()),
        "download_metadata_revision": lines[0],
        "download_metadata_git_blob_id": blob_id,
        "regular_non_symlink": True,
    }


def _source_inventory(source_root: Path) -> dict[str, dict[str, Any]]:
    encoding_dir = source_root / "encoding"
    if not encoding_dir.is_dir():
        raise ContractError(f"encoding source directory is missing: {encoding_dir}")
    actual = sorted(
        path.relative_to(encoding_dir).as_posix()
        for path in encoding_dir.rglob("*")
        if path.is_file()
    )
    expected = sorted(SOURCE_FILES)
    if actual != expected:
        raise ContractError(f"encoding source inventory drift: actual={actual!r} expected={expected!r}")
    bindings = {relative: _source_file_binding(source_root, relative) for relative in SOURCE_FILES}
    return bindings


def _run_exact_upstream_driver(encoding_dir: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "test_encoding_dsv4.py"],
            cwd=str(encoding_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"cannot execute exact upstream encoding test driver: {exc}") from exc
    expected_markers = [f"[PASS] case {case}:" for case in range(1, 5)]
    if result.returncode != 0 or "All 4 tests passed!" not in result.stdout or any(
        marker not in result.stdout for marker in expected_markers
    ):
        raise ContractError(
            "exact upstream encoding test driver failed: "
            f"returncode={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return {
        "entrypoint": "test_encoding_dsv4.py",
        "execution": "exact_hashed_source_driver_via_current_python",
        "python_executable": sys.executable,
        "returncode": result.returncode,
        "all_four_passed": True,
        "stdout_sha256": _sha256_text(result.stdout),
        "stderr_sha256": _sha256_text(result.stderr),
        "stdout_contains_case_pass_markers": expected_markers,
        "source_tree_bytecode_written": False,
    }


def _load_reference_module(source_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_dsv4_hcli_encoding_reference", source_file)
    if spec is None or spec.loader is None:
        raise ContractError("cannot create import spec for exact encoding_dsv4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_messages(case: int, raw: Any) -> tuple[list[dict[str, Any]], str]:
    if case == 1:
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list) or not isinstance(raw.get("tools"), list):
            raise ContractError("upstream test input 1 has unexpected structure")
        messages = copy.deepcopy(raw["messages"])
        messages[0]["tools"] = copy.deepcopy(raw["tools"])
        return messages, CASE_THINKING_MODE[case]
    if not isinstance(raw, list):
        raise ContractError(f"upstream test input {case} must be a message list")
    return copy.deepcopy(raw), CASE_THINKING_MODE[case]


def _grammar_marker_alignment(tokenizer: Tokenizer) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for marker in GRAMMAR_MARKERS:
        token_id = tokenizer.token_to_id(marker)
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if token_id is None or list(encoded.ids) != [token_id]:
            raise ContractError(f"sealed tokenizer does not encode grammar marker as exactly one id: {marker!r}")
        if tokenizer.decode(encoded.ids, skip_special_tokens=False) != marker:
            raise ContractError(f"sealed tokenizer does not lexically round-trip grammar marker: {marker!r}")
        output[marker] = {"id": token_id, "single_id_encode": True, "lexical_round_trip": True}
    return output


def _public_vector_alignment(
    *, source_root: Path, module: Any, tokenizer: Tokenizer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tests_dir = source_root / "encoding" / "tests"
    result: list[dict[str, Any]] = []
    parser_cases: dict[str, Any] = {}
    for case in range(1, 5):
        input_path = tests_dir / f"test_input_{case}.json"
        output_path = tests_dir / f"test_output_{case}.txt"
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        messages, thinking_mode = _case_messages(case, raw)
        golden = output_path.read_text(encoding="utf-8")
        encoded_prompt = module.encode_messages(messages, thinking_mode=thinking_mode)
        if encoded_prompt != golden:
            raise ContractError(f"reference encoder drift against test output {case}")
        no_special = tokenizer.encode(golden, add_special_tokens=False)
        with_special = tokenizer.encode(golden, add_special_tokens=True)
        decoded = tokenizer.decode(no_special.ids, skip_special_tokens=False)
        if decoded != golden:
            raise ContractError(f"sealed tokenizer does not round-trip upstream prompt {case}")
        result.append(
            {
                "case": case,
                "thinking_mode": thinking_mode,
                "source_input_relative_path": f"tests/test_input_{case}.json",
                "source_output_relative_path": f"tests/test_output_{case}.txt",
                "encoder_output_matches_hashed_golden": True,
                "prompt_utf8_bytes": len(golden.encode("utf-8")),
                "prompt_utf8_sha256": _sha256_text(golden),
                "token_count": len(no_special.ids),
                "token_ids_sha256": _canonical_hash(list(no_special.ids)),
                "token_strings_sha256": _canonical_hash(list(no_special.tokens)),
                "add_special_tokens_changes_ids": list(no_special.ids) != list(with_special.ids),
                "sealed_tokenizer_lexical_round_trip": True,
            }
        )

        # These are the exact parse checks in the upstream test driver, kept
        # separately so the receipt does not retain its public prose samples.
        if case == 1:
            marker = "<｜Assistant｜><think>"
            first_start = golden.find(marker) + len(marker)
            first_end = golden.find("<｜User｜>", first_start)
            first = module.parse_message_from_completion_text(golden[first_start:first_end], thinking_mode="thinking")
            last_start = golden.rfind(marker) + len(marker)
            last = module.parse_message_from_completion_text(golden[last_start:], thinking_mode="thinking")
            if (
                first.get("role") != "assistant"
                or len(first.get("tool_calls", [])) != 1
                or first["tool_calls"][0].get("function", {}).get("name") != "get_weather"
                or last.get("role") != "assistant"
                or last.get("tool_calls") != []
            ):
                raise ContractError("upstream test-vector 1 parser contract drift")
            parser_cases["case_1"] = {
                "thinking_mode": "thinking",
                "tool_call_count": 1,
                "first_tool_function_name": "get_weather",
                "final_tool_call_count": 0,
                "source_driver_parser_assertions_rechecked": True,
            }
        elif case == 2:
            marker = "<｜Assistant｜><think>"
            last_start = golden.rfind(marker) + len(marker)
            parsed = module.parse_message_from_completion_text(golden[last_start:], thinking_mode="thinking")
            if parsed.get("role") != "assistant" or parsed.get("tool_calls") != []:
                raise ContractError("upstream test-vector 2 parser contract drift")
            parser_cases["case_2"] = {
                "thinking_mode": "thinking",
                "tool_call_count": 0,
                "source_driver_parser_assertions_rechecked": True,
            }
    return result, parser_cases


def _contract_smoke_probes(module: Any, tokenizer: Tokenizer) -> dict[str, Any]:
    """Exercise public synthetic boundary cases not covered by the four goldens."""
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "status", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
    }
    response_format_prompt = module.encode_messages(
        [
            {"role": "system", "content": "Return a status object.", "response_format": response_format},
            {"role": "user", "content": "Is the service ready?"},
        ],
        thinking_mode="chat",
    )
    response_format_tokenized = tokenizer.encode(response_format_prompt, add_special_tokens=False)
    if "## Response Format:" not in response_format_prompt or tokenizer.decode(
        response_format_tokenized.ids, skip_special_tokens=False
    ) != response_format_prompt:
        raise ContractError("response_format source smoke probe failed")

    structured_completion = '{"ok":true}<｜end▁of▁sentence｜>'
    parsed_structured = module.parse_message_from_completion_text(structured_completion, thinking_mode="chat")
    if parsed_structured.get("content") != '{"ok":true}' or parsed_structured.get("tool_calls") != []:
        raise ContractError("structured completion parser boundary smoke probe failed")

    baseline = module.encode_messages(
        [{"role": "user", "content": "Give a terse answer."}], thinking_mode="thinking", reasoning_effort=None
    )
    high = module.encode_messages(
        [{"role": "user", "content": "Give a terse answer."}], thinking_mode="thinking", reasoning_effort="high"
    )
    maximum = module.encode_messages(
        [{"role": "user", "content": "Give a terse answer."}], thinking_mode="thinking", reasoning_effort="max"
    )
    if high != baseline or not maximum.startswith(module.bos_token + module.REASONING_EFFORT_MAX):
        raise ContractError("reasoning_effort source behavior smoke probe failed")

    extracted_url_rejected = False
    try:
        module.encode_messages(
            [{"role": "user", "content": "https://example.invalid", "task": "extracted_url"}], thinking_mode="chat"
        )
    except AssertionError:
        extracted_url_rejected = True
    if not extracted_url_rejected:
        raise ContractError("source executable unexpectedly accepted README-only extracted_url task")

    return {
        "response_format": {
            "source_behavior": "injects serialized schema instruction into prompt",
            "prompt_utf8_sha256": _sha256_text(response_format_prompt),
            "token_count": len(response_format_tokenized.ids),
            "sealed_tokenizer_round_trip": True,
            "reference_validates_generated_json": False,
        },
        "completion_parser_structured_json": {
            "source_behavior": "returns completion content as text; it does not json-parse or schema-validate ordinary response content",
            "fixture_content_is_json_parseable": True,
            "reference_json_validation_performed": False,
        },
        "reasoning_effort": {
            "accepted_values": [None, "high", "max"],
            "high_equals_none_in_hashed_reference": True,
            "max_prepends_explicit_source_prefix": True,
        },
        "readme_executable_drift": {
            "readme_mentions_task": "extracted_url",
            "executable_valid_tasks": sorted(module.VALID_TASKS),
            "executable_rejects_extracted_url": True,
            "adapter_authority": "HASHED_EXECUTABLE_OVER_DOCUMENTATION_ONLY_TASK_LIST",
        },
    }


def build_receipt(*, artifact_dir: Path, source_root: Path, tokenizer_admission: Path) -> dict[str, Any]:
    manifest, gravity_binding, tokenizer_path = _manifest_binding(artifact_dir)
    prior_tokenizer = _tokenizer_admission_binding(
        tokenizer_admission, expected_tokenizer_sha256=gravity_binding["tokenizer"]["sha256"]
    )
    source_root = source_root.resolve()
    source_bindings = _source_inventory(source_root)
    encoding_dir = source_root / "encoding"
    driver = _run_exact_upstream_driver(encoding_dir)
    module = _load_reference_module(encoding_dir / "encoding_dsv4.py")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    grammar_markers = _grammar_marker_alignment(tokenizer)
    vectors, parser_cases = _public_vector_alignment(source_root=source_root, module=module, tokenizer=tokenizer)
    smoke_probes = _contract_smoke_probes(module, tokenizer)

    if manifest["source"]["revision"] != EXPECTED_REVISION:
        raise ContractError("full artifact revision changed during encoding admission")

    receipt = {
        "schema": SCHEMA,
        "status": "PASS_PINNED_DSV4F_HCLI_ENCODING_CONTRACT_NOT_FULL_RUNTIME",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"repository": EXPECTED_REPOSITORY, "revision": EXPECTED_REVISION},
        "evidence_bindings": {
            "full_streamed_gravity": gravity_binding["manifest"],
            "sealed_tokenizer": gravity_binding["tokenizer"],
            "prior_tokenizer_template_admission": prior_tokenizer,
            "encoding_source_root": str(source_root),
            "encoding_source_files": source_bindings,
            "encoding_source_inventory_exact": sorted(source_bindings),
        },
        "reference_execution": driver,
        "tokenizer_alignment": {
            "implementation": "huggingface_tokenizers_python_binding",
            "tokenizers_version": tokenizers.__version__,
            "custom_tokenizer_implementation_used": False,
            "grammar_markers": grammar_markers,
            "public_upstream_golden_vectors": vectors,
            "manual_bos_eos_schedule_required": True,
            "automatic_special_token_insertion_observed_in_goldens": False,
        },
        "source_parser_coverage": {
            "upstream_driver_parse_cases": parser_cases,
            "completion_input_boundary": (
                "Pass only the generated assistant continuation after the prompt's assistant/thinking prefix; "
                "in thinking mode the reference expects reasoning text followed by </think>."
            ),
            "well_formed_only": True,
            "malformed_completion_recovery": False,
            "tool_calls_output_shape": {
                "role": "assistant",
                "content": "string",
                "reasoning_content": "string",
                "tool_calls": "OpenAI-style [{type:'function', function:{name:string, arguments:JSON-string}}]",
            },
        },
        "hcli_adapter_contract": {
            "encode_entrypoint": "encoding_dsv4.encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None)",
            "parse_entrypoint": "encoding_dsv4.parse_message_from_completion_text(completion_text, thinking_mode)",
            "authorized_input_roles_by_reference": ["system", "user", "assistant", "tool", "latest_reminder", "developer"],
            "developer_role_boundary": (
                "The reference implements developer, while its README says it is internal-search-pipeline-only and "
                "not accepted by the official general API. Do not expose it as ordinary public HCLI chat without a separate policy decision."
            ),
            "thinking_modes": {
                "accepted": ["chat", "thinking"],
                "chat_generation_prefix": "<｜Assistant｜></think>",
                "thinking_generation_prefix": "<｜Assistant｜><think>",
            },
            "bos_and_context": {
                "new_conversation_with_add_default_bos_token": "prepend <｜begin▁of▁sentence｜>",
                "context_nonempty": "do not prepend default BOS",
                "tokenizer_call_for_rendered_prompt": "encode(rendered_prompt, add_special_tokens=False)",
            },
            "thinking_retention": {
                "default_drop_thinking": True,
                "source_auto_disables_drop_thinking_when_any_message_defines_tools": True,
                "tool_reasoning_retention_is_source_behavior": True,
            },
            "tools": {
                "definition_input": "OpenAI-compatible [{'type':'function','function':{name,description,parameters}}] on system or developer message",
                "assistant_call_rendering": "DSML tool_calls/invoke/parameter grammar",
                "tool_message_preprocess": "merge standalone tool messages into user content_blocks as <tool_result>...</tool_result>",
                "result_ordering": "sort tool_result blocks by preceding assistant tool_call id/order when identifiers are available",
                "tool_execution_authorized": False,
            },
            "structured_json": {
                "response_format_input": "JSON-serializable schema object injected into source prompt as instructions",
                "tool_nonstring_parameter_encoding": "JSON text with string=\"false\"",
                "tool_string_parameter_encoding": "raw text with string=\"true\"",
                "ordinary_completion_json_validation_by_reference": False,
                "hcli_json_schema_enforcement_authorized_by_this_receipt": False,
            },
            "quick_tasks": {
                "executable_valid_tasks": sorted(module.VALID_TASKS),
                "README_only_extracted_url_not_authorized": True,
            },
            "source_nonrecovering_boundaries": {
                "unsupported_message_role": "NotImplementedError",
                "malformed_completion": "AssertionError or ValueError; no recovery",
                "unsupported_content_block": "reference renders an [Unsupported type] string; strict HCLI adapter must not advertise this as typed content support",
                "delimiter_escaping": "not implemented by the reference; this receipt does not claim prompt-injection mitigation",
            },
        },
        "source_contract_smoke_probes": smoke_probes,
        "template_authorization": {
            "prior_artifact_metadata_status": "full Gravity metadata did not contain a template file",
            "new_authority": "separately captured and cache-revision-bound upstream encoding_dsv4.py at the exact same pinned revision",
            "now_authorized": (
                "Use the exact hashed reference grammar for rendering supported structured chat/tool prompts and parsing well-formed "
                "assistant continuations, then tokenize the rendered text with the exact sealed tokenizer using add_special_tokens=False."
            ),
            "still_not_authorized": [
                "full 43-layer V4 runtime/HCLI endpoint claim",
                "model completion or tool-execution capability claim",
                "malformed-completion repair",
                "generic structured-JSON schema validation",
                "an inferred tokenizer padding policy",
                "documentation-only extracted_url task",
            ],
        },
        "claim_boundary": {
            "full_43_layer_runtime": False,
            "source_cpu_parity": False,
            "numeric_parity_v2_1": False,
            "first_token_parity": False,
            "base_true_tps": False,
            "gpu_dispatches": 0,
            "hcli_endpoint_started": False,
            "hcli_request_executed_through_model": False,
            "tool_execution": False,
            "model_weights_read": False,
            "kimi_or_glm_inheritance": False,
            "ramanujan_or_odyssey_work": False,
        },
    }
    return seal(receipt)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--encoding-source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"refusing to overwrite existing receipt: {args.out}", file=sys.stderr)
        return 2
    try:
        receipt = build_receipt(
            artifact_dir=args.artifact_dir,
            source_root=args.encoding_source_root,
            tokenizer_admission=args.tokenizer_admission,
        )
        verify(receipt, label="DSV4F HCLI encoding contract")
        _atomic_write_json(args.out, receipt)
    except ContractError as exc:
        print(f"HCLI encoding contract failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "status": receipt["status"], "seal_sha256": receipt["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
