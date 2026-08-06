#!/usr/bin/env python3
"""Seal a source-hash-bound DeepSeek-V4 tokenizer/template admission receipt.

This is deliberately an *asset* admission rather than a model-runtime test.  It
opens the exact ``tokenizer.json`` retained in a sealed streamed Gravity
artifact with the Rust ``tokenizers`` Python bindings, validates it against the
artifact manifest, and records a small public set of deterministic tokenization
fixtures.  It never supplies a chat template: if the source artifact did not
retain one, the resulting receipt says so explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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


SCHEMA = "hawking.gravity.deepseek_v4.tokenizer_template_admission.v1"
EXPECTED_MANIFEST_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
EXPECTED_MANIFEST_STATUS = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY"
EXPECTED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
EXPECTED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

METADATA_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
)

# These are public, synthetic coverage fixtures.  They deliberately contain no
# user prompts, repo text, tool payloads, or model output.
TOKENIZATION_FIXTURES = (
    ("ascii", "Hello, world!"),
    ("numbers", "x = 123_456\n"),
    ("cjk", "你好，世界"),
    ("code", "def f(x):\n    return x + 1\n"),
    ("unicode", "café — π"),
    ("raw_role_markers", "<｜User｜>hello<｜Assistant｜>"),
)

ROLE_AND_EFFECT_MARKERS = (
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|EOT|>",
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁outputs▁begin｜>",
    "<｜tool▁outputs▁end｜>",
    "<｜tool▁output▁begin｜>",
    "<｜tool▁output▁end｜>",
    "<｜tool▁sep｜>",
    "<think>",
    "</think>",
    "<｜begin▁sys｜>",
    "<｜end▁sys｜>",
)


class AdmissionError(RuntimeError):
    """The sealed asset cannot support the stated tokenizer contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdmissionError(f"{label} root must be a JSON object")
    return raw


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise AdmissionError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AdmissionError(f"{label} must be a regular non-symlink file")
    return st


def _metadata_inventory(metadata_dir: Path) -> list[str]:
    inventory: list[str] = []
    for path in sorted(metadata_dir.rglob("*")):
        if path.is_dir():
            continue
        _regular_file(path, label=f"metadata asset {path}")
        inventory.append(path.relative_to(metadata_dir).as_posix())
    return inventory


def _metadata_binding(
    *, artifact_dir: Path, manifest: Mapping[str, Any], relative: str
) -> dict[str, Any]:
    declared_assets = manifest.get("source", {}).get("metadata_assets")
    if not isinstance(declared_assets, dict):
        raise AdmissionError("manifest source.metadata_assets must be an object")
    declared = declared_assets.get(relative)
    if not isinstance(declared, dict):
        raise AdmissionError(f"manifest does not declare metadata asset {relative!r}")
    path = artifact_dir / "metadata" / relative
    st = _regular_file(path, label=f"metadata asset {relative}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != declared.get("sha256"):
        raise AdmissionError(
            f"metadata asset {relative} hash mismatch: actual={actual_sha256} "
            f"manifest={declared.get('sha256')!r}"
        )
    if st.st_size != declared.get("bytes"):
        raise AdmissionError(
            f"metadata asset {relative} size mismatch: actual={st.st_size} "
            f"manifest={declared.get('bytes')!r}"
        )
    if declared.get("path") != relative:
        raise AdmissionError(f"metadata asset {relative} manifest path drift")
    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "bytes": st.st_size,
        "sha256": actual_sha256,
        "manifest_declared_sha256": declared["sha256"],
        "manifest_declared_bytes": declared["bytes"],
        "regular_non_symlink": True,
    }


def _added_token_map(tokenizer_json: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    added = tokenizer_json.get("added_tokens")
    if not isinstance(added, list):
        raise AdmissionError("tokenizer.json added_tokens must be a list")
    result: dict[str, dict[str, Any]] = {}
    ids: set[int] = set()
    for row in added:
        if not isinstance(row, dict):
            raise AdmissionError("tokenizer.json added_tokens contains a non-object")
        token_id = row.get("id")
        content = row.get("content")
        if not isinstance(token_id, int) or token_id < 0 or not isinstance(content, str):
            raise AdmissionError("tokenizer.json added token lacks a nonnegative id/content")
        if token_id in ids or content in result:
            raise AdmissionError("tokenizer.json added token ids/content must be unique")
        ids.add(token_id)
        result[content] = row
    return result


def _required_token(
    tokenizer: Tokenizer, token_rows: Mapping[str, Mapping[str, Any]], content: str
) -> dict[str, Any]:
    row = token_rows.get(content)
    token_id = tokenizer.token_to_id(content)
    if row is None or token_id is None or int(row["id"]) != token_id:
        raise AdmissionError(f"required tokenizer token is absent or inconsistent: {content!r}")
    return {
        "id": token_id,
        "content": content,
        "special": bool(row.get("special")),
        "normalized": bool(row.get("normalized")),
        "lstrip": bool(row.get("lstrip")),
        "rstrip": bool(row.get("rstrip")),
        "single_word": bool(row.get("single_word")),
    }


def _tokenization_trace(tokenizer: Tokenizer, fixture_id: str, text: str) -> dict[str, Any]:
    # Load two independent backend instances so deterministic output is not only
    # a repeat on a single object.
    primary_without = tokenizer.encode(text, add_special_tokens=False)
    primary_with = tokenizer.encode(text, add_special_tokens=True)
    replay = Tokenizer.from_file(str(_TOKENIZER_PATH_FOR_REPLAY))
    replay_without = replay.encode(text, add_special_tokens=False)
    without = {"ids": list(primary_without.ids), "tokens": list(primary_without.tokens)}
    with_special = {"ids": list(primary_with.ids), "tokens": list(primary_with.tokens)}
    replay_value = {"ids": list(replay_without.ids), "tokens": list(replay_without.tokens)}
    decoded = tokenizer.decode(primary_without.ids, skip_special_tokens=False)
    if without != replay_value:
        raise AdmissionError(f"non-deterministic backend tokenization for fixture {fixture_id}")
    if decoded != text:
        raise AdmissionError(f"lossless tokenizer decode failed for fixture {fixture_id}")
    return {
        "fixture_id": fixture_id,
        "public_synthetic_input": text,
        "input_utf8_sha256": _sha256_text(text),
        "add_special_tokens_false": without,
        "add_special_tokens_true": with_special,
        "backend_replay_match": True,
        "automatic_special_tokens_inserted": without != with_special,
        "decoded_utf8_sha256": _sha256_text(decoded),
        "lossless_decode": True,
    }


# This module-level binding avoids hiding a filename inside a custom tokenizer
# implementation.  It is set immediately before fixture evaluation.
_TOKENIZER_PATH_FOR_REPLAY: Path


def _special_round_trip(tokenizer: Tokenizer, token: Mapping[str, Any]) -> dict[str, Any]:
    content = str(token["content"])
    encoded = tokenizer.encode(content, add_special_tokens=False)
    ids = list(encoded.ids)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if ids != [token["id"]] or decoded != content:
        raise AdmissionError(f"special-token lexical round-trip failed for {content!r}")
    return {
        "id": token["id"],
        "content": content,
        "encode_ids": ids,
        "decode_exact": True,
    }


def build_receipt(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    _regular_file(manifest_path, label="full streamed Gravity manifest")
    manifest = _read_json(manifest_path, label="full streamed Gravity manifest")
    try:
        verify(manifest, label="full streamed Gravity manifest")
    except ValueError as exc:
        raise AdmissionError(str(exc)) from exc
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise AdmissionError(f"unexpected artifact schema: {manifest.get('schema')!r}")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise AdmissionError(f"unexpected artifact status: {manifest.get('status')!r}")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise AdmissionError("manifest source must be an object")
    if source.get("repository") != EXPECTED_REPOSITORY or source.get("revision") != EXPECTED_REVISION:
        raise AdmissionError("pinned DeepSeek-V4 source identity drift")
    if source.get("source_parent_persisted") is not False:
        raise AdmissionError("source_parent_persisted must remain false")

    metadata_dir = artifact_dir / "metadata"
    if not metadata_dir.is_dir():
        raise AdmissionError("artifact metadata directory is missing")
    inventory = _metadata_inventory(metadata_dir)
    declared_inventory = sorted(source.get("metadata_assets", {}).keys())
    if inventory != declared_inventory:
        raise AdmissionError(
            "on-disk metadata inventory differs from manifest metadata_assets; "
            f"disk={inventory!r} manifest={declared_inventory!r}"
        )
    bindings = {
        relative: _metadata_binding(artifact_dir=artifact_dir, manifest=manifest, relative=relative)
        for relative in METADATA_FILES
    }

    tokenizer_json_path = artifact_dir / "metadata" / "tokenizer.json"
    tokenizer_config_path = artifact_dir / "metadata" / "tokenizer_config.json"
    model_config_path = artifact_dir / "metadata" / "config.json"
    tokenizer_json = _read_json(tokenizer_json_path, label="tokenizer.json")
    tokenizer_config = _read_json(tokenizer_config_path, label="tokenizer_config.json")
    model_config = _read_json(model_config_path, label="model config.json")

    if tokenizer_json.get("version") != "1.0":
        raise AdmissionError("tokenizer.json version must be '1.0'")
    model = tokenizer_json.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise AdmissionError("tokenizer.json must declare a BPE model")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise AdmissionError("tokenizer.json BPE vocab/merges structure is invalid")
    vocab_ids = list(vocab.values())
    if any(not isinstance(token_id, int) for token_id in vocab_ids):
        raise AdmissionError("tokenizer.json BPE vocab ids must be integers")
    if len(vocab_ids) != len(set(vocab_ids)):
        raise AdmissionError("tokenizer.json BPE vocab ids must be unique")
    token_rows = _added_token_map(tokenizer_json)
    if tokenizer_json.get("truncation") is not None or tokenizer_json.get("padding") is not None:
        raise AdmissionError("tokenizer.json must not silently configure truncation or padding")

    global _TOKENIZER_PATH_FOR_REPLAY
    _TOKENIZER_PATH_FOR_REPLAY = tokenizer_json_path
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    base_vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
    effective_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if base_vocab_size != len(vocab):
        raise AdmissionError("backend base vocab size does not equal tokenizer.json vocab size")
    if effective_vocab_size != int(model_config.get("vocab_size", -1)):
        raise AdmissionError("backend effective vocab size does not equal model config vocab_size")

    bos_content = str(tokenizer_config.get("bos_token", {}).get("content", ""))
    eos_content = str(tokenizer_config.get("eos_token", {}).get("content", ""))
    configured_pad_content = str(tokenizer_config.get("pad_token", {}).get("content", ""))
    bos = _required_token(tokenizer, token_rows, bos_content)
    eos = _required_token(tokenizer, token_rows, eos_content)
    native_pad = _required_token(tokenizer, token_rows, "<｜▁pad▁｜>")
    if bos["id"] != model_config.get("bos_token_id"):
        raise AdmissionError("tokenizer/model config BOS id mismatch")
    if eos["id"] != model_config.get("eos_token_id"):
        raise AdmissionError("tokenizer/model config EOS id mismatch")
    configured_pad_id = tokenizer.token_to_id(configured_pad_content)
    if configured_pad_id is None:
        raise AdmissionError("configured tokenizer pad content does not resolve to an id")

    marker_map = {
        marker: _required_token(tokenizer, token_rows, marker)
        for marker in ROLE_AND_EFFECT_MARKERS
    }
    traces = [_tokenization_trace(tokenizer, fixture_id, text) for fixture_id, text in TOKENIZATION_FIXTURES]
    special_round_trips = [_special_round_trip(tokenizer, token) for token in (bos, eos, native_pad)]

    chat_template_keys = ("chat_template", "default_chat_template")
    present_template_keys = {
        key: tokenizer_config[key] for key in chat_template_keys if key in tokenizer_config
    }
    present_tokenizer_json_template_keys = {
        key: tokenizer_json[key] for key in chat_template_keys if key in tokenizer_json
    }
    template_files = [
        item
        for item in inventory
        if "chat_template" in item.lower() or item.lower().endswith(".jinja")
    ]
    if present_template_keys or present_tokenizer_json_template_keys or template_files:
        raise AdmissionError(
            "this admission is intentionally scoped to the captured no-template source; "
            "a template asset was unexpectedly found"
        )

    special_count = sum(bool(row.get("special")) for row in token_rows.values())
    non_special_count = len(token_rows) - special_count
    normalizer = tokenizer_json.get("normalizer")
    pre_tokenizer = tokenizer_json.get("pre_tokenizer")
    post_processor = tokenizer_json.get("post_processor")
    decoder = tokenizer_json.get("decoder")
    if not all(isinstance(value, dict) for value in (normalizer, pre_tokenizer, post_processor, decoder)):
        raise AdmissionError("tokenizer normalization/pre-tokenization/post-processing/decoder fields must be objects")

    configured_token_normalization = {
        "bos_token_config_normalized": tokenizer_config.get("bos_token", {}).get("normalized"),
        "bos_tokenizer_json_normalized": bos["normalized"],
        "eos_token_config_normalized": tokenizer_config.get("eos_token", {}).get("normalized"),
        "eos_tokenizer_json_normalized": eos["normalized"],
        "pad_token_config_normalized": tokenizer_config.get("pad_token", {}).get("normalized"),
        "native_pad_tokenizer_json_normalized": native_pad["normalized"],
        "wrapper_semantics_executed": False,
        "detail": (
            "The tokenizer_config AddedToken objects set normalized=true while the corresponding "
            "tokenizer.json special rows set normalized=false. This receipt binds both encodings but "
            "does not claim a Transformers-wrapper reconciliation."
        ),
    }

    receipt = {
        "schema": SCHEMA,
        "status": "PASS_TOKENIZER_BACKEND_ADMISSION_CHAT_TEMPLATE_UNSPECIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "repository": source["repository"],
            "revision": source["revision"],
            "source_parent_persisted": False,
        },
        "evidence_bindings": {
            "full_stream_manifest": {
                "path": str(manifest_path.resolve()),
                "file_sha256": _sha256_file(manifest_path),
                "schema": manifest["schema"],
                "status": manifest["status"],
                "seal_sha256": manifest["seal_sha256"],
            },
            "metadata_assets": bindings,
            "metadata_inventory": {
                "exact_manifest_match": True,
                "artifact_relative_files": inventory,
                "manifest_relative_files": declared_inventory,
            },
        },
        "backend_admission": {
            "implementation": "huggingface_tokenizers_python_binding",
            "tokenizers_version": tokenizers.__version__,
            "custom_tokenizer_implementation_used": False,
            "transformers_wrapper_executed": False,
            "backend_loaded_exact_tokenizer_json": True,
            "fixture_count": len(traces),
        },
        "tokenizer_structure": {
            "format_version": tokenizer_json["version"],
            "model_type": model["type"],
            "model_vocab_entries": len(vocab),
            "model_vocab_id_min": min(vocab_ids),
            "model_vocab_id_max": max(vocab_ids),
            "model_merges": len(merges),
            "added_token_entries": len(token_rows),
            "added_token_special_count": special_count,
            "added_token_non_special_count": non_special_count,
            "effective_vocab_size": effective_vocab_size,
            "truncation": "ABSENT",
            "padding": "ABSENT",
            "normalizer": normalizer,
            "pre_tokenizer": pre_tokenizer,
            "post_processor": post_processor,
            "decoder": decoder,
        },
        "tokenizer_config_contract": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "legacy": tokenizer_config.get("legacy"),
            "model_max_length": tokenizer_config.get("model_max_length"),
            "add_bos_token": tokenizer_config.get("add_bos_token"),
            "add_eos_token": tokenizer_config.get("add_eos_token"),
            "clean_up_tokenization_spaces": tokenizer_config.get("clean_up_tokenization_spaces"),
            "unk_token": tokenizer_config.get("unk_token"),
            "bos_token": tokenizer_config.get("bos_token"),
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
        },
        "model_config_crosscheck": {
            "model_type": model_config.get("model_type"),
            "architectures": model_config.get("architectures"),
            "vocab_size": model_config.get("vocab_size"),
            "bos_token_id": model_config.get("bos_token_id"),
            "eos_token_id": model_config.get("eos_token_id"),
            "pad_token_id": model_config.get("pad_token_id"),
            "max_position_embeddings": model_config.get("max_position_embeddings"),
        },
        "special_id_contract": {
            "bos": bos,
            "eos": eos,
            "native_pad_marker": native_pad,
            "configured_pad_token_content": configured_pad_content,
            "configured_pad_token_resolved_id": configured_pad_id,
            "model_config_pad_token_id": model_config.get("pad_token_id"),
            "configured_token_normalization": configured_token_normalization,
            "padding_authority_status": "AMBIGUOUS_DO_NOT_INFER_RUNTIME_PADDING_POLICY",
            "padding_authority_detail": (
                "tokenizer_config pad_token content resolves to the EOS id, while tokenizer.json also "
                "contains a distinct native pad marker and model config has null pad_token_id."
            ),
            "special_token_lexical_round_trips": special_round_trips,
        },
        "raw_role_and_effect_marker_inventory": marker_map,
        "chat_template_contract": {
            "tokenizer_config_chat_template_key": "ABSENT",
            "tokenizer_config_default_chat_template_key": "ABSENT",
            "tokenizer_json_chat_template_key": "ABSENT",
            "tokenizer_json_default_chat_template_key": "ABSENT",
            "artifact_metadata_template_files": template_files,
            "source_chat_template_status": "NO_SOURCE_CHAT_TEMPLATE_CAPTURED_OR_ADMITTED",
            "role_and_tool_markers_are_not_a_template": True,
            "required_runtime_behavior": (
                "Do not derive a structured-chat prompt from role/tool marker token names. "
                "Structured HCLI tokenization remains blocked pending a source-authorized template "
                "or a separately sealed externally supplied template contract."
            ),
        },
        "deterministic_public_tokenization_traces": traces,
        "future_runtime_acceptance_contract": {
            "must_bind_exact_tokenizer_json_sha256": bindings["tokenizer.json"]["sha256"],
            "must_reproduce_fixture_ids_and_tokens": True,
            "must_not_implicitly_insert_bos_or_eos": True,
            "padding_requires_explicit_policy_admission": True,
            "chat_template_requires_separate_admission": True,
            "primary_8k_benchmark_is_a_runtime_limit_not_a_tokenizer_template": True,
        },
        "claim_boundary": {
            "full_43_layer_runtime": False,
            "source_cpu_parity": False,
            "numeric_parity_v2_1": False,
            "first_token_parity": False,
            "base_true_tps": False,
            "gpu_dispatches": 0,
            "hcli_chat_template_parity": False,
            "hcli_endpoint_exercised": False,
            "model_weights_read": False,
            "kimi_or_glm_inheritance": False,
            "ramanujan_or_odyssey_work": False,
        },
        "scope": {
            "admitted": "sealed tokenizer metadata identity, backend load, structure, special IDs, and bounded synthetic tokenization traces",
            "not_admitted": "a source chat template, Transformers wrapper behavior, full model runtime, generation, or TPS",
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--replace-previous-admission",
        action="store_true",
        help=(
            "Replace a prior sealed admission only after its seal, schema, source, and full-manifest "
            "binding have been checked against this run. Intended only to correct an unfinished local issue."
        ),
    )
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(args.artifact_dir)
        verify(receipt, label="DeepSeek-V4 tokenizer/template admission receipt")
        if args.out.exists():
            existing = _read_json(args.out, label="existing output receipt")
            if existing == receipt:
                return 0
            if not args.replace_previous_admission:
                raise AdmissionError(f"refusing to overwrite existing receipt: {args.out}")
            try:
                verify(existing, label="prior tokenizer/template admission receipt")
            except ValueError as exc:
                raise AdmissionError(f"prior receipt does not have a valid canonical seal: {exc}") from exc
            prior_manifest = existing.get("evidence_bindings", {}).get("full_stream_manifest", {})
            new_manifest = receipt["evidence_bindings"]["full_stream_manifest"]
            if (
                existing.get("schema") != SCHEMA
                or existing.get("source") != receipt["source"]
                or prior_manifest.get("seal_sha256") != new_manifest["seal_sha256"]
            ):
                raise AdmissionError("prior receipt identity differs; refusing replacement")
        _atomic_write_json(args.out, receipt)
    except AdmissionError as exc:
        print(f"tokenizer/template admission failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "status": receipt["status"], "seal_sha256": receipt["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
