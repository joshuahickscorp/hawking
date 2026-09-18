"""Fail-closed CPU preflight for Flash-Next E13 vocabulary-row extraction.

E13 asks whether token structure is predictable jointly across frequency and
script strata.  This module does *not* read embedding/head payload bytes or
score a model.  It verifies the exact pinned vocabulary geometry and rejects a
row-extraction request until its corpus-frequency evidence, tokenizer script
labels, disjoint fit/heldout partition, and within-stratum shuffled-token null
are all bound to that source.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from hawking.agentos.flash_tensor_probe import _final_root, load_tensor_header
from hawking.flash_next import PINNED_REVISION, REPO_ID
from hawking.persist import atomic_write_json


SCHEMA = "hawking.agentos.flash_vocabulary_prefilter.v1"
PLAN_SCHEMA = "hawking.agentos.flash_vocabulary_prefilter_plan.v1"
FREQUENCY_EVIDENCE_SCHEMA = "hawking.agentos.flash_vocabulary_frequency_evidence.v1"
SCRIPT_EVIDENCE_SCHEMA = "hawking.agentos.flash_vocabulary_script_evidence.v1"
ROW_UNIVERSE_SCHEMA = "tokenizer_lexical_model_vocab_rows_only.v2"
ROW_UNIVERSE_TAIL_POLICY = (
    "exclude_tokenizer_unaddressable_tensor_tail_until_output_semantics_are_verified.v1"
)
ROW_UNIVERSE_ADDED_TOKEN_POLICY = (
    "exclude_all_tokenizer_added_ids_from_E13_lexical_strata_regardless_of_special_flag.v1"
)
TOKENIZER_TENSOR_PARTITION_SCHEMA = "flash_tokenizer_tensor_partition.v1"
TENSOR_ACCOUNTING_POLICY = "retain_all_declared_tensor_rows_in_complete_artifact_and_byte_accounting.v1"
RAW_LOGIT_POLICY = "preserve_full_declared_tensor_row_axis_for_source_correctness.v1"
TAIL_SAMPLING_POLICY = "withhold_until_admitted_source_or_runtime_control_defines_tail_sampling.v1"
SPLIT_PRESERVING_NULL_ALGORITHM = (
    "within_stratum_split_preserving_token_label_permutation.v2"
)
TOKENIZER_SCRIPT_CLASSIFIER_ALGORITHM = "tokenizer_decoded_text_primary_script.v1"
RUST_E13_PRODUCER_SCHEMA = "hawking.gravity.flash_e13_evidence_producer.v1"
RUST_E13_PRODUCER_RUNTIME = "hawking-core Rust tokenizer; CPU only"
RUST_E13_PRODUCER_REL = Path(
    "crates/hawking-core/src/flash_e13_vocabulary_evidence.rs"
)
DEFAULT_EMIT_NAME = "FLASH_E13_VOCABULARY_PREFLIGHT.json"
DEFAULT_ROW_BUDGET = 4096
MAX_ROW_BUDGET = DEFAULT_ROW_BUDGET
MAX_PLAN_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 1 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
MAX_E13_HEADER_BYTES = 1 * 1024 * 1024
RECEIPT_SEAL_FORMAT = "python_json_compact_utf8_sorted_v1"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_NAME = "tokenizer.json"
INPUT_EMBEDDING = "model.language_model.embed_tokens.weight"
OUTPUT_HEAD = "lm_head.weight"
PINNED_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
PINNED_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
PINNED_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
PINNED_TOKENIZER_MODEL_VOCAB_ENTRY_COUNT = 248044
PINNED_TOKENIZER_ADDED_TOKEN_ENTRY_COUNT = 33
PINNED_TOKENIZER_ADDRESSABLE_TOKEN_COUNT = 248077
PINNED_ROLE_HEADERS = {
    "input_embedding": {
        "shard_name": "model-00130-of-00131.safetensors",
        "shard_size": 2974788136,
        "header_bytes": 1568,
        "dtype": "BF16",
        "shape": [248320, 2560],
        "data_offsets": [0, 1271398400],
        "payload_bytes": 1271398400,
    },
    "output_head": {
        "shard_name": "model-00131-of-00131.safetensors",
        "shard_size": 1271398528,
        "header_bytes": 120,
        "dtype": "BF16",
        "shape": [248320, 2560],
        "data_offsets": [0, 1271398400],
        "payload_bytes": 1271398400,
    },
}


class _Withheld(ValueError):
    """A precondition is absent or invalid; no experiment may proceed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    """The declared compact UTF-8 seal, compatible with lab.receipts.

    The repository's lab receipt owner uses this exact sorted, compact UTF-8
    encoding.  E13 keeps the implementation local because HAWKING's installed
    package does not depend on the repository-only ``research/lab`` source
    root; the format is named and independently verified below rather than
    being mistaken for the historical default-ASCII Python seal.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _seal_receipt(value: Mapping[str, Any]) -> Dict[str, Any]:
    body = dict(value)
    body.pop("seal_sha256", None)
    body["seal_format"] = RECEIPT_SEAL_FORMAT
    return {
        **body,
        "seal_sha256": _sha256_bytes(_canonical_receipt_bytes(body)),
    }


def verify_flash_vocabulary_prefilter_receipt(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify an E13 receipt under its declared compact UTF-8 seal format."""
    if not isinstance(value, Mapping):
        raise ValueError("E13 receipt must be a JSON object")
    if value.get("schema") != SCHEMA:
        raise ValueError(f"E13 receipt schema must be {SCHEMA!r}")
    if value.get("seal_format") != RECEIPT_SEAL_FORMAT:
        raise ValueError(
            f"E13 receipt seal_format must be {RECEIPT_SEAL_FORMAT!r}"
        )
    recorded = value.get("seal_sha256")
    expected = _seal_receipt(value)["seal_sha256"]
    if not isinstance(recorded, str) or recorded != expected:
        raise ValueError(
            f"E13 receipt seal mismatch: recorded={recorded!r} expected={expected}"
        )
    return dict(value)


def verify_flash_vocabulary_prefilter_receipt_path(
        path: str | os.PathLike[str],
) -> Dict[str, Any]:
    """Load and verify a bounded on-disk E13 receipt without source access."""
    raw = _read_bounded_bytes(
        Path(path), maximum=MAX_PLAN_BYTES, label="E13 vocabulary preflight receipt"
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("E13 receipt is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("E13 receipt must be a JSON object")
    return verify_flash_vocabulary_prefilter_receipt(value)


def _producer_identity() -> Dict[str, Any]:
    """Bind the receipt to the source closure enforcing E13's read boundary."""
    path = Path(__file__).resolve()
    repo_root = path.parents[2]
    producer_module = "hawking.agentos.flash_vocabulary_prefilter"
    closure_paths = {
        producer_module: path,
        "hawking.agentos.flash_tensor_probe": repo_root / "hawking" / "agentos" / "flash_tensor_probe.py",
        "tools.odyssey.specimen_open": repo_root / "tools" / "odyssey" / "specimen_open.py",
    }
    try:
        source_closure = {
            module: {
                "path": str(source_path),
                "sha256": _sha256_bytes(source_path.read_bytes()),
            }
            for module, source_path in closure_paths.items()
        }
    except OSError as exc:
        raise RuntimeError(f"cannot bind E13 producer source closure: {path}: {exc}") from exc
    return {
        # ``__name__`` becomes ``__main__`` for the supported CLI entry point.
        # Receipts must retain the importable canonical owner identity in both
        # library and command-line execution.
        "module": producer_module,
        "module_path": str(path),
        "module_sha256": source_closure[producer_module]["sha256"],
        "source_closure": source_closure,
        "python_executable": sys.executable,
    }


def _open_at_no_follow(path: Path, *, allowed_root: Optional[Path], label: str) -> int:
    """Open one regular file beneath an optional opened root without following links.

    A ``Path.is_symlink`` check alone is racy and cannot distinguish a hard
    link to a small safetensors shard.  This uses nofollow descriptors and
    directory-fd traversal so the object read is the object that was checked.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise _Withheld(
            "NOFOLLOW_OPEN_UNAVAILABLE",
            f"{label} cannot be opened safely on this platform",
        )
    file_flags = os.O_RDONLY | nofollow
    target = Path(os.path.abspath(os.fspath(path)))
    if allowed_root is None:
        try:
            return os.open(target, file_flags)
        except OSError as exc:
            code = "SYMLINK_OPEN_REFUSED" if exc.errno == errno.ELOOP else "SOURCE_METADATA_UNAVAILABLE"
            raise _Withheld(code, f"cannot safely open {label}: {target}") from exc

    root = Path(os.path.abspath(os.fspath(allowed_root)))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise _Withheld(
            "SOURCE_METADATA_OUTSIDE_ROOT",
            f"{label} is outside its allowed root: {target}",
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _Withheld("SOURCE_METADATA_OUTSIDE_ROOT", f"unsafe {label} path: {target}")
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    directory_flags = os.O_RDONLY | nofollow | directory_flag
    directory_fd: Optional[int] = None
    try:
        directory_fd = os.open(root, directory_flags)
        root_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise _Withheld("SOURCE_METADATA_UNAVAILABLE", f"allowed root is not a directory: {root}")
        for component in relative.parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise _Withheld("SOURCE_METADATA_UNAVAILABLE", f"{label} parent is not a directory")
        return os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
    except _Withheld:
        raise
    except OSError as exc:
        code = "SYMLINK_OPEN_REFUSED" if exc.errno == errno.ELOOP else "SOURCE_METADATA_UNAVAILABLE"
        raise _Withheld(code, f"cannot safely open {label}: {target}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _read_fd_up_to(fd: int, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum
    while remaining:
        chunk = os.read(fd, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_bounded_bytes(
        path: Path, *, maximum: int, label: str, allowed_root: Optional[Path] = None,
) -> bytes:
    """Read a bounded JSON artifact without accepting link or tensor-body swaps.

    A safetensors file begins with a little-endian header length.  Recognising
    that valid prefix before reading the remainder prevents a small or
    hard-linked tensor body disguised as ``*.json`` from being consumed.
    """
    fd = _open_at_no_follow(path, allowed_root=allowed_root, label=label)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _Withheld("REGULAR_FILE_REQUIRED", f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise _Withheld(
                "HARDLINKED_INPUT_REFUSED",
                f"{label} must not be a hard-linked artifact (nlink={before.st_nlink})",
            )
        if before.st_size > maximum:
            raise _Withheld(
                "INPUT_TOO_LARGE",
                f"{label} is {before.st_size} bytes, above its bounded limit of {maximum} bytes",
            )
        prefix = _read_fd_up_to(fd, maximum=min(8, before.st_size))
        if len(prefix) == 8:
            declared_header_bytes = int.from_bytes(prefix, "little")
            if 0 < declared_header_bytes <= before.st_size - 8:
                raise _Withheld(
                    "SAFETENSORS_INPUT_REFUSED",
                    f"{label} has a safetensors header prefix and is not a JSON control artifact",
                )
        os.lseek(fd, 0, os.SEEK_SET)
        raw = _read_fd_up_to(fd, maximum=maximum + 1)
        after = os.fstat(fd)
    except OSError as exc:
        raise _Withheld("SOURCE_METADATA_UNAVAILABLE", f"cannot read {label}: {path}") from exc
    finally:
        os.close(fd)
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise _Withheld("SOURCE_METADATA_CHANGED_DURING_READ", f"{label} changed during its bounded read")
    if len(raw) > maximum:
        raise _Withheld("INPUT_TOO_LARGE", f"{label} grew beyond its bounded limit")
    return raw


def _sha256_file(
        path: Path, *, maximum: int, label: str, allowed_root: Optional[Path] = None,
) -> str:
    return _sha256_bytes(
        _read_bounded_bytes(path, maximum=maximum, label=label, allowed_root=allowed_root)
    )


def _read_json(
        path: Path, *, maximum: int, label: str, allowed_root: Optional[Path] = None,
) -> Dict[str, Any]:
    raw = _read_bounded_bytes(
        path, maximum=maximum, label=label, allowed_root=allowed_root
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _Withheld("INVALID_JSON", f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise _Withheld("INVALID_JSON", f"{label} must be a JSON object")
    return value


def _safe_metadata_file(root: Path, name: str, *, maximum: int, label: str) -> Path:
    candidate = root / name
    if candidate.is_symlink():
        raise _Withheld("SOURCE_METADATA_SYMLINK_REFUSED", f"{label} may not be a symlink")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise _Withheld("SOURCE_METADATA_OUTSIDE_ROOT", f"{label} resolves outside the selected specimen") from exc
    if not candidate.is_file():
        raise _Withheld("SOURCE_METADATA_UNAVAILABLE", f"{label} is unavailable: {candidate}")
    _read_bounded_bytes(candidate, maximum=maximum, label=label, allowed_root=root)
    return candidate


def _manifest_identity(root: Path) -> Dict[str, Any]:
    """Bind the source root to the canonical pinned ModelLake manifest."""
    # Import the canonical location lazily so synthetic tests can monkeypatch
    # flash_tensor_probe.LAKE_ROOT without a duplicate source-owner constant.
    from hawking.agentos import flash_tensor_probe

    lake_root = flash_tensor_probe.LAKE_ROOT.resolve()
    manifest_path = lake_root / "manifests" / f"{flash_tensor_probe.LAKE_SLUG}.json"
    if manifest_path.is_symlink():
        raise _Withheld("PINNED_MANIFEST_SYMLINK_REFUSED", "ModelLake manifest may not be a symlink")
    try:
        manifest_path = manifest_path.resolve()
        manifest_path.relative_to(lake_root)
    except ValueError as exc:
        raise _Withheld("PINNED_MANIFEST_OUTSIDE_LAKE", "ModelLake manifest resolves outside its lake") from exc
    raw_manifest = _read_bounded_bytes(
        manifest_path,
        maximum=MAX_PLAN_BYTES,
        label="ModelLake manifest",
        allowed_root=lake_root,
    )
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _Withheld("INVALID_JSON", "ModelLake manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise _Withheld("INVALID_JSON", "ModelLake manifest must be a JSON object")
    if (
        manifest.get("repo") != REPO_ID
        or manifest.get("revision") != PINNED_REVISION
        or manifest.get("resolved_sha") != PINNED_REVISION
    ):
        raise _Withheld(
            "PINNED_SOURCE_IDENTITY_MISMATCH",
            "ModelLake manifest does not bind this preflight to the pinned Flash-Next revision",
        )
    declared_path = str(manifest.get("path") or "").strip()
    if not declared_path:
        raise _Withheld(
            "PINNED_SOURCE_PATH_MISSING",
            "ModelLake manifest must name the selected specimen root",
        )
    if Path(declared_path).expanduser().resolve() != root:
        raise _Withheld(
            "PINNED_SOURCE_PATH_MISMATCH",
            "selected specimen root does not match the pinned ModelLake manifest path",
        )
    return {
        "path": str(manifest_path),
        "sha256": _sha256_bytes(raw_manifest),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "resolved_sha": manifest.get("resolved_sha"),
        "bytes": manifest.get("bytes"),
        "n_files": manifest.get("n_files"),
        "n_sha256_verified": manifest.get("n_sha256_verified"),
        "n_size_only_verified": manifest.get("n_size_only_verified"),
    }


def _integer(value: Any, *, field: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise _Withheld("INVALID_CONTROL_PLAN", f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise _Withheld("INVALID_CONTROL_PLAN", f"{field} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise _Withheld("INVALID_CONTROL_PLAN", f"{field} must be >= {minimum}")
    return parsed


def _sha256_text(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise _Withheld("INVALID_CONTROL_PLAN", f"{field} must be a SHA-256 hex digest")
    return text


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Withheld("INVALID_CONTROL_PLAN", f"{field} must be an object")
    return value


def _role_headers(root: Path, *, vocab_size: int, hidden_size: int,
                  tied: bool) -> Dict[str, Dict[str, Any]]:
    if tied:
        raise _Withheld(
            "TIED_EMBEDDING_HEAD_UNSUPPORTED",
            "E13 requires independently declared input-embedding and output-head roles",
        )
    try:
        embedding = load_tensor_header(
            root, INPUT_EMBEDDING, max_header_bytes=MAX_E13_HEADER_BYTES
        )
        head = load_tensor_header(
            root, OUTPUT_HEAD, max_header_bytes=MAX_E13_HEADER_BYTES
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise _Withheld("VOCABULARY_HEADER_UNAVAILABLE", str(exc)) from exc
    expected_shape = [vocab_size, hidden_size]
    for role, header in (("input_embedding", embedding), ("output_head", head)):
        if str(header.get("dtype") or "").upper() != "BF16":
            raise _Withheld("VOCABULARY_HEADER_DTYPE_MISMATCH", f"{role} is not BF16")
        if list(header.get("shape") or []) != expected_shape:
            raise _Withheld(
                "VOCABULARY_HEADER_SHAPE_MISMATCH",
                f"{role} shape {header.get('shape')} != declared vocabulary geometry {expected_shape}",
            )
        if int(header.get("payload_bytes") or 0) != vocab_size * hidden_size * 2:
            raise _Withheld(
                "VOCABULARY_HEADER_PAYLOAD_MISMATCH",
                f"{role} payload does not match BF16 vocabulary geometry",
            )
    if (
        embedding.get("shard") == head.get("shard")
        and embedding.get("data_offsets") == head.get("data_offsets")
    ):
        raise _Withheld(
            "VOCABULARY_ROLES_NOT_INDEPENDENT",
            "untied config points input embedding and output head to the same payload",
        )

    def public(header: Mapping[str, Any]) -> Dict[str, Any]:
        value = {
            key: header.get(key)
            for key in (
                "tensor_name", "shard_name", "shard_size", "header_bytes",
                "dtype", "shape", "data_offsets", "payload_bytes", "index_sha256",
                "index_bytes", "header_read_bytes", "tensor_payload_bytes_read",
            )
        }
        return value

    result = {"input_embedding": public(embedding), "output_head": public(head)}
    for role, expected in PINNED_ROLE_HEADERS.items():
        observed = result[role]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise _Withheld(
                "PINNED_ROLE_HEADER_MISMATCH",
                f"{role} header does not match the pinned Flash-Next vocabulary role",
            )
    return result


def _tokenizer_addressability(
        tokenizer_path: Path, *, metadata_root: Path,
) -> Dict[str, Any]:
    """Map every exact tokenizer surface to its addressable token id.

    The language tensors can be padded beyond the tokenizer's contiguous id
    range.  That is not automatically an error: a padded output row may be
    masked by the source implementation, or it may require a separate output
    policy.  E13 is a token-structure test, however, so it must never silently
    treat a tensor-only row as a corpus-tokenized surface.  This metadata-only
    check records that distinction before a plan is allowed to select rows.
    """
    tokenizer = _read_json(
        tokenizer_path,
        maximum=MAX_TOKENIZER_BYTES,
        label="tokenizer",
        allowed_root=metadata_root,
    )
    model = tokenizer.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("vocab"), Mapping):
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", "tokenizer model.vocab is unavailable")

    by_id: Dict[int, str] = {}
    model_ids: set[int] = set()
    added_ids: set[int] = set()

    def add(token_id: Any, surface: Any, *, field: str, target: set[int]) -> None:
        parsed_id = _integer(token_id, field=field, minimum=0)
        if not isinstance(surface, str):
            raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", f"{field} has a non-text token surface")
        if parsed_id in by_id:
            raise _Withheld(
                "TOKENIZER_SURFACE_AMBIGUOUS",
                f"tokenizer names token id {parsed_id} more than once",
            )
        by_id[parsed_id] = surface
        target.add(parsed_id)

    for surface, token_id in model["vocab"].items():
        add(token_id, surface, field="tokenizer.model.vocab", target=model_ids)
    added_tokens = tokenizer.get("added_tokens") or []
    if not isinstance(added_tokens, list):
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", "tokenizer added_tokens is not an array")
    for position, row in enumerate(added_tokens):
        if not isinstance(row, Mapping):
            raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", f"tokenizer added_tokens[{position}] is invalid")
        add(
            row.get("id"),
            row.get("content"),
            field=f"tokenizer.added_tokens[{position}]",
            target=added_ids,
        )

    if not by_id:
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", "tokenizer has no addressable token ids")
    maximum_id = max(by_id)
    expected = set(range(maximum_id + 1))
    observed = set(by_id)
    if observed != expected:
        missing = sorted(expected - observed)
        raise _Withheld(
            "TOKENIZER_ADDRESSABILITY_GAPPED",
            "tokenizer addressable ids must be contiguous from zero; "
            f"missing ids include {missing[:8]}",
        )
    expected_model_ids = set(range(len(model_ids)))
    if model_ids != expected_model_ids:
        raise _Withheld(
            "TOKENIZER_MODEL_VOCAB_NOT_CONTIGUOUS",
            "tokenizer model-vocab ids must be contiguous from zero",
        )
    expected_added_ids = set(range(len(model_ids), len(by_id)))
    if added_ids != expected_added_ids:
        raise _Withheld(
            "TOKENIZER_ADDED_TOKEN_INTERVAL_MISMATCH",
            "tokenizer added/control ids must immediately follow the lexical model-vocab interval",
        )

    def inclusive_range(ids: set[int]) -> Optional[list[int]]:
        return [min(ids), max(ids)] if ids else None

    return {
        "addressable_token_count": len(by_id),
        "addressable_token_id_range": [0, maximum_id],
        "addressable_ids_contiguous_from_zero": True,
        "tokenizer_model_vocab_entry_count": len(model_ids),
        "tokenizer_model_vocab_id_range": inclusive_range(model_ids),
        "tokenizer_model_vocab_ids_contiguous_from_zero": True,
        "tokenizer_added_token_entry_count": len(added_ids),
        "tokenizer_added_token_id_range": inclusive_range(added_ids),
        "tokenizer_added_token_ids_contiguous_after_model_vocab": True,
        "tokenizer_added_token_ids": sorted(added_ids),
        "tokenizer_added_tokens_sha256": _sha256_text(added_tokens),
    }


def _source_identity(root: Path) -> Dict[str, Any]:
    config_path = _safe_metadata_file(
        root, CONFIG_NAME, maximum=MAX_CONFIG_BYTES, label="Flash config"
    )
    index_path = _safe_metadata_file(
        root, INDEX_NAME, maximum=MAX_INDEX_BYTES, label="safetensors index"
    )
    tokenizer_path = _safe_metadata_file(
        root, TOKENIZER_NAME, maximum=MAX_TOKENIZER_BYTES, label="tokenizer"
    )
    config = _read_json(
        config_path,
        maximum=MAX_CONFIG_BYTES,
        label="Flash config",
        allowed_root=root,
    )
    text_config = _mapping(config.get("text_config"), field="config.text_config")
    vocab_size = _integer(text_config.get("vocab_size"), field="config.text_config.vocab_size", minimum=1)
    hidden_size = _integer(text_config.get("hidden_size"), field="config.text_config.hidden_size", minimum=1)
    tied = bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    hashes = {
        "config_sha256": _sha256_file(
            config_path, maximum=MAX_CONFIG_BYTES, label="Flash config", allowed_root=root
        ),
        "index_sha256": _sha256_file(
            index_path, maximum=MAX_INDEX_BYTES, label="safetensors index", allowed_root=root
        ),
        "tokenizer_sha256": _sha256_file(
            tokenizer_path, maximum=MAX_TOKENIZER_BYTES, label="tokenizer", allowed_root=root
        ),
    }
    expected_hashes = {
        "config_sha256": PINNED_CONFIG_SHA256,
        "index_sha256": PINNED_INDEX_SHA256,
        "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
    }
    for field, expected in expected_hashes.items():
        if hashes[field] != expected:
            raise _Withheld(
                "PINNED_SOURCE_METADATA_MISMATCH",
                f"{field} does not match the pinned Flash-Next source metadata",
            )
    headers = _role_headers(
        root, vocab_size=vocab_size, hidden_size=hidden_size, tied=tied
    )
    addressability = _tokenizer_addressability(tokenizer_path, metadata_root=root)
    expected_addressability = {
        "tokenizer_model_vocab_entry_count": PINNED_TOKENIZER_MODEL_VOCAB_ENTRY_COUNT,
        "tokenizer_added_token_entry_count": PINNED_TOKENIZER_ADDED_TOKEN_ENTRY_COUNT,
        "addressable_token_count": PINNED_TOKENIZER_ADDRESSABLE_TOKEN_COUNT,
    }
    for field, expected in expected_addressability.items():
        if addressability[field] != expected:
            raise _Withheld(
                "PINNED_TOKENIZER_ADDRESSABILITY_MISMATCH",
                f"{field} {addressability[field]} does not match pinned Flash tokenizer {expected}",
            )
    addressable_token_count = _integer(
        addressability["addressable_token_count"],
        field="tokenizer addressable_token_count",
        minimum=1,
    )
    if addressable_token_count > vocab_size:
        raise _Withheld(
            "TOKENIZER_EXCEEDS_TENSOR_ROW_COUNT",
            "tokenizer addressable ids extend beyond the embedding/head tensor row count",
        )
    unaddressable_tensor_rows = vocab_size - addressable_token_count
    tensor_tail_range = (
        [addressable_token_count, vocab_size - 1]
        if unaddressable_tensor_rows
        else None
    )
    tokenizer_tensor_partition = {
        "schema": TOKENIZER_TENSOR_PARTITION_SCHEMA,
        "lexical_model_vocab": {
            "count": addressability["tokenizer_model_vocab_entry_count"],
            "id_range": addressability["tokenizer_model_vocab_id_range"],
            "e13_lexical_eligible": True,
        },
        "added_control": {
            "count": addressability["tokenizer_added_token_entry_count"],
            "id_range": addressability["tokenizer_added_token_id_range"],
            "ids_sha256": _sha256_text(addressability["tokenizer_added_token_ids"]),
            "metadata_sha256": addressability["tokenizer_added_tokens_sha256"],
            "e13_lexical_eligible": False,
        },
        "tokenizer_addressable": {
            "count": addressable_token_count,
            "id_range": addressability["addressable_token_id_range"],
            "input_addressable": True,
        },
        "tensor_only_tail": {
            "count": unaddressable_tensor_rows,
            "id_range": tensor_tail_range,
            "tokenizer_input_addressable": False,
            "e13_lexical_eligible": False,
            "output_sampling_policy": TAIL_SAMPLING_POLICY,
        },
        "tensor_rows": {
            "count": vocab_size,
            "id_range": [0, vocab_size - 1],
            "complete_artifact_accounting_policy": TENSOR_ACCOUNTING_POLICY,
            "raw_source_logit_policy": RAW_LOGIT_POLICY,
        },
    }
    reread_hashes = {
        "config_sha256": _sha256_file(
            config_path, maximum=MAX_CONFIG_BYTES, label="Flash config", allowed_root=root
        ),
        "index_sha256": _sha256_file(
            index_path, maximum=MAX_INDEX_BYTES, label="safetensors index", allowed_root=root
        ),
        "tokenizer_sha256": _sha256_file(
            tokenizer_path, maximum=MAX_TOKENIZER_BYTES, label="tokenizer", allowed_root=root
        ),
    }
    if reread_hashes != hashes:
        raise _Withheld(
            "SOURCE_METADATA_CHANGED_DURING_PREFLIGHT",
            "Flash metadata changed while its vocabulary headers were being checked",
        )
    return {
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(root),
        **hashes,
        "role_headers_sha256": _sha256_text(headers),
        "tokenizer_addressability_sha256": _sha256_text(addressability),
        "vocabulary_geometry": {
            "vocab_size": vocab_size,
            "tensor_row_count": vocab_size,
            "hidden_size": hidden_size,
            "tie_word_embeddings": tied,
            "config_path": str(config_path),
            "index_path": str(index_path),
            "tokenizer_path": str(tokenizer_path),
            **addressability,
            "e13_eligible_model_vocab_token_count": addressability["tokenizer_model_vocab_entry_count"],
            "e13_excluded_added_control_token_count": addressability["tokenizer_added_token_entry_count"],
            "e13_excluded_added_control_token_ids_sha256": _sha256_text(
                addressability["tokenizer_added_token_ids"]
            ),
            "tensor_rows_without_tokenizer_surface": unaddressable_tensor_rows,
            "tensor_only_tail_id_range": tensor_tail_range,
            "tensor_row_tail_status": (
                "NO_TENSOR_ONLY_ROWS"
                if unaddressable_tensor_rows == 0
                else "TENSOR_PADDING_INTERVAL_IDENTIFIED__OUTPUT_SAMPLING_POLICY_WITHHELD"
            ),
            "tokenizer_tensor_partition": tokenizer_tensor_partition,
        },
        "roles": headers,
    }


def _selected_ids(
        strata: Sequence[Mapping[str, Any]], *, vocab_size: int, row_budget: int,
        addressable_token_count: Optional[int] = None,
        ineligible_token_ids: Optional[set[int]] = None,
) -> Tuple[Dict[str, Dict[str, set[int]]], list[int], list[Dict[str, Any]]]:
    if not isinstance(strata, list) or len(strata) < 8:
        raise _Withheld(
            "INSUFFICIENT_STRATA",
            "control plan needs at least a 2-by-2-by-2 frequency/script/byte-length set of strata",
        )
    by_stratum: Dict[str, Dict[str, set[int]]] = {}
    all_ids: list[int] = []
    summaries: list[Dict[str, Any]] = []
    bands: set[str] = set()
    scripts: set[str] = set()
    byte_length_bands: set[str] = set()
    frequency_script_pairs: set[tuple[str, str]] = set()
    frequency_byte_pairs: set[tuple[str, str]] = set()
    script_byte_pairs: set[tuple[str, str]] = set()
    factor_cells: set[tuple[str, str, str]] = set()
    for position, raw in enumerate(strata):
        row = _mapping(raw, field=f"strata[{position}]")
        identifier = str(row.get("id") or "").strip()
        band = str(row.get("frequency_band") or "").strip()
        script = str(row.get("script") or "").strip()
        byte_length_band = str(row.get("byte_length_band") or "").strip()
        if not identifier or not band or not script or not byte_length_band:
            raise _Withheld(
                "INVALID_CONTROL_PLAN",
                f"strata[{position}] needs id, frequency_band, script, and byte_length_band",
            )
        if identifier in by_stratum:
            raise _Withheld("DUPLICATE_STRATUM", f"duplicate stratum id {identifier!r}")
        frequency_range = _mapping(row.get("frequency_range"), field=f"strata[{position}].frequency_range")
        floor = _integer(frequency_range.get("min_inclusive"), field=f"strata[{position}].frequency_range.min_inclusive", minimum=1)
        ceiling = _integer(frequency_range.get("max_inclusive"), field=f"strata[{position}].frequency_range.max_inclusive", minimum=floor)
        byte_length_range = _mapping(row.get("byte_length_range"), field=f"strata[{position}].byte_length_range")
        byte_floor = _integer(
            byte_length_range.get("min_inclusive"),
            field=f"strata[{position}].byte_length_range.min_inclusive",
            minimum=1,
        )
        byte_ceiling = _integer(
            byte_length_range.get("max_inclusive"),
            field=f"strata[{position}].byte_length_range.max_inclusive",
            minimum=byte_floor,
        )
        fit_raw = row.get("fit_token_ids")
        heldout_raw = row.get("heldout_token_ids")
        if not isinstance(fit_raw, list) or not isinstance(heldout_raw, list) or not fit_raw or not heldout_raw:
            raise _Withheld("EMPTY_FIT_OR_HELDOUT", f"stratum {identifier!r} needs nonempty fit and heldout ids")
        fit = [_integer(value, field=f"{identifier}.fit_token_ids", minimum=0) for value in fit_raw]
        heldout = [_integer(value, field=f"{identifier}.heldout_token_ids", minimum=0) for value in heldout_raw]
        ids = fit + heldout
        if any(token_id >= vocab_size for token_id in ids):
            raise _Withheld("TOKEN_ID_OUT_OF_RANGE", f"stratum {identifier!r} names a token outside vocabulary")
        if addressable_token_count is not None and any(token_id >= addressable_token_count for token_id in ids):
            raise _Withheld(
                "TOKEN_ID_NOT_ADDRESSABLE_BY_TOKENIZER",
                f"stratum {identifier!r} selects a tensor row without an exact tokenizer surface",
            )
        if ineligible_token_ids and any(token_id in ineligible_token_ids for token_id in ids):
            raise _Withheld(
                "TOKEN_ID_NOT_ELIGIBLE_FOR_E13_STRATA",
                f"stratum {identifier!r} selects a tokenizer added/control id, which is excluded from E13 lexical strata",
            )
        if len(set(ids)) != len(ids):
            raise _Withheld("OVERLAPPING_FIT_HELDOUT", f"stratum {identifier!r} overlaps fit and heldout ids")
        if vocab_size == 248320 and (len(fit) < 64 or len(heldout) < 64):
            raise _Withheld(
                "INSUFFICIENT_PRODUCTION_SPLIT",
                f"pinned E13 stratum {identifier!r} needs at least 64 fit and 64 heldout ids",
            )
        by_stratum[identifier] = {"fit": set(fit), "heldout": set(heldout)}
        all_ids.extend(ids)
        bands.add(band)
        scripts.add(script)
        byte_length_bands.add(byte_length_band)
        frequency_script_pairs.add((band, script))
        frequency_byte_pairs.add((band, byte_length_band))
        script_byte_pairs.add((script, byte_length_band))
        factor_cells.add((band, script, byte_length_band))
        summaries.append({
            "id": identifier,
            "frequency_band": band,
            "frequency_range": {"min_inclusive": floor, "max_inclusive": ceiling},
            "script": script,
            "byte_length_band": byte_length_band,
            "byte_length_range": {"min_inclusive": byte_floor, "max_inclusive": byte_ceiling},
            "fit_count": len(fit),
            "heldout_count": len(heldout),
        })
    if len(set(all_ids)) != len(all_ids):
        raise _Withheld("OVERLAPPING_STRATA", "a token id appears in more than one stratum")
    if len(all_ids) != row_budget:
        raise _Withheld(
            "ROW_BUDGET_MISMATCH",
            f"strata select {len(all_ids)} rows, but predeclared row budget is {row_budget}",
        )
    if len(bands) < 2 or len(scripts) < 2 or len(byte_length_bands) < 2:
        raise _Withheld(
            "INSUFFICIENT_FREQUENCY_OR_SCRIPT_STRATA",
            "control plan must cover at least two frequency bands, scripts, and byte-length bands",
        )
    if any(sum(pair[0] == band for pair in frequency_script_pairs) < 2 for band in bands) or any(
        sum(pair[1] == script for pair in frequency_script_pairs) < 2 for script in scripts
    ):
        raise _Withheld(
            "CONFOUNDED_FREQUENCY_SCRIPT_STRATA",
            "each frequency band and script must occur with at least two values of the other factor",
        )
    if any(sum(pair[0] == band for pair in frequency_byte_pairs) < 2 for band in bands) or any(
        sum(pair[1] == byte_band for pair in frequency_byte_pairs) < 2 for byte_band in byte_length_bands
    ) or any(sum(pair[0] == script for pair in script_byte_pairs) < 2 for script in scripts) or any(
        sum(pair[1] == byte_band for pair in script_byte_pairs) < 2 for byte_band in byte_length_bands
    ):
        raise _Withheld(
            "CONFOUNDED_BYTE_LENGTH_STRATA",
            "byte length must cross both frequency and script factors",
        )
    if any(
        (band, script, byte_band) not in factor_cells
        for band in bands
        for script in scripts
        for byte_band in byte_length_bands
    ):
        raise _Withheld(
            "INCOMPLETE_FACTORIAL_STRATA",
            "every declared frequency/script/byte-length cell must have fit and heldout tokens",
        )
    return by_stratum, all_ids, summaries


def _control_artifact_path(
        value: Any, *, field: str, control_root: Path,
) -> Path:
    raw_path = str(value or "").strip()
    if not raw_path:
        raise _Withheld("MISSING_CONTROL_ARTIFACT", f"{field} is required")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = control_root / candidate
    candidate = Path(os.path.abspath(str(candidate)))
    if candidate.suffix != ".json":
        raise _Withheld("CONTROL_ARTIFACT_EXTENSION_REFUSED", f"{field} must name a JSON control artifact")
    try:
        relative = candidate.relative_to(control_root)
    except ValueError as exc:
        raise _Withheld(
            "CONTROL_ARTIFACT_OUTSIDE_ROOT",
            f"{field} must stay under {control_root}",
        ) from exc
    cursor = control_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise _Withheld("CONTROL_ARTIFACT_SYMLINK_REFUSED", f"{field} may not traverse a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(control_root)
    except ValueError as exc:
        raise _Withheld("CONTROL_ARTIFACT_OUTSIDE_ROOT", f"{field} resolves outside control root") from exc
    if not resolved.is_file():
        raise _Withheld("CONTROL_ARTIFACT_UNAVAILABLE", f"{field} is unavailable: {resolved}")
    return candidate


def _evidence_artifact(
        plan: Mapping[str, Any], *, field: str, schema: str,
        expected_provenance: str, expected_tokenizer_sha256: str,
        control_root: Path,
) -> tuple[Mapping[str, Any], Dict[str, Any]]:
    declaration = _mapping(plan.get(field), field=field)
    if "rows" in declaration:
        raise _Withheld(
            "INLINE_EVIDENCE_REFUSED",
            f"{field}.rows must live in an independently hashed evidence artifact",
        )
    path = _control_artifact_path(
        declaration.get("artifact_path"),
        field=f"{field}.artifact_path",
        control_root=control_root,
    )
    raw = _read_bounded_bytes(
        path,
        maximum=MAX_PLAN_BYTES,
        label=f"{field} artifact",
        allowed_root=control_root,
    )
    expected_artifact_sha = _require_sha256(
        declaration.get("artifact_sha256"), field=f"{field}.artifact_sha256"
    )
    observed_artifact_sha = _sha256_bytes(raw)
    if observed_artifact_sha != expected_artifact_sha:
        raise _Withheld(
            "EVIDENCE_ARTIFACT_HASH_MISMATCH",
            f"{field} artifact does not match its declared digest",
        )
    try:
        evidence = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _Withheld("INVALID_EVIDENCE_ARTIFACT", f"{field} artifact is not valid JSON") from exc
    evidence = _mapping(evidence, field=f"{field} artifact")
    if evidence.get("schema") != schema:
        raise _Withheld("EVIDENCE_ARTIFACT_SCHEMA_MISMATCH", f"{field} artifact schema is not {schema}")
    if evidence.get("provenance") != expected_provenance:
        raise _Withheld(
            "UNACCEPTABLE_EVIDENCE_PROVENANCE",
            f"{field} artifact provenance must be {expected_provenance!r}; token-id/rank/range proxies are not evidence",
        )
    if str(evidence.get("tokenizer_sha256") or "").lower() != expected_tokenizer_sha256:
        raise _Withheld("TOKENIZER_BINDING_MISMATCH", f"{field} artifact is not bound to this tokenizer")
    if field == "frequency_evidence":
        _require_sha256(evidence.get("corpus_sha256"), field="frequency_evidence artifact corpus_sha256")
        corpus_token_count = _integer(
            evidence.get("corpus_token_count"),
            field="frequency_evidence artifact corpus_token_count",
            minimum=1,
        )
        rows = evidence.get("rows")
        if not isinstance(rows, list):
            raise _Withheld("INVALID_EVIDENCE_ARTIFACT", "frequency evidence artifact rows must be an array")
        selected_count_sum = 0
        for position, raw in enumerate(rows):
            row = _mapping(raw, field=f"frequency_evidence.rows[{position}]")
            count = _integer(
                row.get("count"),
                field=f"frequency_evidence.rows[{position}].count",
                minimum=1,
            )
            if count > corpus_token_count:
                raise _Withheld(
                    "FREQUENCY_EVIDENCE_COUNT_INCONSISTENT",
                    "a token count exceeds the declared corpus token count",
                )
            selected_count_sum += count
        if selected_count_sum > corpus_token_count:
            raise _Withheld(
                "FREQUENCY_EVIDENCE_COUNT_INCONSISTENT",
                "declared evidence row counts exceed the declared corpus token count",
            )
    else:
        classifier = _mapping(evidence.get("classifier"), field="script_evidence artifact classifier")
        if classifier.get("algorithm") != TOKENIZER_SCRIPT_CLASSIFIER_ALGORITHM:
            raise _Withheld(
                "SCRIPT_CLASSIFIER_ALGORITHM_MISMATCH",
                "script evidence must declare the canonical decoded-token-text classifier algorithm",
            )
        _require_sha256(
            classifier.get("implementation_sha256"),
            field="script_evidence artifact classifier.implementation_sha256",
        )
    return evidence, {
        "path": str(path),
        "sha256": observed_artifact_sha,
        "bytes": len(raw),
        "schema": schema,
        "provenance": expected_provenance,
        "tokenizer_sha256": expected_tokenizer_sha256,
    }


def _evidence_map(
        evidence: Mapping[str, Any], *, field: str, vocab_size: int,
        addressable_token_count: Optional[int] = None,
        ineligible_token_ids: Optional[set[int]] = None,
) -> Dict[int, Mapping[str, Any]]:
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise _Withheld("INVALID_EVIDENCE_ARTIFACT", f"{field} artifact rows must be an array")
    result: Dict[int, Mapping[str, Any]] = {}
    for position, raw in enumerate(rows):
        row = _mapping(raw, field=f"{field}.rows[{position}]")
        token_id = _integer(row.get("token_id"), field=f"{field}.rows[{position}].token_id", minimum=0)
        if token_id >= vocab_size:
            raise _Withheld("TOKEN_ID_OUT_OF_RANGE", f"{field} artifact has an out-of-range token id")
        if addressable_token_count is not None and token_id >= addressable_token_count:
            raise _Withheld(
                "EVIDENCE_TOKEN_NOT_ADDRESSABLE_BY_TOKENIZER",
                f"{field} artifact names tensor row {token_id} without an exact tokenizer surface",
            )
        if ineligible_token_ids and token_id in ineligible_token_ids:
            raise _Withheld(
                "EVIDENCE_TOKEN_NOT_ELIGIBLE_FOR_E13",
                f"{field} artifact names added/control token id {token_id}, excluded from E13 lexical evidence",
            )
        if token_id in result:
            raise _Withheld("DUPLICATE_EVIDENCE_TOKEN", f"{field} artifact repeats token id {token_id}")
        result[token_id] = row
    return result


def _admit_e13_evidence_producer(
        frequency_evidence: Mapping[str, Any],
        script_evidence: Mapping[str, Any],
        *,
        control_root: Path,
) -> Dict[str, Any]:
    """Bind both evidence families to the reviewed native producer source."""
    try:
        repo_root = control_root.parents[1]
    except IndexError as exc:
        raise _Withheld(
            "EVIDENCE_PRODUCER_SOURCE_UNAVAILABLE",
            "cannot resolve repository root from the E13 control root",
        ) from exc
    owner_path = repo_root / RUST_E13_PRODUCER_REL
    owner_sha256 = _sha256_file(
        owner_path,
        maximum=2 * 1024 * 1024,
        label="Rust E13 evidence producer",
        allowed_root=repo_root,
    )

    admitted: Optional[Dict[str, Any]] = None
    for field, evidence in (
        ("frequency_evidence", frequency_evidence),
        ("script_evidence", script_evidence),
    ):
        producer = dict(_mapping(evidence.get("producer"), field=f"{field}.producer"))
        expected = {
            "schema": RUST_E13_PRODUCER_SCHEMA,
            "implementation_sha256": owner_sha256,
            "runtime": RUST_E13_PRODUCER_RUNTIME,
            "model_loaded": False,
            "tensor_payload_bytes_read": 0,
            "gpu_session_started": False,
        }
        if producer != expected:
            raise _Withheld(
                "EVIDENCE_PRODUCER_IDENTITY_MISMATCH",
                f"{field} is not bound exactly to the admitted Rust E13 producer",
            )
        if admitted is None:
            admitted = producer
        elif producer != admitted:
            raise _Withheld(
                "EVIDENCE_PRODUCER_IDENTITY_MISMATCH",
                "frequency and script evidence name different producers",
            )
    classifier = _mapping(
        script_evidence.get("classifier"),
        field="script_evidence artifact classifier",
    )
    if str(classifier.get("implementation_sha256") or "").lower() != owner_sha256:
        raise _Withheld(
            "SCRIPT_CLASSIFIER_IMPLEMENTATION_MISMATCH",
            "decoded-script classifier is not bound to the admitted Rust producer source",
        )
    return {
        "status": "ADMITTED_CANONICAL_RUST_PRODUCER",
        "owner_path": str(owner_path),
        "owner_sha256": owner_sha256,
        "schema": RUST_E13_PRODUCER_SCHEMA,
        "runtime": RUST_E13_PRODUCER_RUNTIME,
        "model_loaded": False,
        "tensor_payload_bytes_read": 0,
        "gpu_session_started": False,
    }


def _selected_token_surfaces(
        tokenizer_path: Path, selected: Sequence[int], *, metadata_root: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    tokenizer = _read_json(
        tokenizer_path,
        maximum=MAX_TOKENIZER_BYTES,
        label="tokenizer",
        allowed_root=metadata_root,
    )
    model = tokenizer.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("vocab"), Mapping):
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", "tokenizer model.vocab is unavailable")
    wanted = set(selected)
    surfaces: Dict[int, str] = {}

    def add(token_id: Any, surface: Any, *, field: str) -> None:
        parsed_id = _integer(token_id, field=field, minimum=0)
        if parsed_id not in wanted:
            return
        if not isinstance(surface, str):
            raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", f"{field} has a non-text token surface")
        if parsed_id in surfaces:
            raise _Withheld("TOKENIZER_SURFACE_AMBIGUOUS", f"tokenizer names token id {parsed_id} more than once")
        surfaces[parsed_id] = surface

    for surface, token_id in model["vocab"].items():
        add(token_id, surface, field="tokenizer.model.vocab")
    added_tokens = tokenizer.get("added_tokens") or []
    if not isinstance(added_tokens, list):
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", "tokenizer added_tokens is not an array")
    for position, row in enumerate(added_tokens):
        if not isinstance(row, Mapping):
            raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", f"tokenizer added_tokens[{position}] is invalid")
        add(row.get("id"), row.get("content"), field=f"tokenizer.added_tokens[{position}]")
    missing = sorted(wanted - set(surfaces))
    if missing:
        raise _Withheld("TOKENIZER_SURFACE_UNAVAILABLE", f"tokenizer has no exact surface for selected token ids {missing[:8]}")
    return {
        token_id: {
            # tokenizer.json BPE strings are raw tokenizer wire surfaces.  They
            # are not necessarily decoded text: byte-level forms can render an
            # Arabic or Cyrillic token as apparent Latin codepoints.  Bind them
            # for provenance, but require the native producer to supply a
            # separately decoded-text script/length observation.
            "tokenizer_raw_surface_sha256": _sha256_bytes(surface.encode("utf-8")),
            "tokenizer_raw_surface_utf8_byte_length": len(surface.encode("utf-8")),
        }
        for token_id, surface in surfaces.items()
    }


def _derived_null_assignments(
        by_stratum: Mapping[str, Mapping[str, set[int]]], *, seed: int,
) -> list[Dict[str, Any]]:
    """Derive a seeded derangement without moving labels across fit/heldout.

    The old combined-stratum rotation could map a heldout token's label onto a
    fit token.  That makes a row split nominally disjoint while allowing the
    null itself to cross the split.  The v2 control derives one independent
    non-identity permutation per partition instead.
    """
    assignments: list[Dict[str, Any]] = []
    for identifier in sorted(by_stratum):
        partitions = by_stratum[identifier]
        for split in ("fit", "heldout"):
            token_ids = sorted(partitions[split])
            if len(token_ids) < 2:
                raise _Withheld(
                    "SHUFFLED_NULL_PARTITION_TOO_SMALL",
                    f"stratum {identifier!r} {split} partition needs at least two ids for a non-identity null",
                )
            ranked = sorted(
                token_ids,
                key=lambda token_id: _sha256_bytes(
                    f"{PLAN_SCHEMA}|{SPLIT_PRESERVING_NULL_ALGORITHM}|{seed}|{identifier}|{split}|{token_id}".encode("utf-8")
                ),
            )
            for position, token_id in enumerate(ranked):
                assignments.append({
                    "stratum_id": identifier,
                    "split": split,
                    "token_id": token_id,
                    "shuffled_token_id": ranked[(position + 1) % len(ranked)],
                })
    return sorted(
        assignments,
        key=lambda row: (str(row["stratum_id"]), str(row["split"]), int(row["token_id"])),
    )


def _validate_row_universe(
        plan: Mapping[str, Any], geometry: Mapping[str, Any], *, selected: Sequence[int],
) -> Dict[str, Any]:
    """Require a plan to name the exact tokenizer-addressable E13 row universe."""
    universe = _mapping(plan.get("row_universe"), field="row_universe")
    if universe.get("schema") != ROW_UNIVERSE_SCHEMA:
        raise _Withheld(
            "ROW_UNIVERSE_SCHEMA_MISMATCH",
            f"row_universe.schema must be {ROW_UNIVERSE_SCHEMA}",
        )
    tensor_rows = _integer(geometry.get("tensor_row_count"), field="tensor_row_count", minimum=1)
    addressable = _integer(
        geometry.get("addressable_token_count"),
        field="addressable_token_count",
        minimum=1,
    )
    tail_rows = _integer(
        geometry.get("tensor_rows_without_tokenizer_surface"),
        field="tensor_rows_without_tokenizer_surface",
        minimum=0,
    )
    eligible_model_vocab_rows = _integer(
        geometry.get("e13_eligible_model_vocab_token_count"),
        field="e13_eligible_model_vocab_token_count",
        minimum=1,
    )
    excluded_added_rows = _integer(
        geometry.get("e13_excluded_added_control_token_count"),
        field="e13_excluded_added_control_token_count",
        minimum=0,
    )
    excluded_added_ids_sha256 = _require_sha256(
        geometry.get("e13_excluded_added_control_token_ids_sha256"),
        field="e13_excluded_added_control_token_ids_sha256",
    )
    if tensor_rows != addressable + tail_rows:
        raise _Withheld(
            "ROW_UNIVERSE_GEOMETRY_INCONSISTENT",
            "tensor row count must equal tokenizer-addressable rows plus tokenizer-unaddressable tail rows",
        )
    if eligible_model_vocab_rows + excluded_added_rows != addressable:
        raise _Withheld(
            "ROW_UNIVERSE_GEOMETRY_INCONSISTENT",
            "tokenizer-addressable rows must equal lexical model-vocab rows plus excluded added/control rows",
        )
    expected = {
        "tensor_row_count": tensor_rows,
        "tokenizer_addressable_token_count": addressable,
        "excluded_tensor_tail_row_count": tail_rows,
        "e13_eligible_model_vocab_token_count": eligible_model_vocab_rows,
        "excluded_added_control_token_count": excluded_added_rows,
    }
    for field, observed in expected.items():
        if _integer(universe.get(field), field=f"row_universe.{field}", minimum=0) != observed:
            raise _Withheld(
                "ROW_UNIVERSE_BINDING_MISMATCH",
                f"row_universe.{field} does not match the observed tokenizer/tensor boundary",
            )
    if _require_sha256(
        universe.get("excluded_added_control_token_ids_sha256"),
        field="row_universe.excluded_added_control_token_ids_sha256",
    ) != excluded_added_ids_sha256:
        raise _Withheld(
            "ROW_UNIVERSE_BINDING_MISMATCH",
            "row_universe excluded added/control token ids do not match the tokenizer",
        )
    if universe.get("tail_policy") != ROW_UNIVERSE_TAIL_POLICY:
        raise _Withheld(
            "ROW_UNIVERSE_TAIL_POLICY_MISMATCH",
            f"row_universe.tail_policy must be {ROW_UNIVERSE_TAIL_POLICY}",
        )
    if universe.get("added_token_policy") != ROW_UNIVERSE_ADDED_TOKEN_POLICY:
        raise _Withheld(
            "ROW_UNIVERSE_ADDED_TOKEN_POLICY_MISMATCH",
            f"row_universe.added_token_policy must be {ROW_UNIVERSE_ADDED_TOKEN_POLICY}",
        )
    if any(token_id >= addressable for token_id in selected):
        raise _Withheld(
            "TOKEN_ID_NOT_ADDRESSABLE_BY_TOKENIZER",
            "selected E13 rows extend into the tokenizer-unaddressable tensor tail",
        )
    return {
        "schema": ROW_UNIVERSE_SCHEMA,
        **expected,
        "excluded_added_control_token_ids_sha256": excluded_added_ids_sha256,
        "tail_policy": ROW_UNIVERSE_TAIL_POLICY,
        "added_token_policy": ROW_UNIVERSE_ADDED_TOKEN_POLICY,
        "selected_rows_are_tokenizer_addressable": True,
        "selected_rows_are_lexical_model_vocab_rows": True,
    }


def _validate_controls(
        plan: Mapping[str, Any], source: Mapping[str, Any], *, row_budget: int,
        control_root: Path, plan_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise _Withheld("PLAN_SCHEMA_MISMATCH", f"control plan schema must be {PLAN_SCHEMA}")
    if _integer(plan.get("row_budget"), field="row_budget", minimum=1) != row_budget:
        raise _Withheld("ROW_BUDGET_MISMATCH", "control plan row_budget does not match invocation")
    source_binding = _mapping(plan.get("source"), field="source")
    expected = {
        "repo": source["repo"],
        "pinned_revision": source["pinned_revision"],
        "config_sha256": source["config_sha256"],
        "index_sha256": source["index_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "model_lake_manifest_sha256": source["model_lake_manifest_sha256"],
        "role_headers_sha256": source["role_headers_sha256"],
        "tokenizer_addressability_sha256": source["tokenizer_addressability_sha256"],
    }
    for field, observed in expected.items():
        if str(source_binding.get(field) or "").lower() != str(observed).lower():
            raise _Withheld("SOURCE_BINDING_MISMATCH", f"control plan source.{field} does not match observed source")

    geometry = _mapping(source.get("vocabulary_geometry"), field="observed vocabulary geometry")
    vocab_size = _integer(geometry.get("vocab_size"), field="observed vocab_size", minimum=1)
    addressable_token_count = _integer(
        geometry.get("addressable_token_count"),
        field="observed tokenizer addressable_token_count",
        minimum=1,
    )
    added_token_ids = {
        _integer(token_id, field="observed tokenizer added token id", minimum=0)
        for token_id in geometry.get("tokenizer_added_token_ids", [])
    }
    by_stratum, selected, summaries = _selected_ids(
        plan.get("strata"),
        vocab_size=vocab_size,
        row_budget=row_budget,
        addressable_token_count=addressable_token_count,
        ineligible_token_ids=added_token_ids,
    )
    row_universe = _validate_row_universe(plan, geometry, selected=selected)
    tokenizer_sha = str(source["tokenizer_sha256"]).lower()
    frequency_evidence, frequency_artifact = _evidence_artifact(
        plan,
        field="frequency_evidence",
        schema=FREQUENCY_EVIDENCE_SCHEMA,
        expected_provenance="tokenized_corpus_frequency.v1",
        expected_tokenizer_sha256=tokenizer_sha,
        control_root=control_root,
    )
    script_evidence, script_artifact = _evidence_artifact(
        plan,
        field="script_evidence",
        schema=SCRIPT_EVIDENCE_SCHEMA,
        expected_provenance="tokenizer_script_classification.v1",
        expected_tokenizer_sha256=tokenizer_sha,
        control_root=control_root,
    )
    evidence_producer = _admit_e13_evidence_producer(
        frequency_evidence,
        script_evidence,
        control_root=control_root,
    )
    frequencies = _evidence_map(
        frequency_evidence,
        field="frequency_evidence",
        vocab_size=vocab_size,
        addressable_token_count=addressable_token_count,
        ineligible_token_ids=added_token_ids,
    )
    scripts = _evidence_map(
        script_evidence,
        field="script_evidence",
        vocab_size=vocab_size,
        addressable_token_count=addressable_token_count,
        ineligible_token_ids=added_token_ids,
    )
    surfaces = _selected_token_surfaces(
        Path(str(geometry["tokenizer_path"])),
        selected,
        metadata_root=Path(str(source["root"])),
    )
    for summary in summaries:
        partitions = by_stratum[summary["id"]]
        ids = partitions["fit"] | partitions["heldout"]
        for token_id in ids:
            frequency = frequencies.get(token_id)
            script = scripts.get(token_id)
            if frequency is None or script is None:
                raise _Withheld(
                    "MISSING_FREQUENCY_OR_SCRIPT_EVIDENCE",
                    f"selected token {token_id} lacks corpus frequency or tokenizer script evidence",
                )
            count = _integer(frequency.get("count"), field=f"frequency[{token_id}].count", minimum=1)
            bounds = summary["frequency_range"]
            if not bounds["min_inclusive"] <= count <= bounds["max_inclusive"]:
                raise _Withheld(
                    "FREQUENCY_STRATUM_MISMATCH",
                    f"selected token {token_id} is outside its declared corpus-frequency range",
                )
            if str(script.get("script") or "").strip() != summary["script"]:
                raise _Withheld(
                    "SCRIPT_STRATUM_MISMATCH",
                    f"selected token {token_id} is outside its declared script stratum",
                )
            if str(script.get("tokenizer_raw_surface_sha256") or "").lower() != surfaces[token_id]["tokenizer_raw_surface_sha256"]:
                raise _Withheld(
                    "TOKENIZER_SURFACE_BINDING_MISMATCH",
                    f"script evidence raw surface does not match exact tokenizer token {token_id}",
                )
            byte_length = _integer(
                script.get("decoded_utf8_byte_length"),
                field=f"script[{token_id}].decoded_utf8_byte_length",
                minimum=1,
            )
            _require_sha256(
                script.get("decoded_token_surface_sha256"),
                field=f"script[{token_id}].decoded_token_surface_sha256",
            )
            if byte_length < 1:
                raise _Withheld(
                    "TOKENIZER_SURFACE_BINDING_MISMATCH",
                    f"script evidence decoded byte length is invalid for tokenizer token {token_id}",
                )
            byte_bounds = summary["byte_length_range"]
            if not byte_bounds["min_inclusive"] <= byte_length <= byte_bounds["max_inclusive"]:
                raise _Withheld(
                    "BYTE_LENGTH_STRATUM_MISMATCH",
                    f"selected token {token_id} is outside its declared byte-length range",
                )

    null = _mapping(plan.get("shuffled_token_label_null"), field="shuffled_token_label_null")
    if null.get("algorithm") != SPLIT_PRESERVING_NULL_ALGORITHM:
        raise _Withheld(
            "INVALID_SHUFFLED_NULL",
            f"shuffled-token null must be a {SPLIT_PRESERVING_NULL_ALGORITHM}",
        )
    seed = _integer(null.get("seed"), field="shuffled_token_label_null.seed", minimum=0)
    assignments = _derived_null_assignments(by_stratum, seed=seed)
    provided_assignments = null.get("assignments")
    if provided_assignments is not None:
        if not isinstance(provided_assignments, list):
            raise _Withheld("INVALID_SHUFFLED_NULL", "declared null assignments must be an array")
        normalized: list[Dict[str, Any]] = []
        for position, raw in enumerate(provided_assignments):
            row = _mapping(raw, field=f"shuffled_token_label_null.assignments[{position}]")
            normalized.append({
                "stratum_id": str(row.get("stratum_id") or "").strip(),
                "split": str(row.get("split") or "").strip(),
                "token_id": _integer(row.get("token_id"), field="null token_id", minimum=0),
                "shuffled_token_id": _integer(row.get("shuffled_token_id"), field="null shuffled_token_id", minimum=0),
            })
        normalized.sort(
            key=lambda row: (str(row["stratum_id"]), str(row["split"]), int(row["token_id"]))
        )
        if normalized != assignments:
            raise _Withheld(
                "SHUFFLED_NULL_NOT_SEED_DERIVED",
                "declared null assignments do not equal the deterministic within-stratum seeded derangement",
            )

    return {
        "plan_sha256": _sha256_text(plan),
        "plan_artifact": dict(plan_artifact),
        "row_budget": row_budget,
        "selected_token_ids_sha256": _sha256_text(sorted(selected)),
        "fit_and_heldout_disjoint": True,
        "row_universe": row_universe,
        "frequency_evidence": {
            **frequency_artifact,
            "corpus_sha256": frequency_evidence["corpus_sha256"],
            "corpus_token_count": frequency_evidence["corpus_token_count"],
            "covered_selected_tokens": len(selected),
        },
        "script_evidence": {
            **script_artifact,
            "classifier": dict(_mapping(script_evidence["classifier"], field="script_evidence artifact classifier")),
            "covered_selected_tokens": len(selected),
            "decoded_text_semantics": "VERIFIED_CANONICAL_HAWKING_CORE_SINGLE_TOKEN_DECODE",
        },
        "evidence_producer": evidence_producer,
        "strata": summaries,
        "shuffled_token_label_null": {
            "algorithm": null["algorithm"],
            "seed": seed,
            "within_stratum_bijection": True,
            "fit_and_heldout_partition_preserved": True,
            "seed_derived_derangement": True,
            "assignment_count": len(assignments),
            "assignments_sha256": _sha256_text(assignments),
        },
    }


def _load_plan(
        value: Optional[str | os.PathLike[str] | Mapping[str, Any]], *, control_root: Path,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raise _Withheld(
            "INLINE_CONTROL_PLAN_REFUSED",
            "E13 control plans must be immutable JSON artifacts under receipts/headless",
        )
    path = _control_artifact_path(value, field="E13 vocabulary control plan", control_root=control_root)
    raw = _read_bounded_bytes(
        path,
        maximum=MAX_PLAN_BYTES,
        label="E13 vocabulary control plan",
        allowed_root=control_root,
    )
    try:
        plan = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _Withheld("INVALID_CONTROL_PLAN", "E13 vocabulary control plan is not valid JSON") from exc
    plan = _mapping(plan, field="E13 vocabulary control plan")
    return dict(plan), {"path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw), "inline": False}


def run_flash_vocabulary_prefilter(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    plan: Optional[str | os.PathLike[str] | Mapping[str, Any]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    row_budget: int = DEFAULT_ROW_BUDGET,
) -> Dict[str, Any]:
    """Verify only E13's metadata/control prerequisites; never read row payloads."""
    try:
        budget = _integer(row_budget, field="row_budget", minimum=1)
    except _Withheld as exc:
        budget = DEFAULT_ROW_BUDGET
        input_error: Optional[_Withheld] = exc
    else:
        input_error = None
    if budget > MAX_ROW_BUDGET:
        input_error = _Withheld(
            "ROW_BUDGET_EXCEEDS_CPU_PREFLIGHT_CAP",
            f"row_budget {budget} exceeds the E13 CPU preflight cap of {MAX_ROW_BUDGET}",
        )
    final_root = _final_root(root)
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    control_root = (repo / "receipts" / "headless").resolve()
    destination = (
        Path(emit).expanduser().resolve()
        if emit else repo / "receipts" / "headless" / DEFAULT_EMIT_NAME
    )
    started_ns = time.time_ns()
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "experiment": "E13",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "control_artifact_root": str(control_root),
        "row_budget": budget,
        "model_loaded": False,
        "gpu_session_started": False,
        "tensor_payload_bytes_read": 0,
        "row_extraction_performed": False,
        "producer": _producer_identity(),
        "receipt_integrity": {
            "seal_format": RECEIPT_SEAL_FORMAT,
            "verifier": "hawking.agentos.flash_vocabulary_prefilter.verify_flash_vocabulary_prefilter_receipt",
        },
        "claim_boundary": (
            "header/control admissibility only; no embedding/head payload byte, row, logit, "
            "capability, or runtime claim is measured here; tokenizer-unaddressable tensor rows are "
            "excluded from E13 lexical selection; all declared tensor rows remain in complete byte "
            "accounting and raw-logit correctness surfaces, while tail sampling authority remains "
            "withheld; role payload identity is not rehashed here"
        ),
    }
    try:
        if input_error is not None:
            raise input_error
        if not final_root.is_dir():
            raise _Withheld("SOURCE_ROOT_UNAVAILABLE", f"Flash specimen root is unavailable: {final_root}")
        result["model_lake_manifest"] = _manifest_identity(final_root)
        source = _source_identity(final_root)
        source["model_lake_manifest_sha256"] = result["model_lake_manifest"]["sha256"]
        result["source_identity"] = source
        geometry = _mapping(source["vocabulary_geometry"], field="vocabulary_geometry")
        vocab_size = _integer(geometry["vocab_size"], field="vocab_size", minimum=1)
        if vocab_size == 248320 and budget != DEFAULT_ROW_BUDGET:
            raise _Withheld(
                "ROW_BUDGET_NOT_REGISTERED",
                f"the pinned E13 preflight requires exactly {DEFAULT_ROW_BUDGET} rows",
            )
        hidden_size = _integer(geometry["hidden_size"], field="hidden_size", minimum=1)
        addressable_token_count = _integer(
            geometry["addressable_token_count"],
            field="tokenizer addressable_token_count",
            minimum=1,
        )
        tail_rows = _integer(
            geometry["tensor_rows_without_tokenizer_surface"],
            field="tensor_rows_without_tokenizer_surface",
            minimum=0,
        )
        role_bytes = budget * hidden_size * 2
        result["bounded_extraction_budget"] = {
            "rows_total": budget,
            "tensor_row_count": vocab_size,
            "tokenizer_addressable_token_count": addressable_token_count,
            "e13_eligible_model_vocab_token_count": geometry["e13_eligible_model_vocab_token_count"],
            "e13_excluded_added_control_token_count": geometry["e13_excluded_added_control_token_count"],
            "excluded_tensor_tail_row_count": tail_rows,
            "tensor_tail_output_semantics": "SAMPLING_POLICY_WITHHELD_NOT_AN_E13_LEXICAL_ROW_UNIVERSE",
            "complete_tensor_accounting_policy": TENSOR_ACCOUNTING_POLICY,
            "raw_source_logit_policy": RAW_LOGIT_POLICY,
            "tail_sampling_policy": TAIL_SAMPLING_POLICY,
            "roles": ["input_embedding", "output_head"],
            "dtype": "BF16",
            "bytes_per_row_per_role": hidden_size * 2,
            "bytes_per_role": role_bytes,
            "bytes_total": role_bytes * 2,
            "payload_read_now": 0,
            "authorized_next_action": (
                "none until one predeclared authoritative corpus/threshold declaration is "
                "materialized by hawking gravity e13-evidence and its exact tokenizer-addressable "
                "row universe plus decoded-text controls pass this independent preflight"
            ),
        }
        loaded = _load_plan(plan, control_root=control_root)
        if loaded is None:
            raise _Withheld(
                "MISSING_VOCABULARY_CONTROL_PLAN",
                "use the canonical Rust producer with one predeclared authoritative UTF-8 corpus "
                "and fixed thresholds to supply independently hashed corpus-frequency and "
                "decoded-token-script artifacts, an explicit tokenizer-addressable row universe, "
                "disjoint frequency/script/byte-length strata, and a split-preserving seeded null",
            )
        loaded_plan, plan_artifact = loaded
        result["control_plan"] = _validate_controls(
            loaded_plan,
            source,
            row_budget=budget,
            control_root=control_root,
            plan_artifact=plan_artifact,
        )
        result["evidence_admission"] = dict(
            _mapping(
                result["control_plan"]["evidence_producer"],
                field="control_plan.evidence_producer",
            )
        )
        result["status"] = "READY_FOR_BOUNDED_ROW_EXTRACTION"
        result["next_action"] = (
            "extract only the declared embedding/head rows under this exact plan, "
            "then run the paired-role predictor and split-preserving shuffled null"
        )
    except _Withheld as exc:
        result["status"] = f"WITHHELD_{exc.code}"
        result["withheld_reason"] = str(exc)
        result["next_action"] = "repair the named source/control prerequisite; do not read tensor payload bytes yet"
    except Exception as exc:  # noqa: BLE001 - preflight must persist an honest blocker
        result["status"] = "WITHHELD_UNEXPECTED_PREFLIGHT_ERROR"
        result["withheld_reason"] = f"{type(exc).__name__}: {exc}"
        result["next_action"] = "inspect the source metadata error without loading a model or tensor payload"
    result["started_at_ns"] = started_ns
    result["finished_at_ns"] = time.time_ns()
    result["elapsed_ns"] = result["finished_at_ns"] - started_ns
    result["elapsed_s"] = round(result["elapsed_ns"] / 1_000_000_000, 3)
    result["receipt_path"] = str(destination)
    sealed = _seal_receipt(result)
    # A writer is not a verifier.  Refuse to emit a receipt we cannot validate
    # under the explicitly declared compact UTF-8 serialization.
    verify_flash_vocabulary_prefilter_receipt(sealed)
    atomic_write_json(destination, sealed)
    return sealed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--repo-root")
    parser.add_argument("--plan")
    parser.add_argument("--emit")
    parser.add_argument("--row-budget", type=int, default=DEFAULT_ROW_BUDGET)
    parser.add_argument(
        "--verify-receipt",
        help="verify an existing E13 receipt under its declared compact UTF-8 seal",
    )
    args = parser.parse_args(argv)
    if args.verify_receipt:
        try:
            verified = verify_flash_vocabulary_prefilter_receipt_path(args.verify_receipt)
        except (ValueError, _Withheld) as exc:
            print(json.dumps({"status": "INVALID_E13_RECEIPT", "reason": str(exc)}, indent=2))
            return 1
        print(json.dumps({
            "status": "VERIFIED_E13_RECEIPT",
            "receipt_path": str(Path(args.verify_receipt).expanduser().resolve()),
            "receipt_status": verified.get("status"),
            "seal_format": verified.get("seal_format"),
            "seal_sha256": verified.get("seal_sha256"),
        }, indent=2, sort_keys=True))
        return 0
    report = run_flash_vocabulary_prefilter(
        root=args.root,
        repo_root=args.repo_root,
        plan=args.plan,
        emit=args.emit,
        row_budget=args.row_budget,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "ADMISSIBLE_BOUNDED_CPU_ROW_EXTRACTION" else 1


__all__ = [
    "DEFAULT_ROW_BUDGET",
    "FREQUENCY_EVIDENCE_SCHEMA",
    "PLAN_SCHEMA",
    "SCHEMA",
    "SCRIPT_EVIDENCE_SCHEMA",
    "main",
    "run_flash_vocabulary_prefilter",
    "verify_flash_vocabulary_prefilter_receipt",
    "verify_flash_vocabulary_prefilter_receipt_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
