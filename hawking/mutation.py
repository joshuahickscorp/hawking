from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .persist import atomic_write_bytes, atomic_write_json

MAX_MUTATION_OPERATIONS = 20


class MutationError(Exception):
    pass


class MutationConflict(MutationError):
    """The candidate tree changed after a mutation was admitted."""


MUTATION_REQUEST_SCHEMA = "hawking.mutation.request.v1"
MUTATION_INTENT_SCHEMA = "hawking.mutation.intent.v1"
MUTATION_INTENT_STATUSES = frozenset({
    "NOT_STARTED", "APPLIED", "VERIFIED", "UNKNOWN_OUTCOME",
})
MUTATION_PROPOSAL_SCHEMA = "hawking.mutation_proposal.v1"

# Provider proposals often carry the observations that informed the edit in
# the same ``operations`` array.  Those observations are not mutation effects
# and must not be handed to Engine.apply_typed_mutation, which intentionally
# accepts only effect operations.  Keep this list narrow: unknown or
# effectful operations still reach the canonical engine and fail closed.
_READ_ONLY_PROVIDER_OPERATIONS = frozenset({
    "read", "inspect", "search", "list", "status", "diff",
    "fs.read", "fs.search", "filesystem.read", "filesystem.search",
    "source.read", "source.search", "git.status", "git.diff",
})


def _canonical_provider_operation(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce the admitted provider edit dialect into engine operations.

    Providers are allowed to emit a bounded typed proposal, not engine-owned
    tool arguments.  A few qualified routes use ``file``/``anchor_before``/
    ``replacement`` for the same operation that Hawking's engine represents
    as ``path``/``old_text``/``new_text``.  Normalize that small dialect here;
    untyped prose, unknown operation kinds, paths, and verification gates still
    fail closed downstream.
    """
    row = dict(value)

    def pick(*names: str) -> Any:
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return None

    raw_op = str(
        pick("op", "operation", "type", "kind", "MODE", "mode") or ""
    ).strip().lower()
    action = str(pick("action", "edit_kind", "position", "where") or "").strip().lower()
    provider_tool = str(pick("tool", "tool_name") or "").strip().lower()
    if not raw_op and provider_tool:
        raw_op = "edit" if provider_tool == "repo.edit" else provider_tool
    elif raw_op == "repo.edit":
        raw_op = "edit"
    path = pick("path", "file", "target")
    if path is not None and "path" not in row:
        row["path"] = path

    if raw_op in {
        "edit", "replace", "anchor_edit", "anchor_replace", "replace_range",
        "replace_once", "replace_lines", "patch", "source_edit", "update", "modify",
    }:
        if action in {"insert", "insert_before", "prepend", "insert_after", "append", "append_after", "append_after_function"}:
            raw_op = (
                "insert_before"
                if action in {"insert_before", "prepend"}
                else "insert_after"
            )
            row["op"] = raw_op
            anchor = pick("anchor", "anchor_before", "anchor_after", "old_text", "old", "find")
            text = pick("text", "content", "new_text", "new", "replacement")
            if anchor is not None:
                row["old_text"] = anchor
                row["anchor"] = anchor
            if text is not None:
                row["new_text"] = text
                row["text"] = text
            return row
        row["op"] = "replace"
        old = pick(
            "old_text", "old", "anchor_before", "ANCHOR_BEFORE", "find", "anchor"
        )
        new = pick(
            "new_text", "new", "replacement", "new_content", "content",
            "replacement_text", "replacement_block", "begin_replacement",
        )
        old_lines = pick("old_lines", "before_lines")
        new_lines = pick("new_lines", "after_lines", "replacement_lines")
        if old_lines is not None and "old_text" not in row:
            row["old_lines"] = old_lines
        elif old is not None and "old_text" not in row:
            row["old_text"] = old
        if new_lines is not None and "new_text" not in row:
            row["new_lines"] = new_lines
        elif new is not None and "new_text" not in row:
            row["new_text"] = new
    elif raw_op in {"create_file", "new_file"}:
        row["op"] = "create"
        content = pick("content", "new_text", "new_content")
        if content is None:
            lines = pick("new_lines", "content_lines")
            if isinstance(lines, list):
                content = "\n".join(str(item) for item in lines) + "\n"
        row["content"] = "" if content is None else content
    elif raw_op in {"insert", "insert_before", "insert_after"}:
        # Several remote routes use the compact ``insert`` spelling. Preserve
        # the engine's explicit semantics when the payload names a position;
        # otherwise default to insert-after, the least surprising meaning for
        # an anchor-based append-style patch. The engine still validates the
        # anchor and replacement fail-closed.
        position = str(pick("position", "where", "relative_to") or "").strip().lower()
        if raw_op == "insert":
            raw_op = (
                "insert_before" if position in {"before", "prepend"}
                else "insert_after"
            )
        row["op"] = raw_op
        anchor = pick("anchor", "anchor_before", "anchor_after")
        text = pick("text", "content", "new_text", "replacement")
        if anchor is not None and "old_text" not in row:
            row["old_text"] = anchor
        if text is not None and "new_text" not in row:
            row["new_text"] = text
        if anchor is not None:
            row["anchor"] = anchor
        if text is not None:
            row["text"] = text
    elif raw_op in {"append", "append_after", "append_to_eof"}:
        # Qualified providers commonly describe an append as
        # ``{op: append, content: ...}`` (sometimes with an ``__EOF__``
        # anchor).  The engine's append operation is already EOF-scoped; bind
        # the provider's content to its canonical new_text field and ignore
        # only the non-authoritative sentinel anchor.
        row["op"] = "append"
        text = pick(
            "new_text", "text", "content", "replacement", "new_content",
            "replacement_text", "replacement_block",
        )
        if text is not None:
            row["new_text"] = text
    return row


@dataclass(frozen=True)
class MutationProposal:
    """Provider data describing a possible edit; never an authority grant."""

    objective: str
    operations: Tuple[Mapping[str, Any], ...]
    baseline_evidence_refs: Tuple[str, ...] = field(default_factory=tuple)
    expected_base_hashes: Mapping[str, Optional[str]] = field(default_factory=dict)
    required_tests: Tuple[str, ...] = field(default_factory=tuple)
    broader_tests: Tuple[str, ...] = field(default_factory=tuple)
    canonical_root: str = ""
    goal_id: str = ""
    workunit_id: str = ""
    source_digest: Optional[str] = None
    completion_contract: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MutationProposal":
        if not isinstance(value, Mapping):
            raise MutationError("mutation proposal must be an object")
        if value.get("schema") not in (
            None,
            MUTATION_PROPOSAL_SCHEMA,
            "HAWKING_EDIT_V1",
            "HAWKING_PATCH_V1",
        ):
            raise MutationError("unsupported mutation proposal schema")
        # Qualified provider routes sometimes serialize a one-edit tranche
        # as ``operation`` even though the canonical proposal schema uses the
        # plural ``operations``.  Preserve the strict typed boundary while
        # accepting that unambiguous singleton dialect.  A few routes emit
        # the same typed edit at the proposal top level (``file``, ``mode``,
        # ``anchor_before``, ``replacement``) rather than nesting it.  Compile
        # that dialect here too, but only for an explicitly typed schema and
        # only when it contains a real edit payload; prose and incomplete
        # status objects still normalize to no effects and fail closed.
        raw_ops = (
            value.get("operations")
            or value.get("edits")
            or value.get("operation")
            # Qualified Cloud Auto routes use this explicit name to make the
            # source-vs-receipt distinction visible in a compact proposal.
            # It is still only untrusted provider data and must pass the same
            # canonical operation, scope, anchor, hash, and test gates.
            or value.get("source_operations")
            or value.get("source_operation")
            or ()
        )
        if isinstance(raw_ops, Mapping):
            raw_ops = [raw_ops]
        if not raw_ops and value.get("schema") in {"HAWKING_EDIT_V1", "HAWKING_PATCH_V1"}:
            def pick(*names: str) -> Any:
                for name in names:
                    if name in value and value[name] is not None:
                        return value[name]
                return None

            path = pick("path", "file", "target")
            mode = str(pick("mode", "op", "operation", "type", "kind") or "").strip().lower()
            anchor = pick(
                "old_text", "old", "anchor_before", "ANCHOR_BEFORE", "find", "anchor"
            )
            replacement = pick(
                "new_text", "new", "replacement", "new_content", "content",
                "replacement_text", "replacement_block", "begin_replacement",
            )
            if path and mode in {"create", "create_file", "new_file"} and replacement is not None:
                raw_ops = [{"op": "create", "path": path, "content": replacement}]
            elif path and mode in {"append", "append_after", "append_to_eof"} and replacement is not None:
                raw_ops = [{"op": "append", "path": path, "new_text": replacement}]
            elif path and anchor is not None and replacement is not None:
                raw_ops = [{
                    "op": "replace",
                    "path": path,
                    "old_text": anchor,
                    "new_text": replacement,
                }]
        operations = tuple(
            _canonical_provider_operation(item)
            for item in raw_ops
            if isinstance(item, Mapping)
        )
        expected_base_hashes = dict(value.get("expected_base_hashes") or {})
        def string_tuple(raw: Any) -> Tuple[str, ...]:
            if raw in (None, ""):
                return ()
            if isinstance(raw, (list, tuple, set, frozenset)):
                return tuple(str(x) for x in raw if str(x).strip())
            return (str(raw),)
        for operation in operations:
            path = operation.get("path")
            digest = operation.get("expected_digest")
            if path and digest and path not in expected_base_hashes:
                match = re.search(r"[0-9a-fA-F]{64}", str(digest))
                if match:
                    expected_base_hashes[str(path)] = match.group(0).lower()
        return cls(
            objective=str(value.get("objective") or "").strip(),
            operations=operations,
            baseline_evidence_refs=string_tuple(value.get("baseline_evidence_refs")),
            expected_base_hashes=expected_base_hashes,
            required_tests=string_tuple(value.get("required_tests") or value.get("tests")),
            broader_tests=string_tuple(value.get("broader_tests")),
            canonical_root=str(value.get("canonical_root") or ""),
            goal_id=str(value.get("goal_id") or ""),
            workunit_id=str(value.get("workunit_id") or ""),
            source_digest=value.get("source_digest"),
            completion_contract=dict(value.get("completion_contract") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": MUTATION_PROPOSAL_SCHEMA,
            "status": "MUTATION_PROPOSED",
            "objective": self.objective,
            "operations": [dict(x) for x in self.operations],
            "baseline_evidence_refs": list(self.baseline_evidence_refs),
            "expected_base_hashes": dict(self.expected_base_hashes),
            "required_tests": list(self.required_tests),
            "broader_tests": list(self.broader_tests),
            "canonical_root": self.canonical_root,
            "goal_id": self.goal_id,
            "workunit_id": self.workunit_id,
            "source_digest": self.source_digest,
            "completion_contract": _json_copy(self.completion_contract),
        }


def _envelope_lines(text: Any) -> List[str]:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def normalize_mutation_result(
    value: Any,
    *,
    canonical_root: str,
    goal_id: str = "",
    workunit_id: str = "",
) -> MutationProposal:
    """Compile provider-friendly mutation data into the canonical proposal.

    Only an explicit typed object or a complete Hawking envelope is accepted;
    surrounding prose is deliberately ignored and never becomes executable.
    """
    if isinstance(value, Mapping):
        # Compact workers return semantic intent only.  Hawking resolves the
        # durable source anchor and supplies the exact old bytes locally.
        if str(value.get("s") or value.get("status") or "").strip().upper() == "MUTATE" and value.get("anchor"):
            from .compact_worker import compile_mutation_intent
            return compile_mutation_intent(
                value,
                root=canonical_root,
                goal_id=goal_id,
                workunit_id=workunit_id,
                authority=value.get("authority") if isinstance(value.get("authority"), Mapping) else None,
            )
        status = str(value.get("status") or "").strip().upper()
        # Providers sometimes label a bounded repair as PATCH_PROPOSED,
        # EDIT_PROPOSED, or REPAIR_PROPOSED.  Those are still untrusted data:
        # admit them only when they carry an explicit non-empty operation list
        # and canonicalize the label before the normal authority/scope/hash/test
        # gates run.  Never promote a prose-only blocker or completion claim.
        proposal_aliases = {
            "PATCH_PROPOSED",
            "EDIT_PROPOSED",
            "REPAIR_PROPOSED",
        }
        operations = value.get("operations")
        if (
            status == "MUTATION_PROPOSED"
            or (status in proposal_aliases and isinstance(operations, list) and operations)
        ):
            raw = dict(value)
            raw["status"] = "MUTATION_PROPOSED"
        else:
            raise MutationError("MUTATION_PAYLOAD_MISSING")
    else:
        lines = _envelope_lines(value)
        first = next((line.strip() for line in lines if line.strip()), "")
        if first == "HAWKING_EDIT_V1":
            raw = _parse_edit_envelope(lines)
        elif first == "HAWKING_PATCH_V1":
            raw = _parse_patch_envelope(lines, canonical_root=canonical_root)
        else:
            raise MutationError("MUTATION_PAYLOAD_MISSING")
    raw["canonical_root"] = canonical_root
    raw["goal_id"] = goal_id or raw.get("goal_id") or ""
    raw["workunit_id"] = workunit_id or raw.get("workunit_id") or ""
    return MutationProposal.from_mapping(raw)


def _envelope_field(lines: List[str], name: str) -> str:
    prefix = name.upper() + ":"
    for line in lines:
        if line.upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _section(lines: List[str], begin: str, end: str) -> List[str]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == begin)
        finish = next(i for i in range(start + 1, len(lines)) if lines[i].strip().upper() == end)
    except StopIteration as exc:
        raise MutationError(f"malformed mutation envelope: {begin}") from exc
    return lines[start + 1:finish]


def _parse_edit_envelope(lines: List[str]) -> Dict[str, Any]:
    path = _envelope_field(lines, "PATH")
    mode = _envelope_field(lines, "MODE") or "replace_file"
    if not path:
        raise MutationError("malformed mutation envelope: PATH")
    if "BEGIN_OLD_CONTENT" in lines:
        old_content = "\n".join(_section(lines, "BEGIN_OLD_CONTENT", "END_OLD_CONTENT"))
        content = "\n".join(_section(lines, "BEGIN_CONTENT", "END_CONTENT"))
        op = {"op": "replace", "path": path, "old_text": old_content,
              "new_text": content}
    else:
        content = "\n".join(_section(lines, "BEGIN_CONTENT", "END_CONTENT"))
        op = {"op": {"replace_range": "replace", "replace_file": "replace_file",
                      "create_file": "create", "delete_file": "replace_file"}.get(mode, mode),
              "path": path}
    if "END_CONTENT" in lines and lines.index("END_CONTENT") > 0:
        # Preserve the provider's final newline convention without adding an
        # invisible extra line to an otherwise exact replacement.
        begin = lines.index("BEGIN_CONTENT")
        end = lines.index("END_CONTENT")
        content = "\n".join(lines[begin + 1:end])
    if op["op"] == "replace":
        if "old_text" not in op:
            raise MutationError("replace_range requires BEGIN_OLD_CONTENT")
    if mode == "delete_file":
        content = ""
        op["new_text"] = content
    op["new_text"] = content
    tests = [x.strip() for x in _envelope_field(lines, "TESTS").split(",") if x.strip()]
    digest = _digest_fields(_envelope_field(lines, "EXPECTED_DIGEST"), default_path=path)
    return {"status": "MUTATION_PROPOSED", "objective": "provider edit envelope",
            "operations": [op], "required_tests": tests,
            "expected_base_hashes": digest}


def _parse_keyed_block(lines: List[str], key: str, *, start: int = 0) -> str:
    """Read a bounded ``KEY:`` section without interpreting surrounding prose."""
    prefix = key.upper() + ":"
    for index in range(max(0, start), len(lines)):
        if lines[index].upper().startswith(prefix):
            value = lines[index].split(":", 1)[1].strip()
            if value:
                return value
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                stripped = lines[candidate].strip().upper()
                if stripped.startswith((
                    "FILE:", "PATH:", "MODE:", "TESTS:", "EXPECTED_DIGEST:",
                    "BASE_REVISION:", "ANCHOR_BEFORE:", "ANCHOR_AFTER:",
                    "FIND:", "REPLACE:", "REPLACEMENT:", "BEGIN_",
                    "END_", "END",
                )):
                    end = candidate
                    break
            return "\n".join(lines[index + 1:end]).strip("\n")
    return ""


def _digest_fields(raw: str, *, default_path: str = "") -> Dict[str, str]:
    """Parse ``path sha256=<hex>`` without accepting a non-content identity."""
    value = str(raw or "").strip()
    if not value:
        return {}
    match = re.search(r"(?i)(?:sha256\s*[=:]\s*)?([0-9a-f]{64})\b", value)
    if match is None:
        return {}
    digest = match.group(1).lower()
    before = value[:match.start()].strip()
    path = before.split()[0] if before else str(default_path or "").strip()
    return {path: digest} if path else {}


def _parse_anchor_patch_envelope(
    lines: List[str], *, canonical_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Accept the bounded anchor form emitted by some provider routes.

    This is still a data-only compiler.  A block must name a relative file and
    provide an explicit old anchor (or FIND/OLD section) plus a replacement
    section.  ``ANCHOR_AFTER`` is retained as non-authoritative context so a
    future verifier can display it; it never grants authority or widens scope.
    """
    file_starts = [
        index for index, line in enumerate(lines)
        if line.strip().upper().startswith("FILE:")
    ]
    if not file_starts:
        file_starts = [
            index for index, line in enumerate(lines)
            if line.strip().upper().startswith("PATH:")
        ]
    if not file_starts:
        raise MutationError("malformed mutation envelope: FILE/PATH")
    boundaries = file_starts[1:] + [len(lines)]
    operations: List[Dict[str, Any]] = []
    expected: Dict[str, str] = {}
    global_digest = _envelope_field(lines, "EXPECTED_DIGEST")
    global_tests = [
        item.strip() for item in _envelope_field(lines, "TESTS").split(",")
        if item.strip()
    ]
    for start, finish in zip(file_starts, boundaries):
        block = lines[start:finish]
        path = _envelope_field(block, "FILE") or _envelope_field(block, "PATH")
        if not path:
            raise MutationError("malformed mutation envelope: FILE/PATH")
        mode = (_envelope_field(block, "MODE") or "replace_range").strip().lower()
        digest_raw = _envelope_field(block, "EXPECTED_DIGEST") or global_digest
        expected.update(_digest_fields(digest_raw, default_path=path))

        old_text = ""
        if "BEGIN_OLD_CONTENT" in block:
            old_text = "\n".join(_section(block, "BEGIN_OLD_CONTENT", "END_OLD_CONTENT"))
        elif "FIND:" in "\n".join(block).upper():
            old_text = _parse_keyed_block(block, "FIND")
        else:
            # The observed provider-friendly form uses a one-line exact anchor.
            old_text = _envelope_field(block, "ANCHOR_BEFORE")

        new_text = ""
        for begin, end in (
            ("BEGIN_REPLACEMENT", "END_REPLACEMENT"),
            ("BEGIN_NEW_CONTENT", "END_NEW_CONTENT"),
            ("BEGIN_CONTENT", "END_CONTENT"),
        ):
            if begin in block:
                new_text = "\n".join(_section(block, begin, end))
                break
        if not new_text and "REPLACE:" in "\n".join(block).upper():
            new_text = _parse_keyed_block(block, "REPLACE")
        if not new_text and "REPLACEMENT:" in "\n".join(block).upper():
            new_text = _parse_keyed_block(block, "REPLACEMENT")

        if mode in {"create", "create_file"}:
            operations.append({"op": "create", "path": path, "new_text": new_text})
        elif mode in {"delete", "delete_file"}:
            operations.append({"op": "replace_file", "path": path, "new_text": ""})
        else:
            if not old_text:
                raise MutationError("malformed mutation envelope: missing exact anchor")
            if not new_text:
                raise MutationError("malformed mutation envelope: missing replacement")
            operation = {
                "op": "replace", "path": path,
                "old_text": old_text, "new_text": new_text,
            }
            after = _envelope_field(block, "ANCHOR_AFTER")
            if after:
                operation["provider_context_after"] = after
            operations.append(operation)

    if not operations:
        raise MutationError("malformed mutation envelope: empty patch")
    return {
        "status": "MUTATION_PROPOSED",
        "objective": "provider anchor patch envelope",
        "operations": operations,
        "required_tests": global_tests,
        "expected_base_hashes": expected,
    }


def _parse_patch_envelope(
    lines: List[str], *, canonical_root: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == "BEGIN_PATCH")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().upper() == "END_PATCH")
    except StopIteration as exc:
        # Some qualified routes emit the same explicit version marker with a
        # simpler FILE/ANCHOR/REPLACEMENT envelope.  Parse that data-only form
        # before rejecting it; ordinary prose still has no FILE/PATH block and
        # remains MUTATION_PAYLOAD_MISSING.
        return _parse_anchor_patch_envelope(lines, canonical_root=canonical_root)
    patch = lines[start + 1:end]
    operations: List[Dict[str, Any]] = []
    i = 0
    while i < len(patch):
        if not patch[i].startswith("--- "):
            i += 1
            continue
        old_name = patch[i][4:].strip().split("\t", 1)[0]
        i += 1
        if i >= len(patch) or not patch[i].startswith("+++ "):
            raise MutationError("malformed unified patch header")
        new_name = patch[i][4:].strip().split("\t", 1)[0]
        path = new_name if new_name != "/dev/null" else old_name
        path = path[2:] if path.startswith(("a/", "b/")) else path
        i += 1
        hunk: List[str] = []
        while i < len(patch) and not patch[i].startswith("--- "):
            hunk.append(patch[i]); i += 1
        old = [x[1:] for x in hunk if x.startswith((" ", "-"))]
        new = [x[1:] for x in hunk if x.startswith((" ", "+"))]
        if old_name == "/dev/null":
            operations.append({"op": "create", "path": path, "new_text": "\n".join(new) + "\n"})
        elif new_name == "/dev/null":
            operations.append({"op": "replace_file", "path": path, "new_text": ""})
        elif old and new:
            operations.append({"op": "replace", "path": path,
                               "old_text": "\n".join(old),
                               "new_text": "\n".join(new)})
        else:
            raise MutationError("malformed unified patch hunk")
    if not operations:
        raise MutationError("malformed mutation envelope: empty patch")
    tests = [x.strip() for x in _envelope_field(lines, "TESTS").split(",") if x.strip()]
    base = _envelope_field(lines, "BASE_REVISION")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", base or ""):
        base = ""
    return {"status": "MUTATION_PROPOSED", "objective": "provider patch envelope",
            "operations": operations, "required_tests": tests,
            "source_digest": base or None}


def normalize_mutation_path(
    raw: Any,
    *,
    root: Optional[Union[str, os.PathLike[str]]] = None,
) -> str:
    """Return a safe root-relative path for a typed mutation contract.

    Absolute paths are accepted only when a canonical root is supplied and
    the resolved target is inside it.  Relative ``..`` paths and empty/root
    targets are rejected before a tool or engine gets a chance to write.
    """
    text = str(raw or "").strip()
    if not text or "\x00" in text:
        raise MutationError("mutation path is empty or contains NUL")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if root is None:
            raise MutationError("absolute mutation paths require a canonical root")
        root_path = Path(root).expanduser().resolve()
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(root_path)
        except ValueError as exc:
            raise MutationError(f"mutation path escapes canonical root: {text}") from exc
        parts = relative.parts
    else:
        # Use POSIX separators in the wire contract even on macOS.  Path
        # containment is checked again by MutationExecutor before every write.
        parts = tuple(part for part in text.replace("\\", "/").split("/") if part not in {"", "."})
        if any(part == ".." for part in parts):
            raise MutationError(f"mutation path escape rejected: {text}")
    if not parts or any(part.casefold() in {".git", ".hawking"} for part in parts):
        raise MutationError(f"protected mutation path rejected: {text}")
    return "/".join(str(part) for part in parts)


def _sha256_payload(payload: Optional[bytes]) -> Optional[str]:
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()


def mutation_tree_digest(
    files: Iterable[Mapping[str, Any]],
    *,
    hash_field: str = "sha256_after",
) -> str:
    """Digest the exact path/content view a verifier was run against."""
    rows = []
    for item in files or ():
        if not isinstance(item, Mapping):
            continue
        rows.append({
            "path": str(item.get("path") or ""),
            "sha256": item.get(hash_field),
        })
    rows.sort(key=lambda item: item["path"])
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _path_bytes(path: Path) -> Optional[bytes]:
    if not path.exists():
        return None
    if not path.is_file():
        raise MutationError(f"mutation target is not a file: {path}")
    return path.read_bytes()


def _file_identity(path: Path) -> Optional[Dict[str, Any]]:
    """Capture identity separately from content so replacement races are visible."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mode": int(stat.st_mode),
    }


@dataclass(frozen=True)
class MutationRequest:
    """One typed, attributable mutation admission request."""

    request_id: str
    goal_id: str
    workunit_id: str
    worker_attempt_id: str
    capability_id: str
    capability_schema: str
    worktree_id: str
    canonical_root: str
    allowed_paths: Tuple[str, ...] = field(default_factory=tuple)
    expected_base_hashes: Mapping[str, Optional[str]] = field(default_factory=dict)
    operations: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    payload_ref: Optional[str] = None
    lease_id: Optional[str] = None
    verification_obligations: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    resource_reservation: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", request_id):
            raise MutationError("invalid mutation request_id")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "goal_id", str(self.goal_id or "").strip())
        object.__setattr__(self, "workunit_id", str(self.workunit_id or "").strip())
        object.__setattr__(self, "worker_attempt_id", str(self.worker_attempt_id or "").strip())
        object.__setattr__(self, "capability_id", str(self.capability_id or "hawking.mutation").strip())
        object.__setattr__(self, "capability_schema", str(self.capability_schema or MUTATION_REQUEST_SCHEMA).strip())
        object.__setattr__(self, "worktree_id", str(self.worktree_id or "worktree-unknown").strip())
        canonical_root = str(Path(self.canonical_root).expanduser().resolve())
        object.__setattr__(self, "canonical_root", canonical_root)
        allowed = []
        for raw_path in self.allowed_paths or ():
            if str(raw_path).strip():
                allowed.append(normalize_mutation_path(raw_path, root=canonical_root))
        object.__setattr__(self, "allowed_paths", tuple(sorted(dict.fromkeys(allowed))))
        expected: Dict[str, Optional[str]] = {}
        for raw_path, digest in dict(self.expected_base_hashes or {}).items():
            path = normalize_mutation_path(raw_path, root=canonical_root)
            if not path:
                continue
            if digest is not None:
                digest = str(digest).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise MutationError(f"invalid expected base sha256 for {path}")
            expected[path] = digest
        object.__setattr__(self, "expected_base_hashes", expected)
        normalized_operations = []
        for item in (self.operations or ()):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            if row.get("path") is not None:
                row["path"] = normalize_mutation_path(row.get("path"), root=canonical_root)
            normalized_operations.append(row)
        operations = tuple(normalized_operations)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "verification_obligations", _json_copy(self.verification_obligations))
        object.__setattr__(self, "budget", _json_copy(self.budget))
        object.__setattr__(self, "resource_reservation", _json_copy(self.resource_reservation))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MutationRequest":
        if not isinstance(value, Mapping):
            raise TypeError("mutation request must be an object")
        schema = value.get("schema")
        if schema not in (None, MUTATION_REQUEST_SCHEMA):
            raise MutationError(f"unsupported mutation request schema: {schema}")
        operations = value.get("operations")
        if operations is None and isinstance(value.get("operation"), Mapping):
            operations = [value["operation"]]
        return cls(
            request_id=value.get("request_id") or f"REQ-{uuid.uuid4().hex}",
            goal_id=value.get("goal_id") or "",
            workunit_id=value.get("workunit_id") or "",
            worker_attempt_id=value.get("worker_attempt_id") or f"ATTEMPT-{uuid.uuid4().hex}",
            capability_id=value.get("capability_id") or "hawking.mutation",
            capability_schema=value.get("capability_schema") or MUTATION_REQUEST_SCHEMA,
            worktree_id=value.get("worktree_id") or "worktree-unknown",
            canonical_root=value.get("canonical_root") or ".",
            allowed_paths=value.get("allowed_paths") or (),
            expected_base_hashes=value.get("expected_base_hashes") or {},
            operations=operations or (),
            payload_ref=value.get("payload_ref"),
            lease_id=value.get("lease_id"),
            verification_obligations=value.get("verification_obligations") or {},
            budget=value.get("budget") or {},
            resource_reservation=value.get("resource_reservation") or {},
            created_at=float(value.get("created_at") or time.time()),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": MUTATION_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "goal_id": self.goal_id,
            "workunit_id": self.workunit_id,
            "worker_attempt_id": self.worker_attempt_id,
            "capability_id": self.capability_id,
            "capability_schema": self.capability_schema,
            "worktree_id": self.worktree_id,
            "canonical_root": self.canonical_root,
            "allowed_paths": list(self.allowed_paths),
            "expected_base_hashes": dict(self.expected_base_hashes),
            "operations": [dict(item) for item in self.operations],
            "payload_ref": self.payload_ref,
            "lease_id": self.lease_id,
            "verification_obligations": _json_copy(self.verification_obligations),
            "budget": _json_copy(self.budget),
            "resource_reservation": _json_copy(self.resource_reservation),
            "created_at": self.created_at,
        }
        payload["operation_sha256"] = hashlib.sha256(
            json.dumps(payload["operations"], sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return payload


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def build_mutation_request(
    root: Union[str, os.PathLike[str]],
    operations: Iterable[Mapping[str, Any]],
    tests: Optional[Iterable[Any]] = None,
    *,
    metadata: Optional[Union[MutationRequest, Mapping[str, Any]]] = None,
    capability_id: str = "repo.edit",
) -> MutationRequest:
    """Bind an operation to the current tree and existing policy owners."""
    canonical = Path(root).expanduser().resolve()
    if isinstance(metadata, MutationRequest):
        raw = metadata.to_dict()
    else:
        raw = dict(metadata or {})
    normalized_operations: List[Dict[str, Any]] = []
    touched: List[str] = []
    for operation in operations or ():
        if not isinstance(operation, Mapping):
            continue
        row = dict(operation)
        relative = normalize_mutation_path(row.get("path"), root=canonical)
        row["path"] = relative
        normalized_operations.append(row)
        if relative not in touched:
            touched.append(relative)
    expected: Dict[str, Optional[str]] = {}
    for raw_path, digest in dict(raw.get("expected_base_hashes") or {}).items():
        relative = normalize_mutation_path(raw_path, root=canonical)
        expected[relative] = None if digest is None else str(digest).lower()
    for relative in touched:
        if relative in expected:
            continue
        expected[relative] = _sha256_payload(_path_bytes(canonical / relative))
    test_list = [str(item) for item in (tests or ()) if str(item).strip()]
    obligations = dict(raw.get("verification_obligations") or {})
    obligations.setdefault("tests", test_list)
    obligations.setdefault("inspect_diff", True)
    obligations.setdefault("result_packet", True)
    worktree_id = str(raw.get("worktree_id") or f"worktree-{hashlib.sha256(str(canonical).encode()).hexdigest()[:16]}")
    budget = dict(raw.get("budget") or {})
    budget.setdefault("authority", "existing Hawking Goal/ledger")
    budget.setdefault("cap_usd", None)
    resource_reservation = dict(raw.get("resource_reservation") or {})
    resource_reservation.setdefault("authority", "hawking.resources.ResourceLimits")
    resource_reservation.setdefault("class", "MUTATION")
    resource_reservation.setdefault("slots", 1)
    return MutationRequest(
        request_id=str(raw.get("request_id") or f"REQ-{uuid.uuid4().hex}"),
        goal_id=str(raw.get("goal_id") or ""),
        workunit_id=str(raw.get("workunit_id") or ""),
        worker_attempt_id=str(raw.get("worker_attempt_id") or f"ATTEMPT-{uuid.uuid4().hex}"),
        capability_id=str(raw.get("capability_id") or capability_id),
        capability_schema=str(raw.get("capability_schema") or MUTATION_REQUEST_SCHEMA),
        worktree_id=worktree_id,
        canonical_root=str(canonical),
        allowed_paths=tuple(raw.get("allowed_paths") or touched),
        expected_base_hashes=expected,
        operations=tuple(normalized_operations),
        payload_ref=raw.get("payload_ref") or hashlib.sha256(
            json.dumps(normalized_operations, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest(),
        lease_id=raw.get("lease_id"),
        verification_obligations=obligations,
        budget=budget,
        resource_reservation=resource_reservation,
    )


class MutationIntentStore:
    """Crash-boundary journal for mutation requests, under existing .hawking state."""

    def __init__(self, root: Union[str, os.PathLike[str]]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.directory = self.root / ".hawking" / "mutation-intents"

    def path_for(self, request_id: str) -> Path:
        safe = str(request_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", safe):
            raise MutationError("invalid mutation intent id")
        return self.directory / f"{safe}.json"

    def load(self, request_id: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(request_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            return None
        return value if isinstance(value, dict) else None

    def write(
        self,
        request: MutationRequest,
        *,
        status: str,
        files: Optional[Iterable[Mapping[str, Any]]] = None,
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        if status not in MUTATION_INTENT_STATUSES:
            raise MutationError(f"invalid mutation intent status: {status}")
        payload = {
            "schema": MUTATION_INTENT_SCHEMA,
            "request": request.to_dict(),
            "status": status,
            "updated_at": time.time(),
            "files": [dict(item) for item in (files or ()) if isinstance(item, Mapping)],
            "outcome": _json_copy(outcome or {}),
        }
        destination = self.path_for(request.request_id)
        atomic_write_json(destination, payload)
        return destination

    def update(
        self,
        request: MutationRequest,
        status: str,
        *,
        files: Optional[Iterable[Mapping[str, Any]]] = None,
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        return self.write(request, status=status, files=files, outcome=outcome)


@dataclass(frozen=True)
class PreparedMutation:
    request: MutationRequest
    paths: Tuple[Path, ...]
    snapshots: Mapping[str, Optional[bytes]]
    identities: Mapping[str, Optional[Mapping[str, Any]]]
    intent_path: Path


class MutationExecutor:
    """Shared precondition, atomic-write and intent machinery for all writers."""

    def __init__(self, root: Union[str, os.PathLike[str]]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.intents = MutationIntentStore(self.root)

    def resolve(self, raw: Any, *, allow_missing: bool = True) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise MutationError("mutation path is empty")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if candidate.is_symlink() and not candidate.exists():
            raise MutationError(f"dangling symlink leaf rejected: {text}")
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise MutationError(f"mutation path escapes canonical root: {text}") from exc
        if not relative.parts or any(part.casefold() in {".git", ".hawking"} for part in relative.parts):
            raise MutationError(f"protected mutation path rejected: {text}")
        if not allow_missing and not resolved.exists():
            raise MutationError(f"mutation path does not exist: {text}")
        return resolved

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace(os.sep, "/")

    def _files(self, paths: Iterable[Path], snapshots: Optional[Mapping[str, Optional[bytes]]] = None) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for path in paths:
            key = str(path)
            before = snapshots.get(key) if snapshots is not None else _path_bytes(path)
            after = _path_bytes(path)
            result.append({
                "path": self._relative(path),
                "sha256_before": _sha256_payload(before),
                "sha256_after": _sha256_payload(after),
                "changed": before != after,
            })
        return result

    def _check_expected(self, request: MutationRequest, paths: Iterable[Path]) -> None:
        for path in paths:
            relative = self._relative(path)
            if relative not in request.expected_base_hashes:
                continue
            actual = _sha256_payload(_path_bytes(path))
            expected = request.expected_base_hashes[relative]
            if actual != expected:
                raise MutationConflict(
                    f"expected base hash mismatch for {relative}: expected={expected} actual={actual}"
                )

    def begin(self, request: MutationRequest, paths: Iterable[Path]) -> PreparedMutation:
        if Path(request.canonical_root).expanduser().resolve() != self.root:
            raise MutationError("mutation request canonical root does not match executor root")
        resolved = tuple(self.resolve(path, allow_missing=True) for path in paths)
        unique: List[Path] = []
        seen = set()
        for path in resolved:
            if str(path) not in seen:
                unique.append(path)
                seen.add(str(path))
        # ``allowed_paths`` is the request's capability boundary, not merely
        # descriptive receipt metadata.  The builder defaults it to the
        # operation targets, while callers that provide a narrower admission
        # must explicitly include every target.  Exact-file matching keeps a
        # directory-shaped typo from widening a write lease.
        if not unique:
            raise MutationError("mutation request has no target paths")
        admitted = set(request.allowed_paths)
        if not admitted:
            raise MutationError("mutation request has no admitted allowed_paths")
        outside = [
            self._relative(path)
            for path in unique
            if self._relative(path) not in admitted
        ]
        if outside:
            raise MutationError(
                "mutation path is outside admitted allowed_paths: "
                + ", ".join(outside)
            )
        self._check_expected(request, unique)
        snapshots = {str(path): _path_bytes(path) for path in unique}
        identities = {str(path): _file_identity(path) for path in unique}
        files = [
            {
                "path": self._relative(path),
                "sha256_before": _sha256_payload(snapshots[str(path)]),
                "identity_before": identities[str(path)],
            }
            for path in unique
        ]
        intent_path = self.intents.write(
            request,
            status="NOT_STARTED",
            files=files,
            outcome={"stage": "admitted", "operation_sha256": request.to_dict()["operation_sha256"]},
        )
        return PreparedMutation(
            request=request,
            paths=tuple(unique),
            snapshots=snapshots,
            identities=identities,
            intent_path=intent_path,
        )

    def recheck(self, prepared: PreparedMutation) -> None:
        self._check_expected(prepared.request, prepared.paths)
        for path in prepared.paths:
            before_identity = prepared.identities[str(path)]
            current_identity = _file_identity(path)
            if before_identity != current_identity:
                raise MutationConflict(f"file identity changed before commit: {self._relative(path)}")

    def write_bytes(self, path: Union[str, os.PathLike[str]], payload: bytes) -> None:
        atomic_write_bytes(self.resolve(path), payload)

    def write_text(self, path: Union[str, os.PathLike[str]], text: str, *, encoding: str = "utf-8") -> None:
        try:
            payload = str(text).encode(encoding)
        except (LookupError, UnicodeEncodeError) as exc:
            raise MutationError(f"cannot encode mutation payload: {exc}") from exc
        atomic_write_bytes(self.resolve(path), payload)

    def remove_file(self, path: Union[str, os.PathLike[str]]) -> None:
        """Remove one already-admitted file during rollback.

        Rollback is still a mutation, so it must pass the same root/protected
        path resolver as a forward write.  The caller only uses this for a
        file that was absent at admission; it never recursively removes a
        directory or follows a dangling symlink.
        """
        target = self.resolve(path, allow_missing=True)
        if target.is_symlink():
            raise MutationError(f"rollback symlink target rejected: {path}")
        if target.exists() and not target.is_file():
            raise MutationError(f"rollback target is not a file: {path}")
        if target.is_file():
            target.unlink()

    def finish(
        self,
        prepared: PreparedMutation,
        status: str,
        *,
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        files = self._files(prepared.paths, prepared.snapshots)
        self.intents.update(prepared.request, status, files=files, outcome=outcome)
        return files

    def execute_file(
        self,
        request: MutationRequest,
        path: Union[str, os.PathLike[str]],
        content: str,
        *,
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        target = self.resolve(path, allow_missing=True)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        prepared = self.begin(request, [target])
        commit_started = False
        try:
            self.recheck(prepared)
            # Mark the crash boundary before the first publish.  If the
            # process dies in atomic_write_bytes, recovery must inspect the
            # tree instead of trusting the pre-commit NOT_STARTED marker.
            self.intents.update(
                request,
                "APPLIED",
                files=self._files(prepared.paths, prepared.snapshots),
                outcome={"stage": "commit_started"},
            )
            commit_started = True
            self.write_text(target, content, encoding=encoding)
            files = self.finish(prepared, "APPLIED", outcome={"stage": "atomic_publish"})
            verification = self._files(prepared.paths, prepared.snapshots)
            after = _path_bytes(target)
            before = prepared.snapshots[str(target)]
            expected_bytes = str(content).encode(encoding)
            bytes_match = after == expected_bytes
            if not bytes_match:
                raise MutationError("post-publish bytes differ from the requested content")
            self.intents.update(request, "VERIFIED", files=verification, outcome={"stage": "post_publish_verify"})
            return {
                "path": str(target),
                "bytes": len(after or b""),
                "before_sha256": _sha256_payload(before),
                "sha256": _sha256_payload(after),
                "changed": before != after,
                "atomic_publish": True,
                "files": verification,
                "changed_files": [item for item in verification if item.get("changed")],
                "verification": {"status": "VERIFIED", "bytes_match": bytes_match},
                "intent_status": "VERIFIED",
                "request": request.to_dict(),
            }
        except BaseException as exc:
            self.intents.update(
                request,
                "UNKNOWN_OUTCOME" if commit_started else "NOT_STARTED",
                files=self._files(prepared.paths, prepared.snapshots),
                outcome={"stage": "exception", "error": type(exc).__name__},
            )
            raise


def execute_mutation_proposal(
    proposal: MutationProposal | Mapping[str, Any],
    *,
    engine: Any,
    authority: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authorize provider data, then delegate effects to the canonical engine.

    This is intentionally a narrow adapter: a proposal from a model cannot
    create authority, widen its workspace, or select a different writer.
    """
    try:
        root = str(Path(getattr(engine, "root", "")).resolve())
        if isinstance(proposal, MutationProposal):
            value = proposal
        elif isinstance(proposal, Mapping) and proposal.get("status") == "MUTATION_PROPOSED":
            # Preserve provider-supplied identity fields long enough to reject
            # a lie about Goal/WorkUnit/root.  Canonical Hawking identities are
            # attached only after those comparisons below.
            raw_proposal = dict(proposal)
            # A provider may omit canonical_root because the WorkUnit packet
            # already binds it. Bind omission to the engine root; an explicit
            # conflicting root is still rejected below.
            raw_proposal.setdefault("canonical_root", root)
            value = MutationProposal.from_mapping(raw_proposal)
        elif isinstance(proposal, Mapping) and str(proposal.get("s") or proposal.get("status") or "").strip().upper() == "MUTATE":
            value = normalize_mutation_result(
                {**dict(proposal), "authority": authority},
                canonical_root=root,
                goal_id=str(authority.get("goal_id") or ""),
                workunit_id=str(authority.get("workunit_id") or ""),
            )
        elif isinstance(proposal, Mapping):
            value = MutationProposal.from_mapping(proposal)
        else:
            value = normalize_mutation_result(
                proposal,
                canonical_root=root,
                goal_id=str(authority.get("goal_id") or ""),
                workunit_id=str(authority.get("workunit_id") or ""),
            )
        requested_root = str(Path(value.canonical_root).expanduser().resolve()) if value.canonical_root else root
        required = {str(x) for x in authority.get("capabilities") or ()}
        if "repo.edit" not in required or not authority.get("mutation_lease"):
            return {"status": "rejected", "reason": "MUTATION_AUTHORITY_REQUIRED", "applied": False}
        authority_root = str(
            Path(authority.get("workspace_root") or root).expanduser().resolve()
        )
        if requested_root != root or authority_root != root:
            return {
                "status": "rejected",
                "reason": "WRONG_WORKSPACE",
                "applied": False,
                "detail": {
                    "requested_root": requested_root,
                    "engine_root": root,
                    "authority_root": authority_root,
                },
            }
        if value.goal_id and authority.get("goal_id") and value.goal_id != str(authority.get("goal_id")):
            return {"status": "rejected", "reason": "WORKUNIT_AUTHORITY_MISMATCH", "applied": False}
        if value.workunit_id and authority.get("workunit_id") and value.workunit_id != str(authority.get("workunit_id")):
            return {"status": "rejected", "reason": "WORKUNIT_AUTHORITY_MISMATCH", "applied": False}
        # A provider may serialize read/inspect/search observations next to
        # the actual edit.  They are useful provenance, but are not Engine
        # mutation operations.  Drop only this explicit read-only vocabulary;
        # every unknown/effectful operation remains subject to the canonical
        # engine's fail-closed validation below.
        effect_operations = tuple(
            operation for operation in value.operations
            if str(
                operation.get("op")
                or operation.get("operation")
                or operation.get("tool")
                or operation.get("tool_name")
                or ""
            )
            .strip().lower() not in _READ_ONLY_PROVIDER_OPERATIONS
        )
        if len(effect_operations) != len(value.operations):
            value = MutationProposal(
                objective=value.objective,
                operations=effect_operations,
                baseline_evidence_refs=tuple(value.baseline_evidence_refs),
                expected_base_hashes=dict(value.expected_base_hashes),
                required_tests=tuple(value.required_tests),
                broader_tests=tuple(value.broader_tests),
                canonical_root=value.canonical_root,
                goal_id=value.goal_id,
                workunit_id=value.workunit_id,
                source_digest=value.source_digest,
                completion_contract=dict(value.completion_contract),
            )
        if not value.operations:
            return {"status": "rejected", "reason": "EMPTY_MUTATION_PROPOSAL", "applied": False}
        allowed_paths = authority.get("allowed_paths") or ()
        if isinstance(allowed_paths, str):
            allowed_paths = [allowed_paths]
        if allowed_paths:
            allowed = {
                normalize_mutation_path(path, root=root)
                for path in allowed_paths
                if str(path).strip()
            }
            touched = {
                normalize_mutation_path(item.get("path"), root=root)
                for item in value.operations
                if isinstance(item, Mapping) and item.get("path") is not None
            }
            if not touched.issubset(allowed):
                return {
                    "status": "rejected",
                    "reason": "MUTATION_SCOPE_DENIED",
                    "applied": False,
                    "paths": sorted(touched - allowed),
                }
        expected = dict(value.expected_base_hashes)
        if value.source_digest and expected:
            bound = mutation_tree_digest(
                [{"path": p, "sha256_before": d} for p, d in expected.items()],
                hash_field="sha256_before",
            )
            if bound != value.source_digest:
                return {"status": "rejected", "reason": "STALE_MUTATION_PROPOSAL", "applied": False}
        # Rebuild the proposal with Hawking-owned identity fields before
        # compiling the request.  A provider may omit them or include lies,
        # but it cannot select the Goal/WorkUnit/Workspace authority.
        value = MutationProposal(
            objective=value.objective,
            operations=tuple(value.operations),
            baseline_evidence_refs=tuple(value.baseline_evidence_refs),
            expected_base_hashes=dict(value.expected_base_hashes),
            required_tests=tuple(value.required_tests),
            broader_tests=tuple(value.broader_tests),
            canonical_root=root,
            goal_id=str(authority.get("goal_id") or value.goal_id or ""),
            workunit_id=str(authority.get("workunit_id") or value.workunit_id or ""),
            source_digest=value.source_digest,
            completion_contract=dict(value.completion_contract),
        )
        verification_obligations = {
            "tests": list(value.required_tests),
            "inspect_diff": True,
            "result_packet": True,
            **dict(value.completion_contract),
        }
        # Compact anchored intents use `tests` as verification references;
        # they do not serialize a second literal test-file edit. Keep the
        # focused test execution obligation while removing the legacy paired
        # test-file requirement at this canonical compilation boundary.
        if value.completion_contract.get("compact_worker_schema") == "hawking.compact_worker.v1":
            verification_obligations["require_regression_test_mutation"] = False
        request = {
            "goal_id": value.goal_id,
            "workunit_id": value.workunit_id,
            "capability_id": "repo.edit",
            "canonical_root": root,
            "allowed_paths": [str(x.get("path") or "") for x in value.operations],
            "expected_base_hashes": expected,
            "lease_id": authority.get("mutation_lease"),
            "verification_obligations": verification_obligations,
        }
        result = engine.apply_typed_mutation(list(value.operations), list(value.required_tests), request=request)
        if (str(result.get("status") or "").lower() == "rejected"
                and "base hash mismatch" in str(result.get("reason") or "").lower()):
            return {
                "status": "rejected",
                "reason": "STALE_MUTATION_PROPOSAL",
                "applied": False,
                "execution": result,
            }
        return {
            "schema": "hawking.mutation_execution_packet.v1",
            "proposal": value.to_dict(),
            "status": result.get("status"),
            # Keep the adapter packet self-describing.  The engine is the
            # authority, but callers that need to project a canonical
            # repo.edit receipt should not have to know which fields were
            # nested in its result packet.
            "applied": result.get("applied"),
            "rolled_back": result.get("rolled_back"),
            "execution": result,
            "hawking_owned": True,
            "next_action": "propose_repair" if result.get("status") not in {"accepted", "unproven"} else "complete",
        }
    except MutationConflict:
        return {"status": "rejected", "reason": "STALE_MUTATION_PROPOSAL", "applied": False}
    except Exception as exc:
        # Preserve the fail-closed rejection, but give the bounded recovery
        # turn the concrete validation reason.  A bare exception class made
        # provider repair packets guess at missing anchors, operation shape,
        # or verification obligations and led to repeated unusable turns.
        detail = str(exc).strip().replace("\x00", " ")[:600]
        return {
            "status": "rejected",
            "reason": f"INVALID_MUTATION_PROPOSAL: {type(exc).__name__}",
            "detail": detail,
            "applied": False,
        }


def reconcile_mutation_intent(
    root: Union[str, os.PathLike[str]],
    request_id: str,
) -> Dict[str, Any]:
    """Reconcile an intent after a crash without blindly replaying the effect."""
    store = MutationIntentStore(root)
    document = store.load(request_id)
    if document is None:
        return {"request_id": request_id, "status": "MISSING", "reconciled": False}
    status = str(document.get("status") or "UNKNOWN_OUTCOME")
    if status not in {"UNKNOWN_OUTCOME", "APPLIED"}:
        return {"request_id": request_id, "status": status, "reconciled": True, "document": document}
    request_raw = document.get("request")
    if not isinstance(request_raw, Mapping):
        return {"request_id": request_id, "status": "UNKNOWN_OUTCOME", "reconciled": False, "reason": "request missing"}
    try:
        request = MutationRequest.from_mapping(request_raw)
        executor = MutationExecutor(root)
        paths = [executor.resolve(path, allow_missing=True) for path in request.allowed_paths]
    except Exception as exc:
        return {"request_id": request_id, "status": "UNKNOWN_OUTCOME", "reconciled": False, "reason": type(exc).__name__}
    files = executor._files(paths)
    expected_rows = [
        item for item in files
        if item.get("path") in request.expected_base_hashes
    ]
    before_match = bool(expected_rows) and all(
        item.get("sha256_after") == request.expected_base_hashes.get(item.get("path"))
        for item in expected_rows
    )
    recorded_after = {
        str(item.get("path")): item.get("sha256_after")
        for item in (document.get("files") or ())
        if isinstance(item, Mapping)
    }
    after_match = bool(recorded_after) and all(
        next((row.get("sha256_after") for row in files if row.get("path") == path), object()) == digest
        for path, digest in recorded_after.items()
    )
    outcome = document.get("outcome")
    stage = str(outcome.get("stage") or "") if isinstance(outcome, Mapping) else ""
    verified_stage = stage in {"post_publish_verify", "deterministic_verification"}
    if after_match and verified_stage:
        resolved_status = "VERIFIED"
    elif before_match:
        resolved_status = "NOT_STARTED"
    else:
        resolved_status = "UNKNOWN_OUTCOME"
    if resolved_status != "UNKNOWN_OUTCOME":
        store.update(request, resolved_status, files=files, outcome={"stage": "reconciled", "previous_status": status})
    return {
        "request_id": request_id,
        "status": resolved_status,
        "reconciled": resolved_status != "UNKNOWN_OUTCOME",
        "files": files,
        "candidate_matches_recorded_after": after_match,
        "verification_stage_observed": verified_stage,
    }


def _snapshot_file(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return {"path": path, "content": p.read_text(encoding="utf-8")}


def _restore_file(
    snapshot: Optional[Dict[str, Any]],
    path: Optional[str] = None,
) -> None:
    if snapshot is None:
        target = path
        if not target:
            return
        p = Path(target)
        if p.is_file():
            p.unlink()
        return
    p = Path(snapshot["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(snapshot["content"], encoding="utf-8")


def operation_text(operation: Dict[str, Any], field: str) -> Optional[str]:
    """The text of an operation field, however the model chose to send it.

    `new_text` is one JSON string and therefore carries every newline as an
    escape, which the resident gets wrong: receipt 97444aca burned four calls on
    "unexpected character after line continuation character" and "Invalid
    \\escape", because the model emitted `\\\\n` where it meant `\\n`.
    `new_lines` is the same content as a list of plain lines -- nothing to
    escape, so nothing to get wrong.

    This lives HERE, in the lower-level module, and `engine._operation_text`
    delegates to it, because two readers disagreeing about what an operation
    says is exactly the defect that lets a bad anchor reach an applier with the
    contract reporting no complaint. Before this, `apply_mutation_operations`
    read `old_text`/`new_text` directly and a line-form operation resolved to
    the EMPTY STRING -- an empty anchor, which matches everywhere.
    """
    lines = operation.get(f"{field.split('_')[0]}_lines")
    if isinstance(lines, list) and all(isinstance(x, str) for x in lines):
        return "\n".join(lines) + "\n" if lines else ""
    value = operation.get(field)
    return str(value) if value is not None else None


def _apply_create(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _apply_replace(content: str, old: str, new: str) -> str:
    if old not in content:
        raise MutationError("old_text not found")
    if content.count(old) > 1:
        raise MutationError("old_text not unique")
    if old == new:
        raise MutationError("NO_OP_MUTATION: replace old_text equals new_text")
    return content.replace(old, new, 1)


def _apply_insert(content: str, anchor: str, text: str, mode: str) -> str:
    if anchor not in content:
        raise MutationError("anchor not found")
    if content.count(anchor) > 1:
        raise MutationError("anchor not unique")
    if not text:
        raise MutationError("NO_OP_MUTATION: insert with empty text")
    if mode == "insert_before":
        return content.replace(anchor, text + anchor, 1)
    return content.replace(anchor, anchor + text, 1)


def apply_mutation_operations(guard: Any, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not operations:
        raise MutationError("NO_OP_MUTATION")
    if len(operations) > MAX_MUTATION_OPERATIONS:
        raise MutationError(f"too many operations: {len(operations)}")
    snapshots: Dict[str, Optional[Dict[str, Any]]] = {}
    changed: List[str] = []
    created: List[str] = []
    for op in operations:
        op_type = op.get("op")
        path = op.get("path")
        if not path:
            raise MutationError("missing path")
        full = guard.resolve(path)
        if op_type == "create":
            if full in snapshots:
                raise MutationError("duplicate create")
            snapshots[full] = _snapshot_file(full)
            _apply_create(full, op.get("content", ""))
            created.append(path)
        elif op_type in ("replace", "insert_before", "insert_after"):
            if full not in snapshots:
                snapshots[full] = _snapshot_file(full)
            p = Path(full)
            if not p.exists():
                raise MutationError("file not found")
            content = p.read_text(encoding="utf-8")
            if op_type == "replace":
                new_content = _apply_replace(
                    content,
                    operation_text(op, "old_text") or "",
                    operation_text(op, "new_text") or "",
                )
            else:
                new_content = _apply_insert(content, op.get("anchor", ""), op.get("text", ""), op_type)
            p.write_text(new_content, encoding="utf-8")
            if path not in changed:
                changed.append(path)
        else:
            raise MutationError(f"unknown op: {op_type}")
    files = []
    any_change = False
    for full, snap in snapshots.items():
        before = None if snap is None else snap["content"].encode("utf-8")
        p = Path(full)
        after = p.read_bytes() if p.exists() and p.is_file() else None
        files.append(
            {
                "path": full,
                "sha256_before": hashlib.sha256(before).hexdigest() if before is not None else None,
                "sha256_after": hashlib.sha256(after).hexdigest() if after is not None else None,
            }
        )
        if before != after:
            any_change = True
    if snapshots and not any_change:
        raise MutationError("NO_OP_MUTATION")
    test_paths = discover_tests(changed + created)
    result = {
        "operation_count": len(operations),
        "paths": changed + created,
        "changed": changed,
        "created": created,
        "snapshots": snapshots,
        "files": files,
        "rewrites_tests": bool(test_paths),
        "test_paths": test_paths,
    }
    result["content_hash"] = mutation_content_hash(result)
    return result


def mutation_content_hash(result: Dict[str, Any]) -> str:
    """Stable hash of before/after bytes for every file the mutation touched.

    Scheduler.complete can use this as the fingerprint so a no-op is not
    recorded as progress.
    """
    payload = [
        {
            "path": item.get("path"),
            "sha256_before": item.get("sha256_before"),
            "sha256_after": item.get("sha256_after"),
        }
        for item in (result.get("files") or [])
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_fingerprint(paths: List[str], root: Optional[str] = None) -> str:
    """Hash of current file contents. Identical trees share a fingerprint."""
    digest = hashlib.sha256()
    for raw in sorted(paths):
        digest.update(raw.encode("utf-8"))
        digest.update(b"\0")
        path = Path(raw)
        if root is not None and not path.is_absolute():
            path = Path(root) / raw
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"-")
        digest.update(b"\0")
    return digest.hexdigest()


def rollback_mutation(mutation_result: Dict[str, Any]) -> None:
    for full_path, snap in mutation_result.get("snapshots", {}).items():
        _restore_file(snap, full_path)


def compile_python_file(path: Union[str, Path]) -> Dict[str, Any]:
    """In-process syntax check. Same parser as ``python -m py_compile``.

    Does not write ``.pyc`` and does not spawn an interpreter. Isolation is
    not required: the source is already on disk in this process's workspace.
    """
    target = Path(path)
    try:
        source = target.read_bytes()
    except OSError as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{exc}\n",
        }
    try:
        compile(source, str(target), "exec")
    except SyntaxError as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "".join(traceback.format_exception_only(type(exc), exc)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}\n",
        }
    return {"ok": True, "exit_code": 0, "stdout": "", "stderr": ""}


def validate_python_syntax(path: str) -> bool:
    return bool(compile_python_file(path)["ok"])


def discover_tests(paths: List[str]) -> List[str]:
    patterns = [r"^test_.*\.py$", r"^.*_test\.py$", r"^.*\.test\.js$", r"^.*\.spec\.js$", r"^.*\.test\.ts$", r"^.*\.spec\.ts$"]
    return [p for p in paths if any(re.match(pat, os.path.basename(p)) for pat in patterns)]
