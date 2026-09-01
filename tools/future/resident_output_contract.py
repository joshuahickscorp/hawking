#!/usr/bin/env python3
"""The resident's reply cannot crash the loop, and a partial reply is a re-ask list.

hcli_sovereign.validate has already killed the live loop three times on shapes
the model is free to produce: a missing key, a dict where a list belongs, and
a parse-failure path that omitted fields. This module is the contract that
loop can later adopt. It does not edit the live loop.

admit(raw, schema) is the whole public surface. It takes an arbitrary object
(or None) and a schema, and ALWAYS returns the same keys. It never raises.
A truncated, malformed, or prose-wrapped reply is a result, not an exception.

Narrow re-ask: missing is the exact list of field names the reply still owes,
so the caller can ask for those rather than rerun the whole prompt.

This module does not choose a hypothesis, a tensor, a layer, or a next
experiment. It admits a shape.

    python3 tools/future/resident_output_contract.py --build
    python3 -m pytest tools/future/test_resident_output_contract.py -q
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import random
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/resident_output_contract.py"
RECEIPT_NAME = "RESIDENT_OUTPUT_CONTRACT.json"
SCHEMA_ID = "hawking.future.resident_output_contract.v1"

# Bounds so a junk reply cannot stall the caller. Salvage that would exceed
# these is reported as malformed, not looped until memory or the stack dies.
MAX_INPUT_CHARS = 1_000_000
MAX_DEPTH = 40
MAX_LIST_ITEMS = 256

RESULT_KEYS: tuple[str, ...] = (
    "ok",
    "value",
    "missing",
    "extra",
    "coerced",
    "errors",
    "parse",
    "schema_id",
    "reask",
)
PARSE_KEYS: tuple[str, ...] = ("ok", "kind", "recovered", "source_type")
REASK_KEYS: tuple[str, ...] = ("needed", "fields", "prompt_fragment")
PARSE_KINDS: tuple[str, ...] = (
    "none",
    "empty",
    "already_object",
    "already_array",
    "already_scalar",
    "json_object",
    "json_array",
    "json_scalar",
    "fenced",
    "prose_tail",
    "corrupt_tail",
    "truncated",
    "malformed",
    "non_json",
)

_TYPE_ALIASES = {
    "array": "list",
    "integer": "int",
    "boolean": "bool",
    "object": "object",
    "list": "list",
    "string": "string",
    "int": "int",
    "float": "float",
    "number": "number",
    "bool": "bool",
    "any": "any",
    "null": "null",
}

# The shape the live sovereign pack already asks the body to emit. Copied as
# machinery so a later adopt has a name; this is not a scientific choice.
SOVEREIGN_REPLY_SCHEMA: dict[str, Any] = {
    "id": "hawking.future.hcli_sovereign_reply.v1",
    "type": "object",
    "required": [
        "belief_update",
        "live_hypotheses",
        "selected_work",
        "escalation_needed",
    ],
    "additionalProperties": False,
    "properties": {
        "belief_update": {"type": "string", "nonempty": True},
        "live_hypotheses": {
            "type": "list",
            "items": {
                "type": "object",
                "required": ["id", "claim", "cheapest_falsifier"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "nonempty": True},
                    "claim": {"type": "string", "nonempty": True},
                    "cheapest_falsifier": {"type": "string", "nonempty": True},
                },
            },
        },
        "selected_work": {
            "type": "list",
            "items": {
                "type": "object",
                "required": ["type", "params", "why"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "nonempty": True},
                    "params": {
                        "type": "object",
                        "required": ["tensor", "layer", "side", "fraction"],
                        "additionalProperties": False,
                        "properties": {
                            "tensor": {"type": "string", "nonempty": True},
                            "layer": {"type": "int"},
                            "side": {"type": "string", "nonempty": True},
                            "fraction": {"type": "number"},
                        },
                    },
                    "why": {"type": "string", "nonempty": True},
                },
            },
        },
        "escalation_needed": {"type": "bool"},
    },
}

# An instance of that shape, used as a fixture. Not a recommended experiment.
VALID_SOVEREIGN_REPLY: dict[str, Any] = {
    "belief_update": "the conventional floor still stands",
    "live_hypotheses": [
        {
            "id": "H.example",
            "claim": "a fixture claim, not a selected hypothesis",
            "cheapest_falsifier": "a fixture falsifier, not a selected test",
        }
    ],
    "selected_work": [
        {
            "type": "PERTURB",
            "params": {
                "tensor": "gate",
                "layer": 0,
                "side": "rows",
                "fraction": 0.5,
            },
            "why": "fixture work item for the contract, not a scientific pick",
        }
    ],
    "escalation_needed": False,
}


class ContractRefused(RuntimeError):
    """A schema is missing or unusable; compiling it must not invent a shape."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def compile_schema(schema: Any) -> dict[str, Any]:
    """Refuse a missing schema rather than admitting everything."""
    if isinstance(schema, str):
        blob = schema.strip()
        if not blob:
            raise ContractRefused(
                "schema is missing; a contract with no shape would admit anything"
            )
        try:
            schema = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ContractRefused(f"schema is not JSON: {exc}") from exc
    if not isinstance(schema, Mapping) or not schema:
        raise ContractRefused(
            "schema is missing; a contract with no shape would admit anything"
        )
    if not any(k in schema for k in ("type", "properties", "required", "items")):
        raise ContractRefused(
            "schema has no type/properties/required/items; refusing to infer "
            "a contract from an instance"
        )
    return _normalize_spec(dict(schema), depth=0)


def _normalize_spec(spec: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
    if depth > MAX_DEPTH:
        raise ContractRefused(f"schema nests deeper than {MAX_DEPTH}")
    raw_t = spec.get("type", "object" if "properties" in spec or "required" in spec
                     else "list" if "items" in spec else "any")
    if isinstance(raw_t, list):
        raw_t = raw_t[0] if raw_t else "any"
    t = _TYPE_ALIASES.get(str(raw_t).lower())
    if t is None:
        raise ContractRefused(f"schema type {raw_t!r} is not a contract type")
    out: dict[str, Any] = {
        "type": t,
        "id": str(spec["id"]) if spec.get("id") else "",
        "required": [str(x) for x in (spec.get("required") or [])],
        "nonempty": bool(spec.get("nonempty", t == "string" and bool(spec.get("required")))),
        "additionalProperties": bool(spec.get("additionalProperties", False)),
        "properties": {},
        "items": None,
    }
    if t == "string" and "nonempty" in spec:
        out["nonempty"] = bool(spec["nonempty"])
    props = spec.get("properties") or {}
    if isinstance(props, Mapping):
        out["properties"] = {
            str(k): _normalize_spec(v if isinstance(v, Mapping) else {"type": "any"},
                                    depth=depth + 1)
            for k, v in props.items()
        }
    items = spec.get("items")
    if isinstance(items, Mapping):
        out["items"] = _normalize_spec(items, depth=depth + 1)
    elif items is not None:
        out["items"] = _normalize_spec({"type": "any"}, depth=depth + 1)
    elif t == "list":
        out["items"] = _normalize_spec({"type": "any"}, depth=depth + 1)
    return out


def _default_for(spec: Mapping[str, Any]) -> Any:
    t = spec.get("type") or "any"
    if t == "object":
        props = spec.get("properties") or {}
        return {k: _default_for(v) for k, v in props.items()}
    if t == "list":
        return []
    if t == "string":
        return ""
    if t == "int":
        return 0
    if t == "float" or t == "number":
        return 0.0
    if t == "bool":
        return False
    return None


# ---------------------------------------------------------------------------
# Extract a JSON value from anything, never raising.
# ---------------------------------------------------------------------------


def _source_type(raw: Any) -> str:
    return "NoneType" if raw is None else type(raw).__name__


def _parse_meta(kind: str, *, ok: bool, recovered: bool, source_type: str) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "kind": kind if kind in PARSE_KINDS else "malformed",
        "recovered": bool(recovered),
        "source_type": source_type,
    }


def _strip_fence(text: str) -> tuple[str, bool]:
    start = text.find("```")
    if start < 0:
        return text, False
    rest = text[start + 3:]
    nl = rest.find("\n")
    if nl < 0:
        return rest.strip(), True
    lang = rest[:nl].strip().lower()
    if lang and lang not in ("json", "javascript", "js"):
        # Still a fence; take the body. Language tags other than json happen.
        pass
    body = rest[nl + 1:]
    end = body.find("```")
    if end >= 0:
        body = body[:end]
    return body.strip(), True


def _tail_kind(rest: str) -> str:
    rest = rest.lstrip()
    if not rest:
        return "json_object"
    if rest[0] in ",}]\\" or rest[0] in "{[":
        return "corrupt_tail"
    return "prose_tail"


def _close_truncated(blob: str) -> Any | None:
    """Close open strings/brackets and fill a dangling colon with null."""
    if len(blob) > MAX_INPUT_CHARS:
        blob = blob[:MAX_INPUT_CHARS]
    stack: list[str] = []
    in_str = False
    esc = False
    after_colon = False
    after_comma = False
    for ch in blob:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch.isspace():
            continue
        if ch == '"':
            in_str = True
            after_colon = False
            after_comma = False
            continue
        if ch == "{":
            if len(stack) >= MAX_DEPTH:
                return None
            stack.append("}")
            after_colon = False
            after_comma = False
            continue
        if ch == "[":
            if len(stack) >= MAX_DEPTH:
                return None
            stack.append("]")
            after_colon = False
            after_comma = False
            continue
        if ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            after_colon = False
            after_comma = False
            continue
        if ch == ":":
            after_colon = True
            after_comma = False
            continue
        if ch == ",":
            after_comma = True
            after_colon = False
            continue
        after_colon = False
        after_comma = False
    repaired = blob
    if in_str:
        if esc:
            repaired += "u"
        repaired += '"'
    if after_colon:
        repaired += "null"
    elif after_comma:
        stripped = repaired.rstrip()
        if stripped.endswith(","):
            repaired = stripped[:-1]
    if len(stack) > MAX_DEPTH:
        return None
    repaired += "".join(reversed(stack))
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _kind_of(value: Any) -> str:
    if isinstance(value, dict):
        return "json_object"
    if isinstance(value, list):
        return "json_array"
    return "json_scalar"


# ONE character. The resident produced a complete, correct four-route option
# tree for EXAM A and a single doubled quote before a key - `""evidence_status":`
# - made the whole 1662-character reply unparseable, so every route was
# discarded. This repair is deliberately narrow: a doubled quote IMMEDIATELY
# before a key that is followed by a colon cannot be valid JSON under any
# reading, so collapsing it cannot change the meaning of a document that would
# otherwise have parsed. It is applied only AFTER strict parse, raw_decode and
# truncation salvage have all failed.
_DOUBLED_KEY_QUOTE = re.compile(r'(?<=[,{\s])""(?=[A-Za-z_][^"]*"\s*:)')


def _repair_doubled_key_quote(blob: str) -> Any | None:
    fixed, n = _DOUBLED_KEY_QUOTE.subn('"', blob)
    if not n:
        return None
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, RecursionError, ValueError, MemoryError):
        try:
            value, _end = json.JSONDecoder().raw_decode(fixed)
        except Exception:
            return None
        return value


def _loads_value(blob: str) -> tuple[Any, str, bool] | None:
    try:
        value = json.loads(blob)
        return value, _kind_of(value), False
    except json.JSONDecodeError:
        pass
    except (RecursionError, ValueError, MemoryError):
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(blob)
    except json.JSONDecodeError:
        value = None
    except (RecursionError, ValueError, MemoryError):
        return None
    else:
        rest = blob[end:]
        if rest.strip():
            return value, _tail_kind(rest), True
        return value, _kind_of(value), False
    salvaged = _close_truncated(blob)
    if salvaged is not None:
        return salvaged, "truncated", True
    repaired = _repair_doubled_key_quote(blob)
    if repaired is not None:
        return repaired, "repaired_doubled_quote", True
    try:
        value = ast.literal_eval(blob)
    except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError):
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value, _kind_of(value), True
    return None


def extract(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Best-effort JSON value plus a parse record. Never raises."""
    src = _source_type(raw)
    try:
        return _extract(raw, src)
    except Exception:
        return None, _parse_meta("malformed", ok=False, recovered=False, source_type=src)


def _extract(raw: Any, src: str) -> tuple[Any, dict[str, Any]]:
    if raw is None:
        return None, _parse_meta("none", ok=False, recovered=False, source_type=src)
    if isinstance(raw, dict):
        return raw, _parse_meta("already_object", ok=True, recovered=False, source_type=src)
    if isinstance(raw, list):
        return raw, _parse_meta("already_array", ok=True, recovered=False, source_type=src)
    if isinstance(raw, tuple):
        return list(raw), _parse_meta("already_array", ok=True, recovered=True, source_type=src)
    if isinstance(raw, (bool, int, float)):
        return raw, _parse_meta("already_scalar", ok=True, recovered=False, source_type=src)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raw = bytes(raw).decode("utf-8", errors="replace")
        src = "str"
    if not isinstance(raw, str):
        try:
            raw = str(raw)
        except Exception:
            return None, _parse_meta("non_json", ok=False, recovered=False, source_type=src)
    if len(raw) > MAX_INPUT_CHARS:
        raw = raw[:MAX_INPUT_CHARS]
    text = raw.strip()
    if not text:
        return None, _parse_meta("empty", ok=False, recovered=False, source_type=src)
    if text[0] == "\ufeff":
        text = text.lstrip("\ufeff").strip()
        if not text:
            return None, _parse_meta("empty", ok=False, recovered=False, source_type=src)
    fenced = False
    if "```" in text:
        text, fenced = _strip_fence(text)
        if not text:
            return None, _parse_meta("empty", ok=False, recovered=fenced, source_type=src)
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i >= 0]
    if starts:
        blob = text[min(starts):]
    else:
        blob = text
    got = _loads_value(blob)
    if got is None:
        kind = "non_json" if not starts else "malformed"
        return None, _parse_meta(kind, ok=False, recovered=False, source_type=src)
    value, kind, recovered = got
    if fenced and kind in ("json_object", "json_array", "json_scalar"):
        kind = "fenced"
        recovered = True
    return value, _parse_meta(kind, ok=True, recovered=recovered, source_type=src)


# ---------------------------------------------------------------------------
# Coerce a value onto a compiled schema. Accumulates missing/extra/coerced.
# ---------------------------------------------------------------------------


def _is_index_key(k: Any) -> bool:
    if isinstance(k, int) and k >= 0:
        return True
    return isinstance(k, str) and k.isdigit()


def _join(path: str, name: str) -> str:
    return name if not path else f"{path}.{name}"


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            if "." in s:
                f = float(s)
                if f.is_integer():
                    return int(f)
                return None
            return int(s, 10)
        except ValueError:
            return None
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError):
            return None
    return None


def _dict_to_list(value: dict[str, Any]) -> list[Any]:
    if value and all(_is_index_key(k) for k in value):
        n = max(int(k) for k in value) + 1
        if n > MAX_LIST_ITEMS:
            n = MAX_LIST_ITEMS
        return [value.get(str(i), value.get(i)) for i in range(n)]
    return [value]


def _coerce(value: Any, spec: Mapping[str, Any], path: str, acc: dict[str, Any],
            *, depth: int) -> Any:
    if depth > MAX_DEPTH:
        acc["errors"].append(f"{path or '<root>'}: nested deeper than {MAX_DEPTH}")
        return _default_for(spec)
    t = spec.get("type") or "any"
    if t == "any":
        return value
    if value is None:
        if not path:
            # Top-level null: every required field is missing.
            for name in spec.get("required") or []:
                acc["missing"].append(name)
            acc["errors"].append("value is null")
        else:
            acc["missing"].append(path)
        return _default_for(spec)

    if t == "object":
        got = value
        if isinstance(got, list):
            acc["coerced"].append(path or "<root>")
            if len(got) == 1 and isinstance(got[0], dict):
                got = got[0]
            elif got and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in got):
                try:
                    got = dict(got)
                except (TypeError, ValueError):
                    got = {}
            else:
                got = {}
        if isinstance(got, str):
            inner, meta = extract(got)
            if meta["ok"] and isinstance(inner, (dict, list)):
                acc["coerced"].append(path or "<root>")
                return _coerce(inner, spec, path, acc, depth=depth + 1)
        if not isinstance(got, dict):
            if path:
                acc["missing"].append(path)
            else:
                for name in spec.get("required") or []:
                    acc["missing"].append(name)
            acc["errors"].append(f"{path or '<root>'}: expected object, got {type(value).__name__}")
            return _default_for(spec)
        props: dict[str, Any] = spec.get("properties") or {}
        required = list(spec.get("required") or [])
        allow_extra = bool(spec.get("additionalProperties"))
        out: dict[str, Any] = {}
        for name, sub in props.items():
            here = _join(path, name)
            if name not in got:
                if name in required:
                    acc["missing"].append(here)
                out[name] = _default_for(sub)
                continue
            child = got[name]
            if child is None and name in required:
                acc["missing"].append(here)
                out[name] = _default_for(sub)
                continue
            out[name] = _coerce(child, sub, here, acc, depth=depth + 1)
        for name in required:
            if name not in props and name not in got:
                acc["missing"].append(_join(path, name))
        for name in got:
            if name not in props:
                if not allow_extra:
                    acc["extra"].append(_join(path, str(name)))
        return out

    if t == "list":
        got = value
        if isinstance(got, dict):
            acc["coerced"].append(path or "<root>")
            got = _dict_to_list(got)
        elif isinstance(got, tuple):
            acc["coerced"].append(path or "<root>")
            got = list(got)
        elif isinstance(got, str):
            inner, meta = extract(got)
            if meta["ok"] and isinstance(inner, list):
                acc["coerced"].append(path or "<root>")
                got = inner
            elif meta["ok"] and isinstance(inner, dict):
                acc["coerced"].append(path or "<root>")
                got = [inner]
            else:
                acc["errors"].append(f"{path or '<root>'}: expected list")
                if path:
                    acc["missing"].append(path)
                return []
        elif not isinstance(got, list):
            acc["errors"].append(
                f"{path or '<root>'}: expected list, got {type(value).__name__}"
            )
            if path:
                acc["missing"].append(path)
            return []
        item_spec = spec.get("items") or {"type": "any"}
        out_list: list[Any] = []
        for item in got[:MAX_LIST_ITEMS]:
            out_list.append(_coerce(item, item_spec, path, acc, depth=depth + 1))
        return out_list

    if t == "string":
        s = _as_string(value)
        if s is None:
            acc["missing"].append(path or "<root>")
            acc["errors"].append(f"{path or '<root>'}: expected string")
            return ""
        if s is not value and not isinstance(value, str):
            acc["coerced"].append(path or "<root>")
        if spec.get("nonempty") and not s.strip():
            acc["missing"].append(path or "<root>")
            return ""
        return s

    if t == "int":
        n = _as_int(value)
        if n is None:
            acc["missing"].append(path or "<root>")
            acc["errors"].append(f"{path or '<root>'}: expected int")
            return 0
        if n is not value and not (isinstance(value, int) and not isinstance(value, bool)):
            acc["coerced"].append(path or "<root>")
        return n

    if t == "number" or t == "float":
        n = _as_number(value)
        if n is None:
            acc["missing"].append(path or "<root>")
            acc["errors"].append(f"{path or '<root>'}: expected number")
            return 0.0
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            acc["coerced"].append(path or "<root>")
        return float(n) if t != "number" or isinstance(n, float) else n

    if t == "bool":
        b = _as_bool(value)
        if b is None:
            acc["missing"].append(path or "<root>")
            acc["errors"].append(f"{path or '<root>'}: expected bool")
            return False
        if not isinstance(value, bool):
            acc["coerced"].append(path or "<root>")
        return b

    if t == "null":
        return None
    return value


def _uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prompt_fragment(missing: list[str]) -> str:
    if not missing:
        return ""
    return "Supply only these fields as JSON: " + ", ".join(missing)


def _result(*, ok: bool, value: Any, missing: list[str], extra: list[str],
            coerced: list[str], errors: list[str], parse: dict[str, Any],
            schema_id: str) -> dict[str, Any]:
    missing = _uniq(missing)
    extra = _uniq(extra)
    coerced = _uniq([c for c in coerced if c is not None])
    return {
        "ok": bool(ok) and not missing,
        "value": value,
        "missing": missing,
        "extra": extra,
        "coerced": coerced,
        "errors": list(errors),
        "parse": {
            k: parse[k] for k in PARSE_KEYS
        } if all(k in parse for k in PARSE_KEYS) else _parse_meta(
            str(parse.get("kind") or "malformed"),
            ok=bool(parse.get("ok")),
            recovered=bool(parse.get("recovered")),
            source_type=str(parse.get("source_type") or "unknown"),
        ),
        "schema_id": schema_id or "",
        "reask": {
            "needed": bool(missing),
            "fields": list(missing),
            "prompt_fragment": _prompt_fragment(missing),
        },
    }


def _blank(raw: Any, *, errors: list[str], parse: dict[str, Any] | None = None,
           schema_id: str = "", value: Any = None, missing: list[str] | None = None
           ) -> dict[str, Any]:
    return _result(
        ok=False,
        value={} if value is None else value,
        missing=list(missing or []),
        extra=[],
        coerced=[],
        errors=errors,
        parse=parse or _parse_meta("none", ok=False, recovered=False,
                                   source_type=_source_type(raw)),
        schema_id=schema_id,
    )


# ---------------------------------------------------------------------------
# Public surface: one function, one shape, never raises.
# ---------------------------------------------------------------------------


def admit(raw: Any, schema: Any) -> dict[str, Any]:
    """Admit an arbitrary model reply against a schema. Never raises.

    Always returns RESULT_KEYS. missing is the exact list of field names a
    caller can re-ask for; value is always the schema's shape so a later
    `result["value"]["selected_work"]` cannot KeyError.
    """
    try:
        return _admit(raw, schema)
    except Exception as exc:
        return _blank(
            raw,
            errors=[f"internal {type(exc).__name__}: {exc}"],
            parse=_parse_meta("malformed", ok=False, recovered=False,
                              source_type=_source_type(raw)),
        )


def _admit(raw: Any, schema: Any) -> dict[str, Any]:
    try:
        spec = compile_schema(schema)
    except ContractRefused as exc:
        extracted, parse = extract(raw)
        return _blank(
            raw,
            errors=[str(exc)],
            parse=parse,
            value=extracted if extracted is not None else {},
        )
    schema_id = spec.get("id") or (schema.get("id") if isinstance(schema, Mapping) else "") or ""
    extracted, parse = extract(raw)
    acc: dict[str, Any] = {"missing": [], "extra": [], "coerced": [], "errors": []}
    if not parse["ok"] or extracted is None and spec.get("type") == "object":
        # Unparsed reply: fill the schema shape so callers never see omitted keys.
        value = _coerce(None if extracted is None else extracted, spec, "", acc, depth=0)
        if not acc["missing"] and spec.get("required"):
            acc["missing"] = list(spec["required"])
        if parse["kind"] in ("none", "empty", "malformed", "non_json"):
            acc["errors"].append("reply did not parse as the schema's JSON value")
        return _result(
            ok=False,
            value=value,
            missing=acc["missing"] or list(spec.get("required") or []),
            extra=acc["extra"],
            coerced=acc["coerced"],
            errors=acc["errors"],
            parse=parse,
            schema_id=schema_id,
        )
    value = _coerce(extracted, spec, "", acc, depth=0)
    ok = parse["ok"] and not acc["missing"]
    return _result(
        ok=ok,
        value=value,
        missing=acc["missing"],
        extra=acc["extra"],
        coerced=acc["coerced"],
        errors=acc["errors"],
        parse=parse,
        schema_id=schema_id,
    )


# ---------------------------------------------------------------------------
# Named shapes the receipt and the tests share. Not a scientific menu.
# ---------------------------------------------------------------------------


def named_shapes() -> list[dict[str, Any]]:
    valid = VALID_SOVEREIGN_REPLY
    valid_json = json.dumps(valid, ensure_ascii=False)
    truncated = valid_json[: max(40, len(valid_json) // 2)]
    nested: Any = {"leaf": "junk"}
    for _ in range(24):
        nested = {"n": nested, "xs": [nested]}
    return [
        {"name": "valid_json", "raw": valid_json},
        {"name": "malformed_json", "raw": '{"belief_update": "x", this is not json'},
        {"name": "truncated_json", "raw": truncated},
        {"name": "prose_tail", "raw": valid_json + "\nHope this helps! — a prose tail"},
        {"name": "corrupt_tail", "raw": valid_json + "{not closed, ::: ###"},
        {"name": "dict_where_list_belongs", "raw": {
            **valid, "selected_work": valid["selected_work"][0]
        }},
        {"name": "list_where_dict_belongs", "raw": {
            **valid,
            "selected_work": [{
                **valid["selected_work"][0],
                "params": [valid["selected_work"][0]["params"]],
            }],
        }},
        {"name": "missing_required_fields", "raw": {"belief_update": "only this key"}},
        {"name": "extra_fields", "raw": {**valid, "commentary": "drop me"}},
        {"name": "null_values", "raw": {
            "belief_update": None,
            "live_hypotheses": None,
            "selected_work": None,
            "escalation_needed": None,
        }},
        {"name": "unicode", "raw": {
            **valid,
            "belief_update": "下限は 2.508 BPW — 日本語 🧪 snowman ☃",
        }},
        {"name": "deeply_nested_junk", "raw": {**valid, "junk": nested}},
        {"name": "empty_string", "raw": ""},
        {"name": "none", "raw": None},
        {"name": "fenced_json", "raw": "Here you go:\n```json\n" + valid_json + "\n```\n"},
        {"name": "whitespace_only", "raw": " \n\t "},
        {"name": "json_array_at_top", "raw": [valid]},
        {"name": "bytes_utf8", "raw": valid_json.encode("utf-8")},
        {"name": "parse_failure_omits_nothing", "raw": "not json at all, no braces"},
        {"name": "partially_valid", "raw": {
            "belief_update": "a real sentence",
            "live_hypotheses": [{"id": "H.partial", "claim": "has no falsifier"}],
            "escalation_needed": False,
        }},
    ]


def mutate_payload(rng: random.Random, base: Mapping[str, Any]) -> Any:
    """One random mutation of a valid object, including non-JSON junk."""
    kind = rng.choice((
        "drop_key", "null_key", "list_for_dict", "dict_for_list",
        "truncate", "prose", "corrupt", "fence", "unicode_prefix",
        "extra_key", "empty", "none", "bytes", "int", "deep",
        "malformed", "true_false", "params_list", "scalar_string",
    ))
    obj = json.loads(json.dumps(base))
    keys = list(obj)
    if kind == "drop_key" and keys:
        obj.pop(keys[rng.randrange(len(keys))])
        return obj
    if kind == "null_key" and keys:
        obj[keys[rng.randrange(len(keys))]] = None
        return obj
    if kind == "list_for_dict":
        obj["selected_work"] = obj.get("selected_work") or []
        if isinstance(obj["selected_work"], list) and obj["selected_work"]:
            item = obj["selected_work"][0]
            if isinstance(item, dict) and "params" in item:
                item["params"] = [item["params"]]
        return obj
    if kind == "dict_for_list":
        sw = obj.get("selected_work")
        if isinstance(sw, list) and sw and isinstance(sw[0], dict):
            obj["selected_work"] = sw[0]
        else:
            obj["selected_work"] = {"type": "PERTURB"}
        return obj
    blob = json.dumps(obj, ensure_ascii=False)
    if kind == "truncate":
        cut = rng.randint(1, max(1, len(blob) - 1))
        return blob[:cut]
    if kind == "prose":
        return blob + "\n\n" + rng.choice(("thanks", "Hope this helps.", "— done —"))
    if kind == "corrupt":
        return blob + rng.choice(("{", "}}}[", ",,,", "\\u", '"""'))
    if kind == "fence":
        return "```json\n" + blob + "\n```"
    if kind == "unicode_prefix":
        return "說明：\n" + blob
    if kind == "extra_key":
        obj[rng.choice(("note", "commentary", "debug", "emoji🧪"))] = rng.choice(
            ("x", 1, None, [1, 2], {"k": "v"})
        )
        return obj
    if kind == "empty":
        return rng.choice(("", "   ", "\n"))
    if kind == "none":
        return None
    if kind == "bytes":
        return blob.encode("utf-8")
    if kind == "int":
        return rng.randint(-5, 99)
    if kind == "deep":
        junk: Any = obj
        for _ in range(rng.randint(8, 20)):
            junk = {"n": junk}
        return junk
    if kind == "malformed":
        return blob.replace('"', "", 2) + "{:}"
    if kind == "true_false":
        return blob.replace("false", "False").replace("true", "True")
    if kind == "params_list":
        return obj  # already handled; fall through
    if kind == "scalar_string":
        return rng.choice(("yes", "[]", "null", "0"))
    return obj


def property_probe(n: int = 256, *, seed: int = 0) -> dict[str, Any]:
    """Feed n mutated payloads. Counts only; never a hardware number."""
    rng = random.Random(seed)
    schema = SOVEREIGN_REPLY_SCHEMA
    n_raised = 0
    n_key_mismatch = 0
    kinds: dict[str, int] = {}
    for i in range(n):
        payload = mutate_payload(rng, VALID_SOVEREIGN_REPLY)
        try:
            result = admit(payload, schema)
        except Exception:  # noqa: BLE001 — the probe's whole job is to count this
            n_raised += 1
            continue
        if tuple(result) != RESULT_KEYS:
            n_key_mismatch += 1
        if tuple(result.get("parse") or ()) != PARSE_KEYS:
            n_key_mismatch += 1
        if tuple(result.get("reask") or ()) != REASK_KEYS:
            n_key_mismatch += 1
        kind = str((result.get("parse") or {}).get("kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "n": n,
        "seed": seed,
        "n_raised": n_raised,
        "n_key_mismatch": n_key_mismatch,
        "never_raised": n_raised == 0,
        "always_same_keyset": n_key_mismatch == 0,
        "kinds": dict(sorted(kinds.items())),
    }


def battery() -> tuple[list[dict[str, Any]], int]:
    rows = []
    n_raised = 0
    for shape in named_shapes():
        try:
            result = admit(shape["raw"], SOVEREIGN_REPLY_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — battery exists to count this
            n_raised += 1
            rows.append({
                "name": shape["name"],
                "ok": False,
                "missing": [],
                "extra": [],
                "coerced": [],
                "parse_kind": "malformed",
                "parse_ok": False,
                "reask_fields": [],
                "keys": [],
                "value_keys": [],
                "selected_work_is_list": False,
                "raised": type(exc).__name__,
            })
            continue
        rows.append({
            "name": shape["name"],
            "ok": result["ok"],
            "missing": result["missing"],
            "extra": result["extra"],
            "coerced": result["coerced"],
            "parse_kind": result["parse"]["kind"],
            "parse_ok": result["parse"]["ok"],
            "reask_fields": result["reask"]["fields"],
            "keys": list(result),
            "value_keys": sorted(result["value"]) if isinstance(result["value"], dict) else [],
            "selected_work_is_list": isinstance(
                (result["value"] or {}).get("selected_work"), list
            ) if isinstance(result["value"], dict) else False,
        })
    return rows, n_raised


def build() -> dict[str, Any]:
    rows, n_raised = battery()
    probe = property_probe(256, seed=0)
    keys_ok = all(r["keys"] == list(RESULT_KEYS) for r in rows)
    work_is_list = all(r["selected_work_is_list"] for r in rows)
    partial = next(r for r in rows if r["name"] == "partially_valid")
    return {
        "obligation": "RESIDENT_OUTPUT_CONTRACT",
        "authority": "S033; live hcli_sovereign.validate crash shapes",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "schema": SCHEMA_ID,
        "question": "Can any malformed model reply crash or stall the sovereign loop?",
        "answer": (
            "Not through this contract. admit() never raises, always returns "
            "RESULT_KEYS, fills the schema shape so selected_work is always a "
            "list, and names the exact missing fields for a narrow re-ask. "
            "hcli_sovereign.py is live and is not edited here; adoption is a "
            "later step."
        ),
        "what_this_does_not_do": (
            "choose a hypothesis, a tensor, a layer, a fraction, or a next "
            "experiment. It admits a shape. Scientific selection stays with "
            "the resident."
        ),
        "live_crashes_this_admits": {
            "missing_key": "missing_required_fields: value still has every schema key",
            "dict_where_list_belongs": (
                "selected_work as a dict is coerced to a one-element list; "
                "the live loop crashed slicing a dict"
            ),
            "parse_failure_omitted_fields": (
                "parse_failure_omits_nothing / none / empty_string: the result "
                "still carries every RESULT_KEY; the live loop crashed on a "
                "shape that omitted counts"
            ),
        },
        "result_keys": list(RESULT_KEYS),
        "parse_keys": list(PARSE_KEYS),
        "reask_keys": list(REASK_KEYS),
        "named_shapes": rows,
        "n_named_shapes": len(rows),
        "named_always_same_keyset": keys_ok,
        "named_selected_work_always_list": work_is_list,
        "named_n_raised": n_raised,
        "property_probe": probe,
        "narrow_reask_example": {
            "shape": "partially_valid",
            "missing": partial["missing"],
            "reask_fields": partial["reask_fields"],
            "needed": bool(partial["reask_fields"]),
        },
        "untested": [
            "whether the live hcli_sovereign loop has adopted admit(); it has not",
            "whether a terse re-ask that names only missing fields changes the body's next reply (UNTESTED)",
        ],
        "resident_callable": {
            "entry_point": "tools.future.resident_output_contract.admit",
            "schema": "tools.future.resident_output_contract.SOVEREIGN_REPLY_SCHEMA",
            "receipt": f"receipts/future/{RECEIPT_NAME}",
            "fails_closed": "never raises; missing fields named on the result",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        path = write_receipt(RECEIPT_NAME, doc, RECORDED_BY)
        print(path)
        return 0
    print(json.dumps({
        "question": doc["question"],
        "answer": doc["answer"],
        "n_named_shapes": doc["n_named_shapes"],
        "named_always_same_keyset": doc["named_always_same_keyset"],
        "property_probe": doc["property_probe"],
        "narrow_reask_example": doc["narrow_reask_example"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
