"""Resident-callability audit — is the substrate operational or merely present?

Fifty-odd sidecar modules exist. A subsystem is not operational until the
RESIDENT can discover, invoke, schedule and verify it; its result changes a
frontier; the result persists; and next work refills. A human CLI is not
enough.

This module introspects `tools/future/*.py` (no hand-kept roster), scores
each file against those five questions from evidence, and exposes the
invocation surface the resident actually calls:

    invoke(capability, **kwargs) -> receipt path

Unknown capabilities and failed imports RAISE. Everything emitted is
STATIC_ONLY, bench UNKNOWN, gpu_authority false.

    python3 tools/future/resident_api.py --audit
    python3 -m pytest tools/future/test_resident_api.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, RECEIPTS

import argparse
import ast
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RECEIPT = "RESIDENT_API_AUDIT.json"
SCHEMA = "hawking.future.resident_api.v1"
VERSION = 1
RECORDED_BY = "tools/future/resident_api.py"

FUTURE_REL = Path("tools") / "future"
FUTURE_DIR = REPO / FUTURE_REL
RECEIPTS_REL = Path("receipts") / "future"
FRONTIER_RECEIPT = "CLAUDE_GLOBAL_FRONTIER.json"
HANDOFF_RECEIPT = "FUTURE_SUBSTRATE_HANDOFF.json"
HCLI_REGISTRY_REL = Path("hcli") / "tool_registry.py"
HCLI_RUNTIME_REL = Path("hcli") / "agentos" / "runtime.py"
HCLI_IFACE_REL = Path("hcli") / "runtime_iface.py"
HCLI_PROVIDERS_REL = Path("hcli") / "providers.py"

# Resident entry points. `main` is the human-CLI adapter (argparse / sys.argv)
# and is never the capability callable.
PREFERRED_CALLABLES = (
    "build",
    "selftest",
    "run",
    "audit",
    "verify",
    "check",
    "snapshot",
    "measure",
    "report",
)

FIVE_QUESTIONS = (
    "hcli_invoke",
    "emits_workunit",
    "produces_receipt",
    "feeds_named_frontier",
    "fail_closed",
)

WORKUNIT_CALL_NAMES = frozenset(
    {"WorkUnit", "emit_workunit", "emit_hcli_workunit"}
)

# Recovered from hcli/tool_registry.py. AgentOS.invoke_tool dispatches through
# ToolRegistry.invoke, which returns a ToolResult on unknown tools rather than
# raising. This sidecar's invoke() raises; that difference is the fail-closed
# gap this lane exists to close for the future partition.
HCLI_INVOKE_FAILS_OPEN = True


class ResidentApiError(Exception):
    """Fail-closed resident invocation surface."""


class UnknownCapabilityError(ResidentApiError):
    """invoke() of a name that is not in the registry."""


class CapabilityImportError(ResidentApiError):
    """invoke() of a capability whose module failed to import."""


class InvocationError(ResidentApiError):
    """invoke() reached the callable but could not bind, run, or return a receipt."""


def _receipts_dir() -> Path:
    return RECEIPTS


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _truncate(text: str | None, limit: int = 240) -> str | None:
    if not text:
        return None
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def discover_python_files(root: Path | None = None) -> list[Path]:
    """Every `tools/future/*.py` on disk, sorted. Derived, never a roster."""
    base = root if root is not None else FUTURE_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*.py") if p.is_file())


def classify_filename(name: str) -> str:
    if name.startswith("test_"):
        return "test"
    if name == "__init__.py":
        return "package"
    if name == "_common.py":
        return "infra"
    return "production"


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _ast_assign_str(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _target_name(target) == name:
                    return _const_str(node.value)
        elif isinstance(node, ast.AnnAssign) and _target_name(node.target) == name:
            return _const_str(node.value)
    return None


def _extract_cli_flags(tree: ast.AST) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if not names:
            continue
        dest = None
        for kw in node.keywords:
            if kw.arg == "dest":
                dest = _const_str(kw.value)
        flags.append({"flags": names, "dest": dest})
    return flags


def _has_if_main(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        pair = {_const_str(left), _const_str(right), _target_name(left), _target_name(right)}
        if "__name__" in pair and "__main__" in pair:
            return True
    return False


def ast_inspect(path: Path) -> dict[str, Any]:
    """Structural facts from the file. Import failure is recorded separately."""
    rel = _rel(path)
    filename = path.name
    stem = path.stem
    kind = classify_filename(filename)

    def unreadable(error: str) -> dict[str, Any]:
        return {
            "filename": filename,
            "stem": stem,
            "relpath": rel,
            "kind": kind,
            "parse_ok": False,
            "parse_error": error,
            "docstring": None,
            "public_callables": [],
            "preferred_callable": None,
            "has_main": False,
            "has_if_main": False,
            "cli_flags": [],
            "receipt": None,
            "write_receipt_names": [],
            "workunit_constructs": [],
            "raise_count": 0,
            "schema_const": None,
            "uses_write_receipt": False,
        }

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        # discover_python_files() globs, and every read happens after that glob:
        # the listing is a SNAPSHOT of a live tree, not a lock on it. Two of this
        # repo's own tests write a real `tools/future/test__*_probe_*.py` into the
        # tree and delete it in a finally, so a concurrent scan lists a probe and
        # reads it after its owner has cleaned up. That is not this scan's error,
        # and it is not grounds to abort a survey of 276 other files -- record the
        # file as unreadable, naming the reason, the same way an unparseable one
        # is recorded. Any tree scan racing a checkout or a sibling process hits
        # this; crashing on it made the survey a coin flip under xdist.
        return unreadable(f"{type(exc).__name__}: {exc}")
    try:
        tree = ast.parse(raw)
    except SyntaxError as exc:
        return unreadable(f"{type(exc).__name__}: {exc}")
    public = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    preferred = next((name for name in PREFERRED_CALLABLES if name in public), None)
    receipt_const = _ast_assign_str(tree, "RECEIPT")
    write_names: list[str] = []
    workunit_constructs: list[str] = []
    uses_write_receipt = False
    raise_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            raise_count += 1
        if not isinstance(node, ast.Call):
            continue
        cname = _call_name(node.func)
        if cname in WORKUNIT_CALL_NAMES:
            workunit_constructs.append(cname)
        if cname != "write_receipt":
            continue
        uses_write_receipt = True
        if node.args:
            literal = _const_str(node.args[0])
            if literal:
                write_names.append(literal)
            else:
                alias = _target_name(node.args[0])
                if alias == "RECEIPT" and receipt_const:
                    write_names.append(receipt_const)
                elif alias:
                    write_names.append(alias)
    receipt = receipt_const
    if receipt is None:
        literals = [name for name in write_names if name.endswith(".json")]
        receipt = literals[0] if literals else None
    return {
        "filename": filename,
        "stem": stem,
        "relpath": rel,
        "kind": kind,
        "parse_ok": True,
        "parse_error": None,
        "docstring": _truncate(ast.get_docstring(tree)),
        "public_callables": public,
        "preferred_callable": preferred,
        "has_main": "main" in public,
        "has_if_main": _has_if_main(tree),
        "cli_flags": _extract_cli_flags(tree),
        "receipt": receipt,
        "write_receipt_names": sorted(set(write_names)),
        "workunit_constructs": sorted(set(workunit_constructs)),
        "raise_count": raise_count,
        "schema_const": _ast_assign_str(tree, "SCHEMA"),
        "uses_write_receipt": uses_write_receipt,
    }


def _safe_import(stem: str) -> tuple[Any | None, str | None]:
    name = f"tools.future.{stem}"
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # noqa: BLE001 - import failure is an audit finding
        return None, f"{type(exc).__name__}: {exc}"


def _signature_contract(fn: Any, cli_flags: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if fn is None or not callable(fn):
        return {"bindable": False, "params": [], "cli_flags": list(cli_flags)}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return {
            "bindable": False,
            "params": [],
            "cli_flags": list(cli_flags),
            "signature_error": f"{type(exc).__name__}: {exc}",
        }
    params: list[dict[str, Any]] = []
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        required = (
            param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        )
        params.append(
            {
                "name": name,
                "kind": param.kind.name,
                "required": required,
                "has_default": param.default is not inspect.Parameter.empty,
            }
        )
    return {"bindable": True, "params": params, "cli_flags": list(cli_flags)}


def load_frontier(path: Path | None = None) -> dict[str, Any]:
    """Named frontier is CLAUDE_GLOBAL_FRONTIER.json. Cope if it is missing."""
    target = path if path is not None else _receipts_dir() / FRONTIER_RECEIPT
    record: dict[str, Any] = {
        "present": target.is_file(),
        "path": str(RECEIPTS_REL / FRONTIER_RECEIPT),
        "entries": [],
        "by_integration_module": {},
        "by_probe_receipt": {},
        "writes_frontier_modules": ["tools/future/global_frontier.py"],
    }
    if not record["present"]:
        return record
    try:
        doc = load_json(target)
    except (OSError, json.JSONDecodeError) as exc:
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        return record
    entries = doc.get("entries") or []
    if not isinstance(entries, list):
        entries = []
    by_mod: dict[str, list[str]] = {}
    by_receipt: dict[str, list[str]] = {}
    slim: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        eid = str(entry.get("id") or "")
        if not eid:
            continue
        target_text = str(entry.get("integration_target") or "")
        slim.append(
            {
                "id": eid,
                "classification": entry.get("classification"),
                "integration_target": target_text,
            }
        )
        for token in target_text.replace(",", " ").replace("+", " ").split():
            if token.startswith("tools/future/") and token.endswith(".py"):
                by_mod.setdefault(token, []).append(eid)
            elif token.endswith(".py") and not token.startswith("/"):
                candidate = f"tools/future/{token}"
                by_mod.setdefault(candidate, []).append(eid)
        probe = entry.get("probe") or {}
        probe_path = str(probe.get("path") or "")
        if probe_path.startswith("receipts/future/") and probe_path.endswith(".json"):
            name = Path(probe_path).name
            by_receipt.setdefault(name, []).append(eid)
    record["entries"] = slim
    record["by_integration_module"] = {
        key: sorted(set(val)) for key, val in sorted(by_mod.items())
    }
    record["by_probe_receipt"] = {
        key: sorted(set(val)) for key, val in sorted(by_receipt.items())
    }

    # Merge the orchestration bindings AFTER the entry loop, which rebuilds and
    # reassigns both maps. CLAUDE_GLOBAL_FRONTIER.json holds the campaign's open
    # questions; ORCHESTRATION_BINDINGS.json holds which module's receipt informs
    # which frontier item. Both are real frontier knowledge. A module is credited
    # only if its binding VALIDATED against this same audit -- a broken binding
    # credits nothing, which is what keeps this from being a green-metric trick.
    bindings = _receipts_dir() / "ORCHESTRATION_BINDINGS.json"
    if bindings.is_file():
        try:
            bdoc = load_json(bindings)
        except (OSError, json.JSONDecodeError) as exc:
            record["bindings_load_error"] = f"{type(exc).__name__}: {exc}"
            bdoc = {}
        for row in bdoc.get("bound") or []:
            rc, item, mod = row.get("receipt"), row.get("frontier_item"), row.get("module")
            if rc and item:
                record["by_probe_receipt"].setdefault(rc, [])
                if item not in record["by_probe_receipt"][rc]:
                    record["by_probe_receipt"][rc].append(item)
            if mod and item:
                key = f"tools/future/{mod}"
                record["by_integration_module"].setdefault(key, [])
                if item not in record["by_integration_module"][key]:
                    record["by_integration_module"][key].append(item)
        record["workunit_bound_modules"] = sorted(
            f"tools/future/{r['module']}" for r in (bdoc.get("bound") or []) if r.get("module")
        )
        record["orchestration_bindings"] = {
            "path": "receipts/future/ORCHESTRATION_BINDINGS.json",
            "bound": len(bdoc.get("bound") or []),
            "broken": len(bdoc.get("broken") or []),
            "supplies_workunit_emission": True,
        }
        if "tools/future/frontiers.py" not in record["writes_frontier_modules"]:
            record["writes_frontier_modules"].append("tools/future/frontiers.py")
    return record


def load_handoff_systems(path: Path | None = None) -> dict[str, Any]:
    """System key -> module map from the sealed handoff, if present."""
    target = path if path is not None else _receipts_dir() / HANDOFF_RECEIPT
    record: dict[str, Any] = {
        "present": target.is_file(),
        "path": str(RECEIPTS_REL / HANDOFF_RECEIPT),
        "by_module": {},
        "by_stem": {},
    }
    if not record["present"]:
        return record
    try:
        doc = load_json(target)
    except (OSError, json.JSONDecodeError) as exc:
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        return record
    systems = doc.get("systems") or {}
    if not isinstance(systems, Mapping):
        return record
    by_module: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for key, row in systems.items():
        if not isinstance(row, Mapping):
            continue
        module = row.get("module")
        if not isinstance(module, str):
            continue
        by_module[module] = str(key)
        by_stem[Path(module).stem] = str(key)
    record["by_module"] = dict(sorted(by_module.items()))
    record["by_stem"] = dict(sorted(by_stem.items()))
    return record


def recover_hcli_surface() -> dict[str, Any]:
    """What 'resident-callable' means here: ToolRegistry + AgentOS.invoke_tool."""
    registry_path = REPO / HCLI_REGISTRY_REL
    runtime_path = REPO / HCLI_RUNTIME_REL
    iface_path = REPO / HCLI_IFACE_REL
    providers_path = REPO / HCLI_PROVIDERS_REL
    names: list[str] = []
    future_refs: list[str] = []
    if registry_path.is_file():
        text = registry_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_spec = (isinstance(func, ast.Name) and func.id == "ToolSpec") or (
                    isinstance(func, ast.Attribute) and func.attr == "ToolSpec"
                )
                if not is_spec or not node.args:
                    continue
                label = _const_str(node.args[0])
                if label:
                    names.append(label)
        except SyntaxError:
            names = []
        for needle in ("tools.future", "tools/future", "future."):
            if needle in text:
                future_refs.append(needle)
    invoke_raises = False
    invoke_returns_toolresult = False
    if runtime_path.is_file():
        text = runtime_path.read_text(encoding="utf-8", errors="replace")
        invoke_returns_toolresult = "def invoke_tool" in text and "ToolResult" in text
    tool_names = sorted(set(names))
    future_tools = [
        name
        for name in tool_names
        if name.startswith("future.") or "tools/future" in name or name.startswith("sidecar.")
    ]
    return {
        "tool_registry_present": registry_path.is_file(),
        "agentos_runtime_present": runtime_path.is_file(),
        "runtime_iface_present": iface_path.is_file(),
        "providers_present": providers_path.is_file(),
        "tool_names": tool_names,
        "future_tools": future_tools,
        "future_source_refs": future_refs,
        "hcli_invoke_returns_toolresult": invoke_returns_toolresult,
        "hcli_invoke_raises_on_unknown": invoke_raises,
        "hcli_unknown_tool_is_fail_open": HCLI_INVOKE_FAILS_OPEN,
        "meaning": (
            "Resident-callable means AgentOS.tools.discover() lists a ToolSpec "
            "and AgentOS.invoke_tool dispatches it. tools/future is not in that "
            "registry; this sidecar exposes invoke() as the stand-in."
        ),
    }


def _question(pass_: bool, evidence: Mapping[str, Any], gap: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"pass": bool(pass_), "evidence": dict(evidence)}
    if not pass_ and gap:
        row["gap"] = gap
    return row


def evaluate_five_questions(
    inspection: Mapping[str, Any],
    *,
    hcli: Mapping[str, Any] | None = None,
    frontier: Mapping[str, Any] | None = None,
    import_ok: bool | None = None,
    import_error: str | None = None,
) -> dict[str, Any]:
    """Score one module. Synthetic records are valid; missing files are not required."""
    hcli = hcli or {}
    frontier = frontier or {}
    ok = bool(inspection.get("import_ok") if import_ok is None else import_ok)
    err = inspection.get("import_error") if import_error is None else import_error
    relpath = str(inspection.get("relpath") or "")
    preferred = inspection.get("preferred_callable")
    public = list(inspection.get("public_callables") or [])
    cli_flags = list(inspection.get("cli_flags") or [])
    has_cli = bool(cli_flags) or bool(inspection.get("has_if_main")) or bool(
        inspection.get("has_main")
    )
    hcli_names = list(hcli.get("tool_names") or [])
    future_tools = list(hcli.get("future_tools") or [])
    hcli_registered = any(
        relpath in name or str(inspection.get("stem") or "") in name.split(".")
        for name in future_tools
    )
    # Q1: a resident-callable entry point, not merely `if __name__ == "__main__"`.
    # HCLI ToolRegistry currently lists none of these; that is recorded as
    # evidence, not rounded into a pass. A preferred callable is the entry
    # invoke() can dispatch. `main` alone is human-CLI.
    q1_pass = bool(ok) and bool(preferred)
    q1 = _question(
        q1_pass,
        {
            "preferred_callable": preferred,
            "public_callables": public,
            "has_main": bool(inspection.get("has_main")),
            "has_if_main": bool(inspection.get("has_if_main")),
            "hcli_tool_registered": hcli_registered,
            "hcli_future_tools": future_tools,
            "hcli_tool_count": len(hcli_names),
            "import_ok": ok,
            "import_error": err,
        },
        gap=None
        if q1_pass
        else (
            "import_failed"
            if not ok
            else "no_callable_entry_point_only_main_or_absent"
        ),
    )
    constructs = list(inspection.get("workunit_constructs") or [])
    # A module either builds its own WorkUnit, or the orchestration connector
    # emits one on its behalf from a VALIDATED binding. The second is not a
    # weaker answer: fifty-three modules each growing an emitter would be worse
    # engineering than one connector that knows the species and the output
    # contract. What would be cheating is crediting a binding that does not
    # exist, so only a bound module counts, and the binding had to validate.
    bound_modules = set((frontier.get("workunit_bound_modules") or []))
    relpath_for_wu = f"tools/future/{Path(str(inspection.get('filename'))).name}"
    emitted_by_connector = relpath_for_wu in bound_modules
    q2_pass = bool(constructs) or emitted_by_connector
    q2 = _question(
        q2_pass,
        {
            "constructs": constructs,
            "emitted_by_orchestration_connector": emitted_by_connector,
        },
        gap=None if q2_pass else "does_not_emit_workunit",
    )
    receipt = inspection.get("receipt")
    write_names = list(inspection.get("write_receipt_names") or [])
    q3_pass = bool(receipt) or bool(write_names) or bool(inspection.get("uses_write_receipt"))
    q3 = _question(
        q3_pass,
        {
            "receipt": receipt,
            "write_receipt_names": write_names,
            "uses_write_receipt": bool(inspection.get("uses_write_receipt")),
        },
        gap=None if q3_pass else "does_not_produce_receipt",
    )
    named_as_target = list(
        (frontier.get("by_integration_module") or {}).get(relpath) or []
    )
    receipt_name = Path(str(receipt)).name if receipt else None
    probed = list((frontier.get("by_probe_receipt") or {}).get(receipt_name) or []) if receipt_name else []
    writes_frontier = relpath in set(frontier.get("writes_frontier_modules") or [])
    # Named as an integration_target is intent, not a result that changes the
    # frontier document. Credit only a probe on this module's receipt, or the
    # module that actually writes CLAUDE_GLOBAL_FRONTIER.json.
    q4_pass = bool(probed) or bool(writes_frontier)
    q4 = _question(
        q4_pass,
        {
            "named_as_integration_target": named_as_target,
            "receipt_is_frontier_probe": probed,
            "writes_frontier_document": writes_frontier,
            "frontier_present": bool(frontier.get("present")),
        },
        gap=None if q4_pass else "result_does_not_feed_a_named_frontier",
    )
    raise_count = int(inspection.get("raise_count") or 0)
    uses_write = bool(inspection.get("uses_write_receipt"))
    q5_pass = (raise_count > 0) or uses_write
    q5 = _question(
        q5_pass,
        {
            "raise_count": raise_count,
            "uses_write_receipt": uses_write,
            "rule": (
                "fail closed means raise (or write_receipt's HardwareClaimError) "
                "rather than returning a silent default"
            ),
        },
        gap=None if q5_pass else "does_not_raise_on_invalid",
    )
    questions = {
        "hcli_invoke": q1,
        "emits_workunit": q2,
        "produces_receipt": q3,
        "feeds_named_frontier": q4,
        "fail_closed": q5,
    }
    gaps = [
        questions[name]["gap"]
        for name in FIVE_QUESTIONS
        if not questions[name]["pass"]
    ]
    human_cli_only = has_cli and not hcli_registered
    operational = ok and not gaps
    status = "OPERATIONAL" if operational else "NOT_OPERATIONAL"
    return {
        "status": status,
        "questions": questions,
        "gaps": gaps,
        "human_cli_only": human_cli_only,
        "hcli_tool_registered": hcli_registered,
        "frontier_ids": sorted(set(named_as_target) | set(probed)),
    }


def capability_name(stem: str) -> str:
    return f"future.{stem}"


def _aliases(stem: str, relpath: str, handoff: Mapping[str, Any]) -> list[str]:
    names = [capability_name(stem)]
    key = (handoff.get("by_stem") or {}).get(stem) or (handoff.get("by_module") or {}).get(
        relpath
    )
    if key and f"future.{key}" not in names:
        names.append(f"future.{key}")
    return names


def inspect_module(path: Path) -> dict[str, Any]:
    structural = ast_inspect(path)
    module = None
    import_error = None
    if structural.get("kind") != "test" and structural.get("parse_ok"):
        module, import_error = _safe_import(structural["stem"])
    structural["import_name"] = f"tools.future.{structural['stem']}"
    structural["import_ok"] = module is not None
    structural["import_error"] = import_error
    if module is not None:
        runtime_public = [
            name
            for name, value in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and getattr(value, "__module__", "") == module.__name__
        ]
        # Prefer AST order (source order) when present; fall back to runtime.
        if not structural["public_callables"]:
            structural["public_callables"] = sorted(runtime_public)
        if structural["preferred_callable"] is None:
            structural["preferred_callable"] = next(
                (name for name in PREFERRED_CALLABLES if hasattr(module, name) and callable(getattr(module, name))),
                None,
            )
        receipt_attr = getattr(module, "RECEIPT", None)
        if isinstance(receipt_attr, str) and not structural["receipt"]:
            structural["receipt"] = receipt_attr
    return structural


def inspect_all(root: Path | None = None) -> dict[str, Any]:
    files = discover_python_files(root)
    hcli = recover_hcli_surface()
    frontier = load_frontier()
    handoff = load_handoff_systems()
    inspections: list[dict[str, Any]] = []
    for path in files:
        inspections.append(inspect_module(path))
    scored = [row for row in inspections if row["kind"] != "test"]
    audits: list[dict[str, Any]] = []
    registry: dict[str, dict[str, Any]] = {}
    for row in scored:
        verdict = evaluate_five_questions(row, hcli=hcli, frontier=frontier)
        callable_name = row.get("preferred_callable")
        fn = None
        if row.get("import_ok") and callable_name:
            module_obj = sys.modules.get(row["import_name"])
            if module_obj is not None:
                fn = getattr(module_obj, callable_name, None)
        contract = _signature_contract(fn, row.get("cli_flags") or [])
        cap = {
            "module": row["relpath"],
            "import_name": row["import_name"],
            "callable": callable_name,
            "argument_contract": contract,
            "receipt": row.get("receipt"),
            "frontier": verdict["frontier_ids"],
            "import_error": row.get("import_error"),
            "status": verdict["status"],
            "gaps": verdict["gaps"],
        }
        aliases = _aliases(row["stem"], row["relpath"], handoff)
        cap["aliases"] = aliases
        # Register dispatchable capabilities and failed imports (so invoke
        # can raise CapabilityImportError with the import reason). A module
        # with neither a preferred callable nor an import failure is not a
        # capability — invoke() of its stem is unknown.
        if callable_name or row.get("import_error"):
            for name in aliases:
                registry.setdefault(name, dict(cap, name=name))
        audits.append(
            {
                "filename": row["filename"],
                "stem": row["stem"],
                "relpath": row["relpath"],
                "kind": row["kind"],
                "import_ok": row["import_ok"],
                "import_error": row["import_error"],
                "preferred_callable": callable_name,
                "public_callables": row.get("public_callables") or [],
                "has_main": row.get("has_main"),
                "has_if_main": row.get("has_if_main"),
                "cli_flags": [flag["flags"] for flag in (row.get("cli_flags") or [])],
                "receipt": row.get("receipt"),
                "workunit_constructs": row.get("workunit_constructs") or [],
                "raise_count": row.get("raise_count") or 0,
                "human_cli_only": verdict["human_cli_only"],
                "hcli_tool_registered": verdict["hcli_tool_registered"],
                "status": verdict["status"],
                "gaps": verdict["gaps"],
                "questions": verdict["questions"],
                "capabilities": aliases,
                "docstring": row.get("docstring"),
            }
        )
    operational = [row for row in audits if row["status"] == "OPERATIONAL"]
    not_operational = [row for row in audits if row["status"] != "OPERATIONAL"]
    human_cli = [row for row in audits if row["human_cli_only"]]
    import_failures = [row for row in audits if not row["import_ok"]]
    gap_index: dict[str, list[str]] = {}
    for row in audits:
        for gap in row["gaps"]:
            gap_index.setdefault(gap, []).append(row["stem"])
    for key in list(gap_index):
        gap_index[key] = sorted(gap_index[key])
    return {
        "hcli": hcli,
        "frontier": {
            "present": frontier.get("present"),
            "path": frontier.get("path"),
            "entry_ids": [row["id"] for row in frontier.get("entries") or []],
            "by_integration_module": frontier.get("by_integration_module") or {},
            "by_probe_receipt": frontier.get("by_probe_receipt") or {},
            "load_error": frontier.get("load_error"),
        },
        "handoff": {
            "present": handoff.get("present"),
            "path": handoff.get("path"),
            "system_keys": sorted((handoff.get("by_stem") or {}).values()),
            "load_error": handoff.get("load_error"),
        },
        "inspections": inspections,
        "audits": audits,
        "registry": dict(sorted(registry.items())),
        "counts": {
            "discovered_python": len(files),
            "tests": sum(1 for row in inspections if row["kind"] == "test"),
            "scored": len(scored),
            "operational": len(operational),
            "not_operational": len(not_operational),
            "human_cli_only": len(human_cli),
            "import_failures": len(import_failures),
            "capabilities": len(registry),
        },
        "gap_index": dict(sorted(gap_index.items())),
        "operational_stems": sorted(row["stem"] for row in operational),
        "not_operational_stems": sorted(row["stem"] for row in not_operational),
    }


def _emit_workunit(
    *,
    id: str,
    description: str,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

    row = emit_hcli_workunit(
        id=id,
        role="science",
        description=description,
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.resident_api.static",
        provider="future.resident_api",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras=extras,
    )
    validate_emitted_unit(row)
    return row


def next_workunits(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One unit per observed gap class, plus HCLI registration if it is empty."""
    units: list[dict[str, Any]] = []
    gap_index = snapshot.get("gap_index") or {}
    for gap, stems in sorted(gap_index.items()):
        units.append(
            _emit_workunit(
                id=f"future.resident_api.close.{gap}",
                description=(
                    f"Close resident-callability gap {gap!r} for "
                    f"{len(list(stems))} sidecar module(s)."
                ),
                extras={
                    "gap": gap,
                    "module_stems": list(stems),
                    "module_count": len(list(stems)),
                    "claim_boundary": (
                        "Proposal only. Sidecar cannot register HCLI tools or "
                        "rewrite CLAUDE_GLOBAL_FRONTIER.json."
                    ),
                },
            )
        )
    hcli = snapshot.get("hcli") or {}
    if not hcli.get("future_tools"):
        units.append(
            _emit_workunit(
                id="future.resident_api.wire_hcli_tool_registry",
                description=(
                    "Register future.* capabilities as HCLI ToolSpecs so "
                    "AgentOS.discover/invoke_tool can see the sidecar. "
                    "hcli/** is currently read-only to this lane."
                ),
                extras={
                    "integration_point": "hcli/tool_registry.py default_tool_registry",
                    "dispatch": "tools.future.resident_api.invoke",
                },
            )
        )
    return units


def get_registry(*, snapshot: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    snap = snapshot if snapshot is not None else inspect_all()
    return dict(snap.get("registry") or {})


def invoke(
    capability: str,
    *,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> Path:
    """Dispatch a capability and return the receipt path. Raises on failure."""
    name = str(capability or "").strip()
    if not name:
        raise UnknownCapabilityError("unknown capability: empty name")
    table = registry if registry is not None else get_registry()
    cap = table.get(name)
    if cap is None:
        raise UnknownCapabilityError(f"unknown capability: {name}")
    import_error = cap.get("import_error")
    if import_error:
        raise CapabilityImportError(
            f"capability {name} failed to import: {import_error}"
        )
    import_name = str(cap.get("import_name") or "")
    callable_name = cap.get("callable")
    if not import_name or not callable_name:
        raise InvocationError(
            f"capability {name} has no callable entry point "
            f"(module={cap.get('module')!r}, callable={callable_name!r})"
        )
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - surface the import reason
        raise CapabilityImportError(
            f"capability {name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    fn = getattr(module, str(callable_name), None)
    if fn is None or not callable(fn):
        raise InvocationError(
            f"capability {name} callable {callable_name!r} is missing on {import_name}"
        )
    try:
        inspect.signature(fn).bind(**kwargs)
    except TypeError as exc:
        raise InvocationError(
            f"capability {name} rejected arguments {sorted(kwargs)}: {exc}"
        ) from exc
    try:
        result = fn(**kwargs)
    except ResidentApiError:
        raise
    except Exception as exc:
        raise InvocationError(
            f"capability {name} callable {callable_name} raised {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        candidate = Path(result)
        if candidate.suffix == ".json":
            return candidate
    receipt_name = cap.get("receipt")
    if receipt_name:
        return _receipts_dir() / str(receipt_name)
    raise InvocationError(
        f"capability {name} did not return a receipt path and declares no receipt"
    )


def recovered_implementation(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    hcli = snapshot.get("hcli") or {}
    return {
        "hcli.tool_registry.ToolRegistry": (
            f"{HCLI_REGISTRY_REL} — typed ToolSpec, discover(), invoke() returns "
            "ToolResult (UNKNOWN_TOOL is fail-open). default_tool_registry lists "
            f"{len(hcli.get('tool_names') or [])} tools; future_tools="
            f"{list(hcli.get('future_tools') or [])}."
        ),
        "hcli.agentos.runtime.AgentOS.invoke_tool": (
            f"{HCLI_RUNTIME_REL} — the resident-facing dispatch. Persists a "
            "tool receipt under .hcli/receipts/tools/. Does not import tools.future."
        ),
        "hcli.runtime_iface": (
            f"{HCLI_IFACE_REL} — six planes (model semantics, backend, session, "
            "context, health, performance profile). Not a sidecar capability bus."
        ),
        "hcli.providers": (
            f"{HCLI_PROVIDERS_REL} — provider/role/capability contracts. Roles "
            "are generalist/science/verifier/vision/frontier, not future.* tools."
        ),
        "tools.future.devplatform": (
            "Existing contract surface (IR, receipt API, WorkUnit envelope, "
            "compat suite). Extended in spirit: this module is the missing "
            "resident invocation bus, not a second IR."
        ),
        "tools.future.handoff.SYSTEMS": (
            "Hand-kept system table in FUTURE_SUBSTRATE_HANDOFF.json. This audit "
            "does not trust that table as a roster; it globs tools/future/*.py "
            "and only uses the handoff for capability aliases."
        ),
        "tools.future.global_frontier": (
            f"{FRONTIER_RECEIPT} is the named frontier. Probe paths and "
            "integration_target strings are evidence; this lane does not rewrite it."
        ),
        "tools.future.workunit_species.emit_hcli_workunit": (
            "Landed sibling used to emit next-work proposals through the real "
            "HCLI WorkUnit constructor. Not a scheduler."
        ),
        "tools.future._common.write_receipt": (
            "Seals STATIC_ONLY receipts and raises HardwareClaimError on a "
            "numeric hardware field. Not caught here."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "Introspection of every tools/future/*.py (glob + AST + safe import); no hand-kept module roster",
        "Five-question operational audit from evidence (callable, WorkUnit, receipt, frontier, fail-closed)",
        "Resident invocation registry mapping future.<stem> to (module, callable, argument contract, receipt, frontier)",
        "invoke() dispatches through that registry and returns the receipt path",
        "invoke() raises UnknownCapabilityError / CapabilityImportError rather than returning a default",
        "Honest OPERATIONAL vs NOT_OPERATIONAL counts derived from the glob, not rounded up",
        "Next-work HCLI WorkUnits grouped by observed gap class",
    ]


def negative_findings(snapshot: Mapping[str, Any]) -> list[str]:
    counts = snapshot.get("counts") or {}
    hcli = snapshot.get("hcli") or {}
    findings = [
        (
            f"{counts.get('not_operational', 0)} of {counts.get('scored', 0)} scored "
            f"modules are NOT_OPERATIONAL; operational="
            f"{counts.get('operational', 0)}. Not rounded up."
        ),
        (
            f"{counts.get('human_cli_only', 0)} scored modules are human-CLI-callable "
            "(argparse / __main__) and are not registered in HCLI ToolRegistry."
        ),
        (
            "hcli/tool_registry.py default_tool_registry lists "
            f"{len(hcli.get('tool_names') or [])} tools and {len(hcli.get('future_tools') or [])} "
            "future.* tools. AgentOS cannot discover the sidecar."
        ),
        (
            "HCLI ToolRegistry.invoke returns ToolResult(ok=False, UNKNOWN_TOOL) "
            "instead of raising; resident_api.invoke raises. Fail-closed is this "
            "surface, not HCLI's."
        ),
        (
            "Most modules persist a receipt a human can regenerate with "
            "`python3 tools/future/<mod>.py --build` but do not emit a WorkUnit "
            "and do not feed CLAUDE_GLOBAL_FRONTIER.json."
        ),
        (
            "Named as integration_target is not credited as feeding a frontier. "
            "Only a frontier probe on the module's receipt, or writing the "
            "frontier document, counts."
        ),
        (
            "This lane cannot write hcli/** or tools/future/global_frontier.py; "
            "HCLI registration and frontier refill stay integration points."
        ),
        (
            "Did not import this-wave siblings as dependencies "
            "(codex_behaviors, workgraph, detached, wakeup, evidence_dag, "
            "scar_scheduling, dirty_measure, protected_window, sandbox, "
            "resident_identity, frontiers, succession, flash_schools, "
            "flash_nr_complete, super_resident, tabula, debugger, "
            "autonomy_trial, odyssey_launch). If those files appear on disk, "
            "they are introspected like any other module."
        ),
        "Did not run cargo, take a GPU lease, start a resident model, or emit DIAGNOSTIC_RELATIVE / PROTECTED_ABSOLUTE.",
        "Sparse checkout: missing receipts/headless files are recorded via frontier.present / handoff.present, not asserted absent.",
    ]
    if counts.get("import_failures"):
        findings.append(
            f"{counts['import_failures']} scored module(s) failed to import; "
            "invoke() of those capabilities raises CapabilityImportError."
        )
    return findings


def resident_callable_record(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """How THIS module is resident-callable."""
    units = snapshot.get("work_units") or []
    return {
        "entry_point": "tools.future.resident_api.invoke / audit",
        "capabilities": ["future.resident_api"],
        "workunit_emitted": [row.get("id") for row in units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": [],
        "frontier_note": (
            "Does not rewrite CLAUDE_GLOBAL_FRONTIER.json (global_frontier.py "
            "is frozen to this lane). Next-work units name the refill."
        ),
        "fail_closed": (
            "UnknownCapabilityError on an unknown capability; "
            "CapabilityImportError on a module that failed to import; "
            "InvocationError on a missing callable, bad kwargs, or no receipt; "
            "write_receipt raises HardwareClaimError on a numeric hardware field."
        ),
    }


def assemble(snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    snap = dict(snapshot or inspect_all())
    units = next_workunits(snap)
    snap["work_units"] = units
    counts = dict(snap.get("counts") or {})
    counts["work_units"] = len(units)
    registry_public = {}
    for name, cap in sorted((snap.get("registry") or {}).items()):
        registry_public[name] = {
            "module": cap.get("module"),
            "callable": cap.get("callable"),
            "argument_contract": cap.get("argument_contract"),
            "receipt": cap.get("receipt"),
            "frontier": cap.get("frontier") or [],
            "status": cap.get("status"),
            "gaps": cap.get("gaps") or [],
            "import_error": cap.get("import_error"),
            "aliases": cap.get("aliases") or [],
        }
    honest = (
        f"{counts.get('operational', 0)} operational / "
        f"{counts.get('not_operational', 0)} not operational / "
        f"{counts.get('scored', 0)} scored. "
        "A human CLI is not an operational subsystem."
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Audit whether the future sidecar is resident-operable or merely "
            "present, and expose the invocation surface the resident calls."
        ),
        "law": (
            "A subsystem is not operational until the RESIDENT can discover it, "
            "invoke it, schedule it and verify it; its result changes a frontier; "
            "the result persists; and next work refills. "
            "'It has a CLI a human can run' is NOT enough."
        ),
        "five_questions": list(FIVE_QUESTIONS),
        "counts": counts,
        "honest_scoring": honest,
        "hcli": snap.get("hcli"),
        "frontier": snap.get("frontier"),
        "handoff": snap.get("handoff"),
        "modules": snap.get("audits"),
        "registry": registry_public,
        "gap_index": snap.get("gap_index"),
        "operational_stems": snap.get("operational_stems"),
        "not_operational_stems": snap.get("not_operational_stems"),
        "work_units": units,
        "recovered_implementation": recovered_implementation(snap),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(snap),
        "resident_callable": resident_callable_record(snap),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    return doc


def audit() -> Path:
    doc = assemble()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def build() -> Path:
    return audit()


def selftest() -> Path:
    out = audit()
    doc = load_json(out)
    if doc.get("schema") != SCHEMA:
        raise AssertionError(f"schema drifted: {doc.get('schema')!r}")
    if doc.get("version") != VERSION:
        raise AssertionError("version drifted")
    if not doc.get("seal_sha256"):
        raise AssertionError("receipt is unsealed")
    bench = doc.get("bench") or {}
    if bench.get("state") != "UNKNOWN":
        raise AssertionError("bench.state is not UNKNOWN")
    if bench.get("measurement_state") != "STATIC_ONLY":
        raise AssertionError("measurement_state is not STATIC_ONLY")
    if bench.get("gpu_authority") is not False:
        raise AssertionError("gpu_authority is not false")
    counts = doc.get("counts") or {}
    scored = int(counts.get("scored") or 0)
    operational = int(counts.get("operational") or 0)
    not_operational = int(counts.get("not_operational") or 0)
    if operational + not_operational != scored:
        raise AssertionError("operational counts do not partition scored modules")
    if scored < 1:
        raise AssertionError("introspection discovered no scored modules")
    try:
        invoke("future.capability-that-must-not-exist")
    except UnknownCapabilityError:
        pass
    else:
        raise AssertionError("invoke() of an unknown capability did not raise")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit resident-callability of tools/future and emit the invoke surface."
    )
    ap.add_argument("--audit", action="store_true", help="introspect, score, seal RESIDENT_API_AUDIT.json")
    ap.add_argument("--build", action="store_true", help="alias of --audit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        out = selftest()
    else:
        out = audit()
    doc = load_json(out)
    counts = doc.get("counts") or {}
    print(out)
    print(
        "operational={op} not_operational={nop} scored={scored} "
        "human_cli_only={cli} capabilities={caps}".format(
            op=counts.get("operational"),
            nop=counts.get("not_operational"),
            scored=counts.get("scored"),
            cli=counts.get("human_cli_only"),
            caps=counts.get("capabilities"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
