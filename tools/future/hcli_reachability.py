"""The HCLI scientific reachability graph (G025).

Sixteen autonomy rounds surfaced sixteen defects and most were one family: a
tool HCLI could not reach, a required argument it could not construct from any
reachable tool's output, or a result it could not persist.  Discovering those
one round at a time costs ~810 s each.  This audits the whole graph statically.

The graph is built from the SOURCE, not from round logs.  Harvesting observed
outputs from the twenty rounds that already ran would reproduce their defects
by construction and prove nothing about the ones nobody has hit yet.

For every registered tool it answers:

  discoverable          is the tool in a catalog the model is actually shown
  required_args         which input_schema fields are mandatory
  produced_keys         which keys the handler's `return {...}` literals emit
  arg_sources           for each required arg, which tools can produce it

and flags REQUIRED_ARGUMENT_WITH_NO_SOURCE, ORPHAN_PRODUCER (nothing consumes
what it emits), ORPHAN_CONSUMER, CAPABILITY_WITH_ZERO_CALLER and
OUTPUT_THAT_CANNOT_BE_RECORDED (a measurement with no durable write path).
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "hcli" / "tool_registry.py"
ENGINE = REPO / "hcli" / "engine.py"

# A tool whose result is only ever read back by a human is not "recordable" in
# the sense the autonomy loop needs.  These are the tools that write to disk.
WRITE_DOOR_HINT = re.compile(r"record|write|land|commit|measure|append", re.I)


def _const_str(node: ast.AST) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _schema_required(node: ast.AST) -> List[str]:
    """Pull `required` out of a dict-literal JSON schema. Non-literal -> []."""
    if not isinstance(node, ast.Dict):
        return []
    for k, v in zip(node.keys, node.values):
        if _const_str(k) == "required" and isinstance(v, (ast.List, ast.Tuple)):
            return [s for s in (_const_str(e) for e in v.elts) if s]
    return []


def _schema_props(node: ast.AST) -> List[str]:
    if not isinstance(node, ast.Dict):
        return []
    for k, v in zip(node.keys, node.values):
        if _const_str(k) == "properties" and isinstance(v, ast.Dict):
            return [s for s in (_const_str(e) for e in v.keys) if s]
    return []


def _returned_keys(fn: ast.AST) -> Set[str]:
    """Top-level string keys of every dict literal this function returns."""
    keys: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                s = _const_str(k)
                if s:
                    keys.add(s)
    return keys


def _nested_keys(fn: ast.AST) -> Set[str]:
    """Every string key of every dict literal anywhere in the function.

    A row inside `owed` carries `slug` and `axis`; a top-level-only scan says
    nothing produces them and reports a false REQUIRED_ARGUMENT_WITH_NO_SOURCE
    against odyssey.record_measurement -- which is wired and works.  An
    argument reachable only by digging into a returned row IS reachable; it is
    just more expensive to find, which is a separate cost this records but does
    not call a defect.
    """
    keys: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                s = _const_str(k)
                if s:
                    keys.add(s)
    return keys


def build(repo: Path = REPO) -> Dict[str, Any]:
    registry_path = repo / "hcli" / "tool_registry.py"
    engine_path = repo / "hcli" / "engine.py"
    src = registry_path.read_text()
    tree = ast.parse(src)

    functions: Dict[str, ast.AST] = {}
    schema_vars: Dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    schema_vars[t.id] = node.value

    def resolve_schema(node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Dict):
            return node
        if isinstance(node, ast.Name):
            return schema_vars.get(node.id)
        return None

    # THE TOOL TABLE COMES FROM THE LIVE REGISTRY, not from the AST.
    # An earlier pass walked ToolSpec(...) calls and found 56 tools; discover()
    # returns 71. Twenty-two of them register inside `for name, description in
    # (...)` loops, where the first argument is a loop variable and no literal
    # scan can see it. A reachability graph blind to 22 of 71 tools is not an
    # audit. The live registry is source truth -- it is the same object HCLI
    # calls -- and using it is not the circularity this file avoids: that
    # circularity would be harvesting the ROUND LOGS, which is still not done.
    # Import the registry AT `repo`, not the one this file lives beside: the
    # verify harness builds the graph against a scratch tree with one historical
    # fix reverted, and a build() hardwired to the real repo would ignore it and
    # pass every case. That is the vacuous-test trap this campaign keeps paying
    # for, so the root is a parameter.
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        f"_reach_registry_{abs(hash(str(repo)))}", registry_path)
    mod = importlib.util.module_from_spec(spec_)
    sys.modules[spec_.name] = mod
    sys.path.insert(0, str(repo))
    spec_.loader.exec_module(mod)
    live = mod.default_tool_registry(str(repo), repo_root=str(repo))
    specs = {d["name"]: d for d in live.discover()}

    # Handler return keys still come from the AST: the registry exposes no
    # handler reference. Match by the handler a literal registration named, and
    # for the table-driven ones fall back to the shared factory body.
    ast_handlers: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ToolSpec"):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        nm = _const_str(node.args[0]) if node.args else None
        if not nm:
            continue
        h = kw.get("handler")
        if isinstance(h, ast.Name):
            ast_handlers[nm] = h.id
        elif isinstance(h, ast.Attribute):
            ast_handlers[nm] = h.attr
        elif isinstance(h, ast.Lambda):
            inner = [x for x in ast.walk(h) if isinstance(x, ast.Call)
                     and isinstance(x.func, ast.Name)]
            if inner:
                ast_handlers[nm] = inner[0].func.id
        elif isinstance(h, ast.Call) and isinstance(h.func, ast.Name):
            ast_handlers[nm] = h.func.id

    tools: Dict[str, Dict[str, Any]] = {}
    unresolved_handlers = []
    for name, d in specs.items():
        schema = d.get("input_schema") or {}
        required = list(schema.get("required") or [])
        props = list((schema.get("properties") or {}).keys())
        hname = ast_handlers.get(name)
        produced, nested = set(), set()
        if hname and hname in functions:
            produced = _returned_keys(functions[hname])
            nested = _nested_keys(functions[hname]) - produced
        else:
            unresolved_handlers.append(name)
        tools[name] = {
            "name": name,
            "required_args": required,
            "optional_args": [a for a in props if a not in required],
            "handler": hname,
            "handler_kind": "ast" if (hname and hname in functions) else "unresolved",
            "produced_keys": sorted(produced),
            "nested_keys": sorted(nested),
            "alias_of": d.get("alias_of"),
            "mutation": (d.get("mutation") or "read_only").upper(),
        }

    # Who can produce each key?
    producers: Dict[str, List[str]] = {}
    nested_producers: Dict[str, List[str]] = {}
    for t in tools.values():
        for k in t["produced_keys"]:
            producers.setdefault(k, []).append(t["name"])
        for k in t["nested_keys"]:
            nested_producers.setdefault(k, []).append(t["name"])

    # Which tools does the model actually get shown?
    engine_src = engine_path.read_text()
    m = re.search(r"_AGENTIC_TOOL_CATALOG = \((.*?)\)\n", engine_src, re.S)
    agentic = set(re.findall(r"([a-z_]+\.[a-z_]+)\(", m.group(1))) if m else set()

    # REACHABILITY FIXPOINT. A producer that itself requires the key it emits
    # cannot originate it -- odyssey.anatomy returns `snapshot` and also demands
    # one, so quoting it as the source of `snapshot` is circular. Seed with the
    # tools callable from nothing, then close.
    def _sources(key: str) -> Set[str]:
        return set(producers.get(key, [])) | set(nested_producers.get(key, []))

    callable_tools: Set[str] = {n for n, t in tools.items() if not t["required_args"]}
    reachable_keys: Set[str] = set()
    for n in callable_tools:
        reachable_keys |= set(tools[n]["produced_keys"]) | set(tools[n]["nested_keys"])
    changed = True
    while changed:
        changed = False
        for n, t in tools.items():
            if n in callable_tools:
                continue
            if all(a in reachable_keys for a in t["required_args"]):
                callable_tools.add(n)
                before = len(reachable_keys)
                reachable_keys |= set(t["produced_keys"]) | set(t["nested_keys"])
                changed = changed or len(reachable_keys) != before or True
    unreachable = sorted(set(tools) - callable_tools)

    # NOT every unsourced argument is a defect. `pattern` for a search and
    # `question` for an escalation are the model's job to compose; demanding a
    # tool produce them would be demanding the harness think for it. The
    # dangerous class is narrower and has a mechanical test: an argument the
    # SYSTEM SPEAKS -- some tool emits a key of that name -- but which no tool
    # can ORIGINATE, because every emitter also demands one. There the model
    # will not know it is guessing: it will produce something that looks like a
    # value of that kind and be wrong. Round 15 passed the ledger's JSON path
    # where a snapshot DIRECTORY was wanted, spent the round, and left 5 axes
    # OWED. An argument no tool ever emits is model-authored by construction and
    # is not counted.
    def _has_origin(key: str) -> bool:
        return any(key not in tools[p]["required_args"] for p in _sources(key))

    def _is_state_shaped(key: str) -> bool:
        return bool(_sources(key))

    findings: Dict[str, List[Any]] = {
        "REQUIRED_ARGUMENT_WITH_NO_SOURCE": [],
        "ORPHAN_PRODUCER": [],
        "CAPABILITY_WITH_ZERO_CALLER": [],
        "OUTPUT_THAT_CANNOT_BE_RECORDED": [],
        "ARGUMENT_ONLY_REACHABLE_BY_DIGGING": [],
        "CIRCULAR_ARGUMENT_SOURCE": [],
        "ARGUMENT_WITH_NO_ORIGIN": [],
        "HANDLER_UNRESOLVED": [],
    }
    # What the enumerators -- the tools callable from nothing that return rows --
    # actually name things. If a tool demands a word the enumerators never say,
    # nothing on disk can be pointed at with it.
    enum_vocab = sorted({k for n2, t2 in tools.items()
                         if not t2["required_args"] and t2["nested_keys"]
                         for k in t2["nested_keys"]})
    # CAPABILITY WITH ZERO CALLER. The registry defines the tool; that is not the
    # same as anything ever reaching for it. A name that appears only in its own
    # registration -- not in the agentic catalog, not in a prompt, not in a task
    # file, not in a round challenge -- is a catalogue entry, not a capability.
    # Its own test file does not count as a caller.
    caller_roots = [repo / "hcli", repo / "workspace" / "campaign", repo / "tools" / "future"]
    corpus = []
    for root in caller_roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".txt", ".md", ".json", ".jsonl"} \
                    and f.name != "tool_registry.py" and not f.name.startswith("test_"):
                try:
                    corpus.append(f.read_text(errors="ignore"))
                except OSError:
                    pass
    blob = "\n".join(corpus)
    for n, t in sorted(tools.items()):
        if t["alias_of"] or n in agentic:
            continue
        if n not in blob:
            findings["CAPABILITY_WITH_ZERO_CALLER"].append(
                {"tool": n, "mutation": t["mutation"],
                 "note": "registered, absent from the agentic catalog, and its name appears in no "
                         "prompt, task file, round challenge or engine path -- nothing can reach "
                         "for it except by the model guessing the name exists"})

    for n in unreachable:
        t = tools[n]
        unmet = [a for a in t["required_args"] if a not in reachable_keys]
        trap = [a for a in unmet if _is_state_shaped(a) and not _has_origin(a)]
        authored = [a for a in unmet if a not in trap]
        if trap:
            findings["ARGUMENT_WITH_NO_ORIGIN"].append(
                {"tool": n, "unoriginatable_args": trap, "model_authored_args": authored,
                 "enumerator_vocabulary": enum_vocab,
                 "note": "the registry emits keys of these names, so the model has seen values "
                         "that look like them, but no tool can ORIGINATE one -- every emitter "
                         "also demands one. Whether that is a defect depends on the argument: a "
                         "search `pattern` is the model's to compose, an identifier that must "
                         "match something on disk is not. The enumerator vocabulary is listed so "
                         "the reader can tell which is which."})
        else:
            findings.setdefault("_model_authored_by_design", []).append(
                {"tool": n, "args": authored})

    consumed: Set[str] = set()
    for t in tools.values():
        consumed.update(t["required_args"])
        consumed.update(t["optional_args"])

    for t in sorted(tools.values(), key=lambda x: x["name"]):
        if t["alias_of"]:
            continue
        for arg in t["required_args"]:
            circular = sorted(p for p in _sources(arg)
                              if p != t["name"] and arg in tools[p]["required_args"])
            if circular and arg not in reachable_keys:
                findings["CIRCULAR_ARGUMENT_SOURCE"].append(
                    {"tool": t["name"], "arg": arg, "circular_producers": circular,
                     "note": "every tool that emits this key also demands one, so none of them "
                             "can originate it"})
            srcs = [p for p in producers.get(arg, [])
                    if p != t["name"] and p in callable_tools]
            deep = [p for p in nested_producers.get(arg, [])
                    if p != t["name"] and p in callable_tools]
            if not srcs and deep:
                findings["ARGUMENT_ONLY_REACHABLE_BY_DIGGING"].append(
                    {"tool": t["name"], "arg": arg, "buried_in": deep,
                     "note": "produced only inside a nested row, not at the top level of any "
                             "result; reachable, but the round must know to dig"})
            elif not srcs:
                findings["REQUIRED_ARGUMENT_WITH_NO_SOURCE"].append(
                    {"tool": t["name"], "arg": arg,
                     "note": "no reachable tool emits a key of this name at any depth; HCLI "
                             "must invent it or read it from a prompt"})
        if t["produced_keys"] and not (set(t["produced_keys"]) & consumed):
            findings["ORPHAN_PRODUCER"].append(
                {"tool": t["name"], "produces": t["produced_keys"],
                 "note": "nothing takes any of these as an argument"})
        if t["handler_kind"] == "unresolved":
            findings["HANDLER_UNRESOLVED"].append(
                {"tool": t["name"], "handler": t["handler"],
                 "note": "registered from a loop variable or factory the AST cannot follow; its "
                         "produced keys are UNKNOWN, so every origin conclusion about this tool "
                         "is an upper bound on what it can supply, not a measurement"})

    # A tool that MEASURES needs somewhere to put the measurement.  This is the
    # check that Round 19 paid 810 s to discover the hard way: the round reached
    # a real measurement and there was no door to make it durable.  A door only
    # counts if it is REACHABLE -- every one of its own required arguments must
    # itself be producible, or the door is a locked one.
    write_doors = sorted(n for n, t in tools.items()
                         if t["mutation"] != "READ_ONLY" or WRITE_DOOR_HINT.search(n))
    all_producible = set(producers) | set(nested_producers)
    usable_doors = []
    for n in write_doors:
        t = tools[n]
        if t["mutation"] == "READ_ONLY":
            continue
        unmet = [a for a in t["required_args"] if a not in all_producible]
        if not unmet:
            usable_doors.append(n)
    # "Any write door" is too loose to reproduce Round 19.  filesystem.write
    # existed that whole round; what did not exist was a door onto the LEDGER,
    # and the ledger is the artifact that resolves obligations.  A generic file
    # write leaves the cell OWED, so it is not a door for a measurement.  The
    # ledger's own unit is the (slug, axis) cell -- a door is a mutating tool
    # that accepts one.
    LEDGER_UNIT = {"axis", "slug"}
    ledger_doors = [n for n in usable_doors if LEDGER_UNIT & set(tools[n]["required_args"])]
    # Which tools measure? Ones the campaign ledger recognises as producing a
    # measurable axis -- named by the ledger's own vocabulary, not guessed.
    measurers = sorted(n for n, t in tools.items()
                       if t["mutation"] == "READ_ONLY"
                       and (set(t["produced_keys"]) & {"anatomy", "tally", "rows", "owed"}))
    if not ledger_doors:
        for n in measurers:
            findings["OUTPUT_THAT_CANNOT_BE_RECORDED"].append(
                {"tool": n, "produces": tools[n]["produced_keys"],
                 "note": "this tool measures and NO reachable LEDGER door exists to persist the "
                         "result as a resolved cell; the round will reach a real measurement and "
                         "have nowhere to put it. A generic file write is not a door: it leaves "
                         "the cell OWED.",
                 "generic_write_doors_that_do_not_count": usable_doors})
    else:
        for n in measurers:
            findings.setdefault("_recordable", []).append({"tool": n, "via": ledger_doors})
    return {
        "source_of_truth": "hcli.tool_registry.default_tool_registry(...).discover()",
        "n_tools": len(tools),
        "n_handlers_unresolved": len(unresolved_handlers),
        "handler_coverage": round(1 - len(unresolved_handlers) / max(1, len(tools)), 3),
        "n_aliases": sum(1 for t in tools.values() if t["alias_of"]),
        "agentic_catalog": sorted(agentic),
        "n_shown_in_agentic_catalog": len(agentic),
        "n_registered_not_in_agentic_catalog": len([t for t in tools if t not in agentic]),
        "write_doors": write_doors,
        "usable_write_doors": usable_doors,
        "ledger_doors": ledger_doors,
        "measurers": measurers,
        "n_callable": len(callable_tools),
        "unreachable_tools": unreachable,
        "enumerator_vocabulary": enum_vocab,
        "nested_producers": {k: sorted(v) for k, v in sorted(nested_producers.items())},
        "producers": {k: sorted(v) for k, v in sorted(producers.items())},
        "findings": findings,
        "tools": [tools[n] for n in sorted(tools)],
    }


if __name__ == "__main__":
    g = build()
    out = REPO / "receipts" / "future" / "G025_HCLI_REACHABILITY_GRAPH.json"
    if "--write" in sys.argv:
        out.write_text(json.dumps(g, indent=1) + "\n")
        print("wrote", out)
    print(f"tools {g['n_tools']} ({g['n_aliases']} aliases)")
    print(f"agentic catalog shows {g['n_shown_in_agentic_catalog']}, "
          f"{g['n_registered_not_in_agentic_catalog']} registered tools are not in it")
    for k, v in g["findings"].items():
        print(f"{k}: {len(v)}")
        for item in v[:8]:
            print("   ", item)
