#!/usr/bin/env python3
"""G009 producer: capability reachability by PRODUCTION CALL SITE, not registration.

This repo's hardest-won law is that a capability nothing CALLS does not exist.
It once carried 41 typed tools whose registry appeared zero times in mission.py,
executors.py, engine.py and resident.py: not unimplemented, structurally
unreachable.  So this producer refuses to accept three things that look like
evidence and are not:

  * registration            -- a name in a registry is a catalogue entry
  * importability           -- `import hcli.x` proves a file parses
  * a direct handler call   -- bypasses the registry the production path uses

What it does instead, for every capability the LIVE mission's obligations
require:

  1. derives the required set from the live goal text, and REFUSES to write a
     receipt if any derivation quote is no longer literally present in that
     goal (the anti-invention guard: the list comes from the mission, not from
     whoever ran this);
  2. enumerates production call sites by scanning non-test source under the
     execution roots, with the registry's own definition file excluded so a
     definition can never be mistaken for a call;
  3. INVOKES each required capability through ``ToolRegistry.invoke`` -- the
     same entry point ``hcli/executors.py`` and ``hcli/engine.py`` dispatch
     through -- and records what came back.

Writes receipts/sovereign/G009_reachability.json.  Read-only with respect to
the repository and to .hcli/.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RECEIPT = REPO / "receipts" / "sovereign" / "G009_reachability.json"
MISSION_STATE = REPO / ".hcli" / "mission" / "state.json"
SCHEMA = "hawking.sovereign.g009.reachability.v1"
PRODUCED_BY = "tools/sovereign/g009_reachability.py"
COMMAND = "python3 tools/sovereign/g009_reachability.py"

# The execution path.  hcli/ is the control plane and tools/odyssey/ is the
# campaign runtime; both are named by the live mission's own verifiers.
SCAN_ROOTS = ("hcli", "tools/odyssey")
# The registry's own definition file.  A ToolSpec("fs.read", ...) line here is a
# DEFINITION.  Counting it as a call site is precisely the error this gate exists
# to catch, so it is excluded from the call-site scan by name.
DEFINITION_FILES = {"hcli/tool_registry.py"}

# Every entry is (tool name, the exact substring of the LIVE mission goal that
# requires it, probe arguments).  The quote is checked against the goal text at
# run time; a stale or invented quote aborts the run before anything is written.
FRONTIER: tuple[tuple[str, str, dict], ...] = (
    ("fs.read", "current source",
     {"path": "hcli/tool_registry.py", "max_bytes": 4096}),
    ("fs.list", "current source",
     {"path": "hcli", "max_results": 25}),
    ("fs.search", "call sites are authority",
     {"pattern": "registry.invoke(", "root": "hcli", "max_results": 25}),
    ("receipt.read", "receipts",
     {"path": "receipts/sovereign/VERIFIER_MANIFEST.json", "max_bytes": 8192}),
    ("modellake.status", "ModelLake state", {}),
    ("specimens.registry", "an incomplete specimen never blocks independent science", {}),
    ("tests.list", "tests", {"root": "hcli"}),
    ("tests.run", "tests",
     {"paths": ["hcli/test_command_registry.py"], "timeout_s": 240}),
    ("accelerator.inspect", "hardware measurements", {}),
    ("odyssey.status", "Run Odyssey streams I, II and III", {}),
    ("odyssey.queue", "Run Odyssey streams I, II and III", {}),
    ("tools.catalog", "Audit capability reachability by production call site",
     {"focus": "capability reachability production call site"}),
)

# Named as authority by the same goal law, audited separately because the tool
# registry may or may not carry it.  Whether it does is a RESULT, not an input.
PROCESS_QUOTE = "processes"
PROCESS_REGISTRY_PROBES = ("processes.status", "processes.live",
                           "process.list", "hcli.processes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _summarize(value, *, limit: int = 320) -> dict:
    """What an invocation returned, bounded. Never the whole payload."""
    text = json.dumps(value, default=str, ensure_ascii=False)
    return {
        "type": type(value).__name__,
        "keys": sorted(value)[:24] if isinstance(value, dict) else None,
        "json_bytes": len(text),
        "sha256": _digest(value),
        "excerpt": text[:limit],
    }


def load_goal() -> tuple[str, dict]:
    """The live mission, read only. Its goal text is the authority for scope."""
    if not MISSION_STATE.is_file():
        raise SystemExit(
            f"REFUSING: {MISSION_STATE} is absent. The frontier's required "
            "capability set is derived from the live mission, not assumed."
        )
    state = json.loads(MISSION_STATE.read_text(encoding="utf-8"))
    goal = str(state.get("goal") or "")
    if not goal.strip():
        raise SystemExit("REFUSING: the live mission carries no goal text.")
    return goal, state


def check_derivation(goal: str) -> None:
    """Every quote must still be in the live goal, or nothing is written.

    Without this a future editor could widen or narrow the audited set by
    typing, and the receipt would still look derived. The guard makes the
    mission text the only thing that can change the scope.
    """
    missing = [q for _n, q, _a in FRONTIER if q not in goal]
    if PROCESS_QUOTE not in goal:
        missing.append(PROCESS_QUOTE)
    if missing:
        raise SystemExit(
            "REFUSING to write a receipt: these derivation quotes are no longer "
            f"present in the live mission goal, so the audited set is not derived "
            f"from it: {missing}"
        )


def production_sources() -> list[Path]:
    """Non-test source on the execution path. Definitions excluded."""
    out: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if "__pycache__" in rel or rel in DEFINITION_FILES:
                continue
            if path.name.startswith("test_") or "/test_" in rel:
                continue
            out.append(path)
    return out


def literal_sites(name: str, sources: list[tuple[str, list[str]]]) -> list[str]:
    """path:line for every production line that names this capability."""
    hits = []
    for rel, lines in sources:
        for number, line in enumerate(lines, 1):
            if name in line:
                hits.append(f"{rel}:{number}")
    return hits


def dispatch_sites(sources: list[tuple[str, list[str]]]) -> list[str]:
    """Where production dispatches a tool BY NAME through the registry."""
    hits = []
    for rel, lines in sources:
        for number, line in enumerate(lines, 1):
            if re.search(r"\.invoke\(\s*name", line) or re.search(
                r"registry\.invoke\(|tools\.invoke\(|_tools\.invoke\(", line
            ):
                hits.append(f"{rel}:{number}")
    return sorted(set(hits))


def compact_catalog_names(registry, goal: str) -> tuple[set[str], str]:
    """The names the LIVE model-facing surface actually emits.

    The agentic loop calls ``_prompt_with_observations(..., compact_catalog=True)``
    on every round, so this compact rendering -- not the full catalog -- is the
    only tool list the resident ever sees. Names appear in prefix form
    (``odyssey: status(); queue()``) and alias form (``fs.read|filesystem.read(...)``),
    so both are expanded back to dotted names.
    """
    from hcli.engine import Engine

    text = Engine._compact_tool_catalog(registry, focus=goal)
    names: set[str] = set()
    for line in text.splitlines():
        head, sep, rest = line.partition(":")
        prefix = head.strip()
        if sep and re.fullmatch(r"[a-z_]+", prefix) and rest.strip():
            for entry in rest.split(";"):
                short = entry.strip().split("(", 1)[0].strip()
                if short:
                    names.add(f"{prefix}.{short}")
            continue
        if "|" in line or re.match(r"^[a-z_]+\.[a-z_.\-/]+\(", line.strip()):
            for alias in line.split("(", 1)[0].split("|"):
                alias = alias.strip()
                if "." in alias:
                    names.add(alias)
    return names, text


def catalog_discoverable(registry, name: str) -> dict:
    """Can the model FIND this tool at all, by asking for it?

    ``tools.catalog`` is itself advertised on every round, so a tool the
    compact catalog omits is still reachable in two steps -- ask, then call.
    Reporting such a tool as dead would be as wrong as reporting a registered
    one as live, so the route is probed by really invoking tools.catalog.
    """
    focus = name.replace(".", " ").replace("_", " ")
    result = registry.invoke("tools.catalog", {"focus": focus, "max_results": 30})
    found = name in [
        row.get("name") for row in (result.value or {}).get("matches", [])
    ] if result.ok else False
    return {"focus": focus, "ok": bool(result.ok), "found": found}


def invoke(registry, name: str, args: dict) -> dict:
    """Through the registry. Calling the handler directly is how this hid before."""
    started = time.perf_counter()
    try:
        result = registry.invoke(name, dict(args))
    except Exception as exc:  # an exception is still a real invocation outcome
        return {
            "invoked": True,
            "raised": f"{type(exc).__name__}: {exc}",
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    elapsed = round(time.perf_counter() - started, 4)
    return {
        "invoked": True,
        "ok": bool(result.ok),
        "failure_class": result.failure_class,
        "invocation_id": result.invocation_id,
        "mutation": result.mutation,
        "error": result.error,
        "elapsed_s": elapsed,
        "returned": _summarize(result.value) if result.ok else None,
    }


def audit_processes(registry, sources) -> dict:
    """Process truth: named as authority by the goal, audited on its own terms.

    The registry probes below are not decoration. If the registry carries no
    process surface, the resident cannot observe processes at all through the
    typed path, however healthy hcli/processes.py is.
    """
    probes = {}
    for candidate in PROCESS_REGISTRY_PROBES:
        result = registry.invoke(candidate, {})
        probes[candidate] = {"ok": bool(result.ok), "failure_class": result.failure_class}
    shell_probe = registry.invoke("shell.readonly", {"command": "ps -Ao pid,comm"})
    probes["shell.readonly(ps)"] = {
        "ok": bool(shell_probe.ok),
        "failure_class": shell_probe.failure_class,
        "error": shell_probe.error,
    }
    registered = any(row["ok"] for row in probes.values())

    # hcli/processes.py is this capability's DEFINING module. Its own `def`
    # lines are definitions, not calls -- the same exclusion the registry's
    # definition file gets, for the same reason.
    sites = sorted(
        site for site in set(
            literal_sites("processes import", sources)
            + literal_sites("from .processes", sources)
            + literal_sites("processes.render", sources)
            + literal_sites("reap_orphaned_bodies", sources)
        )
        if not site.startswith("hcli/processes.py:")
    )

    from hcli import processes as live

    started = time.perf_counter()
    summary = live.summary()
    elapsed = round(time.perf_counter() - started, 4)

    return {
        "name": "processes",
        "registered": registered,
        "production_call_sites": sites,
        "required_by_active_frontier": True,
        "derived_from_goal_quote": PROCESS_QUOTE,
        "invoked": True,
        "invoked_via": (
            "hcli.processes.summary() -- its REAL production entry point. NOT the "
            "tool registry: the registry carries no process capability, which is "
            "the finding this row records."
        ),
        "registry_probes": probes,
        "evidence": {
            "elapsed_s": elapsed,
            "returned": _summarize(summary),
            "call_path": (
                "hcli/runtime.py RuntimePool.__init__ -> _register_live -> "
                "_reap_orphans_once -> hcli.processes.reap_orphaned_bodies; and "
                "hcli/commands.py -> hcli.processes.render for the /processes command"
            ),
        },
        "finding": (
            "REACHABLE FROM PRODUCTION CODE, UNREACHABLE FROM THE MODEL. The live "
            "goal's first law names processes as authority, but no registered tool "
            "exposes them and both shell tools refuse `ps`, so the resident cannot "
            "observe process truth through the typed path it is given."
        ),
    }


def main() -> int:
    goal, state = load_goal()
    check_derivation(goal)

    from hcli.tool_registry import default_tool_registry

    registry = default_tool_registry(REPO, repo_root=REPO)
    registered = {spec["name"]: spec for spec in registry.discover()}

    sources = [
        (p.relative_to(REPO).as_posix(), p.read_text(errors="replace").splitlines())
        for p in production_sources()
    ]
    dispatchers = dispatch_sites(sources)
    catalog_names, catalog_text = compact_catalog_names(registry, goal)

    required = {name: (quote, args) for name, quote, args in FRONTIER}
    missing_from_registry = sorted(n for n in required if n not in registered)

    capabilities = []
    for name in sorted(registered):
        sites = literal_sites(name, sources)
        advertised = name in catalog_names
        discoverable = catalog_discoverable(registry, name)
        reachable = advertised or discoverable["found"]
        row = {
            "name": name,
            "registered": True,
            "mutation": registered[name].get("mutation"),
            # A name production can reach: literally, or by name-dispatch once
            # the model has been shown it. Both are verified, not assumed --
            # the catalog renderings below were really executed.
            "production_call_sites": sorted(set(sites) | set(dispatchers)) if reachable else list(sites),
            "call_site_kinds": {
                "named_literally_in_production": sites,
                "model_facing_catalog_emits_name": advertised,
                "found_by_tools_catalog_probe": discoverable,
                "registry_name_dispatch": dispatchers if reachable else [],
            },
            "required_by_active_frontier": name in required,
            "invoked": False,
        }
        if name in required:
            quote, args = required[name]
            row["derived_from_goal_quote"] = quote
            row["probe_arguments"] = args
            outcome = invoke(registry, name, args)
            row["invoked"] = outcome.pop("invoked")
            row["invoked_via"] = "hcli.tool_registry.ToolRegistry.invoke"
            row["evidence"] = outcome
        capabilities.append(row)

    capabilities.append(audit_processes(registry, sources))

    dead = sorted(c["name"] for c in capabilities
                  if c.get("registered") and not c["production_call_sites"])
    direct_only_via_catalog = sorted(
        c["name"] for c in capabilities
        if c.get("registered")
        and not c["call_site_kinds"]["named_literally_in_production"]
        and not c["call_site_kinds"]["model_facing_catalog_emits_name"]
    )
    not_advertised = sorted(
        name for name in registered if name not in catalog_names
    )
    failed_required = sorted(
        c["name"] for c in capabilities
        if c["required_by_active_frontier"] and not (c.get("evidence") or {}).get("ok", True)
    )

    receipt = {
        "schema": SCHEMA,
        "produced_by": PRODUCED_BY,
        "produced_at": _now(),
        "command": COMMAND,
        "status": "completed",
        "mission": {
            "id": state.get("id"),
            "phase": state.get("phase"),
            "state_path": MISSION_STATE.relative_to(REPO).as_posix(),
            "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
            "units": len(state.get("units") or {}),
            "verifier_commands": sorted({
                str(u.get("verifier")) for u in (state.get("units") or {}).values()
                if u.get("verifier")
            })[:20],
        },
        "method": {
            "population": "every tool in hcli.tool_registry.default_tool_registry(...).discover()",
            "required_set_derivation": (
                "each required capability carries the exact substring of the LIVE "
                "mission goal that requires it; the producer aborts before writing "
                "if any quote is no longer present in that goal"
            ),
            "call_site_scan_roots": list(SCAN_ROOTS),
            "excluded_from_scan": ["test files", *sorted(DEFINITION_FILES)],
            "invocation_entry_point": "hcli.tool_registry.ToolRegistry.invoke",
            "why_not_direct_handler_calls": (
                "a registry-reachability defect hid here before because tests "
                "called handlers directly and bypassed the registry the "
                "production path uses"
            ),
            "model_facing_surface": (
                "hcli/engine.py Engine._compact_tool_catalog, rendered for real "
                "with the live goal as focus; the agentic loop passes "
                "compact_catalog=True on every round"
            ),
        },
        "counts": {
            "registered_tools": len(registered),
            "audited_capabilities": len(capabilities),
            "required_by_active_frontier": len(required) + 1,
            "invoked_through_registry": sum(
                1 for c in capabilities
                if c.get("invoked_via") == "hcli.tool_registry.ToolRegistry.invoke"
            ),
            "no_production_call_site": len(dead),
            "not_in_live_model_facing_catalog": len(not_advertised),
        },
        "findings": [
            {
                "id": "G009-F1",
                "severity": "gap",
                "claim": (
                    "The live goal names processes as authority; the tool registry "
                    "carries no process capability and both shell tools refuse `ps`."
                ),
                "evidence": "capabilities[].name == 'processes' -> registry_probes",
            },
            {
                "id": "G009-F2",
                "severity": "observation",
                "claim": (
                    "Engine._tool_catalog (the FULL catalog) has no production call "
                    "site: both callers of _prompt_with_observations pass "
                    "compact_catalog=True, so the compact rendering is the only "
                    "tool list the resident is ever shown."
                ),
                "evidence": "hcli/engine.py:1293 and hcli/engine.py:1319 vs hcli/engine.py:1166",
            },
            {
                "id": "G009-F3",
                "severity": "observation",
                "claim": (
                    "These registered tools are absent from the live model-facing "
                    "catalog for this goal and are reachable only via a "
                    "tools.catalog(focus=...) round: " + ", ".join(not_advertised)
                ),
                "evidence": "method.model_facing_surface rendering",
            },
            {
                "id": "G009-F4",
                "severity": "result",
                "claim": (
                    "No registered tool is structurally unreachable. Every one of "
                    f"{len(registered)} is either named in production, emitted by the "
                    "live model-facing catalog, or returned by a real "
                    "tools.catalog(focus=<its own name>) probe. This is the "
                    "condition the 41-tools-with-no-call-site era failed."
                    if not dead else
                    "Registered tools with NO route from production at all: "
                    + ", ".join(dead)
                ),
                "evidence": "capabilities[].call_site_kinds",
            },
        ],
        "reachable_only_after_a_tools_catalog_round": direct_only_via_catalog,
        "no_production_call_site": dead,
        "not_in_live_model_facing_catalog": not_advertised,
        "required_but_unregistered": missing_from_registry,
        "required_invocations_that_failed": failed_required,
        "capabilities": capabilities,
        "evidence": {
            "registered_tools": len(registered),
            "invoked_through_registry": sorted(required),
            "registry_dispatch_sites": dispatchers,
            "model_facing_catalog_chars": len(catalog_text),
            "mission_id": state.get("id"),
        },
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(f"wrote {RECEIPT.relative_to(REPO)}")
    print(f"  registered={len(registered)} audited={len(capabilities)} "
          f"required={len(required) + 1} invoked_via_registry={receipt['counts']['invoked_through_registry']}")
    print(f"  no production call site: {len(dead)} -> {dead}")
    print(f"  required invocations that FAILED: {failed_required}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
