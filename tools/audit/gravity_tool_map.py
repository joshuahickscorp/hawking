"""Inventory HCLI's registered AgentOS tools before Tool Gravity migration.

This is an audit receipt generator, not a registry rewrite.  It builds the
same default registry used by the engine, records every callable spelling and
the smaller discoverable surface, then attaches implementation, mutation,
resource, caller, test, and compatibility metadata.  No tool is deleted,
renamed, or made callable by this module.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".worktrees", ".preserved", ".hide", ".hcli", ".hcli-legacy",
    "node_modules", "target", "workspace", "artifacts", "reports", "logs",
    "scratchpad", "visionmcp", "receipts",
}
TEXT_EXTENSIONS = {".py", ".rs", ".md", ".toml", ".yaml", ".yml", ".sh"}

# Proposed vocabulary only.  These targets are deliberately not registered by
# this audit: a migration must prove per-op validation and mutation parity
# before changing the public surface.
TARGETS = {
    "fs": "file.read | file.search | file.list",
    "filesystem": "file.read | file.search | file.list | file.write",
    "web": "source.search | source.fetch",
    "github": "source.search | source.fetch",
    "huggingface": "source.search | source.fetch | source.acquire",
    "lake": "model.inspect | model.acquire",
    "modellake": "model.inspect | model.acquire",
    "specimens": "model.inspect",
    "receipt": "evidence.inspect | evidence.read",
    "roadmap": "evidence.inspect",
    "shell": "run.readonly | run.exec",
    "tests": "measure.run",
    "benchmark": "measure.run | measure.inspect",
    "accelerator": "measure.run | measure.inspect",
    "physical": "measure.inspect | measure.measure",
    "odyssey": "campaign.inspect | campaign.record | campaign.advance",
    "campaign": "campaign.inspect | campaign.checkpoint | campaign.advance",
    "audit": "verify.inspect | verify.attack | verify.reachability",
    "capability": "verify.capability",
    "claim": "verify.attack",
    "tool": "verify.reachability",
    "gravity": "gravity.inspect | gravity.transform | gravity.verify | gravity.measure",
    "doctor": "gravity.inspect",
    "nr": "gravity.account",
    "vmcp": "perception.inspect | perception.query",
    "processes": "host.inspect",
    "git": "repo.inspect | repo.land",
    "frontier": "campaign.decide | cognition.delegate",
    "grok": "cognition.delegate",
    "context": "evidence.recall",
    "forbidden_fruit": "experiment.transform",
    "wall": "campaign.record",
}


def _files(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            result.append(path)
    return sorted(result)


def _family(name: str) -> str:
    return name.split(".", 1)[0]


def _source_location(handler: Any) -> dict[str, Any]:
    try:
        source = inspect.getsourcefile(handler)
        lines, line = inspect.getsourcelines(handler)
    except (OSError, TypeError):
        source, line = None, None
    path = None
    if source:
        try:
            path = str(Path(source).resolve().relative_to(ROOT.resolve()))
        except ValueError:
            path = str(Path(source).resolve())
    return {
        "module": getattr(handler, "__module__", None),
        "qualname": getattr(handler, "__qualname__", repr(handler)),
        "path": path,
        "line": line,
        "generated_handler": "<locals>" in str(getattr(handler, "__qualname__", "")),
    }


def _callers(root: Path, names: list[str], files: list[Path]) -> tuple[list[str], list[str]]:
    pattern = re.compile(r"(?<![A-Za-z0-9_.-])(?:" + "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    ) + r")(?![A-Za-z0-9_.-])")
    paths = []
    tests = []
    for path in files:
        try:
            body = path.read_text(errors="replace")
        except OSError:
            continue
        if not pattern.search(body):
            continue
        relative = path.relative_to(root).as_posix()
        paths.append(relative)
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            tests.append(relative)
    return sorted(paths), sorted(tests)


def _risk(spec: dict[str, Any], *, alias: bool) -> str:
    if alias:
        return "LOW: callable compatibility spelling; retire only after caller migration"
    if spec["mutation"] in {"destructive", "external_write"}:
        return "HIGH: preserve explicit authority and destructive-control boundary"
    if spec["mutation"] not in {"read_only", "research"}:
        return "MEDIUM-HIGH: preserve confirmation, timeout, and mutation class"
    return "MEDIUM: preserve schema, result contract, and source provenance"


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    # Import through the same factory as HCLI AgentOS.  This is the authority
    # being inventoried; a hand-maintained list would miss dynamically merged
    # aliases and recreate the defect Tool Gravity is meant to prevent.
    from hcli.tool_registry import default_tool_registry

    registry = default_tool_registry(root, repo_root=root)
    all_specs = registry.discover(include_aliases=True)
    canonical_specs = registry.discover()
    source_files = _files(root)
    names = [str(item["name"]) for item in all_specs]
    caller_cache: dict[str, tuple[list[str], list[str]]] = {}

    def callers_for(item: dict[str, Any]) -> tuple[list[str], list[str]]:
        root_name = str(item.get("alias_of") or item["name"])
        key = root_name
        if key not in caller_cache:
            related = [root_name]
            related.extend(
                str(other["name"])
                for other in all_specs
                if str(other.get("alias_of") or "") == root_name
            )
            caller_cache[key] = _callers(root, related, source_files)
        return caller_cache[key]

    entries = []
    families: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "registered": 0, "discoverable": 0, "aliases": 0, "names": [],
    })
    mutation_counts = Counter(str(item["mutation"]) for item in all_specs)
    for item in all_specs:
        name = str(item["name"])
        family = _family(name)
        alias_of = item.get("alias_of")
        callers, tests = callers_for(item)
        source = _source_location(registry.get(name).handler)
        families[family]["registered"] += 1
        families[family]["names"].append(name)
        if alias_of:
            families[family]["aliases"] += 1
        else:
            families[family]["discoverable"] += 1
        entries.append({
            "name": name,
            "alias_of": alias_of,
            "discoverable": not bool(alias_of),
            "semantic_family": family,
            "proposed_gravity_surface": TARGETS.get(family, "UNREVIEWED"),
            "description": item.get("description"),
            "input_schema": item.get("input_schema"),
            "output_schema": item.get("output_schema"),
            "mutation": item.get("mutation"),
            "deterministic": item.get("deterministic"),
            "timeout_s": item.get("timeout_s"),
            "roles": item.get("roles") or [],
            "resources": item.get("resources") or [],
            "verifier_expectations": item.get("verifier_expectations") or [],
            "implementation": source,
            "callers": callers,
            "tests": tests,
            "authority": {
                "permission_class": item.get("mutation"),
                "registry": "hcli.tool_registry.default_tool_registry",
                "external_resources": item.get("resources") or [],
            },
            "compatibility_risk": _risk(item, alias=bool(alias_of)),
            "replacement": (
                f"compatibility alias -> {alias_of}" if alias_of else
                TARGETS.get(family, "owner review required")
            ),
            "unique_semantic_capability": str(item.get("description") or "")[:240],
        })
    for row in families.values():
        row["names"] = sorted(row["names"])

    return {
        "schema": "hawking.gravity.tool_map.v1",
        "status": "BEFORE_STATE_INVENTORY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(root),
        "registry_authority": "hcli.tool_registry.default_tool_registry",
        "surface_counts": {
            "registered_callable_spellings": len(all_specs),
            "discoverable_model_facing_tools": len(canonical_specs),
            "compatibility_aliases": sum(bool(item.get("alias_of")) for item in all_specs),
            "semantic_families": len(families),
            "mutation_classes": dict(sorted(mutation_counts.items())),
        },
        "semantic_primitive_target": {
            "rule": "Compress the interface, not the capability; preserve typed ops and authority classes.",
            "no_god_tool": True,
            "readonly_mutation_split": ["shell.readonly", "shell.exec"],
            "acquisition_split": ["source.fetch", "source.acquire"],
            "compatibility_rule": "Old names remain callable aliases until callers and tests migrate.",
            "targets": TARGETS,
        },
        "families": dict(sorted(families.items())),
        "entries": sorted(entries, key=lambda row: row["name"]),
        "migration_rule": (
            "Inventory every name, migrate one semantic capability, preserve a regression and receipt, "
            "benchmark before/after, then retire an alias only when its caller count is zero."
        ),
        "claim_boundary": (
            "This is a registry and caller inventory. Proposed Gravity surfaces are targets, not implemented "
            "renames. It proves neither semantic equivalence nor permission equivalence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "GRAVITY_TOOL_MAP.json",
    )
    args = parser.parse_args()
    document = build(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "surface_counts": document["surface_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
