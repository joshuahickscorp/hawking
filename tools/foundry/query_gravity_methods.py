#!/usr/bin/env python3
"""Query Hawking's canonical Gravity method registry by model capabilities.

The registry remains the sole owner of method descriptions and evidence.  This
reader only performs deterministic tag matching; it does not promote a method
or turn a scoped Flash result into a cross-model conclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "hawking.foundry.gravity_method_registry.v2":
        raise ValueError(f"unsupported registry schema: {doc.get('schema')!r}")
    methods = doc.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("canonical registry has no queryable methods")
    owners = doc.get("canonical_owners")
    if not isinstance(owners, dict) or not owners:
        raise ValueError("canonical registry has no ownership map")
    return doc


def select_methods(tags: set[str], path: Path = REGISTRY) -> list[dict[str, Any]]:
    """Return methods whose required architecture tags are a subset of *tags*."""
    selected = []
    for method in load_registry(path)["methods"]:
        required = set(method["applicability"]["architecture_tags"])
        if required.issubset(tags):
            selected.append(method)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags",
        required=False,
        help="comma-separated architecture/capability tags, e.g. moe,routed_experts",
    )
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--json", action="store_true", help="emit machine-readable selected entries")
    parser.add_argument(
        "--owners",
        action="store_true",
        help="emit the canonical implementation-owner map instead of selecting methods",
    )
    args = parser.parse_args()
    if not args.owners and not args.tags:
        parser.error("--tags is required unless --owners is selected")
    if args.owners:
        owners = load_registry(args.registry)["canonical_owners"]
        if args.json:
            print(json.dumps({"canonical_owners": owners}, indent=2, sort_keys=True))
        else:
            for scope, owner in owners.items():
                print(f"{scope}\t{owner}")
        return 0
    tags = {tag.strip() for tag in args.tags.split(",") if tag.strip()}
    selected = select_methods(tags, args.registry)
    if args.json:
        print(json.dumps({"tags": sorted(tags), "methods": selected}, indent=2, sort_keys=True))
        return 0
    for method in selected:
        print(f"{method['id']}\t{method['status']}\t{method['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
