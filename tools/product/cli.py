#!/usr/bin/env python3
"""Product entry path that does not assume the developer checkout.

    python3 -m tools.product entry --config FILE [--artifact SLUG]
    python3 -m tools.product resolve --config FILE --artifact SLUG
    python3 -m tools.product discover-machine
    python3 -m tools.product scoreboard --receipts FILE [--require METRIC]

Runnable from any cwd. Artifact roots come from the config file, never from
cwd and never from this file's git tree. Missing or corrupt config exits 2
and does not guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.odyssey.product_boundary import (  # noqa: E402
    BoundaryError,
    install_plan,
    recover_plan,
    resolve_artifact,
    update_plan,
)
from tools.product.config import (  # noqa: E402
    ConfigClosed,
    machine_inventory,
    require_config,
)
from tools.product.install import (  # noqa: E402
    InstallError,
    install_artifact,
    resolve_install_paths,
    staleness,
    update_artifact,
)
from tools.product.scoreboard import (  # noqa: E402
    UnmeasuredError,
    load_scoreboard,
    qualify,
    require_measured,
)


def _emit(doc: Any) -> int:
    print(json.dumps(doc, indent=1))
    return 0


def cmd_discover_machine(_args: argparse.Namespace) -> int:
    return _emit(machine_inventory())


def cmd_resolve(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    name = args.artifact
    if not name:
        raise ConfigClosed("resolve requires --artifact")
    out = resolve_artifact(name, cfg)
    return _emit(out)


def cmd_install(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    slug = args.artifact
    if not slug:
        raise ConfigClosed("install requires --artifact")
    source, dest = resolve_install_paths(
        slug, cfg, source_root=args.source_root, dest_root=args.dest_root,
    )
    out = install_artifact(source, dest, slug=slug)
    return _emit(out)


def cmd_update(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    slug = args.artifact
    if not slug:
        raise ConfigClosed("update requires --artifact")
    plan = update_plan(cfg)
    source, dest = resolve_install_paths(
        slug, cfg, source_root=args.source_root, dest_root=args.dest_root,
    )
    out = update_artifact(source, dest, slug=slug)
    out["update_plan"] = plan
    return _emit(out)


def cmd_stale(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    slug = args.artifact
    if not slug:
        raise ConfigClosed("stale requires --artifact")
    source, dest = resolve_install_paths(
        slug, cfg, source_root=args.source_root, dest_root=args.dest_root,
    )
    return _emit(staleness(source, dest))


def cmd_scoreboard(args: argparse.Namespace) -> int:
    if not args.receipts:
        raise UnmeasuredError("scoreboard requires --receipts; refusing to search the checkout")
    board = load_scoreboard(args.receipts)
    for metric in args.require or ():
        require_measured(board, metric)
    if args.require:
        board["qualified"] = qualify(board, args.require)
    return _emit(board)


def cmd_recover(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    slug = args.artifact or "unspecified"
    rec = None
    out = recover_plan(slug, cfg, reacquisition=rec)
    out["fail_closed_entry"] = True
    return _emit(out)


def cmd_entry(args: argparse.Namespace) -> int:
    cfg = require_config(explicit=args.config)
    machine = machine_inventory()
    out: dict[str, Any] = {
        "ok": True,
        "schema": "hawking.product.entry.v1",
        "evidence_tier": "STATIC",
        "config_path": cfg.get("_config_path"),
        "cwd": str(Path.cwd()),
        "cwd_is_not_an_artifact_root": True,
        "checkout_independent": True,
        "fail_closed": True,
        "machine": machine,
        "handoff": {
            "schema": "hawking.product.handoff.v1",
            "evidence_tier": "STATIC",
            "config_path": cfg.get("_config_path"),
            "declared_roots": cfg.get("_declared_roots"),
            "machine_hw_model": machine.get("hw_model"),
            "checkout_not_required": True,
        },
        "updates": update_plan(cfg),
    }
    if args.artifact:
        out["artifact"] = resolve_artifact(args.artifact, cfg)
        out["install_plan"] = install_plan(args.artifact, cfg)
        out["recovery"] = recover_plan(args.artifact, cfg)
    if args.receipts:
        board = load_scoreboard(args.receipts)
        if args.require:
            for metric in args.require:
                require_measured(board, metric)
            board["qualified"] = qualify(board, args.require)
        out["scoreboard"] = board
    return _emit(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.product",
        description="Fail-closed Hawking product entry. Config, not cwd.",
    )
    parser.add_argument(
        "cmd",
        choices=(
            "discover-machine",
            "resolve",
            "install",
            "update",
            "stale",
            "scoreboard",
            "recover",
            "entry",
        ),
    )
    parser.add_argument("--config")
    parser.add_argument("--artifact")
    parser.add_argument("--receipts", action="append", default=[])
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--source-root", default="partial")
    parser.add_argument("--dest-root", default="install")
    args = parser.parse_args(argv)
    dispatch = {
        "discover-machine": cmd_discover_machine,
        "resolve": cmd_resolve,
        "install": cmd_install,
        "update": cmd_update,
        "stale": cmd_stale,
        "scoreboard": cmd_scoreboard,
        "recover": cmd_recover,
        "entry": cmd_entry,
    }
    try:
        return dispatch[args.cmd](args)
    except (ConfigClosed, BoundaryError, UnmeasuredError, InstallError) as exc:
        doc: dict[str, Any] = {
            "ok": False,
            "error": str(exc),
            "fail_closed": True,
            "did_not_guess": True,
            "error_type": type(exc).__name__,
        }
        recovery = getattr(exc, "recovery", None)
        if recovery is not None:
            doc["recovery"] = recovery
        print(json.dumps(doc, indent=1))
        print(f"tools.product: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
