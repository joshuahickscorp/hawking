"""Fail-closed product configuration.

`tools.odyssey.product_boundary` already loads a config and resolves relative
roots against the config file's directory. Its callers (modellake CLI) fall
back to `safe_defaults()` when no file is found. This module does not: a
missing or corrupt config is a closed error with a recovery document, never a
guess at artifact roots or at `/Users/scammermike/Downloads/hawking`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.odyssey.product_boundary import (
    SCHEMA,
    BoundaryError,
    discover_config,
    discover_machine,
    load_config,
)

EVIDENCE_TIER = "STATIC"
DEVELOPER_CHECKOUT = "/Users/scammermike/Downloads/hawking"


class ConfigClosed(ValueError):
    """Config is missing or corrupt. Artifact roots were not guessed."""

    def __init__(self, message: str, *, recovery: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.recovery = recovery if recovery is not None else recovery_document("missing")


def recovery_document(
    kind: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """How a fresh process resumes. Does not name a guessed artifact root."""
    return {
        "schema": "hawking.product.recovery.v1",
        "evidence_tier": EVIDENCE_TIER,
        "kind": kind,
        "path": str(path) if path is not None else None,
        "fail_closed": True,
        "did_not_guess_artifact_roots": True,
        "never_restart_healthy_worker": True,
        "developer_checkout_not_used": DEVELOPER_CHECKOUT,
        "write_config": {
            "schema": SCHEMA,
            "artifact_roots": {
                "specimens": "<absolute path, or relative to this file>",
                "partial": "<absolute path, or relative to this file>",
                "install": "<directory this machine owns>",
            },
        },
        "env": ["HAWKING_CONFIG", "HAWKING_HOME"],
        "note": (
            "tools.odyssey.product_boundary.safe_defaults names well-known "
            "product locations. This entry path will not apply them when the "
            "config is missing or corrupt."
        ),
    }


def require_config(
    *,
    explicit: str | Path | None = None,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Load a product config or refuse.

    Calls `discover_config` and `load_config`. Does not call `safe_defaults`
    as a substitute for a missing file. Roots not declared in the file are
    dropped so lake defaults cannot leak in.
    """
    try:
        path = discover_config(explicit=explicit, env=env)
    except BoundaryError as exc:
        raise ConfigClosed(
            str(exc),
            recovery=recovery_document("missing", explicit),
        ) from exc
    if path is None:
        raise ConfigClosed(
            "no product config found; refusing to guess artifact roots",
            recovery=recovery_document("missing"),
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigClosed(
            f"config unreadable or corrupt: {path}: {exc}",
            recovery=recovery_document("corrupt", path),
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigClosed(
            f"config root must be an object: {path}",
            recovery=recovery_document("corrupt", path),
        )
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ConfigClosed(
            f"unsupported or missing schema {schema!r}; want {SCHEMA}",
            recovery=recovery_document("corrupt", path),
        )
    roots = raw.get("artifact_roots")
    if not isinstance(roots, dict) or not roots:
        raise ConfigClosed(
            "config.artifact_roots missing or empty; refusing to guess",
            recovery=recovery_document("corrupt", path),
        )
    try:
        cfg = load_config(path)
    except BoundaryError as exc:
        raise ConfigClosed(
            str(exc),
            recovery=recovery_document("corrupt", path),
        ) from exc
    declared = {str(k): v for k, v in roots.items() if v}
    cfg["artifact_roots"] = {
        key: value
        for key, value in dict(cfg.get("artifact_roots") or {}).items()
        if key in declared
    }
    cfg["_declared_roots"] = sorted(declared)
    cfg["_declared_keys"] = sorted(str(k) for k in raw.keys())
    cfg["evidence_tier"] = EVIDENCE_TIER
    return cfg


def machine_inventory() -> dict[str, Any]:
    """STATIC host inventory. Calls `discover_machine`; does not use cwd."""
    info = dict(discover_machine())
    info["developer_checkout_not_assumed"] = True
    info["developer_checkout"] = DEVELOPER_CHECKOUT
    return info
