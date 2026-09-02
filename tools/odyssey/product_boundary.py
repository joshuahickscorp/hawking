"""Era V-A/V-B product boundary: artifacts come from configuration, not cwd.

Hawking still has entry paths that assume they are running inside this
developer checkout (Path(__file__).parents[2], cwd == repo). This module is
the separation the V-A gene card names: configuration, machine discovery,
artifact installation, updates, safe defaults, recovery.

It does not install a product, does not write the live ModelLake volume, and
does not restart an acquisition worker. Resolution is STATIC path math.

    python3 tools/odyssey/modellake.py resolve --config FILE --artifact SLUG
    python3 tools/odyssey/modellake.py discover-machine
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

SCHEMA = "hawking.product.boundary.v1"
VERSION = 1
EVIDENCE_TIER = "STATIC"

# V-A gene card subgenes this scaffold actually implements a handle for.
# The rest stay named so their absence is a gap, not a silent skip.
SUBGENES_HANDLED = (
    "installability",
    "local execution",
    "hardware discovery",
    "safe defaults",
    "offline paths",
    "diagnostics",
    "upgrade/rollback",
)
SUBGENES_NAMED_NOT_BUILT = (
    "reproducible builds",
    "model/provider choice",
)

# Well-known machine locations. These are not the git checkout.
DEFAULT_HOME = Path.home() / "Library" / "Application Support" / "Hawking"
DEFAULT_LAKE = Path("/Volumes/corpdrive/hawking-modellake")
DEFAULT_STAGE = Path.home() / "noetic" / "stage"

ABSENT_AS_MODEL = ("FPGA/U50", "DGX", "eGPU")


class BoundaryError(ValueError):
    """Config is missing, malformed, or an artifact name cannot be resolved."""


def safe_defaults(*, home: Optional[Path] = None, lake: Optional[Path] = None) -> dict[str, Any]:
    """Defaults that do not point at a developer checkout.

    HAWKING_HOME overrides the product home. The git worktree that contains
    this file is never used as an artifact root.
    """
    env_home = os.environ.get("HAWKING_HOME")
    product_home = Path(home) if home else (
        Path(env_home) if env_home else DEFAULT_HOME
    )
    lake_root = Path(lake) if lake else DEFAULT_LAKE
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_tier": EVIDENCE_TIER,
        "product_home": str(product_home),
        "artifact_roots": {
            "home": str(product_home),
            "lake": str(lake_root),
            "specimens": str(lake_root / "specimens"),
            "partial": str(lake_root / "partial"),
            "lake_manifests": str(lake_root / "manifests"),
            "watch_manifests": str(product_home / "watch-manifests"),
            "nr": str(product_home / "nr"),
            "nx": str(product_home / "nx"),
            "stage": str(DEFAULT_STAGE),
        },
        "install": {
            "atomic_rename": True,
            "overwrite": False,
            "require_manifest": True,
        },
        "updates": {
            "policy": "refuse_unsigned",
            "never_restart_healthy_worker": True,
        },
        "recovery": {
            "reacquire_from_manifest": True,
            "never_restart_healthy_worker": True,
            "partials_are_capital": True,
        },
        "defaults": {
            "offline": True,
            "safe_mode": True,
            "cwd_is_not_an_artifact_root": True,
        },
        "subgenes_handled": list(SUBGENES_HANDLED),
        "subgenes_named_not_built": list(SUBGENES_NAMED_NOT_BUILT),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a boundary config. Relative artifact roots resolve against the
    config file's directory, never against the process cwd."""
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise BoundaryError(f"config is not a file: {cfg_path}")
    try:
        doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryError(f"config unreadable: {cfg_path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise BoundaryError("config root must be an object")
    schema = doc.get("schema")
    if schema is not None and schema != SCHEMA:
        raise BoundaryError(f"unsupported schema {schema!r}; want {SCHEMA}")
    base = safe_defaults()
    merged = {**base, **{k: v for k, v in doc.items() if k != "artifact_roots"}}
    roots = {**base["artifact_roots"], **dict(doc.get("artifact_roots") or {})}
    merged["artifact_roots"] = roots
    merged["_config_path"] = str(cfg_path.resolve())
    merged["_config_dir"] = str(cfg_path.resolve().parent)
    merged["evidence_tier"] = EVIDENCE_TIER
    return merged


def discover_config(
    *,
    explicit: str | Path | None = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Find a config without consulting cwd or this file's checkout.

    Order: explicit path, HAWKING_CONFIG, $HAWKING_HOME/config.json,
    ~/Library/Application Support/Hawking/config.json. A missing file is
    None, not a default-to-repo fallback.
    """
    envmap = dict(env) if env is not None else dict(os.environ)
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        raise BoundaryError(f"explicit config not found: {explicit}")
    env_cfg = envmap.get("HAWKING_CONFIG")
    if env_cfg:
        p = Path(env_cfg).expanduser()
        if p.is_file():
            return p.resolve()
        raise BoundaryError(f"HAWKING_CONFIG is set but not a file: {env_cfg}")
    home = Path(envmap["HAWKING_HOME"]).expanduser() if envmap.get("HAWKING_HOME") else DEFAULT_HOME
    candidate = home / "config.json"
    return candidate.resolve() if candidate.is_file() else None


def _abs_from_config(raw: str | Path, config: Mapping[str, Any]) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    base = config.get("_config_dir")
    if not base:
        raise BoundaryError(
            f"relative artifact root {raw!r} needs a config file directory; "
            "cwd is not an artifact root"
        )
    return (Path(base) / path).resolve()


def resolve_artifact(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one artifact from configuration.

    `name` is a specimen slug, or `root:slug` (specimens/partial/nr/nx/stage/
    lake_manifests/watch_manifests). The returned path is built from
    config['artifact_roots'], never from Path.cwd() or this module's file.
    """
    if not name:
        raise BoundaryError("artifact name is empty")
    roots = dict(config.get("artifact_roots") or {})
    if not roots:
        raise BoundaryError("config has no artifact_roots")

    root_key = None
    slug = name
    if ":" in name:
        root_key, slug = name.split(":", 1)
    aliases = {
        "source": "specimens",
        "specimen": "specimens",
        "cold": "specimens",
        "hot": "stage",
        "manifest": "lake_manifests",
        "watch": "watch_manifests",
    }
    if root_key:
        root_key = aliases.get(root_key, root_key)
        if root_key not in roots:
            raise BoundaryError(f"unknown artifact root {root_key!r}")
        search = (root_key,)
    else:
        search = (
            "specimens", "stage", "partial", "nr", "nx",
            "lake_manifests", "watch_manifests",
        )

    hits = []
    chosen = None
    for key in search:
        raw = roots.get(key)
        if not raw:
            continue
        root = _abs_from_config(raw, config)
        if key.endswith("manifests"):
            candidates = (root / f"{slug}.json", root / slug)
        else:
            candidates = (root / slug,)
        for cand in candidates:
            hit = {
                "root_key": key,
                "root": str(root),
                "path": str(cand),
                "present": cand.exists(),
            }
            hits.append(hit)
            if cand.exists() and chosen is None:
                chosen = hit
                if root_key is None:
                    break
        if chosen is not None and root_key is None:
            break
    if chosen is None:
        chosen = hits[0] if hits else None
    if chosen is None:
        raise BoundaryError(f"no artifact root can host {name!r}")

    return {
        "schema": SCHEMA,
        "artifact": name,
        "slug": slug,
        "path": chosen["path"],
        "present": chosen["present"],
        "root_key": chosen["root_key"],
        "root": chosen["root"],
        "resolved_from": f"config.artifact_roots.{chosen['root_key']}",
        "cwd_independent": True,
        "checkout_independent": True,
        "cwd": str(Path.cwd()),
        "candidates": hits,
        "evidence_tier": EVIDENCE_TIER,
        "config_path": config.get("_config_path"),
    }


def discover_machine() -> dict[str, Any]:
    """STATIC host inventory. Not a kernel measurement, not FPGA presence."""
    hw_model = _sysctl("hw.model")
    mem = _sysctl("hw.memsize")
    ncpu = _sysctl("hw.ncpu")
    try:
        mem_bytes = int(mem) if mem else None
    except ValueError:
        mem_bytes = None
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "schema": "hawking.product.machine.v1",
        "evidence_tier": EVIDENCE_TIER,
        "os": platform.system(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "hw_model": hw_model,
        "ncpu": ncpu,
        "mem_bytes": mem_bytes,
        "present_domains": {
            "CPU": True,
            "GPU_UMA": apple,
            "ANE": apple,
        },
        "present_domains_are": (
            "STATIC host identity from platform/sysctl. GPU_UMA/ANE true "
            "means Apple Silicon is the host, not that a kernel ran."
        ),
        "absent_as_model_not_measurement": list(ABSENT_AS_MODEL),
        "gpu_authority": False,
        "cwd": str(Path.cwd()),
        "cwd_is_not_used_for_artifacts": True,
    }


def install_plan(slug: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe an install. Does not write, does not touch the live lake."""
    src = resolve_artifact(f"partial:{slug}", config)
    dst = resolve_artifact(f"specimens:{slug}", config)
    if src["present"] and not dst["present"]:
        action = "WOULD_RENAME"
    elif dst["present"]:
        action = "ALREADY_INSTALLED"
    else:
        action = "REFUSED"
    return {
        "schema": "hawking.product.install_plan.v1",
        "evidence_tier": EVIDENCE_TIER,
        "slug": slug,
        "source": src["path"],
        "destination": dst["path"],
        "action": action,
        "atomic_rename": True,
        "overwrite": False,
        "wrote": False,
        "why": (
            "promote is tools.odyssey.modellake_promote.promote; this plan "
            "only names the configured paths"
        ),
    }


def update_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe an update policy. Does not fetch, does not restart workers."""
    policy = (config.get("updates") or {}).get("policy") or "refuse_unsigned"
    return {
        "schema": "hawking.product.update_plan.v1",
        "evidence_tier": EVIDENCE_TIER,
        "policy": policy,
        "never_restart_healthy_worker": True,
        "wrote": False,
        "fetched": False,
    }


def recover_plan(
    slug: str,
    config: Mapping[str, Any],
    *,
    reacquisition: str | None = None,
) -> dict[str, Any]:
    """Name how a specimen comes back. Does not download, does not kill PIDs."""
    return {
        "schema": "hawking.product.recover_plan.v1",
        "evidence_tier": EVIDENCE_TIER,
        "slug": slug,
        "reacquisition": reacquisition,
        "never_restart_healthy_worker": True,
        "partials_are_capital": True,
        "wrote": False,
        "spawned": False,
        "config_path": config.get("_config_path"),
    }


def _sysctl(key: str) -> str | None:
    try:
        r = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    val = (r.stdout or "").strip()
    return val or None
