"""Every Hawking body a person could pick, named the way a person would pick it.

WHY A SEPARATE MODULE. `discover_models` already walks roots and identifies
native profiles, MLX directories and GGUF files -- that part is not rebuilt here.
What was missing is the bit a dropdown needs: a STABLE, SHORT, UNIQUE name per
body, and a single list of explicitly admitted Gravity execution artifacts.

NAMES ARE IDENTITY, SO THEY MUST NOT COLLIDE OR DRIFT. A ModelLake specimen
directory is `Qwen--Qwen3-4B-Instruct-2507@cdbee75f17c0`; nobody wants to read
that in a menu, and the bare `Qwen3-4B-Instruct-2507` is what they mean. The
revision is dropped for the label and kept for disambiguation, so two revisions
of one body get distinct names instead of one silently shadowing the other.

THE MODELLAKE IS READ-ONLY AND MAY BE ABSENT. If the volume is not mounted the
catalog simply does not list it. It is never rebuilt, never written to, and its
absence is reported rather than papered over -- a menu that silently loses half
its entries is worse than one that says the drive is unplugged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MODELLAKE = Path("/Volumes/corpdrive/hawking-modellake/specimens")
REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
# Installed HCLI snapshots live under ~/.local/share and therefore cannot
# derive the checkout's workspace registry from __file__.  The shim supplies
# this explicit source path; editable checkouts keep the local default.
_registry_env = os.environ.get("HCLI_GRAVITY_REGISTRY")
GRAVITY_REGISTRY = Path(_registry_env).expanduser() if _registry_env else (
    PROJECT_ROOT / "workspace/campaign/odyssey/gravity-artifacts.json"
)
GRAVITY_STATUSES = frozenset({
    "ADMITTED",
    "OPERATIONAL_DEVELOPMENTAL",
    "FROZEN_OPERATIONAL_DEVELOPMENTAL",
})


@dataclass
class Body:
    """One selectable model."""

    name: str
    path: str
    kind: str                      # noetic_native | mlx | llamacpp | remote
    bytes: int = 0
    revision: str = ""
    source: str = "repo"           # repo | modellake | user
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self, *, loaded: bool = False) -> Dict[str, Any]:
        row = {
            "id": self.name,
            "object": "model",
            "created": 0,
            "owned_by": "hawking",
            "hawking": {
                "path": self.path,
                "backend": self.kind,
                "source": self.source,
                "gb": round(self.bytes / 1e9, 2) if self.bytes else None,
                "loaded": loaded,
            },
        }
        if self.revision:
            row["hawking"]["revision"] = self.revision
        if self.detail:
            row["hawking"]["artifact"] = dict(self.detail)
        return row


def _dir_bytes(path: Path, cap: int = 4000) -> int:
    """Size without walking a whole volume: the shard files only."""
    total = 0
    try:
        for index, entry in enumerate(path.iterdir()):
            if index > cap:
                break
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        return 0
    return total


def _native_profiles(root: Path) -> List[Body]:
    out: List[Body] = []
    for profile in sorted(root.glob("hawking-native.*.json")):
        try:
            data = json.loads(profile.read_text(encoding="utf-8"))
        except Exception:
            continue
        identity = str(data.get("resident_identity") or profile.stem)
        out.append(Body(
            name=identity,
            path=str(profile),
            kind="noetic_native",
            source="repo",
            detail={
                "family": data.get("family"),
                "ebpw": data.get("physical_ebpw"),
                "qualification": data.get("qualification"),
                "greedy": (data.get("generation") or {}).get("do_sample") is False,
            },
        ))
    return out


def _modellake(root: Optional[Path] = None) -> List[Body]:
    # Resolve the default at call time so tests and mounted-volume changes can
    # replace the research root without having to rewrite a bound default.
    root = MODELLAKE if root is None else root
    if not root.is_dir():
        return []
    out: List[Body] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        raw = entry.name
        revision = ""
        label = raw
        if "@" in raw:
            label, revision = raw.rsplit("@", 1)
        # `Qwen--Qwen3-4B-Instruct-2507` -> `Qwen3-4B-Instruct-2507`
        if "--" in label:
            label = label.split("--", 1)[1]
        if not (entry / "config.json").is_file():
            continue
        out.append(Body(
            name=label,
            path=str(entry),
            kind="mlx",
            bytes=_dir_bytes(entry),
            revision=revision,
            source="modellake",
        ))
    return out


def _gravity_artifacts(path: Path = GRAVITY_REGISTRY) -> List[Body]:
    """Read the small admitted-artifact registry used by normal execution.

    The ModelLake is deliberately not the default execution menu.  This
    registry is the one place where a model becomes a user-facing Gravity
    body, with its lineage and qualification receipts kept beside the stable
    selector identity.  Missing or non-executable entries are omitted rather
    than surfaced as choices that will fail only after a click.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    entries = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return []
    out: List[Body] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("id") or "").strip()
        raw_path = str(entry.get("path") or "").strip()
        status = str(entry.get("status") or "").strip()
        if not name or not raw_path or status not in GRAVITY_STATUSES:
            continue
        resolved = Path(os.path.realpath(os.path.expanduser(raw_path)))
        if not resolved.exists():
            continue
        kind = str(entry.get("kind") or "mlx")
        try:
            if kind == "mlx":
                from .backends import is_mlx_model_dir, model_bytes_at
                if not is_mlx_model_dir(str(resolved)):
                    continue
                size = int(model_bytes_at(str(resolved)) or 0)
            elif kind == "noetic_native":
                from .hawking_native import is_hawking_native_path
                if not is_hawking_native_path(str(resolved)):
                    continue
                size = _dir_bytes(resolved)
            else:
                continue
        except Exception:
            continue
        detail = {
            key: value for key, value in entry.items()
            if key not in {"id", "path", "kind", "status", "bytes"}
        }
        detail.update({"status": status, "registry": str(path)})
        out.append(Body(name=name, path=str(resolved), kind=kind,
                        bytes=int(entry.get("bytes") or size),
                        source="gravity", detail=detail))
    return out


def _deduplicate(bodies: List[Body]) -> List[Body]:
    """Two bodies may want the same label. Neither may silently win.

    A collision keeps both and qualifies each with its revision, because a menu
    entry that resolves to whichever body happened to be walked first is the
    kind of false affordance that costs an afternoon.
    """
    counts: Dict[str, int] = {}
    for body in bodies:
        counts[body.name] = counts.get(body.name, 0) + 1
    out = []
    for body in bodies:
        if counts[body.name] > 1 and body.revision:
            body = Body(**{**body.__dict__, "name": f"{body.name}@{body.revision[:8]}"})
        out.append(body)
    return out


def catalog(extra_roots: Optional[List[str]] = None, *, research: bool = False) -> List[Body]:
    """Normal execution bodies, or the explicit research catalog.

    Normal callers see only explicitly admitted Gravity artifacts.
    ``research=True`` is the deliberate advanced door that adds ModelLake
    specimens; it is never used by the OpenAI model picker or normal switching.
    """
    bodies: List[Body] = []
    bodies.extend(_gravity_artifacts())
    if research:
        bodies.extend(_modellake())
    for root in extra_roots or []:
        path = Path(os.path.expanduser(root))
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".json":
            bodies.extend(_native_profiles(path.parent))
        elif (path / "config.json").is_file():
            bodies.append(Body(name=path.name, path=str(path), kind="mlx",
                               bytes=_dir_bytes(path), source="user"))
    return _deduplicate(bodies)


def resolve(name: str, bodies: Optional[List[Body]] = None, *, research: bool = False) -> Optional[Body]:
    """Find a body by name, then by path, then by unique case-insensitive prefix.

    Prefix matching is a convenience that REFUSES when ambiguous rather than
    picking one, so `hcli use qwen3` says which ones it could have meant.
    """
    bodies = bodies if bodies is not None else catalog(research=research)
    text = str(name or "").strip()
    if not text:
        return None
    for body in bodies:
        if body.name == text:
            return body
    target = os.path.realpath(os.path.expanduser(text))
    for body in bodies:
        if os.path.realpath(body.path) == target:
            return body
    lowered = text.lower()
    hits = [b for b in bodies if b.name.lower().startswith(lowered)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise LookupError(
            f"{text!r} matches {len(hits)} bodies: "
            + ", ".join(b.name for b in hits[:8])
            + (" ..." if len(hits) > 8 else ""))
    return None


def missing_sources() -> List[str]:
    """What a person would otherwise notice as an unexplained gap in the menu."""
    out = []
    if not MODELLAKE.is_dir():
        out.append(f"the ModelLake volume is not mounted ({MODELLAKE}), so no "
                   f"specimens are listed")
    return out
