"""Every Hawking body a person could pick, named the way a person would pick it.

WHY A SEPARATE MODULE. `discover_models` already walks roots and identifies
native profiles, MLX directories and GGUF files -- that part is not rebuilt here.
What was missing is the bit a dropdown needs: a STABLE, SHORT, UNIQUE name per
body, and a single list that spans the sealed profiles in the repo and the
specimens on the ModelLake volume.

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
    admitted: bool = True
    admission_reason: str = ""

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
                "admitted": self.admitted,
            },
        }
        if self.revision:
            row["hawking"]["revision"] = self.revision
        if self.admission_reason:
            row["hawking"]["admission_reason"] = self.admission_reason
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
            admitted=bool(data.get("qualification")),
            admission_reason=("" if data.get("qualification") else
                              "native profile has no qualification contract"),
        ))
    return out


def _modellake(root: Path = MODELLAKE) -> List[Body]:
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
            admitted=False,
            admission_reason=("ModelLake preserves the source specimen; no "
                              "qualified Hawking execution binding is recorded"),
        ))
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


def catalog(extra_roots: Optional[List[str]] = None) -> List[Body]:
    """Admitted executable bodies only, repo profiles first."""
    bodies: List[Body] = []
    bodies.extend(_native_profiles(REPO))
    bodies.extend(_modellake())
    for root in extra_roots or []:
        path = Path(os.path.expanduser(root))
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".json":
            bodies.extend(_native_profiles(path.parent))
        elif (path / "config.json").is_file():
            bodies.append(Body(name=path.name, path=str(path), kind="mlx",
                               bytes=_dir_bytes(path), source="user",
                               admitted=False,
                               admission_reason="no qualified execution binding"))
    return _deduplicate([body for body in bodies if body.admitted])


def specimen_catalog() -> List[Body]:
    """Discoverable source specimens kept outside executable selection."""
    return _deduplicate([body for body in _modellake() if not body.admitted])


def resolve(name: str, bodies: Optional[List[Body]] = None) -> Optional[Body]:
    """Find a body by name, then by path, then by unique case-insensitive prefix.

    Prefix matching is a convenience that REFUSES when ambiguous rather than
    picking one, so `hcli use qwen3` says which ones it could have meant.
    """
    bodies = bodies if bodies is not None else catalog()
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
    else:
        out.append("ModelLake source specimens are listed separately and cannot be "
                   "selected until a qualified execution binding exists")
    return out
