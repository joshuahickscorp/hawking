"""Admitted Hawking bodies and separately discoverable source specimens.

WHY A SEPARATE MODULE. `discover_models` already walks roots and identifies
native profiles, MLX directories and GGUF files -- that part is not rebuilt here.
What was missing is the bit a dropdown needs: a STABLE, SHORT, UNIQUE name per
body, and a single list of explicitly admitted Gravity execution artifacts.
ModelLake specimens remain visible as source identities but stay outside the
executable catalog until a qualified execution binding is recorded.

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
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

MODELLAKE = Path("/Volumes/corpdrive/hawking-modellake/specimens")
REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent
# Installed HAWKING snapshots live under ~/.local/share and therefore cannot
# derive the checkout's workspace registry from __file__.  The shim supplies
# this explicit source path; editable checkouts keep the local default.
_registry_env = os.environ.get("HAWKING_GRAVITY_REGISTRY")
GRAVITY_REGISTRY = Path(_registry_env).expanduser() if _registry_env else (
    PROJECT_ROOT / "workspace/campaign/odyssey/gravity-artifacts.json"
)
GRAVITY_STATUSES = frozenset({
    "ADMITTED",
    "OPERATIONAL_DEVELOPMENTAL",
    "FROZEN_OPERATIONAL_DEVELOPMENTAL",
})
ACTION_IDS = frozenset({"execute", "serve", "web"})
ACTION_EXECUTION_INTENTS = {
    "execute": "interactive_execute",
    "serve": "resident_serve",
    "web": "web_surface",
}
RESOLVED_ACTION_SCHEMA = "hawking.resolved_action.v1"
GRAVITY_REGISTRY_SCHEMA = "hawking.gravity.admitted_artifact_registry.v1"
GRAVITY_REGISTRY_STATUS = "ACTIVE"


@dataclass
class Body:
    """One selectable model."""

    name: str
    path: str
    kind: str                      # noetic_native | remote; research may retain mlx
    bytes: int = 0
    revision: str = ""
    source: str = "repo"           # repo | modellake | user
    detail: Dict[str, Any] = field(default_factory=dict)
    admitted: bool = True
    admission_reason: str = ""
    supported_actions: tuple[str, ...] = ()

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
                "supported_actions": list(self.supported_actions),
            },
        }
        if self.revision:
            row["hawking"]["revision"] = self.revision
        if self.detail:
            row["hawking"]["artifact"] = dict(self.detail)
        if self.admission_reason:
            row["hawking"]["admission_reason"] = self.admission_reason
        return row


_REPOSITORY_WRITE_WITHHELD = "autonomous repository writing"


def write_session_denial(body: Body) -> Optional[str]:
    """Return the artifact-authority reason a write surface must be withheld.

    An admitted ``serve`` or ``web`` action grants an executable artifact, not
    automatic repository-mutation authority.  Gravity's artifact record may
    explicitly withhold that authority for a bounded research worker (Kimi P0
    does).  Interpret that existing artifact contract here, beside action
    admission, so every serving front door can reject it before provider
    construction rather than relying on a prompt or a caller convention.

    Bodies without an explicit withheld authority retain the historical
    behavior.  This is deliberately not a Kimi name check: any future artifact
    carrying the same exact withheld authority receives the same fail-closed
    boundary.
    """
    detail = body.detail if isinstance(body.detail, Mapping) else {}
    authority = detail.get("authority")
    if not isinstance(authority, Mapping):
        return None
    withheld = authority.get("withheld")
    if not isinstance(withheld, (list, tuple, set)):
        return None
    normalized = {
        value.strip().casefold()
        for value in withheld
        if isinstance(value, str)
    }
    if _REPOSITORY_WRITE_WITHHELD not in normalized:
        return None
    return (
        f"{body.name!r} does not grant repository-write authority; its admitted "
        "artifact contract withholds autonomous repository writing"
    )


def require_write_session_authority(body: Body) -> None:
    """Fail closed before giving one admitted artifact a write-capable surface."""
    denial = write_session_denial(body)
    if denial is not None:
        raise PermissionError(denial)


def _canonical_path(raw: str) -> str:
    return os.path.realpath(os.path.expanduser(str(raw)))


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _is_lower_sha256(value: object) -> bool:
    text = str(value or "")
    return _is_sha256(text) and text == text.lower()


def _catalog_identity_and_revision(body: Body, path: str) -> tuple[str, str]:
    """Describe the existing admission source without creating a grant store."""
    registry = str((body.detail or {}).get("registry") or "").strip()
    if registry:
        registry_path = _canonical_path(registry)
        try:
            return (
                f"gravity-registry:{registry_path}",
                hashlib.sha256(Path(registry_path).read_bytes()).hexdigest(),
            )
        except OSError as exc:
            raise LookupError(
                f"the Gravity catalog authority is unavailable at {registry_path}"
            ) from exc
    if body.kind == "noetic_native":
        return f"native-profile:{path}", body.revision.lower()
    record = {
        "source": body.source,
        "name": body.name,
        "path": path,
        "kind": body.kind,
        "revision": body.revision.lower(),
        "admitted": bool(body.admitted),
        "supported_actions": sorted(body.supported_actions),
    }
    return f"catalog-body:{body.source}:{path}", _sha256_json(record)


def _grant_digest(
    *,
    name: str,
    path: str,
    kind: str,
    artifact_revision: str,
    artifact_revision_basis: str,
    catalog_identity: str,
    catalog_revision: str,
    grant_identity: str,
    supported_actions: tuple[str, ...],
) -> str:
    return _sha256_json({
        "name": name,
        "path": path,
        "kind": kind,
        "artifact_revision": artifact_revision,
        "artifact_revision_basis": artifact_revision_basis,
        "catalog_identity": catalog_identity,
        "catalog_revision": catalog_revision,
        "grant_identity": grant_identity,
        "admitted": True,
        "supported_actions": list(supported_actions),
    })


@dataclass(frozen=True)
class ResolvedAction:
    """An admitted artifact and the exact action Hawking may perform with it.

    This is a value, not a second catalog.  It carries the output of the
    existing catalog authority across Rust, process, HTTP, and browser
    transport boundaries; every consumer revalidates the value against that
    authority before using it.
    """

    action: str
    name: str
    path: str
    kind: str
    artifact_revision: str
    artifact_revision_basis: str
    catalog_identity: str
    catalog_revision: str
    grant_identity: str
    grant_digest: str
    supported_actions: tuple[str, ...]
    body: Optional[Body] = field(default=None, repr=False, compare=False)

    @property
    def binding(self) -> Dict[str, str]:
        """Identity that must survive resident reuse, independent of action."""
        return {
            "path": self.path,
            "artifact_revision": self.artifact_revision,
            "artifact_revision_basis": self.artifact_revision_basis,
            "catalog_identity": self.catalog_identity,
            "catalog_revision": self.catalog_revision,
            "grant_identity": self.grant_identity,
            "grant_digest": self.grant_digest,
        }

    def has_same_binding(self, other: "ResolvedAction") -> bool:
        return self.binding == other.binding

    def to_wire(self) -> Dict[str, Any]:
        return {
            "schema": RESOLVED_ACTION_SCHEMA,
            "action": self.action,
            "execution_intent": ACTION_EXECUTION_INTENTS[self.action],
            "artifact": {
                "id": self.name,
                "path": self.path,
                "kind": self.kind,
                "revision": self.artifact_revision,
                "revision_basis": self.artifact_revision_basis,
            },
            "catalog": {
                "identity": self.catalog_identity,
                "revision": self.catalog_revision,
            },
            "grant": {
                "identity": self.grant_identity,
                "digest": self.grant_digest,
                "admitted": True,
                "supported_actions": list(self.supported_actions),
            },
            "reuse_binding": self.binding,
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> "ResolvedAction":
        if not isinstance(raw, Mapping):
            raise ValueError("resolved action must be an object")
        if raw.get("schema") != RESOLVED_ACTION_SCHEMA:
            raise ValueError("resolved action has an unknown schema")
        expected_top_level = {
            "schema", "action", "execution_intent", "artifact", "catalog",
            "grant", "reuse_binding",
        }
        if set(raw) != expected_top_level:
            raise ValueError("resolved action has unknown or missing fields")
        action = raw.get("action")
        execution_intent = raw.get("execution_intent")
        artifact = raw.get("artifact")
        catalog_record = raw.get("catalog")
        grant = raw.get("grant")
        reuse_binding = raw.get("reuse_binding")
        if not isinstance(artifact, Mapping) or not isinstance(catalog_record, Mapping) \
                or not isinstance(grant, Mapping) or not isinstance(reuse_binding, Mapping):
            raise ValueError("resolved action omits artifact, catalog, grant, or reuse binding")
        if set(artifact) != {"id", "path", "kind", "revision", "revision_basis"} \
                or set(catalog_record) != {"identity", "revision"} \
                or set(grant) != {"identity", "digest", "admitted", "supported_actions"}:
            raise ValueError("resolved action has an incomplete nested record")
        text_values = {
            "action": action,
            "execution_intent": execution_intent,
            "artifact.id": artifact.get("id"),
            "artifact.path": artifact.get("path"),
            "artifact.kind": artifact.get("kind"),
            "artifact.revision": artifact.get("revision"),
            "artifact.revision_basis": artifact.get("revision_basis"),
            "catalog.identity": catalog_record.get("identity"),
            "catalog.revision": catalog_record.get("revision"),
            "grant.identity": grant.get("identity"),
            "grant.digest": grant.get("digest"),
        }
        if any(not isinstance(value, str) or not value for value in text_values.values()):
            raise ValueError("resolved action has non-string identity fields")
        name = artifact["id"]
        path = artifact["path"]
        kind = artifact["kind"]
        artifact_revision = artifact["revision"]
        artifact_revision_basis = artifact["revision_basis"]
        catalog_identity = catalog_record["identity"]
        catalog_revision = catalog_record["revision"]
        grant_identity = grant["identity"]
        grant_digest = grant["digest"]
        raw_actions = grant.get("supported_actions")
        if not isinstance(raw_actions, list) or not all(isinstance(item, str) for item in raw_actions):
            raise ValueError("resolved action grant omits supported actions")
        supported_actions = tuple(raw_actions)
        if (
            action not in ACTION_IDS
            or not name
            or not path
            or path != _canonical_path(path)
            or not kind
            or not artifact_revision_basis
            or not catalog_identity
            or not grant_identity
            or grant.get("admitted") is not True
            or not _is_lower_sha256(artifact_revision)
            or not _is_lower_sha256(catalog_revision)
            or not _is_lower_sha256(grant_digest)
            or not supported_actions
            or len(set(supported_actions)) != len(supported_actions)
            or any(item not in ACTION_IDS for item in supported_actions)
            or action not in supported_actions
        ):
            raise ValueError("resolved action is incomplete or not admitted")
        expected_digest = _grant_digest(
            name=name,
            path=path,
            kind=kind,
            artifact_revision=artifact_revision,
            artifact_revision_basis=artifact_revision_basis,
            catalog_identity=catalog_identity,
            catalog_revision=catalog_revision,
            grant_identity=grant_identity,
            supported_actions=supported_actions,
        )
        if grant_digest != expected_digest:
            raise ValueError("resolved action grant digest does not match its binding")
        resolved = cls(
            action=action,
            name=name,
            path=path,
            kind=kind,
            artifact_revision=artifact_revision,
            artifact_revision_basis=artifact_revision_basis,
            catalog_identity=catalog_identity,
            catalog_revision=catalog_revision,
            grant_identity=grant_identity,
            grant_digest=grant_digest,
            supported_actions=supported_actions,
        )
        if execution_intent != ACTION_EXECUTION_INTENTS[action] \
                or set(reuse_binding) != set(resolved.binding) \
                or any(not isinstance(value, str) for value in reuse_binding.values()) \
                or dict(reuse_binding) != resolved.binding:
            raise ValueError("resolved action execution intent or reuse binding does not match")
        return resolved


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
        qualification = data.get("qualification")
        admission = data.get("admission") or {}
        evidence = admission.get("evidence") or []
        admitted = (
            data.get("profile_schema") == "hawking.provider.profile.v1"
            and data.get("provider") == "native"
            and data.get("runtime") == "hawking-native"
            and admission.get("status") == "ADMITTED"
            and bool(admission.get("contract"))
            and isinstance(evidence, list)
            and bool(evidence)
            and isinstance(qualification, str)
            and bool(qualification.strip())
        )
        out.append(Body(
            name=identity,
            path=str(profile),
            kind="noetic_native",
            source="repo",
            detail={
                "family": data.get("family"),
                "ebpw": data.get("physical_ebpw"),
                "qualification": qualification,
                "admission_contract": admission.get("contract"),
                "admission_evidence": evidence,
                "greedy": (data.get("generation") or {}).get("do_sample") is False,
            },
            admitted=admitted,
            admission_reason=("" if admitted else
                              "native profile has no explicit admitted contract and evidence"),
            revision=hashlib.sha256(profile.read_bytes()).hexdigest(),
            supported_actions=(("execute", "serve", "web") if admitted else ()),
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
            admitted=False,
            admission_reason=("ModelLake preserves the source specimen; no "
                              "qualified Hawking execution binding is recorded"),
            supported_actions=(),
        ))
    return out


def _gravity_artifacts(path: Optional[Path] = None) -> List[Body]:
    """Read the small admitted-artifact registry used by normal execution.

    The ModelLake is deliberately not the default execution menu.  This
    registry is the one place where a model becomes a user-facing Gravity
    body, with its lineage and qualification receipts kept beside the stable
    selector identity.  Missing or non-executable entries are omitted rather
    than surfaced as choices that will fail only after a click.
    """
    path = GRAVITY_REGISTRY if path is None else path
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(document, dict) \
            or document.get("schema") != GRAVITY_REGISTRY_SCHEMA \
            or document.get("status") != GRAVITY_REGISTRY_STATUS:
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
        kind = str(entry.get("kind") or "")
        try:
            if kind == "noetic_native":
                from .hawking_native import is_hawking_native_path
                if not is_hawking_native_path(str(resolved)):
                    continue
                size = _dir_bytes(resolved)
            else:
                # MLX/GGUF/unknown catalog rows are retained only as source
                # history, never admitted to the live Hawking catalog.
                continue
        except Exception:
            continue
        detail = {
            key: value for key, value in entry.items()
            if key not in {
                "id", "path", "kind", "status", "bytes", "revision",
                "supported_actions",
            }
        }
        detail.update({"status": status, "registry": str(path)})
        revision = str(entry.get("revision") or "").strip()
        revision_basis = entry.get("revision_basis")
        actions = entry.get("supported_actions")
        if not (
            len(revision) == 64
            and all(character in "0123456789abcdefABCDEF" for character in revision)
            and isinstance(actions, list)
            and actions
            and len(set(actions)) == len(actions)
            and all(isinstance(action, str) and action in ACTION_IDS for action in actions)
            and isinstance(revision_basis, str)
            and bool(revision_basis.strip())
        ):
            continue
        out.append(Body(name=name, path=str(resolved), kind=kind,
                        bytes=int(entry.get("bytes") or size),
                        source="gravity", detail=detail,
                        revision=revision.lower(),
                        admitted=True,
                        supported_actions=tuple(actions)))
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
            # A vendor weight directory is a historical specimen, never a
            # live Hawking catalog body. Research callers use specimen_catalog
            # explicitly; normal selection must not expose a dead backend.
            continue
    return _deduplicate([
        body for body in bodies if body.admitted or research
    ])


def specimen_catalog() -> List[Body]:
    """Discoverable source specimens kept outside executable selection."""
    return _deduplicate([body for body in _modellake() if not body.admitted])


def resolve(name: str, bodies: Optional[List[Body]] = None, *, research: bool = False) -> Optional[Body]:
    """Find a body by name, then by path, then by unique case-insensitive prefix.

    Prefix matching is a convenience that REFUSES when ambiguous rather than
    picking one, so `hawking use qwen3` says which ones it could have meant.
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


def _revision_basis(body: Body) -> str:
    """State what the existing revision proves, without inventing a hash claim."""
    if body.source == "gravity":
        basis = (body.detail or {}).get("revision_basis")
        if isinstance(basis, str) and basis.strip():
            return basis.strip()
        raise PermissionError(f"{body.name!r} lacks the Gravity revision basis")
    if body.kind == "noetic_native":
        return "native_profile_bytes_sha256"
    return "catalog_declared"


def _resolved_action_for_body(body: Body, action: str) -> ResolvedAction:
    if action not in ACTION_IDS:
        raise ValueError(f"unsupported Hawking action {action!r}")
    if not body.admitted:
        raise PermissionError(f"{body.name!r} is not admitted for Hawking execution")
    supported_actions = tuple(sorted(body.supported_actions))
    if not supported_actions or any(item not in ACTION_IDS for item in supported_actions):
        raise PermissionError(f"{body.name!r} has no valid admitted action grant")
    if action not in supported_actions:
        raise PermissionError(f"{body.name!r} is not admitted for action {action!r}")
    path = _canonical_path(body.path)
    artifact_revision = body.revision.lower()
    artifact_revision_basis = _revision_basis(body)
    if not _is_sha256(artifact_revision):
        raise PermissionError(f"{body.name!r} lacks an exact catalog revision")
    catalog_identity, catalog_revision = _catalog_identity_and_revision(body, path)
    if not _is_sha256(catalog_revision):
        raise PermissionError(f"{body.name!r} lacks an exact catalog revision")
    grant_identity = f"{catalog_identity}:{body.name}"
    grant_digest = _grant_digest(
        name=body.name,
        path=path,
        kind=body.kind,
        artifact_revision=artifact_revision,
        artifact_revision_basis=artifact_revision_basis,
        catalog_identity=catalog_identity,
        catalog_revision=catalog_revision,
        grant_identity=grant_identity,
        supported_actions=supported_actions,
    )
    return ResolvedAction(
        action=action,
        name=body.name,
        path=path,
        kind=body.kind,
        artifact_revision=artifact_revision,
        artifact_revision_basis=artifact_revision_basis,
        catalog_identity=catalog_identity,
        catalog_revision=catalog_revision,
        grant_identity=grant_identity,
        grant_digest=grant_digest,
        supported_actions=supported_actions,
        body=body,
    )


def resolve_action(
    selector: str,
    action: str,
    bodies: Optional[List[Body]] = None,
    *,
    research: bool = False,
) -> Optional[ResolvedAction]:
    """Resolve one admitted artifact and bind it to one requested action."""
    # Keep the ordinary resolver call shape compatible with catalog adapters;
    # only the explicit research door needs the expanded selection argument.
    body = resolve(selector, bodies, research=True) if research else resolve(selector, bodies)
    if body is None:
        return None
    return _resolved_action_for_body(body, action)


def revalidate_action(
    resolved: ResolvedAction,
    *,
    action: Optional[str] = None,
    bodies: Optional[List[Body]] = None,
    research: bool = False,
) -> ResolvedAction:
    """Re-prove a transported action against the current catalog authority.

    This re-reads the admission record and compares its declared artifact
    revision, grant, and catalog binding.  It does not rehash a multi-shard
    artifact on every action; ``artifact_revision_basis`` states exactly what
    the catalog's declared revision covers.
    """
    if not isinstance(resolved, ResolvedAction):
        raise TypeError("revalidation requires a ResolvedAction value")
    if action is not None and action != resolved.action:
        raise ValueError(
            f"resolved action is bound to {resolved.action!r}, not {action!r}"
        )
    candidates = bodies if bodies is not None else catalog(research=research)
    matches = [
        body for body in candidates
        if body.name == resolved.name and _canonical_path(body.path) == resolved.path
    ]
    if len(matches) != 1:
        raise PermissionError(
            f"resolved action {resolved.name!r} no longer names one admitted catalog body"
        )
    current = _resolved_action_for_body(matches[0], resolved.action)
    if not current.has_same_binding(resolved):
        raise PermissionError(
            f"resolved action {resolved.name!r} no longer matches its admitted "
            "artifact/catalog/grant binding"
        )
    return current


def resolved_action_from_wire(
    raw: Mapping[str, Any], *, action: Optional[str] = None,
) -> ResolvedAction:
    """Parse an untrusted transport value and immediately revalidate it."""
    return revalidate_action(ResolvedAction.from_wire(raw), action=action)


def resolved_action_json(resolved: ResolvedAction) -> str:
    return json.dumps(resolved.to_wire(), sort_keys=True, separators=(",", ":"))


def resolved_action_from_json(
    raw: str, *, action: Optional[str] = None,
) -> ResolvedAction:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("resolved action is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("resolved action JSON must be an object")
    return resolved_action_from_wire(value, action=action)


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
