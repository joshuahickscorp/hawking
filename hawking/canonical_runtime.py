"""Canonical Hawking runtime roles behind the one ``hawkingd`` front door.

This is deliberately a *resolver*, not another catalog or scheduler.  Gravity
continues to admit executable artifacts, and ModelLake scheduling continues to
name the campaign roles.  This module joins those existing authorities at the
public-runtime boundary:

* callers may address any admitted native artifact by its immutable identity,
  or use ``pulsar``, ``magnetar``, ``themis``, or ``hawking-auto``;
* only a Gravity-admitted ``noetic_native`` artifact with an explicit,
  evidence-backed role binding may answer for a *logical role* name;
* a missing qualification is a structured withholding, never a Kimi/MLX or
  remote fallback.

The current machine has a serial single-resident implementation.  The role
contract intentionally does not claim that multiple workers are already
co-resident; it keeps the public API stable while that execution machinery is
earned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from .catalog import Body, ResolvedAction, catalog, resolve_action


ROLE_SCHEMA = "hawking.canonical_runtime_roles.v1"
SCHEDULING_SCHEMA = "hawking.modellake.scheduling_authority.v2"
ROLE_NAMES = ("pulsar", "magnetar", "themis")
AUTO_ROLE = "hawking-auto"
ROLE_BINDINGS_FIELD = "hawkingd_roles"
ROLE_QUALIFICATIONS = {
    "pulsar": frozenset({"RESIDENT_QUALIFIED"}),
    # These roles may begin as bounded workers.  They still need an admitted
    # native serve binding before the public front door can route to them.
    "magnetar": frozenset({"CAPABILITY_QUALIFIED", "RESIDENT_QUALIFIED"}),
    "themis": frozenset({"CAPABILITY_QUALIFIED", "RESIDENT_QUALIFIED"}),
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEDULING_AUTHORITY = (
    PROJECT_ROOT / "workspace/campaign/odyssey/MODELLAKE_SCHEDULING_AUTHORITY.json"
)


class RuntimeRoleUnavailable(RuntimeError):
    """A logical role exists, but no native artifact may serve it yet."""

    def __init__(
        self,
        role: str,
        reason: str,
        *,
        code: str = "ROLE_WITHHELD",
        state: str = "WITHHELD",
    ) -> None:
        self.role = str(role)
        self.reason = str(reason)
        self.code = str(code)
        self.state = str(state)
        super().__init__(f"{self.role}: {self.reason}")

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "state": self.state,
            "code": self.code,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RoleResolution:
    """One evidence-backed logical role to exact-artifact binding."""

    role: str
    action: ResolvedAction
    qualification: str
    evidence: tuple[str, ...]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "state": "READY",
            "qualification": self.qualification,
            "evidence": list(self.evidence),
            "source": self.source,
            "artifact": self.action.to_wire(),
        }


@dataclass(frozen=True)
class CanonicalRoute:
    """The model selector result used by the public OpenAI handler."""

    requested_model: str
    role: str
    action: ResolvedAction
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "role": self.role,
            "reason": self.reason,
            "artifact_id": self.action.name,
            "artifact_revision": self.action.artifact_revision,
            "artifact_kind": self.action.kind,
        }


def _normalise(value: object) -> str:
    return str(value or "").strip().casefold()


def _coerce_evidence(value: object) -> tuple[str, ...]:
    """Normalise a role binding's evidence into an ordered, deduplicated tuple.

    A binding is only evidence-backed when it names at least one non-empty
    evidence token.  Accepting a bare string (rather than iterating it into
    characters) keeps the resolver honest about what the authority declared.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        items = value
    else:
        return ()
    seen: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if token and token not in seen:
            seen.append(token)
    return tuple(seen)


def _authority_roles(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Read role vocabulary from the existing scheduling authority only."""
    campaign = document.get("campaign")
    if not isinstance(campaign, Mapping):
        return ()
    phases = campaign.get("phases")
    if not isinstance(phases, list):
        return ()
    for phase in phases:
        if not isinstance(phase, Mapping) or phase.get("id") != "STAR_SERIES":
            continue
        raw_roles = phase.get("roles")
        if not isinstance(raw_roles, Mapping):
            return ()
        names = tuple(sorted(
            name.casefold()
            for name in raw_roles
            if isinstance(name, str) and name.casefold() in ROLE_NAMES
        ))
        return names
    return ()


def load_scheduling_authority(
    path: Optional[Path | str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load the role vocabulary; absence withholds public role resolution."""
    source = Path(path) if path is not None else DEFAULT_SCHEDULING_AUTHORITY
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"scheduling authority is unavailable at {source}: {type(exc).__name__}"
    if not isinstance(value, dict) or value.get("schema") != SCHEDULING_SCHEMA:
        return None, f"scheduling authority at {source} has an unexpected schema"
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or authority.get("status") != "ACTIVE":
        return None, f"scheduling authority at {source} is not ACTIVE"
    roles = _authority_roles(value)
    if not roles:
        return None, f"scheduling authority at {source} does not declare Star roles"
    return value, None


def _binding_for_role(body: Body, role: str) -> tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """Validate the explicit role declaration stored in the Gravity artifact."""
    if body.source != "gravity":
        return None, "only a Gravity registry artifact can own a public role"
    if body.kind != "noetic_native":
        return None, "public roles require Hawking-native/Noetic execution"
    if not body.admitted:
        return None, "artifact is not admitted"
    detail = body.detail if isinstance(body.detail, Mapping) else {}
    bindings = detail.get(ROLE_BINDINGS_FIELD)
    if not isinstance(bindings, Mapping):
        return None, f"artifact has no explicit {ROLE_BINDINGS_FIELD} declaration"
    claim = bindings.get(role)
    if not isinstance(claim, Mapping):
        return None, f"artifact is not explicitly bound to role {role!r}"
    qualification = str(claim.get("qualification") or "").strip().upper()
    if qualification not in ROLE_QUALIFICATIONS[role]:
        expected = ", ".join(sorted(ROLE_QUALIFICATIONS[role]))
        return None, f"role {role!r} needs qualification {expected}"
    evidence = claim.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        return None, f"role {role!r} has no explicit qualification evidence"
    return claim, None


class CanonicalRoleRouter:
    """Resolve logical models without ever selecting a legacy provider."""

    def __init__(
        self,
        *,
        bodies_provider: Optional[Callable[[], Sequence[Body]]] = None,
        scheduling_authority: Optional[Path | str] = None,
    ) -> None:
        self._bodies_provider = bodies_provider or catalog
        self._scheduling_authority = scheduling_authority

    def _bodies(self) -> list[Body]:
        return list(self._bodies_provider())

    def _authority(self) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        return load_scheduling_authority(self._scheduling_authority)

    def resolve_role(self, role: str) -> RoleResolution:
        normalized = _normalise(role)
        if normalized not in ROLE_NAMES:
            raise RuntimeRoleUnavailable(
                normalized or "(empty)",
                "unknown Hawking logical role",
                code="UNKNOWN_ROLE",
            )
        authority, authority_error = self._authority()
        if authority is None:
            raise RuntimeRoleUnavailable(normalized, authority_error or "authority unavailable")
        if normalized not in _authority_roles(authority):
            raise RuntimeRoleUnavailable(
                normalized,
                "the active scheduling authority does not declare this role",
            )

        matches: list[RoleResolution] = []
        reasons: list[str] = []
        bodies = self._bodies()
        for body in bodies:
            claim, refusal = _binding_for_role(body, normalized)
            if claim is None:
                if body.kind == "noetic_native" or body.name.casefold() == normalized:
                    reasons.append(f"{body.name}: {refusal}")
                continue
            try:
                action = resolve_action(body.name, "serve", bodies=bodies)
            except (LookupError, PermissionError, ValueError) as exc:
                reasons.append(f"{body.name}: {type(exc).__name__}: {exc}")
                continue
            if action is None or action.body is None or action.kind != "noetic_native":
                reasons.append(f"{body.name}: no admitted native serve action")
                continue
            evidence = tuple(str(item).strip() for item in claim["evidence"])
            matches.append(RoleResolution(
                role=normalized,
                action=action,
                qualification=str(claim["qualification"]).strip().upper(),
                evidence=evidence,
                source=f"gravity:{ROLE_BINDINGS_FIELD}",
            ))
        if not matches:
            detail = (
                "no admitted native artifact has an explicit resident role binding"
                + (f" ({'; '.join(reasons[:3])})" if reasons else "")
            )
            raise RuntimeRoleUnavailable(normalized, detail)
        if len(matches) != 1:
            raise RuntimeRoleUnavailable(
                normalized,
                "multiple admitted native artifacts claim the role: "
                + ", ".join(match.action.name for match in matches),
                code="AMBIGUOUS_ROLE_BINDING",
            )
        return matches[0]

    @staticmethod
    def _objective_text(request: Optional[Mapping[str, Any]]) -> str:
        if not isinstance(request, Mapping):
            return ""
        for key in ("hawking_objective", "objective", "purpose"):
            value = request.get(key)
            if isinstance(value, str) and value.strip():
                return value.casefold()
        messages = request.get("messages")
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.casefold()
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, Mapping)
                ).casefold()
        return ""

    def choose_auto_role(self, request: Optional[Mapping[str, Any]]) -> tuple[str, str]:
        """Use deterministic request semantics; no model/provider fallback."""
        objective = self._objective_text(request)
        if any(token in objective for token in (
            "verify", "verification", "audit", "evaluate", "evaluation",
            "falsify", "review", "regression",
        )):
            return "themis", "objective classified as independent verification"
        if any(token in objective for token in (
            "hypothesis", "representation", "novel", "science", "scientific",
            "architecture", "research",
        )):
            return "magnetar", "objective classified as frontier reasoning"
        return "pulsar", "objective classified as routine local work"

    def route(
        self,
        model: object,
        request: Optional[Mapping[str, Any]] = None,
    ) -> CanonicalRoute:
        requested = _normalise(model) or AUTO_ROLE
        if requested == AUTO_ROLE:
            role, reason = self.choose_auto_role(request)
            resolved = self.resolve_role(role)
            return CanonicalRoute(AUTO_ROLE, role, resolved.action, reason)
        if requested in ROLE_NAMES:
            resolved = self.resolve_role(requested)
            return CanonicalRoute(requested, requested, resolved.action, "explicit logical role")

        # Immutable artifact identities remain first-class.  Roles are a
        # convenience/routing layer, never a gate that prevents somebody from
        # running another admitted Hawking-native body.  Admission is still
        # the existing Gravity record, and native-only remains the public
        # runtime boundary.
        bodies = self._bodies()
        try:
            action = resolve_action(str(model), "serve", bodies=bodies)
        except (LookupError, PermissionError, ValueError) as exc:
            raise RuntimeRoleUnavailable(
                str(model) or "(empty)", str(exc), code="MODEL_NOT_CANONICAL"
            ) from exc
        if action is None or action.body is None:
            raise RuntimeRoleUnavailable(
                str(model) or "(empty)",
                "no admitted Hawking artifact has that immutable identity",
                code="MODEL_NOT_CANONICAL",
            )
        if action.kind != "noetic_native":
            raise RuntimeRoleUnavailable(
                str(model),
                "no admitted Hawking artifact has that immutable identity for "
                "canonical native runtime; the artifact is admitted for legacy "
                "evidence but its runtime is "
                f"{action.kind!r}; canonical hawkingd serves native artifacts only",
                code="LEGACY_RUNTIME_WITHDRAWN",
            )
        return CanonicalRoute(
            str(model), "artifact", action,
            "explicit immutable admitted Hawking-native artifact",
        )

    def role_status(self, role: str) -> Dict[str, Any]:
        normalized = _normalise(role)
        try:
            return self.resolve_role(normalized).to_dict()
        except RuntimeRoleUnavailable as exc:
            return exc.to_dict()

    def status(self) -> Dict[str, Any]:
        authority, authority_error = self._authority()
        roles = [self.role_status(role) for role in ROLE_NAMES]
        available = [row["role"] for row in roles if row.get("state") == "READY"]
        native_artifacts = []
        for body in self._bodies():
            if body.kind != "noetic_native":
                continue
            try:
                action = resolve_action(body.name, "serve", bodies=self._bodies())
            except (LookupError, PermissionError, ValueError):
                continue
            if action is not None:
                native_artifacts.append({
                    "id": action.name,
                    "revision": action.artifact_revision,
                    "actions": list(action.supported_actions),
                })
        return {
            "schema": ROLE_SCHEMA,
            "owner": "hawkingd",
            "runtime_mode": "single-resident-serial-router",
            "role_authority": {
                "path": str(self._scheduling_authority or DEFAULT_SCHEDULING_AUTHORITY),
                "valid": authority is not None,
                "error": authority_error,
                "declared_roles": list(_authority_roles(authority)) if authority else [],
            },
            "roles": roles,
            "native_artifacts": native_artifacts,
            "hawking_auto": {
                "id": AUTO_ROLE,
                "state": "READY" if available else "WITHHELD",
                "available_roles": available,
                "policy": "deterministic objective classification; no legacy fallback",
            },
        }

    def public_models(
        self,
        *,
        loaded_action: Optional[ResolvedAction] = None,
    ) -> list[Dict[str, Any]]:
        """OpenAI-shaped logical aliases with explicit availability state."""
        rows: list[Dict[str, Any]] = []
        for role in ROLE_NAMES:
            status = self.role_status(role)
            ready = status.get("state") == "READY"
            artifact = status.get("artifact") if ready else None
            loaded = bool(
                ready and isinstance(artifact, Mapping)
                and loaded_action is not None
                and artifact.get("artifact", {}).get("id") == loaded_action.name
                and artifact.get("artifact", {}).get("revision")
                == loaded_action.artifact_revision
            )
            rows.append({
                "id": role,
                "object": "model",
                "created": 0,
                "owned_by": "hawking",
                "hawking": {
                    "logical_role": True,
                    "availability": "AVAILABLE" if ready else "WITHHELD",
                    "loaded": loaded,
                    "native_only": True,
                    "role_status": status,
                },
            })
        seen = {str(row["id"]).casefold() for row in rows}
        for body in self._bodies():
            if body.kind != "noetic_native":
                continue
            try:
                action = resolve_action(body.name, "serve", bodies=self._bodies())
            except (LookupError, PermissionError, ValueError):
                continue
            if action is None or action.name.casefold() in seen:
                continue
            rows.append({
                "id": action.name,
                "object": "model",
                "created": 0,
                "owned_by": "hawking",
                "hawking": {
                    "logical_role": False,
                    "immutable_artifact": True,
                    "availability": "AVAILABLE",
                    "loaded": bool(
                        loaded_action is not None
                        and action.has_same_binding(loaded_action)
                    ),
                    "native_only": True,
                    "resolved_action": action.to_wire(),
                },
            })
            seen.add(action.name.casefold())
        auto = self.status()["hawking_auto"]
        rows.append({
            "id": AUTO_ROLE,
            "object": "model",
            "created": 0,
            "owned_by": "hawking",
            "hawking": {
                "logical_role": True,
                "availability": "AVAILABLE" if auto["state"] == "READY" else "WITHHELD",
                "loaded": False,
                "native_only": True,
                "role_status": auto,
            },
        })
        return rows


__all__ = [
    "AUTO_ROLE",
    "CanonicalRoleRouter",
    "CanonicalRoute",
    "DEFAULT_SCHEDULING_AUTHORITY",
    "ROLE_BINDINGS_FIELD",
    "ROLE_NAMES",
    "ROLE_SCHEMA",
    "RoleResolution",
    "RuntimeRoleUnavailable",
    "load_scheduling_authority",
]
