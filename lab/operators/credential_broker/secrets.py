"""Credential isolation: models and transport callers never receive the token.

Bible §7: "The models never receive the Hugging Face token."

Proven patterns reused here:
- Kimi/GLM set ``HF_HUB_DISABLE_IMPLICIT_TOKEN=1`` before any Hub import.
- GLM ``hf_hub_download(..., token=False)`` on public paths so the Hub client
  does not pull ambient credentials.
- Tokens, when required for gated sources, live only inside this broker process
  and are applied solely by broker-owned transport — never returned as strings
  to model loaders, teacher runtimes, or Gravity transforms.

This scaffold does **not** perform network I/O. It defines the handle and the
environment contract so future live executors cannot accidentally reintroduce
token leakage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class CredentialBrokerError(RuntimeError):
    """Credential or environment contract failed closed."""


# Environment keys the broker owns. Callers must not inject tokens via these
# after the broker has locked the process environment for a session.
_BROKER_ENV_LOCKS = (
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_OFFLINE",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
)


@dataclass(frozen=True)
class TokenHandle:
    """Opaque handle proving a credential session exists without exposing the secret.

    The raw token is never stored on this object. Possession of a handle is only
    meaningful inside the broker that minted it; serialising a handle does not
    reconstitute a credential.
    """

    session_id: str
    source: str  # e.g. "env:HF_TOKEN" | "none:public" | "keychain:..."
    allows_gated: bool
    minted_for_repository: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "allows_gated": self.allows_gated,
            "minted_for_repository": self.minted_for_repository,
            "token_material_present_on_handle": False,
            "token_export_forbidden": True,
        }


class CredentialBroker:
    """Process-local credential authority for model-source acquisition.

    Capabilities (Bible §7):
    - search public model metadata (no token)
    - resolve official repository (no token for public; token only for gated)
    - pin immutable revision
    - list exact files and sizes
    - download approved ranges (future live transport; token never leaves broker)
    - resume / verify / evict (coordinated with lifecycle)

    Non-capabilities:
    - return token strings
    - inject tokens into model process environments
    - accumulate source bodies (delegated to lifecycle + reclaim)
    """

    def __init__(self) -> None:
        self._locked = False
        self._sessions: dict[str, str] = {}  # session_id -> source label only
        self._token_material: dict[str, str] = {}  # session_id -> secret (never exported)
        self._seq = 0

    def public_environment(self, *, runtime_root: str | None = None) -> dict[str, str]:
        """Environment block for public metadata / public-path streaming.

        Mirrors Kimi ``_configure_environment`` + GLM fetch defaults:
        disable implicit token, disable telemetry, isolate caches when a
        runtime root is supplied.
        """
        env: dict[str, str] = {
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Explicit false-token for public downloads (hub client must not
            # fall back to ambient credentials). Live transport uses this
            # contract; this scaffold only documents and validates it.
            "HAWKING_CREDENTIAL_BROKER_PUBLIC_TOKEN_POLICY": "token=False",
        }
        if runtime_root:
            env["HF_HOME"] = f"{runtime_root}/hf-home"
            env["HF_HUB_CACHE"] = f"{runtime_root}/hub-cache"
            env["HF_XET_CACHE"] = f"{runtime_root}/xet-cache"
            env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
        return env

    def apply_public_environment(self, *, runtime_root: str | None = None) -> Mapping[str, str]:
        """Install the public-path environment contract into ``os.environ``."""
        if self._locked:
            raise CredentialBrokerError("environment is locked for this broker session")
        block = self.public_environment(runtime_root=runtime_root)
        # Refuse to run if ambient tokens are already visible as process env
        # while claiming a public-only path — the broker must own the choice.
        ambient = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if ambient:
            # Do not print or store the value. Record only that ambient existed.
            raise CredentialBrokerError(
                "ambient HF_TOKEN/HUGGING_FACE_HUB_TOKEN is set; refuse public-path "
                "environment install so models cannot inherit an implicit credential. "
                "Use mint_gated_session for gated sources, or unset ambient tokens."
            )
        os.environ.update(block)
        return block

    def mint_public_session(self, *, repository: str | None = None) -> TokenHandle:
        """Mint a session that authorises public metadata/range work only."""
        self._seq += 1
        session_id = f"public-{self._seq:04d}"
        self._sessions[session_id] = "none:public"
        return TokenHandle(
            session_id=session_id,
            source="none:public",
            allows_gated=False,
            minted_for_repository=repository,
        )

    def mint_gated_session(
        self,
        *,
        token: str,
        repository: str,
        source_label: str = "caller_supplied",
    ) -> TokenHandle:
        """Mint a gated session. The token is retained only inside this broker.

        Callers must pass the token once; they must not log it, write it to a
        receipt, or hand it to a model loader. Subsequent broker methods take
        only the ``TokenHandle``.
        """
        if not isinstance(token, str) or not token.strip():
            raise CredentialBrokerError("gated session requires a non-empty token")
        if not isinstance(repository, str) or not repository.strip():
            raise CredentialBrokerError("gated session requires a target repository")
        # Strip so trailing newlines from secret files do not leak into headers.
        secret = token.strip()
        self._seq += 1
        session_id = f"gated-{self._seq:04d}"
        self._sessions[session_id] = source_label
        self._token_material[session_id] = secret
        return TokenHandle(
            session_id=session_id,
            source=source_label,
            allows_gated=True,
            minted_for_repository=repository.strip(),
        )

    def lock_environment(self) -> None:
        """Prevent further ambient env mutation claims for this broker instance."""
        self._locked = True

    def has_session(self, handle: TokenHandle) -> bool:
        return handle.session_id in self._sessions

    def authorization_header(self, handle: TokenHandle) -> Mapping[str, str] | None:
        """Return Authorization headers for broker-owned transport only.

        Intentionally not part of any public model-facing API surface. Scaffold
        returns the mapping only when a gated session exists; live transport
        should consume this inside the same process and never re-export it.
        """
        if handle.session_id not in self._sessions:
            raise CredentialBrokerError("unknown token handle")
        secret = self._token_material.get(handle.session_id)
        if secret is None:
            return None
        return {"Authorization": f"Bearer {secret}"}

    def revoke(self, handle: TokenHandle) -> None:
        """Drop token material for a session. Idempotent."""
        self._sessions.pop(handle.session_id, None)
        self._token_material.pop(handle.session_id, None)

    def revoke_all(self) -> None:
        self._sessions.clear()
        self._token_material.clear()

    def assert_no_token_in_mapping(self, value: Mapping[str, object], *, label: str) -> None:
        """Refuse receipts/payloads that accidentally embed credential material."""
        forbidden_keys = {
            "token",
            "hf_token",
            "huggingface_token",
            "authorization",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_TOKEN",
            "bearer",
        }
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in forbidden_keys or key_l.endswith("_token"):
                raise CredentialBrokerError(
                    f"{label} contains forbidden credential key {key!r}; "
                    "models and receipts never receive the HF token"
                )
            if isinstance(item, Mapping):
                self.assert_no_token_in_mapping(item, label=f"{label}.{key}")  # type: ignore[arg-type]
            elif isinstance(item, str) and item.startswith("hf_"):
                # Hugging Face user access tokens historically start with hf_.
                # Receipts must never carry them.
                raise CredentialBrokerError(
                    f"{label}.{key} looks like an HF user token; refused"
                )
