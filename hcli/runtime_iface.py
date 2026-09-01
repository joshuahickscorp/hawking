"""ONE Runtime interface.

Scheduler policy stays in ``hcli.scheduler.Scheduler``. Session persistence
stays in ``hcli.session.SessionStore``. Context arithmetic stays in
``hcli.context_budget``. This module names the six planes a runtime has —
model semantics, backend, session id, context, health, performance profile —
and selects a backend (MLX, llama.cpp, native, later ones) without copying
those other authorities.

MLX is first-class. llama.cpp Q5_K numbers are archived science. The deleted
GGUF is not required to classify, load a genome, or construct an interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backends import (
    LlamaServerBackend,
    MlxServerBackend,
    OpenAICompatibleBackend,
    RuntimeBackend,
    is_mlx_model_dir,
    is_remote_endpoint,
    mlx_context_length,
    mlx_quantisation_label,
    model_bytes_at,
    quantisation_from_path,
)
from .hawking_native import config_for_model_path, is_hawking_native_path

# Historical llama.cpp artifact. Cited as science. Never a required open().
ARCHIVED_Q5K_GGUF_NAME = "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
ARCHIVED_Q5K_GGUF_REL = (
    "models/qwen3.8-27b-abliterated/" + ARCHIVED_Q5K_GGUF_NAME
)

BACKEND_KINDS = ("mlx", "llamacpp", "noetic_native", "remote")
BACKEND_ALIASES = {
    "native": "noetic_native",
    "hawking-native": "noetic_native",
    "hawking_native": "noetic_native",
    "openai": "remote",
    "openai-compatible": "remote",
    "openai_compatible": "remote",
}
PLANES = (
    "model_semantics",
    "backend",
    "session",
    "context",
    "health",
    "performance_profile",
)
# Owned elsewhere. A RuntimeInterface that grows one of these is a regression.
FOREIGN_AUTHORITIES = (
    "Scheduler",
    "SessionStore",
    "Mission",
    "MemGate",
    "MutationLock",
    "WorkUnitDAG",
)


def archived_q5k_gguf_path() -> Path:
    return Path.home() / ARCHIVED_Q5K_GGUF_REL


def q5k_gguf_required() -> bool:
    """The deleted Q5_K GGUF is never a load-bearing dependency."""
    return False


def artifact_present(path: Optional[str]) -> bool:
    if not path:
        return False
    if is_remote_endpoint(str(path)):
        # A remote selection is present when the endpoint is syntactically
        # valid; readiness is a separate live-health observation.
        return True
    expanded = os.path.realpath(os.path.expanduser(str(path)))
    if os.path.isfile(expanded):
        return True
    return is_mlx_model_dir(expanded) or is_hawking_native_path(expanded)


def classify_backend(
    model_path: Optional[str] = None,
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Name the backend kind from the artifact. Does not open the Q5_K GGUF.

    Env ``HCLI_RUNTIME_BACKEND`` wins when it is one of BACKEND_KINDS.
    MLX directories win over a ``.gguf`` suffix. A missing GGUF is still
    classified ``llamacpp`` by suffix — classification is not existence.
    Unknown paths default to ``mlx`` (first-class), not to llama.cpp, so a
    deleted GGUF cannot become the implicit runtime.
    """
    environ = env if env is not None else os.environ
    forced = (environ.get("HCLI_RUNTIME_BACKEND") or "").strip().lower()
    forced = BACKEND_ALIASES.get(forced, forced)
    if forced in BACKEND_KINDS:
        return forced
    if not model_path:
        return "mlx"
    if is_remote_endpoint(str(model_path)):
        return "remote"
    expanded = os.path.realpath(os.path.expanduser(str(model_path)))
    if is_mlx_model_dir(expanded):
        return "mlx"
    if is_hawking_native_path(expanded):
        return "noetic_native"
    lower = expanded.lower()
    if lower.endswith(".gguf"):
        return "llamacpp"
    if lower.endswith((".gravity", ".nx", ".noetic")):
        return "noetic_native"
    if os.path.isdir(expanded):
        return "mlx"
    return "mlx"


def make_backend_for_model(
    model_path: str,
    *,
    port: Optional[int] = None,
    n_slots: int = 1,
    ctx_size: Optional[int] = None,
    index: Optional[int] = None,
    **_ignored: Any,
) -> RuntimeBackend:
    """Construct the matching RuntimeBackend. Does not spawn a process.

    ``index`` is accepted so RuntimePool's factory calling convention
    (model_path, port, n_slots, index) can land here; backends that do
    not use it ignore it. Scheduler/session kwargs are rejected by not
    being in this signature.
    """
    del index  # pool factory passes it; backends do not own scheduling
    kind = classify_backend(model_path)
    if kind == "mlx":
        return MlxServerBackend(
            model_path=model_path,
            port=port,
            n_slots=n_slots,
        )
    if kind == "noetic_native":
        from .backends import NativeRuntimeBackend

        return NativeRuntimeBackend(model_path=model_path, port=port, n_slots=n_slots)
    if kind == "remote":
        return OpenAICompatibleBackend(model_path=model_path, port=port, n_slots=n_slots)
    return LlamaServerBackend(
        model_path=model_path,
        port=port,
        ctx_size=ctx_size,
        n_slots=n_slots,
    )


@dataclass
class ModelSemantics:
    identity: str
    backend_kind: str
    artifact_kind: str
    path: Optional[str]
    bytes: Optional[int]
    architecture: Optional[str] = None
    quantisation: Optional[str] = None
    context_length: Optional[int] = None
    requires_on_disk: bool = False
    present: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "backend_kind": self.backend_kind,
            "artifact_kind": self.artifact_kind,
            "path": self.path,
            "bytes": self.bytes,
            "architecture": self.architecture,
            "quantisation": self.quantisation,
            "context_length": self.context_length,
            "requires_on_disk": self.requires_on_disk,
            "present": self.present,
        }


def model_semantics_for(path: Optional[str]) -> ModelSemantics:
    if not path:
        return ModelSemantics(
            identity="unspecified",
            backend_kind="mlx",
            artifact_kind="unspecified",
            path=None,
            bytes=None,
            requires_on_disk=False,
            present=False,
        )
    expanded = os.path.realpath(os.path.expanduser(str(path)))
    if is_remote_endpoint(str(path)):
        # Keep URLs opaque.  In particular, do not run them through
        # os.path.realpath or persist query strings that might contain a key.
        remote = OpenAICompatibleBackend(str(path))
        identity = remote.identity()
        return ModelSemantics(
            identity=str(identity.get("model_identity") or "remote"),
            backend_kind="remote",
            artifact_kind="remote_endpoint",
            path=str(identity.get("model_path") or path),
            bytes=None,
            architecture="provider-managed",
            quantisation="provider-managed",
            context_length=None,
            requires_on_disk=False,
            present=True,
        )
    kind = classify_backend(expanded)
    present = artifact_present(expanded)
    if kind == "mlx":
        quant = mlx_quantisation_label(expanded) if present else None
        ctx = mlx_context_length(expanded) if present else None
        size = model_bytes_at(expanded) if present else None
        return ModelSemantics(
            identity=f"mlx:{expanded}",
            backend_kind="mlx",
            artifact_kind="mlx_dir",
            path=expanded,
            bytes=size,
            quantisation=quant,
            context_length=ctx,
            requires_on_disk=False,
            present=present,
        )
    if kind == "llamacpp":
        size = model_bytes_at(expanded) if present else None
        return ModelSemantics(
            identity=f"llamacpp:{expanded}",
            backend_kind="llamacpp",
            artifact_kind="gguf",
            path=expanded,
            bytes=size,
            quantisation=quantisation_from_path(expanded),
            requires_on_disk=False,
            present=present,
        )
    if kind == "noetic_native":
        try:
            native = config_for_model_path(expanded)
            identity = native.identity()
            artifact_root = identity.get("artifact_root") or expanded
            artifact_inventory = identity.get("artifact_inventory") or {}
            size = artifact_inventory.get("artifact_bytes")
            context = identity.get("max_seq_len")
            architecture = identity.get("architecture")
            quantisation = identity.get("quantisation")
        except Exception:
            artifact_root = expanded
            size = model_bytes_at(expanded) if present else None
            context = None
            architecture = None
            quantisation = None
        return ModelSemantics(
            identity=f"noetic_native:{artifact_root}",
            backend_kind="noetic_native",
            artifact_kind="native",
            path=str(artifact_root),
            bytes=int(size) if isinstance(size, (int, float)) else None,
            architecture=str(architecture) if architecture else None,
            quantisation=str(quantisation) if quantisation else None,
            context_length=int(context) if isinstance(context, (int, float)) else None,
            requires_on_disk=False,
            present=present,
        )
    return ModelSemantics(
        identity=f"noetic_native:{expanded}",
        backend_kind="noetic_native",
        artifact_kind="native",
        path=expanded,
        bytes=model_bytes_at(expanded) if present else None,
        requires_on_disk=False,
        present=present,
    )


@dataclass
class RuntimeHealth:
    ready: bool
    reason: str
    backend_kind: str
    attached: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "backend_kind": self.backend_kind,
            "attached": self.attached,
            "details": dict(self.details),
        }


@dataclass
class RuntimeInterface:
    """The one runtime surface used by the pool, the experiment engine, and genomes.

    It does not schedule WorkUnits, persist sessions, or admit GPU memory.
    Those stay Scheduler / SessionStore / MemGate.
    """

    model: ModelSemantics
    backend_kind: str
    session_id: Optional[str] = None
    context: Optional[Any] = None
    health: Optional[RuntimeHealth] = None
    profile: Optional[Dict[str, Any]] = None
    backend: Optional[RuntimeBackend] = None
    persistent_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.health is None:
            self.health = RuntimeHealth(
                ready=False,
                reason="unattached",
                backend_kind=self.backend_kind,
                attached=False,
            )
        if self.persistent_id is None:
            self.persistent_id = id(self)

    @classmethod
    def from_artifact(cls, path: Optional[str]) -> "RuntimeInterface":
        model = model_semantics_for(path)
        return cls(model=model, backend_kind=model.backend_kind)

    @classmethod
    def from_control_set(cls, control: Dict[str, Any]) -> "RuntimeInterface":
        """Build the live MLX interface from CONVENTIONAL_CONTROL_SET.json.

        Does not re-measure. Does not open the archived GGUF.
        """
        live = control.get("live") if isinstance(control.get("live"), dict) else {}
        artifact = live.get("artifact") if isinstance(live.get("artifact"), dict) else {}
        path = artifact.get("path")
        iface = cls.from_artifact(path if isinstance(path, str) else None)
        iface.backend_kind = "mlx"
        iface.model.backend_kind = "mlx"
        iface.model.requires_on_disk = False
        metrics = live.get("metrics") if isinstance(live.get("metrics"), dict) else {}
        iface.profile = extract_live_mlx_profile(metrics, source_receipt=(
            "receipts/headless/CONVENTIONAL_CONTROL_SET.json"
        ))
        iface.health = RuntimeHealth(
            ready=True,
            reason="profile recorded from control set; not a live spawn",
            backend_kind="mlx",
            attached=False,
            details={"remeasured": False, "status": live.get("status")},
        )
        return iface

    def bind_session(self, session_id: Optional[str]) -> None:
        """Record a session id. Does not write SessionStore."""
        self.session_id = session_id

    def bind_context(self, budget: Any) -> None:
        """Hold a ContextBudget produced elsewhere. Does not recompute it."""
        self.context = budget

    def attach(self, backend: RuntimeBackend) -> None:
        self.backend = backend
        ident = {}
        try:
            ident = backend.identity()
        except Exception as exc:  # noqa: BLE001
            ident = {"identity_error": f"{type(exc).__name__}: {exc}"}
        try:
            from .providers import profile_from_backend

            self.profile = profile_from_backend(backend).to_dict()
        except Exception as exc:  # noqa: BLE001 - profile is diagnostic, not admission
            self.profile = {
                "schema": "hcli.provider.profile.v1",
                "provider": self.backend_kind,
                "model_id": self.model.identity,
                "profile_error": f"{type(exc).__name__}: {exc}",
            }
        self.health = RuntimeHealth(
            ready=True,
            reason="backend attached (spawn is the pool's job)",
            backend_kind=self.backend_kind,
            attached=True,
            details={"identity": ident},
        )

    def probe_health(self) -> RuntimeHealth:
        if self.backend is None:
            self.health = RuntimeHealth(
                ready=False,
                reason="unattached",
                backend_kind=self.backend_kind,
                attached=False,
            )
            return self.health
        ready_fn = getattr(self.backend, "ready", None)
        ok = False
        reason = "attached, ready() not probed"
        if callable(ready_fn):
            try:
                ok = bool(ready_fn(0.01))
                reason = "ready" if ok else "backend.ready returned false"
            except Exception as exc:  # noqa: BLE001
                ok = False
                reason = f"{type(exc).__name__}: {exc}"
        self.health = RuntimeHealth(
            ready=ok,
            reason=reason,
            backend_kind=self.backend_kind,
            attached=True,
        )
        return self.health

    def to_dict(self) -> Dict[str, Any]:
        ctx = None
        if self.context is not None:
            ctx = {
                "type": type(self.context).__name__,
                "module": getattr(type(self.context), "__module__", None),
            }
        return {
            "model": self.model.to_dict(),
            "backend_kind": self.backend_kind,
            "session_id": self.session_id,
            "context": ctx,
            "health": self.health.to_dict() if self.health else None,
            "profile": self.profile,
            "backend_attached": self.backend is not None,
            "persistent_id": self.persistent_id,
            "owns": list(PLANES),
            "does_not_own": list(FOREIGN_AUTHORITIES),
        }


def extract_live_mlx_profile(
    metrics: Dict[str, Any],
    *,
    source_receipt: str,
) -> Dict[str, Any]:
    """Copy the seven live MLX metrics. Never re-measure."""

    def _val(name: str) -> Any:
        node = metrics.get(name)
        if isinstance(node, dict):
            return node.get("value")
        return None

    startup = _val("startup")
    prefill = _val("prefill")
    decode = _val("decode_tps")
    context = _val("context_limit")
    peak = _val("peak_memory")
    return {
        "status": "RECORDED",
        "remeasured": False,
        "source_receipt": source_receipt,
        "runtime": "mlx",
        "startup_s": startup,
        "prefill_tps": prefill,
        "decode_tps": decode,
        "context_tokens": context,
        "peak_memory_gb": peak,
        "headline": {
            "startup_s": round(float(startup), 3) if isinstance(startup, (int, float)) else None,
            "prefill_tps": round(float(prefill), 2) if isinstance(prefill, (int, float)) else None,
            "decode_tps": round(float(decode), 2) if isinstance(decode, (int, float)) else None,
            "context_tokens": int(context) if isinstance(context, (int, float)) else None,
            "peak_memory_gb": round(float(peak), 2) if isinstance(peak, (int, float)) else None,
        },
        "metrics_present": sorted(
            k for k in (
                "startup", "prefill", "decode_tps", "context_limit",
                "concurrency", "peak_memory", "tool_shaped_tps",
            )
            if k in metrics
        ),
    }


def runtime_interface_census() -> Dict[str, Any]:
    import inspect

    src = inspect.getsource(RuntimeInterface)
    duplicated = [name for name in FOREIGN_AUTHORITIES if f"class {name}" in src]
    return {
        "interface": "hcli.runtime_iface.RuntimeInterface",
        "pool_member": "hcli.runtime.Runtime",
        "pool": "hcli.runtime.RuntimePool",
        "backend_abc": "hcli.backends.RuntimeBackend",
        "planes": {
            "model_semantics": "hcli.runtime_iface.ModelSemantics",
            "backend": "hcli.backends.RuntimeBackend",
            "session": "hcli.session.Session (id only; SessionStore is not copied)",
            "context": "hcli.context_budget.ContextBudget (held, not recomputed)",
            "health": "hcli.runtime_iface.RuntimeHealth",
            "performance_profile": "hcli.genomes.runtime_genome.RuntimeGenome",
        },
        "backend_kinds": list(BACKEND_KINDS),
        "mlx_first_class": True,
        "llamacpp_status": "archived science when the Q5_K GGUF is absent",
        "noetic_native": "profile-driven native one-shot/resident connector",
        "scheduler_duplicated": "Scheduler" in duplicated,
        "session_store_duplicated": "SessionStore" in duplicated,
        "foreign_classes_defined_here": duplicated,
        "q5k_gguf_required": q5k_gguf_required(),
        "q5k_gguf_path": str(archived_q5k_gguf_path()),
        "q5k_gguf_present": archived_q5k_gguf_path().is_file(),
        "default_kind_without_path": classify_backend(None),
        "missing_q5k_kind": classify_backend(str(archived_q5k_gguf_path())),
    }


def default_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "hcli" / "runtime.py").is_file() and (
            parent / "receipts" / "headless"
        ).is_dir():
            return parent
    return Path.cwd()


def load_control_set(repo: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo) if repo is not None else default_repo_root()
    path = root / "receipts" / "headless" / "CONVENTIONAL_CONTROL_SET.json"
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not an object")
    return data
