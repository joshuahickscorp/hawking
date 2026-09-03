"""Profile-driven native model connector and resident transport.

The HCLI engine already owns prompt construction, schema retry, mutation
transactions, and deterministic verification.  This module only translates
that model payload into the selected native executable's request format and
back again.
The one-shot path is intentionally small and is also the fallback when the
resident executable is not available yet.

The current shipped profile is Hawking's Qwen3.8 resident; the connector and
profile contract are reusable by another native model with the same JSONL
wire shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROTOCOL_SCHEMA = "hawking.native.resident.v1"
QWEN38_PROTOCOL_SCHEMA = "hawking.qwen38.resident.v1"

REQUIRED_FUSION_ENV: Dict[str, str] = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}


class HawkingNativeError(RuntimeError):
    """A native connector or resident protocol failure."""


class HawkingNativeConfigError(HawkingNativeError):
    """The native profile is incomplete or unsafe to run."""


class HawkingNativeProtocolError(HawkingNativeError):
    """The native executable did not return the promised structured record."""


class HawkingNativeTimeout(HawkingNativeError):
    """A native request exceeded its caller-owned timeout."""


def _absolute_path(value: Any, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise HawkingNativeConfigError(f"{field_name} is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise HawkingNativeConfigError(
            f"{field_name} must be an absolute path, got {raw!r}"
        )
    return os.path.realpath(str(path))


def _positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HawkingNativeConfigError(
            f"{field_name} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise HawkingNativeConfigError(f"{field_name} must be positive")
    return parsed


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _safe(value: Any) -> Any:
    """Return a JSON-stable copy for optional provider telemetry."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _sha16(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _artifact_bytes(root: Path) -> Optional[int]:
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        return None
    return total


def _artifact_inventory(root: Path) -> Dict[str, Any]:
    report = root / "MIX_REPORT.json"
    try:
        body = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        body = {}
    declared_bytes = None
    declared_source = None
    if isinstance(body, dict):
        for key in ("artifact_bytes", "payload_bytes", "total_bytes", "size_bytes"):
            value = body.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                declared_bytes = int(value)
                declared_source = f"MIX_REPORT.json:{key}"
                break
    # A status/identity call must not recursively walk a multi-GB model just
    # to produce a receipt. Exact bytes come from the artifact manifest when
    # available; an operator can request a deliberate scan for an unmanifested
    # fixture with HCLI_SCAN_ARTIFACT_BYTES=1.
    scanned_bytes = None
    if declared_bytes is None and os.environ.get("HCLI_SCAN_ARTIFACT_BYTES") == "1":
        scanned_bytes = _artifact_bytes(root)
        if scanned_bytes is not None:
            declared_source = "recursive_scan"
    return {
        "mix_id": body.get("mix_id"),
        "catalog": body.get("catalog"),
        "n_tensors": body.get("n_tensors"),
        "artifact_bytes": declared_bytes if declared_bytes is not None else scanned_bytes,
        "artifact_bytes_source": declared_source,
        "artifact_bytes_exact": declared_bytes is not None or scanned_bytes is not None,
    }


@dataclass
class HawkingNativeConfig:
    """The execution contract for one external native model runtime.

    The current shipped provider is Hawking's Qwen3.8 resident.  The
    transport itself is deliberately described in model-neutral terms so a
    different native executable can be selected by a profile without making
    HCLI, AgentOS, or the verifier know its architecture.
    """

    artifact_root: str
    tokenizer: str
    binary: str
    max_seq_len: int = 8192
    generation: Dict[str, Any] = field(default_factory=dict)
    fusion_env: Dict[str, str] = field(default_factory=dict)
    runtime_env: Dict[str, str] = field(default_factory=dict)
    resident_identity: str = "native-resident"
    mode: str = "auto"
    resident_binary: Optional[str] = None
    family: str = "unknown"
    architecture: str = "unknown"
    param_class: str = "?B"
    quantisation: str = "unknown"
    runtime: str = "native"
    protocol: str = PROTOCOL_SCHEMA
    require_fusion_env: bool = False
    executable_profile: str = "unspecified"
    physical_ebpw: Optional[float] = None
    qualification: str = "profile supplied; qualification not declared"
    current_runtime: Dict[str, Any] = field(default_factory=dict)
    profile_schema: str = "hcli.provider.profile.v1"
    model_id: Optional[str] = None
    compiler: Dict[str, Any] = field(default_factory=dict)
    representation: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    prompt_contract: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)
    fallbacks: List[Any] = field(default_factory=list)
    hot_bytes: Optional[int] = None
    machine_genome: Dict[str, Any] = field(default_factory=dict)
    receipts: List[str] = field(default_factory=list)
    provider: str = "native"

    def __post_init__(self) -> None:
        self.artifact_root = _absolute_path(self.artifact_root, "artifact_root")
        self.tokenizer = _absolute_path(self.tokenizer, "tokenizer")
        self.binary = _absolute_path(self.binary, "binary")
        if self.resident_binary:
            self.resident_binary = _absolute_path(
                self.resident_binary, "resident_binary"
            )
        self.max_seq_len = _positive_int(self.max_seq_len, "max_seq_len", 8192)
        self.generation = dict(self.generation or {})
        self.fusion_env = {
            str(key): str(value) for key, value in dict(self.fusion_env or {}).items()
        }
        self.runtime_env = {
            str(key): str(value) for key, value in dict(self.runtime_env or {}).items()
        }
        self.mode = str(self.mode or "auto").strip().lower()
        if self.mode not in {"auto", "one_shot", "resident"}:
            raise HawkingNativeConfigError(
                f"mode must be auto, one_shot, or resident, got {self.mode!r}"
            )
        self.resident_identity = str(self.resident_identity or "native-resident")
        self.family = str(self.family or "unknown")
        self.architecture = str(self.architecture or self.family)
        self.param_class = str(self.param_class or "?B")
        self.quantisation = str(self.quantisation or "unknown")
        self.runtime = str(self.runtime or "native")
        self.protocol = str(self.protocol or PROTOCOL_SCHEMA)
        self.require_fusion_env = _coerce_bool(self.require_fusion_env, True)
        self.current_runtime = dict(self.current_runtime or {})
        self.profile_schema = str(self.profile_schema or "hcli.provider.profile.v1")
        self.model_id = str(self.model_id) if self.model_id else None
        self.compiler = dict(self.compiler or {})
        self.representation = dict(self.representation or {})
        self.capabilities = dict(self.capabilities or {})
        self.prompt_contract = dict(self.prompt_contract or {})
        self.limits = dict(self.limits or {})
        self.fallbacks = list(self.fallbacks or [])
        if self.hot_bytes is not None:
            try:
                self.hot_bytes = int(self.hot_bytes)
            except (TypeError, ValueError):
                self.hot_bytes = None
        self.machine_genome = dict(self.machine_genome or {})
        self.receipts = [str(item) for item in (self.receipts or [])]
        self.provider = str(self.provider or "native")

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "HawkingNativeConfig":
        if not isinstance(data, dict):
            raise HawkingNativeConfigError("native profile must be a JSON object")
        generation = data.get("generation")
        if not isinstance(generation, dict):
            generation = data.get("generation_settings")
        if not isinstance(generation, dict):
            generation = {}
        fusion = data.get("fusion_env")
        if not isinstance(fusion, dict):
            fusion = data.get("required_fusion_env")
        if not isinstance(fusion, dict):
            fusion = {}
        runtime_env = data.get("runtime_env", data.get("environment", {}))
        if not isinstance(runtime_env, dict):
            runtime_env = {}
        return cls(
            artifact_root=data.get("artifact_root", ""),
            tokenizer=data.get("tokenizer", ""),
            binary=data.get("binary", ""),
            max_seq_len=data.get("max_seq_len", 8192),
            generation=generation,
            fusion_env=fusion,
            runtime_env=runtime_env,
            resident_identity=data.get("resident_identity", data.get("resident", "native-resident")),
            mode=data.get("mode", data.get("execution_mode", "auto")),
            resident_binary=data.get("resident_binary"),
            family=data.get("family", "unknown"),
            architecture=data.get("architecture", data.get("model_architecture", "unknown")),
            param_class=data.get("param_class", data.get("parameters", "?B")),
            quantisation=data.get("quantisation", data.get("quantization", "unknown")),
            runtime=data.get("runtime", "native"),
            protocol=data.get("protocol", PROTOCOL_SCHEMA),
            require_fusion_env=_coerce_bool(data.get("require_fusion_env"), False),
            executable_profile=data.get("executable_profile", "unspecified"),
            physical_ebpw=data.get("physical_ebpw"),
            qualification=data.get(
                "qualification", "profile supplied; qualification not declared"
            ),
            current_runtime=data.get("current_runtime") or {},
            profile_schema=data.get("profile_schema", "hcli.provider.profile.v1"),
            model_id=data.get("model_id"),
            compiler=data.get("compiler") or {},
            representation=data.get("representation") or {},
            capabilities=data.get("capabilities") or {},
            prompt_contract=data.get("prompt_contract") or {},
            limits=data.get("limits") or {},
            fallbacks=data.get("fallbacks") or [],
            hot_bytes=data.get("hot_bytes"),
            machine_genome=data.get("machine_genome") or {},
            receipts=data.get("receipts") or [],
            provider=data.get("provider", "native"),
        )

    @classmethod
    def from_file(cls, path: os.PathLike[str] | str) -> "HawkingNativeConfig":
        profile = Path(path).expanduser()
        if not profile.is_absolute():
            raise HawkingNativeConfigError(
                f"native profile must be an absolute path, got {profile}"
            )
        try:
            data = json.loads(profile.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HawkingNativeConfigError(
                f"cannot read native profile {profile}: {exc}"
            ) from exc
        return cls.from_mapping(data)

    @classmethod
    def defaults(cls, repo_root: Optional[os.PathLike[str] | str] = None) -> "HawkingNativeConfig":
        repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[1]
        artifact = Path(
            os.environ.get(
                "HCLI_HAWKING_ARTIFACT_ROOT",
                str(Path.home() / "noetic" / "NOETIC_PARENT_A"),
            )
        ).expanduser()
        binary = Path(
            os.environ.get(
                "HCLI_HAWKING_BINARY",
                str(repo / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"),
            )
        ).expanduser()
        resident_binary = Path(
            os.environ.get(
                "HCLI_HAWKING_RESIDENT_BINARY",
                str(repo / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"),
            )
        ).expanduser()
        return cls(
            artifact_root=str(artifact),
            tokenizer=os.environ.get("HCLI_HAWKING_TOKENIZER", str(artifact / "tokenizer.json")),
            binary=str(binary),
            resident_binary=str(resident_binary),
            max_seq_len=os.environ.get("HCLI_HAWKING_MAX_SEQ_LEN", 8192),
            generation={
                "max_new_tokens": int(os.environ.get("HCLI_HAWKING_MAX_NEW_TOKENS", "2048")),
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "do_sample": False,
                "enable_thinking": False,
            },
            fusion_env=dict(REQUIRED_FUSION_ENV),
            runtime_env={},
            resident_identity=os.environ.get("HCLI_HAWKING_RESIDENT_IDENTITY", "sealed-3.14"),
            mode=os.environ.get("HCLI_HAWKING_NATIVE_MODE", "auto"),
            family="qwen3.8",
            architecture="qwen3.8",
            param_class="27B",
            quantisation="native-packed",
            runtime="hawking-native",
            protocol=QWEN38_PROTOCOL_SCHEMA,
            require_fusion_env=True,
            executable_profile="release-fast",
            physical_ebpw=3.1393,
            qualification="capability 30/43 historical sealed contract",
            current_runtime={
                "complete_tps_current_measured": 24.4086,
                "complete_tps_historical_qualified": 34.0,
                "fallbacks": 0,
            },
            model_id="qwen3.8-27b-sealed-3.14",
            compiler={
                "language": "rust",
                "profile": "release-fast",
                "source": "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident",
            },
            representation={
                "kind": "native-packed",
                "physical_ebpw": 3.1393,
                "weights_loaded_once": True,
            },
            prompt_contract={
                "renderer": "qwen-chat-template-or-closed-think-fallback",
                "fallback_template": "qwen_closed_think",
                "thinking_parameter": "enable_thinking",
                "supports_thinking": True,
                "token_count_authority": "native-resident",
            },
            limits={"context": 8192, "max_new_tokens": 2048},
            fallbacks=["transformers tokenizer fallback when unavailable"],
            receipts=[
                "receipts/headless/CAPABILITY_noetic-sealed-3.14.json",
                "receipts/headless/QWEN_COMPLETION_RECEIPT.json",
            ],
        )

    def with_mode_override(self) -> "HawkingNativeConfig":
        override = (os.environ.get("HCLI_HAWKING_NATIVE_MODE") or "").strip().lower()
        return replace(self, mode=override) if override else self

    def effective_mode(self) -> str:
        mode = self.with_mode_override().mode
        if mode == "auto":
            return "resident" if self.resident_binary and Path(self.resident_binary).is_file() else "one_shot"
        return mode

    def selected_binary(self) -> str:
        if self.effective_mode() == "resident":
            if not self.resident_binary:
                raise HawkingNativeConfigError(
                    "resident mode requires resident_binary"
                )
            return self.resident_binary
        return self.binary

    def validate(self, *, require_paths: bool = True) -> None:
        mismatches = []
        if self.require_fusion_env:
            if self.protocol == QWEN38_PROTOCOL_SCHEMA:
                mismatches = [
                    f"{key}={self.fusion_env.get(key)!r} (required {value!r})"
                    for key, value in REQUIRED_FUSION_ENV.items()
                    if self.fusion_env.get(key) != value
                ]
            elif not self.fusion_env:
                mismatches = ["profile declares fusion_env required but supplies none"]
        if mismatches:
            raise HawkingNativeConfigError(
                "invalid native fusion configuration: " + "; ".join(mismatches)
            )
        if self.effective_mode() == "resident" and not self.resident_binary:
            raise HawkingNativeConfigError("resident mode has no resident_binary")
        if not require_paths:
            return
        paths = {
            "artifact_root": Path(self.artifact_root),
            "tokenizer": Path(self.tokenizer),
            "binary": Path(self.selected_binary()),
        }
        for name, path in paths.items():
            if not path.exists():
                raise HawkingNativeConfigError(f"{name} does not exist: {path}")
        if not paths["artifact_root"].is_dir():
            raise HawkingNativeConfigError(
                f"artifact_root is not a directory: {paths['artifact_root']}"
            )
        if not paths["tokenizer"].is_file():
            raise HawkingNativeConfigError(
                f"tokenizer is not a file: {paths['tokenizer']}"
            )
        if not os.access(paths["binary"], os.X_OK):
            raise HawkingNativeConfigError(
                f"native executable is not executable: {paths['binary']}"
            )

    def identity(self) -> Dict[str, Any]:
        root = Path(self.artifact_root)
        binary = Path(self.selected_binary())
        return {
            "provider": self.provider,
            "resident_identity": self.resident_identity,
            "family": self.family,
            "architecture": self.architecture,
            "param_class": self.param_class,
            "quantisation": self.quantisation,
            "runtime": self.runtime,
            "protocol": self.protocol,
            "require_fusion_env": self.require_fusion_env,
            "mode": self.effective_mode(),
            "artifact_root": self.artifact_root,
            "tokenizer": self.tokenizer,
            "binary": str(binary),
            "binary_sha256_16": _sha16(binary),
            "tokenizer_sha256_16": _sha16(Path(self.tokenizer)),
            "artifact_inventory": _artifact_inventory(root),
            "max_seq_len": self.max_seq_len,
            "generation": dict(self.generation),
            "fusion_env": dict(self.fusion_env),
            "runtime_env": dict(self.runtime_env),
            "executable_profile": self.executable_profile,
            "physical_ebpw": self.physical_ebpw,
            "qualification": self.qualification,
            "current_runtime": dict(self.current_runtime),
            "profile_schema": self.profile_schema,
            "model_id": self.model_id,
            "compiler": dict(self.compiler),
            "representation": dict(self.representation),
            "capabilities": dict(self.capabilities),
            "prompt_contract": dict(self.prompt_contract),
            "limits": dict(self.limits),
            "fallbacks": list(self.fallbacks),
            "hot_bytes": self.hot_bytes,
            "machine_genome": dict(self.machine_genome),
            "receipts": list(self.receipts),
        }


def is_hawking_native_path(path: Optional[str]) -> bool:
    """Identify an explicit native profile or a native artifact directory."""
    if not path:
        return False
    expanded = Path(path).expanduser()
    if expanded.is_dir():
        return (expanded / "MIX_REPORT.json").is_file() and (
            expanded / "catalog.hq38m20"
        ).is_file()
    if expanded.suffix.lower() in {".gravity", ".nx", ".noetic", ".hawking"} or expanded.name.endswith(
        ".hawking.json"
    ):
        return True
    if expanded.suffix.lower() != ".json" or not expanded.is_file():
        return False
    try:
        data = json.loads(expanded.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and (
        ("artifact_root" in data and "binary" in data)
        or data.get("runtime") == "hawking-native"
    )


def _sealed_profile() -> "Optional[HawkingNativeConfig]":
    """The shipped sealed profile that sits beside this module, or None.

    One source of truth for capabilities: the JSON, not a second copy in
    `defaults()` that can drift away from it silently.
    """
    candidate = Path(__file__).resolve().with_name("hawking-native.sealed-3.14.json")
    if not candidate.is_file():
        return None
    try:
        return HawkingNativeConfig.from_file(str(candidate))
    except HawkingNativeConfigError:
        return None


def config_for_model_path(model_path: Optional[str]) -> HawkingNativeConfig:
    if model_path:
        candidate = Path(model_path).expanduser()
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            try:
                return HawkingNativeConfig.from_file(str(candidate)).with_mode_override()
            except HawkingNativeConfigError:
                raise
        if candidate.is_dir() and is_hawking_native_path(str(candidate)):
            # Start from the SEALED PROFILE, not bare defaults. `defaults()`
            # already calls itself resident_identity "sealed-3.14" but carries
            # capabilities={}, so supports("grammar") was False for the very
            # resident whose profile declares grammar "supported". The grammar
            # channel is built, wired and masked in the running binary, and it
            # was unreachable in the only configuration that ships: an artifact
            # DIRECTORY, which is how the resident is actually launched.
            #
            # Measured cost: replies that could not have broken JSON did --
            # "the reply is NOT valid JSON, the outermost object failed to
            # decode" -- and every receipt reported grammar_enforced as None.
            profile = _sealed_profile() or HawkingNativeConfig.defaults()
            return replace(
                profile,
                artifact_root=str(candidate),
                tokenizer=str(candidate / "tokenizer.json"),
            ).with_mode_override()
        if candidate.suffix.lower() == ".nr":
            raise HawkingNativeConfigError(
                f"NR representation {candidate} is transient/non-executable; compile it to an NX"
            )
        if candidate.suffix.lower() in {".gravity", ".nx", ".noetic", ".hawking"}:
            raise HawkingNativeConfigError(
                f"native model path {candidate} is not a readable profile or artifact; "
                "set HCLI_HAWKING_NATIVE_CONFIG to an explicit profile"
            )
        raise HawkingNativeConfigError(
            f"model path {candidate} is not a Hawking-native profile; "
            "use MLX/llama.cpp discovery or set HCLI_HAWKING_NATIVE_CONFIG"
        )
    configured = os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    if configured:
        return HawkingNativeConfig.from_file(configured).with_mode_override()
    return HawkingNativeConfig.defaults().with_mode_override()


@dataclass
class _RenderedPrompt:
    text: str
    prompt_tokens: int
    thinking_requested: bool
    thinking_qualified: bool
    token_count_source: str


class _TokenizerRenderer:
    def __init__(self, config: HawkingNativeConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._load_error: Optional[str] = None

    def _load(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.config.tokenizer).parent),
                local_files_only=True,
            )
            return self._tokenizer
        except Exception as exc:  # noqa: BLE001 - the fallback is classified below
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, dict) and (
                    "image" in item or "image_url" in item or item.get("type") == "image"
                ):
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif isinstance(item, dict) and (
                    "video" in item or item.get("type") == "video"
                ):
                    parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @classmethod
    def _fallback_render(
        cls,
        messages: Iterable[Dict[str, Any]],
        *,
        thinking_requested: bool,
        contract: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render a dependency-light fallback declared by the profile.

        A native provider is allowed to use a tokenizer-owned chat template,
        but machines that do not have ``transformers`` still need a bounded
        prompt path.  The generic default is deliberately plain role-tagged
        text.  The shipped resident opts into its historical closed-thinking
        template through ``prompt_contract.fallback_template``; that special
        case is profile data, not an HCLI-wide model assumption.
        """
        contract = dict(contract or {})
        template = str(contract.get("fallback_template") or "role_tagged").strip().lower()
        if template in {"qwen_closed_think", "qwen-chat-template-or-closed-think-fallback"}:
            return cls._qwen_fallback_render(messages, thinking_requested=thinking_requested)

        role_open = str(contract.get("role_open") or "[role:{role}]\n")
        role_close = str(contract.get("role_close") or "\n[/role]\n")
        assistant_role = str(contract.get("assistant_role") or "assistant")
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = cls._content_text(message.get("content"))
            parts.append(role_open.format(role=role) + content + role_close.format(role=role))
        parts.append(role_open.format(role=assistant_role))
        if thinking_requested:
            thinking = contract.get("thinking")
            if isinstance(thinking, dict):
                parts.append(str(thinking.get("open") or ""))
        return "".join(parts)

    @classmethod
    def _qwen_fallback_render(
        cls,
        messages: Iterable[Dict[str, Any]],
        *,
        thinking_requested: bool,
    ) -> str:
        """The current resident's explicitly declared fallback template."""
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = cls._content_text(message.get("content"))
            if role == "assistant":
                parts.append(
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    f"{content}<|im_end|>\n"
                )
            elif role == "tool":
                parts.append(
                    "<|im_start|>user\n<tool_response>\n"
                    f"{content}\n</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        if thinking_requested:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    @staticmethod
    def _close_declared_thinking_block(text: str, *, contract: Dict[str, Any], thinking_requested: bool) -> str:
        """Normalize a declared closed-thinking contract when a tokenizer omits it.

        Some resident tokenizer revisions accept ``enable_thinking=False`` but
        still leave the generation prompt at ``<think>\n``.  Only the profile's
        explicitly named Qwen closed-thinking contract gets this repair; a
        generic provider/model never receives Qwen-specific prompt syntax.
        """
        if thinking_requested:
            return text
        template = str(contract.get("fallback_template") or "").strip().lower()
        if template not in {"qwen_closed_think", "qwen-chat-template-or-closed-think-fallback"}:
            return text
        if text.endswith("<think>\n"):
            return text + "\n</think>\n\n"
        if text.endswith("<think>"):
            return text + "\n\n</think>\n\n"
        return text

    def render(
        self,
        messages: List[Dict[str, Any]],
        *,
        thinking_requested: bool,
    ) -> _RenderedPrompt:
        contract = dict(self.config.prompt_contract or {})
        tokenizer = self._load()
        qualified = False
        if tokenizer is not None:
            template_kwargs: Dict[str, Any] = {}
            thinking_key = contract.get("thinking_parameter")
            if thinking_key and _coerce_bool(contract.get("supports_thinking"), False):
                template_kwargs[str(thinking_key)] = thinking_requested
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                text = self._close_declared_thinking_block(
                    str(text),
                    contract=contract,
                    thinking_requested=thinking_requested,
                )
                qualified = bool(template_kwargs) or not thinking_requested
            except TypeError:
                # Older tokenizer wrappers may not expose provider-specific
                # keyword arguments. Keep the functional path, but carry the
                # loss of the qualified contract in the runtime receipt.
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = self._close_declared_thinking_block(
                    str(text),
                    contract=contract,
                    thinking_requested=thinking_requested,
                )
            except Exception as exc:  # noqa: BLE001
                raise HawkingNativeError(
                    f"artifact chat-template rendering failed: {exc}"
                ) from exc
        else:
            text = self._fallback_render(
                messages,
                thinking_requested=thinking_requested,
                contract=contract,
            )

        prompt_tokens = 0
        source = "estimated"
        if tokenizer is not None:
            try:
                encoded = tokenizer(text, add_special_tokens=False)
                ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
                if isinstance(ids, list):
                    prompt_tokens = len(ids)
                    source = "tokenizer"
            except Exception:  # noqa: BLE001
                pass
        if prompt_tokens <= 0:
            prompt_tokens = max(1, (len(text) + 2) // 3)
        if self._load_error and tokenizer is None:
            source = "estimated_without_transformers"
        return _RenderedPrompt(
            text=str(text),
            prompt_tokens=prompt_tokens,
            thinking_requested=thinking_requested,
            thinking_qualified=qualified,
            token_count_source=source,
        )


def _messages_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return [{"role": "user", "content": prompt}]
        raise HawkingNativeProtocolError("model payload has no messages or prompt")
    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise HawkingNativeProtocolError("model message must be an object")
        normalized.append(
            {
                "role": str(item.get("role") or "user"),
                "content": item.get("content") if item.get("content") is not None else "",
            }
        )
    return normalized


class ResidentProcess:
    """A correlated JSONL child process with explicit restart ownership."""

    def __init__(self, config: HawkingNativeConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._ready_payload: Optional[Dict[str, Any]] = None
        self._reader: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()
        self._log_path: Optional[str] = None
        self._log_handle: Any = None
        self.restart_count = 0

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def _alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, timeout: float = 300.0) -> None:
        if self._alive() and self._ready_payload is not None:
            return
        self.stop()
        self.config.validate()
        binary = self.config.selected_binary()
        log = tempfile.NamedTemporaryFile(
            prefix="hcli-hawking-resident-", suffix=".log", mode="w", delete=False
        )
        self._log_path = log.name
        self._log_handle = log
        self._lines = queue.Queue()
        self._pending = {}
        self._ready_payload = None
        env = os.environ.copy()
        env.update(self.config.runtime_env)
        env.update(self.config.fusion_env)
        process = subprocess.Popen(
            [
                binary,
                "--artifact-root",
                self.config.artifact_root,
                "--tokenizer",
                self.config.tokenizer,
                "--max-seq-len",
                str(self.config.max_seq_len),
                "--resident-identity",
                self.config.resident_identity,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            env=env,
            # Native binaries may live in a build tree whose parent depth is
            # not stable.  Use the HCLI checkout as the portable working
            # directory; the artifact and tokenizer remain absolute.
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.process = process

        def read_stdout() -> None:
            # Capture this generation's process. Referencing self.process here
            # lets an old reader accidentally attach to a replacement child
            # during crash recovery, which can cross-wire JSONL responses.
            stream = process.stdout
            try:
                if stream is not None:
                    for line in stream:
                        self._lines.put(line.rstrip("\r\n"))
            finally:
                self._lines.put(None)

        self._reader = threading.Thread(
            target=read_stdout,
            name="hcli-hawking-resident-reader",
            daemon=True,
        )
        self._reader.start()
        self._await_ready(timeout)

    def _await_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while True:
            item = self._next_line(max(0.0, deadline - time.monotonic()))
            if item is None:
                raise HawkingNativeProtocolError(self._dead_message("resident exited before ready"))
            body = self._parse_line(item)
            if body.get("status") == "ready":
                self._ready_payload = body
                return
            request_id = body.get("id")
            if isinstance(request_id, str):
                self._pending[request_id] = body
                continue
            raise HawkingNativeProtocolError(
                f"resident emitted non-ready startup record: {body}"
            )

    def ready(self, timeout: float = 0.0) -> bool:
        if self._ready_payload is not None and self._alive():
            return True
        if not self._alive():
            return False
        try:
            self._await_ready(timeout)
        except HawkingNativeError:
            return False
        return self._ready_payload is not None and self._alive()

    def health(self) -> Dict[str, Any]:
        payload = dict(self._ready_payload or {})
        return {
            "ready": bool(self._ready_payload is not None and self._alive()),
            "pid": self.pid,
            "restart_count": self.restart_count,
            "model_open_count": payload.get("model_open_count"),
            "weight_upload_count": payload.get("weight_upload_count"),
            "resident_identity": payload.get("resident_identity", self.config.resident_identity),
            "protocol": payload.get("protocol", PROTOCOL_SCHEMA),
        }

    def request(self, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        with self._io_lock:
            if not self.ready(0.0):
                raise HawkingNativeProtocolError(self._dead_message("resident is not ready"))
            request_id = str(body.get("id") or "")
            if not request_id:
                raise HawkingNativeProtocolError("resident request has no id")
            if request_id in self._pending:
                return self._pending.pop(request_id)
            try:
                if self.process is None or self.process.stdin is None:
                    raise HawkingNativeProtocolError("resident stdin is unavailable")
                self.process.stdin.write(json.dumps(body, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise HawkingNativeProtocolError(
                    self._dead_message(f"resident write failed: {exc}")
                ) from exc

            deadline = time.monotonic() + float(timeout)
            while True:
                pending = self._pending.pop(request_id, None)
                if pending is not None:
                    return pending
                item = self._next_line(max(0.0, deadline - time.monotonic()))
                if item is None:
                    if time.monotonic() >= deadline:
                        raise HawkingNativeTimeout(
                            f"resident request {request_id} exceeded {timeout}s"
                        )
                    raise HawkingNativeProtocolError(
                        self._dead_message("resident closed stdout")
                    )
                response = self._parse_line(item)
                response_id = response.get("id")
                if response_id == request_id:
                    return response
                if isinstance(response_id, str):
                    self._pending[response_id] = response
                    continue
                raise HawkingNativeProtocolError(
                    f"resident response has no correlated id: {response}"
                )

    def _next_line(self, timeout: float) -> Optional[str]:
        try:
            return self._lines.get(timeout=max(0.001, timeout))
        except queue.Empty as exc:
            raise HawkingNativeTimeout(
                f"resident protocol read exceeded {timeout:.3f}s"
            ) from exc

    @staticmethod
    def _parse_line(line: str) -> Dict[str, Any]:
        try:
            body = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HawkingNativeProtocolError(
                f"resident stdout is not JSONL: {line[:400]!r}"
            ) from exc
        if not isinstance(body, dict):
            raise HawkingNativeProtocolError("resident JSONL record is not an object")
        return body

    def _dead_message(self, prefix: str) -> str:
        code = self.process.returncode if self.process is not None else None
        tail = self.log_tail()
        return f"{prefix} (returncode={code})" + (f"\n{tail}" if tail else "")

    def log_tail(self, limit: int = 4000) -> str:
        if not self._log_path:
            return ""
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            return Path(self._log_path).read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    def stop(self) -> Dict[str, Any]:
        process = self.process
        pid = process.pid if process is not None else None
        gone = True
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    gone = False
        if process is not None and process.poll() is None:
            gone = False
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
        self.process = None
        self._ready_payload = None
        self._pending = {}
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        return {"pid": pid, "gone": gone}


class HawkingNativeConnector:
    """Translate HCLI's OpenAI-shaped payload to a profile's native runtime."""

    def __init__(
        self,
        config: HawkingNativeConfig,
        *,
        renderer: Optional[_TokenizerRenderer] = None,
    ) -> None:
        self.config = config.with_mode_override()
        self.config.validate(require_paths=False)
        self.renderer = renderer or _TokenizerRenderer(self.config)
        self.resident: Optional[ResidentProcess] = (
            ResidentProcess(self.config) if self.config.effective_mode() == "resident" else None
        )
        self.restart_count = 0
        self._one_shot_lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self.config.effective_mode()

    @property
    def pid(self) -> Optional[int]:
        return self.resident.pid if self.resident is not None else None

    @property
    def process(self) -> Optional[subprocess.Popen[str]]:
        return self.resident.process if self.resident is not None else None

    def identity(self) -> Dict[str, Any]:
        identity = self.config.identity()
        identity.update(
            {
                "backend": "hawking_native",
                "protocol": self.config.protocol,
                "pid": self.pid,
                "restart_count": self.restart_count,
            }
        )
        if self.resident is not None:
            identity["resident_health"] = self.resident.health()
        return identity

    def start(self, timeout: float = 300.0) -> None:
        self.config.validate()
        if self.resident is not None:
            self.resident.start(timeout=timeout)

    def ready(self, timeout: float = 0.0) -> bool:
        try:
            self.config.validate()
        except HawkingNativeError:
            return False
        if self.resident is None:
            return True
        return self.resident.ready(timeout)

    def stop(self) -> Dict[str, Any]:
        if self.resident is None:
            return {"gone": True, "pid": None}
        return self.resident.stop()

    def log_tail(self, limit: int = 4000) -> str:
        return self.resident.log_tail(limit) if self.resident is not None else ""

    def _restart_resident(self, timeout: float) -> None:
        if self.resident is None:
            return
        self.resident.stop()
        self.restart_count += 1
        self.resident.restart_count = self.restart_count
        self.resident.start(timeout=timeout)

    def _render(self, payload: Dict[str, Any]) -> _RenderedPrompt:
        messages = _messages_from_payload(payload)
        kwargs = payload.get("chat_template_kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        requested = kwargs.get(
            "enable_thinking",
            self.config.generation.get("enable_thinking", False),
        )
        return self.renderer.render(
            messages,
            thinking_requested=_coerce_bool(requested, False),
        )

    def _limits(
        self,
        payload: Dict[str, Any],
        prompt_tokens: int,
    ) -> Tuple[int, int, bool]:
        requested = payload.get("max_tokens")
        explicit = requested is not None
        if requested is None:
            requested = self.config.generation.get("max_new_tokens", 2048)
        actual = _positive_int(requested, "max_tokens", 2048)
        # `generation.max_new_tokens` is a DEFAULT for callers that ask for
        # nothing, not a ceiling over callers that already did the arithmetic.
        # Treating it as a ceiling silently cut an explicit engine request of
        # 6310 down to the unset-env default of 2048; the truncation was then
        # reported as "hit the 6310-token budget", a number the model never
        # reached, so the diagnosis pointed at the wrong knob for weeks. Only
        # the context window may bound an explicit request.
        if explicit:
            configured_cap = actual
        else:
            configured_cap = _positive_int(
                self.config.generation.get("max_new_tokens", actual),
                "generation.max_new_tokens",
                actual,
            )
        clamped = actual > configured_cap
        actual = min(actual, configured_cap)
        requested_seq = payload.get("max_seq_len")
        position_limit = self.config.max_seq_len
        if requested_seq is not None:
            position_limit = min(
                position_limit,
                _positive_int(requested_seq, "max_seq_len", position_limit),
            )
        available = position_limit - prompt_tokens - 8
        if available < 1:
            raise HawkingNativeError(
                f"prompt is {prompt_tokens} tokens and native max_seq_len is "
                f"{position_limit}; no generation token fits"
            )
        clamped = clamped or actual > available
        return min(actual, available), position_limit, clamped

    def _run_one_shot(
        self,
        prompt: _RenderedPrompt,
        *,
        max_new_tokens: int,
        max_seq_len: int,
        timeout: float,
    ) -> Tuple[Dict[str, Any], float]:
        self.config.validate()
        output_fd, output_name = tempfile.mkstemp(prefix="hcli-hawking-", suffix=".json")
        os.close(output_fd)
        output_path = Path(output_name).resolve()
        env = os.environ.copy()
        env.update(self.config.runtime_env)
        env.update(self.config.fusion_env)
        command = [
            self.config.binary,
            "--artifact-root",
            self.config.artifact_root,
            "--tokenizer",
            self.config.tokenizer,
            "--prompt",
            prompt.text,
            "--raw-prompt",
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-seq-len",
            str(max_seq_len),
            "--out",
            str(output_path),
        ]
        started = time.perf_counter()
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    env=env,
                    cwd=str(Path(__file__).resolve().parents[1]),
                )
            except subprocess.TimeoutExpired as exc:
                raise HawkingNativeTimeout(
                    f"one-shot native request exceeded {timeout}s"
                ) from exc
            if completed.returncode != 0:
                diagnostics = (completed.stderr or completed.stdout or "").strip()
                raise HawkingNativeError(
                    f"native executable exited {completed.returncode}: {diagnostics[-2000:]}"
                )
            try:
                body = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                diagnostics = (completed.stderr or "").strip()
                raise HawkingNativeProtocolError(
                    "native --out artifact was missing or invalid"
                    + (f": {diagnostics[-1000:]}" if diagnostics else "")
                ) from exc
            if not isinstance(body, dict):
                raise HawkingNativeProtocolError("native --out artifact is not an object")
            return body, time.perf_counter() - started
        finally:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    @staticmethod
    def _native_body_to_openai(
        body: Dict[str, Any],
        *,
        config: HawkingNativeConfig,
        prompt: _RenderedPrompt,
        max_new_tokens: int,
        wall_s: float,
        mode: str,
        clamped: bool,
        retry_count: int = 0,
        resident_health: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        generated = body.get("new_token_ids")
        if not isinstance(generated, list):
            generated = []
        text = body.get("generated_text")
        if not isinstance(text, str):
            text = body.get("text")
        if not isinstance(text, str):
            raise HawkingNativeProtocolError(
                "native receipt has no generated_text/text string"
            )
        generated_count = len(generated)
        fallbacks = body.get("fallbacks")
        if fallbacks is not None:
            try:
                fallbacks = int(fallbacks)
            except (TypeError, ValueError) as exc:
                raise HawkingNativeProtocolError("native fallbacks is not numeric") from exc
        prompt_tokens = body.get("prompt_len", body.get("prompt_tokens", prompt.prompt_tokens))
        try:
            prompt_tokens = int(prompt_tokens)
        except (TypeError, ValueError):
            prompt_tokens = prompt.prompt_tokens
        native_wall_ns = body.get("wall_ns")
        try:
            native_wall_s = float(native_wall_ns) / 1_000_000_000.0 if native_wall_ns is not None else None
        except (TypeError, ValueError):
            native_wall_s = None
        decode_wall_ns = body.get("decode_wall_ns")
        try:
            decode_wall_s = float(decode_wall_ns) / 1_000_000_000.0 if decode_wall_ns else None
        except (TypeError, ValueError):
            decode_wall_s = None
        decode_steps = body.get("decode_steps")
        try:
            decode_steps = int(decode_steps) if decode_steps is not None else 0
        except (TypeError, ValueError):
            decode_steps = 0
        decode_tps = None
        if decode_steps > 0 and decode_wall_s and decode_wall_s > 0:
            decode_tps = decode_steps / decode_wall_s
        complete_tps = generated_count / wall_s if wall_s > 0 else None
        runtime_identity = config.identity()
        declared_metrics = body.get("metrics")
        if isinstance(declared_metrics, dict):
            native_metrics = _safe(declared_metrics)
        else:
            # Keep the common connector model-neutral.  A native provider may
            # expose timing/dispatch data either under ``metrics`` or as these
            # stable top-level fields; no Qwen-specific interpretation belongs
            # in the transport layer.
            metric_keys = (
                "complete_wall_ns",
                "complete_wall_ns_per_generated_token",
                "gpu_ns",
                "gpu_ns_per_generated_token",
                "wall_minus_gpu_ns",
                "dispatches",
                "dispatches_per_generated_token",
                "active_bytes_per_token",
                "active_weight_bytes_per_generated_token",
                "active_bytes_scope",
                "resident_weight_bytes",
                "workspace_resident_bytes",
                "actual_read_bytes_per_token",
                "transient_bytes_per_token",
                "prefill",
                "decode",
                "step_trace",
                "kernel_genome",
                "capability",
                "prefill_steps",
                "decode_steps",
            )
            native_metrics = {
                key: _safe(body.get(key)) for key in metric_keys if key in body
            }
        hawking = {
            "protocol": config.protocol,
            "mode": mode,
            "resident_identity": config.resident_identity,
            "runtime_identity": runtime_identity,
            "thinking_requested": prompt.thinking_requested,
            "thinking_mode": (
                "qualified"
                if prompt.thinking_qualified
                else "unqualified_template_fallback"
            ),
            "thinking_qualified": prompt.thinking_qualified,
            "prompt_tokens": prompt_tokens,
            "token_count_source": prompt.token_count_source,
            "generated_tokens": generated_count,
            # A native provider may expose exact generated ids. Keeping them in
            # the provider envelope is useful for paired diagnostics, while the
            # core HCLI contract still treats text/usage as the portable fields.
            "new_token_ids": list(generated),
            "complete_tps": complete_tps,
            "decode_tps": decode_tps,
            "wall_ms": wall_s * 1000.0,
            "generation_wall_s": native_wall_s,
            "fallbacks": fallbacks,
            "dense_w_materialized": body.get("dense_w_materialized"),
            "generation_clamped": clamped,
            "retry_count": retry_count,
            "resident_health": resident_health,
            "native_metrics": native_metrics,
            "grammar_enforced": body.get("grammar_enforced") is True,
            # How much prefill the resident actually skipped. Without this the
            # only evidence of KV reuse is a wall clock, and a wall clock cannot
            # distinguish "reuse worked" from "the prompt was shorter".
            "prefix_reused_tokens": body.get("prefix_reused_tokens"),
            "prefill_tokens_stepped": body.get("prefill_tokens_stepped"),
            # cold / session_append / checkpoint_restore. Which path ran is a
            # fact the resident knows and nothing else can recover.
            "prefix_source": body.get("prefix_source"),
            "prefix_checkpoint_taken_at": body.get("prefix_checkpoint_taken_at"),
            # WHY generation ended, and the body's own layer count. Both are
            # facts only the resident holds. A field has to survive THREE hops
            # to reach a receipt -- the resident emits it, this block relays it,
            # the engine allowlists it -- and adding it at the first and last
            # while missing this one is silent: the receipt simply reads None,
            # exactly as grammar_enforced did before the scar above.
            "stop_reason": body.get("stop_reason"),
            "layers": body.get("layers"),
        }
        return {
            "id": f"hawking-chat-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.resident_identity,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if generated_count >= max_new_tokens else "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_count,
                "total_tokens": prompt_tokens + generated_count,
            },
            "hawking": hawking,
        }

    def complete_payload(
        self,
        payload: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HawkingNativeProtocolError("native model payload must be an object")
        prompt = self._render(payload)
        max_new_tokens, max_seq_len, clamped = self._limits(payload, prompt.prompt_tokens)
        limit = float(timeout if timeout is not None else os.environ.get("HCLI_MODEL_TIMEOUT", "1800"))
        mode = self.mode
        retry_count = 0
        if mode == "one_shot":
            with self._one_shot_lock:
                body, wall_s = self._run_one_shot(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    max_seq_len=max_seq_len,
                    timeout=limit,
                )
            return self._native_body_to_openai(
                body,
                config=self.config,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                wall_s=wall_s,
                mode=mode,
                clamped=clamped,
            )

        if self.resident is None:
            raise HawkingNativeError("resident connector was not initialized")
        request_id = f"req-{uuid.uuid4()}"
        request = {
            "id": request_id,
            "prompt": prompt.text,
            "max_new_tokens": max_new_tokens,
            "max_seq_len": max_seq_len,
        }
        # Either trigger: an explicit `grammar` (what StructuredOutputContract
        # sets once the profile declares the resident honours one) or an
        # OpenAI-shaped json_object request. The contract strips
        # response_format on the degraded path, so deriving it from that field
        # alone left the constrained path unreachable.
        # ONLY "json". The resident implements a JSON syntax mask and nothing
        # else, so any other grammar -- a GBNF string, a custom rule set -- is a
        # field it would ignore. Sending it reads as enforcement in a receipt
        # and enforces nothing, which is the exact lie the degraded path exists
        # to avoid.
        grammar = payload.get("grammar")
        response_format = payload.get("response_format")
        if isinstance(grammar, str) and grammar.strip().lower() == "json":
            request["grammar"] = "json"
        elif (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            request["grammar"] = "json"
        started = time.perf_counter()
        try:
            body = self.resident.request(request, limit)
        except (HawkingNativeTimeout, HawkingNativeProtocolError, BrokenPipeError, OSError):
            # Cognition has no external side effect; retrying after a dead
            # resident is safe. Mission/tool state remains outside this child.
            self._restart_resident(timeout=min(limit, 300.0))
            retry_count = 1
            body = self.resident.request(
                {
                    **request,
                    "id": f"req-{uuid.uuid4()}",
                },
                limit,
            )
        wall_s = time.perf_counter() - started
        if body.get("status") == "error":
            raise HawkingNativeError(str(body.get("error") or "resident request failed"))
        return self._native_body_to_openai(
            body,
            config=self.config,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            wall_s=wall_s,
            mode=mode,
            clamped=clamped,
            retry_count=retry_count,
            resident_health=self.resident.health(),
        )


__all__ = [
    "HawkingNativeConfig",
    "HawkingNativeConfigError",
    "HawkingNativeConnector",
    "HawkingNativeError",
    "HawkingNativeProtocolError",
    "HawkingNativeTimeout",
    "PROTOCOL_SCHEMA",
    "REQUIRED_FUSION_ENV",
    "config_for_model_path",
    "is_hawking_native_path",
]
