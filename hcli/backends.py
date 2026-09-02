"""Runtime backends. The pool talks only to this interface.

LlamaServerBackend and MlxServerBackend both implement RuntimeBackend.
mlx_lm.server keeps chat_template_kwargs (the 44x reasoning budget) and
prefix cache (~25 s/call) and has NO response_format / grammar. supports()
is honest so a caller degrades rather than send a field the backend will
silently ignore.

When a backend cannot enforce a schema, make_structured_output_contract()
returns the prompt-side instruction plus a validator. A response that is
not valid against the schema is rejected and retried a bounded number of
times; exhausting those retries is StructuredOutputExhausted — never a
silent pass.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context_budget import resolve as resolve_context_budget
from .resources import pid_is_alive, process_start_token


def allocate_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def llama_server_binary() -> str:
    explicit = os.environ.get("HCLI_LLAMA_SERVER")
    if explicit:
        return explicit
    binary = shutil.which("llama-server")
    if not binary:
        raise RuntimeError("llama-server not found on PATH")
    return binary


_HELP: Dict[str, str] = {}
_VERSION: Dict[str, str] = {}


def _capture(binary: str, args: List[str], timeout: float = 15.0) -> str:
    try:
        proc = subprocess.run(
            [binary, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()


def llama_help_text(binary: Optional[str] = None) -> str:
    path = binary or llama_server_binary()
    cached = _HELP.get(path)
    if cached is not None:
        return cached
    text = _capture(path, ["--help"])
    _HELP[path] = text
    return text


def llama_version_text(binary: Optional[str] = None) -> str:
    path = binary or llama_server_binary()
    cached = _VERSION.get(path)
    if cached is not None:
        return cached
    text = _capture(path, ["--version"])
    _VERSION[path] = text
    return text


def mlx_server_binary() -> str:
    explicit = os.environ.get("HCLI_MLX_SERVER")
    if explicit:
        return explicit
    binary = shutil.which("mlx_lm.server")
    if binary:
        return binary
    fallback = os.path.expanduser("~/.local/bin/mlx_lm.server")
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    raise RuntimeError("mlx_lm.server not found on PATH")


def mlx_help_text(binary: Optional[str] = None) -> str:
    path = binary or mlx_server_binary()
    cached = _HELP.get(path)
    if cached is not None:
        return cached
    text = _capture(path, ["--help"])
    _HELP[path] = text
    return text


def _shebang_interpreter(script: str) -> Optional[str]:
    try:
        with open(script, "r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return None
    if first.startswith("#!"):
        token = first[2:].strip().split()[0] if first[2:].strip() else ""
        return token or None
    return None


def mlx_version_text(binary: Optional[str] = None) -> str:
    path = binary or mlx_server_binary()
    key = f"mlx:{path}"
    cached = _VERSION.get(key)
    if cached is not None:
        return cached
    text = ""
    interp = _shebang_interpreter(path)
    if interp:
        text = _capture(
            interp,
            [
                "-c",
                "import importlib.metadata as m; print(m.version('mlx-lm'))",
            ],
        )
    text = (text or "").strip().splitlines()[0].strip() if text else ""
    _VERSION[key] = text
    return text


def quantisation_from_path(model_path: str) -> str:
    name = os.path.basename(model_path or "")
    m = re.search(r"((?:IQ|Q)\d+_[A-Za-z0-9_]+)", name, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\b((?:BF16|F16|F32|FP16|FP32))\b", name, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)\s*bit", name, re.I)
    if m:
        return f"{m.group(1)}bit"
    return "unknown"


def is_remote_endpoint(value: Optional[str]) -> bool:
    """Return true for a public OpenAI-compatible endpoint selector.

    A URL is a model provider selection, not a local filesystem path.  Keeping
    this predicate here lets RuntimePool and the model registry agree without
    teaching either one about a particular vendor.
    """
    text = str(value or "").strip().lower()
    return text.startswith(("http://", "https://", "remote://"))


def is_mlx_model_dir(path: str) -> bool:
    """True if path is an MLX weight directory (config.json + safetensors)."""
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    for name in names:
        lower = name.lower()
        if lower.endswith(".safetensors") or lower == "model.safetensors.index.json":
            return True
    return False


def mlx_quantisation_label(model_path: str) -> str:
    """Name the MLX artifact the way a PerformanceLedger row should.

    Prefers config.json quantization {bits, mode, group_size} so a 4-bit
    affine group_size-64 directory becomes '4bit-affine-g64', not 'unknown'.
    """
    cfg_path = None
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "config.json")
        if os.path.isfile(candidate):
            cfg_path = candidate
    elif os.path.isfile(model_path) and model_path.endswith(".json"):
        cfg_path = model_path
    if cfg_path:
        try:
            with open(cfg_path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            cfg = None
        if isinstance(cfg, dict):
            quant = cfg.get("quantization")
            if not isinstance(quant, dict):
                quant = cfg.get("quantization_config")
            if isinstance(quant, dict):
                bits = quant.get("bits")
                mode = quant.get("mode")
                group = quant.get("group_size")
                if bits is not None:
                    parts = [f"{int(bits)}bit"]
                    if mode:
                        parts.append(str(mode))
                    if group is not None:
                        parts.append(f"g{int(group)}")
                    return "-".join(parts)
    labeled = quantisation_from_path(model_path)
    return labeled


def model_bytes_at(path: str) -> Optional[int]:
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        continue
            return total
    except OSError:
        return None
    return None


def mlx_context_length(model_path: str) -> Optional[int]:
    cfg_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else ""
    if not cfg_path or not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    text_cfg = cfg.get("text_config")
    for blob in (text_cfg if isinstance(text_cfg, dict) else None, cfg):
        if not isinstance(blob, dict):
            continue
        for key in ("max_position_embeddings", "max_sequence_length", "context_length"):
            raw = blob.get(key)
            if raw is None:
                continue
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return None


def _reap_zombie(pid: int) -> None:
    """Reap a direct child so a zombie is not mistaken for a live process."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def _wait_pid_dead(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _reap_zombie(pid)
        if not pid_is_alive(pid):
            return True
        time.sleep(0.05)
    _reap_zombie(pid)
    return not pid_is_alive(pid)


def terminate_pid(pid: int, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> Dict[str, Any]:
    """TERM then KILL a pid this pool owns. Never used on a foreign process."""
    report: Dict[str, Any] = {
        "pid": pid,
        "term": False,
        "kill": False,
        "gone": False,
    }
    if not isinstance(pid, int) or pid <= 0:
        report["gone"] = True
        return report
    if not pid_is_alive(pid):
        report["gone"] = True
        return report
    for sig, key in ((signal.SIGTERM, "term"), (signal.SIGKILL, "kill")):
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        report[key] = True
        timeout = term_timeout if sig == signal.SIGTERM else kill_timeout
        if _wait_pid_dead(pid, timeout):
            report["gone"] = True
            return report
    report["gone"] = not pid_is_alive(pid)
    return report


@dataclass
class CompletionResult:
    raw: Any
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    text: Optional[str] = None
    degraded: List[str] = field(default_factory=list)
    runtime_index: Optional[int] = None
    schema_attempts: Optional[int] = None


class RuntimeBackend(ABC):
    """One concrete inference server (process or equivalent).

    MlxServerBackend: spawn mlx_lm.server, the same identity() fields so a
    PerformanceLedger row can name the artifact, complete() that never
    sends response_format/grammar, and supports() that reports those as
    False so the engine degrades to prompt-only structured output with a
    bounded retry. Prefix cache and chat_template_kwargs stay True.
    """

    @abstractmethod
    def identity(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def spawn(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def ready(self, timeout: float) -> bool:
        ...

    @abstractmethod
    def endpoint(self) -> str:
        ...

    @abstractmethod
    def stop(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def complete(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> CompletionResult:
        ...

    @abstractmethod
    def supports(self, feature: str) -> bool:
        ...

    def _prepare_payload(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Strip fields this backend will ignore; never send them silently."""
        prepared = deepcopy(payload)
        degraded: List[str] = []
        if "response_format" in prepared and not self.supports("response_format"):
            schema = schema_from_response_format(prepared.get("response_format"))
            prepared.pop("response_format", None)
            degraded.append("response_format")
            if schema is not None:
                inject_schema_instruction(prepared, schema_instruction(schema))
            else:
                self._nudge_json_instruction(prepared)
        if "grammar" in prepared and not self.supports("grammar"):
            prepared.pop("grammar", None)
            degraded.append("grammar")
        if (
            "chat_template_kwargs" in prepared
            and not self.supports("chat_template_kwargs")
        ):
            prepared.pop("chat_template_kwargs", None)
            degraded.append("chat_template_kwargs")
        return prepared, degraded

    @staticmethod
    def _nudge_json_instruction(payload: Dict[str, Any]) -> None:
        note = (
            "\nReturn exactly one JSON object and nothing else. "
            "Do not wrap it in markdown."
        )
        append_user_text(payload, note, skip_if="JSON object")

    @staticmethod
    def _parse_response(data: Any, degraded: List[str]) -> CompletionResult:
        return completion_from_openai(data, degraded)

    # Provider-neutral adapters.  The native/MLX/llama/remote transports keep
    # their existing CompletionResult API for compatibility, while AgentOS can
    # speak the same contract to every backend.
    def generate(self, request: Any, *, timeout: Optional[float] = None) -> Any:
        from .providers import GenerationRequest, GenerationResponse

        normalized = request if isinstance(request, GenerationRequest) else GenerationRequest.from_mapping(request)
        result = self.complete(normalized.to_payload(), timeout=timeout)
        identity = self.identity()
        response = GenerationResponse.from_completion(
            result,
            provider=str(identity.get("provider") or identity.get("runtime") or type(self).__name__),
        )
        response.request_id = normalized.request_id
        return response

    def capabilities(self) -> Any:
        from .providers import CapabilityContract

        return CapabilityContract.from_backend(self)

    def health(self) -> Any:
        from .providers import ProviderHealth

        identity = self.identity()
        provider = str(identity.get("provider") or identity.get("runtime") or type(self).__name__)
        try:
            ready = bool(self.ready(0.0))
        except Exception as exc:  # noqa: BLE001 - health is an observation
            return ProviderHealth(
                state="unavailable",
                ready=False,
                provider=provider,
                detail=f"{type(exc).__name__}: {exc}",
                recoverable=True,
            )
        return ProviderHealth(
            state="healthy" if ready else "unavailable",
            ready=ready,
            provider=provider,
            detail=None if ready else "backend is not ready",
            recoverable=True,
        )

    def profile(self) -> Any:
        from .providers import ResidentProfile

        return ResidentProfile.from_backend(self)


# Honest capability table. MLX numbers from receipts/headless/BACKEND_CAPABILITY.json.
# llama.cpp: response_format yes, chat_template_kwargs yes (needs --jinja),
#            prefix cache yes, slots yes, grammar yes.
# mlx_lm.server: response_format NO, grammar NO, chat_template_kwargs yes,
#                prefix cache yes. --decode-concurrency is the slot analogue
#                (batch size for simultaneous decode of batchable requests);
#                it does NOT reserve llama.cpp-style slots.
# Canonical names are the values of FEATURE_ALIASES; there is no unused twin tuple.
FEATURE_ALIASES = {
    "response_format": "response_format",
    "json_schema": "response_format",
    "response_format_json_schema": "response_format",
    "chat_template_kwargs": "chat_template_kwargs",
    "chat_template_kwargs_enable_thinking": "chat_template_kwargs",
    "slots": "slots",
    "continuous_batching_slots": "slots",
    "prefix_cache": "prefix_cache",
    "prompt_prefix_cache": "prefix_cache",
    "grammar": "grammar",
    "grammar_gbnf": "grammar",
}

# --decode-concurrency is the MLX analogue of llama.cpp --parallel slots.
# It is the maximum number of *batchable* requests decoded in parallel.
# It does NOT reserve N independent KV sequences, does NOT create
# llama.cpp-style slots, and does NOT imply --cont-batching. Concurrent
# prompt prefills are gated separately by --prompt-concurrency.
MLX_SLOTS_NOTE = (
    "--decode-concurrency is a batch size for simultaneous decode of "
    "batchable requests; it does not reserve llama.cpp-style slots or "
    "guarantee N independent KV sequences"
)

DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS = 3

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)


def canonical_feature(feature: str) -> str:
    key = str(feature or "").strip()
    return FEATURE_ALIASES.get(key, key)


def structured_output_attempts() -> int:
    raw = os.environ.get("HCLI_STRUCTURED_OUTPUT_ATTEMPTS", "")
    if str(raw).strip() == "":
        return DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS


def schema_from_response_format(response_format: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict):
        inner = json_schema.get("schema")
        if isinstance(inner, dict):
            return inner
        if "type" in json_schema or "properties" in json_schema:
            return json_schema
    schema = response_format.get("schema")
    if isinstance(schema, dict):
        return schema
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    if response_format.get("type") == "json_schema" and isinstance(
        response_format.get("json_schema"), dict
    ):
        return None
    return None


def schema_instruction(schema: Dict[str, Any]) -> str:
    rendered = json.dumps(schema, indent=2, sort_keys=True)
    return (
        "\nReturn exactly one JSON object and nothing else. "
        "Do not wrap it in markdown. "
        "The object MUST satisfy this JSON Schema; a response that does "
        "not validate will be rejected:\n"
        f"{rendered}"
    )


def append_user_text(
    payload: Dict[str, Any], note: str, skip_if: Optional[str] = None
) -> None:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            if skip_if and skip_if in last["content"]:
                return
            last["content"] = last["content"] + note
        return
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        if skip_if and skip_if in prompt:
            return
        payload["prompt"] = prompt + note


def inject_schema_instruction(payload: Dict[str, Any], instruction: str) -> None:
    append_user_text(payload, instruction, skip_if="MUST satisfy this JSON Schema")


def _bracket_closing_suffix(prefix: str) -> str:
    """Closing brackets/braces needed to balance an open JSON prefix.

    Scans outside of string literals only; ``prefix`` is assumed to already be
    structurally valid up to this point (the decoder read this far without
    erroring), so a naive escape-aware scan is enough.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in prefix:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    return "".join("}" if ch == "{" else "]" for ch in reversed(stack))


def _close_unterminated_string(text: str, error: "json.JSONDecodeError") -> Optional[str]:
    """Deterministically close a reply whose only defect is an unclosed string.

    json's own scanner reports the exact offset of the opening quote for an
    "Unterminated string starting at" error (``error.pos``): everything from
    there to end-of-text was read as string body without ever finding an
    unescaped closing quote, which is only possible if that body contains no
    unescaped quote or raw control character the scanner would have stopped
    on first. Appending one closing quote plus whatever braces/brackets the
    prefix still has open is therefore a syntax-only fix -- it invents no
    field and no content, only the punctuation the model never emitted
    before it stopped. If the repaired text still does not parse, or the
    parsed object still fails the schema, this bought nothing and the normal
    violation/retry path runs exactly as if it had never been tried.
    """
    if "Unterminated string starting at" not in error.msg:
        return None
    start = error.pos
    if start < 0 or start >= len(text) or text[start] != '"':
        return None
    body = text[start + 1 :]
    # A closing quote right after an ODD run of trailing backslashes would
    # itself read as an escaped quote, not a terminator -- drop the dangling
    # backslash rather than guess what it was escaping.
    trailing_backslashes = len(body) - len(body.rstrip("\\"))
    if trailing_backslashes % 2:
        body = body[:-1]
    closing = _bracket_closing_suffix(text[:start])
    return text[: start + 1] + body + '"' + closing


_DECODE_VIOLATION_MARKERS = (
    "is not a JSON object",
    "not an object",
    "empty response",
    "Unterminated",
    "TRUNCATION_REPAIR",
)


def _is_decode_violation(reason: str) -> bool:
    """True when a violation is broken JSON syntax, not a schema/shape miss.

    Used to pick the retry instruction: a schema miss needs "add this field",
    broken syntax needs "escape your quotes and keep it short" instead --
    repeating the schema back at a model that already produced the right
    shape and just failed to close a string does not fix a syntax error.
    """
    text = str(reason or "")
    return any(marker in text for marker in _DECODE_VIOLATION_MARKERS)


def extract_json_object(content: Any, diag: Optional[List[str]] = None) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply. Raises SchemaViolation.

    ``diag`` collects notes about HOW the object was obtained. It matters when the
    whole reply does not parse and an inner object is salvaged instead: that object
    is a FRAGMENT, and validating it produces a schema error that blames the shape
    when the real fault is a syntax error further up. A "TRUNCATION_REPAIR: " note
    means the object was recovered by closing an unterminated trailing string
    deterministically (see ``_close_unterminated_string``) -- no retry spent.

    Measured on the sealed 27B resident: it emitted an unescaped quote inside a
    string (``"find \"$ROOT/receipts/head" -type f"``), the outer object failed to
    decode, the scan returned the first OBLIGATION object, and the rejection read
    "missing required property 'obligations'". Six retries chased a phantom.
    """
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if not text:
        raise SchemaViolation("empty response", text=text)
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise SchemaViolation(
            f"response JSON is {type(parsed).__name__}, not an object",
            text=text,
        )
    except SchemaViolation:
        raise
    except json.JSONDecodeError as exc:
        repaired_text = _close_unterminated_string(text, exc)
        if repaired_text is not None:
            try:
                repaired = json.loads(repaired_text)
            except Exception:
                repaired = None
            if isinstance(repaired, dict):
                if diag is not None:
                    diag.append(
                        "TRUNCATION_REPAIR: the reply's last string was "
                        f"never closed ({exc}); it "
                        "was closed deterministically (appended a closing "
                        f"quote and {len(repaired_text) - len(text)} "
                        "bracket character(s)), no field content was "
                        "invented, and the model was not asked again for "
                        "this attempt"
                    )
                return repaired
    except Exception:
        pass
    decoder = json.JSONDecoder()
    first_brace = text.find("{")
    outer_error: Optional[str] = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except Exception as exc:
            if index == first_brace:
                outer_error = str(exc)
            continue
        if isinstance(parsed, dict):
            if diag is not None and index != first_brace:
                diag.append(
                    "the reply is NOT valid JSON -- the outermost object failed to "
                    f"decode ({outer_error or 'unknown error'}) and an INNER object "
                    f"at offset {index} was validated instead. Fix the syntax; the "
                    f"schema complaint below is about a fragment.")
            return parsed
    raise SchemaViolation(
        "response is not a JSON object"
        + (f" (outermost object failed to decode: {outer_error})" if outer_error else ""),
        text=text)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, item) for item in expected)
    name = str(expected or "")
    actual = _json_type_name(value)
    if name == "number":
        return actual in {"number", "integer"}
    if name == "integer":
        return actual == "integer"
    return actual == name


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein, small strings only -- these are JSON key names."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _near_miss_key(required: str, instance: Dict[str, Any],
                   schema: Dict[str, Any]) -> Optional[str]:
    """A key the instance HAS that looks like the required one it is missing.

    Only keys the schema does not otherwise know about qualify: a legitimate
    sibling property is not a misspelling of anything. Threshold scales with the
    name's length so short keys cannot collide with each other.
    """
    known = set(schema.get("properties") or {}) | set(schema.get("required") or [])
    # At most a third of the name may differ, capped at 3 edits. A flat
    # distance-1 threshold makes `ix` a misspelling of `id`, which is half the
    # string -- a guess that would be wrong more often than useful on short keys.
    limit = min(3, len(required) // 3)
    if limit < 1:
        return None
    best, best_d = None, limit + 1
    for key in instance:
        if key in known or not isinstance(key, str):
            continue
        d = _edit_distance(required.lower(), key.lower())
        if d < best_d:
            best, best_d = key, d
    return best if best_d <= limit else None


def repair_near_miss_keys(instance: Any, schema: Dict[str, Any],
                         log: List[str], path: str = "$") -> Any:
    """Rename an UNAMBIGUOUS near-miss key onto the required name it misses.

    A grammar-less backend usually produces the right SHAPE and drifts one key.
    Measured on the sealed 27B resident: a correct four-obligation plan was thrown
    away six times over `consequent` for `consequential`, and the retry that named
    the drift produced a different one (` angles`, with a leading space).

    Refusing a recoverable plan over one token is not rigour, but repairing one
    silently would be worse than the drift. Every rename is APPENDED TO log and
    travels into the receipt, and a rename only happens when:

      * the required key is genuinely absent, and
      * exactly one unknown key is within the near-miss threshold.

    A rename whose value does not fit the property is caught by the caller's
    re-validation, which discards the whole repair. An earlier version also
    checked the value type HERE; a mutation deleting that check passed every test,
    because the outer re-validation already refuses the same objects. It was
    redundant, so it is gone -- and the re-validation now has its own test.
    """
    if isinstance(instance, dict) and isinstance(schema, dict):
        # Surrounding whitespace in a key is never intentional and is the same
        # class of drift; strip it first, and only when nothing collides.
        for key in [k for k in instance if isinstance(k, str) and k != k.strip()]:
            if key.strip() and key.strip() not in instance:
                instance[key.strip()] = instance.pop(key)
                log.append(f"{path}: key {key!r} -> {key.strip()!r} (whitespace)")
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key in instance:
                continue
            near = _near_miss_key(key, instance, schema)
            if near is None:
                continue
            instance[key] = instance.pop(near)
            log.append(f"{path}: key {near!r} -> {key!r}")
        for key, value in list(instance.items()):
            if key in props and isinstance(props[key], dict):
                repair_near_miss_keys(value, props[key], log, f"{path}.{key}")
    elif isinstance(instance, list) and isinstance(schema, dict):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, value in enumerate(instance):
                repair_near_miss_keys(value, items, log, f"{path}[{i}]")
    return instance


def validate_against_schema(
    instance: Any, schema: Dict[str, Any], path: str = "$"
) -> Optional[str]:
    """Return a reason string if instance fails schema, else None.

    Covers the JSON Schema subset HCLI actually ships: type, properties,
    required, additionalProperties, enum, items, maxItems, minItems.
    """
    if not isinstance(schema, dict):
        return None
    if "type" in schema and not _type_matches(instance, schema["type"]):
        return (
            f"{path}: expected {schema['type']}, got {_json_type_name(instance)}"
        )
    if "enum" in schema and instance not in schema["enum"]:
        return f"{path}: {instance!r} is not one of {schema['enum']}"
    is_object = schema.get("type") == "object" or "properties" in schema or "required" in schema
    if is_object:
        if not isinstance(instance, dict):
            return f"{path}: expected object, got {_json_type_name(instance)}"
        for key in schema.get("required") or []:
            if key not in instance:
                # NAME THE NEAR MISS. A weak-structured-output backend usually gets
                # the shape right and drifts one key -- measured against the sealed
                # 27B resident, which wrote "consequential" on obligations[0] and
                # "consequent" on [1], [2] and [3]. "missing required property
                # 'consequential'" told it what was absent and nothing about what
                # it had actually written, and six retries never converged.
                near = _near_miss_key(key, instance, schema)
                if near:
                    return (f"{path}: missing required property {key!r} -- you wrote "
                            f"{near!r}, which is not the schema's name for it")
                return f"{path}: missing required property {key!r}"
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props and isinstance(props[key], dict):
                err = validate_against_schema(value, props[key], f"{path}.{key}")
                if err:
                    return err
            elif additional is False:
                return f"{path}: additional property {key!r} is not allowed"
            elif isinstance(additional, dict):
                err = validate_against_schema(value, additional, f"{path}.{key}")
                if err:
                    return err
    is_array = schema.get("type") == "array" or "items" in schema
    if is_array:
        if not isinstance(instance, list):
            return f"{path}: expected array, got {_json_type_name(instance)}"
        if "minItems" in schema:
            try:
                if len(instance) < int(schema["minItems"]):
                    return f"{path}: fewer than {schema['minItems']} items"
            except (TypeError, ValueError):
                pass
        if "maxItems" in schema:
            try:
                if len(instance) > int(schema["maxItems"]):
                    return f"{path}: more than {schema['maxItems']} items"
            except (TypeError, ValueError):
                pass
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(instance):
                err = validate_against_schema(item, items, f"{path}[{index}]")
                if err:
                    return err
    return None


class SchemaViolation(ValueError):
    """A single response failed to parse or failed the JSON Schema."""

    def __init__(self, reason: str, text: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.text = text


class StructuredOutputExhausted(RuntimeError):
    """Bounded structured-output retries ran out. Never a silent pass."""

    def __init__(
        self,
        reason: str,
        *,
        attempts: int,
        last_text: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = int(attempts)
        self.last_text = last_text
        self.errors = list(errors or [])


def backend_supports_response_format(backend: Any) -> bool:
    """True when the caller may treat response_format as enforcement.

    None / missing supports() is True so the llama.cpp engine path keeps
    sending the field. An explicit False is the MLX degrade: do not send
    the field, use the prompt-side contract, validate, retry bounded.
    """
    if backend is None:
        return True
    supports = getattr(backend, "supports", None)
    if not callable(supports):
        return True
    try:
        return bool(supports("response_format"))
    except Exception:
        return True


def structured_output_record(
    *,
    mode: str,
    attempts: int = 1,
    max_attempts: Optional[int] = None,
    exhausted: bool = False,
    last_violation: Optional[str] = None,
    errors: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Receipt shape that distinguishes enforced schema from degraded.

    Enforced: the backend claimed response_format and the field was sent.
    Degraded: the field was not sent; prompt + validate + N retries ran.
    """
    if mode == "enforced":
        return {
            "mode": "enforced",
            "response_format_sent": True,
        }
    rec: Dict[str, Any] = {
        "mode": "degraded",
        "response_format_sent": False,
        "attempts": int(attempts),
        "retries": max(0, int(attempts) - 1),
        "exhausted": bool(exhausted),
    }
    if max_attempts is not None:
        rec["max_attempts"] = int(max_attempts)
    if last_violation:
        rec["last_violation"] = last_violation
    if errors:
        rec["errors"] = list(errors)
    if features:
        rec["features"] = list(features)
    return rec


@dataclass
class StructuredOutputContract:
    """Prompt-side schema instruction + validator + bounded retry.

    Used when the backend cannot enforce response_format / grammar.
    Engine._call_model applies this before the HTTP call. complete()
    on an unsupported backend still strips the field so a direct caller
    cannot silently send it.
    """

    schema: Dict[str, Any]
    instruction: str
    max_attempts: int = DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS
    degraded_features: List[str] = field(default_factory=lambda: ["response_format"])
    # Every key rename this contract performed. Carried into the receipt so a
    # repair is auditable rather than invisible.
    repairs: List[str] = field(default_factory=list)
    # Every deterministic unterminated-string close this contract performed.
    # Same auditability rule: a repair that fixed the reply without spending
    # a retry still has to be visible in the receipt, not silently absorbed.
    truncation_repairs: List[str] = field(default_factory=list)

    def apply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prepared = deepcopy(payload)
        prepared.pop("response_format", None)
        prepared.pop("grammar", None)
        inject_schema_instruction(prepared, self.instruction)
        return prepared

    def validate(self, text: Any) -> Dict[str, Any]:
        diag: List[str] = []
        parsed = extract_json_object(text, diag)
        err = validate_against_schema(parsed, self.schema)
        if err and diag:
            # The syntax error is the cause; the schema error is its shadow.
            err = diag[0] + " || " + err
        if err:
            log: List[str] = []
            repaired = repair_near_miss_keys(deepcopy(parsed), self.schema, log)
            if log and not validate_against_schema(repaired, self.schema):
                self.repairs.extend(log)
                return repaired
            raise SchemaViolation(err, text=str(text) if text is not None else None)
        for note in diag:
            if note.startswith("TRUNCATION_REPAIR: "):
                self.truncation_repairs.append(note[len("TRUNCATION_REPAIR: ") :])
        return parsed

    def enforce(
        self,
        complete_fn: Callable[..., CompletionResult],
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> CompletionResult:
        """Call complete_fn until a reply validates, or fail explicitly.

        complete_fn(payload, timeout) -> CompletionResult
        """
        working = self.apply(payload)
        errors: List[str] = []
        last_text: Optional[str] = None
        attempts = max(1, int(self.max_attempts))
        for attempt in range(1, attempts + 1):
            to_send = working
            if attempt > 1:
                to_send = deepcopy(working)
                prior = errors[-1] if errors else "invalid structured output"
                if _is_decode_violation(prior):
                    # The reply broke JSON syntax, not the schema shape --
                    # repeating "satisfy the schema" back at a model that
                    # already had the right shape and just failed to close a
                    # string does not fix a syntax error. Name the actual
                    # failure class and ask for the thing that avoids it.
                    required = self.schema.get("required") or []
                    close_note = (
                        f" Write every required field ({', '.join(required)}) "
                        "and stop immediately after the final closing brace."
                        if required
                        else " Close every object and array you open."
                    )
                    note = (
                        f"\nAttempt {attempt - 1} was rejected: {prior}. "
                        "That reply broke JSON syntax, not the schema. "
                        'Escape every quote and newline inside a string '
                        'value (\\" and \\n) and keep string values short '
                        "-- a sentence, not an essay." + close_note
                    )
                else:
                    note = (
                        f"\nAttempt {attempt - 1} was rejected: {prior}. "
                        "Return exactly one JSON object that satisfies the "
                        "schema and nothing else."
                    )
                append_user_text(to_send, note)
            try:
                # complete_fn is inside the try on purpose: a caller that can
                # see the reply is truncated knows it violates the schema
                # before the validator does, and that is still a violation
                # this contract owns. Retrying only what validate() raised
                # let a length-truncation escape enforce() entirely, so no
                # attempt was ever counted against max_attempts.
                result = complete_fn(to_send, timeout)
                if not isinstance(result, CompletionResult):
                    result = CompletionResult(raw=result, text=str(result))
                last_text = result.text
                parsed = self.validate(result.text)
            except SchemaViolation as exc:
                errors.append(exc.reason)
                if exc.text is not None:
                    last_text = exc.text
                continue
            result.degraded = list(result.degraded or [])
            for feature in self.degraded_features:
                if feature not in result.degraded:
                    result.degraded.append(feature)
            if "structured_output_prompt_validation" not in result.degraded:
                result.degraded.append("structured_output_prompt_validation")
            result.raw = result.raw
            if isinstance(result.raw, dict):
                result.raw = dict(result.raw)
                result.raw["_structured"] = parsed
                if self.repairs:
                    result.raw["_structured_repairs"] = list(self.repairs)
                    if "structured_output_key_repair" not in result.degraded:
                        result.degraded.append("structured_output_key_repair")
                if self.truncation_repairs:
                    result.raw["_structured_truncation_repairs"] = list(
                        self.truncation_repairs
                    )
                    if "structured_output_truncation_repair" not in result.degraded:
                        result.degraded.append("structured_output_truncation_repair")
            result.schema_attempts = attempt
            return result
        reason = (
            f"structured output rejected after {attempts} attempts: "
            f"{errors[-1] if errors else 'no valid JSON'}"
        )
        raise StructuredOutputExhausted(
            reason,
            attempts=attempts,
            last_text=last_text,
            errors=errors,
        )


def make_structured_output_contract(
    backend: Any,
    schema: Dict[str, Any],
    *,
    max_attempts: Optional[int] = None,
) -> Optional[StructuredOutputContract]:
    """Return a degradation contract if the backend cannot enforce schema.

    None means the backend claims response_format support and the caller
    may send it. A contract means: strip the field, inject the instruction,
    validate every reply, retry up to max_attempts, then fail explicitly.
    """
    supports = getattr(backend, "supports", None)
    if callable(supports) and supports("response_format"):
        return None
    attempts = (
        int(max_attempts)
        if max_attempts is not None
        else structured_output_attempts()
    )
    degraded = ["response_format"]
    if callable(supports) and not supports("grammar"):
        degraded.append("grammar")
    return StructuredOutputContract(
        schema=schema,
        instruction=schema_instruction(schema),
        max_attempts=max(1, attempts),
        degraded_features=degraded,
    )


def completion_from_openai(data: Any, degraded: List[str]) -> CompletionResult:
    finish = None
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    text = None
    if isinstance(data, dict):
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish = choice.get("finish_reason")
            message = choice.get("message")
            if isinstance(message, dict):
                text = message.get("content")
            elif choice.get("text") is not None:
                text = choice.get("text")
        if text is None:
            text = data.get("content")
        if finish is None and data.get("stop") is True:
            finish = "stop"
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        if prompt_tokens is None:
            prompt_tokens = timings.get("prompt_n") or data.get("tokens_evaluated")
        if completion_tokens is None:
            completion_tokens = timings.get("predicted_n") or data.get(
                "tokens_predicted"
            )
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            try:
                total_tokens = int(prompt_tokens) + int(completion_tokens)
            except (TypeError, ValueError):
                pass
    return CompletionResult(
        raw=data,
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        text=text if isinstance(text, str) else None,
        degraded=list(degraded),
    )


def _post_json(url: str, payload: Dict[str, Any], timeout: float, error_label: str) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_label} HTTP {exc.code}: {detail[:1200]}") from exc
    try:
        return json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"{error_label} returned invalid JSON") from exc


class LlamaServerBackend(RuntimeBackend):
    def __init__(
        self,
        model_path: str,
        port: Optional[int] = None,
        ctx_size: Optional[int] = None,
        n_slots: int = 1,
        n_gpu_layers: int = 999,
        binary: Optional[str] = None,
    ) -> None:
        self.model_path = os.path.realpath(os.path.expanduser(model_path))
        self.port = int(port) if port is not None else None
        self.n_slots = max(1, int(n_slots))
        if ctx_size is not None:
            self.ctx_size = int(ctx_size)
        else:
            budget = resolve_context_budget(
                model_path=self.model_path,
                n_parallel=self.n_slots,
            )
            self.ctx_size = int(budget.total_ctx)
        env_layers = os.environ.get("HCLI_N_GPU_LAYERS")
        if env_layers is not None and str(env_layers).strip() != "":
            try:
                n_gpu_layers = int(env_layers)
            except ValueError:
                pass
        self.n_gpu_layers = int(n_gpu_layers)
        self._binary = binary
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.start_time: Optional[str] = None
        self._stopped = False
        self._log_path: Optional[str] = None
        self._log_handle = None

    def _bin(self) -> str:
        return self._binary or llama_server_binary()

    def identity(self) -> Dict[str, Any]:
        binary = None
        version = ""
        try:
            binary = self._bin()
            version = llama_version_text(binary)
        except RuntimeError:
            binary = self._binary or os.environ.get("HCLI_LLAMA_SERVER") or ""
        try:
            model_bytes = (
                os.path.getsize(self.model_path)
                if os.path.isfile(self.model_path)
                else None
            )
        except OSError:
            model_bytes = None
        quant = quantisation_from_path(self.model_path)
        return {
            "provider": "llamacpp",
            "backend": "llama_server",
            "binary": binary,
            "version": version,
            "runtime_build": f"{binary} {version}".strip(),
            "model_path": self.model_path,
            "model_bytes": model_bytes,
            "model_identity": (
                f"{self.model_path}:{model_bytes}:{quant}"
                if model_bytes is not None
                else self.model_path
            ),
            "context": self.ctx_size,
            "quantisation": quant,
            "n_slots": self.n_slots,
            "n_gpu_layers": self.n_gpu_layers,
            "host": "127.0.0.1",
            "port": self.port,
            "pid": self.pid,
        }

    def supports(self, feature: str) -> bool:
        canon = canonical_feature(feature)
        try:
            help_text = llama_help_text(self._bin()).lower()
        except RuntimeError:
            help_text = ""
        if canon == "response_format":
            return "--json-schema" in help_text or "-j," in help_text
        if canon == "chat_template_kwargs":
            return "--chat-template-kwargs" in help_text and "--jinja" in help_text
        if canon == "slots":
            return "--parallel" in help_text
        if canon == "prefix_cache":
            return (
                "--cache-ram" in help_text
                or "--cache-idle-slots" in help_text
                or "cache-ram" in help_text
            )
        if canon == "grammar":
            return "--grammar" in help_text
        return False

    def command(self, port: Optional[int] = None, n_slots: Optional[int] = None) -> List[str]:
        use_port = int(port if port is not None else (self.port or 0))
        slots = max(1, int(n_slots if n_slots is not None else self.n_slots))
        cmd = [
            self._bin(),
            "--model",
            self.model_path,
            "--port",
            str(use_port),
            "--host",
            "127.0.0.1",
            "--ctx-size",
            str(self.ctx_size),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--parallel",
            str(slots),
            "--jinja",
            "--reasoning",
            "off",
        ]
        if slots > 1:
            cmd.append("--cont-batching")
        if (os.environ.get("HCLI_LLAMA_DEVICE") or "").strip().lower() == "none":
            cmd.append("--no-warmup")
        device = (os.environ.get("HCLI_LLAMA_DEVICE") or "").strip()
        if device:
            cmd.extend(["--device", device])
        fit = (os.environ.get("HCLI_LLAMA_FIT") or "").strip()
        if fit:
            cmd.extend(["--fit", fit])
        elif device.lower() == "none":
            # Fitting probes Metal even at -ngl 0; without a command queue
            # llama.cpp then SIGSEGVs. CPU-only must disable the fitter.
            cmd.extend(["--fit", "off"])
        return cmd

    def spawn(self, **kwargs: Any) -> None:
        if kwargs.get("port") is not None:
            self.port = int(kwargs["port"])
        if kwargs.get("n_slots") is not None:
            self.n_slots = max(1, int(kwargs["n_slots"]))
        if self.port is None:
            self.port = allocate_port()
        if self.process is not None and self.process.poll() is None:
            return
        cmd = self.command(port=self.port, n_slots=self.n_slots)
        handle = tempfile.NamedTemporaryFile(
            prefix="hcli-llama-", suffix=".log", delete=False
        )
        self._log_path = handle.name
        self._log_handle = handle
        self.process = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.start_time = process_start_token(self.pid)
        self._stopped = False

    def endpoint(self) -> str:
        if self.port is None:
            raise RuntimeError("backend has no port")
        return f"http://127.0.0.1:{self.port}"

    def log_tail(self, n: int = 2500) -> str:
        path = self._log_path
        if not path:
            return ""
        try:
            if self._log_handle is not None:
                try:
                    self._log_handle.flush()
                except Exception:
                    pass
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()[-n:]
        except OSError:
            return ""

    def ready(self, timeout: float) -> bool:
        if self.port is None:
            return False
        deadline = time.monotonic() + float(timeout)
        url = f"{self.endpoint()}/health"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code == 503:
                    time.sleep(0.4)
                    continue
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def complete(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> CompletionResult:
        prepared, degraded = self._prepare_payload(payload)
        if "messages" in prepared:
            url = f"{self.endpoint()}/v1/chat/completions"
        else:
            url = f"{self.endpoint()}/completion"
        limit = float(
            timeout
            if timeout is not None
            else os.environ.get("HCLI_MODEL_TIMEOUT", "1800")
        )
        data = _post_json(url, prepared, limit, "llama-server")
        return self._parse_response(data, degraded)

    def stop(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"pid": self.pid, "gone": True, "unreaped": []}
        if self._stopped and (self.process is None or self.process.poll() is not None):
            if self.pid and pid_is_alive(self.pid):
                killed = terminate_pid(self.pid)
                report.update(killed)
                if not killed.get("gone"):
                    report["unreaped"] = [self.pid]
                    report["gone"] = False
                else:
                    self.pid = None
                    self.process = None
            return report
        proc = self.process
        pid = self.pid or (proc.pid if proc is not None else None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        if pid and pid_is_alive(pid):
            killed = terminate_pid(int(pid))
            report.update(killed)
            if not killed.get("gone"):
                report["unreaped"] = [pid]
                report["gone"] = False
        self.process = None
        if report.get("gone"):
            self.pid = None
        self._stopped = True
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        return report


class MlxServerBackend(RuntimeBackend):
    """mlx_lm.server backend.

    --decode-concurrency maps to the slot concept as a batch size for
    simultaneous decode of batchable requests. It does not reserve
    llama.cpp-style slots or guarantee N independent KV sequences.
    complete() never sends response_format or grammar.
    """

    SLOTS_NOTE = MLX_SLOTS_NOTE
    # mlx_lm.server maps only "default_model" to --model. Anything else,
    # including the engine's "local", is treated as a Hugging Face repo id.
    DEFAULT_MODEL_NAME = "default_model"
    DEFAULT_MODEL_ALIASES = frozenset({"", "local", "default", "default_model"})

    def __init__(
        self,
        model_path: str,
        port: Optional[int] = None,
        n_slots: int = 1,
        binary: Optional[str] = None,
        max_tokens: Optional[int] = None,
        prompt_cache_size: Optional[int] = None,
        prompt_cache_bytes: Optional[int] = None,
        prompt_concurrency: Optional[int] = None,
        decode_concurrency: Optional[int] = None,
        chat_template_args: Optional[Dict[str, Any]] = None,
        prefill_step_size: Optional[int] = None,
    ) -> None:
        self.model_path = os.path.realpath(os.path.expanduser(model_path))
        self.port = int(port) if port is not None else None
        self.n_slots = max(1, int(n_slots))
        if decode_concurrency is not None:
            self.decode_concurrency = max(1, int(decode_concurrency))
        else:
            self.decode_concurrency = self.n_slots
        if prompt_concurrency is not None:
            self.prompt_concurrency = max(1, int(prompt_concurrency))
        else:
            self.prompt_concurrency = self.n_slots
        env_tokens = os.environ.get("HCLI_MODEL_TOKENS")
        if max_tokens is not None:
            self.max_tokens = max(1, int(max_tokens))
        elif env_tokens is not None and str(env_tokens).strip() != "":
            try:
                self.max_tokens = max(1, int(env_tokens))
            except ValueError:
                self.max_tokens = 8192
        else:
            self.max_tokens = 8192
        env_cache = os.environ.get("HCLI_MLX_PROMPT_CACHE_SIZE")
        if prompt_cache_size is not None:
            self.prompt_cache_size = max(1, int(prompt_cache_size))
        elif env_cache is not None and str(env_cache).strip() != "":
            try:
                self.prompt_cache_size = max(1, int(env_cache))
            except ValueError:
                self.prompt_cache_size = 10
        else:
            self.prompt_cache_size = 10
        self.prompt_cache_bytes = (
            int(prompt_cache_bytes) if prompt_cache_bytes is not None else None
        )
        self.prefill_step_size = (
            int(prefill_step_size) if prefill_step_size is not None else None
        )
        if chat_template_args is None:
            self.chat_template_args: Dict[str, Any] = {"enable_thinking": False}
        else:
            self.chat_template_args = dict(chat_template_args)
        self._binary = binary
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.start_time: Optional[str] = None
        self._stopped = False
        self._log_path: Optional[str] = None
        self._log_handle = None

    def _bin(self) -> str:
        return self._binary or mlx_server_binary()

    def identity(self) -> Dict[str, Any]:
        binary = None
        version = ""
        try:
            binary = self._bin()
            version = mlx_version_text(binary)
        except RuntimeError:
            binary = self._binary or os.environ.get("HCLI_MLX_SERVER") or ""
        model_bytes = model_bytes_at(self.model_path)
        quant = mlx_quantisation_label(self.model_path)
        context = mlx_context_length(self.model_path)
        return {
            "provider": "mlx",
            "backend": "mlx_lm_server",
            "binary": binary,
            "version": version,
            "runtime_build": f"{binary} {version}".strip(),
            "model_path": self.model_path,
            "model_bytes": model_bytes,
            "model_identity": (
                f"{self.model_path}:{model_bytes}:{quant}"
                if model_bytes is not None
                else self.model_path
            ),
            "context": context,
            "quantisation": quant,
            "n_slots": self.decode_concurrency,
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "prompt_cache_size": self.prompt_cache_size,
            "prompt_cache_bytes": self.prompt_cache_bytes,
            "slots_note": MLX_SLOTS_NOTE,
            "served_model": self.DEFAULT_MODEL_NAME,
            "host": "127.0.0.1",
            "port": self.port,
            "pid": self.pid,
        }

    def help_probe_usable(self) -> bool:
        """Whether --help actually answered.

        Under a sandbox `mlx_lm.server --help` can die in Metal detection and
        return a CRASH STRING rather than a usage block. That is not the same
        as a server without the flag, and conflating them made a test skip with
        the reason "does not advertise chat_template_kwargs" on a machine where
        `--chat-template-args` is plainly in --help. Absence of a probe is not
        evidence of absence of a feature.
        """
        try:
            text = mlx_help_text(self._bin()).lower()
        except RuntimeError:
            return False
        return "--model" in text and "--port" in text

    def supports(self, feature: str) -> bool:
        """Probe mlx_lm.server --help. Do not copy the comment table."""
        canon = canonical_feature(feature)
        try:
            help_text = mlx_help_text(self._bin()).lower()
        except RuntimeError:
            help_text = ""
        if canon == "response_format":
            return "--json-schema" in help_text or "--response-format" in help_text
        if canon == "grammar":
            return "--grammar" in help_text
        if canon == "chat_template_kwargs":
            return (
                "--chat-template-args" in help_text
                or "--chat-template-kwargs" in help_text
            )
        if canon == "prefix_cache":
            return (
                "--prompt-cache-size" in help_text
                or "--prompt-cache-bytes" in help_text
            )
        if canon == "slots":
            # Batch decode concurrency, not reserved llama.cpp slots.
            return "--decode-concurrency" in help_text
        return False

    def command(
        self, port: Optional[int] = None, n_slots: Optional[int] = None
    ) -> List[str]:
        use_port = int(port if port is not None else (self.port or 0))
        decode = max(
            1,
            int(
                n_slots
                if n_slots is not None
                else self.decode_concurrency
            ),
        )
        prompt_conc = max(1, int(self.prompt_concurrency))
        args_json = json.dumps(self.chat_template_args, separators=(",", ":"))
        cmd = [
            self._bin(),
            "--model",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(use_port),
            "--max-tokens",
            str(self.max_tokens),
            "--decode-concurrency",
            str(decode),
            "--prompt-concurrency",
            str(prompt_conc),
            "--prompt-cache-size",
            str(self.prompt_cache_size),
            "--chat-template-args",
            args_json,
        ]
        if self.prompt_cache_bytes is not None:
            cmd.extend(["--prompt-cache-bytes", str(self.prompt_cache_bytes)])
        if self.prefill_step_size is not None:
            cmd.extend(["--prefill-step-size", str(self.prefill_step_size)])
        return cmd

    def spawn(self, **kwargs: Any) -> None:
        if kwargs.get("port") is not None:
            self.port = int(kwargs["port"])
        if kwargs.get("n_slots") is not None:
            self.n_slots = max(1, int(kwargs["n_slots"]))
            self.decode_concurrency = self.n_slots
        if self.port is None:
            self.port = allocate_port()
        if self.process is not None and self.process.poll() is None:
            return
        cmd = self.command(port=self.port, n_slots=self.decode_concurrency)
        handle = tempfile.NamedTemporaryFile(
            prefix="hcli-mlx-", suffix=".log", delete=False
        )
        self._log_path = handle.name
        self._log_handle = handle
        self.process = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.start_time = process_start_token(self.pid)
        self._stopped = False

    def endpoint(self) -> str:
        if self.port is None:
            raise RuntimeError("backend has no port")
        return f"http://127.0.0.1:{self.port}"

    def log_tail(self, n: int = 2500) -> str:
        path = self._log_path
        if not path:
            return ""
        try:
            if self._log_handle is not None:
                try:
                    self._log_handle.flush()
                except Exception:
                    pass
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()[-n:]
        except OSError:
            return ""

    def ready(self, timeout: float) -> bool:
        if self.port is None:
            return False
        deadline = time.monotonic() + float(timeout)
        url = f"{self.endpoint()}/health"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code == 503:
                    time.sleep(0.4)
                    continue
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def _canonicalize_model_name(self, prepared: Dict[str, Any]) -> None:
        requested = prepared.get("model")
        if requested is None:
            prepared["model"] = self.DEFAULT_MODEL_NAME
            return
        name = str(requested).strip()
        aliases = set(self.DEFAULT_MODEL_ALIASES)
        aliases.update(
            {
                self.model_path,
                os.path.basename(self.model_path),
                os.path.realpath(self.model_path),
            }
        )
        if name in aliases:
            prepared["model"] = self.DEFAULT_MODEL_NAME

    def complete(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> CompletionResult:
        prepared, degraded = self._prepare_payload(payload)
        self._canonicalize_model_name(prepared)
        if "messages" in prepared:
            url = f"{self.endpoint()}/v1/chat/completions"
        else:
            url = f"{self.endpoint()}/v1/completions"
        # Never send fields mlx_lm.server will silently ignore.
        prepared.pop("response_format", None)
        prepared.pop("grammar", None)
        if "response_format" in payload and "response_format" not in degraded:
            degraded.append("response_format")
        if "grammar" in payload and "grammar" not in degraded:
            degraded.append("grammar")
        limit = float(
            timeout
            if timeout is not None
            else os.environ.get("HCLI_MODEL_TIMEOUT", "1800")
        )
        data = _post_json(url, prepared, limit, "mlx_lm.server")
        return self._parse_response(data, degraded)

    def stop(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"pid": self.pid, "gone": True, "unreaped": []}
        if self._stopped and (self.process is None or self.process.poll() is not None):
            if self.pid and pid_is_alive(self.pid):
                killed = terminate_pid(self.pid)
                report.update(killed)
                if not killed.get("gone"):
                    report["unreaped"] = [self.pid]
                    report["gone"] = False
                else:
                    self.pid = None
                    self.process = None
            return report
        proc = self.process
        pid = self.pid or (proc.pid if proc is not None else None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        if pid and pid_is_alive(pid):
            killed = terminate_pid(int(pid))
            report.update(killed)
            if not killed.get("gone"):
                report["unreaped"] = [pid]
                report["gone"] = False
        self.process = None
        if report.get("gone"):
            self.pid = None
        self._stopped = True
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        return report


class OpenAICompatibleBackend(RuntimeBackend):
    """Remote OpenAI-compatible provider selected by an endpoint URL.

    The remote provider is deliberately a thin transport adapter.  It does
    not claim that a vendor supports JSON schema, grammar, chat-template
    controls, or streaming unless the caller declares those capabilities with
    ``HCLI_REMOTE_SUPPORTS_*``.  Unknown remote behaviour therefore follows
    the same prompt-and-validate degradation path as MLX.
    """

    def __init__(
        self,
        model_path: str,
        port: Optional[int] = None,
        n_slots: int = 1,
        **_ignored: Any,
    ) -> None:
        del port
        raw = str(model_path or "").strip()
        if raw.lower().startswith("remote://"):
            raw = "https://" + raw[len("remote://"):].lstrip("/")
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote provider requires an http(s) endpoint")
        if parsed.username or parsed.password:
            raise ValueError("remote endpoint URL credentials are not allowed")
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        query_model = query.get("model", [None])[0]
        fragment_model = None
        if parsed.fragment.startswith("model="):
            fragment_model = urllib.parse.unquote(parsed.fragment[len("model="):])
        # Query strings are not part of a stable identity and can contain API
        # keys.  Do not preserve them in the provider receipt.
        self.base_url = urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", "")
        )
        self.model_name = (
            os.environ.get("HCLI_REMOTE_MODEL")
            or query_model
            or fragment_model
            or "remote"
        )
        self.n_slots = max(1, int(n_slots))
        self.port = None
        self.process = None
        self.pid = None
        self.start_time = None
        self._started = False

    def _completion_url(self) -> str:
        if self.base_url.endswith("/v1/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    def selection(self) -> str:
        """Return a credential-free URL that preserves the selected model."""
        if self.model_name == "remote":
            return self.base_url
        return self.base_url + "#model=" + urllib.parse.quote(self.model_name, safe="")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "hcli/0.1"}
        token = (
            os.environ.get("HCLI_REMOTE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def identity(self) -> Dict[str, Any]:
        return {
            "provider": "remote",
            "backend": "openai_compatible",
            "runtime": "remote",
            "model_path": self.base_url,
            "model_id": self.model_name,
            "model_identity": f"remote:{self.base_url}:{self.model_name}",
            "context": None,
            "quantisation": "provider-managed",
            "n_slots": self.n_slots,
            "endpoint": self._completion_url(),
            "pid": None,
            "port": None,
            "credential_source": "environment-only" if (
                os.environ.get("HCLI_REMOTE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            ) else "none",
            "credential_value": "[REDACTED]" if (
                os.environ.get("HCLI_REMOTE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            ) else None,
        }

    def supports(self, feature: str) -> bool:
        canon = canonical_feature(feature)
        names = {
            "response_format": "HCLI_REMOTE_SUPPORTS_RESPONSE_FORMAT",
            "grammar": "HCLI_REMOTE_SUPPORTS_GRAMMAR",
            "chat_template_kwargs": "HCLI_REMOTE_SUPPORTS_CHAT_TEMPLATE_KWARGS",
            "prefix_cache": "HCLI_REMOTE_SUPPORTS_PREFIX_CACHE",
            "slots": "HCLI_REMOTE_SUPPORTS_SLOTS",
            "vision": "HCLI_REMOTE_SUPPORTS_VISION",
            "tool_calling": "HCLI_REMOTE_SUPPORTS_TOOL_CALLING",
            "streaming": "HCLI_REMOTE_SUPPORTS_STREAMING",
        }
        env_name = names.get(canon)
        return self._env_bool(env_name, False) if env_name else False

    def spawn(self, **kwargs: Any) -> None:
        del kwargs
        self._started = True

    def ready(self, timeout: float) -> bool:
        if not self._started:
            return False
        if os.environ.get("HCLI_REMOTE_SKIP_HEALTH", "").strip() == "1":
            return True
        parsed = urllib.parse.urlparse(self.base_url)
        origin = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            api_root = path
            service_root = path[:-3].rstrip("/")
        elif path.endswith("/v1/chat/completions"):
            api_root = path[:-len("/chat/completions")].rstrip("/")
            service_root = api_root[:-3].rstrip("/") if api_root.endswith("/v1") else api_root
        else:
            service_root = path
            api_root = path + "/v1"
        candidates = []
        for suffix in (
            f"{service_root}/health",
            f"{service_root}/healthz",
            f"{api_root}/models",
            "/health",
            "/healthz",
            "/v1/models",
        ):
            url = origin + (suffix or "/")
            if url not in candidates:
                candidates.append(url)
        for url in candidates:
            request = urllib.request.Request(url, headers=self._headers(), method="GET")
            try:
                with urllib.request.urlopen(request, timeout=max(0.1, min(3.0, float(timeout)))) as response:
                    if 200 <= response.status < 500:
                        return True
            except urllib.error.HTTPError as exc:
                # 401/403 proves a reachable provider; the completion call
                # will report the missing/invalid credential explicitly.
                if exc.code in {401, 403, 404, 405}:
                    return True
            except Exception:
                continue
        return False

    def endpoint(self) -> str:
        return self._completion_url()

    def complete(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> CompletionResult:
        prepared, degraded = self._prepare_payload(payload)
        # HCLI's common payload carries the model selector in ``model``. For
        # a remote endpoint that selector is an URL, not the provider's model
        # field; replace it with the normalized model id while preserving an
        # explicit per-call model override.
        if (
            not prepared.get("model")
            or is_remote_endpoint(str(prepared.get("model")))
            or prepared.get("model") in {"local", "remote"}
        ):
            prepared["model"] = self.model_name
        body = json.dumps(prepared).encode("utf-8")
        request = urllib.request.Request(
            self._completion_url(), data=body, headers=self._headers(), method="POST"
        )
        limit = float(timeout if timeout is not None else os.environ.get("HCLI_MODEL_TIMEOUT", "1800"))
        try:
            with urllib.request.urlopen(request, timeout=limit) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote provider HTTP {exc.code}: {detail[:1200]}") from exc
        except Exception as exc:
            raise RuntimeError(f"remote provider request failed: {exc}") from exc
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("remote provider returned invalid JSON") from exc
        return self._parse_response(data, degraded)

    def stop(self) -> Dict[str, Any]:
        was_started = self._started
        self._started = False
        return {"gone": True, "pid": None, "remote": True, "was_started": was_started}


class NoeticNativeBackend(RuntimeBackend):
    """A profile-driven native model runtime on the ONE interface.

    The backend deliberately does not add a scheduler or a second agent
    stack.  In one-shot mode each completion invokes the supplied native
    executable.  In resident mode the same backend owns one JSONL child and
    the pool continues to own admission and request serialization.
    """

    def __init__(
        self,
        model_path: str = "",
        port: Optional[int] = None,
        n_slots: int = 1,
        **_ignored: Any,
    ) -> None:
        from .hawking_native import (
            HawkingNativeConfig,
            HawkingNativeConfigError,
            config_for_model_path,
            is_hawking_native_path,
        )

        self.model_path = os.path.realpath(os.path.expanduser(model_path or ""))
        self.port = int(port) if port is not None else None
        self.n_slots = max(1, int(n_slots))
        candidate_exists = bool(
            model_path
            and Path(os.path.expanduser(model_path)).exists()
            and is_hawking_native_path(model_path)
        )
        self._configured = bool(
            not model_path
            or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
            or candidate_exists
        )
        try:
            self.config = config_for_model_path(model_path or None)
        except HawkingNativeConfigError:
            if self._configured:
                raise
            # A forced/legacy native backend may still be constructed for a
            # capability census with a sentinel path. Keep that construction
            # side-effect free and refuse only when inference is requested.
            self.config = HawkingNativeConfig.defaults()
        from .hawking_native import HawkingNativeConnector

        self.connector = HawkingNativeConnector(self.config)
        self.process = None
        self.pid: Optional[int] = None
        self.start_time: Optional[str] = None
        self._stopped = False
        self._spawned = False

    def identity(self) -> Dict[str, Any]:
        identity = self.connector.identity()
        identity.update(
            {
                "provider": self.config.provider,
                "backend": "noetic_native",
                "profile_path": self.model_path or None,
                "model_path": self.config.artifact_root,
                "model_bytes": identity.get("artifact_inventory", {}).get(
                    "artifact_bytes"
                ),
                "context": self.config.max_seq_len,
                "architecture": self.config.architecture,
                "param_class": self.config.param_class,
                "quantisation": self.config.quantisation,
                "n_slots": self.n_slots,
                "pid": self.connector.pid,
                "port": self.port,
            }
        )
        if not self._configured:
            identity["configuration_required"] = True
            identity["note"] = (
                "native profile is not configured; this backend instance is "
                "a side-effect-free capability placeholder"
            )
        return identity

    def spawn(self, **kwargs: Any) -> None:
        if kwargs.get("n_slots") is not None:
            self.n_slots = max(1, int(kwargs["n_slots"]))
        # A native resident is not an HTTP server. RuntimePool may hand every
        # backend a candidate port for the common factory signature, but
        # retaining it would make status/receipts claim a socket exists.
        self.port = None
        if not self._configured:
            self._spawned = True
            self._stopped = False
            return
        self.connector.start(
            timeout=float(os.environ.get("HCLI_READY_TIMEOUT", "300"))
        )
        self.process = self.connector.process
        self.pid = self.connector.pid
        if self.pid:
            self.start_time = process_start_token(self.pid)
        self._spawned = True
        self._stopped = False

    def ready(self, timeout: float) -> bool:
        if not self._configured:
            del timeout
            return self._spawned and not self._stopped
        if self._stopped or not self._spawned:
            return False
        ready = self.connector.ready(timeout)
        self.process = self.connector.process
        self.pid = self.connector.pid
        return ready

    def endpoint(self) -> str:
        return (
            f"hawking-native://{self.config.resident_identity}/"
            f"{self.connector.mode}"
        )

    def stop(self) -> Dict[str, Any]:
        self._stopped = True
        self._spawned = False
        if not self._configured:
            return {"gone": True, "pid": None, "backend": "noetic_native"}
        report = self.connector.stop()
        self.process = None
        self.pid = None
        self.start_time = None
        report["backend"] = "noetic_native"
        return report

    def complete(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> CompletionResult:
        if not self._configured:
            del payload, timeout
            raise RuntimeError(
                "noetic native interface reservation has no configured model "
                "profile; refusing to invent tokens"
            )
        prepared, degraded = self._prepare_payload(payload)
        raw = self.connector.complete_payload(prepared, timeout=timeout)
        hawking = raw.get("hawking")
        if isinstance(hawking, dict) and degraded:
            hawking["degraded_features"] = list(degraded)
        result = self._parse_response(raw, degraded)
        self.process = self.connector.process
        self.pid = self.connector.pid
        self.start_time = process_start_token(self.pid) if self.pid else None
        return result

    def supports(self, feature: str) -> bool:
        canon = canonical_feature(feature)
        declared = self.config.capabilities
        if isinstance(declared, dict):
            declared_features = declared.get("features", declared)
            if isinstance(declared_features, dict) and canon in declared_features:
                value = declared_features[canon]
                if isinstance(value, dict):
                    state = str(value.get("state") or "unknown").strip().lower()
                    if state in {"supported", "true", "yes", "available"}:
                        return True
                    if state in {"unsupported", "false", "no", "unavailable"}:
                        return False
                elif isinstance(value, bool):
                    return value
                elif isinstance(value, (int, float)):
                    return value != 0
                elif value is not None:
                    return str(value).strip().lower() in {"supported", "true", "yes", "available"}
        # The generic native protocol has no implicit structured-output or
        # grammar support. The shipped Qwen resident's only transport-level
        # guarantee is its chat-template keyword; all other support must be
        # declared by the selected profile.
        return canon == "chat_template_kwargs"

    def log_tail(self, n: int = 2500) -> str:
        return self.connector.log_tail(n)

    def health_snapshot(self) -> Dict[str, Any]:
        if not self._configured:
            return {
                "ready": self._spawned and not self._stopped,
                "mode": "unconfigured",
                "pid": None,
            }
        if self.connector.resident is None:
            return {
                "ready": self.ready(0.0),
                "mode": self.connector.mode,
                "pid": None,
            }
        return self.connector.resident.health()


# ``noetic_native`` is retained as the serialized backend kind because it is
# already present in HCLI receipts and capability snapshots.  These aliases
# expose the actual abstraction to new callers: the provider/profile may be
# Hawking's current resident today, while the HCLI runtime contract remains
# usable by any executable that implements the same native JSON protocol.
NativeRuntimeBackend = NoeticNativeBackend
HawkingNativeBackend = NativeRuntimeBackend
