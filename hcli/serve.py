"""An OpenAI-compatible endpoint over the PERSISTENT Hawking resident.

WHY THIS EXISTS. Open WebUI (already installed on this machine) speaks
`/v1/models` + `/v1/chat/completions` and nothing else. Hawking has two prior
OpenAI surfaces and neither can drive a browser chat:

  * `crates/hawking-serve` wants `model-*.gravity` shards. Every body on disk
    is .hq30uq4 / .f32v2 / .hgrafv01, so it has nothing to serve.
  * `tools/hcli_resident/serve_sealed.py` is correct and seal-verified, and its
    own docstring says the binary RELOADS THE MODEL PER CALL. That is a
    plumbing surface, not a chat one -- a ten-gigabyte load per message.

`make_backend_for_model` already returns a backend whose `complete(payload)`
takes an OpenAI-shaped dict, and for the native profile that backend holds a
HawkingNativeConnector wrapping a PERSISTENT ResidentProcess: one load, then
every turn is a pipe round trip. So this server is thin on purpose -- HTTP in,
`backend.complete` out. It adds no second runtime, no second model resolution
and no second state universe.

STREAMING IS SHAPE, NOT TIMING, AND IS LABELLED AS SUCH. The connector returns
a completed string; there is no incremental hook to subscribe to yet. With
`stream: true` this emits ONE content delta followed by `[DONE]` -- a valid SSE
conversation that Open WebUI renders correctly -- and stamps
`hawking.streaming: "single-chunk"` on the non-stream path plus `/health` so
nobody reads the SSE frames as evidence of token-by-token decode. Pretending
otherwise would be a false affordance, and those are bugs.

GREEDY PROFILES REFUSE SAMPLERS RATHER THAN IGNORING THEM. sealed-3.14 decodes
argmax. A request asking for temperature 0.8 is answered with a 400 that names
the accepted values, because silently serving a different sampler than the
caller asked for is how a benchmark ends up measuring something else. That rule
is `serve_sealed.py`'s and it is kept here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .context_budget import resolve
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # the catalog import is deferred so a menu is never
    from .catalog import Body as CatalogBody  # built just to import this module

DEFAULT_PORT = 8011
DEFAULT_HOST = "127.0.0.1"
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
#: Small on purpose. If window resolution ever fails again, the effect
#: is eager compaction that someone notices -- not a plausible constant
#: that silently replaces the measurement.
_WINDOW_UNKNOWN = 2048

#: Values that mean "do not sample", per parameter. A greedy profile accepts
#: these and refuses everything else. `True == 1` in Python, so membership is
#: checked with an explicit type guard rather than a bare `in`.
GREEDY_OK: Dict[str, Tuple[Any, ...]] = {
    "temperature": (0, 0.0),
    "top_p": (1, 1.0),
    "top_k": (0, 1),
    "n": (1,),
    "presence_penalty": (0, 0.0),
    "frequency_penalty": (0, 0.0),
}


def _acceptable(key: str, value: Any) -> bool:
    if value is None:
        return True
    for ok in GREEDY_OK[key]:
        if type(value) is bool or type(ok) is bool:
            if type(value) is type(ok) and value == ok:
                return True
        elif value == ok:
            return True
    return False


def profile_is_greedy(model: str) -> bool:
    """True when the artifact's own profile says it does not sample.

    Read from the profile rather than assumed, so an MLX specimen served
    through the same command is not told it must be greedy.
    """
    try:
        path = Path(os.path.expanduser(str(model)))
        if path.suffix != ".json" or not path.is_file():
            return False
        gen = json.loads(path.read_text(encoding="utf-8")).get("generation") or {}
    except Exception:
        return False
    if gen.get("do_sample") is False:
        return True
    return gen.get("temperature") in (0, 0.0) and gen.get("top_k") in (1,)


def sampler_refusal(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bad = sorted(k for k in GREEDY_OK if k in body and not _acceptable(k, body[k]))
    if not bad:
        return None
    accepted = ", ".join(f"{k}={GREEDY_OK[k][-1]!r}" for k in bad)
    return {"error": {
        "message": (
            f"this resident decodes GREEDY ARGMAX, so {bad} would ask for a sampler "
            f"it does not run. Refused rather than ignored, because silently serving "
            f"a different sampler than you asked for is how a measurement ends up "
            f"describing something else. Accepted here: {accepted} (or leave them "
            f"unset). In Open WebUI these live under Chat Controls > Advanced Params."
        ),
        "type": "unsupported_parameter",
        "param": bad[0],
    }}


def models_payload(identity: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": identity,
        "object": "model",
        # Open WebUI sorts and labels on these two; a missing `created` renders
        # the model as "Invalid Date" in the picker.
        "created": int(time.time()),
        "owned_by": "hawking",
    }
    if extra:
        row.update(extra)
    return {"object": "list", "data": [row]}


def _text_of(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text:
        return text
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def chat_payload(result: Any, identity: str, *, request_id: str) -> Dict[str, Any]:
    text = _text_of(result)
    usage = {
        "prompt_tokens": getattr(result, "prompt_tokens", None) or 0,
        "completion_tokens": getattr(result, "completion_tokens", None) or 0,
        "total_tokens": getattr(result, "total_tokens", None) or 0,
    }
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": identity,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": getattr(result, "finish_reason", None) or "stop",
        }],
        "usage": usage,
        # Named so a reader of this response cannot mistake the SSE shape on the
        # other path for evidence of incremental decode.
        "hawking": {
            "streaming": "single-chunk",
            "degraded": list(getattr(result, "degraded", None) or []),
        },
    }


def stream_frames(result: Any, identity: str, *, request_id: str):
    """SSE frames for one completed answer: role, one content delta, stop, DONE."""
    created = int(time.time())

    def frame(delta: Dict[str, Any], finish: Optional[str]) -> bytes:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": identity,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\n"

    yield frame({"role": "assistant"}, None)
    text = _text_of(result)
    if text:
        yield frame({"content": text}, None)
    yield frame({}, getattr(result, "finish_reason", None) or "stop")
    yield b"data: [DONE]\n\n"


class Resident:
    """The one loaded body, and the ability to become a different one.

    Open WebUI sends `model` on every request and reads `/v1/models` for its
    dropdown, so honouring both is all switching needs -- no reconnection, no
    second endpoint, no restart. That is why this lives behind the OpenAI shape
    rather than beside it.

    ONE BODY AT A TIME, AND THE OLD ONE STOPS FIRST. These are 8-150 GB
    artifacts on a 103 GB machine; spawning the new resident before stopping the
    old one would page the box into the ground and violate the campaign's swap
    ceiling. So a switch is stop-then-start, serialised under a lock, and a
    request that arrives mid-switch waits rather than racing a half-loaded body.

    A BODY THAT CANNOT FIT IS REFUSED, NOT ATTEMPTED. Qwen2.5-72B is 145 GB and
    this machine has 103 GB. Starting it would thrash for many minutes and then
    fail; the refusal names the two numbers instead.
    """

    def __init__(self, body: "CatalogBody", backend: Any, *, ready_timeout: float = 900.0):
        self.body = body
        self.backend = backend
        self.ready_timeout = ready_timeout
        self.state = "ready"
        self.error: Optional[str] = None
        self._lock = threading.RLock()
        # Tool capability belongs to the BODY, not to the server. Qualifying
        # once at startup left a 0.6B's verdict standing after a switch to a
        # different body -- the new model inheriting an old model's unverified
        # action capability, which is the precise failure S034 s15 names.
        self.tools_verdict: Dict[str, Any] = {"qualified": False,
                                              "reason": "not yet qualified"}

    @property
    def identity(self) -> str:
        return self.body.name

    @property
    def greedy(self) -> bool:
        return profile_is_greedy(self.body.path)

    @property
    def native_tools(self) -> bool:
        """Does this body's own artifact template declare tools?

        The sealed artifact's chat_template.jinja has a `tools` slot and states
        the call format the body was trained on. When it does, the template is
        the authority on the tool contract and a hand-written system message is
        both redundant and worse -- it is what made the body invent `shell`.
        """
        return self.body.kind == "noetic_native"

    def catalog(self) -> List[Dict[str, Any]]:
        from .catalog import catalog as _catalog
        return [b.to_openai(loaded=(b.name == self.body.name)) for b in _catalog()]

    def admits(self, body: "CatalogBody") -> Optional[str]:
        if not body.bytes:
            return None
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None
        if body.bytes > total * 0.92:
            return (f"{body.name} is {body.bytes / 1e9:.0f} GB and this machine has "
                    f"{total / 1e9:.0f} GB. Loading it would page the box into swap "
                    f"rather than run, so it is refused here. Bodies that fit are "
                    f"listed by `hcli use`.")
        return None

    def switch(self, name: str) -> Dict[str, Any]:
        from .catalog import resolve
        with self._lock:
            target = resolve(name)
            if target is None:
                raise LookupError(
                    f"no body named {name!r}. `hcli use` lists what is available.")
            if target.name == self.body.name and self.state == "ready":
                return {"switched": False, "resident": self.body.name}
            refusal = self.admits(target)
            if refusal:
                raise MemoryError(refusal)
            from .runtime_iface import make_backend_for_model
            previous = self.body.name
            self.state = "switching"
            try:
                try:
                    self.backend.stop()
                except Exception:
                    pass
                backend = make_backend_for_model(target.path)
                backend.spawn()
                if not backend.ready(self.ready_timeout):
                    raise RuntimeError(
                        f"{target.name} did not become ready within "
                        f"{self.ready_timeout:.0f}s")
                self.backend = backend
                self.body = target
                self.state = "ready"
                self.error = None
                self.qualify_tools()
                return {"switched": True, "from": previous,
                        "resident": target.name,
                        "tools": dict(self.tools_verdict)}
            except Exception as exc:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                raise

    def qualify_tools(self, prefix: Any = None, registry: Any = None) -> Dict[str, Any]:
        """Can THIS body emit a typed action, under its real conditions?

        Same contract the traffic uses: a native body is probed with its own
        template `tools` slot populated, because that is what it will see.
        """
        from .chat_tools import openai_schemas, qualify, system_block
        self.qualify_prefix = prefix if prefix is not None else getattr(
            self, "qualify_prefix", None)
        self.qualify_registry = registry if registry is not None else getattr(
            self, "qualify_registry", None)
        schemas = (openai_schemas(self.qualify_registry)
                   if self.native_tools and self.qualify_registry else None)
        prefix_messages = list(self.qualify_prefix or [])
        if not self.native_tools and self.qualify_registry is not None:
            from .chat_tools import prepend_system
            prefix_messages = prepend_system(
                prefix_messages, system_block(registry=self.qualify_registry))

        def _complete(convo):
            payload: Dict[str, Any] = {"messages": convo, "max_tokens": 96}
            if schemas:
                payload["tools"] = schemas
            return _text_of(self.backend.complete(payload))

        self.tools_verdict = qualify(_complete, prefix=prefix_messages)
        return self.tools_verdict

    def complete(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        wanted = str(payload.get("model") or "").strip()
        with self._lock:
            if wanted and wanted != self.body.name:
                from .catalog import resolve
                try:
                    target = resolve(wanted)
                except LookupError:
                    target = None
                # An unknown model name is NOT a silent fallback to whatever is
                # loaded: answering as a different body than the caller selected
                # is the same lie as serving a different sampler.
                if target is None:
                    raise LookupError(
                        f"no body named {wanted!r} -- the loaded resident is "
                        f"{self.body.name}. `hcli use` lists what is available.")
                if target.name != self.body.name:
                    self.switch(target.name)
            # The `model` field is OURS -- a catalog name that selects which body
            # is loaded. It must not reach the backend: mlx_lm.server reads it as
            # a HuggingFace repo id and goes to the network for it, which turned
            # a correct switch into "Repository Not Found for Qwen3-0.6B". The
            # backend serves the one body it has loaded and needs no name for it.
            inner = {k: v for k, v in payload.items() if k != "model"}
            return self.backend.complete(inner, timeout) if timeout is not None \
                else self.backend.complete(inner)


def make_handler(backend: Any, identity: str, *, greedy: bool,
                 health: Dict[str, Any], repo: Any = None,
                 registry: Any = None, stores: Optional[Dict[str, Any]] = None):
    stores = stores or {}
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802 - CORS preflight from a browser
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _identity(self) -> str:
            return getattr(backend, "identity", None) or identity

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/") or "/"
            if path in ("/v1/models", "/models"):
                # Every body a person could pick, so Open WebUI's dropdown is
                # the model picker rather than a one-entry label.
                rows = getattr(backend, "catalog", None)
                if callable(rows):
                    return self._send(200, {"object": "list", "data": rows()})
                return self._send(200, models_payload(identity))
            if path in ("/health", "/"):
                live = dict(health)
                if hasattr(backend, "body"):
                    live.update(resident=backend.identity,
                                model=backend.body.path,
                                state=backend.state,
                                sampling=("greedy-argmax" if backend.greedy
                                          else "profile-default"))
                    verdict = getattr(backend, "tools_verdict", None)
                    if verdict is not None and "tools" in live:
                        from .chat_tools import CHAT_TOOLS
                        live["tools"] = {
                            "qualified": bool(verdict.get("qualified")),
                            "names": sorted(CHAT_TOOLS) if verdict.get("qualified") else [],
                            "reason": verdict.get("reason"),
                            # The reply that failed. A verdict without the
                            # evidence behind it cannot be argued with -- and
                            # this gate has already produced one false negative
                            # against sealed-3.14.
                            "reply_excerpt": verdict.get("reply_excerpt"),
                        }
                    if backend.error:
                        live["error"] = backend.error
                return self._send(200, live)
            self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            route = self.path.rstrip("/")
            if route in ("/v1/switch", "/switch"):
                # The control door `hcli use` knocks on. Switching is also
                # reachable by naming a model in a chat request; this exists so
                # a person can pay the load cost deliberately instead of
                # discovering it inside their first message.
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except Exception as exc:
                    return self._send(400, {"error": {"message": f"bad json: {exc}"}})
                switch = getattr(backend, "switch", None)
                if not callable(switch):
                    return self._send(409, {"error": {
                        "message": "this surface serves a single fixed body"}})
                try:
                    return self._send(200, switch(str(body.get("model") or "")))
                except LookupError as exc:
                    return self._send(404, {"error": {"message": str(exc)}})
                except MemoryError as exc:
                    return self._send(507, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(502, {"error": {
                        "message": f"{type(exc).__name__}: {exc}"}})
            if route not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {
                    "message": f"no route {self.path}; this server serves "
                               f"/v1/models, /v1/chat/completions and /v1/switch"}})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                return self._send(400, {"error": {"message": f"bad json: {exc}"}})
            if not isinstance(body, dict):
                return self._send(400, {"error": {"message": "body must be an object"}})
            live_greedy = getattr(backend, "greedy", greedy)
            if live_greedy:
                refusal = sampler_refusal(body)
                if refusal is not None:
                    return self._send(400, refusal)
            messages = body.get("messages") or []
            if not messages:
                return self._send(400, {"error": {
                    "message": "no messages: send {'messages':[{'role':'user',"
                               "'content':'...'}]}"}})
            # DURABLE SESSION. Open WebUI sends no id and resends the whole
            # conversation, so the first user turn's digest is the key: stable
            # for the life of that conversation, no protocol change needed.
            session = None
            if stores.get("state_root"):
                from .chat_state import ChatSession, session_key, working_set
                root = stores["state_root"]
                session = ChatSession.load(
                    root, session_key(messages, root,
                                      explicit=body.get("session_id")
                                      or body.get("chat_id")))
                session.resident = getattr(backend, "identity", identity)
                # The session inherits the surface's authority. A session record
                # defaulting to "read" while the surface was started --write left
                # the working set and the response both claiming read on a build
                # session -- true of a fresh record, wrong for this surface.
                session.authority = "write" if stores.get("engine") is not None else "read"
                session.turns += 1
                # RESUME BEFORE ANYTHING ELSE. A checkpoint means a previous
                # invocation yielded mid-objective; the resuming turn must see
                # where it stands and whether the repository moved under it.
                from .chat_continuity import compact, resume, resume_block
                revived = resume(session)
                durable = working_set(session)
                revival = resume_block(revived) if (
                    revived and session.turns <= 1) else ""
                if revival:
                    durable = (durable + "\n\n" + revival) if durable else revival
                if durable:
                    # AFTER the tool contract and repo identity, BEFORE the
                    # conversation: it changes as the plan advances, so it must
                    # not sit in front of the material the prefix cache reuses.
                    messages = [*messages[:1], {"role": "system",
                                                "content": durable},
                                *messages[1:]] if (
                        messages and messages[0].get("role") == "system") else [
                        {"role": "system", "content": durable}, *messages]
            if repo is not None:
                # The folder HCLI was opened in, in front of the conversation.
                # Stable block first so the resident's prefix cache keeps it
                # across turns -- see hcli/repo_context.py for why the obvious
                # order would cost 20s per follow-up instead of 0.5s.
                from .repo_context import inject
                messages = inject(messages, repo)
                body = {**body, "messages": messages}
            if session is not None:
                # COMPACT WHEN THE CONVERSATION OUTGROWS ITS SHARE. Turns leave
                # the working set into the session archive; the invariant
                # leading region is untouched so the prefix cache still hits.
                window = int(health.get("context_window") or 8192)
                messages, compaction = compact(messages, session, window=window)
                if compaction.compacted:
                    body = {**body, "messages": messages}
                    checkpoint_note = (
                        f"compacted {compaction.evicted_turns} turns "
                        f"({compaction.before_tokens}->{compaction.after_tokens} tok)")
                    from .chat_continuity import checkpoint as _ckpt
                    _ckpt(session, resident=session.resident, note=checkpoint_note)
            payload = {k: v for k, v in body.items() if k != "stream"}
            payload.setdefault("model", identity)
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            trace: list = []
            live_tools = getattr(backend, "tools_verdict", None)
            tools_ok = bool((live_tools or {}).get("qualified")) if live_tools is not None else True
            try:
                if registry is not None and tools_ok:
                    # The chat can look things up. One backend, one registry --
                    # the same one `hcli agentos research-gate` uses, with its
                    # read/research permission set, so a browser session cannot
                    # reach a tool that writes.
                    from .chat_tools import (builder_menu, openai_schemas,
                                              prepend_system, run_with_tools,
                                              system_block)

                    native = bool(getattr(backend, "native_tools", False))
                    schemas = openai_schemas(
                        registry, list(builder_menu(
                            stores.get("engine") is not None))) if native else None

                    def _complete(convo):
                        inner = {**payload, "messages": convo}
                        if schemas:
                            inner["tools"] = schemas
                        return _text_of(backend.complete(inner))

                    writing = stores.get("engine") is not None
                    with_contract = (messages if native else prepend_system(
                        messages, system_block(registry=registry,
                                               write=writing)))
                    answer, trace = run_with_tools(
                        _complete, with_contract, registry, native=native,
                        cache=stores.get("cache"),
                        knowledge=stores.get("knowledge"),
                        engine=stores.get("engine"))
                    result = type("R", (), {
                        "text": answer, "finish_reason": "stop", "degraded": [],
                        "prompt_tokens": None, "completion_tokens": None,
                        "total_tokens": None, "raw": {}})()
                else:
                    result = backend.complete(payload)
            except LookupError as exc:
                return self._send(404, {"error": {
                    "message": str(exc), "type": "model_not_found"}})
            except MemoryError as exc:
                return self._send(507, {"error": {
                    "message": str(exc), "type": "insufficient_storage"}})
            except Exception as exc:
                return self._send(502, {"error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "resident_error"}})
            if not body.get("stream"):
                answered = getattr(backend, "identity", identity)
                body_out = chat_payload(result, answered, request_id=request_id)
                if trace:
                    body_out["hawking"]["tools_used"] = trace
                if session is not None:
                    from .chat_state import remember_turn
                    remember_turn(session, messages, _text_of(result),
                                  knowledge=stores.get("knowledge"))
                    session.save()
                    body_out["hawking"]["session"] = {
                        "id": session.id, "turns": session.turns,
                        "active_plan": session.active_plan,
                        "authority": session.authority,
                        "checkpoint": session.last_checkpoint or None}
                    if locals().get("compaction") is not None and compaction.compacted:
                        body_out["hawking"]["compaction"] = compaction.to_dict()
                return self._send(200, body_out)
            # An SSE body has no Content-Length, so under HTTP/1.1 keep-alive a
            # client cannot tell where it ends and waits until it times out --
            # which is exactly what a browser chat looks like when it hangs
            # forever on a reply that was in fact delivered. Close the
            # connection to frame the end of the stream.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True
            for chunk in stream_frames(
                    result, getattr(backend, "identity", identity),
                    request_id=request_id):
                self.wfile.write(chunk)
                self.wfile.flush()

    return Handler


def _context_window(model: str) -> int:
    """The loaded body's usable input budget, from the existing authority.

    `context_budget.resolve` is the repo's single authority on window
    arithmetic. It is KEYWORD-ONLY, and calling it positionally raised a
    TypeError that a bare `except` swallowed -- returning a fallback that
    happened to equal the real value for this profile, so the broken path was
    invisible. The fallback is now deliberately NOT a plausible real number, so
    the next such failure shows up instead of hiding.
    """
    try:
        budget = resolve(model_path=str(model))
    except Exception:
        return _WINDOW_UNKNOWN
    for attr in ("usable_input_tokens", "per_request_ctx", "model_ceiling"):
        value = getattr(budget, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return _WINDOW_UNKNOWN


def build_server(model: str, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 ready_timeout: float = 600.0, repo: Any = None,
                 registry: Any = None, write: bool = False):
    """Spawn the resident, wait for it, and return (httpd, identity, health)."""
    from .catalog import Body, catalog, resolve
    from .runtime_iface import classify_backend, make_backend_for_model

    try:
        body = resolve(model)
    except LookupError:
        body = None
    if body is None:
        # A path that is not in the catalog is still servable; it just has no
        # menu entry. Naming it after its own file keeps identity honest.
        body = Body(name=Path(str(model)).stem, path=str(model),
                    kind=classify_backend(str(model)), source="user")

    backend = make_backend_for_model(body.path)
    backend.spawn()
    if not backend.ready(ready_timeout):
        raise RuntimeError(
            f"the resident did not become ready within {ready_timeout:.0f}s. "
            f"Its log tail is the evidence: "
            f"{getattr(backend, 'log_tail', lambda *_: '(no log)')()!s:.400}"
        )
    resident = Resident(body, backend, ready_timeout=ready_timeout)
    identity = resident.identity
    greedy = resident.greedy
    del catalog
    health = {
        "status": "ok",
        "resident": identity,
        "model": body.path,
        "sampling": "greedy-argmax" if greedy else "profile-default",
        "streaming": "single-chunk (the connector returns a completed string; "
                     "SSE shape is provided so browser clients render, not as "
                     "evidence of incremental decode)",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/health"],
        "switchable": True,
        # The body's real window, so compaction triggers on the actual ceiling
        # rather than a constant. resolve() is the repo's existing authority on
        # this and already reads native profiles and GGUF headers.
        "context_window": _context_window(body.path),
    }
    if repo is not None:
        health["repo"] = {"name": repo.name, "root": str(repo.root),
                          "git": repo.is_git, "branch": repo.branch}
    if registry is not None:
        # CAPABILITY IS EARNED PER BODY. A body that cannot emit a typed action
        # must not be handed a tool contract: it answers in prose that sounds
        # like it searched, and a reader cannot tell that from an answer that
        # did. One cheap probe at load, and again after every switch.
        from .chat_tools import CHAT_TOOLS

        prefix = None
        if repo is not None:
            from .repo_context import inject
            prefix = inject([], repo)
        verdict = resident.qualify_tools(prefix, registry)
        health["tools"] = {
            "qualified": bool(verdict.get("qualified")),
            "names": sorted(CHAT_TOOLS) if verdict.get("qualified") else [],
            "reason": verdict.get("reason"),
            "reply_excerpt": verdict.get("reply_excerpt"),
        }

    stores: Dict[str, Any] = {}
    if registry is not None:
        # DISK-FIRST OBSERVATIONS AND BIDIRECTIONAL EXPANSION, from stores that
        # already exist: PasteCache is a content-addressed text store whose
        # store() had no caller outside its own tests, and KnowledgeStore is a
        # bounded hot index over a gzip cold archive with a real recall() valve.
        # Neither is a new architecture; both were built and left unwired.
        workspace = str(repo.root) if repo is not None else os.getcwd()
        try:
            from .paste_cache import PasteCache
            stores["cache"] = PasteCache(workspace)
        except Exception:
            stores["cache"] = None
        try:
            from .knowledge import KnowledgeStore
            stores["knowledge"] = KnowledgeStore(workspace)
        except Exception:
            stores["knowledge"] = None
        health["observation_store"] = stores.get("cache") is not None
        health["knowledge_store"] = stores.get("knowledge") is not None
        stores["state_root"] = workspace
        health["sessions"] = True
        if write:
            # HANDS, ONLY WHEN ASKED FOR. The engine is what makes repo.edit
            # reachable; a read session simply has none, so the door is absent
            # from the menu rather than refused by persuasion.
            from .engine import Engine
            from .workspace import Workspace
            stores["engine"] = Engine(Workspace(workspace),
                                      runtime_provider=lambda: resident)
        health["authority"] = "write" if write else "read"
    # The body's own capability surface, so the model and the operator can both
    # see what it can do -- named as capability (INSPECT, BUILD, ...) rather
    # than as package, and probed against live machinery rather than asserted.
    try:
        from .capabilities import capability_map
        health["capabilities"] = capability_map()["can"]
    except Exception:
        pass
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(resident, identity, greedy=greedy,
                                   health=health, repo=repo, registry=registry,
                                   stores=stores))
    httpd.backend = resident  # type: ignore[attr-defined]
    return httpd, identity, health


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hcli serve",
        description="OpenAI-compatible endpoint over the persistent Hawking resident.")
    # POSITIONAL, because `hcli serve Qwen3-14B` is what a person types. --model
    # stays as an alias so existing scripts keep working.
    ap.add_argument("model", nargs="?", default=None,
                    help="which body to load: a name from `hcli use`, or a path")
    ap.add_argument("--model", dest="model_flag", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--no-repo", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-tools", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true",
                    help="grant repo-scoped write authority to this session")
    a = ap.parse_args(list(argv or []))
    a.model = a.model or a.model_flag

    model = a.model or str(Path(__file__).resolve().parent / "hawking-native.sealed-3.14.json")
    repo = None
    if not a.no_repo:
        from .repo_context import RepoContext
        repo = RepoContext.detect(os.getcwd())
    registry = None
    if not a.no_tools:
        from .chat_tools import build_registry
        root = str(repo.root) if repo is not None else os.getcwd()
        registry = build_registry(root, root)
    httpd, identity, health = build_server(
        model, host=a.host, port=a.port, ready_timeout=a.ready_timeout,
        repo=repo, registry=registry, write=bool(getattr(a, "write", False)))
    print(json.dumps({"listening": f"http://{a.host}:{a.port}",
                      "openai_base_url": f"http://{a.host}:{a.port}/v1",
                      **health}), flush=True)
    # A MODEL SERVER MUST NOT SURVIVE ITS SURFACE. `hcli stop` sends SIGTERM,
    # whose default handler exits WITHOUT running the finally below, so every
    # stop and every switch orphaned the spawned mlx_lm.server. Measured: 11
    # orphans accumulated across one session, oldest 1h28m, holding ~7 GB of
    # swap-backed state -- swap fell from 9.7 GB to 2.5 GB when they were
    # reaped. Handle the signal so shutdown is shutdown.
    import signal as _signal

    def _shutdown(signum, _frame):  # noqa: ANN001
        try:
            httpd.backend.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        raise SystemExit(0)

    for _sig in (_signal.SIGTERM, _signal.SIGHUP, _signal.SIGINT):
        try:
            _signal.signal(_sig, _shutdown)
        except (ValueError, OSError):
            pass
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            httpd.backend.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
