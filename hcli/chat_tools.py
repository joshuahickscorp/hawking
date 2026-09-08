"""Letting the browser chat look things up, without a second tool system.

WHAT WAS MISSING. hcli/serve.py reached ZERO tools: it stripped everything but a
chat payload and called backend.complete. Meanwhile the registry holds working,
call-sited web and filesystem tools that `hcli agentos research-gate` exercises
on every run. The capability existed and the chat could not reach it -- the
exact shape this campaign keeps finding.

FEW DOORS, NOT FORTY. The registry has ~95 typed tools. Exposing them all would
spend the model's attention on a menu instead of the question, and most are
gated behind permissions a chat session does not hold. Five are offered, chosen
because they answer the two things a person actually asks a local assistant that
its weights cannot know: what is on the public web, and what is in this folder.

READ AND RESEARCH ONLY, BY CONSTRUCTION. The registry is built with
`default_tool_registry`, whose permission set is READ_ONLY / RESEARCH /
REVERSIBLE_*. A browser chat therefore cannot write, land, download a body or
run a benchmark -- not because this module filters those out afterwards, but
because the registry it is handed never admits them. Refusing a capability the
caller could still reach by another spelling is not a control.

A PLAIN JSON LINE, NOT OPENAI FUNCTION-CALLING. Measured this session: a 4B body
could not reliably emit valid JSON for a file edit, and sealed-3.14's own
structured-output contract runs prompt-plus-validate-plus-retry because the
resident enforces JSON syntax and nothing else. A tool protocol that only works
when the model is strong is not wired for the bodies in this catalog, so the
protocol is one object with two keys and the parser is forgiving about what
surrounds it.

EVERY CALL IS ON THE RECORD. The answer carries what was called, with each
tool's own provenance -- source URL and retrieval time for web work. A reader
must be able to tell an answer that consulted a source from one that recalled it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: The doors offered to a chat session. Names must exist in the registry; the
#: test asserts that, so a rename cannot leave a menu entry pointing at nothing.
#: name -> what it is FOR. The argument shape is NOT written here: it is read
#: from the tool's own schema at runtime, because a hand-written one drifts.
#: Measured: this table said fs.search takes {"query": ...} and the tool
#: requires `pattern`, so every search failed and the model burned its whole
#: budget on a menu entry that was itself a false affordance.
CHAT_TOOLS: Dict[str, str] = {
    "web.search": "find pages on the public web",
    "web.fetch": "read one public URL",
    "fs.read": "read a file in this project",
    "fs.list": "list a directory in this project",
    "fs.search": "find text in this project's source",
    "observation.expand": "read more of an earlier large result by its [PASTE id]",
    "context.recall": "recall facts this workspace learned earlier",
    "debug.diagnose": "localize a failure from a test/command output handle",
}

MAX_CALLS = 3
RESULT_CHARS = 3000
# One level of nesting, because the object ALWAYS contains a nested
# "arguments": {...}. A flat [^{}]* class cannot span it, so a call wrapped in
# prose -- exactly what a small body emits -- parsed as no call at all.
_OBJECT = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.S)
#: Qwen bodies wrap their action in this. sealed-3.14 does.
_TOOL_CALL_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
#: A third dialect, emitted by sealed-3.14 when it DECIDES rather than copies:
#:   <function=shell><parameter=command>ls -a</parameter></function>
#: Parsing it is not endorsement -- it is how the refusal gets to say WHICH tool
#: the body reached for instead of "did not emit a parseable action".
_XML_FUNCTION = re.compile(r"<function=([\w.\-]+)\s*>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.\-]+)\s*>(.*?)</parameter>", re.S)


LOCAL_SHAPES = {
    "observation.expand": '{"id": <string>}   optional: query, start, end',
    "debug.diagnose": '{"id": <string>, "command": <string>}   optional: exit_code',
}


def argument_shape(registry: Any, name: str) -> str:
    """The tool's own required/optional arguments, from its schema.

    Generated, never transcribed: the schema is the authority on what the tool
    accepts, and a menu that disagrees with it sends the model into a loop it
    cannot escape.
    """
    if name in LOCAL_SHAPES:
        return LOCAL_SHAPES[name]
    if name in BUILDER_SHAPES:
        return BUILDER_SHAPES[name]
    spec = registry.get(name) if registry is not None else None
    schema = getattr(spec, "input_schema", None) or getattr(spec, "schema", None)
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties") or {}
    required = [k for k in (schema.get("required") or []) if k in props]
    optional = [k for k in props if k not in required]
    parts = [f'"{k}": <{(props[k] or {}).get("type", "value")}>' for k in required]
    shape = "{" + ", ".join(parts) + "}"
    if optional:
        shape += f"   optional: {', '.join(sorted(optional))}"
    return shape


def system_block(names: Optional[Sequence[str]] = None,
                 registry: Any = None, write: bool = False) -> str:
    """The contract, stated once. Stable text, so it stays in the prefix cache."""
    menu = builder_menu(write)
    rows = []
    for name in (names or menu):
        if name not in menu:
            continue
        shape = argument_shape(registry, name)
        rows.append(f"  {name}  --  {menu[name]}"
                    + (f"\n      arguments: {shape}" if shape else ""))
    return (
        "You can look things up before answering. To use a tool, reply with ONLY "
        "a JSON object and nothing else:\n"
        '  {"tool": "web.search", "arguments": {"query": "mlx metal kernels"}}\n'
        "\nAvailable tools:\n" + "\n".join(rows) + "\n"
        "\nWhich to reach for:\n"
        "  about THIS project's files, code, or structure  ->  fs.read / fs.search / fs.list\n"
        "  about the public internet, current events, other projects' docs  ->  web.search / web.fetch\n"
        "  something you already know  ->  no tool, just answer\n"
        "\nMeasured: asked what fields a dataclass in this repo has, a body reached "
        "for web.search. The file was on local disk. Repository questions are "
        "answered from the repository.\n"
        "\nIf files from this project were already included above, they are a "
        "STARTING POINT, not the whole repository -- use fs.search when the answer "
        "needs code you were not shown.\n"
        "\nThe result comes back as the next message and then you answer normally. "
        f"You may use at most {MAX_CALLS} tools per question. If you already know "
        "the answer, just answer -- do not call a tool to confirm something you "
        "are sure of. Never invent a tool result: if a call fails, say so."
    )


def _repair(raw: str) -> Optional[Dict[str, Any]]:
    """Decode an object, tolerating one stray closing brace.

    sealed-3.14 emitted `{"function": "fs.list", "arguments": {"path": "hcli"}}}`
    -- correct except for a third brace. Refusing that is refusing a decision
    the body actually made.
    """
    for candidate in (raw, raw.rstrip().rstrip("}"), raw.rstrip() + "}"):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """A tool call in the reply, or None.

    MEET THE BODY WHERE IT IS. sealed-3.14 does not speak this module's
    protocol; it speaks Qwen's, and under real use it emitted

        <tool_call>
        {"function": "fs.list", "arguments": {"path": "hcli"}}}
        </tool_call>

    -- a correct decision, in its own native convention, with a stray brace.
    The first parser wanted `"tool"` and nothing else, so it read that as prose
    and handed the raw tool-call markup to the user as the answer. A protocol
    that only accepts its own spelling is not robust, it is provincial: accept
    `tool`, `function` and `name`, with or without the `<tool_call>` wrapper.
    """
    if not text:
        return None
    lowered = text
    xml = _XML_FUNCTION.search(text)
    if xml:
        return xml.group(1).strip(), {
            key.strip(): value.strip()
            for key, value in _XML_PARAM.findall(xml.group(2))}
    if not any(key in lowered for key in ('"tool"', '"function"', '"name"')):
        return None
    candidates: List[str] = []
    stripped = text.strip()
    for match in _TOOL_CALL_TAG.findall(text):
        candidates.append(match.strip())
    if stripped.startswith("```"):
        body = stripped.strip("`")
        if body.lower().startswith("json"):
            body = body[4:]
        candidates.append(body.strip())
    candidates.append(stripped)
    candidates.extend(_OBJECT.findall(text))
    for raw in candidates:
        parsed = _repair(raw)
        if parsed is None:
            continue
        name = None
        for key in ("tool", "function", "name"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
            # Qwen also nests: {"function": {"name": "x", "arguments": {...}}}
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                inner = value
                name = inner["name"].strip()
                args = inner.get("arguments")
                return name, args if isinstance(args, dict) else {}
        if not name:
            continue
        arguments = parsed.get("arguments") or parsed.get("parameters")
        if not isinstance(arguments, dict):
            arguments = {k: v for k, v in parsed.items()
                         if k not in ("tool", "function", "name", "arguments",
                                      "parameters")}
        return name, arguments
    return None


def _render(result: Any, name: str, registry: Any = None,
            cache: Any = None) -> str:
    ok = bool(getattr(result, "ok", False))
    if not ok:
        error = getattr(result, "error", None) or "failed"
        failure = str(getattr(result, "failure_class", "") or "")
        # An action rejected for its SHAPE gets the shape back, because
        # "INVALID_ARGUMENTS" alone is the same dead end as a bare
        # NOT_ADMITTED: it names a category and no way into the accepted one.
        if failure in {"INVALID_ARGUMENTS", "SCHEMA", "BAD_REQUEST"} or "argument" in str(error).lower():
            return (f"INVALID ACTION: {name} -- {error}\n"
                    f"{shape_help(name, registry)}\n"
                    f"Send the corrected action, or answer without it.")
        return (f"{name} FAILED: {error}\n"
                f"Tell the user the lookup failed. Do not invent what it "
                f"would have returned.")
    value = getattr(result, "value", None)
    try:
        body = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        body = str(value)
    if len(body) > RESULT_CHARS:
        # DISK FIRST, THEN A POINTER. Truncating at 3000 characters and dropping
        # the rest destroyed the only copy: a test log or a page the model needed
        # ten lines further down was simply gone, and the model could not ask for
        # it because nothing had kept it. PasteCache is a content-addressed store
        # that was already built for exactly this -- store(), slice(), search(),
        # get() -- with store() called from nothing but its own tests. So the
        # detail goes to disk and the context gets the head plus a handle it can
        # expand with observation.expand.
        head = body[:RESULT_CHARS]
        ref = None
        if cache is not None:
            try:
                ref = cache.store(body)
            except Exception:
                ref = None
        if ref is not None:
            body = (f"{head}\n... [{len(body)} chars total; the rest is kept at "
                    f"{ref.id} -- observation.expand "
                    f'{{"id": "{ref.id}", "query": "..."}} or '
                    f'{{"id": "{ref.id}", "start": {RESULT_CHARS // 80}, "end": ...}}]')
        else:
            body = head + f"\n... [truncated at {RESULT_CHARS} chars]"
    warning = ""
    if isinstance(value, dict) and value.get("truncated"):
        seen = value.get("files_seen")
        warning = (f"\nWARNING: this search stopped after {seen} files and did NOT "
                   f"cover the whole project. Absence of a match here is NOT "
                   f"evidence the term is absent -- narrow it with a `path` or "
                   f"`glob` before concluding anything negative.")
    return f"{name} returned:\n{body}{warning}"


#: Provenance a tool reports inside its own value rather than on the envelope.
_VALUE_PROVENANCE = ("source_url", "retrieved_at", "provider", "confidence")


def provenance_of(result: Any) -> Dict[str, Any]:
    """Where this observation came from, and when.

    web.search carries source_url / retrieved_at / provider / confidence in its
    VALUE, not on the result envelope, so lifting only `result.provenance` threw
    away exactly the fields a web answer has to be able to cite. It also lifts
    the result URLs themselves, because "which pages did this answer read" is
    the question a reader asks first.
    """
    raw = getattr(result, "provenance", None)
    out = dict(raw) if isinstance(raw, dict) else {}
    value = getattr(result, "value", None)
    if isinstance(value, dict):
        for key in _VALUE_PROVENANCE:
            if value.get(key) is not None:
                out[key] = value[key]
        rows = value.get("results")
        if isinstance(rows, list) and rows:
            urls = [r.get("url") for r in rows[:5]
                    if isinstance(r, dict) and r.get("url")]
            if urls:
                out["sources"] = urls
    return out


#: Source-ish files, for the narrowed retry below.
SOURCE_GLOB = "*.py"


def _search_needs_escalation(result: Any) -> bool:
    """A search that ran out of budget before it reached the source.

    MEASURED. `fs.search` for `make_backend_for_model` returned TWO hits, both
    in .hcli/receipts/*.json, with files_seen 5001 and truncated True -- the
    per-call file budget was spent walking thousands of generated receipts
    before the walk ever reached hcli/*.py, where the function is defined and
    called three times. The model then reported, faithfully and wrongly, that
    nothing in the project calls it.

    Truncation that only appears as a field inside the payload is not a
    warning; a caller reads the matches and believes them.
    """
    value = getattr(result, "value", None)
    if not isinstance(value, dict) or not value.get("truncated"):
        return False
    return len(value.get("matches") or []) < 20


def escalate_search(registry: Any, arguments: Dict[str, Any]) -> Optional[Any]:
    """Re-run a truncated search narrowed to source. Mechanical, not a decision.

    The repo's own principle: let the harness do the deterministic mechanics and
    let the model spend inference on judgment. Choosing a glob because a walk
    ran out of budget is mechanics.
    """
    if arguments.get("glob"):
        return None
    narrowed = {**arguments, "glob": SOURCE_GLOB}
    try:
        return registry.invoke("fs.search", narrowed)
    except Exception:
        return None


def run_with_tools(
    complete: Callable[[List[Dict[str, str]]], str],
    messages: List[Dict[str, str]],
    registry: Any,
    *,
    max_calls: int = MAX_CALLS,
    native: bool = False,
    cache: Any = None,
    knowledge: Any = None,
    engine: Any = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer, consulting tools when the model asks. Returns (text, trace).

    `complete` takes messages and returns the model's text, so this is testable
    without a resident and the caller keeps ownership of the backend.
    """
    conversation = [dict(m) for m in messages]
    trace: List[Dict[str, Any]] = []
    # The menu for THIS session. Authority decides the door set once, outside
    # the loop, so no turn can widen it.
    offered = builder_menu(engine is not None)
    for _ in range(max(0, max_calls)):
        text = complete(conversation)
        call = parse_call(text)
        if call is None:
            return text, trace
        name, arguments = call
        if name not in offered:
            # Naming what IS available turns a dead end into a retry that can
            # work -- the same rule the engine's test-command refusal follows.
            conversation.append({"role": "assistant", "content": text})
            conversation.append({"role": "user", "content":
                                 f"{name} is not available here. Available: "
                                 f"{', '.join(sorted(offered))}. Use one of "
                                 f"those or answer directly."})
            trace.append({"tool": name, "ok": False, "error": "not offered to chat"})
            continue
        arguments = coerce_arguments(registry, name, arguments)
        if name in BUILDER_TOOLS:
            result = run_builder_tool(name, arguments, engine=engine)
        elif name in LOCAL_TOOLS:
            result = run_local_tool(name, arguments, cache=cache,
                                    knowledge=knowledge)
        else:
            result = registry.invoke(name, arguments)
        escalated = None
        if name == "fs.search" and _search_needs_escalation(result):
            escalated = escalate_search(registry, arguments)
            if escalated is not None and getattr(escalated, "ok", False):
                result = escalated
        entry = {
            "tool": name,
            "arguments": arguments,
            "ok": bool(getattr(result, "ok", False)),
            "error": getattr(result, "error", None),
            "provenance": provenance_of(result),
        }
        if escalated is not None:
            entry["escalated"] = {"glob": SOURCE_GLOB,
                                  "reason": "first search truncated before reaching source"}
        trace.append(entry)
        conversation.append({"role": "assistant", "content": text})
        observation = _render(result, name, registry, cache)
        conversation.append(tool_response(name, observation) if native
                            else {"role": "user", "content": observation})
    # Budget spent. Ask for the answer itself rather than returning the last
    # tool call as though it were one.
    conversation.append({"role": "user", "content":
                         "You have used your tool budget. Answer now with what "
                         "you have, and say what you could not check."})
    return complete(conversation), trace


def build_registry(workspace: str, repo_root: Optional[str] = None) -> Any:
    """The standard read/research registry. No permission widening happens here."""
    from .tool_registry import default_tool_registry

    return default_tool_registry(workspace, repo_root=repo_root or workspace)


#: The probe a body must pass before it is offered tools at all. One exact
#: action, no ambiguity, cheap to run.
QUALIFY_PROMPT = (
    "List the files in the current directory. Use a tool to find out -- "
    "do not answer from memory."
)


def qualify(complete: Callable[[List[Dict[str, str]]], str],
            prefix: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Can this body emit a typed action at all?

    MEASURED REASON THIS EXISTS. The dropdown selects bodies from 0.6B to 30B.
    A 4B failed to emit valid JSON for a file edit three attempts running this
    session, and a body that cannot produce an action must not be handed a tool
    contract -- it will answer in prose that SOUNDS like it searched, and a
    reader cannot tell that from an answer that did. Capability is earned per
    body, not assumed from the fact that tools exist.

    QUALIFY UNDER THE CONDITIONS THE BODY WILL ACTUALLY RUN. `prefix` carries
    the same system context a real request gets. Probing with a bare user turn
    measured a shape no real request ever has, and sealed-3.14 returned an empty
    string to it -- so the flagship body was marked incapable by a probe that
    did not resemble its own traffic.
    """
    # AN EMPTY FIRST REPLY IS A WARM-UP, NOT A VERDICT. Measured on
    # sealed-3.14: backend.ready() returned True, this probe was the first real
    # completion, and it came back "" -- so the flagship resident was marked
    # tool-INCAPABLE while emitting the exact action correctly seconds later.
    # That is this repo's own scar (a server answering /v1/models in 4ms while
    # /v1/chat/completions returned nothing, and readiness saying READY) landing
    # inside a capability gate, where it does the most damage: silently
    # withholding a capability the body has.
    text = ""
    attempts = 0
    for attempt in range(2):
        attempts = attempt + 1
        try:
            text = complete([*(prefix or []),
                             {"role": "user", "content": QUALIFY_PROMPT}])
        except Exception as exc:
            return {"qualified": False, "reason": f"{type(exc).__name__}: {exc}",
                    "attempts": attempts}
        if str(text or "").strip():
            break
    if not str(text or "").strip():
        return {"qualified": False,
                "reason": "the body returned an empty reply twice; it is not "
                          "generating, so tool capability is undetermined rather "
                          "than refused",
                "reply_excerpt": "", "attempts": attempts}
    call = parse_call(text or "")
    if call is None:
        return {"qualified": False,
                "reason": "the body did not emit a parseable action for the "
                          "simplest possible request",
                "reply_excerpt": str(text or "")[:200], "attempts": attempts}
    if call[0] not in CHAT_TOOLS:
        return {"qualified": False,
                "reason": f"chose {call[0]!r}, which is not a tool it was offered",
                "reply_excerpt": str(text or "")[:200], "attempts": attempts}
    return {"qualified": True, "reason": None, "attempts": attempts}


def shape_help(name: str, registry: Any = None) -> str:
    """What a caller must send. Named fields, not a refusal."""
    if name in CHAT_TOOLS:
        shape = argument_shape(registry, name)
        return f"{name} takes: {shape}" if shape else f"{name}: {CHAT_TOOLS[name]}"
    return f"available tools: {', '.join(sorted(CHAT_TOOLS))}"


def prepend_system(messages: Sequence[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    """Put `text` at the front of the conversation's SINGLE system message.

    The native chat template refuses more than one, and refuses one that is not
    first: "artifact chat-template rendering failed: System message must be at
    the beginning." Prepending a second system turn for the tool contract on top
    of the repo context produced exactly that, and only on the native body --
    the MLX backend tolerated it, so every acceptance test passed while the
    flagship resident 502'd.

    Merging also serves prefix reuse: one system block, tool contract first
    (invariant), repo identity next (invariant), question-specific files last.
    """
    rows = [dict(m) for m in messages]
    if rows and rows[0].get("role") == "system":
        rows[0] = {**rows[0], "content": text + "\n\n" + str(rows[0].get("content") or "")}
        return rows
    return [{"role": "system", "content": text}, *rows]


def openai_schemas(registry: Any,
                   names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The offered tools as OpenAI function specs, for the artifact's own template.

    sealed-3.14's chat_template.jinja has a `tools` slot that renders
    "# Tools ... <tools>{schemas}</tools>" and then states the exact call format
    the body was trained on. Describing tools in a plain system message instead
    left that slot empty, and the body did what its training says to do when no
    tools are declared: it invented one (`shell`). Declaring them where the
    artifact expects them is not a workaround -- it is using the contract the
    seal ships with.
    """
    out: List[Dict[str, Any]] = []
    menu = {**CHAT_TOOLS, **BUILDER_TOOLS}
    for name in (names or CHAT_TOOLS):
        if name not in menu:
            continue
        spec = registry.get(name) if registry is not None else None
        schema = getattr(spec, "input_schema", None) or getattr(spec, "schema", None)
        if not isinstance(schema, dict):
            schema = _declared_schema(name)
        out.append({"type": "function", "function": {
            "name": name,
            "description": menu[name],
            "parameters": schema,
        }})
    return out


def tool_response(name: str, observation: str) -> Dict[str, str]:
    """A result in the shape the artifact's template recognises.

    The template inspects user turns for <tool_response>...</tool_response> when
    deciding what the last real query was, so results delivered as bare prose
    are read as new questions from the human.
    """
    return {"role": "user",
            "content": f"<tool_response>\n{observation}\n</tool_response>"}


def coerce_arguments(registry: Any, name: str,
                     arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Make the arguments match the schema's types where that is unambiguous.

    THE XML DIALECT CANNOT EXPRESS A NUMBER. The artifact's own template
    specifies <parameter=max_results>200</parameter> -- every value arrives as
    text, by construction, so a schema that wants an integer rejects a body that
    followed its instructions exactly. Measured: sealed-3.14 called fs.list with
    max_results "200" twice and got "expected integer, got string" both times,
    burning its budget on a shape its own trained format cannot produce.

    Deterministic mechanics belong to the harness, not to inference: "200" ->
    200 is not a judgement call. Only unambiguous conversions are made, and a
    value that does not convert is passed through untouched so the tool's own
    validator still gets to refuse it.
    """
    spec = registry.get(name) if registry is not None else None
    schema = getattr(spec, "input_schema", None) or getattr(spec, "schema", None)
    props = (schema or {}).get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return arguments
    out = dict(arguments)
    for key, value in list(out.items()):
        want = (props.get(key) or {}).get("type") if isinstance(props.get(key), dict) else None
        if not isinstance(value, str) or not want:
            continue
        text = value.strip()
        try:
            if want == "integer":
                out[key] = int(text)
            elif want == "number":
                out[key] = float(text)
            elif want == "boolean" and text.lower() in ("true", "false"):
                out[key] = text.lower() == "true"
            elif want == "array":
                out[key] = json.loads(text) if text.startswith("[") else [text]
        except (TypeError, ValueError):
            continue  # leave it; the tool's validator owns the refusal
    return out


#: Doors served by HCLI itself rather than by the tool registry: retrieval of
#: material this session already put on disk. COMPACTION MUST BE BIDIRECTIONAL
#: -- a summary that cannot be expanded is forgetting, and an evicted
#: observation the model cannot ask for again is lost, not compacted.
#: context.recall is NOT here: the registry already owns it, backed by
#: KnowledgeStore, and duplicating it locally both created a second
#: implementation of one door and let a hand-written shape ('limit')
#: override the real schema (max_results/max_chars) -- the same
#: transcribed-shape defect this module already fixed once.
LOCAL_TOOLS = ("observation.expand", "debug.diagnose")


def run_local_tool(name: str, arguments: Dict[str, Any], *,
                   cache: Any = None, knowledge: Any = None) -> Any:
    """Serve a retrieval door. Returns something shaped like a ToolResult."""

    class _R:
        def __init__(self, ok, value=None, error=None, provenance=None):
            self.ok, self.value, self.error = ok, value, error
            self.provenance = provenance or {}
            self.failure_class = None if ok else "INVALID_ARGUMENTS"

    if name == "observation.expand":
        if cache is None:
            return _R(False, error="no observation store in this session")
        paste_id = str(arguments.get("id") or "").strip()
        if not paste_id:
            return _R(False, error='observation.expand needs {"id": "<paste id>"} '
                                   'plus either "query" or "start"/"end"')
        try:
            if arguments.get("query"):
                hits = cache.search(paste_id, str(arguments["query"]), limit=40)
                return _R(True, value={"id": paste_id, "matches": [
                    {"line": n, "text": t} for n, t in hits]},
                    provenance={"source": "hcli.paste_cache", "paste": paste_id})
            start = int(arguments.get("start") or 1)
            end = int(arguments.get("end") or (start + 60))
            return _R(True, value={"id": paste_id, "start": start, "end": end,
                                   "text": cache.slice(paste_id, start, end)},
                      provenance={"source": "hcli.paste_cache", "paste": paste_id})
        except KeyError:
            return _R(False, error=f"no observation {paste_id!r} is kept in this session")
        except (TypeError, ValueError) as exc:
            return _R(False, error=f"{exc}")

    if name == "debug.diagnose":
        # Turn a captured failure into a localized fact: the deepest project
        # frame, the assertion, the enclosing function -- so the model reasons
        # about a cause, not a 200-line dump (S036 s21).
        if cache is None:
            return _R(False, error="no observation store in this session")
        paste_id = str(arguments.get("id") or "").strip()
        if not paste_id:
            return _R(False, error='debug.diagnose needs {"id": "<output handle>", '
                                   '"command": "<what produced it>"}')
        try:
            output = cache.get(paste_id)
        except (KeyError, ValueError) as exc:
            return _R(False, error=f"cannot read observation {paste_id!r}: {exc}")
        from pathlib import Path as _Path
        from .debug_capability import diagnose as _diagnose
        exit_code = arguments.get("exit_code")
        try:
            exit_code = int(exit_code) if exit_code is not None else 1
        except (TypeError, ValueError):
            exit_code = 1
        failure = _diagnose(output, command=str(arguments.get("command") or "?"),
                            exit_code=exit_code, root=_Path.cwd(), cache=cache)
        return _R(True, value=failure.to_dict(),
                  provenance={"source": "hcli.debug_capability", "paste": paste_id})

    return _R(False, error=f"{name} is not a local tool")


#: BUILDER DOORS. Never in CHAT_TOOLS by default: they enter the menu only when
#: a caller passes explicit write authority, and they are served by
#: Engine.apply_typed_mutation -- the same transaction Engine.execute() runs, so
#: patch application, result-file validation, red-before-green and rollback keep
#: exactly one implementation. There is deliberately no shell door: a typed
#: operation can be refused with a reason, an arbitrary command string cannot.
BUILDER_TOOLS: Dict[str, str] = {
    "repo.edit": ("change this project's source. Typed operations only, each "
                  "validated in its resulting file, rolled back if a named test "
                  "fails"),
}

BUILDER_SHAPES = {
    "repo.edit": (
        '{"operations": [{"op": "create|replace|insert_after", '
        '"path": <string>, "new_lines": [<string>], "old_lines": [<string>]}]}'
        '   optional: tests (list of paths to prove the change)'),
}

#: What a builder verdict means. Kept distinct on purpose: S035 s26 -- a
#: mutation that was rejected must never be reported as "fixed", and a change
#: with no test is "unproven", not "verified".
BUILDER_VERDICTS = ("accepted", "unproven", "rejected")


def builder_menu(write: bool) -> Dict[str, str]:
    """The doors for this session. Read-only unless authority was granted."""
    return {**CHAT_TOOLS, **(BUILDER_TOOLS if write else {})}


def run_builder_tool(name: str, arguments: Dict[str, Any], *,
                     engine: Any = None) -> Any:
    """Serve a builder door through the existing mutation transaction."""

    class _R:
        def __init__(self, ok, value=None, error=None, provenance=None):
            self.ok, self.value, self.error = ok, value, error
            self.provenance = provenance or {}
            self.failure_class = None if ok else "INVALID_ARGUMENTS"

    if name != "repo.edit":
        return _R(False, error=f"{name} is not a builder tool")
    if engine is None:
        return _R(False, error="this session has no write authority")
    operations = arguments.get("operations")
    if isinstance(operations, dict):
        operations = [operations]
    if not isinstance(operations, list) or not operations:
        return _R(False, error=f"repo.edit needs {BUILDER_SHAPES['repo.edit']}")
    tests = arguments.get("tests")
    if isinstance(tests, str):
        tests = [tests]
    verdict = engine.apply_typed_mutation(
        operations, tests if isinstance(tests, list) else None)
    status = verdict.get("status")
    # A rejected mutation is NOT an error the model should paper over: it is a
    # result it must report. So ok=True carries the verdict, and the verdict
    # carries the truth.
    return _R(True, value={
        "status": status,
        "applied": verdict.get("applied"),
        "rolled_back": verdict.get("rolled_back"),
        "reason": verdict.get("reason"),
        "diff": (verdict.get("diff") or "")[:2000],
        "paths": verdict.get("paths"),
    }, provenance={"source": "hcli.engine.apply_typed_mutation",
                   "status": status})


def _declared_schema(name: str) -> Dict[str, Any]:
    """A JSON-Schema for a door HCLI serves itself.

    Written once, here, next to the shape string the contract shows, so the two
    cannot drift the way a transcribed shape did.
    """
    if name == "repo.edit":
        return {
            "type": "object",
            "required": ["operations"],
            "properties": {
                "operations": {"type": "array", "items": {
                    "type": "object",
                    "required": ["op", "path"],
                    "properties": {
                        "op": {"type": "string",
                               "enum": ["create", "replace", "insert_after"]},
                        "path": {"type": "string"},
                        "old_lines": {"type": "array", "items": {"type": "string"}},
                        "new_lines": {"type": "array", "items": {"type": "string"}},
                    }}},
                "tests": {"type": "array", "items": {"type": "string"}},
            },
        }
    if name == "observation.expand":
        return {"type": "object", "required": ["id"], "properties": {
            "id": {"type": "string"}, "query": {"type": "string"},
            "start": {"type": "integer"}, "end": {"type": "integer"}}}
    return {"type": "object", "properties": {}}
