"""Letting the browser chat look things up, without a second tool system.

WHAT WAS MISSING. hcli/serve.py reached ZERO tools: it stripped everything but a
chat payload and called backend.complete. Meanwhile the registry holds working,
call-sited web, filesystem, research, Odyssey, physical, and Gravity tools that
other AgentOS paths exercise. The capability existed and the chat could not
reach it -- the exact shape this campaign keeps finding.

FULL REACHABILITY, COMPACT DISCOVERY. The registry now supplies the complete
permissioned tool surface to a chat session. The prompt shows canonical family
doors plus the small compatibility names; aliases and every other admitted
typed tool remain callable, and `tools.catalog` provides exact signatures on
demand. This preserves context gravity without silently dropping a capability.

PERMISSION IS STILL THE BOUNDARY. A read session receives only READ_ONLY and
RESEARCH. An explicit write session additionally receives reversible,
workspace, repository, and COSTLY tools; their own confirm fields and
verifiers remain in force. DESTRUCTIVE and EXTERNAL_WRITE never enter the chat
surface. Refusing an unpermissioned capability is therefore an explicit
authority result, not a hidden menu omission.

A PLAIN JSON LINE, NOT OPENAI FUNCTION-CALLING. Measured this session: a 4B body
could not reliably emit valid JSON for a file edit, and sealed-3.14's own
structured-output contract runs prompt-plus-validate-plus-retry because the
resident enforces JSON syntax and nothing else. A tool protocol that only works
when the model is strong is not wired for the bodies in this catalog, so the
protocol is one object with two keys and the parser is forgiving about what
surrounds it. A narrow compatibility adapter also accepts an affirmative
English statement for a zero-argument tool such as ``fs.list``; it never
guesses required arguments or turns a negative mention into an action.

EVERY CALL IS ON THE RECORD. The answer carries what was called, with each
tool's own provenance -- source URL and retrieval time for web work. A reader
must be able to tell an answer that consulted a source from one that recalled it.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
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
    "reverse.identify": "identify what a file is by its bytes, without running it",
    "forensics.snapshot": "preserve the failure context of this project before cleanup",
}


def _registry_menu(registry: Any) -> Dict[str, str]:
    """Return every registry name admitted by this session's permissions.

    ``discover(include_aliases=True)`` is intentional here. The canonical
    family door is what the prompt prefers, but old names remain callable for
    bodies trained on the pre-consolidation surface. Filtering by the registry
    context, rather than by a handwritten chat allowlist, is the authority
    check that makes the full surface real.
    """
    if registry is None or not hasattr(registry, "discover"):
        return {}
    context = getattr(registry, "context", None)
    permissions = set(getattr(context, "permissions", ()) or ())
    result: Dict[str, str] = {}
    try:
        items = registry.discover(include_aliases=True)
    except (AttributeError, TypeError):
        return result
    for item in items:
        if item.get("mutation") not in permissions:
            continue
        name = str(item.get("name") or "").strip()
        if name:
            result[name] = str(item.get("description") or name)
    return result


def session_menu(registry: Any = None, *, write: bool = False) -> Dict[str, str]:
    """All tools reachable by this chat session, with compatibility doors."""
    menu = _registry_menu(registry)
    # Local retrieval doors are not ToolSpecs, and the static entries also keep
    # small/fake registries used by the parser tests usable.
    menu.update(CHAT_TOOLS)
    if write:
        menu.update(BUILDER_TOOLS)
    return menu


_PROMPT_CONTROL_TOOLS = (
    # One retrieval/index door per common decision. Every other admitted name
    # remains callable through session_menu and discoverable via tools.catalog.
    "tools.catalog",
    "receipt.read",
    "campaign.state",
    "processes.summary",
    "git.status",
    "audit",
    "gravity.inspect",
    "physical.measure",
    "odyssey.read",
)


def prompt_menu_names(registry: Any = None, *, write: bool = False) -> List[str]:
    """Small canonical prompt surface; the full menu remains callable.

    Serializing every canonical name defeated Tool Gravity: the live Kimi
    contract reached 52 names / 15,614 characters and a direct receipt read
    wandered through vmcp and unrelated session files. Keep high-frequency
    observation doors plus one focused catalog door in context. Direct aliases
    and omitted registry tools are still accepted by ``run_with_tools``; this
    changes discovery cost, not reachability or authority.
    """
    offered = session_menu(registry, write=write)
    names = set(CHAT_TOOLS)
    names.update(name for name in _PROMPT_CONTROL_TOOLS if name in offered)
    if write:
        names.update(BUILDER_TOOLS)
    return sorted(name for name in names if name in offered)

# Twelve bounded rounds cover a real engineering path: orient/search, read,
# discriminator, edit, verification, and recovery decisions. Eight ended a
# fresh P0 liveness run after seven real observations but before it had covered
# the requested objective/runtime categories. A measured five-category blind
# mission then needed one recovery decision per unresolved class; ten rounds
# ended after the first valid recovery. The repeat guard, prompt admission guard
# and focused-test rule still cap cost.
MAX_CALLS = 12
RESULT_CHARS = 3000
# One level of nesting, because the object ALWAYS contains a nested
# "arguments": {...}. A flat [^{}]* class cannot span it, so a call wrapped in
# prose -- exactly what a small body emits -- parsed as no call at all.
_OBJECT = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.S)
#: Qwen bodies wrap their action in this. sealed-3.14 does.
_TOOL_CALL_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_TOOL_PROTOCOL_MARKERS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
)
#: A third dialect, emitted by sealed-3.14 when it DECIDES rather than copies:
#:   <function=shell><parameter=command>ls -a</parameter></function>
#: Parsing it is not endorsement -- it is how the refusal gets to say WHICH tool
#: the body reached for instead of "did not emit a parseable action".
#: SCAR: under greedy-argmax the opener keeps eroding, not just the leading
#: `<`. First observed: `=function=fs.read>` (od -c confirmed, 3x in one
#: reply). Three cycles after THAT fix deployed, every call in every one of
#: those cycles had dropped `function` entirely too -- bare `=fs.read>`, then
#: `==fs.read>` (doubled `=`) -- while <tool_call>, <parameter=...>, and every
#: closer stayed intact throughout. One cycle's ENTIRE reply was exactly one
#: such call, so parse_calls returning [] wasted the whole turn: zero tool
#: executions, same failure class as the original drop. Meet the body where
#: it is: `<` and `function` are each optional, one or more `=` is not.
_XML_FUNCTION = re.compile(r"<?(?:function)?=+([\w.\-]+)\s*>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.\-]+)\s*>(.*?)</parameter>", re.S)
_COMPACT_OPS = {
    "read": "fs.read",
    "list": "fs.list",
    "search": "fs.search",
    # Builder aliases remain authority-neutral here: run_with_tools still
    # rejects them unless the session offered the canonical write/test door,
    # and repo.edit still passes through Engine's transaction/verifier.
    "edit": "repo.edit",
    "test": "tests.run",
    "fetch": "web.fetch",
    "web_search": "web.search",
}


def _compact_tool_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in _COMPACT_OPS:
        return _COMPACT_OPS[key]
    # Dotted names are accepted as canonical registry names. `run_with_tools`
    # still performs the session-menu authority check before dispatch, so this
    # broadens syntax without broadening permissions.
    if re.fullmatch(r"[a-z][\w.-]*\.[a-z][\w.-]*", key):
        return key
    return None
# KIMI-VL has no tool-aware chat template and, under the explicit contract,
# can sometimes state a single zero-argument action in English instead of
# emitting JSON: "I will use the `fs.list` tool ...". Only this affirmative,
# unambiguous form is adapted, and only for tools whose schema permits an empty
# argument object. Negative mentions and tools with required arguments remain
# ordinary prose, never actions.
_NARRATIVE_OPTIONAL_ACTION = re.compile(
    r"^\s*(?:[^\n]{0,180}?\b)?"
    r"(?:i\s+will|i['’]ll|let\s+me|first,?\s+i\s+will|first,?\s+i['’]ll)\s+"
    r"(?:now\s+)?(?:use|call|invoke)\s+(?:the\s+)?[`<]?([\w.\-]+)[`>]?\s+(?:tool|function)\b",
    re.I,
)
_NARRATIVE_OPTIONAL_TOOLS = frozenset({"fs.list", "forensics.snapshot"})

_EXPLICIT_TOOL_IMPERATIVE = re.compile(
    r"(?:^|[.!?]\s+)(?:please\s+)?(?:use|call|invoke)\b.{0,120}"
    r"(?:\btools?\b|\breturned observation\b|\bfs\.|\bfilesystem\.|"
    r"\breceipt\.|\brepository file)",
    re.I | re.S,
)
_EXPLICIT_CAPABILITY_IMPERATIVE = re.compile(
    r"(?:^|[.!?]\s+|\n)\s*(?:please\s+)?"
    r"(?:discover|inventory|enumerate|exercise|use|inspect)\b.{0,180}"
    r"\b(?:admitted\s+)?(?:capabilit(?:y|ies)|tool\s+surface)\b",
    re.I | re.S,
)


def explicit_tool_observation_required(messages: Sequence[Dict[str, str]]) -> bool:
    """Whether the latest human turn explicitly requires a dispatched tool.

    This is not an automatic tool chooser. It only distinguishes an ordinary
    answer from a user contract such as "use the file-reading tool; do not
    guess". In the latter case, unsupported prose must not be returned as if an
    observation happened.
    """
    latest = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            latest = str(message.get("content") or "")
            break
    if not latest:
        return False
    # Both positive expressions begin at a sentence/line boundary and require
    # the imperative verb there.  Therefore "Do not use a tool" does not
    # match, while a later independent sentence such as "Use every admitted
    # capability" still does.  A global negation search used to erase that
    # later positive contract merely because the prompt also excluded one
    # unsafe or irrelevant tool.
    return bool(
        _EXPLICIT_TOOL_IMPERATIVE.search(latest)
        or _EXPLICIT_CAPABILITY_IMPERATIVE.search(latest)
    )


_MISSION_MUTATION_REQUIRED = re.compile(
    r"\b(?:accepted|applied|real|actual)\s+mutation\b|"
    r"\b(?:implement|fix|repair)\b.{0,120}\b(?:code|source|repository|defect)\b",
    re.I | re.S,
)
_MISSION_TEST_REQUIRED = re.compile(
    r"\b(?:focused|passing|proving|regression)\s+tests?\b|"
    r"\b(?:run|execute)\b.{0,60}\btests?\b",
    re.I | re.S,
)
_DISCRIMINATING_TEST_INSPECTION_REQUIRED = re.compile(
    r"\btests?\b.{0,120}\b(?:failed? before|directly discriminates?)\b|"
    r"\b(?:failed? before|directly discriminates?)\b.{0,120}\btests?\b",
    re.I | re.S,
)


def _mission_completion_requirements(
        messages: Sequence[Dict[str, str]]) -> frozenset[str]:
    """Compile explicit write-mission terminals from the latest human turn."""
    latest = next((str(row.get("content") or "") for row in reversed(messages)
                   if row.get("role") == "user"), "")
    required: set[str] = set()
    if _MISSION_MUTATION_REQUIRED.search(latest):
        required.add("accepted mutation")
    if _MISSION_TEST_REQUIRED.search(latest):
        required.add("focused passing test")
    return frozenset(required)


LOCAL_SHAPES = {
    "observation.expand": '{"id": <string>}   optional: query, start, end',
    "debug.diagnose": '{"id": <string>, "command": <string>}   optional: exit_code',
    "reverse.identify": '{"path": <string>}',
    "forensics.snapshot": '{}   optional: note',
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
    menu = session_menu(registry, write=write)
    rows = []
    for name in (names or prompt_menu_names(registry, write=write)):
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
        "\nRepository path rules: paths are relative to the repository shown above. "
        "Use \".\" (or omit path) for the repository root; use fs.search to "
        "discover a filename before fs.read or receipt.read. Receipt paths "
        "must start with receipts/ or come from .hcli state. For a recent "
        "receipt, list path receipts with glob *.json and read one returned "
        "path. For the current objective and recent evidence, campaign.state "
        "is the first read and returns receipt paths. Never use \"/\" or "
        "invent a path such as latest.\n"
        "For live Hawking daemon/runtime state, use processes.summary or "
        "processes.list; do not guess a shell command or PID.\n"
        "tools.catalog focus is a capability keyword or short phrase; to "
        "inspect the broad HCLI/AgentOS surface, use that phrase directly. "
        "observation.expand can only retrieve an exact paste_* id returned by "
        "an earlier observation; it is not a conceptual-state lookup and its "
        "id must never be invented.\n"
        "For an engineering objective, search distinctive source literals "
        "from the objective separately (route fragments, function names, "
        "stream/session identifiers), then read the returned file and line "
        "window. Do not begin with a giant generic engine file or run a whole "
        "test directory; use one focused test file/node as the discriminator.\n"
        "\nMeasured: asked what fields a dataclass in this repo has, a body reached "
        "for web.search. The file was on local disk. Repository questions are "
        "answered from the repository.\n"
        "\nIf files from this project were already included above, they are a "
        "STARTING POINT, not the whole repository -- use fs.search when the answer "
        "needs code you were not shown.\n"
        "\nThe result comes back as the next message and then you answer normally. "
        f"You may use at most {MAX_CALLS} tool rounds per question; one round may "
        "contain a bounded batch of calls. If you already know "
        "the answer, just answer -- do not call a tool to confirm something you "
        "are sure of. For an unlisted signature, call tools.catalog first. Never "
        "invent a tool result: if a call fails, say so. Some compatible bodies "
        "use the compact form {\"op\":\"read\",\"path\":\"...\"}; the "
        "harness maps only these bounded aliases to the matching canonical tools."
    )


def _repair(raw: str) -> Optional[Dict[str, Any]]:
    """Decode an object, tolerating one stray closing brace.

    sealed-3.14 emitted `{"function": "fs.list", "arguments": {"path": "hcli"}}}`
    -- correct except for a third brace. Refusing that is refusing a decision
    the body actually made.
    """
    # Regex arguments are often serialized with JSON-illegal escapes such as
    # ``".*\.json"``. The intended string is unambiguous: JSON only assigns
    # meaning to \" \\ / b f n r t and uXXXX. Preserve every other backslash
    # literally so a correctly chosen tool is not lost before schema checking.
    escaped = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    variants = (raw, escaped)
    for base in variants:
        candidates = (base, base.rstrip().rstrip("}"), base.rstrip() + "}")
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _call_from_object(parsed: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Decode one already-parsed action object."""
    compact = parsed.get("op")
    if isinstance(compact, str):
        name = _compact_tool_name(compact)
        if name:
            arguments = (parsed.get("arguments") or parsed.get("parameters")
                         or parsed.get("args"))
            if not isinstance(arguments, dict):
                arguments = {k: v for k, v in parsed.items()
                             if k not in ("op", "arguments", "parameters")}
            return name, arguments
    name = None
    for key in ("tool", "function", "name", "action", "operation"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            if key in {"action", "operation"}:
                name = _compact_tool_name(name) or name
            break
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            inner = value
            name = inner["name"].strip()
            args = inner.get("arguments") or inner.get("args")
            return name, args if isinstance(args, dict) else {}
    if not name:
        return None
    arguments = (parsed.get("arguments") or parsed.get("parameters")
                 or parsed.get("args"))
    if not isinstance(arguments, dict):
        arguments = {k: v for k, v in parsed.items()
                     if k not in ("tool", "function", "name", "action", "operation",
                                  "arguments", "parameters", "args")}
    return name, arguments


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
    # Some native chat templates return their terminal token as visible text.
    # Strip only the explicit tokenizer markers before JSON decoding.  This is
    # especially important when an otherwise-valid action contains source text
    # such as an f-string: the brace-regex fallback cannot safely understand
    # braces inside JSON strings, whereas json.loads handles them exactly once
    # the trailing protocol token is gone.
    for marker in _TOOL_PROTOCOL_MARKERS:
        text = text.replace(marker, "")
    lowered = text
    xml = _XML_FUNCTION.search(text)
    if xml:
        return xml.group(1).strip(), {
            key.strip(): value.strip()
            for key, value in _XML_PARAM.findall(xml.group(2))}
    narrative = _NARRATIVE_OPTIONAL_ACTION.match(text)
    if narrative and narrative.group(1) in _NARRATIVE_OPTIONAL_TOOLS:
        return narrative.group(1), {}
    if not any(key in lowered for key in (
            '"tool"', '"function"', '"name"', '"action"', '"operation"', '"op"')):
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
        call = _call_from_object(parsed)
        if call is not None:
            return call
    return None


def parse_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Every tool call in the reply, not just the first.

    SCAR: sealed-3.14 routinely batches its whole plan into ONE reply --
    measured, three <tool_call> blocks in a single completion
    (.hcli/selfdev/evidence/cycle-0008.txt). parse_call (singular) returns only
    the first match, so the other two were silently dropped: a body that names
    three reads in one turn got credit for one, spent its remaining turn
    budget re-emitting reads it believed had already run, and never reached an
    edit. Bodies may also batch several JSON actions in prose or code fences.
    Consume each complete action object in order, while ``run_with_tools``
    retains the existing per-turn and total tool budgets.
    """
    matches = list(_XML_FUNCTION.finditer(text or ""))
    if matches:
        return [(m.group(1).strip(),
                 {k.strip(): v.strip() for k, v in _XML_PARAM.findall(m.group(2))})
                for m in matches]
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for raw in _OBJECT.findall(text or ""):
        parsed = _repair(raw)
        if parsed is None:
            continue
        call = _call_from_object(parsed)
        if call is not None:
            calls.append(call)
    if calls:
        return calls
    single = parse_call(text)
    return [single] if single is not None else []


_RECEIPT_HEADLINE_KEYS = frozenset({
    "status", "verdict", "decision", "result", "pass", "qualified",
    "passed", "accepted", "allowed", "measurement", "summary", "scar",
    "law", "id", "identifier", "hash", "sha256", "timestamp",
    "created_at", "updated_at", "written_at", "produced_by", "producer",
    "refused", "reason", "failure_mechanism", "claim_refuted",
    "reopen_condition", "claim_boundary", "role", "scope",
})

_RECEIPT_COMPOUND_KEYS = frozenset({
    "result", "measurement", "summary", "scar", "law", "decision",
})


def _important_key(key: Any) -> bool:
    lowered = str(key).lower()
    return (
        lowered in _RECEIPT_HEADLINE_KEYS
        or lowered.endswith(("_id", "_hash", "_sha256", "_timestamp"))
        or lowered.endswith(("_at", "_time"))
    )


def _bounded_evidence_value(value: Any, *, max_chars: int = 320) -> Any:
    """Keep an exact short scalar or a truthful descriptor of a large one."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if len(text) <= max_chars:
        return text
    return {
        "preview": text[:max_chars].rstrip() + "...",
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
    }


def _receipt_fields(value: Any, *, max_depth: int = 4) -> Dict[str, Any]:
    """Project exact decision/measurement fields out of a receipt wrapper."""
    if not isinstance(value, dict):
        return {}
    document = value.get("document")
    if not isinstance(document, dict):
        return {}
    selected: Dict[str, Any] = {}

    def _walk(node: Any, prefix: str, depth: int,
              compound: bool = False) -> None:
        if depth > max_depth or len(selected) >= 48:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                important = _important_key(key)
                if (important or compound) and not isinstance(child, (dict, list)):
                    selected[path] = _bounded_evidence_value(child)
                _walk(child, path, depth + 1,
                      compound or str(key).lower() in _RECEIPT_COMPOUND_KEYS)
        elif isinstance(node, list):
            for index, child in enumerate(node[:8]):
                _walk(child, f"{prefix}[{index}]", depth + 1, compound)

    _walk(document, "", 0)
    return selected


def _receipt_headline(value: Any, *, max_depth: int = 4,
                      max_chars: int = 2400) -> str:
    """Put exact decision fields before a potentially long receipt body.

    Receipt payloads commonly carry provenance and candidate detail before the
    final status. The chat observation is intentionally bounded, so a raw JSON
    head can omit the very verdict the caller requested. This deterministic
    projection copies existing scalar evidence only; it neither interprets nor
    replaces the complete document retained below.
    """
    selected = _receipt_fields(value, max_depth=max_depth)
    if not selected:
        return ""
    encoded = json.dumps(selected, default=str)
    if len(encoded) > max_chars:
        encoded = encoded[:max_chars] + "... [headline truncated]"
    return "receipt decision fields (exact projection):\n" + encoded


def _value_outline(value: Any, *, depth: int = 0) -> Any:
    """Describe shape without copying payloads into the prompt."""
    if isinstance(value, dict):
        keys = list(value)
        out: Dict[str, Any] = {
            "type": "object", "fields": len(keys), "keys": keys[:24],
        }
        if depth == 0:
            out["children"] = {
                str(key): _value_outline(value[key], depth=depth + 1)
                for key in keys[:12]
            }
        return out
    if isinstance(value, list):
        return {"type": "array", "items": len(value)}
    if isinstance(value, str):
        return {"type": "string", "chars": len(value),
                "lines": value.count("\n") + (1 if value else 0)}
    return {"type": type(value).__name__}


def _array_max(value: Any, *, depth: int = 0) -> int:
    if depth > 6:
        return 0
    if isinstance(value, list):
        return max([len(value), *(_array_max(v, depth=depth + 1)
                                  for v in value[:32])])
    if isinstance(value, dict):
        return max([0, *(_array_max(v, depth=depth + 1)
                         for v in list(value.values())[:64])])
    return 0


def _repeated_string_run(body: str) -> Optional[Dict[str, Any]]:
    """Detect a conspicuous contiguous repeated string in a bounded sample."""
    match = re.search(r"(.{8,64}?)\1{5,}", body[:20000], re.S)
    if match is None:
        return None
    unit = match.group(1)
    return {
        "unit_chars": len(unit),
        "observed_run_chars": len(match.group(0)),
        "observed_repetitions": len(match.group(0)) // len(unit),
        "unit_sha256": hashlib.sha256(
            unit.encode("utf-8", "replace")).hexdigest(),
    }


_SIGNAL_LINE = re.compile(
    r"\b(?:status|verdict|result|passed?|accepted|qualified|measurement|"
    r"summary|error|exception|traceback|failed?|sha256|timestamp)\b",
    re.I,
)

_SOURCE_BEHAVIOR_LINE = re.compile(
    r"^\s*(?:async\s+def|def|class|return|raise|if|elif|for|while|with|try:|"
    r"except)\b|\b(?:exec|Popen|subprocess|os\.environ|PYTHONPATH|symlink|"
    r"copytree|write_text|chmod)\b",
    re.I,
)

_SOURCE_EFFECT_LINE = re.compile(
    r"\bPYTHONPATH\b|\bexec\s|\bPopen\b|\bsubprocess\b|"
    r"\bos\.environ\b|"
    r"^\s*def\s+_[A-Za-z0-9_]+",
    re.I,
)
_TEST_EXPECTATION_LINE = re.compile(r"^\s*(?:assert\b|self\.assert)", re.I)
_SELECTED_OBSERVATION_LINE = re.compile(
    r'"line":\s*(\d+),\s*\n\s*"text":\s*"((?:\\.|[^"\\])*)"'
)


def _selected_observation_lines(row: Dict[str, Any], *, limit: int = 8
                                ) -> List[Tuple[int, str]]:
    """Recover exact selected source lines from a compacted trace row.

    The trace's observation is intentionally a bounded structural projection,
    not the full file. It can itself end with a truncation notice, so parsing
    the whole projection as JSON is not reliable. Each selected-line record is
    nevertheless complete JSON. Decode only those complete scalar records and
    replay them when context eviction would otherwise leave a corrective
    mutation turn with paths but no anchors.
    """
    body = str(row.get("observation") or "")
    selected: List[Tuple[int, str]] = []
    for match in _SELECTED_OBSERVATION_LINE.finditer(body):
        try:
            line = int(match.group(1))
            value = json.loads('"' + match.group(2) + '"')
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        selected.append((line, str(value)))
        if len(selected) >= limit:
            break
    return selected


def _signal_lines(body: str, *, limit: int = 8) -> List[str]:
    """Select useful lines from the whole result, not merely its prefix."""
    selected: List[str] = []
    seen = set()
    for raw in body.splitlines():
        line = " ".join(raw.split())
        if not line or not _SIGNAL_LINE.search(line):
            continue
        clipped = line if len(line) <= 360 else line[:357] + "..."
        if clipped in seen:
            continue
        seen.add(clipped)
        selected.append(clipped)
        if len(selected) >= limit:
            break
    return selected


def _bounded_line(line: Any, *, limit: int = 320) -> str:
    text = " ".join(str(line or "").split())
    if _repeated_string_run(text) is not None:
        return ("[repeated string omitted; exact content is retained in the "
                "full-result handle]")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _content_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Useful lines across a large file body, with exact line coordinates."""
    if not isinstance(value, dict) or not isinstance(value.get("content"), str):
        return None
    content = value["content"]
    lines = content.splitlines()
    chosen: List[Dict[str, Any]] = []
    seen = set()

    first_line = int(value.get("start_line") or 1)

    def _add(index: int, raw: str) -> None:
        text = _bounded_line(raw)
        if not text or text in seen or len(chosen) >= 16:
            return
        seen.add(text)
        chosen.append({"line": first_line + index - 1, "text": text})

    for index, raw in enumerate(lines, 1):
        if raw.strip():
            _add(index, raw)
        if len(chosen) >= 4:
            break
    # A source window is read to understand behavior. Preserve high-signal
    # execution/environment lines across the whole window first, then broader
    # executable anchors, before spending the sample on prose.
    # The former comment-first selection hid the `_script`/`exec` lines that
    # distinguished an installed shim from a cwd-shadowed import.
    for index, raw in enumerate(lines, 1):
        if _SOURCE_EFFECT_LINE.search(raw):
            _add(index, raw)
    for index, raw in enumerate(lines, 1):
        if _TEST_EXPECTATION_LINE.search(raw):
            _add(index, raw)
    for index, raw in enumerate(lines, 1):
        if _SOURCE_BEHAVIOR_LINE.search(raw):
            _add(index, raw)
    for index, raw in enumerate(lines, 1):
        stripped = raw.lstrip()
        if (stripped.startswith(("class ", "def ", "ROLE:", "STATUS:"))
                or _SIGNAL_LINE.search(raw)):
            _add(index, raw)
    return {
        "path": value.get("path"),
        "sha256": value.get("sha256"),
        "total_bytes": value.get("bytes"),
        "shown_bytes": value.get("shown_bytes"),
        "truncated": value.get("truncated"),
        "start_line": value.get("start_line", 1),
        "selected_lines": chosen,
    }


def _listing_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Preserve bounded directory entries when a complete listing is large."""
    if not isinstance(value, dict):
        return None
    files = value.get("files")
    directories = value.get("directories")
    if not isinstance(files, list) or not isinstance(directories, list):
        return None
    def _row(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        return {
            "path": item.get("path"),
            "filename": item.get("filename"),
            "type": item.get("type") or item.get("kind"),
            "size": item.get("size", item.get("bytes")),
        }

    file_sample = files[:8]
    if len(files) > 12:
        file_sample = [*file_sample, *files[-4:]]
    return {
        "root": value.get("root"),
        "glob": value.get("glob"),
        "recursive": bool(value.get("directories_seen", 0) > 1),
        "file_count_returned": len(files),
        "directory_count_returned": len(directories),
        "truncated": bool(value.get("truncated")),
        # Directory rows contain four mostly redundant fields. Keep their
        # repository-relative paths only so the file sample and narrowing
        # instruction remain inside the observation budget. Measured blind:
        # listing receipts spent the first 3,000 rendered characters on 12
        # directory objects and cut off every usable receipt filename.
        "directories": [
            {"path": str(item.get("path") or item.get("filename") or "")}
            if isinstance(item, dict) else {"path": str(item)}
            for item in directories[:12]
        ],
        "file_sample": [_row(item) for item in file_sample],
        "narrowing_hint": (
            "Use fs.list again with a named subdirectory and/or glob when the "
            "wanted entry is not in this bounded sample. Use fs.read only "
            "after choosing a file."
        ),
    }


def _search_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Return diverse bounded matches instead of one file's repeated hits."""
    if not isinstance(value, dict) or not isinstance(value.get("matches"), list):
        return None
    selected: List[Dict[str, Any]] = []
    seen_paths = set()
    for row in value["matches"]:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        selected.append({
            "path": path,
            "line": row.get("line"),
            "text": _bounded_line(row.get("text")),
        })
        if len(selected) >= 12:
            break
    return {
        "root": value.get("root"),
        "pattern": value.get("pattern"),
        "matches_returned": len(value["matches"]),
        "unique_paths_returned": len({
            str(row.get("path") or "") for row in value["matches"]
            if isinstance(row, dict)
        }),
        "truncated": bool(value.get("truncated")),
        "diverse_match_sample": selected,
        "narrowing_hint": (
            "If one file dominates, narrow with path/glob or max_per_file; do "
            "not treat this bounded sample as a complete negative search."
        ),
    }


def _catalog_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Keep the capability names and broad families in a large catalog result."""
    if not isinstance(value, dict) or not isinstance(value.get("names"), list):
        return None
    names = [str(name) for name in value["names"][:32]]
    return {
        "focus": value.get("focus"),
        "shown": value.get("shown"),
        "match_count": value.get("match_count"),
        "truncated": bool(value.get("truncated")),
        "names": names,
        "broad_categories": sorted({name.split(".", 1)[0] for name in names}),
    }


def _command_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Preserve exact command outcome and a bounded, diverse stdout sample."""
    if not isinstance(value, dict) or not {
            "cwd", "returncode", "stdout"}.issubset(value):
        return None
    stdout = str(value.get("stdout") or "")
    lines = stdout.splitlines()
    selected = lines[:8]
    if len(lines) > 12:
        selected.extend(lines[-4:])
    return {
        "cwd": value.get("cwd"),
        "returncode": value.get("returncode"),
        "stdout_lines": len(lines),
        "stdout_sample": [_bounded_line(line) for line in selected],
        "stderr": _bounded_line(value.get("stderr")),
    }


def _runtime_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Keep ownership, model identity and duplicate evidence from process data."""
    if not isinstance(value, dict) or not isinstance(value.get("processes"), list):
        return None
    rows = []
    for process in value["processes"][:16]:
        if not isinstance(process, dict):
            continue
        rows.append({
            "pid": process.get("pid"),
            "ppid": process.get("ppid"),
            "role": process.get("role"),
            "class": process.get("class"),
            "body": process.get("body"),
            "rss_gib": process.get("rss_gib"),
            "command": _bounded_line(process.get("command"), limit=220),
        })
    return {
        "count": value.get("count", value.get("n_processes")),
        "by_class": value.get("by_class"),
        "roles": value.get("roles"),
        "total_rss_bytes": value.get("total_rss_bytes"),
        "processes": rows,
    }


def _campaign_selection(value: Any) -> Optional[Dict[str, Any]]:
    """Retain the current checkpoint and exact evidence paths in a large state."""
    if not isinstance(value, dict) or not (
            "continuation" in value and "recent_evidence" in value):
        return None
    continuation = value.get("continuation")
    if isinstance(continuation, dict):
        continuation = {
            key: _bounded_evidence_value(continuation.get(key), max_chars=480)
            for key in ("objective", "active_specimen", "active_workunit",
                        "next_action", "written_at")
            if continuation.get(key) is not None
        }
    parent = value.get("authoritative_parent")
    if isinstance(parent, dict):
        parent = {
            key: _bounded_evidence_value(parent.get(key), max_chars=(900
                if key == "latest_section_excerpt" else 520))
            for key in ("path", "mtime", "primary_objective",
                        "latest_section", "latest_section_excerpt")
            if parent.get(key) is not None
        }
    return {
        # A current evidence path is the next executable decision after the
        # parent is known. Put these first: the earlier ordering let the long
        # parent excerpt consume the rendered budget and hid all six paths.
        "recent_evidence": value.get("recent_evidence"),
        "authoritative_parent": parent,
        "continuation_older_than_parent": value.get(
            "continuation_older_than_parent"),
        "authority_note": value.get("authority_note"),
        "continuation": continuation,
        "bottleneck": value.get("bottleneck"),
        "ledger": value.get("ledger"),
    }


def _compacted_observation(value: Any, body: str, *, name: str,
                            ref: Any = None) -> str:
    """A bounded structural observation with a lossless retrieval route.

    Large receipts, logs, arrays, stack traces and generated payloads used to
    contribute their first 3,000 characters. That is bounded but not useful:
    one repeated field can evict the verdict. This projection scans the whole
    value for exact authoritative fields, describes its structure and anomaly
    pressure, then points at the complete content-addressed result.
    """
    encoded_bytes = body.encode("utf-8", "replace")
    lines = body.splitlines()
    duplicate_lines = max(0, len(lines) - len(set(lines)))
    control_chars = sum(
        ord(ch) < 32 and ch not in "\n\r\t" for ch in body[:20000]
    )
    record: Dict[str, Any] = {
        "compacted": True,
        "tool": name,
        "original": {
            "chars": len(body),
            "bytes": len(encoded_bytes),
            "lines": len(lines),
            "sha256": hashlib.sha256(encoded_bytes).hexdigest(),
        },
    }
    fields = _receipt_fields(value)
    if fields:
        record["authoritative_fields"] = fields
    listing = _listing_selection(value) if name in {"fs.list", "filesystem.list"} else None
    content = _content_selection(value) if name in {"fs.read", "filesystem.read"} else None
    search = _search_selection(value) if name in {"fs.search", "filesystem.search"} else None
    catalog = _catalog_selection(value) if name == "tools.catalog" else None
    command = _command_selection(value)
    runtime = _runtime_selection(value) if name in {
        "processes.summary", "processes.list"} else None
    campaign = _campaign_selection(value) if name == "campaign.state" else None
    if listing is not None:
        record["discovery"] = listing
    if content is not None:
        record["content_selection"] = content
    if search is not None:
        record["search_selection"] = search
    if catalog is not None:
        record["catalog_selection"] = catalog
    if command is not None:
        record["command_selection"] = command
    if runtime is not None:
        record["runtime_selection"] = runtime
    if campaign is not None:
        record["campaign_selection"] = campaign
    record["structure"] = _value_outline(value)
    record["pathology"] = {
        "duplicate_lines": duplicate_lines,
        "largest_line_chars": max((len(line) for line in lines), default=0),
        "largest_array_items": _array_max(value),
        "binary_looking": bool(control_chars),
    }
    repeated = _repeated_string_run(body)
    if repeated is not None:
        record["pathology"]["repeated_string_run"] = repeated
    signals = _signal_lines(body)
    if signals:
        record["selected_signal_lines"] = signals
    if ref is not None:
        record["full_result"] = {
            "id": ref.id,
            "content_sha256": ref.sha256,
            "retrieve_by_query": {
                "tool": "observation.expand",
                "arguments": {"id": ref.id, "query": "status"},
            },
            "retrieve_by_lines": {
                "tool": "observation.expand",
                "arguments": {"id": ref.id, "start": 1, "end": 20},
            },
        }
    else:
        record["full_result"] = {
            "preserved": False,
            "reason": "no observation store was available",
        }
    rendered = json.dumps(record, default=str, indent=2)
    # The outline and exact evidence fields are independently bounded above.
    # This last guard protects against unexpectedly exotic mapping keys.
    if len(rendered) > RESULT_CHARS:
        rendered = rendered[:RESULT_CHARS] + "\n... [structured summary truncated]"
    pointer = (
        f"\ncomplete result is kept at {ref.id} -- use observation.expand."
        if ref is not None else
        "\ncomplete result could not be retained; observation was truncated."
    )
    if name in {"receipt.read", "receipt.inspect", "benchmark.inspect"}:
        pointer = "\ncomplete receipt follows at the retained result." + pointer
        return ("receipt decision fields (exact projection) are under "
                "authoritative_fields below.\nobservation structural summary:\n"
                + rendered + pointer)
    return "observation structural summary:\n" + rendered + pointer


def _render(result: Any, name: str, registry: Any = None,
            cache: Any = None) -> str:
    ok = bool(getattr(result, "ok", False))
    if not ok:
        error = getattr(result, "error", None) or "failed"
        error_text = str(error)
        if "outside the AgentOS read roots" in error_text:
            return (f"{name} FAILED: {error_text}\n"
                    "Use a repository-relative path. For the repository root, "
                    "fs.list accepts path \".\" or an omitted path; do not use "
                    "\"/\" or invent an absolute path.\n"
                    "Send a corrected action, or answer without it.")
        if "receipt path must be under receipts" in error_text:
            return (f"{name} FAILED: {error_text}\n"
                    "Use the repository-relative path returned by fs.search, "
                    "starting with receipts/ (or a returned .hcli state path); "
                    "do not invent latest.\n"
                    "Send a corrected action, or answer without it.")
        if name == "receipt.read" and "FileNotFoundError" in error_text:
            return (f"{name} FAILED: {error_text}\n"
                    "There is no magic latest receipt. First use campaign.state "
                    "for current objective and recent evidence, or fs.list with "
                    "path receipts and glob *.json; then read a returned path.\n"
                    "Send a corrected action, or answer without it.")
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
        body = (value if isinstance(value, str)
                else json.dumps(value, default=str, indent=2))
    except Exception:
        body = str(value)
    if name in {"receipt.read", "receipt.inspect", "benchmark.inspect"}:
        headline = _receipt_headline(value)
        if headline and len(body) <= RESULT_CHARS:
            body = f"{headline}\ncomplete receipt follows:\n{body}"
    if len(body) > RESULT_CHARS:
        # DISK FIRST, THEN A POINTER. Truncating at 3000 characters and dropping
        # the rest destroyed the only copy: a test log or a page the model needed
        # ten lines further down was simply gone, and the model could not ask for
        # it because nothing had kept it. PasteCache is a content-addressed store
        # that was already built for exactly this -- store(), slice(), search(),
        # get() -- with store() called from nothing but its own tests. So the
        # detail goes to disk and the context gets the head plus a handle it can
        # expand with observation.expand.
        ref = None
        # Expanding an existing pointer must never create a pointer to the
        # expansion wrapper. Measured blind: four rounds became A -> B -> C ->
        # D, each containing the previous escaped JSON, without exposing one
        # additional source line. Continue on the original id/range instead.
        if cache is not None and name != "observation.expand":
            try:
                ref = cache.store(body)
            except Exception:
                ref = None
        if ref is not None:
            body = _compacted_observation(value, body, name=name, ref=ref)
        elif name == "observation.expand" and isinstance(value, dict):
            head = body[:RESULT_CHARS]
            same_id = value.get("id")
            end = value.get("end")
            continuation = (
                f'observation.expand {{"id": "{same_id}", "start": '
                f'{int(end) + 1}, "end": {int(end) + 20}}}'
                if same_id and isinstance(end, int) else
                f'observation.expand with the same id {same_id!r} and a narrow query'
            )
            body = (head + f"\n... [expanded result clipped at {RESULT_CHARS} "
                    f"chars; continue the original observation with {continuation}]")
        else:
            body = _compacted_observation(value, body, name=name, ref=None)
    warning = ""
    if (name in {"fs.search", "filesystem.search"}
            and isinstance(value, dict) and value.get("truncated")):
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


#: Source-ish files, for the narrowed retry below.  This remains the fallback
#: for symbol searches; filename searches choose their own suffix below.
SOURCE_GLOB = "*.py"


def _search_fallback_glob(arguments: Dict[str, Any]) -> str:
    """Choose a safe narrowed glob from the query's explicit file hint.

    A fixed ``*.py`` retry made a truncated search for a JSON receipt look like
    a complete negative result.  Narrowing is deterministic mechanics, so use
    a suffix already present in the requested pattern/path and only fall back
    to source for symbol-like searches.  This never broadens the authority or
    invents a path.
    """
    needle = " ".join(
        str(arguments.get(key) or "") for key in ("pattern", "path")
    ).lower()
    for suffix in (".json", ".md", ".txt", ".toml", ".yaml", ".yml"):
        if suffix in needle:
            return f"*{suffix}"
    return SOURCE_GLOB


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


_SEARCH_PHRASE_STOPWORDS = frozenset({
    "about", "across", "after", "before", "between", "does", "from",
    "have", "into", "preservation", "state", "that", "their", "there",
    "these", "this", "through", "with", "without",
})


def _search_phrase_tokens(arguments: Dict[str, Any]) -> List[str]:
    pattern = str(arguments.get("pattern") or "")
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", pattern)
    return [word for word in words
            if word.lower() not in _SEARCH_PHRASE_STOPWORDS][:4]


def _search_has_no_matches(result: Any) -> bool:
    value = getattr(result, "value", None)
    return (bool(getattr(result, "ok", False))
            and isinstance(value, dict)
            and not value.get("matches")
            and not value.get("truncated"))


def _source_search_score(result: Any) -> int:
    """Prefer active implementation owners over generated/historical hits."""
    value = getattr(result, "value", None)
    matches = value.get("matches") if isinstance(value, dict) else None
    if not isinstance(matches, list) or not matches:
        return -10_000
    best = -10_000
    for match in matches:
        path = str((match or {}).get("path") or "").lower()
        score = 0
        if path.endswith((".py", ".rs", ".go", ".ts", ".tsx", ".js")):
            score += 30
        if path.startswith(("hcli/", "src/", "crates/", "tools/", "tests/")):
            score += 50
        if any(part in path for part in (
                "/receipts/", "receipts/", "/.hcli/", ".hcli/",
                "research/hawking-experiments/", "workspace/campaign/")):
            score -= 100
        best = max(best, score)
    return best + min(len(matches), 20)


def _filename_listing_request(arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate a pure extension regex/glob from content search to file listing.

    Small bodies commonly send ``fs.search(path=..., pattern=.*\\.json)`` when
    they mean "which JSON files are here?".  No file content could satisfy
    that expression reliably because fs.search is literal.  A pattern that
    expresses *only* a filename suffix has one deterministic filesystem
    meaning, so route that narrow shape to the existing read-only fs.list door.
    """
    pattern = str(arguments.get("pattern") or "").strip()
    match = (
        re.fullmatch(r"(?:\.\*)?\\\.([A-Za-z0-9_-]+)\$?", pattern)
        or re.fullmatch(r"\*\.([A-Za-z0-9_-]+)", pattern)
    )
    if match is None:
        return None
    out: Dict[str, Any] = {
        "path": arguments.get("path") or arguments.get("root") or ".",
        "glob": f"*.{match.group(1)}",
        "recursive": bool(arguments.get("recursive", False)),
    }
    if arguments.get("max_results") is not None:
        out["max_results"] = arguments["max_results"]
    return out


def escalate_search(registry: Any, arguments: Dict[str, Any],
                    result: Any = None) -> Optional[Any]:
    """Recover bounded source search failures mechanically.

    A truncated walk gets a source/file-type glob. A complete zero-result query
    containing a prose phrase gets up to four single-token source searches and
    returns the first observed match. This does not invent synonyms or semantic
    conclusions; it makes the exact-match tool tolerate the natural query shape
    small local bodies repeatedly emit.
    """
    current = result
    if _search_needs_escalation(current) and not arguments.get("glob"):
        narrowed_glob = _search_fallback_glob(arguments)
        narrowed = {**arguments, "glob": narrowed_glob}
        try:
            candidate = registry.invoke("fs.search", narrowed)
        except Exception:
            return None
        if not _search_has_no_matches(candidate):
            return candidate
        current = candidate
    if _search_has_no_matches(current):
        tokens = _search_phrase_tokens(arguments)
        if len(str(arguments.get("pattern") or "").split()) < 2 or not tokens:
            return None
        candidates: List[Any] = []
        for token in tokens:
            narrowed = {**arguments, "pattern": token}
            narrowed.setdefault("glob", _search_fallback_glob(arguments))
            narrowed["max_results"] = min(
                80, max(1, int(arguments.get("max_results") or 80)))
            narrowed["max_per_file"] = 2
            try:
                candidate = registry.invoke("fs.search", narrowed)
            except Exception:
                continue
            if not _search_has_no_matches(candidate):
                candidates.append(candidate)
        if candidates:
            return max(candidates, key=_source_search_score)
    return None


def _interactive_test_scope_refusal(arguments: Dict[str, Any]) -> Optional[str]:
    """Keep a browser mission's discriminator focused and bounded.

    The full suite remains available outside the chat loop. Here, an empty path
    or a test-directory root turned one diagnostic step into a 150-second run
    whose unrelated failures the body misread as evidence about its hypothesis.
    """
    paths = arguments.get("paths")
    if not isinstance(paths, list) or not paths:
        return ("interactive tests.run requires at least one focused test file "
                "or node id; discover it with fs.search/tests.list first")
    broad = {".", "hcli", "hcli/tests", "tests", "tools"}
    normalized = [str(path or "").strip().rstrip("/") for path in paths]
    if any(path in broad or path.endswith("/tests") for path in normalized):
        return ("broad test directories are not an interactive discriminator; "
                "name one focused test file or pytest node id, then widen only "
                "after the focused result passes")
    return None


_GENERATION_MARKER_SUFFIX = re.compile(
    r"(?:<\|(?:im_end|endoftext|eot_id)\|>\s*)+$", re.I,
)


def _clean_final_text(value: Any) -> str:
    """Remove provider control tokens that are not user-authored content."""
    clean = _GENERATION_MARKER_SUFFIX.sub("", str(value or "").strip()).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", clean, re.I | re.S)
    return fenced.group(1).strip() if fenced is not None else clean


def _balanced_object(text: str, start: int) -> Optional[str]:
    """Return one brace-balanced JSON candidate, respecting quoted braces."""
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _requested_json_shape(messages: Sequence[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Extract a user-supplied exact JSON skeleton without interpreting values."""
    latest = next((str(row.get("content") or "") for row in reversed(messages)
                   if row.get("role") == "user"), "")
    marker = re.search(
        r"(?:return|reply with|respond with).{0,120}?json object"
        r".{0,100}?exactly (?:this|the following) structure\s*:",
        latest, re.I | re.S,
    )
    if marker is None:
        return None
    start = latest.find("{", marker.end())
    if start < 0:
        return None
    candidate = _balanced_object(latest, start)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_structure_issues(expected: Any, actual: Any,
                           path: str = "$") -> List[str]:
    """Compare only keys/container shape; example scalar values are not answers."""
    issues: List[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path} must be an object"]
        wanted, got = set(expected), set(actual)
        issues.extend(f"missing {path}.{key}" for key in sorted(wanted - got))
        issues.extend(f"unexpected {path}.{key}" for key in sorted(got - wanted))
        for key in sorted(wanted & got):
            issues.extend(_json_structure_issues(
                expected[key], actual[key], f"{path}.{key}"))
    elif isinstance(expected, list) and not isinstance(actual, list):
        issues.append(f"{path} must be an array")
    return issues


def _validate_requested_json(text: Any, expected: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Validate an exact user-requested envelope after stripping provider markers."""
    clean = _clean_final_text(text)
    try:
        parsed = json.loads(clean)
    except (TypeError, ValueError) as exc:
        return clean, [f"response is not one JSON object: {exc}"]
    if not isinstance(parsed, dict):
        return clean, ["$ must be an object"]
    return clean, _json_structure_issues(expected, parsed)


_EVIDENCE_OBLIGATION_RULES: Tuple[
    Tuple[str, re.Pattern[str], frozenset[str]], ...
] = (
    ("repository", re.compile(
        r"\b(?:repository root|repo root|git/project|project repository|"
        r"git status|git evidence|git root|"
        r"actually (?:a )?git|verify.{0,60}repository|"
        r"discover.{0,20}(?:the )?repository)\b", re.I | re.S),
     # A file read proves reachability but not the user's separate requirement
     # that the root really is a Git/project repository. Keep that obligation
     # open until the repository authority itself answers.
     frozenset({"git.status"})),
    ("filesystem", re.compile(
        r"\b(?:repository/filesystem|filesystem evidence|"
        r"evidence from.{0,80}(?:repository|filesystem))\b", re.I | re.S),
     frozenset({"fs.list", "fs.read", "fs.search"})),
    ("source inspection", re.compile(
        r"\b(?:identify the general ownership boundary|"
        r"implement the smallest durable repair|"
        r"reproduce.{0,100}implement)\b", re.I | re.S),
     # Search locates a candidate owner; it does not establish what that owner
     # does. Require the source itself to be read before a write mission can
     # leave this evidence stage.
     frozenset({"fs.read"})),
    ("tool discovery", re.compile(
        r"\b(?:discover|determine|inspect|report)\b.{0,80}"
        r"\b(?:tools?|capabilities|agentos)\b", re.I | re.S),
     frozenset({"tools.catalog"})),
    ("current objective", re.compile(
        r"\b(?:current|authoritative|parent|sovereign)\b.{0,60}"
        r"\b(?:objective|goal|state)\b", re.I | re.S),
     frozenset({"campaign.state", "fs.read"})),
    ("receipt/evidence artifact", re.compile(
        r"\b(?:read|find|inspect|report|obtain evidence from)\b.{0,100}"
        r"\b(?:receipt|evidence artifact)\b", re.I | re.S),
     frozenset({"receipt.read", "fs.read"})),
    ("campaign/receipt", re.compile(
        r"\b(?:current campaign|campaign or receipts?|campaign/receipt)\b",
        re.I | re.S),
     frozenset({"campaign.state", "receipt.read"})),
    ("live runtime", re.compile(
        r"\b(?:live|daemon|hawkingd|runtime|process(?:es)?)\b", re.I),
     frozenset({"processes.summary", "processes.list"})),
    ("diagnostic/forensic inspection", re.compile(
        r"\b(?:audit/debug/forensics|audit.{0,20}debug.{0,20}forensics|"
        r"reverse-inspection)\b", re.I | re.S),
     frozenset({"audit", "debug.diagnose", "forensics.snapshot",
                "reverse.identify"})),
)


def _evidence_obligations(messages: Sequence[Dict[str, str]]) -> Dict[str, frozenset[str]]:
    """Compile explicit evidence nouns into tool families, never arguments."""
    latest = next((str(row.get("content") or "") for row in reversed(messages)
                   if row.get("role") == "user"), "")
    if not explicit_tool_observation_required(messages):
        return {}
    obligations = {
        label: tools for label, pattern, tools in _EVIDENCE_OBLIGATION_RULES
        if pattern.search(latest)
    }
    if _DISCRIMINATING_TEST_INSPECTION_REQUIRED.search(latest):
        # A source read cannot establish that a focused test discriminates the
        # old behavior. Keep this open until the body reads an actual test file.
        obligations["focused test inspection"] = frozenset({"fs.read"})
    return obligations


def _successful_evidence_tools(trace: Sequence[Dict[str, Any]]) -> set[str]:
    return {
        str(row.get("tool")) for row in trace
        if row.get("dispatched") and row.get("ok") and row.get("tool")
    }


def _accepted_mutation_row(
        trace: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The latest mutation whose transaction actually landed and was accepted."""
    return next((
        row for row in reversed(trace)
        if row.get("tool") == "repo.edit"
        and row.get("dispatched") and row.get("ok")
        and row.get("applied") is True
        and str(row.get("verdict") or "").lower() == "accepted"
    ), None)


def _passing_test_row(
        trace: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The latest independently dispatched focused test that actually passed."""
    return next((
        row for row in reversed(trace)
        if row.get("tool") == "tests.run"
        and row.get("dispatched") and row.get("ok")
        and row.get("verified") is True and row.get("returncode") == 0
    ), None)


def _runtime_tree_preserved(process_rows: Sequence[Dict[str, Any]]) -> bool:
    """True when every reported production process descends from one hawkingd.

    A count of one daemon and one provider is not ownership: either process
    can be independently launched. Follow PPIDs through the classified process
    set so additional models, UIs, or worker layers remain valid only under the
    same model-neutral root. Debug-only development processes are deliberately
    outside this production invariant.
    """
    rows = [row for row in process_rows if isinstance(row, dict)]
    roots = [row for row in rows if row.get("role") == "hawkingd"]
    if len(roots) != 1:
        return False
    root_pid = roots[0].get("pid")
    if not isinstance(root_pid, int) or root_pid <= 1:
        return False
    by_pid = {row.get("pid"): row for row in rows
              if isinstance(row.get("pid"), int)}
    production = [row for row in rows
                  if row.get("class") != "DEBUG_ONLY"]
    if not any(row.get("role") in {"resident-provider", "resident-body"}
               for row in production):
        return False
    for row in production:
        pid = row.get("pid")
        if pid == root_pid:
            continue
        seen = set()
        parent = row.get("ppid")
        while isinstance(parent, int) and parent not in seen:
            if parent == root_pid:
                break
            seen.add(parent)
            ancestor = by_pid.get(parent)
            if ancestor is None:
                return False
            parent = ancestor.get("ppid")
        else:
            return False
    return True


def _pending_mission_completion(
        requirements: frozenset[str],
        trace: Sequence[Dict[str, Any]]) -> frozenset[str]:
    pending = set(requirements)
    if _accepted_mutation_row(trace) is not None:
        pending.discard("accepted mutation")
    if _passing_test_row(trace) is not None:
        pending.discard("focused passing test")
    return frozenset(pending)


def _pending_evidence(obligations: Dict[str, frozenset[str]],
                      trace: Sequence[Dict[str, Any]]) -> Dict[str, frozenset[str]]:
    successful = _successful_evidence_tools(trace)
    pending: Dict[str, frozenset[str]] = {}
    for label, alternatives in obligations.items():
        matched = bool(successful & set(alternatives))
        if label == "focused test inspection" and matched:
            matched = any(
                row.get("dispatched") and row.get("ok")
                and row.get("tool") == "fs.read"
                and (
                    Path(str((row.get("arguments") or {}).get("path") or "")).name
                    .startswith("test_")
                    or Path(str((row.get("arguments") or {}).get("path") or "")).name
                    .endswith("_test.py")
                )
                for row in trace
            )
        if label == "receipt/evidence artifact" and matched:
            # A generic fs.read only satisfies this category when it actually
            # read a receipt/evidence path; reading README is not a receipt.
            matched = any(
                row.get("dispatched") and row.get("ok")
                and row.get("tool") in alternatives
                and (row.get("tool") == "receipt.read"
                     or "receipt" in str((row.get("arguments") or {}).get(
                         "path") or "").lower())
                for row in trace
            )
        if label == "current objective" and matched:
            matched = any(
                row.get("dispatched") and row.get("ok")
                and (row.get("tool") == "campaign.state"
                     or any(word in str((row.get("arguments") or {}).get(
                         "path") or "").lower()
                            for word in ("goal", "objective", "continuation")))
                for row in trace
            )
        if not matched:
            pending[label] = alternatives
    return pending


def _trace_grounding_issues(
        text: str, trace: Sequence[Dict[str, Any]],
        obligations: Dict[str, frozenset[str]],
        mission_requirements: frozenset[str] = frozenset()) -> List[str]:
    """Reject structured claims that mechanically contradict dispatch evidence."""
    try:
        document = json.loads(text)
    except (TypeError, ValueError):
        return []  # syntax/shape validation owns this failure
    if not isinstance(document, dict):
        return []
    issues: List[str] = []
    successful = _successful_evidence_tools(trace)
    pending = _pending_evidence(obligations, trace)
    pending_mission = _pending_mission_completion(mission_requirements, trace)
    tools = document.get("tools")
    if isinstance(tools, dict):
        invocations = tools.get("successful_invocations")
        if isinstance(invocations, list) and not invocations and successful:
            issues.append(
                "tools.successful_invocations is empty but HCLI recorded "
                f"successful dispatches: {', '.join(sorted(successful))}")
        discovered = tools.get("discovered_count")
        if ("tools.catalog" in successful and isinstance(discovered, (int, float))
                and discovered <= 0):
            issues.append(
                "tools.discovered_count is zero after a successful tools.catalog call")
    tool_coverage = document.get("tool_coverage")
    if isinstance(tool_coverage, dict):
        recorded = tool_coverage.get("successful")
        if isinstance(recorded, list) and not recorded and successful:
            issues.append(
                "tool_coverage.successful is empty but HCLI recorded successful "
                f"dispatches: {', '.join(sorted(successful))}")
        discovered = tool_coverage.get("discovered_count")
        if ("tools.catalog" in successful and isinstance(discovered, (int, float))
                and discovered <= 0):
            issues.append(
                "tool_coverage.discovered_count is zero after a successful "
                "tools.catalog call")
    git_row = next((
        row for row in reversed(trace)
        if row.get("tool") == "git.status" and row.get("dispatched")
        and row.get("ok") and isinstance(row.get("evidence_projection"), dict)
    ), None)
    if git_row is not None and "git_root_and_branch" in document:
        actual = str(document.get("git_root_and_branch") or "")
        projection = git_row["evidence_projection"]
        branch_line = str(projection.get("stdout") or "").splitlines()
        branch = (branch_line[0].removeprefix("## ").split("...", 1)[0].strip()
                  if branch_line else "")
        root = str(projection.get("cwd") or "")
        if not actual:
            issues.append(
                "git_root_and_branch is empty after a successful git.status observation")
        elif ((branch and branch not in actual)
              or (root and root not in actual and root.rsplit("/", 1)[-1] not in actual)):
            issues.append(
                "git_root_and_branch does not retain git.status's observed root/branch")
    objective = document.get("objective")
    if (isinstance(objective, dict) and objective.get("found") is True
            and "current objective" in pending):
        issues.append(
            "objective.found is true without a successful current-objective observation")
    campaign_row = next((
        row for row in reversed(trace)
        if row.get("tool") == "campaign.state" and row.get("dispatched")
        and row.get("ok") and isinstance(row.get("evidence_projection"), dict)
    ), None)
    if campaign_row is not None and isinstance(objective, dict):
        parent = campaign_row["evidence_projection"].get(
            "authoritative_parent") or {}
        expected_source = str(parent.get("path") or "")
        if objective.get("found") is not True:
            issues.append(
                "objective.found is false although campaign.state returned the "
                "current authoritative parent")
        source = str(objective.get("source") or "")
        if expected_source and not (
                source == expected_source or source.endswith("/" + expected_source)):
            issues.append(
                "objective.source does not match campaign.state's authoritative "
                f"parent path {expected_source}")
        latest = str(parent.get("latest_section") or "").split("—", 1)[0].strip()
        summary = str(objective.get("summary") or "")
        if latest and latest.lower() not in summary.lower():
            issues.append(
                "objective.summary does not identify campaign.state's current "
                f"section {latest!r}")
    evidence_check = document.get("evidence_check")
    receipt_row = next((
        row for row in reversed(trace)
        if row.get("tool") in {"receipt.read", "receipt.inspect", "benchmark.inspect"}
        and row.get("dispatched") and row.get("ok")
    ), None)
    if receipt_row is not None and isinstance(evidence_check, dict):
        expected_path = str((receipt_row.get("arguments") or {}).get("path") or "")
        actual_path = str(evidence_check.get("receipt_or_artifact") or "")
        if not actual_path or (expected_path and not (
                actual_path == expected_path or actual_path.endswith("/" + expected_path))):
            issues.append(
                "evidence_check.receipt_or_artifact does not identify the "
                f"successful receipt.read path {expected_path}")
        if not str(evidence_check.get("measured_fact") or "").strip():
            issues.append(
                "evidence_check.measured_fact is empty after a successful receipt.read")
        if not str(evidence_check.get("provenance") or "").strip():
            issues.append(
                "evidence_check.provenance is empty after a successful receipt.read")
    daemon = document.get("daemon")
    if (isinstance(daemon, dict) and daemon.get("reachable") is True
            and "live runtime" in pending):
        issues.append(
            "daemon.reachable is true without a successful live-runtime observation")
    model = document.get("model_identity")
    if (isinstance(model, dict) and model.get("independently_verified") is True
            and "live runtime" in pending):
        issues.append(
            "model_identity.independently_verified is true without live-runtime evidence")
    runtime_row = next((
        row for row in reversed(trace)
        if row.get("tool") in {"processes.summary", "processes.list"}
        and row.get("dispatched") and row.get("ok")
        and isinstance(row.get("evidence_projection"), dict)
    ), None)
    if runtime_row is not None and isinstance(model, dict):
        processes = runtime_row["evidence_projection"].get("processes") or []
        hawkingd = next((row for row in processes
                        if isinstance(row, dict) and row.get("role") == "hawkingd"), None)
        if hawkingd is not None and "KIMI_P0_OPERATIONAL" in str(
                hawkingd.get("command") or ""):
            if model.get("independently_verified") is not True:
                issues.append(
                    "model_identity.independently_verified is false although "
                    "processes.summary identified hawkingd's served artifact")
            if not str(model.get("evidence") or "").strip():
                issues.append(
                    "model_identity.evidence is empty despite the hawkingd runtime observation")
    if runtime_row is not None and "runtime_ownership_preserved" in document:
        process_rows = runtime_row["evidence_projection"].get("processes") or []
        if (_runtime_tree_preserved(process_rows)
                and document.get("runtime_ownership_preserved") is not True):
            issues.append(
                "runtime_ownership_preserved is false although processes.summary "
                "observed one hawkingd root owning every production descendant")
        if (document.get("runtime_ownership_preserved") is True
                and not _runtime_tree_preserved(process_rows)):
            issues.append(
                "runtime_ownership_preserved is true without one hawkingd root "
                "owning every reported production process")
    not_established = document.get("not_established")
    if isinstance(not_established, list):
        lowered = [str(item).lower().replace("_", " ")
                   for item in not_established]
        if (isinstance(model, dict)
                and model.get("independently_verified") is True
                and any("model identity" in item for item in lowered)):
            issues.append(
                "not_established still lists model identity after runtime verification")
        if (isinstance(daemon, dict)
                and daemon.get("unexpected_duplicate_runtime") in {"YES", "NO"}
                and any("duplicate" in item for item in lowered)):
            issues.append(
                "not_established still lists duplicate runtime after process inspection")
    if "real_action_performed" in document and successful and not str(
            document.get("real_action_performed") or "").strip():
        issues.append(
            "real_action_performed is empty although HCLI recorded successful dispatches")
    if str(document.get("liveness") or "").upper() == "PASS" and pending:
        issues.append(
            "liveness is PASS while requested evidence remains unobserved: "
            + ", ".join(sorted(pending)))
    mutation = document.get("mutation")
    accepted_mutation = _accepted_mutation_row(trace)
    if isinstance(mutation, dict) and "accepted" in mutation:
        if mutation.get("accepted") is True and accepted_mutation is None:
            issues.append(
                "mutation.accepted is true without an applied repo.edit "
                "transaction whose verdict is accepted")
        if mutation.get("accepted") is not True and accepted_mutation is not None:
            issues.append(
                "mutation.accepted is false although HCLI recorded an applied "
                "repo.edit transaction with verdict accepted")
    verification = document.get("verification")
    passing_test = _passing_test_row(trace)
    if isinstance(verification, dict) and "passed" in verification:
        if verification.get("passed") is True and passing_test is None:
            issues.append(
                "verification.passed is true without a dispatched tests.run "
                "observation with verified=true and returncode=0")
        if verification.get("passed") is not True and passing_test is not None:
            issues.append(
                "verification.passed is false although HCLI recorded a passing "
                "tests.run observation")
    mission_verdict = document.get("mission")
    if (isinstance(mission_verdict, str)
            and mission_verdict.upper() == "PASS" and (pending_mission or pending)):
        missing = sorted(set(pending_mission) | set(pending))
        issues.append(
            "mission is PASS while explicit completion evidence remains absent: "
            + ", ".join(missing))
    if (isinstance(mission_verdict, str)
            and mission_requirements and not pending_mission and not pending
            and mission_verdict.upper() != "PASS"):
        issues.append(
            "mission is not PASS although every explicit mutation/test terminal "
            "is present in HCLI's dispatch trace")
    # The user supplied the PASS prerequisites. Once every requested evidence
    # family is actually observed and the receipt fields above retain those
    # observations, PARTIAL is no longer epistemic caution; it contradicts the
    # stated gate. This does not manufacture evidence or waive an optional
    # identity check -- it only applies after all mechanical requirements pass.
    pass_ready = (
        not pending
        and isinstance(document.get("hawking"), dict)
        and document["hawking"].get("repository_found") is True
        and isinstance(objective, dict) and objective.get("found") is True
        and isinstance(evidence_check, dict)
        and bool(evidence_check.get("receipt_or_artifact"))
        and bool(evidence_check.get("measured_fact"))
        and bool(document.get("real_action_performed"))
    )
    if pass_ready and str(document.get("liveness") or "").upper() != "PASS":
        issues.append(
            "liveness is not PASS although every user-stated PASS prerequisite "
            "has a successful observation and a retained receipt fact")
    return issues


def _repair_trace_redundant_fields(
        text: str, trace: Sequence[Dict[str, Any]],
        mission_requirements: frozenset[str] = frozenset(),
        evidence_obligations: Optional[Dict[str, frozenset[str]]] = None,
        ) -> Tuple[str, List[str]]:
    """Repair receipt fields that merely restate HCLI's dispatch ledger.

    A bounded model correction can still copy the user's zero/empty JSON
    examples after it has successfully used tools.  Those fields are not
    judgments: HCLI already knows exactly which calls succeeded and what each
    successful canonical evidence tool returned. Reconcile only values carried
    by those successful projections. A liveness verdict is selected only when
    the user's own explicit PASS prerequisites are all mechanically present.
    Never turn a failed lookup into evidence.
    """
    try:
        document = json.loads(text)
    except (TypeError, ValueError):
        return text, []
    if not isinstance(document, dict):
        return text, []
    tools = document.get("tools")
    repaired: List[str] = []

    def _set(mapping: Any, key: str, value: Any, field: str) -> None:
        if isinstance(mapping, dict) and key in mapping and mapping.get(key) != value:
            mapping[key] = value
            repaired.append(field)

    successful = sorted(_successful_evidence_tools(trace))
    _set(tools, "successful_invocations", successful,
         "tools.successful_invocations")
    tool_coverage = document.get("tool_coverage")
    _set(tool_coverage, "successful", successful,
         "tool_coverage.successful")
    catalog = next((
        row.get("catalog_selection") for row in reversed(trace)
        if row.get("tool") == "tools.catalog" and row.get("dispatched")
        and row.get("ok") and isinstance(row.get("catalog_selection"), dict)
    ), None)
    if catalog is not None:
        count = catalog.get("match_count")
        if not isinstance(count, int):
            count = catalog.get("shown")
        if isinstance(count, int):
            _set(tools, "discovered_count", count, "tools.discovered_count")
            _set(tool_coverage, "discovered_count", count,
                 "tool_coverage.discovered_count")
        categories = catalog.get("broad_categories")
        if isinstance(categories, list):
            exact = [str(item) for item in categories]
            _set(tools, "broad_categories", exact, "tools.broad_categories")

    git_row = next((
        row for row in reversed(trace)
        if row.get("tool") == "git.status" and row.get("dispatched")
        and row.get("ok") and isinstance(row.get("evidence_projection"), dict)
    ), None)
    hawking = document.get("hawking")
    if git_row is not None and isinstance(hawking, dict):
        projection = git_row["evidence_projection"]
        root = str(projection.get("cwd") or "")
        _set(hawking, "repository_found", True, "hawking.repository_found")
        if root:
            _set(hawking, "repository_root", root, "hawking.repository_root")
        if "evidence" in hawking and isinstance(hawking.get("evidence"), list):
            evidence = [
                f"git.status returncode={projection.get('returncode')} cwd={root}"
            ]
            for row in trace:
                if row.get("tool") in {"fs.list", "fs.search"} and row.get("ok"):
                    item = row.get("evidence_projection") or {}
                    paths = item.get("paths") or item.get("files") or []
                    if paths:
                        evidence.append(str(paths[0]))
                        break
            _set(hawking, "evidence", evidence[:2], "hawking.evidence")
    if git_row is not None and "git_root_and_branch" in document:
        projection = git_row["evidence_projection"]
        root = str(projection.get("cwd") or "")
        branch_lines = str(projection.get("stdout") or "").splitlines()
        branch = (branch_lines[0].removeprefix("## ").split("...", 1)[0].strip()
                  if branch_lines else "unknown-branch")
        _set(document, "git_root_and_branch", f"{root} [{branch}]",
             "git_root_and_branch")

    campaign_row = next((
        row for row in reversed(trace)
        if row.get("tool") == "campaign.state" and row.get("dispatched")
        and row.get("ok") and isinstance(row.get("evidence_projection"), dict)
    ), None)
    objective = document.get("objective")
    if campaign_row is not None and isinstance(objective, dict):
        parent = campaign_row["evidence_projection"].get(
            "authoritative_parent") or {}
        source = parent.get("path")
        latest = parent.get("latest_section")
        _set(objective, "found", True, "objective.found")
        if source:
            _set(objective, "source", source, "objective.source")
        if latest:
            _set(objective, "summary", latest, "objective.summary")

    receipt_row = next((
        row for row in reversed(trace)
        if row.get("tool") in {"receipt.read", "receipt.inspect", "benchmark.inspect"}
        and row.get("dispatched") and row.get("ok")
        and isinstance(row.get("evidence_projection"), dict)
    ), None)
    evidence_check = document.get("evidence_check")
    if receipt_row is not None and isinstance(evidence_check, dict):
        projection = receipt_row["evidence_projection"]
        path = ((receipt_row.get("arguments") or {}).get("path")
                or projection.get("path"))
        fields = projection.get("authoritative_fields") or {}
        preferred = next((key for key in
                          ("status", "verdict", "generated_at")
                          if fields.get(key) is not None), None)
        if path:
            _set(evidence_check, "receipt_or_artifact", path,
                 "evidence_check.receipt_or_artifact")
        if preferred is not None:
            fact = (f"{preferred}={fields[preferred]} (recorded receipt field; "
                    "observed, not inferred or targeted)")
            _set(evidence_check, "measured_fact", fact,
                 "evidence_check.measured_fact")
        provenance = "receipt.read"
        if receipt_row.get("evidence_class"):
            provenance += f"; {receipt_row['evidence_class']} routing"
        _set(evidence_check, "provenance", provenance,
             "evidence_check.provenance")

    runtime_row = next((
        row for row in reversed(trace)
        if row.get("tool") in {"processes.summary", "processes.list"}
        and row.get("dispatched") and row.get("ok")
        and isinstance(row.get("evidence_projection"), dict)
    ), None)
    model = document.get("model_identity")
    daemon = document.get("daemon")
    if runtime_row is not None:
        process_rows = runtime_row["evidence_projection"].get("processes") or []
        hawkingd = next((row for row in process_rows
                        if isinstance(row, dict) and row.get("role") == "hawkingd"), None)
        providers = [row for row in process_rows
                     if isinstance(row, dict) and row.get("role") == "resident-provider"]
        served = None
        command = str((hawkingd or {}).get("command") or "")
        match = re.search(r"\[([^\]]+)\]", command)
        if match:
            served = match.group(1)
        if isinstance(model, dict) and served:
            _set(model, "claimed", served, "model_identity.claimed")
            _set(model, "independently_verified", True,
                 "model_identity.independently_verified")
            _set(model, "evidence", f"processes.summary hawkingd command: {command}",
                 "model_identity.evidence")
        if isinstance(daemon, dict) and hawkingd is not None:
            _set(daemon, "reachable", True, "daemon.reachable")
            if served:
                _set(daemon, "served_model_or_artifact", served,
                     "daemon.served_model_or_artifact")
            duplicate_state = ("NO" if len(providers) == 1 else
                               "YES" if len(providers) > 1 else
                               "NOT_ESTABLISHED")
            _set(daemon, "unexpected_duplicate_runtime", duplicate_state,
                 "daemon.unexpected_duplicate_runtime")
        if "runtime_ownership_preserved" in document:
            if _runtime_tree_preserved(process_rows):
                _set(document, "runtime_ownership_preserved", True,
                     "runtime_ownership_preserved")

    if successful:
        _set(document, "real_action_performed", successful[0],
             "real_action_performed")
    verified = document.get("verified_by_tool_or_runtime")
    if isinstance(verified, list):
        _set(document, "verified_by_tool_or_runtime", successful,
             "verified_by_tool_or_runtime")
    not_established = document.get("not_established")
    if isinstance(not_established, list):
        retained = []
        for item in not_established:
            label = str(item).lower().replace("_", " ")
            if (isinstance(model, dict)
                    and model.get("independently_verified") is True
                    and "model identity" in label):
                continue
            if (isinstance(daemon, dict)
                    and daemon.get("unexpected_duplicate_runtime")
                    in {"YES", "NO"}
                    and "duplicate" in label):
                continue
            if (isinstance(daemon, dict) and daemon.get("reachable") is True
                    and (label == "daemon" or "daemon reachable" in label)):
                continue
            retained.append(item)
        _set(document, "not_established", retained, "not_established")

    # A mutation verdict and a test outcome are dispatcher-owned facts, just
    # like the successful tool list above.  Repair only fields that already
    # exist in the caller's requested envelope, and only from a transaction or
    # test that actually crossed the corresponding authority boundary.
    mutation_row = _accepted_mutation_row(trace)
    mutation = document.get("mutation")
    if mutation_row is not None and isinstance(mutation, dict):
        _set(mutation, "accepted", True, "mutation.accepted")
        _set(mutation, "applied", True, "mutation.applied")
        _set(mutation, "verdict", "accepted", "mutation.verdict")
        paths = mutation_row.get("paths")
        if isinstance(paths, list):
            _set(mutation, "paths", paths, "mutation.paths")
            _set(mutation, "changed_paths", paths, "mutation.changed_paths")

    test_row = _passing_test_row(trace)
    verification = document.get("verification")
    if test_row is not None and isinstance(verification, dict):
        _set(verification, "passed", True, "verification.passed")
        _set(verification, "returncode", 0, "verification.returncode")
        test_paths = (test_row.get("arguments") or {}).get("paths")
        if isinstance(test_paths, list):
            _set(verification, "tests", test_paths, "verification.tests")
            _set(verification, "paths", test_paths, "verification.paths")
            if test_paths:
                _set(verification, "focused_test", str(test_paths[0]),
                     "verification.focused_test")

    if (mission_requirements
            and not _pending_mission_completion(mission_requirements, trace)
            and not _pending_evidence(evidence_obligations or {}, trace)):
        _set(document, "mission", "PASS", "mission")

    pending = _pending_evidence(_evidence_obligations([
        {"role": "user", "content": "discover repository root and git/project; "
         "discover tools/capabilities; current sovereign objective; read receipt; "
         "inspect live hawkingd runtime processes using actual tools"}
    ]), trace)
    pass_ready = (
        not pending
        and isinstance(hawking, dict) and hawking.get("repository_found") is True
        and isinstance(objective, dict) and objective.get("found") is True
        and isinstance(evidence_check, dict)
        and bool(evidence_check.get("receipt_or_artifact"))
        and bool(evidence_check.get("measured_fact"))
        and bool(document.get("real_action_performed"))
    )
    if pass_ready:
        _set(document, "liveness", "PASS", "liveness")
    return json.dumps(document, separators=(",", ":")), repaired


_OBSERVATION_LEDGER_PREFIX = "[HCLI OBSERVATION LEDGER -- MECHANICAL, NOT CONCLUSIONS]"


def _trace_evidence_projection(name: str, value: Any,
                               arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Keep the decision-bearing subset of a successful observation in trace.

    This is deliberately smaller than the user-visible rendered observation.
    It lets the final synthesis retain one exact fact per tool instead of the
    opening 420 characters of a structural wrapper.
    """
    if name == "tools.catalog":
        return _catalog_selection(value)
    if name == "campaign.state":
        selected = _campaign_selection(value)
        if selected is None:
            return None
        parent = selected.get("authoritative_parent")
        return {
            "recent_evidence": (selected.get("recent_evidence") or [])[:2],
            "authoritative_parent": {
                key: parent.get(key) for key in ("path", "latest_section")
                if isinstance(parent, dict) and parent.get(key) is not None
            },
            "continuation_older_than_parent": selected.get(
                "continuation_older_than_parent"),
        }
    if name in {"receipt.read", "receipt.inspect", "benchmark.inspect"}:
        fields = _receipt_fields(value)
        return {
            "path": arguments.get("path"),
            "authoritative_fields": dict(list(fields.items())[:8]),
        }
    if name in {"processes.summary", "processes.list"}:
        selected = _runtime_selection(value)
        if selected is None:
            return None
        return {
            "count": selected.get("count"),
            "roles": selected.get("roles"),
            "processes": [
                {key: row.get(key) for key in
                 ("pid", "ppid", "role", "class", "body", "command")}
                for row in (selected.get("processes") or [])
            ],
        }
    if name in {"fs.search", "filesystem.search"}:
        selected = _search_selection(value)
        if selected is not None:
            samples = selected.get("diverse_match_sample") or []
            return {
                "root": selected.get("root"),
                "pattern": selected.get("pattern"),
                "matches": [
                    {"path": row.get("path"), "line": row.get("line")}
                    for row in samples[:3] if row.get("path")
                ],
                "paths": [row.get("path") for row in samples[:3]],
                "truncated": selected.get("truncated"),
            }
    if name in {"fs.list", "filesystem.list"}:
        selected = _listing_selection(value)
        if selected is not None:
            return {
                "root": selected.get("root"),
                "path": selected.get("path"),
                "files": [row.get("path") for row in
                          (selected.get("file_sample") or [])[:5]],
                "directories": [row.get("path") for row in
                                (selected.get("directories") or [])[:5]],
            }
    if name == "git.status" and isinstance(value, dict):
        return {
            key: value.get(key) for key in
            ("cwd", "returncode", "stdout")
            if value.get(key) is not None
        }
    return None


def _observation_ledger(trace: Sequence[Dict[str, Any]], *, limit: int = 2400) -> str:
    """Compact observations that admission may evict; never synthesize facts."""
    rows = [_OBSERVATION_LEDGER_PREFIX]
    # Repeated-call guard entries must not evict the successful observations
    # they refer to. Keep one copy of each successful dispatch, then only the
    # latest few real failures. This is a working-set projection of the trace,
    # not another evidence source.
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in trace:
        if not (item.get("tool") and item.get("dispatched") and item.get("ok")):
            continue
        signature = (
            item.get("tool"),
            json.dumps(item.get("arguments") or {}, sort_keys=True, default=str),
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(item)
    selected = selected[-8:]
    selected.extend([
        item for item in trace[-8:]
        if item.get("tool") and item.get("dispatched") and not item.get("ok")
    ][-2:])
    for item in selected:
        tool = item.get("tool")
        if not tool:
            continue
        status = "SUCCESS" if item.get("dispatched") and item.get("ok") else "FAILED"
        args = json.dumps(item.get("arguments") or {}, sort_keys=True, default=str)
        row = f"- {status} {tool} {args[:260]}"
        projection = item.get("evidence_projection")
        observation = (
            json.dumps(projection, separators=(",", ":"), default=str)
            if isinstance(projection, dict) else
            " ".join(str(item.get("observation_excerpt") or "").split())
        )
        if observation:
            row += f" -> {observation[:520]}"
        elif item.get("error"):
            row += f" -> {str(item.get('error'))[:300]}"
        rows.append(row)
    rows.append(
        "These are calls already dispatched by HCLI. Preserve them when "
        "reporting successful invocations; inspect the retained paste id if "
        "more detail is needed.")
    text = "\n".join(rows)
    return text if len(text) <= limit else text[:limit] + "\n[ledger bounded]"


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
    max_prompt_chars: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer, consulting tools when the model asks. Returns (text, trace).

    `complete` takes messages and returns the model's text, so this is testable
    without a resident and the caller keeps ownership of the backend.

    `max_prompt_chars`, when given, is a PROMPT ADMISSION GUARD: never call
    `complete` again with a conversation bigger than this. SCAR: the caller
    compacts the INCOMING messages exactly once, before this loop starts --
    every completion INSIDE the loop was unchecked. Measured live: once
    batched tool calls started executing (see parse_calls), a real cycle grew
    the conversation to 10203 tokens against the resident's native
    max_seq_len=8192, and the backend raised -- surfaced to the caller as a
    bare 502 with nothing HCLI could act on. A body that never gets a
    response back cannot repair anything, including this. `None` (the
    default) keeps every existing caller's behavior exactly as it was.
    """
    conversation = [dict(m) for m in messages]
    trace: List[Dict[str, Any]] = []
    tool_observation_required = explicit_tool_observation_required(messages)
    requested_json_shape = _requested_json_shape(messages)
    evidence_obligations = _evidence_obligations(messages)
    mission_requirements = _mission_completion_requirements(messages)
    # A real write mission has two additional authority crossings after
    # orientation: an accepted mutation and an independent verification. The
    # ordinary twelve-round budget was measured ending exactly on the first
    # mutation nudge after a six-family evidence sweep. Allocate bounded room
    # for those explicit terminals without increasing ordinary chat cost.
    round_limit = max_calls
    if mission_requirements and max_calls == MAX_CALLS:
        round_limit = min(
            32,
            max_calls + (2 * len(evidence_obligations))
            + (4 * len(mission_requirements)),
        )
    tool_obligation_nudged = False
    # (start, end) of each FULLY COMPLETED turn this loop has appended --
    # never the caller's original messages, which stay outside this list and
    # so are never evicted.
    turn_bounds: List[Tuple[int, int]] = []

    def _admit() -> None:
        if max_prompt_chars is None:
            return
        size = sum(len(str(m.get("content") or "")) for m in conversation)
        # Evict the OLDEST completed turn first -- always leave the most
        # recent one intact, so the model has some grounding to continue
        # from. Runs before every completion, including the first (where
        # there is nothing yet to evict).
        while size > max_prompt_chars and len(turn_bounds) > 1:
            start, end = turn_bounds.pop(0)
            evicted = sum(len(str(m.get("content") or "")) for m in conversation[start:end])
            del conversation[start:end]
            shift = end - start
            turn_bounds[:] = [(s - shift, e - shift) for s, e in turn_bounds]
            size -= evicted
    # The menu for THIS session. Authority decides the door set once, outside
    # the loop, so no turn can widen it.
    offered = session_menu(registry, write=engine is not None)
    # LOOP GUARD. A greedy body can emit the SAME tool call every round -- a
    # small model on an open-ended objective degenerates into re-calling
    # observation.expand (or any tool) on the same arguments forever, churning
    # the whole budget without changing evidence. Re-executing a call it already
    # made teaches it nothing; it just burns rounds. Track each (name, args)
    # signature: hand back the prior result and redirect, and after the same
    # call is emitted a third time, stop the churn and ask for the answer.
    seen: Dict[str, int] = {}
    results: Dict[str, str] = {}
    recovery_nudged = False
    evidence_nudges = 0
    evidence_nudge_limit = max(4, len(evidence_obligations) + 2)
    evidence_nudge_attempts: Dict[str, int] = {}
    mission_nudge_attempts: Dict[str, int] = {}
    controller_routed_label: Optional[str] = None

    def _controller_evidence_call(
            label: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Resolve one explicit read-only evidence obligation after a retry.

        This is serving-contract assistance, not learned-model evidence. It
        never writes, never fabricates a result, and for a receipt only follows
        an exact path returned by a successful live campaign.state observation.
        """
        canonical: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "repository": ("git.status", {}),
            "filesystem": ("fs.list", {"path": "."}),
            "tool discovery": ("tools.catalog", {"focus": "HCLI/AgentOS"}),
            "current objective": ("campaign.state", {}),
            "campaign/receipt": ("campaign.state", {}),
            "live runtime": ("processes.summary", {}),
            "diagnostic/forensic inspection": (
                "forensics.snapshot", {"note": "blind mission evidence"}),
        }
        if label in canonical:
            name, arguments = canonical[label]
            return (name, arguments) if name in offered else None
        if label == "source inspection" and "fs.search" in offered:
            searched = next((
                row for row in reversed(trace)
                if row.get("tool") == "fs.search" and row.get("dispatched")
                and row.get("ok")
                and isinstance(row.get("evidence_projection"), dict)
                and row["evidence_projection"].get("paths")
            ), None)
            if searched is not None and "fs.read" in offered:
                projection = searched["evidence_projection"]
                matches = projection.get("matches") or []
                if matches:
                    first = matches[0]
                    line = first.get("line")
                    arguments: Dict[str, Any] = {"path": str(first["path"])}
                    if isinstance(line, int) and line > 0:
                        # A search hit is a coordinate, not just a filename.
                        # Reading from line one hid the implementation KIMI
                        # had found farther down the source file.
                        arguments.update({
                            "start_line": max(1, line - 20),
                            "end_line": line + 120,
                        })
                    return "fs.read", arguments
                paths = projection.get("paths") or []
                if paths:
                    return "fs.read", {"path": str(paths[0])}
            # Derive a literal query from the user's own objective sentence.
            # This does not supply a target file, symbol, or implementation;
            # it performs the repository-search step the body requested but
            # repeatedly echoed instead of serializing. Exact-phrase miss
            # recovery below will try its first source tokens mechanically.
            latest = next((str(row.get("content") or "")
                           for row in reversed(messages)
                           if row.get("role") == "user"), "")
            objective = re.split(
                r"\b(?:Tool-diversity requirement|Engineering rules)\s*:",
                re.split(r"\bObjective\s*:", latest, maxsplit=1,
                         flags=re.I)[-1],
                maxsplit=1, flags=re.I)[0]
            words = [word for word in re.findall(
                r"[A-Za-z_][A-Za-z0-9_.-]{2,}", objective)
                     if word.lower() not in _SEARCH_PHRASE_STOPWORDS]
            if words:
                return "fs.search", {
                    "path": ".",
                    "pattern": " ".join(words[:4]),
                    "max_results": 80,
                }
        if label == "focused test inspection" and "fs.search" in offered:
            searched = next((
                row for row in reversed(trace)
                if row.get("tool") == "fs.search" and row.get("dispatched")
                and row.get("ok")
                and isinstance(row.get("evidence_projection"), dict)
                and any(
                    Path(str(match.get("path") or "")).name.startswith("test_")
                    or Path(str(match.get("path") or "")).name.endswith("_test.py")
                    for match in (row["evidence_projection"].get("matches") or [])
                    if isinstance(match, dict)
                )
            ), None)
            if searched is not None and "fs.read" in offered:
                match = next(
                    match for match in searched["evidence_projection"]["matches"]
                    if (Path(str(match.get("path") or "")).name.startswith("test_")
                        or Path(str(match.get("path") or "")).name.endswith("_test.py"))
                )
                line = match.get("line")
                arguments = {"path": str(match["path"])}
                if isinstance(line, int) and line > 0:
                    arguments.update({
                        "start_line": max(1, line - 30),
                        "end_line": line + 150,
                    })
                return "fs.read", arguments
            source_row = next((
                row for row in reversed(trace)
                if row.get("tool") == "fs.read" and row.get("dispatched")
                and row.get("ok") and row.get("observation")
            ), None)
            if source_row is not None:
                definitions = re.findall(
                    r"\bdef\s+([A-Za-z][A-Za-z0-9_]*)",
                    str(source_row.get("observation") or ""),
                )
                symbol = next((name for name in definitions
                               if not name.startswith("_")), None)
                if symbol:
                    # The symbol came from source already observed. Searching
                    # test files for it is a mechanical counterpart lookup,
                    # not a hidden target-file hint.
                    return "fs.search", {
                        "path": ".",
                        "pattern": symbol,
                        "glob": "test*.py",
                        "max_results": 80,
                    }
            latest = next((str(row.get("content") or "")
                           for row in reversed(messages)
                           if row.get("role") == "user"), "")
            objective = re.split(
                r"\b(?:Tool-diversity requirement|Engineering rules)\s*:",
                re.split(r"\bObjective\s*:", latest, maxsplit=1,
                         flags=re.I)[-1],
                maxsplit=1, flags=re.I)[0]
            words = [word for word in re.findall(
                r"[A-Za-z_][A-Za-z0-9_.-]{2,}", objective)
                     if word.lower() not in _SEARCH_PHRASE_STOPWORDS]
            if words:
                return "fs.search", {
                    "path": ".",
                    "pattern": " ".join(words[:4]),
                    "glob": "test*.py",
                    "max_results": 80,
                }
        if label == "receipt/evidence artifact" and "receipt.read" in offered:
            for row in reversed(trace):
                if not (row.get("tool") == "campaign.state"
                        and row.get("dispatched") and row.get("ok")):
                    continue
                candidates = row.get("evidence_candidates") or []
                if candidates:
                    return "receipt.read", {"path": str(candidates[0])}
        return None

    def _evidence_nudge(pending: Dict[str, frozenset[str]], *, repeated: bool) -> str:
        """Ask for one missing evidence class, without supplying its answer."""
        label, alternatives = next(iter(pending.items()))
        # Choose the semantic owner, not every mechanically acceptable fallback.
        # A read of *some* file can satisfy repository access, but it is a poor
        # door for current authority or a requested receipt: measured KIMI chose
        # .aider.chat.history.md when offered `fs.read or receipt.read`, despite
        # campaign.state having returned six exact receipt paths. The canonical
        # door still requires the body to choose any required path and emit the
        # call; HCLI supplies neither evidence nor an answer here.
        preferred = {
            "repository": ("git.status",),
            "filesystem": ("fs.list",),
            "source inspection": ("fs.search", "fs.read"),
            "focused test inspection": ("fs.search", "fs.read"),
            "tool discovery": ("tools.catalog",),
            "current objective": ("campaign.state",),
            "receipt/evidence artifact": ("receipt.read",),
            "campaign/receipt": ("campaign.state", "receipt.read"),
            "live runtime": ("processes.summary",),
            "diagnostic/forensic inspection": (
                "audit", "forensics.snapshot", "reverse.identify",
                "debug.diagnose"),
        }.get(label, ())
        canonical = [name for name in preferred if name in alternatives]
        choices = " or ".join(canonical or sorted(alternatives))
        prefix = (
            "The repeated batch added no observation. Do not emit that batch "
            "again. " if repeated else
            "Do not answer the objective yet. "
        )
        return (
            prefix
            + f"The next unobserved evidence category is {label}. Choose ONE "
              f"admitted tool from this family: {choices}. Supply arguments "
              "you discover from prior observations or the repository and "
              "reply only with that new tool call object. Do not invent a "
              "path, observation id, process id, or result."
        )

    def _refresh_observation_ledger() -> None:
        if not any(row.get("tool") for row in trace):
            return
        content = _observation_ledger(trace)
        for row in conversation:
            if str(row.get("content") or "").startswith(
                    _OBSERVATION_LEDGER_PREFIX):
                row["content"] = content
                return
        conversation.append({"role": "user", "content": content})

    def _finalize(candidate: Any) -> Tuple[str, List[Dict[str, Any]]]:
        clean = _clean_final_text(candidate)
        if requested_json_shape is None:
            return clean, trace
        clean, issues = _validate_requested_json(clean, requested_json_shape)
        issues.extend(_trace_grounding_issues(
            clean, trace, evidence_obligations, mission_requirements))
        if not issues:
            return clean, trace
        trace.append({
            "tool": None,
            "ok": False,
            "dispatched": False,
            "error": "requested final JSON contract was incomplete",
            "contract_issues": issues[:24],
            "reply_excerpt": clean[:240],
        })
        # One bounded correction, using only observations already delivered.
        # This enforces the caller's envelope but supplies no fact or answer.
        conversation.append({"role": "assistant", "content": clean[:1200]})
        conversation.append({
            "role": "user",
            "content": (
                "Your final response did not satisfy the exact JSON structure "
                "requested in my original message. Contract issues: "
                + "; ".join(issues[:24])
                + ". Reply now with ONLY one syntactically valid JSON object "
                  "with exactly the requested keys and nested object/array "
                  "shape. Keep it terse enough to complete under 1,800 "
                  "characters: cite tool names, paths, and short facts; do "
                  "not copy observation bodies. Use only evidence already "
                  "observed. Do not call a tool and do not add prose or a "
                  "markdown fence."
            ),
        })
        _admit()
        repaired = _clean_final_text(complete(conversation))
        if parse_calls(repaired):
            trace.append({
                "tool": None, "ok": False, "dispatched": False,
                "error": "tool call emitted during final JSON correction",
                "reply_excerpt": repaired[:240],
            })
            return (
                "HCLI could not satisfy the requested final JSON contract: "
                "the bounded correction emitted another tool call.", trace)
        repaired, remaining = _validate_requested_json(
            repaired, requested_json_shape)
        remaining.extend(_trace_grounding_issues(
            repaired, trace, evidence_obligations, mission_requirements))
        mechanically_repaired: List[str] = []
        if remaining:
            repaired, mechanically_repaired = _repair_trace_redundant_fields(
                repaired, trace, mission_requirements, evidence_obligations)
            repaired, remaining = _validate_requested_json(
                repaired, requested_json_shape)
            remaining.extend(_trace_grounding_issues(
                repaired, trace, evidence_obligations, mission_requirements))
        if remaining:
            trace.append({
                "tool": None, "ok": False, "dispatched": False,
                "error": "requested final JSON contract remained incomplete",
                "contract_issues": remaining[:24],
                "reply_excerpt": repaired[:240],
            })
            return (
                "HCLI could not satisfy the requested final JSON contract "
                "after one bounded correction: " + "; ".join(remaining[:8]),
                trace,
            )
        trace.append({
            "tool": None,
            "ok": True,
            "dispatched": False,
            "event": "requested final JSON contract repaired",
            **({
                "mechanical_fields": mechanically_repaired,
                "evidence_class": "HCLI_SERVING_CONTRACT",
            } if mechanically_repaired else {}),
        })
        return repaired, trace

    for _ in range(max(0, round_limit)):
        _refresh_observation_ledger()
        _admit()
        text = complete(conversation)
        calls = parse_calls(text)
        controller_routed_label = None
        if not calls:
            dispatched = any(bool(row.get("dispatched")) for row in trace)
            pending = _pending_evidence(evidence_obligations, trace)
            # Before any dispatcher success, the existing one-retry fail-closed
            # rule owns the correction. Category nudges are continuation
            # guidance after a real observation, not permission to turn a body
            # that cannot call any tool into a four-retry spinner.
            pending_label = next(iter(pending), None)
            routed = (
                _controller_evidence_call(pending_label)
                if (pending_label is not None and dispatched
                    and evidence_nudge_attempts.get(pending_label, 0) >= 1)
                else None
            )
            if routed is not None:
                calls = [routed]
                controller_routed_label = pending_label
            elif (pending and dispatched
                    and evidence_nudges < evidence_nudge_limit):
                trace.append({
                    "tool": None, "ok": False, "dispatched": False,
                    "error": "explicit evidence categories remain unobserved",
                    "pending_evidence": sorted(pending),
                    "reply_excerpt": str(text or "")[:240],
                })
                conversation.append({
                    "role": "assistant", "content": str(text or "")[:1000],
                })
                conversation.append({
                    "role": "user",
                    "content": _evidence_nudge(pending, repeated=False),
                })
                evidence_nudges += 1
                if pending_label is not None:
                    evidence_nudge_attempts[pending_label] = (
                        evidence_nudge_attempts.get(pending_label, 0) + 1)
                continue
            if not calls and tool_observation_required and not dispatched:
                trace.append({
                    "tool": None,
                    "ok": False,
                    "dispatched": False,
                    "error": (
                        "explicit tool observation required but no parseable "
                        "tool call was emitted"
                    ),
                    "reply_excerpt": str(text or "")[:240],
                })
                if not tool_obligation_nudged:
                    # A body can fill the entire requested completion with a
                    # repeated prose/tool-name loop.  The corrective turn does
                    # not need that whole loop in context; retaining a bounded
                    # excerpt keeps the retry inside the same admission budget.
                    excerpt = str(text or "")[:1000]
                    if len(str(text or "")) > len(excerpt):
                        excerpt += "\n[unsupported reply truncated by HCLI]"
                    conversation.append({"role": "assistant", "content": excerpt})
                    conversation.append({
                        "role": "user",
                        "content": (
                            _evidence_nudge(pending, repeated=False)
                            if pending else
                            "The objective explicitly requires a real tool "
                            "observation, but no tool was dispatched. Do not say "
                            "you used, searched, read, or verified anything yet. "
                            "Reply now with ONLY one JSON tool call using an "
                            "available repository tool and its declared arguments."
                        ),
                    })
                    tool_obligation_nudged = True
                    continue
                return (
                    "HCLI could not satisfy the required tool observation: no "
                    "parseable tool call was emitted after one corrective retry.",
                    trace,
                )
            pending_mission = _pending_mission_completion(
                mission_requirements, trace)
            if not calls and pending_mission:
                terminal = next(iter(sorted(pending_mission)))
                attempts = mission_nudge_attempts.get(terminal, 0)
                malformed_intent = bool(re.search(
                    (r'"(?:op|tool|operation)"\s*:\s*"(?:repo\.edit|edit)"'
                     if terminal == "accepted mutation" else
                     r'"(?:op|tool|operation)"\s*:\s*"(?:tests\.run|test)"'),
                    str(text or ""), re.I))
                rejected_mutation = (
                    terminal == "accepted mutation" and any(
                        row.get("tool") == "repo.edit"
                        and row.get("dispatched")
                        and str(row.get("verdict") or "").lower() == "rejected"
                        for row in trace))
                observed_reads = [
                    row for row in trace
                    if row.get("tool") == "fs.read"
                    and row.get("dispatched") and row.get("ok")
                    and (row.get("arguments") or {}).get("path")
                ]
                observed_test = next((
                    row for row in reversed(observed_reads)
                    if "test" in Path(str(
                        (row.get("arguments") or {}).get("path")
                    )).name.lower()
                ), None)
                observed_source = next((
                    row for row in reversed(observed_reads)
                    if row is not observed_test
                ), observed_reads[-1] if observed_reads else None)
                reminders = []
                for label, row in (("source", observed_source),
                                   ("focused test", observed_test)):
                    if row is None:
                        continue
                    args = row.get("arguments") or {}
                    window = (f" lines {args.get('start_line')}-"
                              f"{args.get('end_line')}"
                              if args.get("start_line") else "")
                    reminders.append(
                        f"{label} {args.get('path')}{window}")
                source_reminder = (
                    " The already observed edit targets are: "
                    + "; ".join(reminders) + "."
                    if reminders else "")
                source_anchors = [
                    item for item in _selected_observation_lines(
                        observed_source or {}, limit=16)
                    if _SOURCE_EFFECT_LINE.search(item[1])
                ][-4:]
                all_test_anchors = [
                    item for item in _selected_observation_lines(
                        observed_test or {}, limit=16)
                    if _TEST_EXPECTATION_LINE.search(item[1])
                ]
                test_anchors = [
                    item for item in all_test_anchors
                    if re.search(r"\b(?:PYTHONPATH|current|shim)\b",
                                 item[1], re.I)
                ][:4] or all_test_anchors[-4:]
                anchor_rows = [
                    *(f"source line {line}: {value}"
                      for line, value in source_anchors),
                    *(f"test line {line}: {value}"
                      for line, value in test_anchors),
                ]
                if anchor_rows:
                    source_reminder += (
                        " Exact anchors retained from those prior reads: "
                        + " | ".join(anchor_rows) + "."
                    )
                trace.append({
                    "tool": None,
                    "ok": False,
                    "dispatched": False,
                    "error": "explicit mission completion evidence remains absent",
                    "pending_mission": sorted(pending_mission),
                    "reply_excerpt": str(text or "")[:240],
                    # A prefix alone made three distinct transport failures
                    # indistinguishable: output truncation, a broken closing
                    # structure, and an invalid string escape all looked like
                    # the same apparently-correct repo.edit intent. Preserve a
                    # bounded tail plus size/hash; never retain an unbounded
                    # generated payload in the mission receipt.
                    "reply_chars": len(str(text or "")),
                    "reply_tail": str(text or "")[-480:],
                    "reply_sha256": hashlib.sha256(
                        str(text or "").encode("utf-8", "replace")
                    ).hexdigest(),
                })
                # One additional recovery is allowed only when the body named
                # the exact required door but its long JSON payload did not
                # parse. That is a transport failure, not another autonomous
                # plan attempt. Keep all other mission nudges single-shot.
                allowed_attempts = (3 if rejected_mutation else
                                    3 if malformed_intent else 1)
                if attempts >= allowed_attempts:
                    return (
                        "HCLI could not satisfy the explicit mission gate: "
                        + ", ".join(sorted(pending_mission))
                        + " was not evidenced after a bounded corrective turn.",
                        trace,
                    )
                conversation.append({
                    "role": "assistant", "content": str(text or "")[:1000],
                })
                if terminal == "accepted mutation":
                    instruction = ((
                        "The latest repo.edit transaction was rejected, so no "
                        "repository change landed. Use the rejection reason and "
                        "the source already read to emit ONE different, small, "
                        "schema-valid repo.edit operation. In compact JSON, op "
                        "is edit; edits is a flat list of actual path, verbatim "
                        "old text, and authored new text rows; each row is an "
                        "array of exactly three short strings, never a nested "
                        "object. Test is the actual focused test path. Do not "
                        "copy schema labels and do not "
                        "switch to an "
                        "unrelated path. Include the focused test already "
                        "inspected in the same call's non-empty `tests` list; "
                        "do not claim success before the transaction returns."
                        + source_reminder
                    ) if rejected_mutation else (
                        "Your repo.edit intent was recognizable, but the JSON "
                        "payload was not parseable and therefore nothing ran. "
                        "Do not repeat or copy the prior payload or the schema. "
                        "Emit ONE materially smaller JSON call whose op is edit, "
                        "whose edits value is a flat list of three-item edit rows, "
                        "and whose test value names the observed focused test. "
                        "Each row must contain the actual repository path, exact "
                        "verbatim existing text, and your authored replacement. "
                        "Each row is an array of exactly three short strings, "
                        "never an object; do not emit nested op/path/old_lines/"
                        "new_lines fields. Keep each old anchor to one line. "
                        "The literal labels path, old, new, and focused/test.py "
                        "are invalid placeholders. Do not add an extra list layer. "
                        "Use two short rows when source and regression test both "
                        "must change. Do not use replace_file/create or inline a "
                        "whole function/file. Do not claim a change "
                        "before the transaction returns."
                        + source_reminder
                    ) if malformed_intent else (
                        "Do not answer the objective yet. It explicitly requires "
                        "an accepted repository mutation, but no repo.edit "
                        "transaction has both applied=true and verdict=accepted. "
                        "Use the observations already gathered to emit ONE "
                        "small repo.edit call through the admitted write door. "
                        "Use compact JSON: op is edit; edits is a flat list of "
                        "three-item rows containing actual path, verbatim old "
                        "text, and authored new text; test is the actual focused "
                        "test path. Rows are arrays of three short strings, never "
                        "objects with nested op/path/old_lines/new_lines fields. "
                        "Keep each old anchor to one line. Never emit the literal "
                        "schema labels path, "
                        "old, new, or focused/test.py. Use two rows when source "
                        "and test change together. "
                        "Do not claim a change before the transaction returns."))
                else:
                    instruction = ((
                        "Your tests.run intent was recognizable, but its JSON "
                        "did not parse and no test ran. Emit ONE shorter "
                        "tests.run call naming only the focused test path."
                    ) if malformed_intent else (
                        "Do not answer the objective yet. The accepted mutation "
                        "must be followed by an independently dispatched focused "
                        "test, but no tests.run observation has verified=true and "
                        "returncode=0. Emit ONE focused tests.run call for the "
                        "smallest relevant test file. Do not claim it passed "
                        "before the runner returns."))
                conversation.append({"role": "user", "content": instruction})
                mission_nudge_attempts[terminal] = attempts + 1
                continue
            if not calls:
                return _finalize(text)
        # ONE reply named however many actions the body batched into it (see
        # parse_calls). The assistant turn happened once; append it once, then
        # settle every call it named before spending another turn on a new
        # completion. Bounded by the turn budget itself -- generous enough for
        # the batching actually observed, never unbounded.
        turn_start = len(conversation)
        # The textual protocol does not need the body to see its action request
        # again: the following observations and mechanical ledger carry what
        # actually ran. Replaying the raw batch primed KIMI to copy the whole
        # plan; replacing it with a compact assistant placeholder merely made
        # KIMI copy that placeholder verbatim for six rounds. Omit the textual
        # assistant turn entirely. Native tool-call transports retain it.
        if native:
            conversation.append({"role": "assistant", "content": text})
        stop_early = False
        recovery_needed = False
        terminal_repeat_seen = False
        new_call_dispatched = False
        repeated_names: List[str] = []
        for name, arguments in calls[:max(1, max_calls)]:
            if name not in offered:
                # Naming what IS available turns a dead end into a retry that
                # can work -- the same rule the engine's test-command refusal
                # follows.
                conversation.append({"role": "user", "content":
                                     f"{name} is not available here. Use a listed "
                                     "canonical tool or call tools.catalog for an "
                                     "admitted capability; answer directly if no "
                                     "tool applies."})
                trace.append({"tool": name, "ok": False,
                              "dispatched": False,
                              "error": "not offered to chat"})
                recovery_needed = True
                continue
            arguments = coerce_arguments(registry, name, arguments)
            sig = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
            if sig in seen:
                seen[sig] += 1
                repeated_names.append(name)
                trace.append({"tool": name, "ok": False,
                              "dispatched": False,
                              "error": "repeated call (loop guard)"})
                if seen[sig] >= 3:  # third emission of the same call -- stop the churn
                    terminal_repeat_seen = True
                # A batch may contain a repeated prefix followed by a new
                # recovery call. Never discard the latter merely because the
                # former crossed its repetition threshold.
                continue
            seen[sig] = 1
            interactive_refusal = (
                _interactive_test_scope_refusal(arguments)
                if name == "tests.run" else None
            )
            dispatch_name = name
            dispatch_arguments = arguments
            listing_request = (
                _filename_listing_request(arguments)
                if name in {"fs.search", "filesystem.search"} else None
            )
            if listing_request is not None and "fs.list" in offered:
                dispatch_name = "fs.list"
                dispatch_arguments = listing_request
            if dispatch_name in MUTATION_TOOLS:
                result = run_builder_tool(
                    dispatch_name, dispatch_arguments, engine=engine,
                    require_tests=("accepted mutation" in mission_requirements))
            elif dispatch_name in LOCAL_TOOLS:
                result = run_local_tool(dispatch_name, dispatch_arguments, cache=cache,
                                        knowledge=knowledge)
            elif interactive_refusal:
                result = type("InteractiveTestRefusal", (), {
                    "ok": False,
                    "value": None,
                    "error": interactive_refusal,
                    "provenance": {"source": "hcli.chat_tools"},
                    "failure_class": "INVALID_ARGUMENTS",
                })()
            else:
                result = registry.invoke(dispatch_name, dispatch_arguments)
            escalated = None
            if dispatch_name == "fs.search" and (
                    _search_needs_escalation(result)
                    or _search_has_no_matches(result)):
                escalated = escalate_search(registry, arguments, result)
                if escalated is not None and getattr(escalated, "ok", False):
                    result = escalated
            entry = {
                "tool": dispatch_name,
                "arguments": dispatch_arguments,
                "ok": bool(getattr(result, "ok", False)),
                # Stronger than `ok`: a rejected result is still evidence that
                # the real dispatcher was reached.
                "dispatched": not bool(interactive_refusal),
                "error": getattr(result, "error", None),
                "provenance": provenance_of(result),
            }
            if controller_routed_label is not None:
                entry["controller_routed_from"] = controller_routed_label
                entry["evidence_class"] = "HCLI_SERVING_CONTRACT"
            if dispatch_name != name:
                entry["requested_tool"] = name
                entry["compatibility_routing"] = (
                    "pure filename-suffix pattern routed to read-only fs.list")
            new_call_dispatched = True
            # WHAT THE MUTATION DID, not merely that it ran. `ok` is True for a
            # REJECTED mutation by design -- a refused edit is a result the body
            # must report, not an error to paper over -- so `ok` has never meant
            # the repository changed. Without the verdict here, every consumer of
            # this trace is structurally unable to tell a landing from a
            # refusal. Campaign cycle 74: the supervisor logged "repo.edit
            # CALLED and accepted" with a clean worktree and an unmoved HEAD.
            if dispatch_name in MUTATION_TOOLS:
                value = getattr(result, "value", None)
                if isinstance(value, dict):
                    entry["verdict"] = value.get("status")
                    entry["applied"] = bool(value.get("applied"))
                    entry["paths"] = value.get("paths")
            # WHETHER THE TESTS PASSED, not merely that the runner ran. Same
            # defect one tool over: `ok` is True for a suite that FAILED, because
            # a failing suite is a result the body must report rather than an
            # error. Campaign cycle 79 labelled "5 passed, returncode 0" as RED
            # and the phase machine advanced on it, demanding an edit to fix a
            # defect its own discriminator had just disproved. `_tests_run`
            # already returns both of these and the trace was discarding them.
            if dispatch_name == "tests.run":
                value = getattr(result, "value", None)
                if isinstance(value, dict):
                    entry["returncode"] = value.get("returncode")
                    entry["verified"] = bool(value.get("verified"))
            if dispatch_name == "campaign.state" and getattr(result, "ok", False):
                value = getattr(result, "value", None)
                recent = value.get("recent_evidence") if isinstance(value, dict) else None
                if isinstance(recent, list):
                    entry["evidence_candidates"] = [
                        str(item.get("receipt")) for item in recent
                        if isinstance(item, dict) and item.get("receipt")
                    ][:8]
            if dispatch_name == "tools.catalog" and getattr(result, "ok", False):
                selection = _catalog_selection(getattr(result, "value", None))
                if selection is not None:
                    entry["catalog_selection"] = selection
            if getattr(result, "ok", False):
                projection = _trace_evidence_projection(
                    dispatch_name, getattr(result, "value", None),
                    dispatch_arguments)
                if projection is not None:
                    entry["evidence_projection"] = projection
            if escalated is not None:
                observed = getattr(escalated, "value", None)
                observed_pattern = (observed.get("pattern")
                                    if isinstance(observed, dict) else None)
                entry["escalated"] = {
                    "glob": _search_fallback_glob(arguments),
                    "pattern": observed_pattern,
                    "reason": (
                        "exact prose phrase had no matches; searched its bounded "
                        "source tokens" if observed_pattern != arguments.get("pattern")
                        else "first search truncated before reaching source"
                    ),
                }
            trace.append(entry)
            observation = _render(result, dispatch_name, registry, cache)
            # The dispatcher result itself is the liveness evidence.  Preserve
            # a small, redacted-by-the-tool excerpt in the receipt so a caller
            # can distinguish "invoked" from "returned a useful observation"
            # without replaying the tool or copying an unbounded payload into
            # the next prompt.
            if getattr(result, "ok", False):
                entry["observation_excerpt"] = observation[:800]
                entry["observation"] = observation
                entry["observation_chars"] = len(observation)
                entry["observation_sha256"] = hashlib.sha256(
                    observation.encode("utf-8", "replace")).hexdigest()
            results[sig] = observation
            # KIMI's template distinguishes machine observations from a new
            # human request only through the tool-response envelope. Bare user
            # text made a rejected edit look like the next objective, and the
            # body copied that rejection for the rest of the bounded mission.
            # The textual JSON-call protocol still needs the same result
            # envelope; native tool schemas describe the *request* side only.
            conversation.append(tool_response(name, observation))
            if not getattr(result, "ok", False):
                recovery_needed = True
        if repeated_names:
            unique_repeats = list(dict.fromkeys(repeated_names))
            conversation.append({
                "role": "user",
                "content": (
                    "HCLI did not rerun these identical requests: "
                    + ", ".join(unique_repeats)
                    + ". Their prior outcomes are in the mechanical "
                      "observation ledger. Do not emit those exact calls "
                      "again; use their observations, choose a different "
                      "bounded call, or answer."
                ),
            })
        if terminal_repeat_seen and not new_call_dispatched:
            pending = _pending_evidence(evidence_obligations, trace)
            if pending and evidence_nudges < evidence_nudge_limit:
                conversation.append({
                    "role": "user",
                    "content": _evidence_nudge(pending, repeated=True),
                })
                trace.append({
                    "tool": None, "ok": False, "dispatched": False,
                    "error": "repeated batch redirected to pending evidence",
                    "pending_evidence": sorted(pending),
                })
                evidence_nudges += 1
            else:
                stop_early = True
        if recovery_needed and not recovery_nudged and not stop_early:
            # A body that sees one failed call often answers with the failure
            # as though it completed the objective. Give it one mechanical
            # recovery turn, without choosing the answer or relaxing guards.
            conversation.append({"role": "user", "content":
                "At least one requested observation failed, so the objective is "
                "not evidenced yet. Before answering, make one different bounded "
                "read-only call that can recover the missing fact. For repository "
                "navigation use fs.list with path \".\" or a named subdirectory; "
                "for a file use fs.search or fs.list before fs.read. Do not repeat "
                "the failed arguments."})
            recovery_nudged = True
        turn_bounds.append((turn_start, len(conversation)))
        if stop_early:
            break
    _refresh_observation_ledger()
    _admit()
    # Budget spent. Ask for the answer itself rather than returning the last
    # tool call as though it were one.
    conversation.append({"role": "user", "content":
                         "You have used your tool budget. Answer now with what "
                         "you have, and say what you could not check. If the "
                         "original request requires JSON, keep the complete "
                         "object terse: cite tool names, paths, and short "
                         "facts rather than copying observation bodies."})
    final = complete(conversation)
    if parse_calls(final):
        trace.append({
            "tool": None,
            "ok": False,
            "dispatched": False,
            "error": "tool call emitted after bounded tool budget",
            "reply_excerpt": str(final or "")[:240],
        })
        return (
            "HCLI reached the bounded tool budget before the objective was "
            "evidenced. No tool call shown after that boundary was dispatched.",
            trace,
        )
    return _finalize(final)


def build_registry(workspace: str, repo_root: Optional[str] = None, *,
                   write: bool = False) -> Any:
    """Build the session's full typed registry under explicit authority.

    A normal browser session is read/research only. ``--write`` is the explicit
    HCLI authority switch: it admits reversible/workspace/repository work and
    costly tools, while the registry still rejects destructive or external-write
    classes and each costly operation keeps its own confirmation/verifier.
    """
    from .tool_registry import default_tool_registry
    from .tool_registry import (COSTLY, READ_ONLY, RESEARCH, REVERSIBLE_REPO,
                                REVERSIBLE_RUNTIME, REPO_WRITE, WORKSPACE_WRITE)
    permissions = {READ_ONLY, RESEARCH}
    if write:
        permissions.update({REVERSIBLE_REPO, REVERSIBLE_RUNTIME,
                            WORKSPACE_WRITE, REPO_WRITE, COSTLY})
    return default_tool_registry(workspace, repo_root=repo_root or workspace,
                                 permissions=permissions)


#: The probe a body must pass before it is offered tools at all. One exact
#: action, no ambiguity, cheap to run.
QUALIFY_PROMPT = (
    "List the files in the current directory. Use a tool to find out -- "
    "do not answer from memory."
)


def qualify(complete: Callable[[List[Dict[str, str]]], str],
            prefix: Optional[Sequence[Dict[str, str]]] = None,
            registry: Any = None, write: bool = False) -> Dict[str, Any]:
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
                    "attempts": attempts,
                    "offered_count": len(session_menu(registry, write=write))}
        if str(text or "").strip():
            break
    if not str(text or "").strip():
        return {"qualified": False,
                "reason": "the body returned an empty reply twice; it is not "
                          "generating, so tool capability is undetermined rather "
                          "than refused",
                "reply_excerpt": "", "attempts": attempts,
                "offered_count": len(session_menu(registry, write=write))}
    call = parse_call(text or "")
    if call is None:
        return {"qualified": False,
                "reason": "the body did not emit a parseable action for the "
                          "simplest possible request",
                "reply_excerpt": str(text or "")[:200], "attempts": attempts,
                "offered_count": len(session_menu(registry, write=write))}
    offered = session_menu(registry, write=write)
    if call[0] not in offered:
        return {"qualified": False,
                "reason": f"chose {call[0]!r}, which is not a tool it was offered",
                "reply_excerpt": str(text or "")[:200], "attempts": attempts,
                "offered_count": len(offered), "parsed_tool": call[0]}
    # Generation qualification does not claim that a dispatcher ran. The live
    # serving path records that stronger fact after the real invoke boundary.
    return {"qualified": True, "reason": None, "attempts": attempts,
            "offered_count": len(offered), "parsed_tool": call[0],
            "reply_excerpt": str(text or "")[:200]}


def shape_help(name: str, registry: Any = None) -> str:
    """What a caller must send. Named fields, not a refusal."""
    menu = session_menu(registry)
    if name in menu:
        shape = argument_shape(registry, name)
        return f"{name} takes: {shape}" if shape else f"{name}: {menu[name]}"
    return "use tools.catalog for the admitted tool names and signatures"


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
                   names: Optional[Sequence[str]] = None,
                   *, write: bool = False) -> List[Dict[str, Any]]:
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
    menu = session_menu(registry, write=write)
    for name in (names or prompt_menu_names(registry, write=write)):
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
    out = dict(arguments)
    # A body may put the compact dialect's operation tag inside the canonical
    # tool arguments. Remove it only when it names this exact canonical tool.
    compact = out.get("op")
    if isinstance(compact, str):
        compact_name = _compact_tool_name(compact)
        if (compact_name == name
                or (compact.strip().lower() == "read"
                    and name.endswith(".read"))):
            out.pop("op", None)
    if name == "repo.edit" and "operations" not in out:
        # Compact chat dialect for small local bodies. This is only syntax:
        # every reconstructed operation still crosses Engine's path checks,
        # syntax validation, snapshot, red-before-green test, and rollback.
        # A measured KIMI mission chose the correct files but overflowed three
        # nested `operations/new_lines` objects before closing their JSON.
        # Keep the wire representation proportional to the edit.
        raw_edits = out.pop("edits", None)
        if raw_edits is None and out.get("path") is not None:
            raw_edits = [{
                "path": out.pop("path"),
                "old": out.pop("old", None),
                "new": out.pop("new", None),
                "edit_op": out.pop("edit_op", None),
            }]
        if isinstance(raw_edits, dict):
            raw_edits = [raw_edits]

        def _line_list(value: Any) -> Optional[List[str]]:
            if value is None:
                return None
            if isinstance(value, list) and all(
                    isinstance(item, str) for item in value):
                return list(value)
            if isinstance(value, str):
                return value.splitlines() or [""]
            return None

        compact_operations: List[Dict[str, Any]] = []
        if isinstance(raw_edits, list):
            for item in raw_edits:
                if isinstance(item, (list, tuple)) and len(item) in (3, 4):
                    item = {
                        "path": item[0], "old": item[1], "new": item[2],
                        "edit_op": item[3] if len(item) == 4 else "replace",
                    }
                if not isinstance(item, dict) or item.get("path") is None:
                    compact_operations = []
                    break
                old = _line_list(item.get("old", item.get("old_lines")))
                new = _line_list(item.get("new", item.get("new_lines")))
                edit_op = str(item.get("edit_op") or item.get("kind") or (
                    "replace" if old is not None else "append"))
                if edit_op not in {
                        "append", "create", "replace", "insert_before",
                        "insert_after"} or new is None:
                    compact_operations = []
                    break
                operation: Dict[str, Any] = {
                    "op": edit_op, "path": item["path"], "new_lines": new,
                }
                if old is not None:
                    operation["old_lines"] = old
                compact_operations.append(operation)
        if compact_operations:
            out["operations"] = compact_operations
        compact_test = out.pop("test", None)
        if compact_test is not None and "tests" not in out:
            out["tests"] = ([compact_test] if isinstance(compact_test, str)
                            else compact_test)
    # Web search conventionally calls its text field `query`; repository
    # search historically calls the same concept `pattern`. A local body used
    # the common spelling in the blind mission and hit schema rejection before
    # any search recovery could run. This one-to-one rename is mechanics, not
    # an inferred search term.
    if (name in {"fs.search", "filesystem.search"}
            and "query" in out
            and "pattern" not in out):
        out["pattern"] = out.pop("query")
    # fs.search reports a one-based line; accepting that same evidence field
    # as fs.read's window start is an unambiguous shape repair, not a path or
    # authority expansion.
    if (name in {"fs.read", "filesystem.read"}
            and "line" in out
            and "start_line" not in out):
        line = out.pop("line")
        try:
            out["start_line"] = int(str(line).strip())
        except (TypeError, ValueError):
            out["start_line"] = line
    if not isinstance(props, dict) or not props:
        # BUILDER DOORS ARE NOT REGISTRY DOORS. repo.edit has no registry spec,
        # so props was empty and every argument was passed through as the text
        # the XML dialect produced -- leaving `operations` a string, which
        # run_builder_tool refuses on an isinstance(list) check. Measured live:
        # after 68 cycles of never reaching for the tool at all, the body
        # finally called repo.edit in cycles 69 and 70 with full new_lines,
        # old_lines and tests, and BOTH were rejected for malformed operations.
        # It was doing exactly what it was asked; the substrate was dropping
        # its arguments. The declared shapes carry the same schema, so fall
        # back to them rather than leaving builder calls uncoercible.
        declared = _declared_schema(name)
        props = (declared or {}).get("properties") if isinstance(declared, dict) else None
    if not isinstance(props, dict) or not props:
        return out
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
                if text.startswith("["):
                    out[key] = json.loads(text)
                elif ((props.get(key) or {}).get("items") or {}).get("type") == "string":
                    # A bare path where a list of paths was wanted: wrapping it
                    # is unambiguous. NOT done for object-item arrays like
                    # repo.edit's `operations` -- wrapping a garbage string into
                    # a one-element list would satisfy run_builder_tool's
                    # isinstance(list) check and hand the engine a list of
                    # strings where it expects operation dicts. Leaving it
                    # untouched keeps the refusal with the validator that can
                    # explain it.
                    out[key] = [text]
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
LOCAL_TOOLS = ("observation.expand", "debug.diagnose",
               "reverse.identify", "forensics.snapshot")


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
            end = int(arguments.get("end") or (start + 19))
            total_lines = len(cache.get(paste_id).splitlines())
            if start > total_lines:
                return _R(
                    False,
                    error=(f"observation {paste_id!r} has {total_lines} lines; "
                           f"requested start={start} is out of range. Use lines "
                           f"1..{total_lines} or search it with a narrow query."),
                    provenance={"source": "hcli.paste_cache", "paste": paste_id},
                )
            return _R(True, value={"id": paste_id, "start": start, "end": end,
                                   "total_lines": total_lines,
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
        from .capabilities import diagnose as _diagnose
        exit_code = arguments.get("exit_code")
        try:
            exit_code = int(exit_code) if exit_code is not None else 1
        except (TypeError, ValueError):
            exit_code = 1
        failure = _diagnose(output, command=str(arguments.get("command") or "?"),
                            exit_code=exit_code, root=_Path.cwd(), cache=cache)
        return _R(True, value=failure.to_dict(),
                  provenance={"source": "hcli.capabilities", "paste": paste_id})

    if name == "reverse.identify":
        from .capabilities import identify
        target = str(arguments.get("path") or "").strip()
        if not target:
            return _R(False, error='reverse.identify needs {"path": "<file>"}')
        result = identify(target)
        return _R("error" not in result, value=result,
                  error=result.get("error"),
                  provenance={"source": "hcli.capabilities"})

    if name == "forensics.snapshot":
        from .capabilities import capture
        import os as _os
        snap = capture(_os.getcwd(), note=str(arguments.get("note") or ""))
        return _R(True, value=snap.to_dict(),
                  provenance={"source": "hcli.capabilities",
                              "preserved": snap.path})

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
    # THE BODY MUST BE ABLE TO RUN ITS OWN DISCRIMINATOR. The capability map
    # advertises TEST ("run admitted tests") and the registry carries a bounded
    # runner, but the chat surface offered 11 doors out of 109 registry tools
    # and not one of them ran anything: debug.diagnose localizes a failure from
    # output it cannot produce, and repo.edit runs tests only as a rollback gate
    # on a mutation already written. So "form a hypothesis, then write a RED
    # discriminator" had no executable middle step -- the step that decides
    # whether a repair is needed at all. A write session gets the runner; a read
    # session does not, because this executes code on the host.
    "tests.run": ("run admitted tests to settle a question BEFORE changing "
                  "anything: pytest/unittest/cargo over given paths"),
}

#: The subset of the builder menu that run_builder_tool itself serves, i.e. the
#: doors that go through the mutation transaction. Everything else in the menu
#: is an ordinary registry tool and must be dispatched as one -- keeping these
#: separate is what lets a builder door exist without pretending to be a
#: mutation.
MUTATION_TOOLS = ("repo.edit",)

#: Name every op the applier implements, and name the CHEAPEST one first.
#: Measured, campaign cycles 69-74: five consecutive `repo.edit` calls died the
#: same way -- `operations` truncated mid-JSON, because the body had chosen
#: `create` and was emitting a whole file inline. It had ~2534 tokens of
#: generation budget and a whole file does not fit. `append` needs no anchor and
#: no file body, `_apply_operations` has always implemented it, and this string
#: never said so; the contract was steering a budget-limited body straight at
#: the most expensive op in the set. `insert_before` and `replace_file` were
#: missing too. test_the_builder_menu_names_every_op pins the agreement.
BUILDER_SHAPES = {
    "repo.edit": (
        'COMPACT PREFERRED: JSON fields op=edit; edits is a flat list of '
        'three-item JSON-string arrays containing repository path, one verbatim '
        'existing line, and authored replacement text; test is one focused '
        'repository test. Edit rows are arrays, never nested objects or '
        'old_lines/new_lines lists. '
        'Literal labels path/old/new/focused/test.py are invalid placeholders. '
        'Use two rows in edits when source and regression test must change '
        'atomically. The authority layer still implements append, create, '
        'replace, insert_before, and insert_after, but the browser contract uses '
        'compact replace rows so a small body does not inline whole files. '
        'Whole-file replacement is intentionally not a normal chat-mission shape. '
        'tests is the list form of test and is REQUIRED when the objective requires '
        'an accepted mutation'),
}

#: What a builder verdict means. Kept distinct on purpose: S035 s26 -- a
#: mutation that was rejected must never be reported as "fixed", and a change
#: with no test is "unproven", not "verified".
BUILDER_VERDICTS = ("accepted", "unproven", "rejected")


def builder_menu(write: bool) -> Dict[str, str]:
    """The doors for this session. Read-only unless authority was granted."""
    return {**CHAT_TOOLS, **(BUILDER_TOOLS if write else {})}


def run_builder_tool(name: str, arguments: Dict[str, Any], *,
                     engine: Any = None, require_tests: bool = False) -> Any:
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
    if isinstance(operations, str):
        # SAY WHICH FAILURE IT WAS. The shape string is correct for a wrong TYPE
        # and actively misleading for a payload that failed to PARSE -- which is
        # what a body emitting a whole file inline actually produces, since one
        # unterminated string in the generated content invalidates everything.
        # Measured: four consecutive cycles sent a repo.edit and got the same
        # shape string back, so the body re-sent the same broken payload each
        # time. It could not tell "wrong kind of thing" from "your JSON broke at
        # character 812". This does not accept bad JSON; it names the failure.
        try:
            operations = json.loads(operations)
        except ValueError as exc:
            return _R(False, error=(
                f"repo.edit's `operations` was a string that is not valid JSON: {exc}. "
                f"Emit a SMALLER payload -- a long inline file is where this breaks. "
                f"Expected {BUILDER_SHAPES['repo.edit']}"))
        if isinstance(operations, dict):
            operations = [operations]
    if not isinstance(operations, list) or not operations:
        return _R(False, error=f"repo.edit needs {BUILDER_SHAPES['repo.edit']}")
    tests = arguments.get("tests")
    if isinstance(tests, str):
        tests = [tests]
    if require_tests and not (isinstance(tests, list) and tests):
        return _R(False, error=(
            "this explicit mission requires an accepted mutation, so repo.edit "
            "must include a non-empty `tests` list in the same transaction; "
            "no mutation was applied"))
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
